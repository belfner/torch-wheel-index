from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag
from packaging.utils import InvalidWheelFilename
from packaging.utils import canonicalize_name
from packaging.utils import parse_wheel_filename
from packaging.version import InvalidVersion
from packaging.version import Version

from torch_wheel_index._client import make_semaphore
from torch_wheel_index._parser import build_package_instance
from torch_wheel_index._parser import cpython_tag_to_version
from torch_wheel_index._parser import filter_wheels
from torch_wheel_index._parser import parse_platform
from torch_wheel_index._parser import wheel_matches_index
from torch_wheel_index._scraper import PYTORCH_INDEX_URL
from torch_wheel_index._scraper import fetch_package_wheels
from torch_wheel_index._scraper import fetch_wheel_metadata
from torch_wheel_index._scraper import fetch_wheel_filenames
from torch_wheel_index._scraper import get_paired_versions
from torch_wheel_index._scraper import get_torch_sub_indexes
from torch_wheel_index.models import Catalog
from torch_wheel_index.models import Platform
from torch_wheel_index.models import SCHEMA_VERSION
from torch_wheel_index.models import PackageInstance
from torch_wheel_index.models import TorchaudioChoice
from torch_wheel_index.paths import default_cache_path

logger = logging.getLogger(__name__)

DEFAULT_CUTOFF_VERSION = Version("2.0.0")
DEFAULT_MAX_CACHE_AGE_SECONDS = 86400.0


async def _latest_at_main_index(package: str, semaphore: asyncio.Semaphore) -> Version | None:
    url = f"{PYTORCH_INDEX_URL}/{package}"
    filenames = await fetch_wheel_filenames(url, semaphore)
    versions: list[Version] = []
    for filename in filenames:
        _, version, _, _ = parse_wheel_filename(filename)
        versions.append(Version(version.base_version))
    if len(versions) == 0:
        return None
    return max(versions)


async def fetch_catalog_async(
    cutoff_version: Version = DEFAULT_CUTOFF_VERSION,
    semaphore: asyncio.Semaphore | None = None,
) -> Catalog:
    """
    Scrape PyTorch indexes and assemble a Catalog.

    Parameters
    ----------
    cutoff_version : Version, optional
        Minimum torch version to include, by default 2.0.0.
    semaphore : asyncio.Semaphore, optional
        Concurrency limiter. A new one is created if not supplied.

    Returns
    -------
    Catalog
        Frozen catalog of available torch wheels and version pairings.
    """
    sem = semaphore or make_semaphore()

    logger.info("discovering sub-indexes")
    sub_indexes = await get_torch_sub_indexes(sem)
    logger.info("found %d valid sub-indexes", len(sub_indexes))

    logger.info("fetching torch, torchvision, and torchaudio wheel listings")
    torch_tasks = [fetch_package_wheels(url, "torch", sem) for url, _ in sub_indexes]
    vision_tasks = [fetch_package_wheels(url, "torchvision", sem) for url, _ in sub_indexes]
    audio_tasks = [fetch_package_wheels(url, "torchaudio", sem) for url, _ in sub_indexes]
    pairs_task = get_paired_versions(sem, cutoff_version)

    torch_listings, vision_listings, audio_listings, official_pairs = await asyncio.gather(
        asyncio.gather(*torch_tasks),
        asyncio.gather(*vision_tasks),
        asyncio.gather(*audio_tasks),
        pairs_task,
    )
    official_vision_pairs, official_audio_pairs = official_pairs

    releases = _build_releases(sub_indexes, torch_listings, cutoff_version)
    vision_versions = _collect_wheel_versions(sub_indexes, vision_listings)
    audio_served, audio_samples = _collect_audio_inventory(sub_indexes, audio_listings)
    audio_requirements = await _fetch_audio_requirements(audio_samples, sem)
    vision_pairs = _resolve_version_pairs(releases, vision_versions, official_vision_pairs)
    audio_choices = _choose_torchaudio(releases, audio_served, audio_requirements, official_audio_pairs)
    audio_pairs: dict[Version, Version] = {}
    for release, choice in audio_choices.items():
        if release.version not in audio_pairs or choice.version > audio_pairs[release.version]:
            audio_pairs[release.version] = choice.version

    logger.info("collected %d torch releases, %d with a torchaudio choice", len(releases), len(audio_choices))
    return Catalog(
        releases=releases,
        torchvision_pairs=vision_pairs,
        torchaudio_pairs=audio_pairs,
        torchaudio_choices=audio_choices,
        torchaudio_latest=max(audio_samples) if len(audio_samples) > 0 else None,
    )


def _build_releases(
    sub_indexes: list[tuple[str, str]],
    torch_listings: list[list[str]],
    cutoff_version: Version,
) -> list[PackageInstance]:
    seen: set[PackageInstance] = set()
    releases: list[PackageInstance] = []
    for (_, key), filenames in zip(sub_indexes, torch_listings, strict=True):
        for version, tags in filter_wheels(filenames, key, cutoff_version):
            instance = build_package_instance(version, key, tags, PYTORCH_INDEX_URL)
            if instance not in seen:
                seen.add(instance)
                releases.append(instance)
    return releases


_AudioKey = tuple[Version | None, Platform]
"""(python version, platform) a torchaudio wheel installs on. None stands for a `py3-none` wheel's any-version."""


@dataclass(frozen=True)
class _AudioRequirements:
    """
    Requirements a torchaudio release declares in its wheel metadata.

    Parameters
    ----------
    torch : SpecifierSet | None
        Intersection of the torch specifiers it declares, None when it
        declares no torch requirement.
    python : SpecifierSet | None
        Its Requires-Python specifier, None when it declares none.
    """

    torch: SpecifierSet | None
    python: SpecifierSet | None


def _audio_wheel_keys(tags: frozenset[Tag]) -> set[_AudioKey]:
    """
    List the (python version, platform) keys a torchaudio wheel installs on.

    torchaudio wheels up to 2.11.0 are built per CPython version (`cp313`);
    2.11.1 ships one `py3-none` wheel per platform, which installs on every
    python version its Requires-Python admits.

    Parameters
    ----------
    tags : frozenset[Tag]
        Wheel tags.

    Returns
    -------
    set[_AudioKey]
        Keys for the wheel, with None as the python version of a `py3` wheel.
    """
    keys: set[_AudioKey] = set()
    for tag in tags:
        try:
            platform = parse_platform(tag.platform)
        except ValueError:
            continue
        if tag.interpreter.startswith("cp"):
            keys.add((cpython_tag_to_version(tag.interpreter), platform))
        elif tag.interpreter.startswith("py3"):
            keys.add((None, platform))
    return keys


def _collect_audio_inventory(
    sub_indexes: list[tuple[str, str]],
    listings: list[list[str]],
) -> tuple[dict[str, dict[_AudioKey, set[Version]]], dict[Version, tuple[str, str]]]:
    """
    Record what torchaudio wheels each sub-index serves.

    Parameters
    ----------
    sub_indexes : list[tuple[str, str]]
        Discovered (url, key) sub-index pairs.
    listings : list[list[str]]
        torchaudio wheel filenames per sub-index, aligned with `sub_indexes`.

    Returns
    -------
    tuple[dict[str, dict[_AudioKey, set[Version]]], dict[Version, tuple[str, str]]]
        Per-index map of (python version, platform) to the torchaudio versions
        served there, and one (sub-index url, filename) sample per version for
        metadata lookups.
    """
    served: dict[str, dict[_AudioKey, set[Version]]] = {}
    samples: dict[Version, tuple[str, str]] = {}
    for (url, key), filenames in zip(sub_indexes, listings, strict=True):
        bucket = served.setdefault(f"{PYTORCH_INDEX_URL}/{key}", {})
        for filename in filenames:
            try:
                _, wheel_version, _, tags = parse_wheel_filename(filename)
            except (InvalidWheelFilename, InvalidVersion):
                continue
            if not wheel_matches_index(wheel_version, tags, key):
                continue
            version = Version(wheel_version.base_version)
            for audio_key in _audio_wheel_keys(tags):
                bucket.setdefault(audio_key, set()).add(version)
            samples.setdefault(version, (url, filename))
    return served, samples


def _parse_audio_requirements(metadata: str) -> _AudioRequirements:
    """
    Extract the torch and python constraints from a torchaudio wheel's METADATA.

    Parameters
    ----------
    metadata : str
        METADATA text from a PEP 658 sidecar.

    Returns
    -------
    _AudioRequirements
        The declared torch specifier and Requires-Python specifier.
    """
    torch: SpecifierSet | None = None
    python: SpecifierSet | None = None
    for line in metadata.splitlines():
        field_name, _, value = line.partition(":")
        match field_name.strip().lower():
            case "requires-dist":
                try:
                    requirement = Requirement(value.strip())
                except InvalidRequirement:
                    continue
                if canonicalize_name(requirement.name) != "torch":
                    continue
                torch = requirement.specifier if torch is None else torch & requirement.specifier
            case "requires-python":
                try:
                    python = SpecifierSet(value.strip())
                except InvalidSpecifier:
                    continue
    return _AudioRequirements(torch=torch, python=python)


async def _fetch_audio_requirements(
    samples: dict[Version, tuple[str, str]],
    semaphore: asyncio.Semaphore,
) -> dict[Version, _AudioRequirements]:
    """
    Read the declared requirements of every torchaudio version.

    torchaudio wheels up to 2.10 pin an exact torch version, 2.11.0 declares
    no torch requirement, and 2.11.1 declares `torch>=2.11`. The PEP 658
    metadata sidecar carries the answer for each version, so pairing follows
    what the wheels declare.

    Parameters
    ----------
    samples : dict[Version, tuple[str, str]]
        One (sub-index url, wheel filename) per torchaudio version.
    semaphore : asyncio.Semaphore
        Concurrency limiter.

    Returns
    -------
    dict[Version, _AudioRequirements]
        Requirements per torchaudio version whose metadata sidecar was
        readable. Versions with an unreadable sidecar are absent.
    """
    versions = list(samples)
    texts = await asyncio.gather(*(fetch_wheel_metadata(*samples[v], semaphore) for v in versions))
    requirements: dict[Version, _AudioRequirements] = {}
    for version, text in zip(versions, texts, strict=True):
        if len(text) == 0:
            logger.info("no metadata sidecar for torchaudio %s; using the pairing table for it", version)
            continue
        requirements[version] = _parse_audio_requirements(text)
    return requirements


def _audio_accepts_torch(
    audio_version: Version,
    torch_version: Version,
    requirements: _AudioRequirements | None,
    official_pairs: dict[Version, Version],
) -> bool:
    """
    Decide whether a torchaudio version installs alongside a torch version.

    A declared torch specifier decides directly. A wheel declaring no torch
    requirement supports the torch release it shipped with and every later
    one, and torchaudio shares torch's major.minor numbering, so it accepts
    torch versions from its own major.minor on. Without readable metadata the
    official pairing table decides, and a torch version the table skips takes
    the torchaudio release sharing its version number.

    Parameters
    ----------
    audio_version : Version
        Candidate torchaudio version.
    torch_version : Version
        Torch release version.
    requirements : _AudioRequirements | None
        The candidate's declared requirements, None when its metadata was
        unreadable.
    official_pairs : dict[Version, Version]
        torch-to-torchaudio pairings scraped from the official table.

    Returns
    -------
    bool
        True when the pair installs together.
    """
    if requirements is None:
        documented = official_pairs.get(torch_version)
        if documented is not None:
            return audio_version == documented
        return audio_version == torch_version
    if requirements.torch is not None:
        return requirements.torch.contains(torch_version)
    return torch_version >= Version(f"{audio_version.major}.{audio_version.minor}")


def _newest_compatible_audio(
    release: PackageInstance,
    wheels: dict[_AudioKey, set[Version]],
    requirements: dict[Version, _AudioRequirements],
    official_pairs: dict[Version, Version],
) -> Version | None:
    """
    Find the newest torchaudio version one index serves for a release.

    Parameters
    ----------
    release : PackageInstance
        Torch release being paired.
    wheels : dict[_AudioKey, set[Version]]
        The index's torchaudio versions per (python version, platform).
    requirements : dict[Version, _AudioRequirements]
        Declared requirements per torchaudio version.
    official_pairs : dict[Version, Version]
        torch-to-torchaudio pairings scraped from the official table.

    Returns
    -------
    Version | None
        Newest torchaudio version with a wheel for the release's python
        version and platform that accepts the release's torch version, or None
        when the index has none.
    """
    candidates = set(wheels.get((release.python_version, release.platform), set()))
    for version in wheels.get((None, release.platform), set()):
        declared = requirements.get(version)
        if declared is None or declared.python is None or declared.python.contains(release.python_version):
            candidates.add(version)
    compatible = [
        version
        for version in candidates
        if _audio_accepts_torch(version, release.version, requirements.get(version), official_pairs)
    ]
    return max(compatible, default=None)


def _choose_torchaudio(
    releases: list[PackageInstance],
    served: dict[str, dict[_AudioKey, set[Version]]],
    requirements: dict[Version, _AudioRequirements],
    official_pairs: dict[Version, Version],
) -> dict[PackageInstance, TorchaudioChoice]:
    """
    Choose the torchaudio requirement for every torch release.

    A release takes the newest compatible torchaudio its own index serves for
    its python version and platform, routed to that index. When its index has
    none, it takes the newest compatible version any scraped index serves for
    that coordinate, routed to the default registry, which publishes the same
    torchaudio versions and build matrix.

    Parameters
    ----------
    releases : list[PackageInstance]
        Torch releases the catalog covers.
    served : dict[str, dict[_AudioKey, set[Version]]]
        Per-index torchaudio inventory from `_collect_audio_inventory`.
    requirements : dict[Version, _AudioRequirements]
        Declared requirements per torchaudio version.
    official_pairs : dict[Version, Version]
        torch-to-torchaudio pairings scraped from the official table.

    Returns
    -------
    dict[PackageInstance, TorchaudioChoice]
        Choice per release that has a compatible torchaudio anywhere.
    """
    choices: dict[PackageInstance, TorchaudioChoice] = {}
    for release in releases:
        own = _newest_compatible_audio(release, served.get(release.index_url, {}), requirements, official_pairs)
        if own is not None:
            choices[release] = TorchaudioChoice(version=own, index_url=release.index_url)
            continue
        elsewhere = [
            found
            for wheels in served.values()
            if (found := _newest_compatible_audio(release, wheels, requirements, official_pairs)) is not None
        ]
        if len(elsewhere) > 0:
            choices[release] = TorchaudioChoice(version=max(elsewhere), index_url="")
    return choices


def _collect_wheel_versions(
    sub_indexes: list[tuple[str, str]],
    listings: list[list[str]],
) -> list[Version]:
    versions: set[Version] = set()
    cutoff = Version("0.0.0")
    for (_, key), filenames in zip(sub_indexes, listings, strict=True):
        for version, _ in filter_wheels(filenames, key, cutoff):
            versions.add(Version(version.base_version))
    return sorted(versions, reverse=True)


def _resolve_version_pairs(
    releases: list[PackageInstance],
    vision_versions: list[Version],
    official_pairs: dict[Version, Version],
) -> dict[Version, Version]:
    pairs: dict[Version, Version] = dict(official_pairs)

    torch_versions = sorted({r.version for r in releases}, reverse=True)
    overlap = min(len(torch_versions), len(vision_versions))
    for torch_v, vision_v in zip(torch_versions[:overlap], vision_versions[:overlap], strict=True):
        if torch_v not in pairs:
            pairs[torch_v] = vision_v

    if len(vision_versions) > 0:
        fallback = vision_versions[0]
        for torch_v in torch_versions:
            if torch_v not in pairs:
                pairs[torch_v] = fallback

    return pairs


def fetch_catalog(cutoff_version: Version = DEFAULT_CUTOFF_VERSION) -> Catalog:
    """
    Synchronous wrapper for `fetch_catalog_async`.

    Parameters
    ----------
    cutoff_version : Version, optional
        Minimum torch version to include, by default 2.0.0.

    Returns
    -------
    Catalog
        Freshly fetched catalog.
    """
    return asyncio.run(fetch_catalog_async(cutoff_version))


async def refresh_if_stale_async(
    cache_path: Path,
    cutoff_version: Version = DEFAULT_CUTOFF_VERSION,
    force: bool = False,
    max_age_seconds: float = DEFAULT_MAX_CACHE_AGE_SECONDS,
) -> Catalog:
    """
    Return a Catalog, refreshing the cache file only when needed.

    Reads `cache_path` and compares the latest torch, torchvision, and
    torchaudio versions against what the PyTorch index serves. If any is newer,
    a full scrape runs and the cache is rewritten. A cache written under an
    older schema, or older than `max_age_seconds`, also triggers a scrape: the
    version probes are blind to a sub-index gaining or losing wheels of a
    version it already served, and the age bound is what puts a ceiling on how
    long such a change stays invisible.

    Parameters
    ----------
    cache_path : Path
        File path of the cached catalog JSON.
    cutoff_version : Version, optional
        Minimum torch version to include when refreshing, by default 2.0.0.
    force : bool, optional
        Refresh unconditionally, by default False.
    max_age_seconds : float, optional
        Rescrape a cache older than this, by default 24 hours.

    Returns
    -------
    Catalog
        Either the up-to-date cached catalog or a freshly fetched one.
    """
    from torch_wheel_index.serialization import load_catalog
    from torch_wheel_index.serialization import save_catalog

    sem = make_semaphore()

    cached: Catalog | None = None
    if not force and cache_path.exists():
        try:
            cached = load_catalog(cache_path)
        except (OSError, ValueError) as exc:
            logger.warning("failed to read cache at %s: %s", cache_path, exc)

    if cached is not None and cached.schema_version < SCHEMA_VERSION:
        logger.info(
            "cache at %s uses schema %d; rescraping for schema %d",
            cache_path,
            cached.schema_version,
            SCHEMA_VERSION,
        )
        cached = None

    if cached is not None:
        age = time.time() - cache_path.stat().st_mtime
        if age > max_age_seconds:
            logger.info("cache at %s is %.0fs old; rescraping", cache_path, age)
            cached = None

    latest_torch, latest_vision, latest_audio = await asyncio.gather(
        _latest_at_main_index("torch", sem),
        _latest_at_main_index("torchvision", sem),
        _latest_at_main_index("torchaudio", sem),
    )

    if (
        cached is not None
        and latest_torch is not None
        and latest_vision is not None
        and latest_audio is not None
        and not force
    ):
        try:
            cached_torch = cached.newest_version()
        except ValueError:
            cached_torch = Version("0.0.0")
        try:
            cached_vision = cached.newest_torchvision_version()
        except ValueError:
            cached_vision = Version("0.0.0")
        cached_audio = cached.torchaudio_latest or Version("0.0.0")

        if latest_torch <= cached_torch and latest_vision <= cached_vision and latest_audio <= cached_audio:
            logger.info(
                "cache up-to-date (torch=%s, torchvision=%s, torchaudio=%s)",
                cached_torch,
                cached_vision,
                cached_audio,
            )
            return cached

        logger.info(
            "cache stale: online torch=%s vision=%s audio=%s, cached torch=%s vision=%s audio=%s",
            latest_torch,
            latest_vision,
            latest_audio,
            cached_torch,
            cached_vision,
            cached_audio,
        )

    catalog = await fetch_catalog_async(cutoff_version, semaphore=sem)
    save_catalog(catalog, cache_path)
    return catalog


def refresh_if_stale(
    cache_path: Path,
    cutoff_version: Version = DEFAULT_CUTOFF_VERSION,
    force: bool = False,
    max_age_seconds: float = DEFAULT_MAX_CACHE_AGE_SECONDS,
) -> Catalog:
    """
    Synchronous wrapper for `refresh_if_stale_async`.

    Parameters
    ----------
    cache_path : Path
        File path of the cached catalog JSON.
    cutoff_version : Version, optional
        Minimum torch version to include when refreshing, by default 2.0.0.
    force : bool, optional
        Refresh unconditionally, by default False.
    max_age_seconds : float, optional
        Rescrape a cache older than this, by default 24 hours.

    Returns
    -------
    Catalog
        Cached or freshly fetched catalog.
    """
    return asyncio.run(refresh_if_stale_async(cache_path, cutoff_version, force, max_age_seconds))


async def get_catalog_async(force: bool = False) -> Catalog:
    """
    Convenience: read or refresh the catalog using the OS cache directory.

    Parameters
    ----------
    force : bool, optional
        Refresh unconditionally, by default False.

    Returns
    -------
    Catalog
        Catalog backed by the default OS cache path.
    """
    return await refresh_if_stale_async(default_cache_path(), force=force)


def get_catalog(force: bool = False) -> Catalog:
    """
    Synchronous wrapper for `get_catalog_async`.

    Parameters
    ----------
    force : bool, optional
        Refresh unconditionally, by default False.

    Returns
    -------
    Catalog
        Catalog backed by the default OS cache path.
    """
    return asyncio.run(get_catalog_async(force))
