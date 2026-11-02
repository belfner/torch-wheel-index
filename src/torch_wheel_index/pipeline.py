from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

from packaging.markers import Marker
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

from torch_wheel_index._client import ScrapeError
from torch_wheel_index._client import make_semaphore
from torch_wheel_index._dependencies import IndexDependencyResolver
from torch_wheel_index._dependencies import resolve_extras
from torch_wheel_index._parser import build_package_instance
from torch_wheel_index._parser import cpython_tag_to_version
from torch_wheel_index._parser import filter_wheels
from torch_wheel_index._parser import parse_compute_version
from torch_wheel_index._parser import parse_platform
from torch_wheel_index._parser import wheel_matches_index
from torch_wheel_index._scraper import PYTORCH_INDEX_URL
from torch_wheel_index._scraper import fetch_package_wheels
from torch_wheel_index._scraper import fetch_wheel_filenames
from torch_wheel_index._scraper import fetch_wheel_metadata
from torch_wheel_index._scraper import get_paired_versions
from torch_wheel_index._scraper import get_torch_sub_indexes
from torch_wheel_index.models import Catalog
from torch_wheel_index.models import IndexDependency
from torch_wheel_index.models import PackageChoice
from torch_wheel_index.models import PackageInstance
from torch_wheel_index.models import Platform
from torch_wheel_index.models import SCHEMA_VERSION
from torch_wheel_index.paths import default_cache_path

logger = logging.getLogger(__name__)

DEFAULT_CUTOFF_VERSION = Version("2.0.0")
DEFAULT_MAX_CACHE_AGE_SECONDS = 86400.0


async def _latest_at_main_index(package: str, semaphore: asyncio.Semaphore) -> Version | None:
    url = f"{PYTORCH_INDEX_URL}/{package}"
    try:
        filenames = await fetch_wheel_filenames(url, semaphore)
    except ScrapeError as exc:
        logger.warning("latest-version probe failed: %s", exc)
        return None
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
    sub_indexes = _known_sub_indexes(await get_torch_sub_indexes(sem))
    logger.info("found %d valid sub-indexes", len(sub_indexes))

    logger.info("fetching torch, torchvision, and torchaudio wheel listings")
    torch_tasks = [fetch_package_wheels(url, "torch", sem) for url, _ in sub_indexes]
    vision_tasks = [fetch_package_wheels(url, "torchvision", sem, missing_ok=True) for url, _ in sub_indexes]
    audio_tasks = [fetch_package_wheels(url, "torchaudio", sem, missing_ok=True) for url, _ in sub_indexes]
    pairs_task = get_paired_versions(sem, cutoff_version)

    torch_listings, vision_listings, audio_listings, official_pairs = await asyncio.gather(
        asyncio.gather(*torch_tasks),
        asyncio.gather(*vision_tasks),
        asyncio.gather(*audio_tasks),
        pairs_task,
    )
    official_vision_pairs, official_audio_pairs = official_pairs

    releases, torch_samples = _build_releases(sub_indexes, torch_listings, cutoff_version)
    vision_versions = _collect_wheel_versions(sub_indexes, vision_listings)
    vision_served, _ = _collect_package_inventory(sub_indexes, vision_listings)
    audio_served, audio_samples = _collect_package_inventory(sub_indexes, audio_listings)
    audio_requirements = await _fetch_audio_requirements(audio_samples, sem)
    vision_pairs = _resolve_version_pairs(releases, vision_versions, official_vision_pairs)
    vision_choices = _choose_torchvision(releases, vision_pairs, vision_served)
    audio_choices = _choose_torchaudio(releases, audio_served, audio_requirements, official_audio_pairs)
    audio_pairs: dict[Version, Version] = {}
    for release, choice in audio_choices.items():
        if release.version not in audio_pairs or choice.version > audio_pairs[release.version]:
            audio_pairs[release.version] = choice.version

    index_dependencies = await _resolve_index_dependencies(releases, torch_samples, sem)

    logger.info(
        "collected %d torch releases, %d with a torchaudio choice, %d with index-only dependencies",
        len(releases),
        len(audio_choices),
        len(index_dependencies),
    )
    return Catalog(
        releases=releases,
        torchvision_pairs=vision_pairs,
        torchvision_choices=vision_choices,
        torchaudio_pairs=audio_pairs,
        torchaudio_choices=audio_choices,
        index_dependencies=index_dependencies,
        torchaudio_latest=max(audio_samples) if len(audio_samples) > 0 else None,
        cutoff_version=cutoff_version,
        generated_at=time.time(),
    )


_TorchSampleKey = tuple[Version, str, Platform]
"""(torch version, sub-index URL, platform) one sample wheel stands for."""


def _known_sub_indexes(sub_indexes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Keep the sub-indexes whose key names a known compute backend.

    PyTorch adds backends over time; a key `parse_compute_version` does not
    recognize is logged and left out so the rest of the catalog still builds.

    Parameters
    ----------
    sub_indexes : list[tuple[str, str]]
        Discovered (url, key) sub-index pairs.

    Returns
    -------
    list[tuple[str, str]]
        The pairs with a recognized key, in input order.
    """
    known: list[tuple[str, str]] = []
    for url, key in sub_indexes:
        try:
            parse_compute_version(key)
        except ValueError:
            logger.warning("skipping sub-index %s: unrecognized compute backend %r", url, key)
            continue
        known.append((url, key))
    return known


def _build_releases(
    sub_indexes: list[tuple[str, str]],
    torch_listings: list[list[str]],
    cutoff_version: Version,
) -> tuple[list[PackageInstance], dict[_TorchSampleKey, tuple[str, str]]]:
    """
    Build the torch release list and pick one sample wheel per coordinate.

    Wheels whose filename, interpreter, or platform tag cannot be classified
    are skipped, so a new platform PyTorch starts publishing for does not stop
    the scrape.

    Parameters
    ----------
    sub_indexes : list[tuple[str, str]]
        Discovered (url, key) sub-index pairs.
    torch_listings : list[list[str]]
        torch wheel filenames per sub-index, aligned with `sub_indexes`.
    cutoff_version : Version
        Minimum torch version to include.

    Returns
    -------
    tuple[list[PackageInstance], dict[_TorchSampleKey, tuple[str, str]]]
        Unique releases in discovery order, and one (sub-index url, filename)
        per (torch version, index URL, platform) for metadata reads.
    """
    seen: set[PackageInstance] = set()
    releases: list[PackageInstance] = []
    samples: dict[_TorchSampleKey, tuple[str, str]] = {}
    for (url, key), filenames in zip(sub_indexes, torch_listings, strict=True):
        for filename in filenames:
            try:
                _, version, _, tags = parse_wheel_filename(filename)
                if version < cutoff_version or not wheel_matches_index(version, tags, key):
                    continue
                instance = build_package_instance(version, key, tags, PYTORCH_INDEX_URL)
            except (InvalidWheelFilename, InvalidVersion, ValueError) as exc:
                logger.debug("skipping torch wheel %s: %s", filename, exc)
                continue
            samples.setdefault((instance.version, instance.index_url, instance.platform), (url, filename))
            if instance not in seen:
                seen.add(instance)
                releases.append(instance)
    return releases, samples


async def _resolve_index_dependencies(
    releases: list[PackageInstance],
    samples: dict[_TorchSampleKey, tuple[str, str]],
    semaphore: asyncio.Semaphore,
) -> dict[PackageInstance, tuple[IndexDependency, ...]]:
    """
    Find, per release, the dependencies only its own index serves.

    Each (torch version, index, platform) is resolved once from its sample
    wheel's metadata sidecar. Wheels published without a sidecar contribute
    nothing.

    Parameters
    ----------
    releases : list[PackageInstance]
        Torch releases the catalog covers.
    samples : dict[_TorchSampleKey, tuple[str, str]]
        One (sub-index url, filename) per coordinate from `_build_releases`.
    semaphore : asyncio.Semaphore
        Concurrency limiter.

    Returns
    -------
    dict[PackageInstance, tuple[IndexDependency, ...]]
        Index-only dependencies per release that has any.
    """
    resolver = IndexDependencyResolver(semaphore)

    pythons_by_key: dict[_TorchSampleKey, set[Version]] = {}
    for release in releases:
        pythons_by_key.setdefault((release.version, release.index_url, release.platform), set()).add(
            release.python_version
        )

    async def resolve(key: _TorchSampleKey) -> tuple[IndexDependency, ...]:
        _, index_url, platform = key
        url, filename = samples[key]
        requirements = await resolver.requirements(f"{url}/{quote(filename)}", True, filename)
        if requirements is None:
            return ()
        roots: list[tuple[Requirement, Marker | None]] = []
        for requirement in requirements:
            applies, marker = resolve_extras(requirement.marker, frozenset())
            if applies:
                roots.append((requirement, marker))
        pythons = tuple(sorted(pythons_by_key.get(key, set())))
        return await resolver.index_only(index_url, platform, pythons, roots)

    keys = list(samples)
    resolved = dict(zip(keys, await asyncio.gather(*(resolve(key) for key in keys)), strict=True))
    dependencies: dict[PackageInstance, tuple[IndexDependency, ...]] = {}
    for release in releases:
        found = resolved.get((release.version, release.index_url, release.platform), ())
        if len(found) > 0:
            dependencies[release] = found
    return dependencies


_WheelKey = tuple[Version | None, Platform]
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


def _wheel_keys(tags: frozenset[Tag]) -> set[_WheelKey]:
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
    set[_WheelKey]
        Keys for the wheel, with None as the python version of a `py3` wheel.
    """
    keys: set[_WheelKey] = set()
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


def _collect_package_inventory(
    sub_indexes: list[tuple[str, str]],
    listings: list[list[str]],
) -> tuple[dict[str, dict[_WheelKey, set[Version]]], dict[Version, tuple[str, str]]]:
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
    tuple[dict[str, dict[_WheelKey, set[Version]]], dict[Version, tuple[str, str]]]
        Per-index map of (python version, platform) to the torchaudio versions
        served there, and one (sub-index url, filename) sample per version for
        metadata lookups.
    """
    served: dict[str, dict[_WheelKey, set[Version]]] = {}
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
            for audio_key in _wheel_keys(tags):
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
            logger.info("metadata for torchaudio %s is unreadable; using the pairing table for it", version)
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
    wheels: dict[_WheelKey, set[Version]],
    requirements: dict[Version, _AudioRequirements],
    official_pairs: dict[Version, Version],
) -> Version | None:
    """
    Find the newest torchaudio version one index serves for a release.

    Parameters
    ----------
    release : PackageInstance
        Torch release being paired.
    wheels : dict[_WheelKey, set[Version]]
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
    served: dict[str, dict[_WheelKey, set[Version]]],
    requirements: dict[Version, _AudioRequirements],
    official_pairs: dict[Version, Version],
) -> dict[PackageInstance, PackageChoice]:
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
    served : dict[str, dict[_WheelKey, set[Version]]]
        Per-index torchaudio inventory from `_collect_package_inventory`.
    requirements : dict[Version, _AudioRequirements]
        Declared requirements per torchaudio version.
    official_pairs : dict[Version, Version]
        torch-to-torchaudio pairings scraped from the official table.

    Returns
    -------
    dict[PackageInstance, PackageChoice]
        Choice per release that has a compatible torchaudio anywhere.
    """
    choices: dict[PackageInstance, PackageChoice] = {}
    for release in releases:
        own = _newest_compatible_audio(release, served.get(release.index_url, {}), requirements, official_pairs)
        if own is not None:
            choices[release] = PackageChoice(version=own, index_url=release.index_url)
            continue
        elsewhere = [
            found
            for wheels in served.values()
            if (found := _newest_compatible_audio(release, wheels, requirements, official_pairs)) is not None
        ]
        if len(elsewhere) > 0:
            choices[release] = PackageChoice(version=max(elsewhere), index_url="")
    return choices


def _choose_torchvision(
    releases: list[PackageInstance],
    pairs: dict[Version, Version],
    served: dict[str, dict[_WheelKey, set[Version]]],
) -> dict[PackageInstance, PackageChoice]:
    """
    Decide where each release's paired torchvision comes from, if anywhere.

    torchvision's build matrix can trail torch's on a coordinate (torch 2.5.1
    ships CPython 3.13 wheels on cu121 while torchvision 0.20.1 stops at
    3.12), so the pairing alone does not make a requirement installable. A
    release whose own index serves the paired version for its python version
    and platform routes to that index; one served only elsewhere routes to
    the default registry; one served nowhere has no choice.

    Parameters
    ----------
    releases : list[PackageInstance]
        Torch releases the catalog covers.
    pairs : dict[Version, Version]
        torch-to-torchvision pairings.
    served : dict[str, dict[_WheelKey, set[Version]]]
        Per-index torchvision inventory from `_collect_package_inventory`.

    Returns
    -------
    dict[PackageInstance, PackageChoice]
        Choice per release that has a servable torchvision.
    """

    def serves(wheels: dict[_WheelKey, set[Version]], release: PackageInstance, version: Version) -> bool:
        return version in wheels.get((release.python_version, release.platform), set()) or version in wheels.get(
            (None, release.platform), set()
        )

    choices: dict[PackageInstance, PackageChoice] = {}
    for release in releases:
        paired = pairs.get(release.version)
        if paired is None:
            continue
        if serves(served.get(release.index_url, {}), release, paired):
            choices[release] = PackageChoice(version=paired, index_url=release.index_url)
        elif any(serves(wheels, release, paired) for wheels in served.values()):
            choices[release] = PackageChoice(version=paired, index_url="")
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


@dataclass(frozen=True)
class CacheStatus:
    """
    Whether a cached catalog can stand in for a fresh scrape, and why not.

    Parameters
    ----------
    path : Path
        Cache file examined.
    catalog : Catalog | None
        The cached catalog, None when the file is missing or unreadable.
    reasons : tuple[str, ...]
        Why the cache is stale, empty when it is fresh.
    cached : dict[str, Version | None]
        Newest cached version per package ('torch', 'torchvision',
        'torchaudio').
    online : dict[str, Version | None]
        Newest version per package on the main index, None when the probe
        failed.
    """

    path: Path
    catalog: Catalog | None
    reasons: tuple[str, ...]
    cached: dict[str, Version | None]
    online: dict[str, Version | None]

    @property
    def stale(self) -> bool:
        """Whether the cache needs a rescrape."""
        return len(self.reasons) > 0


_FRESHNESS_PACKAGES = ("torch", "torchvision", "torchaudio")


def _cached_versions(catalog: Catalog) -> dict[str, Version | None]:
    versions: dict[str, Version | None] = {}
    versions["torch"] = catalog.newest_version() if len(catalog.releases) > 0 else None
    versions["torchvision"] = max(catalog.torchvision_pairs.values()) if len(catalog.torchvision_pairs) > 0 else None
    versions["torchaudio"] = catalog.torchaudio_latest
    return versions


async def check_cache_async(
    cache_path: Path,
    cutoff_version: Version = DEFAULT_CUTOFF_VERSION,
    max_age_seconds: float = DEFAULT_MAX_CACHE_AGE_SECONDS,
    semaphore: asyncio.Semaphore | None = None,
) -> CacheStatus:
    """
    Decide whether a cached catalog is fresh, listing every reason it is not.

    The cache is stale when it is missing or unreadable, was written under an
    older schema, covers torch versions from above `cutoff_version`, is older
    than `max_age_seconds`, or trails the main index's newest torch,
    torchvision, or torchaudio. A failed version probe counts as stale, since
    freshness cannot be confirmed. The age bound covers changes the version
    probes are blind to, such as a sub-index gaining wheels of a version it
    already served.

    Parameters
    ----------
    cache_path : Path
        Cache file to examine.
    cutoff_version : Version, optional
        Minimum torch version the caller needs covered, by default 2.0.0.
    max_age_seconds : float, optional
        Maximum cache age, by default 24 hours.
    semaphore : asyncio.Semaphore, optional
        Concurrency limiter. A new one is created if not supplied.

    Returns
    -------
    CacheStatus
        The cached catalog, the version comparison, and the stale reasons.
    """
    from torch_wheel_index.serialization import load_catalog

    sem = semaphore or make_semaphore()
    online_versions = await asyncio.gather(*(_latest_at_main_index(name, sem) for name in _FRESHNESS_PACKAGES))
    online = dict(zip(_FRESHNESS_PACKAGES, online_versions, strict=True))
    empty: dict[str, Version | None] = dict.fromkeys(_FRESHNESS_PACKAGES)

    if not cache_path.exists():
        return CacheStatus(cache_path, None, ("no cache file",), empty, online)
    try:
        catalog = load_catalog(cache_path)
    except (OSError, ValueError) as exc:
        return CacheStatus(cache_path, None, (f"unreadable cache: {exc}",), empty, online)

    reasons: list[str] = []
    if catalog.schema_version < SCHEMA_VERSION:
        reasons.append(f"schema {catalog.schema_version} predates {SCHEMA_VERSION}")
    if catalog.cutoff_version is None or catalog.cutoff_version > cutoff_version:
        reasons.append(f"covers torch from {catalog.cutoff_version or 'an unknown cutoff'}, {cutoff_version} requested")
    written = catalog.generated_at if catalog.generated_at is not None else cache_path.stat().st_mtime
    age = time.time() - written
    if age > max_age_seconds:
        reasons.append(f"cache is {age:.0f}s old, limit {max_age_seconds:.0f}s")
    cached = _cached_versions(catalog)
    for name in _FRESHNESS_PACKAGES:
        latest = online[name]
        held = cached[name]
        if latest is None:
            reasons.append(f"{name} version probe failed")
        elif held is None or latest > held:
            reasons.append(f"{name} {latest} online, {held} cached")
    return CacheStatus(cache_path, catalog, tuple(reasons), cached, online)


def _narrow_to_cutoff(catalog: Catalog, cutoff_version: Version) -> Catalog:
    """
    Restrict a catalog to torch releases at or above a cutoff.

    Parameters
    ----------
    catalog : Catalog
        Catalog covering `cutoff_version` or a wider range.
    cutoff_version : Version
        Minimum torch version to keep.

    Returns
    -------
    Catalog
        The catalog itself when nothing falls below the cutoff, otherwise a
        copy holding only the releases at or above it.
    """
    if catalog.cutoff_version is not None and catalog.cutoff_version >= cutoff_version:
        return catalog
    kept = [release for release in catalog.releases if release.version >= cutoff_version]
    kept_set = set(kept)
    return replace(
        catalog,
        releases=kept,
        torchvision_choices={r: c for r, c in catalog.torchvision_choices.items() if r in kept_set},
        torchaudio_choices={r: c for r, c in catalog.torchaudio_choices.items() if r in kept_set},
        index_dependencies={r: d for r, d in catalog.index_dependencies.items() if r in kept_set},
        torchvision_pairs={v: p for v, p in catalog.torchvision_pairs.items() if v >= cutoff_version},
        torchaudio_pairs={v: p for v, p in catalog.torchaudio_pairs.items() if v >= cutoff_version},
        cutoff_version=cutoff_version,
    )


async def refresh_if_stale_async(
    cache_path: Path,
    cutoff_version: Version = DEFAULT_CUTOFF_VERSION,
    force: bool = False,
    max_age_seconds: float = DEFAULT_MAX_CACHE_AGE_SECONDS,
) -> Catalog:
    """
    Return a Catalog, refreshing the cache file only when needed.

    `check_cache_async` decides freshness. A fresh cache is returned,
    narrowed to `cutoff_version` when it covers a wider range; a stale one is
    replaced by a full scrape. A scrape that cannot complete raises
    `ScrapeError` and leaves the cache file untouched.

    Parameters
    ----------
    cache_path : Path
        File path of the cached catalog JSON.
    cutoff_version : Version, optional
        Minimum torch version to include, by default 2.0.0.
    force : bool, optional
        Refresh unconditionally, by default False.
    max_age_seconds : float, optional
        Rescrape a cache older than this, by default 24 hours.

    Returns
    -------
    Catalog
        Either the up-to-date cached catalog or a freshly fetched one.

    Raises
    ------
    ScrapeError
        A fetch the catalog depends on kept failing during the scrape.
    """
    from torch_wheel_index.serialization import save_catalog

    sem = make_semaphore()
    if not force:
        status = await check_cache_async(cache_path, cutoff_version, max_age_seconds, sem)
        if not status.stale and status.catalog is not None:
            logger.info("cache up-to-date (%s)", ", ".join(f"{k}={v}" for k, v in status.cached.items()))
            return _narrow_to_cutoff(status.catalog, cutoff_version)
        logger.info("cache stale: %s", "; ".join(status.reasons))

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
