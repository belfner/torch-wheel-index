from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from packaging.requirements import InvalidRequirement
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.utils import parse_wheel_filename
from packaging.version import Version

from torch_wheel_index._client import make_semaphore
from torch_wheel_index._parser import build_package_instance
from torch_wheel_index._parser import filter_wheels
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
    audio_coordinates, audio_versions, audio_samples = _collect_audio_inventory(sub_indexes, audio_listings)
    forward_ok = False
    forward_constraint: SpecifierSet | None = None
    if len(audio_versions) > 0:
        forward_ok, forward_constraint = await _audio_torch_constraint(audio_versions[0], audio_samples, sem)
    vision_pairs = _resolve_version_pairs(releases, vision_versions, official_vision_pairs)
    audio_pairs = _resolve_audio_pairs(
        releases,
        audio_versions,
        official_audio_pairs,
        forward_ok,
        forward_constraint,
    )

    logger.info("collected %d torch releases", len(releases))
    return Catalog(
        releases=releases,
        torchvision_pairs=vision_pairs,
        torchaudio_pairs=audio_pairs,
        torchaudio_index_wheels=audio_coordinates,
        torchaudio_latest=audio_versions[0] if len(audio_versions) > 0 else None,
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


def _collect_audio_inventory(
    sub_indexes: list[tuple[str, str]],
    listings: list[list[str]],
) -> tuple[dict[str, set[tuple[Version, Version, Platform]]], list[Version], dict[Version, tuple[str, str]]]:
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
    tuple[dict[str, set[tuple[Version, Version, Platform]]], list[Version], dict[Version, tuple[str, str]]]
        Per-index wheel coordinates, all versions seen newest first, and one
        (sub-index url, filename) sample per version for metadata lookups.
    """
    coordinates: dict[str, set[tuple[Version, Version, Platform]]] = {}
    versions: set[Version] = set()
    samples: dict[Version, tuple[str, str]] = {}
    cutoff = Version("0.0.0")
    for (url, key), filenames in zip(sub_indexes, listings, strict=True):
        index_url = f"{PYTORCH_INDEX_URL}/{key}"
        bucket = coordinates.setdefault(index_url, set())
        by_version: dict[Version, str] = {}
        for filename in filenames:
            try:
                _, filename_version, _, _ = parse_wheel_filename(filename)
            except Exception:
                continue
            by_version.setdefault(Version(filename_version.base_version), filename)
        for version, tags in filter_wheels(filenames, key, cutoff):
            instance = build_package_instance(version, key, tags, PYTORCH_INDEX_URL)
            bucket.add((instance.version, instance.python_version, instance.platform))
            versions.add(instance.version)
            if instance.version not in samples and instance.version in by_version:
                samples[instance.version] = (url, by_version[instance.version])
    return coordinates, sorted(versions, reverse=True), samples


def _torch_constraint(metadata: str) -> SpecifierSet | None:
    """
    Extract a wheel's declared torch version constraint.

    Parameters
    ----------
    metadata : str
        METADATA text from a PEP 658 sidecar.

    Returns
    -------
    SpecifierSet | None
        The union of every torch specifier the wheel declares, or None when it
        declares no torch requirement at all.
    """
    constraint: SpecifierSet | None = None
    for line in metadata.splitlines():
        if not line.lower().startswith("requires-dist:"):
            continue
        try:
            requirement = Requirement(line.split(":", 1)[1].strip())
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) != "torch":
            continue
        constraint = requirement.specifier if constraint is None else constraint & requirement.specifier
    return constraint


async def _audio_torch_constraint(
    latest_audio: Version,
    samples: dict[Version, tuple[str, str]],
    semaphore: asyncio.Semaphore,
) -> tuple[bool, SpecifierSet | None]:
    """
    Read what torch versions the newest torchaudio wheel accepts.

    torchaudio wheels up to 2.10 declare an exact torch requirement, which
    confines them to the torch release they shipped with. 2.11.0 declares no
    torch requirement at all, which is what lets it pair with later torch
    releases. The wheel's PEP 658 metadata sidecar carries the answer, and the
    declared specifier is evaluated per candidate torch version rather than
    pattern-matched, so a range or compatible-release form constrains pairing
    the same way an exact pin does.

    Parameters
    ----------
    latest_audio : Version
        Newest torchaudio version observed on the sub-indexes.
    samples : dict[Version, tuple[str, str]]
        One (sub-index url, wheel filename) per torchaudio version.
    semaphore : asyncio.Semaphore
        Concurrency limiter.

    Returns
    -------
    tuple[bool, SpecifierSet | None]
        Whether the metadata was readable, and the torch specifier it declares
        (None when it declares no torch requirement).
    """
    sample = samples.get(latest_audio)
    if sample is None:
        return False, None
    metadata = await fetch_wheel_metadata(sample[0], sample[1], semaphore)
    if len(metadata) == 0:
        logger.info("no metadata sidecar for torchaudio %s; skipping forward pairing", latest_audio)
        return False, None
    constraint = _torch_constraint(metadata)
    logger.info("torchaudio %s declares torch requirement: %s", latest_audio, constraint or "none")
    return True, constraint


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


def _resolve_audio_pairs(
    releases: list[PackageInstance],
    audio_versions: list[Version],
    official_pairs: dict[Version, Version],
    forward_ok: bool,
    forward_constraint: SpecifierSet | None,
) -> dict[Version, Version]:
    """
    Extend the official torchaudio pairings past the documented torch range.

    Torch versions the table lists keep its pairing, since torchaudio wheels
    of that era require an exact torch version and the pairing is sometimes
    offset (torch 2.0.1 goes with torchaudio 2.0.2). A version the table skips
    takes the torchaudio wheel sharing its version number when one exists,
    which is the pairing upstream ships for the same-version era. The newest
    torchaudio wheel supports the torch release it shipped alongside and every
    later one, so torch versions past the end of the table take it. A torch
    version left unmatched stays absent so callers can emit an unpinned
    torchaudio requirement. The same-version branch runs whenever wheels are
    available, so a failed scrape of the official table still yields pairings.

    The result says which torchaudio version goes with a torch version, not
    where to get it; `Catalog.torchaudio_index_url` decides that per release.

    Parameters
    ----------
    releases : list[PackageInstance]
        Torch releases the catalog covers.
    audio_versions : list[Version]
        torchaudio wheel versions found on the sub-indexes, newest first.
    official_pairs : dict[Version, Version]
        torch-to-torchaudio pairings scraped from the official table.
    forward_ok : bool
        Whether the newest torchaudio wheel's metadata was readable, which is
        what makes pairing it with undocumented torch versions safe.
    forward_constraint : SpecifierSet | None
        The torch specifier that wheel declares. Each candidate torch version
        is evaluated against it; None means the wheel constrains nothing.

    Returns
    -------
    dict[Version, Version]
        Pairings for every torch version whose torchaudio version is known.
    """
    pairs: dict[Version, Version] = dict(official_pairs)
    if len(audio_versions) == 0:
        return pairs

    available = set(audio_versions)
    newest_documented = max(official_pairs) if len(official_pairs) > 0 else None
    latest_audio = audio_versions[0]
    if len(official_pairs) > 0:
        latest_audio = max(latest_audio, max(official_pairs.values()))

    for release in releases:
        if release.version in pairs:
            continue
        if release.version in available:
            pairs[release.version] = release.version
            continue
        if not forward_ok or newest_documented is None or release.version <= newest_documented:
            continue
        if forward_constraint is None or forward_constraint.contains(release.version):
            pairs[release.version] = latest_audio

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
