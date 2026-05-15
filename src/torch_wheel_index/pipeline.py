from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from packaging.utils import parse_wheel_filename
from packaging.version import Version

from torch_wheel_index._client import make_semaphore
from torch_wheel_index._parser import build_package_instance
from torch_wheel_index._parser import filter_wheels
from torch_wheel_index._scraper import PYTORCH_INDEX_URL
from torch_wheel_index._scraper import fetch_package_wheels
from torch_wheel_index._scraper import fetch_wheel_filenames
from torch_wheel_index._scraper import get_paired_versions
from torch_wheel_index._scraper import get_torch_sub_indexes
from torch_wheel_index.models import Catalog
from torch_wheel_index.models import PackageInstance
from torch_wheel_index.paths import default_cache_path

logger = logging.getLogger(__name__)

DEFAULT_CUTOFF_VERSION = Version("2.0.0")


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

    logger.info("fetching torch and torchvision wheel listings")
    torch_tasks = [fetch_package_wheels(url, "torch", sem) for url, _ in sub_indexes]
    vision_tasks = [fetch_package_wheels(url, "torchvision", sem) for url, _ in sub_indexes]
    pairs_task = get_paired_versions(sem, cutoff_version)

    torch_listings, vision_listings, official_pairs = await asyncio.gather(
        asyncio.gather(*torch_tasks),
        asyncio.gather(*vision_tasks),
        pairs_task,
    )
    official_vision_pairs, audio_pairs = official_pairs

    releases = _build_releases(sub_indexes, torch_listings, cutoff_version)
    vision_versions = _collect_vision_versions(sub_indexes, vision_listings)
    vision_pairs = _resolve_version_pairs(releases, vision_versions, official_vision_pairs)

    logger.info("collected %d torch releases", len(releases))
    return Catalog(releases=releases, torchvision_pairs=vision_pairs, torchaudio_pairs=audio_pairs)


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


def _collect_vision_versions(
    sub_indexes: list[tuple[str, str]],
    vision_listings: list[list[str]],
) -> list[Version]:
    versions: set[Version] = set()
    cutoff = Version("0.0.0")
    for (_, key), filenames in zip(sub_indexes, vision_listings, strict=True):
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
) -> Catalog:
    """
    Return a Catalog, refreshing the cache file only when needed.

    Reads `cache_path` and compares the latest torch and torchvision versions
    against what the PyTorch index serves. If either is newer (or the cache
    is missing or `force` is set), runs a full scrape and writes the cache.

    Parameters
    ----------
    cache_path : Path
        File path of the cached catalog JSON.
    cutoff_version : Version, optional
        Minimum torch version to include when refreshing, by default 2.0.0.
    force : bool, optional
        Refresh unconditionally, by default False.

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

    latest_torch, latest_vision = await asyncio.gather(
        _latest_at_main_index("torch", sem),
        _latest_at_main_index("torchvision", sem),
    )

    if cached is not None and latest_torch is not None and latest_vision is not None and not force:
        try:
            cached_torch = cached.newest_version()
        except ValueError:
            cached_torch = Version("0.0.0")
        try:
            cached_vision = cached.newest_torchvision_version()
        except ValueError:
            cached_vision = Version("0.0.0")

        if latest_torch <= cached_torch and latest_vision <= cached_vision:
            logger.info(
                "cache up-to-date (torch=%s, torchvision=%s)",
                cached_torch,
                cached_vision,
            )
            return cached

        logger.info(
            "cache stale: online torch=%s vision=%s, cached torch=%s vision=%s",
            latest_torch,
            latest_vision,
            cached_torch,
            cached_vision,
        )

    catalog = await fetch_catalog_async(cutoff_version, semaphore=sem)
    save_catalog(catalog, cache_path)
    return catalog


def refresh_if_stale(
    cache_path: Path,
    cutoff_version: Version = DEFAULT_CUTOFF_VERSION,
    force: bool = False,
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

    Returns
    -------
    Catalog
        Cached or freshly fetched catalog.
    """
    return asyncio.run(refresh_if_stale_async(cache_path, cutoff_version, force))


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
