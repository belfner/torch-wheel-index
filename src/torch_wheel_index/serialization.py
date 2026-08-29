from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from typing import TypeVar

from packaging.version import Version

from torch_wheel_index.models import Catalog
from torch_wheel_index.models import ComputeType
from torch_wheel_index.models import PackageInstance
from torch_wheel_index.models import SCHEMA_VERSION
from torch_wheel_index.models import Platform

T = TypeVar("T")


def catalog_to_dict(catalog: Catalog) -> dict[str, Any]:
    """
    Render a Catalog to the `pytorch_info.json` schema.

    Output has two top-level keys:
      - 'schema_version': integer schema version, and 'torchaudio_latest':
        the newest torchaudio version seen on the sub-indexes.
      - 'all_releases': per-release dicts with torchvision_version,
        torchaudio_version, and torchaudio_index_url attached.
        torchaudio_version is an empty string when the pairing is unknown or
        when no index carries that torchaudio version for the release's python
        version and platform. torchaudio_index_url holds the release's
        index URL when that index serves the paired torchaudio version, and an
        empty string when torchaudio comes from the default registry.
      - 'unique_values': dropdown-friendly distinct values, with compute_version
        nested under compute_type.

    Parameters
    ----------
    catalog : Catalog
        Catalog to serialize.

    Returns
    -------
    dict[str, Any]
        Schema-conformant dict ready for json.dump.
    """
    all_releases: list[dict[str, str]] = []
    fallback_vision = ""
    if len(catalog.torchvision_pairs) > 0:
        fallback_vision = str(max(catalog.torchvision_pairs.values()))

    for release in catalog.releases:
        paired_vision = catalog.torchvision_pairs.get(release.version)
        torchvision_str = str(paired_vision) if paired_vision is not None else fallback_vision
        paired_audio = catalog.torchaudio_pairs.get(release.version)
        torchaudio_str = ""
        if paired_audio is not None and catalog.torchaudio_available(release):
            torchaudio_str = str(paired_audio)
        all_releases.append(
            {
                "version": str(release.version),
                "compute_type": release.compute_type.name,
                "compute_version": str(release.compute_version),
                "python_version": str(release.python_version),
                "platform": release.platform.name,
                "index_url": release.index_url,
                "torchvision_version": torchvision_str,
                "torchaudio_version": torchaudio_str,
                "torchaudio_index_url": catalog.torchaudio_index_url(release),
            }
        )

    compute_versions: dict[str, list[str]] = {}
    for release in catalog.releases:
        bucket = compute_versions.setdefault(release.compute_type.name, [])
        cv = str(release.compute_version)
        if cv not in bucket:
            bucket.append(cv)

    return {
        "schema_version": SCHEMA_VERSION,
        "torchaudio_latest": str(catalog.torchaudio_latest) if catalog.torchaudio_latest is not None else "",
        "all_releases": all_releases,
        "unique_values": {
            "compute_type": _stable_unique(r.compute_type.name for r in catalog.releases),
            "python_version": _stable_unique(str(r.python_version) for r in catalog.releases),
            "platform": _stable_unique(r.platform.name for r in catalog.releases),
            "version": _stable_unique(str(r.version) for r in catalog.releases),
            "compute_version": compute_versions,
        },
    }


def catalog_from_dict(data: dict[str, Any]) -> Catalog:
    """
    Construct a Catalog from a `pytorch_info.json`-shaped dict.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed JSON content.

    Returns
    -------
    Catalog
        Reconstructed catalog (releases re-sorted on construction).

    Raises
    ------
    ValueError
        When the input is missing required keys.
    """
    if "all_releases" not in data:
        raise ValueError("missing 'all_releases' key")

    releases: list[PackageInstance] = []
    vision_pairs: dict[Version, Version] = {}
    audio_pairs: dict[Version, Version] = {}
    audio_wheels: dict[str, set[tuple[Version, Version, Platform]]] = {}

    for entry in data["all_releases"]:
        version = Version(entry["version"])
        instance = PackageInstance(
            version=version,
            compute_type=ComputeType[entry["compute_type"]],
            compute_version=Version(entry["compute_version"]),
            python_version=Version(entry["python_version"]),
            platform=Platform[entry["platform"]],
            index_url=entry["index_url"],
        )
        releases.append(instance)

        torchvision_str = entry.get("torchvision_version")
        if torchvision_str:
            vision_pairs[version] = Version(torchvision_str)
        torchaudio_str = entry.get("torchaudio_version")
        if torchaudio_str:
            paired_audio = Version(torchaudio_str)
            audio_pairs[version] = paired_audio
            audio_index_url = entry.get("torchaudio_index_url")
            if audio_index_url:
                coordinate = (paired_audio, instance.python_version, instance.platform)
                audio_wheels.setdefault(audio_index_url, set()).add(coordinate)

    latest_str = data.get("torchaudio_latest")
    torchaudio_latest = Version(latest_str) if latest_str else None
    if torchaudio_latest is None and len(audio_pairs) > 0:
        torchaudio_latest = max(audio_pairs.values())

    return Catalog(
        releases=releases,
        torchvision_pairs=vision_pairs,
        torchaudio_pairs=audio_pairs,
        torchaudio_index_wheels=audio_wheels,
        torchaudio_latest=torchaudio_latest,
        schema_version=int(data.get("schema_version", 0)),
    )


def save_catalog(catalog: Catalog, path: Path) -> None:
    """
    Atomically write a Catalog to `path` as JSON.

    Writes to a temp file in the same directory, then `os.replace` to swap
    into place so concurrent readers never see a half-written file.

    Parameters
    ----------
    catalog : Catalog
        Catalog to write.
    path : Path
        Destination file path. Parent directory is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = catalog_to_dict(catalog)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_catalog(path: Path) -> Catalog:
    """
    Read a catalog JSON file and reconstruct a Catalog.

    Parameters
    ----------
    path : Path
        Source file path.

    Returns
    -------
    Catalog
        Reconstructed catalog.

    Raises
    ------
    FileNotFoundError
        When the path does not exist.
    ValueError
        When the file content is invalid.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return catalog_from_dict(data)


def _stable_unique(values: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    out: list[T] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
