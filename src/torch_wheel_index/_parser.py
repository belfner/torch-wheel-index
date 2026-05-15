from __future__ import annotations

from packaging.tags import Tag
from packaging.utils import parse_wheel_filename
from packaging.version import Version

from torch_wheel_index.models import ComputeType
from torch_wheel_index.models import PackageInstance
from torch_wheel_index.models import Platform


def parse_compute_version(sub_index_key: str) -> tuple[ComputeType, Version]:
    """
    Parse compute type and version from a sub-index key.

    Parameters
    ----------
    sub_index_key : str
        Sub-index key (e.g., 'cu118', 'rocm6.1', 'cpu', 'xpu', 'whl').

    Returns
    -------
    tuple[ComputeType, Version]
        Parsed compute type enum and version.

    Raises
    ------
    ValueError
        If the format is unknown.
    """
    key = sub_index_key.lower()
    if key.startswith("cpu"):
        return ComputeType.CPU, Version("0.0.0")
    if key.startswith("cu"):
        num = key.replace("cu", "")
        return ComputeType.CUDA, Version(f"{num[:-1]}.{num[-1]}")
    if key.startswith("rocm"):
        return ComputeType.ROCM, Version(key.replace("rocm", ""))
    if key.startswith("xpu"):
        return ComputeType.XPU, Version("0.0.0")
    if key == "whl":
        return ComputeType.CPU, Version("0.0.0")
    raise ValueError(f"Unknown compute version: {sub_index_key}")


def parse_platform(platform_tag: str) -> Platform:
    """
    Parse target platform from a wheel platform tag.

    Parameters
    ----------
    platform_tag : str
        Platform tag from a wheel filename.

    Returns
    -------
    Platform
        Parsed platform enum.

    Raises
    ------
    ValueError
        If the platform tag is unknown.
    """
    if "win" in platform_tag:
        return Platform.WINDOWS
    if "linux" in platform_tag:
        return Platform.LINUX
    if "macos" in platform_tag:
        return Platform.MACOS
    raise ValueError(f"Unknown platform: {platform_tag}")


def cpython_tag_to_version(cpython_tag: str) -> Version:
    """
    Convert a CPython interpreter tag to a Version.

    Parameters
    ----------
    cpython_tag : str
        CPython tag (e.g., 'cp310', 'cp311').

    Returns
    -------
    Version
        Version with major.minor (e.g., 3.10).
    """
    num = cpython_tag.replace("cp", "")
    return Version(f"{num[0]}.{num[1:]}")


def filter_wheels(
    wheel_filenames: list[str],
    sub_index_key: str,
    cutoff_version: Version,
) -> list[tuple[Version, frozenset[Tag]]]:
    """
    Select wheels for a sub-index, applying cutoff and local-tag matching.

    Wheels with a local version segment must match `sub_index_key`. Wheels
    without a local segment (the macOS case) are matched by 'macosx' in the
    platform tag.

    Parameters
    ----------
    wheel_filenames : list[str]
        Wheel filenames returned from the index.
    sub_index_key : str
        Sub-index key (e.g., 'cu118').
    cutoff_version : Version
        Minimum version to include.

    Returns
    -------
    list[tuple[Version, frozenset[Tag]]]
        Filtered (version, tags) pairs.
    """
    selected: list[tuple[Version, frozenset[Tag]]] = []
    for filename in wheel_filenames:
        _, version, _, tags = parse_wheel_filename(filename)
        if version < cutoff_version:
            continue
        if version.local is not None:
            if version.local == sub_index_key:
                selected.append((version, tags))
        elif "macosx" in next(iter(tags)).platform:
            selected.append((version, tags))
    return selected


def build_package_instance(
    version: Version,
    sub_index_key: str,
    tags: frozenset[Tag],
    pytorch_index_url: str,
) -> PackageInstance:
    """
    Build a PackageInstance from raw wheel metadata.

    Parameters
    ----------
    version : Version
        Wheel version (with optional local segment).
    sub_index_key : str
        Sub-index key (e.g., 'cu118').
    tags : frozenset[Tag]
        Wheel tags.
    pytorch_index_url : str
        Base URL of the PyTorch wheel index.

    Returns
    -------
    PackageInstance
        Constructed instance with base version, parsed compute, python, and platform.
    """
    tag = next(iter(tags))
    python_version = cpython_tag_to_version(tag.interpreter)
    compute_type, compute_version = parse_compute_version(sub_index_key)
    platform = parse_platform(tag.platform)
    return PackageInstance(
        version=Version(version.base_version),
        compute_type=compute_type,
        compute_version=compute_version,
        python_version=python_version,
        platform=platform,
        index_url=f"{pytorch_index_url}/{sub_index_key}",
    )
