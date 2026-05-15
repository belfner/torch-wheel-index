from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import TypeVar

from packaging.version import Version

T = TypeVar("T")


class Platform(enum.Enum):
    LINUX = 0
    WINDOWS = 1
    MACOS = 2


class ComputeType(enum.Enum):
    CPU = 0
    CUDA = 1
    ROCM = 2
    XPU = 3


@dataclass(frozen=True, eq=True)
class PackageInstance:
    """
    A single torch wheel: one (version, compute, python, platform) coordinate.

    Parameters
    ----------
    version : Version
        Torch release version (e.g., 2.5.0).
    compute_type : ComputeType
        Compute backend (CPU, CUDA, ROCM, XPU).
    compute_version : Version
        Backend version (e.g., 12.4 for CUDA, 6.1 for ROCM, 0.0.0 for CPU/XPU).
    python_version : Version
        CPython version (e.g., 3.12).
    platform : Platform
        Target OS.
    index_url : str
        PyTorch sub-index URL hosting this wheel.
    """

    version: Version
    compute_type: ComputeType
    compute_version: Version
    python_version: Version
    platform: Platform
    index_url: str


def _coerce_version(value: str | Version | None) -> Version | None:
    if value is None:
        return None
    if isinstance(value, Version):
        return value
    return Version(value)


def _coerce_compute_type(value: str | ComputeType | None) -> ComputeType | None:
    if value is None:
        return None
    if isinstance(value, ComputeType):
        return value
    return ComputeType[value.upper()]


def _coerce_platform(value: str | Platform | None) -> Platform | None:
    if value is None:
        return None
    if isinstance(value, Platform):
        return value
    return Platform[value.upper()]


@dataclass(frozen=True)
class Catalog:
    """
    Frozen catalog of available torch wheels at a point in time.

    Holds the full release list along with the torch-to-torchvision and
    torch-to-torchaudio version pairings. Provides query methods for filtering
    and listing unique values.

    Parameters
    ----------
    releases : list[PackageInstance]
        All known torch wheels. Sorted on construction by
        version desc, compute_type asc, compute_version desc,
        python_version desc, platform asc.
    torchvision_pairs : dict[Version, Version]
        Map of torch version to paired torchvision version.
    torchaudio_pairs : dict[Version, Version]
        Map of torch version to paired torchaudio version (from the official
        compatibility table). Wheels are not scraped per sub-index for
        torchaudio; this map is the only source.
    """

    releases: list[PackageInstance] = field(default_factory=list)
    torchvision_pairs: dict[Version, Version] = field(default_factory=dict)
    torchaudio_pairs: dict[Version, Version] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.releases.sort(key=lambda p: p.platform.name.lower())
        self.releases.sort(key=lambda p: p.python_version, reverse=True)
        self.releases.sort(key=lambda p: p.compute_version, reverse=True)
        self.releases.sort(key=lambda p: p.compute_type.name.lower())
        self.releases.sort(key=lambda p: p.version, reverse=True)

    def find(
        self,
        *,
        version: str | Version | None = None,
        compute_type: str | ComputeType | None = None,
        compute_version: str | Version | None = None,
        python_version: str | Version | None = None,
        platform: str | Platform | None = None,
    ) -> list[PackageInstance]:
        """
        Filter releases by any combination of coordinates.

        Parameters
        ----------
        version : str or Version, optional
            Torch version to match exactly.
        compute_type : str or ComputeType, optional
            Compute backend (case-insensitive name when str).
        compute_version : str or Version, optional
            Backend version to match exactly.
        python_version : str or Version, optional
            Python version to match exactly.
        platform : str or Platform, optional
            Platform (case-insensitive name when str).

        Returns
        -------
        list[PackageInstance]
            Matching releases in catalog order.
        """
        v = _coerce_version(version)
        ct = _coerce_compute_type(compute_type)
        cv = _coerce_version(compute_version)
        pv = _coerce_version(python_version)
        plat = _coerce_platform(platform)

        results = []
        for release in self.releases:
            if v is not None and release.version != v:
                continue
            if ct is not None and release.compute_type != ct:
                continue
            if cv is not None and release.compute_version != cv:
                continue
            if pv is not None and release.python_version != pv:
                continue
            if plat is not None and release.platform != plat:
                continue
            results.append(release)
        return results

    def newest_version(self) -> Version:
        """
        Return the newest torch version in the catalog.

        Returns
        -------
        Version
            Highest torch version found.

        Raises
        ------
        ValueError
            When the catalog is empty.
        """
        if len(self.releases) == 0:
            raise ValueError("catalog is empty")
        return max(release.version for release in self.releases)

    def newest_torchvision_version(self) -> Version:
        """
        Return the newest torchvision version known to the pairing map.

        Returns
        -------
        Version
            Highest paired torchvision version.

        Raises
        ------
        ValueError
            When no torchvision pairings are present.
        """
        if len(self.torchvision_pairs) == 0:
            raise ValueError("no torchvision pairings")
        return max(self.torchvision_pairs.values())

    def newest_torchaudio_version(self) -> Version:
        """
        Return the newest torchaudio version known to the pairing map.

        Returns
        -------
        Version
            Highest paired torchaudio version.

        Raises
        ------
        ValueError
            When no torchaudio pairings are present.
        """
        if len(self.torchaudio_pairs) == 0:
            raise ValueError("no torchaudio pairings")
        return max(self.torchaudio_pairs.values())

    def torchvision_for(self, torch_version: str | Version) -> Version | None:
        """
        Look up the paired torchvision version for a given torch version.

        Parameters
        ----------
        torch_version : str or Version
            Torch version to look up.

        Returns
        -------
        Version or None
            Paired torchvision version, or None if not paired.
        """
        v = _coerce_version(torch_version)
        if v is None:
            return None
        return self.torchvision_pairs.get(v)

    def torchaudio_for(self, torch_version: str | Version) -> Version | None:
        """
        Look up the paired torchaudio version for a given torch version.

        Parameters
        ----------
        torch_version : str or Version
            Torch version to look up.

        Returns
        -------
        Version or None
            Paired torchaudio version, or None if the official compatibility
            table does not list torchaudio for this torch version.
        """
        v = _coerce_version(torch_version)
        if v is None:
            return None
        return self.torchaudio_pairs.get(v)

    def versions(self) -> list[Version]:
        """Unique torch versions in catalog order (newest first)."""
        return _stable_unique(release.version for release in self.releases)

    def compute_types(self) -> list[ComputeType]:
        """Unique compute backends present, in catalog order."""
        return _stable_unique(release.compute_type for release in self.releases)

    def compute_versions(self, compute_type: str | ComputeType | None = None) -> list[Version]:
        """
        Unique compute-backend versions, optionally filtered by type.

        Parameters
        ----------
        compute_type : str or ComputeType, optional
            Restrict to a single backend when given.

        Returns
        -------
        list[Version]
            Unique versions in catalog order.
        """
        ct = _coerce_compute_type(compute_type)
        if ct is None:
            return _stable_unique(release.compute_version for release in self.releases)
        return _stable_unique(release.compute_version for release in self.releases if release.compute_type == ct)

    def compute_versions_by_type(self) -> dict[ComputeType, list[Version]]:
        """
        Map of compute type to its unique versions, in catalog order.

        Returns
        -------
        dict[ComputeType, list[Version]]
            Per-backend version lists.
        """
        result: dict[ComputeType, list[Version]] = {}
        for release in self.releases:
            bucket = result.setdefault(release.compute_type, [])
            if release.compute_version not in bucket:
                bucket.append(release.compute_version)
        return result

    def python_versions(self) -> list[Version]:
        """Unique Python versions in catalog order."""
        return _stable_unique(release.python_version for release in self.releases)

    def platforms(self) -> list[Platform]:
        """Unique platforms in catalog order."""
        return _stable_unique(release.platform for release in self.releases)


def _stable_unique(values: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    unique: list[T] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique
