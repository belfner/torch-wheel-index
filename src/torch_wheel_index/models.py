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


SCHEMA_VERSION = 5
"""Current version of the `pytorch_info.json` schema."""


@dataclass(frozen=True, eq=True)
class IndexDependency:
    """
    A dependency of a torch release that only the release's own index serves.

    torch wheels on some sub-indexes depend on packages the default registry
    lacks at the required version (torch+xpu on `triton-xpu`, torch+rocm10.0
    on `rocm` and its `rocm-sdk-*` packages). An explicit index serves only
    the packages a project routes to it, and uv routes direct dependencies
    only, so each of these becomes a direct requirement routed to the index.

    Parameters
    ----------
    name : str
        Normalized project name.
    requirement : str
        PEP 508 requirement without its marker (e.g., 'triton-xpu~=3.9.0').
    marker : str
        PEP 508 marker the requirement carries, with single-quoted values, or
        an empty string when it applies unconditionally.
    """

    name: str
    requirement: str
    marker: str


@dataclass(frozen=True, eq=True)
class PackageChoice:
    """
    A companion package version (torchvision, torchaudio) for one torch release.

    Parameters
    ----------
    version : Version
        Version to pin.
    index_url : str
        Sub-index URL serving that version for the release's python version
        and platform, which is always the release's own index. An empty
        string means the release's index lacks a matching wheel and the
        default registry supplies it.
    """

    version: Version
    index_url: str


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
    torchvision_choices : dict[PackageInstance, PackageChoice]
        Map of release to its paired torchvision version and the index
        serving a wheel of it for the release's python version and platform.
        Releases with no such wheel on any index are absent (torchvision
        0.20.1 ships no CPython 3.13 wheel while torch 2.5.1 does).
    torchaudio_pairs : dict[Version, Version]
        Map of torch version to the newest torchaudio version chosen for any
        of its releases. A summary for version lookups; per-release answers
        live in `torchaudio_choices`.
    torchaudio_choices : dict[PackageInstance, PackageChoice]
        Map of release to the torchaudio version that installs alongside it
        and the index serving that version. Sub-indexes carry different
        torchaudio versions and build matrices (cu130 serves 2.11.0 wheels per
        CPython version while cu132 serves only the 2.11.1 `py3-none` wheel),
        so the choice is made per release rather than per torch version.
    index_dependencies : dict[PackageInstance, tuple[IndexDependency, ...]]
        Map of release to the dependencies only its own index serves,
        including transitive ones, read from wheel and sdist metadata.
        Releases whose dependencies all resolve from the default registry are
        absent.
    torchaudio_latest : Version | None
        Newest torchaudio version observed across the sub-indexes, which is
        what freshness checks compare against.
    cutoff_version : Version | None
        Minimum torch version the scrape covered, None when unknown.
    generated_at : float | None
        Unix time the scrape finished, None when unknown. Freshness checks
        measure age from it, since copying or checking out a cache file
        resets its modification time.
    schema_version : int
        Version of the JSON schema this catalog was loaded from. A freshly
        scraped catalog carries `SCHEMA_VERSION`.
    """

    releases: list[PackageInstance] = field(default_factory=list)
    torchvision_pairs: dict[Version, Version] = field(default_factory=dict)
    torchvision_choices: dict[PackageInstance, PackageChoice] = field(default_factory=dict)
    torchaudio_pairs: dict[Version, Version] = field(default_factory=dict)
    torchaudio_choices: dict[PackageInstance, PackageChoice] = field(default_factory=dict)
    index_dependencies: dict[PackageInstance, tuple[IndexDependency, ...]] = field(default_factory=dict)
    torchaudio_latest: Version | None = None
    cutoff_version: Version | None = None
    generated_at: float | None = None
    schema_version: int = SCHEMA_VERSION

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
        Return the newest torchaudio version chosen for any release.

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

    def torchvision_choice(self, release: PackageInstance) -> PackageChoice | None:
        """
        Return the torchvision requirement that installs alongside a release.

        Parameters
        ----------
        release : PackageInstance
            Release whose torchvision requirement is being looked up.

        Returns
        -------
        PackageChoice or None
            The paired torchvision version and the index serving it, or None
            when no scraped index carries a wheel of it for the release's
            python version and platform.
        """
        return self.torchvision_choices.get(release)

    def torchaudio_choice(self, release: PackageInstance) -> PackageChoice | None:
        """
        Return the torchaudio requirement that installs alongside a release.

        Parameters
        ----------
        release : PackageInstance
            Release whose torchaudio requirement is being looked up.

        Returns
        -------
        PackageChoice or None
            The torchaudio version and the index serving it, or None when no
            scraped index carries a torchaudio wheel that accepts this torch
            version on the release's python version and platform.
        """
        return self.torchaudio_choices.get(release)

    def torchaudio_available(self, release: PackageInstance) -> bool:
        """
        Report whether some index serves a compatible torchaudio for a release.

        Parameters
        ----------
        release : PackageInstance
            Release whose torchaudio availability is being decided.

        Returns
        -------
        bool
            True when a torchaudio choice exists for this release.
        """
        return release in self.torchaudio_choices

    def torchaudio_index_url(self, release: PackageInstance) -> str:
        """
        Return the index URL a release sources its torchaudio from.

        Parameters
        ----------
        release : PackageInstance
            Release whose torchaudio source is being decided.

        Returns
        -------
        str
            The release's own index URL when that index carries the chosen
            torchaudio wheel, otherwise an empty string, which leaves
            torchaudio to the default registry.
        """
        choice = self.torchaudio_choices.get(release)
        if choice is None:
            return ""
        return choice.index_url

    def torchaudio_for(self, torch_version: str | Version) -> Version | None:
        """
        Look up the newest torchaudio version chosen for a torch version.

        Sub-indexes can serve different torchaudio versions for the same torch
        release; `torchaudio_choice` gives the per-release answer.

        Parameters
        ----------
        torch_version : str or Version
            Torch version to look up.

        Returns
        -------
        Version or None
            Newest torchaudio version chosen for any release of this torch
            version, or None when none of its releases has a torchaudio choice.
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
