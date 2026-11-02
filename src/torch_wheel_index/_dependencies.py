from __future__ import annotations

import asyncio
import html
import io
import logging
import re
import tarfile
import zipfile
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from email.parser import HeaderParser
from urllib.parse import urldefrag
from urllib.parse import urljoin

from packaging.markers import InvalidMarker
from packaging.markers import Marker
from packaging.requirements import InvalidRequirement
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag
from packaging.utils import InvalidSdistFilename
from packaging.utils import InvalidWheelFilename
from packaging.utils import canonicalize_name
from packaging.utils import parse_sdist_filename
from packaging.utils import parse_wheel_filename
from packaging.version import InvalidVersion
from packaging.version import Version

from torch_wheel_index._client import MISSING_STATUSES
from torch_wheel_index._client import ScrapeError
from torch_wheel_index._client import fetch
from torch_wheel_index._client import is_transient
from torch_wheel_index._parser import parse_platform
from torch_wheel_index._scraper import fetch_wheel_metadata_at
from torch_wheel_index.models import IndexDependency
from torch_wheel_index.models import Platform

logger = logging.getLogger(__name__)

PYPI_SIMPLE_URL = "https://pypi.org/simple"
"""Default registry uv resolves from when a package has no source."""

MAX_SDIST_BYTES = 5 * 1024 * 1024
"""Largest sdist downloaded to read its PKG-INFO."""

_ANCHOR_RE = re.compile(r"<a\s([^>]*)>([^<]+)</a>", re.IGNORECASE)
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_REQUIRES_PYTHON_RE = re.compile(r"""data-requires-python\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
_EXTRA_COMPARISON_RE = re.compile(
    r"""extra\s*(==|!=)\s*["']([^"']+)["']|["']([^"']+)["']\s*(==|!=)\s*extra""",
)
_MARKER_TRUE = "python_version >= '0'"
"""A clause every environment satisfies, standing in for a resolved extra comparison."""
_MARKER_FALSE = "python_version < '0'"
"""A clause no environment satisfies, standing in for a resolved extra comparison."""


@dataclass(frozen=True)
class _DistFile:
    """
    One file in a simple-index project listing.

    Parameters
    ----------
    filename : str
        File name as listed.
    url : str
        Absolute file URL without its hash fragment.
    version : Version
        Version parsed from the filename.
    is_wheel : bool
        True for a wheel, False for an sdist.
    platforms : frozenset[Platform]
        Platforms a wheel targets. Empty for sdists and for platform-independent
        wheels, which install everywhere.
    tags : frozenset[Tag]
        A wheel's compatibility tags; empty for sdists.
    requires_python : SpecifierSet | None
        The listing's `data-requires-python` value, None when absent.
    """

    filename: str
    url: str
    version: Version
    is_wheel: bool
    platforms: frozenset[Platform]
    tags: frozenset[Tag]
    requires_python: SpecifierSet | None

    def installs_on(self, python: Version, platform: Platform) -> bool:
        """
        Report whether the file installs on a python version and platform.

        Parameters
        ----------
        python : Version
            CPython major.minor version.
        platform : Platform
            Target platform.

        Returns
        -------
        bool
            True when Requires-Python admits `python` and, for a wheel, its
            tags cover the platform and the interpreter (a `py3` tag, the
            matching `cpXY` tag, or an `abi3` tag from an older CPython). An
            sdist installs wherever Requires-Python allows.
        """
        if self.requires_python is not None and not self.requires_python.contains(python, prereleases=True):
            return False
        if not self.is_wheel:
            return True
        if len(self.platforms) > 0 and platform not in self.platforms:
            return False
        own = f"cp{python.major}{python.minor}"
        for tag in self.tags:
            if tag.interpreter in ("py3", f"py{python.major}{python.minor}", own):
                return True
            if tag.abi == "abi3" and tag.interpreter.startswith("cp3"):
                try:
                    if Version(f"3.{tag.interpreter[3:]}") <= python:
                        return True
                except InvalidVersion:
                    continue
        return False


def _tag_platforms(tags: Iterable[Tag]) -> frozenset[Platform]:
    platforms: set[Platform] = set()
    for tag in tags:
        if tag.platform == "any":
            return frozenset()
        try:
            platforms.add(parse_platform(tag.platform))
        except ValueError:
            continue
    return frozenset(platforms)


def _parse_listing(listing_url: str, text: str) -> list[_DistFile]:
    """
    Parse a PEP 503 simple-index project page into its distribution files.

    Parameters
    ----------
    listing_url : str
        URL of the page, which relative links resolve against.
    text : str
        Page HTML.

    Returns
    -------
    list[_DistFile]
        Wheels and sdists with parseable filenames, in page order.
    """
    files: list[_DistFile] = []
    for attributes, label in _ANCHOR_RE.findall(text):
        href = _HREF_RE.search(attributes)
        if href is None:
            continue
        filename = html.unescape(label.strip())
        url, _ = urldefrag(urljoin(listing_url, html.unescape(href.group(1))))
        requires_python: SpecifierSet | None = None
        declared = _REQUIRES_PYTHON_RE.search(attributes)
        if declared is not None and declared.group(1).strip() != "":
            try:
                requires_python = SpecifierSet(html.unescape(declared.group(1)))
            except InvalidSpecifier:
                requires_python = None
        try:
            if filename.endswith(".whl"):
                _, version, _, tags = parse_wheel_filename(filename)
                files.append(_DistFile(filename, url, version, True, _tag_platforms(tags), tags, requires_python))
            elif filename.endswith((".tar.gz", ".zip")):
                _, version = parse_sdist_filename(filename)
                files.append(_DistFile(filename, url, version, False, frozenset(), frozenset(), requires_python))
        except (InvalidWheelFilename, InvalidSdistFilename, InvalidVersion):
            continue
    return files


def _sdist_pkg_info(filename: str, content: bytes) -> str | None:
    """
    Read the top-level PKG-INFO from an sdist archive held in memory.

    Parameters
    ----------
    filename : str
        Archive filename, which selects zip or gzipped tar handling.
    content : bytes
        Archive bytes.

    Returns
    -------
    str | None
        PKG-INFO text, or None when the archive has none or cannot be read.
    """
    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = [n for n in archive.namelist() if n.count("/") == 1 and n.endswith("/PKG-INFO")]
                return archive.read(names[0]).decode("utf-8", errors="replace") if len(names) > 0 else None
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.count("/") == 1 and member.name.endswith("/PKG-INFO"):
                    handle = archive.extractfile(member)
                    return handle.read().decode("utf-8", errors="replace") if handle is not None else None
    except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError):
        return None
    return None


def _requires_dist(metadata: str, from_sdist: bool) -> list[Requirement] | None:
    """
    Parse the Requires-Dist entries of a core metadata document.

    An sdist's PKG-INFO is authoritative from Metadata-Version 2.2 on when
    Requires-Dist is absent from its Dynamic fields. A dynamic list (the
    `rocm` meta-package computes its `rocm-sdk-*` dependencies in setup.py)
    still records what the sdist's own build produced, which is the best
    evidence available short of building it, so it is used and logged.
    Metadata older than 2.2 that lists nothing is treated as unreadable,
    since such sdists routinely leave Requires-Dist out.

    Parameters
    ----------
    metadata : str
        METADATA or PKG-INFO text.
    from_sdist : bool
        True when the text is an sdist's PKG-INFO.

    Returns
    -------
    list[Requirement] | None
        The declared requirements, or None when the document cannot vouch
        for them.
    """
    message = HeaderParser().parsestr(metadata)
    values = message.get_all("Requires-Dist") or []
    if from_sdist:
        try:
            metadata_version = Version(message.get("Metadata-Version", "0"))
        except InvalidVersion:
            return None
        dynamic = {canonicalize_name(value) for value in message.get_all("Dynamic") or []}
        if metadata_version < Version("2.2") and len(values) == 0:
            return None
        if "requires-dist" in dynamic:
            logger.info("%s declares Requires-Dist dynamically; using its recorded list", message.get("Name"))
    requirements: list[Requirement] = []
    for value in values:
        try:
            requirements.append(Requirement(value))
        except InvalidRequirement:
            logger.debug("skipping unparseable requirement %r", value)
    return requirements


def resolve_extras(marker: Marker | None, extras: frozenset[str]) -> tuple[bool, Marker | None]:
    """
    Settle a dependency marker's `extra` comparisons for a set of extras.

    Each `extra == ...` or `extra != ...` comparison is replaced by a clause
    that is always true or always false, which keeps every environment
    condition and the Boolean grouping intact. A marker left with no
    environment condition is evaluated outright.

    Parameters
    ----------
    marker : Marker | None
        The dependency's marker.
    extras : frozenset[str]
        Normalized extras requested on the parent requirement; empty for a
        requirement nothing requested extras of.

    Returns
    -------
    tuple[bool, Marker | None]
        Whether the dependency can apply, and the marker it carries forward
        (None when it applies unconditionally).
    """
    if marker is None:
        return True, None
    text = str(marker)
    if _EXTRA_COMPARISON_RE.search(text) is None:
        return True, marker

    def settle(match: re.Match[str]) -> str:
        operator = match.group(1) or match.group(4)
        named = canonicalize_name(match.group(2) or match.group(3))
        holds = (named in extras) == (operator == "==")
        return _MARKER_TRUE if holds else _MARKER_FALSE

    settled = _EXTRA_COMPARISON_RE.sub(settle, text)
    remainder = settled.replace(_MARKER_TRUE, "").replace(_MARKER_FALSE, "")
    try:
        resolved = Marker(settled)
    except InvalidMarker:
        logger.info("cannot settle extras in marker %r; treating the dependency as inapplicable", text)
        return False, None
    if re.search(r"[a-z_]+\s*(==|!=|<|>|~=|in|not in)", remainder) is None:
        # Only constant clauses remain, so any environment gives the answer.
        return resolved.evaluate(), None
    return True, resolved


_PLATFORM_ENVIRONMENTS: dict[Platform, tuple[dict[str, str], ...]] = {
    Platform.LINUX: tuple(
        {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix", "platform_machine": machine}
        for machine in ("x86_64", "aarch64")
    ),
    Platform.WINDOWS: tuple(
        {"sys_platform": "win32", "platform_system": "Windows", "os_name": "nt", "platform_machine": machine}
        for machine in ("AMD64", "ARM64")
    ),
    Platform.MACOS: tuple(
        {"sys_platform": "darwin", "platform_system": "Darwin", "os_name": "posix", "platform_machine": machine}
        for machine in ("arm64", "x86_64")
    ),
}
"""Marker environments per platform, one per CPU architecture torch ships for it."""


def _marker_outcomes(marker: Marker | None, python: Version, platform: Platform) -> tuple[bool, ...]:
    """
    Evaluate a marker on every architecture of a platform for one python version.

    Parameters
    ----------
    marker : Marker | None
        Marker to evaluate; None always holds.
    python : Version
        CPython major.minor version.
    platform : Platform
        Target platform.

    Returns
    -------
    tuple[bool, ...]
        One outcome per architecture in `_PLATFORM_ENVIRONMENTS`.
    """
    if marker is None:
        return (True,)
    base = {
        "python_version": f"{python.major}.{python.minor}",
        "python_full_version": f"{python.major}.{python.minor}.0",
        "implementation_name": "cpython",
        "implementation_version": f"{python.major}.{python.minor}.0",
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_version": "",
        "extra": "",
    }
    return tuple(marker.evaluate({**base, **machine}) for machine in _PLATFORM_ENVIRONMENTS[platform])


def _python_scope_marker(scope: tuple[Version, ...], pythons: tuple[Version, ...]) -> Marker | None:
    """
    Express a subset of a release's python versions as a marker.

    Parameters
    ----------
    scope : tuple[Version, ...]
        The python versions to select, in ascending order.
    pythons : tuple[Version, ...]
        Every python version the release covers, in ascending order.

    Returns
    -------
    Marker | None
        None when `scope` is all of `pythons`; a range bounded by the
        neighbouring versions when `scope` is contiguous within `pythons`;
        otherwise one equality clause per version.
    """
    if len(scope) == len(pythons):
        return None
    first = pythons.index(scope[0])
    last = pythons.index(scope[-1])
    if last - first + 1 == len(scope):
        clauses = []
        if first > 0:
            clauses.append(f"python_version >= '{scope[0]}'")
        if last < len(pythons) - 1:
            clauses.append(f"python_version < '{pythons[last + 1]}'")
        return Marker(" and ".join(clauses))
    return Marker(" or ".join(f"python_version == '{python}'" for python in scope))


def _and_markers(first: Marker | None, second: Marker | None) -> Marker | None:
    if first is None:
        return second
    if second is None:
        return first
    try:
        return Marker(f"({first}) and ({second})")
    except InvalidMarker:
        return first


def _pick_file(files: list[_DistFile], platform: Platform) -> _DistFile | None:
    """
    Choose the file whose metadata stands for a version on a platform.

    Parameters
    ----------
    files : list[_DistFile]
        Files of a single version.
    platform : Platform
        Platform the dependency installs on.

    Returns
    -------
    _DistFile | None
        A wheel for the platform, then a platform-independent wheel, then an
        sdist, then any wheel; None when there are no files.
    """
    preferences: tuple[Callable[[_DistFile], bool], ...] = (
        lambda f: f.is_wheel and platform in f.platforms,
        lambda f: f.is_wheel and len(f.platforms) == 0,
        lambda f: not f.is_wheel,
        lambda f: f.is_wheel,
    )
    for accept in preferences:
        for file in files:
            if accept(file):
                return file
    return None


class IndexDependencyResolver:
    """
    Find the dependencies of a torch wheel that only its own index serves.

    Listings and metadata are cached per URL, so resolving many releases
    against the same indexes fetches each page once.

    Parameters
    ----------
    semaphore : asyncio.Semaphore
        Concurrency limiter shared with the rest of the scrape.
    """

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore
        self._listings: dict[str, asyncio.Future[list[_DistFile]]] = {}
        self._requirements: dict[str, asyncio.Future[list[Requirement] | None]] = {}

    def _cached_listing(self, url: str) -> asyncio.Future[list[_DistFile]]:
        future = self._listings.get(url)
        if future is None:
            future = asyncio.ensure_future(self._fetch_listing(url))
            self._listings[url] = future
        return future

    async def _fetch_listing(self, url: str) -> list[_DistFile]:
        result = await fetch(url, self._semaphore)
        if result.status == 200:
            return _parse_listing(url, result.text)
        if is_transient(result.status):
            raise ScrapeError(f"listing fetch returned {result.status} for {url}")
        if result.status not in MISSING_STATUSES:
            logger.warning("listing fetch returned %d for %s", result.status, url)
        return []

    async def listing(self, base_url: str, name: str) -> list[_DistFile]:
        """
        Return a project's files on an index.

        Parameters
        ----------
        base_url : str
            Index root (e.g., 'https://pypi.org/simple').
        name : str
            Project name.

        Returns
        -------
        list[_DistFile]
            The project's files, empty when the index lacks the project.

        Raises
        ------
        ScrapeError
            The listing kept failing transiently.
        """
        return await self._cached_listing(f"{base_url}/{canonicalize_name(name)}/")

    def requirements(self, url: str, is_wheel: bool, filename: str) -> asyncio.Future[list[Requirement] | None]:
        """
        Return the requirements a distribution file declares.

        Parameters
        ----------
        url : str
            Absolute file URL.
        is_wheel : bool
            True to read a wheel's METADATA, False to read an sdist's PKG-INFO.
        filename : str
            File name, which selects the sdist archive format.

        Returns
        -------
        asyncio.Future[list[Requirement] | None]
            Resolves to the declared requirements, or None when they cannot be
            read reliably.
        """
        future = self._requirements.get(url)
        if future is None:
            future = asyncio.ensure_future(self._fetch_requirements(url, is_wheel, filename))
            self._requirements[url] = future
        return future

    async def _fetch_requirements(self, url: str, is_wheel: bool, filename: str) -> list[Requirement] | None:
        if is_wheel:
            text = await fetch_wheel_metadata_at(url, self._semaphore)
            if len(text) == 0:
                return None
            return _requires_dist(text, from_sdist=False)
        result = await fetch(url, self._semaphore)
        if is_transient(result.status):
            raise ScrapeError(f"sdist fetch returned {result.status} for {url}")
        if result.status != 200 or len(result.content) > MAX_SDIST_BYTES:
            return None
        pkg_info = _sdist_pkg_info(filename, result.content)
        if pkg_info is None:
            return None
        return _requires_dist(pkg_info, from_sdist=True)

    async def _covered_pythons(
        self,
        base_url: str,
        requirement: Requirement,
        pythons: tuple[Version, ...],
        platform: Platform,
    ) -> tuple[set[Version], list[_DistFile]]:
        """
        Find which python versions an index can satisfy a requirement on.

        Parameters
        ----------
        base_url : str
            Index root.
        requirement : Requirement
            Requirement to satisfy.
        pythons : tuple[Version, ...]
            Python versions in question.
        platform : Platform
            Target platform.

        Returns
        -------
        tuple[set[Version], list[_DistFile]]
            The python versions with an installable matching file, and the
            matching files.
        """
        files = await self.listing(base_url, requirement.name)
        versions = set(requirement.specifier.filter({f.version for f in files}))
        matching = [f for f in files if f.version in versions]
        covered = {python for python in pythons if any(f.installs_on(python, platform) for f in matching)}
        return covered, matching

    async def index_only(
        self,
        index_url: str,
        platform: Platform,
        pythons: tuple[Version, ...],
        requirements: list[tuple[Requirement, Marker | None]],
    ) -> tuple[IndexDependency, ...]:
        """
        Walk a wheel's requirements for packages only its index can satisfy.

        Coverage is decided per python version on the wheel's platform. A
        requirement the default registry covers for every python version in
        question is left to it. For the python versions where only the index
        covers it, it becomes an `IndexDependency` limited to them, and the
        newest matching version's own requirements are walked the same way
        under the extras requested, so packages it pulls in that only the
        index hosts are found too. Markers accumulate down the chain.

        A requirement the torch wheel declares keeps its version specifier.
        A package found further down is listed only so uv routes it to the
        index; it carries no specifier, so its parent alone constrains which
        version installs.

        Parameters
        ----------
        index_url : str
            Sub-index the wheel comes from.
        platform : Platform
            Platform of the wheel.
        pythons : tuple[Version, ...]
            Python versions the wheel's release covers on this platform.
        requirements : list[tuple[Requirement, Marker | None]]
            The wheel's applicable requirements with their settled markers.

        Returns
        -------
        tuple[IndexDependency, ...]
            Index-only dependencies in discovery order.
        """
        found: dict[tuple[str, str, str], IndexDependency] = {}
        seen: set[tuple[str, frozenset[str], str, tuple[Version, ...]]] = set()
        pending: list[tuple[Requirement, Marker | None, tuple[Version, ...], bool]] = [
            (requirement, marker, pythons, True) for requirement, marker in requirements
        ]
        while len(pending) > 0:
            requirement, marker, parent_scope, declared_by_torch = pending.pop(0)
            outcomes = {python: _marker_outcomes(marker, python, platform) for python in parent_scope}
            scope = tuple(python for python in parent_scope if any(outcomes[python]))
            if len(scope) == 0:
                continue
            name = canonicalize_name(requirement.name)
            extras = frozenset(canonicalize_name(extra) for extra in requirement.extras)
            # Cycle identity leaves the accumulated marker out: revisiting a
            # package under a longer chain of the same conditions adds nothing.
            key = (name, extras, str(requirement.specifier), scope)
            if key in seen:
                continue
            seen.add(key)
            on_registry, _ = await self._covered_pythons(PYPI_SIMPLE_URL, requirement, scope, platform)
            if len(on_registry) == len(scope):
                continue
            on_index, matching = await self._covered_pythons(index_url, requirement, scope, platform)
            needed = tuple(python for python in scope if python not in on_registry and python in on_index)
            unserved = [str(python) for python in scope if python not in on_registry and python not in on_index]
            if len(unserved) > 0:
                logger.warning(
                    "no index serves %s for python %s on %s (%s)",
                    requirement,
                    ", ".join(unserved),
                    platform.name,
                    index_url,
                )
            if len(needed) == 0:
                continue
            # When the marker's outcome is the same on every architecture, the
            # python versions that need the index say everything it does on
            # this platform; otherwise the architecture condition is kept.
            uniform = all(len(set(outcomes[python])) == 1 for python in needed)
            scope_marker = _python_scope_marker(needed, pythons)
            emitted = scope_marker if uniform else _and_markers(marker, scope_marker)
            extras_text = f"[{','.join(sorted(extras))}]" if len(extras) > 0 else ""
            specifier = str(requirement.specifier) if declared_by_torch else ""
            dependency = IndexDependency(
                name=name,
                requirement=f"{name}{extras_text}{specifier}",
                marker=str(emitted).replace('"', "'") if emitted is not None else "",
            )
            found.setdefault((dependency.name, dependency.requirement, dependency.marker), dependency)
            newest = max(f.version for f in matching)
            chosen = _pick_file([f for f in matching if f.version == newest], platform)
            if chosen is None:
                continue
            children = await self.requirements(chosen.url, chosen.is_wheel, chosen.filename)
            if children is None:
                logger.info("metadata for %s is unreadable; its dependencies are left to uv", chosen.filename)
                continue
            for child in children:
                applies, carried = resolve_extras(child.marker, extras)
                if applies:
                    pending.append((child, _and_markers(marker, carried), needed, False))
        return tuple(found.values())
