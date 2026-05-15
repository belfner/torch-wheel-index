from __future__ import annotations

import asyncio
import logging
import re
from html.parser import HTMLParser

from packaging.utils import parse_wheel_filename
from packaging.version import InvalidVersion
from packaging.version import Version

from torch_wheel_index._client import fetch

logger = logging.getLogger(__name__)

PYTORCH_INDEX_URL = "https://download.pytorch.org/whl"
PREVIOUS_VERSIONS_URL = "https://pytorch.org/get-started/previous-versions/"

_WHEEL_LINK_RE = re.compile(r">([^<>]+\.whl)<", re.IGNORECASE)
_INSTALL_RE = re.compile(r"install torch==(\S+) torchvision==(\S+)(?: torchaudio==(\S+))?")


async def fetch_wheel_filenames(
    index_url: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """
    Fetch a wheel directory listing and extract .whl filenames.

    Parameters
    ----------
    index_url : str
        URL of the wheel index page.
    semaphore : asyncio.Semaphore
        Concurrency limiter.

    Returns
    -------
    list[str]
        List of .whl filenames in document order.
    """
    result = await fetch(index_url, semaphore)
    if result.status != 200:
        logger.warning("index fetch returned %d for %s", result.status, index_url)
        return []
    return _WHEEL_LINK_RE.findall(result.text)


def _candidate_sub_index_keys(wheel_filenames: list[str]) -> set[str]:
    keys = set()
    for filename in wheel_filenames:
        _, version, _, _ = parse_wheel_filename(filename)
        if version.local is not None:
            keys.add(version.local)
    return keys


async def _verify_sub_index(
    sub_index_key: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str] | None:
    url = f"{PYTORCH_INDEX_URL}/{sub_index_key}"
    result = await fetch(url, semaphore)
    if result.status == 200:
        return url, sub_index_key
    return None


async def get_torch_sub_indexes(
    semaphore: asyncio.Semaphore,
) -> list[tuple[str, str]]:
    """
    Discover and verify all PyTorch sub-indexes.

    Reads the main `torch` index, extracts local-version tags from wheel
    filenames as sub-index keys, and HEAD/GETs each candidate URL to confirm
    it exists.

    Parameters
    ----------
    semaphore : asyncio.Semaphore
        Concurrency limiter.

    Returns
    -------
    list[tuple[str, str]]
        List of (url, sub_index_key) tuples for valid sub-indexes.
    """
    main_index = f"{PYTORCH_INDEX_URL}/torch"
    wheels = await fetch_wheel_filenames(main_index, semaphore)
    candidates = _candidate_sub_index_keys(wheels)

    tasks = [_verify_sub_index(key, semaphore) for key in candidates]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def fetch_package_wheels(
    sub_index_url: str,
    package: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """
    Fetch wheel filenames for a package within a sub-index.

    Parameters
    ----------
    sub_index_url : str
        Base URL of the sub-index.
    package : str
        Package name ('torch' or 'torchvision').
    semaphore : asyncio.Semaphore
        Concurrency limiter.

    Returns
    -------
    list[str]
        Wheel filenames listed in the package index.
    """
    url = f"{sub_index_url}/{package}"
    return await fetch_wheel_filenames(url, semaphore)


class _PreviousVersionsParser(HTMLParser):
    """
    Walk the pytorch.org/get-started/previous-versions article.

    Tracks immediate children of the `pytorch-article`: h3 sets the current
    torch version, h4 with an id containing 'wheel' becomes the active
    subsection, h5 with an id containing 'osx' arms capture for the next
    sibling div, whose text is recorded.
    """

    def __init__(self, cutoff_version: Version):
        super().__init__()
        self._cutoff = cutoff_version
        self._in_article = False
        self._depth = 0
        self._top_tag: str | None = None
        self._top_attrs: dict[str, str] = {}
        self._buffer: list[str] = []
        self._section: Version | None = None
        self._subsection_id: str | None = None
        self._read_next_div = False
        self._stop = False
        self.install_texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._stop:
            return
        attrs_d = {k: v or "" for k, v in attrs}
        if not self._in_article:
            if tag == "article" and "pytorch-article" in attrs_d.get("class", ""):
                self._in_article = True
            return
        self._depth += 1
        if self._depth == 1:
            self._top_tag = tag
            self._top_attrs = attrs_d
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._top_tag is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._in_article:
            return
        if tag == "article" and self._depth == 0:
            self._in_article = False
            return
        if self._depth == 1 and self._top_tag is not None:
            text = "".join(self._buffer).strip()
            top_tag = self._top_tag
            attrs = self._top_attrs
            if top_tag == "h3":
                if text.startswith("v"):
                    try:
                        version = Version(text[1:])
                    except InvalidVersion:
                        pass
                    else:
                        self._section = version
                        if version < self._cutoff:
                            self._stop = True
            elif self._section is not None:
                if top_tag == "h4":
                    self._subsection_id = attrs.get("id")
                elif top_tag == "h5":
                    sub = self._subsection_id or ""
                    if "wheel" in sub and "osx" in attrs.get("id", ""):
                        self._read_next_div = True
                elif top_tag == "div" and self._read_next_div:
                    self.install_texts.append(text)
                    self._read_next_div = False
            self._top_tag = None
            self._top_attrs = {}
            self._buffer = []
        self._depth -= 1


async def get_paired_versions(
    semaphore: asyncio.Semaphore,
    cutoff_version: Version,
) -> tuple[dict[Version, Version], dict[Version, Version]]:
    """
    Scrape the official torch-to-torchvision and torch-to-torchaudio pairings.

    Parameters
    ----------
    semaphore : asyncio.Semaphore
        Concurrency limiter.
    cutoff_version : Version
        Skip pairings older than this version.

    Returns
    -------
    tuple[dict[Version, Version], dict[Version, Version]]
        Two maps from torch version: the first to torchvision, the second to
        torchaudio. The torchaudio map can be smaller when older install
        lines omit the torchaudio token.
    """
    result = await fetch(PREVIOUS_VERSIONS_URL, semaphore)
    if result.status != 200:
        logger.warning("previous-versions fetch returned %d", result.status)
        return {}, {}

    parser = _PreviousVersionsParser(cutoff_version)
    parser.feed(result.text)

    vision_pairs: dict[Version, Version] = {}
    audio_pairs: dict[Version, Version] = {}
    for text in parser.install_texts:
        match = _INSTALL_RE.search(text)
        if match is None:
            continue
        torch_str, vision_str, audio_str = match.groups()
        try:
            torch_v = Version(torch_str)
        except InvalidVersion:
            continue
        try:
            vision_pairs[torch_v] = Version(vision_str)
        except InvalidVersion:
            pass
        if audio_str is not None:
            try:
                audio_pairs[torch_v] = Version(audio_str)
            except InvalidVersion:
                pass
    return vision_pairs, audio_pairs
