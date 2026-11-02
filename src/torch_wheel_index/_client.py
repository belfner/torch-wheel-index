from __future__ import annotations

import asyncio
import io
import logging
import zipfile
from dataclasses import dataclass
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "torch-wheel-index/0.1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_CONCURRENCY = 32
DEFAULT_RETRIES = 3
"""Extra attempts for a transient failure, with exponential backoff from one second."""

MISSING_STATUSES = frozenset({403, 404})
"""Statuses download.pytorch.org and PyPI answer for a path that does not exist."""


class ScrapeError(RuntimeError):
    """A fetch the catalog depends on kept failing, so the scrape is incomplete."""


def is_transient(status: int) -> bool:
    """
    Report whether a fetch status marks a failure worth retrying.

    Parameters
    ----------
    status : int
        HTTP status, or 0 for a network-level failure.

    Returns
    -------
    bool
        True for network failures, rate limiting, and server errors.
    """
    return status == 0 or status == 429 or status >= 500


@dataclass
class FetchResult:
    """
    Result of a single HTTP fetch.

    Parameters
    ----------
    status : int
        HTTP status code.
    text : str
        Decoded response body. Empty when status indicates failure.
    url : str
        Final URL after any redirects.
    content : bytes
        Raw response body, for binary payloads such as sdist archives.
    """

    status: int
    text: str
    url: str
    content: bytes = b""


def _fetch_sync(url: str, timeout: float, method: str) -> FetchResult:
    request = Request(url, method=method, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = b"" if method == "HEAD" else response.read()
            return FetchResult(
                status=response.status,
                text=body.decode("utf-8", errors="replace"),
                url=response.url,
                content=body,
            )
    except HTTPError as exc:
        return FetchResult(status=exc.code, text="", url=url)
    except (URLError, TimeoutError, OSError, HTTPException) as exc:
        # HTTPException covers a body cut short (IncompleteRead) and a
        # dropped connection; both are network failures worth a retry.
        logger.debug("fetch failed for %s: %s", url, exc)
        return FetchResult(status=0, text="", url=url)


async def fetch(
    url: str,
    semaphore: asyncio.Semaphore,
    timeout: float = DEFAULT_TIMEOUT,
    method: str = "GET",
    retries: int = DEFAULT_RETRIES,
) -> FetchResult:
    """
    Concurrent-safe async fetch via stdlib urllib in a worker thread.

    Transient failures (see `is_transient`) are retried with exponential
    backoff, released from the semaphore while waiting.

    Parameters
    ----------
    url : str
        URL to request.
    semaphore : asyncio.Semaphore
        Concurrency limiter.
    timeout : float, optional
        Per-request timeout in seconds, by default 30.0.
    method : str, optional
        HTTP method ('GET' or 'HEAD'), by default 'GET'.
    retries : int, optional
        Extra attempts after a transient failure, by default 3.

    Returns
    -------
    FetchResult
        Status, body, and final URL of the last attempt.
    """
    attempt = 0
    while True:
        async with semaphore:
            result = await asyncio.to_thread(_fetch_sync, url, timeout, method)
        if not is_transient(result.status) or attempt >= retries:
            return result
        logger.debug("fetch of %s returned %d; retrying", url, result.status)
        await asyncio.sleep(2**attempt)
        attempt += 1


def make_semaphore(max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> asyncio.Semaphore:
    """
    Construct a semaphore for throttling concurrent fetches.

    Parameters
    ----------
    max_concurrency : int, optional
        Maximum simultaneous fetches, by default 32.

    Returns
    -------
    asyncio.Semaphore
        Semaphore configured with the given limit.
    """
    return asyncio.Semaphore(max_concurrency)


RANGE_BLOCK_BYTES = 512 * 1024
"""Block size for ranged reads of remote wheels."""

MAX_RANGE_BLOCKS = 64
"""Ceiling on blocks fetched per remote wheel, which bounds a malformed archive."""


class _RemoteZip(io.RawIOBase):
    """
    Seekable read-only view of a remote file, fetched in cached byte ranges.

    `zipfile` reads the end-of-central-directory record, the central
    directory, and then one member, so a wheel's METADATA costs a few ranged
    requests instead of a download of the whole wheel.

    Parameters
    ----------
    url : str
        File URL. The server must honor Range requests.
    size : int
        File size in bytes.
    timeout : float
        Per-request timeout in seconds.
    """

    def __init__(self, url: str, size: int, timeout: float) -> None:
        super().__init__()
        self._url = url
        self._size = size
        self._timeout = timeout
        self._position = 0
        self._blocks: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        match whence:
            case io.SEEK_SET:
                self._position = offset
            case io.SEEK_CUR:
                self._position += offset
            case io.SEEK_END:
                self._position = self._size + offset
            case _:
                raise ValueError(f"invalid whence: {whence}")
        return self._position

    def _block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            return cached
        if len(self._blocks) >= MAX_RANGE_BLOCKS:
            raise OSError(f"ranged read of {self._url} exceeded {MAX_RANGE_BLOCKS} blocks")
        start = index * RANGE_BLOCK_BYTES
        end = min(start + RANGE_BLOCK_BYTES, self._size) - 1
        request = Request(self._url, headers={"User-Agent": DEFAULT_USER_AGENT, "Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=self._timeout) as response:
            if response.status != 206:
                raise OSError(f"server ignored the range request for {self._url}")
            data: bytes = response.read()
        self._blocks[index] = data
        return data

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        if self._position >= self._size:
            return 0
        wanted = min(len(buffer), self._size - self._position)
        written = 0
        while written < wanted:
            index, offset = divmod(self._position, RANGE_BLOCK_BYTES)
            chunk = self._block(index)[offset : offset + wanted - written]
            if len(chunk) == 0:
                break
            buffer[written : written + len(chunk)] = chunk
            written += len(chunk)
            self._position += len(chunk)
        return written


class _TransientReadError(Exception):
    """A ranged read failed in a way a retry can fix."""


def _read_remote_wheel_metadata(url: str, timeout: float) -> str:
    try:
        head = Request(url, method="HEAD", headers={"User-Agent": DEFAULT_USER_AGENT})
        with urlopen(head, timeout=timeout) as response:
            size = int(response.headers.get("Content-Length", "0"))
            ranges = response.headers.get("Accept-Ranges", "")
        if size == 0:
            return ""
        # Accept-Ranges is advisory (RFC 9110 section 14.3); a server that
        # ignores ranges answers 200, which _RemoteZip rejects.
        if ranges.lower() == "none":
            return ""
        with zipfile.ZipFile(_RemoteZip(url, size, timeout)) as archive:
            names = [n for n in archive.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                return ""
            return archive.read(names[0]).decode("utf-8", errors="replace")
    except HTTPError as exc:
        if is_transient(exc.code):
            raise _TransientReadError(f"{url}: HTTP {exc.code}") from exc
        logger.debug("ranged metadata read failed for %s: HTTP %d", url, exc.code)
        return ""
    except (URLError, TimeoutError, ConnectionError, HTTPException) as exc:
        raise _TransientReadError(f"{url}: {exc}") from exc
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        logger.debug("ranged metadata read failed for %s: %s", url, exc)
        return ""


async def fetch_remote_wheel_metadata(
    url: str,
    semaphore: asyncio.Semaphore,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """
    Read a wheel's METADATA from inside the remote archive using byte ranges.

    The fallback for wheels published without a PEP 658 sidecar, which
    covers torch up to 2.5 on every sub-index and later ROCm builds.

    Parameters
    ----------
    url : str
        Wheel URL.
    semaphore : asyncio.Semaphore
        Concurrency limiter, held for the whole read.
    timeout : float, optional
        Per-request timeout in seconds, by default 30.0.

    Returns
    -------
    str
        METADATA text, empty when the server lacks range support or the
        archive cannot be read.

    Raises
    ------
    ScrapeError
        The host kept failing transiently across retries.
    """
    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            async with semaphore:
                return await asyncio.to_thread(_read_remote_wheel_metadata, url, timeout)
        except _TransientReadError as exc:
            if attempt >= DEFAULT_RETRIES:
                raise ScrapeError(f"ranged metadata read kept failing: {exc}") from exc
            await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")
