from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "torch-wheel-index/0.1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_CONCURRENCY = 32


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
    """

    status: int
    text: str
    url: str


def _fetch_sync(url: str, timeout: float, method: str) -> FetchResult:
    request = Request(url, method=method, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = b"" if method == "HEAD" else response.read()
            return FetchResult(
                status=response.status,
                text=body.decode("utf-8", errors="replace"),
                url=response.url,
            )
    except HTTPError as exc:
        return FetchResult(status=exc.code, text="", url=url)
    except (URLError, TimeoutError, OSError) as exc:
        logger.debug("fetch failed for %s: %s", url, exc)
        return FetchResult(status=0, text="", url=url)


async def fetch(
    url: str,
    semaphore: asyncio.Semaphore,
    timeout: float = DEFAULT_TIMEOUT,
    method: str = "GET",
) -> FetchResult:
    """
    Concurrent-safe async fetch via stdlib urllib in a worker thread.

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

    Returns
    -------
    FetchResult
        Status, body text, and final URL.
    """
    async with semaphore:
        return await asyncio.to_thread(_fetch_sync, url, timeout, method)


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
