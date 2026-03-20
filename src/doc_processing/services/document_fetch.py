"""
Fetch document from URL with NSE India session support.

Uses a requests session with browser-like headers and cookies when the URL
is from nseindia.com. Optional temp_dir from DOC_PROCESSING_TEMP_DIR for
writing files; callers must delete after processing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from doc_processing.config import get_settings

logger = logging.getLogger(__name__)

NSE_DOMAIN = "nseindia.com"
CONNECT_TIMEOUT = 15
NSE_COOKIE_DELAY = 0.5


def is_nse_url(url: str) -> bool:
    """Return True if URL is from NSE India (requires session cookies and headers)."""
    try:
        host = urlparse(url).netloc.lower()
        return NSE_DOMAIN in host
    except Exception:
        return False


def _session_for_nse(timeout: int) -> requests.Session:
    """Create a requests session with NSE-required headers and cookies (visit main page first)."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf, application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/",
        "Origin": "https://www.nseindia.com",
    })
    try:
        session.get("https://www.nseindia.com/", timeout=min(15, timeout))
        time.sleep(NSE_COOKIE_DELAY)
    except Exception as e:
        logger.warning("Could not set NSE session cookies: %s", e)
    return session


class DocumentFetchError(Exception):
    """Raised when document cannot be fetched from URL."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def fetch_document(url: str, timeout: int = 120) -> bytes:
    """
    Download document from URL. Uses NSE session when url is from nseindia.com.

    Args:
        url: Document URL (PDF, image, markdown, etc.).
        timeout: Read timeout in seconds; connect timeout is min(15, timeout).

    Returns:
        Raw response body.

    Raises:
        DocumentFetchError: On non-2xx, timeout, connection error, or empty body.
    """
    connect_timeout = min(CONNECT_TIMEOUT, timeout)
    timeouts = (connect_timeout, timeout)

    try:
        if is_nse_url(url):
            session = _session_for_nse(timeout)
            resp = session.get(url, timeout=timeouts, stream=True)
        else:
            resp = requests.get(url, timeout=timeouts, stream=True)
    except requests.Timeout as e:
        raise DocumentFetchError(f"Request timed out: {e}") from e
    except requests.ConnectionError as e:
        raise DocumentFetchError(f"Connection failed: {e}") from e
    except requests.RequestException as e:
        raise DocumentFetchError(str(e)) from e

    if not resp.ok:
        raise DocumentFetchError(
            f"HTTP {resp.status_code}: {resp.reason or 'error'}",
            status_code=resp.status_code,
        )

    try:
        content = resp.content
    except Exception as e:
        raise DocumentFetchError(f"Failed to read response body: {e}") from e

    if not content:
        raise DocumentFetchError("Empty response body")

    return content


def temp_path_for_document(content: bytes, suffix: str = ".bin") -> Path:
    """
    Write content to a file in DOC_PROCESSING_TEMP_DIR (or system temp). Caller must delete.

    Args:
        content: Bytes to write.
        suffix: File suffix (e.g. .pdf, .png).

    Returns:
        Path to the written file.
    """
    import os
    import tempfile
    settings = get_settings()
    if settings.temp_dir:
        base_dir = Path(settings.temp_dir).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=suffix, dir=base_dir)
    else:
        fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
        return Path(path)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
