"""Utility functions for download operations with retry support."""

import logging
import shutil
import ssl
import urllib.request
from pathlib import Path
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_BACKOFF_BASE = 2  # exponential backoff base


def create_retry_session(
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = 1.0,
    timeout: int = DEFAULT_TIMEOUT,
) -> requests.Session:
    """
    Create a requests session with retry logic for transient errors.
    
    Args:
        max_retries: Maximum number of retry attempts.
        backoff_factor: Factor for exponential backoff between retries.
        timeout: Request timeout in seconds.
        
    Returns:
        Configured requests.Session with retry logic.
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.timeout = timeout
    
    return session


# Retry decorator for download functions
def retry_on_timeout(
    max_attempts: int = DEFAULT_MAX_RETRIES,
    min_wait: int = 1,
    max_wait: int = 60,
):
    """
    Decorator factory for retrying functions on timeout/connection errors.
    
    Args:
        max_attempts: Maximum number of retry attempts.
        min_wait: Minimum wait time between retries (seconds).
        max_wait: Maximum wait time between retries (seconds).
        
    Returns:
        Configured retry decorator.
    """
    return retry(
        retry=retry_if_exception_type((
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        )),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def download_file_with_retry(
    url: str,
    target_path: Path,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    ssl_verify: bool = True,
    session: requests.Session | None = None,
) -> bool:
    """
    Download a file with automatic retry on timeout/connection errors.
    
    Args:
        url: URL to download from.
        target_path: Local path to save the file.
        max_retries: Maximum number of retry attempts.
        timeout: Request timeout in seconds.
        ssl_verify: Whether to verify SSL certificates.
        session: Optional pre-configured requests session.
        
    Returns:
        True if download succeeded, False otherwise.
    """
    @retry_on_timeout(max_attempts=max_retries)
    def _download():
        nonlocal session
        if session is None:
            session = create_retry_session(max_retries=1, timeout=timeout)
        
        response = session.get(url, stream=True, verify=ssl_verify, timeout=timeout)
        response.raise_for_status()
        
        # Write to a temp file first, then rename for atomicity
        temp_path = target_path.with_suffix(target_path.suffix + '.tmp')
        try:
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            temp_path.rename(target_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    try:
        _download()
        return True
    except Exception as e:
        logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
        return False


def download_file_urllib_with_retry(
    url: str,
    target_path: Path,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    ssl_context: ssl.SSLContext | None = None,
) -> bool:
    """
    Download a file using urllib with automatic retry on timeout/connection errors.
    
    Useful when you need custom SSL context (e.g., for expired certificates).
    
    Args:
        url: URL to download from.
        target_path: Local path to save the file.
        max_retries: Maximum number of retry attempts.
        timeout: Request timeout in seconds.
        ssl_context: Optional custom SSL context.
        
    Returns:
        True if download succeeded, False otherwise.
    """
    @retry_on_timeout(max_attempts=max_retries)
    def _download():
        with urllib.request.urlopen(url, context=ssl_context, timeout=timeout) as response:
            temp_path = target_path.with_suffix(target_path.suffix + '.tmp')
            try:
                with open(temp_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                temp_path.rename(target_path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise
    
    try:
        _download()
        return True
    except Exception as e:
        logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
        return False


def run_with_retry(
    func: Callable,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **kwargs,
) -> tuple[bool, any]:
    """
    Run a function with retry on timeout/connection errors.
    
    Useful for wrapping third-party download functions.
    
    Args:
        func: Function to call.
        *args: Positional arguments to pass to func.
        max_retries: Maximum number of retry attempts.
        **kwargs: Keyword arguments to pass to func.
        
    Returns:
        Tuple of (success: bool, result: any).
    """
    @retry_on_timeout(max_attempts=max_retries)
    def _run():
        return func(*args, **kwargs)
    
    try:
        result = _run()
        return True, result
    except Exception as e:
        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
        return False, None

