from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "torch-wheel-index"
CACHE_FILENAME = "pytorch_info.json"


def default_cache_dir() -> Path:
    """
    OS-specific cache directory for torch-wheel-index.

    Resolves to:
      - Windows: %LOCALAPPDATA%\\torch-wheel-index\\Cache
      - macOS:   ~/Library/Caches/torch-wheel-index
      - Linux:   $XDG_CACHE_HOME/torch-wheel-index (defaulting to ~/.cache)

    Returns
    -------
    Path
        Directory path. May not exist on disk yet.
    """
    if sys.platform == "win32":
        base_str = os.environ.get("LOCALAPPDATA")
        base = Path(base_str) if base_str else Path.home() / "AppData" / "Local"
        return base / APP_NAME / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / APP_NAME
    base_str = os.environ.get("XDG_CACHE_HOME")
    base = Path(base_str) if base_str else Path.home() / ".cache"
    return base / APP_NAME


def default_cache_path() -> Path:
    """
    Default OS-cache path for the catalog JSON file.

    Returns
    -------
    Path
        Full file path inside the OS cache directory. May not exist yet.
    """
    return default_cache_dir() / CACHE_FILENAME
