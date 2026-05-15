# torch-wheel-index

Scrapes `download.pytorch.org/whl/` and turns it into a catalog you can query for the right index URL, compute backend, Python version, or platform. The catalog is cached in the OS cache directory and only refetched when the live index has a newer torch or torchvision release.

## Features

- Discovers every PyTorch sub-index (CUDA, ROCm, XPU, CPU) by parsing local-version tags.
- Collects `torch` and `torchvision` wheels with version, compute backend, Python version, and platform.
- Pairs each torch release with its matching `torchvision` and `torchaudio` versions (scraped from the official compatibility table). The cache only stores wheel data for `torch`; `torchaudio` versions come from the pairing table.
- Programmatic API and CLI. CLI prints text or JSON.
- One runtime dependency: `packaging`.

## Installation

```bash
pip install torch-wheel-index
```

## Requirements

- Python >= 3.10
- `packaging` >= 22.0

## Usage

### Get a Catalog

`get_catalog` reads the cached catalog, refreshing it first if the live torch or torchvision version is newer than what's stored.

```python
from torch_wheel_index import get_catalog

catalog = get_catalog()
print(catalog.newest_version())
```

### Find a Specific Wheel

```python
from torch_wheel_index import get_catalog

catalog = get_catalog()
matches = catalog.find(
    version="2.5.0",
    compute_type="CUDA",
    compute_version="12.4",
    python_version="3.12",
    platform="LINUX",
)
for m in matches:
    print(m.index_url)
```

### List Unique Values

```python
from torch_wheel_index import get_catalog

catalog = get_catalog()
print(catalog.versions())
print(catalog.compute_types())
print(catalog.python_versions())
print(catalog.compute_versions("CUDA"))
```

### Look Up the Matching torchvision and torchaudio

```python
from torch_wheel_index import get_catalog

catalog = get_catalog()
print(catalog.torchvision_for("2.5.0"))
print(catalog.torchaudio_for("2.5.0"))
```

### Force a Fresh Fetch

`fetch_catalog` skips the cache and scrapes the live index.

```python
from packaging.version import Version
from torch_wheel_index import fetch_catalog

catalog = fetch_catalog(cutoff_version=Version("2.0.0"))
```

### Async Usage

Every sync function has an `_async` counterpart.

```python
import asyncio
from torch_wheel_index import get_catalog_async

catalog = asyncio.run(get_catalog_async())
```

### Save and Load

```python
from pathlib import Path
from torch_wheel_index import fetch_catalog, load_catalog, save_catalog

catalog = fetch_catalog()
save_catalog(catalog, Path("catalog.json"))
loaded = load_catalog(Path("catalog.json"))
```

`save_catalog` writes to a temp file in the same directory and then renames into place, so concurrent readers never see a half-written file.

## Command Line

```bash
# scrape and emit JSON to stdout
torch-wheel-index fetch > catalog.json

# compare cache against live index; exits 1 if stale
torch-wheel-index check
torch-wheel-index check --format json

# list unique values
torch-wheel-index list versions
torch-wheel-index list compute-types
torch-wheel-index list python-versions
torch-wheel-index list platforms
torch-wheel-index list compute-versions --compute-type CUDA
torch-wheel-index list versions --format json

# query
torch-wheel-index find --version 2.5.0 --compute-type CUDA --python 3.12
torch-wheel-index find --version 2.5.0 --format json

# manage the cache
torch-wheel-index cache path
torch-wheel-index cache refresh
torch-wheel-index cache refresh --force
torch-wheel-index cache clear
```

## API Reference

### `get_catalog`

Read or refresh the OS-cached catalog.

```python
get_catalog(force: bool = False) -> Catalog
```

#### Parameters

- **`force`**: Refresh unconditionally instead of using the freshness fast-path (default `False`).

### `fetch_catalog`

Scrape the live index without consulting the cache.

```python
fetch_catalog(cutoff_version: Version = Version("2.0.0")) -> Catalog
```

#### Parameters

- **`cutoff_version`**: Minimum torch version to include.

### `refresh_if_stale`

Read `cache_path`, then rewrite it if the live torch or torchvision version is newer than what's cached.

```python
refresh_if_stale(cache_path: Path, cutoff_version: Version = Version("2.0.0"), force: bool = False) -> Catalog
```

#### Parameters

- **`cache_path`**: Path to the catalog JSON file.
- **`cutoff_version`**: Minimum torch version to include when refreshing.
- **`force`**: Refresh unconditionally.

### `Catalog`

Frozen dataclass holding the release list and the torch-to-torchvision and torch-to-torchaudio pairing maps.

#### Fields

- **`releases`**: List of `PackageInstance`. Sorted on construction.
- **`torchvision_pairs`**: `dict[Version, Version]` mapping torch to torchvision.
- **`torchaudio_pairs`**: `dict[Version, Version]` mapping torch to torchaudio.

#### Methods

- **`find(version=, compute_type=, compute_version=, python_version=, platform=)`**: Filter releases by any combination of coordinates. Each argument accepts a string or the corresponding type (`Version`, `ComputeType`, `Platform`).
- **`versions()`**: Unique torch versions, newest first.
- **`compute_types()`**: Unique compute backends.
- **`compute_versions(compute_type=None)`**: Unique backend versions, optionally restricted to one backend.
- **`compute_versions_by_type()`**: Map of compute type to its version list.
- **`python_versions()`**: Unique Python versions.
- **`platforms()`**: Unique platforms.
- **`newest_version()`**: Highest torch version. Raises `ValueError` on an empty catalog.
- **`newest_torchvision_version()`**: Highest torchvision version in the pairing map.
- **`newest_torchaudio_version()`**: Highest torchaudio version in the pairing map.
- **`torchvision_for(torch_version)`**: Paired torchvision `Version`, or `None`.
- **`torchaudio_for(torch_version)`**: Paired torchaudio `Version`, or `None`.

### `PackageInstance`

Frozen dataclass for a single wheel coordinate.

Fields: `version`, `compute_type`, `compute_version`, `python_version`, `platform`, `index_url`.

### `save_catalog` / `load_catalog`

```python
save_catalog(catalog: Catalog, path: Path) -> None
load_catalog(path: Path) -> Catalog
```

### `default_cache_path` / `default_cache_dir`

Return the cache file and its parent directory. Honors `XDG_CACHE_HOME` on Linux, `~/Library/Caches` on macOS, and `%LOCALAPPDATA%` on Windows.

### Async Counterparts

`get_catalog_async`, `fetch_catalog_async`, `refresh_if_stale_async`. Same signatures, return coroutines.

## License

`torch-wheel-index` is licensed under the [MIT License](LICENSE).

## Author

Ben Elfner
