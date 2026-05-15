from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from packaging.version import Version

from torch_wheel_index._client import make_semaphore
from torch_wheel_index.models import Catalog
from torch_wheel_index.models import ComputeType
from torch_wheel_index.models import PackageInstance
from torch_wheel_index.models import Platform
from torch_wheel_index.paths import default_cache_dir
from torch_wheel_index.paths import default_cache_path
from torch_wheel_index.pipeline import DEFAULT_CUTOFF_VERSION
from torch_wheel_index.pipeline import _latest_at_main_index
from torch_wheel_index.pipeline import fetch_catalog_async
from torch_wheel_index.pipeline import refresh_if_stale_async
from torch_wheel_index.serialization import catalog_to_dict
from torch_wheel_index.serialization import load_catalog
from torch_wheel_index.serialization import save_catalog


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all subcommands attached.
    """
    parser = argparse.ArgumentParser(
        prog="torch-wheel-index",
        description="Discover and query the PyTorch wheel index.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable info-level logging to stderr.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch",
        help="Scrape the PyTorch index and emit catalog JSON to stdout.",
    )
    fetch.add_argument(
        "--cutoff",
        default=str(DEFAULT_CUTOFF_VERSION),
        help="Minimum torch version to include (default: %(default)s).",
    )

    check = sub.add_parser(
        "check",
        help="Compare the cached catalog against the live index. Exits 1 if stale.",
    )
    _add_cache_args(check)
    _add_format_arg(check)

    listing = sub.add_parser("list", help="List unique values from the catalog.")
    listing.add_argument(
        "field",
        choices=["versions", "compute-types", "python-versions", "platforms", "compute-versions"],
        help="Which set of unique values to print.",
    )
    listing.add_argument(
        "--compute-type",
        default=None,
        help="When listing compute-versions, restrict to one backend.",
    )
    _add_cache_args(listing)
    _add_format_arg(listing)

    find = sub.add_parser("find", help="Filter releases by coordinates.")
    find.add_argument("--version", default=None)
    find.add_argument("--compute-type", default=None)
    find.add_argument("--compute-version", default=None)
    find.add_argument("--python", "--python-version", dest="python_version", default=None)
    find.add_argument("--platform", default=None)
    _add_cache_args(find)
    _add_format_arg(find)

    cache = sub.add_parser("cache", help="Inspect, refresh, or clear the OS cache.")
    cache.add_argument("action", choices=["path", "refresh", "clear"])
    cache.add_argument(
        "--cutoff",
        default=str(DEFAULT_CUTOFF_VERSION),
        help="Minimum torch version when refreshing (default: %(default)s).",
    )
    cache.add_argument(
        "--force",
        action="store_true",
        help="When refreshing, scrape unconditionally instead of using the freshness fast-path.",
    )

    return parser


def _add_cache_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cache",
        default=None,
        help="Cache file path (default: OS cache).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip the cache and scrape fresh.",
    )


def _add_format_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: %(default)s).",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entry point.

    Parameters
    ----------
    argv : Sequence[str], optional
        Argument list (used for testing). Defaults to sys.argv[1:].

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )

    handler = {
        "fetch": _cmd_fetch,
        "check": _cmd_check,
        "list": _cmd_list,
        "find": _cmd_find,
        "cache": _cmd_cache,
    }[args.command]
    return handler(args)


def _cmd_fetch(args: argparse.Namespace) -> int:
    cutoff = Version(args.cutoff)
    catalog = asyncio.run(fetch_catalog_async(cutoff))
    json.dump(catalog_to_dict(catalog), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    cache_path = Path(args.cache) if args.cache is not None else default_cache_path()

    if args.no_cache or not cache_path.exists():
        if args.format == "json":
            json.dump({"cached": False, "path": str(cache_path)}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print(f"no cache at {cache_path}", file=sys.stderr)
        return 1

    catalog = load_catalog(cache_path)
    sem = make_semaphore()
    latest_torch, latest_vision = asyncio.run(_check_latest(sem))

    cached_torch = catalog.newest_version()
    try:
        cached_vision = catalog.newest_torchvision_version()
    except ValueError:
        cached_vision = Version("0.0.0")

    stale_torch = latest_torch is not None and latest_torch > cached_torch
    stale_vision = latest_vision is not None and latest_vision > cached_vision
    stale = stale_torch or stale_vision

    if args.format == "json":
        payload = {
            "cached": True,
            "path": str(cache_path),
            "stale": stale,
            "torch": {
                "cached": str(cached_torch),
                "online": str(latest_torch) if latest_torch is not None else None,
                "stale": stale_torch,
            },
            "torchvision": {
                "cached": str(cached_vision),
                "online": str(latest_vision) if latest_vision is not None else None,
                "stale": stale_vision,
            },
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"torch:        cached={cached_torch}  online={latest_torch}  {'STALE' if stale_torch else 'ok'}")
        print(
            f"torchvision:  cached={cached_vision}  online={latest_vision}  {'STALE' if stale_vision else 'ok'}",
        )

    return 1 if stale else 0


async def _check_latest(sem: asyncio.Semaphore) -> tuple[Version | None, Version | None]:
    torch_v, vision_v = await asyncio.gather(
        _latest_at_main_index("torch", sem),
        _latest_at_main_index("torchvision", sem),
    )
    return torch_v, vision_v


def _cmd_list(args: argparse.Namespace) -> int:
    catalog = _load_or_fetch(args)

    if args.field == "versions":
        values: list[str] = [str(v) for v in catalog.versions()]
        _emit_list(args.format, values)
        return 0
    if args.field == "compute-types":
        names = [ct.name for ct in catalog.compute_types()]
        _emit_list(args.format, names)
        return 0
    if args.field == "python-versions":
        values = [str(v) for v in catalog.python_versions()]
        _emit_list(args.format, values)
        return 0
    if args.field == "platforms":
        names = [p.name for p in catalog.platforms()]
        _emit_list(args.format, names)
        return 0
    if args.field == "compute-versions":
        if args.compute_type is None:
            grouped = {ct.name: [str(v) for v in vs] for ct, vs in catalog.compute_versions_by_type().items()}
            if args.format == "json":
                json.dump(grouped, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                for name, versions in grouped.items():
                    print(f"{name}: {', '.join(versions)}")
            return 0
        values = [str(v) for v in catalog.compute_versions(args.compute_type)]
        _emit_list(args.format, values)
        return 0
    return 1


def _emit_list(fmt: str, values: list[str]) -> None:
    if fmt == "json":
        json.dump(values, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for value in values:
            print(value)


def _cmd_find(args: argparse.Namespace) -> int:
    catalog = _load_or_fetch(args)
    matches = catalog.find(
        version=args.version,
        compute_type=args.compute_type,
        compute_version=args.compute_version,
        python_version=args.python_version,
        platform=args.platform,
    )

    if args.format == "json":
        payload = [
            {
                "version": str(m.version),
                "compute_type": m.compute_type.name,
                "compute_version": str(m.compute_version),
                "python_version": str(m.python_version),
                "platform": m.platform.name,
                "index_url": m.index_url,
                "torchvision_version": (
                    str(catalog.torchvision_pairs[m.version]) if m.version in catalog.torchvision_pairs else None
                ),
                "torchaudio_version": (
                    str(catalog.torchaudio_pairs[m.version]) if m.version in catalog.torchaudio_pairs else None
                ),
            }
            for m in matches
        ]
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if len(matches) > 0 else 1

    if len(matches) == 0:
        print("no matches", file=sys.stderr)
        return 1
    _print_table(matches, catalog)
    return 0


def _print_table(matches: list[PackageInstance], catalog: Catalog) -> None:
    headers = ["version", "compute", "cv", "python", "platform", "torchvision", "torchaudio", "index_url"]
    rows: list[list[str]] = []
    for m in matches:
        paired_vision = catalog.torchvision_pairs.get(m.version)
        paired_audio = catalog.torchaudio_pairs.get(m.version)
        rows.append(
            [
                str(m.version),
                m.compute_type.name,
                str(m.compute_version),
                str(m.python_version),
                m.platform.name,
                str(paired_vision) if paired_vision is not None else "",
                str(paired_audio) if paired_audio is not None else "",
                m.index_url,
            ]
        )
    widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(len(headers))]
    sep = "  "
    print(sep.join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    print(sep.join("-" * w for w in widths))
    for row in rows:
        print(sep.join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))


def _cmd_cache(args: argparse.Namespace) -> int:
    if args.action == "path":
        print(default_cache_path())
        return 0
    if args.action == "refresh":
        cutoff = Version(args.cutoff)
        catalog = asyncio.run(refresh_if_stale_async(default_cache_path(), cutoff, force=args.force))
        print(
            f"cache refreshed: {len(catalog.releases)} releases at {default_cache_path()}",
            file=sys.stderr,
        )
        return 0
    if args.action == "clear":
        cache_dir = default_cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"removed {cache_dir}", file=sys.stderr)
        else:
            print(f"nothing to remove at {cache_dir}", file=sys.stderr)
        return 0
    return 1


def _load_or_fetch(args: argparse.Namespace) -> Catalog:
    cache_path = Path(args.cache) if args.cache is not None else default_cache_path()
    if not args.no_cache and cache_path.exists():
        return load_catalog(cache_path)
    catalog = asyncio.run(fetch_catalog_async())
    save_catalog(catalog, cache_path)
    return catalog


__all__ = ["build_parser", "main", "Catalog", "ComputeType", "Platform"]
