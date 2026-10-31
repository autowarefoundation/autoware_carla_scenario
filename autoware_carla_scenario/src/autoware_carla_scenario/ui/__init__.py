"""Scenario Result Viewer - Web UI for browsing scenario test results."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..serve_cli import add_bind_arguments, warn_if_exposed


def _fetch_web_viewer(argv: list[str]) -> None:
    """``viewer --fetch-rerun-web-viewer [--from TGZ] [--version V]``: keep a
    copy of Rerun's web viewer for the viewer to serve itself (offline)."""
    from .recordings import RecordingError, fetch_web_viewer  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="viewer --fetch-rerun-web-viewer",
        description="Keep a copy of Rerun's web viewer (@rerun-io/web-viewer) "
        "in the cache directory, which the viewer then serves itself instead "
        "of loading it from the CDN.",
    )
    parser.add_argument(
        "--from",
        dest="tarball",
        type=Path,
        default=None,
        help="the package's tarball (web-viewer-<version>.tgz) instead of the "
        "npm registry, for a machine without the network",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="the version (default: the one the installed converter writes)",
    )
    args = parser.parse_args(argv)
    try:
        target = fetch_web_viewer(args.version, args.tarball)
    except (RecordingError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Rerun web viewer kept in {target}")  # noqa: T201


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viewer",
        description="Scenario Result Viewer: a web UI for browsing scenario "
        "results and starting runs (FastAPI + Uvicorn).",
        epilog="Each option defaults to its environment variable. "
        "VIEWER_CACHE_DIR, VIEWER_RERUN_WEB_VIEWER, VIEWER_RERUN_MAP_DIR and "
        "VIEWER_SCENARIO_COMMAND are read from the environment only "
        "(docs/usage.md).",
    )
    add_bind_arguments(
        parser, host_env="VIEWER_HOST", port_env="VIEWER_PORT", default_port=9000
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=os.environ.get("VIEWER_BASE_PATH") or None,
        help="directory scanned for outputs/ and multirun/ results "
        "(env VIEWER_BASE_PATH; default the working directory)",
    )
    parser.add_argument(
        "--fetch-rerun-web-viewer",
        dest="fetch_args",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="[--from TGZ] [--version V]",
        help="keep a copy of Rerun's web viewer for offline use, then exit "
        "instead of serving; it takes the arguments after it "
        "(viewer --fetch-rerun-web-viewer --help)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``uv run viewer`` (``viewer --help`` for its options)."""
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.fetch_args is not None:
        _fetch_web_viewer(args.fetch_args)
        return

    # Parsed before uvicorn and the app are imported, so --help and a
    # mistyped option exit without serving.
    import uvicorn  # noqa: PLC0415

    from ..registry import load_scenario_plugins  # noqa: PLC0415
    from .app import app  # noqa: PLC0415, F811

    # External scenario packages (the ``autoware_carla_scenario.scenarios``
    # entry point) register their scenarios and conf dirs, which the scenario
    # list, its globs and the sweep resolution read, as the ``scenario`` CLI
    # loads them; once per process (the scanner and the resolver ask again).
    load_scenario_plugins()

    if args.base_path is not None:
        from . import app as app_module  # noqa: PLC0415

        app_module.BASE_PATH = Path(args.base_path).resolve()

    warn_if_exposed(
        "viewer",
        args.host,
        args.port,
        "browse the results and start scenario runs and SLURM submissions",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
