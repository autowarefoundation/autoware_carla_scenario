"""Scenario Result Viewer - Web UI for browsing scenario test results."""

from __future__ import annotations

import os
from pathlib import Path


def _fetch_web_viewer(argv: list[str]) -> None:
    """``viewer --fetch-rerun-web-viewer [--from TGZ] [--version V]``: keep a
    copy of Rerun's web viewer for the viewer to serve itself (offline)."""
    import argparse  # noqa: PLC0415

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


def main() -> None:
    """Entry point for ``uv run viewer``."""
    import sys  # noqa: PLC0415

    if sys.argv[1:2] == ["--fetch-rerun-web-viewer"]:
        _fetch_web_viewer(sys.argv[2:])
        return

    import uvicorn  # noqa: PLC0415

    from .app import app  # noqa: PLC0415, F811

    # Allow overriding the base path via environment variable.
    env_base = os.environ.get("VIEWER_BASE_PATH")
    if env_base:
        from . import app as app_module  # noqa: PLC0415

        app_module.BASE_PATH = Path(env_base).resolve()

    host = os.environ.get("VIEWER_HOST", "0.0.0.0")  # noqa: S104
    port = int(os.environ.get("VIEWER_PORT", "9000"))

    uvicorn.run(app, host=host, port=port, log_level="info")
