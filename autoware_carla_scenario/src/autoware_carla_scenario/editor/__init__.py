"""Scenario Editor -- a web UI for authoring scenarios declaratively.

Runs as its own application, separate from the Scenario Result Viewer::

    uv run scenario-editor

``uv run viewer`` is untouched: the editor mounts nothing on the viewer's app
and shares none of its routes, so this feature cannot regress the viewer.
Integrating the two is deliberately left to a later step.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ..serve_cli import add_bind_arguments, warn_if_exposed

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scenario-editor",
        description="Scenario Editor: a web UI for authoring scenarios "
        "declaratively and exporting them as packages (FastAPI + Uvicorn).",
        epilog="Each option defaults to its environment variable. "
        "SCENARIO_EDITOR_MAP_ROOTS and SCENARIO_EDITOR_MAP_VIEWER are read "
        "from the environment only (docs/scenario_editor.md).",
    )
    add_bind_arguments(
        parser,
        host_env="SCENARIO_EDITOR_HOST",
        port_env="SCENARIO_EDITOR_PORT",
        # One port up from the viewer, so both can run.
        default_port=9100,
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=os.environ.get("SCENARIO_EDITOR_DRAFTS") or None,
        help="where drafts are stored "
        "(env SCENARIO_EDITOR_DRAFTS; default ./scenario_drafts)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=os.environ.get("SCENARIO_EDITOR_EXPORT_DIR") or None,
        help="where an exported wheelhouse's .zip is staged until the browser "
        "fetches it (env SCENARIO_EDITOR_EXPORT_DIR; default "
        "./scenario_packages)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``uv run scenario-editor``.

    Its options (``scenario-editor --help``) default to the environment, so
    the editor can be pointed at a project's own directories without a config
    file; a flag wins over its variable:

    * ``--drafts`` / ``SCENARIO_EDITOR_DRAFTS`` -- where drafts are stored
      (default ``./scenario_drafts``).
    * ``--export-dir`` / ``SCENARIO_EDITOR_EXPORT_DIR`` -- where an exported
      wheelhouse's ``.zip`` is staged until the browser fetches it (default
      ``./scenario_packages``).  Exports are handed back as downloads, so this
      is a holding area rather than a destination anyone browses.
    * ``--host`` / ``SCENARIO_EDITOR_HOST`` and ``--port`` /
      ``SCENARIO_EDITOR_PORT`` -- bind address (default ``127.0.0.1:9100``,
      one port up from the viewer so both can run).
    """
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)

    # Parsed before uvicorn and the app are imported, so --help and a
    # mistyped option exit without serving.
    import uvicorn  # noqa: PLC0415

    from .app import create_app  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    application = create_app(draft_dir=args.drafts, export_dir=args.export_dir)

    warn_if_exposed(
        "scenario-editor",
        args.host,
        args.port,
        "read and write drafts and export packages",
    )
    uvicorn.run(application, host=args.host, port=args.port, log_level="info")
