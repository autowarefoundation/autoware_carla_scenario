"""Scenario Editor -- a web UI for authoring scenarios declaratively.

Runs as its own application, separate from the Scenario Result Viewer::

    uv run scenario-editor

``uv run viewer`` is untouched: the editor mounts nothing on the viewer's app
and shares none of its routes, so this feature cannot regress the viewer.
Integrating the two is deliberately left to a later step.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

__all__ = ["main"]


def main() -> None:
    """Entry point for ``uv run scenario-editor``.

    Reads its configuration from the environment so the editor can be pointed
    at a project's own directories without a config file:

    * ``SCENARIO_EDITOR_DRAFTS`` -- where drafts are stored
      (default ``./scenario_drafts``).
    * ``SCENARIO_EDITOR_EXPORT_DIR`` -- default export destination
      (default ``./scenario_packages``).
    * ``SCENARIO_EDITOR_HOST`` / ``SCENARIO_EDITOR_PORT`` -- bind address
      (default ``0.0.0.0:9100``, one port up from the viewer so both can run).
    """
    import uvicorn  # noqa: PLC0415

    from .app import create_app  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    draft_dir = os.environ.get("SCENARIO_EDITOR_DRAFTS")
    export_dir = os.environ.get("SCENARIO_EDITOR_EXPORT_DIR")
    application = create_app(
        draft_dir=Path(draft_dir) if draft_dir else None,
        export_dir=Path(export_dir) if export_dir else None,
    )

    host = os.environ.get("SCENARIO_EDITOR_HOST", "0.0.0.0")  # noqa: S104
    port = int(os.environ.get("SCENARIO_EDITOR_PORT", "9100"))
    uvicorn.run(application, host=host, port=port, log_level="info")
