"""FastAPI application for the Scenario Editor.

The editor is a **separate** application from the Scenario Result Viewer.  It is
not mounted on the viewer's app and shares none of its routes, templates or
entry point: integrating the two is a later step, and doing it now would make
every viewer route a place this feature could break.  What the two do share is
the stack -- FastAPI, Jinja partials, htmx, Tailwind -- and a visual language,
so that merging them later is a matter of moving templates rather than porting
an architecture.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..authoring import registry
from ..authoring.persistence import DraftStore, default_draft_dir
from .service import EditorError, EditorService

logger = logging.getLogger(__name__)

__all__ = ["build_service", "create_app", "app"]

_EDITOR_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _EDITOR_DIR / "templates"
_STATIC_DIR = _EDITOR_DIR / "static"

#: Environment variable choosing where exported packages are written.
EXPORT_DIR_ENV = "SCENARIO_EDITOR_EXPORT_DIR"

#: Environment variable pointing at a self-hosted build of the simple_lanelet2
#: web viewer.  See :data:`DEFAULT_MAP_VIEWER_URL`.
MAP_VIEWER_ENV = "SCENARIO_EDITOR_MAP_VIEWER"

#: The Lanelet2 map viewer the spawn preview mounts: a wasm renderer published
#: by ``simple_lanelet2``, the same project that provides the ``lanelet2``
#: Python API this framework runs on.  It gives the preview real map drawing,
#: pan and zoom, and click-to-pick for free -- none of which is worth
#: reimplementing in an SVG.
#:
#: It is loaded from the project's GitHub Pages build because the wasm module is
#: built, not committed.  Point :data:`MAP_VIEWER_ENV` at ``tools/build_web.sh``
#: output to serve it yourself; when it cannot be loaded at all the preview
#: falls back to a server-rendered SVG, so an offline editor still works.
DEFAULT_MAP_VIEWER_URL = "https://hakuturu583.github.io/simple_lanelet2/viewer.js"


def map_viewer_url() -> str:
    """Return the URL the spawn preview loads the map viewer from."""
    return os.environ.get(MAP_VIEWER_ENV, DEFAULT_MAP_VIEWER_URL).strip()


def default_export_dir() -> Path:
    """Return the directory exported scenario packages are written to."""
    env = os.environ.get(EXPORT_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "scenario_packages").resolve()


def build_service(
    draft_dir: str | Path | None = None, export_dir: str | Path | None = None
) -> EditorService:
    """Build the service the app's routes operate through."""
    store = DraftStore(Path(draft_dir) if draft_dir else default_draft_dir())
    return EditorService(
        store, Path(export_dir) if export_dir else default_export_dir()
    )


def create_app(
    draft_dir: str | Path | None = None, export_dir: str | Path | None = None
) -> FastAPI:
    """Create the editor application.

    Args:
        draft_dir: Where drafts are stored.  Defaults to ``./scenario_drafts``.
        export_dir: Default destination for exported packages.

    Returns:
        A configured :class:`FastAPI` application.
    """
    application = FastAPI(title="Scenario Editor")
    application.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    # The templates render primitives from their metadata rather than branching
    # on type, so the registry is the one global they genuinely need.
    templates.env.globals.update(
        map_viewer_url=map_viewer_url,
        action_specs=registry.action_specs,
        condition_specs=registry.condition_specs,
        constraint_specs=registry.constraint_specs,
        binding_specs=registry.binding_specs,
        get_action_spec=registry.get_action_spec,
        get_condition_spec=registry.get_condition_spec,
        get_constraint_spec=registry.get_constraint_spec,
        get_binding_spec=registry.get_binding_spec,
    )

    application.state.templates = templates
    application.state.service = build_service(draft_dir, export_dir)

    @application.exception_handler(EditorError)
    def _editor_error(request: Request, exc: Exception) -> Response:
        """Render a request for something that is not there as a page, not a 500.

        A stale bookmark to a deleted draft is a normal thing to do, and the
        user needs a way back rather than a traceback.
        """
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"heading": "Not found", "message": str(exc), "page": "error"},
            status_code=404,
        )

    from .routes import router  # noqa: PLC0415 -- avoids an import cycle

    application.include_router(router)
    return application


#: Module-level application, used by ``uv run scenario-editor``.
app = create_app()
