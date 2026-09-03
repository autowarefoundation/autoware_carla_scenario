"""HTTP routes for the Scenario Editor.

Every editing route returns an HTML *partial* that htmx swaps into the page:
there is no client-side model of the scenario, so the document on disk is
always what the screen shows.  Mutating routes all funnel through
:class:`~autoware_carla_scenario.editor.service.EditorService` and then re-render
the editor body, which keeps the canvas, the inspector and the validation
banner from ever disagreeing with each other.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
)

from ..authoring.models import ScenarioDocument
from ..authoring.package_export import PackageExportError, export_package
from ..authoring.persistence import Draft, dump_document_yaml
from . import map_preview
from .service import EditorError, EditorService, find_constraint

logger = logging.getLogger(__name__)

router = APIRouter()

__all__ = ["router"]


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _service(request: Request) -> EditorService:
    """Return the app's editor service."""
    return request.app.state.service  # type: ignore[no-any-return]


def _templates(request: Request) -> Any:
    """Return the app's Jinja environment wrapper."""
    return request.app.state.templates


def _resolve_target(
    document: ScenarioDocument, object_id: str
) -> tuple[str, Optional[Any], Optional[str]]:
    """Return ``(kind, object, constraint_owner)`` for an inspector target id.

    ``kind`` is one of ``scenario``, ``entity``, ``action``, ``condition``,
    ``constraint`` or ``missing``.  ``constraint_owner`` is the id of the entity
    whose spawn search holds the object, and is set for constraints only.
    """
    if not object_id or object_id == "scenario":
        return "scenario", document, None
    entity = document.entity(object_id)
    if entity is not None:
        return "entity", entity, None
    action = document.action(object_id)
    if action is not None:
        return "action", action, None
    condition = document.condition(object_id)
    if condition is not None:
        return "condition", condition, None
    owner, constraint = find_constraint(document, object_id)
    if constraint is not None:
        return "constraint", constraint, owner
    return "missing", None, None


def _context(
    request: Request,
    draft: Draft,
    selected: str = "scenario",
    *,
    error: str = "",
    validate: bool = True,
) -> dict[str, Any]:
    """Build the template context shared by every editor render.

    A *selected* id that no longer resolves falls back to the scenario itself,
    so deleting the object the inspector was showing leaves a usable panel
    rather than an empty one.
    """
    service = _service(request)
    document = draft.document
    kind, target, constraint_owner = _resolve_target(document, selected)
    if kind == "missing":
        kind, target, selected = "scenario", document, "scenario"
    return {
        "draft": draft,
        "document": document,
        "report": service.validate(draft) if validate else None,
        "selected": selected,
        "selected_kind": kind,
        "target": target,
        "constraint_owner": constraint_owner,
        "error": error,
        "map_loaded": map_preview.is_map_loaded(document),
        "export_dir": str(service.export_dir),
        "page": "editor",
    }


def _body(request: Request, context: dict[str, Any]) -> HTMLResponse:
    """Render the editor body partial -- the htmx swap target for every edit."""
    return _templates(request).TemplateResponse(
        request=request, name="partials/editor_body.html", context=context
    )


def _apply(request: Request, draft_id: str, selected: str, mutate: Any) -> HTMLResponse:
    """Run *mutate* against the draft, persist it, and re-render the body.

    A rejected edit re-renders the *unmodified* draft with the reason attached,
    so a typo shows an explanation instead of silently doing nothing or storing
    a broken value.
    """
    service = _service(request)
    draft = service.require_draft(draft_id)
    try:
        result = mutate(draft.document)
    except EditorError as exc:
        fresh = service.require_draft(draft_id)
        return _body(request, _context(request, fresh, selected, error=str(exc)))
    service.save(draft)
    focus = result if isinstance(result, str) else selected
    return _body(request, _context(request, draft, focus))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """List the stored drafts."""
    service = _service(request)
    return _templates(request).TemplateResponse(
        request=request,
        name="index.html",
        context={
            "drafts": service.list_drafts(),
            "draft_dir": str(service.store.root),
            "page": "index",
        },
    )


@router.post("/new")
async def create_draft(request: Request) -> RedirectResponse:
    """Create a draft from a starter document and open it."""
    form = await request.form()
    draft = _service(request).create_draft(
        kind=str(form.get("kind") or "cut_in"),
        title=str(form.get("title") or "").strip(),
    )
    return RedirectResponse(url=f"/draft/{draft.id}", status_code=303)


@router.get("/draft/{draft_id}", response_class=HTMLResponse)
def open_draft(
    request: Request, draft_id: str, selected: str = "scenario"
) -> HTMLResponse:
    """Open the editor on a draft."""
    draft = _service(request).require_draft(draft_id)
    return _templates(request).TemplateResponse(
        request=request,
        name="editor.html",
        context=_context(request, draft, selected),
    )


@router.get("/draft/{draft_id}/canvas", response_class=HTMLResponse)
def canvas(request: Request, draft_id: str, selected: str = "scenario") -> HTMLResponse:
    """Render just the swimlane canvas."""
    draft = _service(request).require_draft(draft_id)
    return _templates(request).TemplateResponse(
        request=request,
        name="partials/canvas.html",
        context=_context(request, draft, selected, validate=False),
    )


@router.get("/draft/{draft_id}/inspector/{object_id}", response_class=HTMLResponse)
def inspector(request: Request, draft_id: str, object_id: str) -> HTMLResponse:
    """Render the inspector for one object."""
    draft = _service(request).require_draft(draft_id)
    return _templates(request).TemplateResponse(
        request=request,
        name="partials/inspector.html",
        context=_context(request, draft, object_id, validate=False),
    )


@router.get("/draft/{draft_id}/map.osm")
def map_source(request: Request, draft_id: str) -> FileResponse:
    """Serve the draft's Lanelet2 map so the wasm viewer can render it.

    The viewer parses the ``.osm`` in the browser, which is why the editor
    serves the file rather than a projection of it: the drawing, the pan and
    zoom, and the picking all happen client-side, and the server keeps doing
    the one thing only it can -- evaluating constraints with the real sweeper.

    Raises:
        EditorError: If the draft has no readable Lanelet2 file, which the app
            renders as a 404 rather than a broken viewer.
    """
    draft = _service(request).require_draft(draft_id)
    source = map_preview.lanelet2_source(draft.document)
    if source is None:
        raise EditorError(
            "This scenario has no readable Lanelet2 (.osm) file. Set the path "
            "in the Scenario inspector."
        )
    return FileResponse(
        source,
        media_type="application/xml",
        filename=source.name,
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/draft/{draft_id}/yaml", response_class=PlainTextResponse)
def document_yaml(request: Request, draft_id: str) -> PlainTextResponse:
    """Return the Scenario IR as YAML -- the canonical form, viewable as-is."""
    draft = _service(request).require_draft(draft_id)
    return PlainTextResponse(dump_document_yaml(draft.document))


# ---------------------------------------------------------------------------
# Scenario metadata
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/scenario", response_class=HTMLResponse)
async def update_scenario(request: Request, draft_id: str) -> HTMLResponse:
    """Apply the scenario-level inspector form."""
    form = dict(await request.form())
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "scenario",
        lambda doc: service.update_scenario(doc, form),
    )


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/entity", response_class=HTMLResponse)
async def add_entity(request: Request, draft_id: str) -> HTMLResponse:
    """Add an entity to the scenario."""
    form = dict(await request.form())
    kind = str(form.get("kind", "vehicle"))
    service = _service(request)
    return _apply(
        request, draft_id, "scenario", lambda doc: service.add_entity(doc, kind).id
    )


@router.post("/draft/{draft_id}/entity/{entity_id}", response_class=HTMLResponse)
async def update_entity(
    request: Request, draft_id: str, entity_id: str
) -> HTMLResponse:
    """Apply the entity inspector form."""
    form = dict(await request.form())
    service = _service(request)
    return _apply(
        request,
        draft_id,
        entity_id,
        lambda doc: service.update_entity(doc, entity_id, form),
    )


@router.post("/draft/{draft_id}/entity/{entity_id}/delete", response_class=HTMLResponse)
def delete_entity(request: Request, draft_id: str, entity_id: str) -> HTMLResponse:
    """Delete an entity together with its actions and conditions."""
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "scenario",
        lambda doc: service.delete_entity(doc, entity_id),
    )


# ---------------------------------------------------------------------------
# Spawn constraints
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/constraint", response_class=HTMLResponse)
async def add_constraint(request: Request, draft_id: str) -> HTMLResponse:
    """Add a constraint to an entity's spawn search."""
    form = dict(await request.form())
    entity_id = str(form.get("entity_id", ""))
    type_id = str(form.get("type_id", ""))
    parent_id = str(form.get("parent_id", "")) or None
    service = _service(request)
    return _apply(
        request,
        draft_id,
        entity_id,
        lambda doc: service.add_constraint(doc, entity_id, type_id, parent_id),
    )


@router.post("/draft/{draft_id}/constraint/{node_id}", response_class=HTMLResponse)
async def update_constraint(
    request: Request, draft_id: str, node_id: str
) -> HTMLResponse:
    """Apply the constraint inspector form."""
    form = dict(await request.form())
    selected = str(form.get("selected") or node_id)
    service = _service(request)
    return _apply(
        request,
        draft_id,
        selected,
        lambda doc: service.update_constraint(doc, node_id, form),
    )


@router.post(
    "/draft/{draft_id}/constraint/{node_id}/delete", response_class=HTMLResponse
)
async def delete_constraint(
    request: Request, draft_id: str, node_id: str
) -> HTMLResponse:
    """Delete a constraint subtree."""
    form = dict(await request.form())
    selected = str(form.get("selected") or "scenario")
    service = _service(request)
    return _apply(
        request, draft_id, selected, lambda doc: service.delete_constraint(doc, node_id)
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/action", response_class=HTMLResponse)
async def add_action(request: Request, draft_id: str) -> HTMLResponse:
    """Add an action to an actor's lane."""
    form = dict(await request.form())
    type_id = str(form.get("type_id", ""))
    actor = str(form.get("actor", "")) or None
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "scenario",
        lambda doc: service.add_action(doc, type_id, actor).id,
    )


@router.post("/draft/{draft_id}/action/{action_id}", response_class=HTMLResponse)
async def update_action(
    request: Request, draft_id: str, action_id: str
) -> HTMLResponse:
    """Apply the action inspector form."""
    form = dict(await request.form())
    service = _service(request)
    return _apply(
        request,
        draft_id,
        action_id,
        lambda doc: service.update_action(doc, action_id, form),
    )


@router.post("/draft/{draft_id}/action/{action_id}/delete", response_class=HTMLResponse)
def delete_action(request: Request, draft_id: str, action_id: str) -> HTMLResponse:
    """Delete an action and its trigger."""
    service = _service(request)
    return _apply(
        request, draft_id, "scenario", lambda doc: service.delete_action(doc, action_id)
    )


@router.post("/draft/{draft_id}/action/{action_id}/move", response_class=HTMLResponse)
async def move_action(request: Request, draft_id: str, action_id: str) -> HTMLResponse:
    """Shift an action along its lane.  Layout only, never semantics."""
    form = await request.form()
    try:
        delta = int(str(form.get("delta") or 1))
    except ValueError:
        delta = 1
    service = _service(request)
    return _apply(
        request,
        draft_id,
        action_id,
        lambda doc: service.move_action(doc, action_id, delta),
    )


@router.post("/draft/{draft_id}/actors", response_class=HTMLResponse)
async def reorder_actors(request: Request, draft_id: str) -> HTMLResponse:
    """Set the swimlane order.  Layout only, never semantics."""
    form = await request.form()
    order = [str(value) for value in form.getlist("order")]
    service = _service(request)
    return _apply(
        request, draft_id, "scenario", lambda doc: service.reorder_actors(doc, order)
    )


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/condition", response_class=HTMLResponse)
async def add_condition(request: Request, draft_id: str) -> HTMLResponse:
    """Add a condition to an action trigger, a composition, or an assertion."""
    form = dict(await request.form())
    slot = str(form.get("slot", ""))
    type_id = str(form.get("type_id", ""))
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "scenario",
        lambda doc: service.add_condition(doc, slot, type_id).id,
    )


@router.post("/draft/{draft_id}/condition/{node_id}", response_class=HTMLResponse)
async def update_condition(
    request: Request, draft_id: str, node_id: str
) -> HTMLResponse:
    """Apply the condition inspector form."""
    form = dict(await request.form())
    service = _service(request)
    return _apply(
        request,
        draft_id,
        node_id,
        lambda doc: service.update_condition(doc, node_id, form),
    )


@router.post(
    "/draft/{draft_id}/condition/{node_id}/delete", response_class=HTMLResponse
)
def delete_condition(request: Request, draft_id: str, node_id: str) -> HTMLResponse:
    """Delete a condition subtree."""
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "scenario",
        lambda doc: service.delete_condition(doc, node_id),
    )


# ---------------------------------------------------------------------------
# Spawn preview
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/spawn-preview", response_class=HTMLResponse)
async def spawn_preview(request: Request, draft_id: str) -> HTMLResponse:
    """Evaluate an entity's spawn constraints and draw the matches."""
    form = dict(await request.form())
    entity_id = str(form.get("entity_id", ""))
    load_map = str(form.get("load_map", "")).lower() in ("1", "true", "on", "yes")

    draft = _service(request).require_draft(draft_id)
    entity = draft.document.entity(entity_id)
    if entity is None:
        result = map_preview.PreviewResult(error=f"No entity named {entity_id!r}.")
    else:
        result = map_preview.evaluate_spawn(draft.document, entity, load_map=load_map)
    return _templates(request).TemplateResponse(
        request=request,
        name="partials/spawn_preview.html",
        context={
            "draft": draft,
            "document": draft.document,
            "entity": entity,
            "preview": result,
        },
    )


# ---------------------------------------------------------------------------
# Validate / save / export
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/validate", response_class=HTMLResponse)
def validate(request: Request, draft_id: str) -> HTMLResponse:
    """Re-run validation and render the report."""
    service = _service(request)
    draft = service.require_draft(draft_id)
    return _templates(request).TemplateResponse(
        request=request,
        name="partials/validation.html",
        context={"draft": draft, "report": service.validate(draft), "expanded": True},
    )


@router.post("/draft/{draft_id}/save", response_class=HTMLResponse)
async def save_draft(request: Request, draft_id: str) -> HTMLResponse:
    """Save the draft's working state."""
    form = dict(await request.form())
    service = _service(request)
    draft = service.require_draft(draft_id)
    title = str(form.get("title", "")).strip()
    if title:
        draft.document.title = title
    service.save(draft)
    return _templates(request).TemplateResponse(
        request=request,
        name="partials/status.html",
        context={
            "message": f"Draft saved to {service.store.path_for(draft.id)}",
            "tone": "ok",
        },
    )


@router.post("/draft/{draft_id}/delete")
def delete_draft(request: Request, draft_id: str) -> RedirectResponse:
    """Delete a draft and return to the list."""
    _service(request).delete_draft(draft_id)
    return RedirectResponse(url="/", status_code=303)


@router.post("/draft/{draft_id}/export", response_class=HTMLResponse)
async def export(request: Request, draft_id: str) -> HTMLResponse:
    """Export the draft as a reproducible Scenario Package.

    Declared ``async`` but doing its work through a blocking call would stall
    the event loop for the length of a dependency resolution, so the export runs
    in a worker thread.
    """
    import anyio  # noqa: PLC0415

    form = dict(await request.form())
    service = _service(request)
    draft = service.require_draft(draft_id)

    destination = Path(str(form.get("destination") or service.export_dir)).expanduser()
    options = {
        "dev_mode": _checked(form, "dev_mode"),
        "lock": _checked(form, "lock"),
        "verify": _checked(form, "verify"),
        "run_tests": _checked(form, "run_tests"),
        "force": _checked(form, "force"),
    }

    def _run() -> tuple[Any, str, str]:
        try:
            result = export_package(draft.document, destination, **options)
        except PackageExportError as exc:
            logger.warning("Scenario package export failed: %s", exc)
            # The exporter attaches the failing tool's output to the exception.
            return None, str(exc), exc.log
        return result, "", result.log

    result, error, log = await anyio.to_thread.run_sync(_run)

    return _templates(request).TemplateResponse(
        request=request,
        name="partials/export_result.html",
        context={
            "draft": draft,
            "result": result,
            "error": error,
            "log": log,
            "destination": str(destination),
        },
    )


def _checked(form: dict[str, Any], name: str) -> bool:
    """Return whether a checkbox named *name* was submitted as checked."""
    return str(form.get(name, "")).lower() in ("1", "true", "on", "yes")
