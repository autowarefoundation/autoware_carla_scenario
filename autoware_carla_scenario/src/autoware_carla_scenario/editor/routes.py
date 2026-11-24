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
from typing import Any, Iterator, Mapping, Optional, Sequence
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.types import Receive, Scope, Send

from ..authoring.models import ScenarioDocument
from ..authoring.package_export import PackageExportError
from ..authoring.persistence import Draft, DraftStore, dump_document_yaml
from ..authoring.registry import TRUTHY_VALUES, is_checked
from ..maps import opendrive
from ..web_guard import refuse_cross_site
from . import map_preview, scenario_map
from .revisions import row_is_shown
from .service import (
    EditorError,
    EditorService,
    StagedArchive,
    StaleDraftError,
    as_port,
    find_constraint,
)

logger = logging.getLogger(__name__)

router = APIRouter()

__all__ = ["router"]


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _service(request: Request) -> EditorService:
    """Return the app's editor service."""
    return request.app.state.service  # type: ignore[no-any-return]


#: The hidden field every inspector form carries: the draft revision it was
#: rendered from, which the edit is refused against when it is out of date.
REVISION_FIELD = "draft_revision"

#: The hidden field an inspector form that names its row by position carries:
#: what that row was called when the form was rendered (#2348).
ROW_FIELD = "row"

#: Likewise for the row that row belongs to: a phase's controller.
PARENT_ROW_FIELD = "parent_row"

#: The header ``editor.js`` sends with every request: a token naming the page
#: (one tab's one load of the editor) the request comes from.
CLIENT_HEADER = "X-Editor-Client"

#: The longest client token kept; anything longer is cut, never refused.
_MAX_CLIENT = 64


def _client(request: Request) -> str:
    """The page a request came from (:data:`CLIENT_HEADER`), or ``""``."""
    return request.headers.get(CLIENT_HEADER, "")[:_MAX_CLIENT]


def _templates(request: Request) -> Any:
    """Return the app's Jinja environment wrapper."""
    return request.app.state.templates


def _resolve_target(
    document: ScenarioDocument, object_id: str
) -> tuple[str, Optional[Any], Optional[str]]:
    """Return ``(kind, object, constraint_owner)`` for an inspector target id.

    ``kind`` is one of ``scenario``, ``environment``, ``entity``, ``action``,
    ``condition``, ``constraint`` or ``missing``.  ``constraint_owner`` is the
    id of the object whose lanelet search holds it -- an entity, an action or a
    condition -- and is set for constraints only.

    ``environment`` is the one target that is not a node of the document: it is
    the Environment track, and what it inspects is the part of the world no
    actor performs.  It reads the document like ``scenario`` does, so the two
    share a target.
    """
    if not object_id or object_id == "scenario":
        return "scenario", document, None
    if object_id == "environment":
        return "environment", document, None
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
    notice: str = "",
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
    # Resolved once and shared: the inspector, the map panel and the "is the map
    # parsed yet" badge all describe the same map, and each resolving it for
    # itself meant three walks of the cache per keystroke.
    paths = map_preview.map_paths(document)
    return {
        "draft": draft,
        "document": document,
        "report": service.validate(draft) if validate else None,
        "selected": selected,
        "selected_kind": kind,
        "target": target,
        "constraint_owner": constraint_owner,
        "error": error,
        "notice": notice,
        "map_loaded": map_preview.is_map_loaded(document, paths),
        "map_status": map_preview.map_status(document, paths),
        "carla_host": opendrive.DEFAULT_HOST,
        "carla_port": opendrive.DEFAULT_PORT,
        "page": "editor",
    }


def _body(request: Request, context: dict[str, Any]) -> HTMLResponse:
    """Render the editor body partial -- the htmx swap target for every edit.

    Marked ``partial``, so it also carries the tab title and the top bar's
    crumb, which sit outside the body and are swapped in out of band.
    """
    return _templates(request).TemplateResponse(
        request=request,
        name="partials/editor_body.html",
        context={**context, "partial": True},
    )


#: The longest rendering of a refused value an error message shows.
_MAX_SHOWN_INPUT = 80


def describe_validation_error(exc: ValidationError) -> str:
    """The sentence the editor shows for a value the document's models refused.

    One clause per refused field, naming the model and field it reached and
    the value it was given, so the author can tell which input to correct.
    The value is cut short, so an unexpectedly large one cannot flood the page.

    The handlers are meant to check a form value before storing it, so a
    refusal that reaches here may equally be a bug in a handler: it is logged
    with its traceback, so catching it does not also hide it.
    """
    logger.warning("Edit refused by the document's validation", exc_info=exc)
    clauses = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error.get("loc", ()))
        field = f"{exc.title}.{where}" if where else exc.title
        given = repr(error.get("input"))
        if len(given) > _MAX_SHOWN_INPUT:
            given = given[: _MAX_SHOWN_INPUT - 3] + "..."
        clauses.append(f"{field}: {error['msg']} (got {given})")
    return "Invalid value -- " + "; ".join(clauses) + "."


def _check_row(
    draft_id: str,
    form: Mapping[str, Any],
    rows: Sequence[Any],
    index: int,
    attribute: str,
    *,
    field: str = ROW_FIELD,
    renamed_by: Optional[str] = None,
) -> None:
    """Refuse a form whose row index now points at another row.

    The signal group, controller and phase forms and a waypoint condition's
    vertex post to a URL holding the row's position, and are checked leniently
    (:meth:`~EditorService.check_revision`): a page's own quick edits pass.
    One of those may have removed or reordered the rows, so the form also
    names its row by its *attribute* when rendered (*field*), and is refused
    when the row now at its position has another -- unless that is what the
    form itself posts under *renamed_by*, a rename of this row from this form
    still on its way.  A form without *field* (a script) is not checked, nor a
    position out of range, which the edit itself reports.

    Raises:
        StaleDraftError: If the row at *index* is not the one the form shows.
    """
    if field not in form or not 0 <= index < len(rows):
        return
    current = str(getattr(rows[index], attribute))
    renamed = str(form.get(renamed_by) or "") if renamed_by else ""
    if row_is_shown(current, str(form[field]), renamed):
        return
    raise StaleDraftError(
        "The row this form edits has moved since it was shown -- one before it "
        "was removed or the rows were reordered -- so the change was not "
        "applied. Reload the page to see the draft as it is now, then try again.",
        draft_id,
    )


def _apply(
    request: Request,
    draft_id: str,
    selected: str,
    mutate: Any,
    form: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = False,
) -> HTMLResponse:
    """Run *mutate* against the draft, persist it, and re-render the body.

    A rejected edit re-renders the *unmodified* draft with the reason attached,
    so a typo shows an explanation instead of silently doing nothing or storing
    a broken value.  A returned string is the id to select next.

    The read, the mutation and the save happen under
    :meth:`~EditorService.editing`, so two edits of one draft served at once
    cannot each save a copy that misses the other.  *mutate* must therefore be
    quick; a slow operation goes through :func:`_map_action` instead.

    A *form* that carries :data:`REVISION_FIELD` is an inspector form, which
    posts every field it shows: one rendered from a draft that has been changed
    elsewhere since is refused with :class:`StaleDraftError` (HTTP 409) before
    *mutate* runs, rather than putting back what was changed there (#2345).
    With *strict* -- a button that names a row by its index, or posts a whole
    order -- only the current revision passes, even from the page that made
    the saves since: those saves may have moved the rows (#2348).
    """
    service = _service(request)
    client = _client(request)
    error = ""
    with service.editing(draft_id):
        draft = service.require_draft(draft_id)
        if form is not None and REVISION_FIELD in form:
            service.check_revision(draft, form[REVISION_FIELD], client, strict=strict)
        try:
            result = mutate(draft.document)
        except EditorError as exc:
            error = str(exc)
            draft = service.require_draft(draft_id)
        except ValidationError as exc:
            # The document's models validate on assignment, so a form value a
            # handler stores without checking it first is refused here rather
            # than escaping as a bare 500.
            error = describe_validation_error(exc)
            draft = service.require_draft(draft_id)
        else:
            service.save(draft, client)
    if error:
        return _body(request, _context(request, draft, selected, error=error))
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
    """Open the editor on a draft.

    The map panel is rendered with the page rather than fetched after it, so the
    first thing on screen already says where the scenario happens.  It is the
    only render that builds a map view here: every later one comes through
    ``/map-view``, which is the panel refreshing itself.
    """
    draft = _service(request).require_draft(draft_id)
    context = _context(request, draft, selected)
    context["view"] = scenario_map.build_map_view(draft.document)
    return _templates(request).TemplateResponse(
        request=request, name="editor.html", context=context
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


async def _map_action(
    request: Request, draft_id: str, act: Any, *, bare_error: bool = False
) -> Response:
    """Run a map operation off the event loop, then re-render the editor body.

    Every operation here can clone a repository or load a CARLA town -- tens of
    seconds of blocking work.  Run inline it would stall the whole ASGI loop, so
    no other tab, and no other request from this one, would be served while a
    map downloaded.  Since the draft stays editable meanwhile, it goes through
    :meth:`~EditorService.run_map_operation`, which applies only the map fields
    *act* changed to the draft as it is when *act* finishes.

    *act* returns a sentence about what it did, shown as a notice, because it
    took seconds, reached the network, and can succeed quietly.

    A failure is shown inline on the re-rendered body, with HTTP 200, since the
    editor swaps that body in.  With *bare_error* it is instead the bare reason,
    as plain text with HTTP 422: for a caller that swaps nothing and leaves the
    page on success, so it can tell a failure apart and show the reason (#2206).
    """
    service = _service(request)
    service.require_draft(draft_id)  # An unknown draft is a 404, not a notice.
    client = _client(request)

    def _run() -> Response:
        try:
            draft, result = service.run_map_operation(draft_id, act, client)
        except (EditorError, ValidationError) as exc:
            message = (
                describe_validation_error(exc)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            if bare_error:
                return PlainTextResponse(message, status_code=422)
            fresh = service.require_draft(draft_id)
            return _body(request, _context(request, fresh, "scenario", error=message))
        message = result if isinstance(result, str) else ""
        return _body(request, _context(request, draft, "scenario", notice=message))

    return await run_in_threadpool(_run)


@router.post("/draft/{draft_id}/map/fetch", response_class=HTMLResponse)
async def fetch_map(request: Request, draft_id: str) -> Response:
    """Download the map the scenario names, or refresh what is cached."""
    form = dict(await request.form())
    refresh = is_checked(form, "refresh")
    service = _service(request)
    return await _map_action(
        request, draft_id, lambda doc: service.fetch_map(doc, refresh=refresh)
    )


@router.post("/draft/{draft_id}/map/pin", response_class=HTMLResponse)
async def pin_map(request: Request, draft_id: str) -> Response:
    """Pin the map source to the exact commit it resolves to."""
    service = _service(request)
    return await _map_action(request, draft_id, service.pin_map)


@router.post("/draft/{draft_id}/map/opendrive", response_class=HTMLResponse)
async def fetch_opendrive(request: Request, draft_id: str) -> Response:
    """Read the map's OpenDRIVE from a running CARLA server and cache it."""
    form = dict(await request.form())
    service = _service(request)

    def _act(document: ScenarioDocument) -> str:
        # Parsed here rather than above, so a bad port is refused the way every
        # other bad field on this panel is -- inline, on the re-rendered body.
        # Blank, whitespace included, is unset: strip before the default.
        host = str(form.get("carla_host", "") or "").strip() or opendrive.DEFAULT_HOST
        port = as_port(form.get("carla_port", ""), "CARLA port", opendrive.DEFAULT_PORT)
        return service.fetch_opendrive(document, host=host, port=port)

    return await _map_action(request, draft_id, _act)


@router.post("/draft/{draft_id}/map/use", response_class=HTMLResponse)
async def use_map(request: Request, draft_id: str) -> Response:
    """Point the scenario at a map chosen in the library.

    The library's button posts here and opens the draft once it succeeds, so a
    failure is answered with HTTP 422 and the bare reason, which the library
    shows beside the button instead of opening the draft on the old map (#2206).
    """
    form = dict(await request.form())
    uri = str(form.get("uri", "")).strip()
    service = _service(request)

    def _act(document: ScenarioDocument) -> str:
        service.use_map(document, uri)
        return f"Now editing on {document.map.name}."

    return await _map_action(request, draft_id, _act, bare_error=True)


# ---------------------------------------------------------------------------
# Map library
# ---------------------------------------------------------------------------


def _known_draft_id(store: DraftStore, draft_id: str) -> str:
    """*draft_id* when it is a valid id naming a stored draft, else ``""``.

    The id is checked against the store's id pattern
    (:meth:`~autoware_carla_scenario.authoring.persistence.DraftStore.path_for`)
    and its file must exist, so a value from the URL never reaches a page
    unless it is one of the drafts the editor itself lists (#2053).
    """
    if not draft_id:
        return ""
    try:
        path = store.path_for(draft_id)
    except ValueError:
        return ""
    return draft_id if path.is_file() else ""


@router.get(
    "/maps",
    response_class=HTMLResponse,
    # It runs git against the repository the URL names (#1850).
    dependencies=[Depends(refuse_cross_site)],
)
def map_library(
    request: Request, repo: str = "", draft: str = "", refresh: str = ""
) -> HTMLResponse:
    """Browse the maps a repository offers.

    Reachable on its own and from a draft.  With ``draft`` set, every map
    carries a button that points *that* scenario at it; without one, a map
    starts a new scenario instead.

    A GET, so it can be linked and bookmarked, but one that makes the server
    run ``git ls-remote`` (and may clone or fetch) against whatever repository
    the URL names: a request another site sends is refused with 403
    (:func:`~autoware_carla_scenario.web_guard.refuse_cross_site`).

    ``draft`` comes from the URL, so it is kept only when it is a valid draft
    id naming a stored draft (:func:`_known_draft_id`); anything else is
    dropped, and the page offers new scenarios instead (#2053).
    """
    service = _service(request)
    draft = _known_draft_id(service.store, draft)
    entries: list[Any] = []
    error = ""
    if repo:
        try:
            entries = service.browse_maps(
                repo, refresh=refresh.lower() in TRUTHY_VALUES
            )
        except EditorError as exc:
            error = str(exc)
    return _templates(request).TemplateResponse(
        request=request,
        name="maps.html",
        context={
            "repositories": service.map_repositories(),
            "repo": repo,
            "entries": entries,
            "error": error,
            "draft_id": draft,
            "page": "maps",
        },
    )


@router.post("/maps/new", response_class=HTMLResponse)
async def new_draft_on_map(request: Request) -> Response:
    """Start a scenario on a map chosen in the library.

    A map that cannot be fetched creates no draft: the library is shown again
    with the reason, as ``/draft/{id}/map/use`` shows it beside its button
    (#1429, #2206).
    """
    form = dict(await request.form())
    uri = str(form.get("uri", "")).strip()
    title = str(form.get("title", "")).strip()
    repo = str(form.get("repo", "")).strip()
    service = _service(request)
    try:
        draft = await run_in_threadpool(service.create_draft_for_map, uri, title=title)
    except EditorError as exc:
        return _templates(request).TemplateResponse(
            request=request,
            name="maps.html",
            context={
                "repositories": service.map_repositories(),
                "repo": repo,
                "entries": [],
                "error": str(exc),
                "error_title": "No scenario was created",
                "draft_id": "",
                "page": "maps",
            },
        )
    return RedirectResponse(f"/draft/{draft.id}", status_code=303)


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
        form=form,
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
        form=form,
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
# Lanelet searches
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/lanelet-mode", response_class=HTMLResponse)
async def set_lanelet_mode(request: Request, draft_id: str) -> HTMLResponse:
    """Pin one lanelet field, or hand it to the constraint sweeper.

    The same route for every lanelet in a document -- a spawn, the ego's goal,
    the lanelet a condition watches -- because the choice is the same one; the
    ``slot`` field says which lanelet is being answered for.
    """
    form = dict(await request.form())
    slot_key = str(form.get("slot", ""))
    mode = str(form.get("mode", ""))
    service = _service(request)
    return _apply(
        request,
        draft_id,
        slot_key.rpartition(".")[0],
        lambda doc: service.set_lanelet_mode(doc, slot_key, mode),
    )


@router.post("/draft/{draft_id}/lanelet-binding", response_class=HTMLResponse)
async def set_lanelet_binding(request: Request, draft_id: str) -> HTMLResponse:
    """Say how a lanelet derived from the search follows the lanelet it picks."""
    form = dict(await request.form())
    slot_key = str(form.get("slot", ""))
    service = _service(request)
    return _apply(
        request,
        draft_id,
        slot_key.rpartition(".")[0],
        lambda doc: service.set_lanelet_binding(doc, slot_key, form),
        form=form,
    )


@router.post("/draft/{draft_id}/constraint", response_class=HTMLResponse)
async def add_constraint(request: Request, draft_id: str) -> HTMLResponse:
    """Add a constraint to the search that chooses one lanelet."""
    form = dict(await request.form())
    slot_key = str(form.get("slot", ""))
    type_id = str(form.get("type_id", ""))
    parent_id = str(form.get("parent_id", "")) or None
    service = _service(request)

    def _add(document: ScenarioDocument) -> str:
        service.add_constraint(document, slot_key, type_id, parent_id)
        return service.require_slot(document, slot_key).owner_id

    return _apply(request, draft_id, slot_key.rpartition(".")[0], _add)


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
        form=form,
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
    phase = str(form.get("phase", "")) or None
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "scenario",
        lambda doc: service.add_action(doc, type_id, actor, phase).id,
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
        form=form,
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
    delta = _delta(form)
    service = _service(request)
    return _apply(
        request,
        draft_id,
        action_id,
        lambda doc: service.move_action(doc, action_id, delta),
    )


@router.post(
    "/draft/{draft_id}/action/{action_id}/advance/{index}", response_class=HTMLResponse
)
async def move_vertex_condition(
    request: Request, draft_id: str, action_id: str, index: int
) -> HTMLResponse:
    """Put a waypoint condition on another vertex."""
    form = await request.form()
    vertex = form.get("vertex") or ""
    service = _service(request)

    def _move(document: ScenarioDocument) -> None:
        action = document.action(action_id)
        if action is not None:
            gates = action.advance_conditions
            _check_row(draft_id, form, gates, index, "vertex", renamed_by="vertex")
        service.move_vertex_condition(document, action_id, index, vertex)

    return _apply(request, draft_id, action_id, _move, form=form)


@router.post(
    "/draft/{draft_id}/action/{action_id}/advance/{index}/delete",
    response_class=HTMLResponse,
)
async def delete_vertex_condition(
    request: Request, draft_id: str, action_id: str, index: int
) -> HTMLResponse:
    """Remove a waypoint condition: the vertex goes back to the clock."""
    form = await request.form()
    service = _service(request)
    return _apply(
        request,
        draft_id,
        action_id,
        lambda doc: service.delete_vertex_condition(doc, action_id, index),
        form=form,
        strict=True,
    )


@router.post("/draft/{draft_id}/actors", response_class=HTMLResponse)
async def reorder_actors(request: Request, draft_id: str) -> HTMLResponse:
    """Set the swimlane order.  Layout only, never semantics."""
    form = await request.form()
    order = [str(value) for value in form.getlist("order")]
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "scenario",
        lambda doc: service.reorder_actors(doc, order),
        form=form,
        strict=True,
    )


# ---------------------------------------------------------------------------
# Traffic signals
#
# The panel is the Environment track's inspector, so every one of these
# re-selects "environment": an edit here must leave the panel it was made in
# open, not send the author back to the scenario.
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/signal-group", response_class=HTMLResponse)
def add_signal_group(request: Request, draft_id: str) -> HTMLResponse:
    """Declare one more movement."""
    service = _service(request)
    return _apply(
        request, draft_id, "environment", lambda doc: service.add_signal_group(doc)
    )


@router.post("/draft/{draft_id}/signal-group/{index}", response_class=HTMLResponse)
async def update_signal_group(
    request: Request, draft_id: str, index: int
) -> HTMLResponse:
    """Apply one signal group's form."""
    raw = await request.form()
    # `conflicts_with` is a set of checkboxes, so it arrives repeated; the rest
    # are single values. `dict(form)` would keep only the last conflict.
    form: dict[str, Any] = dict(raw)
    form["conflicts_with"] = [str(value) for value in raw.getlist("conflicts_with")]
    service = _service(request)

    def _update(document: ScenarioDocument) -> None:
        groups = document.map.signal_groups
        _check_row(draft_id, form, groups, index, "name", renamed_by="name")
        service.update_signal_group(document, index, form)

    return _apply(request, draft_id, "environment", _update, form=form)


@router.post(
    "/draft/{draft_id}/signal-group/{index}/delete", response_class=HTMLResponse
)
async def delete_signal_group(
    request: Request, draft_id: str, index: int
) -> HTMLResponse:
    """Remove a group, and every phase state and conflict naming it."""
    form = await request.form()
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "environment",
        lambda doc: service.delete_signal_group(doc, index),
        form=form,
        strict=True,
    )


@router.post("/draft/{draft_id}/signal-controller", response_class=HTMLResponse)
def add_signal_controller(request: Request, draft_id: str) -> HTMLResponse:
    """Declare one more junction cycle."""
    service = _service(request)
    return _apply(
        request, draft_id, "environment", lambda doc: service.add_signal_controller(doc)
    )


@router.post("/draft/{draft_id}/signal-controller/{index}", response_class=HTMLResponse)
async def update_signal_controller(
    request: Request, draft_id: str, index: int
) -> HTMLResponse:
    """Apply one controller's name and offset."""
    form = dict(await request.form())
    service = _service(request)

    def _update(document: ScenarioDocument) -> None:
        controllers = document.map.traffic_signal_controllers
        _check_row(draft_id, form, controllers, index, "name", renamed_by="name")
        service.update_signal_controller(document, index, form)

    return _apply(request, draft_id, "environment", _update, form=form)


@router.post(
    "/draft/{draft_id}/signal-controller/{index}/delete", response_class=HTMLResponse
)
async def delete_signal_controller(
    request: Request, draft_id: str, index: int
) -> HTMLResponse:
    """Remove a controller, and every offset and card that named it."""
    form = await request.form()
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "environment",
        lambda doc: service.delete_signal_controller(doc, index),
        form=form,
        strict=True,
    )


@router.post(
    "/draft/{draft_id}/signal-controller/{index}/phase", response_class=HTMLResponse
)
async def add_signal_phase(request: Request, draft_id: str, index: int) -> HTMLResponse:
    """Add a step to a cycle."""
    form = await request.form()
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "environment",
        lambda doc: service.add_signal_phase(doc, index),
        form=form,
        strict=True,
    )


@router.post(
    "/draft/{draft_id}/signal-controller/{index}/phase/{phase_index}",
    response_class=HTMLResponse,
)
async def update_signal_phase(
    request: Request, draft_id: str, index: int, phase_index: int
) -> HTMLResponse:
    """Apply one phase's name, duration and colours."""
    form = dict(await request.form())
    service = _service(request)

    def _update(document: ScenarioDocument) -> None:
        controllers = document.map.traffic_signal_controllers
        # Phases are named per controller, so the controller is named too:
        # after one before it is removed, index and phase name alike may be
        # another controller's.
        _check_row(draft_id, form, controllers, index, "name", field=PARENT_ROW_FIELD)
        if 0 <= index < len(controllers):
            phases = controllers[index].phases
            _check_row(draft_id, form, phases, phase_index, "name", renamed_by="name")
        service.update_signal_phase(document, index, phase_index, form)

    return _apply(request, draft_id, "environment", _update, form=form)


@router.post(
    "/draft/{draft_id}/signal-controller/{index}/phase/{phase_index}/delete",
    response_class=HTMLResponse,
)
async def delete_signal_phase(
    request: Request, draft_id: str, index: int, phase_index: int
) -> HTMLResponse:
    """Remove one step of a cycle."""
    form = await request.form()
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "environment",
        lambda doc: service.delete_signal_phase(doc, index, phase_index),
        form=form,
        strict=True,
    )


@router.post(
    "/draft/{draft_id}/signal-controller/{index}/phase/{phase_index}/move",
    response_class=HTMLResponse,
)
async def move_signal_phase(
    request: Request, draft_id: str, index: int, phase_index: int
) -> HTMLResponse:
    """Shift a phase along its cycle.  Order is semantics here, not layout."""
    form = await request.form()
    delta = _delta(form)
    service = _service(request)
    return _apply(
        request,
        draft_id,
        "environment",
        lambda doc: service.move_signal_phase(doc, index, phase_index, delta),
        form=form,
        strict=True,
    )


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/condition", response_class=HTMLResponse)
async def add_condition(request: Request, draft_id: str) -> HTMLResponse:
    """Add a condition to an action trigger, a composition, or an assertion."""
    form = dict(await request.form())
    slot = str(form.get("slot", ""))
    if "vertex" in form:
        # The waypoint-condition form asks for the vertex beside the type.
        slot = f"{slot}:{str(form['vertex']).strip()}"
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
        form=form,
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
# Lanelet search preview
# ---------------------------------------------------------------------------


@router.post("/draft/{draft_id}/lanelet-preview", response_class=HTMLResponse)
async def lanelet_preview(request: Request, draft_id: str) -> HTMLResponse:
    """Count the lanelets one field's constraints match.

    The readout beside the map in the lanelet picker: the map itself is already
    open there, and is handed the matching ids to outline.

    The evaluation runs in a worker thread: with ``load_map`` set it parses the
    whole Lanelet2 map -- seconds for a city -- and run inline that would stall
    the event loop, so no other tab would be served until it finished.  The
    draft is read from disk per request, so the document the thread evaluates
    is this request's own copy and shares nothing with concurrent edits.
    """
    form = dict(await request.form())
    slot_key = str(form.get("slot", ""))
    load_map = is_checked(form, "load_map")

    draft = _service(request).require_draft(draft_id)
    slot = draft.document.lanelet_slot(slot_key)
    if slot is None:
        result = map_preview.PreviewResult(
            error=f"No lanelet field named {slot_key!r}."
        )
    else:
        result = await run_in_threadpool(
            map_preview.evaluate_slot, draft.document, slot, load_map=load_map
        )
    return _templates(request).TemplateResponse(
        request=request,
        name="partials/lanelet_preview.html",
        context={
            "draft": draft,
            "document": draft.document,
            "slot": slot,
            "slot_key": slot_key,
            "preview": result,
        },
    )


# ---------------------------------------------------------------------------
# The scenario on the map
# ---------------------------------------------------------------------------


@router.get(
    "/draft/{draft_id}/map-view",
    response_class=HTMLResponse,
    # With load_map it parses a city; without, it still binds on a cached map.
    dependencies=[Depends(refuse_cross_site)],
)
def map_view(
    request: Request, draft_id: str, pattern: int = 0, load_map: bool = False
) -> HTMLResponse:
    """Render the map panel above the timeline: every place the scenario names.

    ``pattern`` binds one match of the searched lanelet, so an abstract
    scenario is drawn as one of the concrete runs a sweep would perform rather
    than as the set it stands for.  It is carried in this panel's own refresh
    URL, which is re-rendered with the answer, so stepping through patterns
    survives an edit somewhere else on the page.

    ``load_map`` is what a first binding costs -- parsing a city -- and is
    therefore asked for rather than assumed; once the map is cached every later
    render binds without it.  Only the panel itself (htmx, same-origin) asks
    for this route, so a request another site sends is refused with 403 rather
    than made to parse a map (#1850).
    """
    draft = _service(request).require_draft(draft_id)
    view = scenario_map.build_map_view(
        draft.document, pattern=pattern, load_map=load_map
    )
    return _templates(request).TemplateResponse(
        request=request,
        name="partials/scenario_map.html",
        context={
            "draft": draft,
            "document": draft.document,
            "view": view,
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
    with service.editing(draft_id):
        draft = service.require_draft(draft_id)
        # Every edit is saved as it is made, so pressing Save Draft usually
        # changes nothing; it then writes nothing and leaves the revision, which
        # its answer -- a status line, not the forms -- would not renumber.
        as_read = service.content(draft)
        title = str(form.get("title", "")).strip()
        if title:
            draft.document.title = title
        service.save(draft, _client(request), unless_same_as=as_read)
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
    """Export the draft as a wheelhouse, ready to download.

    What is handed back is every wheel the scenario needs, so that installing it
    where it runs is ``pip install --no-index --find-links`` and nothing else --
    no uv, no git, no network.  The uv project the wheels were resolved from is
    a build input and does not survive the request.

    It is built and zipped server-side but is not left there: the editor is
    routinely used from another machine on the LAN, where "written to
    ./scenario_packages" means written somewhere the person exporting cannot
    reach.  The response is the report -- warnings, the tool log, whether the
    package's tests passed -- with a link to the archive, because throwing that
    away to make the response a file download would hide the very things an
    export is checked for.

    Declared ``async`` but doing its work through a blocking call would stall
    the event loop for the length of a dependency resolution, so the export runs
    in a worker thread.
    """
    form = dict(await request.form())
    service = _service(request)
    draft = service.require_draft(draft_id)

    # Locking is not offered as a choice: a wheelhouse *is* the lockfile
    # resolved into wheels, so an export without one has nothing to hand back.
    options = {
        "dev_mode": is_checked(form, "dev_mode"),
        "verify": is_checked(form, "verify"),
        "run_tests": is_checked(form, "run_tests"),
    }

    def _run() -> tuple[Any, str, str]:
        try:
            result = service.export_archive(draft, **options)
        except PackageExportError as exc:
            logger.warning("Scenario wheelhouse export failed: %s", exc)
            # The exporter attaches the failing tool's output to the exception.
            return None, str(exc), exc.log
        return result, "", result.log

    result, error, log = await run_in_threadpool(_run)

    return _templates(request).TemplateResponse(
        request=request,
        name="partials/export_result.html",
        context={
            "draft": draft,
            "result": result,
            "error": error,
            "log": log,
        },
    )


#: How much of a staged archive one chunk of the download reads.
_DOWNLOAD_CHUNK = 1024 * 1024


def _attachment(filename: str) -> str:
    """Return the ``content-disposition`` naming a download *filename*, as
    Starlette's ``FileResponse`` writes it."""
    quoted = quote(filename)
    if quoted != filename:
        return f"attachment; filename*=utf-8''{quoted}"
    return f'attachment; filename="{filename}"'


class _ArchiveResponse(StreamingResponse):
    """Stream an archive already opened, and close it however the response ends.

    Starlette's ``FileResponse`` takes a path and opens it only once the
    response is sent, by when the archive may be gone (#1719); this serves the
    handle :meth:`EditorService.open_staged_archive` opened under the archive
    lock, which stays readable after an unlink.  The handle is closed when the
    response finishes, fails or the client disconnects -- neither the body
    iterator nor a background task is guaranteed to run to the end on a
    disconnect, so it is closed around the whole of ``__call__``.
    """

    def __init__(self, archive: StagedArchive) -> None:
        self._archive = archive
        super().__init__(
            self._chunks(archive),
            media_type="application/zip",
            headers={
                "content-length": str(archive.size),
                "content-disposition": _attachment(archive.name),
            },
        )

    @staticmethod
    def _chunks(archive: StagedArchive) -> Iterator[bytes]:
        """Yield *archive* a chunk at a time."""
        while chunk := archive.file.read(_DOWNLOAD_CHUNK):
            yield chunk

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._archive.file.close()


@router.get("/draft/{draft_id}/wheelhouse.zip")
def download_wheelhouse(request: Request, draft_id: str) -> Response:
    """Serve the wheelhouse the last export staged for this draft.

    The archive is opened before the response is built, under the lock that
    staging and deleting it hold, so an archive removed after the link was
    rendered is a 404 rather than a 500 from a path that no longer exists, and
    one removed while it downloads still downloads whole (#1719).

    Raises:
        EditorError: If nothing has been exported yet, or the archive has since
            been cleared -- a stale link is a normal thing to click.
    """
    service = _service(request)
    draft = service.require_draft(draft_id)
    archive = service.open_staged_archive(draft)
    if archive is None:
        raise EditorError("No exported wheelhouse to download. Run the export again.")
    return _ArchiveResponse(archive)


def _delta(form: Mapping[str, Any]) -> int:
    """Return the ``delta`` a move form submitted: a whole number, else 1."""
    try:
        return int(str(form.get("delta") or 1))
    except ValueError:
        return 1
