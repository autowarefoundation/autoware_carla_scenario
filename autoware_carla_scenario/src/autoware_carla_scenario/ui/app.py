"""FastAPI application and route definitions for the scenario result viewer."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import cases, recordings, results, runner, scanner
from .models import RunProgress

logger = logging.getLogger(__name__)

_UI_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _UI_DIR / "templates"
_STATIC_DIR = _UI_DIR / "static"

app = FastAPI(title="Scenario Result Viewer")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Base path for multirun/ and outputs/ directories.
# Resolved at startup; can be overridden via environment variable.
BASE_PATH = Path.cwd()


def _base_path() -> Path:
    """Return the base path for log directories."""
    return BASE_PATH


def _resolve_scenario_names(scenario: str) -> list[str]:
    """Expand a scenario pattern to concrete config names.

    If *scenario* contains glob metacharacters the pattern is matched
    against available configs.  Otherwise the name is returned as-is.
    """
    if any(ch in scenario for ch in ("*", "?", "[")):
        import fnmatch  # noqa: PLC0415

        all_names = scanner.list_scenario_configs()
        return [n for n in all_names if fnmatch.fnmatch(n, scenario)]
    return [scenario]


# ── Page routes ──────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Session list page."""
    sessions = scanner.scan_sessions(_base_path())
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "sessions": sessions,
            "page": "index",
        },
    )


@app.get("/session/{session_type}/{date}/{time}", response_class=HTMLResponse)
async def session_detail(
    request: Request, session_type: str, date: str, time: str
) -> HTMLResponse:
    """Session detail page showing all scenarios in the session."""
    items = scanner.load_session(_base_path(), session_type, date, time)
    passed_count = sum(1 for it in items if it.passed is True)
    total_count = len(items)
    return templates.TemplateResponse(
        request=request,
        name="session.html",
        context={
            "items": items,
            "session_type": session_type,
            "date": date,
            "time": time,
            "passed_count": passed_count,
            "total_count": total_count,
            "page": "session",
        },
    )


@app.get(
    # :int, so /session/.../odd is the ODD view's.
    "/session/{session_type}/{date}/{time}/{index:int}",
    response_class=HTMLResponse,
)
async def scenario_detail(
    request: Request, session_type: str, date: str, time: str, index: int
) -> HTMLResponse:
    """Individual scenario detail page with condition tree."""
    result = scanner.load_scenario(_base_path(), session_type, date, time, index)
    # Also load session items for navigation context
    items = scanner.load_session(_base_path(), session_type, date, time)
    scenario_name = ""
    for it in items:
        if it.index == index:
            scenario_name = it.scenario_name
            break
    return templates.TemplateResponse(
        request=request,
        name="scenario.html",
        context={
            "result": result,
            "scenario_name": scenario_name,
            "session_type": session_type,
            "date": date,
            "time": time,
            "index": index,
            "page": "scenario",
            "recording": _recording_context(session_type, date, time, index),
        },
    )


def _case_path(session_type: str, date: str, time: str, index: int) -> str:
    return f"{session_type}/{date}/{time}/{index}"


def _recording_context(
    session_type: str, date: str, time: str, index: int
) -> dict[str, Any] | None:
    """What the scenario page shows of the case's Rerun recording, or
    ``None`` when the case has no trajectory log."""
    files = cases.case_files(_base_path(), session_type, date, time, index)
    if files is None or not files.trajectory.is_file():
        return None
    made = recordings.recording_path(files.trajectory)
    case = _case_path(session_type, date, time, index)
    return {
        "trajectory": str(files.trajectory.resolve()),
        "rrd": str(made.resolve()) if made.is_file() else None,
        "converter": recordings.converter_available(),
        "install_hint": recordings.INSTALL_HINT,
        "make_url": f"/api/recording/{case}",
        "rrd_url": f"/recording/{case}.rrd",
        "viewer": recordings.web_viewer(),
    }


# ── API routes ───────────────────────────────────────────────────────────


@app.post("/api/refresh", response_class=HTMLResponse)
async def refresh(request: Request) -> HTMLResponse:
    """Clear cache and return updated session table partial."""
    scanner.clear_cache()
    sessions = scanner.scan_sessions(_base_path())
    return templates.TemplateResponse(
        request=request,
        name="partials/session_table.html",
        context={"sessions": sessions},
    )


@app.get("/api/scenarios")
async def list_scenarios() -> list[str]:
    """Return available scenario config names from the conf directory."""
    return scanner.list_scenario_configs()


@app.post("/api/run/preview")
async def run_preview(request: Request) -> dict[str, str]:
    """Return the command that would be executed, for preview."""
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass

    scenario: str = body.get("scenario", "*/*")
    extra_overrides: list[str] = body.get("extra_overrides", [])
    sweeper: str = body.get("sweeper", "")

    cmd, note = runner.build_command(scenario, extra_overrides, sweeper)
    preview = " ".join(cmd)
    if note:
        preview += f"  # {note}"
    return {"command": preview}


@app.post("/api/run")
async def run_scenarios(request: Request) -> dict[str, str]:
    """Start scenario execution in background.

    The request body specifies the scenario pattern and options
    explicitly from the Run Options panel.
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass

    scenario: str = body.get("scenario", "*/*")
    extra_overrides: list[str] = body.get("extra_overrides", [])
    sweeper: str = body.get("sweeper", "")

    if sweeper:
        # Resolve sweep constraints in-process (lightweight, no CARLA).
        # This expands the scenario + constraints into concrete per-job
        # override lists, which are then run as individual subprocesses.
        from .sweep_resolver import resolve_sweep  # noqa: PLC0415

        scenario_names = _resolve_scenario_names(scenario)
        if not scenario_names:
            return {"status": "error", "message": f"No scenarios match '{scenario}'"}

        overrides_list: list[list[str]] = []
        for name in scenario_names:
            try:
                batches = resolve_sweep(name, extra_overrides)
                overrides_list.extend(batches)
            except Exception as exc:  # noqa: BLE001
                logger.error("Sweep resolution failed for %s: %s", name, exc)

        if not overrides_list:
            return {
                "status": "error",
                "message": "No jobs generated (no lanelets matched constraints)",
            }

        # Jobs already contain scenario= and sweep overrides;
        # run without sweeper flag (each is a plain single-scenario run).
        # Group results under a shared multirun/ directory.
        runner.start_run(
            overrides_list,
            base_path=_base_path(),
            extra_overrides=[],
            group_as_multirun=True,
        )
    else:
        # No sweeper: expand globs into individual jobs so progress
        # tracking reports each scenario separately (e.g. [2/5]).
        scenario_names = _resolve_scenario_names(scenario)
        if not scenario_names:
            return {"status": "error", "message": f"No scenarios match '{scenario}'"}
        overrides_list = [[f"scenario={name}"] for name in scenario_names]
        runner.start_run(
            overrides_list,
            base_path=_base_path(),
            extra_overrides=extra_overrides,
            group_as_multirun=len(scenario_names) > 1,
        )

    return {"status": "started"}


@app.get("/api/run/status")
async def run_status() -> dict[str, Any]:
    """Return current run status without SSE streaming."""
    running, progress = runner.get_status()
    return {
        "running": running,
        "progress": progress.model_dump() if progress else None,
    }


@app.get("/api/run/progress")
async def run_progress() -> StreamingResponse:
    """SSE endpoint for run progress updates."""

    async def event_stream() -> Any:
        while True:
            progress = runner.get_progress()
            if progress is None:
                # Not running; send idle event and close
                data = json.dumps(
                    RunProgress(
                        current=0, total=0, scenario_name="", status="idle"
                    ).model_dump()
                )
                yield f"data: {data}\n\n"
                break

            data = json.dumps(progress.model_dump())
            yield f"data: {data}\n\n"

            if progress.status == "done":
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Rerun recordings ─────────────────────────────────────────────────────


def _trajectory_of(session_type: str, date: str, time: str, index: int) -> Path:
    files = cases.case_files(_base_path(), session_type, date, time, index)
    if files is None or not files.trajectory.is_file():
        raise HTTPException(status_code=404, detail="This case has no trajectory log.")
    return files.trajectory


async def _recording_of(trajectory: Path) -> Path:
    """The case's recording, made in a worker thread if need be (the
    converter releases the GIL)."""
    try:
        return await asyncio.to_thread(recordings.ensure_recording, trajectory)
    except recordings.ConverterMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (recordings.RecordingError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/recording/{session_type}/{date}/{time}/{index}")
async def make_recording(
    session_type: str, date: str, time: str, index: int
) -> JSONResponse:
    """Make the case's Rerun recording if there is none yet; where it is.

    404 without a trajectory log, 503 (with the install hint) when a recording
    has to be made and the converter is not installed, 500 when it fails.
    """
    trajectory = _trajectory_of(session_type, date, time, index)
    try:
        path = await _recording_of(trajectory)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "missing_converter": exc.status_code == 503,
                "install_hint": recordings.INSTALL_HINT,
            },
        )
    return JSONResponse(
        {
            "url": f"/recording/{_case_path(session_type, date, time, index)}.rrd",
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
        }
    )


@app.get("/recording/{session_type}/{date}/{time}/{index}.rrd")
async def recording_file(
    session_type: str, date: str, time: str, index: int
) -> FileResponse:
    """The case's Rerun recording (made on the way if need be)."""
    trajectory = _trajectory_of(session_type, date, time, index)
    path = await _recording_of(trajectory)
    return FileResponse(
        path,
        media_type="application/octet-stream",
        # Its name beside the log, also when it is kept in the cache instead.
        filename=recordings.recording_path(trajectory).name,
        content_disposition_type="inline",
    )


#: The web viewer's files, by the name the page asks for: the file, and its type.
#: ``index.js`` imports ``./re_viewer`` with no extension (as a bundler would
#: resolve it), which is ``re_viewer.js``.
_WEB_VIEWER_FILES = {
    "index.js": ("index.js", "text/javascript"),
    "re_viewer": ("re_viewer.js", "text/javascript"),
    "re_viewer.js": ("re_viewer.js", "text/javascript"),
    "re_viewer_bg.wasm": ("re_viewer_bg.wasm", "application/wasm"),
    "package.json": ("package.json", "application/json"),
}


@app.get(recordings.LOCAL_ROUTE + "/{version}/{name}")
async def web_viewer_file(version: str, name: str) -> FileResponse:
    """A file of the web viewer this app serves itself (``recordings.web_viewer``)."""
    viewer = recordings.web_viewer()
    if (
        viewer.source != "local"
        or viewer.directory is None
        or version != viewer.version
        or name not in _WEB_VIEWER_FILES
    ):
        raise HTTPException(status_code=404)
    file_name, media_type = _WEB_VIEWER_FILES[name]
    return FileResponse(
        viewer.directory / file_name,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# ── ODD results and comparison ───────────────────────────────────────────


def _session_ref(session_type: str, date: str, time: str) -> str:
    return f"{session_type}/{date}/{time}"


def _parse_ref(ref: str) -> tuple[str, str, str] | None:
    parts = ref.strip("/").split("/")
    if len(parts) != 3:
        return None
    session_type, date, time = parts
    if cases.session_dir(_base_path(), session_type, date, time) is None:
        return None
    return session_type, date, time


def _family(results_: list[results.CaseResult]) -> set[str]:
    return {r.scenario for r in results_}


def _openodd() -> Any:
    import openodd  # noqa: PLC0415 -- a Rust extension, loaded on first use

    return openodd


def _attribute(value: str | None, attributes: list[str]) -> str | None:
    return value if value and value in attributes else None


@app.get("/session/{session_type}/{date}/{time}/odd", response_class=HTMLResponse)
async def session_odd(
    request: Request, session_type: str, date: str, time: str
) -> HTMLResponse:
    """A session's results on its ODD: openodd's coverage matrix (cells
    coloured by what the runs in them did; a click drills into a cell), its
    plot of the drawn cases, and the cases in the cell drilled into."""
    if cases.session_dir(_base_path(), session_type, date, time) is None:
        raise HTTPException(status_code=404)
    found = await asyncio.to_thread(
        results.load_results, _base_path(), session_type, date, time
    )
    documents = results.coverage_documents(found)
    openodd = _openodd()
    conditions = openodd.view.parse_conditions(request.query_params.getlist("at"))
    ref = _session_ref(session_type, date, time)
    link = f"/session/{ref}/odd?{{conditions}}"
    matrix_html = plot_html = ""
    if documents:
        view = openodd.CoverageView(documents)
        dx, dy = results.default_axes(found)
        x = _attribute(request.query_params.get("x"), view.attributes)
        y = _attribute(request.query_params.get("y"), view.attributes)
        if x is None and y is None and not conditions:
            x, y = _attribute(dx, view.attributes), _attribute(dy, view.attributes)
        matrix_html = view.html(
            x=x, y=y, conditions=conditions, id="odd-matrix", link=link
        )
        plot = openodd.CasePlot(documents)
        if plot.cases:
            px = _attribute(x or dx, plot.attributes)
            py = _attribute(y or dy, plot.attributes)
            plot_html = plot.html(
                x=px, y=py, id="odd-cases", link=f"/session/{ref}/odd?{{axes}}"
            )
    in_cell = results.cases_in_cell(found, conditions)
    return templates.TemplateResponse(
        request=request,
        name="odd.html",
        context={
            "page": "odd",
            "session_type": session_type,
            "date": date,
            "time": time,
            "ref": ref,
            "results": found,
            "in_cell": in_cell,
            "varying": results.varying_attributes(found),
            "conditions": conditions,
            "matrix_html": matrix_html,
            "plot_html": plot_html,
            "oo_styles": openodd.view.styles(),
            "oo_script": openodd.view.script(),
            "metric_names": results.METRICS,
        },
    )


@app.get("/compare", response_class=HTMLResponse)
async def compare_sessions(request: Request) -> HTMLResponse:
    """Sessions of one scenario family compared: pass rates, each case's
    flip (matched by what it ran, not by position), metric deltas, and the
    pass rate of each ODD cell. ``?s=type/date/time`` per session (two or
    more; ordered oldest first, the first being the baseline)."""
    refs = [r for r in (_parse_ref(v) for v in request.query_params.getlist("s")) if r]
    refs = sorted(dict.fromkeys(refs), key=lambda r: (r[1], r[2]))
    loaded = [
        (
            _session_ref(*ref),
            await asyncio.to_thread(results.load_results, _base_path(), *ref),
        )
        for ref in refs
    ]
    family: set[str] = set()
    for _, found in loaded:
        family |= _family(found)
    # The sessions offered: those that ran a scenario of the family (all when
    # none is picked yet).
    offered = []
    for summary in scanner.scan_sessions(_base_path()):
        names = {n.strip().split("#", 1)[0] for n in summary.scenario_name.split(",")}
        if not family or names & family:
            offered.append(
                (
                    _session_ref(summary.session_type, summary.date, summary.time),
                    summary,
                )
            )
    comparison = None
    matrices: list[tuple[str, str]] = []
    if len(loaded) >= 2:
        every = [r for _, found in loaded for r in found]
        dx, dy = results.default_axes(every)
        x = request.query_params.get("x") or dx
        y = request.query_params.get("y") or dy
        comparison = results.compare(loaded, x=x, y=y)
        openodd = _openodd()
        for ref, found in loaded:
            documents = results.coverage_documents(found)
            if not documents:
                matrices.append((ref, ""))
                continue
            view = openodd.CoverageView(documents)
            matrices.append(
                (
                    ref,
                    view.html(
                        x=_attribute(x, view.attributes),
                        y=_attribute(y, view.attributes),
                        id=f"cmp-{len(matrices)}",
                        link=f"/session/{ref}/odd?{{conditions}}",
                    ),
                )
            )
    attributes = sorted({a for _, found in loaded for r in found for a in r.buckets})
    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={
            "page": "compare",
            "selected": [ref for ref, _ in loaded],
            "offered": offered,
            "comparison": comparison,
            "matrices": matrices,
            "attributes": attributes,
            "metric_names": results.METRICS,
            "oo_styles": _openodd().view.styles() if matrices else "",
            "oo_script": _openodd().view.script() if matrices else "",
        },
    )


@app.get("/compare/runs", response_class=HTMLResponse)
async def compare_runs(request: Request) -> HTMLResponse:
    """The same case's runs side by side, each in a Rerun web viewer:
    ``?run=type/date/time/index`` per run."""
    runs: list[dict[str, Any]] = []
    for value in request.query_params.getlist("run")[:4]:
        ref, _, index = value.strip("/").rpartition("/")
        parsed = _parse_ref(ref)
        if parsed is None or not index.isdigit():
            continue
        found = await asyncio.to_thread(
            results.load_results, _base_path(), *parsed, metrics=False
        )
        case = next((r for r in found if r.index == int(index)), None)
        runs.append(
            {
                "ref": ref,
                "index": int(index),
                "case": case,
                "recording": _recording_context(*parsed, int(index)),
            }
        )
    return templates.TemplateResponse(
        request=request,
        name="compare_runs.html",
        context={
            "page": "compare_runs",
            "runs": runs,
            "viewer": recordings.web_viewer(),
            "converter": recordings.converter_available(),
            "install_hint": recordings.INSTALL_HINT,
        },
    )
