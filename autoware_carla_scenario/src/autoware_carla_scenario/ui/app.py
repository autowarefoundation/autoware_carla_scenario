"""FastAPI application and route definitions for the scenario result viewer."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
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

from ..scenario_glob import is_scenario_glob, resolve_scenario_glob
from . import cases, recordings, results, runner, scanner, slurm_launch
from .models import RunProgress, SessionSummary

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


def _session_dir_or_404(session_type: str, date: str, time: str) -> Path:
    """The session's directory, or 404 when the URL's names do not make one
    under the base (:func:`cases.session_dir`: an unknown type, ``..``, a
    separator, or a path that resolves outside ``multirun/`` / ``outputs/``)."""
    directory = cases.session_dir(_base_path(), session_type, date, time)
    if directory is None:
        raise HTTPException(status_code=404)
    return directory


def _resolve_scenario_names(scenario: str) -> list[str]:
    """Expand a scenario pattern to concrete config names.

    If *scenario* contains glob metacharacters
    (:func:`~autoware_carla_scenario.scenario_glob.is_scenario_glob`, the
    CLI's test) it is matched as the
    ``scenario`` CLI matches it
    (:func:`~autoware_carla_scenario.scenario_glob.resolve_scenario_glob`
    over :func:`scanner.scenario_conf_dirs`): ``*`` within one path
    component, a trailing ``**`` everything below, the names sorted as
    strings, ``[]`` when nothing matches.  Otherwise the name is returned
    as-is, as the CLI runs it.

    Raises:
        ValueError: *scenario* is not a valid pattern (a ``**`` that is not a
            whole path component); :func:`run_scenarios` answers it with an
            error response.
    """
    if is_scenario_glob(scenario):
        return resolve_scenario_glob(scenario, scanner.scenario_conf_dirs())
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
    directory = _session_dir_or_404(session_type, date, time)
    items = scanner.load_session(_base_path(), session_type, date, time)
    passed_count = sum(1 for it in items if it.passed is True)
    total_count = len(items)
    # A SLURM batch's cases without an outcome yet.
    pending_count = sum(1 for it in items if it.passed is None)
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
            "pending_count": pending_count,
            "page": "session",
            "slurm": session_type == "batch"
            and (directory / "slurm" / "manifest.json").is_file(),
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
    _session_dir_or_404(session_type, date, time)
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

    if body.get("launcher") == "slurm":
        try:
            options = slurm_launch.SlurmOptions.from_request(body.get("slurm"))
        except ValueError as exc:
            return {"command": f"# {exc}"}
        cmd = slurm_launch.build_command(scenario, extra_overrides, sweeper, options)
        return {"command": " ".join(shlex.quote(c) for c in cmd)}

    cmd, note = runner.build_command(scenario, extra_overrides, sweeper)
    preview = " ".join(cmd)
    if note:
        preview += f"  # {note}"
    return {"command": preview}


#: The answer to a run refused because another is still in progress.
_BUSY = {"status": "busy", "message": "A scenario run is already in progress"}


@app.post("/api/run", response_model=None)
async def run_scenarios(request: Request) -> dict[str, str] | JSONResponse:
    """Start scenario execution in background.

    The request body specifies the scenario pattern and options
    explicitly from the Run Options panel.  A local run is refused with
    409 (``{"status": "busy"}``) while another is in progress.
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        pass

    scenario: str = body.get("scenario", "*/*")
    extra_overrides: list[str] = body.get("extra_overrides", [])
    sweeper: str = body.get("sweeper", "")

    if body.get("launcher") == "slurm":
        return _submit_slurm(scenario, extra_overrides, sweeper, body.get("slurm"))

    try:
        scenario_names = _resolve_scenario_names(scenario)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    if not scenario_names:
        return {"status": "error", "message": f"No scenarios match '{scenario}'"}

    if sweeper:
        # Expand each scenario's sweep in-process (no CARLA), as
        # scenario-batch does (sweeper.expand.expand_config): constraints,
        # route searches and ODD draws alike.  Each job is then run as a
        # plain single-scenario run.
        from ..odd.sampler import OddDraws  # noqa: PLC0415
        from .sweep_resolver import SweepResolutionError, resolve_sweep  # noqa: PLC0415

        overrides_list: list[list[str]] = []
        errors: list[str] = []
        # Scenarios on one ODD go on from each other's draws.
        drawn = OddDraws()
        for name in scenario_names:
            try:
                overrides_list.extend(resolve_sweep(name, extra_overrides, drawn=drawn))
            except SweepResolutionError as exc:
                logger.error("Sweep resolution failed for %s: %s", name, exc)
                errors.append(str(exc))

        if errors:
            # Nothing runs: a run missing some of the scenarios asked for
            # would look like a complete one.
            return {
                "status": "error",
                "message": "Sweep resolution failed: " + "; ".join(errors),
            }
        if not overrides_list:
            return {
                "status": "error",
                "message": "No jobs generated: the sweep matched no lanelets "
                "or routes",
            }

        # The jobs were resolved with the extra overrides, and run with them
        # too, after each job's own (as /api/run/preview shows, and as
        # scenario-batch orders them).  Group results under one multirun/.
        started = runner.start_run(
            overrides_list,
            base_path=_base_path(),
            extra_overrides=extra_overrides,
            group_as_multirun=True,
        )
    else:
        # No sweeper: expand globs into individual jobs so progress
        # tracking reports each scenario separately (e.g. [2/5]).
        overrides_list = [[f"scenario={name}"] for name in scenario_names]
        started = runner.start_run(
            overrides_list,
            base_path=_base_path(),
            extra_overrides=extra_overrides,
            group_as_multirun=len(scenario_names) > 1,
        )

    if not started:
        return JSONResponse(status_code=409, content=_BUSY)
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


def _scenarios(summary: SessionSummary) -> set[str]:
    """The scenarios a session ran, named as :func:`_family` names them: a
    batch's entries name them as its summary does; a multirun or single
    session's are its ``scenario=`` override, which its summary (the result
    file's stem) does not give, so they are read from its overrides
    (:func:`results.session_scenarios`)."""
    if summary.session_type == "batch":
        return {n.strip().split("#", 1)[0] for n in summary.scenario_name.split(",")}
    return results.session_scenarios(
        _base_path(), summary.session_type, summary.date, summary.time
    )


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
    _session_dir_or_404(session_type, date, time)
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
    # The sessions offered: those picked, and those that ran a scenario of the
    # family (all when none is picked yet). Only scanned sessions are offered,
    # so a picked ref is offered only when it names a real session. The scan
    # stays on the event loop, as every other page's does (its cache is not
    # locked); only the reading of the sessions' overrides is threaded.
    selected = {ref for ref, _ in loaded}
    sessions = scanner.scan_sessions(_base_path())

    def offer() -> list[tuple[str, SessionSummary]]:
        out = []
        for summary in sessions:
            ref = _session_ref(summary.session_type, summary.date, summary.time)
            if not family or ref in selected or _scenarios(summary) & family:
                out.append((ref, summary))
        return out

    offered = await asyncio.to_thread(offer)
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
        # isascii(): "²".isdigit() is true, and int() refuses it.
        if parsed is None or not (index.isascii() and index.isdigit()):
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


# ── SLURM ────────────────────────────────────────────────────────────────


def _submit_slurm(
    scenario: str, extra_overrides: list[str], sweeper: str, raw: Any
) -> dict[str, str]:
    """Submit the batch through SLURM (``launcher=slurm --no-wait``)."""
    available, reason = slurm_launch.availability()
    if not available:
        return {"status": "error", "message": reason}
    try:
        options = slurm_launch.SlurmOptions.from_request(
            raw if isinstance(raw, dict) else {}
        )
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    command = slurm_launch.build_command(scenario, extra_overrides, sweeper, options)
    try:
        slurm_launch.start_submit(command, _base_path())
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc)}
    return {"status": "submitting"}


@app.get("/api/launchers")
async def launchers() -> dict[str, Any]:
    """Which launchers a run can use here: ``local`` always, ``slurm`` where
    sbatch is (with the partitions sinfo lists), else why not."""
    available, reason = slurm_launch.availability()
    return {
        "local": {"available": True},
        "slurm": {
            "available": available,
            "reason": reason,
            "partitions": await asyncio.to_thread(slurm_launch.partitions)
            if available
            else [],
        },
    }


def _session_of(batch_dir: str | None) -> str | None:
    """``/session/batch/<date>/<time>`` of a batch directory under the base."""
    if not batch_dir:
        return None
    try:
        rel = Path(batch_dir).resolve().relative_to(_base_path().resolve() / "outputs")
    except ValueError:
        return None
    parts = rel.parts
    return f"/session/batch/{parts[0]}/{parts[1]}" if len(parts) == 2 else None


@app.get("/api/slurm/submission")
async def slurm_submission() -> dict[str, Any]:
    """The batch this viewer last submitted: ``submitting``, ``submitted``
    (with its session's page) or ``failed`` (with the output)."""
    record = slurm_launch.submission()
    if record is None:
        return {"state": "none"}
    out = record.to_dict()
    out["session_url"] = _session_of(record.batch_dir)
    if record.state != "submitting":
        scanner.clear_cache()
    return out


def _slurm_batch(date: str, time: str) -> Path:
    directory = cases.session_dir(_base_path(), "batch", date, time)
    if directory is None or not (directory / "slurm" / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail="Not a SLURM batch")
    return directory


@app.get("/api/slurm/{date}/{time}/status")
async def slurm_status(date: str, time: str) -> dict[str, Any]:
    """A SLURM batch's cases, jobs and tasks (``squeue``)."""
    status = await asyncio.to_thread(
        slurm_launch.batch_status, _slurm_batch(date, time)
    )
    if status is None:
        raise HTTPException(status_code=404, detail="Not a SLURM batch")
    available, reason = slurm_launch.availability()
    status["slurm_available"] = available
    status["slurm_reason"] = reason
    return status


@app.post("/api/slurm/{date}/{time}/{action}")
async def slurm_action(
    request: Request, date: str, time: str, action: str
) -> JSONResponse:
    """``scenario-slurm resubmit|collect|cancel`` on the batch.

    ``resubmit`` takes ``{"failed": bool, "array_parallelism": n,
    "cases_per_task": n}`` and submits without waiting.
    """
    directory = _slurm_batch(date, time)
    if action not in ("resubmit", "collect", "cancel"):
        raise HTTPException(status_code=404)
    if action != "collect":
        available, reason = slurm_launch.availability()
        if not available:
            return JSONResponse(
                status_code=503, content={"ok": False, "output": reason}
            )
    args = [action, str(directory)]
    if action == "resubmit":
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 -- no body: the defaults
            pass
        if body.get("failed"):
            args.append("--failed")
        for key in ("array_parallelism", "cases_per_task"):
            value = body.get(key)
            if value not in (None, ""):
                text = str(value)
                if not (text.isascii() and text.isdigit()) or int(text) < 1:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "ok": False,
                            "output": f"{key} must be a positive integer",
                        },
                    )
                args.extend([f"--{key.replace('_', '-')}", str(int(value))])
        args.append("--no-wait")
    code, output = await asyncio.to_thread(slurm_launch.run_cli, args)
    scanner.clear_cache()
    return JSONResponse(
        {"ok": code == 0 or action == "collect", "code": code, "output": output}
    )
