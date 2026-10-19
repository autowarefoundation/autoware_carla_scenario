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

from . import cases, recordings, runner, scanner, slurm_launch
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
            and (
                _base_path() / "outputs" / date / time / "slurm" / "manifest.json"
            ).is_file()
            and cases.session_dir(_base_path(), session_type, date, time) is not None,
        },
    )


@app.get(
    "/session/{session_type}/{date}/{time}/{index}",
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

    if body.get("launcher") == "slurm":
        return _submit_slurm(scenario, extra_overrides, sweeper, body.get("slurm"))

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
                if not str(value).isdigit() or int(value) < 1:
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
