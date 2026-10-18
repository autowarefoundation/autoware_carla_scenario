"""Render a run's video later, from the CARLA recording it left.

With ``video.enabled=false`` a run renders no video: the replay that makes one
takes about as long as the run itself, and a sweep that is read for its
verdicts (or from its trajectory log) need not wait for it.  The run still
records ``{Scenario}.log`` with CARLA's recorder, and writes beside it
``{Scenario}.video.json``, a *render job*: what the video is rendered with --
the spectator camera, how many ticks the recording holds and how many of them
are the warm-up, the video's size, and the ego's debug markers by frame.
``scenario-render`` replays the recording on a CARLA server and renders the
same video the run would have::

    scenario-render outputs/2026-10-18/08-48-12          # every job in a run's directory
    scenario-render outputs/.../MyScenario-3.video.json  # one
    scenario-render outputs/.../MyScenario-3.log         # the job beside it

The server is the one at ``--host``/``--port`` (started, the way a run starts
it, when none answers there).  Each video is written next to its recording,
``{Scenario}.mp4``, and a job whose video is there already is skipped unless
``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .camera_recorder import check_video_size
from .debug_markers import DebugMarkerLog, DebugMarkers, DebugPolyline
from .scenario_base import SpectatorCameraConfig

logger = logging.getLogger(__name__)

__all__ = [
    "RENDER_JOB_SUFFIX",
    "RenderJob",
    "find_render_jobs",
    "main",
    "read_render_job",
    "render_job",
    "write_render_job",
]

#: The suffix of a render job, after the run's file stem.
RENDER_JOB_SUFFIX = ".video.json"
#: The format of a render job, bumped when it changes.
RENDER_JOB_VERSION = 1


class RenderJob:
    """What rendering one run's video takes (``{Scenario}.video.json``)."""

    def __init__(
        self,
        path: Path,
        log_path: Path,
        spectator_config: SpectatorCameraConfig,
        tick_count: int,
        warmup_ticks: int,
        video_size: "tuple[int, int]",
        debug_log: DebugMarkerLog,
    ) -> None:
        self.path = path
        self.log_path = log_path
        self.spectator_config = spectator_config
        self.tick_count = tick_count
        self.warmup_ticks = warmup_ticks
        self.video_size = video_size
        self.debug_log = debug_log

    @property
    def stem(self) -> str:
        """The run's file stem."""
        return self.path.name[: -len(RENDER_JOB_SUFFIX)]

    @property
    def mp4_path(self) -> Path:
        """Where the video goes: beside the recording, as the run puts it."""
        return self.path.with_name(f"{self.stem}.mp4")


def _markers_by_frame(debug_log: DebugMarkerLog) -> List[Dict[str, Any]]:
    """The log's markers, one entry per run of frames holding the same ones."""
    out: List[Dict[str, Any]] = []
    frames = sorted(debug_log._frames)  # noqa: SLF001 - its own module's data
    index = 0
    while index < len(frames):
        frame = frames[index]
        span = debug_log.span(frame)
        markers = debug_log.at(frame)
        assert markers is not None
        out.append(
            {
                "frame": frame,
                "span": span,
                "polylines": [
                    {
                        # To the tenth of a millimetre: a plan's points
                        # are most of the file.
                        "points": [
                            [round(c, 4) for c in point] for point in polyline.points
                        ],
                        "color": list(polyline.color),
                        "thickness": polyline.thickness,
                        "point_size": polyline.point_size,
                    }
                    for polyline in markers.polylines
                ],
            }
        )
        # The frames the span covered are the next ones in the log.
        index += span
    return out


def _debug_log_from(entries: Sequence[Dict[str, Any]]) -> DebugMarkerLog:
    log = DebugMarkerLog()
    log.start(0)
    for entry in entries:
        markers = DebugMarkers()
        for polyline in entry["polylines"]:
            markers.add_polyline(
                DebugPolyline(
                    [
                        (float(p[0]), float(p[1]), float(p[2]))
                        for p in polyline["points"]
                    ],
                    (
                        int(polyline["color"][0]),
                        int(polyline["color"][1]),
                        int(polyline["color"][2]),
                    ),
                    float(polyline["thickness"]),
                    float(polyline["point_size"]),
                )
            )
        # One object for the whole span, as the run held it: the replay draws
        # it once, to last them all.
        for frame in range(
            int(entry["frame"]), int(entry["frame"]) + int(entry["span"])
        ):
            log.record(frame, markers)
    log.stop()
    return log


def write_render_job(
    path: Path,
    *,
    log_path: Path,
    spectator_config: SpectatorCameraConfig,
    tick_count: int,
    warmup_ticks: int,
    video_size: "tuple[int, int]",
    debug_log: DebugMarkerLog,
) -> None:
    """Write the render job of a run whose video was not rendered."""
    data = {
        "version": RENDER_JOB_VERSION,
        # Relative when beside the job, as the run writes it: the directory
        # can be moved, or copied to the machine that renders.
        "log": log_path.name if log_path.parent == path.parent else str(log_path),
        "spectator": {
            "offset_back": spectator_config.offset_back,
            "offset_up": spectator_config.offset_up,
            "pitch": spectator_config.pitch,
        },
        "tick_count": tick_count,
        "warmup_ticks": warmup_ticks,
        "video": {"width": video_size[0], "height": video_size[1]},
        "debug_markers": _markers_by_frame(debug_log),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def read_render_job(path: Path) -> RenderJob:
    """Read the render job at *path*.

    Raises:
        ValueError: when it is not a render job this version reads.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != RENDER_JOB_VERSION:
        raise ValueError(
            f"{path}: not a render job of version {RENDER_JOB_VERSION} "
            f"(version {data.get('version') if isinstance(data, dict) else None!r})"
        )
    log = Path(data["log"])
    spectator = data["spectator"]
    return RenderJob(
        path=path,
        log_path=log if log.is_absolute() else path.parent / log,
        spectator_config=SpectatorCameraConfig(
            offset_back=float(spectator["offset_back"]),
            offset_up=float(spectator["offset_up"]),
            pitch=float(spectator["pitch"]),
        ),
        tick_count=int(data["tick_count"]),
        warmup_ticks=int(data["warmup_ticks"]),
        video_size=(int(data["video"]["width"]), int(data["video"]["height"])),
        debug_log=_debug_log_from(data.get("debug_markers", [])),
    )


def find_render_jobs(paths: Sequence[Path]) -> List[Path]:
    """The render jobs *paths* name: a job, a recording (the job beside it), or
    a directory (every job in it, recursively), in order, each once.

    Raises:
        FileNotFoundError: when a path names no job.
    """
    found: List[Path] = []
    for path in paths:
        if path.is_dir():
            jobs = sorted(path.rglob(f"*{RENDER_JOB_SUFFIX}"))
            if not jobs:
                raise FileNotFoundError(f"{path}: no {RENDER_JOB_SUFFIX} file in it")
            found.extend(jobs)
        elif path.name.endswith(RENDER_JOB_SUFFIX):
            if not path.is_file():
                raise FileNotFoundError(path)
            found.append(path)
        else:
            job = path.with_name(f"{path.stem}{RENDER_JOB_SUFFIX}")
            if not job.is_file():
                raise FileNotFoundError(
                    f"{path}: no render job beside it ({job.name}); a run writes "
                    "one when video.enabled is off"
                )
            found.append(job)
    unique: List[Path] = []
    seen = set()
    for job in found:
        key = job.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def _recording_map(client: Any, log_path: Path) -> Optional[str]:
    """The map the recording was made on, from its header."""
    info = client.show_recorder_file_info(str(log_path.resolve()), False)
    match = re.search(r"^Map:\s*(\S+)", info, re.MULTILINE)
    return match.group(1) if match else None


#: How long a freshly loaded map is left running before the replay, seconds:
#: UE streams the town's foliage in over the first second or so, and a replay
#: started at once shows its first frames without the trees the run had.
SETTLE_SECONDS = 3.0


def render_job(
    client: Any,
    job: RenderJob,
    *,
    video_size: "Optional[tuple[int, int]]" = None,
    settle_seconds: float = SETTLE_SECONDS,
) -> bool:
    """Render *job*'s video on *client*'s server; whether it was rendered.

    The recording's map is loaded first, every time: a fresh world is what a
    run's replay starts from (its last run's actors are gone with the map
    that held them), and one replay's leftovers -- the actors the replayer
    spawned, which stopping it does not take away -- are not the next one's.
    """
    from .scenario_runner import render_video_from_recording  # noqa: PLC0415
    from .world_reset import reload_map  # noqa: PLC0415

    if not job.log_path.is_file():
        logger.error("%s: the recording %s is not there", job.path, job.log_path)
        return False
    map_name = _recording_map(client, job.log_path)
    if map_name is None:
        logger.error("%s: the server cannot read %s", job.path, job.log_path)
        return False
    reload_map(client, map_name)
    time.sleep(settle_seconds)
    width, height = video_size if video_size is not None else job.video_size
    return render_video_from_recording(
        client,
        job.log_path.resolve(),
        job.mp4_path,
        job.spectator_config,
        job.tick_count,
        job.warmup_ticks,
        job.stem,
        job.debug_log,
        video_width=width,
        video_height=height,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``scenario-render``: render the videos runs left render jobs for."""
    parser = argparse.ArgumentParser(
        prog="scenario-render",
        description=(
            "Render the videos of runs made with video.enabled=false, by "
            "replaying their CARLA recordings."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="run output directories, *.video.json render jobs, or *.log recordings",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument(
        "--width", type=int, default=None, help="video width (default: the run's)"
    )
    parser.add_argument(
        "--height", type=int, default=None, help="video height (default: the run's)"
    )
    parser.add_argument(
        "--force", action="store_true", help="render a video that is there already"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if (args.width is None) != (args.height is None):
        parser.error("--width and --height go together")
    video_size = None
    if args.width is not None:
        check_video_size(args.width, args.height)
        video_size = (args.width, args.height)
    try:
        jobs = [read_render_job(path) for path in find_render_jobs(args.paths)]
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"scenario-render: {error}", file=sys.stderr)  # noqa: T201
        return 2
    todo = [job for job in jobs if args.force or not job.mp4_path.exists()]
    for job in jobs:
        if job not in todo:
            logger.info("%s is there already (--force renders it again)", job.mp4_path)
    if not todo:
        return 0

    import typesafe_carla.carla as carla  # noqa: PLC0415

    from .server import CarlaServerManager  # noqa: PLC0415

    failed: List[Path] = []
    with CarlaServerManager(host=args.host, port=args.port):
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        for number, job in enumerate(todo, 1):
            logger.info("[%d/%d] Rendering %s", number, len(todo), job.mp4_path)
            if not render_job(client, job, video_size=video_size):
                failed.append(job.path)
        # A server this command did not start goes on serving others: the
        # replay left it in synchronous mode, which would hold its world still.
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    for path in failed:
        print(f"scenario-render: {path}: the video was not rendered", file=sys.stderr)  # noqa: T201
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
