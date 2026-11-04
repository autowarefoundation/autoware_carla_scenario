"""The CommonRoad metrics: CriMe's and CommonRoad-Reach's, by the workers,
on the frames a run recorded, once it is over.

A stand-in world on a map roadgen generates -- one straight road heading
36.87 degrees off +x in the OpenDRIVE frame, two lanes forward and one back --
with the ego following a slower car.  The worker is a stand-in too (a script
speaking the worker's protocol), except in the last test, which runs the real
one when the ``commonroad_metrics`` environment is there.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest

from autoware_carla_scenario import metrics
from autoware_carla_scenario.commonroad import frame as cr_frame
from autoware_carla_scenario.commonroad import session as cr_session
from autoware_carla_scenario.commonroad import worker as cr_worker
from autoware_carla_scenario.commonroad.map import MAP_FILE, commonroad_map
from autoware_carla_scenario.conditions import ComparisonRule, MetricCondition
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.failure import FailureCategory
from autoware_carla_scenario.metric_batch import Frame, FrameRecorder, evaluate_metrics
from autoware_carla_scenario.metrics import (
    BUILT_IN_METRICS,
    COMMONROAD_METRICS,
    CRIME_BTN,
    CRIME_HW_M,
    CRIME_TTC_S,
    CRIME_TTR_S,
    CRIME_WTTR_S,
    REACH_DRIVABLE_AREA_M2,
    read_metric,
)

from ._fake_world import ActorsById

REPOSITORY = Path(__file__).resolve().parents[3]
#: The road's heading in the OpenDRIVE frame: (240, 180) / 300.
COS, SIN = 0.8, 0.6


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    """Each test starts its own workers, with its own settings."""
    cr_session.close_commonroad_workers()
    settings = cr_session.commonroad_settings()
    yield
    cr_session.close_commonroad_workers()
    cr_session._POOL.settings = settings


# ---------------------------------------------------------------------------
# The metrics and the condition on them
# ---------------------------------------------------------------------------


def test_the_commonroad_metrics_are_built_in() -> None:
    assert set(COMMONROAD_METRICS) <= set(BUILT_IN_METRICS)
    for key in COMMONROAD_METRICS:
        metric = BUILT_IN_METRICS[key]
        assert metric.title and metric.text and metric.unit
        assert key.startswith(("crime_", "reach_"))
        # The most critical frame of a run: the least time, gap or area, the
        # hardest braking; the most lateral acceleration needed, the greatest
        # threat number.
        most = ("crime_a_lat_req_mps2", "crime_btn", "crime_stn")
        expected = "max" if key in most else "min"
        assert metric.aggregation == expected
        assert metrics._FAILURE_CATEGORIES[key] is FailureCategory.SAFETY_MARGIN


def test_a_commonroad_metric_reads_nothing_off_a_live_world() -> None:
    for key in COMMONROAD_METRICS:
        assert read_metric(key, SimpleNamespace()) is None  # type: ignore[arg-type]


def test_a_metric_condition_judges_a_commonroad_series() -> None:
    ttc = MetricCondition(
        CRIME_TTC_S, ComparisonRule.LESS_THAN, 2.0, max_ng_seconds=0.15, label="ttc"
    )
    assert ttc.failure_category() is FailureCategory.SAFETY_MARGIN
    assert ttc.get_details()["unit"] == "s"
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    # No conflict (inf), then closing in: under 2 s from 0.2 s on.
    values: list[Optional[float]] = [math.inf, None, 1.9, 1.5, 1.2]
    result = ttc.judge(times, values)
    assert result is not None and result.passed
    assert result.elapsed_seconds == pytest.approx(0.35)
    assert result.message.startswith("crime_ttc_s less than 2.000 s for 0.20s in total")
    assert ttc.get_details()["worst"] == 1.2

    blocked = MetricCondition(
        REACH_DRIVABLE_AREA_M2,
        ComparisonRule.LESS_THAN_OR_EQUAL,
        0.0,
        label="no way out",
    )
    assert blocked.judge([0.0, 0.1], [120.0, 0.0]) is not None
    assert blocked.judge([0.0, 0.1], [120.0, 40.0]) is None


# ---------------------------------------------------------------------------
# A stand-in world on a roadgen map
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def opendrive(tmp_path_factory: pytest.TempPathFactory) -> str:
    roadgen = pytest.importorskip("roadgen")
    road_map = roadgen.Map()
    road_map.add_road(
        start=(0.0, 0.0, 0.0),
        end=(240.0, 180.0, 0.0),
        lanes=[
            roadgen.Lane(width=3.5, direction="forward"),
            roadgen.Lane(width=3.5, direction="forward"),
            roadgen.Lane(width=3.5, direction="backward"),
        ],
    )
    path = tmp_path_factory.mktemp("road") / "road.xodr"
    road_map.export_opendrive(str(path))
    return path.read_text()


def _v(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


class _Actor:
    """A car *distance* m along the road, *right* m right of its reference
    line, going *speed* m/s along it -- placed in CARLA's frame."""

    def __init__(
        self,
        actor_id: int,
        distance: float,
        right: float,
        speed: float,
        *,
        role: str = "",
        type_id: str = "vehicle.npc",
        base_type: str = "car",
    ) -> None:
        # The OpenDRIVE frame, then CARLA's: y and the yaw change sign.
        x = distance * COS + right * SIN
        y = distance * SIN - right * COS
        self.id = actor_id
        self.is_alive = True
        self.type_id = type_id
        self.attributes = {"role_name": role, "base_type": base_type}
        self.bounding_box = SimpleNamespace(
            extent=_v(2.25, 0.9, 0.8),
            location=_v(0.0, 0.0, 0.8),
            rotation=SimpleNamespace(yaw=0.0),
        )
        self._location = _v(x, -y, 0.5)
        self._yaw = -math.degrees(math.atan2(SIN, COS))
        self._velocity = _v(speed * COS, -speed * SIN)

    def get_transform(self) -> Any:
        return SimpleNamespace(
            location=self._location, rotation=SimpleNamespace(yaw=self._yaw)
        )

    def get_velocity(self) -> Any:
        return self._velocity

    def get_acceleration(self) -> Any:
        return _v()

    def get_angular_velocity(self) -> Any:
        return _v(0.0, 0.0, 10.0)  # deg/s, left-handed: turning right


class _World(ActorsById):
    def __init__(self, actors: list[_Actor], opendrive: str) -> None:
        self._actors = actors
        self._map = SimpleNamespace(
            name="Carla/Maps/Road", to_opendrive=lambda: opendrive
        )

    def get_actors(self) -> list[_Actor]:
        return self._actors

    def get_map(self) -> Any:
        return self._map


def _following(opendrive: str) -> _World:
    """The ego at 15 m/s, 30 m behind a car at 5 m/s in its lane: a 25.5 m
    gap closing at 10 m/s.  A car in the next lane, and one far off."""
    return _World(
        [
            _Actor(1, 50.0, 1.75, 15.0, role=str(EGO_ROLE_NAME)),
            _Actor(2, 80.0, 1.75, 5.0),
            _Actor(3, 80.0, 5.25, 5.0, type_id="vehicle.truck", base_type="truck"),
            _Actor(4, 280.0, 1.75, 5.0),
            _Actor(5, 60.0, 9.0, 1.0, type_id="walker.pedestrian.0001", base_type=""),
        ],
        opendrive,
    )


def test_the_frame_is_read_in_the_commonroad_frame(opendrive: str) -> None:
    frame = cr_frame.read_frame(_following(opendrive), radius_m=100.0)  # type: ignore[arg-type]
    assert frame is not None
    ego = frame.ego
    assert ego["id"] == 1 and ego["kind"] == "car"
    assert ego["x"] == pytest.approx(50 * COS + 1.75 * SIN)
    assert ego["y"] == pytest.approx(50 * SIN - 1.75 * COS)
    assert ego["yaw"] == pytest.approx(math.atan2(SIN, COS))
    assert (ego["vx"], ego["vy"]) == pytest.approx((15 * COS, 15 * SIN))
    # 10 deg/s to the right in CARLA's frame is clockwise in CommonRoad's.
    assert ego["yaw_rate"] == pytest.approx(-math.radians(10.0))
    assert (ego["length"], ego["width"]) == pytest.approx((4.5, 1.8))
    kinds = {o["id"]: o["kind"] for o in frame.others}
    # The car 230 m off is beyond the radius.
    assert kinds == {2: "car", 3: "truck", 5: "pedestrian"}


def test_no_ego_no_frame(opendrive: str) -> None:
    world = _World([_Actor(2, 80.0, 1.75, 5.0)], opendrive)
    assert cr_frame.read_frame(world, 100.0) is None  # type: ignore[arg-type]


def test_the_map_is_converted_once(opendrive: str, tmp_path: Path) -> None:
    path = commonroad_map(opendrive, tmp_path)
    assert path.name == MAP_FILE
    text = path.read_text()
    # Three lanelets, and the planning problem's goal refers to one.
    assert text.count("<lanelet id=") == 3
    # The OpenDRIVE frame: the forward lanes' bounds run from the origin
    # towards (240, 180) and to their right.
    xs = [float(x) for x in re.findall(r"<x>([^<]+)</x>", text)]
    ys = [float(y) for y in re.findall(r"<y>([^<]+)</y>", text)]
    assert min(xs) == pytest.approx(0.0, abs=10.0) and max(xs) == pytest.approx(
        240.0, abs=10.0
    )
    assert min(ys) == pytest.approx(0.0, abs=10.0) and max(ys) == pytest.approx(
        180.0, abs=10.0
    )
    assert commonroad_map(opendrive, tmp_path) == path
    assert len([p for p in tmp_path.iterdir() if not p.name.startswith(".")]) == 1


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

#: A worker speaking the protocol: TTC from the gap to the first road user in
#: front in the ego's lane, HW refused, and every request logged.
_STAND_IN = textwrap.dedent(
    """
    import json, math, os, sys

    log = open(sys.argv[1], "a")
    for line in sys.stdin:
        request = json.loads(line)
        log.write(json.dumps({**request, "pid": os.getpid()}) + "\\n")
        log.flush()
        op = request["op"]
        if op == "init":
            answer = {"ok": True, "measures": ["crime_ttc_s", "crime_hw_m"]}
        elif op == "evaluate":
            ego = request["ego"]
            heading = math.atan2(ego["vy"], ego["vx"])
            best = math.inf
            for other in request["others"]:
                dx, dy = other["x"] - ego["x"], other["y"] - ego["y"]
                along = dx * math.cos(heading) + dy * math.sin(heading)
                across = -dx * math.sin(heading) + dy * math.cos(heading)
                if along <= 0 or abs(across) > 1.0:
                    continue
                closing = math.hypot(ego["vx"], ego["vy"]) - math.hypot(other["vx"], other["vy"])
                gap = along - (ego["length"] + other["length"]) / 2
                if closing > 0:
                    best = min(best, gap / closing)
            values = {"crime_ttc_s": best, "crime_hw_m": None}
            answer = {"ok": True, "values": values, "errors": {"crime_hw_m": "undefined here"}}
        elif op == "close":
            answer = {"ok": True}
        else:
            answer = {"ok": False, "error": "unknown"}
        sys.stdout.write(json.dumps(answer) + "\\n")
        sys.stdout.flush()
        if op == "close":
            break
    """
)


@pytest.fixture
def stand_in(tmp_path: Path) -> Path:
    script = tmp_path / "worker.py"
    script.write_text(_STAND_IN)
    log = tmp_path / "requests.jsonl"
    cr_session.configure_commonroad_metrics(
        command=(sys.executable, str(script), str(log)),
        cache_dir=tmp_path / "maps",
        period_s=0.5,
        workers=2,
    )
    return log


def _requests(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def _recorded(world: _World, times: list[float]) -> list[Frame]:
    recorder = FrameRecorder()
    for t in times:
        recorder.record(world, t, None)  # type: ignore[arg-type]
    return recorder.frames


TIMES = [round(0.1 * i, 1) for i in range(11)]


def test_the_workers_evaluate_the_recorded_frames(
    opendrive: str, stand_in: Path
) -> None:
    world = _following(opendrive)
    series = evaluate_metrics(
        _recorded(world, TIMES), world.get_map(), [CRIME_TTC_S, CRIME_HW_M]
    )

    # Every 0.5 s of the run: 0.0, 0.5 and 1.0.
    ttc = series[CRIME_TTC_S]
    assert ttc.times == [0.0, 0.5, 1.0]
    assert ttc.values == [pytest.approx(2.55)] * 3
    # What the worker could not read reads nothing.
    assert series[CRIME_HW_M].values == [None, None, None]

    requests = _requests(stand_in)
    inits = [r for r in requests if r["op"] == "init"]
    evaluated = [r for r in requests if r["op"] == "evaluate"]
    # Two workers, the frames shared between them.
    assert len(inits) == 2 and len({r["pid"] for r in evaluated}) == 2
    assert Path(inits[0]["map"]).name == MAP_FILE
    assert inits[0]["settings"]["horizon_steps"] == 30
    assert evaluated[0]["measures"] == [CRIME_TTC_S, CRIME_HW_M]
    assert {o["id"] for o in evaluated[0]["others"]} == {2, 3, 5}


def test_a_slow_metric_is_evaluated_at_its_own_period(
    opendrive: str, stand_in: Path
) -> None:
    cr_session.configure_commonroad_metrics(metric_periods_s={CRIME_HW_M: 1.0})
    world = _following(opendrive)
    series = evaluate_metrics(
        _recorded(world, TIMES), world.get_map(), [CRIME_TTC_S, CRIME_HW_M]
    )

    assert series[CRIME_TTC_S].times == [0.0, 0.5, 1.0]
    assert series[CRIME_HW_M].times == [0.0, 1.0]
    asked = [r["measures"] for r in _requests(stand_in) if r["op"] == "evaluate"]
    # Each frame is asked only what is due on it.
    assert sorted(map(tuple, asked)) == sorted(
        [(CRIME_TTC_S,)] * 3 + [(CRIME_HW_M,)] * 2
    )


def test_ttr_is_evaluated_every_half_second_by_default() -> None:
    settings = cr_session.CommonRoadSettings()
    assert settings.period_for(CRIME_TTR_S) == 0.5
    assert settings.period_for(CRIME_TTC_S) == settings.period_s == 0.1


def test_the_periods_are_replaced_and_checked() -> None:
    settings = cr_session.configure_commonroad_metrics(
        metric_periods_s={CRIME_WTTR_S: 0.3}
    )
    assert settings.period_for(CRIME_WTTR_S) == 0.3
    assert settings.period_for(CRIME_TTR_S) == settings.period_s
    with pytest.raises(ValueError, match="not a CommonRoad metric"):
        cr_session.configure_commonroad_metrics(metric_periods_s={"offroad": 1.0})
    with pytest.raises(ValueError, match="positive and finite"):
        cr_session.configure_commonroad_metrics(metric_periods_s={CRIME_TTR_S: 0.0})
    with pytest.raises(ValueError, match="positive and finite"):
        cr_session.configure_commonroad_metrics(period_s=math.inf)


def test_a_metric_condition_judges_the_workers_ttc(
    opendrive: str, stand_in: Path
) -> None:
    world = _following(opendrive)
    ttc = MetricCondition(CRIME_TTC_S, ComparisonRule.LESS_THAN, 3.0, label="ttc")
    series = evaluate_metrics(_recorded(world, TIMES), world.get_map(), [CRIME_TTC_S])[
        CRIME_TTC_S
    ]

    result = ttc.judge(series.times, series.values)
    assert result is not None and result.elapsed_seconds == 0.0
    # The workers stay up for the next run on the map.
    evaluate_metrics(_recorded(world, TIMES), world.get_map(), [CRIME_TTC_S])
    assert [r["op"] for r in _requests(stand_in)].count("init") == 2


def test_a_run_judging_one_needs_the_workers(opendrive: str, tmp_path: Path) -> None:
    cr_session.configure_commonroad_metrics(
        command=(str(tmp_path / "no-such-worker"),),
        cache_dir=tmp_path / "maps",
        workers=1,
    )
    world = _following(opendrive)
    with pytest.raises(cr_worker.CommonRoadWorkerError, match="could not start"):
        evaluate_metrics(_recorded(world, [0.0]), world.get_map(), [CRIME_TTC_S])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("workers", 0, "workers must be an integer of at least 1, got 0"),
        ("workers", -1, "workers must be an integer of at least 1, got -1"),
        ("workers", True, "workers must be an integer of at least 1, got True"),
        ("workers", 2.0, "workers must be an integer of at least 1, got 2.0"),
        ("dt", 0.0, "dt must be positive and finite, got 0.0"),
        ("dt", -0.1, "dt must be positive and finite, got -0.1"),
        ("dt", math.nan, "dt must be positive and finite, got nan"),
        ("horizon_s", math.inf, "horizon_s must be positive and finite, got inf"),
        ("horizon_s", 0.0, "horizon_s must be positive and finite, got 0.0"),
        ("radius_m", -1.0, "radius_m must be positive and finite, got -1.0"),
        ("radius_m", math.nan, "radius_m must be positive and finite, got nan"),
        ("radius_m", math.inf, "radius_m must be positive and finite, got inf"),
    ],
)
def test_the_settings_refuse_a_bad_value(field: str, value: Any, message: str) -> None:
    """Issue #1341: refused when the settings are built, not later in a run."""
    with pytest.raises(ValueError, match=re.escape(message)):
        cr_session.CommonRoadSettings(**{field: value})
    before = cr_session.commonroad_settings()
    with pytest.raises(ValueError, match=re.escape(message)):
        cr_session.configure_commonroad_metrics(**{field: value})
    assert cr_session.commonroad_settings() is before


def test_the_settings_take_good_values() -> None:
    settings = cr_session.CommonRoadSettings(
        workers=1, dt=0.05, horizon_s=0.01, radius_m=1e-3
    )
    assert settings.worker_settings()["horizon_steps"] == 1


def test_the_workers_may_be_any_integer() -> None:
    np = pytest.importorskip("numpy")
    settings = cr_session.CommonRoadSettings(workers=np.int64(2))
    assert settings.workers == 2 and type(settings.workers) is int


#: A worker that, the first time any is started (no marker file yet), exits
#: (3) on its first ``evaluate``; afterwards the stand-in.
_DIES_ONCE = textwrap.dedent(
    """
    import json, os, runpy, sys

    marker = sys.argv.pop(1)
    if os.path.exists(marker):
        sys.argv = sys.argv[1:]
        runpy.run_path(sys.argv[0], run_name="__main__")
        sys.exit(0)
    for line in sys.stdin:
        op = json.loads(line)["op"]
        if op == "evaluate":
            open(marker, "w").close()
            sys.exit(3)
        sys.stdout.write(json.dumps({"ok": True, "measures": []}) + "\\n")
        sys.stdout.flush()
        if op == "close":
            break
    """
)


def test_a_dead_worker_is_not_reused(opendrive: str, tmp_path: Path) -> None:
    """Issue #1340: the next run on the map starts fresh workers."""
    script = tmp_path / "worker.py"
    script.write_text(_STAND_IN)
    dies = tmp_path / "dies_once.py"
    dies.write_text(_DIES_ONCE)
    log = tmp_path / "requests.jsonl"
    cr_session.configure_commonroad_metrics(
        command=(
            sys.executable,
            str(dies),
            str(tmp_path / "died"),
            str(script),
            str(log),
        ),
        cache_dir=tmp_path / "maps",
        workers=1,
    )
    world = _following(opendrive)
    frames = _recorded(world, [0.0])
    with pytest.raises(cr_worker.CommonRoadWorkerError, match=r"exited \(3\)"):
        evaluate_metrics(frames, world.get_map(), [CRIME_TTC_S])
    # The failed pool is gone, not kept for the next run.
    assert cr_session._POOL.workers == []

    series = evaluate_metrics(frames, world.get_map(), [CRIME_TTC_S])
    assert series[CRIME_TTC_S].values == [pytest.approx(2.55)]


def test_an_exited_pooled_worker_restarts_the_pool(
    opendrive: str, stand_in: Path
) -> None:
    world = _following(opendrive)
    frames = _recorded(world, [0.0])
    evaluate_metrics(frames, world.get_map(), [CRIME_TTC_S])
    first = list(cr_session._POOL.workers)
    first[0]._process.kill()
    first[0]._process.wait()

    series = evaluate_metrics(frames, world.get_map(), [CRIME_TTC_S])
    assert series[CRIME_TTC_S].values == [pytest.approx(2.55)]
    assert not any(worker in cr_session._POOL.workers for worker in first)
    # Every worker of the old pool was closed, the one alive too.
    assert not any(worker.running() for worker in first)
    assert [r["op"] for r in _requests(stand_in)].count("init") == 4


def test_a_worker_that_cannot_be_closed_does_not_mask_the_failure(
    opendrive: str, stand_in: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _following(opendrive)
    frames = _recorded(world, [0.0])
    evaluate_metrics(frames, world.get_map(), [CRIME_TTC_S])
    stuck, other = cr_session._POOL.workers
    close = stuck.close

    def cannot_close() -> None:
        raise BrokenPipeError("cannot close")

    def fails(*_: Any) -> Any:
        raise cr_worker.CommonRoadWorkerError("the worker failed: boom")

    monkeypatch.setattr(stuck, "close", cannot_close)
    monkeypatch.setattr(other, "evaluate", fails)
    monkeypatch.setattr(stuck, "evaluate", fails)
    try:
        with pytest.raises(cr_worker.CommonRoadWorkerError, match="boom"):
            evaluate_metrics(frames, world.get_map(), [CRIME_TTC_S])
        # The other worker was still closed, and the pool reset.
        assert not other.running()
        assert cr_session._POOL.workers == []
    finally:
        close()


#: A worker that, the first time any is started, exits before its ``init``;
#: the others are the stand-in.
_ONE_FAILS_TO_START = textwrap.dedent(
    """
    import os, runpy, sys

    marker = sys.argv.pop(1)
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        sys.argv = sys.argv[1:]
        runpy.run_path(sys.argv[0], run_name="__main__")
        sys.exit(0)
    sys.exit(4)
    """
)


def test_the_workers_started_are_closed_when_one_cannot_start(
    opendrive: str, tmp_path: Path
) -> None:
    script = tmp_path / "worker.py"
    script.write_text(_STAND_IN)
    fails = tmp_path / "one_fails.py"
    fails.write_text(_ONE_FAILS_TO_START)
    log = tmp_path / "requests.jsonl"
    cr_session.configure_commonroad_metrics(
        command=(
            sys.executable,
            str(fails),
            str(tmp_path / "failed"),
            str(script),
            str(log),
        ),
        cache_dir=tmp_path / "maps",
        workers=3,
    )
    world = _following(opendrive)
    with pytest.raises(cr_worker.CommonRoadWorkerError, match=r"exited \(4\)"):
        evaluate_metrics(_recorded(world, [0.0]), world.get_map(), [CRIME_TTC_S])
    assert cr_session._POOL.workers == []
    ops = [r["op"] for r in _requests(log)]
    # The two that started were asked to exit.
    assert ops.count("init") == 2 and ops.count("close") == 2


def test_the_worker_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        cr_worker.WORKER_ENV, "my-python -m commonroad_metrics --x 'a b'"
    )
    assert cr_worker.worker_command() == [
        "my-python",
        "-m",
        "commonroad_metrics",
        "--x",
        "a b",
    ]
    monkeypatch.delenv(cr_worker.WORKER_ENV)
    script = tmp_path / cr_worker.WORKER_SCRIPT
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert cr_worker.worker_command() == [str(script)]
    # Neither: the checkout's project, run with uv -- or nothing.
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(
        cr_worker.CommonRoadWorkerError, match="commonroad-metrics-worker"
    ):
        cr_worker.worker_command()


def test_the_checkout_s_project_is_found() -> None:
    assert cr_worker._checkout_project() == REPOSITORY / "commonroad_metrics"


# ---------------------------------------------------------------------------
# The real worker, when its environment is there
# ---------------------------------------------------------------------------


def _real_worker() -> Optional[list[str]]:
    configured = os.environ.get(cr_worker.WORKER_ENV)
    if configured:
        return cr_worker.worker_command()
    script = (
        REPOSITORY / "commonroad_metrics" / ".venv" / "bin" / cr_worker.WORKER_SCRIPT
    )
    return [str(script)] if script.is_file() else None


@pytest.mark.slow
def test_crime_and_reach_through_the_real_worker(
    opendrive: str, tmp_path: Path
) -> None:
    command = _real_worker()
    if command is None:
        reason = (
            "the commonroad_metrics environment is not there "
            "(`uv sync` in commonroad_metrics/, or $" + cr_worker.WORKER_ENV + ")"
        )
        # CI's slow-tests job installs the worker: this is the only end-to-end
        # test of its protocol, which a workflow change must not drop by
        # skipping it unnoticed. CommonRoad's wheels are for x86_64 only, so
        # an aarch64 runner (test-arm64's slow tests) still skips.
        if os.environ.get("CI") and platform.machine() == "x86_64":
            pytest.fail(reason + " on CI, where the slow-tests job installs it")
        pytest.skip(reason)
    cr_session.configure_commonroad_metrics(
        command=tuple(command), cache_dir=tmp_path / "maps", workers=2
    )
    world = _following(opendrive)
    keys = [
        CRIME_TTC_S,
        CRIME_HW_M,
        CRIME_BTN,
        CRIME_TTR_S,
        CRIME_WTTR_S,
        REACH_DRIVABLE_AREA_M2,
    ]
    series = evaluate_metrics(_recorded(world, [0.0, 0.1, 0.2]), world.get_map(), keys)

    assert series[CRIME_TTC_S].values == [pytest.approx(2.55, abs=0.05)] * 3
    assert series[CRIME_HW_M].values == [pytest.approx(25.5, abs=0.1)] * 3
    assert all(v is not None and v > 0.0 for v in series[REACH_DRIVABLE_AREA_M2].values)
    # 1.96 m/s^2 to brake of CriMe's 11.5.
    assert series[CRIME_BTN].values == [pytest.approx(0.17, abs=0.01)] * 3
    # Time to react, but less than until they would meet.
    assert all(v is not None and 0.0 < v < 2.55 for v in series[CRIME_WTTR_S].values)
    # TTR, every 0.5 s: once in these 0.2 s; no later than WTTR.
    assert series[CRIME_TTR_S].times == [0.0]
    ttr = series[CRIME_TTR_S].values[0]
    assert ttr is not None and 0.0 < ttr <= series[CRIME_WTTR_S].values[0]  # type: ignore[operator]
    for condition in (
        MetricCondition(CRIME_TTC_S, ComparisonRule.LESS_THAN, 3.0, label="ttc"),
        MetricCondition(CRIME_HW_M, ComparisonRule.LESS_THAN, 30.0, label="hw"),
    ):
        metric = series[condition.metric]
        assert condition.judge(metric.times, metric.values) is not None


# ---------------------------------------------------------------------------
# Turning the heavy metrics off
# ---------------------------------------------------------------------------


def test_the_heavy_metrics_are_on_unless_the_environment_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cr_session.HEAVY_METRICS == {CRIME_TTR_S}
    monkeypatch.delenv(cr_session.HEAVY_METRICS_ENV, raising=False)
    assert cr_session.CommonRoadSettings().heavy_metrics
    for off in ("0", "false", "OFF", "no"):
        monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, off)
        settings = cr_session.CommonRoadSettings()
        assert not settings.heavy_metrics
        assert not settings.evaluates(CRIME_TTR_S)
        assert settings.evaluates(CRIME_WTTR_S)
    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "1")
    assert cr_session.CommonRoadSettings().evaluates(CRIME_TTR_S)


def test_a_condition_on_a_heavy_metric_turned_off_is_not_judged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.conditions import ScenarioResult

    from .test_metrics import _hit_at_two, _judged_run

    # Workers that cannot start: the run fails if they are asked for.
    cr_session.configure_commonroad_metrics(
        command=(str(tmp_path / "no-such-worker"),),
        cache_dir=tmp_path / "maps",
        heavy_metrics=False,
    )
    passed = ScenarioResult(passed=True, message="done", elapsed_seconds=3.0)
    with caplog.at_level(logging.WARNING):
        result = _judged_run(
            MetricCondition(CRIME_TTR_S, ComparisonRule.LESS_THAN, 1.0, label="ttr"),
            result=passed,
            worlds=_hit_at_two(),
        )
    assert result.passed
    assert "crime_ttr_s is a heavy metric, and the heavy metrics are off" in caplog.text


def test_metrics_heavy_turns_them_off_for_the_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import apply_metrics

    monkeypatch.delenv(cr_session.HEAVY_METRICS_ENV, raising=False)
    assert not apply_metrics(OmegaConf.create({"metrics": {"heavy": False}}))
    assert not cr_session.commonroad_settings().heavy_metrics
    # Carried to the runs' own processes, the user's variable left unset.
    assert os.environ[cr_session.HEAVY_METRICS_DECIDED_ENV] == "0"
    assert cr_session.HEAVY_METRICS_ENV not in os.environ
    assert not cr_session.CommonRoadSettings().heavy_metrics
    # On when no config sets it and neither did the user: the "0" above is
    # the last run's, not the environment's.
    assert apply_metrics(OmegaConf.create({}))
    assert cr_session.commonroad_settings().heavy_metrics
    assert os.environ[cr_session.HEAVY_METRICS_DECIDED_ENV] == "1"
    assert cr_session.HEAVY_METRICS_ENV not in os.environ


@pytest.mark.parametrize("env", [None, "0", "1"])
def test_one_runs_metrics_heavy_does_not_carry_over_to_the_next(
    monkeypatch: pytest.MonkeyPatch, env: Optional[str]
) -> None:
    """Hydra's --multirun runs its cases in one process, one apply_metrics each."""
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import apply_metrics

    if env is None:
        monkeypatch.delenv(cr_session.HEAVY_METRICS_ENV, raising=False)
    else:
        monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, env)
    from_env = env != "0"
    for heavy in (True, False, True):
        assert apply_metrics(OmegaConf.create({"metrics": {"heavy": heavy}})) is heavy
        assert apply_metrics(OmegaConf.create({"metrics": {"heavy": None}})) is from_env
        assert os.environ[cr_session.HEAVY_METRICS_DECIDED_ENV] == (
            "1" if from_env else "0"
        )
        assert os.environ.get(cr_session.HEAVY_METRICS_ENV) == env


@pytest.mark.parametrize(
    "cfg", [{}, {"metrics": {}}, {"metrics": {"heavy": None}}, {"metrics": None}]
)
def test_the_environment_decides_when_no_config_sets_metrics_heavy(
    monkeypatch: pytest.MonkeyPatch, cfg: dict[str, Any]
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import apply_metrics

    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "0")
    assert not apply_metrics(OmegaConf.create(cfg))
    assert not cr_session.commonroad_settings().heavy_metrics
    assert not cr_session.commonroad_settings().evaluates(CRIME_TTR_S)
    assert os.environ[cr_session.HEAVY_METRICS_DECIDED_ENV] == "0"

    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "1")
    assert apply_metrics(OmegaConf.create(cfg))
    assert cr_session.commonroad_settings().heavy_metrics


@pytest.mark.parametrize(("env", "heavy"), [("0", True), ("1", False), ("", False)])
def test_metrics_heavy_set_in_a_config_overrides_the_environment(
    monkeypatch: pytest.MonkeyPatch, env: str, heavy: bool
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import apply_metrics

    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, env)
    # The first config that sets it, past one that leaves it to the environment.
    cfgs = [
        OmegaConf.create({"metrics": {"heavy": None}}),
        OmegaConf.create({"metrics": {"heavy": heavy}}),
        OmegaConf.create({"metrics": {"heavy": not heavy}}),
    ]
    assert apply_metrics(*cfgs) is heavy
    assert cr_session.commonroad_settings().heavy_metrics is heavy
    assert os.environ[cr_session.HEAVY_METRICS_DECIDED_ENV] == ("1" if heavy else "0")
    assert os.environ[cr_session.HEAVY_METRICS_ENV] == env


@pytest.mark.parametrize("value", ["1", "0"])
def test_a_value_the_user_sets_equal_to_the_last_decision_is_honoured(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """The user's variable is the user's, even when it reads what an earlier
    run decided (#1164)."""
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import apply_metrics

    heavy = value == "1"
    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "1" if not heavy else "0")
    # An earlier run decides `value`, overriding the user's opposite one.
    assert apply_metrics(OmegaConf.create({"metrics": {"heavy": heavy}})) is heavy
    # The user (a test's monkeypatch, an embedding runner) now sets `value`.
    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, value)
    assert apply_metrics(OmegaConf.create({"metrics": {"heavy": None}})) is heavy
    assert cr_session.commonroad_settings().heavy_metrics is heavy
    assert os.environ[cr_session.HEAVY_METRICS_ENV] == value


def test_the_runs_one_after_another_of_issue_1164(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import apply_metrics

    def cfg(heavy: Optional[bool]) -> Any:
        return OmegaConf.create({"metrics": {"heavy": heavy}})

    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "0")
    assert apply_metrics(cfg(True))
    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "1")
    assert apply_metrics(cfg(None))
    monkeypatch.delenv(cr_session.HEAVY_METRICS_ENV)
    assert not apply_metrics(cfg(False))
    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "0")
    assert not apply_metrics(cfg(None))


def test_the_runs_own_processes_follow_the_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process the run starts reads what it decided, not the user's variable."""
    import subprocess

    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import apply_metrics

    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "1")
    probe = (
        "from autoware_carla_scenario.commonroad.session import CommonRoadSettings;"
        "print(CommonRoadSettings().heavy_metrics)"
    )
    for heavy in (False, True):
        apply_metrics(OmegaConf.create({"metrics": {"heavy": heavy}}))
        out = subprocess.run(  # noqa: S603
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == str(heavy)


def test_the_decision_is_carried_to_the_slurm_jobs() -> None:
    from autoware_carla_scenario.slurm.settings import (
        DEFAULT_ENV_PASSTHROUGH,
        snapshot_environment,
    )

    env = {cr_session.HEAVY_METRICS_DECIDED_ENV: "0", cr_session.HEAVY_METRICS_ENV: "1"}
    assert snapshot_environment(DEFAULT_ENV_PASSTHROUGH, env) == env


def test_a_slurm_task_follows_the_batchs_decision_not_its_own_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task runs only some of the batch's cases: the one that sets
    ``metrics.heavy`` may be another task's, and the task still runs with the
    decision the submitter made for the whole batch."""
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.slurm.worker import _apply_metrics

    monkeypatch.delenv(cr_session.HEAVY_METRICS_ENV, raising=False)
    unset = OmegaConf.create({"metrics": {"heavy": None}})
    off = OmegaConf.create({"metrics": {"heavy": False}})
    # The submitter: the batch's second case turns them off.
    assert not run.apply_metrics(unset, off)
    snapshot = os.environ[cr_session.HEAVY_METRICS_DECIDED_ENV]
    assert snapshot == "0"
    # A task with only the first case, on a node whose process starts with
    # them on, and the snapshot in its environment.
    cr_session.configure_commonroad_metrics(heavy_metrics=True)
    assert not _apply_metrics(run, [unset])
    assert not cr_session.commonroad_settings().heavy_metrics
    assert cr_session.HEAVY_METRICS_ENV not in os.environ
    # A snapshot without the decision: the task decides from its cases.
    monkeypatch.delenv(cr_session.HEAVY_METRICS_DECIDED_ENV)
    assert _apply_metrics(run, [unset])
    assert cr_session.commonroad_settings().heavy_metrics


def _shipped_config(overrides: list[str]) -> Any:
    from hydra import compose, initialize_config_dir

    from autoware_carla_scenario.examples import conf as conf_package

    conf_dir = str(Path(conf_package.__file__).parent.resolve())
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_the_shipped_config_leaves_the_heavy_metrics_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.examples.run import apply_metrics

    cfg = _shipped_config([])
    assert cfg.metrics.heavy is None
    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "0")
    assert not apply_metrics(cfg)
    assert os.environ[cr_session.HEAVY_METRICS_DECIDED_ENV] == "0"
    monkeypatch.delenv(cr_session.HEAVY_METRICS_ENV)
    assert apply_metrics(cfg)

    # The command line still overrides it, either way (no `+`: the key exists).
    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "0")
    assert apply_metrics(_shipped_config(["metrics.heavy=true"]))
    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "1")
    assert not apply_metrics(_shipped_config(["metrics.heavy=false"]))
