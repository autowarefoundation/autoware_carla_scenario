"""The CommonRoad metrics: CriMe's and CommonRoad-Reach's, by the workers,
on the frames a run recorded, once it is over.

A stand-in world on a map roadgen generates -- one straight road heading
36.87 degrees off +x in the OpenDRIVE frame, two lanes forward and one back --
with the ego following a slower car.  The worker is a stand-in too (a script
speaking the worker's protocol), except in the last test, which runs the real
one when the ``commonroad_metrics`` environment is there.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import platform
import re
import signal
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest

from autoware_carla_scenario import heavy_metrics, metrics
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
            location=self._location,
            rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=self._yaw),
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


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "#2065: roadgen refuses the fixture map: 520 validation issues, "
        "most of them lane links and junction connections in the OpenDRIVE "
        "that join lanes which do not meet"
    ),
)
def test_the_fixture_map_is_converted(tmp_path: Path) -> None:
    pytest.importorskip("roadgen")
    xodr = REPOSITORY / "data" / "nishishinjuku_carla.xodr"
    path = commonroad_map(xodr.read_text(encoding="utf-8"), tmp_path)
    assert path.read_text().count("<lanelet id=") > 0


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

#: A worker speaking the protocol: TTC from the gap to the first road user in
#: front in the ego's lane, HW failing on its first frame and reading nothing
#: on the others, and every request logged.
_STAND_IN = textwrap.dedent(
    """
    import json, math, os, sys

    log = open(sys.argv[1], "a")
    evaluated = 0
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
            errors = {"crime_hw_m": "undefined here"} if evaluated == 0 else {}
            evaluated += 1
            answer = {"ok": True, "values": values, "errors": errors}
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
        (
            "request_timeout_s",
            0.0,
            "request_timeout_s must be positive and finite, got 0.0",
        ),
        (
            "request_timeout_s",
            math.nan,
            "request_timeout_s must be positive and finite, got nan",
        ),
        ("start_timeout_s", -1, "start_timeout_s must be positive and finite, got -1"),
        (
            "request_timeout_s",
            True,
            "request_timeout_s must be positive and finite, or None, got True",
        ),
        (
            "start_timeout_s",
            False,
            "start_timeout_s must be positive and finite, or None, got False",
        ),
        # Too big for a float (#1652).
        pytest.param(
            "dt",
            10**400,
            f"dt must be positive and finite, got {10**400}",
            id="dt-huge",
        ),
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


def test_the_timeouts_may_be_none_and_are_floats() -> None:
    """#1954: a worker is given wall-clock time, by default generous enough
    for TTR's 2 s a frame; ``None`` waits as long as it takes."""
    default = cr_session.CommonRoadSettings()
    assert default.request_timeout_s is not None
    assert default.request_timeout_s >= 60
    assert default.start_timeout_s is not None
    assert default.start_timeout_s >= default.request_timeout_s
    settings = cr_session.CommonRoadSettings(request_timeout_s=None, start_timeout_s=2)
    assert settings.request_timeout_s is None
    assert settings.start_timeout_s == 2.0 and type(settings.start_timeout_s) is float


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
        sys.stdout.write(json.dumps({"ok": True, "measures": ["crime_ttc_s", "crime_hw_m"]}) + "\\n")
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

    def fails(*_: Any, **__: Any) -> Any:
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


def test_the_heavy_metrics_settings_are_heavy_metrics_reexported() -> None:
    # Moved to the checked heavy_metrics (#1689); session keeps the names.
    for name in (
        "DEFAULT_METRIC_PERIODS_S",
        "HEAVY_METRICS",
        "HEAVY_METRICS_DECIDED_ENV",
        "HEAVY_METRICS_ENV",
        "_check_period",
        "_heavy_metrics_default",
        "_heavy_metrics_from_user",
        "_heavy_metrics_on",
        "evaluates",
    ):
        assert getattr(cr_session, name) is getattr(heavy_metrics, name), name
    assert heavy_metrics.HEAVY_METRICS_ENV == "AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS"
    assert heavy_metrics.HEAVY_METRICS_DECIDED_ENV == (
        "AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS_DECIDED"
    )


def test_heavy_metrics_holds_the_key_metrics_defines() -> None:
    # heavy_metrics imports the key from metrics, which the library check
    # reads through the module (#1693; #1689 wrote it out).
    assert heavy_metrics.CRIME_TTR_S is metrics.CRIME_TTR_S
    assert heavy_metrics.HEAVY_METRICS == frozenset({metrics.CRIME_TTR_S})
    assert isinstance(heavy_metrics.HEAVY_METRICS, frozenset)
    assert heavy_metrics.HEAVY_METRICS <= set(metrics.COMMONROAD_METRICS)
    assert heavy_metrics.DEFAULT_METRIC_PERIODS_S == {metrics.CRIME_TTR_S: 0.5}


@pytest.mark.parametrize(
    ("value", "on"),
    [
        (None, True),
        ("", True),
        ("   ", True),
        ("1", True),
        ("true", True),
        ("on", True),
        ("anything", True),
        ("0", False),
        (" 0 ", False),
        ("false", False),
        ("False", False),
        ("OFF", False),
        ("no", False),
        ("No\n", False),
    ],
)
def test_heavy_metrics_on_reads_an_environment_value(
    value: Optional[str], on: bool
) -> None:
    assert heavy_metrics._heavy_metrics_on(value) is on


@pytest.mark.parametrize(
    ("user", "decided", "on"),
    [
        (None, None, True),
        ("0", None, False),
        ("1", None, True),
        # The scenario command's decision wins over the user's variable.
        ("0", "1", True),
        ("1", "0", False),
        (None, "off", False),
        # A blank decision is no decision: the user's variable says.
        ("0", "", False),
        ("0", "  ", False),
        ("1", " ", True),
    ],
)
def test_heavy_metrics_default_reads_the_decision_then_the_user(
    monkeypatch: pytest.MonkeyPatch,
    user: Optional[str],
    decided: Optional[str],
    on: bool,
) -> None:
    for name, value in (
        (heavy_metrics.HEAVY_METRICS_ENV, user),
        (heavy_metrics.HEAVY_METRICS_DECIDED_ENV, decided),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert heavy_metrics._heavy_metrics_from_user() is (user not in ("0",))
    assert heavy_metrics._heavy_metrics_default() is on
    assert cr_session.CommonRoadSettings().heavy_metrics is on


@pytest.mark.parametrize("heavy", [True, False])
def test_evaluates_leaves_out_only_a_heavy_metric_when_they_are_off(
    heavy: bool,
) -> None:
    settings = cr_session.CommonRoadSettings(heavy_metrics=heavy)
    for key in metrics.COMMONROAD_METRICS:
        expected = heavy or key != CRIME_TTR_S
        assert heavy_metrics.evaluates(heavy, key) is expected, key
        assert settings.evaluates(key) is expected, key


@pytest.mark.parametrize(
    ("period", "message"),
    [
        (0.0, "p must be positive and finite, got 0.0"),
        (-1.0, "p must be positive and finite, got -1.0"),
        (math.inf, "p must be positive and finite, got inf"),
        (math.nan, "p must be positive and finite, got nan"),
        (10**400, "p must be positive and finite, got 1"),
    ],
)
def test_check_period_refuses_a_period_not_positive_and_finite(
    period: float, message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        heavy_metrics._check_period("p", period)
    heavy_metrics._check_period("p", 0.5)
    heavy_metrics._check_period("p", 1)


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


def test_a_version_reads_ascii_digits_only() -> None:
    """#1612: ``"²".isdigit()`` is true but ``int("²")`` raises."""
    from autoware_carla_scenario.commonroad.map import _version

    assert _version("2.1rc3") == (2, 13)
    assert _version("2².1") == (2, 1)


#: A worker that answers its ``init`` ok unless told otherwise, and then each
#: request with the line it was given (``sys.argv[2]``).  With ``init`` as
#: ``sys.argv[1]``, it answers its ``init`` with that line too.
_ANSWERS_OUT_OF_PROTOCOL = textwrap.dedent(
    """
    import json, sys

    when, answer = sys.argv[1], sys.argv[2]
    for line in sys.stdin:
        op = json.loads(line)["op"]
        if op == "close":
            break
        if op == "init" and when != "init":
            sys.stdout.write(json.dumps({"ok": True, "measures": ["crime_ttc_s", "crime_hw_m"]}) + "\\n")
        else:
            sys.stdout.write(answer + "\\n")
        sys.stdout.flush()
    """
)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ("Loading CommonRoad...", "not with JSON: 'Loading CommonRoad...'"),
        ("{not json", "not with JSON: '{not json'"),
        ('["ok", true]', """not with a JSON object: '["ok", true]'"""),
        ('"ok"', """not with a JSON object: '"ok"'"""),
        ("1", "not with a JSON object: '1'"),
        ("null", "not with a JSON object: 'null'"),
    ],
)
def test_an_answer_out_of_protocol_is_a_worker_error(
    tmp_path: Path, answer: str, message: str
) -> None:
    """#1636: a line that is not a JSON object raises CommonRoadWorkerError,
    not JSONDecodeError or AttributeError."""
    script = tmp_path / "worker.py"
    script.write_text(_ANSWERS_OUT_OF_PROTOCOL)
    worker = cr_worker.CommonRoadWorker(
        [sys.executable, str(script), "evaluate", answer], tmp_path / "map.xml"
    )
    try:
        with pytest.raises(cr_worker.CommonRoadWorkerError) as raised:
            worker.evaluate([CRIME_TTC_S], {}, [])
        assert "'evaluate' out of protocol" in str(raised.value)
        assert message in str(raised.value)
    finally:
        worker.close()
    assert not worker.running()


def test_an_answer_out_of_protocol_is_quoted_cut_short(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text(_ANSWERS_OUT_OF_PROTOCOL)
    worker = cr_worker.CommonRoadWorker(
        [sys.executable, str(script), "evaluate", "x" * 5000], tmp_path / "map.xml"
    )
    try:
        with pytest.raises(cr_worker.CommonRoadWorkerError) as raised:
            worker.evaluate([CRIME_TTC_S], {}, [])
    finally:
        worker.close()
    assert "'" + "x" * cr_worker._EXCERPT_CHARS + "...'" in str(raised.value)
    assert "x" * (cr_worker._EXCERPT_CHARS + 1) not in str(raised.value)


def test_a_worker_answering_its_init_out_of_protocol_is_not_left_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "worker.py"
    script.write_text(_ANSWERS_OUT_OF_PROTOCOL)
    started: list[Any] = []
    popen = cr_worker.subprocess.Popen

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(cr_worker.subprocess, "Popen", recording_popen)
    with pytest.raises(cr_worker.CommonRoadWorkerError, match="'init' out of"):
        cr_worker.CommonRoadWorker(
            [sys.executable, str(script), "init", "a banner"], tmp_path / "map.xml"
        )
    (process,) = started
    assert process.poll() is not None


#: A worker that answers its ``init`` and then neither reads nor answers
#: anything more, as one deep in a long (or deadlocked) evaluation would.
_STUCK = textwrap.dedent(
    """
    import json, sys, time

    json.loads(sys.stdin.readline())
    sys.stdout.write(json.dumps({"ok": True, "measures": ["crime_ttc_s", "crime_hw_m"]}) + "\\n")
    sys.stdout.flush()
    while True:
        time.sleep(60)
    """
)

#: A worker that answers its ``init`` and exits at the end of its input,
#: answering nothing else (not even ``close``).
_EXITS_AT_EOF = textwrap.dedent(
    """
    import json, sys

    json.loads(sys.stdin.readline())
    sys.stdout.write(json.dumps({"ok": True, "measures": ["crime_ttc_s", "crime_hw_m"]}) + "\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        pass
    """
)


def _closed_within(worker: cr_worker.CommonRoadWorker, seconds: float) -> float:
    """How long ``worker.close()`` took; fails if it took over *seconds*,
    without waiting for it longer than that."""
    import threading
    import time

    errors: list[BaseException] = []

    def close() -> None:
        try:
            worker.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    start = time.monotonic()
    closing = threading.Thread(target=close, daemon=True)
    closing.start()
    closing.join(seconds)
    took = time.monotonic() - start
    assert not closing.is_alive(), f"close() still blocked after {seconds} s"
    assert not errors, errors
    return took


def _started_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> tuple[cr_worker.CommonRoadWorker, Any]:
    """A worker running *source*, and its process."""
    script = tmp_path / "worker.py"
    script.write_text(source)
    started: list[Any] = []
    popen = cr_worker.subprocess.Popen

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(cr_worker.subprocess, "Popen", recording_popen)
    worker = cr_worker.CommonRoadWorker(
        [sys.executable, str(script)], tmp_path / "map.xml"
    )
    (process,) = started
    return worker, process


def _kill(process: Any) -> None:
    """Make sure the test leaves *process* (one it started) not running."""
    if process.poll() is None:
        process.kill()
        process.wait()


def test_a_stuck_worker_is_killed_by_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1775: close() does not wait for an answer a busy worker never gives;
    the worker is killed once the close timeout is up."""
    worker, process = _started_worker(tmp_path, monkeypatch, _STUCK)
    try:
        took = _closed_within(worker, cr_worker._CLOSE_TIMEOUT_S + 10)
        assert took >= cr_worker._CLOSE_TIMEOUT_S - 0.5
        assert process.poll() is not None
        assert not worker.running()
        # Closing it again does nothing more.
        assert _closed_within(worker, 5) < 1
    finally:
        _kill(process)


def test_a_worker_exiting_at_the_end_of_its_input_closes_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, process = _started_worker(tmp_path, monkeypatch, _EXITS_AT_EOF)
    try:
        assert _closed_within(worker, cr_worker._CLOSE_TIMEOUT_S + 10) < (
            cr_worker._CLOSE_TIMEOUT_S
        )
        assert process.returncode == 0
    finally:
        _kill(process)


def test_a_worker_exiting_on_close_closes_at_once_and_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, process = _started_worker(
        tmp_path,
        monkeypatch,
        _ANSWERS_OUT_OF_PROTOCOL.replace(
            "when, answer = sys.argv[1], sys.argv[2]", "when, answer = 'never', ''"
        ),
    )
    try:
        assert _closed_within(worker, cr_worker._CLOSE_TIMEOUT_S + 10) < (
            cr_worker._CLOSE_TIMEOUT_S
        )
        assert process.returncode == 0
        assert _closed_within(worker, 5) < 1
    finally:
        _kill(process)


def test_closing_a_worker_that_has_exited_is_fine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, process = _started_worker(tmp_path, monkeypatch, _STUCK)
    try:
        process.kill()
        process.wait()
        assert _closed_within(worker, 5) < 1
        assert not worker.running()
        with pytest.raises(cr_worker.CommonRoadWorkerError, match="has exited"):
            worker.evaluate([CRIME_TTC_S], {}, [])
    finally:
        _kill(process)


# ---------------------------------------------------------------------------
# Interrupts, and workers under a wrapper (#1780)
# ---------------------------------------------------------------------------

#: A worker run by :data:`_WRAPPER`: it answers its ``init``, having written
#: its pid to ``sys.argv[1]``, and then ignores the end of its input and
#: sleeps, as a worker deep in an evaluation would.
_IGNORES_EOF = textwrap.dedent(
    """
    import json, os, sys, time

    json.loads(sys.stdin.readline())
    with open(sys.argv[1] + ".tmp", "w") as f:
        f.write(str(os.getpid()))
    os.replace(sys.argv[1] + ".tmp", sys.argv[1])
    sys.stdout.write(json.dumps({"ok": True, "measures": ["crime_ttc_s", "crime_hw_m"]}) + "\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        pass
    while True:
        time.sleep(60)
    """
)

#: A wrapper that, as ``uv run`` does, runs the worker as its child and waits
#: for it: killing the wrapper alone leaves the worker running.
_WRAPPER = textwrap.dedent(
    """
    import subprocess, sys

    sys.exit(subprocess.call([sys.executable] + sys.argv[1:]))
    """
)

#: A worker that reads its requests and answers none, ``init`` included,
#: exiting at the end of its input.
_NEVER_ANSWERS = textwrap.dedent(
    """
    import sys

    for line in sys.stdin:
        pass
    """
)


def _gone(pid: int) -> bool:
    """Whether *pid* no longer runs (a zombie waiting for its reaper does
    not)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] == "Z"


def _eventually(check: Any, seconds: float) -> bool:
    """Whether *check()* holds within *seconds*."""
    import time

    deadline = time.monotonic() + seconds
    while not check():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _kill_pid_running(pid: int, script: Path) -> None:
    """Kill *pid* if it still runs *script* (a process the test started)."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return
    if str(script).encode() in cmdline:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def _bounded(call: Any, seconds: float) -> tuple[Optional[BaseException], float]:
    """Run *call()* in a thread: what it raised, if anything, and how long it
    took; fails if it is still running after *seconds*, without waiting
    longer."""
    import threading
    import time

    raised: list[BaseException] = []

    def run() -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    start = time.monotonic()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    took = time.monotonic() - start
    assert not thread.is_alive(), f"still running after {seconds} s"
    return (raised[0] if raised else None), took


def _recording_popen(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The processes the workers start from now on."""
    started: list[Any] = []
    popen = cr_worker.subprocess.Popen

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(cr_worker.subprocess, "Popen", recording_popen)
    return started


def _wrapped_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[cr_worker.CommonRoadWorker, Any, int, Path]:
    """A worker run by :data:`_WRAPPER`, the wrapper's process, the worker's
    pid, and the worker's script."""
    child = tmp_path / "ignores_eof.py"
    child.write_text(_IGNORES_EOF)
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(_WRAPPER)
    pid_file = tmp_path / "worker.pid"
    started = _recording_popen(monkeypatch)
    worker = cr_worker.CommonRoadWorker(
        [sys.executable, str(wrapper), str(child), str(pid_file)],
        tmp_path / "map.xml",
    )
    (process,) = started
    return worker, process, int(pid_file.read_text()), child


def test_a_worker_runs_in_a_session_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, process = _started_worker(tmp_path, monkeypatch, _EXITS_AT_EOF)
    try:
        assert os.getsid(process.pid) == process.pid
        assert os.getpgid(process.pid) == process.pid
    finally:
        worker.close()
        _kill(process)


def test_close_kills_the_worker_under_a_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1780: the kill reaches the worker a ``uv run`` started, not
    only ``uv``."""
    worker, process, pid, child = _wrapped_worker(tmp_path, monkeypatch)
    try:
        _closed_within(worker, cr_worker._CLOSE_TIMEOUT_S + 10)
        assert process.poll() is not None
        assert _eventually(lambda: _gone(pid), 10), f"worker {pid} still runs"
    finally:
        _kill_pid_running(pid, child)
        _kill(process)


def test_close_returns_while_a_request_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1780: close() does not wait for the output a thread blocked in
    request() holds; that request raises once the worker is gone."""
    import threading

    worker, process, pid, child = _wrapped_worker(tmp_path, monkeypatch)
    raised: list[BaseException] = []

    def evaluate() -> None:
        try:
            worker.evaluate([CRIME_TTC_S], {}, [])
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    requesting = threading.Thread(target=evaluate, daemon=True)
    try:
        requesting.start()
        requesting.join(0.5)  # in request(), waiting for an answer
        _closed_within(worker, cr_worker._CLOSE_TIMEOUT_S + 10)
        requesting.join(10)
        assert not requesting.is_alive()
        assert len(raised) == 1
        assert isinstance(raised[0], cr_worker.CommonRoadWorkerError)
        assert _eventually(lambda: _gone(pid), 10), f"worker {pid} still runs"
        # Neither pipe is left open for the garbage collector.
        assert process.stdout.closed and process.stdin.closed
    finally:
        _kill_pid_running(pid, child)
        _kill(process)


def test_close_returns_while_a_request_is_blocked_writing(
    tmp_path: Path,
) -> None:
    """Issue #1780: a request blocked writing to a worker that reads nothing
    does not hold close() up; it raises once the worker is killed, its pipes
    closed (no ResourceWarning when the worker is collected)."""
    import gc
    import threading
    import warnings
    import weakref

    script = tmp_path / "worker.py"
    script.write_text(_STUCK)
    worker = cr_worker.CommonRoadWorker(
        [sys.executable, str(script)], tmp_path / "map.xml"
    )
    pid = worker._process.pid
    raised: list[BaseException] = []

    def evaluate(worker: cr_worker.CommonRoadWorker) -> None:
        try:
            # Far more than a pipe holds: the write blocks.
            worker.evaluate([CRIME_TTC_S], {}, [{"pad": "x" * 1_000_000}])
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    # The worker as an argument, not a closure's: the thread lets go of it.
    requesting = threading.Thread(target=evaluate, args=(worker,), daemon=True)
    try:
        requesting.start()
        requesting.join(0.5)  # blocked writing the request
        _closed_within(worker, cr_worker._CLOSE_TIMEOUT_S + 10)
        requesting.join(10)
        assert not requesting.is_alive()
        assert len(raised) == 1
        assert isinstance(raised[0], cr_worker.CommonRoadWorkerError)
        process = worker._process
        assert process.poll() is not None
        assert process.stdin is not None and process.stdin.closed
        assert process.stdout is not None and process.stdout.closed
        collected = weakref.ref(process)
        del process
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            del worker
            raised.clear()  # the error's traceback holds the worker
            gc.collect()
        assert collected() is None  # so its pipes would have warned
        assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
    finally:
        _kill_pid_running(pid, script)


def test_an_interrupted_evaluation_closes_its_workers_at_once(
    opendrive: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1780: an interrupt does not wait for the remaining frames; the
    stuck workers are closed together, within about one close timeout."""
    from concurrent.futures import FIRST_EXCEPTION

    script = tmp_path / "worker.py"
    script.write_text(_STUCK)
    cr_session.configure_commonroad_metrics(
        command=(sys.executable, str(script)),
        cache_dir=tmp_path / "maps",
        workers=3,
    )
    world = _following(opendrive)
    frames = _recorded(world, TIMES)
    carla_map = world.get_map()
    # Started before the interrupt is armed.
    started = _recording_popen(monkeypatch)
    cr_session._workers_for(carla_map)
    real_wait = cr_session.wait

    def interrupted(fs: Any, timeout: Any = None, return_when: Any = None) -> Any:
        if return_when == FIRST_EXCEPTION:
            # Each share is blocked in its first request by now.
            real_wait(fs, timeout=1.0, return_when=FIRST_EXCEPTION)
            raise KeyboardInterrupt
        return real_wait(fs, timeout=timeout)

    monkeypatch.setattr(cr_session, "wait", interrupted)
    try:
        raised, took = _bounded(
            lambda: cr_session.evaluate_commonroad(frames, carla_map, [CRIME_TTC_S]),
            cr_worker._CLOSE_TIMEOUT_S * 3 + 10,
        )
        assert isinstance(raised, KeyboardInterrupt)
        assert len(started) == 3
        assert all(process.poll() is not None for process in started)
        assert cr_session._POOL.workers == []
        # One close timeout for all three, not one each.
        assert took < cr_worker._CLOSE_TIMEOUT_S * 2
    finally:
        for process in started:
            _kill(process)


def test_an_interrupt_while_starting_closes_the_workers_started(
    opendrive: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1780: workers that are still waiting for their ``init`` answer
    when an interrupt comes are closed, not left to the end of the run."""
    script = tmp_path / "worker.py"
    script.write_text(_NEVER_ANSWERS)
    cr_session.configure_commonroad_metrics(
        command=(sys.executable, str(script)),
        cache_dir=tmp_path / "maps",
        workers=2,
    )
    world = _following(opendrive)
    started = _recording_popen(monkeypatch)
    real_wait = cr_session.wait

    def interrupted(fs: Any, timeout: Any = None, return_when: Any = None) -> Any:
        if timeout is None and return_when is None:
            _eventually(lambda: len(started) == 2, 30)
            real_wait(fs, timeout=0.5)  # each is waiting for its answer
            raise KeyboardInterrupt
        return real_wait(fs, timeout=timeout)

    monkeypatch.setattr(cr_session, "wait", interrupted)
    try:
        raised, _ = _bounded(
            lambda: cr_session.evaluate_commonroad(
                _recorded(world, [0.0]), world.get_map(), [CRIME_TTC_S]
            ),
            cr_worker._CLOSE_TIMEOUT_S * 2 + 30,
        )
        assert isinstance(raised, KeyboardInterrupt)
        assert len(started) == 2
        assert _eventually(
            lambda: all(process.poll() is not None for process in started), 10
        )
        assert cr_session._POOL.workers == []
    finally:
        for process in started:
            _kill(process)


def test_a_worker_answering_out_of_protocol_is_not_reused(
    opendrive: str, tmp_path: Path
) -> None:
    """The run fails with the worker's error, and the pool is reset: the
    worker's answers are out of step, so the next run starts afresh."""
    script = tmp_path / "worker.py"
    script.write_text(_ANSWERS_OUT_OF_PROTOCOL)
    cr_session.configure_commonroad_metrics(
        command=(sys.executable, str(script), "evaluate", "[]"),
        cache_dir=tmp_path / "maps",
        workers=1,
    )
    world = _following(opendrive)
    (worker,) = cr_session._workers_for(world.get_map())
    with pytest.raises(cr_worker.CommonRoadWorkerError, match="not with a JSON"):
        evaluate_metrics(_recorded(world, [0.0]), world.get_map(), [CRIME_TTC_S])
    assert cr_session._POOL.workers == []
    assert not worker.running()


@pytest.mark.parametrize(
    ("value", "heavy"),
    [
        ("false", False),
        (" Off ", False),
        ("no", False),
        (0, False),
        ("0", False),
        ("true", True),
        ("YES", True),
        (1, True),
        ("1", True),
    ],
)
def test_metrics_heavy_is_read_strictly(
    monkeypatch: pytest.MonkeyPatch, value: Any, heavy: bool
) -> None:
    """#1761: ``bool("false")`` turned them on; the spelling is read."""
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import apply_metrics

    # The environment says the opposite: the config decides.
    monkeypatch.setenv(cr_session.HEAVY_METRICS_ENV, "0" if heavy else "1")
    assert apply_metrics(OmegaConf.create({"metrics": {"heavy": value}})) is heavy
    assert cr_session.commonroad_settings().heavy_metrics is heavy
    # What the SLURM tasks and the runs' processes read.
    assert os.environ[cr_session.HEAVY_METRICS_DECIDED_ENV] == ("1" if heavy else "0")
    assert heavy_metrics._heavy_metrics_default() is heavy


@pytest.mark.parametrize("value", ["ture", "", 2, -1, 0.0, 1.5, [True]])
def test_a_metrics_heavy_that_is_no_switch_is_refused(
    monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import apply_metrics

    monkeypatch.delenv(cr_session.HEAVY_METRICS_DECIDED_ENV, raising=False)
    with pytest.raises(ValueError, match=r"^metrics\.heavy must be true or false"):
        apply_metrics(OmegaConf.create({"metrics": {"heavy": value}}))
    # Nothing was decided.
    assert cr_session.HEAVY_METRICS_DECIDED_ENV not in os.environ


@pytest.mark.parametrize("value", ["ture", "2"])
def test_the_batch_check_refuses_a_bad_metrics_heavy_in_any_case(
    value: str,
) -> None:
    """#1761: refused before the first case runs, though apply_metrics reads
    only the first case that sets it."""
    from autoware_carla_scenario.examples import run

    good = run._compose_config("intersection_passing/straight", [])
    run.check_batch_configs([good])
    falsy = run._compose_config(
        "intersection_passing/straight", ["metrics.heavy=false"]
    )
    run.check_batch_configs([falsy])
    bad = run._compose_config(
        "intersection_passing/straight", [f"metrics.heavy={value}"]
    )
    with pytest.raises(
        run.RunConfigError, match=r"^b: metrics\.heavy must be true or false"
    ):
        run.check_batch_configs([falsy, bad], ["a", "b"])


# ---------------------------------------------------------------------------
# A metric the worker cannot read (#1953), a worker out of time (#1954)
# ---------------------------------------------------------------------------

#: A worker that knows the measures ``sys.argv[1]`` (a JSON list), answers
#: each ``evaluate`` with ``sys.argv[2]`` (a JSON object, ``ok`` added) after
#: sleeping ``sys.argv[3]`` seconds, and logs every request to ``sys.argv[4]``.
_ANSWERS_WITH = textwrap.dedent(
    """
    import json, sys, time

    measures, answer = json.loads(sys.argv[1]), json.loads(sys.argv[2])
    delay, log = float(sys.argv[3]), open(sys.argv[4], "a")
    for line in sys.stdin:
        request = json.loads(line)
        log.write(line)
        log.flush()
        op = request["op"]
        if op == "init":
            reply = {"ok": True, "measures": measures}
        elif op == "evaluate":
            time.sleep(delay)
            reply = {"ok": True, **answer}
        else:
            reply = {"ok": True}
        sys.stdout.write(json.dumps(reply) + "\\n")
        sys.stdout.flush()
        if op == "close":
            break
    """
)


def _answering(
    tmp_path: Path,
    measures: list[str],
    answer: dict[str, Any],
    delay: float = 0.0,
    **settings: Any,
) -> Path:
    """Configure the workers to run :data:`_ANSWERS_WITH`; its request log."""
    script = tmp_path / "answers_with.py"
    script.write_text(_ANSWERS_WITH)
    log = tmp_path / "requests.jsonl"
    cr_session.configure_commonroad_metrics(
        command=(
            sys.executable,
            str(script),
            json.dumps(measures),
            json.dumps(answer),
            str(delay),
            str(log),
        ),
        cache_dir=tmp_path / "maps",
        **{"workers": 1, **settings},
    )
    return log


def test_a_worker_that_does_not_know_a_metric_fails_the_run(
    opendrive: str, tmp_path: Path
) -> None:
    """#1953: an older worker (on the PATH) that does not know a metric fails
    the run, rather than read nothing in every frame and pass it."""
    log = _answering(
        tmp_path,
        [CRIME_TTC_S],
        {"values": {CRIME_TTC_S: 1.0}, "errors": {CRIME_HW_M: "unknown measure"}},
    )
    world = _following(opendrive)
    with pytest.raises(
        cr_worker.CommonRoadWorkerError,
        match="does not know the metrics crime_hw_m",
    ):
        evaluate_metrics(
            _recorded(world, TIMES), world.get_map(), [CRIME_TTC_S, CRIME_HW_M]
        )
    # Refused before a frame is sent.
    assert [r["op"] for r in _requests(log)] == ["init"]


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (
            {"values": {CRIME_TTC_S: 1.0}, "errors": {CRIME_HW_M: "unknown measure"}},
            "crime_hw_m unknown",
        ),
        (
            {"values": {CRIME_TTC_S: 1.0}, "errors": {}},
            "no value nor error for crime_hw_m",
        ),
    ],
)
def test_an_answer_leaving_a_metric_out_is_out_of_protocol(
    opendrive: str, tmp_path: Path, answer: dict[str, Any], message: str
) -> None:
    """#1953: an answer that calls a metric unknown, or has neither a value
    nor an error for it, is out of protocol, and the pool is reset."""
    _answering(tmp_path, [CRIME_TTC_S, CRIME_HW_M], answer)
    world = _following(opendrive)
    (worker,) = cr_session._workers_for(world.get_map())
    with pytest.raises(cr_worker.CommonRoadWorkerError) as raised:
        evaluate_metrics(
            _recorded(world, [0.0]), world.get_map(), [CRIME_TTC_S, CRIME_HW_M]
        )
    assert "'evaluate' out of protocol" in str(raised.value)
    assert message in str(raised.value)
    assert cr_session._POOL.workers == []
    assert not worker.running()


def test_a_metric_failing_in_every_frame_fails_the_run(
    opendrive: str, tmp_path: Path
) -> None:
    """#1953: a metric that failed in every frame read nothing at all, and a
    fail condition on it would pass the run unjudged: the run fails."""
    _answering(
        tmp_path,
        [CRIME_TTC_S, CRIME_HW_M],
        {
            "values": {CRIME_TTC_S: 1.0, CRIME_HW_M: None},
            "errors": {CRIME_HW_M: "RuntimeError: boom"},
        },
        workers=2,
        period_s=0.5,
    )
    world = _following(opendrive)
    frames = _recorded(world, TIMES)
    with pytest.raises(
        cr_worker.CommonRoadWorkerError,
        match=re.escape(
            "crime_hw_m (failed: RuntimeError: boom) failed in every frame "
            "evaluated (3)"
        ),
    ):
        evaluate_metrics(frames, world.get_map(), [CRIME_TTC_S, CRIME_HW_M])
    # The workers are fine, and kept: the metric is the failure, not they.
    workers = cr_session._POOL.workers
    assert len(workers) == 2 and all(worker.running() for worker in workers)
    # The metric that reads is read.
    series = evaluate_metrics(frames, world.get_map(), [CRIME_TTC_S])
    assert series[CRIME_TTC_S].values == [1.0, 1.0, 1.0]


def test_a_metric_reading_nothing_with_no_one_to_measure_from_passes(
    opendrive: str, tmp_path: Path
) -> None:
    """A metric reading ``null`` without an error (no one ahead) in every
    frame is no failure: it reads nothing, and is not NG."""
    _answering(
        tmp_path,
        [CRIME_TTC_S],
        {"values": {CRIME_TTC_S: None}, "errors": {}},
    )
    world = _following(opendrive)
    series = evaluate_metrics(_recorded(world, TIMES), world.get_map(), [CRIME_TTC_S])
    assert set(series[CRIME_TTC_S].values) == {None}


def test_a_worker_out_of_time_is_killed_and_fails_the_run(
    opendrive: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1954: a worker stuck on a frame does not hold up the run (and the
    batch after it) forever: it is killed, and the run fails."""
    _answering(
        tmp_path,
        [CRIME_TTC_S],
        {"values": {CRIME_TTC_S: 1.0}, "errors": {}},
        delay=60.0,
        request_timeout_s=0.5,
    )
    world = _following(opendrive)
    started = _recording_popen(monkeypatch)
    carla_map = world.get_map()
    cr_session._workers_for(carla_map)
    frames = _recorded(world, TIMES)
    try:
        raised, took = _bounded(
            lambda: cr_session.evaluate_commonroad(frames, carla_map, [CRIME_TTC_S]),
            30,
        )
        assert isinstance(raised, cr_worker.CommonRoadWorkerError)
        assert "did not answer 'evaluate' within 0.5 s" in str(raised)
        # Killed at once, not asked to exit and given the close timeout.
        assert took < cr_worker._CLOSE_TIMEOUT_S
        (process,) = started
        assert process.poll() is not None
        assert cr_session._POOL.workers == []
    finally:
        for process in started:
            _kill(process)


def test_a_worker_that_does_not_start_in_time_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "worker.py"
    script.write_text(_NEVER_ANSWERS)
    started = _recording_popen(monkeypatch)
    try:
        raised, took = _bounded(
            lambda: cr_worker.CommonRoadWorker(
                [sys.executable, str(script)], tmp_path / "map.xml", timeout=0.3
            ),
            30,
        )
        assert isinstance(raised, cr_worker.CommonRoadWorkerError)
        assert "did not answer 'init' within 0.3 s" in str(raised)
        assert took < cr_worker._CLOSE_TIMEOUT_S
        (process,) = started
        assert process.poll() is not None
    finally:
        for process in started:
            _kill(process)


#: A worker that answers each request in two writes a while apart, and its
#: ``init`` together with a line of the next answer's.
_ANSWERS_IN_PIECES = textwrap.dedent(
    """
    import json, sys, time

    for line in sys.stdin:
        op = json.loads(line)["op"]
        if op == "close":
            break
        if op == "init":
            init = json.dumps({"ok": True, "measures": ["crime_ttc_s"]})
            sys.stdout.write(init + "\\n" + '{"ok": true, ')
            sys.stdout.flush()
            continue
        time.sleep(0.3)
        sys.stdout.write('"values": {"crime_ttc_s": 2.5}, "errors": {}}\\n')
        sys.stdout.flush()
    """
)


def test_an_answer_written_in_pieces_is_read_whole(tmp_path: Path) -> None:
    """The output is read a chunk at a time, to wait with a deadline: a line
    that comes in pieces is read whole, and what comes past a line is kept
    for the next."""
    script = tmp_path / "worker.py"
    script.write_text(_ANSWERS_IN_PIECES)
    worker = cr_worker.CommonRoadWorker(
        [sys.executable, str(script)], tmp_path / "map.xml", timeout=30
    )
    try:
        assert worker.measures == [CRIME_TTC_S]
        values, errors = worker.evaluate([CRIME_TTC_S], {}, [], timeout=30)
        assert values == {CRIME_TTC_S: 2.5} and errors == {}
    finally:
        worker.close()


def test_a_timeout_longer_than_poll_waits_at_once_is_fine(
    opendrive: str, tmp_path: Path
) -> None:
    """A timeout meant as "forever" (past poll()'s C int of milliseconds)
    passes the settings' check, and a normal answer is read under it."""
    _answering(
        tmp_path,
        [CRIME_TTC_S],
        {"values": {CRIME_TTC_S: 1.0}, "errors": {}},
        delay=0.2,
        request_timeout_s=1e9,
        start_timeout_s=1e12,
    )
    world = _following(opendrive)
    series = evaluate_metrics(_recorded(world, [0.0]), world.get_map(), [CRIME_TTC_S])
    assert series[CRIME_TTC_S].values == [1.0]
