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
import math
import os
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
    CRIME_HW_M,
    CRIME_TTC_S,
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
        # hardest braking; the most lateral acceleration needed.
        expected = "max" if key == "crime_a_lat_req_mps2" else "min"
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
        pytest.skip(
            "the commonroad_metrics environment is not there "
            "(`uv sync` in commonroad_metrics/, or $" + cr_worker.WORKER_ENV + ")"
        )
    cr_session.configure_commonroad_metrics(
        command=tuple(command), cache_dir=tmp_path / "maps", workers=2
    )
    world = _following(opendrive)
    keys = [CRIME_TTC_S, CRIME_HW_M, REACH_DRIVABLE_AREA_M2]
    series = evaluate_metrics(_recorded(world, [0.0, 0.1, 0.2]), world.get_map(), keys)

    assert series[CRIME_TTC_S].values == [pytest.approx(2.55, abs=0.05)] * 3
    assert series[CRIME_HW_M].values == [pytest.approx(25.5, abs=0.1)] * 3
    assert all(v is not None and v > 0.0 for v in series[REACH_DRIVABLE_AREA_M2].values)
    for condition in (
        MetricCondition(CRIME_TTC_S, ComparisonRule.LESS_THAN, 3.0, label="ttc"),
        MetricCondition(CRIME_HW_M, ComparisonRule.LESS_THAN, 30.0, label="hw"),
    ):
        metric = series[condition.metric]
        assert condition.judge(metric.times, metric.values) is not None
