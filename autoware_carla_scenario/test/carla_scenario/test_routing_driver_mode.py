"""RoutingAction in ``driver`` mode, its pacing, and the NPC collision log.

On fakes: the pacer is plain arithmetic; the action is driven against a
duck-typed vehicle that records what it is told; the collision monitor is
fed sensor events and snapshots by hand.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Optional

import jsonschema
import pytest

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    Lanelet2Pose,
    RouteProgressCondition,
    RoutingAction,
    RoutingMode,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions._route_pacing import (
    HOLD_MARGIN_M,
    PaceMark,
    RoutePacer,
)
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    Entity,
    ScenarioDocument,
    SpawnSpec,
)
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.conditions import AlwaysTrueCondition
from autoware_carla_scenario.conditions.base import BaseCondition, ScenarioResult
from autoware_carla_scenario.npc_collisions import NpcCollisionMonitor
from autoware_carla_scenario.trajectory_recorder import SCHEMA_PATH, TrajectoryRecorder
from autoware_carla_scenario.waypoint_arrival import WaypointArrival

_WORLD: Any = object()


class _Never(BaseCondition):
    """A departure condition that never holds and cannot be predicted."""

    def __init__(self) -> None:
        super().__init__(label="never")

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        return None


class _Progress(RouteProgressCondition):
    """A rising route_progress whose entity's progress the test sets."""

    def __init__(self, target: float) -> None:
        super().__init__(value=target, label="progress")
        self.now = 0.0
        self._threshold = target

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        self._last = self.now
        self._target = self._threshold
        if self.now < self._threshold:
            return None
        return ScenarioResult(passed=True, message="", elapsed_seconds=elapsed)


# ---------------------------------------------------------------------------
# The pacer
# ---------------------------------------------------------------------------


class TestPacer:
    def test_a_timed_waypoint_is_met_by_distance_over_time(self) -> None:
        pacer = RoutePacer("npc1", "driver", [PaceMark(1, 100.0, time=10.0)], 1.0, 10.0)
        speed = pacer.speed(_WORLD, 0.0, 0.0, cruise=8.0, limit=15.0)
        # 10 m/s needed; the command starts there.
        assert speed == pytest.approx(10.0)

    def test_the_command_is_smoothed_towards_the_target(self) -> None:
        pacer = RoutePacer("npc1", "driver", [PaceMark(1, 50.0, time=10.0)], 1.0, 0.0)
        first = pacer.speed(_WORLD, 0.0, 0.0, cruise=8.0, limit=15.0)
        assert 0.0 < first < 5.0
        for _ in range(30):
            last = pacer.speed(_WORLD, 0.0, 0.0, cruise=8.0, limit=15.0)
        assert last == pytest.approx(5.0, abs=0.01)

    def test_a_late_entity_catches_up_at_the_limit(self) -> None:
        pacer = RoutePacer("npc1", "driver", [PaceMark(1, 50.0, time=10.0)], 1.0, 15.0)
        assert pacer.speed(_WORLD, 0.0, 10.0, cruise=8.0, limit=15.0) == 15.0

    def test_the_speed_is_clamped_to_the_limit(self) -> None:
        pacer = RoutePacer("npc1", "driver", [PaceMark(1, 500.0, time=10.0)], 1.0, 15.0)
        assert pacer.speed(_WORLD, 0.0, 0.0, cruise=8.0, limit=15.0) == 15.0

    def test_with_nothing_due_it_cruises(self) -> None:
        pacer = RoutePacer("npc1", "driver", [PaceMark(1, 50.0)], 1.0, 8.0)
        assert pacer.speed(_WORLD, 0.0, 0.0, cruise=8.0, limit=15.0) == 8.0

    def test_an_unpredictable_condition_is_held_short_of_the_waypoint(self) -> None:
        mark = PaceMark(1, 20.0, gate=_Never())
        pacer = RoutePacer("npc1", "driver", [mark], 1.0, 0.0)
        for _ in range(40):
            speed = pacer.speed(_WORLD, 20.0 - HOLD_MARGIN_M, 0.0, 8.0, 15.0)
        assert speed == 0.0
        pacer.observe(19.5, 3.0)  # within the tolerance: reached, still closed
        assert pacer.speed(_WORLD, 19.5, 3.0, 8.0, 15.0) == 0.0
        assert pacer.take_arrivals() == []  # its condition has not held

    def test_a_route_progress_condition_is_predicted(self) -> None:
        gate = _Progress(target=50.0)
        pacer = RoutePacer("npc1", "driver", [PaceMark(1, 40.0, gate=gate)], 1.0, 4.0)
        # The ego comes 5 m/s closer to 50 m; 40 m to go for the entity.
        speed = 0.0
        for tick in range(60):
            t = tick * 0.05
            gate.now = 5.0 * t
            speed = pacer.speed(_WORLD, 0.0, t, cruise=8.0, limit=15.0)
        # The ego needs (50 - 14.75) / 5 = 7.05 s more; 40 m over that is 5.7 m/s.
        assert speed == pytest.approx(40.0 / ((50.0 - 5.0 * 2.95) / 5.0), rel=0.15)

    def test_arrivals_are_logged_with_their_targets(self) -> None:
        gate = _Progress(target=10.0)
        marks = [PaceMark(1, 10.0, time=5.0), PaceMark(2, 20.0, gate=gate)]
        pacer = RoutePacer("npc1", "follow", marks, 1.0, 0.0)
        pacer.observe(9.5, 6.0)
        gate.now = 10.0
        pacer.note_open(2, 7.0)
        pacer.observe(19.5, 6.5)
        arrivals = pacer.take_arrivals()
        assert arrivals == [
            WaypointArrival("npc1", 1, "time", "follow", 5.0, 6.0),
            WaypointArrival("npc1", 2, "condition", "follow", 7.0, 6.5),
        ]
        assert arrivals[0].error_s == pytest.approx(1.0)
        assert pacer.take_arrivals() == []


# ---------------------------------------------------------------------------
# The action
# ---------------------------------------------------------------------------


class _Vehicle:
    """A backend-driven vehicle that records what it is told."""

    use_autopilot = True

    def __init__(self) -> None:
        self.routes: list[tuple[Lanelet2Pose, tuple[Lanelet2Pose, ...]]] = []
        self.speeds: list[float] = []
        self.actor = None

    def route_to(
        self, world, goal, *, initial_pose=None, ground_projection=None, waypoints=()
    ):  # noqa: ANN001, ANN201
        self.routes.append((goal, tuple(waypoints)))

    def set_desired_speed(self, world, speed_kmh):  # noqa: ANN001, ANN201
        self.speeds.append(speed_kmh)


@pytest.fixture
def vehicle(monkeypatch: pytest.MonkeyPatch) -> _Vehicle:
    from autoware_carla_scenario.entity import registry

    monkeypatch.setattr(registry, "_VEHICLES", {})
    monkeypatch.setattr(registry, "_PEDESTRIANS", {})
    entity = _Vehicle()
    registry.register_entity("npc1", entity)
    return entity


class TestDriverMode:
    def test_an_untimed_route_is_handed_over_and_ends_at_once(
        self, vehicle: _Vehicle
    ) -> None:
        via = [Lanelet2Pose(1, 0.0)]
        action = RoutingAction("npc1", Lanelet2Pose(3, 7.0), waypoints=via)
        assert action.mode is RoutingMode.DRIVER
        assert not action.needs_tick_loop
        action.tick(_WORLD, 0.0)
        action.tick(_WORLD, 0.05)
        assert vehicle.routes == [(Lanelet2Pose(3, 7.0), tuple(via))]
        assert vehicle.speeds == []
        assert action.state in (ActionState.END_TRANSITION, ActionState.COMPLETE)

    def test_a_cruise_speed_is_set_once(self, vehicle: _Vehicle) -> None:
        action = RoutingAction("npc1", Lanelet2Pose(3, 7.0), speed=10.0)
        action.tick(_WORLD, 0.0)
        assert vehicle.speeds == [pytest.approx(36.0)]

    def test_an_external_stack_takes_the_route_and_a_warning(
        self, vehicle: _Vehicle, caplog: pytest.LogCaptureFixture
    ) -> None:
        from autoware_carla_scenario.trajectory.model import (
            Trajectory,
            TrajectoryVertex,
        )

        vehicle.use_autopilot = False
        gated = Trajectory(
            "r",
            [
                TrajectoryVertex(Lanelet2Pose(1, 0.0)),
                TrajectoryVertex(Lanelet2Pose(3, 7.0), _Never()),
            ],
        )
        action = RoutingAction("npc1", trajectory=gated)
        assert action.needs_tick_loop
        with caplog.at_level(logging.WARNING):
            action.tick(_WORLD, 0.0)
        assert vehicle.routes == [(Lanelet2Pose(3, 7.0), (Lanelet2Pose(1, 0.0),))]
        assert vehicle.speeds == []
        assert sum("cannot be met" in r.getMessage() for r in caplog.records) == 1

    def test_a_route_needs_a_goal_or_a_trajectory(self) -> None:
        with pytest.raises(ValueError, match="goal"):
            RoutingAction("npc1")


class TestTheCardIsGone:
    def test_a_follow_trajectory_card_is_refused_pointing_to_routing(self) -> None:
        document = ScenarioDocument(
            id="s",
            entities=[
                Entity(id="ego", kind="ego", spawn=SpawnSpec(lanelet_id=1)),
                Entity(id="npc1", kind="vehicle", spawn=SpawnSpec(lanelet_id=2)),
            ],
            actions=[
                ActionNode(
                    id="a1",
                    type="follow_trajectory",
                    actor="npc1",
                    params={"following_mode": "follow"},
                )
            ],
        )
        messages = [i.message for i in validate_document(document).errors]
        assert any("routing" in m and "removed" in m for m in messages)


# ---------------------------------------------------------------------------
# The NPC collision log
# ---------------------------------------------------------------------------


def _actor(actor_id: int, type_id: str, role: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=actor_id,
        type_id=type_id,
        attributes={"role_name": role},
        bounding_box=SimpleNamespace(extent=SimpleNamespace(z=0.9)),
    )


class _Sensor:
    def __init__(self) -> None:
        self.listener: Any = None
        self.stopped = False

    def listen(self, callback: Any) -> None:
        self.listener = callback

    def stop(self) -> None:
        self.stopped = True

    def destroy(self) -> None:
        pass


class _MonitoredWorld:
    def __init__(self, actors: list[SimpleNamespace]) -> None:
        self.actors = actors
        self.sensors: list[_Sensor] = []
        self.poses: dict[int, tuple[float, float, float, float]] = {}
        self.frame = 1

    def get_actors(self) -> list[SimpleNamespace]:
        return self.actors

    def get_blueprint_library(self) -> Any:
        return SimpleNamespace(find=lambda name: name)

    def spawn_actor(self, blueprint: Any, transform: Any, attach_to: Any) -> _Sensor:
        sensor = _Sensor()
        self.sensors.append(sensor)
        return sensor

    def get_snapshot(self) -> Any:
        def find(actor_id: int) -> Any:
            if actor_id not in self.poses:
                return None
            x, y, z, pitch = self.poses[actor_id]
            transform = SimpleNamespace(
                location=SimpleNamespace(x=x, y=y, z=z),
                rotation=SimpleNamespace(pitch=pitch, roll=0.0),
            )
            return SimpleNamespace(get_transform=lambda: transform)

        return SimpleNamespace(find=find, frame=self.frame)


class TestNpcCollisions:
    def test_a_collision_is_logged_once_per_pair_and_contact(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ego = _actor(1, "vehicle.mini", str(EGO_ROLE_NAME))
        a = _actor(2, "vehicle.mini", "npc1")
        b = _actor(3, "vehicle.mini", "npc2")
        world = _MonitoredWorld([ego, a, b])
        monitor = NpcCollisionMonitor()
        monitor.tick(world, 0.0)  # type: ignore[arg-type]
        assert len(world.sensors) == 2  # not on the ego
        # Each side's sensor hears the other.
        for sensor, other in zip(world.sensors, (b, a)):
            sensor.listener(
                SimpleNamespace(
                    frame=7,
                    other_actor=other,
                    normal_impulse=SimpleNamespace(x=3.0, y=4.0, z=0.0),
                )
            )
        with caplog.at_level(logging.WARNING):
            collisions, _ = monitor.tick(world, 0.05)  # type: ignore[arg-type]
        assert len(collisions) == 1
        assert collisions[0].actors[1] == "npc2" and collisions[0].impulse == 5.0
        assert not collisions[0].ego
        assert any("Collision" in r.getMessage() for r in caplog.records)
        monitor.close()
        assert all(sensor.stopped for sensor in world.sensors)

    def test_a_walker_that_falls_is_logged_once(self) -> None:
        walker = _actor(5, "walker.pedestrian.0001", "w1")
        world = _MonitoredWorld([walker])
        monitor = NpcCollisionMonitor()
        falls: list[Any] = []
        for tick in range(80):
            world.poses[5] = (0.1 * tick, 0.0, 1.0, 80.0 if tick >= 60 else 0.0)
            falls += monitor.tick(world, tick * 0.05)[1]  # type: ignore[arg-type]
        assert [f.role for f in falls] == ["w1"]

    def test_a_walker_that_appears_is_not_fallen(self) -> None:
        walker = _actor(5, "walker.pedestrian.0001", "w1")
        world = _MonitoredWorld([walker])
        monitor = NpcCollisionMonitor()
        falls: list[Any] = []
        for tick in range(80):
            z = -500.0 if tick < 20 else (2.0 if tick < 25 else 1.0)
            world.poses[5] = (0.0, 0.0, z, 0.0)
            falls += monitor.tick(world, tick * 0.05)[1]  # type: ignore[arg-type]
        assert falls == []


def test_the_new_records_follow_the_schema(tmp_path: Any) -> None:
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    recorder = TrajectoryRecorder(tmp_path / "t.jsonl")
    recorder._out = (tmp_path / "t.jsonl").open("wb")  # noqa: SLF001
    recorder.record_waypoint_arrival(
        "r", WaypointArrival("npc1", 2, "condition", "driver", 5.0, 6.5)
    )
    recorder.record_collision(
        SimpleNamespace(
            frame=3, t=1.0, actors=["a", "b"], ids=[1, 2], impulse=9.0, ego=False
        )
    )
    recorder.record_walker_fallen(
        SimpleNamespace(frame=3, t=1.0, id=4, role="w", drop=0.8, tilt=0.0)
    )
    recorder.close()
    lines = [
        json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()
    ]
    assert [line["type"] for line in lines] == [
        "waypoint_arrival",
        "collision",
        "walker_fallen",
    ]
    for line in lines:
        validator.validate(line)
    assert lines[0]["error_s"] == pytest.approx(1.5)


def test_an_always_true_gate_never_holds_the_entity() -> None:
    mark = PaceMark(1, 20.0, gate=AlwaysTrueCondition())
    pacer = RoutePacer("npc1", "driver", [mark], 1.0, 8.0)
    assert pacer.speed(_WORLD, 0.0, 0.0, cruise=8.0, limit=15.0) == 8.0
