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
            collisions, _, _ = monitor.tick(world, 0.05)  # type: ignore[arg-type]
        assert len(collisions) == 1
        assert collisions[0].actors[1] == "npc2" and collisions[0].impulse == 5.0
        assert collisions[0].phase == "start"
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

    def test_a_walker_bumped_into_the_air_is_not_fallen(self) -> None:
        """Pushed up half a metre, it lands where it stood: no fall; one that
        goes down below where it stood has fallen."""
        walker = _actor(5, "walker.pedestrian.0001", "w1")
        world = _MonitoredWorld([walker])
        monitor = NpcCollisionMonitor()
        falls: list[Any] = []
        for tick in range(200):
            z = 0.94
            if 100 <= tick < 105:
                z = 1.43  # in the air a moment
            if tick >= 170:
                z = 0.2  # down
            world.poses[5] = (0.01 * tick, 0.0, z, 0.0)
            falls += monitor.tick(world, tick * 0.05)[1]  # type: ignore[arg-type]
        assert [round(f.t, 2) for f in falls] == [8.5]

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
            frame=3,
            t=1.0,
            actors=["a", "b"],
            ids=[1, 2],
            impulse=9.0,
            ego=False,
            phase="end",
            duration=2.5,
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


def test_a_hold_in_the_measured_entitys_way_is_let_go(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Held for the ego's progress, in the ego's lane just ahead of it, while
    the ego stands behind it: after BLOCK_WAIT_S the hold is let go."""
    import autoware_carla_scenario.actions._route_pacing as pacing

    monkeypatch.setattr(
        pacing, "route_s_of_carla_point", lambda x, y, near=None: (x, 0.5)
    )
    gate = _Progress(target=50.0)
    gate.now = 10.0  # the ego, standing at route s 10
    mark = PaceMark(1, 1.0, gate=gate)
    pacer = RoutePacer("npc1", "driver", [mark], 0.5, 0.0)
    pacer.observe(1.0, 0.0)
    with caplog.at_level(logging.WARNING):
        t = 0.0
        while t < pacing.BLOCK_WAIT_S + 0.5:
            pacer.speed(_WORLD, 1.0, t, 8.0, 15.0, position=(18.0, 0.0))
            t += 0.05
    assert mark.opened_at is not None
    assert sum("stops holding" in r.getMessage() for r in caplog.records) == 1


def test_a_hold_out_of_the_measured_entitys_way_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autoware_carla_scenario.actions._route_pacing as pacing

    # Beside the route (8 m off its line): not in the ego's way.
    monkeypatch.setattr(
        pacing, "route_s_of_carla_point", lambda x, y, near=None: (x, 8.0)
    )
    gate = _Progress(target=50.0)
    gate.now = 10.0
    mark = PaceMark(1, 1.0, gate=gate)
    pacer = RoutePacer("npc1", "driver", [mark], 0.5, 0.0)
    pacer.observe(1.0, 0.0)
    t = 0.0
    while t < 10.0:
        assert pacer.speed(_WORLD, 1.0, t, 8.0, 15.0, position=(18.0, 0.0)) == 0.0
        t += 0.05
    assert mark.opened_at is None


def test_a_contact_is_logged_when_it_starts_and_when_it_ends() -> None:
    a = _actor(2, "vehicle.mini", "npc1")
    b = _actor(3, "vehicle.mini", "npc2")
    world = _MonitoredWorld([a, b])
    monitor = NpcCollisionMonitor()
    monitor.tick(world, 0.0)  # type: ignore[arg-type]
    records: list[Any] = []
    t = 0.05
    while t < 6.0:
        if t < 3.0:  # in contact for 3 s, an event every tick
            world.sensors[0].listener(
                SimpleNamespace(
                    frame=int(t * 20),
                    other_actor=b,
                    normal_impulse=SimpleNamespace(x=1.0, y=0.0, z=0.0),
                )
            )
        records += monitor.tick(world, t)[0]  # type: ignore[arg-type]
        t += 0.05
    assert [r.phase for r in records] == ["start", "end"]
    assert records[1].duration == pytest.approx(2.95, abs=0.06)


def test_a_fallen_walker_is_despawned_after_it_has_lain_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autoware_carla_scenario.actions.despawn as despawn

    gone: list[str] = []

    def despawn_entity(world: Any, name: str, label: str) -> int:
        gone.append(name)
        return 9

    monkeypatch.setattr(despawn, "despawn_entity", despawn_entity)
    walker = _actor(5, "walker.pedestrian.0001", "w1")
    world = _MonitoredWorld([walker])
    monitor = NpcCollisionMonitor(despawn_fallen_after_s=5.0)
    despawned: list[Any] = []
    fell_at = None
    for tick in range(300):
        t = tick * 0.05
        world.poses[5] = (0.0, 0.0, 1.0, 80.0 if tick >= 60 else 0.0)
        _, falls, out = monitor.tick(world, t)  # type: ignore[arg-type]
        if falls:
            fell_at = t
        despawned += [(name, t) for name, _ in out]
    assert fell_at is not None
    assert gone == ["w1"]
    ((name, at),) = despawned
    assert name == "w1" and 5.0 <= at - fell_at < 5.5


class _SnapshotWorld:
    """A world whose snapshot holds the actors of *poses*."""

    id = 1

    def __init__(self, actors: dict[int, tuple[str, float, float, float]]) -> None:
        self.actors = actors

    def get_snapshot(self) -> Any:
        def seen(actor_id: int, x: float, y: float, speed: float) -> Any:
            return SimpleNamespace(
                id=actor_id,
                get_transform=lambda: SimpleNamespace(
                    location=SimpleNamespace(x=x, y=y, z=0.0),
                    rotation=SimpleNamespace(yaw=0.0),
                ),
                get_velocity=lambda: SimpleNamespace(x=speed, y=0.0, z=0.0),
            )

        class _Snapshot(list):
            frame = 1

        return _Snapshot(seen(i, x, y, v) for i, (_, x, y, v) in self.actors.items())

    def get_actor(self, actor_id: int) -> Any:
        type_id = self.actors[actor_id][0]
        extent = (2.0, 1.0) if type_id.startswith("vehicle.") else (0.3, 0.3)
        return SimpleNamespace(
            type_id=type_id,
            bounding_box=SimpleNamespace(
                extent=SimpleNamespace(x=extent[0], y=extent[1], z=1.0)
            ),
        )


def test_a_walker_ahead_is_found_and_one_beside_is_not() -> None:
    from autoware_carla_scenario.utils import clearance

    clearance._CACHE.frame = -1
    world = _SnapshotWorld(
        {
            1: ("vehicle.mini", 0.0, 0.0, 0.0),
            2: ("walker.pedestrian.0001", 6.0, 0.5, 0.0),
            3: ("walker.pedestrian.0001", 6.0, 5.0, 0.0),
        }
    )
    found = clearance.first_in_way(world, 1, 0.0, 0.0, 0.0, "walker", 8.0, 1.5)
    assert found is not None and found.id == 2
    assert clearance.first_in_way(world, 1, 0.0, 0.0, 180.0, "walker", 8.0, 1.5) is None


class _Controller:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def start(self) -> None:
        self.calls.append("start")

    def go_to_location(self, location: Any) -> None:
        self.calls.append(("go", round(location.x, 1)))

    def set_max_speed(self, speed: float) -> None:
        self.calls.append(("speed", speed))

    def stop(self) -> None:
        self.calls.append("stop")

    def destroy(self) -> None:
        self.calls.append("destroy")


class _AiWorld:
    def __init__(self, navmesh: bool, world_id: int) -> None:
        self.id = world_id
        self.navmesh = navmesh
        self.controller = _Controller()

    def set_pedestrians_cross_factor(self, factor: float) -> None:
        self.cross_factor = factor

    def get_random_location_from_navigation(self) -> Any:
        return SimpleNamespace(x=0.0, y=0.0, z=0.0) if self.navmesh else None

    def get_blueprint_library(self) -> Any:
        return SimpleNamespace(find=lambda name: name)

    def spawn_actor(self, blueprint: Any, transform: Any, attach_to: Any) -> Any:
        assert blueprint == "controller.ai.walker"
        return self.controller


def path_of(length: float) -> Any:
    from autoware_carla_scenario.trajectory.model import ResolvedTrajectory

    return ResolvedTrajectory([0.0, length], [0.0, 0.0], [0.0, 0.0], [None, None])


def test_a_routed_walker_is_walked_by_carlas_controller() -> None:
    from autoware_carla_scenario.entity import _walker_ai

    world = _AiWorld(True, 101)
    where = SimpleNamespace(x=0.0, y=0.0, z=-500.0)
    walker = SimpleNamespace(
        id=55,
        get_world=lambda: world,
        get_location=lambda: where,
        apply_control=lambda control: world.controller.calls.append("stand"),
    )
    path = path_of(25.0)
    # Still under the map (brought in, not yet moved there): not started yet.
    assert _walker_ai.drive(walker, path, 0.0, 1.4, where)
    assert world.controller.calls == []
    where.z = 0.0
    assert _walker_ai.drive(walker, path, 0.0, 1.4, where)
    assert _walker_ai.drive(walker, path, 9.0, 1.4, where)  # 2 m short: reached
    assert _walker_ai.drive(walker, path, 9.5, 0.0, where)  # held: speed 0
    assert _walker_ai.drive(walker, path, 9.5, 1.0, where)  # on: target again
    _walker_ai.release(55)
    assert world.controller.calls == [
        "start",
        ("go", 10.0),
        ("speed", 1.4),
        ("go", 20.0),
        ("speed", 0.0),
        ("go", 20.0),
        ("speed", 1.0),
        "stop",
        "destroy",
    ]


def test_a_walkers_own_cross_factor_is_set_as_its_controller_starts() -> None:
    from autoware_carla_scenario.entity import _walker_ai

    world = _AiWorld(True, 103)
    factors: list[float] = []
    world.set_pedestrians_cross_factor = factors.append  # type: ignore[assignment]
    where = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    for walker_id, factor in ((57, 1.0), (58, 0.0), (60, None)):
        walker = SimpleNamespace(id=walker_id, get_world=lambda: world)
        assert _walker_ai.drive(walker, path_of(25.0), 0.0, 1.4, where, factor)
        _walker_ai.release(walker_id)
    # Each set just before its start and put back after; None leaves it.
    assert factors == [1.0, 0.0, -1.0, 0.0]


def test_without_a_navmesh_the_walker_is_stepped_and_it_is_said_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from autoware_carla_scenario.entity import _walker_ai
    from autoware_carla_scenario.trajectory.model import ResolvedTrajectory

    world = _AiWorld(False, 102)
    walker = SimpleNamespace(id=56, get_world=lambda: world)
    path = ResolvedTrajectory([0.0, 25.0], [0.0, 0.0], [0.0, 0.0], [None, None])
    with caplog.at_level(logging.WARNING):
        assert not _walker_ai.drive(walker, path, 0.0, 1.4)
        assert not _walker_ai.drive(walker, path, 1.0, 1.4)
    assert sum("navigation mesh" in r.getMessage() for r in caplog.records) == 1
    assert world.controller.calls == []


def test_a_target_near_enough_counts_as_reached_off_the_walk() -> None:
    """Taken off the line of its walk, the walker projects short of the
    target; 2 m from it in a straight line, it has reached it all the same."""
    from autoware_carla_scenario.entity import _walker_ai

    world = _AiWorld(True, 104)
    where = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    walker = SimpleNamespace(id=59, get_world=lambda: world, get_location=lambda: where)
    assert _walker_ai.drive(walker, path_of(25.0), 0.0, 1.4, where)
    beside = SimpleNamespace(x=9.0, y=1.5, z=0.0)
    assert _walker_ai.drive(walker, path_of(25.0), 3.0, 1.4, beside)
    _walker_ai.release(59)
    assert ("go", 20.0) in world.controller.calls


def _road_between(monkeypatch: pytest.MonkeyPatch, a: float, b: float) -> None:
    """A map whose road runs across the walk from x=a to x=b; walkway else."""
    from autoware_carla_scenario.entity import _walker_ai

    monkeypatch.setattr(
        _walker_ai, "_kind_at", lambda x, y: "road" if a < x < b else "walk"
    )


def test_a_walks_road_stretches_are_found_along_its_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.entity import _walker_ai

    _road_between(monkeypatch, 10.0, 17.0)
    assert _walker_ai.road_stretches(path_of(30.0)) == [(10.5, 16.5)]
    # Ending on the road: the stretch runs to the walk's end.
    _road_between(monkeypatch, 10.0, 99.0)
    assert _walker_ai.road_stretches(path_of(30.0)) == [(10.5, 30.0)]
    # Starting on the road (a walk along its edge from where it appears).
    _road_between(monkeypatch, -5.0, 12.0)
    assert _walker_ai.road_stretches(path_of(30.0)) == [(0.0, 11.5)]
    # Along the road's edge all the way, however long.
    _road_between(monkeypatch, -5.0, 999.0)
    assert _walker_ai.road_stretches(path_of(240.0)) == [(0.0, 240.0)]


def test_a_walker_that_may_cross_is_stepped_over_the_road_and_handed_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the controller to the kerb, stepped across, back on it only on the
    far pavement a metre on; one that may not cross is never stepped."""
    from autoware_carla_scenario.entity import _walker_ai

    _road_between(monkeypatch, 10.0, 17.0)
    world = _AiWorld(True, 105)
    path = path_of(30.0)
    steps: dict[float, bool] = {}
    for factor, walker_id in ((1.0, 61), (0.0, 62)):
        walker = SimpleNamespace(
            id=walker_id, get_world=lambda: world, apply_control=lambda c: None
        )
        for here in (0.0, 5.0, 10.2, 14.0, 17.0, 17.6, 19.0, 25.0):
            where = SimpleNamespace(x=here, y=0.0, z=0.0)
            on = _walker_ai.drive(walker, path, here, 1.4, where, factor)
            if factor == 1.0:
                steps[here] = not on
            else:
                assert on  # never stepped
        _walker_ai.release(walker_id)
    # Stepped from just before the kerb to a metre past the far one.
    assert [here for here, stepped in steps.items() if stepped] == [10.2, 14.0, 17.0]


def test_a_walk_ending_on_the_road_is_stepped_to_its_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.entity import _walker_ai

    _road_between(monkeypatch, 10.0, 99.0)
    world = _AiWorld(True, 106)
    walker = SimpleNamespace(
        id=63, get_world=lambda: world, apply_control=lambda c: None
    )
    path = path_of(30.0)
    stepped = []
    for here in (0.0, 12.0, 29.0):
        where = SimpleNamespace(x=here, y=0.0, z=0.0)
        stepped.append(not _walker_ai.drive(walker, path, here, 1.4, where, 1.0))
    _walker_ai.release(63)
    assert stepped == [False, True, True]


def test_a_jump_as_the_controller_starts_is_said(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from autoware_carla_scenario.entity import _walker_ai

    world = _AiWorld(True, 107)
    walker = SimpleNamespace(
        id=64, get_world=lambda: world, apply_control=lambda c: None
    )
    path = path_of(30.0)
    with caplog.at_level(logging.WARNING):
        _walker_ai.drive(walker, path, 0.0, 1.4, SimpleNamespace(x=0.0, y=0.0, z=0.0))
        _walker_ai.drive(walker, path, 0.0, 1.4, SimpleNamespace(x=2.0, y=0.0, z=0.0))
    _walker_ai.release(64)
    assert sum("jumped" in r.getMessage() for r in caplog.records) == 1


def test_a_target_on_the_road_is_moved_to_the_nearest_walkway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A walker that may not use the road is never sent where CARLA finds it
    no route (it would walk to a random point instead): a target on the road
    goes to the nearest walkway; at its walk's end so moved, it is held."""
    from autoware_carla_scenario.entity import _walker_ai

    monkeypatch.setattr(
        _walker_ai, "_kind_at", lambda x, y: "road" if 5.0 < x < 22.0 else "walk"
    )
    monkeypatch.setattr(_walker_ai, "_nearest_walk", lambda x, y: (x, 3.0))
    world = _AiWorld(True, 107)
    sent: list[Any] = []
    world.controller.go_to_location = lambda location: sent.append(  # type: ignore[method-assign]
        (round(location.x, 1), round(location.y, 1))
    )
    where = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    walker = SimpleNamespace(id=62, get_world=lambda: world)
    path = path_of(20.0)
    assert _walker_ai.drive(walker, path, 0.0, 1.4, where, 0.0)
    assert sent == [(10.0, 3.0)]
    where.x, where.y = 20.0, 2.0  # near its end, as near as it goes
    assert _walker_ai.drive(walker, path, 19.0, 1.4, where, 0.0)
    assert sent[-1] == (20.0, 3.0)
    assert world.controller.calls[-1] == ("speed", 0.0)
    _walker_ai.release(62)
    # A walker that may cross keeps its targets (its road stretches stepped).
    world = _AiWorld(True, 108)
    sent.clear()
    world.controller.go_to_location = lambda location: sent.append(  # type: ignore[method-assign]
        (round(location.x, 1), round(location.y, 1))
    )
    walker = SimpleNamespace(id=63, get_world=lambda: world)
    monkeypatch.setattr(_walker_ai, "road_stretches", lambda path: [])
    assert _walker_ai.drive(
        walker, path, 0.0, 1.4, SimpleNamespace(x=0.0, y=0.0, z=0.0), 1.0
    )
    assert sent == [(10.0, 0.0)]
    _walker_ai.release(63)


def test_a_stepped_walker_gives_way_to_a_walker_in_its_way() -> None:
    """Off its controller no crowd steers it round others: it stands for a
    walker just ahead on its walk, for at most GIVE_WAY_MAX_S at a time."""
    from autoware_carla_scenario.entity import _walker_ai

    world = _SnapshotWorld(
        {
            70: ("walker.pedestrian.0001", 0.0, 0.0, 0.0),
            71: ("walker.pedestrian.0001", 1.2, 0.1, 0.0),
        }
    )
    world.id = 709  # type: ignore[misc]
    walker = SimpleNamespace(id=70, get_world=lambda: world)
    here = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    path = path_of(20.0)
    assert _walker_ai.give_way(walker, path, 0.0, here)
    ticks = 1
    while _walker_ai.give_way(walker, path, 0.0, here):
        ticks += 1
    assert abs(ticks - _walker_ai.GIVE_WAY_MAX_S / 0.05) <= 1.0
    # The way clear, and then a walker beside the walk: walks on.
    world.actors[71] = ("walker.pedestrian.0001", 1.2, 1.5, 0.0)
    world2 = _SnapshotWorld(dict(world.actors))
    world2.id = 710  # type: ignore[misc]
    walker2 = SimpleNamespace(id=70, get_world=lambda: world2)
    assert not _walker_ai.give_way(walker2, path, 0.0, here)
    _walker_ai.release(70)


def test_a_walker_off_its_controller_has_its_physics_and_collisions_back() -> None:
    """Its controller's start set both off; stepping without collisions drops
    the walker through the ground."""
    from autoware_carla_scenario.entity import _walker_ai

    calls: list[tuple[str, bool]] = []
    walker = SimpleNamespace(
        set_simulate_physics=lambda on: calls.append(("physics", on)),
        set_collisions=lambda on: calls.append(("collisions", on)),
    )
    _walker_ai._own_steps(walker)
    assert calls == [("physics", True), ("collisions", True)]


def test_a_road_user_in_the_lane_beside_does_not_take_a_spot() -> None:
    """Outlines, not circles: a car 3.5 m to the side leaves the spot clear,
    one overlapping it (a margin added) takes it."""
    from autoware_carla_scenario.utils import clearance

    world = _SnapshotWorld(
        {
            1: ("vehicle.mini", 0.0, 3.5, 0.0),
            2: ("vehicle.mini", 30.0, 0.0, 0.0),
        }
    )
    world.id = 811  # type: ignore[misc]
    assert clearance.overlaps(world, 99, 0.0, 0.0, 0.0, 2.0, 1.0) is None
    assert clearance.overlaps(world, 99, 26.0, 0.5, 0.0, 2.0, 1.0) == 2
    assert clearance.overlaps(world, 99, 25.4, 0.0, 0.0, 2.0, 1.0) is None


def test_a_spawn_whose_place_is_taken_takes_one_further_on_if_allowed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """SpawnAction does not wait (its trigger does): a place taken as it fires
    is given up for the nearest clear one within shift_max along its lane,
    else it spawns where it was due; each said, naming what stood there."""
    from autoware_carla_scenario.actions import SpawnAction
    from autoware_carla_scenario.actions import spawn as spawn_module
    from autoware_carla_scenario.coordinate import Lanelet2Pose

    taken_until = {"m": 7.0}
    monkeypatch.setattr(
        spawn_module,
        "resolve_position",
        lambda position, reference: (float(position.s), 0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        spawn_module,
        "_along_lane",
        lambda pose, d, lanes=None: Lanelet2Pose(pose.lanelet_id, pose.s + d),
    )
    monkeypatch.setattr(
        spawn_module,
        "_taken",
        lambda world, actor_id, x, y, yaw, hl, hw, ahead=0.0: (
            5 if x < taken_until["m"] else -1
        ),
    )
    held: list[float] = []
    monkeypatch.setattr(
        spawn_module, "_hold", lambda world, actor_id, x, y, yaw, hl, hw: held.append(x)
    )

    def place(shift: float) -> Any:
        action = SpawnAction("npc1", position=Lanelet2Pose(3, 0.0), shift_max=shift)
        return action._place(None, 1, 2.0, 1.0)  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING):
        assert place(30.0)[0] == 8.0
        assert place(0.0)[0] == 0.0
        taken_until["m"] = 999.0
        assert place(30.0)[0] == 0.0
    said = [r.getMessage() for r in caplog.records]
    assert any("8 m on along its lane" in m and "actor 5" in m for m in said)
    assert sum("where it was due, on actor 5" in m for m in said) == 2
    assert held == [8.0, 0.0, 0.0]  # each place claimed for the frame
    taken_until["m"] = -1.0
    assert place(30.0)[0] == 0.0


def test_what_a_spawn_refuses() -> None:
    from autoware_carla_scenario.actions import SpawnAction
    from autoware_carla_scenario.coordinate import Lanelet2Pose

    with pytest.raises(ValueError, match="shift_max"):
        SpawnAction("npc1", position=Lanelet2Pose(1, 0.0), shift_max=-1.0)
    with pytest.raises(ValueError, match="ego"):
        SpawnAction("Ego")
    with pytest.raises(ValueError, match="one of the two"):
        SpawnAction()


def test_a_spawn_with_no_place_brings_the_entity_up_where_it_was_declared() -> None:
    """Parked 500 m under its declared spawn: brought straight up."""
    from autoware_carla_scenario.actions import SpawnAction
    from autoware_carla_scenario.entity.registry import (
        register_entity,
        unregister_entity,
    )

    transforms: list[Any] = []
    actor = SimpleNamespace(
        id=31,
        type_id="vehicle.mini.cooper",
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2.0, y=1.0, z=0.8)),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=4.0, y=5.0, z=-499.8),
            rotation=SimpleNamespace(yaw=90.0),
        ),
        set_simulate_physics=lambda on: None,
        set_transform=transforms.append,
        set_target_velocity=lambda v: None,
    )
    register_entity("npc31", SimpleNamespace(actor=actor))
    try:
        SpawnAction("npc31").tick(None, 0.0)  # type: ignore[arg-type]
    finally:
        unregister_entity("npc31")
    (put,) = transforms
    assert (put.location.x, put.location.y, put.location.z) == pytest.approx(
        (4.0, 5.0, 0.2)
    )
    assert put.rotation.yaw == pytest.approx(90.0)


def test_a_place_clear_condition_holds_while_nothing_stands_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic place (a lanelet pose) and a footprint: passes while no road
    user overlaps it, says once what stands there."""
    from autoware_carla_scenario.conditions import PlaceClearCondition
    from autoware_carla_scenario.conditions import place_clear as place_module
    from autoware_carla_scenario.coordinate import Lanelet2Pose
    from autoware_carla_scenario.coordinate import transform as transform_module

    monkeypatch.setattr(
        transform_module,
        "to_carla_world",
        lambda pose: SimpleNamespace(x=10.0, y=2.0, z=0.0, yaw=90.0),
    )
    seen: list[tuple[float, ...]] = []
    occupant = {"id": 7}

    def look(
        world: Any, x: float, y: float, yaw: float, hl: float, hw: float, m: float
    ) -> int:
        seen.append((x, y, yaw, hl, hw, m))
        return occupant["id"]

    monkeypatch.setattr(place_module, "_occupant", look)
    condition = PlaceClearCondition(Lanelet2Pose(3, 5.0, 1.0), 4.0, 2.0, label="p")
    assert condition.check(None, 1.0) is None  # type: ignore[arg-type]
    assert seen[-1] == (10.0, 2.0, 90.0, 2.0, 1.0, 0.3)
    occupant["id"] = -1
    result = condition.check(None, 2.0)  # type: ignore[arg-type]
    assert result is not None and result.passed
    assert condition.get_details()["lanelet_id"] == 3
    with pytest.raises(ValueError):
        PlaceClearCondition(Lanelet2Pose(3, 5.0), 0.0, 2.0, label="p")


def test_the_place_clear_card_builds_its_condition() -> None:
    """The card's lanelet pose and footprint reach the condition."""
    from autoware_carla_scenario.authoring import _builders_generated as builders
    from autoware_carla_scenario.conditions import PlaceClearCondition

    compiled = SimpleNamespace(
        label="p",
        params={
            "lanelet_id": 12,
            "s": 3.0,
            "t": -1.5,
            "heading": 0.0,
            "length": 0.5,
            "width": 0.4,
            "margin": 0.3,
        },
    )
    built = builders.build_place_clear_condition(compiled, [], None)  # type: ignore[arg-type]
    assert isinstance(built, PlaceClearCondition)
    details = built.get_details()
    assert (details["lanelet_id"], details["s"], details["t"]) == (12, 3.0, -1.5)
    assert (details["length"], details["width"], details["margin"]) == (0.5, 0.4, 0.3)


def test_a_walker_the_crowd_took_for_run_over_goes_on_as_a_new_one(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CARLA's crowd kills a walker a vehicle merely came near (and removes
    it ten seconds on): a new walker of the same role takes its place and
    walks on."""
    import typesafe_carla.carla as tcarla

    from autoware_carla_scenario.entity import (
        PedestrianEntity,
        PedestrianEntityConfig,
        SpawnTransform,
    )
    from autoware_carla_scenario.entity import pedestrian_entity as pe

    spawned: list[Any] = []

    class _Blueprint:
        def __init__(self) -> None:
            self.attributes: dict[str, str] = {}

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

        def has_attribute(self, key: str) -> bool:
            return True

    class _Walker:
        def __init__(self, walker_id: int) -> None:
            self.id = walker_id
            self.destroyed = False

        def get_world(self) -> Any:
            return world

        def get_transform(self) -> Any:
            return tcarla.Transform(tcarla.Location(3.0, 4.0, 1.0), tcarla.Rotation())

        def destroy(self) -> None:
            self.destroyed = True

        def apply_control(self, control: Any) -> None:
            pass

    def try_spawn_actor(blueprint: Any, transform: Any) -> Any:
        new = _Walker(80 + len(spawned))
        spawned.append((blueprint.attributes["role_name"], transform.location.x, new))
        return new

    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: _Blueprint()),
        try_spawn_actor=try_spawn_actor,
    )
    entity = PedestrianEntity(
        PedestrianEntityConfig(
            role_name="w9", spawn_location=SpawnTransform(tcarla.Transform())
        )
    )
    old = _Walker(70)
    entity._walker = old  # type: ignore[assignment]
    entity.walk_along(path_of(20.0), 1.2)
    killed = {70}
    monkeypatch.setattr(pe, "_killed_by_crowd", lambda walker: int(walker.id) in killed)
    monkeypatch.setattr(pe, "_walk_by_ai", lambda *args: True)
    with caplog.at_level(logging.WARNING):
        entity.walk_on()
    assert old.destroyed
    assert [(role, x) for role, x, _ in spawned] == [("w9", 3.0)]
    assert entity.actor is spawned[0][2]
    assert entity.walking
    assert any("goes on in its place" in r.getMessage() for r in caplog.records)


def test_a_driver_run_that_has_not_begun_is_tried_again_each_tick() -> None:
    """One that could not begin on its trigger's tick (its spot taken) is
    executed again on the next; once begun, only while it paces or walks."""
    from autoware_carla_scenario.actions.routing import RoutingAction, RoutingMode

    action = RoutingAction.__new__(RoutingAction)
    action._mode = RoutingMode.DRIVER
    action._stepping = False
    action._begun = False
    assert action._reissues_by_default()
    action._begun = True
    assert not action._reissues_by_default()
    action._stepping = True
    assert action._reissues_by_default()


def test_a_spot_claimed_this_frame_is_taken_before_the_snapshot_shows_it() -> None:
    """Two entities due at one spot on one tick: the second waits."""
    from autoware_carla_scenario.utils import clearance

    world = _SnapshotWorld({})
    world.id = 812  # type: ignore[misc]
    assert clearance.overlaps(world, 2, 10.0, 0.0, 0.0, 2.0, 1.0) is None
    clearance.claim_spot(world, 1, 10.5, 0.0, 0.0, 2.0, 1.0)
    assert clearance.overlaps(world, 2, 10.0, 0.0, 0.0, 2.0, 1.0) == 1
    assert clearance.overlaps(world, 1, 10.0, 0.0, 0.0, 2.0, 1.0) is None  # its own
    clearance._CACHE.claims_frame = -5  # a later frame: shown by the snapshot
    assert clearance.overlaps(world, 2, 10.0, 0.0, 0.0, 2.0, 1.0) is None


def test_a_spawn_at_speed_needs_room_to_stop_ahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A car entering at 5.7 m/s 0.3 m behind a slower one ran into it: the
    place checked is stretched forward by its stopping distance and 2 m."""
    from autoware_carla_scenario.actions import spawn as spawn_module

    looked: list[tuple[float, float, float]] = []

    def overlaps(
        world: Any, actor_id: int, x: float, y: float, yaw: float, hl: float, hw: float
    ) -> Any:
        looked.append((round(x, 3), round(hl, 3), round(hw, 3)))
        return None

    from autoware_carla_scenario.utils import clearance

    monkeypatch.setattr(clearance, "overlaps", overlaps)
    assert spawn_module._taken(None, 1, 10.0, 0.0, 0.0, 2.3, 1.0, 6.0) == -1  # type: ignore[arg-type]
    assert looked == [(13.0, 5.3, 1.0)]


def test_a_shift_runs_along_the_lanes_given_and_no_further(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Along its route's lanelets, not the straightest way on: vehicle_165,
    shifted 10 m on a 4 m route, and vehicle_12, shifted down another branch,
    left their routes."""
    from autoware_carla_scenario import coordinate
    from autoware_carla_scenario.actions import spawn as spawn_module
    from autoware_carla_scenario.coordinate import Lanelet2Pose

    monkeypatch.setattr(coordinate, "lanelet_length", lambda lanelet: 10.0)
    on = spawn_module._along_lane(Lanelet2Pose(7, 6.0), 12.0, [5, 7, 9])
    assert on is not None and (on.lanelet_id, on.s) == (9, 8.0)
    assert spawn_module._along_lane(Lanelet2Pose(7, 6.0), 15.0, [5, 7, 9]) is None
    assert spawn_module._along_lane(Lanelet2Pose(8, 6.0), 1.0, [5, 7, 9]) is None


def test_a_bicycle_spawn_checks_the_place_of_its_recorded_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ridden by a walker mesh (0.43 x 0.16 m), a bicycle needs 1.8 x 0.7 m:
    its footprint, not its actor's outline, is what the place is checked for."""
    from autoware_carla_scenario.actions import SpawnAction
    from autoware_carla_scenario.actions import spawn as spawn_module
    from autoware_carla_scenario.coordinate import Lanelet2Pose
    from autoware_carla_scenario.entity.registry import (
        register_entity,
        unregister_entity,
    )

    sizes: list[tuple[float, float]] = []

    def place(self: Any, world: Any, actor_id: int, hl: float, hw: float) -> Any:
        sizes.append((hl, hw))
        return (0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(spawn_module.SpawnAction, "_place", place)
    monkeypatch.setattr(spawn_module, "_put", lambda *args: None)
    actor = SimpleNamespace(
        id=41,
        type_id="walker.pedestrian.0015",
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=0.215, y=0.08, z=0.9)),
    )
    register_entity("npc41", SimpleNamespace(actor=actor))
    try:
        SpawnAction("npc41", position=Lanelet2Pose(1, 0.0), length=1.8, width=0.7).tick(
            None, 0.0
        )  # type: ignore[arg-type]
        SpawnAction("npc41", position=Lanelet2Pose(1, 0.0)).tick(None, 0.0)  # type: ignore[arg-type]
    finally:
        unregister_entity("npc41")
    assert sizes == [(0.9, 0.35), (0.215, 0.08)]
