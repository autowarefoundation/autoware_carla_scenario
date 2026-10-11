"""Waypoint conditions: each vertex is departed on its condition.

A vertex is departed when its time is reached on the trajectory's clock, when
any other condition (``advance``) holds, or -- with neither -- on arrival.
Between a vertex departed at ``T`` and the next one the entity arrives at the
next vertex's time if it has one not before ``T``, and goes at the action's
speed otherwise.  These tests run the action against the simulated vehicle
(and walker) of ``test_follow_trajectory_action``; trajectories with times
alone or nothing at all are that module's, and stay as they were.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Optional

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    AlwaysTrueCondition,
    AndCondition,
    BaseCondition,
    CarlaWorldPose,
    ComparisonRule,
    ElapsedTimeCondition,
    EntityDistanceCondition,
    FollowTrajectoryAction,
    MapPose,
    ReferenceContext,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectoryTimeCondition,
    TrajectoryTiming,
    TrajectoryVertex,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions._departures import (
    DepartureTimeline,
    TimelineVertex,
)
from autoware_carla_scenario.actions.follow_trajectory import HIDDEN_DEPTH_M
from autoware_carla_scenario.conditions.base import ScenarioResult
from autoware_carla_scenario.entity.pedestrian_entity import _UE5_WALKER_SPEED_SCALE
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity

from .test_follow_trajectory_action import _DT, _Actor, _Entity, _line, _run, _World


class _Recorder(BaseCondition):
    """A condition that remembers when it was asked, and answers as told.

    ``inner`` answers when given; otherwise :attr:`open` does.
    """

    def __init__(
        self, inner: Optional[BaseCondition] = None, *, open: bool = False
    ) -> None:
        super().__init__(label="recorder")
        self.inner = inner
        self.open = open
        self.asked: list[float] = []
        self.answers: list[bool] = []

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        self.asked.append(elapsed)
        if self.inner is not None:
            result = self.inner.check(world, elapsed)
        else:
            result = (
                ScenarioResult(passed=True, message="open", elapsed_seconds=elapsed)
                if self.open
                else None
            )
        self.answers.append(result is not None)
        return result

    @property
    def released_at(self) -> float:
        """The scenario time the condition first held."""
        return self.asked[self.answers.index(True)]


@pytest.fixture
def entity() -> Iterator[_Entity]:
    registered = _Entity(_Actor())
    register_entity("npc1", registered)
    yield registered
    unregister_entity("npc1")


def _x(x: float) -> CarlaWorldPose:
    return CarlaWorldPose(x, 0.0, 0.0, yaw=0.0)


def _path(*vertices: tuple[float, Any]) -> Trajectory:
    """A line along CARLA +x: ``(x, departure)`` per vertex.

    A number is a time, a condition is an ``advance``, ``None`` is nothing.
    """
    return Trajectory(
        "path",
        [
            TrajectoryVertex(
                _x(x),
                TrajectoryTimeCondition(departure)
                if isinstance(departure, (int, float))
                else departure,
            )
            for x, departure in vertices
        ],
    )


def _trace(
    action: FollowTrajectoryAction, actor: _Actor, seconds: float, start: float = 0.0
) -> list[tuple[float, float, float]]:
    """Run like ``_run`` and return ``(elapsed, x, speed)`` after every tick."""
    world = _World()
    elapsed = start
    out = []
    for _ in range(int(round(seconds / _DT))):
        action.tick(world, elapsed)
        out.append((elapsed, actor.x, actor.speed))
        actor.step(_DT, driven=False)
        elapsed += _DT
    return out


def _at(trace: list[tuple[float, float, float]], when: float) -> tuple[float, float]:
    """``(x, speed)`` on the tick nearest *when*."""
    _, x, speed = min(trace, key=lambda row: abs(row[0] - when))
    return x, speed


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class TestTheModel:
    def test_a_time_is_a_condition(self) -> None:
        vertex = TrajectoryVertex(MapPose(0.0, 0.0), TrajectoryTimeCondition(2.0))
        assert vertex.time == 2.0 and vertex.gate is None
        assert vertex == TrajectoryVertex(
            MapPose(0.0, 0.0), TrajectoryTimeCondition(2.0)
        )
        gate = AlwaysTrueCondition()
        other = TrajectoryVertex(MapPose(0.0, 0.0), gate)
        assert other.time is None and other.gate is gate
        assert TrajectoryVertex(MapPose(0.0, 0.0)).advance is None

    def test_one_condition_per_vertex_and_no_bare_time(self) -> None:
        with pytest.raises(TypeError, match="takes no time any more"):
            TrajectoryVertex(MapPose(0.0, 0.0), 1.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="takes no time any more"):
            TrajectoryVertex(MapPose(0.0, 0.0), time=1.0)  # type: ignore[call-arg]

    def test_an_advance_is_a_condition(self) -> None:
        with pytest.raises(TypeError, match="advance must be a condition"):
            TrajectoryVertex(MapPose(0.0, 0.0), advance="go")  # type: ignore[arg-type]

    def test_a_timed_path_departs_every_vertex_on_some_condition(self) -> None:
        gate = AlwaysTrueCondition()
        path = _path((0.0, 0.0), (10.0, gate), (20.0, 2.0))
        assert path.is_gated and path.has_times and not path.is_timed
        assert not _line().is_gated and _line().is_timed
        with pytest.raises(ValueError, match="either every vertex has a time"):
            _path((0.0, 0.0), (10.0, None), (20.0, 2.0))
        with pytest.raises(ValueError, match="times decrease"):
            _path((0.0, 2.0), (10.0, gate), (20.0, 1.0))

    def test_the_last_vertex_may_carry_a_condition(self) -> None:
        assert _path((0.0, None), (10.0, AlwaysTrueCondition())).is_gated

    def test_a_closed_path_carries_no_time(self) -> None:
        with pytest.raises(ValueError, match="closed trajectory cannot carry times"):
            Trajectory(
                "loop",
                [
                    TrajectoryVertex(_x(0.0), TrajectoryTimeCondition(0.0)),
                    TrajectoryVertex(_x(10.0), AlwaysTrueCondition()),
                ],
                closed=True,
            )

    def test_gated_replaces_what_the_vertices_named_depart_on(self) -> None:
        gate = AlwaysTrueCondition()
        untimed = _line(duration=None).gated({2: gate})
        assert untimed.vertices[2].advance is gate
        # A time is replaced like any other condition.
        timed = _line().gated({2: gate})
        assert timed.vertices[2].gate is gate and timed.vertices[2].time is None
        assert timed.vertices[3].time == _line().vertices[3].time
        with pytest.raises(ValueError, match="either every vertex has a time"):
            _line(duration=None).gated({1: TrajectoryTimeCondition(5.0)})
        with pytest.raises(IndexError, match="no vertex 9"):
            _line(duration=None).gated({9: gate})


class TestTheTimeline:
    """The arithmetic, without an entity."""

    def _timeline(self, *vertices: TimelineVertex, speed: float = 10.0) -> Any:
        return DepartureTimeline(list(vertices), vertices[-1].distance, False, speed)

    def test_times_alone_are_the_timed_interpolation(self) -> None:
        timeline = self._timeline(
            TimelineVertex(0.0, 0.0),
            TimelineVertex(10.0, 1.0),
            TimelineVertex(30.0, 3.0),
        )
        timeline.start(0, 0.0, 0.0)
        assert timeline.end_kind == "end" and timeline.end_arrival == 3.0
        assert timeline.at(0.5) == (pytest.approx(5.0), pytest.approx(10.0))
        assert timeline.at(2.0) == (pytest.approx(20.0), pytest.approx(10.0))

    def test_a_late_departure_goes_at_the_speed(self) -> None:
        timeline = self._timeline(
            TimelineVertex(0.0, 0.0),
            TimelineVertex(10.0, None, gate=AlwaysTrueCondition()),
            TimelineVertex(20.0, 2.0),
            TimelineVertex(30.0, 6.0),
        )
        timeline.start(0, 0.0, 0.0)
        assert timeline.end_kind == "gate" and timeline.end_vertex == 1
        assert timeline.end_arrival == pytest.approx(1.0)  # 10 m at 10 m/s
        timeline.depart(3.0)  # past vertex 2's time: at the speed
        assert timeline.times[-2:] == [pytest.approx(4.0), 6.0]  # 6.0 is not past
        assert timeline.at(5.0) == (pytest.approx(25.0), pytest.approx(5.0))


# ---------------------------------------------------------------------------
# POSITION mode
# ---------------------------------------------------------------------------


class TestPosition:
    def test_time_condition_time(self, entity: _Entity) -> None:
        """Departed before the next time: interpolated to arrive on it."""
        gate = _Recorder(ElapsedTimeCondition(1.5, label="go"))
        action = FollowTrajectoryAction(
            "npc1",
            _path((0.0, 0.0), (10.0, gate), (20.0, 3.0), (30.0, 4.0)),
            TrajectoryTiming(),
            speed=10.0,
        )
        trace = _trace(action, entity.actor, 4.5)
        # At the speed to the condition vertex (no time there), arriving at 1 s.
        assert gate.asked[0] == pytest.approx(1.0, abs=_DT + 1e-9)
        assert _at(trace, 0.5) == (pytest.approx(5.0), pytest.approx(10.0))
        assert _at(trace, 1.25) == (pytest.approx(10.0), 0.0)
        released = gate.released_at
        assert released == pytest.approx(1.5, abs=_DT + 1e-9)
        # Then over to the next time: 10 m in 3.0 - departure.
        x, speed = _at(trace, 2.25)
        pace = 10.0 / (3.0 - released)
        assert x == pytest.approx(10.0 + pace * (2.25 - released), abs=1e-6)
        assert speed == pytest.approx(pace)
        assert _at(trace, 3.5) == (pytest.approx(25.0), pytest.approx(10.0))
        assert action.finished
        assert entity.actor.x == pytest.approx(30.0)

    def test_a_late_departure_goes_at_the_speed(self, entity: _Entity) -> None:
        """Departed after the next times have passed: at the action's speed."""
        gate = _Recorder(ElapsedTimeCondition(3.5, label="late"))
        action = FollowTrajectoryAction(
            "npc1",
            _path((0.0, 0.0), (10.0, gate), (20.0, 3.0), (30.0, 4.0)),
            TrajectoryTiming(),
            speed=10.0,
        )
        trace = _trace(action, entity.actor, 6.0)
        released = gate.released_at
        x, speed = _at(trace, 4.0)
        assert x == pytest.approx(10.0 + 10.0 * (4.0 - released), abs=1e-6)
        assert speed == pytest.approx(10.0)
        # 20 m more at 10 m/s: at the end 2 s after departing.
        ended = next(e for e, x, _ in trace if x >= 30.0 - 1e-6)
        assert ended == pytest.approx(released + 2.0, abs=_DT + 1e-9)
        assert action.finished

    def test_a_condition_holding_on_arrival_does_not_stop_it(
        self, entity: _Entity
    ) -> None:
        gate = _Recorder(AlwaysTrueCondition())
        action = FollowTrajectoryAction(
            "npc1",
            _path((0.0, 0.0), (10.0, gate), (20.0, 3.0)),
            TrajectoryTiming(),
            speed=10.0,
        )
        trace = _trace(action, entity.actor, 2.9)
        assert len(gate.asked) == 1
        assert all(speed > 0.0 for _, _, speed in trace)

    def test_a_condition_on_the_first_vertex_holds_the_start(
        self, entity: _Entity
    ) -> None:
        action = FollowTrajectoryAction(
            "npc1",
            _path((0.0, ElapsedTimeCondition(1.0, label="go")), (10.0, None)),
            speed=5.0,
        )
        trace = _trace(action, entity.actor, 2.0)
        assert _at(trace, 0.5) == (pytest.approx(0.0), 0.0)
        x, speed = _at(trace, 1.5)
        assert speed == pytest.approx(5.0)
        assert x == pytest.approx(2.5, abs=5.0 * _DT + 1e-6)

    def test_a_condition_on_the_last_vertex_ends_the_run_when_it_holds(
        self, entity: _Entity
    ) -> None:
        gate = _Recorder(open=False)
        action = FollowTrajectoryAction(
            "npc1", _path((0.0, None), (10.0, None), (20.0, gate)), speed=10.0
        )
        _run(action, entity.actor, 3.0)
        assert not action.finished
        assert action.held_vertex == 2
        assert entity.actor.x == pytest.approx(20.0)
        gate.open = True
        _run(action, entity.actor, 0.2, start=3.0)
        assert action.finished
        assert action.state is ActionState.COMPLETE

    def test_absolute_scale_and_offset(self, entity: _Entity) -> None:
        """Times go through the timing; conditions read the scenario clock."""
        action = FollowTrajectoryAction(
            "npc1",
            _path(
                (0.0, 0.0), (10.0, ElapsedTimeCondition(2.0, label="go")), (20.0, 2.0)
            ),
            TrajectoryTiming(ReferenceContext.ABSOLUTE, scale=2.0, offset=1.0),
            speed=10.0,
        )
        trace = _trace(action, entity.actor, 6.0)
        # Vertex 0 departs at 0 * 2 + 1 = 1 s; 10 m at 10 m/s; waits for 2 s
        # on the scenario clock; vertex 2 is at 2 * 2 + 1 = 5 s.
        assert _at(trace, 0.5) == (pytest.approx(0.0), 0.0)
        assert _at(trace, 1.5)[0] == pytest.approx(5.0)
        x, speed = _at(trace, 3.5)
        assert speed == pytest.approx(10.0 / 3.0)
        assert x == pytest.approx(10.0 + 1.5 * 10.0 / 3.0, abs=1e-6)

    def test_an_initial_offset_past_a_condition_skips_it(self, entity: _Entity) -> None:
        gate = _Recorder(open=False)
        action = FollowTrajectoryAction(
            "npc1",
            _line(duration=None).gated({2: gate}),
            initial_distance_offset=12.0,
            speed=10.0,
        )
        _run(action, entity.actor, 1.0)
        assert gate.asked == []
        assert action.finished

    def test_an_initial_offset_before_a_condition_reaches_it_sooner(
        self, entity: _Entity
    ) -> None:
        gate = _Recorder(open=False)
        action = FollowTrajectoryAction(
            "npc1",
            _line(duration=None).gated({2: gate}),
            initial_distance_offset=5.0,
            speed=10.0,
        )
        _run(action, entity.actor, 1.0)
        assert gate.asked[0] == pytest.approx(0.5, abs=_DT + 1e-9)
        assert entity.actor.x == pytest.approx(10.0)

    def test_waiting_is_on_the_vertex_itself(self, entity: _Entity) -> None:
        """Waiting before a zero-duration jump stays on the waiting vertex."""
        action = FollowTrajectoryAction(
            "npc1",
            _path(
                (0.0, 0.0),
                (10.0, _Recorder(open=False)),
                (20.0, 2.0),
                (20.0, 2.0),
                (30.0, 4.0),
            ),
            TrajectoryTiming(),
            speed=10.0,
        )
        _run(action, entity.actor, 3.0)
        assert action.held_vertex == 1
        assert entity.actor.x == pytest.approx(10.0)
        assert entity.actor.speed == 0.0

    def test_a_run_ended_while_waiting_holds_nothing(self, entity: _Entity) -> None:
        action = FollowTrajectoryAction(
            "npc1",
            _line(duration=None).gated({1: _Recorder(open=False)}),
            speed=10.0,
            until=ElapsedTimeCondition(2.0, label="enough"),
        )
        _run(action, entity.actor, 1.5)
        assert action.held_vertex == 1
        _run(action, entity.actor, 1.5, start=1.5)
        assert action.state is ActionState.COMPLETE
        assert action.held_vertex is None

    def test_a_closed_path_checks_its_conditions_on_every_lap(
        self, entity: _Entity
    ) -> None:
        at_start = _Recorder(AlwaysTrueCondition())
        corner = _Recorder(AlwaysTrueCondition())
        action = FollowTrajectoryAction(
            "npc1", _square().gated({0: at_start, 1: corner}), speed=10.0
        )
        # A lap is 40 m, 4 s: the start is reached at 0, 4 and 8 s, the
        # corner at 1 and 5 s.
        _run(action, entity.actor, 8.5)
        assert at_start.asked == pytest.approx([0.0, 4.0, 8.0], abs=_DT + 1e-9)
        assert corner.asked == pytest.approx([1.0, 5.0], abs=_DT + 1e-9)
        assert not action.finished

    def test_a_closed_path_waits_on_its_last_vertex(self, entity: _Entity) -> None:
        action = FollowTrajectoryAction(
            "npc1", _square().gated({3: _Recorder(open=False)}), speed=10.0
        )
        _run(action, entity.actor, 6.0)
        assert (entity.actor.x, entity.actor.y) == (
            pytest.approx(0.0),
            pytest.approx(10.0),
        )
        assert action.held_vertex == 3

    def test_every_run_rearms_its_conditions(self, entity: _Entity) -> None:
        gate = _Recorder(AlwaysTrueCondition())
        action = FollowTrajectoryAction(
            "npc1",
            _path((0.0, 0.0), (10.0, gate), (20.0, 2.0)),
            TrajectoryTiming(),
            once=False,
            speed=10.0,
        )
        world = _World()
        elapsed = 0.0
        ends = 0
        while ends < 2 and elapsed < 10.0:
            was = action.finished
            action.tick(world, elapsed)
            entity.actor.step(_DT, driven=False)
            elapsed += _DT
            if action.finished and not was:
                ends += 1
        assert ends == 2
        assert len(gate.asked) == 2


class TestHidden:
    def test_hidden_until_its_first_time_and_after_its_end(
        self, entity: _Entity
    ) -> None:
        gate = _Recorder(ElapsedTimeCondition(3.0, label="go"))
        action = FollowTrajectoryAction(
            "npc1",
            _path((0.0, 1.0), (10.0, gate), (20.0, 4.0)),
            TrajectoryTiming(ReferenceContext.ABSOLUTE),
            hidden_outside_trajectory=True,
            speed=10.0,
        )
        _run(action, entity.actor, 0.5)
        assert entity.actor.z == pytest.approx(-HIDDEN_DEPTH_M)
        _run(action, entity.actor, 2.0, start=0.5)
        # 2.5 s: waiting at the condition vertex, in the world.
        assert entity.actor.physics is True
        assert entity.actor.x == pytest.approx(10.0)
        assert action.held_vertex == 1
        _run(action, entity.actor, 1.2, start=2.5)
        assert not action.finished and entity.actor.physics is True
        _run(action, entity.actor, 0.5, start=3.7)
        assert action.finished
        assert entity.actor.physics is False


# ---------------------------------------------------------------------------
# Other conditions
# ---------------------------------------------------------------------------


class TestOtherConditions:
    def test_a_condition_on_the_distance_to_another_entity(self) -> None:
        """Wait at the vertex until the lead is more than 30 m away."""
        lead = _Located(x=35.0, speed=2.0)
        lead.attributes = {"role_name": "lead"}
        npc = _Located(speed=5.0)
        npc.attributes = {"role_name": "npc2"}
        world = _ActorsWorld([npc, lead])
        gate = _Recorder(
            EntityDistanceCondition(
                "npc2", "lead", 30.0, ComparisonRule.GREATER_THAN, label="clear"
            )
        )
        register_entity("npc2", _Entity(npc))
        register_entity("lead", _Entity(lead))
        try:
            action = FollowTrajectoryAction(
                "npc2", _line(length=60.0, duration=None, count=7).gated({1: gate})
            )
            elapsed = 0.0
            for _ in range(int(round(6.0 / _DT))):
                action.tick(world, elapsed)
                npc.step(_DT, driven=False)
                lead.step(_DT, driven=False)
                elapsed += _DT
        finally:
            unregister_entity("npc2")
            unregister_entity("lead")
        # At the vertex (10 m) from 2 s, with the lead 25 m and closing the
        # gap at 2 m/s: departed once it is past 40 m, after 2.5 s.
        released = gate.released_at
        assert gate.asked[0] == pytest.approx(2.0, abs=_DT + 1e-9)
        assert released == pytest.approx(2.5, abs=2 * _DT)
        assert npc.x == pytest.approx(10.0 + 5.0 * (6.0 - released), abs=1e-6)

    def test_a_composed_condition(self, entity: _Entity) -> None:
        both = AndCondition(
            [
                ElapsedTimeCondition(1.0, label="settled"),
                ElapsedTimeCondition(2.0, label="late_enough"),
            ]
        )
        action = FollowTrajectoryAction(
            "npc1", _line(duration=None).gated({1: both}), speed=10.0
        )
        _run(action, entity.actor, 1.9)
        assert entity.actor.x == pytest.approx(5.0)
        _run(action, entity.actor, 0.6, start=1.9)
        assert entity.actor.x > 5.0


# ---------------------------------------------------------------------------
# FOLLOW mode
# ---------------------------------------------------------------------------


def _square() -> Trajectory:
    return Trajectory(
        "square",
        [
            TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0)),
            TrajectoryVertex(CarlaWorldPose(10.0, 0.0, 0.0)),
            TrajectoryVertex(CarlaWorldPose(10.0, 10.0, 0.0)),
            TrajectoryVertex(CarlaWorldPose(0.0, 10.0, 0.0)),
        ],
        closed=True,
    )


def _drive(
    trajectory: Trajectory,
    *,
    seconds: float,
    timed: bool = False,
    speed: Optional[float] = None,
    start_speed: float = 0.0,
    stop_when_finished: bool = True,
) -> tuple[FollowTrajectoryAction, _Actor, float, list[float]]:
    """Drive a fresh vehicle (FOLLOW); the action, the actor, the clock, its speeds."""
    actor = _Actor(speed=start_speed)
    register_entity("npc9", _Entity(actor))
    speeds: list[float] = []
    try:
        action = FollowTrajectoryAction(
            "npc9",
            trajectory,
            TrajectoryTiming() if timed else None,
            TrajectoryFollowingMode.FOLLOW,
            speed=speed,
        )
        world = _World()
        elapsed = 0.0
        while elapsed < seconds and not (stop_when_finished and action.finished):
            action.tick(world, elapsed)
            actor.step(_DT, driven=True)
            elapsed += _DT
            speeds.append(actor.speed)
    finally:
        unregister_entity("npc9")
    return action, actor, elapsed, speeds


class TestFollow:
    def test_a_vehicle_stops_at_a_waiting_vertex_and_drives_on(
        self, entity: _Entity
    ) -> None:
        actor = entity.actor
        actor.vx = actor.speed = 5.0
        gate = _Recorder(open=False)
        action = FollowTrajectoryAction(
            "npc1",
            _line(length=100.0, duration=None, count=11).gated({3: gate}),
            following_mode=TrajectoryFollowingMode.FOLLOW,
        )
        world = _World()
        elapsed = 0.0
        furthest = 0.0
        for _ in range(int(round(15.0 / _DT))):
            action.tick(world, elapsed)
            actor.step(_DT, driven=True)
            elapsed += _DT
            furthest = max(furthest, actor.x)
        assert action.held_vertex == 3
        assert actor.x == pytest.approx(30.0, abs=1.0)
        assert furthest < 31.0
        assert actor.speed < 0.1
        assert actor.control.brake > 0.0

        gate.open = True
        _run(action, actor, 12.0, start=elapsed, driven=True)
        assert action.held_vertex is None
        assert actor.x > 60.0
        assert actor.speed == pytest.approx(5.0, abs=0.7)

    def test_a_timed_vehicle_waits_then_goes_at_the_speed(self) -> None:
        # 60 m in 10 s (6 m/s); the vertex at 30 m waits for 12 s, by when
        # every later time has passed: on at the action's 6 m/s.
        line = _line(length=60.0, duration=10.0, count=7)
        vertices = list(line.vertices)
        vertices[3] = TrajectoryVertex(
            vertices[3].position, advance=ElapsedTimeCondition(12.0, label="go")
        )
        action, actor, elapsed, _ = _drive(
            Trajectory("line", vertices), seconds=30.0, timed=True, speed=6.0
        )
        assert action.finished
        # Departed at 12 s with 30 m to go at 6 m/s.
        assert 12.0 + 5.0 - 1.0 < elapsed < 12.0 + 5.0 + 3.0

    def test_a_timed_vehicle_waiting_is_on_the_brake(self) -> None:
        line = _line(length=60.0, duration=10.0, count=7)
        vertices = list(line.vertices)
        vertices[3] = TrajectoryVertex(
            vertices[3].position, advance=ElapsedTimeCondition(12.0, label="go")
        )
        action, actor, _, _ = _drive(
            Trajectory("line", vertices), seconds=11.9, timed=True, speed=6.0
        )
        assert action.held_vertex == 3
        assert actor.x == pytest.approx(30.0, abs=1.0)
        assert actor.speed < 0.1

    def test_a_condition_already_holding_costs_the_schedule_nothing(self) -> None:
        line = _line(length=120.0, duration=12.0, count=13)
        _, plain_actor, plain_end, _ = _drive(line, seconds=40.0, timed=True)
        vertices = list(line.vertices)
        gate = _Recorder(AlwaysTrueCondition())
        vertices[4] = TrajectoryVertex(vertices[4].position, advance=gate)
        action, actor, gated_end, _ = _drive(
            Trajectory("line", vertices), seconds=40.0, timed=True, speed=10.0
        )
        assert gate.asked
        assert action.finished
        assert gated_end == pytest.approx(plain_end, abs=0.3)
        assert actor.x == pytest.approx(plain_actor.x, abs=0.5)

    def test_an_untimed_open_vertex_is_driven_through_at_speed(self) -> None:
        line = _line(length=100.0, duration=None, count=11)
        _, _, _, plain = _drive(line, seconds=10.0, start_speed=5.0)
        _, _, _, gated = _drive(
            line.gated({3: AlwaysTrueCondition()}), seconds=10.0, start_speed=5.0
        )
        assert min(gated[60:]) == pytest.approx(min(plain[60:]), abs=0.2)

    def test_a_condition_on_the_last_vertex_is_driven_up_to(self) -> None:
        """Holding already, it ends the run at the end, not from braking distance."""
        path = _path((0.0, None), (25.0, None), (60.0, AlwaysTrueCondition()))
        action, actor, _, _ = _drive(path, seconds=30.0, speed=8.0, start_speed=8.0)
        assert action.finished
        assert actor.x > 58.0

    def test_a_walker_walks_up_to_a_last_vertex_whose_condition_holds(self) -> None:
        walker = _Walker(speed=1.4)
        register_entity("walker1", _Entity(walker))
        try:
            action = FollowTrajectoryAction(
                "walker1",
                _path((0.0, None), (10.0, None), (20.0, AlwaysTrueCondition())),
                following_mode=TrajectoryFollowingMode.FOLLOW,
            )
            _run(action, walker, 30.0)
        finally:
            unregister_entity("walker1")
        assert action.finished
        assert walker.x > 19.0

    def test_a_walker_stops_at_a_waiting_vertex_and_walks_on(self) -> None:
        walker = _Walker(speed=1.4)
        register_entity("walker1", _Entity(walker))
        gate = _Recorder(open=False)
        try:
            action = FollowTrajectoryAction(
                "walker1",
                _line(duration=None).gated({2: gate}),
                following_mode=TrajectoryFollowingMode.FOLLOW,
            )
            _run(action, walker, 15.0)
            assert action.held_vertex == 2
            assert 9.6 <= walker.x <= 10.05
            assert walker.walker_controls[-1].speed == 0.0
            gate.open = True
            _run(action, walker, 15.0, start=15.0)
        finally:
            unregister_entity("walker1")
        assert action.finished
        assert walker.x > 18.5


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Located(_Actor):
    """An actor conditions can locate."""

    def get_location(self) -> carla.Location:
        return carla.Location(x=self.x, y=self.y, z=self.z)


class _ActorsWorld(_World):
    def __init__(self, actors: list[Any]) -> None:
        self._actors = actors

    def get_actors(self) -> list[Any]:
        return self._actors


class _Walker(_Actor):
    """A walker that goes where its ``WalkerControl`` sends it."""

    def __init__(self, speed: float) -> None:
        super().__init__(speed=speed, type_id="walker.pedestrian.0001")

    def apply_control(self, control: Any) -> None:
        super().apply_control(control)
        if isinstance(control, carla.WalkerControl):
            speed = control.speed / _UE5_WALKER_SPEED_SCALE
            self.vx = control.direction.x * speed
            self.vy = control.direction.y * speed
            self.speed = math.hypot(self.vx, self.vy)
