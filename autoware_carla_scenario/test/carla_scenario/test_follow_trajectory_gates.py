"""Waypoint conditions: a trajectory vertex that is left only once a condition holds.

A vertex's ``advance`` condition holds the entity at that vertex until the
condition is satisfied.  These tests run the action against the simulated
vehicle (and walker) of ``test_follow_trajectory_action``, in every mode the
action has, and check the three promises the feature makes: the entity waits
at the vertex, it goes on afterwards as if the trajectory had been paused
there, and a trajectory with no gate -- or with a gate already open -- moves
exactly as it did before.
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
    TrajectoryTiming,
    TrajectoryVertex,
)
from autoware_carla_scenario.action_state import ActionState
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


class _Once(BaseCondition):
    """Holds on the first check only."""

    def __init__(self) -> None:
        super().__init__(label="once")
        self.checks = 0

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        self.checks += 1
        if self.checks == 1:
            return ScenarioResult(passed=True, message="", elapsed_seconds=elapsed)
        return None


@pytest.fixture
def entity() -> Iterator[_Entity]:
    registered = _Entity(_Actor())
    register_entity("npc1", registered)
    yield registered
    unregister_entity("npc1")


def _gated(trajectory: Trajectory, **advance: BaseCondition) -> Trajectory:
    """*trajectory* with conditions on the vertices named ``v<index>``."""
    return trajectory.gated({int(key[1:]): value for key, value in advance.items()})


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


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class TestTheModel:
    def test_a_vertex_takes_a_condition_and_stays_hashable(self) -> None:
        gate = AlwaysTrueCondition()
        vertex = TrajectoryVertex(MapPose(0.0, 0.0), 1.0, advance=gate)
        assert vertex.advance is gate
        # Hashable whenever its position is: a condition hashes by identity.
        assert hash(vertex) == hash(TrajectoryVertex(MapPose(0.0, 0.0), 1.0, gate))
        assert vertex != TrajectoryVertex(MapPose(0.0, 0.0), 1.0)
        assert TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0)).advance is None

    def test_a_vertex_refuses_something_that_is_not_a_condition(self) -> None:
        with pytest.raises(TypeError, match="advance must be a condition"):
            TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0), advance=3.0)  # type: ignore[arg-type]

    def test_is_gated(self) -> None:
        assert not _line().is_gated
        assert _gated(_line(), v1=AlwaysTrueCondition()).is_gated

    def test_the_last_vertex_of_an_open_path_cannot_be_gated(self) -> None:
        with pytest.raises(ValueError, match="last vertex of an open trajectory"):
            _gated(_line(), v4=AlwaysTrueCondition())
        with pytest.raises(ValueError, match="last vertex of an open trajectory"):
            _line().gated({-1: AlwaysTrueCondition()})

    def test_the_last_vertex_of_a_closed_path_can(self) -> None:
        square = _square()
        assert square.gated({3: AlwaysTrueCondition()}).is_gated

    def test_gated_keeps_the_rest_and_refuses_a_missing_vertex(self) -> None:
        line = _line()
        gate = AlwaysTrueCondition()
        gated = line.gated({2: gate})
        assert gated.vertices[2].advance is gate
        assert gated.vertices[2].time == line.vertices[2].time
        assert gated.vertices[2].position == line.vertices[2].position
        assert [v.advance for v in gated.vertices[:2]] == [None, None]
        with pytest.raises(IndexError, match="no vertex 9"):
            line.gated({9: gate})


# ---------------------------------------------------------------------------
# POSITION mode, timed
# ---------------------------------------------------------------------------


class TestPositionTimed:
    def test_it_waits_at_the_vertex_with_its_clock_paused(
        self, entity: _Entity
    ) -> None:
        gate = _Recorder(ElapsedTimeCondition(3.0, label="go"))
        action = FollowTrajectoryAction(
            "npc1", _gated(_line(), v2=gate), TrajectoryTiming()
        )
        _run(action, entity.actor, 2.9)
        assert entity.actor.x == pytest.approx(10.0)
        assert entity.actor.speed == 0.0
        assert action.held_vertex == 2
        assert action.state is ActionState.RUNNING
        # First asked on the tick it got there -- vertex time 1.0 -- not before.
        assert gate.asked[0] == pytest.approx(1.0, abs=_DT / 2)

        trace = _trace(action, entity.actor, 2.0, start=2.9)
        released = gate.released_at
        # Within a tick of 3 s: the clock is a sum of ticks.
        assert released == pytest.approx(3.0, abs=_DT + 1e-9)
        assert action.held_vertex is None
        # Each segment after the gate keeps its recorded duration: 10 m/s,
        # starting from the vertex on the tick the condition held.
        for elapsed, x, speed in trace:
            if released <= elapsed < released + 1.0 - 1e-6:
                assert x == pytest.approx(10.0 + 10.0 * (elapsed - released))
                assert speed == pytest.approx(10.0)
        ended = next(e for e, _, _ in trace if e >= released + 1.0 - 1e-6)
        assert action.finished
        # The end came the hold later than the recording had it.
        assert ended == pytest.approx(released + 1.0, abs=1e-6)
        assert entity.actor.x == pytest.approx(20.0)

    def test_a_condition_true_on_arrival_does_not_stop_it(
        self, entity: _Entity
    ) -> None:
        plain = _trace(
            FollowTrajectoryAction("npc1", _line(), TrajectoryTiming()),
            entity.actor,
            2.5,
        )
        entity.actor.x = entity.actor.vx = entity.actor.speed = 0.0
        gate = _Recorder(AlwaysTrueCondition())
        gated = _trace(
            FollowTrajectoryAction(
                "npc1",
                _gated(_line(), v1=gate, v2=AlwaysTrueCondition()),
                TrajectoryTiming(),
            ),
            entity.actor,
            2.5,
        )
        assert gated == plain
        assert len(gate.asked) == 1

    def test_once_open_a_gate_stays_open(self, entity: _Entity) -> None:
        once = _Once()
        action = FollowTrajectoryAction(
            "npc1", _gated(_line(), v1=once), TrajectoryTiming()
        )
        _run(action, entity.actor, 2.5)
        assert once.checks == 1
        assert action.finished

    def test_several_gates_in_turn(self, entity: _Entity) -> None:
        first = _Recorder(ElapsedTimeCondition(1.0, label="first"))
        second = _Recorder(ElapsedTimeCondition(3.0, label="second"))
        action = FollowTrajectoryAction(
            "npc1", _gated(_line(), v1=first, v3=second), TrajectoryTiming()
        )
        _run(action, entity.actor, 2.0)
        # Held at 5 m from 0.5 s to 1.0 s, so 15 m is reached at 2.0 s.
        assert first.released_at == pytest.approx(1.0, abs=_DT + 1e-9)
        assert entity.actor.x == pytest.approx(15.0, abs=0.51)
        _run(action, entity.actor, 0.8, start=2.0)
        assert entity.actor.x == pytest.approx(15.0)
        assert action.held_vertex == 3
        _run(action, entity.actor, 1.0, start=2.8)
        assert second.released_at == pytest.approx(3.0, abs=_DT + 1e-9)
        assert action.finished

    def test_absolute_timing_pauses_the_scenario_clock_too(
        self, entity: _Entity
    ) -> None:
        gate = _Recorder(ElapsedTimeCondition(2.0, label="go"))
        action = FollowTrajectoryAction(
            "npc1",
            _gated(_line(), v2=gate),
            TrajectoryTiming(ReferenceContext.ABSOLUTE),
        )
        trace = _trace(action, entity.actor, 2.5)
        released = gate.released_at
        last_elapsed, last_x, _ = trace[-1]
        assert last_x == pytest.approx(10.0 + 10.0 * (last_elapsed - released))

    def test_absolute_timing_started_past_a_gate_waits_at_it(
        self, entity: _Entity
    ) -> None:
        """Begun after the gate's time, it is held there, not let through."""
        action = FollowTrajectoryAction(
            "npc1",
            _gated(_line(), v2=ElapsedTimeCondition(5.0, label="go")),
            TrajectoryTiming(ReferenceContext.ABSOLUTE),
        )
        _run(action, entity.actor, _DT, start=1.5)
        assert entity.actor.x == pytest.approx(10.0)
        assert action.held_vertex == 2

    def test_scale_applies_after_the_hold(self, entity: _Entity) -> None:
        gate = _Recorder(ElapsedTimeCondition(3.0, label="go"))
        action = FollowTrajectoryAction(
            "npc1", _gated(_line(), v2=gate), TrajectoryTiming(scale=2.0)
        )
        trace = _trace(action, entity.actor, 3.5)
        # Reached at scenario time 2.0 (vertex time 1.0, played at half speed).
        assert gate.asked[0] == pytest.approx(2.0, abs=_DT / 2)
        released = gate.released_at
        elapsed, x, speed = trace[-1]
        assert x == pytest.approx(10.0 + 5.0 * (elapsed - released))
        assert speed == pytest.approx(5.0)

    def test_an_initial_offset_past_a_gate_skips_it(self, entity: _Entity) -> None:
        gate = _Recorder(open=False)
        action = FollowTrajectoryAction(
            "npc1",
            _gated(_line(), v2=gate),
            TrajectoryTiming(),
            initial_distance_offset=12.0,
        )
        _run(action, entity.actor, 1.0)
        assert gate.asked == []
        assert action.finished

    def test_an_initial_offset_before_a_gate_reaches_it_sooner(
        self, entity: _Entity
    ) -> None:
        gate = _Recorder(open=False)
        action = FollowTrajectoryAction(
            "npc1",
            _gated(_line(), v2=gate),
            TrajectoryTiming(),
            initial_distance_offset=5.0,
        )
        _run(action, entity.actor, 1.0)
        assert gate.asked[0] == pytest.approx(0.5, abs=_DT / 2)
        assert entity.actor.x == pytest.approx(10.0)


class TestHiddenWithGates:
    def _late(self, gate: BaseCondition) -> Trajectory:
        return Trajectory(
            "late",
            [
                TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0), 1.0),
                TrajectoryVertex(CarlaWorldPose(10.0, 0.0, 0.0), 2.0, gate),
                TrajectoryVertex(CarlaWorldPose(20.0, 0.0, 0.0), 3.0),
            ],
        )

    def test_it_is_in_the_world_while_held_and_leaves_the_hold_later(
        self, entity: _Entity
    ) -> None:
        gate = _Recorder(ElapsedTimeCondition(4.0, label="go"))
        action = FollowTrajectoryAction(
            "npc1",
            self._late(gate),
            TrajectoryTiming(ReferenceContext.ABSOLUTE),
            hidden_outside_trajectory=True,
        )
        _run(action, entity.actor, 0.5)
        assert entity.actor.z == pytest.approx(-HIDDEN_DEPTH_M)
        _run(action, entity.actor, 3.0, start=0.5)
        # 3.5 s: past the recording's end, but held at the gate and visible.
        assert entity.actor.physics is True
        assert entity.actor.z == pytest.approx(0.0)
        assert entity.actor.x == pytest.approx(10.0)
        _run(action, entity.actor, 1.0, start=3.5)
        assert entity.actor.physics is True
        assert not action.finished
        _run(action, entity.actor, 1.0, start=4.5)
        assert action.finished
        assert entity.actor.physics is False

    def test_held_on_the_end_time_it_stays(self, entity: _Entity) -> None:
        """A gate whose time is the last one's holds it in the world, unfinished."""
        stop = Trajectory(
            "stop",
            [
                TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0), 0.0),
                TrajectoryVertex(
                    CarlaWorldPose(10.0, 0.0, 0.0), 1.0, _Recorder(open=False)
                ),
                TrajectoryVertex(CarlaWorldPose(10.0, 0.0, 0.0), 1.0),
            ],
        )
        action = FollowTrajectoryAction(
            "npc1",
            stop,
            TrajectoryTiming(ReferenceContext.ABSOLUTE),
            hidden_outside_trajectory=True,
        )
        _run(action, entity.actor, 3.0)
        assert not action.finished
        assert entity.actor.physics is True
        assert entity.actor.x == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# POSITION mode, untimed
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


class TestPositionUntimed:
    def test_the_distance_stops_at_the_gate_and_goes_on(self, entity: _Entity) -> None:
        entity.actor.vx = entity.actor.speed = 4.0
        gate = _Recorder(open=False)
        action = FollowTrajectoryAction("npc1", _gated(_line(duration=None), v2=gate))
        _run(action, entity.actor, 5.0)
        assert entity.actor.x == pytest.approx(10.0)
        assert entity.actor.speed == 0.0
        assert action.held_vertex == 2
        assert gate.asked[0] == pytest.approx(2.5, abs=_DT + 1e-9)
        gate.open = True
        _run(action, entity.actor, 1.0, start=5.0)
        # From the vertex, at the speed it had: 4 m in a second.
        assert entity.actor.x == pytest.approx(14.0)
        assert entity.actor.speed == pytest.approx(4.0)

    def test_without_gates_it_is_as_before(self, entity: _Entity) -> None:
        entity.actor.vx = entity.actor.speed = 4.0
        plain = _trace(
            FollowTrajectoryAction("npc1", _line(duration=None)), entity.actor, 3.0
        )
        entity.actor.x, entity.actor.vx, entity.actor.speed = 0.0, 4.0, 4.0
        open_gate = _trace(
            FollowTrajectoryAction(
                "npc1", _gated(_line(duration=None), v1=AlwaysTrueCondition())
            ),
            entity.actor,
            3.0,
        )
        assert open_gate == plain

    def test_a_closed_path_checks_its_gates_on_every_lap(self, entity: _Entity) -> None:
        entity.actor.vx = entity.actor.speed = 10.0
        at_start = _Recorder(AlwaysTrueCondition())
        corner = _Recorder(AlwaysTrueCondition())
        action = FollowTrajectoryAction(
            "npc1", _square().gated({0: at_start, 1: corner})
        )
        # A lap is 40 m, 4 s: the start is reached at 0, 4 and 8 s, the
        # corner at 1 and 5 s.
        _run(action, entity.actor, 8.5)
        assert at_start.asked == pytest.approx([0.0, 4.0, 8.0], abs=_DT / 2)
        assert corner.asked == pytest.approx([1.0, 5.0], abs=_DT / 2)
        assert not action.finished

    def test_a_closed_path_waits_on_its_last_vertex(self, entity: _Entity) -> None:
        entity.actor.vx = entity.actor.speed = 10.0
        action = FollowTrajectoryAction(
            "npc1", _square().gated({3: _Recorder(open=False)})
        )
        _run(action, entity.actor, 6.0)
        assert (entity.actor.x, entity.actor.y) == (
            pytest.approx(0.0),
            pytest.approx(10.0),
        )
        assert action.held_vertex == 3


class TestRepeatedRuns:
    def test_every_run_rearms_its_gates(self, entity: _Entity) -> None:
        gate = _Recorder(AlwaysTrueCondition())
        action = FollowTrajectoryAction(
            "npc1", _gated(_line(), v2=gate), TrajectoryTiming(), once=False
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


class TestOtherConditions:
    def test_a_gate_on_the_distance_to_another_entity(self) -> None:
        """Wait at the vertex until the lead is more than 30 m away."""
        lead = _Located(x=35.0, speed=2.0)
        lead.attributes = {"role_name": "lead"}
        npc = _Located(speed=5.0)
        world = _ActorsWorld([npc, lead])
        gate = _Recorder(
            EntityDistanceCondition(
                "npc2", "lead", 30.0, ComparisonRule.GREATER_THAN, label="clear"
            )
        )
        npc.attributes = {"role_name": "npc2"}
        register_entity("npc2", _Entity(npc))
        register_entity("lead", _Entity(lead))
        try:
            action = FollowTrajectoryAction(
                "npc2", _gated(_line(length=60.0, duration=None, count=7), v1=gate)
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
        # gap at 2 m/s: released once it is past 40 m, after 2.5 s.
        released = gate.released_at
        assert gate.asked[0] == pytest.approx(2.0, abs=_DT + 1e-9)
        assert released == pytest.approx(2.5, abs=2 * _DT)
        assert npc.x == pytest.approx(10.0 + 5.0 * (6.0 - released), abs=1e-6)

    def test_a_composed_gate(self, entity: _Entity) -> None:
        """A composition gates a vertex like any condition: both must hold."""
        entity.actor.vx = entity.actor.speed = 10.0
        both = AndCondition(
            [
                ElapsedTimeCondition(1.0, label="settled"),
                ElapsedTimeCondition(2.0, label="late_enough"),
            ]
        )
        action = FollowTrajectoryAction("npc1", _gated(_line(duration=None), v1=both))
        _run(action, entity.actor, 1.9)
        assert entity.actor.x == pytest.approx(5.0)
        _run(action, entity.actor, 0.6, start=1.9)
        assert entity.actor.x > 5.0


# ---------------------------------------------------------------------------
# FOLLOW mode
# ---------------------------------------------------------------------------


class TestFollow:
    def test_a_vehicle_stops_at_a_held_gate_and_drives_on(
        self, entity: _Entity
    ) -> None:
        actor = entity.actor
        actor.vx = actor.speed = 5.0
        gate = _Recorder(open=False)
        action = FollowTrajectoryAction(
            "npc1",
            _gated(_line(length=100.0, duration=None, count=11), v3=gate),
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
        assert gate.asked  # reached, and asked every tick since

        gate.open = True
        _run(action, actor, 12.0, start=elapsed, driven=True)
        assert action.held_vertex is None
        assert actor.x > 60.0
        assert actor.speed == pytest.approx(5.0, abs=0.7)

    def test_a_timed_vehicle_waits_and_its_schedule_shifts(
        self, entity: _Entity
    ) -> None:
        actor = entity.actor
        # 60 m in 10 s, 6 m/s; the gate is the vertex at 30 m, 5 s.
        line = _line(length=60.0, duration=10.0, count=7)
        gate = _Recorder(ElapsedTimeCondition(12.0, label="go"))
        action = FollowTrajectoryAction(
            "npc1",
            _gated(line, v3=gate),
            TrajectoryTiming(),
            TrajectoryFollowingMode.FOLLOW,
        )
        world = _World()
        elapsed = 0.0
        while elapsed < 11.9:
            action.tick(world, elapsed)
            actor.step(_DT, driven=True)
            elapsed += _DT
        assert action.held_vertex == 3
        assert actor.x == pytest.approx(30.0, abs=1.0)
        assert actor.speed < 0.1
        while elapsed < 30.0 and not action.finished:
            action.tick(world, elapsed)
            actor.step(_DT, driven=True)
            elapsed += _DT
        assert action.finished
        # Released at 12 s with 5 s of recording left.
        assert 12.0 + 5.0 - 1.0 < elapsed < 12.0 + 5.0 + 3.0

    def test_an_open_gate_lets_a_vehicle_through(self, entity: _Entity) -> None:
        actor = entity.actor
        actor.vx = actor.speed = 5.0
        action = FollowTrajectoryAction(
            "npc1",
            _gated(
                _line(length=100.0, duration=None, count=11), v3=AlwaysTrueCondition()
            ),
            following_mode=TrajectoryFollowingMode.FOLLOW,
        )
        _run(action, actor, 15.0, driven=True)
        assert actor.x > 50.0

    def test_a_walker_stops_at_a_held_gate_and_walks_on(self) -> None:
        walker = _Walker(speed=1.4)
        holder = _Entity(walker)
        register_entity("walker1", holder)
        gate = _Recorder(open=False)
        try:
            action = FollowTrajectoryAction(
                "walker1",
                _gated(_line(duration=None), v2=gate),
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
