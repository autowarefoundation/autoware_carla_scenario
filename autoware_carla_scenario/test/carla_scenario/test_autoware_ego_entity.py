"""Unit tests for :class:`AutowareEgoEntity`.

The CARLA world/actor are faked with lightweight stand-ins so no live CARLA
server is required.  These tests verify the attach-instead-of-spawn behaviour,
the no-destroy lifecycle, and the readiness wait driven through the
:class:`~autoware_carla_scenario.entity.ego.EgoVehicle` lifecycle hooks
(``on_scenario_start`` / ``on_tick`` / ``on_scenario_end``).
"""

from __future__ import annotations

import math
from typing import List

import carla
import pytest

from autoware_carla_scenario.autoware_bridge import (
    AutowareBridgeConfig,
    BridgePose,
    FakeAutowareBridge,
)
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.entity import AutowareEgoEntity, AutowareEntity

#: Where the fake ego actually stands, and the same pose in the map frame.
_CARLA_SPAWN = carla.Transform(
    carla.Location(x=1.0, y=-2.0, z=0.5), carla.Rotation(yaw=-28.6479)
)


@pytest.fixture(autouse=True)
def _map_frame_without_a_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert CARLA poses to the map frame without loading a map.

    The real :func:`to_map_frame` needs the projector offsets a loaded map
    carries; what these tests are about is which pose the entity picks, so the
    conversion is stood in for by the y-flip it reduces to on a map whose
    offsets are zero.
    """

    def _flip_y(pose):
        return BridgePose.from_yaw(
            x=pose.x, y=-pose.y, z=pose.z, yaw=math.radians(-pose.yaw)
        )

    monkeypatch.setattr(
        "autoware_carla_scenario.entity.autoware_entity.to_map_frame", _flip_y
    )


_INITIAL = BridgePose.from_yaw(x=1.0, y=2.0, z=0.5, yaw=0.5)
_GOAL = BridgePose.from_yaw(x=10.0, y=20.0, z=3.0, yaw=1.5)


# ---------------------------------------------------------------------------
# Fake CARLA world / actor
# ---------------------------------------------------------------------------


class _FakeActor:
    def __init__(
        self, actor_id: int, role_name: str, transform: "carla.Transform | None" = None
    ) -> None:
        self.id = actor_id
        self.attributes = {"role_name": role_name}
        self.destroyed = False
        self._transform = transform if transform is not None else _CARLA_SPAWN

    def get_transform(self) -> "carla.Transform":
        return self._transform

    def destroy(self) -> None:
        self.destroyed = True


class _FakeActorList:
    def __init__(self, actors: List[_FakeActor]) -> None:
        self._actors = actors

    def __iter__(self):
        return iter(self._actors)


class _FakeWorld:
    def __init__(self, actors: List[_FakeActor]) -> None:
        self._actors = actors

    def get_actors(self) -> _FakeActorList:
        return _FakeActorList(self._actors)


def _make_entity(bridge=None, **config_kwargs) -> AutowareEgoEntity:
    bridge = bridge if bridge is not None else FakeAutowareBridge()
    config = AutowareBridgeConfig(**config_kwargs) if config_kwargs else None
    return AutowareEgoEntity(
        config, bridge=bridge, initial_pose=_INITIAL, goal_pose=_GOAL
    )


def _assert_pose_close(actual, expected) -> None:
    """Compare two poses component-wise; the derived one is exact only to float."""
    assert actual is not None
    for got, want in (
        (actual.position.x, expected.position.x),
        (actual.position.y, expected.position.y),
        (actual.position.z, expected.position.z),
        (actual.rotation.w, expected.rotation.w),
        (actual.rotation.x, expected.rotation.x),
        (actual.rotation.y, expected.rotation.y),
        (actual.rotation.z, expected.rotation.z),
    ):
        assert got == pytest.approx(want, abs=1e-6)


def _drive_to_ready(entity: AutowareEgoEntity, world: _FakeWorld, max_ticks=50) -> None:
    entity.on_scenario_start(world)
    for _ in range(max_ticks):
        entity.on_tick(world, 0.0)
        if entity.is_initialized or entity.termination_requested:
            break


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_use_autopilot_is_false() -> None:
    # The runner relies on this to skip TrafficManager autopilot on the ego.
    assert AutowareEgoEntity.use_autopilot is False


def test_constructor_requires_bridge() -> None:
    # bridge is a required keyword argument until the live gRPC transport lands.
    with pytest.raises(TypeError):
        AutowareEgoEntity(initial_pose=_INITIAL, goal_pose=_GOAL)  # type: ignore[call-arg]


def test_autoware_entity_placeholder_still_exists() -> None:
    # The bare placeholder is kept for backwards compatibility (used by run.py).
    assert AutowareEntity is not AutowareEgoEntity
    assert AutowareEntity.use_autopilot is False


# ---------------------------------------------------------------------------
# Attach behaviour
# ---------------------------------------------------------------------------


def test_spawn_attaches_to_existing_ego_actor() -> None:
    ego_actor = _FakeActor(42, str(EGO_ROLE_NAME))
    other = _FakeActor(1, "npc1")
    world = _FakeWorld([other, ego_actor])
    entity = _make_entity()

    attached = entity.spawn(world, config=None)  # type: ignore[arg-type]

    assert attached is ego_actor
    assert entity.actor is ego_actor


def test_spawn_times_out_when_ego_absent() -> None:
    world = _FakeWorld([_FakeActor(1, "npc1")])
    entity = _make_entity(attach_timeout=0.0)

    with pytest.raises(RuntimeError, match="No ego actor"):
        entity.spawn(world, config=None)  # type: ignore[arg-type]


def test_destroy_does_not_destroy_actor() -> None:
    ego_actor = _FakeActor(42, str(EGO_ROLE_NAME))
    world = _FakeWorld([ego_actor])
    entity = _make_entity()
    entity.spawn(world, config=None)  # type: ignore[arg-type]

    entity.destroy()

    # The interface node owns the actor lifecycle; we must not destroy it.
    assert ego_actor.destroyed is False
    assert entity.actor is None


# ---------------------------------------------------------------------------
# Readiness wait via lifecycle hooks
# ---------------------------------------------------------------------------


def test_lifecycle_configures_and_reaches_ready() -> None:
    bridge = FakeAutowareBridge(ready_after=2)
    ego_actor = _FakeActor(42, str(EGO_ROLE_NAME))
    world = _FakeWorld([ego_actor])
    entity = _make_entity(bridge=bridge)
    entity.spawn(world, config=None)  # type: ignore[arg-type]

    assert not entity.is_initialized
    _drive_to_ready(entity, world)

    assert entity.is_initialized
    assert not entity.termination_requested
    # Autoware was handed the mission once, then readiness was polled.
    _assert_pose_close(bridge.configured_initial_pose, _INITIAL)
    assert bridge.configured_goal == _GOAL
    assert bridge.calls.count("configure") == 1


def test_on_scenario_start_before_spawn_raises() -> None:
    entity = _make_entity()
    world = _FakeWorld([])

    with pytest.raises(RuntimeError, match="before spawn"):
        entity.on_scenario_start(world)


def test_on_scenario_start_requires_poses() -> None:
    ego_actor = _FakeActor(42, str(EGO_ROLE_NAME))
    world = _FakeWorld([ego_actor])
    entity = AutowareEgoEntity(bridge=FakeAutowareBridge())  # no poses
    entity.spawn(world, config=None)  # type: ignore[arg-type]

    # The message has to name the way out: the poses come from the scenario's
    # setup(), which is the only place they exist.
    with pytest.raises(ValueError, match="set_mission"):
        entity.on_scenario_start(world)


def test_on_tick_before_start_is_noop() -> None:
    entity = _make_entity()
    world = _FakeWorld([])
    # Not configured yet -> must not raise or poll.
    entity.on_tick(world, 0.0)
    assert not entity.is_initialized


def test_not_ready_in_time_requests_termination() -> None:
    # Autoware never reports ready within the tick budget.
    bridge = FakeAutowareBridge(ready_after=10_000)
    ego_actor = _FakeActor(42, str(EGO_ROLE_NAME))
    world = _FakeWorld([ego_actor])
    entity = _make_entity(bridge=bridge, ready_timeout_ticks=3)
    entity.spawn(world, config=None)  # type: ignore[arg-type]

    entity.on_scenario_start(world)
    for _ in range(50):
        entity.on_tick(world, 0.0)
        if entity.termination_requested:
            break

    assert entity.termination_requested
    assert not entity.is_initialized


def test_on_scenario_end_closes_bridge() -> None:
    bridge = FakeAutowareBridge()
    world = _FakeWorld([_FakeActor(42, str(EGO_ROLE_NAME))])
    entity = _make_entity(bridge=bridge)

    entity.on_scenario_end(world)

    assert bridge.closed is True


class TestInitialPoseIsCheckedAgainstTheEgo:
    """Where the ego is is a different question from where it should be."""

    def test_the_attached_actor_supplies_a_pose_no_one_set(self) -> None:
        # A scenario knows its goal in setup(); the ego it will attach to does
        # not exist yet, and its pose is chosen by whoever spawns it.
        bridge = FakeAutowareBridge()
        entity = AutowareEgoEntity(bridge=bridge, goal_pose=_GOAL)
        world = _FakeWorld([_FakeActor(1, str(EGO_ROLE_NAME))])
        entity.spawn(world, config=None)  # type: ignore[arg-type]

        entity.on_scenario_start(world)

        _assert_pose_close(bridge.configured_initial_pose, _INITIAL)
        assert bridge.configured_goal is _GOAL

    def test_the_scenario_pose_wins_when_it_has_one(self) -> None:
        # It is a spawn snapped onto the road surface -- what base_link means --
        # while the actor's transform is measured from the actor's own origin.
        bridge = FakeAutowareBridge()
        entity = _make_entity(bridge=bridge)
        world = _FakeWorld([_FakeActor(1, str(EGO_ROLE_NAME))])
        entity.spawn(world, config=None)  # type: ignore[arg-type]

        entity.on_scenario_start(world)

        assert bridge.configured_initial_pose is _INITIAL

    def test_a_matching_expectation_is_quiet(self, caplog) -> None:
        entity = _make_entity()
        world = _FakeWorld([_FakeActor(1, str(EGO_ROLE_NAME))])
        entity.spawn(world, config=None)  # type: ignore[arg-type]

        with caplog.at_level("WARNING"):
            entity.on_scenario_start(world)

        assert "where the scenario expected it" not in caplog.text

    def test_an_ego_somewhere_else_is_reported(self, caplog) -> None:
        # The interface node's spawn_point and the scenario's spawn are set on
        # opposite sides of the run; when they disagree, that is said out loud
        # instead of becoming a localization error nobody can place.
        entity = _make_entity()
        elsewhere = carla.Transform(carla.Location(x=50.0, y=-2.0, z=0.5))
        world = _FakeWorld([_FakeActor(1, str(EGO_ROLE_NAME), elsewhere)])
        entity.spawn(world, config=None)  # type: ignore[arg-type]

        with caplog.at_level("WARNING"):
            entity.on_scenario_start(world)

        assert "49.00 m from where the scenario expected it" in caplog.text

    def test_a_height_difference_alone_is_not_a_disagreement(self, caplog) -> None:
        # The two heights are measured from different places: the scenario's
        # from the road surface, the actor's from its own origin.
        entity = _make_entity()
        higher = carla.Transform(carla.Location(x=1.0, y=-2.0, z=2.0))
        world = _FakeWorld([_FakeActor(1, str(EGO_ROLE_NAME), higher)])
        entity.spawn(world, config=None)  # type: ignore[arg-type]

        with caplog.at_level("WARNING"):
            entity.on_scenario_start(world)

        assert "where the scenario expected it" not in caplog.text
