"""Hidden entities wait on spots of their own and are brought in where they
were declared (#2014).

Two hidden entities declared at one spawn used to be spawned on one spot
500 m under it, and CARLA refused the second, aborting the setup; parked ones
waited 10 m apart under the ego's spawn, closer than two buses are long.  A
Spawn card with no place lifted the entity 500 m from where it waited: for a
parked one, onto a spot along world x from the ego's spawn, unchecked.  Now
each hidden entity waits deeper than any other it would overlap, and a Spawn
card with no place brings it in at its declared (snapped) spawn through the
clearance check -- or, with no spawn of its own, is rejected by the validator
and refused at run time.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from typing import Any, Optional
from unittest import mock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import declarative
from autoware_carla_scenario.actions import _trajectory_run
from autoware_carla_scenario.actions import spawn as spawn_module
from autoware_carla_scenario.actions._trajectory_run import HIDDEN_DEPTH_M
from autoware_carla_scenario.actions.spawn import SpawnAction
from autoware_carla_scenario.authoring.compiler import compile_document
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    Assertions,
    ConditionNode,
    Entity,
    EntitySize,
    ScenarioDocument,
    SpawnSpec,
    SValue,
)
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.trajectory.model import RelativeLanePose
from autoware_carla_scenario.entity import vehicle_entity

from .test_entity_field_parity import _fake_world

#: The height (m) of the tallest CARLA road user, about: two waiting spots
#: this far apart vertically cannot overlap.
_TALLEST_M = 4.0


def _document(
    *npcs: Entity, spawns: tuple[ActionNode, ...] = (), route: Any = None
) -> ScenarioDocument:
    raw: dict[str, Any] = {}
    if route is not None:
        raw["route"] = route
    return ScenarioDocument(
        id="s",
        entities=[Entity(id="ego", kind="ego", spawn=SpawnSpec(lanelet_id=1)), *npcs],
        actions=list(spawns),
        assertions=Assertions.model_validate(
            {
                "pass": [
                    ConditionNode(type="elapsed_time", params={"duration_seconds": 5.0})
                ]
            }
        ),
        **raw,
    )


def _hidden(
    entity_id: str, lanelet_id: int = 2, heading: float = 0.0, **fields: Any
) -> Entity:
    return Entity(
        id=entity_id,
        spawn=SpawnSpec(
            lanelet_id=lanelet_id, s=SValue(value=5.0), heading=heading, hidden=True
        ),
        **fields,
    )


def _spawn_card(actor: str, **params: Any) -> ActionNode:
    return ActionNode(id=f"{actor}_enters", type="spawn", actor=actor, params=params)


@pytest.fixture
def maps() -> Iterator[None]:
    """Every map conversion and snap, made visible (``_fake_world``)."""
    with (
        mock.patch(
            "autoware_carla_scenario.coordinate.transform.to_carla_world",
            side_effect=_fake_world,
        ),
        mock.patch(
            "autoware_carla_scenario.coordinate.transform.to_opendrive",
            side_effect=repr,
        ),
        mock.patch.object(
            declarative,
            "snap_to_carla_road",
            side_effect=lambda pose, world, **_: _fake_world(pose),
        ),
    ):
        yield


def _scenario(
    document: ScenarioDocument, *, unplaced: tuple[str, ...] = ()
) -> declarative.DeclarativeScenario:
    """A DeclarativeScenario of *document* that reaches no CARLA; the
    entities *unplaced* on lanelet 0, as a sweep or a logical scenario's
    route leaves them (a fixed one on lanelet 0 does not compile)."""
    scenario = object.__new__(declarative.DeclarativeScenario)
    scenario._document = document
    scenario._compiled = compile_document(document)
    for entity in scenario._compiled.npcs:
        if entity.id in unplaced:
            entity.spawn.lanelet_id = 0
    scenario._ground_projection = None  # type: ignore[assignment]
    scenario._spawn_pose = Lanelet2Pose(1, 0.0)
    scenario._waiting = []
    scenario._declared = {}
    scenario.ego_config = mock.Mock(  # type: ignore[assignment]
        spawn_retry_max_count=0, spawn_retry_t_step=0.0, spawn_retry_z_step=0.0
    )
    return scenario


def _built(scenario: declarative.DeclarativeScenario) -> dict[str, Any]:
    """The entity built for every NPC of *scenario*, by id (``maps`` on)."""
    built: dict[str, Any] = {}
    for entity in scenario._compiled.npcs:
        if entity.kind == "pedestrian":
            built[entity.id] = scenario._build_pedestrian(entity, object())
        else:
            built[entity.id] = scenario._build_npc(entity, object())
    return built


def _location(entity: Any) -> Any:
    return entity._config.spawn_location.value.location


def _apart(a: Any, b: Any, half_length: float) -> bool:
    """Whether two vehicles *half_length* long at CARLA *a* and *b* (any
    yaw) cannot overlap."""
    flat = math.hypot(a.x - b.x, a.y - b.y)
    return flat >= 2.0 * half_length or abs(a.z - b.z) >= _TALLEST_M


# ---------------------------------------------------------------------------
# Where hidden entities wait
# ---------------------------------------------------------------------------


class TestWaitingSpots:
    def test_two_at_one_spot_wait_one_under_the_other(self) -> None:
        at = carla.Transform(
            carla.Location(x=1.0, y=2.0, z=3.0), carla.Rotation(yaw=90.0)
        )
        waiting: declarative.WaitingSpots = []
        first = declarative._hidden_if_asked(_hidden("a"), at, waiting)
        second = declarative._hidden_if_asked(_hidden("b"), at, waiting)
        assert first.location.z == pytest.approx(3.0 - HIDDEN_DEPTH_M)
        assert second.location.z == pytest.approx(
            3.0 - HIDDEN_DEPTH_M - declarative.HIDDEN_STACK_M
        )
        # Under the same spot, facing the same way.
        assert (second.location.x, second.location.y) == (1.0, 2.0)
        assert second.rotation.yaw == pytest.approx(90.0)

    def test_ones_far_apart_wait_at_one_depth(self) -> None:
        waiting: declarative.WaitingSpots = []
        spots = [
            declarative._hidden_if_asked(
                _hidden(f"e{i}"),
                carla.Transform(carla.Location(x=100.0 * i, y=0.0, z=0.0)),
                waiting,
            )
            for i in range(3)
        ]
        assert {s.location.z for s in spots} == {-HIDDEN_DEPTH_M}

    def test_a_third_takes_the_free_depth_between(self) -> None:
        # One far away is not in the way of one stacked under another.
        waiting: declarative.WaitingSpots = [
            (0.0, 0.0, -HIDDEN_DEPTH_M, 7.0),
            (0.0, 0.0, -HIDDEN_DEPTH_M - 2 * declarative.HIDDEN_STACK_M, 7.0),
            (500.0, 0.0, -HIDDEN_DEPTH_M - declarative.HIDDEN_STACK_M, 7.0),
        ]
        spot = declarative._hidden_if_asked(
            _hidden("c"), carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)), waiting
        )
        assert spot.location.z == pytest.approx(
            -HIDDEN_DEPTH_M - declarative.HIDDEN_STACK_M
        )

    def test_a_visible_one_is_where_it_was_declared(self) -> None:
        at = carla.Transform(carla.Location(x=1.0, y=2.0, z=3.0))
        waiting: declarative.WaitingSpots = []
        shown = Entity(id="v", spawn=SpawnSpec(lanelet_id=2))
        assert declarative._hidden_if_asked(shown, at, waiting) is at
        assert waiting == []

    @pytest.mark.parametrize("sized", [True, False])
    def test_long_vehicles_parked_under_the_ego_do_not_overlap(
        self, maps: None, sized: bool
    ) -> None:
        # Sized (vehicle_type: auto), or a bus blueprint named outright.
        fields: dict[str, Any] = (
            {"vehicle_type": "auto", "size": EntitySize(length=10.3, width=2.5)}
            if sized
            else {"vehicle_type": "vehicle.mitsubishi.fusorosa"}
        )
        buses = [_hidden(f"bus{i}", **fields) for i in range(3)]
        scenario = _scenario(_document(*buses), unplaced=tuple(b.id for b in buses))
        with mock.patch.object(
            declarative, "_vehicle_type", return_value="vehicle.bus"
        ):
            built = _built(scenario)
        spots = [_location(built[b.id]) for b in buses]
        for i, a in enumerate(spots):
            assert a.z < -100.0  # out of the scene
            for b in spots[i + 1 :]:
                assert _apart(a, b, 10.3 / 2.0)

    def test_an_unsized_vehicle_is_taken_as_long_as_any(self) -> None:
        # Its blueprint carries no bounding box: longer than a bus.
        radius = declarative._waiting_radius(_hidden("v"))
        assert 2.0 * radius > 10.3
        walker = declarative._waiting_radius(_hidden("w", kind="pedestrian"))
        assert walker < radius


# ---------------------------------------------------------------------------
# Two hidden NPCs at one spawn are both set up
# ---------------------------------------------------------------------------


class _Blueprint:
    def __init__(self, blueprint_id: str) -> None:
        self.id = blueprint_id

    def set_attribute(self, key: str, value: str) -> None:
        pass


class _Library:
    def filter(self, pattern: str) -> list[_Blueprint]:
        return [_Blueprint("vehicle.mini.cooper")]

    def find(self, blueprint_id: str) -> _Blueprint:
        return _Blueprint(blueprint_id)


class _CarlaWorld:
    """Refuses a vehicle spawned overlapping one there already, as CARLA's
    ``try_spawn_actor`` does."""

    def __init__(self) -> None:
        self.placed: list[Any] = []

    def get_blueprint_library(self) -> _Library:
        return _Library()

    def try_spawn_actor(self, blueprint: Any, transform: Any) -> Optional[Any]:
        here = transform.location
        if any(not _apart(here, there, 2.5) for there in self.placed):
            return None
        self.placed.append(here)
        return mock.Mock(id=len(self.placed))


class TestSameSpawn:
    def test_two_hidden_npcs_at_one_spawn_are_both_spawned(self, maps: None) -> None:
        scenario = _scenario(_document(_hidden("a"), _hidden("b")))
        built = _built(scenario)
        world = _CarlaWorld()
        with mock.patch.object(vehicle_entity, "_warmup_done", False):
            for entity in built.values():
                assert entity.spawn(world) is not None  # type: ignore[arg-type]
        assert len(world.placed) == 2
        a, b = (_location(built[e]) for e in ("a", "b"))
        assert (a.x, a.y) == (b.x, b.y)  # under their one spawn
        assert a.z != b.z


# ---------------------------------------------------------------------------
# Brought in with no place: where it was declared
# ---------------------------------------------------------------------------


class _World:
    id = 2014
    frame = 3

    def get_snapshot(self) -> Any:
        return mock.Mock(frame=self.frame)

    def get_actors(self) -> list[Any]:
        return []


@pytest.fixture
def bring_in(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Bringing an entity in, recorded: where it was put (``put``), each
    place looked at (``looked``) and claimed (``claimed``); the actor of a
    role is ``actors[role]``."""
    record: dict[str, Any] = {"put": {}, "looked": [], "claimed": [], "actors": {}}

    def actor(role: str) -> Any:
        found = record["actors"].get(role)
        if found is None:
            found = mock.Mock(id=len(record["actors"]) + 1)
            found.bounding_box.extent = carla.Vector3D(2.0, 1.0, 0.8)
            found.get_transform.return_value = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=-HIDDEN_DEPTH_M)
            )
            found.role = role
            record["actors"][role] = found
        return found

    def put(
        world: Any, found: Any, x: float, y: float, z: float, yaw: float, speed: float
    ) -> float:
        record["put"][found.role] = (x, y, z, yaw)
        return z

    def taken(world: Any, actor_id: int, x: float, y: float, *a: Any, **k: Any) -> int:
        record["looked"].append((x, y))
        return -1

    def hold(world: Any, actor_id: int, x: float, y: float, *a: Any, **k: Any) -> None:
        record["claimed"].append((x, y))

    monkeypatch.setattr(_trajectory_run, "_entity_actor", actor)
    monkeypatch.setattr(_trajectory_run, "_is_walker", lambda found: False)
    monkeypatch.setattr(spawn_module, "_put", put)
    monkeypatch.setattr(spawn_module, "_taken", taken)
    monkeypatch.setattr(spawn_module, "_hold", hold)
    monkeypatch.setattr(spawn_module, "_keep_entry_speed", lambda found, speed: None)
    spawn_module._BROUGHT_IN.clear()
    yield record
    spawn_module._BROUGHT_IN.clear()
    spawn_module._ENTERED.clear()


def _declared(scenario: declarative.DeclarativeScenario, entity_id: str) -> SpawnAction:
    """A Spawn card with no place for *entity_id*, told its declared spawn as
    the scenario's setup tells it."""
    action = SpawnAction(scenario._compiled.role_of(entity_id), label=entity_id)
    scenario._declare_spawn(action)
    return action


class TestBroughtInWhereDeclared:
    def test_each_of_two_at_one_spawn_appears_at_its_declared_pose(
        self, maps: None, bring_in: dict[str, Any]
    ) -> None:
        scenario = _scenario(_document(_hidden("a"), _hidden("b", heading=0.5)))
        _built(scenario)
        for entity_id in ("a", "b"):
            assert _declared(scenario, entity_id).apply(_World())  # type: ignore[arg-type]
        role_a, role_b = (scenario._compiled.role_of(e) for e in ("a", "b"))
        # _fake_world: x is the lanelet, y its s, z its t, yaw its heading.
        assert bring_in["put"][role_a] == pytest.approx((2.0, 5.0, 0.0, 0.0))
        assert bring_in["put"][role_b] == pytest.approx(
            (2.0, 5.0, 0.0, math.degrees(0.5))
        )
        # Through the clearance check, as a place of its own is.
        assert bring_in["looked"] == [(2.0, 5.0), (2.0, 5.0)]
        assert bring_in["claimed"] == [(2.0, 5.0), (2.0, 5.0)]

    def test_a_parked_one_appears_at_its_declared_spawn(
        self, maps: None, bring_in: dict[str, Any]
    ) -> None:
        # Parked (a logical scenario's) though it names a lanelet: waits
        # under the ego's spawn, comes up at its own.
        scenario = _scenario(_document(_hidden("p", lanelet_id=7)))
        with mock.patch.object(
            declarative.DeclarativeScenario, "_parked", return_value=True
        ):
            built = _built(scenario)
        waiting = _location(built["p"])
        assert (waiting.x, waiting.y) == (1.0, 0.0)  # under the ego's spawn
        assert _declared(scenario, "p").apply(_World())  # type: ignore[arg-type]
        put = bring_in["put"][scenario._compiled.role_of("p")]
        assert put == pytest.approx((7.0, 5.0, 0.0, 0.0))

    def test_one_with_no_spawn_of_its_own_is_refused(
        self, maps: None, bring_in: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        scenario = _scenario(_document(_hidden("p")), unplaced=("p",))
        _built(scenario)
        with caplog.at_level(logging.WARNING, logger=spawn_module.__name__):
            assert not _declared(scenario, "p").apply(_World())  # type: ignore[arg-type]
        assert bring_in["put"] == {}
        assert "it has no place of its own, and no spawn was declared" in caplog.text

    def test_one_whose_own_place_fails_is_refused_saying_so(
        self, maps: None, bring_in: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        # Its place is against an entity not in the scene, and it has no
        # spawn of its own to fall back to.
        scenario = _scenario(_document(_hidden("p")), unplaced=("p",))
        _built(scenario)
        action = SpawnAction(
            scenario._compiled.role_of("p"),
            position=RelativeLanePose(ds=5.0, entity_ref="npc77"),
        )
        scenario._declare_spawn(action)
        with caplog.at_level(logging.WARNING, logger=spawn_module.__name__):
            assert not action.apply(_World())  # type: ignore[arg-type]
        assert bring_in["put"] == {}
        assert "its own place could not be resolved, and no spawn" in caplog.text

    def test_one_whose_lanelet_this_map_lacks_is_refused(
        self, bring_in: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        scenario = _scenario(_document(_hidden("p", lanelet_id=7)))
        with (
            mock.patch.object(
                declarative.DeclarativeScenario, "_parked", return_value=True
            ),
            mock.patch(
                "autoware_carla_scenario.coordinate.transform.to_carla_world",
                side_effect=_fake_world,
            ),
            mock.patch.object(
                declarative, "snap_to_carla_road", side_effect=KeyError(7)
            ),
            mock.patch.object(
                declarative, "_vehicle_type", return_value="vehicle.mini.cooper"
            ),
            caplog.at_level(logging.WARNING),
        ):
            _built(scenario)
            assert not _declared(scenario, "p").apply(_World())  # type: ignore[arg-type]
        assert "is not on this map" in caplog.text
        assert "or one this map does not have" in caplog.text
        assert bring_in["put"] == {}

    def test_a_place_of_its_own_is_kept(
        self, maps: None, bring_in: dict[str, Any]
    ) -> None:
        scenario = _scenario(_document(_hidden("a")))
        _built(scenario)
        action = SpawnAction(
            scenario._compiled.role_of("a"),
            position=CarlaWorldPose(40.0, 50.0, 1.0, yaw=10.0),
        )
        scenario._declare_spawn(action)
        assert action.apply(_World())  # type: ignore[arg-type]
        put = bring_in["put"][scenario._compiled.role_of("a")]
        assert put == pytest.approx((40.0, 50.0, 1.0, 10.0))

    def test_with_nothing_declared_it_comes_up_where_it_waits(
        self, bring_in: dict[str, Any]
    ) -> None:
        # A hand-written scenario's, hidden under its own place.
        assert SpawnAction("npc9").apply(_World())  # type: ignore[arg-type]
        assert bring_in["put"]["npc9"] == pytest.approx((0.0, 0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


def _issues(document: ScenarioDocument, severity: str = "error") -> list[str]:
    report = validate_document(document)
    found = report.errors if severity == "error" else report.warnings
    return [i.message for i in found if "where it was declared" in i.message]


class TestValidator:
    def test_no_place_on_one_with_no_spawn_of_its_own_is_rejected(self) -> None:
        document = _document(_hidden("p", lanelet_id=0), spawns=(_spawn_card("p"),))
        assert len(_issues(document)) == 1

    def test_a_place_of_its_own_is_accepted(self) -> None:
        document = _document(
            _hidden("p", lanelet_id=0),
            spawns=(_spawn_card("p", lanelet_id=3, s=1.0),),
        )
        assert _issues(document) == []

    def test_no_place_on_one_with_a_spawn_is_accepted(self) -> None:
        document = _document(_hidden("p", lanelet_id=4), spawns=(_spawn_card("p"),))
        assert _issues(document) == []
        assert _issues(document, "warning") == []
