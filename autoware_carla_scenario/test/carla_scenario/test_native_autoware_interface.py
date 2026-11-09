"""The Autoware ego spawned for CARLA 1.0's native Autoware interface (#2038).

CARLA serves Autoware natively only for a vehicle spawned with
``ros2_autoware_control=true`` and the role name ``ego`` or ``hero``.  The
framework spawns its ego that way on request, and still finds it by
``EGO_ROLE_NAME`` (``"Ego"``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from autoware_carla_scenario.autoware_bridge import FakeAutowareBridge
from autoware_carla_scenario.conditions.base import find_actor_by_role_name
from autoware_carla_scenario.conditions.entity_filter import (
    _role_name as filter_role_name,
)
from autoware_carla_scenario.constants import EGO_ROLE_NAME, EGO_ROLES
from autoware_carla_scenario.entity import AutowareEgoEntity
from autoware_carla_scenario.entity._spawn import (
    ROS2_AUTOWARE_CONTROL_ATTRIBUTE,
    SpawnTransform,
    spawn_vehicle_actor,
)
from autoware_carla_scenario.entity.ego import EgoConfig, EgoVehicle
from autoware_carla_scenario.entity_role import (
    EGO_ROLE_ALIASES,
    NATIVE_AUTOWARE_EGO_ROLE_NAME,
    EntityRole,
    _role_name,
    canonical_role_name,
    find_actor_in_list,
    find_actor_in_world,
)

_BLUEPRINT = "vehicle.mini.cooper"

_CONF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
)


# ---------------------------------------------------------------------------
# Role names
# ---------------------------------------------------------------------------


class TestCanonicalRoleName:
    @pytest.mark.parametrize("role", ["Ego", "ego", "EGO", "hero", "Hero", "HERO"])
    def test_every_name_of_the_ego_reads_as_ego(self, role: str) -> None:
        assert canonical_role_name(role) == "Ego"

    @pytest.mark.parametrize("role", ["npc1", "vehicle2", "egos", "superhero", ""])
    def test_any_other_name_is_kept(self, role: str) -> None:
        assert canonical_role_name(role) == role

    def test_the_aliases_are_the_ones_the_trajectory_log_reads(self) -> None:
        assert EGO_ROLES == EGO_ROLE_ALIASES == ("ego", "hero")
        assert NATIVE_AUTOWARE_EGO_ROLE_NAME in EGO_ROLE_ALIASES

    @pytest.mark.parametrize("role", ["ego", "hero"])
    def test_no_entity_role_is_an_alias(self, role: str) -> None:
        # An NPC can never be mistaken for the ego.
        with pytest.raises(ValueError):
            EntityRole(role)


class _Actor:
    def __init__(self, actor_id: int, role_name: Optional[str]) -> None:
        self.id = actor_id
        self.attributes: Dict[str, str] = (
            {} if role_name is None else {"role_name": role_name}
        )
        self.is_alive = True
        self.type_id = "vehicle.mini.cooper"


class _World:
    def __init__(self, actors: List[_Actor], world_id: int = 2038) -> None:
        self.id = world_id
        self._actors = actors

    def get_actors(self) -> List[_Actor]:
        return list(self._actors)

    def get_actor(self, actor_id: int) -> Optional[_Actor]:
        return next((a for a in self._actors if a.id == actor_id), None)


class TestTheEgoIsFoundUnderItsNativeName:
    @pytest.mark.parametrize("role", ["ego", "hero", "Ego"])
    def test_in_a_list(self, role: str) -> None:
        npc = _Actor(1, "npc1")
        ego = _Actor(2, role)
        assert find_actor_in_list([npc, ego], EGO_ROLE_NAME) is ego  # type: ignore[arg-type]
        assert find_actor_in_list([npc, ego], "npc1") is npc  # type: ignore[arg-type]

    def test_in_a_world(self) -> None:
        ego = _Actor(7, NATIVE_AUTOWARE_EGO_ROLE_NAME)
        world = _World([_Actor(1, "npc1"), ego], world_id=203801)
        assert find_actor_in_world(world, EGO_ROLE_NAME) is ego  # type: ignore[arg-type]
        # Remembered by id, and still the ego the second time.
        assert find_actor_in_world(world, EGO_ROLE_NAME) is ego  # type: ignore[arg-type]
        assert find_actor_by_role_name(world, EGO_ROLE_NAME) is ego  # type: ignore[arg-type]

    def test_a_lookup_by_an_alias_finds_the_ego_spawned_as_ego(self) -> None:
        ego = _Actor(3, "Ego")
        assert find_actor_in_list([ego], "hero") is ego  # type: ignore[arg-type]

    def test_the_actor_s_role_name_reads_as_ego(self) -> None:
        assert _role_name(_Actor(1, "ego")) == "Ego"  # type: ignore[arg-type]
        assert _role_name(_Actor(1, "npc1")) == "npc1"  # type: ignore[arg-type]
        assert _role_name(_Actor(1, None)) is None  # type: ignore[arg-type]
        assert filter_role_name(_Actor(1, "hero")) == "Ego"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------


class _Blueprint:
    def __init__(self, native: bool) -> None:
        self.id = _BLUEPRINT
        self.attributes: Dict[str, str] = {"role_name": ""}
        if native:
            self.attributes[ROS2_AUTOWARE_CONTROL_ATTRIBUTE] = "false"

    def has_attribute(self, name: str) -> bool:
        return name in self.attributes

    def set_attribute(self, name: str, value: str) -> None:
        if name not in self.attributes:
            # CARLA raises for an attribute the blueprint does not have.
            raise IndexError(name)
        self.attributes[name] = value


class _SpawnWorld:
    def __init__(self, native: bool) -> None:
        self.blueprint = _Blueprint(native)
        self._library = SimpleNamespace(
            filter=lambda _pattern: [self.blueprint],
            find=lambda _id: self.blueprint,
        )

    def get_blueprint_library(self) -> Any:
        return self._library

    def try_spawn_actor(self, blueprint: _Blueprint, transform: Any) -> Any:
        return SimpleNamespace(attributes=dict(blueprint.attributes))


def _transform() -> Any:
    location = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    return SimpleNamespace(location=location, rotation=SimpleNamespace(yaw=0.0))


class TestSpawnVehicleActor:
    def test_a_native_server_registers_the_vehicle(self) -> None:
        world = _SpawnWorld(native=True)
        actor = spawn_vehicle_actor(
            world,  # type: ignore[arg-type]
            _BLUEPRINT,
            "Ego",
            SpawnTransform(_transform()),
            ros2_autoware_role_name="ego",
        )
        assert actor.attributes[ROS2_AUTOWARE_CONTROL_ATTRIBUTE] == "true"
        assert actor.attributes["role_name"] == "ego"

    def test_an_older_server_spawns_as_before_and_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        world = _SpawnWorld(native=False)
        with caplog.at_level(logging.WARNING):
            actor = spawn_vehicle_actor(
                world,  # type: ignore[arg-type]
                _BLUEPRINT,
                "Ego",
                SpawnTransform(_transform()),
                ros2_autoware_role_name="ego",
            )
        assert actor.attributes == {"role_name": "Ego"}
        assert ROS2_AUTOWARE_CONTROL_ATTRIBUTE in caplog.text

    def test_without_it_the_attribute_is_left_alone(self) -> None:
        world = _SpawnWorld(native=True)
        actor = spawn_vehicle_actor(
            world,  # type: ignore[arg-type]
            _BLUEPRINT,
            "Ego",
            SpawnTransform(_transform()),
        )
        assert actor.attributes[ROS2_AUTOWARE_CONTROL_ATTRIBUTE] == "false"
        assert actor.attributes["role_name"] == "Ego"


def _spawn_config() -> EgoConfig:
    return EgoConfig(spawn_location=SpawnTransform(_transform()))


def _spawned_kwargs(
    entity: EgoVehicle, monkeypatch: pytest.MonkeyPatch
) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}

    def _fake_spawn(world, vehicle_type, role_name, spawn_location, **kwargs):
        seen.update(kwargs, role_name=role_name)
        return SimpleNamespace()

    monkeypatch.setattr(
        "autoware_carla_scenario.entity.ego.spawn_vehicle_actor", _fake_spawn
    )
    entity.spawn(_SpawnWorld(native=True), _spawn_config())  # type: ignore[arg-type]
    return seen


class TestAutowareEgoEntity:
    def test_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entity = AutowareEgoEntity(bridge=FakeAutowareBridge())
        assert entity.native_autoware_interface is False
        seen = _spawned_kwargs(entity, monkeypatch)
        assert seen["role_name"] == str(EGO_ROLE_NAME)
        assert seen["ros2_autoware_role_name"] is None

    def test_on_request_the_ego_is_spawned_for_the_native_interface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entity = AutowareEgoEntity(
            bridge=FakeAutowareBridge(), native_autoware_interface=True
        )
        assert entity.native_autoware_interface is True
        seen = _spawned_kwargs(entity, monkeypatch)
        assert seen["ros2_autoware_role_name"] == NATIVE_AUTOWARE_EGO_ROLE_NAME

    def test_other_egos_never_are(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spawned_kwargs(EgoVehicle(), monkeypatch)
        assert seen["ros2_autoware_role_name"] is None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _composed(overrides: List[str]) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(_CONF), version_base=None):
        return compose(config_name="config", overrides=overrides)


class TestTheSetting:
    @pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
    def test_ego_native_autoware_interface_is_applied(
        self, value: str, expected: bool
    ) -> None:
        from autoware_carla_scenario.examples.run import build_ego_entity

        entity = build_ego_entity(
            _composed(
                [
                    "ego.entity=autoware",
                    f"ego.native_autoware_interface={value}",
                    "autoware.launcher.type=none",
                ]
            )
        )
        assert isinstance(entity, AutowareEgoEntity)
        assert entity.native_autoware_interface is expected

    def test_it_is_off_by_default(self) -> None:
        from autoware_carla_scenario.examples.run import build_ego_entity

        entity = build_ego_entity(
            _composed(["ego.entity=autoware", "autoware.launcher.type=none"])
        )
        assert isinstance(entity, AutowareEgoEntity)
        assert entity.native_autoware_interface is False

    def test_a_value_that_is_not_a_boolean_is_refused(self) -> None:
        from autoware_carla_scenario.examples.run import build_ego_entity

        with pytest.raises(ValueError, match="ego.native_autoware_interface"):
            build_ego_entity(
                _composed(
                    [
                        "ego.entity=autoware",
                        "ego.native_autoware_interface=yes",
                        "autoware.launcher.type=none",
                    ]
                )
            )
