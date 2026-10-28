"""vehicle_type: auto -- the vehicle of the running server nearest a size."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from autoware_carla_scenario.authoring.models import (
    AUTO_MODEL,
    Entity,
    EntitySize,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.entity import fit

#: id -> (length, width, base_type) of the fake server's vehicles.
_SERVER = {
    "vehicle.small.car": (4.18, 1.99, "car"),
    "vehicle.saloon.car": (4.67, 1.81, "car"),
    "vehicle.box.truck": (8.0, 2.91, "truck"),
    "vehicle.city.bus": (11.5, 2.55, "bus"),
    "vehicle.firetruck.actors": (11.0, 2.5, "truck"),
    "vehicle.road.bike": (1.8, 0.7, "bicycle"),
}


class _Attribute:
    def __init__(self, value: str) -> None:
        self._value = value

    def as_str(self) -> str:
        return self._value


class _Blueprint:
    def __init__(self, blueprint_id: str) -> None:
        self.id = blueprint_id

    def has_attribute(self, name: str) -> bool:
        return name in ("base_type", "role_name")

    def get_attribute(self, name: str) -> _Attribute:
        return _Attribute(_SERVER[self.id][2])

    def set_attribute(self, name: str, value: str) -> None:
        pass


class _Library(list):
    def filter(self, pattern: str) -> "_Library":
        return self


class _World:
    def __init__(self, world_id: int) -> None:
        self.id = world_id
        self.spawned: list[str] = []

    def get_blueprint_library(self) -> _Library:
        return _Library(_Blueprint(b) for b in _SERVER)

    def try_spawn_actor(self, blueprint: _Blueprint, transform: Any) -> Any:
        self.spawned.append(blueprint.id)
        length, width, _kind = _SERVER[blueprint.id]
        extent = SimpleNamespace(x=length / 2.0, y=width / 2.0, z=0.8)
        return SimpleNamespace(
            bounding_box=SimpleNamespace(extent=extent), destroy=lambda: None
        )


@pytest.fixture
def world() -> _World:
    fit._SIZES.clear()
    return _World(7)


class TestFitting:
    @pytest.mark.parametrize(
        ("length", "width", "chosen"),
        [
            (4.5, 1.8, "vehicle.saloon.car"),
            (8.0, 2.4, "vehicle.box.truck"),
            (11.0, 2.5, "vehicle.city.bus"),  # not the fire engine
        ],
    )
    def test_the_nearest_of_this_servers_vehicles(
        self, world: _World, length: float, width: float, chosen: str
    ) -> None:
        assert fit.fitted_vehicle(world, length, width) == chosen

    def test_no_emergency_vehicle_or_bicycle_is_a_candidate(
        self, world: _World
    ) -> None:
        assert set(fit.vehicle_sizes(world)) == {
            "vehicle.small.car",
            "vehicle.saloon.car",
            "vehicle.box.truck",
            "vehicle.city.bus",
        }

    def test_each_box_is_read_once_per_world(self, world: _World) -> None:
        fit.fitted_vehicle(world, 4.5, 1.8)
        fit.fitted_vehicle(world, 8.0, 2.4)
        assert sorted(world.spawned) == sorted(fit.vehicle_sizes(world))

    def test_no_size_no_choice(self, world: _World) -> None:
        assert fit.fitted_vehicle(world, 0.0, 1.8) is None


def _document(entity: dict[str, Any]) -> ScenarioDocument:
    return ScenarioDocument.model_validate(
        {
            "id": "auto_model",
            "map": {"name": "Town01"},
            "entities": [
                {"id": "ego", "kind": "ego", "spawn": {"lanelet_id": 1}},
                entity,
            ],
        }
    )


def _errors(document: ScenarioDocument) -> list[str]:
    return [i.message for i in validate_document(document).errors]


class TestTheDocument:
    def test_auto_with_a_size_is_valid(self) -> None:
        document = _document(
            {
                "id": "car",
                "vehicle_type": AUTO_MODEL,
                "size": {"length": 4.5, "width": 1.8},
                "spawn": {"lanelet_id": 1},
            }
        )
        assert not [m for m in _errors(document) if "vehicle_type" in m or "size" in m]
        assert document.entities[1].size == EntitySize(length=4.5, width=1.8)

    def test_auto_wants_a_size(self) -> None:
        document = _document(
            {"id": "car", "vehicle_type": AUTO_MODEL, "spawn": {"lanelet_id": 1}}
        )
        assert any("has none" in m for m in _errors(document))

    def test_a_pedestrian_names_its_blueprint(self) -> None:
        document = _document(
            {
                "id": "walker",
                "kind": "pedestrian",
                "vehicle_type": AUTO_MODEL,
                "size": {"length": 0.5, "width": 0.5},
                "spawn": {"lanelet_id": 1},
            }
        )
        assert any("Only a vehicle" in m for m in _errors(document))


class TestTheSpawn:
    def test_auto_is_resolved_on_the_world(self) -> None:
        from autoware_carla_scenario.declarative import _vehicle_type

        entity = Entity(
            id="car", vehicle_type=AUTO_MODEL, size=EntitySize(length=4.5, width=1.8)
        )
        with patch.object(
            fit, "fitted_vehicle", return_value="vehicle.saloon.car"
        ) as f:
            assert _vehicle_type(entity, object()) == "vehicle.saloon.car"
        assert f.call_args.args[1:] == (4.5, 1.8)

    def test_none_fitting_falls_back_to_the_kinds_default(self) -> None:
        from autoware_carla_scenario.declarative import _vehicle_type

        entity = Entity(
            id="car", vehicle_type=AUTO_MODEL, size=EntitySize(length=4.5, width=1.8)
        )
        with patch.object(fit, "fitted_vehicle", return_value=None):
            assert _vehicle_type(entity, object()) == "vehicle.mini.cooper"

    def test_a_named_blueprint_is_kept(self) -> None:
        from autoware_carla_scenario.declarative import _vehicle_type

        assert _vehicle_type(Entity(id="car", vehicle_type="vehicle.a.b"), None) == (
            "vehicle.a.b"
        )
