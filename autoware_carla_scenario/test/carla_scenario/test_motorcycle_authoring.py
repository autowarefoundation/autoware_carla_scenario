"""A motorcycle in the Scenario IR: ``kind: motorcycle``.

An NPC vehicle on a motorcycle's blueprint.  Its ``vehicle_type`` is ``auto``
by default -- whichever motorcycle the running CARLA has -- it may do whatever
a vehicle may, and the runtime spawns it as a
:class:`~autoware_carla_scenario.entity.MotorcycleEntity`.  The editor's half
is in ``test_editor_routes.py``.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from autoware_carla_scenario import declarative
from autoware_carla_scenario.authoring.compiler import compile_document
from autoware_carla_scenario.authoring.models import (
    AUTO_MODEL,
    ScenarioDocument,
    actor_kind,
)
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.entity import ANY_MOTORCYCLE, MotorcycleEntity
from autoware_carla_scenario.entity.vehicle_entity import VehicleEntity

from .test_entity_field_parity import built_entities


def _document(
    actions: tuple[dict[str, Any], ...] = (), **motorcycle: Any
) -> ScenarioDocument:
    return ScenarioDocument.model_validate(
        {
            "entities": [
                {"id": "ego", "kind": "ego", "spawn": {"lanelet_id": 242}},
                {"id": "car", "kind": "vehicle", "spawn": {"lanelet_id": 243}},
                {
                    "id": "moto",
                    "kind": "motorcycle",
                    "spawn": {"lanelet_id": 244},
                    **motorcycle,
                },
            ],
            "actions": list(actions),
            "assertions": {
                "pass": [{"type": "elapsed_time", "params": {"duration_seconds": 5.0}}],
                "fail": [{"type": "timeout", "params": {"timeout_seconds": 30.0}}],
            },
        }
    )


def _errors_on(document: ScenarioDocument, field: str) -> list[str]:
    return [
        i.message
        for i in validate_document(document).errors
        if i.path == f"entities[2].{field}"
    ]


class TestTheDocument:
    def test_its_blueprint_is_the_servers_motorcycle_by_default(self) -> None:
        moto = _document().entities[2]
        assert moto.vehicle_type == AUTO_MODEL
        assert _errors_on(_document(), "vehicle_type") == []
        assert _errors_on(_document(), "size") == []

    def test_a_named_vehicle_blueprint_is_valid(self) -> None:
        document = _document(vehicle_type="vehicle.yamaha.yzf")
        assert _errors_on(document, "vehicle_type") == []

    def test_a_walker_blueprint_is_refused(self) -> None:
        document = _document(vehicle_type="walker.pedestrian.0015")
        assert _errors_on(document, "vehicle_type")

    @pytest.mark.parametrize("vehicle_type", [AUTO_MODEL, "vehicle.yamaha.yzf"])
    def test_a_size_is_refused(self, vehicle_type: str) -> None:
        """A motorcycle is never chosen by its size, so nothing would read one."""
        document = _document(
            vehicle_type=vehicle_type, size={"length": 2.0, "width": 0.8}
        )
        assert _errors_on(document, "size")

    def test_it_is_an_npc_with_an_npc_role(self) -> None:
        compiled = compile_document(_document())
        assert [e.id for e in compiled.npcs] == ["car", "moto"]
        assert compiled.role_of("moto") == "npc2"

    def test_it_round_trips(self) -> None:
        document = _document()
        again = ScenarioDocument.model_validate(document.model_dump(mode="json"))
        assert again.entities[2].kind == "motorcycle"
        assert again.entities[2].vehicle_type == AUTO_MODEL


def _action(type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": "a1", "type": type_id, "actor": "moto", "params": params}


class TestItsActions:
    def test_it_acts_as_a_vehicle(self) -> None:
        """Matched against an action's kinds as a vehicle, so every action
        for a vehicle is a motorcycle's too, with no spec naming it."""
        assert actor_kind("motorcycle") == "vehicle"
        assert actor_kind("pedestrian") == "pedestrian"

    def test_a_vehicles_action_on_it_is_valid(self) -> None:
        document = _document((_action("lane_change", direction="left"),))
        assert not [
            i
            for i in validate_document(document).errors
            if "cannot be performed" in i.message
        ]

    def test_a_pedestrians_action_on_it_is_refused(self) -> None:
        document = _document((_action("walk_straight", speed_ms=1.0),))
        assert any(
            "cannot be performed by a motorcycle" in i.message
            for i in validate_document(document).errors
        )


_built = built_entities


class TestTheRun:
    def test_it_is_spawned_as_a_motorcycle_entity(self) -> None:
        built = _built(_document())
        assert type(built["car"]) is VehicleEntity
        moto = built["moto"]
        assert isinstance(moto, MotorcycleEntity)
        # Resolved on the server when it spawns.
        assert moto.vehicle_type == ANY_MOTORCYCLE
        assert moto.role_name == "npc2"

    def test_a_named_blueprint_is_the_one_asked_for(self) -> None:
        built = _built(_document(vehicle_type="vehicle.yamaha.yzf"))
        assert built["moto"].vehicle_type == "vehicle.yamaha.yzf"

    def test_a_hidden_one_is_parked_as_a_motorcycle_too(self) -> None:
        with (
            mock.patch.object(
                declarative.DeclarativeScenario, "_parked", return_value=True
            ),
            mock.patch.object(
                declarative.DeclarativeScenario,
                "_parking_transform",
                return_value=object(),
            ),
        ):
            built = _built(_document(spawn={"hidden": True, "lanelet_id": 244}))
        assert isinstance(built["moto"], MotorcycleEntity)
