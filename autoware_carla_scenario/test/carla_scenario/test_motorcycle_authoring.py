"""A motorcycle in the Scenario IR: ``kind: motorcycle``.

An NPC vehicle on a motorcycle's blueprint.  Its ``vehicle_type`` is ``auto``
by default -- whichever motorcycle the running CARLA has -- it may do whatever
a vehicle may, and the runtime spawns it as a
:class:`~autoware_carla_scenario.entity.MotorcycleEntity`.  The editor's half
is in ``test_editor_routes.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from autoware_carla_scenario import declarative
from autoware_carla_scenario.authoring.compiler import compile_document
from autoware_carla_scenario.authoring.models import (
    AUTO_MODEL,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.registry import _ACTION_SPECS
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.entity import ANY_MOTORCYCLE, MotorcycleEntity
from autoware_carla_scenario.entity.vehicle_entity import VehicleEntity


def _document(**motorcycle: Any) -> ScenarioDocument:
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


class TestItsActions:
    def test_it_may_do_whatever_a_vehicle_may(self) -> None:
        for spec in _ACTION_SPECS.values():
            assert ("vehicle" in spec.actor_kinds) == (
                "motorcycle" in spec.actor_kinds
            ), spec.type_id

    def test_a_pedestrians_action_on_it_is_refused(self) -> None:
        document = _document()
        document = ScenarioDocument.model_validate(
            {
                **document.model_dump(mode="json"),
                "actions": [
                    {
                        "id": "walk",
                        "type": "walk_straight",
                        "actor": "moto",
                        "params": {"speed_ms": 1.0},
                    }
                ],
            }
        )
        assert any(
            "cannot be performed by a motorcycle" in i.message
            for i in validate_document(document).errors
        )


def _fake_world(pose: Lanelet2Pose) -> CarlaWorldPose:
    return CarlaWorldPose(x=float(pose.lanelet_id), y=pose.s, z=0.0, yaw=0.0)


def _built(document: ScenarioDocument) -> dict[str, Any]:
    """What the runtime builds for every NPC of *document*."""
    scenario = object.__new__(declarative.DeclarativeScenario)
    scenario._document = document
    scenario._compiled = compile_document(document)
    scenario._ground_projection = None  # type: ignore[assignment]
    scenario.ego_config = SimpleNamespace(  # type: ignore[assignment]
        spawn_retry_max_count=0, spawn_retry_t_step=0.0, spawn_retry_z_step=0.0
    )
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
        return {
            entity.id: scenario._build_npc(entity, object())
            for entity in scenario._compiled.npcs
        }


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
