"""An entity config stores its role name as CARLA's ``role_name`` string.

``VehicleEntityConfig``, ``PedestrianEntityConfig`` and ``EgoConfig`` accept an
:class:`EntityRole` or a ``str`` and hold the string either way (#72), so what
a config holds has one type, which the static check needs (docs/typecheck.md).
The subclasses that add state of their own compare and print it too (#2534).
"""

from __future__ import annotations

import copy
import dataclasses

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    EgoConfig,
    EmergencyVehicleEntityConfig,
    EntityRole,
    PedestrianEntity,
    PedestrianEntityConfig,
    SpawnPointIndex,
    SpawnTransform,
    VehicleEntity,
    VehicleEntityConfig,
)
from autoware_carla_scenario.coordinate import Lanelet2Pose
from autoware_carla_scenario.entity.vehicle_entity import (
    _same_vehicle_config,
    _vehicle_config_repr_fields,
)


@pytest.mark.parametrize("role", [EntityRole.npc(3), "npc3"])
def test_a_vehicle_config_stores_the_role_as_its_string(role: EntityRole | str) -> None:
    # The field is typed `str`; an EntityRole is still taken at run time.
    config = VehicleEntityConfig(role_name=role, spawn_location=SpawnPointIndex(0))  # type: ignore[arg-type]
    assert type(config.role_name) is str
    assert config.role_name == "npc3"
    assert VehicleEntity(config).role_name == "npc3"


@pytest.mark.parametrize("role", [EntityRole("pedestrian1"), "pedestrian1"])
def test_a_pedestrian_config_stores_the_role_as_its_string(
    role: EntityRole | str,
) -> None:
    config = PedestrianEntityConfig(
        role_name=role,  # type: ignore[arg-type]
        spawn_location=SpawnTransform(carla.Transform()),
    )
    assert type(config.role_name) is str
    assert config.role_name == "pedestrian1"
    assert PedestrianEntity(config).role_name == "pedestrian1"


def test_the_ego_config_stores_the_ego_role_as_its_string() -> None:
    config = EgoConfig(SpawnPointIndex(0))
    assert type(config.role_name) is str
    assert config.role_name == str(EGO_ROLE_NAME) == "Ego"


def test_a_role_name_is_not_newly_validated() -> None:
    # Only EntityRole validates; a plain string is stored as it was given.
    config = VehicleEntityConfig(
        role_name="Not A Role", spawn_location=SpawnPointIndex(0)
    )
    assert config.role_name == "Not A Role"


def test_a_config_rebuilt_from_its_fields_keeps_the_string() -> None:
    config = VehicleEntityConfig(
        role_name=EntityRole.npc(1),  # type: ignore[arg-type]
        spawn_location=SpawnPointIndex(0),
    )
    copy = dataclasses.replace(config, vehicle_type="vehicle.tesla.model3")
    assert copy.role_name == "npc1"
    assert copy == dataclasses.replace(config, vehicle_type="vehicle.tesla.model3")


class TestASubclassComparesItsOwnState:
    """EmergencyVehicleEntityConfig and EgoConfig are not dataclasses
    themselves, so the dataclass's ``==`` and repr left their own state out
    (#2534)."""

    def test_emergency_takes_part_in_eq_and_repr(self) -> None:
        on_call = EmergencyVehicleEntityConfig(
            "amb1", SpawnPointIndex(1), emergency=True
        )
        off_duty = EmergencyVehicleEntityConfig(
            "amb1", SpawnPointIndex(1), emergency=False
        )
        assert on_call != off_duty
        assert on_call == EmergencyVehicleEntityConfig(
            "amb1", SpawnPointIndex(1), emergency=True
        )
        assert repr(on_call) == (
            "EmergencyVehicleEntityConfig(role_name='amb1', "
            "spawn_location=SpawnPointIndex(value=1), vehicle_type='', "
            "initial_speed_kmh=0.0, spawn_retry_max_count=0, "
            "spawn_retry_t_step=0.1, spawn_retry_z_step=0.5, od_pose=None, "
            "ground_projection=None, emergency=True)"
        )

    def test_the_emergency_config_still_compares_its_vehicle_fields(self) -> None:
        a = EmergencyVehicleEntityConfig("amb1", SpawnPointIndex(1))
        assert a != EmergencyVehicleEntityConfig("amb2", SpawnPointIndex(1))
        assert a != EmergencyVehicleEntityConfig("amb1", SpawnPointIndex(2))
        assert a != EmergencyVehicleEntityConfig(
            "amb1", SpawnPointIndex(1), initial_speed_kmh=10.0
        )
        # Not equal to a plain vehicle config of the same fields, as the
        # dataclass's == compares only configs of one class.
        assert a != VehicleEntityConfig(
            "amb1", SpawnPointIndex(1), vehicle_type=a.vehicle_type
        )

    def test_the_goal_and_waypoints_take_part_in_eq_and_repr(self) -> None:
        goal = Lanelet2Pose(lanelet_id=1, s=2.0, t=0.0)
        via = Lanelet2Pose(lanelet_id=3, s=4.0, t=0.0)
        config = EgoConfig(SpawnPointIndex(0), goal_pose=goal, waypoint_poses=[via])
        assert config == EgoConfig(
            SpawnPointIndex(0), goal_pose=goal, waypoint_poses=[via]
        )
        assert config != EgoConfig(SpawnPointIndex(0), waypoint_poses=[via])
        assert config != EgoConfig(SpawnPointIndex(0), goal_pose=goal)
        assert config != EgoConfig(
            SpawnPointIndex(0),
            goal_pose=Lanelet2Pose(lanelet_id=1, s=5.0, t=0.0),
            waypoint_poses=[via],
        )
        assert config != EgoConfig(
            SpawnPointIndex(1), goal_pose=goal, waypoint_poses=[via]
        )
        text = repr(config)
        assert text.startswith("EgoConfig(role_name='Ego', ")
        assert f"goal_pose={goal!r}" in text
        assert text.endswith(f", waypoint_poses=[{via!r}])")

    def test_the_configs_stay_unhashable(self) -> None:
        # As a non-frozen dataclass is: its fields can change.
        with pytest.raises(TypeError):
            hash(EgoConfig(SpawnPointIndex(0)))
        with pytest.raises(TypeError):
            hash(EmergencyVehicleEntityConfig("amb1", SpawnPointIndex(1)))

    def test_the_helpers_cover_every_vehicle_config_field(self) -> None:
        # The helpers name the fields one by one (Codon has no
        # dataclasses.fields()), so a field added to VehicleEntityConfig must
        # be added to them too: this fails until it is.
        config = VehicleEntityConfig(
            "npc1",
            SpawnPointIndex(2),
            vehicle_type="vehicle.tesla.model3",
            initial_speed_kmh=12.5,
            spawn_retry_max_count=3,
            spawn_retry_t_step=0.25,
            spawn_retry_z_step=0.75,
        )
        assert repr(config) == (
            f"VehicleEntityConfig({_vehicle_config_repr_fields(config)})"
        )
        assert _same_vehicle_config(config, copy.copy(config))
        for field in dataclasses.fields(VehicleEntityConfig):
            changed = copy.copy(config)
            setattr(changed, field.name, object())
            assert not _same_vehicle_config(config, changed), field.name
