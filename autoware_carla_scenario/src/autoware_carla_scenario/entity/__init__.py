"""entity – vehicle entity types for CARLA scenarios.

Usage::

    from autoware_carla_scenario.entity import (
        EgoVehicle,
        SpawnPointIndex,
        SpawnTransform,
        VehicleEntity,
        VehicleEntityConfig,
    )

    # NPC vehicle at spawn-point index 5
    config = VehicleEntityConfig(
        role_name="npc_vehicle_1",
        spawn_location=SpawnPointIndex(5),
        vehicle_type="vehicle.mini.cooper",
    )
    npc = VehicleEntity(config)
    actor = npc.spawn(world)
    # Autopilot is enabled automatically by ScenarioRunner after warm-up ticks
"""

from ._spawn import SpawnLocation, SpawnPointIndex, SpawnTransform
from .autoware_entity import AutowareEgoEntity, AutowareEntity
from .carla_driver_entity import CarlaDriverEntity
from .ego import EgoVehicle
from .registry import (
    clear_entities,
    find_entity_by_role_name,
    register_entity,
    unregister_entity,
)
from .vehicle_entity import VehicleEntity, VehicleEntityConfig

__all__ = [
    "AutowareEgoEntity",
    "AutowareEntity",
    "CarlaDriverEntity",
    "EgoVehicle",
    "clear_entities",
    "find_entity_by_role_name",
    "register_entity",
    "unregister_entity",
    "SpawnLocation",
    "SpawnPointIndex",
    "SpawnTransform",
    "VehicleEntity",
    "VehicleEntityConfig",
]
