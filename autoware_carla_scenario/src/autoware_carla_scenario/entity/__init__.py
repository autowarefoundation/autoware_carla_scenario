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

    # NPC motorcycle: whichever motorcycle blueprint the server has
    rider = MotorcycleEntity(
        MotorcycleEntityConfig(role_name="rider", spawn_location=SpawnPointIndex(6))
    )

    # NPC emergency vehicle: siren lights on and driven as one while on a call
    ambulance = EmergencyVehicleEntity(
        EmergencyVehicleEntityConfig(role_name="ambulance", spawn_location=SpawnPointIndex(7))
    )
    ambulance.set_emergency(True)
"""

from ._spawn import SpawnLocation, SpawnPointIndex, SpawnTransform
from .autoware_entity import AutowareEgoEntity
from .bridged_ego import BridgedEgoEntity
from .carla_driver_entity import CarlaDriverEntity
from .ego import EgoVehicle
from .externally_driven import ExternallyDrivenEgo
from .pedestrian_entity import PedestrianEntity, PedestrianEntityConfig
from .registry import (
    clear_entities,
    find_entity_by_role_name,
    register_entity,
    unregister_entity,
)
from .vehicle_entity import (
    ANY_EMERGENCY_VEHICLE,
    ANY_MOTORCYCLE,
    EmergencyVehicleEntity,
    EmergencyVehicleEntityConfig,
    MotorcycleEntity,
    MotorcycleEntityConfig,
    VehicleEntity,
    VehicleEntityConfig,
)

__all__ = [
    "AutowareEgoEntity",
    "BridgedEgoEntity",
    "CarlaDriverEntity",
    "EgoVehicle",
    "ExternallyDrivenEgo",
    "ANY_EMERGENCY_VEHICLE",
    "EmergencyVehicleEntity",
    "EmergencyVehicleEntityConfig",
    "ANY_MOTORCYCLE",
    "MotorcycleEntity",
    "MotorcycleEntityConfig",
    "PedestrianEntity",
    "PedestrianEntityConfig",
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
