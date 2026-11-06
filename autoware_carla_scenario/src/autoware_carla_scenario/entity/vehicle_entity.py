"""NPC vehicle entity for CARLA scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..coordinate.poses import OpenDrivePose
    from ..coordinate.snap import GroundProjectionConfig

from ..traffic.base import DrivenEntity
from ..traffic.driven import BackendDriven
from ..utils.vehicles import is_emergency_vehicle_type, set_siren_lights
from ._spawn import SpawnLocation, spawn_vehicle_actor

#: Module-level flag set by :class:`~autoware_carla_scenario.scenario_runner.ScenarioRunner`
#: after warm-up ticks complete.  When ``True``, :meth:`VehicleEntity.spawn`
#: raises :class:`RuntimeError` to prevent spawning NPCs too late.
_warmup_done: bool = False


def _set_warmup_done(done: bool) -> None:
    """Set :data:`_warmup_done`, for the runner: Codon has no module objects
    to assign it through (docs/typecheck.md)."""
    global _warmup_done
    _warmup_done = done


def warmup_done() -> bool:
    """Whether the clock has started (:data:`_warmup_done`)."""
    return _warmup_done


#: The speeds (m/s, by actor id) of the vehicles a spawn card brought in
#: before the clock started -- in the init phase, which holds every vehicle
#: still: the runner gives them these with the initial speeds.
_entry_speeds: dict[int, float] = {}


def hold_entry_speed(actor_id: int, speed: float) -> None:
    """Have the vehicle *actor_id*, brought in before the clock started,
    start the run at *speed* (m/s)."""
    _entry_speeds[actor_id] = speed


def entry_speeds() -> dict[int, float]:
    """:data:`_entry_speeds`, a copy."""
    return dict(_entry_speeds)


def _clear_entry_speeds() -> None:
    """Forget the entry speeds, for the runner at the start and end of a run."""
    _entry_speeds.clear()


@dataclass
class VehicleEntityConfig:
    """Configuration for spawning a vehicle entity (ego or NPC).

    *spawn_location* determines where the vehicle is placed — either an
    explicit :class:`SpawnTransform` or a :class:`SpawnPointIndex`.
    """

    #: CARLA's ``role_name`` string.  An :class:`EntityRole` is accepted and
    #: stored as its string (:meth:`__post_init__`), so the config holds one
    #: type whatever it was built with.
    role_name: str
    spawn_location: SpawnLocation
    vehicle_type: str = "vehicle.mini.cooper"
    initial_speed_kmh: float = 0.0
    spawn_retry_max_count: int = 0
    spawn_retry_t_step: float = 0.1
    spawn_retry_z_step: float = 0.5
    od_pose: Optional["OpenDrivePose"] = None
    ground_projection: Optional["GroundProjectionConfig"] = None

    def __post_init__(self) -> None:
        # Stored as a string: CARLA's role_name is one, every reader converts
        # it to one anyway, and the static check (docs/typecheck.md) needs an
        # attribute of one type.
        self.role_name = str(self.role_name)
        # A speed is never below zero: a negative or NaN one is skipped by the
        # runner's `> 0.0` and the vehicle starts still with no error, an
        # infinite one breaks the actor's physics.  A retry step is shifted by
        # on both sides, so a negative one means nothing; 0 keeps that axis
        # (#1809).
        _check_non_negative_finite("initial_speed_kmh", self.initial_speed_kmh)
        _check_non_negative_finite("spawn_retry_t_step", self.spawn_retry_t_step)
        _check_non_negative_finite("spawn_retry_z_step", self.spawn_retry_z_step)


def _check_non_negative_finite(name: str, value: float) -> None:
    """Refuse *value* of the config field *name* unless finite and at least 0.

    Raises:
        ValueError: naming the field.
    """
    # NaN fails `value >= 0.0` as well.
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and at least 0, got {value}")


class VehicleEntity(BackendDriven):
    """Manages the lifecycle of an NPC vehicle actor in CARLA.

    This class handles spawning, destroying, and providing access to an NPC
    vehicle actor.  It mirrors the :class:`EgoVehicle` API but is intended
    for non-ego vehicles that participate in a scenario.
    """

    # Declared for the static check (docs/typecheck.md): Codon types an
    # attribute from its declaration; to Python it is a bare annotation.
    _config: VehicleEntityConfig
    _vehicle: Optional["carla.Actor"]

    def __init__(self, config: VehicleEntityConfig) -> None:
        self._config = config
        self._vehicle: Optional["carla.Actor"] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def role_name(self) -> str:
        """Return the role name that identifies this entity."""
        return self._config.role_name

    @property
    def vehicle_type(self) -> str:
        """Return the CARLA blueprint ID for this vehicle."""
        return self._config.vehicle_type

    @property
    def initial_speed_kmh(self) -> float:
        """Return the configured initial speed in km/h."""
        return self._config.initial_speed_kmh

    @property
    def actor(self) -> Optional["carla.Actor"]:
        """Return the underlying CARLA actor, or ``None`` if not spawned."""
        return self._vehicle

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def spawn(self, world: "carla.World") -> "carla.Actor":
        """Spawn the NPC vehicle in the CARLA world.

        Must be called during :meth:`BaseScenario.setup`, **before** the
        warm-up ticks run.  Spawning after warm-up raises :class:`RuntimeError`.

        Args:
            world: The CARLA world instance.

        Returns:
            The spawned vehicle actor.

        Raises:
            RuntimeError: If called after the warm-up phase has completed.
            ValueError: If the vehicle blueprint is not available or the
                spawn index is out of range.
            RuntimeError: If the vehicle could not be spawned at the
                requested location.
        """
        if _warmup_done:
            raise RuntimeError(
                "CARLA NPCs require ~5 ticks to stabilise after spawning. "
                "Spawn operations must be performed in setup() before "
                "the warm-up phase."
            )

        self._vehicle = spawn_vehicle_actor(
            world,
            self._config.vehicle_type,
            self._config.role_name,
            self._config.spawn_location,
            od_pose=self._config.od_pose,
            spawn_retry_max_count=self._config.spawn_retry_max_count,
            spawn_retry_t_step=self._config.spawn_retry_t_step,
            spawn_retry_z_step=self._config.spawn_retry_z_step,
            ground_projection=self._config.ground_projection,
        )
        return self._vehicle

    def _replace_actor(self, actor: Optional["carla.Actor"]) -> bool:
        """Accept a traffic backend taking this NPC out of the world while it
        simulates it mesoscopically, and the actor it spawns when it brings it
        back (:class:`~autoware_carla_scenario.traffic.SimulationFidelity`).

        An NPC is the scenario's only by role name, which the backend gives the
        new actor too, so nothing the scenario holds goes stale -- except the
        lane change in progress, which was the old actor's.
        """
        self.forget_lane_change()
        self._vehicle = actor
        return True

    def destroy(self) -> None:
        """Destroy the vehicle actor and release resources."""
        self.forget_lane_change()
        if self._vehicle is not None:
            self._vehicle.destroy()
            self._vehicle = None


#: The ``base_type`` CARLA gives a motorcycle's blueprint
#: (``vehicle.yamaha.yzf``, ``vehicle.kawasaki.ninja``, ...).
MOTORCYCLE_BASE_TYPE: str = "motorcycle"
#: :class:`MotorcycleEntityConfig`'s ``vehicle_type`` when it names no
#: blueprint: the first motorcycle, by blueprint id, of the server the run is
#: on -- which motorcycles a CARLA build has differs between builds.
ANY_MOTORCYCLE: str = ""


def is_motorcycle_blueprint(blueprint: "carla.ActorBlueprint") -> bool:
    """Whether *blueprint* is a motorcycle's: its ``base_type`` says so."""
    if not blueprint.has_attribute("base_type"):
        return False
    base_type = blueprint.get_attribute("base_type").as_str()
    return base_type.lower() == MOTORCYCLE_BASE_TYPE


def motorcycle_blueprints(world: "carla.World") -> list[str]:
    """The ids of *world*'s motorcycle blueprints, sorted."""
    found: list[str] = []
    for blueprint in world.get_blueprint_library().filter("vehicle.*"):
        if is_motorcycle_blueprint(blueprint):
            found.append(str(blueprint.id))
    return sorted(found)


def resolve_motorcycle_type(world: "carla.World", vehicle_type: str) -> str:
    """The motorcycle blueprint a :class:`MotorcycleEntity` of
    *vehicle_type* is spawned as on *world*.

    :data:`ANY_MOTORCYCLE` (empty) is the first of
    :func:`motorcycle_blueprints`; a blueprint named is checked to be one.

    Raises:
        ValueError: If *world* has no motorcycle, or *vehicle_type* names a
            blueprint that is not one of its motorcycles.
    """
    return _resolve_among(
        motorcycle_blueprints(world),
        vehicle_type,
        "a motorcycle",
        "motorcycles",
        "This CARLA server has no motorcycle blueprint (no vehicle blueprint "
        f"has base_type {MOTORCYCLE_BASE_TYPE!r})",
    )


def _resolve_among(
    found: list[str], vehicle_type: str, kind: str, kinds: str, none_found: str
) -> str:
    """*vehicle_type* if it is one of the blueprints *found* (each *kind*),
    or the first of them for an empty one (``ANY_*``).

    Raises:
        ValueError: *none_found* if there is none, or *vehicle_type* is not
            one of them.
    """
    if not vehicle_type:
        if not found:
            raise ValueError(none_found)
        return found[0]
    if vehicle_type not in found:
        raise ValueError(
            f"Vehicle blueprint {vehicle_type!r} is not {kind} of this CARLA "
            f"server. Available {kinds}: {found}"
        )
    return vehicle_type


class MotorcycleEntityConfig(VehicleEntityConfig):
    """Configuration for spawning a :class:`MotorcycleEntity`.

    A :class:`VehicleEntityConfig` whose ``vehicle_type`` must be a
    motorcycle's blueprint (:func:`is_motorcycle_blueprint`), and is by
    default :data:`ANY_MOTORCYCLE`: whichever motorcycle the server has.
    """

    def __init__(
        self,
        role_name: str,
        spawn_location: SpawnLocation,
        vehicle_type: str = ANY_MOTORCYCLE,
        initial_speed_kmh: float = 0.0,
        spawn_retry_max_count: int = 0,
        spawn_retry_t_step: float = 0.1,
        spawn_retry_z_step: float = 0.5,
        od_pose: Optional["OpenDrivePose"] = None,
        ground_projection: Optional["GroundProjectionConfig"] = None,
    ) -> None:
        super().__init__(
            role_name=role_name,
            spawn_location=spawn_location,
            vehicle_type=vehicle_type,
            initial_speed_kmh=initial_speed_kmh,
            spawn_retry_max_count=spawn_retry_max_count,
            spawn_retry_t_step=spawn_retry_t_step,
            spawn_retry_z_step=spawn_retry_z_step,
            od_pose=od_pose,
            ground_projection=ground_projection,
        )


class MotorcycleEntity(VehicleEntity):
    """An NPC motorcycle: a :class:`VehicleEntity` ridden on a motorcycle's
    blueprint.

    It is driven as every NPC vehicle is, by the run's traffic backend: the
    TrafficManager puts it on autopilot like any other vehicle, and SUMO
    takes it on as a vehicle of its ``motorcycle`` class rather than a
    passenger car (docs/traffic_backends.md).  :meth:`spawn` refuses a
    blueprint that is not a motorcycle's, and picks one when the config names
    none (:data:`ANY_MOTORCYCLE`); :attr:`vehicle_type` is the one spawned
    from then on.
    """

    def spawn(self, world: "carla.World") -> "carla.Actor":
        """Spawn the motorcycle in the CARLA world, as
        :meth:`VehicleEntity.spawn` spawns a vehicle.

        Raises:
            ValueError: If the world has no motorcycle, or the configured
                blueprint is not one (:func:`resolve_motorcycle_type`).
        """
        self._config.vehicle_type = resolve_motorcycle_type(
            world, self._config.vehicle_type
        )
        return super().spawn(world)


#: :class:`EmergencyVehicleEntityConfig`'s ``vehicle_type`` when it names no
#: blueprint: the first emergency vehicle, by blueprint id, of the server the
#: run is on -- which ones a CARLA build has differs between builds.
ANY_EMERGENCY_VEHICLE: str = ""


def is_emergency_vehicle_blueprint(blueprint: "carla.ActorBlueprint") -> bool:
    """Whether *blueprint* is an emergency vehicle's: its ``special_type``
    says so (CARLA 0.9), or, with none, its id (CARLA 0.10;
    :func:`~autoware_carla_scenario.utils.vehicles.is_emergency_vehicle_type`)."""
    special_type = ""
    if blueprint.has_attribute("special_type"):
        special_type = blueprint.get_attribute("special_type").as_str()
    return is_emergency_vehicle_type(str(blueprint.id), special_type)


def emergency_vehicle_blueprints(world: "carla.World") -> list[str]:
    """The ids of *world*'s emergency vehicle blueprints, sorted."""
    found: list[str] = []
    for blueprint in world.get_blueprint_library().filter("vehicle.*"):
        if is_emergency_vehicle_blueprint(blueprint):
            found.append(str(blueprint.id))
    return sorted(found)


def resolve_emergency_vehicle_type(world: "carla.World", vehicle_type: str) -> str:
    """The emergency vehicle blueprint an :class:`EmergencyVehicleEntity` of
    *vehicle_type* is spawned as on *world*.

    :data:`ANY_EMERGENCY_VEHICLE` (empty) is the first of
    :func:`emergency_vehicle_blueprints`; a blueprint named is checked to be
    one.

    Raises:
        ValueError: If *world* has no emergency vehicle, or *vehicle_type*
            names a blueprint that is not one of its emergency vehicles.
    """
    return _resolve_among(
        emergency_vehicle_blueprints(world),
        vehicle_type,
        "an emergency vehicle",
        "emergency vehicles",
        "This CARLA server has no emergency vehicle blueprint (none has "
        "special_type 'emergency', nor a police car's, an ambulance's or a "
        "fire engine's id)",
    )


class EmergencyVehicleEntityConfig(VehicleEntityConfig):
    """Configuration for spawning an :class:`EmergencyVehicleEntity`.

    A :class:`VehicleEntityConfig` whose ``vehicle_type`` must be an emergency
    vehicle's blueprint (:func:`is_emergency_vehicle_blueprint`), and is by
    default :data:`ANY_EMERGENCY_VEHICLE`: whichever one the server has.
    *emergency* puts it on a call from its spawn: its siren lights on, and
    driven as :meth:`EmergencyVehicleEntity.set_emergency` drives it.
    """

    # Declared for the static check (docs/typecheck.md).
    emergency: bool

    def __init__(
        self,
        role_name: str,
        spawn_location: SpawnLocation,
        vehicle_type: str = ANY_EMERGENCY_VEHICLE,
        initial_speed_kmh: float = 0.0,
        spawn_retry_max_count: int = 0,
        spawn_retry_t_step: float = 0.1,
        spawn_retry_z_step: float = 0.5,
        od_pose: Optional["OpenDrivePose"] = None,
        ground_projection: Optional["GroundProjectionConfig"] = None,
        emergency: bool = False,
    ) -> None:
        super().__init__(
            role_name=role_name,
            spawn_location=spawn_location,
            vehicle_type=vehicle_type,
            initial_speed_kmh=initial_speed_kmh,
            spawn_retry_max_count=spawn_retry_max_count,
            spawn_retry_t_step=spawn_retry_t_step,
            spawn_retry_z_step=spawn_retry_z_step,
            od_pose=od_pose,
            ground_projection=ground_projection,
        )
        self.emergency = emergency


class EmergencyVehicleEntity(VehicleEntity):
    """An NPC emergency vehicle -- a police car, an ambulance, a fire engine:
    a :class:`VehicleEntity` on an emergency vehicle's blueprint, which can be
    put on a call (:meth:`set_emergency`).

    On a call its siren lights (CARLA's ``VehicleLightState.Special1``, the
    light bar) flash, and the run's traffic backend drives it as one
    (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.set_emergency`):
    the TrafficManager has it run red lights and stop signs, and SUMO, which
    takes it on as a vehicle of its ``emergency`` class with a bluelight
    device, has its own vehicles make way for it too (docs/traffic_backends.md).
    Off duty it is lit and driven as any other vehicle.

    :meth:`spawn` refuses a blueprint that is not an emergency vehicle's, and
    picks one when the config names none (:data:`ANY_EMERGENCY_VEHICLE`);
    :attr:`vehicle_type` is the one spawned from then on.
    """

    # Declared for the static check (docs/typecheck.md).
    _emergency: bool

    def __init__(self, config: EmergencyVehicleEntityConfig) -> None:
        super().__init__(config)
        self._emergency = config.emergency

    @property
    def in_emergency(self) -> bool:
        """Whether it is on a call: its siren lights on, driven as one."""
        return self._emergency

    def spawn(self, world: "carla.World") -> "carla.Actor":
        """Spawn the emergency vehicle in the CARLA world, as
        :meth:`VehicleEntity.spawn` spawns a vehicle, with its siren lights
        on if it is on a call (:attr:`in_emergency`).

        Raises:
            ValueError: If the world has no emergency vehicle, or the
                configured blueprint is not one
                (:func:`resolve_emergency_vehicle_type`).
        """
        self._config.vehicle_type = resolve_emergency_vehicle_type(
            world, self._config.vehicle_type
        )
        actor = super().spawn(world)
        if self._emergency:
            set_siren_lights(actor, True)
            self._tell_backend()
        return actor

    def set_emergency(self, active: bool) -> None:
        """Put it on a call (*active*) or off duty, from now on.

        Switches its siren lights, keeping its other lights as they are, and
        has the run's traffic backend drive it as an emergency vehicle on a
        call or as any other vehicle
        (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.set_emergency`).
        Before :meth:`spawn` it only sets the state it spawns in.
        """
        self._emergency = active
        actor = self._vehicle
        if actor is None:
            return
        # A typed local: Codon unwraps the Optional here.
        vehicle: "carla.Actor" = actor
        set_siren_lights(vehicle, active)
        self._tell_backend()

    def _replace_actor(self, actor: Optional["carla.Actor"]) -> bool:
        """Accept the actor a traffic backend brings it back as
        (:meth:`VehicleEntity._replace_actor`), lit and driven as it was."""
        accepted = super()._replace_actor(actor)
        if actor is not None and self._emergency:
            # A typed local: Codon unwraps the Optional here.
            vehicle: "carla.Actor" = actor
            set_siren_lights(vehicle, True)
            self._tell_backend()
        return accepted

    def _tell_backend(self) -> None:
        """Have the backend drive it as :attr:`in_emergency` says; none is
        told before one is given (:meth:`set_traffic_backend`,
        :meth:`set_client`), nor before the actor exists.  A backend given
        it later reads :attr:`in_emergency` when it adopts it
        (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.adopt`):
        it overrides no method of the mixin, which Codon cannot dispatch
        (docs/typecheck.md)."""
        if self._vehicle is None:
            return
        if self._traffic_backend is None and self._fallback_backend is None:
            return
        entity: DrivenEntity = self
        self._resolve_backend().set_emergency(entity, self._emergency)
