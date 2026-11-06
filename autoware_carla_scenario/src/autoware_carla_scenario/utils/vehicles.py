"""Vehicle-wide helpers that read nothing but the world they are handed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Collection

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "EMERGENCY_BLUEPRINT_WORDS",
    "EMERGENCY_SPECIAL_TYPE",
    "as_vehicle",
    "hold_vehicles_still",
    "is_emergency_vehicle_type",
    "release_vehicle",
    "set_siren_lights",
    "with_siren_lights",
]

#: The ``special_type`` CARLA 0.9 gives an emergency vehicle's blueprint (a
#: police car's, an ambulance's, a fire engine's), whatever its ``base_type``.
EMERGENCY_SPECIAL_TYPE: str = "emergency"
#: What an emergency vehicle's blueprint id holds, for CARLA 0.10, which gives
#: none a ``special_type`` (``entity.fit.FIT_EXCLUDED`` leaves the same out).
EMERGENCY_BLUEPRINT_WORDS: tuple[str, ...] = ("firetruck", "ambulance", "cop", "police")


def is_emergency_vehicle_type(type_id: str, special_type: str) -> bool:
    """Whether the vehicle blueprint *type_id*, of ``special_type``
    *special_type* (empty when it has none), is an emergency vehicle's: its
    special type says so, or, with none, its id does."""
    if special_type:
        return special_type.lower() == EMERGENCY_SPECIAL_TYPE
    lowered = type_id.lower()
    for word in EMERGENCY_BLUEPRINT_WORDS:
        if word in lowered:
            return True
    return False


def as_vehicle(actor: "carla.Actor") -> "carla.Vehicle":
    """*actor* as the vehicle a control, a light state or a physics control is
    read from and applied to.

    At run time that is *actor* itself, which typesafe_carla's Python API
    takes as it is.  Its Codon library has these on a ``carla.Vehicle`` only,
    which the check gets with the checked conversion: Codon reads
    ``TYPE_CHECKING`` as a compile-time true (docs/typecheck.md), and Python
    as false.
    """
    if TYPE_CHECKING:
        return actor.as_vehicle()
    else:
        return actor


def _siren_bit() -> int:
    """CARLA's ``VehicleLightState.Special1``: the light bar (the beacons) of
    an emergency vehicle's blueprint -- a police car's, an ambulance's, a fire
    engine's.  A vehicle without one shows nothing for it."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    return int(carla.VehicleLightState.Special1)


def with_siren_lights(light_state: int, on: bool) -> int:
    """*light_state* with its siren lights (:func:`_siren_bit`) *on* or off,
    every other bit kept."""
    bit = _siren_bit()
    if on:
        return light_state | bit
    return light_state & ~bit


def set_siren_lights(actor: "carla.Actor", on: bool) -> None:
    """Switch vehicle *actor*'s siren lights *on* or off, its other lights
    (the turn indicators, the brake lights, ...) as they are."""
    vehicle = as_vehicle(actor)
    state = int(vehicle.get_light_state())
    lit = with_siren_lights(state, on)
    if lit != state:
        vehicle.set_light_state(lit)


def hold_vehicles_still(world: "carla.World", spare: Collection[int] = ()) -> None:
    """Keep every vehicle stopped while the run is still being set up.

    The init phase has to advance simulation time -- an autonomy stack only
    localizes, routes and engages while the clock ticks, and its sensors only
    publish then -- but nothing should have moved before the run starts.  Two
    things would move otherwise: a car parked on a slope rolls, and an ego
    engages partway through the wait and drives off before the scenario has
    begun measuring anything.

    The hold is the brakes, not frozen physics: a stopped car with its handbrake
    on is a state the simulation and the stack both understand, while a vehicle
    with physics disabled reports poses no suspension has settled.  It is
    re-applied every tick because whatever drives the ego applies its own
    control every tick too.

    The vehicles ``spare`` names by actor id are left alone: ones the ego's
    entity is already moving (see
    :attr:`~autoware_carla_scenario.entity.ego.EgoVehicle.carried_actor_ids`).
    """
    import typesafe_carla.carla as carla  # noqa: PLC0415 -- this helper is CARLA-side by definition

    stopped = carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id in spare:
            continue
        actor.apply_control(stopped)


def release_vehicle(actor: "carla.Actor", throttle: float = 0.0) -> None:
    """Take the init phase's brakes off one vehicle, at *throttle*.

    CARLA's client sends a vehicle control only when it differs from the last
    one sent through the same actor handle, and a handle that has sent nothing
    holds the default control -- which is exactly the released one.  The hold
    goes out through other handles (a fresh :meth:`World.get_actors` each tick),
    so a plain release from any handle is dropped and the brakes stay on.  This
    release names first gear, which an automatic gearbox ignores, so it differs
    from the default and is sent.
    """
    import typesafe_carla.carla as carla  # noqa: PLC0415 -- this helper is CARLA-side by definition

    actor.apply_control(
        carla.VehicleControl(
            throttle=float(throttle),
            brake=0.0,
            hand_brake=False,
            manual_gear_shift=False,
            gear=1,
        )
    )
