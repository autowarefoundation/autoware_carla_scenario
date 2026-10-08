"""Vehicle-wide helpers that read nothing but the world they are handed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Collection

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["hold_vehicles_still", "release_vehicle"]


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
            throttle=float(throttle), brake=0.0, hand_brake=False, manual_gear_shift=False, gear=1
        )
    )
