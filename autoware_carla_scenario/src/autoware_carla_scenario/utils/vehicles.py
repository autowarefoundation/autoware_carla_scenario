"""Vehicle-wide helpers that read nothing but the world they are handed."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla

__all__ = ["hold_vehicles_still"]


def hold_vehicles_still(world: "carla.World") -> None:
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
    """
    import carla  # noqa: PLC0415 -- this helper is CARLA-side by definition

    stopped = carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
    for actor in world.get_actors().filter("vehicle.*"):
        actor.apply_control(stopped)
