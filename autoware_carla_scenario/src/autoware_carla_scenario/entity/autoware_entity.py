"""Autoware ego vehicles.

Two entities live here:

* :class:`AutowareEntity` - a bare placeholder that spawns the ego, opts out of
  TrafficManager, and leaves the actor standing still for an external stack to
  control out of band.  Kept for backwards compatibility.

* :class:`AutowareEgoEntity` - the closed-loop entity, a
  :class:`~autoware_carla_scenario.entity.bridged_ego.BridgedEgoEntity` whose
  stack is Autoware.  Autoware (via the
  ``autoware_carla_interface`` ROS 2 node) reads CARLA sensors and applies
  control to the ego **directly**, so unlike
  :class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`
  the framework is *not* in the control loop.  This entity **spawns** the ego at
  the scenario's own spawn pose, hands Autoware the scenario's initial pose and
  goal, and waits for Autoware to become ready.  The whole startup sequence
  (localization init, routing, engage) is owned by the Autoware side; see
  :mod:`autoware_carla_scenario.autoware_bridge`.

  The scenario places the ego because the scenario is what knows where the run
  starts.  When the interface node placed it instead, the scenario's pose
  reached it afterwards, as an initial pose it moved the ego onto -- a teleport
  every other node could observe, and one that left the ego at the interface's
  own spawn point whenever the pose went missing.  The interface node must
  therefore be launched to attach to this ego rather than spawn its own.

Autoware's lifecycle: given an
:class:`~autoware_carla_scenario.autoware_stack.AutowareLauncher`, the entity
starts a fresh Autoware stack for its scenario once the mission is being served
and removes it when the scenario ends, so scenario after scenario starts from
an Autoware nothing has run in.  Without one, Autoware is someone else's to
start -- and to clean up between scenarios.

Tick ownership: the scenario framework remains the tick master; the interface
node runs as a non-ticking, asynchronous I/O bridge (``sync_mode:=false``).

Role name: the framework identifies the ego by
:data:`~autoware_carla_scenario.constants.EGO_ROLE_NAME` (``"Ego"``).  Launch
the interface node with ``ego_vehicle_role_name:=Ego`` so it looks for the
actor this entity spawns under that name.
"""

from __future__ import annotations

from .bridged_ego import BridgedEgoEntity
from .externally_driven import ExternallyDrivenEgo

__all__ = ["AutowareEgoEntity", "AutowareEntity"]

#: What both Autoware entities call the stack that drives them.
_AUTOWARE: str = "Autoware"


class AutowareEntity(ExternallyDrivenEgo):
    """Ego vehicle controlled by Autoware instead of TrafficManager.

    After spawning, the :class:`ScenarioRunner` reads
    :attr:`EgoVehicle.use_autopilot` and skips ``set_autopilot(True)``
    for this actor, leaving it free for external (Autoware) control.

    The lifecycle hooks inherited from :class:`EgoVehicle` stay no-ops, so the
    vehicle stands still unless something outside the scenario drives it.  For a
    closed loop that waits for Autoware to become ready, use
    :class:`AutowareEgoEntity`.
    """

    stack_name: str = _AUTOWARE


class AutowareEgoEntity(BridgedEgoEntity):
    """Ego vehicle spawned at the scenario's spawn pose and driven by Autoware.

    Everything about the run -- spawning the ego, handing over the mission,
    waiting for readiness, launching and stopping the stack -- is
    :class:`~autoware_carla_scenario.entity.bridged_ego.BridgedEgoEntity`'s; see
    there for the arguments.  The ``bridge`` is a required keyword argument: the
    :class:`~autoware_carla_scenario.autoware_bridge.GrpcAutowareBridgeServer`
    ``autoware_carla_interface`` dials, or ``FakeAutowareBridge`` in tests.
    """

    stack_name: str = _AUTOWARE
