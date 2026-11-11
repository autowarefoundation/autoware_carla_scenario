"""The Autoware ego vehicle.

:class:`AutowareEgoEntity` is a
:class:`~autoware_carla_scenario.entity.bridged_ego.BridgedEgoEntity` whose
stack is Autoware.  Autoware (via the ``autoware_carla_interface`` ROS 2 node)
reads CARLA sensors and applies control to the ego **directly**, so unlike
:class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`
the framework is *not* in the control loop.  This entity **spawns** the ego at
the scenario's own spawn pose, hands Autoware the scenario's initial pose and
goal, and waits for Autoware to become ready.  The whole startup sequence
(localization init, routing, engage) is owned by the Autoware side; see
:mod:`autoware_carla_scenario.autoware_bridge`.

The scenario places the ego because the scenario is what knows where the run
starts.  When the interface node placed it instead, the scenario's pose reached
it afterwards, as an initial pose it moved the ego onto -- a teleport every
other node could observe, and one that left the ego at the interface's own
spawn point whenever the pose went missing.  The interface node must therefore
be launched to attach to this ego rather than spawn its own.

Autoware's lifecycle: given an
:class:`~autoware_carla_scenario.autoware_stack.AutowareLauncher`, the entity
starts a fresh Autoware stack for its scenario once the mission is being served
and removes it when the scenario ends, so scenario after scenario starts from
an Autoware nothing has run in.  Without one, Autoware is someone else's to
start -- and to clean up between scenarios.

Tick ownership: the scenario framework remains the tick master; the interface
node runs as a non-ticking, asynchronous I/O bridge (``sync_mode:=false``).

Role name: the framework identifies the ego by
:data:`~autoware_carla_scenario.constants.EGO_ROLE_NAME` (``"Ego"``), and spawns
it as ``Ego`` -- or as ``ego`` with ``native_autoware_interface`` (below).
Launch the interface node with the role name the ego is spawned with
(``ego_vehicle_role_name:=Ego``, or ``ego_vehicle_role_name:=ego`` with
``native_autoware_interface``) so it looks for the actor this entity spawns.

CARLA's native Autoware interface: a CARLA 1.0 server started with ``-ros2``
applies ``/control/command/control_cmd`` and publishes ``/vehicle/status/*``
itself, with no client in the control loop -- but only for a vehicle spawned
with the blueprint attribute ``ros2_autoware_control=true`` and the role name
``ego`` or ``hero``.  ``native_autoware_interface=True``
(``ego.native_autoware_interface``) spawns the ego that way, as ``ego``
(:data:`~autoware_carla_scenario.entity_role.NATIVE_AUTOWARE_EGO_ROLE_NAME`);
the framework still finds it by ``EGO_ROLE_NAME``, which matches ``ego`` and
``hero`` too (:func:`~autoware_carla_scenario.entity_role.canonical_role_name`).
A server whose blueprints have no ``ros2_autoware_control`` spawns the ego as
``Ego``, as without the setting, with a warning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..entity_role import NATIVE_AUTOWARE_EGO_ROLE_NAME
from .bridged_ego import BridgedEgoEntity

if TYPE_CHECKING:
    from ..autoware_bridge.base import (
        AutowareBridge,
        AutowareBridgeConfig,
        BridgePose,
    )
    from ..autoware_stack.launcher import AutowareLauncher

__all__ = ["AutowareEgoEntity"]


class AutowareEgoEntity(BridgedEgoEntity):
    """Ego vehicle spawned at the scenario's spawn pose and driven by Autoware.

    Everything about the run -- spawning the ego, handing over the mission,
    waiting for readiness, launching and stopping the stack -- is
    :class:`~autoware_carla_scenario.entity.bridged_ego.BridgedEgoEntity`'s; see
    there for the arguments.  The ``bridge`` is a required keyword argument: the
    :class:`~autoware_carla_scenario.autoware_bridge.GrpcAutowareBridgeServer`
    ``autoware_carla_interface`` dials, or ``FakeAutowareBridge`` in tests.

    Args:
        native_autoware_interface: Spawn the ego for CARLA 1.0's native
            Autoware interface: with ``ros2_autoware_control=true`` and the
            role name ``ego``, when the server's blueprint has that attribute
            (see the module docstring).  ``False`` (the default) spawns it as
            ``Ego``, for an interface node that drives it from the client.
    """

    stack_name: str = "Autoware"

    _native_autoware_interface: bool

    def __init__(
        self,
        config: Optional["AutowareBridgeConfig"] = None,
        *,
        bridge: "AutowareBridge",
        initial_pose: Optional["BridgePose"] = None,
        goal_pose: Optional["BridgePose"] = None,
        launcher: Optional["AutowareLauncher"] = None,
        episode_name: str = "",
        native_autoware_interface: bool = False,
    ) -> None:
        super().__init__(
            config,
            bridge=bridge,
            initial_pose=initial_pose,
            goal_pose=goal_pose,
            launcher=launcher,
            episode_name=episode_name,
        )
        self._native_autoware_interface = native_autoware_interface

    @property
    def native_autoware_interface(self) -> bool:
        """Whether the ego is spawned for CARLA's native Autoware interface."""
        return self._native_autoware_interface

    def _ros2_autoware_role_name(self) -> Optional[str]:
        if self._native_autoware_interface:
            return NATIVE_AUTOWARE_EGO_ROLE_NAME
        return None
