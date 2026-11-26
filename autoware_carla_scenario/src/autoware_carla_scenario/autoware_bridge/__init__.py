"""autoware_bridge - minimal readiness contract with the Autoware stack.

The Autoware side of the bridge is the framework's own ``scenario_bridge`` ROS 2
node (:mod:`.ros_bridge.node`).  It owns the whole startup sequence through the
AD API -- it initializes localization (``/api/localization/initialize``), sets
the route and engages (``change_to_autonomous``) itself, and runs the readiness
aggregator in-process -- so the framework only hands over the initial pose and
goal and waits for a single readiness flag.  See :mod:`.base` for the
rationale.

The framework core depends only on the abstract :class:`AutowareBridge`.  The
concrete gRPC transport hosts the ``AutowareBridge`` *server*; the
``scenario_bridge`` node is the *client* that dials it (see the topology note in
``proto/autoware_bridge/v0/autoware_bridge.proto``).  Only that node, under
:mod:`.ros_bridge`, imports ROS 2 / rclpy; importing this package does not.

Usage::

    from autoware_carla_scenario.autoware_bridge import (
        AutowareBridge,
        AutowareBridgeConfig,
        BridgePose,
        FakeAutowareBridge,
    )
"""

from __future__ import annotations

from .base import (
    AutowareBridge,
    AutowareBridgeConfig,
    BridgePose,
    Quaternion,
    Vector3,
)
from .fake import FakeAutowareBridge
from .grpc_server import GrpcAutowareBridgeServer

__all__ = [
    "AutowareBridge",
    "AutowareBridgeConfig",
    "BridgePose",
    "FakeAutowareBridge",
    "GrpcAutowareBridgeServer",
    "Quaternion",
    "Vector3",
]
