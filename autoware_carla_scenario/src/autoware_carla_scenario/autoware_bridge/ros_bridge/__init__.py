"""The ``scenario_bridge`` ROS 2 node: Autoware's end of the ``AutowareBridge``.

The framework hosts the ``AutowareBridge`` gRPC server
(:mod:`autoware_carla_scenario.autoware_bridge.grpc_server`); this node dials it
from Autoware's ROS 2 environment, pulls the scenario's mission (initial pose and
goal) with ``GetMission``, drives localization initialization, routing and
engaging through the AD API, and pushes a single readiness flag back with
``ReportReadiness``.

It runs where ROS 2 and Autoware's message packages are -- the launcher starts it
next to the stack (:mod:`autoware_carla_scenario.autoware_stack`) -- and imports
nothing of the framework beyond :mod:`autoware_carla_scenario.autoware_bridge`,
which needs only ``grpcio`` and ``protobuf``.  So the directory of that package is
all a container needs, mounted as ``autoware_carla_scenario/autoware_bridge``
under a directory on ``PYTHONPATH``::

    python3 -m autoware_carla_scenario.autoware_bridge.ros_bridge \\
        --ros-args -p bridge_address:=localhost:50052 -p use_sim_time:=true

Importing this package does not import ``rclpy``: :mod:`.ad_api` and
:mod:`.client` are plain Python, and only :mod:`.node` needs ROS 2.
"""
