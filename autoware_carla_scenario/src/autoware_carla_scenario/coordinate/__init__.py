"""coordinate – mutual conversion between Lanelet2, OpenDRIVE, and CARLA world poses.

Usage::

    from autoware_carla_scenario.coordinate import (
        CarlaWorldPose, Lanelet2Pose, OpenDrivePose,
        MapManager,
        to_carla_world, to_lanelet2, to_opendrive,
    )

    # Initialize maps once
    mm = MapManager.get_instance()
    mm.initialize(xodr_path=Path("map.xodr"), lanelet2_path=Path("map.osm"))

    # Convert poses
    carla_pose = to_carla_world(Lanelet2Pose(lanelet_id=1234, s=10.0, t=0.5))
    od_pose    = to_opendrive(carla_pose)
    ll2_pose   = to_lanelet2(od_pose)
"""

from .frames import CoordinateFrame, FrameMismatchError, frame_of
from .map_manager import MapManager
from .poses import AnyPose, CarlaWorldPose, Lanelet2Pose, OpenDrivePose
from .snap import GroundProjectionConfig, snap_to_carla_road
from .traffic_light import (
    find_nearest_traffic_light,
    get_signal_ids_for_controller,
    lanelet2_traffic_light_id_to_opendrive_controller_id,
)
from .stop_line import (
    get_stop_line_linestrings,
    get_stop_line_linestrings_with_following,
    get_stop_line_poses,
    get_stop_line_poses_with_following,
)
from .transform import (
    project_onto_road,
    to_carla_location,
    to_carla_world,
    to_lanelet2,
    to_map_frame,
    to_opendrive,
)

__all__ = [
    "GroundProjectionConfig",
    "AnyPose",
    "CarlaWorldPose",
    "CoordinateFrame",
    "FrameMismatchError",
    "Lanelet2Pose",
    "OpenDrivePose",
    "MapManager",
    "frame_of",
    "find_nearest_traffic_light",
    "get_signal_ids_for_controller",
    "get_stop_line_linestrings",
    "get_stop_line_linestrings_with_following",
    "get_stop_line_poses",
    "get_stop_line_poses_with_following",
    "lanelet2_traffic_light_id_to_opendrive_controller_id",
    "project_onto_road",
    "snap_to_carla_road",
    "to_carla_location",
    "to_carla_world",
    "to_lanelet2",
    "to_map_frame",
    "to_opendrive",
]
