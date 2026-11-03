"""Unit tests for the CARLA ground-truth payload sent to a driver policy.

The payload is what carries traffic lights and other vehicles, neither of which
the alpasim contract has a field for. A policy reads it defensively -- a missing
one means "no light, no traffic" rather than an error -- so a mistake here does
not fail loudly. These tests pin the behaviour that would otherwise rot quietly.
"""

from __future__ import annotations

import fnmatch
import math
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from carla_driver_interface.protocol import carla_driver_pb2
from autoware_carla_scenario.driver.base import DriverClientConfig
from autoware_carla_scenario.driver.geometry import Pose
from autoware_carla_scenario.driver.renderer import RendererDataBuilder


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _waypoint(
    x: float,
    *,
    road_id: int = 1,
    lane_id: int = -1,
    is_junction: bool = False,
    junction_at: Optional[float] = None,
) -> MagicMock:
    """A waypoint on a straight road along +x (CARLA coordinates).

    *junction_at* makes every waypoint at or beyond that x a junction waypoint,
    which is how the lane walk decides where to stop.
    """
    waypoint = MagicMock()
    waypoint.road_id = road_id
    waypoint.lane_id = lane_id
    waypoint.is_junction = is_junction or (junction_at is not None and x >= junction_at)
    waypoint.transform = SimpleNamespace(
        location=SimpleNamespace(x=x, y=0.0, z=0.0),
        rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0),
    )
    waypoint.next.side_effect = lambda step: [
        _waypoint(x + step, road_id=road_id, lane_id=lane_id, junction_at=junction_at)
    ]
    return waypoint


def _light(state: str = "Red", stop_x: float = 30.0, light_id: int = 7) -> MagicMock:
    """A traffic light whose stop waypoint sits at *stop_x*, junction just past."""
    light = MagicMock()
    light.id = light_id
    # A plain int, as typesafe_carla's TrafficLight.get_state() returns.
    light.get_state.return_value = int(getattr(carla.TrafficLightState, state, 99))
    light.type_id = "traffic.traffic_light"
    light.get_stop_waypoints.return_value = [
        _waypoint(stop_x, junction_at=stop_x + 5.0)
    ]
    light.get_transform.return_value = SimpleNamespace(
        location=SimpleNamespace(x=stop_x, y=0.0, z=0.0),
        rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0),
    )
    return light


def _vehicle(
    actor_id: int,
    x: float,
    y: float = 0.0,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    extent: tuple = (2.4, 1.0, 0.75),
    box_location: tuple = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
    z: float = 0.0,
    type_id: str = "vehicle.tesla.model3",
) -> MagicMock:
    actor = MagicMock()
    actor.id = actor_id
    actor.type_id = type_id
    actor.get_transform.return_value = SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=z),
        rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=yaw),
    )
    actor.get_velocity.return_value = SimpleNamespace(x=vx, y=vy, z=0.0)
    actor.bounding_box.extent = SimpleNamespace(x=extent[0], y=extent[1], z=extent[2])
    actor.bounding_box.location = SimpleNamespace(
        x=box_location[0], y=box_location[1], z=box_location[2]
    )
    return actor


class _Actors(list):
    """Stands in for CARLA's ActorList, which filters type_ids by wildcard."""

    def filter(self, pattern: str) -> List[MagicMock]:
        return [a for a in self if fnmatch.fnmatchcase(a.type_id, pattern)]


def _world(
    *,
    actors: Optional[List[MagicMock]] = None,
    ego_x: float = 0.0,
    junction_at: Optional[float] = None,
    ego_in_junction: bool = False,
) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = _Actors(actors or [])
    world.get_map.return_value.name = "Town10HD_Opt"
    world.get_map.return_value.get_waypoint.return_value = _waypoint(
        ego_x, is_junction=ego_in_junction, junction_at=junction_at
    )
    world.get_snapshot.return_value.frame = 4242
    world.get_weather.return_value = SimpleNamespace(
        cloudiness=10.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=90.0,
        sun_altitude_angle=45.0,
        fog_density=2.0,
        wetness=0.0,
    )
    return world


def _ego(actor_id: int = 1, speed_limit_kmh: float = 30.0) -> MagicMock:
    ego = _vehicle(actor_id, 0.0)
    ego.get_speed_limit.return_value = speed_limit_kmh
    ego.is_at_traffic_light.return_value = False
    ego.get_location.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    return ego


def _build(world: MagicMock, ego: MagicMock, ego_pose=None, **overrides):
    """Build and parse the payload for an ego at the origin facing +x."""
    config = DriverClientConfig(**overrides)
    builder = RendererDataBuilder(world, ego, config)
    pose = ego_pose if ego_pose is not None else Pose.identity()
    blob = builder.build(1_000, pose)
    message = carla_driver_pb2.CarlaRendererData()
    message.ParseFromString(blob)
    return message


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def test_snapshot_scalars_are_reported() -> None:
    message = _build(_world(), _ego())
    assert message.snapshot_timestamp_us == 1_000
    assert message.frame_id == 4242
    assert message.map_name == "Town10HD_Opt"


def test_speed_limit_is_converted_to_mps() -> None:
    message = _build(_world(), _ego(speed_limit_kmh=30.0))
    assert message.speed_limit_mps == pytest.approx(30.0 / 3.6, rel=1e-5)


def test_missing_speed_limit_reports_zero() -> None:
    ego = _ego()
    ego.get_speed_limit.return_value = None
    assert _build(_world(), ego).speed_limit_mps == pytest.approx(0.0)


def test_weather_is_copied_field_by_field() -> None:
    message = _build(_world(), _ego())
    assert message.weather.cloudiness == pytest.approx(10.0)
    assert message.weather.sun_altitude_angle == pytest.approx(45.0)


def test_weather_fields_absent_in_this_carla_read_as_zero() -> None:
    """CARLA 0.10 dropped some of 0.9's weather fields."""
    world = _world()
    world.get_weather.return_value = SimpleNamespace(cloudiness=20.0)
    message = _build(world, _ego())
    assert message.weather.cloudiness == pytest.approx(20.0)
    assert message.weather.wetness == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Traffic lights
# ---------------------------------------------------------------------------


def test_light_ahead_on_our_lane_governs_us() -> None:
    """The lane graph is walked; the light need not be in its trigger volume."""
    light = _light("Red", stop_x=30.0)
    message = _build(_world(actors=[light], junction_at=35.0), _ego())

    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_RED
    # The reported point is the junction mouth, past the stop waypoint.
    assert message.ego_traffic_light_distance_m > 30.0
    assert message.ego_traffic_light_distance_m < 40.0


def test_light_state_mapping() -> None:
    for name, expected in [
        ("Red", carla_driver_pb2.TRAFFIC_LIGHT_STATE_RED),
        ("Yellow", carla_driver_pb2.TRAFFIC_LIGHT_STATE_YELLOW),
        ("Green", carla_driver_pb2.TRAFFIC_LIGHT_STATE_GREEN),
        ("Off", carla_driver_pb2.TRAFFIC_LIGHT_STATE_OFF),
    ]:
        message = _build(_world(actors=[_light(name)], junction_at=35.0), _ego())
        assert message.ego_traffic_light == expected


def test_unknown_light_state_is_reported_as_unknown() -> None:
    message = _build(_world(actors=[_light("Sideways")], junction_at=35.0), _ego())
    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_UNKNOWN


def test_no_light_reports_none_and_a_negative_distance() -> None:
    """The proto documents the distance as negative when no line applies."""
    message = _build(_world(), _ego())
    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_NONE
    assert message.ego_traffic_light_distance_m < 0.0


def test_a_light_already_behind_reports_a_negative_distance() -> None:
    """Crossing the line must flip the sign, not restart a growing distance.

    A policy that stopped inside the trigger volume would otherwise be told
    forever that there is a line ahead of it.
    """
    light = _light("Red", stop_x=30.0)
    # Ego is 50 m along, so the line (and its junction mouth) is behind it.
    ego_pose = Pose.from_xyz_yaw(50.0, 0.0, 0.0, 0.0)
    message = _build(
        _world(actors=[light], ego_x=50.0, junction_at=35.0), _ego(), ego_pose
    )
    assert message.ego_traffic_light_distance_m < 0.0


def test_nothing_governs_us_inside_a_junction() -> None:
    """Having crossed the line, the thing to do is clear the box."""
    message = _build(_world(actors=[_light("Red")], ego_in_junction=True), _ego())
    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_NONE


def test_zero_sight_distance_falls_back_to_trigger_volumes() -> None:
    light = _light("Red")
    ego = _ego()
    ego.is_at_traffic_light.return_value = True
    ego.get_traffic_light.return_value = light

    message = _build(
        _world(actors=[light], junction_at=35.0),
        ego,
        traffic_light_sight_distance_m=0.0,
    )
    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_RED


def test_a_light_behind_us_on_our_own_lane_does_not_govern_us() -> None:
    """The stop-line index is keyed by lane only, so a light behind the ego on the
    current lane shares its bucket -- it must be filtered out by distance."""
    light = _light("Red", stop_x=5.0)  # stop line (and mouth) behind the ego
    ego_pose = Pose.from_xyz_yaw(20.0, 0.0, 0.0, 0.0)
    message = _build(
        _world(actors=[light], ego_x=20.0, junction_at=60.0), _ego(), ego_pose
    )
    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_NONE


def test_the_nearest_light_ahead_wins_regardless_of_actor_order() -> None:
    """A lane bucket holds its lights in arbitrary order; the nearer one governs."""
    far = _light("Green", stop_x=50.0, light_id=1)
    near = _light("Red", stop_x=20.0, light_id=2)
    # ``far`` is listed first, so a first-hit selection would pick the wrong light.
    message = _build(_world(actors=[far, near], junction_at=80.0), _ego())
    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_RED
    assert message.ego_traffic_light_distance_m < 30.0


def test_a_light_beyond_the_sight_distance_does_not_govern_us() -> None:
    light = _light("Red", stop_x=80.0)
    message = _build(
        _world(actors=[light], junction_at=85.0),
        _ego(),
        traffic_light_sight_distance_m=40.0,
    )
    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_NONE


def test_light_on_another_lane_does_not_govern_us() -> None:
    light = _light("Red")
    light.get_stop_waypoints.return_value = [_waypoint(30.0, lane_id=+1)]
    message = _build(_world(actors=[light], junction_at=35.0), _ego())
    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_NONE


def test_a_light_without_stop_waypoints_uses_its_own_position() -> None:
    light = _light("Red")
    light.get_stop_waypoints.return_value = []
    ego = _ego()
    ego.is_at_traffic_light.return_value = True
    ego.get_traffic_light.return_value = light

    message = _build(_world(actors=[light]), ego, traffic_light_sight_distance_m=0.0)
    assert message.ego_traffic_light == carla_driver_pb2.TRAFFIC_LIGHT_STATE_RED
    assert message.ego_traffic_light_distance_m == pytest.approx(30.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Other vehicles
# ---------------------------------------------------------------------------


def test_other_vehicles_are_reported_in_the_local_frame() -> None:
    """CARLA's y is South; the local frame's y is North."""
    other = _vehicle(2, x=20.0, y=4.0, vx=6.0)
    message = _build(_world(actors=[other]), _ego())

    assert len(message.actors) == 1
    actor = message.actors[0]
    assert actor.track_id == "2"
    assert actor.type_id == "vehicle.tesla.model3"
    assert actor.pose_local_to_aabb.vec.x == pytest.approx(20.0)
    assert actor.pose_local_to_aabb.vec.y == pytest.approx(-4.0)
    assert actor.dynamic_state.linear_velocity.x == pytest.approx(6.0)


def test_the_pose_is_the_centre_of_the_bounding_box() -> None:
    """#854: a vehicle's origin sits behind and below its box centre."""
    other = _vehicle(2, x=20.0, box_location=(1.3, 0.0, 0.75))
    message = _build(_world(actors=[other]), _ego())
    vec = message.actors[0].pose_local_to_aabb.vec
    assert (vec.x, vec.y, vec.z) == pytest.approx((21.3, 0.0, 0.75))


def test_the_box_offset_turns_with_the_actor_and_keeps_its_yaw() -> None:
    # CARLA yaw 90 degrees faces South (+y in CARLA), i.e. -y in the local
    # frame; a box 1.3 m ahead and 0.2 m to the right (CARLA +y, towards West).
    other = _vehicle(2, x=20.0, y=4.0, box_location=(1.3, 0.2, 0.75), yaw=90.0)
    message = _build(_world(actors=[other]), _ego())
    pose = message.actors[0].pose_local_to_aabb
    assert (pose.vec.x, pose.vec.y, pose.vec.z) == pytest.approx(
        (20.0 - 0.2, -4.0 - 1.3, 0.75)
    )
    # The quaternion is the actor's own yaw, -90 degrees in the local frame.
    yaw = 2.0 * math.atan2(pose.quat.z, pose.quat.w)
    assert yaw == pytest.approx(-math.pi / 2.0)


def test_actor_velocity_flips_with_the_handedness() -> None:
    other = _vehicle(2, x=10.0, vy=3.0)
    message = _build(_world(actors=[other]), _ego())
    assert message.actors[0].dynamic_state.linear_velocity.y == pytest.approx(-3.0)


def test_aabb_is_the_full_size_not_the_half_extent() -> None:
    other = _vehicle(2, x=10.0, extent=(2.4, 1.0, 0.75))
    message = _build(_world(actors=[other]), _ego())
    aabb = message.actors[0].aabb
    assert aabb.size_x == pytest.approx(4.8)
    assert aabb.size_y == pytest.approx(2.0)
    assert aabb.size_z == pytest.approx(1.5)


def test_the_ego_is_not_reported_as_traffic() -> None:
    ego = _ego()
    message = _build(_world(actors=[ego, _vehicle(2, x=10.0)]), ego)
    assert [a.track_id for a in message.actors] == ["2"]


def test_vehicles_beyond_the_horizon_are_dropped() -> None:
    message = _build(
        _world(actors=[_vehicle(2, x=40.0), _vehicle(3, x=400.0)]),
        _ego(),
        actor_horizon_m=150.0,
    )
    assert [a.track_id for a in message.actors] == ["2"]


def test_actor_ground_truth_can_be_switched_off() -> None:
    message = _build(
        _world(actors=[_vehicle(2, x=10.0)]),
        _ego(),
        send_actor_ground_truth=False,
    )
    assert list(message.actors) == []
    # The light and speed limit still come through.
    assert message.map_name == "Town10HD_Opt"


def test_traffic_lights_are_not_reported_as_vehicles() -> None:
    message = _build(_world(actors=[_light("Red"), _vehicle(2, x=10.0)]), _ego())
    assert [a.track_id for a in message.actors] == ["2"]


def _walker(actor_id: int, x: float, y: float = 0.0, **kwargs: Any) -> MagicMock:
    """A pedestrian: a box half a metre square and 1.8 m tall."""
    kwargs.setdefault("extent", (0.25, 0.25, 0.9))
    return _vehicle(actor_id, x, y, type_id="walker.pedestrian.0015", **kwargs)


def _other(actor_id: int, type_id: str, x: float = 10.0) -> MagicMock:
    return _vehicle(actor_id, x, type_id=type_id)


def test_pedestrians_are_reported_with_their_walking_velocity() -> None:
    """#1244: a policy that never sees a walker plans straight through it."""
    walker = _walker(2, x=10.0, y=1.0, vy=-1.4, yaw=-90.0, box_location=(0.0, 0.0, 0.9))
    message = _build(_world(actors=[walker]), _ego())

    assert [a.type_id for a in message.actors] == ["walker.pedestrian.0015"]
    actor = message.actors[0]
    assert actor.track_id == "2"
    vec = actor.pose_local_to_aabb.vec
    assert (vec.x, vec.y, vec.z) == pytest.approx((10.0, -1.0, 0.9))
    assert (actor.aabb.size_x, actor.aabb.size_y, actor.aabb.size_z) == pytest.approx(
        (0.5, 0.5, 1.8)
    )
    # CARLA y is South: walking North in CARLA is +y in the local frame.
    assert actor.dynamic_state.linear_velocity.y == pytest.approx(1.4)


def test_props_are_reported_standing_still() -> None:
    cone = _vehicle(
        3, x=15.0, extent=(0.2, 0.2, 0.4), type_id="static.prop.constructioncone"
    )
    message = _build(_world(actors=[cone]), _ego())
    assert [a.type_id for a in message.actors] == ["static.prop.constructioncone"]
    velocity = message.actors[0].dynamic_state.linear_velocity
    assert (velocity.x, velocity.y, velocity.z) == pytest.approx((0.0, 0.0, 0.0))


def test_vehicles_pedestrians_and_props_are_reported_together() -> None:
    ego = _ego()
    actors = [
        ego,
        _vehicle(2, x=20.0),
        _walker(3, x=10.0),
        _other(4, "static.prop.streetbarrier"),
    ]
    message = _build(_world(actors=actors), ego)
    assert sorted(a.track_id for a in message.actors) == ["2", "3", "4"]


@pytest.mark.parametrize(
    "type_id",
    [
        "controller.ai.walker",
        "sensor.camera.rgb",
        "sensor.other.collision",
        "spectator",
        "traffic.traffic_light",
        "traffic.stop",
        "static.trigger.friction",
    ],
)
def test_what_is_not_a_body_on_the_road_is_not_reported(type_id: str) -> None:
    """A walker's AI controller stands where its walker does; it is not a second
    pedestrian.  Sensors ride on actors, and the rest are not in the way."""
    message = _build(_world(actors=[_other(5, type_id), _walker(6, x=10.0)]), _ego())
    assert [a.track_id for a in message.actors] == ["6"]


def test_the_ego_is_excluded_among_pedestrians_and_props() -> None:
    ego = _ego()
    message = _build(
        _world(actors=[_walker(2, x=5.0), ego, _other(3, "static.prop.box01")]), ego
    )
    assert "1" not in [a.track_id for a in message.actors]


def test_a_hidden_pedestrian_is_beyond_the_horizon() -> None:
    """An entity spawned hidden is parked 500 m under the map."""
    message = _build(
        _world(actors=[_walker(2, x=0.0, z=-500.0), _walker(3, x=10.0)]), _ego()
    )
    assert [a.track_id for a in message.actors] == ["3"]


def test_pedestrians_beyond_the_horizon_are_dropped() -> None:
    message = _build(
        _world(actors=[_walker(2, x=40.0), _walker(3, x=200.0)]),
        _ego(),
        actor_horizon_m=150.0,
    )
    assert [a.track_id for a in message.actors] == ["2"]


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_a_carla_error_yields_an_empty_payload_rather_than_raising() -> None:
    """Ground truth is best-effort; losing it must not abort a scenario."""
    world = _world()
    world.get_weather.side_effect = RuntimeError("connection lost")
    builder = RendererDataBuilder(world, _ego(), DriverClientConfig())
    assert builder.build(0, Pose.identity()) == b""


def test_stop_line_geometry_is_cached() -> None:
    """Lights do not move, and the lane walk behind them is not free.

    Two caches read the stop waypoints -- the lane index and the stop points --
    so what matters is that neither refills, not the exact count.
    """
    light = _light("Red")
    world = _world(actors=[light], junction_at=35.0)
    builder = RendererDataBuilder(world, _ego(), DriverClientConfig())

    builder.build(0, Pose.identity())
    after_first = light.get_stop_waypoints.call_count

    for _ in range(4):
        builder.build(0, Pose.identity())

    assert light.get_stop_waypoints.call_count == after_first


def test_the_payload_round_trips_through_the_wire() -> None:
    world = _world(actors=[_light("Red"), _vehicle(2, x=12.0, vx=3.0)])
    builder = RendererDataBuilder(world, _ego(), DriverClientConfig())
    blob = builder.build(500, Pose.identity())

    restored = carla_driver_pb2.CarlaRendererData.FromString(blob)
    assert restored.snapshot_timestamp_us == 500
    assert len(restored.actors) == 1


def test_yawed_ego_measures_the_light_along_its_heading() -> None:
    """Distance is longitudinal in the rig frame, not a straight line."""
    light = _light("Red", stop_x=30.0)
    world = _world(actors=[light], junction_at=35.0)
    # Ego sits 10 m to the CARLA-south of the road, still facing +x. The stop
    # point is then off to one side, so the along-heading distance is shorter
    # than the straight-line distance.
    ego_pose = Pose.from_xyz_yaw(0.0, 10.0, 0.0, 0.0)
    message = _build(world, _ego(), ego_pose)

    straight = math.hypot(35.0, 10.0)
    assert message.ego_traffic_light_distance_m < straight


def test_a_t4_scene_rides_along() -> None:
    builder = RendererDataBuilder(_world(), _ego(), DriverClientConfig())
    message = carla_driver_pb2.CarlaRendererData()
    message.ParseFromString(builder.build(1_000, Pose.identity(), t4_scene=b"scene"))
    assert message.t4_scene == b"scene"
    assert message.snapshot_timestamp_us == 1_000


def test_a_t4_scene_rides_along_even_without_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = RendererDataBuilder(_world(), _ego(), DriverClientConfig())

    def fail(*args: object) -> None:
        raise RuntimeError("CARLA hiccup")

    monkeypatch.setattr(builder, "_build", fail)
    message = carla_driver_pb2.CarlaRendererData()
    message.ParseFromString(builder.build(1_000, Pose.identity(), t4_scene=b"scene"))
    assert message.t4_scene == b"scene"
