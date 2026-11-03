"""Build the CARLA ground-truth payload a driver policy reads.

The alpasim contract carries no traffic lights and no other road users.  Both ride
inside ``egodriver.DriveRequest.renderer_data``, an upstream-sanctioned ``bytes``
extension point, as a serialized ``carla_driver.v0.CarlaRendererData``.  A policy
that ignores those bytes stays fully compatible; one that wants CARLA ground
truth parses them.

Sending this matters more than it looks.  A policy such as
`stl_driver <https://github.com/hakuturu583/stl_driver>`_ reads the payload
defensively -- a missing one yields "no light applies" and "no other road users"
rather than an error -- so an empty ``renderer_data`` does not fail loudly.  It
silently disables every rule that depends on the world outside the ego.

The logic here is ported from ``carla_driver_interface``'s reference runtime
(``runtime/carla_world.py`` at ``af1dcd3``, Apache-2.0) so that a policy tuned
against that runtime sees the same numbers here.  The two subtleties worth
keeping are documented at
:meth:`~autoware_carla_scenario.driver.stop_lines.StopLines.governing_light` and
:meth:`~autoware_carla_scenario.driver.stop_lines.StopLines.stop_line_points`
(:mod:`~autoware_carla_scenario.driver.stop_lines`, which the library check
compiles); both were derived from measurements on ``Town10HD_Opt`` upstream, and
re-deriving them by intuition produces a policy that brakes too late and stops
too early.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..coordinate.map_manager import run_carla_map
from ..kinematics.walker_motion import WalkerVelocities, is_walker
from ..utils.clearance import _HIDDEN_BELOW_M
from ..utils.traffic_light import traffic_light_state_name
from carla_driver_interface.protocol import carla_driver_pb2, common_pb2

from .base import DriverClientConfig
from .geometry import Pose
from .observation import to_local_pose, to_local_vector
from .stop_lines import StopLines, stop_waypoints

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


logger = logging.getLogger(__name__)

__all__ = ["RendererDataBuilder"]

#: CARLA reports posted speed limits in km/h.
_KMH_TO_MPS: float = 1.0 / 3.6

#: Microseconds in a second: :meth:`RendererDataBuilder.build`'s timestamp is
#: in microseconds, :class:`WalkerVelocities` measures in seconds.
_US_PER_S: float = 1e6

#: Proto states by :func:`traffic_light_state_name`.
_LIGHT_STATES: Dict[str, "carla_driver_pb2.TrafficLightState"] = {
    "Red": carla_driver_pb2.TRAFFIC_LIGHT_STATE_RED,
    "Yellow": carla_driver_pb2.TRAFFIC_LIGHT_STATE_YELLOW,
    "Green": carla_driver_pb2.TRAFFIC_LIGHT_STATE_GREEN,
    "Off": carla_driver_pb2.TRAFFIC_LIGHT_STATE_OFF,
}

#: Weather fields, read defensively in case a CARLA build lacks one.
_WEATHER_FIELDS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "fog_density",
    "wetness",
)

#: The road users and obstacles reported to the policy, as CARLA blueprint
#: patterns: other vehicles, pedestrians and the props a scenario puts on the
#: road (cones, barriers, boxes).  Nothing else in the world is something to
#: drive around: a walker's AI controller (``controller.ai.walker``) is not a
#: body, sensors ride on actors, and the spectator and traffic lights and signs
#: are not in the way.
_ACTOR_PATTERNS: Tuple[str, ...] = ("vehicle.*", "walker.*", "static.prop.*")


class RendererDataBuilder:
    """Collects CARLA ground truth for one ego, once per policy step.

    Traffic-light geometry is cached: lights and junctions do not move, and the
    lane-graph walks behind them are not cheap enough to repeat at policy rate.

    Walkers' velocities are measured from one build to the next
    (:class:`~autoware_carla_scenario.kinematics.walker_motion.WalkerVelocities`),
    so the builder keeps where each walker stood; :meth:`reset` forgets it.  A
    builder serves one driver session.

    Args:
        world: The CARLA world.
        ego_actor: The ego vehicle actor.
        config: Driver settings; supplies the sight distance, the actor horizon
            and the lane-walk step.
        map_id: The map file set written for the policy
            (:func:`~autoware_carla_scenario.driver.hdmap_export.export_map`); with
            one, every light in the world is sent with where to stop for it.
            ``""`` sends neither.
    """

    def __init__(
        self,
        world: "carla.World",
        ego_actor: "carla.Actor",
        config: DriverClientConfig,
        map_id: str = "",
    ) -> None:
        self._world = world
        self._ego = ego_actor
        self._config = config
        # The run's map: get_map() builds a whole new one on every call.
        self._map = run_carla_map(world)
        self._map_id = map_id
        self._lights: Optional[
            List[Tuple["carla.TrafficLight", str, List[carla_driver_pb2.StopPoint]]]
        ] = None
        # The lane index and the stop points, cached there.
        self._stop_lines = StopLines(world, ego_actor, self._map, config)
        # Where each walker stood at the last build: CARLA reports 0 for one
        # walking on its controller, so its velocity is measured (#1252).
        self._walkers = WalkerVelocities()
        self._walker_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget where every walker stood (the session ended), so no
        velocity is measured across runs."""
        self._walkers.clear()
        self._walker_ids.clear()

    def build(
        self,
        timestamp_us: int,
        ego_pose: Pose,
        lidar: Sequence[carla_driver_pb2.LidarSweep] = (),
        t4_scene: bytes = b"",
    ) -> bytes:
        """Return the serialized payload for this instant.

        Args:
            timestamp_us: Simulation time of the snapshot.
            ego_pose: The ego's rig pose in the local frame, used to resolve
                distances along the ego's heading.
            lidar: This step's LiDAR sweeps, already in the rig frame.
            t4_scene: The rollout so far as a serialized T4Scene
                (``driver.t4_scene``), or empty.

        Returns:
            The serialized ``CarlaRendererData``.  If collecting the ground truth
            failed -- a hiccup that must not abort a scenario -- only the sweeps, or
            ``b""`` without any.
        """
        try:
            data = self._build(timestamp_us, ego_pose)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            # Best-effort by design: the payload is an optional extension, and a
            # CARLA hiccup here must not take the scenario down with it -- nor
            # cost the policy its sweep, which is sensor data, not ground truth.
            logger.warning("Failed to collect CARLA ground truth", exc_info=True)
            if not lidar and not t4_scene:
                return b""
            data = carla_driver_pb2.CarlaRendererData(
                snapshot_timestamp_us=timestamp_us
            )
        data.lidar.extend(lidar)
        if t4_scene:
            data.t4_scene = t4_scene
        return data.SerializeToString()

    def _build(
        self, timestamp_us: int, ego_pose: Pose
    ) -> carla_driver_pb2.CarlaRendererData:
        """Assemble the payload."""
        light = self._governing_traffic_light(ego_pose)
        return carla_driver_pb2.CarlaRendererData(
            snapshot_timestamp_us=timestamp_us,
            frame_id=self._frame_id(),
            map_name=str(self._map.name),
            weather=self._weather(),
            ego_traffic_light=self._light_state(light),
            ego_traffic_light_distance_m=self._light_distance(light, ego_pose),
            speed_limit_mps=self._speed_limit_mps(),
            actors=(
                self._actor_states(timestamp_us, ego_pose)
                if self._config.send_actor_ground_truth
                else []
            ),
            map_id=self._map_id,
            traffic_lights=self._traffic_lights() if self._map_id else [],
        )

    # ------------------------------------------------------------------
    # Traffic lights
    # ------------------------------------------------------------------

    def _governing_traffic_light(
        self, ego_pose: Pose
    ) -> Optional["carla.TrafficLight"]:
        """The light the ego must obey, found by looking down its own lane.

        The walk is :meth:`StopLines.governing_light
        <autoware_carla_scenario.driver.stop_lines.StopLines.governing_light>`'s,
        which says why: CARLA's trigger volumes tell a policy of a red light only
        when it arrives at it, and inside a junction nothing governs us.

        Falls back to ``is_at_traffic_light()`` when the sight distance is zero,
        and, outside a junction, whenever the walk finds nothing.
        """
        decided, light = self._stop_lines.governing_light(ego_pose)
        if decided:
            return light
        try:
            if self._ego.is_at_traffic_light():
                return self._ego.get_traffic_light()
        except (RuntimeError, AttributeError):
            logger.debug("is_at_traffic_light() unavailable", exc_info=True)
        return None

    def _traffic_lights(self) -> List[carla_driver_pb2.TrafficLight]:
        """Every light in the world with its state now -- the map's dynamic layer.

        Where to stop for each is cached: lights do not move.  The stop points are
        CARLA's stop waypoints as they are, named by OpenDRIVE road, lane section
        and lane, which is how the policy finds them in its map file.
        """
        if self._lights is None:
            self._lights = [
                (
                    light,
                    str(light.get_opendrive_id()),
                    [
                        carla_driver_pb2.StopPoint(
                            road_id=int(waypoint.road_id),
                            section_id=int(waypoint.section_id),
                            lane_id=int(waypoint.lane_id),
                            position_local=_vec3(
                                to_local_pose(waypoint.transform).position
                            ),
                        )
                        for waypoint in stop_waypoints(light)
                    ],
                )
                for light in self._world.get_actors().filter("traffic.traffic_light*")
            ]
        return [
            carla_driver_pb2.TrafficLight(
                opendrive_id=opendrive_id,
                state=self._light_state(light),
                stop_points=stops,
            )
            for light, opendrive_id, stops in self._lights
        ]

    def _light_state(
        self, light: Optional["carla.TrafficLight"]
    ) -> "carla_driver_pb2.TrafficLightState":
        """Map a CARLA light state onto the proto enum."""
        if light is None:
            return carla_driver_pb2.TRAFFIC_LIGHT_STATE_NONE
        return _LIGHT_STATES.get(
            traffic_light_state_name(light.get_state()),
            carla_driver_pb2.TRAFFIC_LIGHT_STATE_UNKNOWN,
        )

    def _light_distance(
        self, light: Optional["carla.TrafficLight"], ego_pose: Pose
    ) -> float:
        """Distance the ego still has to travel before *light*'s stop line.

        Along the ego's heading, negative once the line is behind
        (:meth:`StopLines.light_distance
        <autoware_carla_scenario.driver.stop_lines.StopLines.light_distance>`).
        """
        return self._stop_lines.light_distance(light, ego_pose)

    # ------------------------------------------------------------------
    # Ego and world state
    # ------------------------------------------------------------------

    def _frame_id(self) -> int:
        """The CARLA snapshot frame counter, for correlating with recordings."""
        try:
            return int(self._world.get_snapshot().frame)
        except (RuntimeError, AttributeError):
            return 0

    def _speed_limit_mps(self) -> float:
        """The posted speed limit for the ego lane, in m/s.  0 when unknown."""
        try:
            return float(self._ego.get_speed_limit() or 0.0) * _KMH_TO_MPS
        except (RuntimeError, AttributeError):
            return 0.0

    def _weather(self) -> carla_driver_pb2.CarlaWeather:
        """The current weather, reading each field defensively."""
        weather = self._world.get_weather()
        return carla_driver_pb2.CarlaWeather(
            **{
                name: float(getattr(weather, name, 0.0) or 0.0)
                for name in _WEATHER_FIELDS
            }
        )

    def _actor_states(
        self, timestamp_us: int, ego_pose: Pose
    ) -> List[carla_driver_pb2.CarlaActorState]:
        """Every other road user and prop within the horizon, in the local frame.

        Vehicles (``vehicle.*``), pedestrians (``walker.*``) and the props a
        scenario places (``static.prop.*``, :data:`_ACTOR_PATTERNS`), each told
        apart by its ``type_id``.  Not the ego, not a walker's AI controller,
        a sensor or the spectator; and not an actor parked under the map
        (hidden): one whose CARLA ``z`` is below ``_HIDDEN_BELOW_M``
        (:mod:`~autoware_carla_scenario.utils.clearance`'s, the threshold the
        other consumers of the world share) is skipped explicitly, whatever
        the horizon (#1248).

        Poses carry yaw only.  A policy reads this quaternion to orient the
        actor's bounding box in the ground plane, which roll and pitch do not
        affect for a road user on a road.

        The position is the centre of the bounding box, as the field's name
        says and a policy reads it (stl_driver's ``forecast_actors`` takes it
        for the box centre), not the actor's origin: a CARLA vehicle's origin
        is often well behind its box centre, and at ground level under it
        (#854).  The box's offset is rotated by the actor's yaw only, like
        the pose itself.

        A vehicle's and a prop's velocity is the one CARLA reports, a prop's
        zero unless its physics has it moving.  A walker's is measured
        (:class:`~autoware_carla_scenario.kinematics.walker_motion.WalkerVelocities`,
        as the trajectory log, the T4 dataset, the metrics and the conditions
        measure it): CARLA reports 0 for a walker walking on its controller,
        which places it rather than driving it (#1252).  It is measured from
        where the walker stood at the previous build to where it stands now,
        over the simulated time between them (*timestamp_us*), so it is the
        walker's average velocity over the policy step.  At the first build,
        and once a walker has been teleported -- brought in from under the
        map, put where its walk starts -- there is nothing to measure from and
        it is the one CARLA reports.  Hidden walkers are read too, though not
        reported, so the one brought in is no walk.  Every velocity is
        resolved in the local frame.
        """
        states: List[carla_driver_pb2.CarlaActorState] = []
        horizon = self._config.actor_horizon_m
        ego_position = ego_pose.position
        # One fetch from the server; CARLA's filter matches type_ids locally.
        world_actors = self._world.get_actors()
        actors = [
            actor
            for pattern in _ACTOR_PATTERNS
            for actor in world_actors.filter(pattern)
        ]
        elapsed = timestamp_us / _US_PER_S
        walker_ids: set[int] = set()
        for actor in actors:
            if actor.id == self._ego.id:
                continue
            transform = actor.get_transform()
            walker_velocity = None
            if is_walker(actor):
                # Measured before any skip -- hidden, beyond the horizon -- so
                # that where it stood is always the last build's: a walker
                # brought in from under the map is then a teleport, no walk.
                walker_ids.add(int(actor.id))
                walker_velocity = self._walkers.walker(
                    actor, elapsed, transform.location
                )
            # Parked under the map until brought in: not on the road (#1248).
            # Compared in CARLA's world frame, as every other consumer does.
            if float(transform.location.z) < _HIDDEN_BELOW_M:
                continue
            box = actor.bounding_box
            centre = box.location
            # The offset is in the actor's own CARLA frame (y right): its y
            # flips into the right-handed local frame, like the pose's.
            pose = to_local_pose(transform) @ Pose.from_xyz_yaw(
                float(centre.x), -float(centre.y), float(centre.z), 0.0
            )
            if float(np.linalg.norm(pose.position - ego_position)) > horizon:
                continue

            velocity = (
                walker_velocity if walker_velocity is not None else actor.get_velocity()
            )
            extent = box.extent
            states.append(
                carla_driver_pb2.CarlaActorState(
                    track_id=str(actor.id),
                    type_id=str(actor.type_id),
                    pose_local_to_aabb=pose.to_proto(),
                    aabb=common_pb2.AABB(
                        size_x=2.0 * extent.x,
                        size_y=2.0 * extent.y,
                        size_z=2.0 * extent.z,
                    ),
                    dynamic_state=common_pb2.DynamicState(
                        # Resolved in the local frame, not the actor's own: a
                        # policy rotates these into its rig frame and would
                        # double-count an actor-frame velocity.
                        linear_velocity=_vec3(
                            to_local_vector(velocity.x, velocity.y, velocity.z)
                        ),
                    ),
                )
            )
        # Walkers gone from the world: nothing more to measure.
        for gone in self._walker_ids - walker_ids:
            self._walkers.forget(gone)
        self._walker_ids = walker_ids
        return states


def _vec3(values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> common_pb2.Vec3:
    """Return *values* as a protobuf ``Vec3``."""
    return common_pb2.Vec3(x=values[0], y=values[1], z=values[2])
