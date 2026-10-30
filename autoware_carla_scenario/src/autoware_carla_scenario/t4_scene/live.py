"""The rollout so far as a T4Scene, built while it runs, for a T4Scene driver.

A driver policy that plans from T4Scenes (``carla_driver_interface``'s
``T4SceneDriver``, ``driver.t4_scene``) is sent the run's history with every
drive.  :class:`LiveT4Scene` builds it the way a finished run's trajectory log
is converted (:func:`~autoware_carla_scenario.t4_scene.convert.log_to_scene`):
a trajectory recorder writes the same records, here into an in-memory log
instead of a file, and each scene is that log's last rows.  The run's scene and
its file export cannot drift apart.

On top of the log, each scene carries what a live policy is given and a log
does not hold, on the rows it has them:

* the ego's velocity and acceleration as the policy is told them (with the
  egodriver's observations: of ``base_link``, the acceleration an estimate),
  its angular velocity and front tyres' angle, and the odometry times;
* the current LiDAR sweep, ``[N, 5]`` in the rig (``base_link``) frame, T4's
  layout: intensity on the 0-255 scale, the fifth column zero;
* the route's lanelets -- the road ahead the policy is sent
  (``route_waypoints_in_rig``) walked along the map's lanelets, with the few
  just passed -- and the goal.

History only: the current row is the last.  Before the run has
``history_rows`` rows, the scene has fewer -- the driver decides what a short
history means, nothing is made up here.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Optional, Sequence

import numpy as np
import numpy.typing as npt

from ..constants import FIXED_DELTA_SECONDS
from ..trajectory_recorder import TrajectoryRecorder
from ._proto import t4_scene_pb2 as pb
from .convert import (
    _TICK_NS,
    _array,
    _Log,
    _load_map,
    _Map,
    _read_map,
    log_to_scene,
    map_of,
)
from .route import RouteGraph

__all__ = ["EgoExtras", "LiveT4Scene"]

logger = logging.getLogger(__name__)

#: Route lanelets the ego has passed that a scene's route keeps, as the closed
#: loop's policies did (the route Autoware's planner is given starts behind
#: the ego).
_ROUTE_KEPT_BEHIND = 4
#: How far the ego may be from a route lanelet's centreline and be on it.
_ON_ROUTE_M = 3.0


@dataclass(frozen=True)
class EgoExtras:
    """What the policy is told of the ego on one tick beyond its pose and velocity.

    Attributes:
        acceleration_rig: ``(2,)`` acceleration in the rig frame, m/s^2.
        velocity_rig: ``(2,)`` velocity in the rig frame (of ``base_link``,
            not the actor's origin the log records), m/s; ``None`` keeps the
            log's.
        angular_velocity: ``(3,)`` body angular velocity, rad/s.
        tire_steering_angle: The front tyres' angle, rad, positive to the left;
            ``None`` when the vehicle cannot say.
    """

    acceleration_rig: npt.NDArray[np.float64]
    angular_velocity: npt.NDArray[np.float64]
    tire_steering_angle: Optional[float] = None
    velocity_rig: Optional[npt.NDArray[np.float64]] = None


class _Discard:
    """A stream that keeps nothing: the live recorder's records go to its log."""

    def write(self, data: bytes) -> int:
        return len(data)

    def close(self) -> None:
        pass


class _LiveRecorder(TrajectoryRecorder):
    """A trajectory recorder whose records feed a log in memory, not a file."""

    def __init__(self, log: _Log, **kwargs: Any) -> None:
        super().__init__(Path(os.devnull), **kwargs)
        self._log = log

    def _open(self) -> IO[bytes]:
        return _Discard()  # type: ignore[return-value]

    def _emit(self, record: dict[str, Any]) -> None:
        self._log.feed(record)

    @staticmethod
    def _state_value(value: float) -> float:
        """Unrounded: a scene's poses are the ones the policy is told, not a
        file's millimetres."""
        return float(value)


class LiveT4Scene:
    """The run's history as a T4Scene, on a 0.1 s grid, as it runs.

    Args:
        history_rows: Rows a scene holds, the current one included.
        step_s: The grid's step: a multiple of the tick.
        max_distance_m: Road users this close to the ego are its objects.
        rear_axle_offset_m: Where the ego's ``base_link`` is from its actor
            origin, as the run takes it; by default as the recorder reads it.
        wheel_base_m: The ego's wheel base, written in its shape; ``None``
            leaves it out (CARLA 0.10 reports no wheel positions).
        map_source, lanelet2_path, opendrive_path, projector_type: The run's
            map, as the trajectory recorder takes them.
    """

    def __init__(
        self,
        history_rows: int,
        *,
        step_s: float = 0.1,
        max_distance_m: Optional[float] = 100.0,
        rear_axle_offset_m: Optional[float] = None,
        wheel_base_m: Optional[float] = None,
        map_source: Optional[str] = None,
        lanelet2_path: Optional[Path] = None,
        opendrive_path: Optional[Path] = None,
        projector_type: Optional[str] = None,
    ) -> None:
        step_frames = round(step_s / FIXED_DELTA_SECONDS)
        if history_rows < 1:
            raise ValueError(f"history_rows must be at least 1, not {history_rows}")
        if step_frames < 1 or not math.isclose(
            step_frames * FIXED_DELTA_SECONDS, step_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"step_s={step_s} is not a multiple of the tick ({FIXED_DELTA_SECONDS} s)"
            )
        self.history_rows = history_rows
        self.max_distance_m = max_distance_m
        self.rear_axle_offset_m = rear_axle_offset_m
        self.wheel_base_m = wheel_base_m
        self._log = _Log(step_frames=step_frames)
        self._recorder = _LiveRecorder(
            self._log,
            map_source=map_source,
            lanelet2_path=lanelet2_path,
            opendrive_path=opendrive_path,
            projector_type=projector_type,
        )
        self._extras: dict[int, EgoExtras] = {}
        self._route: list[int] = []
        self._map: Optional[_Map] = None
        self._map_read = False
        self._map_path: Optional[Path] = None

    # -- recording ------------------------------------------------------------

    def start(self, world: Any) -> None:
        """Begin with the world's traffic lights and map, before the first tick."""
        self._recorder.start(world)

    def record(
        self, world: Any, elapsed: float, extras: Optional[EgoExtras] = None
    ) -> None:
        """Take the tick the world was just stepped to, and what the policy is
        told of the ego then; keep only the rows a scene can still read."""
        self._recorder.record(world, elapsed)
        log = self._log
        if log.last_frame < 0:
            return
        if extras is not None:
            self._extras[log.last_frame] = extras
        # From the row a scene now ends on (frames()), on the grid: a tick
        # between grid rows must not take the oldest row a scene still shows.
        current = log.last_frame - (log.last_frame - log.first_frame) % log.step_frames
        oldest = current - (self.history_rows - 1) * log.step_frames
        log.forget_before(oldest)
        for frame in [f for f in self._extras if f < oldest]:
            del self._extras[frame]

    def use_map(self, lanelet2_path: Path) -> None:
        """Put the scenes on the Lanelet2 map *lanelet2_path*, in the run's
        local frame (as a map set's is), instead of the run's own map."""
        self._map_path = Path(lanelet2_path)
        self._map = None
        self._map_read = False

    @property
    def map_resource(self) -> Optional[pb.MapResource]:
        """The run's map as the scenes reference it, read once; ``None`` without one."""
        loaded = self._loaded_map()
        return None if loaded is None else loaded.resource

    def _loaded_map(self) -> Optional[_Map]:
        """The map the files the recorder names hold; else the run's own, as
        the map manager loaded it (the frame the recorder's offsets are of)."""
        if not self._map_read:
            self._map_read = True
            try:
                if self._map_path is not None:
                    self._map = _local_map(self._log, self._map_path)
                else:
                    self._map = _load_map(self._log, None, None, None)
                if self._map is None:
                    self._map = _manager_map()
            except Exception:  # noqa: BLE001 - scenes go on without a map
                logger.warning("No map for the live T4Scenes", exc_info=True)
        return self._map

    # -- the scene --------------------------------------------------------------

    def frames(self) -> list[int]:
        """The CARLA frames of the rows a scene now holds, oldest first."""
        log = self._log
        if log.last_frame < 0:
            return []
        current = log.last_frame - (log.last_frame - log.first_frame) % log.step_frames
        first = max(
            log.first_frame, current - (self.history_rows - 1) * log.step_frames
        )
        return list(range(first, current + 1, log.step_frames))

    def scene(
        self,
        stem: str,
        *,
        lidar_points: Optional[npt.NDArray[np.float32]] = None,
        lidar_age_ns: int = 0,
        route_lanelet_ids: Optional[Sequence[int]] = None,
        road_ahead: Optional[npt.NDArray[np.float64]] = None,
        road_ahead_origin: Optional[npt.NDArray[np.float64]] = None,
        goal_pose: Optional[Sequence[float]] = None,
    ) -> Optional[pb.T4Scene]:
        """The scene of the rows so far, the current one last; ``None`` before
        the first tick.

        Args:
            stem: The scene's id.
            lidar_points: The current sweep, ``[N, 4]`` rig-frame x, y, z and
                intensity in ``[0, 1]``.
            lidar_age_ns: How long before the current row it was taken.
            route_lanelet_ids: The route's lanelets, in order.
            road_ahead: Instead, the road ahead as ``(N, 2)`` map-frame
                points from the ego on: the route is the lanelets it runs
                along, after the few the ego has just passed.
            road_ahead_origin: Where the ego was, map frame, as the policy
                was told when the road ahead was sent; by default the
                current row's pose.
            goal_pose: ``(x, y, yaw)`` in the map frame.
        """
        frames = self.frames()
        if not frames:
            return None
        scene = log_to_scene(
            self._log,
            frames,
            len(frames) - 1,
            stem,
            self._loaded_map(),
            max_distance_m=self.max_distance_m,
            rear_axle_offset_m=self.rear_axle_offset_m,
            source="the run",
        )
        self._fill_ego_extras(scene, frames)
        rows = len(frames)
        if lidar_points is not None:
            scene.lidar.rows.extend(pb.LidarRow() for _ in range(rows - 1))
            now_ns = (frames[-1] - self._log.first_frame) * _TICK_NS
            sweep = pb.Lidar(
                lidar_pc=_array(_t4_points(lidar_points), "f4"),
                timestamp_ns=now_ns - int(lidar_age_ns),
            )
            scene.lidar.rows.append(pb.LidarRow(value=sweep))
        if road_ahead is not None and route_lanelet_ids is None:
            loaded = self._loaded_map()
            if loaded is not None and loaded.route is not None:
                poses = np.frombuffer(scene.ego.poses.data, dtype="<f8").reshape(
                    rows, 3
                )
                route_lanelet_ids = self._route_along(
                    loaded.route,
                    np.asarray(road_ahead, dtype=np.float64),
                    poses[-1, :2]
                    if road_ahead_origin is None
                    else np.asarray(road_ahead_origin, dtype=np.float64),
                )
        if route_lanelet_ids is not None:
            route = pb.RouteData(lane_ids=_array(list(route_lanelet_ids), "i8"))
            scene.routes.rows.extend(pb.RouteRow(value=route) for _ in range(rows))
        if goal_pose is not None:
            x, y, yaw = (float(v) for v in goal_pose)
            scene.goal_poses.CopyFrom(
                _array(np.tile([x, y, math.cos(yaw), math.sin(yaw)], (rows, 1)), "f8")
            )
        return scene

    def _route_along(
        self,
        graph: RouteGraph,
        points: npt.NDArray[np.float64],
        ego: npt.NDArray[np.float64],
    ) -> list[int]:
        """The road ahead's lanelets after the few of the last route the ego
        has passed; the walk starts from the one it is on when it can."""
        here = graph.nearest(self._route, ego, _ON_ROUTE_M)
        behind = (
            []
            if here is None
            else self._route[max(0, here - _ROUTE_KEPT_BEHIND) : here]
        )
        pin = None if here is None else self._route[here]
        self._route = behind + graph.match(points, pin)
        return self._route

    def _fill_ego_extras(self, scene: pb.T4Scene, frames: Sequence[int]) -> None:
        """The ego's acceleration, angular velocity, tyre angle and odometry
        times, on the rows they were told; masked on the others."""
        ego = scene.ego
        rows = len(frames)
        poses = np.frombuffer(ego.poses.data, dtype="<f8").reshape(rows, 3)
        velocities = (
            np.frombuffer(ego.velocities.data, dtype="<f8").reshape(rows, 2).copy()
        )
        velocity_mask = (
            np.frombuffer(ego.velocities.mask, dtype=bool).reshape(rows, 2).copy()
            if ego.velocities.HasField("mask")
            else np.zeros((rows, 2), dtype=bool)
        )
        accelerations = np.zeros((rows, 2))
        angular = np.zeros((rows, 3))
        steering = np.zeros(rows)
        told = np.zeros(rows, dtype=bool)
        steered = np.zeros(rows, dtype=bool)
        for row, frame in enumerate(frames):
            extras = self._extras.get(frame)
            if extras is None:
                continue
            told[row] = True
            c, s = math.cos(poses[row, 2]), math.sin(poses[row, 2])
            ax, ay = (float(v) for v in extras.acceleration_rig[:2])
            # Rig axes into the scene's: the row's heading.
            accelerations[row] = (c * ax - s * ay, s * ax + c * ay)
            if extras.velocity_rig is not None:
                vx, vy = (float(v) for v in extras.velocity_rig[:2])
                velocities[row] = (c * vx - s * vy, s * vx + c * vy)
                velocity_mask[row] = False
            angular[row] = extras.angular_velocity[:3]
            if extras.tire_steering_angle is not None:
                steering[row] = extras.tire_steering_angle
                steered[row] = True
        ego.velocities.CopyFrom(_array(velocities, "f8", velocity_mask))
        ego.accelerations.CopyFrom(_array(accelerations, "f8", ~told[:, None]))
        ego.angular_velocities.CopyFrom(_array(angular, "f8", ~told[:, None]))
        ego.tire_steering_angles.CopyFrom(_array(steering, "f8", ~steered))
        ego.odometry_timestamps_ns.CopyFrom(
            _array([(f - self._log.first_frame) * _TICK_NS for f in frames], "i8")
        )
        if self.wheel_base_m is not None:
            ego.shape.wheel_base = float(self.wheel_base_m)


def _local_map(log: _Log, lanelet2_path: Path) -> Optional[_Map]:
    """A Lanelet2 map in the run's local frame (``local_x``/``local_y``), on
    the scenes' ``map`` frame; ``None`` -- the run's own map then -- when that
    frame is offset from the local one."""
    record = log.map or {}
    if any(float(v) for v in (record.get("mgrs_offset") or ())) or float(
        record.get("z_offset") or 0.0
    ):
        logger.warning(
            "%s is in the run's local frame, which its map frame is offset from; "
            "the scenes keep the run's own map",
            lanelet2_path,
        )
        return None
    path = lanelet2_path.resolve()
    return _read_map(path, None, "local_cartesian", (path.stat().st_mtime_ns, None))


def _manager_map() -> Optional[_Map]:
    """The run's Lanelet2 map as the map manager holds it; ``None`` before one
    is loaded.  Named by its resource's digest: there is no file to name it by."""
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415

    manager = MapManager.get_instance()
    try:
        lanelet_map = manager.lanelet_map
    except RuntimeError:
        return None
    try:
        network = manager.road_network
    except RuntimeError:
        network = None
    loaded = map_of(lanelet_map, "", network)
    # Its points left where they are: the manager's map is everyone's.
    return loaded._replace(
        digest=loaded.resource.sha256, route=RouteGraph.of(lanelet_map, owned=False)
    )


def _t4_points(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """``[N, 4]`` rig-frame x, y, z, intensity in ``[0, 1]`` as T4's ``[N, 5]``:
    intensity on the 0-255 scale, the fifth (source-defined) column zero."""
    raw = np.asarray(points, dtype=np.float32).reshape(-1, 4)
    out = np.zeros((len(raw), 5), dtype=np.float32)
    out[:, :3] = raw[:, :3]
    out[:, 3] = raw[:, 3] * 255.0
    return out
