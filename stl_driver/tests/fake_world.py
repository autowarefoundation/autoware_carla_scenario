# SPDX-License-Identifier: Apache-2.0
"""A closed loop without a simulator, over real gRPC.

carla_driver_interface 2.x keeps only the policy half of the contract; its
``testing.FakeLoop`` drives a straight road and sends no CARLA payload, so it
can exercise neither the route-following nor the light and actor rules. This
loop plays the runtime the same way -- session start, egomotion, route, CARLA
ground truth, ``drive`` -- along any route, with a ``CarlaRendererData`` a test
subclass can rewrite each step.

The ego follows the returned plan exactly: its pose at the next step is the
plan's pose at that time, and its speed and acceleration the plan's there.
What is tested is the policy and the wire, not a controller; vehicle dynamics
belong in a scenario against CARLA (stl_driver/README.md, "Running it in a
scenario").
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import numpy as np
from carla_driver_interface.contract import contract_metadata, negotiate
from carla_driver_interface.driver import BaseDriver
from carla_driver_interface.geometry import Pose, Trajectory, waypoints_to_proto
from carla_driver_interface.protocol import (
    CarlaRendererData,
    DriveRequest,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    DynamicState,
    EgodriverServiceStub,
    RolloutEgoTrajectory,
    Route,
    RouteRequest,
    TrafficLightState,
    Vec3,
    channel_options,
    pack_renderer_data,
)
from carla_driver_interface.server import serving

__all__ = ["FakeWorld", "Rollout", "straight_then_curve_route"]


def straight_then_curve_route(
    straight_m: float = 40.0,
    radius_m: float = 30.0,
    turn_rad: float = math.pi / 2,
    resolution_m: float = 1.0,
) -> np.ndarray:
    """A route in the local frame that goes straight, then turns left at constant radius."""
    points = [
        np.array([distance, 0.0, 0.0]) for distance in np.arange(0.0, straight_m, resolution_m)
    ]
    centre = np.array([straight_m, radius_m, 0.0])
    n_arc = max(2, int(round(radius_m * turn_rad / resolution_m)))
    for i in range(n_arc + 1):
        theta = turn_rad * i / n_arc
        points.append(centre + radius_m * np.array([math.sin(theta), -math.cos(theta), 0.0]))
    return np.stack(points)


@dataclass
class Rollout:
    """What happened in one :meth:`FakeWorld.run`."""

    #: Ego poses in the local frame, one per step.
    poses: list[Pose] = field(default_factory=list)
    #: Ego speed at each pose [m/s].
    speeds_mps: list[float] = field(default_factory=list)
    #: Distance from the route at each pose [m].
    route_errors_m: list[float] = field(default_factory=list)
    #: Arc length along the route of each pose's projection [m].
    stations_m: list[float] = field(default_factory=list)
    route_length_m: float = 0.0
    terminated_by_policy: bool = False

    @property
    def positions(self) -> np.ndarray:
        return np.stack([pose.position[:2] for pose in self.poses])

    @property
    def speeds(self) -> np.ndarray:
        return np.asarray(self.speeds_mps)

    @property
    def distance_travelled_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    @property
    def route_completion(self) -> float:
        return self.stations_m[-1] / self.route_length_m

    @property
    def max_route_error_m(self) -> float:
        return max(self.route_errors_m)


class FakeWorld:
    """Drive a policy along ``route`` and record where it went.

    Args:
        route: ``(N, 3)`` route in the local frame; the ego starts at its first
            point, headed along it, at rest.
        speed_limit_mps: Posted in every step's ``CarlaRendererData``.
        step_s: Simulated time between drive calls.
        horizon_m: How far ahead of the ego the route sent each step reaches,
            as a runtime trims it.
    """

    def __init__(
        self,
        route: np.ndarray | None = None,
        *,
        speed_limit_mps: float = 20.0,
        step_s: float = 0.1,
        horizon_m: float = 80.0,
    ) -> None:
        self.route = (
            straight_then_curve_route() if route is None else np.asarray(route, dtype=np.float64)
        )
        self.speed_limit_mps = speed_limit_mps
        self.step_s = step_s
        self.horizon_m = horizon_m
        segments = np.linalg.norm(np.diff(self.route[:, :2], axis=0), axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(segments)])

    def environment(self, pose: Pose, now_us: int) -> CarlaRendererData:
        """The CARLA ground truth for the ego at ``pose``; override to add lights or actors."""
        return CarlaRendererData(
            snapshot_timestamp_us=now_us,
            map_name="FakeTown",
            ego_traffic_light=TrafficLightState.TRAFFIC_LIGHT_STATE_NONE,
            ego_traffic_light_distance_m=-1.0,
            speed_limit_mps=self.speed_limit_mps,
        )

    def station_of(self, xy: np.ndarray) -> tuple[float, float]:
        """Arc length of ``xy``'s projection onto the route, and its distance from it."""
        best = (0.0, math.inf)
        for i in range(len(self.route) - 1):
            a, b = self.route[i, :2], self.route[i + 1, :2]
            ab = b - a
            t = float(np.clip(np.dot(xy - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
            distance = float(np.linalg.norm(xy - (a + t * ab)))
            if distance < best[1]:
                best = (float(self._arc[i] + t * (self._arc[i + 1] - self._arc[i])), distance)
        return best

    def run(self, policy: BaseDriver, steps: int) -> Rollout:
        """Serve ``policy`` and drive one session of ``steps`` drive calls against it."""
        import grpc

        with serving(policy, port=0, host="127.0.0.1") as port:
            with grpc.insecure_channel(f"127.0.0.1:{port}", options=channel_options()) as channel:
                return self._run(EgodriverServiceStub(channel), steps)

    def _run(self, stub: EgodriverServiceStub, steps: int) -> Rollout:
        session = str(uuid.uuid4())
        _, revision = negotiate(stub, timeout=60.0)
        request = DriveSessionRequest(session_uuid=session, random_seed=0)
        request.debug_info.scene_id = "fake:closed_loop"
        stub.start_session(request, timeout=60.0, metadata=contract_metadata(revision))

        heading = self.route[1, :2] - self.route[0, :2]
        pose = Pose.from_xyz_yaw(
            float(self.route[0, 0]),
            float(self.route[0, 1]),
            0.0,
            math.atan2(float(heading[1]), float(heading[0])),
        )
        speed = accel = 0.0
        step_us = int(round(self.step_s * 1e6))
        result = Rollout(route_length_m=float(self._arc[-1]))
        try:
            for k in range(steps):
                now_us = k * step_us
                station, error = self.station_of(pose.position[:2])
                result.poses.append(pose)
                result.speeds_mps.append(speed)
                result.stations_m.append(station)
                result.route_errors_m.append(error)
                self._observe(stub, session, now_us, pose, speed, accel, station)

                response = stub.drive(
                    DriveRequest(
                        session_uuid=session,
                        time_now_us=now_us,
                        time_query_us=now_us + step_us,
                        renderer_data=pack_renderer_data(self.environment(pose, now_us)),
                    ),
                    timeout=60.0,
                )
                if response.terminate_session:
                    result.terminated_by_policy = True
                    break
                plan = Trajectory.from_proto(response.trajectory)
                following = plan.interpolate(now_us + step_us) if len(plan) >= 2 else None
                if following is None:
                    # An empty plan means "hold": stay put.
                    speed = accel = 0.0
                    continue
                speed, accel = _speed_and_accel(plan, now_us + step_us)
                pose = following
        finally:
            stub.close_session(DriveSessionCloseRequest(session_uuid=session), timeout=60.0)
        return result

    def _observe(
        self,
        stub: EgodriverServiceStub,
        session: str,
        now_us: int,
        pose: Pose,
        speed: float,
        accel: float,
        station: float,
    ) -> None:
        egomotion = RolloutEgoTrajectory(session_uuid=session)
        egomotion.trajectory.CopyFrom(Trajectory([now_us], [pose]).to_proto())
        egomotion.dynamic_states.append(
            DynamicState(
                linear_velocity=Vec3(x=speed, y=0.0, z=0.0),
                linear_acceleration=Vec3(x=accel, y=0.0, z=0.0),
            )
        )
        stub.submit_egomotion_observation(egomotion, timeout=60.0)
        # The route ahead of the ego's projection, resampled every metre.
        stations = np.arange(station, min(station + self.horizon_m, self._arc[-1]) + 1e-9, 1.0)
        ahead = np.stack(
            [np.interp(stations, self._arc, self.route[:, axis]) for axis in range(3)], axis=1
        )
        in_rig = pose.inverse().transform_points(ahead)
        stub.submit_route(
            RouteRequest(
                session_uuid=session,
                route=Route(timestamp_us=now_us, waypoints=waypoints_to_proto(in_rig)),
            ),
            timeout=60.0,
        )


def _speed_and_accel(plan: Trajectory, time_us: int, delta_us: int = 20_000) -> tuple[float, float]:
    """The plan's speed and longitudinal acceleration at ``time_us``, by central differences.

    The instantaneous values, not the step's average: from rest, the average over
    the first step is half the speed the plan reaches at its end.
    """
    poses = [plan.interpolate(time_us + k * delta_us) for k in (-1, 0, 1)]
    if any(pose is None for pose in poses):
        return 0.0, 0.0
    before, now, after = (pose.position[:2] for pose in poses)  # type: ignore[union-attr]
    dt = delta_us * 1e-6
    speed_before = float(np.linalg.norm(now - before)) / dt
    speed_after = float(np.linalg.norm(after - now)) / dt
    return 0.5 * (speed_before + speed_after), (speed_after - speed_before) / dt
