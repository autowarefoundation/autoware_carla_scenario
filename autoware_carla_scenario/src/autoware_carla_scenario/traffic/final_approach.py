"""Drive the last metres of a given path where the TrafficManager would not.

The TrafficManager plans a few seconds past the end of a path it was given,
on the lanes beyond it, and destroys a vehicle whose plan finds no lane to go
on.  A path that ends a few metres before a dead end -- a route's goal next
to the map's edge -- would get its vehicle destroyed, or (#523) held braking
well short of its goal.  :class:`FinalApproach` drives that vehicle the rest
of its path itself, with the controller a ``RoutingAction`` in ``follow``
mode drives with (:class:`~autoware_carla_scenario.driver.control.TrajectoryFollower`),
at :data:`APPROACH_SPEED_MPS`, and stops it on the path's last point.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

from ..trajectory.model import ResolvedTrajectory
from ..utils.vehicles import as_vehicle

if TYPE_CHECKING:
    import typesafe_carla.carla as carla
    from carla_driver_interface.geometry import Trajectory as _Plan

    from ..driver.control import TrajectoryFollower

__all__ = ["APPROACH_SPEED_MPS", "FinalApproach"]

#: The speed (m/s) the last metres are driven at.
APPROACH_SPEED_MPS: float = 4.0
#: How near (m) the path's last point the approach has arrived.
_ARRIVED_M: float = 1.0
#: How far (m) ahead of the vehicle the controller is shown the path.
_HORIZON_M: float = 30.0
#: Metres between the points of the plan.
_STEP_M: float = 1.0
#: How long (s) a vehicle driven out of its junction waits for a vehicle in
#: its way before it is handed to the TrafficManager all the same (two such
#: vehicles facing each other would wait for ever).  Long: handed over still
#: in the junction, the TrafficManager may take any way out of it (vehicle_178
#: waited 5 s behind a car standing at a light and went straight on).
LEAD_OUT_WAIT_S: float = 30.0
#: How long (s) it waits for a walker in its way before it is handed on: a
#: walker held on the road (at a waypoint, by its route's timing) may stand
#: there far longer, and the TrafficManager drives round it.
LEAD_OUT_WALKER_WAIT_S: float = 5.0
#: How long (s) a vehicle driven out of its junction may make no way along
#: its path, nothing in its way, before it is handed on (wedged on a kerb, it
#: would push against it for the rest of the run).
LEAD_OUT_STUCK_S: float = 5.0
#: Progress (m) along its path that counts as making way.
_PROGRESS_M: float = 0.2


class FinalApproach:
    """The rest of a given path, driven to its end and stopped there.

    Args:
        actor: The vehicle, off autopilot.
        start: Where it is now.
        points: The path's points still ahead of it, the last one its end.
        stop: Whether it stops on the path's end (a final approach), or is
            only done there, at speed, for the TrafficManager to take it on
            (a way out of the junction it started in).
    """

    _actor: "carla.Actor"
    _path: ResolvedTrajectory
    _follower: "TrajectoryFollower"
    _here: float
    _stop: bool
    _blocked_s: float
    _stuck_s: float
    _progress_at: float
    done: bool

    def __init__(
        self,
        actor: "carla.Actor",
        start: "carla.Location",
        points: List["carla.Location"],
        stop: bool = True,
    ) -> None:
        from ..driver.control import ControlConfig, TrajectoryFollower  # noqa: PLC0415
        from ..utils.powertrain import ChaosPowertrain  # noqa: PLC0415

        xs = [start.x]
        ys = [start.y]
        zs = [start.z]
        yaws: List[Optional[float]] = [None]
        for point in points:
            if math.hypot(point.x - xs[-1], point.y - ys[-1]) < 0.1:
                continue
            xs.append(point.x)
            ys.append(point.y)
            zs.append(point.z)
            yaws.append(None)
        if len(xs) < 2:
            # Already on its end: a path of one step on the spot.
            xs.append(start.x + 0.1)
            ys.append(start.y)
            zs.append(start.z)
            yaws.append(None)
        self._actor = actor
        self._path = ResolvedTrajectory(xs, ys, zs, yaws)
        # The MKZ's throttle floor, as a FOLLOW run uses: a crawl needs it.
        self._follower = TrajectoryFollower(
            ControlConfig(throttle_deadband=0.25),
            ChaosPowertrain.from_physics_control(
                as_vehicle(actor).get_physics_control()
            ),
        )
        self._here = 0.0
        self._stop = stop
        self._blocked_s = 0.0
        self._stuck_s = 0.0
        self._progress_at = 0.0
        self.done = False

    @property
    def length(self) -> float:
        """How long (m) the path driven is."""
        return self._path.length

    def step(self, dt: float, elapsed: float) -> None:
        """One tick: drive on along the path, or stop on its end."""
        import typesafe_carla.carla as carla  # noqa: PLC0415
        from carla_driver_interface.geometry import Pose  # noqa: PLC0415

        actor = self._actor
        vehicle = as_vehicle(actor)
        if self.done:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return
        transform = actor.get_transform()
        self._here = self._path.project(
            transform.location.x, transform.location.y, near=self._here
        )
        if self._here >= self._path.length - _ARRIVED_M:
            self.done = True
            if self._stop:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return
        velocity = actor.get_velocity()
        from ..actions._trajectory_run import (  # noqa: PLC0415
            _vehicle_in_way,
            _walker_in_way,
        )

        speed = math.hypot(velocity.x, velocity.y)
        walker = _walker_in_way(actor, transform, speed)
        if walker or _vehicle_in_way(actor, transform, speed):
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            self._blocked_s += dt
            wait = LEAD_OUT_WALKER_WAIT_S if walker else LEAD_OUT_WAIT_S
            if not self._stop and self._blocked_s >= wait:
                self.done = True  # the TrafficManager takes it on as it is
            return
        self._blocked_s = 0.0
        if not self._stop:
            # Making no way, nothing in its way: wedged; handed on.
            if self._here > self._progress_at + _PROGRESS_M:
                self._progress_at = self._here
                self._stuck_s = 0.0
            else:
                self._stuck_s += dt
                if self._stuck_s >= LEAD_OUT_STUCK_S:
                    self.done = True
                    return
        angular = actor.get_angular_velocity()
        control = vehicle.get_control()
        try:
            gear = control.gear
        except AttributeError:
            gear = 0
        command = self._follower.step(
            self._plan(elapsed),
            # The plan's frame is CARLA's with y flipped: right-handed.
            Pose.from_xyz_yaw(
                transform.location.x,
                -transform.location.y,
                transform.location.z,
                -math.radians(transform.rotation.yaw),
            ),
            math.hypot(velocity.x, velocity.y),
            dt,
            yaw_rate_rps=-math.radians(angular.z),
            now_us=int(round(elapsed * 1e6)),
            gear=int(gear),
        )
        vehicle.apply_control(command.to_carla_control())

    def _plan(self, elapsed: float) -> "_Plan":
        """The path ahead, at :data:`APPROACH_SPEED_MPS`, slowing onto its end."""
        from carla_driver_interface.geometry import Pose  # noqa: PLC0415
        from carla_driver_interface.geometry import Trajectory as Plan  # noqa: PLC0415

        path = self._path
        plan = Plan.empty()
        distance = max(0.0, self._here - 2.0)
        last = min(path.length, self._here + _HORIZON_M)
        stamp = elapsed
        previous = distance
        previous_us: Optional[int] = None
        while distance <= last + 1e-9:
            # Slower towards the end: the speed braking at 1.5 m/s² allows
            # (not where it only hands the vehicle on).
            room = max(path.length - distance, 0.0)
            speed = APPROACH_SPEED_MPS
            if self._stop:
                speed = min(APPROACH_SPEED_MPS, max(math.sqrt(3.0 * room), 0.3))
            stamp += (distance - previous) / speed
            previous = distance
            sample = path.at_distance(distance)
            stamp_us = int(round(stamp * 1e6))
            if previous_us is not None and stamp_us <= previous_us:
                stamp_us = previous_us + 1
            plan.append(
                stamp_us,
                Pose.from_xyz_yaw(
                    sample.x, -sample.y, sample.z, -math.radians(sample.yaw)
                ),
            )
            previous_us = stamp_us
            distance += _STEP_M
        return plan
