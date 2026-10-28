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

import logging
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
#: The room (m) a vehicle driven out of its junction keeps to each side of it
#: for a walker: less than the PASSING_CLEARANCE_M a vehicle the framework
#: drives keeps (actions._road_threat), as it drives out at no more than
#: APPROACH_SPEED_MPS -- with that, vehicle_58 waited at a walker standing
#: 2.3 m (centre to centre) off its side, out of its way.
LEAD_OUT_WALKER_CLEARANCE_M: float = 0.5
#: The pieces (m) the path ahead is looked along for something in the way.
_IN_WAY_STEP_M: float = 1.0
#: The room (m) a vehicle driven out of its junction keeps ahead of it for a
#: walker, beyond its stopping distance: less than the 4 m a vehicle the
#: framework drives keeps (actions._trajectory_run._walker_in_way), as it
#: drives out at a crawl at first.
LEAD_OUT_WALKER_AHEAD_M: float = 2.0
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


logger = logging.getLogger(__name__)

#: The room (m) kept to a vehicle's side: the recording's own, passing a bus's
#: corner, was 0.25 m (vehicle_96).
_VEHICLE_CLEARANCE_M = 0.15
#: A vehicle given more throttle than this (0..1) that makes less way than
#: this (m) along its path this long (s) is wedged, and given full throttle.
_WEDGED_THROTTLE = 0.2
_WEDGED_PROGRESS_M = 0.3
_WEDGED_S = 1.5
#: How long (s) more full throttle is tried before it stops where it is.
_CLIMB_S = 3.0
#: A sideways step (m) between two points of a path this long or longer is a
#: crossing to the lane beside.
LANE_STEP_M: float = 2.0
#: A step (m) longer than this is no lane change: a gap a junction left.
_LANE_CHANGE_SPAN_M: float = 12.0
#: How near parallel (cosine, 15 deg) the path goes on after a lane change.
_PARALLEL_COS: float = math.cos(math.radians(15.0))


def lane_changes(xs: List[float], ys: List[float]) -> List[tuple[int, float]]:
    """Where the path through ``(xs, ys)`` crosses to the lane beside: the
    index of each point it steps sideways from by :data:`LANE_STEP_M` or
    more, against the way it was going, no longer a step than
    :data:`_LANE_CHANGE_SPAN_M`, and going on parallel to the way it came
    (not a turn, nor the gap a junction left in it) -- with how far across
    (m; positive to the left of the way in a right-handed frame, to the
    right in CARLA's left-handed one)."""
    out: List[tuple[int, float]] = []
    for i in range(1, len(xs) - 2):
        hx, hy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
        span = math.hypot(hx, hy)
        if span < 1e-6:
            continue
        hx, hy = hx / span, hy / span
        sx, sy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        if math.hypot(sx, sy) > _LANE_CHANGE_SPAN_M:
            continue
        across = hx * sy - hy * sx
        if abs(across) < LANE_STEP_M:
            continue
        ox, oy = xs[i + 2] - xs[i + 1], ys[i + 2] - ys[i + 1]
        on = math.hypot(ox, oy)
        if on < 1e-6 or (hx * ox + hy * oy) / on < _PARALLEL_COS:
            continue
        out.append((i, across))
    return out


class FinalApproach:
    """The rest of a given path, driven to its end and stopped there.

    Args:
        actor: The vehicle, off autopilot.
        start: Where it is now.
        points: The path's points still ahead of it, the last one its end.
        stop: Whether it stops on the path's end (a final approach), or is
            only done there, at speed, for the TrafficManager to take it on
            (a way out of the junction it started in).
        speed: How fast (m/s) it is driven: for a way out of its junction the
            speed its route asks for (:meth:`set_speed`), so a car that pulled
            away from the junction in the recording is not held to a crawl
            while a vehicle beside it drives on; for a final approach
            :data:`APPROACH_SPEED_MPS`, or the speed it came at, slowing onto
            its end.
    """

    _actor: "carla.Actor"
    _path: ResolvedTrajectory
    _follower: "TrajectoryFollower"
    _here: float
    _stop: bool
    _waits_for: int
    _speed: float
    _blocked_s: float
    _stuck_s: float
    _progress_at: float
    _wedged_s: float
    _wedged_at: float
    _climbing: bool
    done: bool

    def __init__(
        self,
        actor: "carla.Actor",
        start: "carla.Location",
        points: List["carla.Location"],
        stop: bool = True,
        speed: float = APPROACH_SPEED_MPS,
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
        #: The actor it last said it waits for (``-1``: none).
        self._waits_for = -1
        self._speed = max(float(speed), 0.1)
        self._blocked_s = 0.0
        self._stuck_s = 0.0
        self._progress_at = 0.0
        self._wedged_s = 0.0
        self._wedged_at = 0.0
        self._climbing = False
        self.done = False

    def set_speed(self, speed: float) -> None:
        """Drive a way out of its junction at *speed* (m/s) from now on."""
        self._speed = max(float(speed), 0.1)

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
        if TYPE_CHECKING:
            pass  # utils.clearance is not compiled (docs/typecheck.md)
        else:
            speed = math.hypot(velocity.x, velocity.y)
            extent = actor.bounding_box.extent
            # Along the path it drives -- a turn out of its junction, say -- not
            # straight ahead, where a walker on the corner it turns away from
            # stood "in its way" (vehicle_56 and pedestrian_66, each waiting on
            # the other): within its stopping distance, as _walker_in_way and
            # _vehicle_in_way measure it.
            walker = self._in_way(
                "walker",
                speed * speed / 8.0 + LEAD_OUT_WALKER_AHEAD_M,
                float(extent.y) + LEAD_OUT_WALKER_CLEARANCE_M,
            )
            if walker or self._in_way(
                "vehicle",
                speed * speed / 8.0 + float(extent.x) + 2.0,
                float(extent.y) + _VEHICLE_CLEARANCE_M,
            ):
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
        control = command.to_carla_control()
        # Throttle on, nothing in its way, and making no way along its path
        # -- grinding at a crawl, maybe: wedged on a kerb its corner swung
        # onto (vehicle_41, an 11 m bus turning where the recording -- SUMO's,
        # with no kerbs -- had it turn).  Full throttle takes it over.
        if control.throttle > _WEDGED_THROTTLE:
            if self._here > self._wedged_at + _WEDGED_PROGRESS_M:
                self._wedged_at = self._here
                self._wedged_s = 0.0
                self._climbing = False
            else:
                self._wedged_s += dt
        else:
            self._wedged_at = self._here
            self._wedged_s = 0.0
        if self._wedged_s >= _WEDGED_S + _CLIMB_S:
            # Not over it at full throttle either: it stops where it is, its
            # way blocked (the recording's own path ran over the kerb).
            self.done = True
            logger.warning(
                "actor %d: wedged %.1f m along its %.1f m path, and not over "
                "it at full throttle; stops there",
                int(actor.id),
                self._here,
                self._path.length,
            )
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return
        if self._wedged_s >= _WEDGED_S:
            if not self._climbing:
                self._climbing = True
                logger.info(
                    "actor %d: wedged %.1f m along its path; full throttle",
                    int(actor.id),
                    self._here,
                )
            control.throttle = 1.0
        vehicle.apply_control(control)

    def _in_way(self, kind: str, reach: float, half_width: float) -> bool:
        """Whether a *kind* (``"vehicle"`` or ``"walker"``) actor stands on the
        path from where the vehicle is to *reach* m on, within *half_width* of
        it (its own size added); straight on past the path's end."""
        path = self._path
        try:
            world = self._actor.get_world()
            own = int(self._actor.id)
            transform = self._actor.get_transform()
            behind = (
                float(transform.location.x),
                float(transform.location.y),
                float(transform.rotation.yaw),
                # Its centre: one whose centre is behind its own is not in
                # its way going on, however near its corner (vehicle_56 at
                # vehicle_58's rear, each waiting on the other).
                0.0,
            )
        except (AttributeError, RuntimeError):
            return False  # no world to ask (a test double)
        try:
            body = float(self._actor.bounding_box.extent.y)
            half_length = float(self._actor.bounding_box.extent.x)
        except (AttributeError, RuntimeError):
            body = 0.0
            half_length = 0.0
        distance = self._here
        end = self._here + reach
        while distance < end - 1e-6:
            on_path = distance < path.length - 1e-6
            span = end - distance
            if on_path:
                span = min(span, _IN_WAY_STEP_M, path.length - distance)
            at = path.at_distance(distance, 1.0)
            # The piece's own direction, end to end: a short first segment
            # (to the route's next point) may point anywhere.
            ahead = path.at_distance(distance + span, 1.0)
            if on_path and math.hypot(ahead.x - at.x, ahead.y - at.y) > 0.5:
                heading = math.degrees(math.atan2(ahead.y - at.y, ahead.x - at.x))
            else:
                heading = math.degrees(math.atan2(at.vy, at.vx))
            # Its corners swing out where it faces across the piece: as wide
            # as its half length times the sine of that (vehicle_134, turning
            # across into the lane beside, caught a lorry's corner).
            # Only along its own front half: further on it faces along its path.
            looked = half_width
            if distance - self._here < half_length:
                swing = abs(math.sin(math.radians(heading - behind[2])))
                looked += swing * half_length
            if TYPE_CHECKING:
                pass  # utils.clearance is not compiled (docs/typecheck.md)
            else:
                from ..actions._trajectory_run import _in_way_ahead  # noqa: PLC0415

                found = _in_way_ahead(
                    world,
                    own,
                    at.x,
                    at.y,
                    heading,
                    kind,
                    span,
                    looked,
                    behind,
                    body,
                )
                if found >= 0:
                    if found != self._waits_for:
                        self._waits_for = found
                        logger.info(
                            "actor %d: led out of its junction, waits for %s %d, "
                            "%.1f m on along its path",
                            own,
                            kind,
                            found,
                            distance - self._here,
                        )
                    return True
            distance += span
        return False

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
            speed = self._speed
            if self._stop:
                speed = min(self._speed, max(math.sqrt(3.0 * room), 0.3))
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
