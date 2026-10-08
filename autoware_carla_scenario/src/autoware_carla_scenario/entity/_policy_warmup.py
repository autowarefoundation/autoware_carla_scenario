"""A run-up that fills a driver policy's history before the scenario starts.

A policy that reads a history -- a stack of past LiDAR maps, the ego's past
poses -- has none at the first step of a scenario and plans as if the ego had
stood still.  An ego spawned moving brakes on the first plans, and a scenario
that checks its speed from the first second fails on that alone.

The run-up plays the ``warmup_s`` before the scenario's first frame instead,
by rule rather than by the policy, so that every actor arrives at its
first-frame pose at its initial speed:

* **Vehicles** (the ego among them) are moved back along their lane by as far
  as their initial speed covers in ``warmup_s``, keeping their Frenet offset
  from the lane, and carried forward onto their pose.  Each is placed once and
  then moved by its velocity, set each tick and aimed at the path's next point,
  its physics left on, so it is handed over as a car that has been driving.  It
  is driven meanwhile on the throttle that holds its speed, by its powertrain
  model, so it arrives in the gear and at the engine speed it would be in; carried
  on a released throttle it would arrive in neutral, the engine idling.  A
  vehicle with no initial speed stands on its pose, held on its brakes like
  any vehicle in the init phase.
* **Pedestrians** are moved in a straight line, in world coordinates, along
  their heading at their initial speed, placed afresh each tick; one with no
  initial speed stands where it starts.

The policy is asked to plan all the while, and sees the road and the traffic
it would have come through, but its plans are not applied.  The traffic lights
are frozen meanwhile, so the run-up spends none of their phases, and thawed on
arrival.  The runner's init-phase brakes spare what the run-up carries
(``carried_actor_ids``).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, FrozenSet, List, Mapping, Tuple

from ..utils.powertrain import ChaosPowertrain
from ..utils.vehicles import release_vehicle

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


logger = logging.getLogger(__name__)

#: Spacing of the path sampled back along a lane.
_PATH_STEP_M: float = 0.5

#: The share of its distance off the path a vehicle is aimed to close each tick.
_PULL: float = 0.5


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class _LanePath:
    """Poses back along a vehicle's lane from its first-frame pose.

    The pose's Frenet offset from its lane is kept all the way back: its
    distance to the side of the centre line (along each sample's own
    right-hand direction, so it holds through a curve), its heading relative to
    the lane, and its height above the road.  At a split, the predecessor
    heading most like the lane walked so far is taken.
    """

    def __init__(self, carla_map: "carla.Map", start: "carla.Transform", length_m: float, name: str) -> None:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        waypoint = carla_map.get_waypoint(start.location, project_to_road=True)
        centre, lane_yaw = waypoint.transform.location, waypoint.transform.rotation.yaw
        right = math.radians(lane_yaw + 90.0)
        lateral = (start.location.x - centre.x) * math.cos(right) + (
            start.location.y - centre.y
        ) * math.sin(right)
        height = start.location.z - centre.z
        heading = _wrap_deg(start.rotation.yaw - lane_yaw)
        self._samples: List[Tuple[float, "carla.Transform"]] = [(0.0, start)]
        travelled = 0.0
        while travelled < length_m:
            behind = waypoint.previous(_PATH_STEP_M)
            if not behind:
                logger.warning(
                    "Policy warm-up: %s's lane ends %.1f m behind its first-frame pose, "
                    "short of its %.1f m run-up; it waits there until it is due to leave",
                    name,
                    travelled,
                    length_m,
                )
                break
            yaw = waypoint.transform.rotation.yaw
            waypoint = min(behind, key=lambda w: abs(_wrap_deg(w.transform.rotation.yaw - yaw)))
            travelled += _PATH_STEP_M
            at = waypoint.transform
            right = math.radians(at.rotation.yaw + 90.0)
            self._samples.append(
                (
                    travelled,
                    carla.Transform(
                        carla.Location(
                            x=at.location.x + lateral * math.cos(right),
                            y=at.location.y + lateral * math.sin(right),
                            z=at.location.z + height,
                        ),
                        carla.Rotation(
                            pitch=at.rotation.pitch,
                            yaw=at.rotation.yaw + heading,
                            roll=start.rotation.roll,
                        ),
                    ),
                )
            )

    @property
    def length_m(self) -> float:
        return self._samples[-1][0]

    def at(self, back_m: float) -> "carla.Transform":
        """The pose ``back_m`` behind the first-frame pose, between two samples."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        if len(self._samples) == 1:
            return self._samples[0][1]
        back_m = min(max(back_m, 0.0), self.length_m)
        index = min(int(back_m / _PATH_STEP_M), len(self._samples) - 2)
        (d0, a), (d1, b) = self._samples[index], self._samples[index + 1]
        t = (back_m - d0) / (d1 - d0)
        return carla.Transform(
            carla.Location(
                x=a.location.x + t * (b.location.x - a.location.x),
                y=a.location.y + t * (b.location.y - a.location.y),
                z=a.location.z + t * (b.location.z - a.location.z),
            ),
            carla.Rotation(
                pitch=a.rotation.pitch + t * (b.rotation.pitch - a.rotation.pitch),
                yaw=a.rotation.yaw + t * _wrap_deg(b.rotation.yaw - a.rotation.yaw),
                roll=a.rotation.roll + t * (b.rotation.roll - a.rotation.roll),
            ),
        )


class _CarriedVehicle:
    """A vehicle driven onto its first-frame pose along its lane."""

    def __init__(
        self, carla_map: "carla.Map", actor: "carla.Actor", speed_mps: float, ticks: int, step_s: float
    ) -> None:
        self.actor = actor
        self.speed = speed_mps
        self._ticks = ticks
        self._step_s = step_s
        self.start = actor.get_transform()
        self.path = _LanePath(carla_map, self.start, speed_mps * ticks * step_s, actor.type_id)
        self._powertrain = ChaosPowertrain.from_physics_control(actor.get_physics_control())

    def back_m(self, tick: int) -> float:
        """How far behind its first-frame pose the vehicle is at ``tick``."""
        return self.speed * (self._ticks - tick) * self._step_s

    def place(self, tick: int) -> None:
        """Move the vehicle towards where its path has it one tick after ``tick``.

        Placed once, at the start; after that it is moved by its velocity alone,
        aimed each tick at where the path has it on the next.
        """
        import typesafe_carla.carla as carla  # noqa: PLC0415

        if tick == 0:
            here = self.path.at(self.back_m(0))
            self.actor.set_transform(here)
        else:
            here = self.actor.get_transform()
        if tick < self._ticks:
            # The path's own step, plus half the way back onto the path: aiming
            # the whole error at one tick overshoots, tick after tick.
            due, target = self.path.at(self.back_m(tick)), self.path.at(self.back_m(tick + 1))
            velocity = carla.Vector3D(
                x=(target.location.x - due.location.x + _PULL * (due.location.x - here.location.x))
                / self._step_s,
                y=(target.location.y - due.location.y + _PULL * (due.location.y - here.location.y))
                / self._step_s,
                z=self.actor.get_velocity().z,
            )
            turn = _wrap_deg(target.rotation.yaw - here.rotation.yaw)
        else:  # arrived: carry on at the speed the scenario starts it at
            yaw = math.radians(self.start.rotation.yaw)
            velocity = carla.Vector3D(
                x=self.speed * math.cos(yaw),
                y=self.speed * math.sin(yaw),
                z=self.actor.get_velocity().z,
            )
            turn = 0.0
        # Vertically, and in pitch and roll, the vehicle is left to its suspension:
        # holding those at zero tick after tick keeps the springs off their rest,
        # and at the handover they bounce the car, which a policy reading its
        # acceleration back takes for braking.
        spin = self.actor.get_angular_velocity()
        self.actor.set_target_velocity(velocity)
        self.actor.set_target_angular_velocity(
            carla.Vector3D(x=spin.x, y=spin.y, z=turn / self._step_s)
        )
        throttle, _ = self._powertrain.pedals(0.0, self.speed, int(self.actor.get_control().gear))
        release_vehicle(self.actor, throttle)


class _CarriedWalker:
    """A pedestrian moved in a straight line onto its first-frame pose."""

    def __init__(self, actor: "carla.Actor", speed_mps: float, ticks: int, step_s: float) -> None:
        self.actor = actor
        self._ticks = ticks
        self._step_s = step_s
        self.start = actor.get_transform()
        yaw = math.radians(self.start.rotation.yaw)
        self._vx, self._vy = speed_mps * math.cos(yaw), speed_mps * math.sin(yaw)

    def place(self, tick: int) -> None:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        back_s = (self._ticks - tick) * self._step_s
        at = self.start.location
        self.actor.set_transform(
            carla.Transform(
                carla.Location(x=at.x - self._vx * back_s, y=at.y - self._vy * back_s, z=at.z),
                self.start.rotation,
            )
        )


class PolicyWarmup:
    """The run-up onto the scenario's first frame; see the module docstring.

    Args:
        world: The CARLA world, every actor on its first-frame pose.
        ego: The ego actor.
        carla_map: The world's map.
        duration_s: Length of the run-up.
        initial_speeds: The speed each actor starts the scenario at, by actor
            id, in metres per second; an actor not named starts still.
        step_s: The simulation step.
    """

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Actor",
        carla_map: "carla.Map",
        duration_s: float,
        initial_speeds: Mapping[int, float],
        step_s: float,
    ) -> None:
        self._ticks = max(1, int(round(duration_s / step_s)))
        self._tick = 0
        self._step_s = step_s
        actors = world.get_actors()
        self._ego = _CarriedVehicle(
            carla_map, ego, max(0.0, initial_speeds.get(ego.id, 0.0)), self._ticks, step_s
        )
        self._vehicles: List[_CarriedVehicle] = [self._ego]
        for actor in actors.filter("vehicle.*"):
            speed = initial_speeds.get(actor.id, 0.0)
            if actor.id != ego.id and speed > 0.0:
                self._vehicles.append(_CarriedVehicle(carla_map, actor, speed, self._ticks, step_s))
        self._walkers: List[_CarriedWalker] = [
            _CarriedWalker(actor, speed, self._ticks, step_s)
            for actor in actors.filter("walker.pedestrian.*")
            if (speed := initial_speeds.get(actor.id, 0.0)) > 0.0
        ]
        self._frozen = self._freeze_lights(world)
        self._place(0)
        logger.info(
            "Policy warm-up: %.1f s run-up onto the first frame; the ego from %.1f m back "
            "at %.1f m/s, %d other vehicle(s) and %d pedestrian(s) moving with it; "
            "%d light(s) frozen",
            self._ticks * step_s,
            self._ego.path.length_m,
            self._ego.speed,
            len(self._vehicles) - 1,
            len(self._walkers),
            len(self._frozen),
        )

    @property
    def done(self) -> bool:
        """Whether every actor has arrived on its first-frame pose."""
        return self._tick >= self._ticks

    @property
    def carried_actor_ids(self) -> FrozenSet[int]:
        """The vehicles the run-up is moving, which the init-phase brakes must spare."""
        return frozenset(v.actor.id for v in self._vehicles if v.speed > 0.0)

    # -- what the ego reports during the run-up ------------------------------

    def velocity(self) -> "carla.Vector3D":
        import typesafe_carla.carla as carla  # noqa: PLC0415

        yaw = math.radians(self._ego.actor.get_transform().rotation.yaw)
        speed = self._ego.speed
        return carla.Vector3D(x=speed * math.cos(yaw), y=speed * math.sin(yaw), z=0.0)

    def angular_velocity(self) -> "carla.Vector3D":
        """The yaw rate the ego's path turns at, in degrees per second as CARLA reports it."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        here = self._ego.path.at(self._ego.back_m(self._tick))
        ahead = self._ego.path.at(self._ego.back_m(self._tick + 1))
        gap = _wrap_deg(ahead.rotation.yaw - here.rotation.yaw)
        return carla.Vector3D(x=0.0, y=0.0, z=gap / self._step_s)

    def acceleration(self) -> "carla.Vector3D":
        import typesafe_carla.carla as carla  # noqa: PLC0415

        return carla.Vector3D(x=0.0, y=0.0, z=0.0)

    # -- stepping ---------------------------------------------------------------

    def advance(self) -> None:
        """Move every carried actor on by one tick; on arrival, thaw the lights."""
        self._tick += 1
        self._place(self._tick)
        if self.done:
            self.restore()

    def _place(self, tick: int) -> None:
        for vehicle in self._vehicles:
            try:
                vehicle.place(tick)
            except RuntimeError:  # destroyed meanwhile
                logger.debug("vehicle %d is gone", vehicle.actor.id)
        for walker in self._walkers:
            try:
                walker.place(tick)
            except RuntimeError:
                logger.debug("pedestrian %d is gone", walker.actor.id)

    def restore(self) -> None:
        """Thaw the lights the run-up froze."""
        for light in self._frozen:
            light.freeze(False)
        self._frozen = []

    @staticmethod
    def _freeze_lights(world: "carla.World") -> List["carla.Actor"]:
        frozen = []
        for light in world.get_actors().filter("traffic.traffic_light*"):
            if not light.is_frozen():
                light.freeze(True)
                frozen.append(light)
        return frozen
