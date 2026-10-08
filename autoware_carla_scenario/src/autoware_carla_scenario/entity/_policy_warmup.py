"""A run-up that fills a driver policy's history before the scenario starts.

A policy that reads a history -- a stack of past LiDAR maps, the ego's past
poses -- has none at the first step of a scenario and plans as if the ego had
stood still.  An ego spawned moving brakes on the first plans, and a scenario
that checks its speed from the first second fails on that alone.

The run-up drives the ego onto its spawn pose instead, by rule rather than by
the policy: it is moved back along its lane by as far as the initial speed
covers in ``warmup_s``, keeping its Frenet offset from the lane, and carried
forward, tick by tick, so it arrives at the spawn pose at the initial speed when
the scenario starts.  The policy is asked to plan all the while, and sees the
road it would have come down, but its plans are not applied.  The ego is
placed once, at the start of the run-up, and moved by its velocity from then
on with its physics on, so it is handed over as a car that has been driving.
The runner's init-phase brakes spare it (``moves_while_waiting``).

Nothing else is on the road: every other vehicle and pedestrian is put out of
sight (well below the road, gravity off)
and every traffic light is frozen, so the run-up neither appears in the
policy's history as traffic that is not there when the scenario starts nor
spends the lights' phases before it.  Both are put back as they were.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Tuple

from ..utils.vehicles import release_vehicle

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


logger = logging.getLogger(__name__)

#: Spacing of the path sampled back along the lane.
_PATH_STEP_M: float = 0.5

#: The share of its distance off the path the ego is aimed to close each tick.
_PULL: float = 0.5

#: How far below its pose an actor is put out of sight: beyond any sensor range.
_OUT_OF_SIGHT_M: float = 500.0


def _yaw_gap(a_deg: float, b_deg: float) -> float:
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


class PolicyWarmup:
    """The ego's run-up onto its spawn pose; see the module docstring.

    Args:
        world: The CARLA world.
        actor: The ego actor, standing on its spawn pose.
        carla_map: The world's map.
        duration_s: Length of the run-up.
        speed_mps: The speed the ego is to arrive at.
        step_s: The simulation step.
    """

    def __init__(
        self,
        world: "carla.World",
        actor: "carla.Actor",
        carla_map: "carla.Map",
        duration_s: float,
        speed_mps: float,
        step_s: float,
    ) -> None:
        self._actor = actor
        self._speed = max(0.0, speed_mps)
        self._step_s = step_s
        self._ticks = max(1, int(round(duration_s / step_s)))
        self._tick = 0
        self._start = actor.get_transform()
        self._path = self._lane_behind(carla_map, self._speed * self._ticks * step_s)
        self._hidden = self._hide_others(world, actor)
        self._frozen = self._freeze_lights(world)
        self._place(0)
        logger.info(
            "Policy warm-up: %.1f s run-up of %.1f m onto the spawn pose at %.1f m/s; "
            "%d other actor(s) out of sight, %d light(s) frozen",
            self._ticks * step_s,
            self._path[-1][0],
            self._speed,
            len(self._hidden),
            len(self._frozen),
        )

    @property
    def done(self) -> bool:
        """Whether the ego has arrived on its spawn pose."""
        return self._tick >= self._ticks

    # -- what the ego reports during the run-up ------------------------------

    def velocity(self) -> "carla.Vector3D":
        import typesafe_carla.carla as carla  # noqa: PLC0415

        yaw = math.radians(self._actor.get_transform().rotation.yaw)
        return carla.Vector3D(
            x=self._speed * math.cos(yaw), y=self._speed * math.sin(yaw), z=0.0
        )

    def angular_velocity(self) -> "carla.Vector3D":
        """The yaw rate the path turns at, in degrees per second as CARLA reports it."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        here = self._back_m(self._tick)
        ahead = self._back_m(self._tick + 1)
        gap = (
            self._at(ahead).rotation.yaw - self._at(here).rotation.yaw + 180.0
        ) % 360.0 - 180.0
        return carla.Vector3D(x=0.0, y=0.0, z=gap / self._step_s)

    def acceleration(self) -> "carla.Vector3D":
        import typesafe_carla.carla as carla  # noqa: PLC0415

        return carla.Vector3D(x=0.0, y=0.0, z=0.0)

    # -- stepping ---------------------------------------------------------------

    def advance(self) -> None:
        """Place the ego where it is one tick later; on arrival, hand it back."""
        self._tick += 1
        self._place(self._tick)
        if self.done:
            self.restore()

    def _place(self, tick: int) -> None:
        """Move the ego towards where the path has it one tick after ``tick``.

        Placed once, at the start; after that the ego is moved by its velocity
        alone, aimed each tick at where the path has it on the next.
        """
        import typesafe_carla.carla as carla  # noqa: PLC0415

        if tick == 0:
            here = self._pose(0)
            self._actor.set_transform(here)
        else:
            here = self._actor.get_transform()
        if tick < self._ticks:
            # The path's own step, plus half the way back onto the path: aiming
            # the whole error at one tick overshoots, tick after tick.
            due, target = self._at(self._back_m(tick)), self._at(self._back_m(tick + 1))
            velocity = carla.Vector3D(
                x=(target.location.x - due.location.x + _PULL * (due.location.x - here.location.x))
                / self._step_s,
                y=(target.location.y - due.location.y + _PULL * (due.location.y - here.location.y))
                / self._step_s,
                z=0.0,
            )
            turn = (target.rotation.yaw - here.rotation.yaw + 180.0) % 360.0 - 180.0
        else:  # arrived: carry on at the speed the scenario starts at
            yaw = math.radians(self._start.rotation.yaw)
            velocity = carla.Vector3D(
                x=self._speed * math.cos(yaw), y=self._speed * math.sin(yaw), z=0.0
            )
            turn = 0.0
        self._actor.set_target_velocity(velocity)
        self._actor.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=turn / self._step_s))
        release_vehicle(self._actor)

    def restore(self) -> None:
        """Put the other actors and the lights back as they were."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        for actor, transform in self._hidden:
            try:
                actor.set_transform(transform)
                actor.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
                actor.set_enable_gravity(True)
            except RuntimeError:  # destroyed meanwhile
                logger.debug("actor %d is gone", actor.id)
        self._hidden = []
        for light in self._frozen:
            light.freeze(False)
        self._frozen = []

    # -- the path ---------------------------------------------------------------

    def _back_m(self, tick: int) -> float:
        """How far behind the spawn pose the ego is at ``tick``."""
        return self._speed * (self._ticks - tick) * self._step_s

    def _pose(self, tick: int) -> "carla.Transform":
        return self._at(self._back_m(tick))

    def _lane_behind(
        self, carla_map: "carla.Map", length_m: float
    ) -> List[Tuple[float, "carla.Transform"]]:
        """``(distance back, transform)`` along the ego's lane, nearest first.

        The spawn pose's Frenet offset from its lane is kept all the way back:
        its distance to the side of the centre line (along each sample's own
        right-hand direction, so it holds through a curve), its heading
        relative to the lane, and its height above the road.  At a split, the
        predecessor heading most like the lane walked so far is taken.
        """
        import typesafe_carla.carla as carla  # noqa: PLC0415

        start = self._start
        waypoint = carla_map.get_waypoint(start.location, project_to_road=True)
        centre, lane_yaw = waypoint.transform.location, waypoint.transform.rotation.yaw
        right = math.radians(lane_yaw + 90.0)
        lateral = (start.location.x - centre.x) * math.cos(right) + (
            start.location.y - centre.y
        ) * math.sin(right)
        height = start.location.z - centre.z
        heading = (start.rotation.yaw - lane_yaw + 180.0) % 360.0 - 180.0
        path: List[Tuple[float, "carla.Transform"]] = [(0.0, start)]
        travelled = 0.0
        while travelled < length_m:
            behind = waypoint.previous(_PATH_STEP_M)
            if not behind:
                logger.warning(
                    "Policy warm-up: the lane ends %.1f m behind the spawn pose, short of "
                    "the %.1f m run-up; the ego waits there until it is due to leave",
                    travelled,
                    length_m,
                )
                break
            yaw = waypoint.transform.rotation.yaw
            waypoint = min(behind, key=lambda w: _yaw_gap(w.transform.rotation.yaw, yaw))
            travelled += _PATH_STEP_M
            at = waypoint.transform
            right = math.radians(at.rotation.yaw + 90.0)
            path.append(
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
        return path

    def _at(self, back_m: float) -> "carla.Transform":
        """The pose ``back_m`` behind the spawn pose, between two path samples."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        back_m = min(max(back_m, 0.0), self._path[-1][0])
        index = min(int(back_m / _PATH_STEP_M), len(self._path) - 2) if len(self._path) > 1 else 0
        if len(self._path) == 1:
            return self._path[0][1]
        (d0, a), (d1, b) = self._path[index], self._path[index + 1]
        t = (back_m - d0) / (d1 - d0)
        yaw_gap = (b.rotation.yaw - a.rotation.yaw + 180.0) % 360.0 - 180.0
        return carla.Transform(
            carla.Location(
                x=a.location.x + t * (b.location.x - a.location.x),
                y=a.location.y + t * (b.location.y - a.location.y),
                z=a.location.z + t * (b.location.z - a.location.z),
            ),
            carla.Rotation(
                pitch=a.rotation.pitch + t * (b.rotation.pitch - a.rotation.pitch),
                yaw=a.rotation.yaw + t * yaw_gap,
                roll=a.rotation.roll + t * (b.rotation.roll - a.rotation.roll),
            ),
        )

    # -- the rest of the world ---------------------------------------------------

    @staticmethod
    def _hide_others(
        world: "carla.World", ego: "carla.Actor"
    ) -> List[Tuple["carla.Actor", "carla.Transform"]]:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        hidden = []
        actors = world.get_actors()
        for pattern in ("vehicle.*", "walker.*"):
            for actor in actors.filter(pattern):
                if actor.id == ego.id:
                    continue
                transform = actor.get_transform()
                actor.set_enable_gravity(False)
                actor.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
                actor.set_transform(
                    carla.Transform(
                        carla.Location(
                            x=transform.location.x,
                            y=transform.location.y,
                            z=transform.location.z - _OUT_OF_SIGHT_M,
                        ),
                        transform.rotation,
                    )
                )
                hidden.append((actor, transform))
        return hidden

    @staticmethod
    def _freeze_lights(world: "carla.World") -> List["carla.Actor"]:
        frozen = []
        for light in world.get_actors().filter("traffic.traffic_light*"):
            if not light.is_frozen():
                light.freeze(True)
                frozen.append(light)
        return frozen
