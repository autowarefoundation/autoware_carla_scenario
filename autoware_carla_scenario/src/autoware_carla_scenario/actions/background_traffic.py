"""Background traffic: vehicles that come and go without a scenario entity behind them.

Modelled on OpenSCENARIO's ``TrafficSourceAction`` and ``TrafficSinkAction``,
with one change: where OpenSCENARIO places a source or a sink at a *position*
with a *radius*, these take **lanelet constraints** -- the vocabulary a sweep
picks its cases with (``sweep.constraints``) -- so the same action describes
"every lane outside a junction" or "the dead ends of the map" on any map::

    TrafficSourceAction(
        [{"type": "not", "constraint": {"type": "is_junction"}}],
        initial_vehicles=20,
        vehicles_per_minute=6.0,
    )

Who drives the vehicles is the run's traffic backend
(:meth:`~autoware_carla_scenario.traffic.TrafficBackend.spawn_background`):
under ``traffic=sumo`` they are SUMO's own traffic, mirrored into CARLA; under
``traffic=traffic_manager`` they are CARLA vehicles on autopilot.

**When they act.**  Both are ordinary actions, so a trigger decides when they
start and ``until`` when they stop.  Registered with ``register_init`` a source
places its ``initial_vehicles`` during the scenario's initialization -- before
SUMO's warm-up, which spreads them along their routes -- and stops there,
because the initialization phase runs once.  Registered with
``register_pre_tick`` it keeps the rate going through the run until ``until``
fires, e.g. ``until=ElapsedTimeCondition(10.0, label=...)`` for "only during
the first ten seconds".  A sink is the same: it removes whatever background
vehicle is on its lanelets on every tick it runs.
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from ..conditions import BaseCondition, ScenarioResult
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

    from ..traffic import TrafficBackend

logger = logging.getLogger(__name__)

__all__ = [
    "LaneletRegion",
    "TrafficSinkAction",
    "TrafficSourceAction",
    "constraints_from_text",
]

#: A spawn point is kept this far from either end of its lanelet, metres.
_END_MARGIN_M = 3.0
#: How many places a source tries for one vehicle before giving up this tick.
_SPAWN_ATTEMPTS = 8

ConstraintSpec = Union[Mapping[str, Any], Any]


def constraints_from_text(text: str) -> list[dict[str, Any]]:
    """Sweep constraints written as YAML (or JSON): a list, or a single mapping.

    How the scenario editor hands a source or sink its constraints -- the same
    text a sweep's ``constraints:`` holds.  Each is parsed once here, so a
    constraint the sweeper does not know is refused when the scenario is built
    rather than when the first vehicle is due.
    """
    import yaml  # noqa: PLC0415

    from ..sweeper.constraints import parse_constraint  # noqa: PLC0415

    loaded = yaml.safe_load(text) if text and text.strip() else None
    if isinstance(loaded, Mapping):
        loaded = [loaded]
    if not isinstance(loaded, list) or not loaded:
        raise ValueError(f"expected a list of lanelet constraints, got {text!r}")
    constraints = [dict(item) for item in loaded]
    for item in constraints:
        parse_constraint(item)
    return constraints


class LaneletRegion:
    """The lanelets a list of sweep constraints matches on the run's map.

    Resolved on first use, against the map the run loaded, so an action can be
    built before the map is -- from a Hydra config, for instance.
    """

    def __init__(self, constraints: Sequence[ConstraintSpec]) -> None:
        if not constraints:
            raise ValueError("a lanelet region needs at least one constraint")
        self._constraints = list(constraints)
        self._ids: Optional[list[int]] = None

    @property
    def lanelet_ids(self) -> list[int]:
        """The matching lanelet IDs, sorted."""
        if self._ids is None:
            from ..coordinate.map_manager import MapManager  # noqa: PLC0415
            from ..sweeper.constraints import (  # noqa: PLC0415
                find_matching_lanelets,
                parse_constraint,
            )

            parsed = [
                parse_constraint(dict(c)) if isinstance(c, Mapping) else c
                for c in self._constraints
            ]
            manager = MapManager.get_instance()
            self._ids = find_matching_lanelets(
                parsed, manager.lanelet_map, manager.routing_graph
            )
            logger.info("Background traffic region: %d lanelet(s)", len(self._ids))
        return self._ids

    def contains(self, x: float, y: float) -> bool:
        """Whether the CARLA position ``(x, y)`` lies on one of the lanelets."""
        from ..coordinate import CarlaWorldPose, on_lanelet  # noqa: PLC0415

        pose = CarlaWorldPose(x=x, y=y, z=0.0, yaw=0.0)
        return any(on_lanelet(pose, lanelet_id) for lanelet_id in self.lanelet_ids)


class _Never(BaseCondition):
    """Never fires: a source or sink with no ``until`` runs for the whole run."""

    def __init__(self) -> None:
        super().__init__(label="until_the_run_ends")

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        del world, elapsed
        return None


class _BackendAction(BaseAction):
    """An action that acts through the run's traffic backend.

    Like OpenSCENARIO's traffic actions it lasts until it is stopped: with no
    *until* it runs for the rest of the run, where an ordinary action with
    none is over on the tick after it fired.
    """

    REISSUES_BY_DEFAULT = True

    def __init__(
        self, label: str, *, until: Optional[BaseCondition] = None, **kwargs: Any
    ) -> None:
        super().__init__(
            label, until=until if until is not None else _Never(), **kwargs
        )
        self._backend: Optional["TrafficBackend"] = None

    def set_traffic_backend(self, backend: "TrafficBackend") -> None:
        """Inject the run's traffic backend; the scenario does this on registration."""
        self._backend = backend

    def _require_backend(self) -> Optional["TrafficBackend"]:
        if self._backend is None:
            logger.warning(
                "%s '%s': no traffic backend; register it on a scenario",
                type(self).__name__,
                self.label,
            )
        return self._backend


class TrafficSourceAction(_BackendAction):
    """Create background vehicles on the lanelets *constraints* match.

    Args:
        constraints: Sweep constraints, as YAML mappings or parsed constraints.
            Every one has to hold (an implicit ``and``).
        initial_vehicles: Vehicles placed at once on the first tick the action
            runs.
        vehicles_per_minute: Then, the rate at which more are added while it
            runs, in simulated time.  ``0`` places only the initial ones.
        max_vehicles: The most background vehicles on the road at once -- this
            source's and any other's -- past which it adds none; ``None`` for
            no limit.
        speed_kmh: The speed each drives at, at most: the traffic model may
            hold it below that (a queue, a bend, a red light).
        min_gap_m: The least distance from a new vehicle to any other one.
        blueprint: CARLA blueprint, or ``None`` for the backend's own.
        seed: Seeds the choice of lanelet and position, for a reproducible run.
        label, condition, timing, once, until, reissue: as for
            :class:`~autoware_carla_scenario.actions.base.BaseAction`, except
            that with no *until* it runs until the scenario ends.  It reissues
            by default: each tick of its run is a tick it may add a vehicle on.
    """

    def __init__(
        self,
        constraints: Sequence[ConstraintSpec],
        *,
        initial_vehicles: int = 0,
        vehicles_per_minute: float = 0.0,
        max_vehicles: Optional[int] = None,
        speed_kmh: float = 30.0,
        min_gap_m: float = 15.0,
        blueprint: Optional[str] = None,
        seed: int = 0,
        label: str = "traffic_source",
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        once: bool = True,
        until: Optional[BaseCondition] = None,
        reissue: Optional[bool] = None,
    ) -> None:
        if initial_vehicles < 0 or vehicles_per_minute < 0:
            raise ValueError("vehicle counts and rates must not be negative")
        if max_vehicles is not None and max_vehicles < 0:
            raise ValueError("max_vehicles must not be negative")
        super().__init__(
            label,
            condition=condition,
            timing=timing,
            once=once,
            until=until,
            reissue=reissue,
        )
        self._region = LaneletRegion(constraints)
        self._initial = initial_vehicles
        self._rate = vehicles_per_minute / 60.0
        self._max = max_vehicles
        self._speed_kmh = speed_kmh
        self._min_gap = min_gap_m
        self._blueprint = blueprint
        self._rng = random.Random(seed)
        self._handles: set[str] = set()
        self._started = False
        self._last_time: Optional[float] = None
        self._owed = 0.0

    @property
    def spawned(self) -> int:
        """How many vehicles this source has created so far."""
        return len(self._handles)

    def execute(self, world: "carla.World") -> None:
        """Add the vehicles owed since the last tick, up to the limit."""
        backend = self._require_backend()
        if backend is None:
            return
        now = _simulated_seconds(world)
        if not self._started:
            self._started = True
            self._last_time = now
            due = self._initial
        else:
            if self._last_time is not None and now is not None:
                self._owed += self._rate * max(now - self._last_time, 0.0)
            self._last_time = now
            due = int(self._owed)
            self._owed -= due
        if due <= 0:
            return
        live = backend.background_vehicles(world)
        if self._max is not None:
            due = min(due, max(self._max - len(live), 0))
        occupied = [*live.values(), *_vehicle_positions(world)]
        added = 0
        for _ in range(due):
            handle = self._spawn_one(world, backend, occupied)
            if handle is None:
                continue
            self._handles.add(handle)
            added += 1
        if added:
            logger.info("%s: %d background vehicle(s) added", self.label, added)

    def _spawn_one(
        self,
        world: "carla.World",
        backend: "TrafficBackend",
        occupied: list[tuple[float, float]],
    ) -> Optional[str]:
        from ..coordinate import Lanelet2Pose, lanelet_length, to_carla_world  # noqa: PLC0415

        ids = self._region.lanelet_ids
        if not ids:
            return None
        for _ in range(_SPAWN_ATTEMPTS):
            lanelet_id = self._rng.choice(ids)
            length = lanelet_length(lanelet_id)
            if length > 2 * _END_MARGIN_M:
                s = self._rng.uniform(_END_MARGIN_M, length - _END_MARGIN_M)
            else:
                s = length / 2
            pose = to_carla_world(Lanelet2Pose(lanelet_id=lanelet_id, s=s))
            if any(math.dist((pose.x, pose.y), p) < self._min_gap for p in occupied):
                continue
            handle = backend.spawn_background(
                world,
                pose.to_carla_transform(),
                speed_kmh=self._speed_kmh,
                blueprint=self._blueprint,
            )
            if handle is not None:
                occupied.append((pose.x, pose.y))
                return handle
        return None


class TrafficSinkAction(_BackendAction):
    """Remove the background vehicles that are on the lanelets *constraints* match.

    Every background vehicle the run's backend has -- whichever source made it --
    is checked on each tick the sink runs.

    Args:
        constraints: Sweep constraints, as for :class:`TrafficSourceAction`.
        label, condition, timing, once, until, reissue: as for
            :class:`~autoware_carla_scenario.actions.base.BaseAction`, except
            that with no *until* it runs until the scenario ends.
    """

    def __init__(
        self,
        constraints: Sequence[ConstraintSpec],
        *,
        label: str = "traffic_sink",
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        once: bool = True,
        until: Optional[BaseCondition] = None,
        reissue: Optional[bool] = None,
    ) -> None:
        super().__init__(
            label,
            condition=condition,
            timing=timing,
            once=once,
            until=until,
            reissue=reissue,
        )
        self._region = LaneletRegion(constraints)
        self.removed = 0

    def execute(self, world: "carla.World") -> None:
        """Take every background vehicle on the region off the road."""
        backend = self._require_backend()
        if backend is None:
            return
        for handle, (x, y) in backend.background_vehicles(world).items():
            if self._region.contains(x, y):
                backend.remove_background(world, handle)
                self.removed += 1


def _simulated_seconds(world: Any) -> Optional[float]:
    try:
        return float(world.get_snapshot().timestamp.elapsed_seconds)
    except Exception:  # noqa: BLE001 - a fake or a closing world
        return None


def _vehicle_positions(world: Any) -> list[tuple[float, float]]:
    try:
        actors = world.get_actors().filter("vehicle.*")
    except Exception:  # noqa: BLE001
        return []
    out = []
    for actor in actors:
        try:
            location = actor.get_location()
        except Exception:  # noqa: BLE001
            continue
        out.append((location.x, location.y))
    return out
