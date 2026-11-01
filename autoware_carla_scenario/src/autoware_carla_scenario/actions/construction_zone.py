"""TeraSim's construction zone as an action: cones, barriers and signs on a lane (#402).

:class:`ConstructionZoneAction` lays out TeraSim's ``ConstructionAdversity``
or, with ``layout="urban"``, its ``UrbanConstructionAdversity`` on one
intersection approach (:mod:`~..adversity.construction`) along a lane of the Lanelet2 map, from
``start_s`` to ``end_s`` metres along it -- past its lanelet's end, on along
the straightest successors -- and from ``start_time`` to ``end_time``
(scenario seconds; ``None`` for good) it stands the objects there as CARLA
props, each added and removed by a :class:`~.spawn.SpawnAction` (named
``<label>/<zone>/<n>``).  Then, as TeraSim's ``initialize``:

* the traffic backend's background vehicles in the zone are taken off the
  road (``clear_traffic``);
* with ``close_lane`` (TeraSim's ``full_lane``), the backend is told the lane
  is closed (:meth:`~..traffic.base.TrafficBackend.close_lane`: SUMO stops
  routing traffic onto it; the TrafficManager cannot, and says so).

TeraSim's ``full_lane`` places no object, only closes the lane; in CARLA a
closure nobody sees is one Autoware drives into, so the objects are always
placed.  At ``end_time`` the props are removed and the lane reopened; when the
run ends, the lane is reopened and the runner takes the props away with every
other (:func:`~.spawn.clear_props`).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import lanelet2.core
import lanelet2.geometry

from ..adversity.construction import (
    BARRIER,
    CONE,
    DEFAULT_CONSTRUCTION,
    SIGN,
    ConstructionSettings,
    construction_layout,
)
from ..conditions.base import BaseCondition, ScenarioResult
from ..coordinate.map_manager import MapManager, run_carla_map
from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose
from ..coordinate.transform import on_lanelet, project_onto_lanelet, to_carla_world
from ..route.geometry import MapFeatures
from ..traffic.base import TrafficBackend
from ..trajectory.relative_lane import _centreline, _point_at, advance_along_lane
from ..trajectory.resolve import road_height
from .background_traffic import _BackendAction
from .base import BaseAction, TickTiming
from .spawn import SpawnAction, remove_prop

if TYPE_CHECKING:
    import typesafe_carla.carla as carla
    from lanelet2.core import ConstLanelet

__all__ = ["DEFAULT_PROP_BLUEPRINTS", "ConstructionZoneAction"]

logger = logging.getLogger(__name__)

#: The CARLA prop each kind of object is.
DEFAULT_PROP_BLUEPRINTS: dict[str, str] = {
    CONE: "static.prop.constructioncone",
    BARRIER: "static.prop.streetbarrier",
    SIGN: "static.prop.warningconstruction",
}

#: The lane width (m) a lanelet whose bounds cannot be read is taken to be.
_DEFAULT_LANE_WIDTH_M = 3.5


class ConstructionZoneAction(_BackendAction):
    """A construction zone on lanelet *lanelet_id*'s lane, from *start_s* to
    *end_s*, from *start_time* until *end_time*.

    Args:
        lanelet_id: The lanelet the lane is measured from.
        start_s: Where the work starts, metres along it; for an urban
            layout (``settings.layout == "urban"``), where the approach
            starts, the warning signs from there.
        end_s: Where it ends; past the lanelet's end, along the straightest
            successors.  Urban: the intersection the approach ends at.
        settings: TeraSim's layout numbers (zones, spacing, tapers, side,
            and the layout: :data:`~..adversity.construction.URBAN_CONSTRUCTION`
            is TeraSim's ``UrbanConstructionAdversity``).
        start_time: When the zone appears (s from the scenario's start).
        end_time: When it is cleared away; ``None`` never.
        close_lane: Tell the traffic backend the lane is closed (TeraSim's
            ``full_lane``).
        clear_traffic: Take the backend's background vehicles in the zone off
            the road when it appears.
        blueprints: The CARLA prop of each kind of object
            (:data:`DEFAULT_PROP_BLUEPRINTS`).
        condition: Starts the action; at once by default.
        timing: Tick phase.
        label: Human-readable identifier.

    Raises:
        ValueError: *end_s* not after *start_s*, a negative *start_time*, or an
            *end_time* not after it.
    """

    # Declared for the static check (docs/typecheck.md).
    _lanelet_id: int
    _start_s: float
    _end_s: float
    _settings: ConstructionSettings
    _start_time: float
    _end_time: Optional[float]
    _close_lane: bool
    _clear_traffic: bool
    _blueprints: dict[str, str]
    _props: list[str]
    _closed: list[str]
    _placed: bool
    _cleared: bool
    _elapsed: float

    def __init__(
        self,
        lanelet_id: int,
        start_s: float,
        end_s: float,
        settings: ConstructionSettings = DEFAULT_CONSTRUCTION,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        *,
        close_lane: bool = True,
        clear_traffic: bool = True,
        blueprints: Optional[dict[str, str]] = None,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        label: str = "construction_zone",
    ) -> None:
        if not end_s > start_s:
            raise ValueError(f"{label}: end_s must be after start_s")
        if start_time < 0.0:
            raise ValueError(f"{label}: start_time must not be negative")
        if end_time is not None and not end_time > start_time:
            raise ValueError(f"{label}: end_time must be after start_time")
        end: BaseCondition = _Cleared(self, label)
        super().__init__(
            label, condition=condition, timing=timing, once=True, until=end
        )
        self._lanelet_id = int(lanelet_id)
        self._start_s = float(start_s)
        self._end_s = float(end_s)
        self._settings = settings
        self._start_time = float(start_time)
        self._end_time = end_time
        self._close_lane = close_lane
        self._clear_traffic = clear_traffic
        chosen = dict(DEFAULT_PROP_BLUEPRINTS)
        if blueprints is not None:
            chosen.update(blueprints)
        self._blueprints = chosen
        self._props = []
        self._closed = []
        self._placed = False
        self._cleared = False
        self._elapsed = 0.0

    @property
    def needs_tick_loop(self) -> bool:
        """Always: the zone appears at *start_time* and is cleared away at
        *end_time*, which only a tick reaches."""
        return True

    @property
    def cleared(self) -> bool:
        """Whether the zone has come and gone."""
        return self._cleared

    @property
    def prop_count(self) -> int:
        """How many of the zone's props stand in the world now."""
        return len(self._props)

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Remember the scenario time, which :meth:`execute` is not given."""
        self._elapsed = elapsed
        # Named, not super(): Codon 0.19 cannot dispatch super() two classes up.
        BaseAction.tick(self, world, elapsed)

    def execute(self, world: "carla.World") -> None:
        """Place the zone at its start time, clear it away at its end time."""
        if self._cleared:
            return
        # elapsed is a running sum of the tick, so a time that is a whole
        # number of ticks can come out a hair short of it; without the
        # tolerance the zone would appear (and go) one tick late.
        if not self._placed and self._elapsed >= self._start_time - 1e-9:
            self._place(world)
        end = self._end_time
        if self._placed and end is not None and self._elapsed >= end - 1e-9:
            self._clear(world)

    def close(self) -> None:
        """Reopen the lane if the run ends with the zone standing; the runner
        takes its props away with every other."""
        BaseAction.close(self)  # named, as in tick
        self._props = []
        self._reopen(None)

    def reset(self) -> None:
        """Place the zone afresh in the next run (:meth:`BaseAction.reset`);
        a lane still closed is reopened first, as :meth:`close` does."""
        BaseAction.reset(self)  # named, as in tick
        self._props = []
        self._reopen(None)
        self._placed = False
        self._cleared = False
        self._elapsed = 0.0

    # -- placing -------------------------------------------------------------

    def _place(self, world: "carla.World") -> None:
        self._placed = True
        manager = MapManager.get_instance()
        features = MapFeatures(manager.lanelet_map, manager.routing_graph)
        start: ConstLanelet = manager.lanelet_map.laneletLayer[self._lanelet_id]
        width = _lane_width(start, self._start_s)
        objects = construction_layout(self._start_s, self._end_s, width, self._settings)
        backend = (
            self._require_backend() if self._clear_traffic or self._close_lane else None
        )
        if backend is not None:
            driver: TrafficBackend = backend
            if self._clear_traffic:
                self._clear_zone(
                    world, driver, _spans(features, start, self._start_s, self._end_s)
                )
            closed = [item.s for item in objects if item.kind != SIGN]
            if self._close_lane and closed:
                self._close(
                    world, driver, _spans(features, start, min(closed), max(closed))
                )
        # Fetched once for every prop of the zone, and a missing prop said once.
        library = world.get_blueprint_library()
        ground = road_height(run_carla_map(world))
        available: set[str] = set()
        for kind, name in self._blueprints.items():
            if len(library.filter(name)) == 0:  # no truth value in Codon
                logger.warning(
                    "%s: CARLA has no prop %r for a %s", self.label, name, kind
                )
            else:
                available.add(kind)
        for index, item in enumerate(objects):
            if item.kind not in available:
                continue
            try:
                lanelet_id, along = advance_along_lane(
                    manager.lanelet_map,
                    manager.routing_graph,
                    self._lanelet_id,
                    0.0,
                    item.s,
                )
            except ValueError:
                continue  # past the lane's last lanelet
            name = f"{self.label}/{item.zone}/{index}"
            added = SpawnAction(
                name=name,
                position=Lanelet2Pose(lanelet_id, along, item.t),
                blueprint=self._blueprints[item.kind],
                label=self.label,
            )
            if added.apply(world, library, ground):
                self._props.append(name)
        logger.info(
            "%s: %d of %d objects placed on lanelet %d from s=%.1f to %.1f",
            self.label,
            len(self._props),
            len(objects),
            self._lanelet_id,
            self._start_s,
            self._end_s,
        )

    def _clear_zone(
        self,
        world: "carla.World",
        backend: TrafficBackend,
        spans: list[tuple[int, float, float]],
    ) -> None:
        """Take the backend's background vehicles in *spans* off the road."""
        for handle, (x, y) in backend.background_vehicles(world).items():
            pose = CarlaWorldPose(x, y, 0.0)
            for lanelet_id, s_from, s_to in spans:
                if not on_lanelet(pose, lanelet_id):
                    continue
                s = project_onto_lanelet(pose, lanelet_id).s
                if s_from <= s <= s_to:
                    backend.remove_background(world, handle)
                    break

    def _close(
        self,
        world: "carla.World",
        backend: TrafficBackend,
        spans: list[tuple[int, float, float]],
    ) -> None:
        """Have the backend close the lane of each of *spans*, at its middle on
        the centreline."""
        for lanelet_id, s_from, s_to in spans:
            middle = to_carla_world(
                Lanelet2Pose(lanelet_id, (s_from + s_to) / 2.0, 0.0)
            )
            handle = backend.close_lane(world, middle.to_carla_transform())
            if handle is not None and handle not in self._closed:
                self._closed.append(handle)

    # -- clearing away -------------------------------------------------------

    def _clear(self, world: "carla.World") -> None:
        self._teardown(world)
        self._cleared = True
        logger.info("%s: cleared away", self.label)

    def _teardown(self, world: "carla.World") -> None:
        """Remove the props and reopen the lane."""
        for name in self._props:
            remove_prop(name)
        self._props = []
        self._reopen(world)

    def _reopen(self, world: Optional["carla.World"]) -> None:
        if self._backend is not None:
            backend: TrafficBackend = self._backend
            for handle in self._closed:
                backend.reopen_lane(world, handle)
        self._closed = []


class _Cleared(BaseCondition):
    """Holds once the zone it watches has come and gone."""

    _action: ConstructionZoneAction

    def __init__(self, action: ConstructionZoneAction, label: str) -> None:
        super().__init__(label=f"{label}_cleared")
        self._action = action

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        if not self._action.cleared:
            return None
        return ScenarioResult(
            passed=True,
            message=f"{self._action.label} cleared away",
            elapsed_seconds=elapsed,
        )


def _spans(
    features: MapFeatures, start: "ConstLanelet", s_from: float, s_to: float
) -> list[tuple[int, float, float]]:
    """The lanelets from *s_from* to *s_to* along the lane, as ``(id, from,
    to)`` in each one's own ``s``."""
    spans: list[tuple[int, float, float]] = []
    lanelet: ConstLanelet = start
    begins = 0.0
    for _hop in range(64):
        length = features.length(lanelet)
        if begins + length >= s_from:
            spans.append(
                (
                    int(lanelet.id),
                    max(s_from - begins, 0.0),
                    min(s_to - begins, length),
                )
            )
        if begins + length >= s_to:
            break
        following = features.straightest_following(lanelet)
        if following is None:
            break
        ahead: ConstLanelet = following
        begins += length
        lanelet = ahead
    return spans


def _bound_distance(
    bound: "lanelet2.core.ConstLineString3d", centre: "lanelet2.core.BasicPoint2d"
) -> float:
    """How far (m) *centre* is off *bound*; the arc coordinates' distance is
    signed."""
    return abs(
        float(
            lanelet2.geometry.toArcCoordinates(
                lanelet2.geometry.to2D(bound), centre
            ).distance
        )
    )


def _lane_width(lanelet: "ConstLanelet", s: float) -> float:
    """The width (m) of *lanelet* at *s*, between its bounds."""
    x, y = _point_at(_centreline(lanelet), s)
    centre = lanelet2.core.BasicPoint2d(x, y)
    width = _bound_distance(lanelet.leftBound, centre) + _bound_distance(
        lanelet.rightBound, centre
    )
    if not math.isfinite(width) or width <= 0.0:
        return _DEFAULT_LANE_WIDTH_M
    return width
