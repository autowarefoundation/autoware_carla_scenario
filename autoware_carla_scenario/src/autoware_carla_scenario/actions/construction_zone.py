"""TeraSim's construction zone as an action: cones, barriers and signs on a lane (#402).

:class:`ConstructionZoneAction` lays out TeraSim's ``ConstructionAdversity``
(:mod:`~..adversity.construction`) along a lane of the Lanelet2 map, from
``start_s`` to ``end_s`` metres along it -- past its lanelet's end, on along
the straightest successors -- and from ``start_time`` to ``end_time``
(scenario seconds; ``None`` for good) it stands the objects there as CARLA
props (``static.prop.*``, which, unlike a vehicle, may appear while a run is
under way).  Then, as TeraSim's ``initialize``:

* the traffic backend's background vehicles in the zone are taken off the
  road (``clear_traffic``);
* with ``close_lane`` (TeraSim's ``full_lane``), the backend is told the lane
  is closed (:meth:`~..traffic.base.TrafficBackend.close_lane`: SUMO stops
  routing traffic onto it; the TrafficManager cannot, and says so).

TeraSim's ``full_lane`` places no object, only closes the lane; in CARLA a
closure nobody sees is one Autoware drives into, so the objects are always
placed.  At ``end_time`` -- or when the run ends -- the props are destroyed and
the lane reopened.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Callable, Optional

import lanelet2.core
import lanelet2.geometry

from ..adversity.construction import (
    BARRIER,
    CONE,
    DEFAULT_CONSTRUCTION,
    SIGN,
    ConstructionObject,
    ConstructionSettings,
    construction_layout,
)
from ..conditions.base import BaseCondition, ScenarioResult
from ..coordinate.map_manager import MapManager
from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose
from ..coordinate.transform import on_lanelet, project_onto_lanelet, to_carla_world
from ..route.geometry import MapFeatures
from ..traffic.base import TrafficBackend
from ..trajectory.relative_lane import _centreline, _point_at
from ..trajectory.resolve import road_height
from .base import BaseAction, TickTiming

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


class ConstructionZoneAction(BaseAction):
    """A construction zone on lanelet *lanelet_id*'s lane, from *start_s* to
    *end_s*, from *start_time* until *end_time*.

    Args:
        lanelet_id: The lanelet the lane is measured from.
        start_s: Where the work starts, metres along it.
        end_s: Where it ends; past the lanelet's end, along the straightest
            successors.
        settings: TeraSim's layout numbers (zones, spacing, tapers, side).
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

    REISSUES_BY_DEFAULT = True

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
    _backend: Optional[TrafficBackend]
    _props: list["carla.Actor"]
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
            label=label, condition=condition, timing=timing, once=True, until=end
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
        self._backend = None
        self._props = []
        self._closed = []
        self._placed = False
        self._cleared = False
        self._elapsed = 0.0

    def set_traffic_backend(self, backend: TrafficBackend) -> None:
        """The run's traffic backend, told of the closure; the scenario hands
        it over when the action is registered."""
        self._backend = backend

    @property
    def cleared(self) -> bool:
        """Whether the zone has come and gone."""
        return self._cleared

    @property
    def prop_count(self) -> int:
        """How many props stand in the world now."""
        return len(self._props)

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Remember the scenario time, which :meth:`execute` is not given."""
        self._elapsed = elapsed
        super().tick(world, elapsed)

    def execute(self, world: "carla.World") -> None:
        """Place the zone at its start time, clear it away at its end time."""
        if self._cleared:
            return
        if not self._placed and self._elapsed >= self._start_time:
            self._place(world)
        end = self._end_time
        if self._placed and end is not None and self._elapsed >= end:
            self._clear(world)

    def close(self) -> None:
        """Clear the zone away if the run ends with it standing."""
        super().close()
        self._destroy_props()
        self._reopen(None)

    # -- placing -------------------------------------------------------------

    def _place(self, world: "carla.World") -> None:
        self._placed = True
        manager = MapManager.get_instance()
        features = MapFeatures(manager.lanelet_map, manager.routing_graph)
        start: ConstLanelet = manager.lanelet_map.laneletLayer[self._lanelet_id]
        width = _lane_width(start, self._start_s, features)
        objects = construction_layout(self._start_s, self._end_s, width, self._settings)
        if self._clear_traffic:
            self._clear_zone(world, features, start)
        library = world.get_blueprint_library()
        height = road_height(world.get_map())
        missing: set[str] = set()
        for item in objects:
            pose = _carla_pose(features, start, item)
            if pose is None:
                continue
            placed: CarlaWorldPose = pose
            spawned = self._spawn(world, library, item.kind, placed, height, missing)
            if spawned is not None:
                actor: "carla.Actor" = spawned
                self._props.append(actor)
            if self._close_lane and item.kind != SIGN:
                self._close(world, placed)
        logger.info(
            "%s: %d of %d objects placed on lanelet %d from s=%.1f to %.1f",
            self.label,
            len(self._props),
            len(objects),
            self._lanelet_id,
            self._start_s,
            self._end_s,
        )

    def _spawn(
        self,
        world: "carla.World",
        library: "carla.BlueprintLibrary",
        kind: str,
        pose: CarlaWorldPose,
        height: Callable[[float, float], Optional[float]],
        missing: set[str],
    ) -> Optional["carla.Actor"]:
        """One prop of *kind* at *pose*, on the road surface; None if CARLA
        has no such prop or the spot is taken."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        name = self._blueprints.get(kind, "")
        found = library.filter(name)
        if len(found) == 0:  # A BlueprintLibrary has no truth value in Codon.
            if name not in missing:
                missing.add(name)
                logger.warning(
                    "%s: CARLA has no prop %r for a %s", self.label, name, kind
                )
            none: Optional["carla.Actor"] = None
            return none
        z = pose.z
        ground = height(pose.x, pose.y)
        if ground is not None:
            z = float(ground)
        transform = carla.Transform(
            carla.Location(x=pose.x, y=pose.y, z=z + 0.05),
            carla.Rotation(yaw=pose.yaw),
        )
        return world.try_spawn_actor(found[0], transform)

    def _clear_zone(
        self, world: "carla.World", features: MapFeatures, start: "ConstLanelet"
    ) -> None:
        """Take the backend's background vehicles in the zone off the road."""
        if self._backend is None:
            return
        backend: TrafficBackend = self._backend
        spans = _spans(features, start, self._start_s, self._end_s)
        for handle, (x, y) in backend.background_vehicles(world).items():
            pose = CarlaWorldPose(x, y, 0.0)
            for lanelet_id, s_from, s_to in spans:
                if not on_lanelet(pose, lanelet_id):
                    continue
                s = project_onto_lanelet(pose, lanelet_id).s
                if s_from <= s <= s_to:
                    backend.remove_background(world, handle)
                    break

    def _close(self, world: "carla.World", pose: CarlaWorldPose) -> None:
        """Have the backend close the lane at *pose*, once per lane."""
        if self._backend is None:
            return
        backend: TrafficBackend = self._backend
        handle = backend.close_lane(world, pose.to_carla_transform())
        if handle is not None and handle not in self._closed:
            self._closed.append(handle)

    # -- clearing away -------------------------------------------------------

    def _clear(self, world: "carla.World") -> None:
        self._destroy_props()
        self._reopen(world)
        self._cleared = True
        logger.info("%s: cleared away", self.label)

    def _destroy_props(self) -> None:
        for actor in self._props:
            try:
                actor.destroy()
            except RuntimeError:
                pass  # gone with the world already
        self._props = []

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


def _locate(
    features: MapFeatures, start: "ConstLanelet", s: float
) -> Optional[tuple["ConstLanelet", float]]:
    """The lanelet *s* metres along the lane from *start*'s start, and how far
    along it; past the last, None."""
    lanelet: ConstLanelet = start
    along = s
    for _hop in range(64):
        length = features.length(lanelet)
        if along <= length:
            return lanelet, max(along, 0.0)
        following = features.straightest_following(lanelet)
        if following is None:
            return None
        ahead: ConstLanelet = following
        along -= length
        lanelet = ahead
    return None


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


def _lane_width(lanelet: "ConstLanelet", s: float, features: MapFeatures) -> float:
    """The width (m) of *lanelet* at *s*, between its bounds."""
    points = _centreline(lanelet)
    x, y = _point_at(points, min(max(s, 0.0), features.length(lanelet)))
    centre = lanelet2.core.BasicPoint2d(x, y)
    # The arc coordinates' distance is signed; its size is how far off it is.
    left = abs(
        float(
            lanelet2.geometry.toArcCoordinates(
                lanelet2.geometry.to2D(lanelet.leftBound), centre
            ).distance
        )
    )
    right = abs(
        float(
            lanelet2.geometry.toArcCoordinates(
                lanelet2.geometry.to2D(lanelet.rightBound), centre
            ).distance
        )
    )
    width = left + right
    if not math.isfinite(width) or width <= 0.0:
        return _DEFAULT_LANE_WIDTH_M
    return width


def _carla_pose(
    features: MapFeatures, start: "ConstLanelet", item: ConstructionObject
) -> Optional[CarlaWorldPose]:
    """Where *item* stands in CARLA's world, facing along the lane; None past
    the lane's last lanelet."""
    found = _locate(features, start, item.s)
    if found is None:
        none: Optional[CarlaWorldPose] = None
        return none
    lanelet, along = found
    return to_carla_world(Lanelet2Pose(int(lanelet.id), along, item.t))
