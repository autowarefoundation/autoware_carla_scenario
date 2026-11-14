"""CARLA's TrafficManager near the ego, a mesoscopic model far from it
(``traffic.backend=hybrid``).

The TrafficManager drives every vehicle that is in CARLA, as under
``traffic_manager``.  A background vehicle (:meth:`spawn_background`) far from
every *focus point* -- the ego and every other vehicle that is not background
traffic -- leaves CARLA for a mesoscopic model of the map's roads, and comes
back, an actor on autopilot again, as it nears one
(:class:`~autoware_carla_scenario.traffic.fidelity.MesoscopicConfig` decides
which, as for the ``sumo`` backend).

The TrafficManager plans no route it tells anyone of, and has no road graph to
read, so the mesoscopic side takes both from SUMO's network of the map
(:func:`~.network.build_network`, the ``sumo`` extra): a vehicle leaving CARLA
is put on the SUMO road it is on and given a route on from there, chosen at
random at each junction as the TrafficManager chooses its own, and is put back
in CARLA where that route has taken it.  The model is SUMO's own mesoscopic
simulation (``mesoscopic.engine: sumo``, the default) or the in-process queue
model (``engine: queue``).

Only background traffic changes level: the scenario's own vehicles, whose
routes, lane changes and conditions the TrafficManager backend keeps per
actor, stay in CARLA (``mesoscopic.scenario_vehicles`` is refused).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional, Union

from ...coordinate.map_manager import run_carla_map
from ...utils.clearance import actor_box_centre
from ..base import (
    SimulationFidelity,
    TrafficBackendError,
    TrafficBackendUnavailable,
    TrafficContext,
)
from ..config import TrafficManagerBackendConfig
from ..fidelity import FidelityCandidate, MesoscopicConfig, plan_fidelity
from ..mesoscopic import MesoLink, MesoscopicTraffic
from ..parameters import VEHICLE_PARAMETER_BACKENDS, VehicleParameters
from ..traffic_manager import TrafficManagerBackend, _background_lane
from .geometry import Pose2D, carla_to_sumo, sumo_to_carla
from .mesosim import SumoMesoscopicTraffic
from .network import build_network, sumo_home
from .options import _net_offset, run_output_dir

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["HybridBackendConfig", "HybridTrafficBackend", "MesoRoads"]

#: Roads a route is given ahead of a vehicle, and given again as it nears its
#: end: the TrafficManager plans no further than this either.
ROUTE_EDGES = 6
#: How far from a road's centre line a vehicle may be and still be on it, m:
#: half of a wide lane, and some.
LOCATE_RADIUS_M = 3.5
#: How far a vehicle's heading may be from its road's to be on it, degrees:
#: a vehicle changing lanes is still on its road, one on the other
#: carriageway is not.
LOCATE_HEADING_DEG = 60.0
#: The side of a cell of the index of lanes by place, m.
_CELL_M = 50.0
#: A vehicle's length where its actor gives none, m: SUMO's default car's.
_DEFAULT_LENGTH_M = 5.0


@dataclass
class HybridBackendConfig:
    """Options of the ``hybrid`` backend.

    Attributes:
        port: Port the TrafficManager is reached on, as ``traffic_manager``'s.
        mesoscopic: When a background vehicle leaves CARLA for the
            mesoscopic model, and which model (``engine``).  On by default
            here -- choosing this backend is choosing it --
            ``mesoscopic.enabled: false`` makes it ``traffic_manager``.
            ``scenario_vehicles`` is refused: only background traffic changes
            level.
        net_path: A SUMO network (``.net.xml``) of the map to use as it is,
            instead of converting the world's OpenDRIVE.
        cache_dir: Where converted networks are cached; ``None``: the
            ``sumo`` backend's cache.
    """

    port: Optional[int] = None
    mesoscopic: MesoscopicConfig = field(
        default_factory=lambda: MesoscopicConfig(enabled=True)
    )
    net_path: Optional[str] = None
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        # The TrafficManager's own check, and its default when unset.
        if self.port is None:
            self.port = TrafficManagerBackendConfig().port
        else:
            self.port = TrafficManagerBackendConfig(port=self.port).port
        if isinstance(self.mesoscopic, Mapping):
            self.mesoscopic = _mesoscopic(self.mesoscopic)
        if self.mesoscopic.scenario_vehicles:
            raise ValueError(
                "traffic.options.mesoscopic.scenario_vehicles is not supported by "
                "the 'hybrid' backend: only its background traffic leaves CARLA"
            )

    @property
    def traffic_manager(self) -> TrafficManagerBackendConfig:
        """The options the TrafficManager side is built with."""
        assert self.port is not None
        return TrafficManagerBackendConfig(port=self.port)

    def resolved_cache_dir(self) -> Path:
        """The directory converted networks are cached in (the ``sumo``
        backend's, unless ``cache_dir`` says otherwise)."""
        if self.cache_dir:
            return Path(self.cache_dir).expanduser()
        return Path.home() / ".cache" / "autoware_carla_scenario" / "sumo"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HybridBackendConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        from ...utils.config import checked_options  # noqa: PLC0415

        checked = checked_options(cls, mapping)
        if checked.get("mesoscopic") is not None:
            checked["mesoscopic"] = _mesoscopic(checked["mesoscopic"])
        else:
            checked.pop("mesoscopic", None)
        return cls(**checked)


def _mesoscopic(mapping: Mapping[str, Any]) -> MesoscopicConfig:
    """*mapping* as the mesoscopic options, on unless it says otherwise."""
    if isinstance(mapping, MesoscopicConfig):
        return mapping
    if not isinstance(mapping, Mapping):
        raise ValueError(
            f"traffic.options.mesoscopic must be a mapping (got {mapping!r})"
        )
    return MesoscopicConfig.from_mapping({"enabled": True, **dict(mapping)})


class MesoRoads:
    """The roads of a SUMO network a car may drive, for a backend whose
    vehicles are not in SUMO: the mesoscopic model's links, where a point is
    on them, where a place on them is, and routes on.

    Positions are in SUMO's frame (:mod:`.geometry` turns CARLA's into it):
    a vehicle's front bumper, as SUMO places a vehicle, and the edge offset
    along its road's rightmost lane a car may use -- the mesoscopic model has
    no lanes.
    """

    def __init__(self, net: Any) -> None:
        """Index *net* (a ``sumolib.net.Net``)."""
        self._net = net
        self.links: dict[str, MesoLink] = {}
        #: Each road's rightmost lane a car may use: where a vehicle on it is.
        self._shapes: dict[str, list[tuple[float, float]]] = {}
        #: Every lane a car may use, as ``(edge, shape)``, and those near each
        #: cell of the plane, so a point is matched against a few.
        self._lanes: list[tuple[str, list[tuple[float, float]]]] = []
        self._cells: dict[tuple[int, int], list[int]] = {}
        for edge in net.getEdges():
            if edge.getFunction() != "" or not edge.allows("passenger"):
                continue
            lanes = [lane for lane in edge.getLanes() if lane.allows("passenger")]
            if not lanes:
                continue
            edge_id = str(edge.getID())
            self.links[edge_id] = MesoLink(
                edge_id, float(edge.getLength()), len(lanes), float(edge.getSpeed())
            )
            self._shapes[edge_id] = _shape(lanes[0])
            for lane in lanes:
                self._index(edge_id, _shape(lane))

    def _index(self, edge_id: str, shape: list[tuple[float, float]]) -> None:
        index = len(self._lanes)
        self._lanes.append((edge_id, shape))
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        r = LOCATE_RADIUS_M
        for i in range(_cell(min(xs) - r), _cell(max(xs) + r) + 1):
            for j in range(_cell(min(ys) - r), _cell(max(ys) + r) + 1):
                self._cells.setdefault((i, j), []).append(index)

    def locate(
        self, x: float, y: float, heading_deg: float
    ) -> Optional[tuple[str, float]]:
        """The road a vehicle at SUMO's ``(x, y)``, heading *heading_deg*
        (SUMO's angle: clockwise from north), is on, and how far along it;
        ``None`` in a junction or off every road, or facing against it."""
        import sumolib  # noqa: PLC0415

        geom = sumolib.geomhelper
        best: Optional[tuple[float, str, float]] = None
        for index in self._cells.get((_cell(x), _cell(y)), ()):
            edge_id, shape = self._lanes[index]
            distance = float(geom.distancePointToPolygon((x, y), shape))
            if distance > LOCATE_RADIUS_M or (best is not None and distance >= best[0]):
                continue
            offset = float(
                geom.polygonOffsetWithMinimumDistanceToPoint((x, y), shape, False)
            )
            angle = 90.0 - math.degrees(
                geom.rotationAtShapeOffset(shape, offset) or 0.0
            )
            if _angle_between(angle, heading_deg) > LOCATE_HEADING_DEG:
                continue
            best = (distance, edge_id, offset)
        if best is None:
            return None
        _distance, edge_id, offset = best
        length = self.links[edge_id].length
        return edge_id, min(max(offset, 0.0), max(length - 0.1, 0.0))

    def pose(self, edge_id: str, offset: float) -> Pose2D:
        """Where *offset* metres along *edge_id* is, in SUMO's frame, facing
        along the road (SUMO's angle)."""
        import sumolib  # noqa: PLC0415

        geom = sumolib.geomhelper
        shape = self._shapes[edge_id]
        # On the shape, whose length is not quite the edge's.
        offset = min(max(offset, 0.0), max(geom.polyLength(shape) - 0.01, 0.0))
        x, y = geom.positionAtShapeOffset(shape, offset)
        rotation = geom.rotationAtShapeOffset(shape, offset) or 0.0
        return Pose2D(float(x), float(y), 0.0, (90.0 - math.degrees(rotation)) % 360.0)

    def route_on(
        self, route: list[str], rng: random.Random, edges: int = ROUTE_EDGES
    ) -> list[str]:
        """*route* extended by up to *edges* roads a car may drive, each taken
        at random where a road branches; shorter at a dead end."""
        route = list(route)
        for _ in range(edges):
            outgoing = sorted(
                str(to_edge.getID())
                for to_edge in self._net.getEdge(route[-1]).getOutgoing()
                if str(to_edge.getID()) in self.links
            )
            if not outgoing:
                break
            route.append(rng.choice(outgoing))
        return route


def _shape(lane: Any) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in lane.getShape()]


def _cell(value: float) -> int:
    return int(math.floor(value / _CELL_M))


def _angle_between(a: float, b: float) -> float:
    """How far apart two headings are, degrees, 0 to 180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class _Parked:
    """What a background vehicle out of CARLA needs to come back as itself."""

    blueprint: str
    #: The speed it was capped at, km/h (its spawn speed), if any.
    speed_kmh: Optional[float]


class HybridTrafficBackend(TrafficManagerBackend):
    """The TrafficManager, with background traffic far from everything else
    simulated mesoscopically (module docstring)."""

    name: ClassVar[str] = "hybrid"

    _hybrid: HybridBackendConfig
    _roads: Optional[MesoRoads]
    _meso: Optional[Union[MesoscopicTraffic, SumoMesoscopicTraffic]]
    _parked: dict[str, _Parked]
    _offset: tuple[float, float]

    def __init__(
        self,
        config: Optional[HybridBackendConfig] = None,
        *,
        client: Optional["carla.Client"] = None,
        vehicles: Optional[VehicleParameters] = None,
    ) -> None:
        self._hybrid = config or HybridBackendConfig()
        self._roads = None
        self._meso = None
        self._parked = {}
        self._offset = (0.0, 0.0)
        self._route_rng = random.Random(0)
        self._step_s = 0.05
        self._steps = 0
        self._fidelity_due = -math.inf
        self._level_changes = [0, 0]
        super().__init__(self._hybrid.traffic_manager, client=client, vehicles=vehicles)

    def supported_vehicle_parameters(self) -> set[str]:
        """The TrafficManager's: it drives every vehicle in CARLA."""
        return {
            name
            for name, backends in VEHICLE_PARAMETER_BACKENDS.items()
            if "traffic_manager" in backends
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self, context: TrafficContext) -> None:
        """Prepare the TrafficManager, then the mesoscopic model on SUMO's
        network of the map."""
        super().prepare(context)
        self._close_meso()
        config = self._hybrid.mesoscopic
        if not config.enabled:
            return
        try:
            import sumolib  # noqa: PLC0415
        except ImportError as exc:
            raise TrafficBackendUnavailable(
                "the 'hybrid' backend's mesoscopic model needs sumolib: install "
                "the `sumo` extra"
            ) from exc
        if self._hybrid.net_path:
            net_file = Path(self._hybrid.net_path).expanduser()
            if not net_file.is_file():
                raise TrafficBackendUnavailable(f"net_path {net_file} does not exist")
        else:
            net_file = build_network(
                _opendrive(context), self._hybrid.resolved_cache_dir()
            ).net_file
        self._offset = _net_offset(net_file)
        self._roads = MesoRoads(sumolib.net.readNet(str(net_file)))
        self._step_s = float(context.fixed_delta_seconds)
        # Routes from a stream of their own: the same seed, the same routes.
        self._route_rng = random.Random(context.random_seed + 2)
        if config.engine == "sumo":
            out = run_output_dir(context)
            out.mkdir(parents=True, exist_ok=True)
            try:
                # Its own signal programs: CARLA's lights are not matched to
                # SUMO's here, as the sumo backend matches them.
                self._meso = SumoMesoscopicTraffic(
                    self._roads.links,
                    net_file=net_file,
                    binary=str(sumo_home() / "bin" / "sumo"),
                    step_length=self._step_s,
                    seed=context.random_seed,
                    segment_length_m=config.segment_length_m,
                    junction_control=config.junction_control,
                    log_file=out / "sumo_meso.log",
                )
            except (OSError, RuntimeError) as exc:
                raise TrafficBackendError(
                    f"SUMO's mesoscopic simulation did not start: {exc}"
                ) from exc
        else:
            self._meso = MesoscopicTraffic(
                self._roads.links,
                headway_s=config.headway_s,
                jam_spacing_m=config.jam_spacing_m,
            )
        logger.info(
            "hybrid: far background traffic is mesoscopic (%s) on %s",
            config.engine,
            net_file,
        )

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """The TrafficManager's tick, then a step of the mesoscopic model and,
        when due, a decision of which background vehicles change level."""
        super().tick(world, elapsed)
        meso = self._meso
        if meso is None:
            return
        self._steps += 1
        now = round(self._steps * self._step_s, 6)
        assert self._roads is not None
        for handle in list(meso):
            vehicle = meso.vehicle(handle)
            if vehicle.index >= len(vehicle.route) - 2:
                meso.set_route(
                    handle,
                    self._roads.route_on(vehicle.remaining_route, self._route_rng),
                )
        for vehicle in meso.step(now):
            # The end of a dead end: gone, as the TrafficManager's are.
            self._parked.pop(vehicle.id, None)
        if now >= self._fidelity_due:
            self._fidelity_due = now + self._hybrid.mesoscopic.update_interval_s
            self._rebalance(world, now)

    def close(self) -> None:
        """Close the TrafficManager side and the mesoscopic model."""
        super().close()
        if self._meso is not None and self._level_changes != [0, 0]:
            logger.info(
                "hybrid: %d vehicle change(s) to mesoscopic, %d to microscopic this run",
                *self._level_changes,
            )
        self._close_meso()

    def _close_meso(self) -> None:
        if isinstance(self._meso, SumoMesoscopicTraffic):
            self._meso.close()
        self._meso = None
        self._roads = None
        self._parked = {}
        self._steps = 0
        self._fidelity_due = -math.inf
        self._level_changes = [0, 0]

    def describe(self) -> dict[str, Any]:
        """The TrafficManager's, and the mesoscopic model's so far."""
        out = super().describe()
        meso = self._hybrid.mesoscopic
        if meso.enabled:
            out["mesoscopic"] = {
                "engine": meso.engine,
                "micro_radius_m": meso.micro_radius_m,
                "meso_radius_m": meso.demote_radius_m,
                "to_mesoscopic": self._level_changes[0],
                "to_microscopic": self._level_changes[1],
            }
        return out

    def fidelity_census(self) -> dict[str, int]:
        """The vehicles the TrafficManager drives are microscopic, the
        mesoscopic model's mesoscopic."""
        micro = 0
        for actor in list(self._autopiloted.values()):
            try:
                if actor.is_alive:
                    micro += 1
            except RuntimeError:
                continue
        return {
            SimulationFidelity.MICROSCOPIC.value: micro,
            SimulationFidelity.MESOSCOPIC.value: len(self._meso or ()),
        }

    # ------------------------------------------------------------------
    # Background traffic
    # ------------------------------------------------------------------

    def background_vehicles(
        self, world: "carla.World"
    ) -> dict[str, tuple[float, float, Optional[float]]]:
        """Every background vehicle, in CARLA or not: one out of it where the
        mesoscopic model has it, with no height."""
        out = super().background_vehicles(world)
        if self._meso is not None:
            now = self._now()
            for handle in self._meso:
                if handle in self._parked:
                    pose = self._carla_pose(handle, now)
                    out[handle] = (pose.x, pose.y, None)
        return out

    def remove_background(self, world: "carla.World", handle: str) -> None:
        """Take a background vehicle out of the run, wherever it is."""
        if handle in self._parked:
            del self._parked[handle]
            if self._meso is not None:
                self._meso.remove(handle)
            return
        super().remove_background(world, handle)

    # ------------------------------------------------------------------
    # Changing level
    # ------------------------------------------------------------------

    def _now(self) -> float:
        return round(self._steps * self._step_s, 6)

    def _rebalance(self, world: "carla.World", now: float) -> None:
        """Send the background vehicles far from every focus point out of
        CARLA, and bring back those near one."""
        meso = self._meso
        if meso is None:
            return
        config = self._hybrid.mesoscopic
        ahead = config.horizon_s > 0.0
        background = {int(actor.id) for actor in self._background.values()}
        focus: list[tuple[float, float]] = []
        focus_velocities: list[tuple[float, float]] = []
        for actor in world.get_actors().filter("vehicle.*"):
            if int(actor.id) in background:
                continue
            try:
                location = actor.get_location()
                velocity = actor.get_velocity() if ahead else None
            except RuntimeError:  # destroyed under us
                continue
            focus.append((float(location.x), float(location.y)))
            if velocity is not None:
                focus_velocities.append((float(velocity.x), float(velocity.y)))
        if not focus:
            return
        candidates = []
        for handle, actor in list(self._background.items()):
            try:
                location = actor.get_location()
                velocity = actor.get_velocity() if ahead else None
            except RuntimeError:
                continue
            vx, vy = (
                (float(velocity.x), float(velocity.y))
                if velocity is not None
                else (0.0, 0.0)
            )
            candidates.append(
                FidelityCandidate(
                    handle,
                    SimulationFidelity.MICROSCOPIC,
                    float(location.x),
                    float(location.y),
                    vx,
                    vy,
                )
            )
        for handle in meso:
            if handle not in self._parked:
                continue
            pose = self._carla_pose(handle, now)
            vx = vy = 0.0
            if ahead:
                speed = meso.speed(handle, now)
                yaw = math.radians(pose.heading_deg)
                vx, vy = speed * math.cos(yaw), speed * math.sin(yaw)
            candidates.append(
                FidelityCandidate(
                    handle, SimulationFidelity.MESOSCOPIC, pose.x, pose.y, vx, vy
                )
            )
        to_meso, to_micro = plan_fidelity(config, candidates, focus, focus_velocities)
        for handle in to_meso:
            self._to_mesoscopic(handle, now)
        for handle in to_micro:
            self._to_microscopic(world, handle, now)

    def _to_mesoscopic(self, handle: str, now: float) -> bool:
        """Move background vehicle *handle* out of CARLA into the mesoscopic
        model, on the road it is on.  Not from a junction, where it is on no
        road of the model: that waits for the next decision.  Answers whether
        it moved."""
        meso, roads = self._meso, self._roads
        actor = self._background.get(handle)
        if meso is None or roads is None or actor is None:
            return False
        try:
            transform = actor.get_transform()
            x, y = actor_box_centre(actor, transform)
            length = _length_of(actor)
            blueprint = str(actor.type_id)
        except RuntimeError:
            return False
        pose = carla_to_sumo(
            Pose2D(x, y, float(transform.location.z), float(transform.rotation.yaw)),
            length,
            self._offset,
        )
        found = roads.locate(pose.x, pose.y, pose.heading_deg)
        if found is None:
            return False
        edge, offset = found
        speed_kmh = self._exact_kmh.get(int(actor.id))
        try:
            meso.add(
                handle,
                roads.route_on([edge], self._route_rng),
                now=now,
                offset_m=offset,
                max_speed=speed_kmh / 3.6 if speed_kmh is not None else math.inf,
                length=length,
            )
        except ValueError as exc:
            logger.debug("hybrid: %s stays in CARLA: %s", handle, exc)
            return False
        self._parked[handle] = _Parked(blueprint, speed_kmh)
        del self._background[handle]
        self._forget_background_actor(actor)
        self._level_changes[0] += 1
        return True

    def _to_microscopic(self, world: "carla.World", handle: str, now: float) -> bool:
        """Put background vehicle *handle* back in CARLA where the mesoscopic
        model has it, on autopilot.  One that does not fit there (the spot is
        taken) stays out until the next decision.  Answers whether it moved."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        meso = self._meso
        parked = self._parked.get(handle)
        if meso is None or parked is None or handle not in meso:
            return False
        centre = self._carla_pose(handle, now)
        lane = _background_lane(
            world,
            carla.Transform(
                carla.Location(centre.x, centre.y, 0.0),
                carla.Rotation(0.0, centre.heading_deg, 0.0),
            ),
        )
        if lane is None:
            return False
        on_lane: "carla.Transform" = lane
        found = world.get_blueprint_library().filter(parked.blueprint)
        if len(found) == 0:
            logger.warning("hybrid: no blueprint %r for %s", parked.blueprint, handle)
            return False
        blueprint = found[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", handle)
        lifted = carla.Transform(
            carla.Location(
                on_lane.location.x, on_lane.location.y, on_lane.location.z + 0.5
            ),
            carla.Rotation(0.0, on_lane.rotation.yaw, 0.0),
        )
        actor = world.try_spawn_actor(blueprint, lifted)
        if actor is None:
            return False
        speed = meso.speed(handle, now)
        forward = lifted.get_forward_vector()
        actor.set_target_velocity(
            carla.Vector3D(forward.x * speed, forward.y * speed, 0.0)
        )
        meso.remove(handle)
        del self._parked[handle]
        self._background[handle] = actor
        if self._started:
            self._autopilot(actor, keep=False)
        if parked.speed_kmh is not None:
            tm = self._require_tm("hybrid")
            if tm is not None:
                self._give_speed(tm, actor, parked.speed_kmh)
        self._level_changes[1] += 1
        return True

    def _carla_pose(self, handle: str, now: float) -> Pose2D:
        """Where the mesoscopic model has *handle*: its box centre and yaw in
        CARLA's frame."""
        assert self._meso is not None and self._roads is not None
        edge, offset = self._meso.position(handle, now)
        return sumo_to_carla(
            self._roads.pose(edge, offset),
            self._meso.vehicle(handle).length,
            self._offset,
        )

    def _forget_background_actor(self, actor: "carla.Actor") -> None:
        """Destroy *actor*, and forget what the TrafficManager side kept of
        it: its id may be another vehicle's next."""
        actor_id = int(actor.id)
        for kept in (
            self._autopiloted,
            self._exact_kmh,
            self._speed_differences,
            self._held,
            self._own_ignores,
        ):
            kept.pop(actor_id, None)  # type: ignore[attr-defined]
        self._emergencies.discard(actor_id)
        self._forget_overrides(actor_id)
        try:
            actor.destroy()
        except RuntimeError:
            pass


def _length_of(actor: Any) -> float:
    """*actor*'s length, m, from its bounding box."""
    box = getattr(actor, "bounding_box", None)
    extent = getattr(box, "extent", None)
    if extent is None or not float(extent.x) > 0.0:
        return _DEFAULT_LENGTH_M
    return 2.0 * float(extent.x)


def _opendrive(context: TrafficContext) -> str:
    """The OpenDRIVE of the run's world, as the ``sumo`` backend reads it."""
    if context.xodr_path is not None and Path(context.xodr_path).is_file():
        return Path(context.xodr_path).read_text()
    if context.world is None:
        raise TrafficBackendUnavailable(
            "the 'hybrid' backend needs the world's OpenDRIVE, and the run has "
            "neither a world nor an OpenDRIVE file"
        )
    return str(run_carla_map(context.world).to_opendrive())
