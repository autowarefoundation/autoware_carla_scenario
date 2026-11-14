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

from ...utils.clearance import actor_box_centre
from ..base import SimulationFidelity, TrafficBackendUnavailable, TrafficContext
from ..config import TrafficManagerBackendConfig
from ..fidelity import FidelityCandidate, MesoscopicConfig, plan_fidelity
from ..mesoscopic import MesoscopicTraffic
from ..parameters import VEHICLE_PARAMETER_BACKENDS, VehicleParameters
from ..traffic_manager import TrafficManagerBackend
from .backend import world_opendrive
from .config import default_cache_dir
from .geometry import Pose2D, carla_to_sumo, sumo_to_carla
from .mesosim import (
    MesoRoads,
    SumoMesoscopicTraffic,
    build_mesoscopic_model,
)
from .network import build_network, sumo_home
from .options import _net_offset, run_output_dir

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["HybridBackendConfig", "HybridTrafficBackend"]


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
    #: The options the TrafficManager side is built with.
    traffic_manager: TrafficManagerBackendConfig = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # The TrafficManager's own check, and its default when unset.
        self.traffic_manager = (
            TrafficManagerBackendConfig()
            if self.port is None
            else TrafficManagerBackendConfig(port=self.port)
        )
        self.port = self.traffic_manager.port
        self.mesoscopic = _mesoscopic(self.mesoscopic)
        if self.mesoscopic.scenario_vehicles:
            raise ValueError(
                "traffic.options.mesoscopic.scenario_vehicles is not supported by "
                "the 'hybrid' backend: only its background traffic leaves CARLA"
            )

    def resolved_cache_dir(self) -> Path:
        """The directory converted networks are cached in (the ``sumo``
        backend's, unless ``cache_dir`` says otherwise)."""
        if self.cache_dir:
            return Path(self.cache_dir).expanduser()
        return default_cache_dir()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HybridBackendConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        from ...utils.config import checked_options  # noqa: PLC0415

        checked = checked_options(cls, mapping)
        if checked.get("mesoscopic") is None:
            checked.pop("mesoscopic", None)
        return cls(**checked)


def _mesoscopic(mapping: Any) -> MesoscopicConfig:
    """*mapping* as the mesoscopic options, on unless it says otherwise."""
    if isinstance(mapping, MesoscopicConfig):
        return mapping
    if not isinstance(mapping, Mapping):
        raise ValueError(
            f"traffic.options.mesoscopic must be a mapping (got {mapping!r})"
        )
    return MesoscopicConfig.from_mapping({"enabled": True, **dict(mapping)})


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
    _route_rng: random.Random
    _step_s: float
    _steps: int
    _fidelity_due: float
    #: Changes of level this run: to mesoscopic, to microscopic.
    _level_changes: list[int]

    def __init__(
        self,
        config: Optional[HybridBackendConfig] = None,
        *,
        client: Optional["carla.Client"] = None,
        vehicles: Optional[VehicleParameters] = None,
    ) -> None:
        self._hybrid = config or HybridBackendConfig()
        self._meso = None
        self._close_meso()
        self._offset = (0.0, 0.0)
        self._route_rng = random.Random(0)
        self._step_s = 0.05
        super().__init__(self._hybrid.traffic_manager, client=client, vehicles=vehicles)

    def supported_vehicle_parameters(self) -> set[str]:
        """The TrafficManager's: it drives every vehicle in CARLA."""
        return {
            name
            for name, backends in VEHICLE_PARAMETER_BACKENDS.items()
            if TrafficManagerBackend.name in backends
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
                world_opendrive(context, "the 'hybrid' backend"),
                self._hybrid.resolved_cache_dir(),
            ).net_file
        self._offset = _net_offset(net_file)
        self._roads = MesoRoads(sumolib.net.readNet(str(net_file)))
        self._step_s = float(context.fixed_delta_seconds)
        # Routes from a stream of their own: the same seed, the same routes.
        self._route_rng = random.Random(context.random_seed + 2)
        out = run_output_dir(context)
        if config.engine == "sumo":
            out.mkdir(parents=True, exist_ok=True)
        # No signals: CARLA's lights are matched to SUMO's only by the sumo
        # backend, so SUMO's mesoscopic model runs its own programs and the
        # queue model holds nobody at a junction.
        self._meso = build_mesoscopic_model(
            config,
            self._roads.links,
            net_file=net_file,
            binary=str(sumo_home() / "bin" / "sumo"),
            step_length=self._step_s,
            seed=context.random_seed,
            log_file=out / "sumo_meso.log",
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
        now = self._now()
        assert self._roads is not None
        for handle in meso:
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
        if self._meso is not None:
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
            self._to_mesoscopic(world, handle, now)
        for handle in to_micro:
            self._to_microscopic(world, handle, now)

    def _to_mesoscopic(self, world: "carla.World", handle: str, now: float) -> bool:
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
            length = max(2.0 * float(actor.bounding_box.extent.x), 1.0)
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
        # Out of CARLA as any background vehicle leaves it.
        super().remove_background(world, handle)
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
        spawned = self._spawn_background_actor(
            world,
            carla.Transform(
                carla.Location(centre.x, centre.y, 0.0),
                carla.Rotation(0.0, centre.heading_deg, 0.0),
            ),
            parked.blueprint,
            parked.speed_kmh,
            meso.speed(handle, now),
            handle,
        )
        if spawned is None:
            return False
        meso.remove(handle)
        del self._parked[handle]
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
