"""SUMO as a traffic backend.

SUMO runs in-process (libsumo, or TraCI when libsumo is missing) on the road
network of the world the run is in (:mod:`.network`), one SUMO step per CARLA
tick, with the step length set to the world's ``fixed_delta_seconds``.  The
coupling follows TeraSim's CARLA co-simulation
(https://github.com/autowarefoundation/TeraSim, Apache-2.0); its adversarial
behaviour generation is left out -- that is what scenarios are for.

Every vehicle in the run falls in one of three groups:

* **Published** -- vehicles CARLA's side drives: an Autoware or policy-driven
  ego (the runner's ``skip_actor_ids``) and, by default
  (``scenario_vehicles=traffic_manager``), the scenario's own vehicles, which
  the TrafficManager backend this one composes drives exactly as the
  ``traffic_manager`` backend would.  Each is added to SUMO with SUMO's speed
  and lane-change control switched off, and moved to its CARLA pose every
  tick, so SUMO traffic sees it and reacts to it.  The scenario owns what it
  authored.
* **Driven** -- with ``scenario_vehicles=sumo``, the scenario's vehicles (its
  authored NPCs and an ego with ``ego.entity=autopilot``) are driven by SUMO
  instead: each is added to SUMO where it stands when the run starts, its CARLA
  actor follows its SUMO vehicle every tick, and its manoeuvres become TraCI
  calls.
* **Ambient** -- SUMO's own traffic (``randomTrips.py`` demand).  Each is
  mirrored into CARLA while it is in SUMO (and, with
  ``spawn_radius_m``/``max_vehicles``, near the ego), and destroyed when it
  leaves.

How an actor follows its SUMO vehicle differs from TeraSim, which moves
physics-off actors: CARLA reports no velocity for those, so every speed
condition, and anything else reading ``get_velocity()``, would see parked cars.
Here the actor keeps its physics; each tick it is put where its SUMO vehicle
was at the start of the step and given SUMO's speed as a constant velocity
(``enable_constant_velocity``), which carries it to where SUMO is at the end of
the step.  Height, pitch and roll stay CARLA's, so the car sits on CARLA's road.

The CARLA coupling is one step behind SUMO, as in TeraSim: SUMO steps after
the world has ticked, and its new state is applied to the actors for the next
tick.
"""

from __future__ import annotations

import logging
import math
import random
import re
from pathlib import Path
from typing import Any, ClassVar, Collection, Optional

from ...constants import EGO_ROLE_NAME
from ...utils.traffic_light import traffic_light_state_name
from ..base import (
    LaneChangeDirection,
    TrafficBackend,
    TrafficBackendError,
    TrafficBackendUnavailable,
    TrafficContext,
    TurnDirection,
)
from ..config import TrafficManagerBackendConfig
from ..traffic_manager import TrafficManagerBackend
from .config import SumoBackendConfig
from .geometry import Pose2D, carla_to_sumo, sumo_to_carla
from .network import SumoNetwork, build_network, generate_trips, sumo_home
from .physics_control import (
    longitudinal,
    lookahead_distance,
    pursuit_curvature,
    steer_command,
    to_vehicle_frame,
)

logger = logging.getLogger(__name__)

__all__ = ["SumoTrafficBackend"]

#: SUMO's lane-change duration for a manoeuvre a scenario asks for, seconds.
LANE_CHANGE_DURATION_S: float = 3.0
#: How close to its lane's centre a vehicle must be for a lane change to count
#: as settled, metres.
LANE_CHANGE_SETTLED_M: float = 0.3
#: How many edges ahead a turn looks for a junction offering it.
TURN_SEARCH_EDGES: int = 8
#: Speed mode bits of a vehicle SUMO does not control (all checks off).
_SPEED_MODE_EXTERNAL = 0
#: SUMO connection directions that count as each turn, by TurnDirection value.
_TURNS = {"left": ("l", "L"), "right": ("r", "R"), "straight": ("s",)}
#: How the run's own vehicles look in sumo-gui, apart from SUMO's traffic.
_PUBLISHED_COLOR = (220, 30, 40, 255)
_DRIVEN_COLOR = (30, 90, 220, 255)
#: moveToXY's keepRoute for a person: placed exactly where it is (2), on
#: whatever lane is there whoever may use it (4).  Without the 4 a pedestrian
#: crossing the road is put on the nearest footway, where no vehicle sees it.
_KEEP_ROUTE_ANYWHERE = 2 | 4
#: The length SUMO is told a background vehicle has before its CARLA mirror
#: exists, m, and how many junctions its route runs through.
_BACKGROUND_LENGTH_M = 4.6
_BACKGROUND_ROUTE_EDGES = 40
#: How far from a lane a background vehicle may be asked for and still be put on it, m.
_BACKGROUND_MAX_OFFSET_M = 5.0
#: How far from a pedestrian an edge to add its person on is looked for, m.
_PERSON_EDGE_RADIUS_M = 50.0
_CARLA_TO_SUMO_SIGNAL = {"Red": "r", "Yellow": "y", "Green": "G"}
_SUMO_TO_CARLA_SIGNAL = {"r": "Red", "y": "Yellow", "g": "Green", "G": "Green"}


def _load_traci(use_libsumo: bool) -> Any:
    if use_libsumo:
        try:
            import libsumo  # noqa: PLC0415

            return libsumo
        except ImportError:
            logger.warning("libsumo is not installed; running SUMO through TraCI")
    try:
        import traci  # noqa: PLC0415

        return traci
    except ImportError as exc:
        raise TrafficBackendUnavailable(
            "SUMO's Python bindings are not installed: install the `sumo` extra "
            "(autoware-carla-scenario[sumo])"
        ) from exc


def _net_offset(net_file: Path) -> tuple[float, float]:
    """The ``netOffset`` of *net_file*, read from its ``<location>``."""
    with net_file.open() as f:
        for line in f:
            found = re.search(r'netOffset="([-\d.eE+]+),([-\d.eE+]+)"', line)
            if found:
                return float(found[1]), float(found[2])
            if "<edge" in line:
                break
    return 0.0, 0.0


class SumoTrafficBackend(TrafficBackend):
    """Traffic driven by SUMO, co-simulated with the CARLA world."""

    name: ClassVar[str] = "sumo"

    def __init__(self, config: Optional[SumoBackendConfig] = None) -> None:
        self._config = config or SumoBackendConfig()
        #: Drives the scenario's own vehicles with scenario_vehicles=traffic_manager.
        self._tm = TrafficManagerBackend(
            TrafficManagerBackendConfig(port=self._config.tm_port)
        )
        self._traci: Any = None
        self._context: Optional[TrafficContext] = None
        self._network: Optional[SumoNetwork] = None
        self._net: Any = None  # sumolib's view of the network, for routes
        self._offset: tuple[float, float] = (0.0, 0.0)
        self._rng = random.Random(0)
        self._entities: list[Any] = []
        #: SUMO id -> the entity SUMO drives.
        self._driven: dict[str, Any] = {}
        #: SUMO id -> the actor CARLA's side drives, published into SUMO.
        self._external: dict[str, Any] = {}
        #: CARLA actor id -> the SUMO person standing for a CARLA pedestrian.
        self._walkers: dict[int, str] = {}
        #: CARLA pedestrians no edge was found near, so not yet in SUMO.
        self._walkers_off_network: set[int] = set()
        #: SUMO ids of the background vehicles spawn_background added.
        self._background: set[str] = set()
        self._background_count = 0
        #: SUMO id -> the CARLA blueprint asked for its mirror.
        self._blueprints: dict[str, str] = {}
        #: SUMO id -> the CARLA actor mirroring an ambient SUMO vehicle.
        self._ambient: dict[str, Any] = {}
        #: SUMO id -> lane index a lane change aims for.
        self._lane_targets: dict[str, int] = {}
        #: (tls id, link index) -> the CARLA traffic light controlling it.
        self._signals: dict[tuple[str, int], Any] = {}
        self._ego_actor: Any = None
        self._running = False
        self._failures = 0
        #: With fcd_output: SUMO time, CARLA simulation time and frame per tick.
        self._clock_log: Any = None
        #: Actor id -> what the physics follower keeps between ticks.
        self._physics: dict[int, dict[str, Any]] = {}
        #: Commands for this tick, sent in one batch.
        self._commands: list[Any] = []
        #: How often a SUMO vehicle was moved to its car (physics feedback).
        self._feedback_count = 0

    @property
    def port(self) -> int:
        """The TrafficManager port the scenario's vehicles are driven through."""
        return self._config.tm_port

    @property
    def _tm_drives_scenario(self) -> bool:
        return self._config.scenario_vehicles == "traffic_manager"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self, context: TrafficContext) -> None:
        """Build the network of the world, generate demand and start SUMO."""
        self.close()
        self._context = context
        self._traci = _load_traci(self._config.use_libsumo)
        self._rng = random.Random(context.random_seed)
        if self._config.net_path:
            net_file = Path(self._config.net_path).expanduser()
            if not net_file.is_file():
                raise TrafficBackendUnavailable(f"net_path {net_file} does not exist")
            self._network = SumoNetwork(net_file, net_file.with_suffix(".safe"))
        else:
            self._network = build_network(
                self._opendrive(context),
                self._config.resolved_cache_dir(),
                self._config.curve_lateral_acceleration,
            )
        self._offset = _net_offset(self._network.net_file)
        try:
            import sumolib  # noqa: PLC0415
        except ImportError as exc:
            raise TrafficBackendUnavailable(
                "sumolib is not installed: install the `sumo` extra"
            ) from exc
        # With internal lanes: a look-ahead point inside a junction lies on one.
        self._net = sumolib.net.readNet(str(self._network.net_file), withInternal=True)

        out = Path(context.output_dir) / "sumo"
        out.mkdir(parents=True, exist_ok=True)
        binary = self._config.binary
        if "/" not in binary:
            binary = str(sumo_home() / "bin" / binary)
        cmd = [
            binary,
            "-n",
            str(self._network.net_file),
            "--step-length",
            f"{context.fixed_delta_seconds:g}",
            "--seed",
            str(context.random_seed),
            "--collision.action",
            "warn",
            # Lane changes take time, as on the road (SUMO's default is one
            # step), so a scenario waiting for one sees it happen.
            "--lanechange.duration",
            f"{LANE_CHANGE_DURATION_S:g}",
            "--no-step-log",
            "true",
            "--log",
            str(out / "sumo.log"),
        ]
        routes = []
        ambient = self._config.ambient
        if ambient.enabled:
            routes.append(
                generate_trips(
                    self._network,
                    out / "ambient.rou.xml",
                    seed=context.random_seed,
                    period=ambient.period(),
                    end=ambient.end_s,
                    safe_weights=ambient.use_safe_weights,
                    fringe_factor=ambient.fringe_factor,
                )
            )
        if self._config.route_path:
            routes.append(Path(self._config.route_path).expanduser())
        if routes:
            cmd += ["-r", ",".join(str(r) for r in routes)]
        if self._config.fcd_output:
            cmd += ["--fcd-output", str(out / "fcd.xml"), "--fcd-output.geo", "false"]
            # How SUMO's clock maps onto CARLA's, to line the FCD up with
            # anything else recorded on CARLA's (camera frames, a rosbag).
            self._clock_log = (out / "clock.csv").open("w")
            self._clock_log.write("sumo_time,carla_elapsed_seconds,carla_frame\n")
        cmd += self._config.sumo_args
        try:
            self._traci.start(cmd)
        except Exception as exc:
            raise TrafficBackendError(f"SUMO did not start: {exc}") from exc
        self._running = True
        logger.info("SUMO started on %s", self._network.net_file)
        if self._tm_drives_scenario:
            self._tm.prepare(context)

    def adopt(self, entity: Any) -> None:
        """Remember *entity*; it joins SUMO when the run starts."""
        self._entities.append(entity)

    def start(self, world: Any, *, skip_actor_ids: Collection[int] = ()) -> None:
        """Put the run's vehicles into SUMO, warm the traffic up and go."""
        if not self._running:
            return
        tc = self._traci
        skip = set(skip_actor_ids)
        if self._tm_drives_scenario:
            # Before any ambient vehicle exists, so the TrafficManager takes
            # only the scenario's own.
            self._tm.start(world, skip_actor_ids=skip)
        by_actor = {
            entity.actor.id: entity
            for entity in self._entities
            if getattr(entity, "actor", None) is not None
        }
        for actor in world.get_actors().filter("vehicle.*"):
            entity = by_actor.get(actor.id)
            sumo_id = self._sumo_id(actor, entity)
            if not self._add_vehicle(sumo_id, actor):
                continue
            if actor.attributes.get("role_name") == str(EGO_ROLE_NAME):
                self._ego_actor = actor
            if actor.id in skip or entity is None or self._tm_drives_scenario:
                tc.vehicle.setSpeedMode(sumo_id, _SPEED_MODE_EXTERNAL)
                tc.vehicle.setLaneChangeMode(sumo_id, 0)
                tc.vehicle.setColor(sumo_id, _PUBLISHED_COLOR)
                self._external[sumo_id] = actor
            else:
                tc.vehicle.setColor(sumo_id, _DRIVEN_COLOR)
                self._driven[sumo_id] = entity
        logger.info(
            "SUMO drives %d of the scenario's vehicle(s); %d driven in CARLA are "
            "published into it",
            len(self._driven),
            len(self._external),
        )
        if self._config.traffic_light_authority != "none":
            self._signals = self._match_signals(world)
            if self._config.traffic_light_authority == "sumo":
                for light in {id(tl): tl for tl in self._signals.values()}.values():
                    light.freeze(True)
        self._warm_up(world)

    def tick(self, world: Any, elapsed: float) -> None:
        """Step SUMO once and bring CARLA's actors in line with it."""
        if self._tm_drives_scenario:
            self._tm.tick(world, elapsed)
        if not self._running:
            return
        try:
            self._push_external()
            self._push_walkers(world)
            if self._config.traffic_light_authority == "carla":
                self._signals_to_sumo()
            self._traci.simulationStep()
            if self._clock_log is not None:
                stamp = world.get_snapshot().timestamp
                self._clock_log.write(
                    f"{self._traci.simulation.getTime():.3f},"
                    f"{stamp.elapsed_seconds:.3f},{stamp.frame}\n"
                )
            self._pull_driven()
            self._sync_ambient(world)
            self._flush_commands()
            if self._config.traffic_light_authority == "sumo":
                self._signals_to_carla()
            self._failures = 0
        except Exception as exc:  # must not end the run
            self._failures += 1
            logger.warning("SUMO step failed (%d in a row): %s", self._failures, exc)
            if self._failures >= 20:
                logger.error("SUMO keeps failing; traffic stops for this run")
                self._running = False

    def close(self) -> None:
        """Stop SUMO, destroy the ambient actors and forget the run."""
        self._tm.close()
        if self._clock_log is not None:
            self._clock_log.close()
            self._clock_log = None
        if self._running and self._traci is not None:
            try:
                if self._traci.__name__ == "libsumo":
                    self._traci.close()
                else:
                    # Without waiting: a sumo-gui left open (no --quit-on-end)
                    # would otherwise hold the run's teardown until it is closed.
                    self._traci.close(False)
            except Exception as exc:  # noqa: BLE001 - teardown must go on
                logger.debug("SUMO close: %s", exc)
        self._running = False
        for actor in self._ambient.values():
            try:
                actor.destroy()
            except RuntimeError:
                pass
        for entity in self._driven.values():
            actor = getattr(entity, "actor", None)
            if actor is not None:
                try:
                    actor.disable_constant_velocity()
                except RuntimeError:
                    pass
        self._entities = []
        self._driven = {}
        self._external = {}
        self._walkers = {}
        self._walkers_off_network = set()
        self._background = set()
        self._background_count = 0
        self._blueprints = {}
        self._ambient = {}
        self._lane_targets = {}
        self._signals = {}
        self._ego_actor = None
        self._failures = 0
        if self._feedback_count:
            logger.info(
                "SUMO: moved SUMO vehicles to where CARLA had them %d time(s) this run",
                self._feedback_count,
            )
        self._physics = {}
        self._commands = []
        self._feedback_count = 0

    def describe(self) -> dict[str, Any]:
        """The backend, its seed, network and demand, for the result record."""
        out: dict[str, Any] = {"backend": self.name}
        if self._context is not None:
            out["random_seed"] = self._context.random_seed
        if self._network is not None:
            out["network"] = str(self._network.net_file)
        out["scenario_vehicles"] = self._config.scenario_vehicles
        out["vehicle_control"] = self._config.vehicle_control
        out["publish_walkers"] = self._config.publish_walkers
        if self._config.vehicle_control == "physics":
            out["feedback_distance_m"] = self._config.feedback_distance_m
        out["ambient_period_s"] = (
            self._config.ambient.period() if self._config.ambient.enabled else None
        )
        out["traffic_light_authority"] = self._config.traffic_light_authority
        return out

    # ------------------------------------------------------------------
    # Manoeuvres
    # ------------------------------------------------------------------

    def change_lane(
        self, entity: Any, world: Any, direction: LaneChangeDirection
    ) -> None:
        """Ask SUMO for a lane change one lane to *direction*."""
        if self._id_of(entity) is None and self._tm_drives_scenario:
            self._tm.change_lane(entity, world, direction)
            return
        sumo_id = self._driven_id(entity, "change_lane")
        if sumo_id is None:
            return
        tc = self._traci
        try:
            lane = tc.vehicle.getLaneIndex(sumo_id)
            # SUMO counts lanes from the right-hand kerb.
            step = 1 if direction is LaneChangeDirection.LEFT else -1
            allowed = tc.lane.getChangePermissions(tc.vehicle.getLaneID(sumo_id), step)
            if tc.vehicle.getVehicleClass(sumo_id) not in allowed:
                # SUMO obeys the line (a solid one, as roadgen reads the map),
                # where CARLA's TrafficManager would cross it.
                logger.warning(
                    "SUMO: the marking on %s forbids %s a lane change %s; it will "
                    "not happen",
                    tc.vehicle.getLaneID(sumo_id),
                    sumo_id,
                    direction.value,
                )
            tc.vehicle.changeLane(sumo_id, lane + step, LANE_CHANGE_DURATION_S)
            self._lane_targets[sumo_id] = lane + step
        except Exception as exc:  # noqa: BLE001
            logger.warning("SUMO could not change lane for %s: %s", sumo_id, exc)

    def lane_change_finished(self, entity: Any, world: Any) -> bool:
        """Whether *entity* is on its target lane, near the centre."""
        sumo_id = self._id_of(entity)
        if sumo_id is None and self._tm_drives_scenario:
            return self._tm.lane_change_finished(entity, world)
        if sumo_id is None or sumo_id not in self._lane_targets:
            return False
        try:
            import sumolib  # noqa: PLC0415

            vehicle = self._traci.vehicle
            if vehicle.getLaneIndex(sumo_id) != self._lane_targets[sumo_id]:
                return False
            # Measured from where the vehicle is drawn: a continuous lane change
            # moves onto the new lane halfway across, and SUMO's own lateral
            # position says nothing of the rest of the way.
            shape = self._net.getLane(vehicle.getLaneID(sumo_id)).getShape()
            distance = sumolib.geomhelper.distancePointToPolygon(
                vehicle.getPosition(sumo_id), shape
            )
            return bool(distance < LANE_CHANGE_SETTLED_M)
        except Exception:  # noqa: BLE001
            return False

    def set_desired_speed(self, entity: Any, world: Any, speed_kmh: float) -> None:
        """Hold *speed_kmh* (SUMO still keeps its distance to the leader)."""
        if self._id_of(entity) is None and self._tm_drives_scenario:
            self._tm.set_desired_speed(entity, world, speed_kmh)
            return
        sumo_id = self._driven_id(entity, "set_desired_speed")
        if sumo_id is None:
            return
        try:
            self._traci.vehicle.setSpeed(sumo_id, max(speed_kmh, 0.0) / 3.6)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SUMO could not set the speed of %s: %s", sumo_id, exc)

    def turn_at_junction(
        self, entity: Any, world: Any, direction: TurnDirection, **kwargs: Any
    ) -> None:
        """Route *entity* *direction* at the next junction offering that turn."""
        if self._id_of(entity) is None and self._tm_drives_scenario:
            self._tm.turn_at_junction(entity, world, direction, **kwargs)
            return
        sumo_id = self._driven_id(entity, "turn_at_junction")
        if sumo_id is None:
            return
        try:
            edge = self._traci.vehicle.getRoadID(sumo_id)
            route = self._turn_route(edge, direction)
            if route is None:
                logger.warning(
                    "SUMO: no %s turn within %d edges of %s for %s",
                    direction.value,
                    TURN_SEARCH_EDGES,
                    edge,
                    sumo_id,
                )
                return
            self._traci.vehicle.setRoute(sumo_id, route)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SUMO could not turn %s: %s", sumo_id, exc)

    # ------------------------------------------------------------------
    # Internals: the run's vehicles
    # ------------------------------------------------------------------

    def _opendrive(self, context: TrafficContext) -> str:
        if context.xodr_path is not None and Path(context.xodr_path).is_file():
            return Path(context.xodr_path).read_text()
        if context.world is None:
            raise TrafficBackendUnavailable(
                "the SUMO backend needs the world's OpenDRIVE, and the run has neither "
                "a world nor an OpenDRIVE file"
            )
        return str(context.world.get_map().to_opendrive())

    @staticmethod
    def _sumo_id(actor: Any, entity: Any) -> str:
        role = actor.attributes.get("role_name") or ""
        if entity is not None and getattr(entity, "role_name", None) is not None:
            role = str(entity.role_name)
        return f"{role or 'carla'}#{actor.id}"

    def _id_of(self, entity: Any) -> Optional[str]:
        actor = getattr(entity, "actor", None)
        for sumo_id, driven in self._driven.items():
            if driven is entity or (
                actor is not None and getattr(driven, "actor", None) is actor
            ):
                return sumo_id
        return None

    def _driven_id(self, entity: Any, what: str) -> Optional[str]:
        sumo_id = self._id_of(entity)
        if sumo_id is None or not self._running:
            self._unavailable(what, entity)
        return sumo_id

    def _length(self, actor: Any) -> float:
        return max(2.0 * float(actor.bounding_box.extent.x), 1.0)

    def _carla_pose_to_sumo(self, actor: Any) -> Pose2D:
        tf = actor.get_transform()
        return carla_to_sumo(
            Pose2D(tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw),
            self._length(actor),
            self._offset,
        )

    def _road_position(
        self, pose: Pose2D, sumo_id: str
    ) -> Optional[tuple[str, float, int]]:
        """The (edge, position, lane index) a car at SUMO *pose* starts from.

        ``None``, with a warning, where there is no road for a car there.
        """
        tc = self._traci
        try:
            edge, pos, lane = tc.simulation.convertRoad(
                pose.x, pose.y, False, "passenger"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s is off the SUMO network: %s", sumo_id, exc)
            return None
        if edge.startswith(":"):
            # Inside a junction: a route has to start on a normal edge, so it
            # starts at the beginning of the lane the junction lane leads to
            # -- not lane 0, which is a footway on a road that has one.
            links = tc.lane.getLinks(f"{edge}_{lane}")
            if not links:
                logger.warning("%s is off the SUMO network", sumo_id)
                return None
            edge, index = links[0][0].rsplit("_", 1)
            pos, lane = 0.0, int(index)
        return edge, pos, lane

    def _add_vehicle(self, sumo_id: str, actor: Any) -> bool:
        """Add *actor* to SUMO where it stands; ``False`` if it is off the network."""
        tc = self._traci
        place = self._road_position(self._carla_pose_to_sumo(actor), sumo_id)
        if place is None:
            return False
        edge, pos, lane = place
        route_id = f"route:{sumo_id}"
        try:
            tc.route.add(route_id, self._continue_route([edge]))
            tc.vehicle.add(
                sumo_id,
                route_id,
                typeID="DEFAULT_VEHTYPE",
                departLane=str(lane),
                departPos=f"{pos:.2f}",
                departSpeed="0",
            )
        except Exception as exc:  # noqa: BLE001 - one vehicle must not end the run
            logger.warning(
                "SUMO could not add %s on %s_%d: %s", sumo_id, edge, lane, exc
            )
            return False
        tc.vehicle.setLength(sumo_id, self._length(actor))
        tc.vehicle.setWidth(sumo_id, max(2.0 * float(actor.bounding_box.extent.y), 0.5))
        return True

    # ------------------------------------------------------------------
    # Background traffic
    # ------------------------------------------------------------------

    def spawn_background(
        self,
        world: Any,
        transform: Any,
        *,
        speed_kmh: float,
        blueprint: Optional[str] = None,
    ) -> Optional[str]:
        """Add a SUMO vehicle at *transform*; it is mirrored into CARLA like
        the rest of SUMO's traffic, and wanders the network until its route
        runs out."""
        del world
        if not self._running:
            return None
        tc = self._traci
        self._background_count += 1
        sumo_id = f"bg{self._background_count}"
        location, rotation = transform.location, transform.rotation
        pose = carla_to_sumo(
            Pose2D(location.x, location.y, location.z, rotation.yaw),
            _BACKGROUND_LENGTH_M,
            self._offset,
        )
        place = self._road_position(pose, sumo_id)
        if place is None:
            return None
        edge, pos, lane = place
        # convertRoad answers with the nearest lane however far away it is;
        # a background vehicle goes only where there is one.
        lane_x, lane_y = tc.simulation.convert2D(edge, pos, lane)
        if math.hypot(lane_x - pose.x, lane_y - pose.y) > _BACKGROUND_MAX_OFFSET_M:
            logger.debug("No lane for background vehicle %s near %s", sumo_id, pose)
            return None
        route_id = f"route:{sumo_id}"
        try:
            tc.route.add(route_id, self._wander(edge))
            tc.vehicle.add(
                sumo_id,
                route_id,
                typeID="DEFAULT_VEHTYPE",
                departLane=str(lane),
                departPos=f"{pos:.2f}",
                # The fastest it can safely enter at: a fixed speed near a
                # junction is one SUMO refuses to insert the vehicle at.
                departSpeed="max" if speed_kmh > 0 else "0",
            )
            if speed_kmh > 0:
                tc.vehicle.setMaxSpeed(sumo_id, max(speed_kmh / 3.6, 1.0))
        except Exception as exc:  # noqa: BLE001 - one vehicle must not end the run
            logger.warning(
                "SUMO could not add %s on %s_%d: %s", sumo_id, edge, lane, exc
            )
            return None
        self._background.add(sumo_id)
        if blueprint:
            self._blueprints[sumo_id] = blueprint
        return sumo_id

    def background_vehicles(self, world: Any) -> dict[str, tuple[float, float]]:
        """Every background vehicle SUMO still has, at its CARLA ``(x, y)``."""
        del world
        if not self._running:
            return {}
        vehicle = self._traci.vehicle
        present = set(vehicle.getIDList())
        # A vehicle added this step is pending until SUMO inserts it; one that
        # is neither has reached the end of its route and left.
        pending = set(self._traci.simulation.getPendingVehicles())
        self._background &= present | pending
        out: dict[str, tuple[float, float]] = {}
        for sumo_id in self._background & present:
            x, y, z = vehicle.getPosition3D(sumo_id)
            pose = sumo_to_carla(
                Pose2D(x, y, z, vehicle.getAngle(sumo_id)),
                vehicle.getLength(sumo_id),
                self._offset,
            )
            out[sumo_id] = (pose.x, pose.y)
        return out

    def remove_background(self, world: Any, handle: str) -> None:
        """Take a background vehicle out of SUMO; its CARLA mirror goes with it."""
        del world
        if handle not in self._background or not self._running:
            return
        self._background.discard(handle)
        try:
            self._traci.vehicle.remove(handle)
        except Exception as exc:  # noqa: BLE001 - it may have left on its own
            logger.debug("SUMO: removing %s: %s", handle, exc)

    def _wander(self, edge: str, edges: int = _BACKGROUND_ROUTE_EDGES) -> list[str]:
        """A route from *edge* that takes a random way out at every junction."""
        route = [edge]
        for _ in range(edges):
            outgoing = [
                to_edge.getID()
                for to_edge in self._net.getEdge(route[-1]).getOutgoing()
                if to_edge.allows("passenger")
            ]
            if not outgoing:
                break
            route.append(self._rng.choice(sorted(outgoing)))
        return route

    def _continue_route(self, route: list[str], edges: int = 3) -> list[str]:
        """*route* extended by up to *edges* edges, straight on where possible."""
        route = list(route)
        for _ in range(edges):
            last = self._net.getEdge(route[-1])
            outgoing = [
                (to_edge, conns)
                for to_edge, conns in last.getOutgoing().items()
                if to_edge.allows("passenger")
            ]
            if not outgoing:
                break
            straight = [
                e
                for e, conns in outgoing
                if any(c.getDirection() == "s" for c in conns)
            ]
            nxt = straight[0] if straight else self._rng.choice(outgoing)[0]
            route.append(nxt.getID())
        return route

    def _turn_route(self, edge: str, direction: TurnDirection) -> Optional[list[str]]:
        if edge.startswith(":"):
            return None
        route = [edge]
        for _ in range(TURN_SEARCH_EDGES):
            last = self._net.getEdge(route[-1])
            for to_edge, conns in last.getOutgoing().items():
                if any(c.getDirection() in _TURNS[direction.value] for c in conns):
                    return self._continue_route([*route, to_edge.getID()])
            ahead = self._continue_route(route, 1)
            if len(ahead) == len(route):
                return None
            route = ahead
        return None

    def _keep_route_ahead(self, sumo_id: str) -> None:
        """Give a vehicle that must not arrive more road before it runs out."""
        vehicle = self._traci.vehicle
        route = list(vehicle.getRoute(sumo_id))
        index = vehicle.getRouteIndex(sumo_id)
        if 0 <= index >= len(route) - 2:
            # From the edge it is on: SUMO refuses a route without it.
            vehicle.setRoute(sumo_id, self._continue_route(route[index:]))

    def _push_external(self) -> None:
        tc = self._traci
        for sumo_id, actor in self._external.items():
            pose = self._carla_pose_to_sumo(actor)
            # keepRoute=0: onto the nearest lane wherever it is, so SUMO traffic
            # keeps yielding to it (TeraSim's finding: with 2 it can end up off
            # every lane and stop being seen).
            tc.vehicle.moveToXY(sumo_id, "", -1, pose.x, pose.y, pose.heading_deg, 0)
            velocity = actor.get_velocity()
            speed = math.hypot(velocity.x, velocity.y)
            tc.vehicle.setPreviousSpeed(sumo_id, speed)

    def _push_walkers(self, world: Any) -> None:
        """Put every CARLA pedestrian into SUMO where it stands, as a person.

        Placed again each step -- a person SUMO is not moving any more would
        walk off on its own -- and removed once its actor is gone.
        """
        if not self._config.publish_walkers:
            return
        tc = self._traci
        walkers = {a.id: a for a in world.get_actors().filter("walker.pedestrian.*")}
        for actor_id in [i for i in self._walkers if i not in walkers]:
            try:
                tc.person.remove(self._walkers.pop(actor_id))
            except Exception as exc:  # noqa: BLE001 - it may have left already
                logger.debug("SUMO: removing a person: %s", exc)
        for actor_id, actor in walkers.items():
            tf = actor.get_transform()
            pose = carla_to_sumo(
                Pose2D(tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw),
                0.0,
                self._offset,
            )
            person = self._walkers.get(actor_id) or self._add_person(actor, pose)
            if person is not None:
                tc.person.moveToXY(
                    person, "", pose.x, pose.y, pose.heading_deg, _KEEP_ROUTE_ANYWHERE
                )

    def _add_person(self, actor: Any, pose: Pose2D) -> Optional[str]:
        """Add a person for *actor* on the nearest edge; ``None`` if none is near."""
        edges = [
            (distance, edge)
            for edge, distance in self._net.getNeighboringEdges(
                pose.x, pose.y, _PERSON_EDGE_RADIUS_M
            )
            # A person has to start where people walk: SUMO stops the whole
            # simulation, not just the call, when it is moved off an edge with
            # no footway.  moveToXY then puts it on the road as well.
            if edge.getFunction() == "" and edge.allows("pedestrian")
        ]
        if not edges:
            if actor.id not in self._walkers_off_network:
                self._walkers_off_network.add(actor.id)
                logger.warning(
                    "SUMO: no footway within %.0f m of pedestrian %d; SUMO traffic "
                    "will not see it",
                    _PERSON_EDGE_RADIUS_M,
                    actor.id,
                )
            return None
        self._walkers_off_network.discard(actor.id)
        person = f"carla_walker{actor.id}"
        tc = self._traci
        tc.person.add(person, min(edges, key=lambda e: e[0])[1].getID(), 0.0)
        # Stands where it is put; moveToXY moves it every step.
        tc.person.appendWaitingStage(person, 1e7)
        extent = actor.bounding_box.extent
        tc.person.setLength(person, max(2.0 * float(extent.x), 0.2))
        tc.person.setWidth(person, max(2.0 * float(extent.y), 0.2))
        tc.person.setColor(person, _PUBLISHED_COLOR)
        self._walkers[actor.id] = person
        return person

    def _follow(self, actor: Any, sumo_id: str, length: float) -> None:
        """Carry *actor* towards where its SUMO vehicle is after this step."""
        if self._config.vehicle_control == "physics":
            self._drive(actor, sumo_id, length)
        else:
            self._teleport(actor, sumo_id, length)

    def _flush_commands(self) -> None:
        """Send this tick's physics commands in one batch."""
        if (
            self._commands
            and self._context is not None
            and self._context.client is not None
        ):
            self._context.client.apply_batch(self._commands)
        self._commands = []

    def _lookahead(
        self, sumo_id: str, distance: float
    ) -> Optional[tuple[float, float]]:
        """The point *distance* metres ahead of the vehicle along its lanes (SUMO xy).

        Walks the current lane's shape, then on through the junction lane and
        onto the next edge of the vehicle's route.  ``None`` when SUMO's lanes
        cannot be followed (the caller extrapolates along the heading).
        """
        import sumolib  # noqa: PLC0415

        tc = self._traci
        try:
            lane = self._net.getLane(tc.vehicle.getLaneID(sumo_id))
        except Exception:  # noqa: BLE001 - off every lane, or a lane sumolib lacks
            return None
        # Mid lane change, SUMO's vehicle is off its lane's centre by this much
        # (to the left when positive); the point aimed at is too, or the car
        # would cut the lane change short of where SUMO has it.
        lateral = float(tc.vehicle.getLateralLanePosition(sumo_id))
        offset = tc.vehicle.getLanePosition(sumo_id) + distance
        route = list(tc.vehicle.getRoute(sumo_id))
        index = tc.vehicle.getRouteIndex(sumo_id)

        def at(shape: Any, along: float) -> tuple[float, float]:
            x0, y0 = sumolib.geomhelper.positionAtShapeOffset(shape, along)
            x1, y1 = sumolib.geomhelper.positionAtShapeOffset(shape, along + 0.5)
            norm = math.hypot(x1 - x0, y1 - y0)
            if norm < 1e-6 or lateral == 0.0:
                return x0, y0
            # Left of the lane's direction, in SUMO's right-handed frame.
            return x0 - (y1 - y0) / norm * lateral, y0 + (x1 - x0) / norm * lateral

        for _ in range(4):
            if offset <= lane.getLength():
                return at(lane.getShape(), offset)
            offset -= lane.getLength()
            edge = lane.getEdge()
            if edge.getFunction() == "internal":
                successors = [c.getToLane() for c in lane.getOutgoing()]
            else:
                nxt = route[index + 1] if 0 <= index < len(route) - 1 else None
                index += 1
                conns = [
                    c
                    for c in lane.getOutgoing()
                    if nxt is None or c.getTo().getID() == nxt
                ]
                successors = [
                    self._net.getLane(c.getViaLaneID())
                    if c.getViaLaneID()
                    else c.getToLane()
                    for c in conns
                ]
            if not successors:
                return at(lane.getShape(), max(lane.getLength() - 0.5, 0.0))
            lane = successors[0]
        return None

    def _geometry(self, actor: Any) -> dict[str, Any]:
        """Wheelbase, rear-axle offset and steering range of *actor*, read once."""
        state = self._physics.get(actor.id)
        if state is not None:
            return state

        # From the bounding box unless CARLA reports the wheels: CARLA 0.10's
        # wheel locations read as zero.  A passenger car's wheelbase is about
        # 0.58 of its length (a Lincoln MKZ's: 2.85 m of 4.9 m).
        length = 2.0 * float(actor.bounding_box.extent.x)
        wheel_base, rear_offset, max_steer = 0.58 * length, -0.29 * length, 70.0
        try:
            import typesafe_carla.carla as carla  # noqa: PLC0415

            physics = actor.get_physics_control()
            wheels = physics.wheels
            tf = actor.get_transform()
            fwd = tf.get_forward_vector()

            def ahead(w: Any) -> float:
                # `location` on CARLA 0.10, `position` before; world centimetres.
                p: Any = getattr(w, "location", None) or getattr(w, "position")
                return (p.x / 100.0 - tf.location.x) * fwd.x + (
                    p.y / 100.0 - tf.location.y
                ) * fwd.y

            front = (ahead(wheels[0]) + ahead(wheels[1])) / 2.0
            rear = (ahead(wheels[2]) + ahead(wheels[3])) / 2.0
            if 1.0 < front - rear < length:
                wheel_base, rear_offset = front - rear, rear
            max_steer = float(wheels[0].max_steer_angle) or max_steer
            # CARLA 0.10 ships a corrupt speed-based steering curve that cuts the
            # steering angle at driving speeds (the Autoware interface flattens
            # it for its ego for the same reason); a flat one makes the steer
            # command map linearly onto the wheel's range, as the follower
            # assumes.
            physics.steering_curve = [
                carla.Vector2D(0.0, 1.0),
                carla.Vector2D(120.0, 1.0),
            ]
            actor.apply_physics_control(physics)
        except Exception as exc:  # noqa: BLE001 - keep the estimates
            logger.debug("SUMO: no physics control for %s: %s", actor.id, exc)
        state = {
            "wheel_base": wheel_base,
            "rear_offset": rear_offset,
            "max_steer_rad": math.radians(max_steer),
            "steer": None,
            "integral": 0.0,
            # Not yet moving at its SUMO vehicle's speed: SUMO inserts cars at
            # speed, and a velocity given at spawn does not survive CARLA
            # placing the car.
            "launched": False,
        }
        self._physics[actor.id] = state
        return state

    def _drive(self, actor: Any, sumo_id: str, length: float) -> None:
        """Steer and accelerate *actor* along its SUMO vehicle's trajectory.

        CARLA's physics moves the car; SUMO only says where it should be.  A car
        that has drifted further than ``resync_distance_m`` (a collision, a kerb)
        is put back on its SUMO vehicle.
        """
        import typesafe_carla.carla as carla  # noqa: PLC0415

        tc = self._traci
        state = self._geometry(actor)
        dt = self._context.fixed_delta_seconds if self._context is not None else 0.05
        target_speed = float(tc.vehicle.getSpeed(sumo_id))
        x, y, z = tc.vehicle.getPosition3D(sumo_id)
        angle = tc.vehicle.getAngle(sumo_id)
        target = sumo_to_carla(Pose2D(x, y, z, angle), length, self._offset)
        ahead = lookahead_distance(target_speed)
        look = self._lookahead(sumo_id, ahead)
        if look is None:
            heading = math.radians(90.0 - angle)
            look = (x + math.cos(heading) * ahead, y + math.sin(heading) * ahead)
        look_x, look_y = look[0] - self._offset[0], -(look[1] - self._offset[1])

        tf = actor.get_transform()
        velocity = actor.get_velocity()
        speed = math.hypot(velocity.x, velocity.y)
        long_error, side_error = to_vehicle_frame(
            tf.location.x, tf.location.y, tf.rotation.yaw, target.x, target.y
        )
        if math.hypot(long_error, side_error) > self._config.resync_distance_m:
            logger.info(
                "SUMO: %s is %.1f m off its SUMO vehicle on %s (%.1f m along, %.1f m "
                "across, %.1f vs %.1f m/s); putting it back",
                sumo_id,
                math.hypot(long_error, side_error),
                tc.vehicle.getLaneID(sumo_id),
                long_error,
                side_error,
                speed,
                target_speed,
            )
            yaw = math.radians(target.heading_deg)
            self._commands += [
                carla.command.ApplyTransform(
                    actor.id,
                    carla.Transform(
                        carla.Location(target.x, target.y, tf.location.z + 0.2),
                        carla.Rotation(yaw=target.heading_deg),
                    ),
                ),
                carla.command.ApplyTargetVelocity(
                    actor.id,
                    carla.Vector3D(
                        math.cos(yaw) * target_speed, math.sin(yaw) * target_speed, 0.0
                    ),
                ),
            ]
            state["steer"], state["integral"], state["steer_integral"] = None, 0.0, 0.0
            return
        feedback = self._config.feedback_distance_m
        if feedback > 0 and math.hypot(long_error, side_error) > feedback:
            self._move_sumo_to_car(sumo_id, tf, speed, length)
        curvature = pursuit_curvature(
            x=tf.location.x,
            y=tf.location.y,
            yaw_deg=tf.rotation.yaw,
            lookahead_x=look_x,
            lookahead_y=look_y,
            rear_axle_offset=state["rear_offset"],
        )
        # CARLA's yaw rate is degrees per second, positive turning right.
        yaw_rate = math.radians(actor.get_angular_velocity().z)
        steering = steer_command(
            curvature=curvature,
            actual_curvature=yaw_rate / speed if speed > 1.0 else curvature,
            wheel_base=state["wheel_base"],
            integral=state.get("steer_integral", 0.0),
            dt=dt,
        )
        state["steer_integral"] = steering.integral
        if not state["launched"]:
            state["launched"] = True
            yaw = math.radians(tf.rotation.yaw)
            self._commands.append(
                carla.command.ApplyTargetVelocity(
                    actor.id,
                    carla.Vector3D(
                        math.cos(yaw) * target_speed, math.sin(yaw) * target_speed, 0.0
                    ),
                )
            )
        drive = longitudinal(
            speed=speed,
            sumo_speed=target_speed,
            sumo_acceleration=float(tc.vehicle.getAcceleration(sumo_id)),
            longitudinal_error=long_error,
            integral=state["integral"],
            dt=dt,
        )
        state["steer"], state["integral"] = steering.steer, drive.integral
        self._commands.append(
            carla.command.ApplyVehicleControl(
                actor.id,
                carla.VehicleControl(
                    throttle=drive.throttle,
                    brake=drive.brake,
                    steer=steering.steer,
                ),
            )
        )

    def _move_sumo_to_car(
        self, sumo_id: str, tf: Any, speed: float, length: float
    ) -> None:
        """Move a SUMO vehicle to where its car really is, keeping its speed.

        ``moveTo`` along a lane of the vehicle's route takes effect at once and
        SUMO plans the vehicle's next step from there.  Inside a junction, or
        off the route, nothing is corrected: ``moveToXY`` would re-route the
        vehicle every tick it fired, and SUMO's plan through the junction would
        come apart.
        """
        tc = self._traci
        pose = carla_to_sumo(
            Pose2D(tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw),
            length,
            self._offset,
        )
        try:
            edge, pos, lane = tc.simulation.convertRoad(
                pose.x, pose.y, False, "passenger"
            )
            if edge.startswith(":") or edge not in tc.vehicle.getRoute(sumo_id):
                return
            tc.vehicle.moveTo(sumo_id, f"{edge}_{lane}", pos)
            tc.vehicle.setPreviousSpeed(sumo_id, speed)
        except Exception as exc:  # noqa: BLE001 - SUMO refused: leave it be
            logger.debug("SUMO: could not move %s to its car: %s", sumo_id, exc)
            return
        self._feedback_count += 1

    def _teleport(self, actor: Any, sumo_id: str, length: float) -> None:
        """Put *actor* where its SUMO vehicle was at the start of this step.

        With SUMO's speed as a constant velocity for the tick to come (see the
        module docstring for why not a physics-off actor).
        """
        import typesafe_carla.carla as carla  # noqa: PLC0415

        tc = self._traci
        speed = float(tc.vehicle.getSpeed(sumo_id))
        x, y, z = tc.vehicle.getPosition3D(sumo_id)
        pose = sumo_to_carla(
            Pose2D(x, y, z, tc.vehicle.getAngle(sumo_id)), length, self._offset
        )
        dt = self._context.fixed_delta_seconds if self._context is not None else 0.05
        yaw = math.radians(pose.heading_deg)
        now = actor.get_transform()
        actor.set_transform(
            carla.Transform(
                carla.Location(
                    pose.x - math.cos(yaw) * speed * dt,
                    pose.y - math.sin(yaw) * speed * dt,
                    now.location.z,
                ),
                carla.Rotation(
                    pitch=now.rotation.pitch,
                    yaw=pose.heading_deg,
                    roll=now.rotation.roll,
                ),
            )
        )
        actor.enable_constant_velocity(carla.Vector3D(speed, 0.0, 0.0))

    def _pull_driven(self) -> None:
        present = set(self._traci.vehicle.getIDList())
        for sumo_id, entity in self._driven.items():
            actor = getattr(entity, "actor", None)
            if actor is None or sumo_id not in present:
                continue
            self._follow(actor, sumo_id, self._length(actor))
            self._keep_route_ahead(sumo_id)
        for sumo_id in self._external:
            if sumo_id in present:
                self._keep_route_ahead(sumo_id)

    def _ambient_ids(self) -> list[str]:
        """The ambient SUMO vehicles to mirror, nearest to the ego first."""
        tc = self._traci
        own = set(self._driven) | set(self._external)
        ids = [i for i in tc.vehicle.getIDList() if i not in own]
        if self._ego_actor is None:
            # Nothing to measure from: the first ones SUMO lists.
            return (
                ids[: self._config.max_vehicles] if self._config.max_vehicles else ids
            )
        if not (self._config.spawn_radius_m or self._config.max_vehicles):
            return ids
        ego = self._carla_pose_to_sumo(self._ego_actor)
        dist = {i: math.dist((ego.x, ego.y), tc.vehicle.getPosition(i)) for i in ids}
        ids.sort(key=dist.__getitem__)
        if self._config.spawn_radius_m:
            ids = [i for i in ids if dist[i] <= self._config.spawn_radius_m]
        if self._config.max_vehicles:
            ids = ids[: self._config.max_vehicles]
        return ids

    def _sync_ambient(self, world: Any) -> None:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        tc = self._traci
        wanted = self._ambient_ids()
        wanted_set = set(wanted)
        for sumo_id in [i for i in self._ambient if i not in wanted_set]:
            gone = self._ambient.pop(sumo_id)
            self._physics.pop(gone.id, None)
            try:
                gone.destroy()
            except RuntimeError:
                pass
        library = world.get_blueprint_library()
        for sumo_id in wanted:
            length = tc.vehicle.getLength(sumo_id)
            actor = self._ambient.get(sumo_id)
            if actor is None:
                x, y, z = tc.vehicle.getPosition3D(sumo_id)
                pose = sumo_to_carla(
                    Pose2D(x, y, z, tc.vehicle.getAngle(sumo_id)), length, self._offset
                )
                blueprint = library.find(
                    self._blueprints.get(sumo_id, self._config.blueprint)
                )
                if blueprint.has_attribute("role_name"):
                    blueprint.set_attribute("role_name", f"sumo:{sumo_id}")
                # A little above the road: at road height the wheels touch the
                # kerb and the spawn is refused.  Retried next tick if refused.
                actor = world.try_spawn_actor(
                    blueprint,
                    carla.Transform(
                        carla.Location(pose.x, pose.y, pose.z + 0.5),
                        carla.Rotation(yaw=pose.heading_deg),
                    ),
                )
                if actor is None:
                    continue
                self._ambient[sumo_id] = actor
                if self._config.vehicle_control == "physics":
                    # Moving at SUMO's speed from its first tick; steered from
                    # the next, once CARLA has placed it.
                    yaw = math.radians(pose.heading_deg)
                    speed = tc.vehicle.getSpeed(sumo_id)
                    actor.set_target_velocity(
                        carla.Vector3D(
                            math.cos(yaw) * speed, math.sin(yaw) * speed, 0.0
                        )
                    )
                    continue
            self._follow(actor, sumo_id, length)

    # ------------------------------------------------------------------
    # Internals: traffic lights
    # ------------------------------------------------------------------

    def _match_signals(self, world: Any) -> dict[tuple[str, int], Any]:
        """Each SUMO signal link the CARLA light standing over its lane controls."""
        tc = self._traci
        links: dict[str, tuple[str, int]] = {}
        for tls in tc.trafficlight.getIDList():
            for index, controlled in enumerate(tc.trafficlight.getControlledLinks(tls)):
                for in_lane, _out, _via in controlled:
                    links.setdefault(in_lane, (tls, index))
                    links[f"{in_lane}#{index}"] = (tls, index)
        matched: dict[tuple[str, int], Any] = {}
        for light in world.get_actors().filter("traffic.traffic_light"):
            for waypoint in light.get_affected_lane_waypoints():
                back = waypoint.previous(2.0)
                where = (back[0] if back else waypoint).transform.location
                sx, sy = where.x + self._offset[0], -where.y + self._offset[1]
                try:
                    edge, _pos, lane = tc.simulation.convertRoad(
                        sx, sy, False, "passenger"
                    )
                except Exception:  # noqa: BLE001
                    continue
                lane_id = f"{edge}_{lane}"
                for key, (tls, index) in links.items():
                    if key == lane_id or key.startswith(f"{lane_id}#"):
                        matched[(tls, index)] = light
        logger.info("Matched %d SUMO signal link(s) to CARLA lights", len(matched))
        return matched

    def _signals_to_sumo(self) -> None:
        tc = self._traci
        by_tls: dict[str, dict[int, Any]] = {}
        for (tls, index), light in self._signals.items():
            by_tls.setdefault(tls, {})[index] = light
        for tls, lights in by_tls.items():
            state = list(tc.trafficlight.getRedYellowGreenState(tls))
            for index, light in lights.items():
                char = _CARLA_TO_SUMO_SIGNAL.get(
                    traffic_light_state_name(light.get_state())
                )
                if char is None:
                    continue
                if char == "G" and state[index] == "g":
                    char = "g"  # a permissive green stays one
                state[index] = char
            tc.trafficlight.setRedYellowGreenState(tls, "".join(state))

    def _signals_to_carla(self) -> None:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        tc = self._traci
        states: dict[str, str] = {}
        for (tls, index), light in self._signals.items():
            if tls not in states:
                states[tls] = tc.trafficlight.getRedYellowGreenState(tls)
            name = _SUMO_TO_CARLA_SIGNAL.get(states[tls][index])
            if name is not None:
                light.set_state(getattr(carla.TrafficLightState, name))

    def _warm_up(self, world: Any) -> None:
        """Run SUMO alone for a while, holding the run's own vehicles still."""
        if self._context is None:
            return
        steps = int(
            self._config.warmup_s / max(self._context.fixed_delta_seconds, 1e-3)
        )
        if steps <= 0:
            return
        tc = self._traci
        # Only SUMO's own drivers are held: a published vehicle is pinned to its
        # CARLA pose by moveToXY already, and a speed forced on it here would
        # outlive the warm-up -- SUMO would report it standing still forever.
        for sumo_id in self._driven:
            tc.vehicle.setSpeed(sumo_id, 0.0)
        for _ in range(steps):
            self._push_external()
            self._push_walkers(world)
            tc.simulationStep()
        for sumo_id in self._driven:
            tc.vehicle.setSpeed(sumo_id, -1)  # back to SUMO's own speed
        self._sync_ambient(world)
        self._flush_commands()
        logger.info(
            "SUMO warmed up for %.0f s: %d ambient vehicle(s) on the road",
            self._config.warmup_s,
            len(self._ambient),
        )
