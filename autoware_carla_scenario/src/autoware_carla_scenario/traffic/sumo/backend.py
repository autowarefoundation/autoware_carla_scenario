"""SUMO as a traffic backend.

SUMO runs in-process (libsumo, or TraCI when libsumo is missing) on the road
network of the world the run is in (:mod:`.network`), one SUMO step per CARLA
tick, with the step length set to the world's ``fixed_delta_seconds`` (or
several shorter ones per tick, with ``step_length_s``).  The
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

Pedestrians go both ways too: CARLA's are published into SUMO as persons
(``publish_walkers``), and SUMO's own (``pedestrians``) are mirrored into
CARLA as walkers while they are near the ego or a vehicle CARLA drives.  The
scenario's own pedestrians are published, like its vehicles, unless
``scenario_pedestrians=sumo``: then each is a SUMO person its walker follows,
which a ``WalkStraightAction`` walks straight on through TraCI
(:meth:`SumoTrafficBackend.walk_straight`).

Bicycles are SUMO vehicles of the class ``bicycle``.  SUMO's own
(``bicycles``) ride the carriageway and, on a network built with
``cycle_on_sidewalks``, the sidewalks too, and are mirrored into CARLA as
bicycles near the ego, with limits of their own.  A scenario's own vehicle
whose blueprint is a bicycle joins SUMO as one, published or driven like the
rest, and SUMO routes it where bicycles may go.  How SUMO's bicycles ride is
``bicycles``' own, not the drivers' parameters.

Nothing joins the run that the run did not ask for: SUMO's own vehicles
(``ambient``), pedestrians (``pedestrians``) and bicycles (``bicycles``) are
off by default, so the road holds the scenario's vehicles and pedestrians, and
the background vehicles its traffic sources add (``TrafficSourceAction``).

With ``mesoscopic.enabled``, a driven or ambient vehicle far from every
published one (the ego among them) leaves SUMO for a mesoscopic queue model
(:mod:`..mesoscopic`) and CARLA for good measure: SUMO steps only the traffic
near what CARLA drives, and CARLA holds an actor only for that.  It is put back
into SUMO, and into CARLA, where the queue model has it once it comes near
again.  A scenario's own vehicle does this only with
``mesoscopic.scenario_vehicles``, and its entity's ``simulation_fidelity`` says
which it is at (:class:`..base.SimulationFidelity`).

How an actor follows its SUMO vehicle differs from TeraSim, which moves
physics-off actors: CARLA reports no velocity for those, so every speed
condition, and anything else reading ``get_velocity()``, would see parked cars.
Here the actor keeps its physics; each tick it is put where its SUMO vehicle
was at the start of the step and given SUMO's speed as a constant velocity
(``enable_constant_velocity``), which carries it to where SUMO is at the end of
the step.  Height, pitch and roll stay CARLA's, so the car sits on CARLA's road.

How SUMO's drivers behave is the run's
:class:`~autoware_carla_scenario.traffic.parameters.VehicleParameters`
(those :data:`~autoware_carla_scenario.traffic.parameters.VEHICLE_PARAMETER_BACKENDS`
gives ``sumo``): each vehicle SUMO drives is given them as it
departs, and the TrafficManager driving the scenario's vehicles is given the
ones it shares.  How its pedestrians walk is the run's
:class:`~autoware_carla_scenario.traffic.parameters.PedestrianParameters`:
SUMO's default pedestrian type is a distribution of types, each its own draw
of them, which every pedestrian of SUMO's own demand is.  Every numeric
parameter is drawn per vehicle and per pedestrian from its spread.  SUMO's own traffic sees only as far as the run's weather lets
its drivers see (``follow_visibility``, on by default,
:mod:`autoware_carla_scenario.visibility`): night, rain and fog hold each
vehicle to the speed it can stop from within that distance, worked out from
its ``reaction_time_s`` and ``decel``.  The TrafficManager has no notion of
sight, and only logs it.

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
from ...coordinate.map_manager import run_carla_map
from ...utils.traffic_light import traffic_light_state_name
from ...visibility import sight_speed, world_visibility
from ...world_reset import episode_of
from ..base import (
    LaneChangeDirection,
    SimulationFidelity,
    TrafficBackend,
    TrafficBackendError,
    TrafficBackendUnavailable,
    TrafficContext,
    TurnDirection,
)
from ..config import TrafficManagerBackendConfig
from ..fidelity import FidelityCandidate, MesoscopicConfig, plan_fidelity
from ..parameters import (
    PEDESTRIAN_BOUNDS,
    VEHICLE_BOUNDS,
    PedestrianParameters,
    VehicleParameters,
)
from ..mesoscopic import MesoLink, MesoscopicTraffic
from ..traffic_manager import TrafficManagerBackend
from .config import BICYCLE_BOUNDS, SumoBackendConfig
from .geometry import Pose2D, carla_to_sumo, sumo_to_carla
from .network import (
    SignalTable,
    SumoNetwork,
    build_network,
    crossing_edges,
    generate_rides,
    generate_trips,
    generate_walks,
    sumo_home,
)
from .physics_control import (
    longitudinal,
    lookahead_distance,
    pursuit_curvature,
    steer_command,
    to_vehicle_frame,
)

logger = logging.getLogger(__name__)

__all__ = ["SumoTrafficBackend", "is_bicycle"]

#: SUMO's lane-change duration for a manoeuvre a scenario asks for, seconds.
LANE_CHANGE_DURATION_S: float = 3.0
#: How close to its lane's centre a vehicle must be for a lane change to count
#: as settled, metres.
LANE_CHANGE_SETTLED_M: float = 0.3
#: How many edges ahead a turn looks for a junction offering it.
TURN_SEARCH_EDGES: int = 8
#: Speed mode bits of a vehicle SUMO does not control (all checks off).
_SPEED_MODE_EXTERNAL = 0
#: Lane-change mode bits of a vehicle SUMO does not control (none).
_LANE_CHANGE_MODE_EXTERNAL = 0
#: Lane-change mode bits of a vehicle that changes lane only where its route
#: needs it: SUMO's default (1621) without the cooperative, speed-gain and
#: keep-right changes.
_LANE_CHANGE_MODE_STRATEGIC = 1537
#: SUMO's own lane-change mode bits, which ``lane_changes: true`` puts back.
_LANE_CHANGE_MODE_DEFAULT = 1621
#: The pedestrian parameters a run can change: the walking speed and the gap
#: are set on each person, but ``jmDriveAfterRedTime`` only on a type, and
#: the types are written before SUMO starts.
_PEDESTRIAN_OVERRIDES = ("walking_speed_mps", "min_gap_m")
#: Vehicle parameter -> the TraCI setter that gives a vehicle its draw.
_VEHICLE_SETTERS = {
    "speed_factor": "setSpeedFactor",
    "min_gap_m": "setMinGap",
    "accel": "setAccel",
    "decel": "setDecel",
    "reaction_time_s": "setTau",
    "imperfection": "setImperfection",
    "max_lateral_speed_mps": "setMaxSpeedLat",
}
#: Vehicle parameter -> the lane-change model parameter that gives a vehicle
#: its draw, for one with no setter of its own (``laneChangeModel.<name>``).
_LANE_CHANGE_MODEL_PARAMETERS = {"max_lateral_accel": "lcAccelLat"}
#: How many pedestrian types, each its own draw of the pedestrian
#: parameters, SUMO's default pedestrian type is a distribution of.
PEDESTRIAN_TYPES = 100
#: The same for SUMO's default bicycle type, and the vType attribute each
#: riding parameter of ``bicycles`` sets.
BICYCLE_TYPES = 100
_BICYCLE_TYPE = "DEFAULT_BIKETYPE"
_BICYCLE_ATTRIBUTES = {
    "max_speed_mps": "maxSpeed",
    "accel": "accel",
    "decel": "decel",
    "min_gap_m": "minGap",
    "reaction_time_s": "tau",
    "imperfection": "sigma",
    "max_lateral_speed_mps": "maxSpeedLat",
    "red_light_violation_s": "jmDriveAfterRedTime",
}
#: SUMO connection directions that count as each turn, by TurnDirection value.
_TURNS = {"left": ("l", "L"), "right": ("r", "R"), "straight": ("s",)}
#: How the run's own vehicles look in sumo-gui, apart from SUMO's traffic.
_PUBLISHED_COLOR = (220, 30, 40, 255)
#: The role name of an ambient SUMO vehicle's (or person's) CARLA mirror
#: begins with this, then its SUMO id.
_MIRROR_PREFIX = "sumo:"
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


def _sumo_step(tick: float, step_length: Optional[float]) -> tuple[float, int]:
    """SUMO's step length and how many steps it makes per CARLA *tick*.

    Raises:
        ValueError: If *step_length* is longer than the tick, does not divide
            it, or is not a whole number of milliseconds -- SUMO's clock counts
            those, so it would round the step and drift from CARLA's.
    """
    if step_length is None:
        return tick, 1
    steps = round(tick / step_length)
    millis = step_length * 1000.0
    if (
        steps < 1
        or abs(steps * step_length - tick) > 1e-6 * max(tick, 1.0)
        or abs(millis - round(millis)) > 1e-6
    ):
        raise ValueError(
            f"traffic.options.step_length_s ({step_length:g} s) must divide the "
            f"world's fixed_delta_seconds ({tick:g} s) in whole milliseconds: "
            "SUMO makes a whole number of steps per CARLA tick, and its clock "
            "counts milliseconds"
        )
    return tick / steps, steps


def is_bicycle(actor: Any) -> bool:
    """Whether the CARLA vehicle *actor* is a bicycle: its blueprint's
    ``base_type`` says so, as CARLA's bicycles' do."""
    try:
        base_type = str(actor.attributes.get("base_type", ""))
    except Exception:  # noqa: BLE001 - an actor without attributes is a car
        return False
    return base_type.lower() == "bicycle"


def _ride_state(follows: str, states: list[str]) -> str:
    """The state of a bicycle ride's signal link, from the *states* of the
    vehicle links it *follows* (``with``) or must be clear of (``clear``)."""
    if follows == "clear":
        return "G" if all(state in "rR" for state in states) else "r"
    if any(state in "Gg" for state in states):
        return "g"  # with the traffic, giving way as a bicycle does
    if any(state in "yY" for state in states):
        return "y"
    return "r"


class SumoTrafficBackend(TrafficBackend):
    """Traffic driven by SUMO, co-simulated with the CARLA world."""

    name: ClassVar[str] = "sumo"

    def __init__(
        self,
        config: Optional[SumoBackendConfig] = None,
        *,
        vehicles: Optional[VehicleParameters] = None,
        pedestrians: Optional[PedestrianParameters] = None,
    ) -> None:
        """Create the backend.

        Args:
            config: Backend options; defaults are used when omitted.
            vehicles: How SUMO's drivers behave; SUMO's defaults (and
                ``follow_visibility``) when omitted.
            pedestrians: How SUMO's pedestrians walk; SUMO's defaults when
                omitted.

        Raises:
            TrafficParameterError: If *vehicles* or *pedestrians* sets a field
                SUMO does not support.
        """
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
        self._draws = random.Random(1)
        self._entities: list[Any] = []
        #: ``id()`` of the entities taken back with release(): published, not driven.
        self._released: set[int] = set()
        #: ``id()`` of an entity released while SUMO drove it -> its SUMO id
        #: and the speed and lane-change modes resume() puts back.
        self._handed: dict[int, tuple[str, int, int]] = {}
        #: SUMO id -> the speed and lane-change modes of an ambient vehicle
        #: released to an adversity (#417); its actor is in ``_external``
        #: meanwhile.
        self._handed_ambient: dict[str, tuple[int, int]] = {}
        #: SUMO id -> the speed (m/s) set_desired_speed() last held it at.
        self._commanded: dict[str, float] = {}
        #: SUMO id -> the entity SUMO drives.
        self._driven: dict[str, Any] = {}
        #: SUMO id -> the actor CARLA's side drives, published into SUMO.
        self._external: dict[str, Any] = {}
        #: CARLA actor id -> the SUMO person standing for a CARLA pedestrian.
        self._walkers: dict[int, str] = {}
        #: CARLA pedestrians no edge was found near, so not yet in SUMO.
        self._walkers_off_network: set[int] = set()
        #: The scenario's pedestrians, adopted with adopt_pedestrian().
        self._pedestrian_entities: list[Any] = []
        #: ``id()`` of the scenario's pedestrians an action moves itself
        #: (release_pedestrian()): published, not walked.
        self._released_pedestrians: set[int] = set()
        #: SUMO person id -> the scenario's pedestrian SUMO walks
        #: (``scenario_pedestrians=sumo``).
        self._walked: dict[str, Any] = {}
        #: The persons of ``_walked`` out on a walk, whose walkers follow them:
        #: where each is (SUMO's frame), its heading and its speed (m/s).
        self._walking: dict[str, tuple[float, float, float, float]] = {}
        #: SUMO ids of the background vehicles spawn_background added.
        self._background: set[str] = set()
        #: Lane id -> the vehicle classes it allowed before close_lane().
        self._closed_lanes: dict[str, list[str]] = {}
        self._background_count = 0
        #: SUMO id -> the CARLA blueprint asked for its mirror.
        self._blueprints: dict[str, str] = {}
        #: SUMO id -> the CARLA actor mirroring an ambient SUMO vehicle.
        self._ambient: dict[str, Any] = {}
        #: SUMO id -> its vehicle class, read once.
        self._vclasses: dict[str, str] = {}
        #: The bicycle blueprints the world has, of ``bicycles.blueprints``;
        #: looked up on the first bicycle mirrored.
        self._bicycle_blueprints: Optional[list[str]] = None
        #: tls id -> each signal link of a bicycle ride CARLA's lights do not
        #: switch: its index, how it follows (``with`` its companion vehicle
        #: links, or ``clear`` of every vehicle link it never goes with in
        #: SUMO's program) and those links.
        self._bicycle_rides: dict[str, list[tuple[int, str, list[int]]]] = {}
        #: SUMO id -> lane index a lane change aims for.
        self._lane_targets: dict[str, int] = {}
        #: (tls id, link index) -> the CARLA traffic light controlling it.
        self._signals: dict[tuple[str, int], Any] = {}
        #: tls id -> each pedestrian crossing link it controls, with the
        #: vehicle links into or out of a road the crossing crosses.
        self._crossing_links: dict[str, list[tuple[int, list[int]]]] = {}
        self._ego_actor: Any = None
        self._running = False
        self._failures = 0
        #: With fcd_output: SUMO time, CARLA simulation time and frame per tick.
        self._clock_log: Any = None
        #: Actor id -> what the physics follower keeps between ticks.
        self._physics: dict[int, dict[str, Any]] = {}
        #: Commands for this tick, sent in one batch.
        self._commands: list[Any] = []
        #: Actor id -> the constant velocity (m/s) teleport mode last gave it.
        self._constant_speed: dict[int, float] = {}
        #: How often a SUMO vehicle was moved to its car (physics feedback).
        self._feedback_count = 0
        #: With mesoscopic.enabled: the queue model far traffic runs in.
        self._meso: Optional[MesoscopicTraffic] = None
        #: SUMO id -> the scenario entity the queue model holds.
        self._meso_entities: dict[str, Any] = {}
        #: SUMO id -> the blueprint of a mesoscopic entity's actor, and the
        #: width of a mesoscopic vehicle, to bring it back as it was.
        self._meso_blueprints: dict[str, str] = {}
        self._meso_widths: dict[str, float] = {}
        #: SUMO time at which the next level decision is due.
        self._fidelity_due = -math.inf
        #: SUMO ids of driven vehicles that stay microscopic: their entity
        #: keeps its actor, or something asked for it (require_microscopic).
        self._stays_microscopic: set[str] = set()
        #: (from edge, to edge) -> the signal links of the move, and each
        #: signal's state this step (read once, for the queue model's gate).
        self._edge_signals: dict[tuple[str, str], list[tuple[str, int]]] = {}
        self._signal_states: dict[str, str] = {}
        #: Edge id -> the lane shape a mesoscopic vehicle on it is placed on.
        self._edge_shapes: dict[str, list[tuple[float, float]]] = {}
        self._route_count = 0
        #: How many vehicles became mesoscopic, and microscopic, this run.
        self._level_changes = [0, 0]
        #: SUMO person id -> the CARLA walker mirroring one of SUMO's own
        #: pedestrians.
        self._persons: dict[str, Any] = {}
        #: How far drivers see, m, as last applied; ``None``: as far as SUMO lets them.
        self._visibility: Optional[float] = None
        #: SUMO id -> its maximum speed (m/s) before the visibility held it
        #: down, its deceleration (m/s²) and its tau (s): read once.
        self._max_speeds: dict[str, tuple[float, float, float]] = {}
        #: SUMO steps per CARLA tick (``step_length_s``).
        self._substeps = 1
        #: The vehicles that departed in this step, read once per step.
        self._departed: Optional[list[str]] = None
        #: The pedestrian parameters changed during the run, which each
        #: person who departs from then on draws.
        self._pedestrian_overrides: Optional[PedestrianParameters] = None
        if vehicles is not None:
            self.set_vehicle_parameters(vehicles)
        if pedestrians is not None:
            self.set_pedestrian_parameters(pedestrians)

    def set_vehicle_parameters(self, parameters: VehicleParameters) -> None:
        """Drive SUMO's vehicles with *parameters*, and the scenario's that
        the TrafficManager drives with the ones it shares: SUMO's own (a
        reaction time, a braking) have no TrafficManager counterpart, and the
        scenario's vehicles under ``scenario_vehicles=traffic_manager`` drive
        as they would under the ``traffic_manager`` backend."""
        super().set_vehicle_parameters(parameters)
        shared = parameters.restricted_to(self._tm.supported_vehicle_parameters())
        self._tm.set_vehicle_parameters(shared)
        dropped = sorted(set(parameters.set_fields()) - set(shared.set_fields()))
        if dropped and self._tm_drives_scenario:
            logger.info(
                "SUMO: %s apply to SUMO's vehicles only; the scenario's, which "
                "the TrafficManager drives, do without them",
                ", ".join(dropped),
            )

    def supported_vehicle_parameters(self) -> set[str]:
        """SUMO's, but those of :meth:`parameter_requirements`."""
        return super().supported_vehicle_parameters() - set(
            self.parameter_requirements()
        )

    def parameter_requirements(self) -> dict[str, str]:
        """A lateral acceleration only with the sublane model: SUMO's
        lane-based model has none."""
        if self._config.lateral_resolution_m is not None:
            return {}
        return {
            "max_lateral_accel": (
                "traffic.options.lateral_resolution_m (SUMO's sublane model, the "
                "only one with a lateral acceleration), which is null"
            )
        }

    def supported_vehicle_overrides(self) -> set[str]:
        """Every vehicle parameter it supports: each is set per vehicle."""
        return self.supported_vehicle_parameters()

    def supported_pedestrian_overrides(self) -> set[str]:
        """The walking speed and the gap; the red light only before the run."""
        return set(_PEDESTRIAN_OVERRIDES)

    def override_entity_parameters(
        self, world: Any, overrides: VehicleParameters, entity: Any
    ) -> None:
        """Give *entity*'s vehicle alone a new draw of the fields *overrides*
        sets: in SUMO when SUMO drives it, else through the TrafficManager."""
        sumo_id = self._id_of(entity)
        if sumo_id is not None:
            if self._vclass(sumo_id) == "bicycle":
                logger.warning(
                    "SUMO: %s is a bicycle, which rides as traffic.options."
                    "bicycles says; the vehicle parameters do not reach it",
                    sumo_id,
                )
                return
            self._draw_vehicle(sumo_id, overrides)
        elif self._tm_drives_scenario:
            shared = overrides.restricted_to(self._tm.supported_vehicle_parameters())
            self._warn_dropped(overrides, shared)
            self._tm.override_entity_parameters(world, shared, entity)

    def override_vehicle_parameters(
        self, world: Any, overrides: VehicleParameters
    ) -> None:
        """Give every vehicle SUMO drives (and the TrafficManager drives for
        the scenario) a new draw of the fields *overrides* sets, and the ones
        that depart later the new parameters."""
        shared = overrides.restricted_to(self._tm.supported_vehicle_parameters())
        self._remember_parameters()
        self._vehicle_parameters = self.vehicle_parameters.overridden_by(overrides)
        if self._tm_drives_scenario:
            self._warn_dropped(overrides, shared)
            self._tm.override_vehicle_parameters(world, shared)
        if not self._running:
            return
        for sumo_id in self._traci.vehicle.getIDList():
            if sumo_id not in self._external and self._vclass(sumo_id) != "bicycle":
                self._draw_vehicle(sumo_id, overrides)
        # The visibility is held anew, from the new draws (or not at all).
        for sumo_id in list(self._max_speeds):
            self._hold_to_sight(sumo_id, None)
        self._visibility = None

    def override_pedestrian_parameters(
        self, world: Any, overrides: PedestrianParameters
    ) -> None:
        """Give every pedestrian of SUMO's own a new draw of the fields
        *overrides* sets, and those who depart later too."""
        del world
        self._remember_parameters()
        self._pedestrian_parameters = self.pedestrian_parameters.overridden_by(
            overrides
        )
        previous = self._pedestrian_overrides
        self._pedestrian_overrides = (
            overrides if previous is None else previous.overridden_by(overrides)
        )
        if not self._running:
            return
        for person in self._traci.person.getIDList():
            self._draw_person(person, overrides)

    @staticmethod
    def _warn_dropped(overrides: VehicleParameters, shared: VehicleParameters) -> None:
        dropped = sorted(set(overrides.set_fields()) - set(shared.set_fields()))
        if dropped:
            logger.warning(
                "SUMO: %s do not reach the scenario's vehicles the TrafficManager "
                "drives",
                ", ".join(dropped),
            )

    def _step_sumo(self, world: Any) -> None:
        """Advance SUMO by one CARLA tick: its steps, each giving the vehicles
        that departed in it their parameters and their sight."""
        for _ in range(self._substeps):
            self._traci.simulationStep()
            self._departed = None
            self._apply_parameters()
            self._follow_visibility(world)

    def _departed_ids(self) -> list[str]:
        """The vehicles that departed in the step SUMO just made."""
        if self._departed is None:
            self._departed = list(self._traci.simulation.getDepartedIDList())
        return self._departed

    @property
    def _follows_visibility(self) -> bool:
        """Whether SUMO's own traffic is held to what its drivers see."""
        follow = self.vehicle_parameters.follow_visibility
        return True if follow is None else follow

    @property
    def port(self) -> int:
        """The TrafficManager port the scenario's vehicles are driven through."""
        return self._config.tm_port

    @property
    def _tm_drives_scenario(self) -> bool:
        return self._config.scenario_vehicles == "traffic_manager"

    @property
    def _walks_scenario_pedestrians(self) -> bool:
        return self._config.scenario_pedestrians == "sumo"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self, context: TrafficContext) -> None:
        """Build the network of the world, generate demand and start SUMO."""
        self.close()
        # Before anything is built: a step that does not fit is a config error.
        step_length, self._substeps = _sumo_step(
            context.fixed_delta_seconds, self._config.step_length_s
        )
        self._context = context
        self._traci = _load_traci(self._config.use_libsumo)
        self._rng = random.Random(context.random_seed)
        # The road users' parameter draws, apart from the routes' stream so
        # that setting a parameter does not change where traffic goes.
        self._draws = random.Random(context.random_seed + 1)
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
                pedestrian_crossings=self._config.pedestrian_crossings,
                cycle_on_sidewalks=self._config.cycle_on_sidewalks,
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
        if self._config.mesoscopic.enabled:
            self._meso = self._build_mesoscopic()

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
            f"{step_length:g}",
            "--seed",
            str(context.random_seed),
            "--collision.action",
            "warn",
            "--no-step-log",
            "true",
            "--log",
            str(out / "sumo.log"),
        ]
        if self._config.lateral_resolution_m is not None:
            cmd += ["--lateral-resolution", f"{self._config.lateral_resolution_m:g}"]
        else:
            # Lane changes take time, as on the road (SUMO's default is one
            # step), so a scenario waiting for one sees it happen.
            duration = self._config.lane_change_duration_s
            cmd += [
                "--lanechange.duration",
                f"{LANE_CHANGE_DURATION_S if duration is None else duration:g}",
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
        pedestrians = self._config.pedestrians
        if pedestrians.enabled:
            routes.append(
                generate_walks(
                    self._network,
                    out / "pedestrians.rou.xml",
                    seed=context.random_seed,
                    period=pedestrians.period(),
                    end=pedestrians.end_s,
                    max_distance=pedestrians.max_distance_m,
                )
            )
        pedestrian_type = self._pedestrian_type(out)
        bicycle_type = self._bicycle_type(out)
        bicycles = self._config.bicycles
        if bicycles.enabled:
            routes.append(
                generate_rides(
                    self._network,
                    out / "bicycles.rou.xml",
                    seed=context.random_seed,
                    period=bicycles.period(),
                    end=bicycles.end_s,
                    min_distance=bicycles.min_distance_m,
                    max_distance=bicycles.max_distance_m,
                    bicycle_type=bicycle_type,
                )
            )
        if self._config.route_path:
            routes.append(Path(self._config.route_path).expanduser())
        types = [
            str(path) for path in (pedestrian_type, bicycle_type) if path is not None
        ]
        if types:
            # Before the demand: SUMO's default types are redefined only
            # before anything uses them.
            cmd += ["-a", ",".join(types)]
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
            if (
                actor.id in skip
                or entity is None
                or self._tm_drives_scenario
                or id(entity) in self._released
            ):
                self._publish(sumo_id, actor)
            else:
                tc.vehicle.setColor(sumo_id, _DRIVEN_COLOR)
                self._driven[sumo_id] = entity
        logger.info(
            "SUMO drives %d of the scenario's vehicle(s); %d driven in CARLA are "
            "published into it",
            len(self._driven),
            len(self._external),
        )
        if self._walks_scenario_pedestrians:
            for entity in self._pedestrian_entities:
                self._take_pedestrian(entity)
            logger.info(
                "SUMO walks %d of the scenario's %d pedestrian(s)",
                len(self._walked),
                len(self._pedestrian_entities),
            )
        if self._config.traffic_light_authority != "none":
            self._signals = self._match_signals(world)
            if self._config.traffic_light_authority == "carla":
                self._crossing_links = self._match_crossings()
                self._bicycle_rides = self._match_bicycle_rides()
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
            self._step_sumo(world)
            if self._clock_log is not None:
                stamp = world.get_snapshot().timestamp
                self._clock_log.write(
                    f"{self._traci.simulation.getTime():.3f},"
                    f"{stamp.elapsed_seconds:.3f},{stamp.frame}\n"
                )
            if self._meso is not None:
                self._step_mesoscopic(world, entities=True)
            self._pull_driven()
            self._sync_ambient(world)
            self._sync_persons(world)
            self._follow_walked()
            self._flush_commands()
            if self._config.traffic_light_authority == "sumo":
                self._signals_to_carla()
            self._failures = 0
        except Exception as exc:  # must not end the run
            # What this step queued is stale by the next one: not sent with it.
            self._commands = []
            self._failures += 1
            logger.warning("SUMO step failed (%d in a row): %s", self._failures, exc)
            if self._failures >= 20:
                logger.error("SUMO keeps failing; traffic stops for this run")
                self._running = False

    def close(self) -> None:
        """Stop SUMO, destroy the ambient actors and forget the run."""
        self._tm.close()
        # The next run starts from its config, not from this one's overrides.
        self._restore_parameters()
        self._pedestrian_overrides = None
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
        if self._meso is not None and self._level_changes != [0, 0]:
            logger.info(
                "SUMO: %d vehicle change(s) to mesoscopic, %d to microscopic this run",
                *self._level_changes,
            )
        for entity in self._meso_entities.values():
            # Out of the run, an entity without an actor is one not spawned.
            self._mark_fidelity(entity, SimulationFidelity.MICROSCOPIC)
        self._meso = None
        self._meso_entities = {}
        self._meso_blueprints = {}
        self._meso_widths = {}
        self._fidelity_due = -math.inf
        self._stays_microscopic = set()
        self._edge_signals = {}
        self._signal_states = {}
        self._edge_shapes = {}
        self._level_changes = [0, 0]
        handed_ambient = [
            self._external[sumo_id]
            for sumo_id in self._handed_ambient
            if sumo_id in self._external
        ]
        for actor in [
            *self._ambient.values(),
            *self._persons.values(),
            *handed_ambient,
        ]:
            try:
                actor.destroy()
            except RuntimeError:
                pass
        self._persons = {}
        self._handed_ambient = {}
        for entity in self._driven.values():
            actor = getattr(entity, "actor", None)
            if actor is not None:
                try:
                    actor.disable_constant_velocity()
                except RuntimeError:
                    pass
        self._entities = []
        self._released = set()
        self._handed = {}
        self._commanded = {}
        self._driven = {}
        self._external = {}
        self._walkers = {}
        self._walkers_off_network = set()
        self._pedestrian_entities = []
        self._released_pedestrians = set()
        self._walked = {}
        self._walking = {}
        self._background = set()
        self._background_count = 0
        self._closed_lanes = {}
        self._blueprints = {}
        self._ambient = {}
        self._vclasses = {}
        self._bicycle_blueprints = None
        self._lane_targets = {}
        self._signals = {}
        self._crossing_links = {}
        self._bicycle_rides = {}
        self._ego_actor = None
        self._failures = 0
        if self._feedback_count:
            logger.info(
                "SUMO: moved SUMO vehicles to where CARLA had them %d time(s) this run",
                self._feedback_count,
            )
        self._physics = {}
        self._commands = []
        self._constant_speed = {}
        self._feedback_count = 0
        self._visibility = None
        self._max_speeds = {}

    def describe(self) -> dict[str, Any]:
        """The backend, its seed, network and demand, for the result record."""
        out = super().describe()
        if self._context is not None:
            out["random_seed"] = self._context.random_seed
        if self._network is not None:
            out["network"] = str(self._network.net_file)
        out["scenario_vehicles"] = self._config.scenario_vehicles
        out["scenario_pedestrians"] = self._config.scenario_pedestrians
        out["vehicle_control"] = self._config.vehicle_control
        out["publish_walkers"] = self._config.publish_walkers
        out["pedestrian_crossings"] = self._config.pedestrian_crossings
        out["cycle_on_sidewalks"] = self._config.cycle_on_sidewalks
        if self._config.vehicle_control == "physics":
            out["feedback_distance_m"] = self._config.feedback_distance_m
        out["ambient_period_s"] = (
            self._config.ambient.period() if self._config.ambient.enabled else None
        )
        out["traffic_light_authority"] = self._config.traffic_light_authority
        if self._config.pedestrians.enabled:
            out["pedestrians_period_s"] = self._config.pedestrians.period()
        bicycles = self._config.bicycles
        if bicycles.enabled:
            out["bicycles_period_s"] = bicycles.period()
            out["bicycle_control"] = bicycles.vehicle_control
        riding = bicycles.riding()
        if riding:
            out["bicycles"] = {
                name: spread.describe() for name, spread in riding.items()
            }
        meso = self._config.mesoscopic
        if meso.enabled:
            out["mesoscopic"] = {
                "micro_radius_m": meso.micro_radius_m,
                "meso_radius_m": meso.demote_radius_m,
                "scenario_vehicles": meso.scenario_vehicles,
                # So far this run: what a result read at its end records.
                "to_mesoscopic": self._level_changes[0],
                "to_microscopic": self._level_changes[1],
            }
        out["follow_visibility"] = self._follows_visibility
        out["visibility_m"] = self._visibility
        return out

    def fidelity_census(self) -> dict[str, int]:
        """SUMO's vehicles are microscopic, the queue model's mesoscopic."""
        if not self._running:
            return {}
        return {
            SimulationFidelity.MICROSCOPIC.value: int(self._traci.vehicle.getIDCount()),
            SimulationFidelity.MESOSCOPIC.value: len(self._meso or ()),
        }

    def require_microscopic(self, entity: Any, world: Any) -> bool:
        """Bring a mesoscopic *entity* back into SUMO and CARLA now, and keep
        it there for the rest of the run."""
        sumo_id = self._meso_id_of(entity)
        if sumo_id is None:
            sumo_id = self._id_of(entity)
            if sumo_id is not None:
                self._stays_microscopic.add(sumo_id)
            return super().require_microscopic(entity, world)
        if not self._to_microscopic(world, sumo_id, self._now()):
            return False
        self._stays_microscopic.add(sumo_id)
        return True

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
        if self._meso_id_of(entity) is not None:
            logger.warning(
                "SUMO: %s is simulated mesoscopically, which has no lanes; it "
                "does not change lane",
                self._meso_id_of(entity),
            )
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
        speed = max(speed_kmh, 0.0) / 3.6
        meso_id = self._meso_id_of(entity)
        if meso_id is not None and self._meso is not None:
            self._meso.set_speed_cap(meso_id, speed, self._now())
            self._commanded[meso_id] = speed
            return
        sumo_id = self._driven_id(entity, "set_desired_speed")
        if sumo_id is None:
            return
        try:
            self._traci.vehicle.setSpeed(sumo_id, speed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SUMO could not set the speed of %s: %s", sumo_id, exc)
            return
        self._commanded[sumo_id] = speed

    def turn_at_junction(
        self, entity: Any, world: Any, direction: TurnDirection, **kwargs: Any
    ) -> None:
        """Route *entity* *direction* at the next junction offering that turn."""
        if self._id_of(entity) is None and self._tm_drives_scenario:
            self._tm.turn_at_junction(entity, world, direction, **kwargs)
            return
        meso_id = self._meso_id_of(entity)
        if meso_id is not None and self._meso is not None:
            turn = self._turn_route(self._meso.vehicle(meso_id).link, direction)
            if turn is None:
                logger.warning("SUMO: no %s turn ahead of %s", direction.value, meso_id)
            else:
                self._meso.set_route(meso_id, turn)
            return
        sumo_id = self._driven_id(entity, "turn_at_junction")
        if sumo_id is None:
            return
        try:
            edge = self._traci.vehicle.getRoadID(sumo_id)
            route = self._turn_route(edge, direction, self._vclass(sumo_id))
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

    def release(self, entity: Any, world: Any) -> None:
        """Stop driving *entity* in SUMO and publish it there instead.

        The vehicle stays in SUMO -- SUMO's own traffic still has to see it --
        but as one CARLA drives, like the ego: its pose is pushed in every tick
        rather than pulled out.
        """
        if self._tm_drives_scenario:
            self._tm.release(entity, world)
            return
        self._released.add(id(entity))
        if self._meso_id_of(entity) is not None:
            # CARLA drives it from now on, which needs it in CARLA.
            self.require_microscopic(entity, world)
        sumo_id = self._id_of(entity)
        if sumo_id is None or not self._running:
            # Not in SUMO yet: start() publishes it rather than driving it.
            return
        self._driven.pop(sumo_id, None)
        actor = getattr(entity, "actor", None)
        if actor is None:
            return
        modes = self._hand_over(sumo_id, actor)
        if modes is not None:
            self._handed[id(entity)] = (sumo_id, modes[0], modes[1])
        logger.info("SUMO: %s released; it is published from CARLA now", sumo_id)

    def resume(self, entity: Any, world: Any) -> None:
        """Drive *entity* in SUMO again, from where CARLA has it now.

        While released its pose was pushed into SUMO every tick, so SUMO
        takes it over where the action left it, with the speed and
        lane-change modes it had and the speed set_desired_speed() last held
        it at; its pose is pulled out each tick again.
        """
        if self._tm_drives_scenario:
            self._tm.resume(entity, world)
            return
        self._released.discard(id(entity))
        handed = self._handed.pop(id(entity), None)
        if handed is None or not self._running:
            # Never released while SUMO drove it: start() drives it, or it
            # was never SUMO's to drive.
            return
        sumo_id, speed_mode, lane_change_mode = handed
        self._take_back(
            sumo_id,
            (speed_mode, lane_change_mode),
            self._commanded.get(sumo_id, -1.0),
            _DRIVEN_COLOR,
        )
        self._driven[sumo_id] = entity
        logger.info("SUMO: %s resumed; SUMO drives it again", sumo_id)

    # ------------------------------------------------------------------
    # The scenario's pedestrians
    # ------------------------------------------------------------------

    def adopt_pedestrian(self, entity: Any) -> None:
        """Remember *entity*; with ``scenario_pedestrians=sumo`` SUMO walks it
        from when the run starts."""
        self._pedestrian_entities.append(entity)

    def walk_straight(
        self, entity: Any, speed_ms: float, yaw_deg: Optional[float] = None
    ) -> bool:
        """Walk a scenario pedestrian SUMO walks straight ahead at *speed_ms*,
        the way its walker faces (or along *yaw_deg*, a CARLA yaw); zero
        stands it where it is.

        The walk is TraCI's, not SUMO's pedestrian model: each step the person
        is moved on along the line (``person.moveToXY``, onto whatever lane is
        there, so SUMO's traffic sees it on the road), and its walker follows
        it -- off the kerb and across the road where it stands, as
        ``WalkStraightAction`` means, until it is stopped.
        """
        person = self._walked_id(entity)
        if person is None or not self._running:
            return False
        actor = getattr(entity, "actor", None)
        if actor is None:
            return False
        role = str(getattr(entity, "role_name", person))
        if speed_ms <= 0.0:
            self._walking.pop(person, None)
            self._walk(actor, self._person_carla_pose(person), 0.0)
            logger.info("SUMO: pedestrian '%s' stands where it is", role)
            return True
        pose = self._walker_pose(actor, yaw_deg)
        self._walking[person] = (pose.x, pose.y, pose.heading_deg, float(speed_ms))
        logger.info(
            "SUMO walks pedestrian '%s' straight on at %.2f m/s", role, speed_ms
        )
        return True

    def release_pedestrian(self, entity: Any, world: Any) -> None:
        """Stop walking *entity* in SUMO and publish it there instead: an
        action moves its walker itself from now on."""
        del world
        self._released_pedestrians.add(id(entity))
        person = self._walked_id(entity)
        if person is None or not self._running:
            # Not in SUMO yet: start() leaves it to be published.
            return
        self._publish_walked(person)
        logger.info(
            "SUMO: pedestrian '%s' released; it is published from CARLA now",
            str(getattr(entity, "role_name", person)),
        )

    def _walked_id(self, entity: Any) -> Optional[str]:
        for person, walked in self._walked.items():
            if walked is entity:
                return person
        return None

    def _take_pedestrian(self, entity: Any) -> None:
        """Put the scenario's pedestrian *entity* into SUMO as a person SUMO
        walks.  Until it is sent on a walk it stands where its walker stands
        (:meth:`_push_walkers` puts it there every tick)."""
        actor = getattr(entity, "actor", None)
        if actor is None or id(entity) in self._released_pedestrians:
            return
        person = self._new_person(actor, self._walker_pose(actor))
        if person is None:
            # No footway near (warned): CARLA walks it, and SUMO cannot see it.
            return
        self._traci.person.setColor(person, _DRIVEN_COLOR)
        self._walked[person] = entity

    def _publish_walked(self, person: str) -> None:
        """Stop walking *person*: its walker is published into SUMO from now
        on, as any CARLA pedestrian is (or taken out, without
        ``publish_walkers``)."""
        entity = self._walked.pop(person)
        self._walking.pop(person, None)
        actor = getattr(entity, "actor", None)
        if actor is None or not self._config.publish_walkers:
            self._traci.person.remove(person)
            return
        self._traci.person.setColor(person, _PUBLISHED_COLOR)
        self._walkers[actor.id] = person

    def _walked_pose(self, person: str, actor: Any) -> Pose2D:
        """Where *person*, SUMO walks, is to be this step (SUMO's frame): on
        along its walk, or, standing, where its walker is."""
        walk = self._walking.get(person)
        if walk is None:
            return self._walker_pose(actor)
        x, y, heading_deg, speed = walk
        dt = self._context.fixed_delta_seconds if self._context is not None else 0.05
        # SUMO's angle turns clockwise from north.
        heading = math.radians(90.0 - heading_deg)
        x += speed * dt * math.cos(heading)
        y += speed * dt * math.sin(heading)
        self._walking[person] = (x, y, heading_deg, speed)
        return Pose2D(x, y, 0.0, heading_deg)

    def _follow_walked(self) -> None:
        """Walk the walkers of the scenario's pedestrians out on a walk to
        where SUMO has their persons."""
        for person, (_x, _y, _heading, speed) in self._walking.items():
            entity = self._walked.get(person)
            actor = getattr(entity, "actor", None) if entity is not None else None
            if actor is not None:
                self._walk(actor, self._person_carla_pose(person), speed)

    def background_actors(self, world: Any) -> dict[str, Any]:
        """The CARLA mirror of every ambient SUMO vehicle, by its role name
        (``sumo:<id>``): what an adversity may take over (#417)."""
        del world
        if not self._running:
            return {}
        return {
            f"{_MIRROR_PREFIX}{sumo_id}": actor
            for sumo_id, actor in self._ambient.items()
        }

    def _hand_over(self, sumo_id: str, actor: Any) -> Optional[tuple[int, int]]:
        """Have CARLA drive *sumo_id*'s *actor* from now on, published into
        SUMO; the speed and lane-change modes it had, or None when SUMO
        could not say (it is published all the same)."""
        self._constant_speed.pop(actor.id, None)
        try:
            actor.disable_constant_velocity()
        except RuntimeError:
            pass
        try:
            tc = self._traci
            modes = (
                int(tc.vehicle.getSpeedMode(sumo_id)),
                int(tc.vehicle.getLaneChangeMode(sumo_id)),
            )
            self._publish(sumo_id, actor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SUMO could not hand %s over: %s", sumo_id, exc)
            self._external[sumo_id] = actor
            return None
        return modes

    def _take_back(
        self,
        sumo_id: str,
        modes: tuple[int, int],
        speed: float,
        color: Optional[tuple[int, int, int, int]] = None,
    ) -> bool:
        """Have SUMO drive *sumo_id* again with *modes* (``-1`` keeps them)
        at *speed*; whether it could (the vehicle may have left SUMO)."""
        self._external.pop(sumo_id, None)
        try:
            tc = self._traci
            if modes[0] >= 0:
                tc.vehicle.setSpeedMode(sumo_id, modes[0])
                tc.vehicle.setLaneChangeMode(sumo_id, modes[1])
            tc.vehicle.setSpeed(sumo_id, speed)
            if color is not None:
                tc.vehicle.setColor(sumo_id, color)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SUMO could not take %s back: %s", sumo_id, exc)
            return False
        return True

    def release_actor(self, world: Any, actor: Any) -> None:
        """Take an ambient vehicle's mirror over for CARLA (#417).  It stays
        in SUMO, published like a released entity, and is no longer followed
        nor removed by the ambient sync."""
        del world
        sumo_id = self._ambient_id_of(actor)
        if sumo_id is None:
            return
        self._ambient.pop(sumo_id)
        self._physics.pop(actor.id, None)
        modes = self._hand_over(sumo_id, actor)
        self._handed_ambient[sumo_id] = (-1, -1) if modes is None else modes
        logger.info(
            "SUMO: ambient %s released; it is published from CARLA now", sumo_id
        )

    def resume_actor(self, world: Any, actor: Any) -> None:
        """Give a released ambient vehicle back to SUMO, which drives it on
        from where CARLA has it with its modes as they were; the ambient sync
        follows (or removes) its mirror again."""
        del world
        sumo_id = self._mirrored_id(actor)
        if sumo_id is None or sumo_id not in self._handed_ambient:
            return
        modes = self._handed_ambient.pop(sumo_id)
        if not self._take_back(sumo_id, modes, -1.0):
            # Gone from SUMO while CARLA drove it (its route ended): the
            # mirror has nothing left to follow.
            try:
                actor.destroy()
            except RuntimeError:
                pass
            return
        self._ambient[sumo_id] = actor
        logger.info("SUMO: ambient %s resumed; SUMO drives it again", sumo_id)

    @staticmethod
    def _mirrored_id(actor: Any) -> Optional[str]:
        """The SUMO id a mirror's role name names, if it is one."""
        role = str(actor.attributes.get("role_name", ""))
        if not role.startswith(_MIRROR_PREFIX):
            return None
        return role[len(_MIRROR_PREFIX) :]

    def _ambient_id_of(self, actor: Any) -> Optional[str]:
        """*actor*'s SUMO id, if it is an ambient vehicle's mirror now."""
        sumo_id = self._mirrored_id(actor)
        if sumo_id is None or self._ambient.get(sumo_id) is not actor:
            return None
        return sumo_id

    def _publish(self, sumo_id: str, actor: Any) -> None:
        """Have SUMO stop driving *sumo_id*: CARLA's *actor* is pushed into it."""
        tc = self._traci
        tc.vehicle.setSpeedMode(sumo_id, _SPEED_MODE_EXTERNAL)
        tc.vehicle.setLaneChangeMode(sumo_id, _LANE_CHANGE_MODE_EXTERNAL)
        tc.vehicle.setColor(sumo_id, _PUBLISHED_COLOR)
        self._external[sumo_id] = actor

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
        # The run's map where one is held: get_map() builds a whole new one.
        return str(run_carla_map(context.world).to_opendrive())

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
        self, pose: Pose2D, sumo_id: str, vclass: str = "passenger"
    ) -> Optional[tuple[str, float, int]]:
        """The (edge, position, lane index) a vehicle of *vclass* at SUMO
        *pose* starts from.

        ``None``, with a warning, where there is no road for it there.
        """
        tc = self._traci
        try:
            edge, pos, lane = tc.simulation.convertRoad(pose.x, pose.y, False, vclass)
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
        """Add *actor* to SUMO where it stands; ``False`` if it is off the network.

        A bicycle (:func:`is_bicycle`) as one of SUMO's default bicycle type,
        on the nearest lane a bicycle may use -- a sidewalk's, when bicycles
        ride them.
        """
        tc = self._traci
        bicycle = is_bicycle(actor)
        vclass = "bicycle" if bicycle else "passenger"
        place = self._road_position(self._carla_pose_to_sumo(actor), sumo_id, vclass)
        if place is None:
            return False
        edge, pos, lane = place
        route_id = f"route:{sumo_id}"
        try:
            tc.route.add(route_id, self._continue_route([edge], vclass=vclass))
            tc.vehicle.add(
                sumo_id,
                route_id,
                typeID=_BICYCLE_TYPE if bicycle else "DEFAULT_VEHTYPE",
                departLane=str(lane),
                departPos=f"{pos:.2f}",
                departSpeed="0",
            )
        except Exception as exc:  # noqa: BLE001 - one vehicle must not end the run
            logger.warning(
                "SUMO could not add %s on %s_%d: %s", sumo_id, edge, lane, exc
            )
            return False
        self._vclasses[sumo_id] = vclass
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

    def close_lane(self, world: Any, transform: Any) -> Optional[str]:
        """Disallow every vehicle class on the SUMO lane at *transform*, as
        TeraSim's full-lane construction zone does (``setDisallowed``); the
        lane's id, or None off the network."""
        del world
        if not self._running:
            return None
        location, rotation = transform.location, transform.rotation
        pose = carla_to_sumo(
            Pose2D(location.x, location.y, location.z, rotation.yaw),
            0.0,
            self._offset,
        )
        tc = self._traci
        try:
            # Whatever the lane allows: one this zone (or another) closed
            # already is no passenger lane any more, and a passenger match
            # would close the lane beside it.
            edge, _pos, lane = tc.simulation.convertRoad(
                pose.x, pose.y, False, "ignoring"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("A construction zone is off the SUMO network: %s", exc)
            return None
        lane_id = f"{edge}_{lane}"
        if lane_id in self._closed_lanes:
            return lane_id
        try:
            allowed = list(tc.lane.getAllowed(lane_id))
            tc.lane.setDisallowed(lane_id, ["all"])
        except Exception as exc:  # noqa: BLE001 - a closure must not end the run
            logger.warning("SUMO could not close lane %s: %s", lane_id, exc)
            return None
        self._closed_lanes[lane_id] = allowed
        logger.info("SUMO: lane %s closed", lane_id)
        return lane_id

    def reopen_lane(self, world: Any, handle: str) -> None:
        """Allow on lane *handle* again what it allowed before it was closed."""
        del world
        allowed = self._closed_lanes.pop(handle, None)
        if allowed is None or not self._running:
            return
        try:
            self._traci.lane.setAllowed(handle, allowed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SUMO could not reopen lane %s: %s", handle, exc)
            return
        logger.info("SUMO: lane %s reopened", handle)

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
        meso = self._meso if self._meso is not None else ()
        self._background &= (
            present | pending | {i for i in self._background if i in meso}
        )
        out: dict[str, tuple[float, float]] = {}
        if self._meso is not None:
            now = self._now()
            for sumo_id in self._background:
                if sumo_id in self._meso:
                    pose = self._meso_carla_pose(sumo_id, now)
                    out[sumo_id] = (pose.x, pose.y)
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
        if self._meso is not None and self._meso.remove(handle) is not None:
            self._blueprints.pop(handle, None)
            self._meso_widths.pop(handle, None)
            return
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

    def _continue_route(
        self, route: list[str], edges: int = 3, vclass: str = "passenger"
    ) -> list[str]:
        """*route* extended by up to *edges* edges a vehicle of *vclass* may
        use, straight on where possible."""
        route = list(route)
        for _ in range(edges):
            last = self._net.getEdge(route[-1])
            outgoing = [
                (to_edge, conns)
                for to_edge, conns in last.getOutgoing().items()
                if to_edge.allows(vclass)
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

    def _turn_route(
        self, edge: str, direction: TurnDirection, vclass: str = "passenger"
    ) -> Optional[list[str]]:
        if edge.startswith(":"):
            return None
        route = [edge]
        for _ in range(TURN_SEARCH_EDGES):
            last = self._net.getEdge(route[-1])
            for to_edge, conns in last.getOutgoing().items():
                if not to_edge.allows(vclass):
                    continue
                if any(c.getDirection() in _TURNS[direction.value] for c in conns):
                    return self._continue_route(
                        [*route, to_edge.getID()], vclass=vclass
                    )
            ahead = self._continue_route(route, 1, vclass=vclass)
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
            vehicle.setRoute(
                sumo_id,
                self._continue_route(route[index:], vclass=self._vclass(sumo_id)),
            )

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
        """Put every CARLA pedestrian into SUMO where it stands, as a person,
        and each of the scenario's pedestrians SUMO walks where it is to be.

        Placed again each step -- a person SUMO is not moving any more would
        walk off on its own -- and removed once its actor is gone.
        """
        tc = self._traci
        present = {a.id: a for a in world.get_actors().filter("walker.pedestrian.*")}
        # The scenario's pedestrians SUMO walks leave SUMO with their walkers.
        for walked, entity in list(self._walked.items()):
            actor = getattr(entity, "actor", None)
            if actor is None or actor.id not in present:
                del self._walked[walked]
                self._walking.pop(walked, None)
                try:
                    tc.person.remove(walked)
                except Exception as exc:  # noqa: BLE001 - it may have left already
                    logger.debug("SUMO: removing a person: %s", exc)
                continue
            pose = self._walked_pose(walked, actor)
            tc.person.moveToXY(
                walked, "", pose.x, pose.y, pose.heading_deg, _KEEP_ROUTE_ANYWHERE
            )
        if not self._config.publish_walkers:
            return
        # Not the walkers mirroring SUMO's own pedestrians, nor the ones
        # following a person SUMO walks: those are in SUMO.
        mirrors = {actor.id for actor in self._persons.values()}
        mirrors.update(entity.actor.id for entity in self._walked.values())
        walkers = {
            actor_id: actor
            for actor_id, actor in present.items()
            if actor_id not in mirrors
        }
        for actor_id in [i for i in self._walkers if i not in walkers]:
            try:
                tc.person.remove(self._walkers.pop(actor_id))
            except Exception as exc:  # noqa: BLE001 - it may have left already
                logger.debug("SUMO: removing a person: %s", exc)
        for actor_id, actor in walkers.items():
            pose = self._walker_pose(actor)
            person = self._walkers.get(actor_id) or self._add_person(actor, pose)
            if person is not None:
                tc.person.moveToXY(
                    person, "", pose.x, pose.y, pose.heading_deg, _KEEP_ROUTE_ANYWHERE
                )

    def _walker_pose(self, actor: Any, yaw_deg: Optional[float] = None) -> Pose2D:
        """Where CARLA's walker *actor* is, in SUMO's frame; facing *yaw_deg*
        (CARLA's) when given, else the way it faces."""
        tf = actor.get_transform()
        yaw = tf.rotation.yaw if yaw_deg is None else yaw_deg
        return carla_to_sumo(
            Pose2D(tf.location.x, tf.location.y, tf.location.z, yaw),
            0.0,
            self._offset,
        )

    def _add_person(self, actor: Any, pose: Pose2D) -> Optional[str]:
        """Publish *actor* as a person on the nearest edge; ``None`` if none is near."""
        person = self._new_person(actor, pose)
        if person is not None:
            self._traci.person.setColor(person, _PUBLISHED_COLOR)
            self._walkers[actor.id] = person
        return person

    def _new_person(self, actor: Any, pose: Pose2D) -> Optional[str]:
        """Add a person for *actor* on the nearest edge, standing; ``None`` if
        none is near."""
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
        # Stands where it is put: a published person is moved every step, and
        # one SUMO walks stands until walk_straight() sends it on a walk.
        tc.person.appendWaitingStage(person, 1e7)
        extent = actor.bounding_box.extent
        tc.person.setLength(person, max(2.0 * float(extent.x), 0.2))
        tc.person.setWidth(person, max(2.0 * float(extent.y), 0.2))
        return person

    def _follow(
        self, actor: Any, sumo_id: str, length: float, control: Optional[str] = None
    ) -> None:
        """Carry *actor* towards where its SUMO vehicle is after this step,
        as *control* says (``vehicle_control`` when ``None``)."""
        if (control or self._config.vehicle_control) == "physics":
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
            # On a lane of its own class: a bicycle's may be a sidewalk's.
            edge, pos, lane = tc.simulation.convertRoad(
                pose.x, pose.y, False, self._vclass(sumo_id)
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
        module docstring for why not a physics-off actor).  The pose goes into
        this tick's batch; the velocity, which has no batch command, is sent
        only when it is not the one the actor already has -- it is kept in the
        actor's own frame, so the new pose does not change it.  That holds as
        long as only this backend turns it off (release(), close(),
        which forget it): whatever else disables an actor's constant velocity
        must forget its entry in _constant_speed too.
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
        transform = carla.Transform(
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
        if self._context is not None and self._context.client is not None:
            self._commands.append(carla.command.ApplyTransform(actor.id, transform))
        else:
            actor.set_transform(transform)
        if self._constant_speed.get(actor.id) != speed:
            actor.enable_constant_velocity(carla.Vector3D(speed, 0.0, 0.0))
            self._constant_speed[actor.id] = speed

    def _pull_driven(self) -> None:
        present = set(self._traci.vehicle.getIDList())
        for sumo_id, entity in self._driven.items():
            actor = getattr(entity, "actor", None)
            if actor is None or sumo_id not in present:
                continue
            self._follow(actor, sumo_id, self._length(actor), self._control(sumo_id))
            self._keep_route_ahead(sumo_id)
        for sumo_id in self._external:
            if sumo_id in present:
                self._keep_route_ahead(sumo_id)

    def _ambient_ids(self) -> list[str]:
        """The ambient SUMO vehicles to mirror, nearest to the ego first.

        Cars within ``spawn_radius_m``, at most ``max_vehicles`` of them, and
        bicycles within ``bicycles.radius_m``, at most
        ``bicycles.max_bicycles``: each counts against its own limit.
        """
        tc = self._traci
        own = set(self._driven) | set(self._external)
        ids = [i for i in tc.vehicle.getIDList() if i not in own]
        bicycles = self._config.bicycles
        bikes = [i for i in ids if self._vclass(i) == "bicycle"]
        cars = [i for i in ids if self._vclass(i) != "bicycle"]
        radius = bicycles.radius_m or self._config.spawn_radius_m
        return self._nearest(
            cars, self._config.spawn_radius_m, self._config.max_vehicles
        ) + self._nearest(bikes, radius, bicycles.max_bicycles)

    def _nearest(
        self, ids: list[str], radius: Optional[float], limit: Optional[int]
    ) -> list[str]:
        """Those of *ids* within *radius* of the ego, at most *limit* of them,
        nearest first."""
        tc = self._traci
        if self._ego_actor is None:
            # Nothing to measure from: the first ones SUMO lists.
            return ids[:limit] if limit else ids
        if not (radius or limit):
            return ids
        ego = self._carla_pose_to_sumo(self._ego_actor)
        dist = {i: math.dist((ego.x, ego.y), tc.vehicle.getPosition(i)) for i in ids}
        ids = sorted(ids, key=dist.__getitem__)
        if radius:
            ids = [i for i in ids if dist[i] <= radius]
        if limit:
            ids = ids[:limit]
        return ids

    def _vclass(self, sumo_id: str) -> str:
        """The vehicle class of *sumo_id*, read once; ``passenger`` if SUMO
        cannot say."""
        vclass = self._vclasses.get(sumo_id)
        if vclass is None:
            try:
                vclass = str(self._traci.vehicle.getVehicleClass(sumo_id))
            except Exception:  # noqa: BLE001 - gone, or never in
                return "passenger"
            self._vclasses[sumo_id] = vclass
        return vclass

    def _control(self, sumo_id: str) -> str:
        """How the CARLA actor of *sumo_id* follows it: ``bicycles``' way for
        a bicycle, ``vehicle_control`` for the rest."""
        if self._vclass(sumo_id) == "bicycle":
            return self._config.bicycles.vehicle_control
        return self._config.vehicle_control

    def _mirror_blueprint(self, library: Any, sumo_id: str) -> Optional[str]:
        """The blueprint *sumo_id*'s CARLA mirror is spawned as; ``None`` for
        a bicycle when the world has none of ``bicycles.blueprints``."""
        blueprint = self._blueprints.get(sumo_id)
        if blueprint is not None:
            return blueprint
        if self._vclass(sumo_id) != "bicycle":
            return self._config.blueprint
        if self._bicycle_blueprints is None:
            self._bicycle_blueprints = []
            for name in self._config.bicycles.blueprints:
                try:
                    library.find(name)
                except (IndexError, RuntimeError):
                    continue
                self._bicycle_blueprints.append(name)
            if not self._bicycle_blueprints:
                logger.warning(
                    "SUMO: the world has none of the bicycle blueprints %s; "
                    "SUMO's bicycles ride in SUMO only",
                    ", ".join(self._config.bicycles.blueprints) or "(none given)",
                )
        if not self._bicycle_blueprints:
            return None
        # Kept, so a bicycle mirrored again comes back as the same bicycle.
        blueprint = self._blueprints[sumo_id] = self._rng.choice(
            self._bicycle_blueprints
        )
        return blueprint

    def _sync_ambient(self, world: Any) -> None:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        tc = self._traci
        wanted = self._ambient_ids()
        wanted_set = set(wanted)
        for sumo_id in [i for i in self._ambient if i not in wanted_set]:
            gone = self._ambient.pop(sumo_id)
            self._physics.pop(gone.id, None)
            self._constant_speed.pop(gone.id, None)
            try:
                gone.destroy()
            except RuntimeError:
                pass
        # The whole actor-definition list, over RPC: fetched on a tick that
        # spawns, not on every tick.
        library = None
        for sumo_id in wanted:
            length = tc.vehicle.getLength(sumo_id)
            control = self._control(sumo_id)
            actor = self._ambient.get(sumo_id)
            if actor is None:
                if library is None:
                    library = world.get_blueprint_library()
                blueprint_id = self._mirror_blueprint(library, sumo_id)
                if blueprint_id is None:
                    continue
                x, y, z = tc.vehicle.getPosition3D(sumo_id)
                pose = sumo_to_carla(
                    Pose2D(x, y, z, tc.vehicle.getAngle(sumo_id)), length, self._offset
                )
                blueprint = library.find(blueprint_id)
                if blueprint.has_attribute("role_name"):
                    blueprint.set_attribute("role_name", f"{_MIRROR_PREFIX}{sumo_id}")
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
                if control == "physics":
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
            self._follow(actor, sumo_id, length, control)

    # ------------------------------------------------------------------
    # Internals: mesoscopic traffic
    # ------------------------------------------------------------------

    def _build_mesoscopic(self) -> MesoscopicTraffic:
        """The queue model on every road of the network a car may drive."""
        links: dict[str, MesoLink] = {}
        signals: dict[tuple[str, str], list[tuple[str, int]]] = {}
        for edge in self._net.getEdges():
            if edge.getFunction() != "" or not edge.allows("passenger"):
                continue
            lanes = [lane for lane in edge.getLanes() if lane.allows("passenger")]
            links[edge.getID()] = MesoLink(
                edge.getID(),
                float(edge.getLength()),
                len(lanes),
                float(edge.getSpeed()),
            )
            for to_edge, conns in edge.getOutgoing().items():
                for conn in conns:
                    if conn.getTLSID():
                        signals.setdefault((edge.getID(), to_edge.getID()), []).append(
                            (conn.getTLSID(), int(conn.getTLLinkIndex()))
                        )
        self._edge_signals = signals
        config = self._config.mesoscopic
        return MesoscopicTraffic(
            links,
            headway_s=config.headway_s,
            jam_spacing_m=config.jam_spacing_m,
            gate=self._meso_gate if config.junction_control else None,
        )

    def _meso_gate(self, from_edge: str, to_edge: str) -> bool:
        """Whether SUMO's signals let a vehicle from *from_edge* on to *to_edge*.

        Green on any lane of the move will do: the queue model has no lanes.
        """
        links = self._edge_signals.get((from_edge, to_edge))
        if not links:
            return True
        for tls, index in links:
            state = self._signal_states.get(tls)
            if state is None:
                state = str(self._traci.trafficlight.getRedYellowGreenState(tls))
                self._signal_states[tls] = state
            if index < len(state) and state[index] in "Gg":
                return True
        return False

    def _now(self) -> float:
        return float(self._traci.simulation.getTime()) if self._running else 0.0

    def _meso_id_of(self, entity: Any) -> Optional[str]:
        for sumo_id, held in self._meso_entities.items():
            if held is entity:
                return sumo_id
        return None

    def _step_mesoscopic(self, world: Any, *, entities: bool) -> None:
        """Step the queue model, and move vehicles between it and SUMO when due.

        Args:
            world: The CARLA world.
            entities: Whether the scenario's vehicles may change level now.
        """
        meso = self._meso
        if meso is None:
            return
        now = self._now()
        self._signal_states = {}
        for sumo_id in self._meso_entities:
            # A scenario's vehicle must not arrive, as in SUMO.
            vehicle = meso.vehicle(sumo_id)
            if vehicle.index >= len(vehicle.route) - 2:
                meso.set_route(sumo_id, self._continue_route(vehicle.remaining_route))
        for vehicle in meso.step(now):
            if vehicle.id in self._meso_entities:
                # A dead end: it waits where its road ends.
                meso.add(
                    vehicle.id,
                    [vehicle.link],
                    now=now,
                    offset_m=meso.link(vehicle.link).length,
                    max_speed=vehicle.max_speed,
                    length=vehicle.length,
                    type_id=vehicle.type_id,
                    speed_cap=0.0,
                )
                continue
            self._background.discard(vehicle.id)
            self._blueprints.pop(vehicle.id, None)
            self._commanded.pop(vehicle.id, None)
            self._meso_widths.pop(vehicle.id, None)
        if now >= self._fidelity_due:
            self._fidelity_due = now + self._config.mesoscopic.update_interval_s
            self._rebalance_fidelity(world, now, entities=entities)

    def _rebalance_fidelity(self, world: Any, now: float, *, entities: bool) -> None:
        """Make the vehicles near a published one microscopic, the rest not.

        The focus points are the ego, whoever drives it, and every vehicle
        CARLA drives -- the scenario's vehicles the TrafficManager or an action
        drives: SUMO's traffic has to react to each of them, which takes lanes.
        """
        meso = self._meso
        if meso is None:
            return
        focus = self._focus_points()
        if not focus:
            return
        tc = self._traci
        candidates = []
        for sumo_id in tc.vehicle.getIDList():
            # A bicycle stays in SUMO: the queue model's roads are the
            # carriageways, and a bicycle rides the sidewalks too.
            if sumo_id in self._external or self._vclass(sumo_id) == "bicycle":
                continue
            if sumo_id in self._driven and not (
                entities and self._may_leave_world(sumo_id)
            ):
                continue
            x, y = tc.vehicle.getPosition(sumo_id)
            candidates.append(
                FidelityCandidate(sumo_id, SimulationFidelity.MICROSCOPIC, x, y)
            )
        for sumo_id in meso:
            if sumo_id in self._meso_entities and not entities:
                continue
            pose = self._meso_pose(sumo_id, now)
            candidates.append(
                FidelityCandidate(
                    sumo_id, SimulationFidelity.MESOSCOPIC, pose.x, pose.y
                )
            )
        to_meso, to_micro = plan_fidelity(self._config.mesoscopic, candidates, focus)
        for sumo_id in to_meso:
            self._to_mesoscopic(sumo_id, now)
        for sumo_id in to_micro:
            self._to_microscopic(world, sumo_id, now)

    def _focus_points(self) -> list[tuple[float, float]]:
        """Where traffic must be microscopic around, in SUMO's frame: the ego,
        whoever drives it, and every vehicle CARLA drives."""
        focus = []
        actors = list(self._external.values())
        if self._ego_actor is not None and self._ego_actor not in actors:
            actors.append(self._ego_actor)  # an autopilot ego SUMO drives
        for actor in actors:
            try:
                pose = self._carla_pose_to_sumo(actor)
            except RuntimeError:  # destroyed under us
                continue
            focus.append((pose.x, pose.y))
        return focus

    def _may_leave_world(self, sumo_id: str) -> bool:
        """Whether the scenario's vehicle *sumo_id* may become mesoscopic."""
        return (
            self._config.mesoscopic.scenario_vehicles
            and sumo_id not in self._stays_microscopic
            and hasattr(self._driven[sumo_id], "_replace_actor")
        )

    def _to_mesoscopic(self, sumo_id: str, now: float) -> bool:
        """Move *sumo_id* from SUMO, and CARLA, into the queue model.

        Not from inside a junction -- the queue model's roads are SUMO's
        normal edges -- which leaves it for the next decision.  Answers
        whether it moved.
        """
        meso = self._meso
        if meso is None:
            return False
        tc = self._traci
        try:
            edge = str(tc.vehicle.getRoadID(sumo_id))
            if not meso.has_link(edge):
                return False
            route = list(tc.vehicle.getRoute(sumo_id))
            index = int(tc.vehicle.getRouteIndex(sumo_id))
            offset = float(tc.vehicle.getLanePosition(sumo_id))
            max_speed = float(tc.vehicle.getMaxSpeed(sumo_id))
            length = float(tc.vehicle.getLength(sumo_id))
            width = float(tc.vehicle.getWidth(sumo_id))
            # A vehicle given a length of its own has a type of its own
            # (`<type>@<id>`), which goes with it; its length and width are
            # put back when it returns.
            type_id = str(tc.vehicle.getTypeID(sumo_id)).split("@", 1)[0]
        except Exception as exc:  # noqa: BLE001 - one vehicle must not end the run
            logger.debug("SUMO: %s stays microscopic: %s", sumo_id, exc)
            return False
        if not (0 <= index < len(route) and route[index] == edge):
            route, index = [edge], 0
        entity = self._driven.get(sumo_id)
        if entity is not None:
            actor = getattr(entity, "actor", None)
            if not entity._replace_actor(None):
                self._stays_microscopic.add(sumo_id)
                return False
            del self._driven[sumo_id]
            self._lane_targets.pop(sumo_id, None)
            self._meso_entities[sumo_id] = entity
            self._meso_blueprints[sumo_id] = str(
                getattr(actor, "type_id", None) or self._config.blueprint
            )
            self._forget_actor(actor)
        else:
            self._forget_actor(self._ambient.pop(sumo_id, None))
        try:
            tc.vehicle.remove(sumo_id)
        except Exception as exc:  # noqa: BLE001 - it may have left this step
            logger.debug("SUMO: removing %s: %s", sumo_id, exc)
        meso.add(
            sumo_id,
            route[index:],
            now=now,
            offset_m=offset,
            max_speed=max_speed,
            length=length,
            type_id=type_id,
            speed_cap=self._commanded.get(sumo_id),
        )
        self._meso_widths[sumo_id] = width
        if entity is not None:
            self._mark_fidelity(entity, SimulationFidelity.MESOSCOPIC)
            logger.info("SUMO: %s is mesoscopic now; out of CARLA", sumo_id)
        self._level_changes[0] += 1
        return True

    def _to_microscopic(self, world: Any, sumo_id: str, now: float) -> bool:
        """Put *sumo_id* back into SUMO where the queue model has it.

        SUMO inserts it on the best lane for its route, at the fastest speed
        that is safe there, or holds it until it is; an ambient vehicle is
        mirrored into CARLA once it is in, like the rest of SUMO's traffic,
        while an entity gets its actor here, at once, so the scenario never
        holds an entity that is microscopic without one.  Answers whether it
        moved; one that did not (no room in CARLA, or SUMO refused it) stays
        mesoscopic until the next decision.
        """
        meso = self._meso
        if meso is None or sumo_id not in meso:
            return False
        vehicle = meso.vehicle(sumo_id)
        link, offset = meso.position(sumo_id, now)
        speed = meso.speed(sumo_id, now)
        if speed <= 0.1:
            depart_speed = "0"
        elif vehicle.speed_cap is not None:
            # At the speed it was told to keep, which SUMO's "max" knows nothing of.
            depart_speed = f"{speed:.2f}"
        else:
            depart_speed = "max"  # the fastest that is safe where it goes in
        entity = self._meso_entities.get(sumo_id)
        actor = None
        if entity is not None:
            actor = self._spawn_actor(
                world,
                self._meso_blueprints.get(sumo_id, self._config.blueprint),
                str(getattr(entity, "role_name", None) or sumo_id),
                self._meso_carla_pose(sumo_id, now),
            )
            if actor is None:
                return False
        tc = self._traci
        self._route_count += 1
        route_id = f"route:{sumo_id}:{self._route_count}"
        try:
            tc.route.add(route_id, vehicle.remaining_route)
            tc.vehicle.add(
                sumo_id,
                route_id,
                typeID=vehicle.type_id or "DEFAULT_VEHTYPE",
                departLane="best",
                departPos=f"{min(offset, max(meso.link(link).length - 0.1, 0.0)):.2f}",
                departSpeed=depart_speed,
            )
            if math.isfinite(vehicle.max_speed):
                tc.vehicle.setMaxSpeed(sumo_id, vehicle.max_speed)
            tc.vehicle.setLength(sumo_id, vehicle.length)
            if sumo_id in self._meso_widths:
                tc.vehicle.setWidth(sumo_id, self._meso_widths[sumo_id])
            if vehicle.speed_cap is not None:
                tc.vehicle.setSpeed(sumo_id, vehicle.speed_cap)
        except Exception as exc:  # noqa: BLE001 - one vehicle must not end the run
            # The scenario's own vehicle is worth a line; SUMO's traffic is not.
            log = logger.warning if entity is not None else logger.debug
            log("SUMO: %s stays mesoscopic: %s", sumo_id, exc)
            try:
                tc.vehicle.remove(sumo_id)
            except Exception:  # noqa: BLE001 - it was never added
                pass
            self._forget_actor(actor)
            return False
        meso.remove(sumo_id)
        self._meso_widths.pop(sumo_id, None)
        if entity is not None:
            del self._meso_entities[sumo_id]
            self._meso_blueprints.pop(sumo_id, None)
            entity._replace_actor(actor)
            if id(entity) in self._released:
                self._publish(sumo_id, actor)
            else:
                tc.vehicle.setColor(sumo_id, _DRIVEN_COLOR)
                self._driven[sumo_id] = entity
            self._mark_fidelity(entity, SimulationFidelity.MICROSCOPIC)
            logger.info("SUMO: %s is microscopic again; back in CARLA", sumo_id)
        self._level_changes[1] += 1
        return True

    def _meso_pose(self, sumo_id: str, now: float) -> Pose2D:
        """Where the queue model has *sumo_id*, in SUMO's frame (front bumper).

        On the rightmost lane a car may use: the queue model has no lanes.
        """
        import sumolib  # noqa: PLC0415

        assert self._meso is not None
        link, offset = self._meso.position(sumo_id, now)
        shape = self._edge_shapes.get(link)
        if shape is None:
            lanes = self._net.getEdge(link).getLanes()
            lane = next((x for x in lanes if x.allows("passenger")), lanes[0])
            shape = [(float(p[0]), float(p[1])) for p in lane.getShape()]
            self._edge_shapes[link] = shape
        # On the shape, whose length is not quite the edge's.
        offset = min(offset, max(sumolib.geomhelper.polyLength(shape) - 0.01, 0.0))
        x, y = sumolib.geomhelper.positionAtShapeOffset(shape, offset)
        rotation = sumolib.geomhelper.rotationAtShapeOffset(shape, offset) or 0.0
        return Pose2D(float(x), float(y), 0.0, (90.0 - math.degrees(rotation)) % 360.0)

    def _meso_carla_pose(self, sumo_id: str, now: float) -> Pose2D:
        """Where the queue model has *sumo_id*, in CARLA's frame (box centre)."""
        assert self._meso is not None
        return sumo_to_carla(
            self._meso_pose(sumo_id, now),
            self._meso.vehicle(sumo_id).length,
            self._offset,
        )

    def _spawn_actor(
        self, world: Any, blueprint_id: str, role_name: str, pose: Pose2D
    ) -> Any:
        """Spawn *blueprint_id* as *role_name* at the CARLA *pose*; ``None`` if
        the spot is taken."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        blueprint = world.get_blueprint_library().find(blueprint_id)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)
        # A little above the road, as for SUMO's own traffic.
        return world.try_spawn_actor(
            blueprint,
            carla.Transform(
                carla.Location(pose.x, pose.y, pose.z + 0.5),
                carla.Rotation(yaw=pose.heading_deg),
            ),
        )

    def _forget_actor(self, actor: Any) -> None:
        """Destroy *actor* and what the followers kept for it."""
        if actor is None:
            return
        self._physics.pop(actor.id, None)
        self._constant_speed.pop(actor.id, None)
        try:
            actor.destroy()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Internals: SUMO's pedestrians in CARLA
    # ------------------------------------------------------------------

    def _person_policy(self) -> Optional[MesoscopicConfig]:
        """The radii a SUMO pedestrian is mirrored into CARLA within, or
        ``None`` to mirror every one."""
        pedestrians = self._config.pedestrians
        if pedestrians.radius_m is not None:
            return MesoscopicConfig(enabled=True, micro_radius_m=pedestrians.radius_m)
        if self._config.mesoscopic.enabled:
            return self._config.mesoscopic
        return None

    def _sync_persons(self, world: Any) -> None:
        """Mirror SUMO's own pedestrians near a focus point into CARLA.

        Like the traffic: one within the near radius gets a walker, one
        beyond the far radius loses it, and one between keeps what it has;
        the rest walk on in SUMO only.  A walker follows its person as an
        ambient car follows its vehicle under ``teleport``: put where the
        person was at the start of the step and walked at its speed, so
        CARLA animates it.
        """
        if not self._config.pedestrians.enabled:
            return
        tc = self._traci
        published = set(self._walkers.values()) | set(self._walked)
        policy = self._person_policy()
        focus = self._focus_points() if policy is not None else []
        wanted: list[tuple[float, str]] = []
        for person in tc.person.getIDList():
            if person in published:
                continue
            if policy is None or not focus:
                wanted.append((0.0, person))
                continue
            x, y = tc.person.getPosition(person)
            distance = min(math.hypot(x - fx, y - fy) for fx, fy in focus)
            current = (
                SimulationFidelity.MICROSCOPIC
                if person in self._persons
                else SimulationFidelity.MESOSCOPIC
            )
            if policy.decide(current, distance) is SimulationFidelity.MICROSCOPIC:
                wanted.append((distance, person))
        wanted.sort()
        limit = self._config.pedestrians.max_persons
        if limit is not None:
            wanted = wanted[:limit]
        keep = {person for _d, person in wanted}
        for person in [p for p in self._persons if p not in keep]:
            self._forget_actor(self._persons.pop(person))
        library = None
        for _d, person in wanted:
            pose = self._person_carla_pose(person)
            walker = self._persons.get(person)
            if walker is None:
                if library is None:
                    library = world.get_blueprint_library()
                walker = self._spawn_walker(world, library, person, pose)
                if walker is not None:
                    self._persons[person] = walker
                continue
            self._walk(walker, pose, float(tc.person.getSpeed(person)))

    def _person_carla_pose(self, person: str) -> Pose2D:
        tc = self._traci
        x, y, z = tc.person.getPosition3D(person)
        return sumo_to_carla(
            Pose2D(x, y, z, tc.person.getAngle(person)), 0.0, self._offset
        )

    def _spawn_walker(self, world: Any, library: Any, person: str, pose: Pose2D) -> Any:
        """A walker for *person* at *pose*; ``None`` if the spot is taken
        (tried again next tick)."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        blueprint = library.find(self._config.pedestrians.blueprint)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", f"{_MIRROR_PREFIX}{person}")
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        # A little above the footway, as CARLA's own walker spawns are.
        return world.try_spawn_actor(
            blueprint,
            carla.Transform(
                carla.Location(pose.x, pose.y, pose.z + 1.0),
                carla.Rotation(yaw=pose.heading_deg),
            ),
        )

    def _walk(self, walker: Any, pose: Pose2D, speed: float) -> None:
        """Put *walker* where its person was at the start of this step, facing
        and walking the way it goes."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        dt = self._context.fixed_delta_seconds if self._context is not None else 0.05
        yaw = math.radians(pose.heading_deg)
        now = walker.get_transform()
        transform = carla.Transform(
            carla.Location(
                pose.x - math.cos(yaw) * speed * dt,
                pose.y - math.sin(yaw) * speed * dt,
                now.location.z,
            ),
            carla.Rotation(yaw=pose.heading_deg),
        )
        from ...entity.pedestrian_entity import (  # noqa: PLC0415
            _UE5_WALKER_MAX_SPEED_MS,
            walker_speed_command,
        )

        # The command CARLA 0.10 walks it at *speed* with, capped at what it
        # can walk rather than warned about: it is sent every tick.
        control = carla.WalkerControl(
            direction=carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0),
            speed=walker_speed_command(min(speed, _UE5_WALKER_MAX_SPEED_MS)),
        )
        if self._context is not None and self._context.client is not None:
            self._commands += [
                carla.command.ApplyTransform(walker.id, transform),
                carla.command.ApplyWalkerControl(walker.id, control),
            ]
        else:
            walker.set_transform(transform)
            walker.apply_control(control)

    # ------------------------------------------------------------------
    # Internals: how drivers drive, and pedestrians walk
    # ------------------------------------------------------------------

    def _pedestrian_type(self, out: Path) -> Optional[Path]:
        """Write SUMO's default pedestrian type as a distribution of
        :data:`PEDESTRIAN_TYPES` types, each with its own draw of the run's
        pedestrian parameters, into *out*; ``None`` when none are set.

        A type, not each person: SUMO reads the junction model's
        ``jmDriveAfterRedTime`` off the type alone, and TraCI cannot set it on
        a person.  SUMO gives each person one of the distribution's types as
        it is added, so each walks as its own draw says.  SUMO's own
        pedestrians (``pedestrians``, and a ``route_path`` demand's that names
        no type of its own) are of it.
        """
        parameters = self.pedestrian_parameters
        if not parameters.set_fields():
            return None
        members = []
        for index in range(PEDESTRIAN_TYPES):
            attributes: dict[str, Optional[float]] = {}
            walking = parameters.walking_speed_mps
            if walking is not None:
                attributes["maxSpeed"] = walking.draw(
                    self._draws, PEDESTRIAN_BOUNDS["walking_speed_mps"]
                )
                # The draw is the person's speed: no second spread of SUMO's.
                attributes["speedDev"] = 0.0
            gap = parameters.min_gap_m
            if gap is not None:
                attributes["minGap"] = gap.draw(
                    self._draws, PEDESTRIAN_BOUNDS["min_gap_m"]
                )
            red = parameters.red_light_violation_s
            if red is not None:
                attributes["jmDriveAfterRedTime"] = red.draw(
                    self._draws, PEDESTRIAN_BOUNDS["red_light_violation_s"]
                )
            written = " ".join(
                f'{key}="{value:.4g}"' for key, value in attributes.items()
            )
            members.append(
                f'        <vType id="pedestrian{index}" vClass="pedestrian" {written}/>'
            )
        path = out / "pedestrian_type.add.xml"
        path.write_text(
            "<additional>\n"
            '    <vTypeDistribution id="DEFAULT_PEDTYPE">\n'
            + "\n".join(members)
            + "\n    </vTypeDistribution>\n"
            "</additional>\n"
        )
        return path

    def _bicycle_type(self, out: Path) -> Optional[Path]:
        """Write SUMO's default bicycle type as a distribution of
        :data:`BICYCLE_TYPES` types, each with its own draw of ``bicycles``'
        riding parameters, into *out*; ``None`` when none are set.

        As for pedestrians (:meth:`_pedestrian_type`): SUMO gives each bicycle
        one of the distribution's types as it is added, so each rides as its
        own draw says.  SUMO's own bicycles (``bicycles``), a ``route_path``
        demand's of SUMO's default bicycle type and the scenario's bicycles
        SUMO drives or is told of are all of it.
        """
        riding = self._config.bicycles.riding()
        if not riding:
            return None
        members = []
        for index in range(BICYCLE_TYPES):
            attributes = {
                _BICYCLE_ATTRIBUTES[name]: spread.draw(
                    self._draws, BICYCLE_BOUNDS[name]
                )
                for name, spread in riding.items()
            }
            if "max_speed_mps" in riding:
                # The speed it rides at is desiredMaxSpeed (SUMO's 20 km/h for
                # a bicycle), below a maxSpeed of 50 km/h: both are the draw,
                # with no second spread of SUMO's.
                attributes["desiredMaxSpeed"] = attributes["maxSpeed"]
                attributes["speedDev"] = 0.0
            written = " ".join(
                f'{key}="{value:.4g}"' for key, value in attributes.items()
            )
            members.append(
                f'        <vType id="bicycle{index}" vClass="bicycle" {written}/>'
            )
        path = out / "bicycle_type.add.xml"
        path.write_text(
            "<additional>\n"
            f'    <vTypeDistribution id="{_BICYCLE_TYPE}">\n'
            + "\n".join(members)
            + "\n    </vTypeDistribution>\n"
            "</additional>\n"
        )
        return path

    def _apply_parameters(self) -> None:
        """Give each vehicle that has just departed, and that SUMO drives,
        its own draw of the run's parameters.

        On the vehicle, not its type: a demand file (``route_path``) brings
        types of its own, and its vehicles are SUMO's to drive all the same.
        Before :meth:`_follow_visibility` reads the vehicle's ``decel`` and
        ``tau``, so the sight speed is worked out from its draws.  Not a
        bicycle: a rider draws ``bicycles``' parameters, off its type.
        """
        parameters = self.vehicle_parameters
        draws = bool(parameters.set_fields())
        sublane = self._config.lateral_resolution_m is not None
        if draws or sublane:
            for sumo_id in self._departed_ids():
                if sumo_id in self._external:  # CARLA's side drives those
                    continue
                if sublane:
                    self._no_sidestep_standing(sumo_id)
                if draws and self._vclass(sumo_id) != "bicycle":
                    self._draw_vehicle(sumo_id, parameters)
        overrides = self._pedestrian_overrides
        if overrides is not None:
            for person in self._traci.simulation.getDepartedPersonIDList():
                self._draw_person(person, overrides)

    def _no_sidestep_standing(self, sumo_id: str) -> None:
        """Keep *sumo_id* from moving sideways while it stands.

        The sublane model lets a standing vehicle crab sideways at its full
        lateral speed: held behind a pedestrian, it edges round them a hand's
        breadth away, and no car under CARLA's physics can follow that.
        """
        try:
            self._traci.vehicle.setParameter(
                sumo_id, "laneChangeModel.lcMaxSpeedLatStanding", "0"
            )
        except Exception as exc:  # noqa: BLE001 - one vehicle must not end the run
            logger.warning("SUMO could not hold %s sideways: %s", sumo_id, exc)

    def _draw_vehicle(self, sumo_id: str, parameters: VehicleParameters) -> None:
        """Give *sumo_id* its own draw of each field *parameters* sets."""
        vehicle = self._traci.vehicle
        rng = self._draws
        try:
            for name, value in (
                ("speed_factor", parameters.speed_factor),
                ("min_gap_m", parameters.min_gap_m),
                ("accel", parameters.accel),
                ("decel", parameters.decel),
                ("reaction_time_s", parameters.reaction_time_s),
                ("imperfection", parameters.imperfection),
                ("max_lateral_speed_mps", parameters.max_lateral_speed_mps),
                ("max_lateral_accel", parameters.max_lateral_accel),
            ):
                if value is None:
                    continue
                draw = value.draw(rng, VEHICLE_BOUNDS[name])
                model = _LANE_CHANGE_MODEL_PARAMETERS.get(name)
                if model is not None:
                    vehicle.setParameter(
                        sumo_id, f"laneChangeModel.{model}", f"{draw:g}"
                    )
                else:
                    getattr(vehicle, _VEHICLE_SETTERS[name])(sumo_id, draw)
            if parameters.lane_changes is not None:
                vehicle.setLaneChangeMode(
                    sumo_id,
                    _LANE_CHANGE_MODE_DEFAULT
                    if parameters.lane_changes
                    else _LANE_CHANGE_MODE_STRATEGIC,
                )
        except Exception as exc:  # noqa: BLE001 - one vehicle must not end the run
            logger.warning(
                "SUMO could not set the traffic parameters of %s: %s", sumo_id, exc
            )

    def _draw_person(self, person: str, parameters: PedestrianParameters) -> None:
        """Give SUMO's pedestrian *person* its own draw of the fields
        *parameters* sets that a person can be given during a run."""
        if person in self._walkers.values():
            return  # one of CARLA's, placed in SUMO where it stands
        tc = self._traci
        try:
            walking = parameters.walking_speed_mps
            if walking is not None:
                tc.person.setMaxSpeed(
                    person,
                    walking.draw(self._draws, PEDESTRIAN_BOUNDS["walking_speed_mps"]),
                )
                tc.person.setSpeedFactor(person, 1.0)  # the draw is the speed
            gap = parameters.min_gap_m
            if gap is not None:
                tc.person.setMinGap(
                    person, gap.draw(self._draws, PEDESTRIAN_BOUNDS["min_gap_m"])
                )
        except Exception as exc:  # noqa: BLE001 - one person must not end the run
            logger.warning(
                "SUMO could not set the pedestrian parameters of %s: %s", person, exc
            )

    # ------------------------------------------------------------------
    # Internals: what drivers see
    # ------------------------------------------------------------------

    def _follow_visibility(self, world: Any) -> None:
        """Hold SUMO's own traffic to the speed it stops from within what its
        drivers see: every vehicle when the visibility changes, and the ones
        that have just departed otherwise."""
        if not self._follows_visibility:
            return
        seen = world_visibility(episode_of(world))
        if seen is None and self._visibility is None:
            return  # nothing set, nothing held: no TraCI call at all
        tc = self._traci
        if self._max_speeds:
            for sumo_id in tc.simulation.getArrivedIDList():
                self._max_speeds.pop(sumo_id, None)
        if seen == self._visibility:
            ids = self._departed_ids()
        elif seen is None:
            self._visibility = None
            logger.info("SUMO: drivers see as far as SUMO lets them again")
            # Only the held ones have anything to get back, if still there.
            present = set(tc.vehicle.getIDList())
            ids = [i for i in self._max_speeds if i in present]
            self._max_speeds = {i: self._max_speeds[i] for i in ids}
        else:
            self._visibility = seen
            logger.info("SUMO: drivers see %.0f m; traffic is held to it", seen)
            ids = list(tc.vehicle.getIDList())
        for sumo_id in ids:
            if sumo_id not in self._driven and sumo_id not in self._external:
                self._hold_to_sight(sumo_id, seen)

    def _hold_to_sight(self, sumo_id: str, seen: Optional[float]) -> None:
        """Cap *sumo_id*'s maximum speed at the one it stops from within
        *seen* metres; ``None`` gives it its own back."""
        vehicle = self._traci.vehicle
        if seen is None:
            if sumo_id in self._max_speeds:
                vehicle.setMaxSpeed(sumo_id, self._max_speeds.pop(sumo_id)[0])
            return
        held = self._max_speeds.get(sumo_id)
        if held is None:
            # The vehicle's own: its draws of the parameters, when set.
            held = self._max_speeds[sumo_id] = (
                vehicle.getMaxSpeed(sumo_id),
                vehicle.getDecel(sumo_id),
                vehicle.getTau(sumo_id),
            )
        own, decel, tau = held
        limit = sight_speed(seen, decel, tau)
        # Never to a standstill: a car in thick fog still creeps on.
        vehicle.setMaxSpeed(sumo_id, max(min(own, limit), 1.0))

    # ------------------------------------------------------------------
    # Internals: traffic lights
    # ------------------------------------------------------------------

    def _match_signals(self, world: Any) -> dict[tuple[str, int], Any]:
        """Each SUMO signal link, and the CARLA light that switches it.

        Read off roadgen's traces when the network was built here: a CARLA light
        is the OpenDRIVE signal of its ``get_opendrive_id()``, and the traces
        follow that signal through the map to the SUMO links of every movement
        it governs.  A network given as ``net_path`` has no traces, and is
        matched by where each light's lanes fall in it instead.
        """
        table = SignalTable.load(self._network) if self._network else None
        if table is None:
            logger.info(
                "No roadgen traces beside the SUMO network; matching CARLA lights "
                "to SUMO signal links by position"
            )
            return self._match_signals_by_position(world)
        matched: dict[tuple[str, int], Any] = {}
        unmatched = []
        for light in world.get_actors().filter("traffic.traffic_light"):
            links = table.links(light.get_opendrive_id())
            if not links:
                unmatched.append(light.get_opendrive_id())
            for link in links:
                matched[link] = light
        if unmatched:
            logger.warning(
                "No SUMO signal link for CARLA light(s) %s; SUMO keeps its own "
                "program there",
                ", ".join(sorted(unmatched)),
            )
        logger.info("Matched %d SUMO signal link(s) to CARLA lights", len(matched))
        return matched

    def _match_signals_by_position(self, world: Any) -> dict[tuple[str, int], Any]:
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
            # A bicycle ride goes with the traffic beside it, or crosses while
            # the traffic it crosses is held; before the crossings, which wait
            # for it as for any vehicle over their road.
            for index, follows, links in self._bicycle_rides.get(tls, ()):
                state[index] = _ride_state(follows, [state[i] for i in links])
            # A crossing walks while nothing drives over the road it crosses:
            # the CARLA lights decide the vehicles, and with them the people.
            for index, foes in self._crossing_links.get(tls, ()):
                clear = all(state[foe] in "rR" for foe in foes)
                state[index] = "G" if clear else "r"
            tc.trafficlight.setRedYellowGreenState(tls, "".join(state))

    def _match_crossings(self) -> dict[str, list[tuple[int, list[int]]]]:
        """Each signalled pedestrian crossing, and the vehicle links it waits for.

        netconvert appends a crossing's link to its junction's program after
        the vehicle links (``pedestrian_crossings``), so the vehicle links keep
        the indices CARLA's lights are matched to and only the crossings' are
        left without a CARLA light: each takes its state from the vehicle links
        into or out of a road it crosses.
        """
        if self._network is None:
            return {}
        crossings = crossing_edges(self._network.net_file)
        if not crossings:
            return {}
        tc = self._traci
        out: dict[str, list[tuple[int, list[int]]]] = {}
        for tls in tc.trafficlight.getIDList():
            vehicles: list[tuple[int, str, str]] = []
            walks: list[tuple[int, frozenset[str]]] = []
            for index, controlled in enumerate(tc.trafficlight.getControlledLinks(tls)):
                for in_lane, out_lane, _via in controlled:
                    to_edge = out_lane.rsplit("_", 1)[0]
                    if to_edge in crossings:
                        walks.append((index, crossings[to_edge]))
                    else:
                        vehicles.append((index, in_lane.rsplit("_", 1)[0], to_edge))
            links = [
                (
                    index,
                    sorted(
                        {i for i, src, dst in vehicles if src in roads or dst in roads}
                    ),
                )
                for index, roads in walks
            ]
            if links:
                out[tls] = links
        if out:
            logger.info(
                "SUMO: %d signalled pedestrian crossing(s) follow CARLA's lights",
                sum(len(links) for links in out.values()),
            )
        return out

    def _match_bicycle_rides(self) -> dict[str, list[tuple[int, str, list[int]]]]:
        """Each signal link of a bicycle ride no CARLA light switches, and the
        vehicle links its state follows.

        With ``cycle_on_sidewalks``, roadgen puts a signalised junction's
        bicycle rides in its program: a bicycle comes off a sidewalk onto the
        road in a green of its own, ahead of the traffic, gets off the road
        with the traffic it leaves, and rides across a road while that road
        is held.  CARLA's lights switch the traffic, so each ride follows the
        vehicle links of the same movement instead:

        * onto the road, from a sidewalk: the vehicle links into the same edge
          from the approach whose lanes end nearest the sidewalk's;
        * off the road, onto a sidewalk: the vehicle links from the same lane;
        * across the road, sidewalk to sidewalk: clear of every vehicle link
          SUMO's own program never lets go together with it.

        Read off SUMO's programs before anything sets a light's state.
        """
        tc = self._traci
        out: dict[str, list[tuple[int, str, list[int]]]] = {}
        switched = set(self._signals)
        for tls in tc.trafficlight.getIDList():
            lanes: dict[int, tuple[str, str]] = {}
            for index, controlled in enumerate(tc.trafficlight.getControlledLinks(tls)):
                for in_lane, out_lane, _via in controlled:
                    lanes.setdefault(index, (in_lane, out_lane))
            kinds = {
                index: (self._lane_kind(i), self._lane_kind(o))
                for index, (i, o) in lanes.items()
            }
            vehicles = [
                index for index, kind in kinds.items() if kind == ("road", "road")
            ]
            rides: list[tuple[int, str, list[int]]] = []
            for index, kind in kinds.items():
                if (tls, index) in switched or "bicycle" not in kind:
                    continue
                in_lane, out_lane = lanes[index]
                if kind == ("bicycle", "road"):
                    rides.append(
                        (
                            index,
                            "with",
                            self._approach(in_lane, out_lane, lanes, vehicles),
                        )
                    )
                elif kind == ("road", "bicycle"):
                    same = [i for i in vehicles if lanes[i][0] == in_lane]
                    rides.append((index, "with", same))
                elif kind == ("bicycle", "bicycle"):
                    together = self._green_together(tls, index)
                    foes = [i for i in vehicles if i not in together]
                    rides.append((index, "clear", foes))
            rides = [ride for ride in rides if ride[2]]
            if rides:
                out[tls] = rides
        if out:
            logger.info(
                "SUMO: %d signalled bicycle ride(s) follow CARLA's lights",
                sum(len(rides) for rides in out.values()),
            )
        return out

    def _lane_kind(self, lane_id: str) -> str:
        """``road`` for a lane a car may use, ``bicycle`` for one only a
        bicycle (and pedestrians) may, ``other`` for the rest -- a footway, a
        crossing, a walking area."""
        try:
            lane = self._net.getLane(lane_id)
        except Exception:  # noqa: BLE001 - a lane sumolib lacks
            return "other"
        if lane.getEdge().getFunction() not in ("", "internal"):
            return "other"
        if lane.allows("passenger"):
            return "road"
        return "bicycle" if lane.allows("bicycle") else "other"

    def _approach(
        self,
        in_lane: str,
        out_lane: str,
        lanes: dict[int, tuple[str, str]],
        vehicles: list[int],
    ) -> list[int]:
        """The vehicle links into *out_lane*'s edge from the approach whose
        lane ends nearest *in_lane*'s: the traffic a bicycle coming off that
        sidewalk joins."""
        to_edge = out_lane.rsplit("_", 1)[0]
        into = [i for i in vehicles if lanes[i][1].rsplit("_", 1)[0] == to_edge]
        if not into:
            return []

        def end(lane_id: str) -> tuple[float, float]:
            point = self._net.getLane(lane_id).getShape()[-1]
            return float(point[0]), float(point[1])

        start = end(in_lane)
        nearest = min(into, key=lambda i: math.dist(start, end(lanes[i][0])))
        from_edge = lanes[nearest][0].rsplit("_", 1)[0]
        return [i for i in into if lanes[i][0].rsplit("_", 1)[0] == from_edge]

    def _green_together(self, tls: str, index: int) -> set[int]:
        """The links SUMO's own programs of *tls* have green in a phase where
        link *index* is green."""
        together: set[int] = set()
        for logic in self._traci.trafficlight.getAllProgramLogics(tls):
            for phase in logic.phases:
                state = phase.state
                if index < len(state) and state[index] in "Gg":
                    together.update(i for i, c in enumerate(state) if c in "Gg")
        return together

    def _signals_to_carla(self) -> None:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        tc = self._traci
        states: dict[str, str] = {}
        # Each light once, with the state of its last link (a light switching
        # several links was set once per link, the last one winning) ...
        wanted: dict[int, tuple[Any, Any]] = {}
        for (tls, index), light in self._signals.items():
            if tls not in states:
                states[tls] = tc.trafficlight.getRedYellowGreenState(tls)
            name = _SUMO_TO_CARLA_SIGNAL.get(states[tls][index])
            if name is not None:
                wanted[id(light)] = (light, getattr(carla.TrafficLightState, name))
        # ... and only when it shows another: set_state() is an RPC, and the
        # state a light shows is read off the client's snapshot. A light a
        # scenario switched is still set back to SUMO's state.
        for light, state in wanted.values():
            if int(light.get_state()) != int(state):
                light.set_state(state)

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
            # The queues SUMO builds while it warms up are the ones CARLA's lights
            # would have built.
            if self._config.traffic_light_authority == "carla":
                self._signals_to_sumo()
            self._step_sumo(world)
            if self._meso is not None:
                # Not the scenario's vehicles: they are held still for now.
                self._step_mesoscopic(world, entities=False)
        for sumo_id in self._driven:
            tc.vehicle.setSpeed(sumo_id, -1)  # back to SUMO's own speed
        self._sync_ambient(world)
        self._sync_persons(world)
        self._flush_commands()
        logger.info(
            "SUMO warmed up for %.0f s: %d ambient vehicle(s) on the road",
            self._config.warmup_s,
            len(self._ambient),
        )
