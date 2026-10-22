"""Options of the SUMO traffic backend.

Like :mod:`..config`, this module imports nothing heavier than the standard
library, so the config layer and the editor can name the backend's options
without SUMO installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from ...constants import DEFAULT_TM_PORT
from ..fidelity import MesoscopicConfig

__all__ = ["AmbientTrafficConfig", "PedestrianConfig", "SumoBackendConfig"]

_LIGHT_AUTHORITIES = ("carla", "sumo", "none")
_SCENARIO_VEHICLE_DRIVERS = ("traffic_manager", "sumo")
_SCENARIO_PEDESTRIAN_DRIVERS = ("traffic_manager", "sumo")
_VEHICLE_CONTROLS = ("physics", "teleport")


@dataclass
class AmbientTrafficConfig:
    """SUMO's own traffic, generated with ``randomTrips.py``.

    Attributes:
        enabled: Generate any at all.  Off (the default), SUMO carries only
            the scenario's vehicles, the background vehicles its traffic
            sources add (``TrafficSourceAction``), and a ``route_path``
            demand, when one is given: nothing joins the run that the run did
            not ask for.
        vehicles_per_hour: Trips generated per hour of simulated time.
        period_s: Seconds between two trips; overrides ``vehicles_per_hour``.
        end_s: Seconds of demand to generate.
        fringe_factor: ``randomTrips.py``'s weight on trips starting and ending
            at the edge of the network; used without ``use_safe_weights``.
        use_safe_weights: Weight sources and sinks with the
            ``network.safe.*`` files roadgen writes beside the network, which
            keep trips off dead ends.
    """

    enabled: bool = False
    vehicles_per_hour: float = 900.0
    period_s: Optional[float] = None
    end_s: float = 3600.0
    fringe_factor: float = 5.0
    use_safe_weights: bool = True

    def period(self) -> float:
        """Seconds between two generated trips."""
        if self.period_s is not None:
            return float(self.period_s)
        return 3600.0 / max(float(self.vehicles_per_hour), 1e-6)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AmbientTrafficConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        # Imported here: utils.config reads a dataclass's fields from its class,
        # which Codon cannot express, and the library check compiles no import
        # in a function it does not call (typecheck/library.py, UNCALLED).
        from ...utils.config import checked_options  # noqa: PLC0415

        return cls(**checked_options(cls, mapping))


@dataclass
class PedestrianConfig:
    """SUMO's own pedestrians, generated with ``randomTrips.py --pedestrians``
    and mirrored into CARLA as walkers.

    Attributes:
        enabled: Generate any at all.  They walk the network's footways, so a
            network without any has none to walk.
        persons_per_hour: Walks generated per hour of simulated time.
        period_s: Seconds between two walks; overrides ``persons_per_hour``.
        end_s: Seconds of demand to generate.
        max_distance_m: The longest walk generated, metres.
        radius_m: Mirror only the pedestrians this close to the ego (or to a
            vehicle CARLA drives) into CARLA, and remove one that walks more
            than 1.25 times as far away; the rest walk on in SUMO only, where
            SUMO's traffic still sees them.  ``None``: the radii of
            ``mesoscopic`` when that is enabled, everywhere otherwise.
        max_persons: Most pedestrians mirrored into CARLA at once, the nearest
            first; ``None`` for no limit.
        blueprint: CARLA blueprint of a mirrored pedestrian.
    """

    enabled: bool = False
    persons_per_hour: float = 600.0
    period_s: Optional[float] = None
    end_s: float = 3600.0
    max_distance_m: float = 500.0
    radius_m: Optional[float] = None
    max_persons: Optional[int] = None
    blueprint: str = "walker.pedestrian.0001"

    def __post_init__(self) -> None:
        if self.radius_m is not None and not self.radius_m > 0.0:
            raise ValueError(
                f"radius_m must be positive or null (got {self.radius_m!r})"
            )

    def period(self) -> float:
        """Seconds between two generated walks."""
        if self.period_s is not None:
            return float(self.period_s)
        return 3600.0 / max(float(self.persons_per_hour), 1e-6)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PedestrianConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        from ...utils.config import checked_options  # noqa: PLC0415

        return cls(**checked_options(cls, mapping))


@dataclass
class SumoBackendConfig:
    """Options of the ``sumo`` traffic backend.

    How SUMO's drivers behave -- their speed, gap, reaction and braking, and
    whether they follow the visibility -- and how its pedestrians walk is not
    here but in the parameters every backend shares
    (:class:`~autoware_carla_scenario.traffic.parameters.VehicleParameters`,
    ``traffic.vehicles``;
    :class:`~autoware_carla_scenario.traffic.parameters.PedestrianParameters`,
    ``traffic.pedestrians``).  ``pedestrians`` here is how many SUMO
    generates.

    Attributes:
        scenario_vehicles: What drives the scenario's own vehicles (its
            authored NPCs, and an ego with ``ego.entity=autopilot``).
            ``traffic_manager`` (the default): CARLA's TrafficManager, as under
            the ``traffic_manager`` backend, so a scenario's NPCs behave the
            same whichever backend fills the road -- they are published into
            SUMO, whose traffic reacts to them.  ``sumo``: SUMO drives them, and
            their manoeuvres become TraCI calls.
        scenario_pedestrians: What walks the scenario's own pedestrians (the
            ``PedestrianEntity`` it registers).  ``traffic_manager`` (the
            default): CARLA's side, as under the ``traffic_manager`` backend
            -- a ``WalkStraightAction`` walks the walker straight ahead with a
            ``carla.WalkerControl`` -- and each is published into SUMO
            (``publish_walkers``).  ``sumo``: each is a SUMO person, and its
            CARLA walker follows it.  A ``WalkStraightAction`` walks the
            person straight on through TraCI (``person.moveToXY`` each step,
            onto the road where the line crosses it, so SUMO's traffic sees
            it there), and the walker follows: the same walk as on CARLA's
            side, with SUMO holding the pedestrian.
        tm_port: TrafficManager port, for ``scenario_vehicles=traffic_manager``.
        vehicle_control: How a CARLA vehicle SUMO drives follows its SUMO
            vehicle.  ``physics`` (the default): with CARLA's vehicle physics,
            steered along SUMO's path (pure pursuit, as TeraSim's
            ``ackermann_physics`` mode does) and driven at SUMO's speed by a PI
            throttle/brake controller.
            ``teleport``: put where SUMO has it every tick, with SUMO's speed as
            a constant velocity.
        feedback_distance_m: With ``physics``, a car further than this from its
            SUMO vehicle moves the SUMO vehicle to where the car really is, so
            the gaps SUMO keeps are kept to the car CARLA has.  ``moveTo``
            takes effect at once and leaves SUMO planning the vehicle's next
            step; inside a junction nothing is corrected.  ``0`` never
            corrects SUMO.
        resync_distance_m: With ``physics``, a car further than this from its
            SUMO vehicle (after a collision, say) is put back on it in CARLA.
        traffic_light_authority: Which simulator's signals the other follows.
            ``carla``: SUMO's signals take the states of CARLA's lights every
            step, so SUMO traffic obeys the lights the ego sees and a scenario
            that sets them (``TrafficSignalAction``) sets them for SUMO too.
            ``sumo``: CARLA's lights follow SUMO's programs.  ``none``: each
            keeps its own.
        pedestrian_crossings: Build the network with a pedestrian crossing
            across each road at a junction where footways meet it, and walking
            areas joining them (netconvert's ``--crossings.guess``), so SUMO's
            pedestrians cross roads on crossings, where SUMO's vehicles yield
            to them.  At a signalised junction, with
            ``traffic_light_authority=carla``, a crossing is green while every
            vehicle movement into or out of a road it crosses is red.
            Ignored with ``net_path``.
        publish_walkers: Put CARLA's pedestrians into SUMO as persons where
            they stand, every step, so SUMO traffic sees them: a SUMO vehicle
            brakes for a pedestrian on its lane -- on a crossing or anywhere
            else on the road -- or changes lanes round it.  Not the scenario's
            pedestrians SUMO walks (``scenario_pedestrians=sumo``): those are
            in SUMO already.
        curve_lateral_acceleration: Lateral acceleration (m/s²) SUMO traffic is
            held to in bends when the world's OpenDRIVE is converted: roadgen
            cuts each edge at its bends and gives every piece, and every path
            across a junction, the speed ``sqrt(a / curvature)`` its sharpest
            point allows, where that is below the limit.  SUMO's models read no
            curvature, so without it traffic takes Town10's bends at up to
            ~10 m/s² sideways, which a car under ``physics`` cannot follow.
            ``None`` converts the map as it is.  Needs a roadgen with
            ``export_sumo(curve_lateral_acceleration=...)``; an older one
            converts without it, with a warning.  Ignored with ``net_path``.
        lateral_resolution_m: Width of a sublane, metres
            (``--lateral-resolution``; 0.4 by default): SUMO's sublane model,
            where a vehicle's lateral position is continuous -- it changes
            lanes smoothly, at its own lateral speed and acceleration
            (``traffic.vehicles.max_lateral_speed_mps`` and
            ``max_lateral_accel``), never in one step.  ``None``: SUMO's
            lane-based model, with ``lane_change_duration_s``.
        lane_change_duration_s: With ``lateral_resolution_m: null`` only, how
            long a lane change of SUMO's lane-based model takes, seconds
            (``--lanechange.duration``; 3 s when unset): a vehicle moves
            across continuously over this time, at no more than its
            ``max_lateral_speed_mps``.  SUMO takes only one of the two.
        step_length_s: SUMO's step, seconds (``--step-length``).  ``None``:
            the world's ``fixed_delta_seconds``, one SUMO step per CARLA tick.
            A shorter one must divide the tick in whole milliseconds (SUMO's
            clock counts those): SUMO then makes that many steps per tick, so
            movement within a tick, a lane change above all, is smoother.
        net_path: An existing ``.net.xml`` to run on, instead of converting the
            world's OpenDRIVE.  Its x/y must be OpenDRIVE's plus its
            ``netOffset``, as netconvert writes a network built from it.
        route_path: Explicit SUMO demand (``.rou.xml``), added to the ambient
            traffic.
        ambient: SUMO's own generated traffic.  Off by default.
        pedestrians: SUMO's own generated pedestrians, mirrored into CARLA
            near the ego.  Off by default.
        warmup_s: SUMO seconds simulated when the run starts, before the
            scenario clock, so ambient traffic is already on the road; the
            scenario's vehicles are held still meanwhile.
        spawn_radius_m: Mirror only SUMO vehicles this close to the ego into
            CARLA; ``None`` mirrors them wherever they are.
        max_vehicles: Most ambient vehicles mirrored into CARLA at once, the
            nearest to the ego first; ``None`` for no limit.
        mesoscopic: Simulate SUMO's vehicles far from the ego (and from every
            vehicle CARLA drives) in a mesoscopic queue model instead of in
            SUMO, with no actor in CARLA, and put them back into SUMO -- and
            CARLA -- when they come near
            (:class:`~autoware_carla_scenario.traffic.fidelity.MesoscopicConfig`).
            Off by default.
        blueprint: CARLA blueprint of an ambient vehicle.
        binary: SUMO executable, when not in-process (``sumo-gui`` to watch).
        use_libsumo: Run SUMO in-process through libsumo (faster, no GUI);
            otherwise as a TraCI subprocess running ``binary``.
        cache_dir: Where converted networks are kept, keyed by the OpenDRIVE
            they came from; ``~/.cache/autoware_carla_scenario/sumo`` by
            default.
        fcd_output: Record every SUMO vehicle's state each step (SUMO's
            floating car data) to ``<output_dir>/sumo/fcd.xml``, for replaying
            or plotting a run's traffic afterwards, and the SUMO time each CARLA
            tick maps to in ``<output_dir>/sumo/clock.csv``.
        sumo_args: Extra command-line arguments for SUMO.
    """

    scenario_vehicles: str = "traffic_manager"
    scenario_pedestrians: str = "traffic_manager"
    tm_port: int = DEFAULT_TM_PORT
    vehicle_control: str = "physics"
    feedback_distance_m: float = 0.0
    resync_distance_m: float = 8.0
    traffic_light_authority: str = "carla"
    publish_walkers: bool = True
    pedestrian_crossings: bool = True
    curve_lateral_acceleration: Optional[float] = 3.0
    lateral_resolution_m: Optional[float] = 0.4
    lane_change_duration_s: Optional[float] = None
    step_length_s: Optional[float] = None
    net_path: Optional[str] = None
    route_path: Optional[str] = None
    ambient: AmbientTrafficConfig = field(default_factory=AmbientTrafficConfig)
    pedestrians: PedestrianConfig = field(default_factory=PedestrianConfig)
    warmup_s: float = 30.0
    spawn_radius_m: Optional[float] = None
    max_vehicles: Optional[int] = None
    mesoscopic: MesoscopicConfig = field(default_factory=MesoscopicConfig)
    blueprint: str = "vehicle.lincoln.mkz"
    binary: str = "sumo"
    use_libsumo: bool = True
    cache_dir: Optional[str] = None
    fcd_output: bool = False
    sumo_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.traffic_light_authority not in _LIGHT_AUTHORITIES:
            raise ValueError(
                "traffic_light_authority must be one of "
                f"{', '.join(_LIGHT_AUTHORITIES)} (got {self.traffic_light_authority!r})"
            )
        if self.vehicle_control not in _VEHICLE_CONTROLS:
            raise ValueError(
                "vehicle_control must be one of "
                f"{', '.join(_VEHICLE_CONTROLS)} (got {self.vehicle_control!r})"
            )
        if self.curve_lateral_acceleration is not None and not (
            self.curve_lateral_acceleration > 0.0
        ):
            raise ValueError(
                "curve_lateral_acceleration must be positive or null "
                f"(got {self.curve_lateral_acceleration!r})"
            )
        for name, value in (
            ("lateral_resolution_m", self.lateral_resolution_m),
            ("lane_change_duration_s", self.lane_change_duration_s),
            ("step_length_s", self.step_length_s),
        ):
            if value is not None and not value > 0.0:
                raise ValueError(f"{name} must be positive or null (got {value!r})")
        if (
            self.lane_change_duration_s is not None
            and self.lateral_resolution_m is not None
        ):
            raise ValueError(
                "lane_change_duration_s needs lateral_resolution_m: null: SUMO "
                "changes lanes either over a duration, in its lane-based model, "
                "or in its sublane model (the default), where "
                "max_lateral_speed_mps and max_lateral_accel (traffic.vehicles) "
                "say how fast"
            )
        if self.scenario_vehicles not in _SCENARIO_VEHICLE_DRIVERS:
            raise ValueError(
                "scenario_vehicles must be one of "
                f"{', '.join(_SCENARIO_VEHICLE_DRIVERS)} (got {self.scenario_vehicles!r})"
            )
        if self.scenario_pedestrians not in _SCENARIO_PEDESTRIAN_DRIVERS:
            raise ValueError(
                "scenario_pedestrians must be one of "
                f"{', '.join(_SCENARIO_PEDESTRIAN_DRIVERS)} "
                f"(got {self.scenario_pedestrians!r})"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SumoBackendConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        from ...utils.config import checked_options  # noqa: PLC0415

        if "follow_visibility" in mapping:
            raise ValueError(
                "traffic.options.follow_visibility has moved to "
                "traffic.vehicles.follow_visibility, with the other vehicle "
                "parameters every backend shares (VehicleParameters)"
            )
        checked = checked_options(cls, mapping)
        if isinstance(checked.get("ambient"), Mapping):
            checked["ambient"] = AmbientTrafficConfig.from_mapping(checked["ambient"])
        if isinstance(checked.get("pedestrians"), Mapping):
            checked["pedestrians"] = PedestrianConfig.from_mapping(
                checked["pedestrians"]
            )
        if isinstance(checked.get("mesoscopic"), Mapping):
            checked["mesoscopic"] = MesoscopicConfig.from_mapping(checked["mesoscopic"])
        if "sumo_args" in checked:
            checked["sumo_args"] = [str(a) for a in checked["sumo_args"] or []]
        if checked.get("tm_port") is None:
            checked.pop("tm_port", None)  # `${oc.select:...,null}`: the default
        return cls(**checked)

    def resolved_cache_dir(self) -> Path:
        """The directory converted networks are cached in."""
        if self.cache_dir:
            return Path(self.cache_dir).expanduser()
        return Path.home() / ".cache" / "autoware_carla_scenario" / "sumo"
