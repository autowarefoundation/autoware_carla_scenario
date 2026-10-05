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
from ...utils.config import checked_options

__all__ = ["AmbientTrafficConfig", "SumoBackendConfig"]

_LIGHT_AUTHORITIES = ("carla", "sumo", "none")
_SCENARIO_VEHICLE_DRIVERS = ("traffic_manager", "sumo")
_VEHICLE_CONTROLS = ("physics", "teleport")


@dataclass
class AmbientTrafficConfig:
    """SUMO's own traffic, generated with ``randomTrips.py``.

    Attributes:
        enabled: Generate any at all.  Off, SUMO carries only the scenario's
            vehicles (and a ``route_path`` demand, when one is given).
        vehicles_per_hour: Trips generated per hour of simulated time.
        period_s: Seconds between two trips; overrides ``vehicles_per_hour``.
        end_s: Seconds of demand to generate.
        fringe_factor: ``randomTrips.py``'s weight on trips starting and ending
            at the edge of the network; used without ``use_safe_weights``.
        use_safe_weights: Weight sources and sinks with the
            ``network.safe.*`` files roadgen writes beside the network, which
            keep trips off dead ends.
    """

    enabled: bool = True
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
        return cls(**checked_options(cls, mapping))


@dataclass
class SumoBackendConfig:
    """Options of the ``sumo`` traffic backend.

    Attributes:
        scenario_vehicles: What drives the scenario's own vehicles (its
            authored NPCs, and an ego with ``ego.entity=autopilot``).
            ``traffic_manager`` (the default): CARLA's TrafficManager, as under
            the ``traffic_manager`` backend, so a scenario's NPCs behave the
            same whichever backend fills the road -- they are published into
            SUMO, whose traffic reacts to them.  ``sumo``: SUMO drives them, and
            their manoeuvres become TraCI calls.
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
        publish_walkers: Put CARLA's pedestrians into SUMO as persons where
            they stand, every step, so SUMO traffic sees them: a SUMO vehicle
            brakes for a pedestrian on its lane -- on a crossing or anywhere
            else on the road -- or changes lanes round it.
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
        net_path: An existing ``.net.xml`` to run on, instead of converting the
            world's OpenDRIVE.  Its x/y must be OpenDRIVE's plus its
            ``netOffset``, as netconvert writes a network built from it.
        route_path: Explicit SUMO demand (``.rou.xml``), added to the ambient
            traffic.
        ambient: SUMO's own generated traffic.
        warmup_s: SUMO seconds simulated when the run starts, before the
            scenario clock, so ambient traffic is already on the road; the
            scenario's vehicles are held still meanwhile.
        spawn_radius_m: Mirror only SUMO vehicles this close to the ego into
            CARLA; ``None`` mirrors them wherever they are.
        max_vehicles: Most ambient vehicles mirrored into CARLA at once, the
            nearest to the ego first; ``None`` for no limit.
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
    tm_port: int = DEFAULT_TM_PORT
    vehicle_control: str = "physics"
    feedback_distance_m: float = 0.0
    resync_distance_m: float = 8.0
    traffic_light_authority: str = "carla"
    publish_walkers: bool = True
    curve_lateral_acceleration: Optional[float] = 3.0
    net_path: Optional[str] = None
    route_path: Optional[str] = None
    ambient: AmbientTrafficConfig = field(default_factory=AmbientTrafficConfig)
    warmup_s: float = 30.0
    spawn_radius_m: Optional[float] = None
    max_vehicles: Optional[int] = None
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
        if self.scenario_vehicles not in _SCENARIO_VEHICLE_DRIVERS:
            raise ValueError(
                "scenario_vehicles must be one of "
                f"{', '.join(_SCENARIO_VEHICLE_DRIVERS)} (got {self.scenario_vehicles!r})"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SumoBackendConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        checked = checked_options(cls, mapping)
        if isinstance(checked.get("ambient"), Mapping):
            checked["ambient"] = AmbientTrafficConfig.from_mapping(checked["ambient"])
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
