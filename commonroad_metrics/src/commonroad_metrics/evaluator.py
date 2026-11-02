"""Criticality of one frame of a run, read with CriMe and CommonRoad-Reach.

A frame is the road users where they are now, in the CommonRoad frame of the
map (the OpenDRIVE inertial frame roadgen writes the scenario in: x east, y
north, angles counter-clockwise).  The evaluator turns it into a CommonRoad
scenario on the map's lanelet network -- every road user a dynamic obstacle
whose trajectory runs on from where it is at its current velocity and
acceleration, over the horizon -- and evaluates the measures asked for on the
ego at its first time step:

* CriMe's (``commonroad_crime``) time, distance and acceleration measures --
  TTC, THW, TTB and the rest -- each the most critical over the other road
  users, as CriMe folds them (the least TTC, the greatest required
  deceleration);
* CommonRoad-Reach's (``commonroad_reach``) reachable set of the ego, its
  drivable area at the end of the horizon: what is left of the road the ego can
  still reach without hitting anyone.  ``0`` means none: a collision can no
  longer be avoided within the horizon.

The lanelet network is read once, and cut down every frame to the lanelets
within :attr:`EvaluatorSettings.radius_m` of the ego, so that a measure deep-
copying the scenario (CriMe's do) copies a block, not a city.  That block is
moved so the ego is at the origin and turned so the lane under it runs along
+x (:meth:`Evaluator._alignment` says why).
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: The CriMe measures, by the key the framework reads them under, and the
#: CriMe class that computes each (its name in ``commonroad_crime.measure``).
CRIME_MEASURES: dict[str, str] = {
    "crime_ttc_s": "TTC",
    "crime_thw_s": "THW",
    "crime_hw_m": "HW",
    "crime_ttb_s": "TTB",
    "crime_ttce_s": "TTCE",
    "crime_dce_m": "DCE",
    "crime_a_long_req_mps2": "ALongReq",
    "crime_a_lat_req_mps2": "ALatReq",
}
#: CommonRoad-Reach's: the area of the ego's drivable area at the end of the
#: horizon (m^2), and summed over the horizon's steps (CriMe's DA, m^2 * steps).
REACH_DRIVABLE_AREA_M2 = "reach_drivable_area_m2"
REACH_DRIVABLE_AREA_SUM_M2 = "reach_drivable_area_sum_m2"
REACH_MEASURES: tuple[str, ...] = (REACH_DRIVABLE_AREA_M2, REACH_DRIVABLE_AREA_SUM_M2)

MEASURES: tuple[str, ...] = (*CRIME_MEASURES, *REACH_MEASURES)

#: How far off +x the ego's lane is turned (rad): some of CriMe's measures
#: (ALongReq) take a lane heading of exactly 0 for one they could not read.
_TILT = 1e-3

#: The CommonRoad obstacle type of each kind of road user the frame names.
_OBSTACLE_TYPES: dict[str, str] = {
    "car": "CAR",
    "truck": "TRUCK",
    "bus": "BUS",
    "motorcycle": "MOTORCYCLE",
    "bicycle": "BICYCLE",
    "pedestrian": "PEDESTRIAN",
}


@dataclass(frozen=True)
class Actor:
    """A road user in the frame, in the CommonRoad frame of the map.

    Attributes:
        id: Its id, unique in the frame; not ``0`` (CommonRoad's ids are
            positive).
        kind: ``car``, ``truck``, ``bus``, ``motorcycle``, ``bicycle`` or
            ``pedestrian``.
        x, y: Where its centre is (m).
        yaw: Where it heads (rad, counter-clockwise from +x).
        vx, vy: Its velocity (m/s), in the map's frame.
        ax, ay: Its acceleration (m/s^2), in the map's frame.
        yaw_rate: How fast it turns (rad/s).
        length, width: Its box (m).
    """

    id: int
    kind: str
    x: float
    y: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    yaw_rate: float = 0.0
    length: float = 4.5
    width: float = 1.8

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Actor:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in value.items() if k in known})


@dataclass(frozen=True)
class EvaluatorSettings:
    """How a frame is evaluated.

    Attributes:
        dt: The time step of the scenario built (s).
        horizon_steps: How many steps the road users' trajectories run on, and
            the reachable set is computed over.
        radius_m: How far from the ego the lanelets are kept (m).
        reach_settings: Overrides of CommonRoad-Reach's configuration, by
            section and key (``{"planning": {"size_grid": 0.3}}``).
        share_frame: Let CriMe's measures share the frame's scenario rather
            than each copy it (:func:`_shared_frame`); the same values, in a
            fifth of the time.  Off only to check that.
    """

    dt: float = 0.1
    horizon_steps: int = 30
    radius_m: float = 150.0
    reach_settings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    share_frame: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluatorSettings:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in value.items() if k in known})


def _finite_or_none(value: Any) -> float | None:
    """A measure's value as the framework reads it: ``inf`` is kept (no
    conflict: TTC never comes), NaN and ``None`` (nothing to measure) are
    ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


class Evaluator:
    """Evaluates frames on one map.

    Args:
        map_xml: The CommonRoad 2020a scenario of the map (roadgen's
            ``Map.export_commonroad``); only its lanelet network is read.
        settings: How a frame is evaluated.
    """

    def __init__(self, map_xml: str, settings: EvaluatorSettings | None = None) -> None:
        from commonroad.common.file_reader import CommonRoadFileReader

        self.settings = settings or EvaluatorSettings()
        self._config: Any = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scenario, _ = CommonRoadFileReader(map_xml).open(lanelet_assignment=False)
        self._network = scenario.lanelet_network
        self._scenario_id = scenario.scenario_id
        # The map's ids and the road users' share one space: a road user's id
        # is offset past the map's.
        used = [lanelet.lanelet_id for lanelet in self._network.lanelets]
        used += [sign.traffic_sign_id for sign in self._network.traffic_signs]
        used += [light.traffic_light_id for light in self._network.traffic_lights]
        for intersection in self._network.intersections:
            used.append(intersection.intersection_id)
            used += [incoming.incoming_id for incoming in intersection.incomings]
        self._id_base = 10 ** len(str(max(used, default=0) + 1))
        self._centres = {
            lanelet.lanelet_id: np.asarray(lanelet.center_vertices)
            for lanelet in self._network.lanelets
        }

    # ------------------------------------------------------------------
    # The scenario of a frame
    # ------------------------------------------------------------------

    def _local_network(self, x: float, y: float, angle: float) -> Any:
        """The lanelets within the radius of ``(x, y)``, as a network of their
        own, moved so that ``(x, y)`` is the origin and turned by *angle*."""

        from commonroad.scenario.lanelet import LaneletNetwork

        radius = self.settings.radius_m
        point = np.array([x, y])
        kept = {
            lanelet_id
            for lanelet_id, centre in self._centres.items()
            if float(np.min(np.linalg.norm(centre - point, axis=1))) <= radius
        }
        # Each part is turned before it is added: the network indexes where its
        # lanelets are as they are added, and turning it after would leave the
        # index where the map was.
        network = LaneletNetwork()
        for lanelet_id in sorted(kept):
            lanelet = _without_dangling(self._network.find_lanelet_by_id(lanelet_id), kept)
            lanelet.translate_rotate(-point, angle)
            network.add_lanelet(lanelet)
        for sign in self._network.traffic_signs:
            if sign.position is None or np.linalg.norm(sign.position - point) <= radius:
                sign = copy.deepcopy(sign)
                sign.translate_rotate(-point, angle)
                network.add_traffic_sign(sign, set())
        for light in self._network.traffic_lights:
            if light.position is None or np.linalg.norm(light.position - point) <= radius:
                light = copy.deepcopy(light)
                light.translate_rotate(-point, angle)
                network.add_traffic_light(light, set())
        return network

    def _alignment(self, ego: Actor) -> float | None:
        """The angle that turns the lane under the ego to run along +x (nearly:
        :data:`_TILT` off it); ``None`` when the ego is on no lanelet.

        CriMe projects a road user's speed onto the lane with the cosine of the
        lane's own heading, which is the speed along the lane only where the
        lane runs along +x: on a lane heading north every road user would read
        as standing still.  A frame turned this way reads as CriMe means it,
        and distances, times and areas do not change with the turn.
        """
        found = self._network.find_lanelet_by_position([np.array([ego.x, ego.y])])[0]
        if not found:
            return None
        centre = self._centres[_best_lanelet(self._network, found, ego.yaw)]
        point = np.array([ego.x, ego.y])
        i = int(np.argmin(np.linalg.norm(centre - point, axis=1)))
        a, b = (centre[i], centre[i + 1]) if i + 1 < len(centre) else (centre[i - 1], centre[i])
        return _TILT - math.atan2(b[1] - a[1], b[0] - a[0])

    def _obstacle(self, actor: Actor, network: Any) -> Any:
        from commonroad.geometry.shape import Rectangle
        from commonroad.prediction.prediction import TrajectoryPrediction
        from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
        from commonroad.scenario.state import CustomState, InitialState
        from commonroad.scenario.trajectory import Trajectory

        dt = self.settings.dt
        shape = Rectangle(length=max(actor.length, 0.1), width=max(actor.width, 0.1))
        speed = math.hypot(actor.vx, actor.vy)
        heading = math.atan2(actor.vy, actor.vx) if speed > 0.1 else actor.yaw

        def state(step: int) -> dict[str, Any]:
            t = step * dt
            # Constant acceleration along where it goes, never backwards: a
            # braking road user stops rather than reversing.
            along = actor.ax * math.cos(heading) + actor.ay * math.sin(heading)
            stop_t = -speed / along if along < 0.0 else math.inf
            tt = min(t, stop_t)
            v = max(speed + along * tt, 0.0)
            s = speed * tt + 0.5 * along * tt * tt
            a = along if t < stop_t else 0.0
            return {
                "time_step": step,
                "position": np.array(
                    [actor.x + s * math.cos(heading), actor.y + s * math.sin(heading)]
                ),
                "orientation": actor.yaw,
                "velocity": v * math.cos(heading),
                "velocity_y": v * math.sin(heading),
                "acceleration": a * math.cos(heading),
                "acceleration_y": a * math.sin(heading),
                "yaw_rate": actor.yaw_rate if step == 0 else 0.0,
                "slip_angle": 0.0,
            }

        first = state(0)
        initial = InitialState(
            **{k: v for k, v in first.items() if k not in ("velocity_y", "acceleration_y")}
        )
        # Not InitialState's fields, but CriMe's point-mass reading of a state
        # takes them where it finds them, as it sets them itself.
        initial.velocity_y = first["velocity_y"]
        initial.acceleration_y = first["acceleration_y"]
        trajectory = Trajectory(
            1, [CustomState(**state(k)) for k in range(1, self.settings.horizon_steps + 1)]
        )
        obstacle_type = ObstacleType(_OBSTACLE_TYPES.get(actor.kind, "CAR").lower())
        obstacle = DynamicObstacle(
            self._id_base + actor.id,
            obstacle_type,
            shape,
            initial,
            TrajectoryPrediction(trajectory, shape),
        )
        _assign_lanelets(obstacle, network)
        return obstacle

    def scenario(self, ego: Actor, others: Iterable[Actor], angle: float = 0.0) -> Any:
        """The CommonRoad scenario of a frame: the lanelets around the ego and
        every road user on them, moved so that the ego is at the origin and
        turned by *angle*."""
        from commonroad.scenario.scenario import Scenario

        network = self._local_network(ego.x, ego.y, angle)
        scenario = Scenario(self.settings.dt, self._scenario_id)
        scenario.add_objects(network)
        for actor in [ego, *(a for a in others if a.id != ego.id)]:
            scenario.add_objects(self._obstacle(_moved(actor, ego.x, ego.y, angle), network))
        return scenario

    # ------------------------------------------------------------------
    # The measures
    # ------------------------------------------------------------------

    def evaluate(
        self, ego: Actor, others: Iterable[Actor], measures: Iterable[str]
    ) -> tuple[dict[str, float | None], dict[str, str]]:
        """The measures asked for in a frame; with the error of each that
        failed (its value then ``None``)."""
        wanted = list(dict.fromkeys(measures))
        unknown = [m for m in wanted if m not in MEASURES]
        values: dict[str, float | None] = {}
        errors: dict[str, str] = {m: "unknown measure" for m in unknown}
        wanted = [m for m in wanted if m not in unknown]
        if not wanted:
            return values, errors
        angle = self._alignment(ego)
        if angle is None:
            # Off the map: nothing CriMe or Reach measure is defined.
            for m in wanted:
                values[m] = None
            return values, errors
        scenario = self.scenario(ego, list(others), angle)
        ego_id = self._id_base + ego.id
        ego = _moved(ego, ego.x, ego.y, angle)
        if any(m in CRIME_MEASURES for m in wanted):
            self._crime(
                scenario, ego_id, [m for m in wanted if m in CRIME_MEASURES], values, errors
            )
        if any(m in REACH_MEASURES for m in wanted):
            self._reach(scenario, ego, [m for m in wanted if m in REACH_MEASURES], values, errors)
        return values, errors

    def _crime(
        self,
        scenario: Any,
        ego_id: int,
        wanted: list[str],
        values: dict[str, float | None],
        errors: dict[str, str],
    ) -> None:
        import commonroad_crime.measure as crime_measure

        config = self._crime_config()
        config.update(ego_id=ego_id, sce=scenario)
        with _shared_frame() if self.settings.share_frame else nullcontext():
            for key in wanted:
                try:
                    measure = getattr(crime_measure, CRIME_MEASURES[key])(config)
                    values[key] = _finite_or_none(measure.compute_criticality(0, verbose=False))
                except Exception as exc:  # a measure undefined here, not a broken worker
                    values[key] = None
                    errors[key] = f"{type(exc).__name__}: {exc}"

    def _crime_config(self) -> Any:
        """CriMe's configuration, made once: making one builds the vehicle's
        parameters through omegaconf, a third of a frame's time when it was
        made for every measure."""
        if self._config is None:
            from commonroad_crime.data_structure.configuration import CriMeConfiguration

            config = CriMeConfiguration()
            config.debug.save_plots = False
            config.debug.draw_visualization = False
            self._config = config
        return self._config

    def _reach(
        self,
        scenario: Any,
        ego: Actor,
        wanted: list[str],
        values: dict[str, float | None],
        errors: dict[str, str],
    ) -> None:
        try:
            areas = self._reachable_areas(scenario, ego)
        except Exception as exc:
            for key in wanted:
                values[key] = None
                errors[key] = f"{type(exc).__name__}: {exc}"
            return
        for key in wanted:
            if key == REACH_DRIVABLE_AREA_M2:
                values[key] = areas[-1] if areas else 0.0
            else:
                values[key] = float(sum(areas))

    def _reachable_areas(self, scenario: Any, ego: Actor) -> list[float]:
        """The area of the ego's drivable area at each step of the horizon."""
        from commonroad.common.util import Interval
        from commonroad.geometry.shape import Rectangle
        from commonroad.planning.goal import GoalRegion
        from commonroad.planning.planning_problem import PlanningProblem
        from commonroad.scenario.state import InitialState
        from commonroad_clcs.pycrccosy import CurvilinearCoordinateSystem
        from commonroad_crime.utility.general import generate_reference_path
        from commonroad_reach.data_structure.configuration_builder import (
            ConfigurationBuilder,
        )
        from commonroad_reach.data_structure.reach.reach_interface import (
            ReachableSetInterface,
        )
        from commonroad_reach.utility.reach_operation import compute_area_of_reach_nodes

        lanelet_ids = scenario.lanelet_network.find_lanelet_by_position([np.array([ego.x, ego.y])])[
            0
        ]
        reference = generate_reference_path(
            _best_lanelet(scenario.lanelet_network, lanelet_ids, ego.yaw),
            scenario.lanelet_network,
        )
        clcs = CurvilinearCoordinateSystem(reference, 25.0, 0.1)
        speed = math.hypot(ego.vx, ego.vy)
        initial = InitialState(
            time_step=0,
            position=np.array([ego.x, ego.y]),
            orientation=ego.yaw,
            velocity=speed,
            acceleration=0.0,
            yaw_rate=0.0,
            slip_angle=0.0,
        )
        goal = GoalRegion(
            [
                InitialState(
                    time_step=Interval(0, self.settings.horizon_steps),
                    position=Rectangle(1.0, 1.0, np.array([ego.x, ego.y])),
                )
            ]
        )
        # Past every road user's id.
        problem = PlanningProblem(self._id_base * 2 + 1 + ego.id, initial, goal)
        others = scenario.__class__(scenario.dt, scenario.scenario_id)
        others.add_objects(scenario.lanelet_network)
        for obstacle in scenario.obstacles:
            if obstacle.obstacle_id != self._id_base + ego.id:
                others.add_objects(obstacle)

        config = ConfigurationBuilder().build_configuration(str(scenario.scenario_id))
        # Reach saves its configuration under ./output/ every computation.
        config.debug.save_config = False
        config.planning.dt = self.settings.dt
        config.planning.steps_computation = self.settings.horizon_steps
        config.vehicle.ego.length = ego.length
        config.vehicle.ego.width = ego.width
        config.vehicle.ego.v_max = max(config.vehicle.ego.v_max, speed + 1.0)
        config.vehicle.ego.v_lon_max = max(config.vehicle.ego.v_lon_max, speed + 1.0)
        for section, overrides in self.settings.reach_settings.items():
            target = getattr(config, section)
            for name, value in overrides.items():
                setattr(target, name, value)
        config.update(scenario=others, planning_problem=problem, CLCS=clcs)
        interface = ReachableSetInterface(config)
        interface.compute_reachable_sets(verbose=False)
        steps = sorted(interface.reachable_set)
        return [float(compute_area_of_reach_nodes(interface.reachable_set[t])) for t in steps]


@contextmanager
def _shared_frame() -> Iterator[None]:
    """Let CriMe's measures share one frame's scenario instead of copying it.

    Every CriMe measure deep-copies the scenario as it is made, and every
    other road user it measures against: half a frame's time.  A frame's
    scenario is made for that frame alone and no measure the worker evaluates
    changes it (CriMe fills in state fields the worker already sets), so
    while this holds CriMe's base class copies nothing; the copies CriMe's
    simulations make of the states they step stay.  The ego's reference path
    and curvilinear frame, which each measure built anew, are built once per
    ego lanelet, and a lanelet's bounds are smoothed once (CriMe smoothed them
    every time it read a road user's heading on the lane, a third of what was
    left).
    """
    import commonroad_crime.data_structure.base as base
    import commonroad_crime.utility.solver as solver
    from commonroad_clcs.pycrccosy import CurvilinearCoordinateSystem
    from commonroad_crime.utility.general import generate_reference_path

    frames: dict[tuple[int, int], Any] = {}

    def update_clcs(self: Any) -> None:
        ego = self.ego_vehicle
        lanelet_id = next(
            iter(ego.prediction.center_lanelet_assignment[ego.initial_state.time_step])
        )
        key = (id(self.sce), lanelet_id)
        if key not in frames:
            path = generate_reference_path(lanelet_id, self.sce.lanelet_network)
            frames[key] = CurvilinearCoordinateSystem(path)
        self.configuration.update(CLCS=frames[key])

    smooth = solver.smoothing_reference_path
    smoothed: dict[tuple[Any, ...], Any] = {}

    def smoothing_reference_path(
        path: Any, smooth_factor: Any = None, weight_coefficient: Any = None
    ) -> Any:
        points = np.ascontiguousarray(path, dtype=float)
        key = (points.shape, points.tobytes(), smooth_factor, weight_coefficient)
        if key not in smoothed:
            smoothed[key] = smooth(points, smooth_factor, weight_coefficient)
        return np.array(smoothed[key], copy=True)

    module_copy, method = base.copy, base.CriMeBase._update_clcs
    base.copy = SimpleNamespace(deepcopy=lambda value, memo=None: value, copy=copy.copy)
    base.CriMeBase._update_clcs = update_clcs
    solver.smoothing_reference_path = smoothing_reference_path
    try:
        yield
    finally:
        base.copy, base.CriMeBase._update_clcs = module_copy, method
        solver.smoothing_reference_path = smooth


def _moved(actor: Actor, x: float, y: float, angle: float) -> Actor:
    """*actor* in the frame whose origin is ``(x, y)``, turned by *angle*."""
    c, s = math.cos(angle), math.sin(angle)

    def turned(u: float, v: float) -> tuple[float, float]:
        return c * u - s * v, s * u + c * v

    px, py = turned(actor.x - x, actor.y - y)
    vx, vy = turned(actor.vx, actor.vy)
    ax, ay = turned(actor.ax, actor.ay)
    return replace(actor, x=px, y=py, yaw=actor.yaw + angle, vx=vx, vy=vy, ax=ax, ay=ay)


def _best_lanelet(network: Any, lanelet_ids: list[int], yaw: float) -> int:
    """Of the lanelets under the ego, the one heading most its way."""

    def misalignment(lanelet_id: int) -> float:
        centre = network.find_lanelet_by_id(lanelet_id).center_vertices
        dx, dy = centre[-1] - centre[0]
        return -math.cos(math.atan2(dy, dx) - yaw)

    return min(lanelet_ids, key=misalignment)


def _without_dangling(lanelet: Any, kept: set[int]) -> Any:
    """*lanelet* with its links to lanelets not *kept* dropped, so a network of
    the kept ones refers to nothing it lacks."""

    copied = copy.deepcopy(lanelet)
    copied.predecessor = [i for i in lanelet.predecessor if i in kept]
    copied.successor = [i for i in lanelet.successor if i in kept]
    if lanelet.adj_left is not None and lanelet.adj_left not in kept:
        copied.adj_left = None
        copied.adj_left_same_direction = None
    if lanelet.adj_right is not None and lanelet.adj_right not in kept:
        copied.adj_right = None
        copied.adj_right_same_direction = None
    return copied


def _assign_lanelets(obstacle: Any, network: Any) -> None:
    """Which lanelets each state of *obstacle*'s occupies and has its centre in,
    as CriMe reads them (``prediction.center_lanelet_assignment``)."""
    states = [obstacle.initial_state, *obstacle.prediction.trajectory.state_list]
    positions = [s.position for s in states]
    found = network.find_lanelet_by_position(positions)
    obstacle.prediction.center_lanelet_assignment = {
        s.time_step: set(ids) for s, ids in zip(states, found, strict=True)
    }
    obstacle.prediction.shape_lanelet_assignment = dict(
        obstacle.prediction.center_lanelet_assignment
    )
    obstacle.initial_center_lanelet_ids = set(found[0])
    obstacle.initial_shape_lanelet_ids = set(found[0])
