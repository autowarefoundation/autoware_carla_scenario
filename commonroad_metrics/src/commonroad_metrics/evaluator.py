"""Criticality of one frame of a run, read with CriMe and CommonRoad-Reach.

A frame is the road users where they are now, in the CommonRoad frame of the
map (the OpenDRIVE inertial frame roadgen writes the scenario in: x east, y
north, angles counter-clockwise).  The evaluator turns it into a CommonRoad
scenario on the map's lanelet network -- every road user a dynamic obstacle
whose trajectory runs on from where it is at its current velocity and
acceleration, over the horizon -- and evaluates the measures asked for on the
ego at its first time step:

* CriMe's (``commonroad_crime``) time, distance and acceleration measures --
  TTC, THW, TTB, the threat numbers BTN and STN and the rest -- each the most
  critical over the other road users, as CriMe folds them (the least TTC, the
  greatest required deceleration) -- and the time-to-react TTR, which
  simulates the ego's evasive manoeuvres (seconds a frame);
* CommonRoad-Reach's (``commonroad_reach``) reachable set of the ego, its
  drivable area at the end of the horizon: what is left of the road the ego can
  still reach without hitting anyone.  ``0`` means none: a collision can no
  longer be avoided within the horizon;
* the worst-case time-to-react (WTTR, :meth:`Evaluator._wttr`): how long the
  ego can go on as it goes before no way it can go is free of collisions to
  the end of the horizon, by the reachable set.

The lanelet network is read once, and cut down every frame to the lanelets
within :attr:`EvaluatorSettings.radius_m` of the ego, so that a measure deep-
copying the scenario (CriMe's do) copies a block, not a city.  That block is
moved so the ego is at the origin and turned so the lane under it runs along
+x (:meth:`Evaluator._alignment` says why), every lanelet of it reversed when
that lane runs against the ego (:meth:`Evaluator._local_network`).
"""

from __future__ import annotations

import copy
import fcntl
import functools
import logging
import math
import warnings
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
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
    "crime_btn": "BTN",
    "crime_stn": "STN",
    "crime_ttr_s": "TTR",
}
#: CriMe's TTR: the latest time an evasive manoeuvre (braking, kick-down,
#: steering left or right) still avoids the collision.  CriMe reads ``-inf``
#: when none does even now, read ``0`` here as :data:`CRIME_WTTR_S` is.
CRIME_TTR_S = "crime_ttr_s"
#: CriMe's TTB: the latest time braking at full still avoids the collision.
#: A TTM, as TTR is, CriMe reads it ``-inf`` when braking no longer avoids it
#: even now (no step does, or the ego is in contact already: TTC* ``0``),
#: read ``0`` here as TTR is.
CRIME_TTB_S = "crime_ttb_s"
#: The measures CriMe reads ``-inf`` when no manoeuvre is left even now, read
#: ``0`` here: the time left is none, not less than none, and ``min``
#: aggregation must not report ``-inf`` as a run's least time left.
_NO_TIME_LEFT: frozenset[str] = frozenset({CRIME_TTR_S, CRIME_TTB_S})
#: CommonRoad-Reach's: the area of the ego's drivable area at the end of the
#: horizon (m^2), and summed over the horizon's steps (CriMe's DA, m^2 * steps).
REACH_DRIVABLE_AREA_M2 = "reach_drivable_area_m2"
REACH_DRIVABLE_AREA_SUM_M2 = "reach_drivable_area_sum_m2"
#: The worst-case time-to-react (s), after CriMe's WTTR (Sontges, Koschi and
#: Althoff, IV 2018), read here: CriMe's own loads CommonRoad-Reach's scenario
#: from a file rather than the frame's.
CRIME_WTTR_S = "crime_wttr_s"
REACH_MEASURES: tuple[str, ...] = (
    REACH_DRIVABLE_AREA_M2,
    REACH_DRIVABLE_AREA_SUM_M2,
    CRIME_WTTR_S,
)

MEASURES: tuple[str, ...] = (*CRIME_MEASURES, *REACH_MEASURES)

#: How far off +x the ego's lane is turned (rad): some of CriMe's measures
#: (ALongReq) take a lane heading of exactly 0 for one they could not read.
_TILT = 1e-3

#: The least length and width a box is given (m): CommonRoad's shapes and
#: Reach's discs need a positive extent.  Obstacles and the ego alike.
_MIN_EXTENT_M = 0.1

#: Actor's fields that must be finite numbers (every one but id and kind).
_ACTOR_NUMBERS = ("x", "y", "yaw", "vx", "vy", "ax", "ay", "yaw_rate", "length", "width")


class NonFiniteError(ValueError):
    """A number that must be finite is not: a NaN, an infinity or an int too
    big for a float.  A ``ValueError`` as any other refusal; told apart so
    that the worker can answer a frame holding one with its measures failed,
    a reading gone bad in one frame, rather than the request refused
    (:func:`failed_frame`)."""


def _finite(name: str, value: Any) -> float:
    """*value* as a float, refused (``ValueError`` naming *name*) unless it is
    a finite number -- :class:`NonFiniteError` when it is a number but not
    finite.  A NaN read as a speed or a headway gives the least critical value
    (TTC ``inf``, ALongReq ``0``), and one read as a position makes every
    measure ``None``: neither says the input was bad."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:  # an int too big for a float
        raise NonFiniteError(f"{name} must be finite, got an int too big for a float") from None
    if not math.isfinite(number):
        raise NonFiniteError(f"{name} must be finite, got {number!r}")
    return number


#: The error a measure that is not one of :data:`MEASURES` fails with.
UNKNOWN_MEASURE = "unknown measure"


def failed_frame(
    measures: Iterable[str], error: str
) -> tuple[dict[str, float | None], dict[str, str]]:
    """What :meth:`Evaluator.evaluate` gives for a frame none of whose
    *measures* could be evaluated: each known one ``None``, failed with
    *error*; each unknown one failed with :data:`UNKNOWN_MEASURE`, as ever."""
    wanted = list(dict.fromkeys(measures))
    values: dict[str, float | None] = {m: None for m in wanted if m in MEASURES}
    errors = {m: error if m in MEASURES else UNKNOWN_MEASURE for m in wanted}
    return values, errors


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
        id: Its id, a positive int (CommonRoad's ids are positive), unique
            among the road users of a frame (:meth:`Evaluator.evaluate`).
        kind: ``car``, ``truck``, ``bus``, ``motorcycle``, ``bicycle`` or
            ``pedestrian``.
        x, y: Where its centre is (m).
        yaw: Where it heads (rad, counter-clockwise from +x).
        vx, vy: Its velocity (m/s), in the map's frame.
        ax, ay: Its acceleration (m/s^2), in the map's frame.
        yaw_rate: How fast it turns (rad/s).
        length, width: Its box (m).

    Every number but ``id`` must be finite: a NaN or an infinity is refused
    with a :class:`NonFiniteError` naming the field (the worker answers the
    frame with each measure ``null`` and failed with that message), anything
    but a number with a ``ValueError`` (the worker answers ``ok: false``), as
    is an ``id`` that is not a positive int (a float, a bool, ``0``).
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

    def __post_init__(self) -> None:
        if isinstance(self.id, bool) or not isinstance(self.id, int) or self.id < 1:
            raise ValueError(f"id must be a positive int, got {self.id!r}")
        for name in _ACTOR_NUMBERS:
            _finite(name, getattr(self, name))

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
            section and key (``{"reachable_set": {"size_grid": 0.3}}``, the
            grid the drivable area is computed on); a section or key Reach's
            configuration does not have is refused
            (:func:`_apply_reach_settings`).
        share_frame: Let CriMe's measures share the frame's scenario rather
            than each copy it (:func:`_shared_frame`); the same values, in a
            fifth of the time.  Off only to check that.

    ``dt`` and ``radius_m`` must be finite and positive, ``horizon_steps`` an
    int of at least 1; anything else is refused with a ``ValueError`` (the
    worker's ``init`` answers ``ok: false``) rather than breaking every frame
    after it.
    """

    dt: float = 0.1
    horizon_steps: int = 30
    radius_m: float = 150.0
    reach_settings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    share_frame: bool = True

    def __post_init__(self) -> None:
        for name in ("dt", "radius_m"):
            if not _finite(name, getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        steps = self.horizon_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"horizon_steps must be an int of at least 1, got {steps!r}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluatorSettings:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in value.items() if k in known})


@functools.cache
def _import_reach() -> None:
    """Import CommonRoad-Reach, which CriMe imports too, one process at a time.

    Two of its modules compile a Cython module with pyximport the first
    time they are imported. The module is built into a directory every
    process shares (``~/.pyxbld``). Workers started together build it
    together, and one then loads the other's half-written module: "file too
    short", or "does not define module export function". Holding a lock on
    that directory across the import makes the first worker build it and the
    others load what it built.
    """
    build = Path.home() / ".pyxbld"
    build.mkdir(parents=True, exist_ok=True)
    with (build / ".commonroad_metrics.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            import commonroad_reach.data_structure.reach.reach_interface  # noqa: F401
            import commonroad_reach.data_structure.regular_grid  # noqa: F401
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


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

    Raises:
        ValueError: The settings' ``reach_settings`` name a section or key
            CommonRoad-Reach's configuration does not have.
    """

    def __init__(self, map_xml: str, settings: EvaluatorSettings | None = None) -> None:
        from commonroad.common.file_reader import CommonRoadFileReader

        self.settings = settings or EvaluatorSettings()
        self._config: Any = None
        self._reach_config: Any = None
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
        # Every segment of every lanelet's centre line, and whose it is: a
        # straight lanelet is just its two end points, and its middle can be
        # near the ego with both ends far off.
        starts, ends, owners = [], [], []
        for lanelet_id, centre in self._centres.items():
            if len(centre) == 1:
                centre = np.vstack([centre, centre])
            starts.append(centre[:-1])
            ends.append(centre[1:])
            owners.append(np.full(len(centre) - 1, lanelet_id))
        self._segment_starts = np.concatenate(starts) if starts else np.zeros((0, 2))
        self._segment_ends = np.concatenate(ends) if ends else np.zeros((0, 2))
        self._segment_owners = np.concatenate(owners) if owners else np.zeros(0, dtype=int)
        if self.settings.reach_settings:
            # A section or key Reach does not have is refused now, by init,
            # rather than failing every frame's Reach measures.
            self._reach_base_config()

    # ------------------------------------------------------------------
    # The scenario of a frame
    # ------------------------------------------------------------------

    def _local_network(self, x: float, y: float, angle: float, reverse: bool = False) -> Any:
        """The lanelets within the radius of ``(x, y)``, as a network of their
        own, moved so that ``(x, y)`` is the origin and turned by *angle*;
        with *reverse*, every lanelet runs the other way (:func:`_reversed`).

        The ego's lane is the lanelet under it heading most its way
        (:func:`_best_lanelet`), and every measure reads along it: CriMe's
        curvilinear frame, whose "ahead" is along the lanelet, and Reach's.
        When even that one runs against the ego -- the ego overtaking through
        the oncoming lane, or driving the wrong way -- its "ahead" is behind
        the ego: a car coming at it head-on reads as behind (TTC ``inf``,
        ALongReq ``0``), and Reach reads the ego as reversing.  The network
        is reversed then, all of it, so that the lane runs the ego's way and
        every link stays consistent: predecessors and successors swap, left
        and right swap, and two lanelets reversed together keep their
        driving directions to one another.
        """

        from commonroad.scenario.lanelet import LaneletNetwork

        radius = self.settings.radius_m
        point = np.array([x, y])
        # A lanelet is kept when its centre line comes within the radius --
        # anywhere along it, not only at its vertices -- and always when
        # (x, y) is on it.
        kept = {int(i) for i in self._segment_owners[self._segment_distances(point) <= radius]}
        kept.update(self._network.find_lanelet_by_position([point])[0])
        # Each part is turned before it is added: the network indexes where its
        # lanelets are as they are added, and turning it after would leave the
        # index where the map was.
        network = LaneletNetwork()
        for lanelet_id in sorted(kept):
            lanelet = _without_dangling(self._network.find_lanelet_by_id(lanelet_id), kept)
            if reverse:
                lanelet = _reversed(lanelet)
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

    def _segment_distances(self, point: np.ndarray) -> np.ndarray:
        """How far *point* is from each segment of the lanelets' centre lines."""
        a, b = self._segment_starts, self._segment_ends
        ab = b - a
        length2 = np.einsum("ij,ij->i", ab, ab)
        t = np.einsum("ij,ij->i", point - a, ab) / np.where(length2 > 0.0, length2, 1.0)
        nearest = a + np.clip(t, 0.0, 1.0)[:, None] * ab
        return np.asarray(np.linalg.norm(nearest - point, axis=1))

    def _alignment(self, ego: Actor) -> tuple[float, bool] | None:
        """The angle that turns the lane under the ego to run along +x (nearly:
        :data:`_TILT` off it), and whether that lane runs against the ego
        (``cos(heading - yaw) < 0`` with the lanelet's heading from its first
        to its last centre point, as :func:`_best_lanelet` reads it): the
        network is then reversed (:meth:`_local_network`), and the angle
        turns the lane as reversed to run along +x.  ``None`` when the ego is
        on no lanelet.

        CriMe projects a road user's speed onto the lane with the cosine of the
        lane's own heading, which is the speed along the lane only where the
        lane runs along +x: on a lane heading north every road user would read
        as standing still.  A frame turned this way reads as CriMe means it,
        and distances, times and areas do not change with the turn.
        """
        found = self._network.find_lanelet_by_position([np.array([ego.x, ego.y])])[0]
        if not found:
            return None
        best = _best_lanelet(self._network, found, ego.yaw)
        dx, dy = self._centres[best][-1] - self._centres[best][0]
        against = math.cos(math.atan2(dy, dx) - ego.yaw) < 0.0
        if against:
            # Reversed, the one heading most against the ego heads most its
            # way: the one the measures read (_best_lanelet on the network).
            best = _best_lanelet(self._network, found, ego.yaw + math.pi)
        centre = self._centres[best]
        point = np.array([ego.x, ego.y])
        i = int(np.argmin(np.linalg.norm(centre - point, axis=1)))
        a, b = (centre[i], centre[i + 1]) if i + 1 < len(centre) else (centre[i - 1], centre[i])
        if against:
            a, b = b, a
        return _TILT - math.atan2(b[1] - a[1], b[0] - a[0]), against

    def _obstacle(self, actor: Actor, network: Any) -> Any:
        from commonroad.geometry.shape import Rectangle
        from commonroad.prediction.prediction import TrajectoryPrediction
        from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
        from commonroad.scenario.state import CustomState, InitialState
        from commonroad.scenario.trajectory import Trajectory

        dt = self.settings.dt
        shape = Rectangle(
            length=max(actor.length, _MIN_EXTENT_M), width=max(actor.width, _MIN_EXTENT_M)
        )
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

    def scenario(
        self, ego: Actor, others: Iterable[Actor], angle: float = 0.0, reverse: bool = False
    ) -> Any:
        """The CommonRoad scenario of a frame: the lanelets around the ego and
        every road user on them, moved so that the ego is at the origin and
        turned by *angle*, and with *reverse* running the other way
        (:meth:`_local_network`)."""
        from commonroad.scenario.scenario import Scenario

        network = self._local_network(ego.x, ego.y, angle, reverse)
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
        failed (its value then ``None``).

        Raises:
            ValueError: Two of *others* share an id (one sharing the ego's is
                the ego, and left out).
        """
        others = list(others)
        seen: set[int] = set()
        for actor in others:
            if actor.id in seen and actor.id != ego.id:
                raise ValueError(f"road users share the id {actor.id!r}: ids must be unique")
            seen.add(actor.id)
        wanted = list(dict.fromkeys(measures))
        unknown = [m for m in wanted if m not in MEASURES]
        values: dict[str, float | None] = {}
        errors: dict[str, str] = {m: UNKNOWN_MEASURE for m in unknown}
        wanted = [m for m in wanted if m not in unknown]
        if not wanted:
            return values, errors
        alignment = self._alignment(ego)
        if alignment is None:
            # Off the map: nothing CriMe or Reach measure is defined.
            for m in wanted:
                values[m] = None
            return values, errors
        angle, reverse = alignment
        scenario = self.scenario(ego, others, angle, reverse)
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
        _import_reach()
        import commonroad_crime.measure as crime_measure

        config = self._crime_config()
        config.update(ego_id=ego_id, sce=scenario)
        signed = _signed_crime_measures()
        shared = _shared_frame() if self.settings.share_frame else nullcontext()
        with _unwrapped_lane_headings(), shared:
            for key in wanted:
                name = CRIME_MEASURES[key]
                try:
                    measure = signed.get(name, getattr(crime_measure, name))(config)
                    value = _finite_or_none(measure.compute_criticality(0, verbose=False))
                    if key in _NO_TIME_LEFT and value == -math.inf:
                        value = 0.0
                    values[key] = value
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
            frame = _ReachFrame.of(scenario, ego, self._id_base)
        except Exception as exc:
            for key in wanted:
                values[key] = None
                errors[key] = f"{type(exc).__name__}: {exc}"
            return
        areas: list[float] | None = None
        for key in wanted:
            try:
                if key == CRIME_WTTR_S:
                    values[key] = self._wttr(frame, ego)
                    continue
                if areas is None:
                    areas = self._reachable_areas(frame, ego)
                if key == REACH_DRIVABLE_AREA_M2:
                    values[key] = areas[-1] if areas else 0.0
                else:
                    values[key] = float(sum(areas))
            except Exception as exc:
                values[key] = None
                errors[key] = f"{type(exc).__name__}: {exc}"

    def _reachable_set(self, frame: _ReachFrame, ego: Actor, start: int = 0) -> Any:
        """CommonRoad-Reach's reachable set of the ego, from where *ego* is at
        step *start* to the end of the horizon, among the other road users as
        they go on."""
        _import_reach()
        from commonroad.common.util import Interval
        from commonroad.geometry.shape import Rectangle
        from commonroad.planning.goal import GoalRegion
        from commonroad.planning.planning_problem import PlanningProblem
        from commonroad.scenario.state import InitialState
        from commonroad_reach.data_structure.reach import reach_set_cpp
        from commonroad_reach.data_structure.reach.reach_interface import (
            ReachableSetInterface,
        )

        horizon = self.settings.horizon_steps
        speed = math.hypot(ego.vx, ego.vy)
        # Signed along where the ego heads: a reversing ego goes backwards.
        velocity = math.copysign(speed, _along_lane(ego.vx, ego.vy, ego.yaw))
        initial = InitialState(
            time_step=start,
            position=np.array([ego.x, ego.y]),
            orientation=ego.yaw,
            velocity=velocity,
            acceleration=0.0,
            yaw_rate=0.0,
            slip_angle=0.0,
        )
        goal = GoalRegion(
            [
                InitialState(
                    time_step=Interval(start, horizon),
                    position=Rectangle(1.0, 1.0, np.array([ego.x, ego.y])),
                )
            ]
        )
        # Past every road user's id.
        problem = PlanningProblem(self._id_base * 2 + 1 + ego.id, initial, goal)

        config = copy.deepcopy(self._reach_base_config())
        config.planning.steps_computation = horizon - start
        # Floored as the obstacles' boxes are (Evaluator._obstacle).
        _set_ego_footprint(config, max(ego.length, _MIN_EXTENT_M), max(ego.width, _MIN_EXTENT_M))
        config.vehicle.ego.v_max = max(config.vehicle.ego.v_max, speed + 1.0)
        config.vehicle.ego.v_lon_max = max(config.vehicle.ego.v_lon_max, speed + 1.0)
        config.vehicle.ego.v_lon_min = min(config.vehicle.ego.v_lon_min, velocity - 1.0)
        config.update(scenario=frame.others, planning_problem=problem, CLCS=frame.clcs)
        if frame.checker is None:
            # One collision checker for every reachable set of the frame: the
            # road users' occupancies over the whole horizon, by absolute step,
            # do not depend on where the set starts.
            planning = config.planning
            planned = planning.step_start, planning.steps_computation
            planning.step_start, planning.steps_computation = 0, horizon
            frame.checker = reach_set_cpp.CollisionChecker(config)
            planning.step_start, planning.steps_computation = planned
        made = reach_set_cpp.CollisionChecker
        reach_set_cpp.CollisionChecker = lambda _config: frame.checker  # type: ignore[assignment,misc]
        try:
            interface = ReachableSetInterface(config)
        finally:
            reach_set_cpp.CollisionChecker = made  # type: ignore[misc]
        interface.compute_reachable_sets(verbose=False)
        return interface

    def _reach_base_config(self) -> Any:
        """CommonRoad-Reach's configuration with the settings applied, made
        once (from its YAML files, through omegaconf) and copied for each
        reachable set.

        Raises:
            ValueError: :attr:`EvaluatorSettings.reach_settings` names a
                section or key Reach's configuration does not have
                (:func:`_apply_reach_settings`).
        """
        if self._reach_config is None:
            _import_reach()
            from commonroad_reach.data_structure.configuration_builder import (
                ConfigurationBuilder,
            )

            config = ConfigurationBuilder().build_configuration(str(self._scenario_id))
            # Reach saves its configuration under ./output/ every computation.
            config.debug.save_config = False
            config.planning.dt = self.settings.dt
            _apply_reach_settings(config, self.settings.reach_settings)
            self._reach_config = config
        return self._reach_config

    def _reachable_areas(self, frame: _ReachFrame, ego: Actor) -> list[float]:
        """The area of the ego's drivable area at each step of the horizon."""
        _import_reach()
        from commonroad_reach.utility.reach_operation import compute_area_of_reach_nodes

        interface = self._reachable_set(frame, ego)
        steps = sorted(interface.reachable_set)
        return [float(compute_area_of_reach_nodes(interface.reachable_set[t])) for t in steps]

    def _wttr(self, frame: _ReachFrame, ego: Actor) -> float:
        """The worst-case time-to-react (s), as CriMe's WTTR reads it.

        The ego goes on as it goes until it would hit a road user (TTC*, by
        its predicted trajectory; ``inf`` when it hits no one within the
        horizon).  Before that, the latest step from which a way to the end of
        the horizon free of collisions is still left -- the reachable set
        from the ego's predicted state there is not empty at the end -- is
        found by bisection, as CriMe finds it; WTTR is the time to the step
        after it: ``0`` when no way is left even now.
        """
        hit = frame.first_collision_step(self.settings.horizon_steps)
        if hit is None:
            return math.inf
        low, high = 0, hit
        while low < high:
            mid = (low + high) // 2
            interface = self._reachable_set(frame, frame.ego_at(ego, mid), mid)
            if interface.reachable_set_at_step(self.settings.horizon_steps):
                low = mid + 1
            else:
                high = mid
        return round(low * self.settings.dt, 6)


@dataclass
class _ReachFrame:
    """What CommonRoad-Reach reads of a frame, made once for every reachable
    set computed on it: the ego's predicted trajectory, the other road users
    without it, and the curvilinear frame along the ego's lane."""

    ego_obstacle: Any
    others: Any
    clcs: Any
    #: CommonRoad-Reach's collision checker of the frame, made by its first
    #: reachable set (:meth:`Evaluator._reachable_set`).
    checker: Any = None

    @classmethod
    def of(cls, scenario: Any, ego: Actor, id_base: int) -> _ReachFrame:
        from commonroad_clcs.pycrccosy import CurvilinearCoordinateSystem
        from commonroad_crime.utility.general import generate_reference_path

        network = scenario.lanelet_network
        lanelet_ids = network.find_lanelet_by_position([np.array([ego.x, ego.y])])[0]
        reference = generate_reference_path(_best_lanelet(network, lanelet_ids, ego.yaw), network)
        clcs = CurvilinearCoordinateSystem(reference, 25.0, 0.1)
        ego_id = id_base + ego.id
        others = scenario.__class__(scenario.dt, scenario.scenario_id)
        others.add_objects(network)
        ego_obstacle = None
        for obstacle in scenario.obstacles:
            if obstacle.obstacle_id == ego_id:
                ego_obstacle = obstacle
            else:
                others.add_objects(obstacle)
        return cls(ego_obstacle, others, clcs)

    def ego_at(self, ego: Actor, step: int) -> Actor:
        """*ego* where its predicted trajectory has it at *step*."""
        state = self.ego_obstacle.state_at_time(step)
        vx = float(state.velocity)
        vy = float(getattr(state, "velocity_y", 0.0) or 0.0)
        return replace(
            ego,
            x=float(state.position[0]),
            y=float(state.position[1]),
            yaw=float(state.orientation),
            vx=vx,
            vy=vy,
        )

    def first_collision_step(self, horizon: int) -> int | None:
        """The first step at which the ego, on its predicted trajectory, hits
        another road user on theirs (TTC* over the time step); ``None`` when
        it hits no one within *horizon* steps.  Unlike CriMe's TTC*, the road's
        boundary is no obstacle: the ego's prediction goes straight on, and
        would leave every bend."""
        from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
            create_collision_object,
        )
        from commonroad_dc.collision.trajectory_queries.trajectory_queries import (
            trajectories_collision_dynamic_obstacles,
        )

        if not self.others.dynamic_obstacles:
            return None
        step = trajectories_collision_dynamic_obstacles(
            [create_collision_object(self.ego_obstacle)],
            [create_collision_object(o) for o in self.others.dynamic_obstacles],
            method="fcl",
        )[0]
        return None if step < 0 or step > horizon else int(step)


def _lane_heading(position: float, path_length: Any, orientation: Any) -> float:
    """The lane's heading *position* (m) along it, from the headings of its
    centre line's segments: CriMe's ``get_orientation_point``, wrap-safe.

    CriMe 0.4 stores those headings in [0, 2pi) (``convert_to_0_2pi``) and
    interpolates between two of them linearly, so where they straddle 0 --
    on any bend of a lane the frame turned to run along +x, next to the ego
    -- it reads anything between the two, ~3 rad off (and a heading of
    ``2pi - 0.001`` in place of ``0.001``, not just the same angle).  Its own
    wrap handling only looks for the jump in [-pi, pi], where they never are.
    Here the headings are unwrapped first, so the interpolation runs between
    neighbours; the result is put back in [0, 2pi), as CriMe returns them.
    Off either end the end heading holds, as ``np.interp`` holds it.
    """
    headings = np.unwrap(np.asarray(orientation, dtype=float))
    heading = float(np.interp(position, path_length, headings))
    return heading % (2.0 * math.pi)


@contextmanager
def _unwrapped_lane_headings() -> Iterator[None]:
    """Make CriMe read lane headings with :func:`_lane_heading`.

    Every CriMe reading of a lane's heading at a point goes through
    ``compute_lanelet_width_orientation``, which looks ``get_orientation_point``
    up in its module when it runs: replacing it there covers TTC, ALongReq,
    ALatReq, MSD, the jerk measures, the simulations behind TTR, and the
    signed measures here (:func:`_signed_crime_measures`).
    """
    import commonroad_crime.utility.solver as solver

    original = solver.get_orientation_point
    solver.get_orientation_point = _lane_heading
    try:
        yield
    finally:
        solver.get_orientation_point = original


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
        # The lanelet the frame was turned for (Evaluator._alignment), not
        # any of the ones under the ego: in a junction a turning connecting
        # lanelet overlaps the straight one.
        lanelet_id = _best_lanelet(
            self.sce.lanelet_network,
            ego.prediction.center_lanelet_assignment[ego.initial_state.time_step],
            ego.initial_state.orientation,
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


def _along_lane(x: float, y: float, heading: float) -> float:
    """The component of the vector ``(x, y)`` along a lane heading *heading*,
    signed: negative against the lane."""
    return x * math.cos(heading) + y * math.sin(heading)


def _stopping_time(v: float, a: float) -> float:
    """When a road user at velocity *v* and acceleration *a* along the lane
    stops (s): it decelerates to a stop and stays there rather than
    reversing, as :meth:`Evaluator._obstacle` predicts it; ``0`` for one
    standing with its acceleration backwards (it never reverses), ``inf``
    when it does not decelerate."""
    if v * a < 0.0:
        return -v / a
    if v == 0.0 and a < 0.0:
        return 0.0
    return math.inf


def _quadratic_roots(c0: float, c1: float, c2: float) -> list[float]:
    """The real roots of ``c0 + c1 t + c2 t^2``, in the numerically stable
    form (no cancellation between ``-c1`` and the discriminant's root)."""
    if abs(c2) < 1e-12:
        return [-c0 / c1] if c1 != 0.0 else []
    disc = c1 * c1 - 4.0 * c2 * c0
    if disc < 0.0:
        return []
    q = -0.5 * (c1 + math.copysign(math.sqrt(disc), c1))
    roots = [q / c2]
    if q != 0.0:
        roots.append(c0 / q)
    return roots


def _contact_time(gap: float, v_ego: float, a_ego: float, v_other: float, a_other: float) -> float:
    """When the ego, *gap* behind the road user ahead, reaches it (s); ``inf``
    when it never does.

    Each holds its acceleration along the lane until it stops, and then
    stays (:func:`_stopping_time`): Jansson's constant acceleration TTC
    ((5.24)-(5.26) in "Collision Avoidance Theory", CriMe's TTC) with no one
    reversing.  Unlike CriMe's, a road user braking to a stop is not
    extrapolated backwards: a car creeping to a halt ahead of a stopped ego
    never meets it.  The gap is piecewise quadratic in time, between the
    times either stops; its first zero is the contact.
    """
    t_ego, t_other = _stopping_time(v_ego, a_ego), _stopping_time(v_other, a_other)
    users = ((1.0, v_other, a_other, t_other), (-1.0, v_ego, a_ego, t_ego))

    def coefficients(lo: float) -> tuple[float, float, float]:
        # gap(t) = c0 + c1 t + c2 t^2 from *lo* to the next stop.
        c0, c1, c2 = gap, 0.0, 0.0
        for sign, v, a, stop in users:
            if lo >= stop:
                c0 += sign * (v * stop + 0.5 * a * stop * stop)
            else:
                c1 += sign * v
                c2 += sign * 0.5 * a
        return c0, c1, c2

    if gap <= 0.0:
        _, c1, c2 = coefficients(0.0)
        if c1 < 0.0 or (c1 == 0.0 and c2 < 0.0):
            return 0.0
    bounds = sorted({0.0, t_ego, t_other, math.inf})
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        roots = _quadratic_roots(*coefficients(lo))
        found = [t for t in roots if t > 0.0 and lo - 1e-9 <= t <= hi + 1e-9]
        if found:
            return min(found)
    return math.inf


def _required_deceleration(
    gap: float,
    v_ego: float,
    v_other: float,
    a_other: float,
    mode: int,
    horizon: float = math.inf,
) -> float:
    """ALongReq of the ego *gap* behind the road user ahead (m/s^2, ``0`` or
    negative).

    In CriMe's mode 1 (its default), the least constant deceleration that
    keeps the ego off the other, each holding its acceleration until it
    stops and then staying (:func:`_contact_time`).  A contact counts while
    the ego is still moving, at any time, and within *horizon* (s; the
    frame's prediction horizon) after it has stopped: braking that stops the
    ego short of a car still coming at it does not avoid that car, but a
    road user that reaches a standing ego only after the horizon -- a
    pedestrian crossing the lane a little towards the ego, a car rolling back
    -- is not one braking must avoid.

    It is CriMe's ``min(a_other - v_rel^2 / (2 gap), 0)`` -- matching the
    other's speed as the ego reaches it -- wherever that holds: a closing
    gap, the other going the ego's way and matched before it stops.
    Elsewhere it is not: a gap opening for good needs ``0`` (CriMe's formula
    reads ``-v_rel^2 / (2 gap)``), one whose lead brakes to a stop the
    deceleration that stops the ego behind where it stops, and a car coming
    the other way the deceleration that stops the ego far enough back that
    the car does not reach it within the horizon after the ego has stopped
    (at 10 m/s, a car 40 m ahead coming at 10 m/s: -15).  ``-inf`` when no
    deceleration does: the other reaches the ego within the horizon even
    stopped where it is, or the ego is not going forward and the other
    reaches it within the horizon.  Monotone in the ego's deceleration
    (braking harder brings the ego nowhere further, and stops it sooner), so
    it is found by bisection to well under the 0.01 m/s^2 it is rounded to.
    No road user ahead (*gap* ``inf``: CriMe's HW of one behind) needs none.
    Mode 2 is CriMe's piecewise constant motion, as it writes it.
    """
    if not math.isfinite(gap):
        return 0.0
    if mode != 1:
        # Never reached from the worker, which leaves CriMe's
        # acceleration_mode at 1.  ``v_other**2 / 2 * a_other`` is
        # ``(v_other**2 / 2) * a_other``, not ``v_other**2 / (2 * a_other)``:
        # CriMe's precedence, copied on purpose so that mode 2 reads what
        # CriMe's ALongReq reads.  Do not "fix" it here alone.
        return -(v_ego**2) / (2 * (gap - v_other**2 / 2 * a_other))

    def avoids(a_ego: float) -> bool:
        contact = _contact_time(gap, v_ego, a_ego, v_other, a_other)
        moving = _stopping_time(v_ego, a_ego) if v_ego > 0.0 else 0.0
        return contact == math.inf or contact > moving + horizon

    if avoids(0.0):
        return 0.0
    standing = _contact_time(gap, 0.0, 0.0, v_other, a_other)
    if v_ego <= 0.0 or (standing < math.inf and standing <= horizon):
        # Braking cannot hold the ego further back than standing where it is.
        return -math.inf
    low, high = 0.0, 1.0  # magnitudes: too little, enough
    while not avoids(-high):
        low, high = high, high * 2.0
        if high > 1e9:
            return -math.inf
    for _ in range(60):
        middle = 0.5 * (low + high)
        if avoids(-middle):
            high = middle
        else:
            low = middle
    return -high


@functools.cache
def _signed_crime_measures() -> dict[str, type]:
    """CriMe's HW, THW, TTC and ALongReq, and the measures built on them, reading the
    other road user's velocity and acceleration along the lane with their sign.

    CriMe 0.4 reads the other road user's velocity and acceleration along the
    lane as their magnitude times the cosine of the lane's heading, applying
    the sign to the ego's alone: a braking lead reads as speeding up (TTC
    ``inf``, ALongReq ``0``) and a car coming the other way in the ego's lane
    as going its way.  These subclasses compute what CriMe's ``compute`` does
    with the other's components projected onto the lane instead
    (:func:`_along_lane`; in the turned frame the lane runs along +x, so this
    is its ``state.velocity`` / ``state.acceleration``, as CriMe reads the
    ego's).  The ego's are read as CriMe reads them in TTC (signed), in
    ALongReq too.  BTN is ALongReq over the most a car can brake, and ALatReq
    and STN take TTC: each is given the signed one in place of CriMe's.

    With the sign comes a road user braking to a stop, which CriMe's constant
    acceleration would carry on into reversing: TTC and ALongReq take it to
    stop and stay, as the worker predicts every road user
    (:func:`_contact_time`), and ALongReq is the least deceleration that
    keeps the ego off it in that model (:func:`_required_deceleration`;
    ``-inf``, and BTN ``inf``, when none does).  A road user behind the ego
    (CriMe's HW ``inf``) needs no ALongReq.

    CriMe's HW and THW read ``inf`` -- a road user behind -- for one whose
    box overlaps the ego's from ahead (a negative bumper-to-bumper headway):
    a collision read as no conflict.  Here HW and THW read ``0`` for it (its
    centre ahead of the ego's along the lane, their boxes touching), and TTC
    and ALongReq, which take HW, read the gap as closed: TTC ``0``, ALongReq
    ``-inf`` and BTN ``inf`` while the ego closes on it, as for a gap closing
    to ``0``, and ALatReq and STN, which take TTC, ``inf``.  A road user
    whose centre is behind the ego's, or one beside it whose box does not
    touch the ego's (CriMe takes any lanelet their boxes share as the same
    lane), still reads HW ``inf``.
    """
    import commonroad_crime.utility.general as utils_gen
    import commonroad_crime.utility.solver as utils_sol
    from commonroad.scenario.obstacle import DynamicObstacle
    from commonroad_crime.measure import BTN, HW, STN, THW, TTC, ALatReq, ALongReq

    def heading_at(lanelet: Any, position: Any) -> float | None:
        # The lane's heading where *position* projects onto it, as CriMe reads
        # it, or None where there is none; wrap-safe under
        # _unwrapped_lane_headings, which Evaluator._crime holds.  CriMe
        # answers (None, None) for a position out of the projection's domain,
        # and raises ValueError for a lanelet it cannot build the projection of
        # (a degenerate centre line: smoothing it fails, and the raw one has
        # fewer than the 3 points a CurvilinearCoordinateSystem needs).  Both
        # are None here, and every measure below reads None the same way: the
        # ego's heading missing is NaN (nothing to measure), the other's is a
        # road user far away.
        try:
            heading = utils_sol.compute_lanelet_width_orientation(lanelet, position)[1]
        except ValueError:
            return None
        return None if heading is None else float(heading)

    def ego_along(state: Any, heading: float, name: str) -> float:
        # CriMe's TTC reading of the ego: its magnitude, signed as its
        # first component, along the lane.
        x = float(getattr(state, name))
        y = float(getattr(state, f"{name}_y", 0.0) or 0.0)
        return float(np.sign(x)) * math.hypot(x, y) * math.cos(heading)

    def other_along(state: Any, heading: float, name: str) -> float:
        x = float(getattr(state, name))
        y = float(getattr(state, f"{name}_y", 0.0) or 0.0)
        return _along_lane(x, y, heading)

    def overlapping_ahead(measure: Any, bumpers: bool) -> bool:
        # Whether the other's box touches or overlaps the ego's, its centre
        # not behind the ego's along the ego's lane (CriMe's curvilinear
        # frame) -- and, with *bumpers*, the ego's front bumper past the
        # other's rear one, as CriMe's THW reads them.  A negative headway
        # alone is no contact: CriMe takes a road user in the ego's lane by
        # any lanelet their boxes share, so one beside the ego (straddling a
        # lane line, or the ego changing lanes) overlaps it along the lane
        # without touching it.
        time_step = measure.time_step
        try:
            ego_position = measure.ego_vehicle.state_at_time(time_step).position
            other_position = measure.other_vehicle.state_at_time(time_step).position
            ego_s, _ = measure.clcs.convert_to_curvilinear_coords(*ego_position)
            other_s, _ = measure.clcs.convert_to_curvilinear_coords(*other_position)
            ego_add_on = measure._compute_vehicle_add_on(measure.ego_vehicle)
            other_add_on = measure._compute_vehicle_add_on(measure.other_vehicle)
        except ValueError:  # out of the projection's domain, or no box
            return False
        if other_s < ego_s or (bumpers and ego_s + ego_add_on <= other_s - other_add_on):
            return False
        ego_occupancy = measure.ego_vehicle.occupancy_at_time(time_step)
        other_occupancy = measure.other_vehicle.occupancy_at_time(time_step)
        if ego_occupancy is None or other_occupancy is None:
            return False
        # CriMe's HW moves each centre by half the length along x and y both,
        # which in a frame turned _TILT off the lane overshoots the bumpers by
        # that much: boxes this close read a negative headway too.
        touching = (ego_add_on + other_add_on) * 2.0 * _TILT + 1e-6
        gap = ego_occupancy.shape.shapely_object.distance(other_occupancy.shape.shapely_object)
        return bool(gap <= touching)

    class SignedHW(HW):
        # CriMe's HW, but for a road user overlapping the ego from ahead
        # (their boxes touching, overlapping_ahead): CriMe reads its negative
        # bumper-to-bumper headway as one behind (inf, no conflict), here it
        # is 0 -- touching -- so TTC reads 0, ALongReq -inf and BTN inf for a
        # collision, as they do for a gap closing to 0 (_contact_time).  0
        # rather than the negative gap: a negative gap opening would read as
        # a contact still to come.  CriMe's other infs (a projection failing,
        # a road user beside the ego or behind it) stay inf.
        def cal_headway(self, verbose: bool = True) -> Any:
            headway = super().cal_headway(verbose=verbose)
            if headway == math.inf and overlapping_ahead(self, bumpers=False):
                return 0.0
            return headway

    class SignedTHW(THW):
        # CriMe's THW, but 0 rather than inf for a road user overlapping the
        # ego from ahead, as SignedHW: the ego's front is already past its rear.
        # (CriMe's inf also means not reached within the horizon.)
        def cal_headway(self, verbose: bool = True) -> Any:
            headway = super().cal_headway(verbose=verbose)
            if headway == math.inf and overlapping_ahead(self, bumpers=True):
                return 0.0
            return headway

    class SignedTTC(TTC):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self._hw_object = SignedHW(config)

        def compute(self, vehicle_id: int, time_step: int = 0, verbose: bool = True) -> Any:
            if not self.validate_update_states_log(vehicle_id, time_step, verbose):
                return np.nan
            state = self.ego_vehicle.state_at_time(time_step)
            state_other = self.other_vehicle.state_at_time(time_step)
            network = self.sce.lanelet_network
            lanelet_ids = network.find_lanelet_by_position([state.position])[0]
            delta_d = self._hw_object.compute(vehicle_id, time_step, verbose)
            if delta_d == math.inf:
                self.value = math.inf
                return self.value
            lanelet = network.find_lanelet_by_id(
                _best_lanelet(network, lanelet_ids, state.orientation)
            )
            ego_heading = heading_at(lanelet, state.position)
            if ego_heading is None:
                self.value = math.nan
                return self.value
            other_heading = heading_at(lanelet, state_other.position)
            if other_heading is None:
                # Out of the projection's domain: the other is far away.
                self.value = math.inf
                return self.value
            v_ego = ego_along(state, ego_heading, "velocity")
            a_ego = ego_along(state, ego_heading, "acceleration")
            if isinstance(self.other_vehicle, DynamicObstacle):
                v_other = other_along(state_other, other_heading, "velocity")
                a_other = other_along(state_other, other_heading, "acceleration")
            else:
                v_other = a_other = 0.0
            ttc = _contact_time(delta_d, v_ego, a_ego, v_other, a_other)
            self.value = ttc if ttc == math.inf else float(utils_gen.int_round(ttc, 2))
            return self.value

    class SignedALongReq(ALongReq):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self._hw_object = SignedHW(config)

        def compute(self, vehicle_id: int, time_step: int = 0, verbose: bool = True) -> Any:
            if not self.validate_update_states_log(vehicle_id, time_step, verbose):
                return np.nan
            if self._except_obstacle_in_same_lanelet(expected_value=0.0, verbose=verbose):
                return self.value
            state = self.ego_vehicle.state_at_time(time_step)
            state_other = self.other_vehicle.state_at_time(time_step)
            network = self.sce.lanelet_network
            lanelet = network.find_lanelet_by_id(
                _best_lanelet(
                    network,
                    network.find_lanelet_by_position([state.position])[0],
                    state.orientation,
                )
            )
            ego_heading = heading_at(lanelet, state.position)
            if ego_heading is None:
                # Nothing to measure from, as SignedTTC: NaN, not the least
                # critical 0.
                self.value = math.nan
                return self.value
            other_heading = heading_at(lanelet, state_other.position)
            if other_heading is None:
                # Out of the projection's domain: the other is far away.  (A
                # heading of exactly 0.0 is a heading, not a missing one.)
                a_req = 0.0
            else:
                a_req = _required_deceleration(
                    self._hw_object.compute(vehicle_id, time_step, verbose=verbose),
                    ego_along(state, ego_heading, "velocity"),
                    other_along(state_other, other_heading, "velocity"),
                    other_along(state_other, other_heading, "acceleration"),
                    self.configuration.acceleration.acceleration_mode,
                    # The frame's prediction horizon: the steps the ego's
                    # trajectory runs on (EvaluatorSettings.horizon_steps).
                    (self.ego_vehicle.prediction.final_time_step - time_step) * self.dt,
                )
            if a_req > 0:
                self.value = 0.0
            elif math.isfinite(a_req):
                self.value = utils_gen.int_round(a_req, 2)
            else:
                self.value = a_req  # no braking avoids it
            return self.value

    class SignedBTN(BTN):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self._a_long_req_object = SignedALongReq(config)

        def compute(self, vehicle_id: int, time_step: int = 0, verbose: bool = True) -> Any:
            # CriMe's, but for an ALongReq of -inf (no braking avoids the
            # collision), which CriMe's rounding cannot take: BTN inf.
            if not self.validate_update_states_log(vehicle_id, time_step, verbose):
                return np.nan
            a_long_req = self._a_long_req_object.compute(vehicle_id, time_step, verbose=verbose)
            btn = a_long_req / -self.configuration.vehicle.curvilinear.a_lon_max
            self.value = utils_gen.int_round(btn, 4) if math.isfinite(btn) else btn
            return self.value

    class SignedALatReq(ALatReq):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self._ttc_object = SignedTTC(config)

        def compute(self, vehicle_id: int, time_step: int = 0, verbose: bool = True) -> Any:
            # CriMe 0.4's, but reading the lane headings on the ego's lanelet
            # as _best_lanelet picks it (the one the frame was turned for)
            # rather than the first one found under the ego, which in a
            # junction can be a turning connecting lanelet (#2211).
            if not self.validate_update_states_log(vehicle_id, time_step, verbose):
                return np.nan
            if self._except_obstacle_in_same_lanelet(expected_value=0.0, verbose=verbose):
                return self.value
            state = self.ego_vehicle.state_at_time(time_step)
            state_other = self.other_vehicle.state_at_time(time_step)
            network = self.sce.lanelet_network
            lanelet = network.find_lanelet_by_id(
                _best_lanelet(
                    network,
                    network.find_lanelet_by_position([state.position])[0],
                    state.orientation,
                )
            )
            ego_orientation = heading_at(lanelet, state.position)
            if ego_orientation is None:
                # Nothing to measure from, as SignedTTC (math.sin(None) would
                # fail the measure for the whole frame).
                self.value = math.nan
                return self.value
            other_orientation = heading_at(lanelet, state_other.position)
            if other_orientation is None:
                # Out of the projection's domain: the other is far away.
                self.value = 0.0
            else:
                ttc = self._ttc_object.compute(vehicle_id, time_step, verbose=verbose)
                if ttc == math.inf:
                    self.value = 0.0
                    return self.value
                if ttc == 0.0:
                    # In contact now (an overlapping road user, SignedHW): no
                    # lateral acceleration avoids it.
                    self.value = math.inf
                    return self.value
                a_obj_lat = math.sqrt(
                    state_other.acceleration_y**2 + state_other.acceleration**2
                ) * math.sin(other_orientation)
                # CriMe's, ego to ego, kept as it is.
                d_rel_lat = utils_sol.compute_clcs_distance(
                    self.clcs, state.position, state.position
                )[1]
                v_rel_lat = (
                    math.sqrt(state_other.velocity**2 + state_other.velocity_y**2)
                    * math.sin(other_orientation)
                    - math.sqrt(state.velocity**2 + state.velocity_y**2)
                ) * math.sin(ego_orientation)
                self.value = min(
                    abs(self._compute_a_lat(a_obj_lat, d_rel_lat, v_rel_lat, ttc, "left")),
                    abs(self._compute_a_lat(a_obj_lat, d_rel_lat, v_rel_lat, ttc, "right")),
                )
            if self.value != math.inf:
                self.value = utils_gen.int_round(self.value, 2)
            return self.value

    class SignedSTN(STN):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self._a_lat_req_object = SignedALatReq(config)

        def compute(self, vehicle_id: int, time_step: int = 0, verbose: bool = True) -> Any:
            # CriMe's, but for an ALatReq of inf (in contact now), which
            # CriMe's rounding cannot take: STN inf, as SignedBTN.
            if not self.validate_update_states_log(vehicle_id, time_step, verbose):
                return np.nan
            a_lat_req = self._a_lat_req_object.compute(vehicle_id, time_step, verbose=verbose)
            stn = abs(a_lat_req / self.configuration.vehicle.curvilinear.a_lat_max)
            self.value = utils_gen.int_round(stn, 4) if math.isfinite(stn) else stn
            return self.value

    return {
        "HW": SignedHW,
        "THW": SignedTHW,
        "TTC": SignedTTC,
        "ALongReq": SignedALongReq,
        "BTN": SignedBTN,
        "ALatReq": SignedALatReq,
        "STN": SignedSTN,
    }


#: The sections of CommonRoad-Reach's configuration ``reach_settings`` may
#: override (``Configuration``'s attributes that are sections: not the
#: scenario, the planning problem or the omegaconf it was read from).
_REACH_SECTIONS: tuple[str, ...] = ("general", "planning", "vehicle", "reachable_set", "debug")

#: The keys of CommonRoad-Reach's configuration the worker sets itself, which
#: ``reach_settings`` may not: the time step (``planning.dt``, the
#: :attr:`EvaluatorSettings.dt` the road users' trajectories step by -- a
#: Reach step of another length would run past their end, and they would drop
#: out of its collision check), the steps of each reachable set (from the
#: horizon and where the set starts), and what Reach derives from the frame's
#: scenario, planning problem and curvilinear frame for every one.
_REACH_OWNED: dict[str, frozenset[str]] = {
    "planning": frozenset(
        {
            "dt",
            "step_start",
            "steps_computation",
            "p_lon_initial",
            "p_lat_initial",
            "v_lon_initial",
            "v_lat_initial",
            "o_initial",
            "route",
            "reference_path",
            "lanelet_network",
            "list_ids_lanelets",
            "CLCS",
        }
    ),
}


def _apply_reach_settings(config: Any, reach_settings: Mapping[str, Any]) -> None:
    """Set *reach_settings*, overrides by section and key, on CommonRoad-
    Reach's *config*.

    Reach's sections are plain objects, where setting a name they do not
    have adds it, read by nothing: a misspelled or misplaced key
    (``planning.size_grid`` for ``reachable_set.size_grid``) would do
    nothing.  Every section and key must be one the configuration has, and a
    key must hold a value, not a section of its own (``vehicle.ego``).  A key
    the worker sets itself (:data:`_REACH_OWNED`: ``planning.dt``,
    ``planning.step_start``, ``planning.steps_computation``, ...) is refused
    too: the worker's value would overwrite it, or it the worker's.

    Raises:
        ValueError: A section or key it does not have, or one the worker
            sets, naming it as ``reach_settings[section][key]``.
    """
    for section, overrides in reach_settings.items():
        if section not in _REACH_SECTIONS:
            raise ValueError(
                f"reach_settings[{section!r}]: CommonRoad-Reach's configuration has "
                f"no section {section!r} (its sections: {', '.join(_REACH_SECTIONS)})"
            )
        if not isinstance(overrides, Mapping):
            raise ValueError(
                f"reach_settings[{section!r}] must be a mapping of keys to values, "
                f"got {overrides!r}"
            )
        target = getattr(config, section)
        for name, value in overrides.items():
            where = f"reach_settings[{section!r}][{name!r}]"
            if not isinstance(name, str) or name.startswith("_") or not hasattr(target, name):
                known = ", ".join(sorted(k for k in vars(target) if not k.startswith("_")))
                raise ValueError(
                    f"{where}: CommonRoad-Reach's {section!r} section has no key "
                    f"{name!r} (its keys: {known})"
                )
            if name in _REACH_OWNED.get(section, ()):
                hint = {
                    "dt": " (use the settings' dt)",
                    "steps_computation": " (use the settings' horizon_steps)",
                }.get(name, "")
                raise ValueError(
                    f"{where}: {section}.{name} is set by the worker for every "
                    f"reachable set and cannot be overridden{hint}"
                )
            if type(getattr(target, name)).__module__.startswith("commonroad_reach"):
                raise ValueError(
                    f"{where}: {section}.{name} is a section of CommonRoad-Reach's "
                    "configuration, not a value"
                )
            setattr(target, name, value)


def _set_ego_footprint(config: Any, length: float, width: float) -> None:
    """Give CommonRoad-Reach's ego the box ``length`` x ``width`` (m).

    Reach derives the discs that approximate the ego, and the radius the
    other road users are inflated by for its collision check, from the box
    once, as its configuration is built (from its default vehicle, a BMW 320i
    of 4.508 x 1.61 m); ``Configuration.update`` does not derive them again.
    They are derived here as Reach does, by its inflation mode and reference
    point.
    """
    from commonroad_reach.utility.configuration import (
        compute_disc_radius_and_distance,
        compute_inflation_radius,
    )

    vehicle = config.vehicle.ego
    vehicle.length = length
    vehicle.width = width
    vehicle.radius_disc, vehicle.circle_distance = compute_disc_radius_and_distance(
        length,
        width,
        ref_point=config.planning.reference_point,
        dist_axle_rear=vehicle.wb_rear_axle,
    )
    vehicle.radius_inflation = compute_inflation_radius(
        config.reachable_set.mode_inflation, length, width, vehicle.radius_disc
    )


def _moved(actor: Actor, x: float, y: float, angle: float) -> Actor:
    """*actor* in the frame whose origin is ``(x, y)``, turned by *angle*."""
    c, s = math.cos(angle), math.sin(angle)

    def turned(u: float, v: float) -> tuple[float, float]:
        return c * u - s * v, s * u + c * v

    px, py = turned(actor.x - x, actor.y - y)
    vx, vy = turned(actor.vx, actor.vy)
    ax, ay = turned(actor.ax, actor.ay)
    return replace(actor, x=px, y=py, yaw=actor.yaw + angle, vx=vx, vy=vy, ax=ax, ay=ay)


def _best_lanelet(network: Any, lanelet_ids: Iterable[int], yaw: float) -> int:
    """Of the lanelets under the ego, the one heading most its way.

    The frame's turn (:meth:`Evaluator._alignment`), Reach's curvilinear
    frame (:class:`_ReachFrame`) and CriMe's measures (:func:`_shared_frame`,
    :func:`_signed_crime_measures`) all read the ego's lane through this, so
    in a junction they all read the same one of the overlapping lanelets.
    """

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


def _reversed(lanelet: Any) -> Any:
    """*lanelet* running the other way: its bounds and centre line reversed,
    its left bound the right one and its right the left, its predecessors its
    successors and its successors its predecessors, its left neighbour its
    right one and its right its left.

    For a network every lanelet of which is reversed
    (:meth:`Evaluator._local_network`): each link then still joins the
    lanelets it joined, and two neighbours keep their driving directions to
    one another, so the ``adj_*_same_direction`` flags are kept as they are.
    The stop line is dropped: it is at the lanelet's end, where traffic the
    other way enters it.
    """
    from commonroad.scenario.lanelet import Lanelet

    return Lanelet(
        left_vertices=np.flip(lanelet.right_vertices, 0).copy(),
        center_vertices=np.flip(lanelet.center_vertices, 0).copy(),
        right_vertices=np.flip(lanelet.left_vertices, 0).copy(),
        lanelet_id=lanelet.lanelet_id,
        predecessor=list(lanelet.successor),
        successor=list(lanelet.predecessor),
        adjacent_left=lanelet.adj_right,
        adjacent_left_same_direction=lanelet.adj_right_same_direction,
        adjacent_right=lanelet.adj_left,
        adjacent_right_same_direction=lanelet.adj_left_same_direction,
        line_marking_left_vertices=lanelet.line_marking_right_vertices,
        line_marking_right_vertices=lanelet.line_marking_left_vertices,
        lanelet_type=set(lanelet.lanelet_type),
        user_one_way=set(lanelet.user_one_way),
        user_bidirectional=set(lanelet.user_bidirectional),
        traffic_signs=set(lanelet.traffic_signs),
        traffic_lights=set(lanelet.traffic_lights),
        adjacent_areas=set(lanelet.adjacent_areas),
    )


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
