"""Metrics computed once the run is over, from the frames it recorded.

A run with a :class:`~autoware_carla_scenario.conditions.MetricCondition`
records its road users every tick
(:class:`~autoware_carla_scenario.metric_frames.FrameRecorder`, re-exported
here): each vehicle's and walker's place, velocity, acceleration and box, the
ego's steering and light state, the run's importance-sampling weight.  When the run ends, while its
CARLA map is still loaded, :func:`evaluate_metrics` replays those frames:

* AlpaSim's metrics, and the run's own, by the very code that reads them off a
  live world (:func:`~autoware_carla_scenario.metrics.read_metric`), on a
  stand-in world (:class:`ReplayWorld`) that answers with the recorded actors
  and the run's map -- every tick, in order, so the weaving metrics see the
  run as it went -- a metric that raises in every frame failing the
  evaluation (:class:`MetricEvaluationError`);
* CriMe's and CommonRoad-Reach's (:data:`~autoware_carla_scenario.metrics.COMMONROAD_METRICS`)
  by the CommonRoad workers, every
  :attr:`~autoware_carla_scenario.commonroad.CommonRoadSettings.period_s` of
  simulated time -- a slow metric less often, at its own period
  (:attr:`~autoware_carla_scenario.commonroad.CommonRoadSettings.metric_periods_s`)
  -- the frames shared among several workers that run at once
  (:func:`~autoware_carla_scenario.commonroad.session.evaluate_commonroad`).

The simulation never waits on a metric, and the frames, being independent of
one another, are evaluated in parallel.
"""

from __future__ import annotations

import fnmatch
import logging
from types import SimpleNamespace
from typing import Any, Optional

from .metric_frames import (
    ActorFrame,
    Frame,
    FrameRecorder,
    MetricSeries,
    _finite,
    sampled,
)
from . import metrics as _metrics
from .metrics import (
    BUILT_IN_METRICS,
    COMMONROAD_METRICS,
    set_importance_sampling_weight,
    track_ego_motion,
)

__all__ = [
    "ActorFrame",
    "Frame",
    "FrameRecorder",
    "MetricEvaluationError",
    "MetricSeries",
    "ReplayWorld",
    "evaluate_metrics",
]


class MetricEvaluationError(RuntimeError):
    """A metric raised in every frame it was evaluated in, so read nothing:
    a fail condition on it would otherwise pass the run without having
    judged anything."""


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replaying a frame as a world
# ---------------------------------------------------------------------------


def _vector(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


class _ReplayActor:
    """A recorded road user, answering as a CARLA actor does."""

    def __init__(self, state: ActorFrame) -> None:
        self._state = state
        self.id = state.id
        self.type_id = state.type_id
        self.is_alive = True
        self.attributes = {"role_name": state.role, "base_type": state.base_type}
        self.bounding_box = SimpleNamespace(
            extent=_vector(*state.extent),
            location=_vector(*state.centre),
            rotation=SimpleNamespace(yaw=state.box_yaw, pitch=0.0, roll=0.0),
        )

    def get_transform(self) -> Any:
        s = self._state
        return SimpleNamespace(
            location=_vector(s.x, s.y, s.z),
            rotation=SimpleNamespace(roll=s.roll, pitch=s.pitch, yaw=s.yaw),
        )

    def get_location(self) -> Any:
        s = self._state
        return _vector(s.x, s.y, s.z)

    def get_velocity(self) -> Any:
        s = self._state
        return _vector(s.vx, s.vy, s.vz)

    def get_acceleration(self) -> Any:
        s = self._state
        return _vector(s.ax, s.ay, s.az)

    def get_angular_velocity(self) -> Any:
        s = self._state
        return _vector(s.wx, s.wy, s.wz)

    def get_control(self) -> Any:
        return SimpleNamespace(
            steer=0.0 if self._state.steer is None else self._state.steer
        )

    def get_light_state(self) -> int:
        return 0 if self._state.light_state is None else self._state.light_state


class _ActorList(list):  # type: ignore[type-arg]
    """A list with ``ActorList.filter()``: a wildcard on ``type_id``."""

    def filter(self, pattern: str) -> _ActorList:
        return _ActorList(a for a in self if fnmatch.fnmatchcase(a.type_id, pattern))


class ReplayWorld:
    """A recorded frame, answering as a CARLA world does: its actors, its
    frame number, and the run's map (``get_map``), which is still loaded.

    One world replays every frame of a run (:meth:`show`), so what reads it
    keeps what it holds of the world -- the map above all -- across frames.
    """

    def __init__(self, carla_map: Any) -> None:
        self._map = carla_map
        self._frame = Frame(0, 0.0, None, ())
        self._actors = _ActorList()
        self._by_id: dict[int, _ReplayActor] = {}

    @property
    def id(self) -> int:
        return id(self)

    def show(self, frame: Frame) -> None:
        """Answer with *frame* from now on."""
        self._frame = frame
        self._actors = _ActorList(_ReplayActor(a) for a in frame.actors)
        self._by_id = {a.id: a for a in self._actors}

    def get_actors(self) -> _ActorList:
        return self._actors

    def get_actor(self, actor_id: int) -> Optional[_ReplayActor]:
        return self._by_id.get(int(actor_id))

    def get_snapshot(self) -> Any:
        return SimpleNamespace(
            frame=self._frame.frame,
            timestamp=SimpleNamespace(elapsed_seconds=self._frame.t),
        )

    def get_map(self) -> Any:
        return self._map


# ---------------------------------------------------------------------------
# Evaluating
# ---------------------------------------------------------------------------


def _replay(
    frames: list[Frame],
    world: ReplayWorld,
    carla_map: Any,
    keys: list[str],
    series: dict[str, MetricSeries],
) -> None:
    """Read the world metrics *keys* in every one of *frames*, in order, on
    *world* replaying them on *carla_map*, into *series*.

    The ego's motion and the weight read are the replay's own, and what the
    module held before is put back after: the live run's state survives it.

    Raises:
        MetricEvaluationError: A key raised in every frame.
    """
    failed: dict[str, int] = {}
    first_error: dict[str, str] = {}
    held_motion = _metrics._MOTION[0]  # noqa: SLF001
    held_weight = _metrics._RUN["weight"]  # noqa: SLF001
    try:
        _metrics._MOTION[0] = _metrics._Motion()  # noqa: SLF001
        _metrics._use_map(world, carla_map)  # type: ignore[arg-type]  # noqa: SLF001
        for frame in frames:
            world.show(frame)
            set_importance_sampling_weight(frame.weight)
            track_ego_motion(world, frame.t)  # type: ignore[arg-type]
            for key in keys:
                value: Optional[float] = None
                try:
                    value = BUILT_IN_METRICS[key].read(world)  # type: ignore[arg-type]
                except Exception as error:  # noqa: BLE001 - counted, logged once
                    failed[key] = failed.get(key, 0) + 1
                    if key not in first_error:
                        first_error[key] = f"{type(error).__name__}: {error}"
                        logger.warning(
                            "Metric %s reads nothing at %.3fs: it raised",
                            key,
                            frame.t,
                            exc_info=error,
                        )
                series[key].times.append(frame.t)
                series[key].values.append(_finite(value))
    finally:
        _metrics._MOTION[0] = held_motion  # noqa: SLF001
        _metrics._RUN["weight"] = held_weight  # noqa: SLF001
        # Nor is the replay's world and map kept: a live world read next
        # reads its own (and the recorded frames are let go).
        _metrics._hold(None)  # type: ignore[arg-type]  # noqa: SLF001
    # A metric that raised in every frame read nothing at all: a fail
    # condition on it would pass the run without having judged it.
    always = [key for key in keys if frames and failed.get(key, 0) == len(frames)]
    if always:
        raise MetricEvaluationError(
            "the metrics "
            + ", ".join(f"{key} (failed: {first_error[key]})" for key in always)
            + f" failed in every frame evaluated ({len(frames)}), so read nothing"
        )


def evaluate_metrics(
    frames: list[Frame],
    carla_map: Any,
    keys: list[str],
) -> dict[str, MetricSeries]:
    """The metrics *keys* over the recorded *frames*, on the run's
    *carla_map* (the map-based ones on it, whichever map
    :class:`~autoware_carla_scenario.coordinate.map_manager.MapManager`
    holds).

    A frame a metric raises in reads nothing (``None``), as one with nothing
    to read does; the first exception of each metric is logged.  The ego's
    motion and the weight the replay reads are its own: what the module
    holds of the live run (its progress,
    :func:`~autoware_carla_scenario.metrics.ego_progress`) is left as it is.

    Raises:
        ValueError: A key is not a built-in metric's.
        MetricEvaluationError: A metric that is not a CommonRoad one raised in
            every frame evaluated.
        ~autoware_carla_scenario.commonroad.CommonRoadWorkerError: A
            CommonRoad metric is asked for and its workers cannot be started
            or do not know it, one of them fails (exits, answers out of
            protocol, or does not answer in time), or the metric fails in
            every frame evaluated
            (:func:`~autoware_carla_scenario.commonroad.session.evaluate_commonroad`).
    """
    unknown = [key for key in keys if key not in BUILT_IN_METRICS]
    if unknown:
        raise ValueError(f"not metrics: {unknown}")
    wanted = list(dict.fromkeys(keys))
    series = {key: MetricSeries(key) for key in wanted}
    world_keys = [key for key in wanted if key not in COMMONROAD_METRICS]
    commonroad_keys = [key for key in wanted if key in COMMONROAD_METRICS]
    world = ReplayWorld(carla_map)

    if world_keys:
        _replay(frames, world, carla_map, world_keys, series)

    if commonroad_keys:
        from .commonroad.session import (  # noqa: PLC0415
            commonroad_settings,
            evaluate_commonroad,
        )

        # The metrics sampled alike are evaluated together: one request a
        # frame for all of them.
        settings = commonroad_settings()
        by_period: dict[float, list[str]] = {}
        for key in commonroad_keys:
            by_period.setdefault(settings.period_for(key), []).append(key)
        for period, keys_at in sorted(by_period.items()):
            picked = sampled(frames, period)
            values = evaluate_commonroad(picked, carla_map, keys_at)
            for key in keys_at:
                series[key].times = [frame.t for frame in picked]
                series[key].values = [_finite(v) for v in values[key]]
    return series
