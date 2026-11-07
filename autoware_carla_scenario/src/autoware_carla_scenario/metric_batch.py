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
  run as it went;
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
from .metrics import (
    BUILT_IN_METRICS,
    COMMONROAD_METRICS,
    read_metric,
    reset_ego_motion,
    set_importance_sampling_weight,
    track_ego_motion,
)

__all__ = [
    "ActorFrame",
    "Frame",
    "FrameRecorder",
    "MetricSeries",
    "ReplayWorld",
    "evaluate_metrics",
]

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


def evaluate_metrics(
    frames: list[Frame],
    carla_map: Any,
    keys: list[str],
) -> dict[str, MetricSeries]:
    """The metrics *keys* over the recorded *frames*, on the run's
    *carla_map*.

    Raises:
        ValueError: A key is not a built-in metric's.
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
        reset_ego_motion()
        try:
            for frame in frames:
                world.show(frame)
                set_importance_sampling_weight(frame.weight)
                track_ego_motion(world, frame.t)  # type: ignore[arg-type]
                for key in world_keys:
                    series[key].times.append(frame.t)
                    series[key].values.append(_finite(read_metric(key, world)))  # type: ignore[arg-type]
        finally:
            reset_ego_motion()
            set_importance_sampling_weight(None)

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
