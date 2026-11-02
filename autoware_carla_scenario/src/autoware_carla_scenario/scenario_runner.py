"""Top-level scenario runner integrating all sub-components."""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Optional

import typesafe_carla.carla as carla

from .actions._trajectory_run import forget_walkers
from .actions.environment import forget_tire_friction
from .actions.spawn import clear_props
from .entity.ego import EgoVehicle
from .adversity_activity import clear_adversities
from .entity.registry import (
    clear_entities,
    register_entity,
    register_vehicle_entity,
)
from .conditions import (
    EntityExistenceCondition,
    OrCondition,
    ScenarioResult,
    StickyCondition,
    TimeoutCondition,
)
from .conditions.base import (
    BaseCondition,
    ConditionStatus,
    find_actor_by_role_name,
    is_post_run,
)
from .actions.base import reset_control_slots
from .constants import DEFAULT_TM_PORT, EGO_ROLE_NAME, FIXED_DELTA_SECONDS
from .debug_markers import DebugMarkers
from .failure import Failure, FailureCategory, progress_detail
from .maps.opendrive import map_asset_env_var
from .coordinate.map_manager import run_carla_map
from .coordinate.poses import CarlaWorldPose
from .coverage.collector import CoverageCollector, prepare_measures
from .measures import set_measured_scenario
from .metrics import (
    ego_progress,
    reset_ego_motion,
    set_importance_sampling_weight,
    track_ego_motion,
)
from .odd import OddCase, OddDefinition, reset_probes, resolve_odd
from .coordinate.transform import to_opendrive
from .entity.vehicle_entity import _clear_entry_speeds, _set_warmup_done
from .ego_base_link import clear_ego_base_link
from .scenario_base import BaseScenario
from .seed import set_run_seed
from .server import CarlaServerManager
from .traffic.base import TrafficBackend, TrafficContext
from .traffic.driven import BackendDriven
from .world_reset import (
    RESET_RELOAD,
    RESET_SOFT,
    WorldBaseline,
    capture_baseline,
    check_world_reset,
    episode_of,
    reload_current_map,
    same_episode,
    short_map_name,
    soft_reset_world,
)
from .traffic.config import TrafficManagerBackendConfig
from .traffic.traffic_manager import TrafficManagerBackend
from .rerun import RerunConverter, recording_path
from .run_outputs import run_outputs_exist
from .t4 import T4Recorder, T4Recording
from .t4_scene import T4SceneExporter
from .visibility import forget_visibility
from .traffic.parameters import record_run_parameters
from .trajectory_recorder import TrajectoryRecorder

if TYPE_CHECKING:
    from .npc_collisions import NpcCollisionMonitor
    from .prepare import BackendPreparer

logger = logging.getLogger(__name__)


#: Log ego OpenDRIVE position every N ticks (~1 s at 20 Hz).
_CONDITION_LOG_INTERVAL: int = 20


def _bounds_the_run(condition: BaseCondition) -> bool:
    """Whether *condition*, as a fail condition, ends the run by a timeout.

    Exactly these count:

    - a bare :class:`TimeoutCondition`;
    - an :class:`OrCondition` with a child that counts -- an OR fires as soon
      as any child does;
    - a :class:`StickyCondition` whose child counts -- it hands on the
      child's failure as it comes (it latches only a *pass*).

    Nothing else does, on purpose.  An AND or a NOT holding a timeout fires
    only on some *other* state of the world, which is a condition on the
    timeout, not a limit on the run; a :class:`PersistentCondition` resets on
    every failure its child reports, so around a timeout it never fires at
    all.

    A :class:`TimeoutCondition` whose timeout is not finite (NaN, or
    infinity) does not count either: it never fires, so it bounds nothing,
    and the default timeout is still added.  Its constructor refuses one;
    this guards a timeout changed after construction.
    """
    if isinstance(condition, TimeoutCondition):
        return math.isfinite(condition.timeout_seconds)
    if isinstance(condition, OrCondition):
        return any(_bounds_the_run(child) for child in condition._conditions)
    if isinstance(condition, StickyCondition):
        return _bounds_the_run(condition._condition)
    return False


def _register_default_timeout(
    scenario: BaseScenario, timeout_seconds: float, scenario_name: str
) -> bool:
    """Give *scenario* the runner's default timeout, unless it has its own.

    A scenario that registers a timeout fail condition has said how long it
    runs -- a red light that turns green after a drawn delay needs that delay
    and then some, however short the run's default is -- so the default would
    only cut it short.  A scenario that registers none is given the default,
    so that no run goes on forever.  What counts as a timeout of its own is
    :func:`_bounds_the_run`.

    So the runner's ``timeout_seconds``, ``run_scenario(timeout_seconds=...)``
    and ``ScenarioQueue.add(timeout_seconds=...)`` -- the run's
    ``scenario.timeout_seconds`` -- do not apply to a scenario with a timeout
    of its own (a :class:`~autoware_carla_scenario.declarative.DeclarativeScenario`
    always has one: its document's).  A sweep still has the wall-clock
    ``sweep.job_timeout_seconds`` guard over every job.

    Registered after ``ego_existence``, as the last fail condition: on a tick
    where several fire, the result names the first.  A retry of the same
    scenario object starts without the earlier attempt's registrations
    (:meth:`BaseScenario._reset_registrations`), so it is given the default
    again, once.

    Returns:
        Whether the default was registered.
    """
    if any(_bounds_the_run(c) for c in scenario._fail_conditions):
        logger.info(
            "[%s] Scenario registers its own timeout; the default of %.1fs "
            "is not added",
            scenario_name,
            timeout_seconds,
        )
        return False
    scenario.register_fail_condition(
        TimeoutCondition(timeout_seconds, label="default_timeout")
    )
    return True


def _map_name_of(world: "carla.World") -> str:
    """Return the loaded map's name, or an empty string if CARLA will not say.

    Only ever used to tell a traffic backend which map it joined, so a server
    that refuses the question is worth a blank rather than a failed run.
    """
    try:
        # The run's map, held for the batch: get_map() would build a whole new
        # one (hundreds of MB on a large map) just for its name.
        return str(run_carla_map(world).name)
    except Exception:  # pragma: no cover - defensive; CARLA-side failure
        logger.warning("Could not read the loaded map name", exc_info=True)
        return ""


def _log_ego_opendrive_position(
    world: "carla.World",
    scenario_name: str,
    elapsed: float,
    tick_count: int,
) -> None:
    """Log the ego vehicle's current OpenDRIVE road/lane position."""
    ego = find_actor_by_role_name(world, EGO_ROLE_NAME)
    if ego is None:
        logger.warning(
            "[%s] tick=%d t=%.2fs ego actor NOT FOUND (destroyed?)",
            scenario_name,
            tick_count,
            elapsed,
        )
        return

    loc = ego.get_location()
    try:
        od = to_opendrive(CarlaWorldPose(x=loc.x, y=loc.y, z=loc.z))
        logger.info(
            "[%s] tick=%d t=%.2fs ego CARLA(%.1f, %.1f, %.1f) -> "
            "OpenDRIVE road='%s' lane=%d s=%.1f",
            scenario_name,
            tick_count,
            elapsed,
            loc.x,
            loc.y,
            loc.z,
            od.road_id,
            od.lane_id,
            od.s,
        )
    except (ValueError, KeyError, IndexError, RuntimeError) as exc:
        logger.warning(
            "[%s] tick=%d t=%.2fs ego at (%.1f, %.1f, %.1f) — "
            "failed to convert to OpenDRIVE: %s",
            scenario_name,
            tick_count,
            elapsed,
            loc.x,
            loc.y,
            loc.z,
            exc,
        )


def _status_label(role: str, index: int, cond: BaseCondition) -> str:
    """A condition's label in the run's statuses: ``fail[2](late)``."""
    return f"{role}[{index}]({cond.label})"


def _failure(
    condition: str,
    category: FailureCategory,
    at: float,
    message: str,
    statuses: list[ConditionStatus],
) -> Failure:
    """Why a run failed: by *condition* (its :func:`_status_label`, ``""``
    for none), of *category*, at *at* seconds, with *message*; the other fail
    conditions that held, from *statuses*.  A run out of time says how the ego ended it
    (:func:`~autoware_carla_scenario.failure.progress_detail`)."""
    detail: Optional[str] = None
    if category == FailureCategory.PROGRESS:
        progress = ego_progress()
        if progress is not None:
            detail = progress_detail(progress[0], progress[1])
    also = [
        status.label
        for status in statuses
        if status.role == "fail" and status.satisfied and status.label != condition
    ]
    return Failure(
        condition=condition,
        category=category,
        at=at,
        message=message,
        detail=detail,
        also_triggered=also,
    )


def _build_condition_status(
    cond: "BaseCondition",
    index: int,
    role: str,
    world: "carla.World",
    elapsed: float,
) -> ConditionStatus:
    """Build a :class:`ConditionStatus` for a single condition.

    Args:
        cond: The condition instance.
        index: Positional index within its role group (pass or fail).
        role: ``"pass"`` or ``"fail"``.
        world: CARLA world.
        elapsed: Elapsed time in seconds.
    """
    cond_type = type(cond).__name__
    if is_post_run(cond):
        # Judged once the run is over (_judge_metrics), not checked here:
        # what the judgement found, or that there has been none yet.
        details = cond.get_details()
        if cond._last_satisfied:
            message = cond._last_message
        elif "ng_seconds" in details:
            message = "not triggered"
        else:
            message = "judged after the run"
        return ConditionStatus(
            label=_status_label(role, index, cond),
            satisfied=cond._last_satisfied,
            message=message,
            condition_type=cond_type,
            role=role,
            details=details,
        )
    check = cond.check(world, elapsed)
    details = cond.get_details()

    if check is not None:
        satisfied = True
        message = check.message
    else:
        satisfied = False
        message = "not yet satisfied" if role == "pass" else "not triggered"

    return ConditionStatus(
        label=_status_label(role, index, cond),
        satisfied=satisfied,
        message=message,
        condition_type=cond_type,
        role=role,
        details=details,
    )


def _collect_condition_statuses(
    scenario: "BaseScenario",
    world: "carla.World",
    elapsed: float,
    scenario_name: str,
) -> list[ConditionStatus]:
    """Snapshot current status of all pass and fail conditions.

    Each condition is checked once and the result is recorded as a
    :class:`ConditionStatus`.  The statuses are also logged for
    real-time visibility.
    """
    statuses: list[ConditionStatus] = []

    for i, cond in enumerate(scenario._pass_conditions):
        status = _build_condition_status(cond, i, "pass", world, elapsed)
        if status.satisfied:
            logger.info(
                "[%s]   %s: OK — %s", scenario_name, status.label, status.message
            )
        else:
            logger.info("[%s]   %s: PENDING", scenario_name, status.label)
        statuses.append(status)

    for i, cond in enumerate(scenario._fail_conditions):
        status = _build_condition_status(cond, i, "fail", world, elapsed)
        if status.satisfied:
            logger.info(
                "[%s]   %s: TRIGGERED — %s",
                scenario_name,
                status.label,
                status.message,
            )
        statuses.append(status)

    return statuses


def _condition_statuses(
    scenario: "BaseScenario",
    world: "carla.World",
    elapsed: float,
    scenario_name: str,
) -> list[ConditionStatus]:
    """The condition statuses a result records (:func:`_collect_condition_statuses`).

    A status holds its condition's details (``get_details()``, a
    ``dict[str, Any]`` of mixed value types), which Codon cannot express, so
    the check (docs/typecheck.md) does not compile how they are collected.
    """
    if TYPE_CHECKING:
        statuses: list[ConditionStatus] = []
        return statuses
    else:
        return _collect_condition_statuses(scenario, world, elapsed, scenario_name)


def _npc_collision_monitor() -> "NpcCollisionMonitor":
    """The run's :class:`~autoware_carla_scenario.npc_collisions.NpcCollisionMonitor`:
    the road users' own collisions and falls, logged, not judged."""
    from .npc_collisions import NpcCollisionMonitor  # noqa: PLC0415

    return NpcCollisionMonitor()


def _metric_recorder(scenario: BaseScenario) -> Optional[Any]:
    """What records the run's frames for its MetricConditions, judged once it
    is over (:mod:`~autoware_carla_scenario.metric_batch`); ``None`` when it
    has none.  Recorded frames are duck-typed actors, which the check
    (docs/typecheck.md) does not compile."""
    if not any(is_post_run(c) for c in scenario._fail_conditions):
        return None
    from .metric_batch import FrameRecorder  # noqa: PLC0415

    return FrameRecorder()


def _record_metrics(
    trajectory: Any,
    frames: list[Any],
    series: dict[str, Any],
    judged: list[tuple[int, BaseCondition]],
) -> None:
    """Write the metrics of *series* into the *trajectory* log, a ``metric``
    record per frame evaluated, and each condition of *judged* as
    ``condition`` records: whether its metric was NG, on its first frame and
    whenever that changed."""
    frame_at = {frame.t: frame.frame for frame in frames}
    rows: dict[float, dict[str, Optional[float]]] = {}
    for key, metric in series.items():
        for t, value in zip(metric.times, metric.values):
            rows.setdefault(t, {})[key] = value
    for t in sorted(rows):
        trajectory.record_metric(frame_at.get(t, 0), t, rows[t])
    for i, condition in judged:
        metric = series[condition.metric]  # type: ignore[attr-defined]
        last: Optional[bool] = None
        for t, value in zip(metric.times, metric.values):
            ng = condition.is_ng(value)  # type: ignore[attr-defined]
            if ng != last:
                trajectory.record_condition_at(
                    frame_at.get(t, 0), t, "fail", i, condition.label, ng
                )
                last = ng


def _judge_metrics(
    scenario: BaseScenario,
    recorder: Optional[Any],
    world: "carla.World",
    result: ScenarioResult,
    scenario_name: str,
    trajectory: Optional[Any] = None,
) -> ScenarioResult:
    """Judge the run's MetricConditions on the metrics of the frames it
    recorded, computed now, on the run's map (still loaded); the run's result,
    failed by the first that fails when it fails earlier than the run did.

    The metrics, and each condition's NG frames, go into the run's
    *trajectory* log (its :class:`TrajectoryRecorder`), for its Rerun
    recording (:mod:`~autoware_carla_scenario.rerun_metrics`).

    Raises:
        ~autoware_carla_scenario.commonroad.CommonRoadWorkerError: A CommonRoad
            metric is judged and its workers cannot be started.
    """
    if recorder is None:
        return result
    from .metric_batch import evaluate_metrics  # noqa: PLC0415

    judged = [(i, c) for i, c in enumerate(scenario._fail_conditions) if is_post_run(c)]
    keys = [c.metric for _, c in judged]  # type: ignore[attr-defined]
    logger.info(
        "[%s] Computing %d metric(s) over %d recorded frames: %s",
        scenario_name,
        len(set(keys)),
        len(recorder.frames),
        ", ".join(sorted(set(keys))),
    )
    series = evaluate_metrics(recorder.frames, run_carla_map(world), keys)
    if trajectory is not None:
        _record_metrics(trajectory, recorder.frames, series, judged)
    first: Optional[tuple[int, BaseCondition, ScenarioResult]] = None
    for i, condition in judged:
        metric = series[condition.metric]  # type: ignore[attr-defined]
        verdict = condition.judge(metric.times, metric.values)  # type: ignore[attr-defined]
        if verdict is None:
            logger.info(
                "[%s] Metric condition [%d](%s) held: %s",
                scenario_name,
                i,
                condition.label,
                condition.get_details(),
            )
            continue
        logger.info(
            "[%s] Metric condition [%d](%s) TRIGGERED: %s",
            scenario_name,
            i,
            condition.label,
            verdict.message,
        )
        if first is None or verdict.elapsed_seconds < first[2].elapsed_seconds:
            first = (i, condition, verdict)
    statuses = _condition_statuses(
        scenario, world, result.elapsed_seconds, scenario_name
    )
    if first is None or (
        not result.passed and result.elapsed_seconds <= first[2].elapsed_seconds
    ):
        # The run's own outcome stands; its statuses now hold the judgements.
        judged_labels = {_status_label("fail", i, c) for i, c in judged}
        result.condition_statuses = [
            s for s in result.condition_statuses if s.label not in judged_labels
        ] + [s for s in statuses if s.label in judged_labels]
        if result.failure is not None:
            also = result.failure.also_triggered
            also.extend(
                s.label
                for s in statuses
                if s.label in judged_labels and s.satisfied and s.label not in also
            )
        return result
    i, condition, verdict = first
    return ScenarioResult(
        passed=False,
        message=verdict.message,
        elapsed_seconds=verdict.elapsed_seconds,
        condition_statuses=statuses,
        failure=_failure(
            _status_label("fail", i, condition),
            scenario.fail_category(i),
            verdict.elapsed_seconds,
            verdict.message,
            statuses,
        ),
    )


def _traffic_record(backend: TrafficBackend, scenario_name: str) -> Optional[Any]:
    """What drove the run's traffic, as the result records it: the backend's
    :meth:`~TrafficBackend.describe`, with how many vehicles it simulated at
    each level when the run ended (:meth:`~TrafficBackend.fidelity_census`).

    Read before the backend closes, which forgets both.  ``None`` when the
    backend cannot say: a record must not end a run.  Dicts of mixed value
    types, which Codon cannot express: the check (docs/typecheck.md) does not
    compile it.
    """
    try:
        record = dict(backend.describe())
        census = backend.fidelity_census()
    except Exception:  # noqa: BLE001 - a third party's backend
        logger.warning(
            "[%s] Could not read what drove the traffic", scenario_name, exc_info=True
        )
        return None
    if census:
        record["fidelity"] = dict(census)
        logger.info("[%s] Traffic at the end: %s", scenario_name, census)
    return record


def _write_result(
    json_path: Path,
    output_dir: Path,
    file_stem: str,
    scenario: "BaseScenario",
    result: ScenarioResult,
    traffic: Optional[Any] = None,
) -> None:
    """Write *result* to *json_path*, with the environment the run had.

    What it ran with: the lock and the environment beside the result, and
    their summary (with the environment's id) in it, and what drove the
    traffic (*traffic*, :func:`_traffic_record`).  It merges values of mixed
    types (the traffic record, ``Any``, and :mod:`.environment`'s record, a
    module the check does not compile: platform, importlib.metadata) into the
    result document, which Codon cannot express: the check (docs/typecheck.md)
    does not compile it.
    """
    import json  # noqa: PLC0415

    from .environment import write as write_environment  # noqa: PLC0415

    environment = write_environment(output_dir, file_stem, scenario)
    data = json.loads(result.to_json())
    if environment is not None:
        data["environment"] = environment
    if traffic is not None:
        data["traffic"] = traffic
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _record_run(
    coverage: CoverageCollector,
    case: Optional[OddCase],
    result: ScenarioResult,
    weight: Optional[float] = None,
) -> None:
    """Record in *coverage* the case the run ran, if it was drawn from the ODD,
    and its outcome: what the merged report assesses the flakiness of and a
    later sweep's flaky search learns from, with openodd's collector (a Rust
    extension), which the check (docs/typecheck.md) compiles against the
    coverage model.

    A failed run also records when it failed -- ``result.elapsed_seconds``,
    on the simulated clock the collector was started and ticked on -- so the
    report can say which ODD cell the ego was in. Every run records its
    environment's id (what ``{stem}_requirements.lock`` pins, the id its
    header and the result carry): the report never pools runs of different
    environments, where a regression would read as flakiness.

    An importance-sampled run also records its likelihood ratio, *weight*
    (:attr:`BaseScenario.outcome_weight`), from which the merged report
    estimates the naturalistic failure rate (openodd's weighted outcomes).
    """
    from .environment import capture  # noqa: PLC0415

    if case is not None:
        coverage.record_case(*case)
    try:
        environment: Optional[str] = capture().id
    except Exception:  # noqa: BLE001 - a run's outcome matters more than this
        logger.warning("could not read the run's environment", exc_info=True)
        environment = None
    coverage.record_outcome(
        result.passed,
        failed_at=None if result.passed else result.elapsed_seconds,
        environment=environment,
        weight=weight,
    )


def _run_file_stem(output_dir: Path, scenario_name: str) -> str:
    """The stem a run's files are named by: *scenario_name*, or
    ``{scenario_name}-N`` once an earlier run in *output_dir* has taken it.

    A batch can run one scenario class more than once -- two configs of it, or
    the cases of a sweep -- and each run keeps a result, trajectory, recording
    and coverage file of its own, all under one stem.  A stem is taken while
    any output of a run (``run_outputs``) exists under it.
    """
    stem = scenario_name
    counter = 1
    while run_outputs_exist(output_dir, stem):
        stem = f"{scenario_name}-{counter}"
        counter += 1
    return stem


def _destroy_all_dynamic_actors(
    world: "carla.World",
    scenario_name: str,
    keep_role_names: "frozenset[str]" = frozenset[str](),
) -> None:
    """Destroy the vehicles, walkers and sensors in the world for a clean state.

    *keep_role_names* spares actors this scenario does not own -- an ego spawned
    by an ``autoware_carla_interface`` node, say, which the scenario only
    attaches to. Their sensors are spared with them: destroying those would
    leave the owner driving blind.

    Walkers are swept with the vehicles rather than left alone.  A queue runs
    several scenarios against one world, and a pedestrian left standing from
    the last one is a pedestrian the next one did not put there.  The
    ``keep`` set is matched on vehicles only, because everything spared is an
    externally owned ego.
    """
    actors = world.get_actors()
    kept_ids = {
        actor.id
        for actor in actors.filter("vehicle.*")
        if actor.attributes.get("role_name") in keep_role_names
    }
    destroyed = 0
    for actor in [
        *actors.filter("vehicle.*"),
        *actors.filter("walker.*"),
        *actors.filter("controller.*"),
        *actors.filter("sensor.*"),
    ]:
        if actor.id in kept_ids:
            continue
        try:
            parent = actor.parent
        except AttributeError:
            parent = None
        if parent is not None and parent.id in kept_ids:
            continue
        try:
            actor.destroy()
            destroyed += 1
        except RuntimeError:
            pass
    if destroyed:
        world.tick()
        logger.info("[%s] Destroyed %d leftover actor(s)", scenario_name, destroyed)


def _destroy_leftovers_before_reload(world: "carla.World", scenario_name: str) -> None:
    """Destroy what a run left in *world* through the server, before a map
    reload would remove it.

    The reload removes these actors too, but it does not release a sensor
    that is still listening. Its stream stays open in the server and in the
    client and keeps waking both of them. A ``CollisionCondition`` used to
    leave such a sensor in every run, because nothing destroyed it. Every case
    then added thousands of context switches per tick to each process, and
    after a few cases the tick loop was a fifth slower (#290). Destroying the
    actors through the server first releases the server's end of the stream.
    The video's replay used to do that before every reload, as a side effect,
    which hid the leak.

    Only the server's end: the handles taken here from ``world.get_actors()``
    never listened, and LibCarla unsubscribes a sensor's stream on the client
    only through the handle that listened. That is why the run's own cleanup
    closes its conditions first (``BaseScenario.close_conditions``), which
    stops and destroys their sensors through their own handles; what is
    destroyed here is what nothing of the run owned (#371).

    Everything goes, an attached ego included, because the reload would
    remove it anyway. A failure is logged, and the reload still runs.
    """
    try:
        _destroy_all_dynamic_actors(world, scenario_name)
    except Exception:  # noqa: BLE001 - the reload must still happen
        logger.warning(
            "[%s] Destroying the run's actors before the reload failed",
            scenario_name,
            exc_info=True,
        )


def _ego_markers(ego: "EgoVehicle") -> Optional[DebugMarkers]:
    """What *ego* shows on the tick the world has just ticked to (its plan),
    for the trajectory log; None when the entity fails to say."""
    try:
        return ego.debug_markers()
    except Exception:  # noqa: BLE001 - must never fail the run
        return None


class _ScenarioClock:
    """The run's clock: simulated seconds since it began.

    A scenario is measured on the world's clock and on nothing else.  The world
    advances by ``fixed_delta_seconds`` per tick and the loop steps it as fast
    as the slowest client allows, so how much simulated time fits into a second
    of real time is a property of the machine.  A duration read off the wall
    clock would make the same scenario fire its triggers in different places on
    a fast host and a slow one, which is the determinism that synchronous mode
    exists to provide.

    That holds for the runner's own timeout as much as for a condition an
    author wrote: a scenario that is given sixty seconds is given sixty of its
    own.  The wall-clock protection a run needs against a *stuck* simulator
    lives where it belongs -- the CARLA client's RPC timeout, which raises when
    ``world.tick()`` stops returning, and the sweeper's ``job_timeout_seconds``,
    which bounds a whole job.

    It starts when the clock is built, which is after the ego is ready: an
    entity that needs an autonomy stack to come up would otherwise spend most
    of the scenario's timeout booting.  Simulation time does advance during
    that wait, which is why the start is measured rather than assumed to be
    zero.

    Args:
        world: The CARLA world whose simulated time is read.
    """

    _world: carla.World
    _simulated_start: float

    def __init__(self, world: "carla.World") -> None:
        self._world = world
        self._simulated_start = self._simulated_now()

    def _simulated_now(self) -> float:
        return float(self._world.get_snapshot().timestamp.elapsed_seconds)

    @property
    def simulated(self) -> float:
        """Simulated seconds since the run began."""
        return self._simulated_now() - self._simulated_start


class ScenarioRunner:
    """Orchestrates scenario execution: map loading, tick loop, and recording.

    A single instance can run multiple scenarios sequentially. Each call to
    :meth:`run_scenario` spawns/destroys the ego vehicle and saves a recording.

    Example::

        with CarlaServerManager() as server:
            runner = ScenarioRunner(server)
            runner.load_map_by_name("Town10HD_Opt")
            result = runner.run_scenario(MyScenario(ego_config))
            assert result.passed
    """

    odd: OddDefinition
    timeout_seconds: float
    output_dir: Path
    _tm_port: int
    _min_tick_interval: float
    _next_tick_at: float
    _traffic_backend: TrafficBackend
    _xodr_path: Optional[Path]
    _client: carla.Client
    _world: Optional[carla.World]
    #: The map the runs are for: the one load_map_by_name() last loaded (its
    #: short name), or "" when this runner loaded none.
    _batch_map: str
    #: The OpenDRIVE load_map_by_overwriting_xodr() installed for it, if any.
    _batch_xodr: Optional[Path]
    #: The episode this runner last loaded _batch_map into, if it knows it.
    _batch_episode: Optional[int]
    #: How the world is reset between runs: RESET_RELOAD or RESET_SOFT.
    _world_reset: str
    #: With RESET_SOFT, load the map instead on every this many-th reset (0:
    #: only when a soft reset cannot be done).
    _reload_every: int
    #: Soft resets since the map was last loaded.
    _soft_resets: int
    #: What a soft reset puts the world back to, read when the map was last
    #: loaded; None when there is none to soft-reset to (RESET_RELOAD, or the
    #: map was not loaded by this runner yet).
    _baseline: Optional[WorldBaseline]
    #: The map the runs are on, as the trajectory log's ``map`` record names
    #: it: where it came from (``map.source``), the Lanelet2 file read, and the
    #: projection the config overrode its own with.  The queue sets them.
    map_source: Optional[str]
    lanelet2_path: Optional[Path]
    projector_type: Optional[str]
    #: What converts each run's trajectory log into a Rerun recording once
    #: the log is closed (``rerun.enabled``), or None for no recording.  The
    #: queue sets it.
    rerun: Optional[RerunConverter]
    #: What writes each run as a T4 dataset, with sensors of its own on the
    #: ego (``t4.enabled``), or None for no dataset.  The queue sets it.
    t4: Optional[T4Recorder]
    #: What writes each run's trajectory log as a T4Scene once the log is
    #: closed (``t4scene.enabled``), or None for none.  The queue sets it.
    t4scene: Optional[T4SceneExporter]
    #: What starts the backends a run's actions need (``@need_prepare``) that
    #: the queue did not start before it, or None.  The queue sets it.
    preparer: Optional["BackendPreparer"]

    def __init__(
        self,
        server: CarlaServerManager,
        host: str = "localhost",
        port: int = 2000,
        tm_port: int = DEFAULT_TM_PORT,
        timeout_seconds: float = 60.0,
        output_dir: Path = Path("scenario_outputs"),
        max_tick_rate_hz: Optional[float] = None,
        traffic_backend: Optional[TrafficBackend] = None,
        odd: Optional[OddDefinition] = None,
        world_reset: str = RESET_RELOAD,
        reload_every: int = 0,
    ) -> None:
        """Initialize the scenario runner.

        Args:
            server: An already-started (or context-managed) server manager.
            host: CARLA server hostname.
            port: CARLA server RPC port.
            tm_port: CARLA TrafficManager port.
            timeout_seconds: Default ``TimeoutCondition`` registered on every
                scenario that registers no timeout fail condition of its own,
                in **simulated** seconds -- the same clock every other
                duration is on, so a scenario given sixty seconds is given
                sixty of its own however fast the host runs.  A scenario with
                its own timeout keeps it, longer or shorter than this one, and
                this one does not apply to it (see
                ``_register_default_timeout`` for what counts).
            output_dir: Directory where CARLA recording logs are saved.
            max_tick_rate_hz: Upper bound on how fast the tick loop steps the
                world, or *None* to step as fast as the server allows.  Slow
                that down to the rate of the slowest client reading the
                simulation: a client that cannot service every tick sees the
                world jump, not step.
            traffic_backend: What drives the vehicles the scenario did not
                author.  *None* selects CARLA's TrafficManager on *tm_port*,
                which is what every scenario written before the backend seam
                existed expects.
            odd: The ODD every run is measured against: its attributes are
                the ODD coverage, its modules say which ticks were outside it
                (docs/odd.md).  *None* selects the built-in ``default`` ODD.
            world_reset: How :meth:`reset_world` resets the world between
                runs.  ``"reload"`` (the default) loads the map again.
                ``"soft"`` keeps the loaded map and destroys what the run
                spawned, restoring the weather, the episode settings, the
                traffic lights and the spectator as they were when the map
                was loaded (:func:`~autoware_carla_scenario.world_reset.soft_reset_world`);
                the map is loaded instead when the world is not that one any
                more, or the soft reset fails or finds anything left over
                (docs/usage.md, "Resetting the world between cases").
            reload_every: With ``world_reset="soft"``, load the map instead of
                a soft reset on every *reload_every*-th reset; 0 (the default)
                only when a soft reset cannot be done.

        Raises:
            ValueError: *world_reset* is not ``"reload"`` or ``"soft"``, or
                *reload_every* is negative.
        """
        check_world_reset(world_reset, reload_every)
        self._world_reset = world_reset
        self._reload_every = reload_every
        self._soft_resets = 0
        self._baseline = None
        self.map_source = None
        self.lanelet2_path = None
        self.projector_type = None
        self.rerun = None
        self.t4 = None
        self.t4scene = None
        self.preparer = None
        self.odd = resolve_odd(odd)
        self.timeout_seconds = timeout_seconds
        self.output_dir = output_dir
        self._tm_port = tm_port
        self._min_tick_interval = (
            1.0 / max_tick_rate_hz
            if max_tick_rate_hz is not None and max_tick_rate_hz > 0.0
            else 0.0
        )
        self._next_tick_at = 0.0
        # `is not None` rather than truthiness: a backend is a third party's
        # object and may define __bool__ or __len__ (no vehicles yet, say), and
        # one that was explicitly passed must drive the run whatever it reports.
        if traffic_backend is not None:
            self._traffic_backend = traffic_backend
        else:
            self._traffic_backend = TrafficManagerBackend(
                TrafficManagerBackendConfig(port=tm_port)
            )
        self._xodr_path = None

        self._client = carla.Client(host, port)
        # Loading a town on CARLA 0.10 (UE5) takes ~25 s, so a 10 s client
        # timeout fails the load with a bare 'std::exception'.
        self._client.set_timeout(60.0)
        self._world = None
        self._batch_map = ""
        self._batch_xodr = None
        self._batch_episode = None

    @property
    def traffic_backend(self) -> TrafficBackend:
        """Return what drives this runner's traffic."""
        return self._traffic_backend

    @property
    def _backend_tm_port(self) -> int:
        """Return the TrafficManager port in play, as the backend sees it.

        The port belongs to the backend, and the ``tm_port`` this runner was
        constructed with only builds the default one.  Reading it back keeps
        :meth:`BaseScenario.set_client` -- which hands a scenario the client it
        spawns actors with, and the port its own compatibility surface exposes
        as ``scenario.tm_port`` -- naming the same TrafficManager the run's
        traffic is actually on, rather than a second value nothing reconciles.

        Entities are not given a port at all: the runner injects the backend
        into them, and the backend knows its own port.
        """
        if hasattr(self._traffic_backend, "port"):
            return int(self._traffic_backend.port)
        return self._tm_port

    @property
    def xodr_path(self) -> Optional[Path]:
        """Return the OpenDRIVE of the world in play, when one is known.

        A backend that derives its own road network from the map -- a traffic
        simulator with its own network format -- reads it from
        :class:`~autoware_carla_scenario.traffic.base.TrafficContext`.
        """
        return self._xodr_path

    @xodr_path.setter
    def xodr_path(self, path: Optional[Path]) -> None:
        """Tell the runner which OpenDRIVE the loaded world is running.

        Set by :meth:`load_map_by_overwriting_xodr` for a map this runner
        installs, and by :class:`~autoware_carla_scenario.ScenarioQueue` for one
        whose roads CARLA already ships -- the queue is where that file is
        resolved (or captured from the server) for the MapManager anyway.
        """
        self._xodr_path = path

    # ------------------------------------------------------------------
    # Tick pacing
    # ------------------------------------------------------------------

    def _pace_tick(self) -> None:
        """Sleep until the next tick is due under ``max_tick_rate_hz``.

        No-op when the rate is uncapped.  The schedule is kept in absolute
        time so that a tick which overruns its slot does not push every later
        tick back with it.
        """
        if self._min_tick_interval <= 0.0:
            return
        now = time.monotonic()
        if now < self._next_tick_at:
            time.sleep(self._next_tick_at - now)
            now = self._next_tick_at
        self._next_tick_at = now + self._min_tick_interval

    # ------------------------------------------------------------------
    # Ego readiness
    # ------------------------------------------------------------------

    @staticmethod
    def _hold_vehicles_still(world: "carla.World", spare: Collection[int] = ()) -> None:
        """Keep every vehicle stopped while the run is still being set up.

        See :func:`~autoware_carla_scenario.utils.vehicles.hold_vehicles_still`;
        the ego attach wait applies the same hold, so it lives there.
        """
        from .utils.vehicles import hold_vehicles_still  # noqa: PLC0415

        hold_vehicles_still(world, spare)

    @staticmethod
    def _release_vehicles(world: "carla.World") -> None:
        """Let go of the brakes the init phase held on."""
        from .utils.vehicles import release_vehicle  # noqa: PLC0415

        for actor in world.get_actors().filter("vehicle.*"):
            release_vehicle(actor)

    def _run_init(self, world: "carla.World", scenario: "BaseScenario") -> None:
        """Run the scenario's init actions, then tick once so the world shows them.

        A client reads an actor's state -- a light's ``is_frozen()`` and
        ``get_state()``, an actor's transform -- from the snapshot of the last
        tick, so what an init action changed reads as it was until the world
        ticks.  An ego started on the same tick would take the world as it was:
        the policy run-up (``warmup_s``) read every light the scenario had just
        frozen as running, froze it as its own, and thawed it on arrival, so
        CARLA's cycle took over lights the scenario holds red.  The tick holds
        every vehicle still, like the ones before it.
        """
        scenario.run_init(world)
        self._hold_vehicles_still(world)
        self._pace_tick()
        world.tick()

    def _warn_if_route_leaves_odd(
        self, scenario_name: str, scenario: "BaseScenario"
    ) -> None:
        """Warn when the ego's planned route leaves the ODD (docs/odd.md).

        The route is planned on the loaded Lanelet2 map from the spawn pose,
        through the waypoints, to the goal that ``setup()`` has settled.  A
        warning, never a failure: the run is what measures the ODD, and
        anything that keeps the route from being planned (no goal, no map, no
        path) is only logged at debug level.  An ODD with no modules rules
        nothing out, so it is not even planned.
        """
        try:
            if not self.odd.modules or scenario.goal_pose is None:
                return
            from .odd.route import plan_route_coverage  # noqa: PLC0415

            coverage = plan_route_coverage(self.odd, scenario)
        except Exception as error:  # noqa: BLE001 - must never fail the run
            logger.debug(
                "[%s] Planned route not checked against the ODD: %s",
                scenario_name,
                error,
            )
            return
        outside = coverage.outside()
        if outside:
            logger.warning(
                "[%s] The planned route leaves ODD %s: %.1f m of %.1f m on "
                "lanelets %s (modules: %s)",
                scenario_name,
                coverage.odd,
                coverage.outside_m,
                coverage.length_m,
                ", ".join(str(ll.lanelet_id) for ll in outside),
                ", ".join(sorted({m for ll in outside for m in ll.failing_modules})),
            )

    def _wait_for_ego(
        self,
        world: "carla.World",
        ego: "EgoVehicle",
        scenario_name: str,
    ) -> None:
        """Tick the world until the ego is ready to be judged.

        An entity backed by a whole autonomy stack is not ready when its actor
        appears: localization, routing and engagement come minutes later, and
        that stack only makes progress while simulation time advances -- so this
        waits by ticking, not by sleeping. Running the scenario's conditions
        during that time would judge a vehicle that has not been asked to move
        yet, and a condition already true at the initial pose (standing still,
        for one) would record a result before the run began.

        Returns as soon as the entity reports readiness, or when it gives up
        waiting: the tick loop reads ``termination_requested`` and ends the run.
        Entities that are ready with their actor (the default) return here
        immediately.
        """
        if ego.is_initialized:
            return
        logger.info("[%s] Waiting for the ego to be ready ...", scenario_name)
        waited_ticks = 0
        while not ego.is_initialized and not ego.termination_requested:
            self._hold_vehicles_still(world, ego.carried_actor_ids)
            self._pace_tick()
            world.tick()
            ego.on_tick(world, 0.0)
            waited_ticks += 1
        logger.info(
            "[%s] Ego %s after %d tick(s)",
            scenario_name,
            "is ready" if ego.is_initialized else "gave up waiting",
            waited_ticks,
        )

    # ------------------------------------------------------------------
    # Map loading
    # ------------------------------------------------------------------

    def load_map_by_overwriting_xodr(self, xodr_path: Path, map_name: str) -> None:
        """Load a built-in CARLA map after overwriting its internal ``.xodr`` file.

        The destination path is read from an environment variable derived from
        *map_name* by converting CamelCase to ``UPPER_SNAKE_CASE_PATH``
        (e.g. ``NishishinjukuMap`` → ``NISHISHINJUKU_MAP_PATH``).

        This allows using the full CARLA map assets (meshes, textures, etc.)
        while replacing only the road network definition.

        Args:
            xodr_path: Path to the ``.xodr`` file that will overwrite the
                internal map file.
            map_name: Built-in CARLA map name (e.g. ``"NishishinjukuMap"``).
                Must exist in the server's available maps.

        Raises:
            RuntimeError: If the derived environment variable is not set.
            FileNotFoundError: If *xodr_path* does not exist.
        """
        env_var = map_asset_env_var(map_name)
        dest_str = os.environ.get(env_var)
        if not dest_str:
            raise RuntimeError(
                f"Environment variable '{env_var}' is not set. "
                f"Set it to the path of the internal .xodr file for map '{map_name}' "
                f"inside the CARLA installation "
                f"(e.g. /opt/carla/CarlaUE5/Content/Carla/Maps/{map_name}.xodr)."
            )
        dest = Path(dest_str)
        shutil.copy2(xodr_path, dest)
        self._xodr_path = xodr_path
        self.load_map_by_name(map_name)
        # After load_map_by_name(), which forgets any OpenDRIVE: the map is to
        # be loaded again the way it was here, installed file and all.
        self._batch_xodr = xodr_path

    def load_map_by_name(self, map_name: str) -> None:
        """Load a built-in CARLA map by name (e.g. ``"Town10HD_Opt"``).

        Available maps are checked before loading.  Both short names
        (``"Town10HD_Opt"``) and full asset paths (``"/Game/Carla/Maps/Town10HD_Opt"``)
        are accepted.

        Args:
            map_name: Name of the CARLA map asset.

        Raises:
            ValueError: If *map_name* is not found in the server's available
                maps, with the full list of valid names included in the message.
        """
        available: list[str] = self._client.get_available_maps()

        # Accept either a full asset path or the short name (last path segment).
        def _matches(candidate: str) -> bool:
            return candidate == map_name or candidate.split("/")[-1] == map_name

        if not any(_matches(m) for m in available):
            short_names = sorted(m.split("/")[-1] for m in available)
            raise ValueError(
                f"Map {map_name!r} is not available on the CARLA server. "
                f"Available maps: {short_names}"
            )

        loaded = self._client.load_world(map_name)
        self._world = loaded
        # The map the runs are for, to reset the world to -- not whatever map
        # the server happens to run then (see _world_for_next_run).
        self._batch_map = short_map_name(map_name)
        self._batch_xodr = None
        self._batch_episode = episode_of(loaded)
        self._baseline = self._capture_baseline(loaded, map_name)

    def _load_batch_map(self) -> None:
        """Load the runs' map again, the way it was loaded first."""
        if self._batch_xodr is not None:
            # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
            xodr: Path = self._batch_xodr
            self.load_map_by_overwriting_xodr(xodr, self._batch_map)
        else:
            self.load_map_by_name(self._batch_map)

    def _world_for_next_run(self, scenario_name: str) -> "carla.World":
        """Return the world the next run sets up on, loading the runs' map if
        the server has gone over to another.

        The server is shared: another client loading a world of its own on it
        replaces this runner's (the ego vanishes, the clock restarts), and a
        scenario set up on that map would snap its poses against roads it was
        not written for -- or, had it reloaded this very map, among its
        actors, settings and weather.  Every load starts a new episode, so the
        episode id -- which the client holds, and costs nothing to read --
        says whether anyone loaded a world since this runner last did; then
        the batch's map is loaded again, OpenDRIVE and all, whatever map the
        server runs.

        The id is the one the client last heard of on its tick stream: a load
        the stream has not told it about yet goes unseen, the run fails on the
        foreign world, and its retry sees the new episode.

        A runner that loaded no map itself runs on whatever world there is,
        as it always has.
        """
        if not self._batch_map:
            if self._world is None:
                self._world = self._client.get_world()
            # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
            held: "carla.World" = self._world
            return held
        world = self._client.get_world()
        episode = episode_of(world)
        if same_episode(episode, self._batch_episode):
            self._world = world
            return world
        logger.warning(
            "[%s] The CARLA world is not the one these runs loaded (another "
            "client loaded a world on this server, or a reset failed): "
            "loading %r again",
            scenario_name,
            self._batch_map,
        )
        self._load_batch_map()
        reloaded = self._client.get_world()
        self._world = reloaded
        return reloaded

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------

    @property
    def world_reset(self) -> str:
        """How :meth:`reset_world` resets the world: ``"reload"`` or ``"soft"``."""
        return self._world_reset

    def _capture_baseline(
        self, world: "carla.World", scenario_name: str
    ) -> Optional[WorldBaseline]:
        """What a soft reset of *world*, freshly loaded, puts back; None when
        the runner reloads, or the world cannot be read (the next reset then
        loads the map)."""
        self._soft_resets = 0
        if self._world_reset != RESET_SOFT:
            return None
        try:
            return capture_baseline(world)
        except Exception:  # noqa: BLE001 - the next reset reloads instead
            logger.warning(
                "[%s] Reading the world for a soft reset failed: the next "
                "reset loads the map",
                scenario_name,
                exc_info=True,
            )
            return None

    def _soft_reset(self, scenario_name: str) -> bool:
        """Reset the world without loading the map, if it can be done.

        Returns False -- the caller then loads the map -- when there is no
        baseline, a reload is due (*reload_every*), the server runs another
        episode than the one the baseline was read in (another client loaded
        a world), or the soft reset raises or finds anything left over.
        """
        if self._baseline is None:
            logger.info(
                "[%s] No baseline to soft-reset to: loading the map", scenario_name
            )
            return False
        baseline: WorldBaseline = self._baseline
        if self._reload_every > 0 and self._soft_resets + 1 >= self._reload_every:
            logger.info(
                "[%s] Reload due (every %d resets): loading the map",
                scenario_name,
                self._reload_every,
            )
            return False
        try:
            world = self._client.get_world()
            if not same_episode(episode_of(world), baseline.episode):
                logger.warning(
                    "[%s] The CARLA world is not the one the soft reset knows "
                    "(another client loaded a world on this server): loading "
                    "the map",
                    scenario_name,
                )
                return False
            soft_reset_world(self._client, world, baseline, scenario_name)
            self._world = world
        except Exception:  # noqa: BLE001 - the map is loaded instead
            logger.warning(
                "[%s] The soft reset failed: loading the map",
                scenario_name,
                exc_info=True,
            )
            return False
        self._soft_resets += 1
        return True

    def reset_world(self, scenario_name: str = "") -> None:
        """Reset the world, so the next scenario starts on a clean one.

        With ``world_reset="reload"`` (the default) the map is loaded again:
        the one :meth:`load_map_by_name` loaded, when this runner loaded one,
        and the server's current map otherwise.  With ``world_reset="soft"``
        the world is put back as it was loaded, without a load, when it can
        be (:meth:`_soft_reset`), and the map is loaded otherwise.
        """
        if self._world_reset == RESET_SOFT and self._soft_reset(scenario_name):
            return
        self._reload_world(scenario_name)

    def _reload_world(self, scenario_name: str) -> None:
        """Load the map again (see :meth:`reset_world`)."""
        # Reload the world to guarantee a completely clean state (all
        # actors, sensors, physics state, settings and weather are reset).
        # This is more reliable than destroying actors individually, which
        # can fail with "failed to destroy actor" errors from the CARLA
        # server.  Flush any pending server-side operations (e.g. replayer
        # shutdown) before reloading.  The run's actors are still destroyed
        # first, on a best-effort basis: the reload alone leaves a listening
        # sensor's stream open (_destroy_leftovers_before_reload).
        try:
            world = self._client.get_world()
            world.tick()
            _destroy_leftovers_before_reload(world, scenario_name)
            if self._batch_map:
                # The runs' map, not the current one, and its OpenDRIVE
                # installed again: another client may have loaded a world (or
                # installed an OpenDRIVE) of its own on this server meanwhile.
                self._load_batch_map()
            else:
                reload_current_map(self._client)
            # After reload, get the fresh world and tick it once so that
            # CARLA fully initialises the new world state.
            fresh = self._client.get_world()
            fresh.tick()
            self._world = fresh
            self._batch_episode = episode_of(fresh)
            self._baseline = self._capture_baseline(fresh, scenario_name)
            logger.info("[%s] World reloaded and ticked", scenario_name)
        except Exception:
            # Whatever the world is now, it is not one to soft-reset to.
            self._baseline = None
            logger.warning(
                "[%s] Reloading the map failed — world may retain residual state",
                scenario_name,
                exc_info=True,
            )
            # Ensure self._world is refreshed even on failure so that the
            # next scenario does not operate on a stale world reference.
            try:
                self._world = self._client.get_world()
            except Exception:
                logger.warning(
                    "[%s] Failed to refresh world reference after reload failure",
                    scenario_name,
                    exc_info=True,
                )

    def run_scenario(
        self,
        scenario: BaseScenario,
        *,
        timeout_seconds: float | None = None,
        case: OddCase | None = None,
        reset_world: bool = True,
        file_stem: str | None = None,
    ) -> ScenarioResult:
        """Execute a single scenario from setup to teardown.

        Steps:
        1. Enable synchronous mode on the world
        2. Prepare the traffic backend (synchronous mode, seed, whatever it
           needs to start), before anything is spawned
        3. Call ``scenario.setup(world)``
        4. Spawn the ego vehicle
        5. Start the CARLA native recorder
        6. Register the default timeout fail condition, unless the scenario
           registered a timeout of its own
        7. Run the tick loop
        8. Stop the recorder and destroy the ego vehicle
        9. Close the traffic backend and restore the world

        Args:
            scenario: The scenario to run.
            timeout_seconds: The default timeout for this scenario; the
                runner's ``timeout_seconds`` when ``None``.  Not registered
                when the scenario's setup registers a ``TimeoutCondition`` of
                its own (bare, or inside an ``OrCondition`` or a
                ``StickyCondition``): such a scenario runs to its own limit,
                whatever is passed here.
            case: The case drawn from the ODD that the scenario runs, and the
                space it was drawn in: recorded in the coverage file with the
                run's outcome, so that the merged report can assess its
                flakiness and the next sweep's flaky search learn from it.
            reset_world: Reset the world at the end (:meth:`reset_world`: a
                map reload, or a soft reset), so the next scenario starts on a
                clean world.  A queue's last scenario has no next one, and
                whatever runs on the server after it loads its map anyway.
            file_stem: The stem the run's files are named by.  ``None`` (the
                default) names them by the scenario's class, numbered past the
                runs already in the output directory: ``CutInScenario``, then
                ``CutInScenario-1`` and so on.  A batch run by several processes
                (docs/slurm.md) gives each case the name it would have had.

        Returns:
            The outcome of the scenario execution.
        """
        if timeout_seconds is None:
            timeout_seconds = self.timeout_seconds
        scenario_name = type(scenario).__name__
        world = self._world_for_next_run(scenario_name)
        if file_stem is None:
            file_stem = _run_file_stem(self.output_dir, scenario_name)

        # A batch runs several scenarios against one world, so the entities of
        # the previous one are cleared before this one registers its own -- an
        # entity from the last run still answering to "npc1" would be worse
        # than no entity at all.  So are the props a SpawnAction placed, and
        # the entry speeds and warm-up flag: the end of a run clears them too,
        # but a spawn card ticked outside any run (a test, a tool) holds a
        # speed for an actor id this run's world may reuse.
        clear_entities()
        # What a scenario before carried out is not under way in this one.
        clear_adversities()
        clear_props()
        _set_warmup_done(False)
        _clear_entry_speeds()
        # And where the last run's ego had its base_link: this run's ego says
        # where its own is when its scenario starts.
        clear_ego_base_link()

        # The ego is built before the cleanup, not after it: an entity that
        # attaches to an actor someone else spawned has to be able to say so
        # before that actor would be destroyed.
        if TYPE_CHECKING:
            # create_ego() instantiates the ego's class (`ego_type`), which
            # Codon cannot hold as a value: the check takes the entity a
            # scenario was given, or a plain EgoVehicle.
            ego = EgoVehicle()
            if scenario.ego_entity is not None:
                ego = scenario.ego_entity
        else:
            ego = scenario.create_ego()
        # Handed on as the vehicle it is (a typed local: the static check
        # upcasts there); anything else (a test double) goes through
        # register_entity, which keeps it as it always has.  Codon resolves
        # isinstance() by the static type, so it compiles the typed branch.
        if isinstance(ego, EgoVehicle):
            driven: BackendDriven = ego
            register_vehicle_entity(EGO_ROLE_NAME, driven)
        else:
            register_entity(EGO_ROLE_NAME, ego)
        backend = self._traffic_backend
        ego.set_traffic_backend(backend)

        # Destroy any leftover actors from a previous scenario that may
        # have survived a failed map reload.  On a clean world this
        # is a no-op.
        _destroy_all_dynamic_actors(
            world,
            scenario_name,
            keep_role_names=(
                frozenset({str(EGO_ROLE_NAME)})
                if ego.attaches_to_existing_actor
                else frozenset[str]()
            ),
        )

        # Enable synchronous mode so we control the simulation tick rate.
        # Original settings are not saved because the reset at the end of
        # this method (a map reload, or a soft reset to the settings the map
        # was loaded with) puts them back.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        recording_started = False
        # The world frame the recording starts at, and how many frames of it
        # are the warm-up before the scenario's first tick.
        recording_start_frame = 0
        warmup_ticks = 0
        trajectory = TrajectoryRecorder(
            self.output_dir / f"{file_stem}.trajectory.jsonl",
            self.map_source,
            self.lanelet2_path,
            self._xodr_path,
            self.projector_type,
        )
        tick_count = 0
        result: Optional[ScenarioResult] = None
        coverage: Optional[CoverageCollector] = None
        t4_recording: Optional[T4Recording] = None
        npc_events: Optional["NpcCollisionMonitor"] = None
        # The log the Rerun converter takes once the result JSON is written.
        rerun_log: Optional[Path] = None
        # The scenario's seed is the run's (autoware_carla_scenario.seed): what
        # setup() builds with no seed of its own draws from it.  The one before
        # comes back when the run ends.
        seed = int(scenario.random_seed)
        previous_seed = set_run_seed(seed)
        logger.info("[%s] Seed: %d", scenario_name, seed)

        try:
            # Let the traffic backend get ready before anything is spawned: this
            # is where the TrafficManager is put in step with the world and
            # seeded (a seed only decides anything if it is set before the first
            # autopilot call), and where a backend that cannot run at all -- a
            # simulator that is not installed -- refuses.  Inside the try, so a
            # refusal still gets the cleanup every other failure gets.
            # How far drivers see, and the grip the road leaves the tires, are
            # what this run's weather leaves, not what the last run's left in
            # the same episode (a soft world reset keeps it).
            forget_visibility()
            forget_tire_friction(world)
            # CARLA's pedestrian navigation (the AI walker controllers' choice
            # of where to go) from the run's seed, before any walker is spawned.
            world.set_pedestrians_seed(seed)
            backend.prepare(
                TrafficContext(
                    client=self._client,
                    world=world,
                    map_name=_map_name_of(world),
                    xodr_path=self._xodr_path,
                    # The step the world actually got, not a constant that
                    # happens to match it: a backend stepping a second simulator
                    # against the wrong number drifts silently.
                    fixed_delta_seconds=settings.fixed_delta_seconds,
                    random_seed=scenario.random_seed,
                    output_dir=self.output_dir,
                )
            )
            # What the run's road users draw their parameters from, for the
            # ODD's probes of them (odd.traffic_probes).
            record_run_parameters(
                backend.vehicle_parameters,
                backend.pedestrian_parameters,
                backend.bicycle_parameters,
            )
            if TYPE_CHECKING:
                # describe() is a dict of mixed value types, which Codon
                # cannot express (UNCALLED in typecheck/library.py).
                pass
            else:
                logger.info("[%s] Traffic: %s", scenario_name, backend.describe())

            logger.info("[%s] === Setup start ===", scenario_name)
            # A retry, or a re-run, of the same scenario object calls setup()
            # on it again: what the earlier attempt registered -- conditions
            # that may have latched, actions, callbacks, entities, coverage,
            # measures -- goes, and what __init__ registered stays.  Before
            # set_traffic_backend(), so the backend is handed only to actions
            # this run has.
            scenario._reset_registrations()
            scenario.set_client(self._client, tm_port=self._backend_tm_port)
            # Before setup(), because setup() is where a scenario spawns and
            # registers its NPCs, and register_entity() passes the backend on.
            scenario.set_traffic_backend(backend)
            # A logical scenario's route is process-wide, like the signal
            # controllers: one left by the previous scenario of a batch must
            # not be read as this one's.  A scenario with a route sets its own
            # in setup().
            from .route import clear_scenario_route  # noqa: PLC0415

            clear_scenario_route()
            # Who controls which entity is process-wide too: an action of the
            # previous scenario that never finished holds nothing in this one.
            reset_control_slots()
            scenario.setup()
            # What setup() registered may need a backend the queue could not
            # see before (a Python scenario's actions): started now, before
            # the run ticks, or the run is refused.
            if self.preparer is not None:
                preparer: BackendPreparer = self.preparer
                preparer.prepare_actions(
                    scenario.registered_actions(), log_dir=self.output_dir
                )
            # Setup is where a scenario may still name a destination the config
            # did not, so an ego that cannot start without one is checked once
            # setup returns -- and here rather than inside `_setup_ego_spawn`,
            # which a scenario that snaps its own spawn never calls.
            scenario.require_goal()
            self._warn_if_route_leaves_odd(scenario_name, scenario)
            # What the ODD's measures read is built now, not on the first tick
            # that reads them: the vehicle-ahead measures' routing graph takes
            # tens of seconds on a large map.  Not for a measure the scenario
            # replaced, which reads what its own reader reads.
            prepare_measures(self.odd, scenario)
            logger.info("[%s] Spawning ego vehicle ...", scenario_name)
            if TYPE_CHECKING:
                # EgoVehicle.spawn() takes the model's EgoConfig (entity.ego
                # imports scenario_base through the model: MODELLED_IMPORTS in
                # typecheck/library.py), which the checked BaseScenario's is
                # not, so the check takes the actor the entity holds once
                # spawned.
                spawned = ego.actor
                assert spawned is not None
                ego_actor = spawned
            else:
                ego_actor = ego.spawn(world, scenario.ego_config)
            logger.info(
                "[%s] Ego spawned: id=%d blueprint=%s",
                scenario_name,
                ego_actor.id,
                ego_actor.type_id,
            )
            backend.adopt(ego)

            # Register ego existence fail condition so the scenario fails
            # immediately if the ego is destroyed (e.g. falls through map).
            scenario.register_fail_condition(
                EntityExistenceCondition(EGO_ROLE_NAME, label="ego_existence")
            )

            # Start the native CARLA recorder now that the ego exists, so a
            # replay of it (CARLA's own replayer) shows the warm-up below too;
            # the trajectory log's warmup record says how many frames that is.
            output_path = self.output_dir / f"{file_stem}.log"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._client.start_recorder(str(output_path))
            except RuntimeError as error:
                # CARLA 0.10 (UE5) can refuse start_recorder; a missing replay
                # log must not fail an otherwise healthy scenario run.
                logger.warning(
                    "[%s] CARLA recorder unavailable (%s); continuing without a replay log",
                    scenario_name,
                    error,
                )
            else:
                recording_started = True
                recording_start_frame = int(world.get_snapshot().frame)
                logger.info("[%s] Recording to %s", scenario_name, output_path)

            # Warm-up ticks: let physics and TrafficManager stabilise
            # before the main loop begins.
            for _ in range(scenario.STABILIZE_TICKS):
                self._hold_vehicles_still(world)
                self._pace_tick()
                world.tick()

            # The scenario's initialization phase: the actor exists and
            # physics have settled, but the clock has not started and no
            # condition has been evaluated.  Everything registered with
            # register_init() runs here -- putting the world in the state the
            # run starts from, and handing an ego that plans its own route the
            # mission it needs before the wait below asks whether it is ready.
            logger.info("[%s] === Init actions ===", scenario_name)
            self._run_init(world, scenario)

            # Let the ego entity bring up whatever it needs now that the actor
            # exists and physics have settled (e.g. sensors and an external
            # driver session).  It is told first what every vehicle starts at,
            # for an entity that plays the run-up to the first frame.
            ego.set_initial_speeds(scenario.initial_speeds_mps(ego_actor))
            ego.on_scenario_start(world)

            self._wait_for_ego(world, ego, scenario_name)

            # The init phase is over: the ego is ready and the clock is about to
            # start, so the brakes that held everything still come off before
            # anything is handed to TrafficManager or given a speed.
            self._release_vehicles(world)

            # Traffic last, and only now: the world has been ticking through
            # the wait above, and a car the backend was already driving would
            # have spent that time going -- away from the scenario it was placed
            # for, before its own pre-tick actions had said how it should drive.
            # When the ego opts out (an ExternallyDrivenEgo), its actor is named
            # here so external control can drive it instead: one authority per
            # vehicle, passed as data rather than assumed by the backend.
            skip_ids: set[int] = set()
            if not ego.use_autopilot and ego_actor is not None:
                skip_ids.add(ego_actor.id)

            if TYPE_CHECKING:
                # Codon 0.19 calls an overridden method through a dispatch
                # thunk, which takes no keyword argument (docs/typecheck.md):
                # the check calls TrafficBackend's own, and checks each
                # backend's start() on its own, through its class.
                TrafficBackend.start(backend, world, skip_actor_ids=skip_ids)
            else:
                backend.start(world, skip_actor_ids=skip_ids)

            # Initial speeds last of all, so they are the speeds the scenario
            # starts at rather than ones a long wait has bled off.
            scenario.set_initial_speed(ego_actor)

            _set_warmup_done(True)

            if recording_started:
                warmup_ticks = int(world.get_snapshot().frame) - recording_start_frame
                logger.info(
                    "[%s] Warm-up took %d tick(s) of the recording",
                    scenario_name,
                    warmup_ticks,
                )

            # The default timeout, unless the scenario registered its own.
            _register_default_timeout(scenario, timeout_seconds, scenario_name)

            logger.info("[%s] === Tick loop start ===", scenario_name)
            # The clock starts here, after the ego is ready: an entity that
            # needs a stack to come up would otherwise spend most of the
            # scenario's timeout booting.
            clock = _ScenarioClock(world)
            # The motion of everything on the road, recorded from the ticks
            # this loop steps -- in-process, so no second client has to follow
            # the world from outside.
            trajectory.start(world)
            clock_start_frame = int(world.get_snapshot().frame)
            # The file starts where the clock does; the recording, warmup_ticks
            # frames earlier.
            trajectory.record_warmup(
                warmup_ticks,
                recording_start_frame if recording_started else -1,
                clock_start_frame,
            )
            if self.rerun is not None:
                # The ego's sensors, from the clock's first frame, beside the
                # trajectory log (rerun.lidar / rerun.cameras; nothing otherwise).
                lidar_converter: RerunConverter = self.rerun
                lidar_converter.start_sensors(
                    trajectory.path, file_stem, clock_start_frame
                )
            if self.t4 is not None:
                # The run as a T4 dataset, from the clock's first frame, with
                # sensors of its own on the ego (t4.enabled).
                t4_recorder: T4Recorder = self.t4
                t4_recording = t4_recorder.start(
                    self.output_dir,
                    file_stem,
                    world,
                    ego_actor,
                    clock_start_frame,
                    scenario_name,
                    short_map_name(_map_name_of(world)),
                    self.lanelet2_path,
                )
            trajectory.record(world, clock.simulated)
            trajectory.record_markers(_ego_markers(ego))
            # What the run covers: the attributes of the run's ODD, sampled
            # on every tick and judged against its modules, then the
            # scenario's own items, declared in setup() with register_cover().
            reset_probes()
            # An ODD that maps its taxonomy onto scenario measures reads this
            # scenario's, its own replacements included.
            if TYPE_CHECKING:
                # measures holds the model's BaseScenario (it imports
                # scenario_base through the model: MODELLED_IMPORTS in
                # typecheck/library.py), which the checked one is not.
                pass
            else:
                set_measured_scenario(scenario)
            set_importance_sampling_weight(scenario.importance_sampling_weight)
            reset_ego_motion()
            metric_recorder: Optional[Any] = None
            if TYPE_CHECKING:
                # Duck-typed recorded actors (UNCALLED in typecheck/library.py).
                pass
            else:
                metric_recorder = _metric_recorder(scenario)
            # Walkers measured outside a run (or by a run that never reached
            # its cleanup) are not this run's.
            forget_walkers()
            coverage = CoverageCollector(
                scenario._cover_items, scenario._cross_items, odd=self.odd
            )
            coverage.start(world, clock.simulated)

            # The road users' own collisions and falls, logged (not judged).
            npc_events = _npc_collision_monitor()

            # Tick loop
            while not scenario.is_done():
                # Read once before the world advances and once after, because
                # on the simulated clock those are a whole `fixed_delta_seconds`
                # apart rather than the sliver of real time a tick takes.  Each
                # half of the loop is told the time of the world it is looking
                # at: a pre-tick action acts on the world as it stands, and
                # everything below `world.tick()` observes the world it has
                # become.  Handing the earlier reading to both would date every
                # post-tick observation one step early.
                pre_tick_elapsed = clock.simulated
                tick_count += 1

                # Pre-tick actions (receive the time they act at)
                for action in scenario._pre_tick_actions:
                    action.tick(world, pre_tick_elapsed)

                # Pre-tick callbacks
                for cb in scenario._pre_tick_callbacks:
                    cb(world)

                self._pace_tick()
                world.tick()

                # The world has advanced; everything from here reads the time
                # it advanced to.
                elapsed = clock.simulated
                trajectory.record(world, elapsed)

                # Give the ego entity a chance to drive itself before the
                # scenario's own post-tick hooks observe the new state.
                ego.on_tick(world, elapsed)
                # After the ego's tick: a driver policy's ego has just handed
                # the dataset what its sensors took on it.
                if t4_recording is not None:
                    t4_recording.record(world)
                trajectory.record_markers(_ego_markers(ego))

                # Then the traffic, which reacts to where the ego now is.
                # See TrafficBackend.tick for what a backend does with this.
                backend.tick(world, elapsed)

                # Post-tick actions (receive elapsed)
                for action in scenario._post_tick_actions:
                    action.tick(world, elapsed)

                # Post-tick callbacks
                for cb in scenario._post_tick_callbacks:
                    cb(world)

                # Where every action is in its lifecycle after this tick: the
                # pre-tick ones acted on the frame just recorded, the post-tick
                # ones on what it showed.
                for phase, actions in (
                    ("pre_tick", scenario._pre_tick_actions),
                    ("post_tick", scenario._post_tick_actions),
                ):
                    for i, action in enumerate(actions):
                        trajectory.record_action(
                            phase,
                            i,
                            action.label,
                            type(action).__name__,
                            action.state.value,
                        )
                        # What it sampled this tick (NADE's adversities).
                        for event in action.take_adversity_events():
                            trajectory.record_adversity(action.label, event)
                        for arrival in action.take_waypoint_arrivals():
                            trajectory.record_waypoint_arrival(action.label, arrival)
                        for entity, actor_id in action.take_spawns():
                            trajectory.record_spawn(action.label, entity, actor_id)
                        for entity, actor_id in action.take_despawns():
                            trajectory.record_despawn(action.label, entity, actor_id)
                if npc_events is not None:
                    collisions, falls, fallen_gone = npc_events.tick(world, elapsed)
                    for collision in collisions:
                        trajectory.record_collision(collision)
                    for fall in falls:
                        trajectory.record_walker_fallen(fall)
                    for entity, actor_id in fallen_gone:
                        trajectory.record_despawn(
                            "fallen walker", entity, actor_id, "fallen"
                        )

                # The weight the actions above leave the run with, for the
                # metric that reads it.
                set_importance_sampling_weight(scenario.importance_sampling_weight)
                # The ego's motion, which the weaving metrics read.
                track_ego_motion(world, elapsed)
                if TYPE_CHECKING:
                    # Duck-typed recorded actors (UNCALLED in
                    # typecheck/library.py).
                    pass
                else:
                    # The frames the MetricConditions are judged on once the
                    # run is over.
                    if metric_recorder is not None:
                        metric_recorder.record(
                            world, elapsed, scenario.importance_sampling_weight
                        )

                # Coverage samples the world the hooks above have acted on.
                coverage.tick(world, elapsed)

                # Periodic ego OpenDRIVE position log
                if tick_count % _CONDITION_LOG_INTERVAL == 0:
                    _log_ego_opendrive_position(
                        world, scenario_name, elapsed, tick_count
                    )

                # Check pass conditions
                for i, condition in enumerate(scenario._pass_conditions):
                    check = condition.check(world, elapsed)
                    trajectory.record_condition(
                        "pass", i, condition.label, check is not None
                    )
                    if check is not None:
                        logger.info(
                            "[%s] Pass condition [%d](%s) SATISFIED: %s",
                            scenario_name,
                            i,
                            condition.label,
                            check.message,
                        )
                        result = ScenarioResult(
                            passed=True,
                            message=check.message,
                            elapsed_seconds=check.elapsed_seconds,
                            condition_statuses=_condition_statuses(
                                scenario, world, elapsed, scenario_name
                            ),
                        )
                        break
                    if tick_count % _CONDITION_LOG_INTERVAL == 0:
                        logger.info(
                            "[%s] Pass condition [%d](%s) pending at t=%.2fs (tick %d)",
                            scenario_name,
                            i,
                            condition.label,
                            elapsed,
                            tick_count,
                        )

                if result is not None:
                    break

                # Check fail conditions
                for i, condition in enumerate(scenario._fail_conditions):
                    if is_post_run(condition):
                        # A MetricCondition: judged once the run is over.
                        continue
                    check = condition.check(world, elapsed)
                    trajectory.record_condition(
                        "fail", i, condition.label, check is not None
                    )
                    if check is not None:
                        logger.info(
                            "[%s] Fail condition [%d](%s) TRIGGERED: %s",
                            scenario_name,
                            i,
                            condition.label,
                            check.message,
                        )
                        statuses = _condition_statuses(
                            scenario, world, elapsed, scenario_name
                        )
                        result = ScenarioResult(
                            passed=False,
                            message=check.message,
                            elapsed_seconds=check.elapsed_seconds,
                            condition_statuses=statuses,
                            failure=_failure(
                                _status_label("fail", i, condition),
                                scenario.fail_category(i),
                                check.elapsed_seconds,
                                check.message,
                                statuses,
                            ),
                        )
                        break

                if result is not None:
                    break

                # An ego entity can ask to stop early (e.g. the driver policy
                # signalled termination).  Conditions are checked first so a
                # pass or fail firing on the same tick still decides the outcome;
                # on its own, an early stop is not a pass.
                if ego.termination_requested:
                    logger.info(
                        "[%s] Ego entity requested termination at t=%.2fs (tick %d)",
                        scenario_name,
                        elapsed,
                        tick_count,
                    )
                    reason = ego.termination_reason
                    message = "Ego entity requested session termination" + (
                        "" if reason is None else f": {reason}"
                    )
                    statuses = _condition_statuses(
                        scenario, world, elapsed, scenario_name
                    )
                    result = ScenarioResult(
                        passed=False,
                        message=message,
                        elapsed_seconds=elapsed,
                        condition_statuses=statuses,
                        failure=_failure(
                            "", FailureCategory.DRIVER, elapsed, message, statuses
                        ),
                    )
                    break

            # If the loop exited via is_done() with no result, treat as passed
            if result is None:
                elapsed = clock.simulated
                result = ScenarioResult(
                    passed=True,
                    message="Scenario completed successfully",
                    elapsed_seconds=elapsed,
                )

            if TYPE_CHECKING:
                # Duck-typed recorded actors, and worker processes spoken to
                # in JSON (UNCALLED in typecheck/library.py).
                pass
            else:
                # The MetricConditions, on the metrics of the recorded frames,
                # while the run's map is still loaded.
                result = _judge_metrics(
                    scenario,
                    metric_recorder,
                    world,
                    result,
                    scenario_name,
                    trajectory,
                )

            # Sampled before teardown, while the ego still exists.
            coverage.end(world, clock.simulated)

        finally:
            set_run_seed(previous_seed)
            set_measured_scenario(None)
            set_importance_sampling_weight(None)
            reset_ego_motion()
            # The walkers the trajectory runs and the vehicles led out of their
            # junctions measured were this run's.
            forget_walkers()
            logger.info("[%s] === Cleanup start ===", scenario_name)
            _set_warmup_done(False)
            _clear_entry_speeds()

            # Let the ego entity tear down its own resources (driver session,
            # sensors) while the world is still alive.
            try:
                ego.on_scenario_end(world)
            except Exception:
                logger.warning(
                    "[%s] Ego entity teardown failed",
                    scenario_name,
                    exc_info=True,
                )
            # The ego's base_link was this run's: the ego forgets it in its
            # on_scenario_end, which may have failed before it did.  Right
            # here, before any later step of the cleanup can raise past it.
            clear_ego_base_link()

            # Stop and destroy the sensors the conditions started listening,
            # through their own handles, while the world is still the run's:
            # a sensor destroyed through any other handle (the leftovers'
            # sweep, a reload) leaves its stream subscribed on the client,
            # which can abort the process later (BaseCondition.close).
            try:
                scenario.close_conditions()
            except Exception:
                logger.warning(
                    "[%s] Closing the conditions failed",
                    scenario_name,
                    exc_info=True,
                )
            if npc_events is not None:
                # Its sensors, through their own handles, as the conditions'.
                npc_events.close()
            # And the props a SpawnAction placed: the leftovers' sweep takes
            # vehicles, walkers and sensors, not static props.
            clear_props()

            # The T4 dataset's sensors go before the ego they are on.
            if t4_recording is not None:
                t4_recording.close()

            # Explicitly destroy the ego vehicle so it does not persist
            # if the map reload fails later.
            try:
                ego.destroy()
                logger.info("[%s] Ego vehicle destroyed", scenario_name)
            except Exception:
                logger.warning(
                    "[%s] Failed to destroy ego vehicle",
                    scenario_name,
                    exc_info=True,
                )

            if recording_started:
                try:
                    self._client.stop_recorder()
                    logger.info("[%s] Recorder stopped", scenario_name)
                except Exception:
                    # The trajectory log's verdict and close below still go.
                    logger.warning(
                        "[%s] Stopping the CARLA recorder failed",
                        scenario_name,
                        exc_info=True,
                    )
            if self.rerun is not None:
                lidar_recording: RerunConverter = self.rerun
                lidar_recording.stop_sensors()
            if trajectory.active:
                if result is not None:
                    trajectory.record_result(result)
                trajectory.close()
                logger.info(
                    "[%s] Trajectory written to %s", scenario_name, trajectory.path
                )
                if self.rerun is not None:
                    # Converted in the background while the next case runs;
                    # the run waits for it at its end (rerun.py).  A log
                    # without its result record (the recorder failed writing
                    # it) is converted with the result JSON beside it, so a
                    # run with a result hands its log over once that is
                    # written, below: handed over now, the converter could
                    # read it while it is written.
                    if result is not None:
                        result.rerun_recording = str(recording_path(trajectory.path))
                        rerun_log = trajectory.path
                    else:
                        # An exception is on its way: no result is written.
                        converter: RerunConverter = self.rerun
                        converter.submit(trajectory.path, file_stem)
                if self.t4scene is not None:
                    # Best effort, as the log itself: a failure is logged.
                    exporter: T4SceneExporter = self.t4scene
                    exporter.export(trajectory.path)

            if TYPE_CHECKING:
                # describe() and fidelity_census() are dicts of mixed value
                # types, which Codon cannot express (UNCALLED in
                # typecheck/library.py).
                pass
            else:
                # Before close(), which forgets what it counts.
                traffic_record = _traffic_record(backend, scenario_name)
            # Close the backend so the next run starts with a fresh one --
            # for the TrafficManager that resets its InMemoryMap cache and
            # internal state, for a backend with a process of its own it is
            # that process going away.
            try:
                backend.close()
            except Exception:
                logger.warning(
                    "[%s] Traffic backend teardown failed",
                    scenario_name,
                    exc_info=True,
                )
            logger.info("[%s] === Cleanup done ===", scenario_name)

        if result is not None:
            status = "PASSED" if result.passed else "FAILED"
            logger.info(
                "[%s] Result: %s — %s (%.2fs)",
                scenario_name,
                status,
                result.message,
                result.elapsed_seconds,
            )
            json_path = self.output_dir / f"{file_stem}_result.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if TYPE_CHECKING:
                    # The result document of mixed value types, with the
                    # traffic and environment records (UNCALLED in
                    # typecheck/library.py).
                    pass
                else:
                    _write_result(
                        json_path,
                        self.output_dir,
                        file_stem,
                        scenario,
                        result,
                        traffic_record,
                    )
            finally:
                if rerun_log is not None and self.rerun is not None:
                    written: RerunConverter = self.rerun
                    written.submit(rerun_log, file_stem)
            logger.info("[%s] Result JSON written to: %s", scenario_name, json_path)
            if coverage is not None:
                _record_run(coverage, case, result, scenario.outcome_weight)
                coverage_path = self.output_dir / f"{file_stem}_coverage.json"
                coverage.write(coverage_path, scenario_name)
                logger.info(
                    "[%s] Coverage written to: %s", scenario_name, coverage_path
                )

        if reset_world:
            self.reset_world(scenario_name)
        else:
            logger.info("[%s] No scenario follows: world left as it is", scenario_name)

        return result
