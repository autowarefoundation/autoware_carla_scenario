"""Run an authored :class:`ScenarioDocument` on the existing scenario runtime.

:class:`DeclarativeScenario` is an ordinary :class:`BaseScenario`.  It owns no
tick loop, no condition evaluation and no result handling of its own: it reads a
compiled document and calls the same :meth:`~BaseScenario.register_pre_tick`,
:meth:`~BaseScenario.register_post_tick`,
:meth:`~BaseScenario.register_pass_condition` and
:meth:`~BaseScenario.register_fail_condition` hooks a hand-written scenario
would.  That is the whole point -- an authored scenario and a hand-written one
are the same thing to ``ScenarioRunner``.

Importing this module pulls in CARLA (via :class:`BaseScenario`), so the editor
never imports it; only a live scenario run does.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .authoring.builders import instantiate_action, instantiate_condition
from .authoring.compiler import BuildContext, CompiledScenario, compile_document
from .authoring.models import Entity, ScenarioDocument
from .authoring.persistence import load_document
from .authoring.registry import get_action_spec, get_condition_spec
from .actions.base import TickTiming
from .conditions import TimeoutCondition
from .signals import (
    build_controllers,
    clear_signal_controllers,
    register_signal_controller,
)
from .actions.spawn import SpawnAction
from .coordinate import (
    CarlaWorldPose,
    GroundProjectionConfig,
    Lanelet2Pose,
    snap_to_carla_road,
)
from .entity._spawn import SpawnTransform
from .entity.pedestrian_entity import PedestrianEntity, PedestrianEntityConfig
from .entity.vehicle_entity import (
    ANY_MOTORCYCLE,
    MotorcycleEntity,
    VehicleEntity,
    VehicleEntityConfig,
)
from .gate.probe import GateProbe
from .scenario_base import BaseScenario, EgoConfig

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from .conditions import ScenarioResult
    from .coordinate import OpenDrivePose
    from .gate.record import GateRecord
    from .route.model import RouteMatch

logger = logging.getLogger(__name__)

__all__ = [
    "DeclarativeScenario",
    "DeclarativeScenarioConfig",
    "TURN_LOOKAHEAD_M",
    "document_digest",
]

#: How far (m) before a route junction an ego the TrafficManager drives is told
#: which way to turn there -- within the turn action's own search distance.
TURN_LOOKAHEAD_M = 100.0

#: How much deeper (m) a hidden entity waits than another one waiting where
#: it would overlap it -- taller than any CARLA road user.
HIDDEN_STACK_M = 6.0
#: The room (m) kept between two hidden entities waiting side by side.
_HIDDEN_GAP_M = 1.0
#: The length (m) a hidden entity with no size of its own is taken to have
#: where it waits, by kind: a vehicle's blueprint carries no bounding box, so
#: one longer than any CARLA vehicle (a bus is about 10.3 m).
_WAITING_LENGTH_M: dict[str, float] = {
    "pedestrian": 1.0,
    "motorcycle": 3.0,
    "vehicle": 14.0,
}
#: How much longer (m) than the size it is to fill a ``vehicle_type: auto``
#: vehicle is taken to be: the nearest one the server has may be longer.
_SIZE_MARGIN_M = 4.0
#: Where hidden entities wait: ``(x, y, z, radius)`` in CARLA world, the
#: radius (m) of the circle each one's footprint fits in, whatever its yaw.
WaitingSpots = list[tuple[float, float, float, float]]


def document_digest(document: ScenarioDocument) -> str:
    """The SHA-256 of *document*'s content, its ``ui`` layout left out.

    What a gate record names the document by (``docs/gate.md``): records of
    one scenario id with different digests ran different versions of it.
    """
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    content = document.to_yaml_dict()
    content.pop("ui", None)
    text = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class DeclarativeScenarioConfig:
    """Hydra config group for a declarative scenario.

    An exported scenario package's YAML sets these; everything else about the
    scenario lives in the document, which is version-controlled next to it.

    Attributes:
        name: Registry name, matching what ``register_scenario`` was given.
        document_path: Path to ``document.yaml``.  Relative paths resolve
            against the current working directory; an exported package passes
            an absolute path from its ``register()``.
        timeout_seconds: Overrides the document's own timeout when set, so the
            usual ``scenario.timeout_seconds=...`` CLI override keeps working.
        spawn_overrides: Per-entity spawn overrides, ``{entity_id: {lanelet_id,
            s}}``.  The ego spawns through the framework's own
            ``ego.spawn_lanelet_id`` / ``ego.spawn_s`` keys; this sub-tree gives
            every *other* entity an equivalent addressable key so the
            lanelet-constraint sweeper can drive an NPC spawn with the same
            plain ``key=value`` overrides.  An exported package declares the
            keys in its YAML, so Hydra's struct mode accepts them.
        param_overrides: Per-node parameter overrides, ``{node_id: {field:
            value}}``.  The same channel as *spawn_overrides*, for the lanelet
            an action or a condition names: a document may leave that lanelet to
            the constraint sweeper too, and the sweeper only knows how to write
            a Hydra key.
        route: A logical scenario's route match, as ``scenario-expand``
            writes it (:meth:`~autoware_carla_scenario.route.model.RouteMatch.to_config`).
            Empty -- a run that was not expanded -- searches the map the run
            is on and takes the document's ``route.match_index``.
    """

    name: str = "declarative"
    document_path: Optional[str] = None
    timeout_seconds: Optional[float] = None
    spawn_overrides: dict[str, Any] = field(default_factory=dict)
    param_overrides: dict[str, Any] = field(default_factory=dict)
    route: dict[str, Any] = field(default_factory=dict)


class DeclarativeScenario(BaseScenario):
    """A :class:`BaseScenario` whose content comes from a :class:`ScenarioDocument`.

    Args:
        ego_config: Ego spawn configuration, built by the runner as usual.
        spawn_pose: Ego spawn pose from the Hydra config (``ego.spawn_lanelet_id``
            / ``ego.spawn_s``).  The lanelet-constraint sweeper overrides those
            keys, so a swept run reaches the scenario through the same path a
            hand-written one does.
        config: The Hydra config group; supplies ``document_path`` when
            *document* is not passed directly.
        ground_projection: Ground-projection settings for spawn snapping.
        document: A pre-loaded document, which takes precedence over
            ``config.document_path``.

    Raises:
        ValueError: If neither *document* nor a readable ``document_path`` is given.
    """

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: DeclarativeScenarioConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
        document: ScenarioDocument | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or DeclarativeScenarioConfig()
        self._document = document or self._load_document(self._config)
        # Before the overrides fold a case's lanelets into the document: every
        # case of one logical scenario has the digest of the one document.
        # Only a gate record reads it.
        self._document_digest = (
            document_digest(self._document) if self._document.gate is not None else ""
        )
        self._route_match: Optional[RouteMatch] = None
        # Filled as the entities spawn (_spawn_npcs): where each hidden one
        # waits, and where each one was declared to spawn, by role -- None
        # for one with no spawn of its own to bring it in at.
        self._waiting: WaitingSpots = []
        self._declared: dict[str, Optional[tuple["carla.Transform", Lanelet2Pose]]] = {}
        self._case_count = 0
        self._gate_probe: Optional[GateProbe] = None
        self._apply_spawn_overrides()
        self._apply_param_overrides()
        self._apply_ego_goal()
        # Compiling here (not in setup) surfaces an invalid document before the
        # runner has spent anything on a CARLA session.
        self._compiled: CompiledScenario = compile_document(self._document)
        for warning in self._compiled.warnings:
            logger.warning("%s: %s", warning.path, warning.message)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_document(config: DeclarativeScenarioConfig) -> ScenarioDocument:
        """Load the document named by *config*."""
        if not config.document_path:
            raise ValueError(
                "DeclarativeScenario needs either a document or config.document_path."
            )
        path = Path(config.document_path)
        if not path.is_file():
            raise ValueError(f"Scenario document not found: {path}")
        return load_document(path)

    def _apply_spawn_overrides(self) -> None:
        """Fold ``config.spawn_overrides`` into the document's entity spawns.

        This is how a swept NPC spawn reaches the scenario: the sweeper writes
        ``scenario.spawn_overrides.<entity>.lanelet_id=<id>`` and the value
        lands on the entity here, before compilation.
        """
        for entity_id, override in (self._config.spawn_overrides or {}).items():
            entity = self._document.entity(str(entity_id))
            if entity is None:
                logger.warning(
                    "spawn_overrides names unknown entity %r; ignoring.", entity_id
                )
                continue
            if override is None:
                continue
            lanelet_id = override.get("lanelet_id")
            if lanelet_id is not None:
                entity.spawn.lanelet_id = int(lanelet_id)
            offset = override.get("s")
            if offset is not None:
                entity.spawn.s.value = float(offset)

    def _apply_param_overrides(self) -> None:
        """Fold ``config.param_overrides`` into the document's action/condition params.

        How a swept lanelet reaches a card: the sweeper writes
        ``scenario.param_overrides.<node>.<field>=<id>`` and the value lands on
        the node here, before compilation coerces it.  A node the document no
        longer has is logged and skipped rather than raised on -- an override is
        a run's opinion about a document, and a stale one must not make the
        scenario unrunnable.

        A list of records (a ``record_list`` field) is overridden one record's
        field at a time, ``...<field>.<index>.<sub>=<value>``: the value
        arrives as ``{index: {sub: value}}`` and is merged into that record.
        A whole list replaces the field as any other value does.
        """
        for node_id, override in (self._config.param_overrides or {}).items():
            if override is None:
                continue
            node = self._document.action(str(node_id)) or self._document.condition(
                str(node_id)
            )
            if node is None:
                logger.warning(
                    "param_overrides names unknown node %r; ignoring.", node_id
                )
                continue
            spec = get_action_spec(node.type) or get_condition_spec(node.type)
            records = {
                f.name for f in (spec.fields if spec else ()) if f.kind == "record_list"
            }
            for name, value in override.items():
                if value is None:
                    continue
                current = node.params.get(str(name))
                if (
                    str(name) in records
                    and isinstance(current, list)
                    and isinstance(value, Mapping)
                ):
                    _merge_records(str(node_id), str(name), current, value)
                else:
                    node.params[str(name)] = value

    def _apply_ego_goal(self) -> None:
        """Give the ego the goal the document names, unless the run named one.

        Where the ego is going belongs to the ego, so it lands on the scenario's
        ego config.  An exported package renders this same goal into
        ``ego.goal_lanelet_id``, which means a Hydra run arrives with it already
        on the config -- and a CLI override naming a different goal has to win,
        hence "unless".  Applying it here is what makes a document
        self-sufficient when the scenario is built directly, with an ego config
        the caller assembled itself.
        """
        ego = self._document.ego
        if self.goal_pose is not None or ego is None or ego.goal is None:
            return
        self.goal_pose = Lanelet2Pose(lanelet_id=ego.goal.lanelet_id, s=ego.goal.s)

    @property
    def document(self) -> ScenarioDocument:
        """The document this scenario runs."""
        return self._document

    @property
    def compiled(self) -> CompiledScenario:
        """The compiled plan built from :attr:`document`."""
        return self._compiled

    @property
    def timeout_seconds(self) -> float:
        """Effective timeout: the Hydra override when set, else the document's.

        :meth:`setup` registers it as a fail condition, so it bounds the run
        whatever timeout the runner would otherwise default to.
        """
        if self._config.timeout_seconds is not None:
            return float(self._config.timeout_seconds)
        return float(self._document.timeout_seconds)

    # ------------------------------------------------------------------
    # BaseScenario interface
    # ------------------------------------------------------------------

    def _start_signal_controllers(self) -> None:
        """Build the document's signal controllers and set them running.

        Registered as pre-tick callbacks rather than as actions: a controller
        is not something the storyboard does, it is a part of the road that
        keeps running underneath it, and registering it as an action would put
        it in the list an ``action_completed`` condition can name.

        Cleared first, because the registry outlives a single scenario and a
        controller left over from the previous one would go on driving lights
        that this one never declared -- which reads in a report as this
        junction misbehaving.
        """
        clear_signal_controllers()
        controllers = build_controllers(
            self._document.map.traffic_signal_controllers,
            self._document.map.signal_groups,
        )
        for controller in controllers:
            register_signal_controller(controller)
            self.register_pre_tick(controller.tick)
        if controllers:
            logger.info(
                "DeclarativeScenario '%s': running %d signal controller(s): %s",
                self._document.id,
                len(controllers),
                ", ".join(c.name for c in controllers),
            )

    def _prepare_route(self) -> None:
        """Choose a logical scenario's route, and put the ego on it.

        The match an expansion wrote (``scenario.route``) is taken as it is;
        the ego's spawn and goal came with it.  Without one the route search
        runs on the loaded map and match ``route.match_index`` is taken, the
        ego spawning ``ego_spawn_s`` along it and -- unless the run named a
        goal -- sent to its end.  Either way the route becomes the scenario's
        (:func:`~autoware_carla_scenario.route.set_scenario_route`), and the ego
        is sent along it: an Autoware ego through the route's lanelets as
        waypoints, an ego the TrafficManager drives by a turn at each of the
        route's junctions as it comes to it.

        Raises:
            ValueError: If the map has no matching route, or fewer matches than
                ``match_index`` asks for.
        """
        from .coordinate.map_manager import MapManager  # noqa: PLC0415
        from .route import clear_scenario_route, set_scenario_route  # noqa: PLC0415
        from .route.model import RouteMatch, parse_route_search  # noqa: PLC0415

        clear_scenario_route()
        search = self._document.route
        if search is None:
            return
        from .route.frame import RouteFrame, ego_placement  # noqa: PLC0415

        manager = MapManager.get_instance()
        lanelet_map, routing_graph = manager.lanelet_map, manager.routing_graph
        spec = parse_route_search(search.to_sweep_dict())
        match = RouteMatch.from_config(self._config.route or {})
        # How many cases the expansion found on this map: what a gate's case
        # coverage is counted against (docs/gate.md).
        self._case_count = int((self._config.route or {}).get("case_count") or 0)
        if match is None:
            from .route.search import find_route_matches  # noqa: PLC0415

            matches = find_route_matches(spec, lanelet_map, routing_graph)
            if not matches:
                raise ValueError(
                    f"DeclarativeScenario '{self._document.id}': no route of "
                    "this map matches the document's route search"
                )
            if spec.match_index >= len(matches):
                raise ValueError(
                    f"DeclarativeScenario '{self._document.id}': route.match_index "
                    f"is {spec.match_index}, but this map has {len(matches)} "
                    "matching route(s)"
                )
            match = matches[spec.match_index]
            # Unexpanded, the run counts the matches itself (those the ego
            # cannot be placed on included, which an expansion skips).
            self._case_count = len(matches)
            frame = RouteFrame(match, lanelet_map, routing_graph)
            (spawn_id, spawn_s), goal = ego_placement(
                frame, spec.ego_spawn_s, spec.ego_goal, spec.ego_goal_margin
            )
            # The route places the ego along it; across the lane and turned
            # against it is still what the run asked for.
            authored = self._spawn_pose
            self._spawn_pose = Lanelet2Pose(
                lanelet_id=spawn_id,
                s=spawn_s,
                t=authored.t if authored is not None else 0.0,
                heading=authored.heading if authored is not None else 0.0,
            )
            if goal is not None and self.goal_pose is None:
                self.goal_pose = Lanelet2Pose(lanelet_id=goal[0], s=goal[1])
        set_scenario_route(match, lanelet_map, routing_graph)
        self._route_match = match
        logger.info(
            "DeclarativeScenario '%s': route match %d through lanelets %s "
            "(%.1f m, %d segment(s))",
            self._document.id,
            match.index,
            list(match.lanelet_ids),
            match.length,
            len(match.segments),
        )
        self._send_ego_along(match)

    def _send_ego_along(self, match: "RouteMatch") -> None:
        """Make the route the ego's: waypoints with a goal, turns without one.

        With a goal, the route's lanelets between the ego's spawn and goal
        lanelets become its waypoints (unless some were configured), which the
        ego's :class:`~autoware_carla_scenario.actions.RoutingAction` hands it:
        an Autoware ego plans the matched route rather than the shortest one,
        and one a traffic backend drives is handed the whole route as one path
        (:meth:`~autoware_carla_scenario.traffic.driven.BackendDriven.route_to`).
        Without one -- only an ego driven for it may have none -- nothing is
        handed over, and the ego is turned at each junction instead.
        """
        if self.ego_requires_goal or self.goal_pose is not None:
            if self.waypoint_poses:
                return
            ids = list(match.lanelet_ids)
            spawn = self._spawn_pose.lanelet_id if self._spawn_pose else None
            goal = self.goal_pose.lanelet_id if self.goal_pose else None
            first = ids.index(spawn) + 1 if spawn in ids else 0
            last = ids.index(goal) if goal in ids else len(ids)
            self.waypoint_poses = [
                Lanelet2Pose(lanelet_id=lanelet_id, s=0.0)
                for lanelet_id in ids[first:last]
            ]
            return
        from .conditions import RouteProgressCondition  # noqa: PLC0415
        from .actions import TurnAction  # noqa: PLC0415
        from .constants import EGO_ROLE_NAME  # noqa: PLC0415
        from .traffic import TurnDirection  # noqa: PLC0415

        previous_exit = 0.0
        for index, junction in enumerate(match.junctions):
            if junction.turn not in ("left", "right", "straight"):
                logger.warning(
                    "Route junction %d turns both ways (%s); the ego is not "
                    "steered through it",
                    index,
                    junction.turn,
                )
                previous_exit = junction.end
                continue
            arm_at = max(previous_exit, junction.start - TURN_LOOKAHEAD_M)
            self.register_pre_tick(
                TurnAction(
                    EGO_ROLE_NAME,
                    TurnDirection(junction.turn),
                    condition=RouteProgressCondition(
                        EGO_ROLE_NAME, arm_at, label=f"route_junction_{index}_ahead"
                    ),
                    label=f"route_turn_{index}",
                )
            )
            previous_exit = junction.end

    def setup(self) -> None:
        """Spawn the entities and register the document's actions and assertions."""
        self._prepare_route()
        if self._document.gate is not None and self._route_match is not None:
            # The ego's drive in the route's frame, for the gate's trajectory
            # criteria (docs/gate.md).
            self._gate_probe = GateProbe()
            self.register_post_tick(self._gate_probe)
        od_pose: OpenDrivePose = self._setup_ego_spawn()
        logger.info(
            "DeclarativeScenario '%s': ego spawned on OpenDRIVE road '%s'",
            self._document.id,
            od_pose.road_id,
        )

        self._spawn_npcs()
        self._start_signal_controllers()

        ctx = BuildContext(scenario=self)

        for compiled_action in self._compiled.actions:
            action = instantiate_action(compiled_action, ctx)
            # Published under the document's own id so an ``action_completed``
            # condition -- possibly one built earlier, in another action's
            # trigger -- can find it.
            ctx.actions[compiled_action.node.id] = action
            if isinstance(action, SpawnAction) and action.entity_name:
                self._declare_spawn(action)
            # The phase is read from the document, not from the action: `init`
            # is not a tick position, and an action that has to happen before
            # the loop -- a goal the ego needs to become ready -- would never be
            # performed if it were registered on the loop.
            phase = compiled_action.node.phase
            if phase == "init":
                self.register_init(action)
            elif action.timing is TickTiming.POST_TICK:
                self.register_post_tick(action)
            else:
                self.register_pre_tick(action)
            logger.info(
                "Registered action %s (%s) on %s in %s",
                action.label,
                compiled_action.spec.type_id,
                compiled_action.actor_role or "world",
                phase,
            )

        for compiled_condition in self._compiled.pass_conditions:
            self.register_pass_condition(instantiate_condition(compiled_condition, ctx))
        for compiled_condition in self._compiled.fail_conditions:
            self.register_fail_condition(instantiate_condition(compiled_condition, ctx))
        # The document's own timeout, registered here rather than left to the
        # runner's default: a document with a ``timeout`` node of its own
        # would otherwise lose it, since the runner adds no default to a
        # scenario that already has a timeout.
        self.register_fail_condition(
            TimeoutCondition(self.timeout_seconds, label="document_timeout")
        )

        logger.info(
            "DeclarativeScenario '%s': %d action(s), %d pass / %d fail condition(s)",
            self._document.id,
            len(self._compiled.actions),
            len(self._compiled.pass_conditions),
            len(self._compiled.fail_conditions),
        )

    def is_done(self) -> bool:
        """Always ``False`` -- termination is driven by the pass/fail conditions."""
        return False

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def gate_record(
        self, result: "ScenarioResult", file_stem: str
    ) -> "Optional[GateRecord]":
        """What this run leaves for its logical scenario's gate, or ``None``.

        ``None`` for a document with no gate, and for a run that never chose
        a route (it raised before).  The runner names the outputs it wrote
        beside it and writes it as ``{file_stem}_gate.json``
        (``docs/gate.md``).
        """
        from .coordinate.map_manager import MapManager  # noqa: PLC0415
        from .gate.record import GateCase, GateRecord  # noqa: PLC0415
        from .world_reset import short_map_name  # noqa: PLC0415

        gate = self._document.gate
        match = self._route_match
        if gate is None or match is None:
            return None
        map_name = self._document.map.name
        try:
            carla_map = MapManager.get_instance().carla_map
            if carla_map is not None:
                map_name = short_map_name(carla_map.name)
        except Exception:  # noqa: BLE001 -- the document's name stands in
            logger.debug("no CARLA map to name the gate case by", exc_info=True)
        failure = result.failure
        return GateRecord(
            scenario=self._document.id,
            title=self._document.title,
            digest=self._document_digest,
            gate=gate.to_dict(),
            case=GateCase(
                map=map_name,
                index=match.index,
                lanelet_ids=tuple(int(v) for v in match.lanelet_ids),
                case_count=self._case_count,
                segment_kinds=tuple(seg.kind for seg in match.segments),
                segment_ends=tuple(float(seg.end) for seg in match.segments),
                segment_lanelets=tuple(len(seg.lanelet_ids) for seg in match.segments),
            ),
            passed=result.passed,
            message=result.message,
            elapsed_seconds=result.elapsed_seconds,
            failure_category=(
                None if failure is None else failure.to_dict()["category"]
            ),
            ego=list(self._gate_probe.rows) if self._gate_probe is not None else [],
        )

    # ------------------------------------------------------------------
    # Entity spawning
    # ------------------------------------------------------------------

    def _parked(self, entity: Entity) -> bool:
        """Whether *entity* waits out of the world with no lanelet of its own.

        A logical scenario's road users are brought in by their actions, on
        whatever map the route was found; the lanelet a hidden one spawns on
        would name a lanelet of one map only, so it is not asked for.
        """
        return entity.spawn.hidden and (
            self._document.route is not None or entity.spawn.lanelet_id <= 0
        )

    def _parking_transform(self, entity: Entity) -> "carla.Transform":
        """Where a parked entity waits: under the ego's spawn, deeper than
        any other waiting there (:func:`_hidden_if_asked`)."""
        from .coordinate.transform import to_carla_world  # noqa: PLC0415

        assert self._spawn_pose is not None  # noqa: S101 -- set before NPCs spawn
        base = to_carla_world(self._spawn_pose).to_carla_transform()
        return _hidden_if_asked(entity, base, self._waiting)

    def _parked_declared(
        self, entity: Entity, world: "object"
    ) -> Optional[tuple["carla.Transform", Lanelet2Pose]]:
        """Where a parked *entity* was declared to spawn, as a pose it is
        brought in at would be (a vehicle snapped to the road, a pedestrian
        not): ``None`` on lanelet 0 or less, or on one this map does not have
        (said) -- a logical scenario's lanelet names a lanelet of one map."""
        from .coordinate.transform import to_carla_world  # noqa: PLC0415

        if entity.spawn.lanelet_id <= 0:
            return None
        pose = _spawn_pose(entity)
        try:
            if entity.kind == "pedestrian":
                at = to_carla_world(pose)
            else:
                at = snap_to_carla_road(
                    pose, world, ground_projection=self._ground_projection
                )
        except Exception:  # noqa: BLE001 -- said; it is just not brought in
            logger.warning(
                "%s: its spawn, lanelet %d, is not on this map: a Spawn card "
                "with no place will not bring it in",
                entity.id,
                entity.spawn.lanelet_id,
                exc_info=True,
            )
            return None
        return (at.to_carla_transform(), pose)

    def _declare_spawn(self, action: SpawnAction) -> None:
        """Tell an entity's *action* where the entity was declared to spawn,
        which it is brought in at with no place of its own (#2014)."""
        declared = self._declared.get(action.entity_name)
        if declared is None:
            action.declare_spawn(None)
            return
        transform, lane = declared
        action.declare_spawn(CarlaWorldPose.from_carla_transform(transform), lane)

    def _spawn_npcs(self) -> None:
        """Spawn every non-ego entity at its document spawn position."""
        world = self.world
        self._waiting = []
        self._declared = {}
        for entity in self._compiled.npcs:
            if entity.kind == "pedestrian":
                walker = self._build_pedestrian(entity, world)
                actor = walker.spawn(world)
                self.register_pedestrian(walker)
            else:
                npc_entity = self._build_npc(entity, world)
                actor = npc_entity.spawn(world)
                self.register_entity(npc_entity)
                if entity.spawn.hidden:
                    # Kept off autopilot until its action brings it in: the
                    # TrafficManager deletes a vehicle it finds under the map
                    # as stuck at a dead end.
                    npc_entity.release_from_traffic(world)
            if entity.spawn.hidden:
                # Under the map, where it would otherwise fall for as long as
                # it waits to be brought in.
                actor.set_simulate_physics(False)
            logger.info(
                "Spawned %s %s (%s) on lanelet %d at s=%.1f",
                entity.kind,
                entity.id,
                self._compiled.role_of(entity.id),
                entity.spawn.lanelet_id,
                entity.spawn.s.value,
            )

    def _build_pedestrian(self, entity: Entity, world: "object") -> PedestrianEntity:
        """Return the :class:`PedestrianEntity` for *entity*.

        Deliberately **not** snapped to the road.  ``snap_to_carla_road`` puts a
        pose on the nearest driving surface, which for a pedestrian is the one
        place it must not start: the lanelet an author picks for a walker is a
        crossing or a footway, and snapping would move it into the carriageway
        and call that a spawn.

        The cost is that a pedestrian's z comes from the Lanelet2 map rather
        than from CARLA's mesh, so a map whose footway heights are wrong puts
        the walker slightly above or below the pavement.  That is visible and
        fixable; a pedestrian standing in the road is neither.
        """
        from .coordinate.transform import to_carla_world  # noqa: PLC0415

        role = self._compiled.role_of(entity.id)
        if self._parked(entity):
            transform = self._parking_transform(entity)
            self._declared[role] = self._parked_declared(entity, world)
        else:
            pose = _spawn_pose(entity)
            declared = to_carla_world(pose).to_carla_transform()
            self._declared[role] = (declared, pose)
            transform = _hidden_if_asked(entity, declared, self._waiting)
        return PedestrianEntity(
            PedestrianEntityConfig(
                role_name=self._compiled.role_of(entity.id),
                spawn_location=SpawnTransform(transform),
                walker_type=entity.vehicle_type,
            )
        )

    def _build_npc(self, entity: Entity, world: "object") -> VehicleEntity:
        """Return the :class:`VehicleEntity` for *entity*, snapped to the road:
        a :class:`MotorcycleEntity` for a ``motorcycle``."""
        from .coordinate.transform import to_opendrive  # noqa: PLC0415

        vehicle_type = _vehicle_type(entity, world)
        npc_class = MotorcycleEntity if entity.kind == "motorcycle" else VehicleEntity
        role = self._compiled.role_of(entity.id)
        if self._parked(entity):
            self._declared[role] = self._parked_declared(entity, world)
            return npc_class(
                VehicleEntityConfig(
                    role_name=self._compiled.role_of(entity.id),
                    spawn_location=SpawnTransform(self._parking_transform(entity)),
                    vehicle_type=vehicle_type,
                    initial_speed_kmh=entity.initial_speed_kmh,
                    spawn_retry_max_count=self.ego_config.spawn_retry_max_count,
                    spawn_retry_t_step=self.ego_config.spawn_retry_t_step,
                    spawn_retry_z_step=self.ego_config.spawn_retry_z_step,
                    od_pose=None,
                    ground_projection=self._ground_projection,
                )
            )
        pose = _spawn_pose(entity)
        # Snapped as the Lanelet2 pose it was authored as; the OpenDRIVE pose is
        # carried on to the entity only to enable the spawn retries (which
        # offset the snapped transform, not this pose).
        od_pose = to_opendrive(pose)
        snapped = snap_to_carla_road(
            pose, world, ground_projection=self._ground_projection
        )
        declared = snapped.to_carla_transform()
        self._declared[role] = (declared, pose)
        return npc_class(
            VehicleEntityConfig(
                # The role the compiler already resolved every condition to.
                # Deriving it a second time here is how a vehicle ends up
                # spawned under a name no condition is watching.
                role_name=self._compiled.role_of(entity.id),
                spawn_location=SpawnTransform(
                    _hidden_if_asked(entity, declared, self._waiting)
                ),
                vehicle_type=vehicle_type,
                initial_speed_kmh=entity.initial_speed_kmh,
                spawn_retry_max_count=self.ego_config.spawn_retry_max_count,
                spawn_retry_t_step=self.ego_config.spawn_retry_t_step,
                spawn_retry_z_step=self.ego_config.spawn_retry_z_step,
                # Under the map nothing is in the way, so no retries either.
                od_pose=None if entity.spawn.hidden else od_pose,
                ground_projection=self._ground_projection,
            )
        )


def _vehicle_type(entity: Entity, world: "object") -> str:
    """*entity*'s blueprint: the one it names, or with ``vehicle_type: auto``
    the vehicle of *world* nearest its size (the kind's default where there
    is none)."""
    from .authoring.models import AUTO_MODEL, DEFAULT_MODELS  # noqa: PLC0415

    if entity.vehicle_type != AUTO_MODEL:
        return entity.vehicle_type
    if entity.kind == "motorcycle":
        # MotorcycleEntity picks the server's motorcycle when it spawns.
        return ANY_MOTORCYCLE
    from .entity.fit import fitted_vehicle  # noqa: PLC0415

    size = entity.size
    if size is not None:
        chosen = fitted_vehicle(world, size.length, size.width)
        if chosen is not None:
            logger.info(
                "%s: %s, the vehicle nearest %.2f x %.2f m",
                entity.id,
                chosen,
                size.length,
                size.width,
            )
            return chosen
    fallback = DEFAULT_MODELS[entity.kind]
    logger.warning(
        "%s: no vehicle of this server fits its size; spawned as %s",
        entity.id,
        fallback,
    )
    return fallback


def _waiting_radius(entity: Entity) -> float:
    """The radius (m) of the circle *entity* fits in where it waits hidden:
    half its size's length (and a margin, for the nearest vehicle the server
    has) where it has one, else half its kind's longest."""
    least = _WAITING_LENGTH_M.get(entity.kind, _WAITING_LENGTH_M["vehicle"])
    size = entity.size
    if size is None:
        return least / 2.0
    length = max(size.length, size.width) + _SIZE_MARGIN_M
    if entity.kind == "pedestrian":
        length = max(length, least)
    return length / 2.0


def _hidden_if_asked(
    entity: Entity,
    transform: "carla.Transform",
    waiting: Optional[WaitingSpots] = None,
) -> "carla.Transform":
    """*transform*, moved under the map when *entity* spawns out of the world.

    With *waiting* (where the hidden entities spawned before it wait), deeper
    by :data:`HIDDEN_STACK_M` for as long as it would overlap one of them --
    two declared at one spawn, or parked under the ego's, would otherwise be
    spawned on one spot, and CARLA refuses the second (#2014) -- and noted
    there.
    """
    if not entity.spawn.hidden:
        return transform
    import typesafe_carla.carla as carla  # noqa: PLC0415

    from .actions._trajectory_run import HIDDEN_DEPTH_M  # noqa: PLC0415

    location = transform.location
    x, y = float(location.x), float(location.y)
    z = float(location.z) - HIDDEN_DEPTH_M
    if waiting is not None:
        radius = _waiting_radius(entity)
        # Those it would overlap at its depth, the highest first: below each
        # it meets, until one is far enough below it.
        under = sorted(
            (
                wz
                for wx, wy, wz, wr in waiting
                if math.hypot(x - wx, y - wy) < radius + wr + _HIDDEN_GAP_M
            ),
            reverse=True,
        )
        for wz in under:
            if abs(z - wz) < HIDDEN_STACK_M - 1e-6:
                z = wz - HIDDEN_STACK_M
        waiting.append((x, y, z, radius))
    return carla.Transform(carla.Location(x=x, y=y, z=z), transform.rotation)


def _spawn_pose(entity: Entity) -> Lanelet2Pose:
    """Return *entity*'s spawn as the Lanelet2 pose it was authored as."""
    spawn = entity.spawn
    return Lanelet2Pose(
        lanelet_id=spawn.lanelet_id,
        s=spawn.s.value,
        t=spawn.t,
        heading=spawn.heading,
    )


def _merge_records(
    node_id: str, name: str, records: list[Any], override: Mapping[Any, Any]
) -> None:
    """Merge ``{index: {sub: value}}`` into *records*, a record list's rows.

    An index the list does not have is logged and skipped, as an unknown node
    is: the override is the run's opinion, the document says how many rows
    there are.
    """
    for key, fields in override.items():
        text = str(key)
        index = int(text) if text.isascii() and text.isdigit() else -1
        if not 0 <= index < len(records) or not isinstance(records[index], dict):
            logger.warning(
                "param_overrides names no record %r of %s.%s; ignoring.",
                key,
                node_id,
                name,
            )
            continue
        if isinstance(fields, Mapping):
            records[index].update(
                {str(sub): value for sub, value in fields.items() if value is not None}
            )
