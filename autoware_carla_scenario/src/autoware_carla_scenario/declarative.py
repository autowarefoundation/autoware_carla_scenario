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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .authoring.builders import instantiate_action, instantiate_condition
from .authoring.compiler import BuildContext, CompiledScenario, compile_document
from .authoring.models import Entity, ScenarioDocument
from .authoring.persistence import load_document
from .actions.base import TickTiming
from .signals import (
    build_controllers,
    clear_signal_controllers,
    register_signal_controller,
)
from .coordinate import GroundProjectionConfig, Lanelet2Pose, snap_to_carla_road
from .entity._spawn import SpawnTransform
from .entity.pedestrian_entity import PedestrianEntity, PedestrianEntityConfig
from .entity.vehicle_entity import VehicleEntity, VehicleEntityConfig
from .scenario_base import BaseScenario, EgoConfig

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from .coordinate import OpenDrivePose
    from .route.model import RouteMatch

logger = logging.getLogger(__name__)

__all__ = ["DeclarativeScenario", "DeclarativeScenarioConfig", "TURN_LOOKAHEAD_M"]

#: How far (m) before a route junction an ego the TrafficManager drives is told
#: which way to turn there -- within the turn action's own search distance.
TURN_LOOKAHEAD_M = 100.0


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
            for name, value in override.items():
                if value is not None:
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
        """Effective timeout: the Hydra override when set, else the document's."""
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
            frame = RouteFrame(match, lanelet_map, routing_graph)
            (spawn_id, spawn_s), goal = ego_placement(
                frame, spec.ego_spawn_s, spec.ego_goal, spec.ego_goal_margin
            )
            self._spawn_pose = Lanelet2Pose(lanelet_id=spawn_id, s=spawn_s)
            if goal is not None and self.goal_pose is None:
                self.goal_pose = Lanelet2Pose(lanelet_id=goal[0], s=goal[1])
        set_scenario_route(match, lanelet_map, routing_graph)
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
        """Make the route the ego's: waypoints for Autoware, turns for the rest."""
        if self.ego_requires_goal:
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
        """Where a parked entity waits: under the ego's spawn, 10 m apart."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        from .actions.follow_trajectory import HIDDEN_DEPTH_M  # noqa: PLC0415
        from .coordinate.transform import to_carla_world  # noqa: PLC0415

        assert self._spawn_pose is not None  # noqa: S101 -- set before NPCs spawn
        base = to_carla_world(self._spawn_pose)
        slot = 1 + [e.id for e in self._compiled.npcs].index(entity.id)
        return carla.Transform(
            carla.Location(x=base.x + 10.0 * slot, y=base.y, z=base.z - HIDDEN_DEPTH_M),
            carla.Rotation(yaw=base.yaw),
        )

    def _spawn_npcs(self) -> None:
        """Spawn every non-ego entity at its document spawn position."""
        world = self.world
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

        del world
        transform = (
            self._parking_transform(entity)
            if self._parked(entity)
            else _hidden_if_asked(
                entity, to_carla_world(_spawn_pose(entity)).to_carla_transform()
            )
        )
        return PedestrianEntity(
            PedestrianEntityConfig(
                role_name=self._compiled.role_of(entity.id),
                spawn_location=SpawnTransform(transform),
                walker_type=entity.vehicle_type,
            )
        )

    def _build_npc(self, entity: Entity, world: "object") -> VehicleEntity:
        """Return the :class:`VehicleEntity` for *entity*, snapped to the road."""
        from .coordinate.transform import to_opendrive  # noqa: PLC0415

        if self._parked(entity):
            return VehicleEntity(
                VehicleEntityConfig(
                    role_name=self._compiled.role_of(entity.id),
                    spawn_location=SpawnTransform(self._parking_transform(entity)),
                    vehicle_type=entity.vehicle_type,
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
        return VehicleEntity(
            VehicleEntityConfig(
                # The role the compiler already resolved every condition to.
                # Deriving it a second time here is how a vehicle ends up
                # spawned under a name no condition is watching.
                role_name=self._compiled.role_of(entity.id),
                spawn_location=SpawnTransform(
                    _hidden_if_asked(entity, snapped.to_carla_transform())
                ),
                vehicle_type=entity.vehicle_type,
                initial_speed_kmh=entity.initial_speed_kmh,
                spawn_retry_max_count=self.ego_config.spawn_retry_max_count,
                spawn_retry_t_step=self.ego_config.spawn_retry_t_step,
                spawn_retry_z_step=self.ego_config.spawn_retry_z_step,
                # Under the map nothing is in the way, so no retries either.
                od_pose=None if entity.spawn.hidden else od_pose,
                ground_projection=self._ground_projection,
            )
        )


def _hidden_if_asked(entity: Entity, transform: "carla.Transform") -> "carla.Transform":
    """*transform*, moved under the map when *entity* spawns out of the world."""
    if not entity.spawn.hidden:
        return transform
    import typesafe_carla.carla as carla  # noqa: PLC0415

    from .actions.follow_trajectory import HIDDEN_DEPTH_M  # noqa: PLC0415

    location = transform.location
    return carla.Transform(
        carla.Location(x=location.x, y=location.y, z=location.z - HIDDEN_DEPTH_M),
        transform.rotation,
    )


def _spawn_pose(entity: Entity) -> Lanelet2Pose:
    """Return *entity*'s spawn as the Lanelet2 pose it was authored as."""
    spawn = entity.spawn
    return Lanelet2Pose(
        lanelet_id=spawn.lanelet_id,
        s=spawn.s.value,
        t=spawn.t,
        heading=spawn.heading,
    )
