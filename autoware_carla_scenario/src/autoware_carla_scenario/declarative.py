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
from .coordinate import GroundProjectionConfig, Lanelet2Pose, snap_to_carla_road
from .entity._spawn import SpawnTransform
from .entity.vehicle_entity import VehicleEntity, VehicleEntityConfig
from .entity_role import EntityRole
from .scenario_base import BaseScenario, EgoConfig

if TYPE_CHECKING:
    from .coordinate import OpenDrivePose

logger = logging.getLogger(__name__)

__all__ = ["DeclarativeScenario", "DeclarativeScenarioConfig"]


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
    """

    name: str = "declarative"
    document_path: Optional[str] = None
    timeout_seconds: Optional[float] = None
    spawn_overrides: dict[str, Any] = field(default_factory=dict)


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

    def setup(self) -> None:
        """Spawn the entities and register the document's actions and assertions."""
        od_pose: OpenDrivePose = self._setup_ego_spawn()
        logger.info(
            "DeclarativeScenario '%s': ego spawned on OpenDRIVE road '%s'",
            self._document.id,
            od_pose.road_id,
        )

        self._spawn_npcs()

        ctx = BuildContext(scenario=self, client=self._client, tm_port=self.tm_port)

        for compiled_action in self._compiled.actions:
            action = instantiate_action(compiled_action, ctx)
            if compiled_action.node.timing == "post_tick":
                self.register_post_tick(action)
            else:
                self.register_pre_tick(action)
            logger.info(
                "Registered action %s (%s) on %s",
                action.label,
                compiled_action.spec.type_id,
                compiled_action.actor_role or "world",
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

    def _spawn_npcs(self) -> None:
        """Spawn every non-ego entity at its document spawn position."""
        world = self.world
        for index, entity in enumerate(self._compiled.npcs, start=1):
            npc_entity = self._build_npc(entity, index, world)
            npc_entity.spawn(world)
            self.register_entity(npc_entity)
            logger.info(
                "Spawned %s (%s) on lanelet %d at s=%.1f",
                entity.id,
                EntityRole.npc(index),
                entity.spawn.lanelet_id,
                entity.spawn.s.value,
            )

    def _build_npc(self, entity: Entity, index: int, world: "object") -> VehicleEntity:
        """Return the :class:`VehicleEntity` for *entity*, snapped to the road."""
        from .coordinate.transform import to_opendrive  # noqa: PLC0415

        pose = Lanelet2Pose(lanelet_id=entity.spawn.lanelet_id, s=entity.spawn.s.value)
        od_pose = to_opendrive(pose)
        snapped = snap_to_carla_road(
            od_pose, world, ground_projection=self._ground_projection
        )
        return VehicleEntity(
            VehicleEntityConfig(
                role_name=EntityRole.npc(index),
                spawn_location=SpawnTransform(snapped.to_carla_transform()),
                vehicle_type=entity.vehicle_type,
                initial_speed_kmh=entity.initial_speed_kmh,
                spawn_retry_max_count=self.ego_config.spawn_retry_max_count,
                spawn_retry_t_step=self.ego_config.spawn_retry_t_step,
                spawn_retry_z_step=self.ego_config.spawn_retry_z_step,
                od_pose=od_pose,
                ground_projection=self._ground_projection,
            )
        )
