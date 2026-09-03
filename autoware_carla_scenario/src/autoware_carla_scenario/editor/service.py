"""Document mutations behind the editor's HTTP routes.

Routes parse a request and render a template; every change to a
:class:`~autoware_carla_scenario.authoring.models.ScenarioDocument` happens
here, so the rules about *what* an edit means live in one place and can be
tested without a web client.

Mutations are metadata-driven: adding a node seeds it from its spec's field
defaults and updating one parses the form through the same spec, so a newly
registered primitive is editable with no change to this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from ..authoring.models import (
    ActionNode,
    BindingRef,
    ConditionNode,
    ConstraintNode,
    Entity,
    ScenarioDocument,
    SpawnSpec,
)
from ..authoring.persistence import Draft, DraftStore
from ..authoring.registry import (
    default_params,
    get_action_spec,
    get_binding_spec,
    get_condition_spec,
    get_constraint_spec,
)
from ..authoring.starter import blank_document, new_document
from ..authoring.validator import ValidationReport, validate_document
from .forms import parse_params

logger = logging.getLogger(__name__)

__all__ = ["EditorError", "EditorService", "SLOT_FAIL", "SLOT_PASS"]

#: Condition slots that are not attached to an action.
SLOT_PASS = "pass"
SLOT_FAIL = "fail"


class EditorError(Exception):
    """Raised when a request asks for something the document cannot do."""


class EditorService:
    """Draft storage plus every document mutation the editor performs."""

    def __init__(self, store: DraftStore, export_dir: Path | None = None) -> None:
        self.store = store
        self.export_dir = (
            Path(export_dir)
            if export_dir is not None
            else Path.cwd() / "scenario_packages"
        )

    # ------------------------------------------------------------------
    # Draft lifecycle
    # ------------------------------------------------------------------

    def list_drafts(self) -> list[Draft]:
        """Return every stored draft, newest first."""
        return self.store.list()

    def create_draft(self, kind: str = "cut_in", title: str = "") -> Draft:
        """Create a draft from a starter document.

        Args:
            kind: ``"cut_in"`` for the worked example, anything else for the
                minimal document.
            title: Optional title override.
        """
        document = (
            new_document() if kind == "cut_in" else blank_document(title=title or "")
        )
        if title:
            document.title = title
        return self.store.create(document, title=document.title)

    def require_draft(self, draft_id: str) -> Draft:
        """Return the draft, or raise.

        Raises:
            EditorError: If no draft with that id is stored.
        """
        try:
            draft = self.store.get(draft_id)
        except ValueError as exc:
            raise EditorError(str(exc)) from exc
        if draft is None:
            raise EditorError(f"No draft named {draft_id!r}.")
        return draft

    def save(self, draft: Draft) -> Draft:
        """Normalise the layout and persist *draft*."""
        draft.document.sync_layout()
        return self.store.save(draft)

    def delete_draft(self, draft_id: str) -> bool:
        """Delete a draft.  Returns whether anything was removed."""
        return self.store.delete(draft_id)

    def validate(self, draft: Draft) -> ValidationReport:
        """Return the validation report for a draft's document."""
        return validate_document(draft.document)

    # ------------------------------------------------------------------
    # Scenario metadata
    # ------------------------------------------------------------------

    def update_scenario(
        self, document: ScenarioDocument, form: Mapping[str, Any]
    ) -> None:
        """Apply the scenario-level inspector form.

        Raises:
            EditorError: If a submitted value is not usable.
        """
        if "title" in form:
            document.title = str(form["title"]).strip() or document.title
        if "description" in form:
            document.description = str(form["description"]).strip()
        if "scenario_id" in form:
            new_id = str(form["scenario_id"]).strip()
            if new_id and new_id != document.id:
                try:
                    document.id = new_id
                except Exception as exc:  # pydantic validation
                    raise EditorError(f"Scenario id: {exc}") from exc
        if "timeout_seconds" in form:
            document.timeout_seconds = _as_float(
                form["timeout_seconds"], "Timeout", document.timeout_seconds
            )
        for attribute in ("group", "name", "xodr_path", "lanelet2_path"):
            key = f"map_{attribute}"
            if key in form:
                setattr(document.map, attribute, str(form[key]).strip() or None)
        # ``group`` and ``name`` are plain strings, never None.
        document.map.group = document.map.group or "nishishinjuku"
        document.map.name = document.map.name or "Town10HD_Opt"
        if "map_no_3d_model_lanelet_ids" in form:
            from .forms import parse_int_list  # noqa: PLC0415

            parsed = parse_int_list(form["map_no_3d_model_lanelet_ids"])
            document.map.no_3d_model_lanelet_ids = (
                parsed if isinstance(parsed, list) else []
            )

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def add_entity(self, document: ScenarioDocument, kind: str = "vehicle") -> Entity:
        """Append a new entity and return it."""
        if kind == "ego" and document.ego is not None:
            raise EditorError("The scenario already has an ego entity.")
        entity_id = _unique_entity_id(document, "ego" if kind == "ego" else "npc")
        entity = Entity(
            id=entity_id,
            kind="ego" if kind == "ego" else "vehicle",
            title="Ego" if kind == "ego" else entity_id.upper(),
            spawn=SpawnSpec(lanelet_id=document.entities[0].spawn.lanelet_id)
            if document.entities
            else SpawnSpec(),
        )
        document.entities.append(entity)
        document.sync_layout()
        return entity

    def delete_entity(self, document: ScenarioDocument, entity_id: str) -> None:
        """Remove an entity, its actions, and every condition that referenced it.

        Leaving a dangling reference behind would turn a delete into a
        validation error the user did not cause, so the references go too.
        """
        entity = document.entity(entity_id)
        if entity is None:
            raise EditorError(f"No entity named {entity_id!r}.")
        document.entities.remove(entity)
        document.actions = [a for a in document.actions if a.actor != entity_id]
        for action in document.actions:
            if action.trigger is not None and _references_entity(
                action.trigger, entity_id
            ):
                action.trigger = None
        document.assertions.pass_conditions = [
            c
            for c in document.assertions.pass_conditions
            if not _references_entity(c, entity_id)
        ]
        document.assertions.fail_conditions = [
            c
            for c in document.assertions.fail_conditions
            if not _references_entity(c, entity_id)
        ]
        document.sync_layout()

    def update_entity(
        self, document: ScenarioDocument, entity_id: str, form: Mapping[str, Any]
    ) -> None:
        """Apply the entity inspector form, including its spawn definition."""
        entity = document.entity(entity_id)
        if entity is None:
            raise EditorError(f"No entity named {entity_id!r}.")

        if "title" in form:
            entity.title = str(form["title"]).strip()
        if "vehicle_type" in form:
            entity.vehicle_type = (
                str(form["vehicle_type"]).strip() or entity.vehicle_type
            )
        if "initial_speed_kmh" in form:
            entity.initial_speed_kmh = _as_float(
                form["initial_speed_kmh"], "Initial speed", entity.initial_speed_kmh
            )

        spawn = entity.spawn
        if "spawn_mode" in form:
            mode = str(form["spawn_mode"])
            if mode in ("fixed", "constraint_search"):
                spawn.mode = mode  # type: ignore[assignment]
        if "spawn_lanelet_id" in form:
            spawn.lanelet_id = _as_int(
                form["spawn_lanelet_id"], "Lanelet ID", spawn.lanelet_id
            )
        if "spawn_s_mode" in form:
            s_mode = str(form["spawn_s_mode"])
            if s_mode in ("fixed", "derived"):
                spawn.s.mode = s_mode  # type: ignore[assignment]
        if "spawn_s" in form:
            spawn.s.value = _as_float(form["spawn_s"], "Offset", spawn.s.value)

        # Only a derived offset needs a binding, and switching back to Fixed
        # leaves the old one in place: it is inert (nothing emits it) and it
        # means flipping the radio back does not lose what was configured.
        if spawn.s.mode == "derived":
            binding_type = str(form.get("binding_type") or "").strip()
            if not binding_type and spawn.s.binding is not None:
                binding_type = spawn.s.binding.type
            if not binding_type:
                binding_type = "stop_line_offset"
            spec = get_binding_spec(binding_type)
            if spec is None:
                raise EditorError(f"Unknown binding type {binding_type!r}.")
            existing = (
                spawn.s.binding.params
                if spawn.s.binding is not None and spawn.s.binding.type == binding_type
                else default_params(spec.fields)
            )
            params = dict(existing)
            params.update(_parse(spec.fields, form, prefix="binding_"))
            spawn.s.binding = BindingRef(type=binding_type, params=params)

    # ------------------------------------------------------------------
    # Spawn constraints
    # ------------------------------------------------------------------

    def add_constraint(
        self,
        document: ScenarioDocument,
        entity_id: str,
        type_id: str,
        parent_id: str | None = None,
    ) -> ConstraintNode:
        """Add a constraint to an entity's spawn search."""
        entity = document.entity(entity_id)
        if entity is None:
            raise EditorError(f"No entity named {entity_id!r}.")
        spec = get_constraint_spec(type_id)
        if spec is None:
            raise EditorError(f"Unknown constraint type {type_id!r}.")

        node = ConstraintNode(type=type_id, params=default_params(spec.fields))
        if parent_id:
            _, parent = find_constraint(document, parent_id)
            if parent is None:
                raise EditorError(f"No constraint named {parent_id!r}.")
            parent_spec = get_constraint_spec(parent.type)
            if parent_spec is None or not parent_spec.accepts_children:
                raise EditorError(f"{parent.type!r} does not take child constraints.")
            if (
                parent_spec.max_children is not None
                and len(parent.constraints) >= parent_spec.max_children
            ):
                raise EditorError(
                    f"{parent_spec.title} takes at most "
                    f"{parent_spec.max_children} child constraint(s)."
                )
            parent.constraints.append(node)
        else:
            entity.spawn.constraints.append(node)
        return node

    def update_constraint(
        self, document: ScenarioDocument, node_id: str, form: Mapping[str, Any]
    ) -> None:
        """Apply the constraint inspector form."""
        _, node = find_constraint(document, node_id)
        if node is None:
            raise EditorError(f"No constraint named {node_id!r}.")
        spec = get_constraint_spec(node.type)
        if spec is None:
            raise EditorError(f"Unknown constraint type {node.type!r}.")
        node.params.update(_parse(spec.fields, form))

    def delete_constraint(self, document: ScenarioDocument, node_id: str) -> None:
        """Remove a constraint subtree."""
        for entity in document.entities:
            if _remove_constraint(entity.spawn.constraints, node_id):
                return
        raise EditorError(f"No constraint named {node_id!r}.")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def add_action(
        self, document: ScenarioDocument, type_id: str, actor: str | None
    ) -> ActionNode:
        """Append an action to an actor's lane."""
        spec = get_action_spec(type_id)
        if spec is None:
            raise EditorError(f"Unknown action type {type_id!r}.")
        if spec.actor_required and not actor:
            raise EditorError(f"{spec.title} needs an actor.")
        if actor and document.entity(actor) is None:
            raise EditorError(f"No entity named {actor!r}.")

        action = ActionNode(
            type=type_id,
            title=spec.title,
            actor=actor or None,
            params=default_params(spec.fields),
            timing=spec.default_timing,
        )
        document.actions.append(action)
        document.sync_layout()
        return action

    def update_action(
        self, document: ScenarioDocument, action_id: str, form: Mapping[str, Any]
    ) -> None:
        """Apply the action inspector form."""
        action = document.action(action_id)
        if action is None:
            raise EditorError(f"No action named {action_id!r}.")
        spec = get_action_spec(action.type)
        if spec is None:
            raise EditorError(f"Unknown action type {action.type!r}.")

        if "title" in form:
            action.title = str(form["title"]).strip()
        if "actor" in form:
            actor = str(form["actor"]).strip()
            if actor and document.entity(actor) is None:
                raise EditorError(f"No entity named {actor!r}.")
            action.actor = actor or None
        if "timing" in form:
            timing = str(form["timing"])
            if timing in ("pre_tick", "post_tick"):
                action.timing = timing  # type: ignore[assignment]
        action.once = "once" in form
        action.params.update(_parse(spec.fields, form))

    def delete_action(self, document: ScenarioDocument, action_id: str) -> None:
        """Remove an action and its trigger."""
        action = document.action(action_id)
        if action is None:
            raise EditorError(f"No action named {action_id!r}.")
        document.actions.remove(action)
        document.ui.nodes.pop(action_id, None)
        document.sync_layout()

    def move_action(
        self, document: ScenarioDocument, action_id: str, delta: int
    ) -> None:
        """Shift an action along its lane.

        This writes ``ui.column_hint`` only -- causal progression is presentation,
        so moving a card never changes what the scenario does.
        """
        action = document.action(action_id)
        if action is None:
            raise EditorError(f"No action named {action_id!r}.")
        lane = (
            document.actions_for(action.actor)
            if action.actor
            else document.world_actions()
        )
        if action not in lane:
            return
        index = lane.index(action)
        target = max(0, min(len(lane) - 1, index + delta))
        if target == index:
            return
        lane.insert(target, lane.pop(index))
        for column, node in enumerate(lane):
            document.ui.set_column(node.id, column)

    def reorder_actors(self, document: ScenarioDocument, order: list[str]) -> None:
        """Set the swimlane order.  Presentation only."""
        known = [e.id for e in document.entities]
        document.ui.actor_order = [e for e in order if e in known]
        document.sync_layout()

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def add_condition(
        self, document: ScenarioDocument, slot: str, type_id: str
    ) -> ConditionNode:
        """Add a condition into *slot*.

        Slots are ``trigger:<action_id>`` (the action's trigger),
        ``node:<node_id>`` (a child of a composition), ``pass`` or ``fail``.
        Attaching a second condition to an action whose trigger is a single leaf
        wraps both in an ``ALL`` -- the reading the swimlane already implies.
        """
        spec = get_condition_spec(type_id)
        if spec is None:
            raise EditorError(f"Unknown condition type {type_id!r}.")
        node = ConditionNode(type=type_id, params=default_params(spec.fields))

        if slot == SLOT_PASS:
            document.assertions.pass_conditions.append(node)
            return node
        if slot == SLOT_FAIL:
            document.assertions.fail_conditions.append(node)
            return node

        target, _, identifier = slot.partition(":")
        if target == "trigger":
            action = document.action(identifier)
            if action is None:
                raise EditorError(f"No action named {identifier!r}.")
            action.trigger = _attach_trigger(action.trigger, node)
            return node
        if target == "node":
            parent = document.condition(identifier)
            if parent is None:
                raise EditorError(f"No condition named {identifier!r}.")
            parent_spec = get_condition_spec(parent.type)
            if parent_spec is None or not parent_spec.accepts_children:
                raise EditorError(f"{parent.type!r} does not take child conditions.")
            if (
                parent_spec.max_children is not None
                and len(parent.children) >= parent_spec.max_children
            ):
                raise EditorError(
                    f"{parent_spec.title} already has its "
                    f"{parent_spec.max_children} child condition(s)."
                )
            parent.children.append(node)
            return node

        raise EditorError(f"Unknown condition slot {slot!r}.")

    def update_condition(
        self, document: ScenarioDocument, node_id: str, form: Mapping[str, Any]
    ) -> None:
        """Apply the condition inspector form."""
        node = document.condition(node_id)
        if node is None:
            raise EditorError(f"No condition named {node_id!r}.")
        spec = get_condition_spec(node.type)
        if spec is None:
            raise EditorError(f"Unknown condition type {node.type!r}.")
        node.params.update(_parse(spec.fields, form))

    def delete_condition(self, document: ScenarioDocument, node_id: str) -> None:
        """Remove a condition subtree from wherever it sits."""
        for action in document.actions:
            trigger = action.trigger
            if trigger is None:
                continue
            if trigger.id == node_id:
                action.trigger = None
                return
            if trigger.remove(node_id):
                return
        for bucket in (
            document.assertions.pass_conditions,
            document.assertions.fail_conditions,
        ):
            for index, root in enumerate(bucket):
                if root.id == node_id:
                    del bucket[index]
                    return
                if root.remove(node_id):
                    return
        raise EditorError(f"No condition named {node_id!r}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(fields: Any, form: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Parse the subset of *form* that names fields, honouring an optional prefix.

    Raises:
        EditorError: If a value cannot be parsed into its declared type.
    """
    if prefix:
        stripped = {
            key[len(prefix) :]: value
            for key, value in form.items()
            if key.startswith(prefix)
        }
    else:
        stripped = dict(form)
    present = [f for f in fields if f.name in stripped or f.kind == "bool"]
    try:
        return parse_params(present, stripped)
    except ValueError as exc:
        raise EditorError(str(exc)) from exc


def _as_float(raw: Any, label: str, fallback: float) -> float:
    """Return *raw* as a float, or raise a user-facing error."""
    text = str(raw).strip()
    if not text:
        return fallback
    try:
        return float(text)
    except ValueError as exc:
        raise EditorError(f"{label} must be a number, got {raw!r}.") from exc


def _as_int(raw: Any, label: str, fallback: int) -> int:
    """Return *raw* as an int, or raise a user-facing error."""
    text = str(raw).strip()
    if not text:
        return fallback
    try:
        return int(text)
    except ValueError as exc:
        raise EditorError(f"{label} must be a whole number, got {raw!r}.") from exc


def _unique_entity_id(document: ScenarioDocument, stem: str) -> str:
    """Return an entity id based on *stem* that nothing else uses.

    Non-ego entities are always numbered (``npc1``, ``npc2``, ...) so that ids
    line up with the ``npc<N>`` CARLA role names the compiler assigns.
    """
    taken = {e.id for e in document.entities}
    if stem == "ego":
        if stem not in taken:
            return stem
    index = 1
    while f"{stem}{index}" in taken:
        index += 1
    return f"{stem}{index}"


def _references_entity(node: ConditionNode, entity_id: str) -> bool:
    """Whether any node in this subtree names *entity_id*.

    Entity-typed parameters are discovered from the spec, so this keeps working
    when a new condition introduces a differently named entity field.
    """
    for candidate in node.walk():
        spec = get_condition_spec(candidate.type)
        if spec is None:
            continue
        for field_spec in spec.fields:
            if field_spec.kind != "entity":
                continue
            if str(candidate.params.get(field_spec.name) or "") == entity_id:
                return True
    return False


def _attach_trigger(
    existing: ConditionNode | None, node: ConditionNode
) -> ConditionNode:
    """Return the action's new trigger after adding *node* to *existing*."""
    if existing is None:
        return node
    spec = get_condition_spec(existing.type)
    if spec is not None and spec.kind == "composite":
        existing.children.append(node)
        return existing
    return ConditionNode(type="all", children=[existing, node])


def find_constraint(
    document: ScenarioDocument, node_id: str
) -> tuple[str | None, ConstraintNode | None]:
    """Find a constraint by id, with the id of the entity whose spawn holds it."""
    for entity in document.entities:
        for root in entity.spawn.constraints:
            for candidate in root.walk():
                if candidate.id == node_id:
                    return entity.id, candidate
    return None, None


def _remove_constraint(nodes: list[ConstraintNode], node_id: str) -> bool:
    """Remove a constraint subtree from a forest.  Returns success."""
    for index, node in enumerate(nodes):
        if node.id == node_id:
            del nodes[index]
            return True
        if _remove_constraint(node.constraints, node_id):
            return True
    return False
