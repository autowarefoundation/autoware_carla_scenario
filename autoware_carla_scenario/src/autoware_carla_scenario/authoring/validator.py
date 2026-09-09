"""Validation of a :class:`ScenarioDocument` against the primitive metadata.

Validation is metadata-driven for the same reason the GUI is: every rule below
is expressed in terms of :mod:`autoware_carla_scenario.authoring.registry`
specs, so a newly registered primitive is validated without touching this file.

Like the rest of :mod:`autoware_carla_scenario.authoring`, this module imports
neither CARLA nor lanelet2 -- the editor validates on every keystroke.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from .models import (
    ActionNode,
    ConditionNode,
    ConstraintNode,
    Entity,
    ScenarioDocument,
    condition_refs,
)
from .registry import (
    INT_KINDS,
    INT_LIST_KINDS,
    ConditionSpec,
    FieldSpec,
    get_action_spec,
    get_binding_spec,
    get_condition_spec,
    get_constraint_spec,
)

__all__ = ["Issue", "Severity", "ValidationReport", "validate_document"]

Severity = Literal["error", "warning"]

#: Value accepted by ``in_set`` in place of a literal list, resolved from the
#: map config exactly as the existing sweep YAML does.
MAP_EXCLUSION_REF = "${map.no_3d_model_lanelet_ids}"


@dataclass(frozen=True)
class Issue:
    """One validation finding.

    Attributes:
        severity: ``error`` blocks compilation and export; ``warning`` does not.
        path: Dotted location inside the document (e.g. ``actions[0].trigger``).
        message: What is wrong, phrased for the person editing the scenario.
        object_id: Id of the offending object, so the canvas can highlight it.
    """

    severity: Severity
    path: str
    message: str
    object_id: Optional[str] = None


@dataclass(frozen=True)
class ValidationReport:
    """The result of validating one document."""

    issues: tuple[Issue, ...] = ()

    @property
    def errors(self) -> tuple[Issue, ...]:
        """Only the blocking issues."""
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[Issue, ...]:
        """Only the non-blocking issues."""
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        """Whether the document can be compiled and exported."""
        return not self.errors

    def summary(self) -> str:
        """Return a one-line human summary."""
        if self.ok and not self.warnings:
            return "Valid."
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)


class _Collector:
    """Accumulates issues while walking a document."""

    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def error(self, path: str, message: str, object_id: str | None = None) -> None:
        self.issues.append(Issue("error", path, message, object_id))

    def warn(self, path: str, message: str, object_id: str | None = None) -> None:
        self.issues.append(Issue("warning", path, message, object_id))


# ---------------------------------------------------------------------------
# Field checking
# ---------------------------------------------------------------------------


def _is_blank(value: Any) -> bool:
    """Return ``True`` for values a form submits when nothing was entered."""
    return value is None or (isinstance(value, str) and not value.strip())


@dataclass(frozen=True)
class _Refs:
    """Ids a node is allowed to point at.

    Both sets are complete before any node is checked, so a trigger may name an
    action declared later in the document -- an ego reaction to an NPC's
    manoeuvre is written exactly that way.
    """

    entities: set[str]
    actions: set[str]


#: Spawn constraints and offset bindings describe lanelets, never entities or
#: actions, so nothing in them may point at either.
_NO_REFS = _Refs(entities=set(), actions=set())


def _check_field(
    out: _Collector,
    path: str,
    spec: FieldSpec,
    params: dict[str, Any],
    refs: _Refs,
    object_id: str | None,
) -> None:
    """Validate one parameter against its :class:`FieldSpec`."""
    value = params.get(spec.name, spec.default)

    if _is_blank(value):
        if spec.required:
            out.error(f"{path}.{spec.name}", f"{spec.label} is required.", object_id)
        return

    if spec.kind in INT_KINDS:
        if not _coercible_int(value):
            out.error(
                f"{path}.{spec.name}",
                f"{spec.label} must be a whole number.",
                object_id,
            )
    elif spec.kind == "number":
        if not _coercible_float(value):
            out.error(
                f"{path}.{spec.name}", f"{spec.label} must be a number.", object_id
            )
    elif spec.kind == "select":
        allowed = {o.value for o in spec.options}
        if str(value) not in allowed:
            out.error(
                f"{path}.{spec.name}",
                f"{spec.label} must be one of {sorted(allowed)}, got {value!r}.",
                object_id,
            )
    elif spec.kind == "entity":
        if str(value) not in refs.entities:
            out.error(
                f"{path}.{spec.name}",
                f"{spec.label} references unknown entity {value!r}.",
                object_id,
            )
    elif spec.kind == "action":
        if str(value) not in refs.actions:
            out.error(
                f"{path}.{spec.name}",
                f"{spec.label} references unknown action {value!r}.",
                object_id,
            )
    elif spec.kind in INT_LIST_KINDS:
        if not _is_int_list(value):
            out.error(
                f"{path}.{spec.name}",
                f"{spec.label} must be a list of whole numbers.",
                object_id,
            )
    elif spec.kind == "int_list_or_ref":
        if value != MAP_EXCLUSION_REF and not _is_int_list(value):
            out.error(
                f"{path}.{spec.name}",
                f"{spec.label} must be a list of whole numbers or {MAP_EXCLUSION_REF}.",
                object_id,
            )


def _coercible_int(value: Any) -> bool:
    try:
        int(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def _coercible_float(value: Any) -> bool:
    try:
        float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def _is_int_list(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    return all(_coercible_int(v) for v in value)


def _check_unknown_params(
    out: _Collector,
    path: str,
    fields: "tuple[FieldSpec, ...]",
    params: dict[str, Any],
    object_id: str | None,
) -> None:
    """Warn about parameters no field spec claims (usually a stale rename)."""
    known = {f.name for f in fields}
    for key in params:
        if key not in known:
            out.warn(
                f"{path}.{key}",
                f"Parameter {key!r} is not used by this type and will be dropped.",
                object_id,
            )


# ---------------------------------------------------------------------------
# Node checking
# ---------------------------------------------------------------------------


def _check_condition(
    out: _Collector,
    path: str,
    node: ConditionNode,
    refs: _Refs,
    owner: str | None = None,
) -> None:
    """Validate one condition subtree.

    *owner* is the id of the action this subtree triggers, when it is a trigger
    at all.  An action whose trigger waits on its own completion can never fire,
    so that is an error rather than a scenario that quietly does nothing.
    """
    spec: ConditionSpec | None = get_condition_spec(node.type)
    if spec is None:
        out.error(path, f"Unknown condition type {node.type!r}.", node.id)
        return

    for field_spec in spec.fields:
        _check_field(out, path, field_spec, node.params, refs, node.id)
    _check_unknown_params(out, path, spec.fields, node.params, node.id)

    if owner is not None and owner in condition_refs(node, "action"):
        out.error(
            f"{path}.action",
            f"{spec.title} waits on the action it triggers, which can never fire.",
            node.id,
        )

    count = len(node.children)
    if count < spec.min_children:
        out.error(
            path,
            f"{spec.title} needs at least {spec.min_children} child condition(s), "
            f"has {count}.",
            node.id,
        )
    if spec.max_children is not None and count > spec.max_children:
        out.error(
            path,
            f"{spec.title} accepts at most {spec.max_children} child condition(s), "
            f"has {count}.",
            node.id,
        )

    for index, child in enumerate(node.children):
        _check_condition(out, f"{path}.children[{index}]", child, refs, owner)


def _check_constraint(out: _Collector, path: str, node: ConstraintNode) -> None:
    """Validate one spawn-constraint subtree."""
    spec = get_constraint_spec(node.type)
    if spec is None:
        out.error(path, f"Unknown constraint type {node.type!r}.", node.id)
        return

    for field_spec in spec.fields:
        _check_field(out, path, field_spec, node.params, _NO_REFS, node.id)
    _check_unknown_params(out, path, spec.fields, node.params, node.id)

    count = len(node.constraints)
    limit = spec.max_children
    if limit == 0:
        if count:
            out.error(path, f"{spec.title} does not take child constraints.", node.id)
    elif not count:
        out.error(path, f"{spec.title} needs at least one child constraint.", node.id)
    elif limit is not None and count > limit:
        out.error(
            path, f"{spec.title} takes at most {limit} child constraint(s).", node.id
        )

    for index, child in enumerate(node.constraints):
        _check_constraint(out, f"{path}.constraints[{index}]", child)


def _check_action(out: _Collector, path: str, node: ActionNode, refs: _Refs) -> None:
    """Validate one action and its trigger."""
    spec = get_action_spec(node.type)
    if spec is None:
        out.error(path, f"Unknown action type {node.type!r}.", node.id)
        return

    if spec.actor_required:
        if _is_blank(node.actor):
            out.error(f"{path}.actor", f"{spec.title} needs an actor.", node.id)
        elif node.actor not in refs.entities:
            out.error(
                f"{path}.actor",
                f"{spec.title} references unknown entity {node.actor!r}.",
                node.id,
            )
    elif node.actor is not None and node.actor not in refs.entities:
        out.error(
            f"{path}.actor",
            f"Action references unknown entity {node.actor!r}.",
            node.id,
        )

    for field_spec in spec.fields:
        _check_field(out, path, field_spec, node.params, refs, node.id)
    _check_unknown_params(out, path, spec.fields, node.params, node.id)

    if node.trigger is not None:
        _check_condition(out, f"{path}.trigger", node.trigger, refs, node.id)


def _check_entity(out: _Collector, path: str, entity: Entity) -> None:
    """Validate one entity, its spawn definition and its goal."""
    _check_goal(out, path, entity)
    spawn = entity.spawn
    if spawn.mode == "fixed":
        if spawn.lanelet_id <= 0:
            out.error(
                f"{path}.spawn.lanelet_id",
                "A fixed spawn needs a positive lanelet ID.",
                entity.id,
            )
    else:
        if not spawn.constraints:
            out.error(
                f"{path}.spawn.constraints",
                "A constraint search needs at least one constraint.",
                entity.id,
            )
        if spawn.lanelet_id <= 0:
            out.warn(
                f"{path}.spawn.lanelet_id",
                "No default lanelet ID: the scenario cannot run without a sweep.",
                entity.id,
            )
        for index, constraint in enumerate(spawn.constraints):
            _check_constraint(out, f"{path}.spawn.constraints[{index}]", constraint)

    if spawn.s.mode == "derived":
        binding = spawn.s.binding
        if binding is None:
            out.error(
                f"{path}.spawn.s.binding",
                "A derived offset needs a binding.",
                entity.id,
            )
        else:
            binding_spec = get_binding_spec(binding.type)
            if binding_spec is None:
                out.error(
                    f"{path}.spawn.s.binding",
                    f"Unknown binding type {binding.type!r}.",
                    entity.id,
                )
            else:
                for field_spec in binding_spec.fields:
                    _check_field(
                        out,
                        f"{path}.spawn.s.binding",
                        field_spec,
                        binding.params,
                        _NO_REFS,
                        entity.id,
                    )


def _check_goal(out: _Collector, path: str, entity: Entity) -> None:
    """Check the entity's goal: the ego may have one, and an Autoware ego must.

    A goal is read by a vehicle that plans its own route, and the ego is the
    only one that can have such a stack behind it -- the export renders the
    ego's goal and nothing else.  A goal stored on another vehicle would
    therefore be dropped in silence, so it is reported instead.

    Whether the ego needs one follows what drives it.  Autoware localizes at the
    spawn, plans a route to the goal and only then engages, so an ``autoware``
    ego without a goal never moves and the document is wrong.  An ego the
    TrafficManager drives reads no goal at all, and a scenario about what
    happens on the way -- a cut-in, a red light -- may legitimately name no
    destination.
    """
    if entity.kind != "ego":
        if entity.goal is not None:
            out.error(
                f"{path}.goal",
                "Only the ego is routed to a goal; this vehicle would ignore "
                "it. Use a Set Goal card if something should act on it during "
                "the run.",
                entity.id,
            )
        return
    if entity.goal is None:
        if entity.driven_by == "autoware":
            out.error(
                f"{path}.goal",
                "An Autoware ego has no goal. Pick one in the Goal section of "
                "its inspector: Autoware plans its route to it and will not "
                "move without one.",
                entity.id,
            )
    elif entity.goal.lanelet_id <= 0:
        out.error(
            f"{path}.goal.lanelet_id",
            "A goal needs a positive lanelet ID.",
            entity.id,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _check_untriggered_actions(out: _Collector, document: ScenarioDocument) -> None:
    """Warn about an action drawn late that in fact fires on the first tick.

    An action with no trigger gets ``AlwaysTrueCondition``, and every action is
    armed from the first tick -- so a card in step 5 with an empty trigger runs
    immediately, while its position says otherwise.  The canvas writes "fires
    immediately" under the card, but the step number is louder.

    Only actions drawn past the first step are worth saying this about: one in
    step 1 with no trigger is doing exactly what it looks like.  An init action
    is not drawn in a step at all -- its phase runs once, before the loop -- so
    firing without a trigger is what it is for, not a discrepancy.
    """
    for index, action in enumerate(document.actions):
        if action.trigger is not None or action.phase == "init":
            continue
        column = document.ui.column_of(action.id)
        if column <= 0:
            continue
        out.warn(
            f"actions[{index}].trigger",
            f"{action.title or action.type!r} has no trigger, so it fires on the "
            f"first tick even though it is drawn in step {column + 1}. Give it a "
            f"trigger, or move it to step 1.",
            action.id,
        )


def _check_init_triggers(out: _Collector, document: ScenarioDocument) -> None:
    """An init action must not be gated by a trigger.

    An error and not a warning: `run_init` evaluates a trigger once, at elapsed
    0.0, so an action whose trigger is not already true at that instant never
    runs at all.  A document that says "do this during initialization, but only
    once X" describes something the phase cannot do, and exporting it would
    ship a scenario with a step that quietly does not happen.
    """
    for index, action in enumerate(document.actions):
        if action.trigger is None or action.takes_trigger:
            continue
        out.error(
            f"actions[{index}].trigger",
            f"{action.title or action.type!r} is in the initialization phase, "
            "which runs once and begins immediately, so it cannot wait for a "
            "condition. Remove the trigger, or move the action onto the tick "
            "loop.",
            action.id,
        )


def _check_duplicate_ego_routing(out: _Collector, document: ScenarioDocument) -> None:
    """Warn when an init Set Goal card repeats the goal the ego already carries.

    The ego is routed to :attr:`~...models.Entity.goal` during initialization
    whether or not a card says so, so an init ``routing`` card owned by the ego
    sends it a second destination in the same phase, and which of the two
    survives is the order the two are registered in.  A warning rather than an
    error: the run is well defined, it just says one thing twice.
    """
    ego = document.ego
    if ego is None or ego.goal is None:
        return
    for index, action in enumerate(document.actions):
        if action.type != "routing" or action.actor != ego.id or action.phase != "init":
            continue
        out.warn(
            f"actions[{index}].phase",
            f"{action.title or 'Set Goal'} routes the ego during initialization, "
            "where its own goal is already delivered. Move this card onto the "
            "tick loop to change the destination mid-run.",
            action.id,
        )


def _check_step_order(out: _Collector, document: ScenarioDocument) -> None:
    """Warn when a stored layout draws an order the runtime cannot honour.

    A step is a set of actions that nothing orders -- they are all armed from
    the first tick -- so an action level with, or ahead of, its own dependency
    is a picture of a sequence that does not exist.  The editor repairs this on
    every save; this catches a hand-edited ``ui`` block that says otherwise.

    A **warning**, never an error, and only for actions whose column the
    document actually states.  ``ui`` is presentation: deleting the block must
    not change what the scenario does, and :func:`compile_document` and
    :func:`export_package` both validate without normalising the layout first,
    so an error here would stop a perfectly good scenario from being exported
    for saying nothing about where to draw it.
    """
    stated = {
        action.id: document.ui.nodes[action.id].column_hint
        for action in document.actions
        if action.id in document.ui.nodes
    }
    titles = {a.id: a.title or a.type for a in document.actions}
    for action_id, needs in document.action_dependencies().items():
        if action_id not in stated:
            continue
        for need in sorted(needs):
            if need not in stated or stated[action_id] > stated[need]:
                continue
            out.warn(
                f"ui.nodes.{action_id}.column_hint",
                f"{titles[action_id]!r} waits on {titles[need]!r} but is drawn "
                f"in step {stated[action_id] + 1}, not after step "
                f"{stated[need] + 1}. Nothing inside a step is ordered; the "
                f"canvas repairs this on the next save.",
                action_id,
            )


def validate_document(document: ScenarioDocument) -> ValidationReport:
    """Return every problem found in *document*.

    Errors block compilation and package export; warnings are advisory (an
    incomplete draft is still saveable).
    """
    out = _Collector()

    entity_ids: set[str] = set()
    for index, entity in enumerate(document.entities):
        path = f"entities[{index}]"
        if entity.id in entity_ids:
            out.error(path, f"Duplicate entity id {entity.id!r}.", entity.id)
        entity_ids.add(entity.id)
        _check_entity(out, path, entity)

    egos = [e for e in document.entities if e.kind == "ego"]
    if not egos:
        out.error("entities", "The scenario needs exactly one ego entity.")
    elif len(egos) > 1:
        out.error(
            "entities",
            f"Only one ego entity is allowed, found {len(egos)}: "
            f"{[e.id for e in egos]}.",
        )

    # Every id is gathered before anything is checked, so a forward reference
    # to an action declared further down is not a validation error.
    refs = _Refs(entities=entity_ids, actions={a.id for a in document.actions})

    seen_actions: set[str] = set()
    for index, action in enumerate(document.actions):
        path = f"actions[{index}]"
        if action.id in seen_actions:
            out.error(path, f"Duplicate action id {action.id!r}.", action.id)
        seen_actions.add(action.id)
        _check_action(out, path, action, refs)

    _check_step_order(out, document)
    _check_untriggered_actions(out, document)
    _check_init_triggers(out, document)
    _check_duplicate_ego_routing(out, document)

    for index, condition in enumerate(document.assertions.pass_conditions):
        _check_condition(out, f"assertions.pass[{index}]", condition, refs)
    for index, condition in enumerate(document.assertions.fail_conditions):
        _check_condition(out, f"assertions.fail[{index}]", condition, refs)

    if not document.assertions.pass_conditions:
        out.error(
            "assertions.pass",
            "The scenario needs at least one PASS condition, or it can never succeed.",
        )
    if not document.assertions.fail_conditions:
        out.warn(
            "assertions.fail",
            "No FAIL condition: without a timeout the scenario can run forever.",
        )

    if document.timeout_seconds <= 0:
        out.error("timeout_seconds", "The scenario timeout must be positive.")

    _check_sweep_shape(out, document)

    return ValidationReport(issues=tuple(out.issues))


def _check_sweep_shape(out: _Collector, document: ScenarioDocument) -> None:
    """Check the document against the lanelet-constraint sweeper's limits.

    The sweeper enumerates one target key per run, so a scenario can search for
    at most one entity's spawn lanelet, and only that entity's offset can be
    derived from a binding.  Both are warnings rather than errors: the scenario
    still runs, it just runs with the extra entities pinned to their defaults.
    """
    searching = [e for e in document.entities if e.spawn.mode == "constraint_search"]
    for entity in searching[1:]:
        out.warn(
            f"entities[{document.entities.index(entity)}].spawn",
            "The sweeper searches one entity's spawn per run; "
            f"{searching[0].id!r} is used and {entity.id!r} keeps its default "
            "lanelet.",
            entity.id,
        )

    swept = searching[0] if searching else None
    for entity in document.entities:
        if entity.spawn.s.mode != "derived":
            continue
        if swept is None or entity.id != swept.id:
            out.warn(
                f"entities[{document.entities.index(entity)}].spawn.s",
                f"A derived offset only resolves for the searched entity; "
                f"{entity.id!r} keeps its fixed value of {entity.spawn.s.value}.",
                entity.id,
            )
