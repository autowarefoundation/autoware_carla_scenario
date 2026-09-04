"""Compile a :class:`ScenarioDocument` down to the existing runtime primitives.

The pipeline is::

    ScenarioDocument -> ScenarioCompiler -> BaseAction / BaseCondition -> BaseScenario

and it is deliberately split in two halves:

* **Compilation** (this module) resolves entity ids to CARLA role names, checks
  every node against its metadata spec, and coerces form strings into typed
  parameters.  It imports neither CARLA nor lanelet2, so a document can be
  compiled -- and a scenario package's tests can prove it compiles -- on a
  machine with no simulator.
* **Instantiation** (:mod:`autoware_carla_scenario.authoring.builders`) turns
  the compiled plan into live ``BaseAction`` / ``BaseCondition`` objects.  That
  half needs CARLA and only runs inside
  :meth:`~autoware_carla_scenario.declarative.DeclarativeScenario.setup`.

There is no editor-only runtime: every builder returns a class that already
existed in the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..constants import EGO_ROLE_NAME
from .models import ActionNode, ConditionNode, Entity, ScenarioDocument
from .registry import (
    INT_KINDS,
    INT_LIST_KINDS,
    TRUTHY_VALUES,
    ActionSpec,
    ConditionSpec,
    FieldSpec,
    get_action_spec,
    get_condition_spec,
)
from .validator import Issue, validate_document

__all__ = [
    "BuildContext",
    "CompilationError",
    "CompiledAction",
    "CompiledCondition",
    "CompiledScenario",
    "coerce_params",
    "compile_document",
]

#: Role name the framework reserves for the ego vehicle.  ``constants`` pulls in
#: only ``entity_role``, so taking it from there costs the editor nothing and
#: leaves one definition of the name.
EGO_ROLE = str(EGO_ROLE_NAME)


class CompilationError(Exception):
    """Raised when a document cannot be compiled.

    Attributes:
        issues: The blocking :class:`~autoware_carla_scenario.authoring.validator.Issue`
            list, so callers can render the same detail the editor shows.
    """

    def __init__(self, issues: "tuple[Issue, ...]") -> None:
        detail = "; ".join(f"{i.path}: {i.message}" for i in issues)
        super().__init__(f"Scenario document is not valid: {detail}")
        self.issues = issues


# ---------------------------------------------------------------------------
# Parameter coercion
# ---------------------------------------------------------------------------


def _coerce_one(spec: FieldSpec, value: Any) -> Any:
    """Return *value* converted to the type *spec* describes."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return spec.default

    if spec.kind in INT_KINDS:
        return int(str(value).strip())
    if spec.kind == "number":
        return float(str(value).strip())
    if spec.kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in TRUTHY_VALUES
    if spec.kind in (*INT_LIST_KINDS, "int_list_or_ref"):
        if isinstance(value, str):
            # A reference such as ${map.no_3d_model_lanelet_ids} passes through
            # untouched; the sweep YAML resolves it.
            return value
        return [int(str(v).strip()) for v in value]
    return value


def coerce_params(
    fields: "tuple[FieldSpec, ...]", params: dict[str, Any]
) -> dict[str, Any]:
    """Return *params* with every known field converted to its declared type.

    Unknown keys are dropped: form posts and hand-edited YAML both go through
    here, and a stale key must not reach a constructor as a surprise keyword.
    """
    return {spec.name: _coerce_one(spec, params.get(spec.name)) for spec in fields}


# ---------------------------------------------------------------------------
# Compiled plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledCondition:
    """A validated condition node with its spec, typed params and children."""

    spec: ConditionSpec
    node: ConditionNode
    params: dict[str, Any]
    children: "tuple[CompiledCondition, ...]" = ()

    @property
    def label(self) -> str:
        """Stable, human-readable label handed to the runtime condition."""
        return f"{self.spec.type_id}:{self.node.id}"


@dataclass(frozen=True)
class CompiledAction:
    """A validated action node with its spec, typed params and trigger."""

    spec: ActionSpec
    node: ActionNode
    params: dict[str, Any]
    actor_role: Optional[str]
    trigger: Optional[CompiledCondition] = None

    @property
    def label(self) -> str:
        """Stable, human-readable label handed to the runtime action."""
        return self.node.title or f"{self.spec.type_id}:{self.node.id}"


@dataclass(frozen=True)
class CompiledScenario:
    """The document, resolved and type-checked, ready to be instantiated.

    Attributes:
        document: The source document (kept for spawn data and map paths).
        roles: Entity id -> CARLA ``role_name``.
        actions: Compiled actions in document order.
        pass_conditions: Compiled PASS assertions.
        fail_conditions: Compiled FAIL assertions.
    """

    document: ScenarioDocument
    roles: dict[str, str]
    actions: "tuple[CompiledAction, ...]" = ()
    pass_conditions: "tuple[CompiledCondition, ...]" = ()
    fail_conditions: "tuple[CompiledCondition, ...]" = ()
    warnings: "tuple[Issue, ...]" = field(default=())

    # -- entity helpers -------------------------------------------------

    @property
    def ego(self) -> Entity:
        """The ego entity (guaranteed present by validation)."""
        ego = self.document.ego
        assert ego is not None  # noqa: S101 -- validate_document enforces this
        return ego

    @property
    def npcs(self) -> list[Entity]:
        """Every non-ego entity, in document order."""
        return [e for e in self.document.entities if e.kind != "ego"]

    def role_of(self, entity_id: str) -> str:
        """Return the CARLA ``role_name`` assigned to *entity_id*."""
        return self.roles[entity_id]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _assign_roles(document: ScenarioDocument) -> dict[str, str]:
    """Map entity ids onto CARLA role names (``Ego`` plus ``npc1``, ``npc2``...)."""
    roles: dict[str, str] = {}
    npc_index = 0
    for entity in document.entities:
        if entity.kind == "ego":
            roles[entity.id] = EGO_ROLE
        else:
            npc_index += 1
            roles[entity.id] = f"npc{npc_index}"
    return roles


def _compile_condition(node: ConditionNode, roles: dict[str, str]) -> CompiledCondition:
    """Compile one condition subtree (already validated)."""
    spec = get_condition_spec(node.type)
    assert spec is not None  # noqa: S101 -- validate_document enforces this
    params = coerce_params(spec.fields, node.params)
    # Entity references are resolved to role names here so that no builder --
    # and no runtime object -- has to know about document ids.
    for field_spec in spec.fields:
        if field_spec.kind == "entity":
            entity_id = params.get(field_spec.name)
            if entity_id is not None:
                params[field_spec.name] = roles[str(entity_id)]
    children = tuple(_compile_condition(c, roles) for c in node.children)
    return CompiledCondition(spec=spec, node=node, params=params, children=children)


def _compile_action(node: ActionNode, roles: dict[str, str]) -> CompiledAction:
    """Compile one action (already validated)."""
    spec = get_action_spec(node.type)
    assert spec is not None  # noqa: S101 -- validate_document enforces this
    params = coerce_params(spec.fields, node.params)
    actor_role = roles.get(node.actor) if node.actor else None
    trigger = _compile_condition(node.trigger, roles) if node.trigger else None
    return CompiledAction(
        spec=spec,
        node=node,
        params=params,
        actor_role=actor_role,
        trigger=trigger,
    )


def compile_document(document: ScenarioDocument) -> CompiledScenario:
    """Validate and compile *document*.

    Args:
        document: The scenario IR to compile.

    Returns:
        A :class:`CompiledScenario` whose nodes are guaranteed well-formed.

    Raises:
        CompilationError: If validation reported any error.
    """
    report = validate_document(document)
    if not report.ok:
        raise CompilationError(report.errors)

    roles = _assign_roles(document)
    return CompiledScenario(
        document=document,
        roles=roles,
        actions=tuple(_compile_action(a, roles) for a in document.actions),
        pass_conditions=tuple(
            _compile_condition(c, roles) for c in document.assertions.pass_conditions
        ),
        fail_conditions=tuple(
            _compile_condition(c, roles) for c in document.assertions.fail_conditions
        ),
        warnings=report.warnings,
    )


# ---------------------------------------------------------------------------
# Instantiation context
# ---------------------------------------------------------------------------


@dataclass
class BuildContext:
    """Everything a builder needs from the live scenario.

    ``client`` and ``world`` are typed ``Any`` on purpose: this module is
    imported without CARLA installed, so it must not name ``carla.Client``.
    """

    scenario: Any
    client: Any = None
    tm_port: int = 8000
    #: Action id -> live action, filled in as each one is instantiated.  An
    #: ``action_completed`` condition keeps this mapping rather than an action,
    #: so a trigger may name an action that is built after it.
    actions: dict[str, Any] = field(default_factory=dict)
