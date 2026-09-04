"""Scenario IR -- the canonical representation edited by the Scenario Editor.

The editor never edits Python.  It edits a :class:`ScenarioDocument`, a
declarative, serialisable description of a scenario that
:mod:`autoware_carla_scenario.authoring.compiler` turns into the *existing*
runtime primitives (``BaseAction`` / ``BaseCondition`` / ``BaseScenario``).

Two rules shape the model:

* **Semantics and layout are separate.**  Everything under :attr:`ScenarioDocument.ui`
  is presentation only -- ``column_hint`` and ``actor_order`` never reach the
  runtime.  Deleting the whole ``ui`` block must not change what the scenario does.
* **Primitives are named, not modelled.**  Actions, conditions, spawn constraints
  and bindings are stored as a ``type`` string plus a free-form ``params`` mapping,
  and are given meaning by the metadata in
  :mod:`autoware_carla_scenario.authoring.registry`.  Adding a primitive is a
  Python-side change; neither this module nor the HTML templates need to know
  about it.

This module deliberately has **no** CARLA or lanelet2 import, so a document can
be loaded, validated and compiled anywhere.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ActionNode",
    "Assertions",
    "BindingRef",
    "ConditionNode",
    "ConstraintNode",
    "Entity",
    "EntityKind",
    "MapRef",
    "ScenarioDocument",
    "SpawnMode",
    "SpawnSpec",
    "SValue",
    "TickTimingName",
    "UiLayout",
    "UiNode",
    "DOCUMENT_FORMAT_VERSION",
    "new_object_id",
]

#: Bumped whenever the on-disk shape of a document changes incompatibly.
DOCUMENT_FORMAT_VERSION = 1

EntityKind = Literal["ego", "vehicle"]
SpawnMode = Literal["fixed", "constraint_search"]
TickTimingName = Literal["pre_tick", "post_tick"]

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def new_object_id(prefix: str) -> str:
    """Return a short, unique, identifier-safe object id with *prefix*."""
    import uuid  # noqa: PLC0415

    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class _Node(BaseModel):
    """Base for every IR node.

    ``extra="forbid"`` makes a typo in a hand-edited YAML fail loudly rather
    than being silently dropped, and ``validate_assignment=True`` extends the
    same guarantee to the editor: a rejected value has to raise where it is
    assigned, not corrupt the draft and surface as a parse error the next time
    it is opened.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------


class ConstraintNode(_Node):
    """One node of a lanelet spawn-constraint tree.

    The shape mirrors the ``sweep.constraints`` YAML syntax consumed by
    :mod:`autoware_carla_scenario.sweeper.constraints` one-for-one, so
    :meth:`to_sweep_dict` is a pure re-shaping step -- the editor never gets
    its own constraint engine.

    Children always live in :attr:`constraints`, however many a type takes.
    The sweeper spells a unary ``not`` child as a singular ``constraint`` key,
    so that difference lives in :meth:`to_sweep_dict` alone rather than in a
    second field every walk, edit and validation would have to branch on.

    Attributes:
        type: Constraint ``type`` string (``has_adjacent``, ``and``, ``not``, ...).
        params: Leaf parameters (e.g. ``{"value": "left"}``).
        constraints: Child constraints, bounded by the type's
            :attr:`~autoware_carla_scenario.authoring.registry.ConstraintSpec.max_children`.
    """

    id: str = Field(default_factory=lambda: new_object_id("c"))
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    constraints: list[ConstraintNode] = Field(default_factory=list)

    def to_sweep_dict(self) -> dict[str, Any]:
        """Return this subtree in ``sweep.constraints`` YAML form."""
        from .registry import get_constraint_spec

        out: dict[str, Any] = {"type": self.type}
        out.update(self.params)
        if not self.constraints:
            return out
        spec = get_constraint_spec(self.type)
        if spec is not None and spec.kind == "wrapper":
            out["constraint"] = self.constraints[0].to_sweep_dict()
        else:
            out["constraints"] = [c.to_sweep_dict() for c in self.constraints]
        return out

    def walk(self) -> "list[ConstraintNode]":
        """Return this node followed by every descendant, depth first."""
        found = [self]
        for child in self.constraints:
            found.extend(child.walk())
        return found


class BindingRef(_Node):
    """Reference to a ``sweep.bindings`` entry that derives a value.

    Attributes:
        type: Binding ``type`` string (e.g. ``stop_line_offset``).
        params: Binding parameters (e.g. ``{"offset": 15.0}``).
    """

    type: str
    params: dict[str, Any] = Field(default_factory=dict)

    def to_sweep_dict(self) -> dict[str, Any]:
        """Return this binding in ``sweep.bindings`` YAML form."""
        return {"type": self.type, **self.params}


class SValue(_Node):
    """Longitudinal spawn offset, either given outright or derived from the map.

    ``mode="fixed"`` uses :attr:`value` verbatim.  ``mode="derived"`` hands the
    value to a :class:`BindingRef` (``sweep.bindings``), which resolves it
    per matched lanelet -- "15 m before the stop line" rather than "s = 12.5".
    :attr:`value` is still carried in derived mode: it is the concrete number
    written into the Hydra config as the starting point the sweeper overrides.
    """

    mode: Literal["fixed", "derived"] = "fixed"
    value: float = 0.0
    binding: Optional[BindingRef] = None


class SpawnSpec(_Node):
    """Where an entity starts.

    ``mode="fixed"`` pins :attr:`lanelet_id`.  ``mode="constraint_search"``
    leaves the lanelet to the existing lanelet-constraint sweeper: the
    :attr:`constraints` tree is emitted as ``sweep.constraints`` and
    :attr:`lanelet_id` becomes the concrete default the sweep overrides.
    """

    mode: SpawnMode = "fixed"
    lanelet_id: int = 0
    s: SValue = Field(default_factory=SValue)
    constraints: list[ConstraintNode] = Field(default_factory=list)

    def sweep_constraint_dicts(self) -> list[dict[str, Any]]:
        """Return the constraint tree in ``sweep.constraints`` YAML form."""
        if self.mode != "constraint_search":
            return []
        return [c.to_sweep_dict() for c in self.constraints]


class Entity(_Node):
    """A vehicle taking part in the scenario.

    Exactly one entity must have ``kind="ego"``; its
    :attr:`role_name` is fixed to the framework's ego role by the compiler.
    Other entities map onto ``npc<N>`` roles in registration order.
    """

    id: str
    kind: EntityKind = "vehicle"
    title: str = ""
    vehicle_type: str = "vehicle.mini.cooper"
    initial_speed_kmh: float = 0.0
    spawn: SpawnSpec = Field(default_factory=SpawnSpec)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(
                f"Entity id {value!r} must be lower_snake_case starting with a letter."
            )
        return value

    @property
    def display_name(self) -> str:
        """Human-facing name: the explicit title, else the id."""
        return self.title or self.id


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


class ConditionNode(_Node):
    """One node of a condition tree: a leaf check, or a composition of them.

    A single recursive node type covers leaves (``entity_distance``, ``ttc``,
    ``speed``, ...) and compositions (``all`` / ``any`` / ``not`` / ``sticky`` /
    ``persistent``).  Which of :attr:`params` and :attr:`children` matter is
    decided by the node's :class:`~autoware_carla_scenario.authoring.registry.ConditionSpec`,
    which is also what the templates render from -- that is what keeps the GUI
    free of per-type branching.
    """

    id: str = Field(default_factory=lambda: new_object_id("p"))
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    children: list[ConditionNode] = Field(default_factory=list)

    def walk(self) -> "list[ConditionNode]":
        """Return this node followed by every descendant, depth first."""
        found = [self]
        for child in self.children:
            found.extend(child.walk())
        return found

    def find(self, node_id: str) -> Optional["ConditionNode"]:
        """Return the descendant (or self) whose id is *node_id*."""
        for node in self.walk():
            if node.id == node_id:
                return node
        return None

    def remove(self, node_id: str) -> bool:
        """Remove the descendant identified by *node_id*.  Returns success."""
        for index, child in enumerate(self.children):
            if child.id == node_id:
                del self.children[index]
                return True
            if child.remove(node_id):
                return True
        return False


def condition_refs(node: "ConditionNode", kind: str) -> list[str]:
    """Return the ids *node* itself names through fields of *kind*, in order.

    The one place that knows how a condition points at something else.  Which
    parameters are references is the spec's business, so a newly registered
    condition is picked up without changing any caller, and the canvas, the step
    ordering and the delete path cannot come to disagree about what a reference
    is -- a disagreement that shows up as a causal arrow drawn for a dependency
    the layout rules never saw.

    Only this node is read, never its children, so a caller gets the leaf that
    actually names the id rather than the composition wrapped around it.
    """
    from .registry import get_condition_spec  # noqa: PLC0415

    spec = get_condition_spec(node.type)
    if spec is None:
        return []
    found: list[str] = []
    for field in spec.fields:
        if field.kind != kind:
            continue
        value = str(node.params.get(field.name) or "")
        if value and value not in found:
            found.append(value)
    return found


class Assertions(_Node):
    """The scenario's verdict: what makes it pass and what makes it fail."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, populate_by_name=True
    )

    pass_conditions: list[ConditionNode] = Field(default_factory=list, alias="pass")
    fail_conditions: list[ConditionNode] = Field(default_factory=list, alias="fail")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class ActionNode(_Node):
    """Something an actor does, gated by an optional trigger condition.

    There is no separate "events" concept: a trigger is simply the condition
    tree attached to the action it fires, which is what the swimlane draws
    underneath the action card.
    """

    id: str = Field(default_factory=lambda: new_object_id("a"))
    type: str
    title: str = ""
    actor: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    trigger: Optional[ConditionNode] = None
    timing: TickTimingName = "pre_tick"
    once: bool = True


# ---------------------------------------------------------------------------
# Presentation (never read by the runtime)
# ---------------------------------------------------------------------------


class UiNode(_Node):
    """Layout hints for one object on the canvas."""

    column_hint: int = 0


class UiLayout(_Node):
    """Everything the canvas needs and the runtime must ignore."""

    actor_order: list[str] = Field(default_factory=list)
    nodes: dict[str, UiNode] = Field(default_factory=dict)

    def column_of(self, object_id: str) -> int:
        """Return the stored column hint for *object_id* (0 when unset)."""
        node = self.nodes.get(object_id)
        return node.column_hint if node is not None else 0

    def set_column(self, object_id: str, column: int) -> None:
        """Store a column hint for *object_id*."""
        node = self.nodes.setdefault(object_id, UiNode())
        node.column_hint = column


# ---------------------------------------------------------------------------
# Map reference
# ---------------------------------------------------------------------------


class MapRef(_Node):
    """The map the scenario runs on.

    Mirrors the framework's ``map`` Hydra config group.  ``group`` names an
    existing built-in group (e.g. ``nishishinjuku``) so an exported package can
    just select it; the paths are carried too so the editor can load the
    Lanelet2 map for spawn previews without composing Hydra.
    """

    group: str = "nishishinjuku"
    name: str = "NishishinjukuMap"
    xodr_path: Optional[str] = None
    lanelet2_path: Optional[str] = None
    no_3d_model_lanelet_ids: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class ScenarioDocument(_Node):
    """The complete, canonical description of one authored scenario."""

    version: int = DOCUMENT_FORMAT_VERSION
    id: str = "new_scenario"
    title: str = "New scenario"
    description: str = ""
    timeout_seconds: float = 30.0
    map: MapRef = Field(default_factory=MapRef)
    entities: list[Entity] = Field(default_factory=list)
    actions: list[ActionNode] = Field(default_factory=list)
    assertions: Assertions = Field(default_factory=Assertions)
    ui: UiLayout = Field(default_factory=UiLayout)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(
                f"Scenario id {value!r} must be lower_snake_case starting with a letter."
            )
        return value

    # -- lookups --------------------------------------------------------

    def entity(self, entity_id: str) -> Optional[Entity]:
        """Return the entity with *entity_id*, or ``None``."""
        return next((e for e in self.entities if e.id == entity_id), None)

    def action(self, action_id: str) -> Optional[ActionNode]:
        """Return the action with *action_id*, or ``None``."""
        return next((a for a in self.actions if a.id == action_id), None)

    @property
    def ego(self) -> Optional[Entity]:
        """Return the ego entity, or ``None`` when the document has none."""
        return next((e for e in self.entities if e.kind == "ego"), None)

    def condition(self, node_id: str) -> Optional[ConditionNode]:
        """Return any condition node in the document by id."""
        for root in self.condition_roots():
            found = root.find(node_id)
            if found is not None:
                return found
        return None

    def condition_roots(self) -> list[ConditionNode]:
        """Return every condition tree root: action triggers and assertions."""
        roots = [a.trigger for a in self.actions if a.trigger is not None]
        roots.extend(self.assertions.pass_conditions)
        roots.extend(self.assertions.fail_conditions)
        return roots

    # -- layout ---------------------------------------------------------

    def ordered_entities(self) -> list[Entity]:
        """Return entities in the canvas' actor order, ego first by default."""
        order = {eid: i for i, eid in enumerate(self.ui.actor_order)}
        fallback = len(order)
        return sorted(
            self.entities,
            key=lambda e: (order.get(e.id, fallback), 0 if e.kind == "ego" else 1),
        )

    def actions_for(self, entity_id: str) -> list[ActionNode]:
        """Return the actions owned by *entity_id*, in column order."""
        owned = [a for a in self.actions if a.actor == entity_id]
        return sorted(owned, key=lambda a: (self.ui.column_of(a.id), a.id))

    def world_actions(self) -> list[ActionNode]:
        """Return the actions no actor owns, in column order.

        Some actions act on the environment rather than on a vehicle -- setting
        traffic lights, for instance.  They still belong on the canvas, so the
        editor gives them a lane of their own rather than hiding them.
        """
        unowned = [a for a in self.actions if not a.actor]
        return sorted(unowned, key=lambda a: (self.ui.column_of(a.id), a.id))

    def action_dependencies(self) -> dict[str, set[str]]:
        """Action id -> the ids of the actions its trigger waits on.

        Only triggers are walked: an assertion that waits on an action says
        nothing about when the action runs.
        """
        found: dict[str, set[str]] = {}
        for action in self.actions:
            if action.trigger is None:
                continue
            needs = {
                ref
                for node in action.trigger.walk()
                for ref in condition_refs(node, "action")
            }
            if needs:
                found[action.id] = needs
        return found

    def action_slots(self, entity_id: Optional[str]) -> "list[list[ActionNode]]":
        """Return the lane as one list of actions per step, empty steps included.

        A step is a *set* of actions, not one action: every action is armed from
        the first tick and fires when its own trigger says so, so actions that
        nothing sequences really do run alongside each other.  Drawing them
        stacked in one column says that; spreading them across columns would
        draw an order the runtime does not have.

        The gaps matter too.  A reaction belongs in a later step than the action
        that provokes it even when its own actor does nothing in between -- an
        ego that swerves *after* NPC1 cuts in reads as a reaction only if its
        card is further right.
        """
        lane = self.actions_for(entity_id) if entity_id else self.world_actions()
        slots: list[list[ActionNode]] = []
        for action in lane:
            column = self.ui.column_of(action.id)
            while len(slots) <= column:
                slots.append([])
            slots[column].append(action)
        return slots

    def step_count(self) -> int:
        """Return how many steps the ruler has to number.

        As long as the busiest track, or the lane that runs past the last
        numbered step would have unlabelled slots.  An actor's track carries two
        columns that are not steps of its own -- the spawn marker and the "add
        action" control -- and the world and verdict lanes carry only the
        latter.
        """
        actor_extras, other_extras = 2, 1
        widths = [
            len(self.action_slots(entity.id)) + actor_extras
            for entity in self.ordered_entities()
        ]
        widths += [
            len(self.action_slots(None)) + other_extras,
            len(self.assertions.pass_conditions) + other_extras,
            len(self.assertions.fail_conditions) + other_extras,
        ]
        return max([1, *widths])

    def enforce_dependency_order(self) -> None:
        """Push actions right until every trigger's dependencies are behind it.

        The step axis only means something if a dependency is visibly earlier
        than what waits on it: an action triggered by "the cut-in has completed"
        cannot share a step with the cut-in, because within one step nothing is
        ordered.  Actions are only ever moved *later*, so a layout is repaired
        rather than rearranged.

        A cycle -- two actions each waiting on the other, which can never fire
        at runtime either -- leaves the layout untouched for
        :mod:`.validator` to report.
        """
        dependencies = self.action_dependencies()
        if not dependencies:
            return

        columns = {a.id: self.ui.column_of(a.id) for a in self.actions}
        for _ in range(len(self.actions) + 1):
            moved = False
            for action_id, needs in dependencies.items():
                floor = max(
                    (columns[need] for need in needs if need in columns), default=-1
                )
                if columns[action_id] <= floor:
                    columns[action_id] = floor + 1
                    moved = True
            if not moved:
                break
        else:
            return

        for action_id, column in columns.items():
            if column != self.ui.column_of(action_id):
                self.ui.set_column(action_id, column)

    def sync_layout(self) -> None:
        """Make :attr:`ui` consistent with the semantic content.

        Adds newly created entities to the actor order, drops stale entries,
        and gives every action a column hint.  Purely cosmetic -- it is safe
        to call at any time and never changes scenario semantics.
        """
        known = [e.id for e in self.entities]
        self.ui.actor_order = [e for e in self.ui.actor_order if e in known]
        self.ui.actor_order.extend(e for e in known if e not in self.ui.actor_order)

        object_ids = {a.id for a in self.actions}
        self.ui.nodes = {k: v for k, v in self.ui.nodes.items() if k in object_ids}
        for owner in [*known, None]:
            for column, action in enumerate(
                [a for a in self.actions if (a.actor or None) == owner]
            ):
                if action.id not in self.ui.nodes:
                    self.ui.set_column(action.id, column)
        self.enforce_dependency_order()

    # -- serialisation --------------------------------------------------

    def to_yaml_dict(self) -> dict[str, Any]:
        """Return a plain, YAML-friendly dict (aliases applied, defaults kept)."""
        return self.model_dump(mode="json", by_alias=True)
