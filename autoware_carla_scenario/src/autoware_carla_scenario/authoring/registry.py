"""Metadata that gives the Scenario IR its meaning -- and drives the whole GUI.

Actions, conditions, spawn constraints and bindings all grow over time.  If the
editor's templates branched on ``type`` the HTML would have to grow with them,
so instead every primitive is described *here*, in Python, by a small spec
object.  The templates render specs, not types: adding a primitive means adding
one entry below (and a builder in :mod:`autoware_carla_scenario.authoring.builders`),
with no template change at all.

Each spec carries three things:

* how to **present** it -- title, category and, for conditions, a
  :class:`ConditionVisual` saying which parameters read as *subject*, *target*,
  *metric* and *value* so the swimlane can render ``NPC1 -> Ego | Distance | < 20 m``;
* how to **edit** it -- an ordered tuple of :class:`FieldSpec` the inspector
  turns into form controls;
* how to **build** it -- the name of a factory in
  :mod:`autoware_carla_scenario.authoring.builders` that returns the *existing*
  runtime object.

Importing this module must stay free of CARLA and lanelet2: it is imported by
the editor process, which never talks to a simulator.
"""

from __future__ import annotations

import re

# Free to import: `action_state` holds nothing but the enum, which is why it
# is a top-level module. Spelling these values out again -- as the comparison
# and traffic-light mirrors below have to, because their sources drag in CARLA
# -- would let a renamed state leave a dead option in the editor that only
# fails when the scenario is built.
from ..action_state import ActionState
from dataclasses import dataclass
from typing import Any, Literal, Optional

__all__ = [
    "ActionSpec",
    "BindingSpec",
    "ConditionSpec",
    "ConstraintSpec",
    "FieldKind",
    "INT_KINDS",
    "INT_LIST_KINDS",
    "FieldSpec",
    "ConditionVisual",
    "REFERENCE_PATTERN",
    "SelectOption",
    "COMPARISON_RULES",
    "TRAFFIC_LIGHT_STATES",
    "TRUTHY_VALUES",
    "action_specs",
    "binding_specs",
    "condition_specs",
    "constraint_specs",
    "get_action_spec",
    "get_binding_spec",
    "get_condition_spec",
    "get_constraint_spec",
    "register_action_spec",
    "register_binding_spec",
    "register_condition_spec",
    "register_constraint_spec",
]

#: Form control the inspector renders for a field.  The template switches on
#: this closed set -- never on the primitive's ``type`` -- so the set only grows
#: when a genuinely new *kind of input* is needed.
FieldKind = Literal[
    "text",
    "int",
    "number",
    "bool",
    "select",
    "entity",
    # Ids of Lanelet2 primitives, singly or as a set.  They behave as an ``int``
    # and an ``int_list`` everywhere except the inspector, which replaces the
    # text box with the map: an id is not something anyone knows by heart, and
    # picking one off the drawing is the only way to choose it that does not
    # involve a second window.
    "lanelet",
    "lanelet_list",
    # Names another action in the document by id.  The only field kind that
    # makes an edge *between* two nodes of the AST rather than describing one,
    # which is what lets "after NPC1 finished cutting in" be a fact the document
    # states rather than something the canvas infers from card positions.
    "action",
    "int_list",
    "int_list_or_ref",
]

#: Field kinds that hold one whole number, and those that hold a list of them.
#:
#: Named here because the compiler, the form parser and the validator each have
#: to branch on the same grouping: adding a kind meant editing three inline
#: tuples in lockstep, and missing one let an un-coerced string reach a builder,
#: or left the field unchecked entirely, with nothing to say which difference
#: was deliberate.
INT_KINDS: tuple[FieldKind, ...] = ("int", "lanelet")
INT_LIST_KINDS: tuple[FieldKind, ...] = ("int_list", "lanelet_list")

#: Values a checked HTML checkbox may submit, and the strings a hand-edited
#: document may spell a ``bool`` field with.  One set, so form parsing and
#: document coercion cannot drift on what counts as true.
TRUTHY_VALUES = frozenset({"1", "true", "on", "yes"})

#: An OmegaConf interpolation such as ``${map.no_3d_model_lanelet_ids}``.  A
#: field whose kind ends in ``_or_ref`` keeps one of these verbatim so the
#: composed YAML resolves it at run time.
REFERENCE_PATTERN = re.compile(r"^\$\{[^}]+\}$")


#: Condition node shapes.  ``leaf`` takes no children, ``composite`` takes many,
#: ``wrapper`` takes exactly one.
ConditionKind = Literal["leaf", "composite", "wrapper"]

#: Constraint node shapes, mirroring the sweeper's parser.
ConstraintKind = Literal["leaf", "composite", "wrapper"]


@dataclass(frozen=True)
class SelectOption:
    """One choice of a ``select`` field."""

    value: str
    label: str


@dataclass(frozen=True)
class FieldSpec:
    """One editable parameter of a primitive.

    Attributes:
        name: Key inside the node's ``params`` mapping.
        label: Human-facing label shown in the inspector.
        kind: Which control to render (see :data:`FieldKind`).
        default: Value used when the parameter is absent.
        options: Choices for ``select`` fields.
        unit: Unit suffix shown after the control (e.g. ``m``, ``s``).
        help: One-line hint rendered under the control.
        required: Whether the compiler rejects a missing/empty value.
    """

    name: str
    label: str
    kind: FieldKind = "text"
    default: Any = None
    options: tuple[SelectOption, ...] = ()
    unit: str = ""
    help: str = ""
    required: bool = True


@dataclass(frozen=True)
class ConditionVisual:
    """How a condition reads as ``subject -> target | metric | rule value``.

    ``subject`` / ``target`` / ``value`` name *fields*; the canvas substitutes
    the node's current parameter values.  ``value_label`` supplies constant
    text for conditions whose right-hand side is not a number (``inside``,
    ``occurred``).  ``rule`` names the field holding the comparison operator.

    ``target_prefix`` says what a target that is not an entity *is*.  A bare
    number on a card is unreadable in a framework that speaks two coordinate
    systems: ``NPC1 -> 183`` could be a Lanelet2 lanelet or an OpenDRIVE road,
    and those are different things that happen to be written the same way.

    ``detail`` names a further field to show beside the metric, labelled by its
    own :class:`FieldSpec`.  It is for the part of a condition that would
    otherwise be invisible on the canvas -- two conditions differing only in an
    unshown parameter must not render identically.
    """

    metric: str
    subject: Optional[str] = None
    target: Optional[str] = None
    target_prefix: str = ""
    rule: Optional[str] = None
    value: Optional[str] = None
    value_label: str = ""
    unit: str = ""
    detail: Optional[str] = None


@dataclass(frozen=True)
class ActionSpec:
    """Everything the editor and compiler need to know about one action type."""

    type_id: str
    title: str
    category: str
    builder: str
    fields: tuple[FieldSpec, ...] = ()
    actor_required: bool = True
    #: ``instant`` renders as a diamond on the swimlane, ``continuous`` as a bar.
    visual_kind: Literal["instant", "continuous"] = "instant"
    default_timing: Literal["pre_tick", "post_tick"] = "pre_tick"
    description: str = ""


@dataclass(frozen=True)
class ConditionSpec:
    """Everything the editor and compiler need to know about one condition type."""

    type_id: str
    title: str
    category: str
    builder: str
    visual: ConditionVisual
    fields: tuple[FieldSpec, ...] = ()
    kind: ConditionKind = "leaf"
    min_children: int = 0
    max_children: Optional[int] = 0
    description: str = ""

    @property
    def accepts_children(self) -> bool:
        """Whether the editor should offer an *Add condition* control."""
        return self.max_children is None or self.max_children > 0


@dataclass(frozen=True)
class ConstraintSpec:
    """One lanelet spawn-constraint type, mirroring the sweeper's parser."""

    type_id: str
    title: str
    category: str
    fields: tuple[FieldSpec, ...] = ()
    kind: ConstraintKind = "leaf"
    description: str = ""

    @property
    def max_children(self) -> Optional[int]:
        """How many child constraints this type takes (``None`` for any)."""
        if self.kind == "composite":
            return None
        return 1 if self.kind == "wrapper" else 0

    @property
    def accepts_children(self) -> bool:
        """Whether this constraint nests other constraints."""
        return self.kind in ("composite", "wrapper")


@dataclass(frozen=True)
class BindingSpec:
    """One ``sweep.bindings`` type that derives a value from the matched lanelet."""

    type_id: str
    title: str
    fields: tuple[FieldSpec, ...] = ()
    description: str = ""


# ---------------------------------------------------------------------------
# Shared option sets
# ---------------------------------------------------------------------------

#: Mirrors ``ComparisonRule``.  Spelled out rather than imported because
#: importing ``autoware_carla_scenario.conditions`` pulls in CARLA, and the
#: editor process has no simulator.  ``test_authoring_registry`` asserts the two
#: stay in step.
COMPARISON_RULES: tuple[SelectOption, ...] = (
    SelectOption("less_than", "<"),
    SelectOption("less_than_or_equal", "<="),
    SelectOption("greater_than", ">"),
    SelectOption("greater_than_or_equal", ">="),
    SelectOption("equal_to", "="),
)

#: Mirrors ``carla.TrafficLightState`` (same reason as above).
TRAFFIC_LIGHT_STATES: tuple[SelectOption, ...] = (
    SelectOption("Green", "Green"),
    SelectOption("Yellow", "Yellow"),
    SelectOption("Red", "Red"),
    SelectOption("Off", "Off"),
    SelectOption("Unknown", "Unknown"),
)

_DIRECTIONS: tuple[SelectOption, ...] = (
    SelectOption("left", "Left"),
    SelectOption("right", "Right"),
)

_RULE_FIELD = FieldSpec(
    name="rule",
    label="Predicate",
    kind="select",
    default="less_than",
    options=COMPARISON_RULES,
)


def _entity_field(name: str, label: str) -> FieldSpec:
    """Return an entity-reference field; options come from the document."""
    return FieldSpec(name=name, label=label, kind="entity", default=None)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_ACTION_SPECS: dict[str, ActionSpec] = {}
_CONDITION_SPECS: dict[str, ConditionSpec] = {}
_CONSTRAINT_SPECS: dict[str, ConstraintSpec] = {}
_BINDING_SPECS: dict[str, BindingSpec] = {}


def register_action_spec(spec: ActionSpec) -> None:
    """Make an action type available to the editor and the compiler."""
    _ACTION_SPECS[spec.type_id] = spec


def register_condition_spec(spec: ConditionSpec) -> None:
    """Make a condition type available to the editor and the compiler."""
    _CONDITION_SPECS[spec.type_id] = spec


def register_constraint_spec(spec: ConstraintSpec) -> None:
    """Make a spawn-constraint type available to the constraint builder."""
    _CONSTRAINT_SPECS[spec.type_id] = spec


def register_binding_spec(spec: BindingSpec) -> None:
    """Make a spawn binding available to the spawn editor."""
    _BINDING_SPECS[spec.type_id] = spec


def action_specs() -> list[ActionSpec]:
    """Return every registered action spec, ordered by category then title."""
    return sorted(_ACTION_SPECS.values(), key=lambda s: (s.category, s.title))


def condition_specs() -> list[ConditionSpec]:
    """Return every registered condition spec, ordered by category then title."""
    return sorted(_CONDITION_SPECS.values(), key=lambda s: (s.category, s.title))


def constraint_specs() -> list[ConstraintSpec]:
    """Return every registered constraint spec, ordered by category then title."""
    return sorted(_CONSTRAINT_SPECS.values(), key=lambda s: (s.category, s.title))


def binding_specs() -> list[BindingSpec]:
    """Return every registered binding spec, ordered by title."""
    return sorted(_BINDING_SPECS.values(), key=lambda s: s.title)


def get_action_spec(type_id: str) -> Optional[ActionSpec]:
    """Return the action spec for *type_id*, or ``None``."""
    return _ACTION_SPECS.get(type_id)


def get_condition_spec(type_id: str) -> Optional[ConditionSpec]:
    """Return the condition spec for *type_id*, or ``None``."""
    return _CONDITION_SPECS.get(type_id)


def get_constraint_spec(type_id: str) -> Optional[ConstraintSpec]:
    """Return the constraint spec for *type_id*, or ``None``."""
    return _CONSTRAINT_SPECS.get(type_id)


def get_binding_spec(type_id: str) -> Optional[BindingSpec]:
    """Return the binding spec for *type_id*, or ``None``."""
    return _BINDING_SPECS.get(type_id)


def default_params(fields: "tuple[FieldSpec, ...]") -> dict[str, Any]:
    """Return a fresh params mapping seeded with each field's default."""
    return {f.name: f.default for f in fields}


# ---------------------------------------------------------------------------
# Built-in actions
# ---------------------------------------------------------------------------

register_action_spec(
    ActionSpec(
        type_id="lane_change",
        title="Lane Change",
        category="Vehicle / Motion",
        builder="build_lane_change_action",
        visual_kind="instant",
        fields=(
            FieldSpec(
                name="direction",
                label="Direction",
                kind="select",
                default="left",
                options=_DIRECTIONS,
            ),
        ),
        description="Force a lane change through the CARLA TrafficManager.",
    )
)

register_action_spec(
    ActionSpec(
        type_id="turn",
        title="Turn at Junction",
        category="Vehicle / Motion",
        builder="build_turn_action",
        visual_kind="instant",
        fields=(
            FieldSpec(
                name="direction",
                label="Direction",
                kind="select",
                default="left",
                options=_DIRECTIONS,
            ),
            FieldSpec(
                name="search_distance",
                label="Search distance",
                kind="number",
                default=200.0,
                unit="m",
                required=False,
                help="How far ahead to look for the next junction.",
            ),
        ),
        description="Route the actor through the next junction in a direction.",
    )
)

register_action_spec(
    ActionSpec(
        type_id="traffic_signal",
        title="Set Traffic Signal",
        category="Environment",
        builder="build_traffic_signal_action",
        actor_required=False,
        visual_kind="instant",
        fields=(
            FieldSpec(
                name="state",
                label="State",
                kind="select",
                default="Green",
                options=TRAFFIC_LIGHT_STATES,
            ),
            FieldSpec(
                name="target",
                label="Target",
                kind="select",
                default="all",
                options=(
                    SelectOption("all", "All traffic lights"),
                    SelectOption("ids", "Selected Lanelet2 IDs"),
                ),
            ),
            FieldSpec(
                name="lanelet2_traffic_light_ids",
                label="Lanelet2 regulatory element IDs",
                kind="int_list",
                default=[],
                required=False,
                help="Only used when Target is 'Selected Lanelet2 IDs'.",
            ),
            FieldSpec(
                name="freeze",
                label="Freeze lights",
                kind="bool",
                default=True,
                required=False,
                help="Stop the TrafficManager from overriding the state.",
            ),
        ),
        description="Set traffic lights by Lanelet2 regulatory element ID.",
    )
)


# ---------------------------------------------------------------------------
# Built-in conditions -- compositions
# ---------------------------------------------------------------------------

register_condition_spec(
    ConditionSpec(
        type_id="all",
        title="ALL",
        category="Composition",
        builder="build_and_condition",
        kind="composite",
        visual=ConditionVisual(metric="ALL"),
        min_children=2,
        max_children=None,
        description="Every child condition must hold (AndCondition).",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="any",
        title="ANY",
        category="Composition",
        builder="build_or_condition",
        kind="composite",
        visual=ConditionVisual(metric="ANY"),
        min_children=2,
        max_children=None,
        description="At least one child condition must hold (OrCondition).",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="not",
        title="NOT",
        category="Composition",
        builder="build_not_condition",
        kind="wrapper",
        visual=ConditionVisual(metric="NOT"),
        min_children=1,
        max_children=1,
        description="Invert the child condition (NotCondition).",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="sticky",
        title="Sticky",
        category="Composition",
        builder="build_sticky_condition",
        kind="wrapper",
        visual=ConditionVisual(metric="Sticky"),
        min_children=1,
        max_children=1,
        description="Latch the child once it has held (StickyCondition).",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="persistent",
        title="Persistent",
        category="Composition",
        builder="build_persistent_condition",
        kind="wrapper",
        visual=ConditionVisual(metric="Held for", value="duration", unit="s"),
        fields=(
            FieldSpec(
                name="duration",
                label="Duration",
                kind="number",
                default=1.0,
                unit="s",
            ),
        ),
        min_children=1,
        max_children=1,
        description="The child must hold continuously (PersistentCondition).",
    )
)


# ---------------------------------------------------------------------------
# Built-in conditions -- relational
# ---------------------------------------------------------------------------

register_condition_spec(
    ConditionSpec(
        type_id="entity_distance",
        title="Distance",
        category="Relative",
        builder="build_entity_distance_condition",
        visual=ConditionVisual(
            metric="Distance",
            subject="source",
            target="target",
            rule="rule",
            value="distance",
            unit="m",
        ),
        fields=(
            _entity_field("source", "Subject"),
            _entity_field("target", "Target"),
            _RULE_FIELD,
            FieldSpec(
                name="distance", label="Distance", kind="number", default=20.0, unit="m"
            ),
        ),
        description="Distance from the subject to the target.",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="ttc",
        title="TTC",
        category="Relative",
        builder="build_ttc_condition",
        visual=ConditionVisual(
            metric="TTC",
            subject="source",
            target="target",
            rule="rule",
            value="seconds",
            unit="s",
        ),
        fields=(
            _entity_field("source", "Subject"),
            _entity_field("target", "Target"),
            _RULE_FIELD,
            FieldSpec(
                name="seconds", label="TTC", kind="number", default=4.0, unit="s"
            ),
        ),
        description="Time to collision from the subject to the target.",
    )
)


# ---------------------------------------------------------------------------
# Built-in conditions -- single entity
# ---------------------------------------------------------------------------

register_condition_spec(
    ConditionSpec(
        type_id="speed",
        title="Speed",
        category="Entity",
        builder="build_speed_condition",
        visual=ConditionVisual(
            metric="Speed",
            subject="entity",
            rule="rule",
            value="value",
            unit="m/s",
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="rule",
                label="Predicate",
                kind="select",
                default="greater_than",
                options=COMPARISON_RULES,
            ),
            FieldSpec(
                name="value", label="Speed", kind="number", default=10.0, unit="m/s"
            ),
            FieldSpec(
                name="direction",
                label="Component",
                kind="select",
                default="MAGNITUDE",
                options=(
                    SelectOption("MAGNITUDE", "Magnitude"),
                    SelectOption("LONGITUDINAL", "Longitudinal"),
                    SelectOption("LATERAL", "Lateral"),
                ),
                required=False,
            ),
        ),
        description="Speed of a single entity.",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="standstill",
        title="Standstill",
        category="Entity",
        builder="build_standstill_condition",
        visual=ConditionVisual(
            metric="Standstill", subject="entity", value="duration", unit="s"
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="duration", label="Duration", kind="number", default=1.0, unit="s"
            ),
            FieldSpec(
                name="speed_threshold",
                label="Speed threshold",
                kind="number",
                default=0.1,
                unit="m/s",
                required=False,
            ),
        ),
        description="The entity stays (almost) stopped for a duration.",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="entity_lane_position",
        title="Position (Lanelet2)",
        category="Entity",
        builder="build_entity_lane_position_condition",
        visual=ConditionVisual(
            metric="Position",
            subject="entity",
            target="lanelet_id",
            target_prefix="Lanelet",
            value_label="inside",
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="lanelet_id",
                label="Lanelet",
                kind="lanelet",
                default=0,
                help=(
                    "Resolved to its OpenDRIVE road *and lane* at scenario "
                    "setup. A lanelet already names one lane, so there is "
                    "nothing further to pin -- use Position (OpenDRIVE) to "
                    "address a road directly."
                ),
            ),
        ),
        description=(
            "The entity is on the lane a Lanelet2 lanelet describes.  Pick this "
            "when you are thinking in lanelets; everything else on the canvas "
            "already is."
        ),
    )
)

# A separate condition rather than a coordinate-system switch on the one above:
# the two frames need different fields, and offering both at once is what let a
# Lanelet2 lanelet and an OpenDRIVE lane be set to contradict each other.
register_condition_spec(
    ConditionSpec(
        type_id="entity_road_position",
        title="Position (OpenDRIVE)",
        category="Entity",
        builder="build_entity_road_position_condition",
        visual=ConditionVisual(
            metric="Position",
            subject="entity",
            target="road_id",
            target_prefix="Road",
            detail="lane_id",
            value_label="inside",
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="road_id",
                label="Road",
                kind="text",
                default="",
                help="OpenDRIVE road ID, as written in the .xodr.",
            ),
            FieldSpec(
                name="lane_id",
                label="Lane",
                kind="int",
                default=None,
                required=False,
                help=(
                    "A road carries several lanes, so a road on its own does "
                    "not name a place: leave this empty only when any lane of "
                    "the road will do."
                ),
            ),
        ),
        description=(
            "The entity is on an OpenDRIVE road, optionally on one of its "
            "lanes.  Road and lane together are what address a lane uniquely."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="entity_existence",
        title="Exists",
        category="Entity",
        builder="build_entity_existence_condition",
        visual=ConditionVisual(
            metric="Existence", subject="entity", value_label="missing"
        ),
        fields=(_entity_field("entity", "Subject"),),
        description="Fires while the entity is absent from the world.",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="waypoint",
        title="Road ends ahead",
        category="Entity",
        builder="build_waypoint_condition",
        visual=ConditionVisual(
            metric="Waypoints ahead", subject="entity", value_label="none"
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="distance",
                label="Look-ahead",
                kind="number",
                default=10.0,
                unit="m",
            ),
        ),
        description="No CARLA waypoint exists the given distance ahead.",
    )
)


# ---------------------------------------------------------------------------
# Built-in conditions -- world
# ---------------------------------------------------------------------------

register_condition_spec(
    ConditionSpec(
        type_id="elapsed_time",
        title="Elapsed time",
        category="World",
        builder="build_elapsed_time_condition",
        visual=ConditionVisual(
            metric="Elapsed time", rule="rule", value="duration_seconds", unit="s"
        ),
        fields=(
            FieldSpec(
                name="rule",
                label="Predicate",
                kind="select",
                default="greater_than_or_equal",
                options=COMPARISON_RULES,
            ),
            FieldSpec(
                name="duration_seconds",
                label="Duration",
                kind="number",
                default=5.0,
                unit="s",
            ),
        ),
        description="Scenario clock reaches a threshold (passes).",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="timeout",
        title="Timeout",
        category="World",
        builder="build_timeout_condition",
        visual=ConditionVisual(metric="Timeout", value="timeout_seconds", unit="s"),
        fields=(
            FieldSpec(
                name="timeout_seconds",
                label="Timeout",
                kind="number",
                default=30.0,
                unit="s",
            ),
        ),
        description="Scenario clock exceeds a limit (fails).",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="collision",
        title="Collision",
        category="World",
        builder="build_collision_condition",
        visual=ConditionVisual(metric="Collision", value_label="occurred"),
        fields=(
            FieldSpec(
                name="min_impulse",
                label="Minimum impulse",
                kind="number",
                default=0.0,
                unit="N s",
                required=False,
            ),
        ),
        description="The ego vehicle collided with any actor.",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="traffic_signal",
        title="Traffic signal state",
        category="World",
        builder="build_traffic_signal_condition",
        visual=ConditionVisual(
            metric="Signal",
            target="lanelet2_regulatory_element_id",
            target_prefix="Regulatory element",
            value="state",
        ),
        fields=(
            FieldSpec(
                name="lanelet2_regulatory_element_id",
                label="Lanelet2 regulatory element ID",
                kind="int",
                default=0,
                help=(
                    "Typed, not picked: the viewer's regulatory layer reports "
                    "the linestring that draws a sign, not the id of the "
                    "regulatory element itself."
                ),
            ),
            FieldSpec(
                name="state",
                label="Expected state",
                kind="select",
                default="Green",
                options=TRAFFIC_LIGHT_STATES,
            ),
        ),
        description="A traffic light is in the expected state.",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="action_state",
        title="Action state",
        category="World",
        builder="build_action_state_condition",
        visual=ConditionVisual(metric="Action", subject="action", value="state"),
        fields=(
            FieldSpec(
                name="action",
                label="Action",
                kind="action",
            ),
            FieldSpec(
                name="state",
                label="State",
                kind="select",
                default=ActionState.COMPLETE.value,
                options=(
                    SelectOption(
                        ActionState.STANDBY.value, "Standby -- waiting to be triggered"
                    ),
                    SelectOption(
                        ActionState.START_TRANSITION.value,
                        "Start -- the tick it fired on",
                    ),
                    SelectOption(ActionState.RUNNING.value, "Running -- under way"),
                    SelectOption(
                        ActionState.END_TRANSITION.value,
                        "End -- the tick it finished on",
                    ),
                    SelectOption(ActionState.COMPLETE.value, "Complete -- finished"),
                ),
                help=(
                    "OpenSCENARIO storyboard element states.  Complete means the "
                    "action met its own completion criteria -- a forced lane "
                    "change is only complete once the vehicle has settled onto "
                    "the next lane, not when the command was issued."
                ),
            ),
        ),
        description=(
            "Another action has reached a lifecycle state.  Use this to make "
            "one actor react to another finishing a manoeuvre, rather than to "
            "a world state that merely coincides with it."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="temporary_stop",
        title="Temporary stop",
        category="Entity",
        builder="build_temporary_stop_condition",
        visual=ConditionVisual(
            metric="Stops at",
            subject="entity",
            target="stop_lanelets",
            target_prefix="Lanelet",
            value="stop_duration",
            unit="s",
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="stop_lanelets",
                label="Stop lanelets",
                kind="lanelet_list",
                default=[],
                help="Stopping at any one of them is enough.",
            ),
            FieldSpec(
                name="s_margin",
                label="Margin",
                kind="number",
                default=5.0,
                unit="m",
                help="How far either side of a lanelet's start still counts.",
            ),
            FieldSpec(
                name="stop_duration",
                label="Held for",
                kind="number",
                default=1.0,
                unit="s",
            ),
            FieldSpec(
                name="speed_threshold",
                label="Counts as stopped below",
                kind="number",
                default=0.1,
                unit="m/s",
            ),
        ),
        description=(
            "The entity came to a stop at one of these lanelets.  The editor "
            "offers the start of a lanelet, widened by the margin; the runtime "
            "condition itself accepts any pose."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="always_true",
        title="Always",
        category="World",
        builder="build_always_true_condition",
        visual=ConditionVisual(metric="Always", value_label="true"),
        description="Fires unconditionally -- the default action trigger.",
    )
)


# ---------------------------------------------------------------------------
# Built-in spawn constraints (the sweeper's own vocabulary)
# ---------------------------------------------------------------------------

register_constraint_spec(
    ConstraintSpec(
        type_id="and",
        title="ALL",
        category="Composition",
        kind="composite",
        description="Every child constraint must match.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="or",
        title="ANY",
        category="Composition",
        kind="composite",
        description="At least one child constraint must match.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="not",
        title="NOT",
        category="Composition",
        kind="wrapper",
        description="Invert the child constraint.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="previous_of",
        title="Previous of",
        category="Topology",
        kind="composite",
        description="Lanelets immediately before the inner matches.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="following_of",
        title="Following of",
        category="Topology",
        kind="composite",
        description="Lanelets immediately after the inner matches.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="has_adjacent",
        title="Has adjacent lane",
        category="Topology",
        fields=(
            FieldSpec(
                name="value",
                label="Side",
                kind="select",
                default="left",
                options=_DIRECTIONS,
            ),
        ),
        description="A same-direction neighbour lane exists on that side.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="is_junction",
        title="Is junction",
        category="Topology",
        description="The lanelet carries a turn_direction tag.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="lanelet_length",
        title="Lanelet length",
        category="Geometry",
        fields=(
            FieldSpec(
                name="rule",
                label="Predicate",
                kind="select",
                default="greater_than_or_equal",
                options=COMPARISON_RULES,
            ),
            FieldSpec(
                name="value", label="Length", kind="number", default=10.0, unit="m"
            ),
        ),
        description="2-D centerline length of the lanelet.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="has_stop_line",
        title="Has stop line",
        category="Regulatory",
        description="The lanelet owns a stop-line regulatory element.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="has_traffic_light_stop_line",
        title="Has traffic-light stop line",
        category="Regulatory",
        description="The stop line comes from a traffic-light element.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="equals",
        title="Lanelet ID equals",
        category="Identity",
        fields=(
            FieldSpec(
                name="value",
                label="Lanelet ID",
                kind="text",
                default="any",
                help="A lanelet ID, or 'any' to match every lanelet.",
            ),
        ),
        description="Match one specific lanelet by ID.",
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="in_set",
        title="In set",
        category="Identity",
        fields=(
            FieldSpec(
                name="values",
                label="Lanelet IDs",
                kind="int_list_or_ref",
                default=[],
                help=(
                    "A list of IDs, or ${map.no_3d_model_lanelet_ids} to reuse "
                    "the map's exclusion list."
                ),
            ),
        ),
        description="Match lanelets whose ID is in a given set.",
    )
)


# ---------------------------------------------------------------------------
# Built-in spawn bindings
# ---------------------------------------------------------------------------

register_binding_spec(
    BindingSpec(
        type_id="stop_line_offset",
        title="Before stop line",
        fields=(
            FieldSpec(
                name="offset",
                label="Distance before the stop line",
                kind="number",
                default=15.0,
                unit="m",
            ),
        ),
        description=(
            "Place the entity a fixed distance upstream of the nearest stop "
            "line, walking back through predecessor lanelets when needed."
        ),
    )
)
