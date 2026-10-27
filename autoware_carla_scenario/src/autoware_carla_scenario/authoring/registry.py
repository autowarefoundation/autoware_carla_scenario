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

import copy
import re

# Free to import: `action_state` holds nothing but the enum, which is why it
# is a top-level module. Spelling these values out again -- as the comparison
# and traffic-light mirrors below have to, because their sources drag in CARLA
# -- would let a renamed state leave a dead option in the editor that only
# fails when the scenario is built.
from ..action_state import ActionState

# Free to import for the same reason: metrics reads CARLA only when a metric
# is read, so the Metric card's options are the metrics themselves.
from ..metrics import BUILT_IN_METRICS
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any, Literal, Optional

__all__ = [
    "ActionSpec",
    "ArgumentCase",
    "BuiltArgument",
    "BuiltPart",
    "ArgumentChoice",
    "BindingSpec",
    "ConditionSpec",
    "ConstraintSpec",
    "FieldKind",
    "INT_KINDS",
    "LaneletPlace",
    "INT_LIST_KINDS",
    "SEARCHABLE_LANELET_KINDS",
    "searchable_lanelet_fields",
    "FieldSpec",
    "field_holders",
    "ConditionVisual",
    "REFERENCE_PATTERN",
    "SelectOption",
    "ACTION_PHASES",
    "COMPARISON_RULES",
    "PHASE_HELP",
    "rule_symbol",
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
    # Names a controller, or one of its phases, that the document declares under
    # `map.traffic_signal_controllers`.  Both behave as plain text everywhere
    # except the inspector, which offers the declared names instead of a box to
    # retype them into: they are the document's own vocabulary, so a typo is a
    # mistake the editor can decline to let anyone make.
    "signal_controller",
    "signal_phase",
    "int_list",
    "int_list_or_ref",
    # A polyline of map-frame vertices, ``[x, y, yaw]`` per row (see
    # :mod:`autoware_carla_scenario.trajectory.authoring`).  Kept in the
    # document as data and edited as text, one vertex per line: a recording
    # transcribes to hundreds of them, which no row of form controls could hold.
    "trajectory",
    # Vertices relative to an entity's lane, ``[ds, offset, d_lane, yaw]``
    # per row (OpenSCENARIO's RelativeLanePosition; see
    # :func:`autoware_carla_scenario.trajectory.authoring.parse_relative_vertices`).
    # Edited as text like ``trajectory``, with its own columns.
    "relative_lane_trajectory",
    # Vertices placed against the scenario's route (a logical scenario), one
    # mapping per vertex with a ``kind`` (see
    # :func:`autoware_carla_scenario.trajectory.authoring.parse_route_vertices`).
    # Edited as text, ``kind key=value ...`` per line.
    "route_trajectory",
    # An ordered list of records, each a mapping of its own typed fields
    # (:attr:`FieldSpec.item_fields`), edited as rows that can be added,
    # removed and reordered.  For a list whose order means something and whose
    # entries repeat -- NADE's adversities, the first to fire winning -- which
    # one field per entry cannot say.
    "record_list",
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

#: Field kinds whose lanelet may be handed to the lanelet-constraint sweeper
#: instead of being pinned.
#:
#: A search names *one* lanelet per run -- the sweeper enumerates matches and
#: runs the scenario once for each -- so it answers the question a ``lanelet``
#: field asks and not the one ``lanelet_list`` asks.  A set of lanelets ("stop
#: at any of these") is a different statement, and giving it the same control
#: would promise that a search fills the whole set.
SEARCHABLE_LANELET_KINDS: frozenset[str] = frozenset({"lanelet"})

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


#: What a lanelet-valued field names: where something starts, where something is
#: sent, or a place the scenario only watches.
#:
#: Declared by the spec rather than guessed from whatever holds the field: a
#: "Set Goal" action names a destination exactly as an ego's own goal does, and
#: a map that inferred the answer from the holder drew the two with different
#: glyphs.
LaneletPlace = Literal["spawn", "goal", "watched"]


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
        place: For a lanelet field, what kind of place it names -- see
            :data:`LaneletPlace`.  Read by the editor's map, which draws a
            destination differently from a place merely watched.
        item_fields: For a ``record_list`` field, the fields of each record,
            checked and coerced as a primitive's own are.
    """

    name: str
    label: str
    kind: FieldKind = "text"
    default: Any = None
    options: tuple[SelectOption, ...] = ()
    unit: str = ""
    help: str = ""
    required: bool = True
    place: LaneletPlace = "watched"
    item_fields: tuple["FieldSpec", ...] = ()


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

    ``details`` names further fields to show beside the metric, each labelled by
    its own :class:`FieldSpec`.  They are for the parts of a condition that
    would otherwise be invisible on the canvas -- two conditions differing only
    in an unshown parameter must not render identically.
    """

    metric: str
    subject: Optional[str] = None
    target: Optional[str] = None
    target_prefix: str = ""
    rule: Optional[str] = None
    value: Optional[str] = None
    value_label: str = ""
    unit: str = ""
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuiltPart:
    """One call to a built argument's constructor.

    ``args`` maps that constructor's keywords to the spec fields feeding them;
    ``constants`` supplies the keywords no field does, as source text.  A part
    with ``when_present`` set is left out at runtime when that field has no
    value, which is how an unset bound says "no rule" rather than "a rule
    comparing against nothing".
    """

    args: tuple[tuple[str, str], ...] = ()
    constants: tuple[tuple[str, str], ...] = ()
    when_present: str = ""
    #: Keywords filled from the compiled node rather than from a field, by
    #: name: ``("advance", "advance_conditions")`` passes the action's
    #: waypoint conditions, built into runtime conditions.  The names the
    #: generator knows are listed in its ``COMPILED_SOURCES``.
    sources: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BuiltArgument:
    """A constructor argument the generator assembles instead of passing through.

    Most arguments are a field handed over as it stands, which the signature
    alone describes.  These are the ones it cannot: a pose built out of two
    fields and a literal, a list of comparison rules built out of four bounds,
    a lanelet id per item of a list field.  Saying it here rather than in a
    hand-written builder keeps the generator's checks -- fields that reach no
    argument, arguments nothing fills -- applying to these specs too.

    Three shapes, by which of the last three attributes is set:

    * one ``parts`` entry, no ``over`` -- a single value;
    * several ``parts`` -- a list, each part one item, absent ones dropped;
    * ``over`` -- a list, one item per element of that list field, which feeds
      ``item_kwarg``.
    """

    #: The constructor keyword this fills.
    kwarg: str
    #: What to call, as ``<relative module>:<name>``.
    target: str
    parts: tuple[BuiltPart, ...] = ()
    #: A list field to map over, one call per element.
    over: str = ""
    #: The keyword each element of ``over`` is passed as.
    item_kwarg: str = ""


@dataclass(frozen=True)
class ArgumentCase:
    """One branch of a discriminated constructor argument.

    Exactly one of ``constant`` and ``field`` says where the value comes from:
    a dotted path such as ``..actions:TrafficLightTarget.ALL``, or the name of
    a :class:`FieldSpec` whose value is passed through.
    """

    when: str
    constant: str = ""
    field: str = ""


@dataclass(frozen=True)
class ArgumentChoice:
    """A constructor argument whose source a ``select`` field chooses.

    Some runtime arguments are a union: ``TrafficSignalAction`` takes either a
    list of ids or ``TrafficLightTarget.ALL``.  An author picks between them
    with a select, so the mapping from that select to the argument is what the
    spec has to say -- and saying it here rather than in a hand-written builder
    is what lets the generator check that *every* option has a branch.  A new
    option with no case is the failure this exists to prevent: it used to fall
    into a builder's ``else`` and quietly produce the wrong kind of value.

    Attributes:
        kwarg: Constructor parameter being fed.
        discriminator: Name of the ``select`` field that chooses the branch.
        cases: One per option of *discriminator*, exhaustively.
    """

    kwarg: str
    discriminator: str
    cases: tuple[ArgumentCase, ...]


@dataclass(frozen=True)
class ActionSpec:
    """Everything the editor and compiler need to know about one action type."""

    type_id: str
    title: str
    category: str
    builder: str
    #: Runtime class this builds, as ``<relative module>:<name>``.  When it is
    #: set the builder named above is *generated* from the constructor's
    #: signature; when it is empty the builder is hand-written in
    #: :mod:`autoware_carla_scenario.authoring.builders`.  Exactly one of the
    #: two, which ``test_authoring_registry`` enforces.
    target: str = ""
    #: Field name -> constructor keyword, for the pairs that differ.  Fields
    #: whose name already matches the constructor need no entry.
    argmap: tuple[tuple[str, str], ...] = ()
    #: Constructor arguments a select field chooses the source of.
    choices: tuple[ArgumentChoice, ...] = ()
    #: Constructor arguments the generator assembles rather than passes through.
    builds: tuple[BuiltArgument, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    #: What the action acts on. ``actor`` actions are performed *by* a vehicle
    #: and belong to its track; ``environment`` actions change the world around
    #: every vehicle -- traffic lights, for instance -- and belong to the
    #: environment track. The distinction is the action type's, not the
    #: author's: giving a traffic light an actor used to move its card into that
    #: vehicle's lane while the runtime, which never reads the field, went on
    #: setting the same lights.
    scope: Literal["actor", "environment"] = "actor"
    #: Entity kinds that may perform this action.  Read by the validator, so a
    #: pedestrian asked to change lane is a document error rather than a
    #: warning in a run's log -- and so the rule lives with the action rather
    #: than as a list of type ids the validator would have to be kept in step
    #: with.  Ignored for an ``environment`` action, which has no actor.
    actor_kinds: tuple[str, ...] = ("ego", "vehicle")
    #: ``instant`` renders as a diamond on the swimlane, ``continuous`` as a bar.
    visual_kind: Literal["instant", "continuous"] = "instant"
    #: An ``actor`` action whose actor may be left out: it then acts on the
    #: world (a Spawn or Despawn card for a prop, by name) and belongs to the
    #: environment track.
    actor_optional: bool = False
    #: The phase a freshly added card lands in.  ``init`` for anything the run
    #: has to be set up with rather than driven by -- a goal, the starting state
    #: of the lights.
    default_phase: Literal["init", "pre_tick", "post_tick"] = "pre_tick"
    #: Whether the action follows a list of vertices that can carry waypoint
    #: conditions (:attr:`~autoware_carla_scenario.authoring.models.ActionNode.advance_conditions`).
    #: The editor offers them, and the validator accepts them, only then.
    vertex_conditions: bool = False
    description: str = ""

    @property
    def actor_required(self) -> bool:
        """Whether this action needs a vehicle to perform it."""
        return self.scope == "actor" and not self.actor_optional


def field_holders(
    fields: "tuple[FieldSpec, ...]", params: "dict[str, Any]"
) -> "Iterator[tuple[FieldSpec, dict[str, Any]]]":
    """Yield every field of *fields* with the mapping that holds its value.

    A ``record_list`` field is not yielded itself: each of its records yields
    its own fields, with the record as the holder.  For whatever looks for a
    kind of field -- entity references, action references -- so a reference
    written in a record is found as one written at the top level is.
    """
    for spec in fields:
        if spec.kind == "record_list":
            for row in params.get(spec.name) or ():
                if isinstance(row, dict):
                    yield from field_holders(spec.item_fields, row)
        else:
            yield spec, params


@dataclass(frozen=True)
class ConditionSpec:
    """Everything the editor and compiler need to know about one condition type."""

    type_id: str
    title: str
    category: str
    builder: str
    visual: ConditionVisual
    #: See :attr:`ActionSpec.target`.
    target: str = ""
    #: See :attr:`ActionSpec.argmap`.
    argmap: tuple[tuple[str, str], ...] = ()
    #: See :attr:`ActionSpec.choices`.
    choices: tuple[ArgumentChoice, ...] = ()
    #: See :attr:`ActionSpec.builds`.
    builds: tuple[BuiltArgument, ...] = ()
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
    #: What the binding works out: a spawn's ``s``, or a lanelet id for a
    #: ``derived`` lanelet slot.  The editor offers each only where it fits.
    produces: Literal["s", "lanelet"] = "s"
    #: Parameters the binding is always given and the editor does not offer.
    fixed: tuple[tuple[str, Any], ...] = ()


# ---------------------------------------------------------------------------
# Shared option sets
# ---------------------------------------------------------------------------

#: Mirrors ``ComparisonRule``.  Spelled out rather than imported because
#: importing ``autoware_carla_scenario.conditions`` pulls in CARLA, and the
#: editor process has no simulator.  ``test_authoring_registry`` asserts the two
#: stay in step.
#: The phases an action can be placed in, in the order a run performs them.
#: One table, so the inspector's select, the form parser and the canvas cannot
#: disagree about which phases exist.
ACTION_PHASES: tuple[SelectOption, ...] = (
    SelectOption("init", "Initialization"),
    SelectOption("pre_tick", "Pre-tick"),
    SelectOption("post_tick", "Post-tick"),
)

#: What each phase means, for the one place that has room to say it.
PHASE_HELP: dict[str, str] = {
    "init": (
        "Before the run: from the ego spawning until it reports ready. The "
        "world ticks and sensors publish, but the clock has not started and no "
        "condition has been evaluated."
    ),
    "pre_tick": "Each tick of the running scenario, before the world advances.",
    "post_tick": "Each tick of the running scenario, after the world advances.",
}

COMPARISON_RULES: tuple[SelectOption, ...] = (
    SelectOption("less_than", "<"),
    SelectOption("less_than_or_equal", "≤"),
    SelectOption("greater_than", ">"),
    SelectOption("greater_than_or_equal", "≥"),
    SelectOption("equal_to", "="),
)


def rule_symbol(rule: object) -> str:
    """Return the operator glyph for a comparison rule value.

    The canvas and the inspector select read the same table, so a rule added
    to :data:`COMPARISON_RULES` cannot render as its raw name on one of them.
    """
    wanted = str(rule)
    for option in COMPARISON_RULES:
        if option.value == wanted:
            return option.label
    return wanted


#: A vehicle's turn indicators, by ``TurnIndicatorState`` member.
_TURN_INDICATOR_STATES: tuple[SelectOption, ...] = (
    SelectOption("OFF", "Off"),
    SelectOption("LEFT", "Left"),
    SelectOption("RIGHT", "Right"),
    SelectOption("HAZARD", "Hazard (both)"),
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

#: Mirrors ``hils.protocol.LINK_DIRECTIONS`` (same reason as above).
_LINK_DIRECTIONS: tuple[SelectOption, ...] = (
    SelectOption("both", "Both ways"),
    SelectOption("to_ecu", "To the ECU"),
    SelectOption("from_ecu", "From the ECU"),
)

#: A junction is also driven straight through, which a lane is not changed.
_TURN_DIRECTIONS: tuple[SelectOption, ...] = (
    *_DIRECTIONS,
    SelectOption("straight", "Straight"),
)


def _coordinate_system_field() -> FieldSpec:
    """Return the frame select the three distance conditions share.

    Defaulting to ``entity`` keeps every document written before this existed
    meaning what it meant.  ``lane`` is the measurement OpenSCENARIO calls
    ``coordinateSystem: lane``, and the one the transpiler needs in order to
    carry that attribute across rather than dropping it.

    Named ``measure`` in the document and mapped to the constructor's
    ``coordinate_system`` by each spec's ``argmap``, the way ``seconds`` maps
    to ``value``: the short name is what an author reads, and it is also what
    keeps the generated call inside the width the formatter wraps at.
    """
    return FieldSpec(
        name="measure",
        label="Measured",
        kind="select",
        default="entity",
        options=(
            SelectOption("entity", "Straight line"),
            SelectOption("lane", "Along the road"),
        ),
        help=(
            "In a straight line between the two, or along the road they "
            "share.  They agree on a straight road and differ on a curve, "
            "where the straight line reads short.  Along the road needs a "
            "loaded map, and has no answer while the two are on different "
            "roads."
        ),
    )


def _rule_field(default: str = "less_than") -> FieldSpec:
    """Return the comparison-operator select, defaulting to *default*.

    Taking the default as an argument rather than fixing it: a condition whose
    natural reading is "at least" had to write the whole field out again to say
    so, which is three more copies of the one place `COMPARISON_RULES` is used.
    """
    return FieldSpec(
        name="rule",
        label="Predicate",
        kind="select",
        default=default,
        options=COMPARISON_RULES,
    )


def _edge_to_edge_field() -> FieldSpec:
    """Return the bounding-box-versus-centres switch.

    Shared because the choice means the same thing wherever a distance is
    measured, and a second copy would be a second place for its help text to
    drift from what the measurement does.

    OpenSCENARIO spells this ``freespace``, and this library does not.  The
    standard's name says what is being measured *through* rather than what the
    measurement runs *between*, so as a bare boolean it tells a reader nothing
    about bounding boxes; the mapping belongs in the transpiler, next to the
    other names it translates, rather than in the vocabulary an author reads.
    """
    return FieldSpec(
        name="edge_to_edge",
        label="Between bounding boxes",
        kind="bool",
        default=False,
        required=False,
        help=(
            "Measure between the vehicles' bounding boxes rather than between "
            "their centres.  The difference is about a vehicle length, which "
            "matters most where the answer is a handful of seconds.  This is "
            "OpenSCENARIO's freespace, and it cannot be combined with "
            "measuring along the road."
        ),
    )


def _entity_field(name: str, label: str, *, required: bool = True) -> FieldSpec:
    """Return an entity-reference field; options come from the document.

    *required* is taken as an argument for the same reason ``_rule_field``
    takes its default: a condition where naming a vehicle is genuinely
    optional -- a collision with anything -- would otherwise have to write the
    whole field out again to say so.
    """
    return FieldSpec(
        name=name, label=label, kind="entity", default=None, required=required
    )


def _temporary_stop_fields(place: str) -> tuple[FieldSpec, ...]:
    """Return what makes a temporary stop: how near *place*, how still, how long."""
    return (
        FieldSpec(
            name="s_margin",
            label="Margin",
            kind="number",
            default=5.0,
            unit="m",
            help=f"How far either side of {place} still counts.",
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
    )


def _extent_fields(noun: str) -> tuple[FieldSpec, ...]:
    """Return the s/t range fields that bound a position along *noun*.

    Both position conditions accept the same stretch, so they share one
    definition -- separate copies could disagree on a unit or on whether a
    bound is required while ``build_*_condition`` reads them through one path.

    What they differ in is the frame, and the help says which: a lanelet's
    ``s`` runs along its direction of travel, a road's along its reference
    line, and on a lane against that line the two run opposite ways.
    """
    direction = (
        "its direction of travel"
        if noun == "lanelet"
        else "its reference line, whichever way its lanes run"
    )
    return (
        FieldSpec(
            name="s_min",
            label="s from",
            kind="number",
            default=None,
            required=False,
            unit="m",
            help=(
                f"Along the {noun} from its start, following {direction}.  "
                "Leave both empty to accept the whole length."
            ),
        ),
        FieldSpec(
            name="s_max",
            label="s to",
            kind="number",
            default=None,
            required=False,
            unit="m",
        ),
        FieldSpec(
            name="t_min",
            label="t from",
            kind="number",
            default=None,
            required=False,
            unit="m",
            help=f"Across the {noun}. Positive is left of {direction}.",
        ),
        FieldSpec(
            name="t_max",
            label="t to",
            kind="number",
            default=None,
            required=False,
            unit="m",
        ),
    )


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


def binding_specs(produces: Optional[str] = None) -> list[BindingSpec]:
    """Return the registered binding specs, ordered by title.

    *produces* narrows them to the ones that work out an ``s`` or a lanelet,
    which is what decides where each one may be offered.
    """
    return sorted(
        (s for s in _BINDING_SPECS.values() if produces in (None, s.produces)),
        key=lambda s: s.title,
    )


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
    """Return a fresh params mapping seeded with each field's default.

    Copied, so no two nodes share a default list -- or a record list's
    records -- that one of them then edits in place.
    """
    return {f.name: copy.deepcopy(f.default) for f in fields}


def searchable_lanelet_fields(
    spec: "ActionSpec | ConditionSpec | None",
) -> "tuple[FieldSpec, ...]":
    """Return the fields of *spec* whose lanelet may be searched for.

    The one place that answers "can this field be handed to the sweeper", so
    the inspector, the validator and the Hydra config cannot come to disagree
    about which parameters a document may attach a constraint tree to.
    """
    if spec is None:
        return ()
    return tuple(f for f in spec.fields if f.kind in SEARCHABLE_LANELET_KINDS)


# ---------------------------------------------------------------------------
# Built-in actions
# ---------------------------------------------------------------------------

register_action_spec(
    ActionSpec(
        type_id="environment",
        title="Set Weather",
        category="Environment",
        builder="build_environment_action",
        target="..actions:EnvironmentAction",
        scope="environment",
        visual_kind="instant",
        # Most scenarios set the weather once, before anything moves.  A card
        # on the tick loop is what makes it change during a run.
        default_phase="init",
        fields=(
            FieldSpec(
                name="cloudiness",
                label="Cloudiness",
                kind="number",
                default=None,
                required=False,
                unit="%",
                help="Leave empty to keep what the world already has.",
            ),
            FieldSpec(
                name="precipitation",
                label="Rain",
                kind="number",
                default=None,
                required=False,
                unit="%",
                help="Rain intensity.",
            ),
            FieldSpec(
                name="precipitation_deposits",
                label="Standing water",
                kind="number",
                default=None,
                required=False,
                unit="%",
                help="Water left on the road surface.",
            ),
            FieldSpec(
                name="wetness",
                label="Wetness",
                kind="number",
                default=None,
                required=False,
                unit="%",
                help="How wet surfaces look.",
            ),
            FieldSpec(
                name="wind_intensity",
                label="Wind",
                kind="number",
                default=None,
                required=False,
                unit="%",
                help="Wind strength.",
            ),
            FieldSpec(
                name="fog_density",
                label="Fog density",
                kind="number",
                default=None,
                required=False,
                unit="%",
                help="Thickness of the fog.",
            ),
            FieldSpec(
                name="fog_distance",
                label="Fog distance",
                kind="number",
                default=None,
                required=False,
                unit="m",
                help="How far away the fog starts.",
            ),
            FieldSpec(
                name="sun_altitude_angle",
                label="Sun altitude",
                kind="number",
                default=None,
                required=False,
                unit="deg",
                help="Degrees above the horizon; negative is night.",
            ),
            FieldSpec(
                name="sun_azimuth_angle",
                label="Sun azimuth",
                kind="number",
                default=None,
                required=False,
                unit="deg",
                help="Compass direction of the sun.",
            ),
        ),
        description=(
            "Set the weather and the position of the sun.  Every field left "
            "empty keeps what the world already has, so a scenario that wants "
            "rain does not have to restate the time of day and reset it by "
            "accident.  Time of day is the sun's angle: CARLA has no clock."
        ),
    )
)

register_action_spec(
    ActionSpec(
        type_id="lane_change",
        title="Lane Change",
        category="Vehicle / Motion",
        builder="build_lane_change_action",
        target="..actions:LaneChangeAction",
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
        type_id="turn_indicator",
        title="Turn Indicator",
        category="Vehicle / Lights",
        builder="build_turn_indicator_action",
        target="..actions:TurnIndicatorAction",
        visual_kind="instant",
        fields=(
            FieldSpec(
                name="state",
                label="Indicators",
                kind="select",
                default="LEFT",
                options=_TURN_INDICATOR_STATES,
            ),
        ),
        description=(
            "Light the vehicle's turn indicators (or hazard lights, or neither); "
            "they stay so until set again."
        ),
    )
)

register_action_spec(
    ActionSpec(
        type_id="communication_loss",
        title="Communication Loss",
        category="Vehicle / HILS",
        builder="build_communication_loss_action",
        target="..actions:CommunicationLossAction",
        actor_kinds=("ego",),
        visual_kind="continuous",
        fields=(
            FieldSpec(
                name="device",
                label="Device",
                kind="text",
                default=None,
                required=False,
                help=(
                    "The HILS bridge configuration's device whose data stops. "
                    "Leave empty for every device."
                ),
            ),
            FieldSpec(
                name="messages",
                label="Messages",
                kind="text",
                default=None,
                required=False,
                help=(
                    "The messages that stop, comma-separated, by the names the "
                    "device's codec gives them (VehicleSpeed, scan). Leave empty "
                    "to stop everything the device carries."
                ),
            ),
            FieldSpec(
                name="direction",
                label="Direction",
                kind="select",
                default="both",
                options=_LINK_DIRECTIONS,
            ),
            FieldSpec(
                name="duration",
                label="Duration",
                kind="number",
                default=None,
                required=False,
                unit="s",
                help="Leave empty to keep the data stopped until the scenario ends.",
            ),
        ),
        description=(
            "Stop data between the HILS bridge and an ECU under test "
            "(ego.entity=hils) while the card runs, as a break in the "
            "communication path would: a CAN frame, a LiDAR's or a camera's "
            "packets.  No effect on any other ego."
        ),
    )
)

register_action_spec(
    ActionSpec(
        type_id="turn",
        title="Turn at Junction",
        category="Vehicle / Motion",
        builder="build_turn_action",
        target="..actions:TurnAction",
        visual_kind="instant",
        fields=(
            FieldSpec(
                name="direction",
                label="Direction",
                kind="select",
                default="left",
                options=_TURN_DIRECTIONS,
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

_BACKGROUND_CONSTRAINTS_HELP = (
    "Lanelet constraints, as a sweep writes them, in one line of YAML: "
    "a list, every item of which has to hold."
)
_NOT_IN_A_JUNCTION = "[{type: not, constraint: {type: is_junction}}]"
_ROAD_ENDS = (
    "[{type: not, constraint: {type: previous_of, constraints: "
    "[{type: lanelet_length, rule: greater_than_or_equal, value: 0.0}]}}]"
)

register_action_spec(
    ActionSpec(
        type_id="traffic_source",
        title="Traffic Source",
        category="Environment",
        builder="build_traffic_source_action",
        target="..actions:TrafficSourceAction",
        scope="environment",
        visual_kind="continuous",
        builds=(
            BuiltArgument(
                kwarg="constraints",
                target="..actions.background_traffic:constraints_from_text",
                parts=(BuiltPart(args=(("text", "constraints"),)),),
            ),
        ),
        fields=(
            FieldSpec(
                name="constraints",
                label="Where vehicles appear",
                kind="text",
                default=_NOT_IN_A_JUNCTION,
                help=_BACKGROUND_CONSTRAINTS_HELP,
            ),
            FieldSpec(
                name="initial_vehicles",
                label="Vehicles at start",
                kind="int",
                default=10,
                help="Placed at once, on the first tick it runs.",
            ),
            FieldSpec(
                name="vehicles_per_minute",
                label="Then, per minute",
                kind="number",
                default=0.0,
                help="Added while it runs; 0 places only the ones at start.",
            ),
            FieldSpec(
                name="max_vehicles",
                label="At most on the road",
                kind="int",
                default=None,
                required=False,
                help="Every background vehicle counts, whichever source made it.",
            ),
            FieldSpec(
                name="speed_kmh",
                label="Speed",
                kind="number",
                default=30.0,
                unit="km/h",
            ),
            FieldSpec(
                name="min_gap_m",
                label="Least gap",
                kind="number",
                default=15.0,
                unit="m",
            ),
            FieldSpec(name="seed", label="Seed", kind="int", default=0),
        ),
        description=(
            "Background traffic, as OpenSCENARIO's TrafficSourceAction, placed "
            "on the lanelets the constraints match.  The traffic backend drives "
            "it.  Runs until its end condition fires, or the scenario ends."
        ),
    )
)

register_action_spec(
    ActionSpec(
        type_id="traffic_sink",
        title="Traffic Sink",
        category="Environment",
        builder="build_traffic_sink_action",
        target="..actions:TrafficSinkAction",
        scope="environment",
        visual_kind="continuous",
        builds=(
            BuiltArgument(
                kwarg="constraints",
                target="..actions.background_traffic:constraints_from_text",
                parts=(BuiltPart(args=(("text", "constraints"),)),),
            ),
        ),
        fields=(
            FieldSpec(
                name="constraints",
                label="Where vehicles disappear",
                kind="text",
                default=_ROAD_ENDS,
                help=_BACKGROUND_CONSTRAINTS_HELP,
            ),
        ),
        description=(
            "Removes background traffic on the lanelets the constraints match, "
            "as OpenSCENARIO's TrafficSinkAction.  Runs until its end condition "
            "fires, or the scenario ends."
        ),
    )
)

#: The turn-back thresholds of WalkStraight, and of a routed pedestrian
#: stepped over a road stretch: one test (actions._road_threat).
_TURN_BACK_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="turn_back_ttc_s",
        label="Turn back at TTC",
        kind="number",
        default=None,
        required=False,
        unit="s",
        help=(
            "Turn back when a vehicle's time to collision, along the "
            "road, is at most this.  Empty: not by TTC."
        ),
    ),
    FieldSpec(
        name="turn_back_distance_m",
        label="Turn back at distance",
        kind="number",
        default=None,
        required=False,
        unit="m",
        help=(
            "Turn back when a vehicle is at most this far away along "
            "the road.  Empty: not by distance."
        ),
    ),
    FieldSpec(
        name="turn_back_within_m",
        label="Turn back within",
        kind="number",
        default=1.0,
        unit="m",
        help=(
            "How far onto the road, from the kerb, the pedestrian may "
            "be and still turn back; further out it keeps going."
        ),
    ),
)

register_action_spec(
    ActionSpec(
        type_id="walk_straight",
        title="Walk Straight",
        category="Pedestrian",
        builder="build_walk_straight_action",
        target="..actions:WalkStraightAction",
        actor_kinds=("pedestrian",),
        visual_kind="continuous",
        fields=(
            FieldSpec(
                name="speed_ms",
                label="Walking speed",
                kind="number",
                default=1.4,
                unit="m/s",
                help=(
                    "Zero stops the pedestrian where it stands, so one card "
                    "can start a crossing and a second can end it."
                ),
            ),
            *_TURN_BACK_FIELDS,
        ),
        description=(
            "Send a pedestrian forward the way it is facing, with no "
            "destination.  It keeps going rather than routing around the "
            "vehicle the scenario is about, which is what makes it usable "
            "for a crossing."
        ),
    )
)

#: Mirrors ``TrajectoryFollowingMode``; ``test_authoring_registry`` keeps the
#: two in step (the codegen refuses an option that names no member).
_FOLLOWING_MODES: tuple[SelectOption, ...] = (
    SelectOption("position", "Position -- placed on it every tick (replay)"),
    SelectOption("follow", "Follow -- a controller drives along it"),
)

register_action_spec(
    ActionSpec(
        type_id="despawn",
        title="Despawn",
        category="Vehicle / Motion",
        builder="build_despawn_action",
        target="..actions:DespawnAction",
        actor_kinds=("vehicle", "pedestrian"),
        actor_optional=True,
        visual_kind="instant",
        fields=(
            FieldSpec(
                name="name",
                label="Prop name",
                kind="text",
                default=None,
                required=False,
                help="With no actor: the name of the prop a Spawn card added.",
            ),
        ),
        description=(
            "Take a vehicle or pedestrian out of the world for good: its "
            "traffic backend lets go of it, its actor is destroyed and it is "
            "forgotten. Trigger it on its Set Goal card's Action state "
            "(Complete) to have a road user leave where its route ends, "
            "rather than drive on wherever its traffic takes it."
        ),
    )
)

register_action_spec(
    ActionSpec(
        type_id="spawn",
        title="Spawn",
        category="Vehicle / Motion",
        builder="build_spawn_action",
        target="..actions:SpawnAction",
        argmap=(("speed_ms", "speed"),),
        actor_kinds=("vehicle", "pedestrian"),
        actor_optional=True,
        visual_kind="instant",
        builds=(
            BuiltArgument(
                kwarg="position",
                target="..coordinate:Lanelet2Pose",
                parts=(
                    BuiltPart(
                        args=(
                            ("lanelet_id", "lanelet_id"),
                            ("s", "s"),
                            ("t", "t"),
                            ("heading", "heading"),
                        ),
                    ),
                ),
            ),
        ),
        fields=(
            FieldSpec(
                name="lanelet_id",
                label="Lanelet",
                kind="lanelet",
                default=0,
                required=False,
                help=(
                    "The lanelet of the place it enters at. None (0): where "
                    "it was declared to spawn, which it waits under."
                ),
            ),
            FieldSpec(
                name="s",
                label="s",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="Along the lanelet from its start.",
            ),
            FieldSpec(
                name="t",
                label="t",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="Off the centreline, positive to the left.",
            ),
            FieldSpec(
                name="heading",
                label="Heading",
                kind="number",
                default=0.0,
                required=False,
                unit="rad",
                help="Turned from the lane's direction, counter-clockwise.",
            ),
            FieldSpec(
                name="name",
                label="Prop name",
                kind="text",
                default=None,
                required=False,
                help=(
                    "With no actor: the prop's name, which a Despawn card "
                    "removes it by (one standing of that name is replaced)."
                ),
            ),
            FieldSpec(
                name="blueprint",
                label="Prop",
                kind="text",
                default="static.prop.constructioncone",
                required=False,
                help="With no actor: the CARLA prop (static.prop.*) it is.",
            ),
            FieldSpec(
                name="on_ground",
                label="On the road surface",
                kind="bool",
                default=True,
                required=False,
                help=(
                    "A prop stands on the road surface under its place rather "
                    "than at the place's own height."
                ),
            ),
            FieldSpec(
                name="shift_max",
                label="Or up to this far on",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help=(
                    "If a road user stands on the place as the card fires, "
                    "the nearest clear place up to this far on along the lane "
                    "(0: there all the same). To wait for the place first, "
                    "put a Place clear condition in the trigger."
                ),
            ),
            FieldSpec(
                name="length",
                label="Footprint length",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help=(
                    "The length of the place it needs where its actor's outline "
                    "does not say it (a bicycle on a walker mesh); 0: its "
                    "actor's bounding box."
                ),
            ),
            FieldSpec(
                name="width",
                label="Footprint width",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="Its width, likewise; 0: its actor's bounding box.",
            ),
            FieldSpec(
                name="lanes",
                label="Shift along lanelets",
                kind="lanelet_list",
                default=[],
                required=False,
                help=(
                    "The lanelets, in order, a shift runs along -- the way its "
                    "route goes on -- and no further than their last. Empty: on "
                    "along the lane, the straightest way where it branches."
                ),
            ),
            FieldSpec(
                name="speed_ms",
                label="Speed",
                kind="number",
                default=0.0,
                required=False,
                unit="m/s",
                help="The speed it enters at, along its heading.",
            ),
        ),
        description=(
            "Bring a vehicle or pedestrian spawned out of the world (Spawn "
            "hidden) into the scene at a place, facing along its lane, at a "
            "speed -- or, with no actor, add a prop (a cone, a barrier...) by "
            "name. The counterpart of Despawn. Trigger an entity's Set Goal "
            "card on this card's Action state (Complete) to drive it on."
        ),
    )
)

register_action_spec(
    ActionSpec(
        type_id="stalled_object",
        title="Stalled Object",
        category="Adversity",
        builder="build_stalled_object_action",
        target="..actions.stalled_object:StalledObjectAction",
        actor_kinds=("vehicle",),
        visual_kind="continuous",
        builds=(
            BuiltArgument(
                kwarg="pose",
                target="..coordinate:Lanelet2Pose",
                parts=(
                    BuiltPart(
                        args=(
                            ("lanelet_id", "lanelet_id"),
                            ("s", "s"),
                            ("t", "t"),
                            ("heading", "heading"),
                        )
                    ),
                ),
            ),
        ),
        fields=(
            FieldSpec(
                name="lanelet_id",
                label="Lanelet",
                kind="lanelet",
                default=0,
                help="The lanelet the vehicle stands on.",
            ),
            FieldSpec(
                name="s",
                label="s",
                kind="number",
                default=0.0,
                unit="m",
                help="Along the lanelet from its start.",
            ),
            FieldSpec(
                name="t",
                label="t",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="Off the centreline, positive to the left.",
            ),
            FieldSpec(
                name="heading",
                label="Heading",
                kind="number",
                default=0.0,
                required=False,
                unit="rad",
                help="Turned from the lane's direction, counter-clockwise.",
            ),
            FieldSpec(
                name="start_time",
                label="Appears at",
                kind="number",
                default=0.0,
                unit="s",
                help="From the scenario's start.",
            ),
            FieldSpec(
                name="end_time",
                label="Removed at",
                kind="number",
                default=None,
                required=False,
                unit="s",
                help="From the scenario's start; empty: never.",
            ),
        ),
        description=(
            "TeraSim's stalled object: the vehicle stands still at a pose "
            "from one time to another, out of the world before and after.  "
            "Spawn it out of the world; two of them make a collision's "
            "aftermath."
        ),
    )
)

#: Mirrors ``AdversityLocation``; the codegen refuses an option that names no
#: member.  No default: an adversity fires anywhere only when told so.
_ADVERSITY_LOCATIONS: tuple[SelectOption, ...] = (
    SelectOption("any", "Anywhere"),
    SelectOption("intersection", "In or 25 m before a junction"),
    SelectOption("highway", "On a highway"),
    SelectOption("roundabout", "In or 25 m before a roundabout"),
)


#: Every adversarial action's fields: where it may fire, how it drives.
_ADVERSITY_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="location",
        label="Where",
        kind="select",
        options=_ADVERSITY_LOCATIONS,
        help="Where the adversity may fire; 'Anywhere' has to be chosen.",
    ),
    FieldSpec(
        name="following_mode",
        label="Following mode",
        kind="select",
        default="follow",
        options=_FOLLOWING_MODES,
    ),
)


_ADVERSITY_NOTE = (
    "  Fires when its trigger holds at the chosen location (TeraSim's "
    "numbers), drives the planned path, then hands the entity back to traffic."
)

register_action_spec(
    ActionSpec(
        type_id="adversarial_brake",
        title="Adversarial Brake",
        category="Adversity",
        builder="build_adversarial_brake_action",
        target="..actions.adversarial:AdversarialBrakeAction",
        actor_kinds=("vehicle",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Brake hard in front of a follower that keeps coming."
        + _ADVERSITY_NOTE,
    )
)

register_action_spec(
    ActionSpec(
        type_id="ignore_leader",
        title="Ignore Leader",
        category="Adversity",
        builder="build_ignore_leader_action",
        target="..actions.adversarial:IgnoreLeaderAction",
        actor_kinds=("vehicle",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Drive on as if the close, slower leader were not there."
        + _ADVERSITY_NOTE,
    )
)

register_action_spec(
    ActionSpec(
        type_id="adversarial_cut_in",
        title="Adversarial Cut-in",
        category="Adversity",
        builder="build_adversarial_cut_in_action",
        target="..actions.adversarial:AdversarialCutInAction",
        actor_kinds=("vehicle",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Change lane in front of a vehicle too close behind in the next lane."
        + _ADVERSITY_NOTE,
    )
)

register_action_spec(
    ActionSpec(
        type_id="adversarial_merge",
        title="Adversarial Merge",
        category="Adversity",
        builder="build_adversarial_merge_action",
        target="..actions.adversarial:AdversarialMergeAction",
        actor_kinds=("vehicle",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Beside a merging lane, speed up or change in to cut a "
        "merging vehicle off." + _ADVERSITY_NOTE,
    )
)

register_action_spec(
    ActionSpec(
        type_id="lane_change_abort",
        title="Lane-change Abort",
        category="Adversity",
        builder="build_lane_change_abort_action",
        target="..actions.adversarial:LaneChangeAbortAction",
        actor_kinds=("vehicle",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Start into the next lane in front of a vehicle too close "
        "behind there, then swerve back." + _ADVERSITY_NOTE,
    )
)

register_action_spec(
    ActionSpec(
        type_id="head_on",
        title="Head-on",
        category="Adversity",
        builder="build_head_on_action",
        target="..actions.adversarial:HeadOnAction",
        actor_kinds=("vehicle",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Drive on at a leader coming the other way." + _ADVERSITY_NOTE,
    )
)

register_action_spec(
    ActionSpec(
        type_id="run_traffic_rule",
        title="Run Traffic Rule",
        category="Adversity",
        builder="build_run_traffic_rule_action",
        target="..actions.adversarial:RunTrafficRuleAction",
        actor_kinds=("vehicle",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Run the red or yellow light the vehicle is stopping for."
        + _ADVERSITY_NOTE,
    )
)

register_action_spec(
    ActionSpec(
        type_id="jaywalk",
        title="Jaywalk",
        category="Adversity",
        builder="build_jaywalk_action",
        target="..actions.adversarial:JaywalkAction",
        actor_kinds=("pedestrian",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Walk straight across the nearest road, off any crosswalk."
        + _ADVERSITY_NOTE,
    )
)

register_action_spec(
    ActionSpec(
        type_id="pedestrian_run_red_light",
        title="Pedestrian Runs a Red Light",
        category="Adversity",
        builder="build_pedestrian_run_red_light_action",
        target="..actions.adversarial:PedestrianRunRedLightAction",
        actor_kinds=("pedestrian",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Cross on red at the crosswalk the pedestrian waits at."
        + _ADVERSITY_NOTE,
    )
)

register_action_spec(
    ActionSpec(
        type_id="stop_crossing",
        title="Stop Crossing",
        category="Adversity",
        builder="build_stop_crossing_action",
        target="..actions.adversarial:StopCrossingAction",
        actor_kinds=("pedestrian",),
        visual_kind="continuous",
        fields=_ADVERSITY_FIELDS,
        description="Stop on the crosswalk a moment, then hurry across."
        + _ADVERSITY_NOTE,
    )
)

#: Mirrors ``AdversityKind``; the codegen refuses an option that names no
#: member.
_ADVERSITY_KINDS: tuple[SelectOption, ...] = (
    SelectOption("brake_check", "Brake check"),
    SelectOption("ignore_leader", "Ignore leader"),
    SelectOption("cut_in", "Cut in"),
    SelectOption("traffic_rule", "Run a red light"),
    SelectOption("jaywalking", "Jaywalk"),
    SelectOption("lane_change_abort", "Abort a lane change"),
    SelectOption("head_on", "Head-on"),
    SelectOption("running_red_light", "Cross on red"),
    SelectOption("stop_crossing", "Stop while crossing"),
    SelectOption("merge", "Cut off a merge"),
)

_CONSTRUCTION_SETTINGS = (
    "construction_type",
    "spacing",
    "work_zone_offset",
    "taper_length_in",
    "taper_length_out",
    "taper_type",
    "warning_zone_length",
    "buffer_zone_length",
    "termination_zone_length",
    "side",
)

register_action_spec(
    ActionSpec(
        type_id="construction_zone",
        title="Construction Zone",
        category="Adversity",
        builder="build_construction_zone_action",
        target="..actions.construction_zone:ConstructionZoneAction",
        scope="environment",
        visual_kind="continuous",
        builds=(
            BuiltArgument(
                kwarg="settings",
                target="..adversity.construction:ConstructionSettings",
                parts=(
                    BuiltPart(
                        args=tuple((name, name) for name in _CONSTRUCTION_SETTINGS)
                    ),
                ),
            ),
        ),
        fields=(
            FieldSpec(
                name="lanelet_id",
                label="Lanelet",
                kind="lanelet",
                default=0,
                help="The lane the zone is on is measured from this lanelet's start.",
            ),
            FieldSpec(
                name="start_s",
                label="Work starts at",
                kind="number",
                default=100.0,
                unit="m",
                help="Along the lane; the warning signs stand before it.",
            ),
            FieldSpec(
                name="end_s",
                label="Work ends at",
                kind="number",
                default=250.0,
                unit="m",
                help="Along the lane, on past the lanelet's end if need be.",
            ),
            FieldSpec(
                name="start_time",
                label="Appears at",
                kind="number",
                default=0.0,
                unit="s",
                help="From the scenario's start.",
            ),
            FieldSpec(
                name="end_time",
                label="Cleared away at",
                kind="number",
                default=None,
                required=False,
                unit="s",
                help="From the scenario's start; empty: never.",
            ),
            FieldSpec(
                name="construction_type",
                label="Work zone of",
                kind="select",
                default="cone",
                options=(
                    SelectOption("cone", "Cones"),
                    SelectOption("barrier", "Barriers"),
                    SelectOption("mixed", "Cones and barriers"),
                ),
            ),
            FieldSpec(
                name="side",
                label="Closed from",
                kind="select",
                default="right",
                options=(
                    SelectOption("right", "The right"),
                    SelectOption("left", "The left"),
                ),
            ),
            FieldSpec(
                name="spacing",
                label="Spacing",
                kind="number",
                default=20.0,
                unit="m",
                help="Between the objects; 0.7 of it in a taper, 0.8 in the buffer.",
            ),
            FieldSpec(
                name="work_zone_offset",
                label="Work zone offset",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="Off the lane's centreline, positive to the left.",
            ),
            FieldSpec(
                name="taper_length_in",
                label="Entry taper",
                kind="number",
                default=60.0,
                required=False,
                unit="m",
            ),
            FieldSpec(
                name="taper_length_out",
                label="Exit taper",
                kind="number",
                default=30.0,
                required=False,
                unit="m",
            ),
            FieldSpec(
                name="taper_type",
                label="Taper",
                kind="select",
                default="linear",
                required=False,
                options=(
                    SelectOption("linear", "Linear"),
                    SelectOption("curved", "Curved"),
                ),
            ),
            FieldSpec(
                name="warning_zone_length",
                label="Warning zone",
                kind="number",
                default=100.0,
                required=False,
                unit="m",
            ),
            FieldSpec(
                name="buffer_zone_length",
                label="Buffer zone",
                kind="number",
                default=10.0,
                required=False,
                unit="m",
            ),
            FieldSpec(
                name="termination_zone_length",
                label="Termination zone",
                kind="number",
                default=30.0,
                required=False,
                unit="m",
            ),
            FieldSpec(
                name="close_lane",
                label="Close the lane to the traffic backend",
                kind="bool",
                default=True,
                help="SUMO routes no traffic onto it; the TrafficManager cannot.",
            ),
            FieldSpec(
                name="clear_traffic",
                label="Take background traffic in the zone off the road",
                kind="bool",
                default=True,
            ),
        ),
        description=(
            "TeraSim's construction zone: warning signs, an entry taper of "
            "cones, a buffer, the work zone, an exit taper and termination "
            "signs along a lane, from one time to another."
        ),
    )
)

_NADE_SETTINGS = (
    "max_controlled",
    "control_radius",
    "magnitude",
    "max_is_probability",
    "fixed_is_probability",
    "cooldown",
    "challenge_npcs",
)

register_action_spec(
    ActionSpec(
        type_id="adversarial_traffic",
        title="Adversarial Traffic (NADE)",
        category="Adversity",
        builder="build_adversarial_traffic_action",
        target="..actions.adversarial_traffic:AdversarialTrafficAction",
        scope="environment",
        visual_kind="continuous",
        builds=(
            BuiltArgument(
                kwarg="settings",
                target="..adversity.nade:NadeSettings",
                parts=(BuiltPart(args=tuple((name, name) for name in _NADE_SETTINGS)),),
            ),
        ),
        fields=(
            FieldSpec(
                name="adversities",
                label="Adversities",
                kind="record_list",
                default=(),
                help=(
                    "In order: an agent carries out the first whose trigger "
                    "fires where it is."
                ),
                item_fields=(
                    FieldSpec(
                        "kind", "Kind", "select", "brake_check", _ADVERSITY_KINDS
                    ),
                    FieldSpec(
                        "location", "Where", "select", "any", _ADVERSITY_LOCATIONS
                    ),
                    FieldSpec(
                        "probability",
                        "Probability",
                        "number",
                        0.001,
                        help="Naturalistic, a tick, when its trigger holds.",
                    ),
                ),
            ),
            FieldSpec(name="seed", label="Seed", kind="int", default=0),
            FieldSpec(
                name="max_controlled",
                label="Vehicles considered",
                kind="int",
                default=5,
                help="The nearest this many vehicles around the ego.",
            ),
            FieldSpec(
                name="control_radius",
                label="Around the ego",
                kind="number",
                default=50.0,
                unit="m",
            ),
            FieldSpec(
                name="magnitude",
                label="IS magnitude",
                kind="number",
                default=20.0,
                help="q = clip(p x magnitude, 0, the most q).",
            ),
            FieldSpec(
                name="max_is_probability",
                label="Most q",
                kind="number",
                default=0.05,
            ),
            FieldSpec(
                name="fixed_is_probability",
                label="Fixed q",
                kind="number",
                default=0.0,
                help="Above 0, every adversity's q, in place of the clip.",
            ),
            FieldSpec(
                name="cooldown",
                label="Cooldown",
                kind="number",
                default=5.5,
                unit="s",
                help="After an adversity, how long before the next.",
            ),
            FieldSpec(
                name="challenge_npcs",
                label="Challenge NPCs too",
                kind="bool",
                default=False,
                help=(
                    "Not the ego only: an NPC victim avoids or accepts the "
                    "adversity, its own weight on the run's."
                ),
            ),
            FieldSpec(
                name="following_mode",
                label="Following mode",
                kind="select",
                default="follow",
                options=_FOLLOWING_MODES,
            ),
        ),
        description=(
            "TeraSim's NADE: every tick, the NPC or pedestrian near the ego "
            "whose adversity would challenge it carries it out, sampled with "
            "an importance-sampling probability, and the run keeps the weight "
            "that unbiases its statistics. One adversity at a time."
        ),
    )
)

register_action_spec(
    ActionSpec(
        type_id="set_speed",
        title="Set Speed",
        category="Vehicle / Motion",
        builder="build_set_speed_action",
        target="..actions:SetSpeedAction",
        visual_kind="continuous",
        fields=(
            FieldSpec(
                name="target_speed_kmh",
                label="Target speed",
                kind="number",
                default=30.0,
                unit="km/h",
                help=(
                    "A target, not a jump: whatever drives the vehicle gets "
                    "it there under its own acceleration limits."
                ),
            ),
            FieldSpec(
                name="rate_kmh_s",
                label="Rate",
                kind="number",
                default=None,
                required=False,
                unit="km/h/s",
                help=(
                    "Most the speed may change per second.  Leave empty to "
                    "command the new speed at once."
                ),
            ),
        ),
        description=(
            "Command a vehicle to drive at a new speed.  Given a rate, the "
            "card stays running until the vehicle has actually reached the "
            "speed, so another action can wait for the manoeuvre rather than "
            "for the command."
        ),
    )
)

# The traffic's parameters (TrafficParametersAction): one field per number, a
# mean and a std per parameter, assembled into the parameter objects the action
# takes by traffic.authoring.
_VEHICLE_SPREADS: tuple[tuple[str, str, str, str], ...] = (
    ("speed_factor", "Speed factor", "", "Desired speed as a multiple of the limit."),
    ("min_gap_m", "Gap when standing", "m", "Gap kept to the vehicle ahead."),
    ("ignore_lights", "Run red lights", "", "Probability (0-1); traffic_manager."),
    ("ignore_signs", "Ignore stop signs", "", "Probability (0-1); traffic_manager."),
    ("ignore_vehicles", "Ignore vehicles", "", "Probability (0-1); traffic_manager."),
    ("ignore_walkers", "Ignore pedestrians", "", "Probability (0-1); traffic_manager."),
    (
        "random_lane_changes",
        "Random lane changes",
        "",
        "Probability (0-1); traffic_manager.",
    ),
    ("accel", "Acceleration", "m/s^2", "sumo."),
    ("decel", "Braking", "m/s^2", "sumo; also the visibility's."),
    ("reaction_time_s", "Reaction time", "s", "sumo; also the visibility's."),
    ("imperfection", "Imperfection", "", "Driver imperfection (0-1); sumo."),
    ("max_lateral_speed_mps", "Max lateral speed", "m/s", "sumo."),
    (
        "max_lateral_accel",
        "Max lateral acceleration",
        "m/s^2",
        "sumo, with its sublane model (lateral_resolution_m).",
    ),
)
_PEDESTRIAN_SPREADS: tuple[tuple[str, str, str, str], ...] = (
    ("walking_speed_mps", "Walking speed", "m/s", "sumo."),
    ("pedestrian_min_gap_m", "Pedestrian gap", "m", "Gap to the person ahead; sumo."),
    (
        "red_light_violation_s",
        "Cross on red within",
        "s",
        "Set before the run only (traffic.pedestrians); sumo.",
    ),
)
# The bicycles' fields are all ``bicycle_*``, apart from the vehicles' and the
# pedestrians' of the same name on the same form.
_BICYCLE_SPREADS: tuple[tuple[str, str, str, str], ...] = (
    (
        "bicycle_max_speed_mps",
        "Bicycle speed",
        "m/s",
        "Set before the run only (traffic.bicycles); sumo.",
    ),
    ("bicycle_accel", "Bicycle acceleration", "m/s^2", "sumo."),
    ("bicycle_decel", "Bicycle braking", "m/s^2", "sumo."),
    ("bicycle_min_gap_m", "Bicycle gap", "m", "Gap to the one ahead; sumo."),
    ("bicycle_reaction_time_s", "Rider reaction time", "s", "sumo."),
    ("bicycle_imperfection", "Rider imperfection", "", "0-1; sumo."),
    ("bicycle_max_lateral_speed_mps", "Bicycle lateral speed", "m/s", "sumo."),
    (
        "bicycle_red_light_violation_s",
        "Ride on red within",
        "s",
        "Set before the run only (traffic.bicycles); sumo.",
    ),
)


def _spread_fields(
    spreads: "tuple[tuple[str, str, str, str], ...]",
) -> "tuple[FieldSpec, ...]":
    """A mean and a std field per parameter, both optional."""
    fields: list[FieldSpec] = []
    for name, label, unit, note in spreads:
        fields.append(
            FieldSpec(
                name=f"{name}_mean",
                label=f"{label} (mean)",
                kind="number",
                default=None,
                required=False,
                unit=unit,
                help=f"{note}  Leave empty to keep the run's.",
            )
        )
        fields.append(
            FieldSpec(
                name=f"{name}_std",
                label=f"{label} (std)",
                kind="number",
                default=None,
                required=False,
                unit=unit,
                help="Spread between road users; empty or 0: all the same.",
            )
        )
    return tuple(fields)


def _spread_args(
    spreads: "tuple[tuple[str, str, str, str], ...]",
) -> "tuple[tuple[str, str], ...]":
    return tuple(
        (f"{name}_{stat}", f"{name}_{stat}")
        for name, _, _, _ in spreads
        for stat in ("mean", "std")
    )


_TRAFFIC_BOOLEANS = (
    FieldSpec(
        name="lane_changes",
        label="Lane changes of their own accord",
        kind="bool",
        default=None,
        required=False,
        help="Empty: keep the run's.",
    ),
    FieldSpec(
        name="follow_visibility",
        label="Slow to what drivers see",
        kind="bool",
        default=None,
        required=False,
        help="traffic_manager, sumo.  Empty: keep the run's.",
    ),
)
_VEHICLES_ARGUMENT = BuiltArgument(
    kwarg="vehicles",
    target="..traffic.authoring:authored_vehicle_parameters",
    parts=(
        BuiltPart(
            args=_spread_args(_VEHICLE_SPREADS)
            + (
                ("lane_changes", "lane_changes"),
                ("follow_visibility", "follow_visibility"),
            ),
        ),
    ),
)
_TRAFFIC_PARAMETERS_HELP = (
    "Each road user draws its own value of a parameter from a normal "
    "distribution of the mean and std given; the fields left empty keep their "
    "values. A field the run's traffic backend cannot change during a run is "
    "refused before the run starts."
)

register_action_spec(
    ActionSpec(
        type_id="traffic_parameters",
        title="Traffic Parameters",
        category="Traffic",
        builder="build_traffic_parameters_action",
        target="..actions:TrafficParametersAction",
        scope="environment",
        visual_kind="instant",
        builds=(
            _VEHICLES_ARGUMENT,
            BuiltArgument(
                kwarg="pedestrians",
                target="..traffic.authoring:authored_pedestrian_parameters",
                parts=(BuiltPart(args=_spread_args(_PEDESTRIAN_SPREADS)),),
            ),
            BuiltArgument(
                kwarg="bicycles",
                target="..traffic.authoring:authored_bicycle_parameters",
                parts=(BuiltPart(args=_spread_args(_BICYCLE_SPREADS)),),
            ),
        ),
        fields=_spread_fields(_VEHICLE_SPREADS)
        + _TRAFFIC_BOOLEANS
        + _spread_fields(_PEDESTRIAN_SPREADS)
        + _spread_fields(_BICYCLE_SPREADS),
        description=(
            "Change the run's traffic -- every vehicle, pedestrian and bicycle -- from "
            "this point on: they draw the changed parameters again. "
            + _TRAFFIC_PARAMETERS_HELP
        ),
    )
)

register_action_spec(
    ActionSpec(
        type_id="vehicle_parameters",
        title="Vehicle Parameters",
        category="Vehicle / Motion",
        builder="build_vehicle_parameters_action",
        target="..actions:TrafficParametersAction",
        # An NPC: the ego is driven by what it is (Autoware, a policy), not by
        # the traffic backend's driver model.
        actor_kinds=("vehicle",),
        visual_kind="instant",
        builds=(_VEHICLES_ARGUMENT,),
        fields=_spread_fields(_VEHICLE_SPREADS) + _TRAFFIC_BOOLEANS,
        description=(
            "Change one vehicle's driving from this point on: it draws the "
            "changed parameters again, and the rest of the traffic keeps its. "
            + _TRAFFIC_PARAMETERS_HELP
        ),
    )
)

#: Mirrors ``RoutingMode``; ``test_authoring_registry`` keeps the two in step.
_ROUTING_MODES: tuple[SelectOption, ...] = (
    SelectOption("driver", "Driver -- its own driver, Lanelet2 routes between"),
    SelectOption("follow", "Follow -- a controller, straight legs between"),
    SelectOption("position", "Position -- placed on the legs every tick"),
)

register_action_spec(
    ActionSpec(
        type_id="routing",
        title="Set Goal",
        category="Vehicle / Motion",
        builder="build_routing_action",
        target="..actions:RoutingAction",
        argmap=(("speed_ms", "speed"),),
        actor_kinds=("ego", "vehicle", "pedestrian"),
        visual_kind="continuous",
        vertex_conditions=True,
        # The route is a goal and via lanelets, or written-out waypoints, and
        # the time reference is several fields: what the constructor's
        # signature alone cannot say.
        builds=(
            BuiltArgument(
                kwarg="goal",
                target="..trajectory.authoring:authored_goal",
                parts=(
                    BuiltPart(
                        args=(
                            ("path_source", "path_source"),
                            ("lanelet_id", "goal_lanelet_id"),
                            ("s", "goal_s"),
                        ),
                    ),
                ),
            ),
            BuiltArgument(
                kwarg="waypoints",
                target="..trajectory.authoring:authored_waypoints",
                parts=(
                    BuiltPart(
                        args=(
                            ("path_source", "path_source"),
                            ("lanelet_ids", "waypoint_lanelet_ids"),
                        ),
                    ),
                ),
            ),
            BuiltArgument(
                kwarg="trajectory",
                target="..trajectory.authoring:authored_route",
                parts=(
                    BuiltPart(
                        args=(
                            ("path_source", "path_source"),
                            ("vertices", "vertices"),
                            ("lanelet_ids", "lanelet_ids"),
                            ("speed_kmh", "speed_kmh"),
                            ("lateral_offset_m", "lateral_offset_m"),
                            ("relative_vertices", "relative_vertices"),
                            ("reference_entity", "reference_entity"),
                            ("route_vertices", "route_vertices"),
                        ),
                        sources=(("advance", "advance_conditions"),),
                    ),
                ),
            ),
            BuiltArgument(
                kwarg="time_reference",
                target="..trajectory.authoring:authored_timing",
                parts=(
                    BuiltPart(
                        args=(
                            ("time_domain", "time_domain"),
                            ("scale", "time_scale"),
                            ("offset", "time_offset"),
                        ),
                    ),
                ),
            ),
        ),
        # A goal is what an ego needs in order to become ready, and the runner
        # waits for ready before the loop starts.  Landing this card on the tick
        # loop would deliver the goal to a stack nobody is listening to any more.
        default_phase="init",
        fields=(
            FieldSpec(
                name="path_source",
                label="Path",
                kind="select",
                default="goal",
                options=(
                    SelectOption("goal", "Goal lanelet, via lanelets"),
                    SelectOption("vertices", "Vertices (map frame)"),
                    SelectOption("lanelets", "Along lanelets"),
                    SelectOption("relative_lane", "Relative to an entity's lane"),
                    SelectOption("route", "Along the scenario's route (logical)"),
                ),
                help=(
                    "A goal is a lanelet to go to, through the via lanelets. "
                    "Vertices are written out below -- a recording transcribes "
                    "to them; a lanelet path follows the centrelines of the "
                    "lanelets picked on the map; relative vertices are lanes "
                    "and metres from where an entity is when the action starts; "
                    "route vertices are placed against the route the "
                    "scenario's route search found, on any map."
                ),
            ),
            FieldSpec(
                name="goal_lanelet_id",
                place="goal",
                label="Goal lanelet",
                kind="lanelet",
                default=None,
                required=False,
                help=(
                    "Where this vehicle is being sent. An Autoware ego plans "
                    "its route there, and will not move without a goal; a "
                    "vehicle the traffic backend drives, ego or NPC, is handed "
                    "the route there; a driver-policy ego ignores it. The "
                    "ego's own goal is set on the ego itself, beside its spawn."
                ),
            ),
            FieldSpec(
                name="goal_s",
                label="s",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="Along the goal lanelet from its start.",
            ),
            FieldSpec(
                name="waypoint_lanelet_ids",
                label="Via lanelets",
                kind="lanelet_list",
                default=[],
                required=False,
                help=(
                    "Lanelets the route passes through on the way, in order. "
                    "Empty: the shortest way to the goal."
                ),
            ),
            FieldSpec(
                name="mode",
                label="Driven by",
                kind="select",
                default="driver",
                options=_ROUTING_MODES,
                help=(
                    "Driver: the entity's own driver (the traffic backend, the "
                    "pedestrian route walker, Autoware) drives Lanelet2 routes "
                    "between the waypoints, its speed set to meet their times. "
                    "Follow: a controller drives straight legs between them. "
                    "Position: placed on the legs every tick (a replay)."
                ),
            ),
            FieldSpec(
                name="vertices",
                label="Vertices",
                kind="trajectory",
                default=None,
                required=False,
                help=(
                    "One vertex per line: x, y[, yaw] in the map frame (m, "
                    "rad).  Leave yaw empty to face along the path.  A vertex's "
                    "time is its Trajectory time waypoint condition, below."
                ),
            ),
            FieldSpec(
                name="lanelet_ids",
                label="Lanelets",
                kind="lanelet_list",
                default=None,
                required=False,
                help="The route, in the order it is driven.",
            ),
            FieldSpec(
                name="speed_kmh",
                label="Path speed",
                kind="number",
                default=30.0,
                required=False,
                unit="km/h",
                help="Times the lanelet path; empty or 0 leaves it untimed.",
            ),
            FieldSpec(
                name="lateral_offset_m",
                label="Lateral offset",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="From the lanelets' centreline, positive to the left.",
            ),
            FieldSpec(
                name="reference_entity",
                label="Relative to",
                kind="entity",
                default=None,
                required=False,
                help=(
                    "The entity relative vertices are measured from; leave "
                    "empty for the entity that follows the trajectory."
                ),
            ),
            FieldSpec(
                name="relative_vertices",
                label="Relative vertices",
                kind="relative_lane_trajectory",
                default=None,
                required=False,
                help=(
                    "One vertex per line: ds[, offset][, d_lane][, yaw] -- "
                    "metres along the reference's lane (negative behind), "
                    "metres from the lane centre (positive left), lanes across "
                    "(+1 left, -1 right), heading from the lane (rad, positive "
                    "left; empty faces along the path).  Times are waypoint "
                    "conditions, below.  Placed when the action starts; whether "
                    "the lanes exist is only known then."
                ),
            ),
            FieldSpec(
                name="route_vertices",
                label="Route vertices",
                kind="route_trajectory",
                default=None,
                required=False,
                help=(
                    "One vertex per line, 'kind key=value ...': "
                    "lane ds= offset= d_lane= yaw= anchor= | "
                    "opposite ds= lane= offset= yaw= anchor= | "
                    "crossing junction= approach=same|left|right|opposite distance= "
                    "turn= offset= yaw= | "
                    "crosswalk junction= leg=entry|exit side=left|right along= "
                    "yaw= | roadside ds= side= kerb_distance= yaw= anchor=.  "
                    "ds counts from the anchor (start, end, segment:K:start|end, "
                    "junction:K:entry|exit) or, without one, from the ego.  "
                    "Placed when the action starts."
                ),
            ),
            FieldSpec(
                name="speed_ms",
                label="Speed where no time paces",
                kind="number",
                default=None,
                required=False,
                unit="m/s",
                help=(
                    "Where no waypoint time paces the entity: without a time "
                    "reference, and after a waypoint condition once the next "
                    "time has passed. Empty: its driver's own speed (Driver), "
                    "the speed it has when the action starts (Follow, Position)."
                ),
            ),
            FieldSpec(
                name="time_domain",
                label="Time reference",
                kind="select",
                default="none",
                options=(
                    SelectOption("relative", "From when the action starts"),
                    SelectOption("absolute", "From when the scenario starts"),
                    SelectOption("none", "None -- keep the entity's own speed"),
                ),
                help=(
                    "A vertex at time t is reached at t * scale + offset, "
                    "counted from the start this names."
                ),
            ),
            FieldSpec(
                name="time_scale",
                label="Time scale",
                kind="number",
                default=1.0,
                help="2 plays the trajectory at half speed.",
            ),
            FieldSpec(
                name="time_offset",
                label="Time offset",
                kind="number",
                default=0.0,
                unit="s",
            ),
            FieldSpec(
                name="initial_distance_offset",
                label="Start this far along",
                kind="number",
                default=0.0,
                unit="m",
            ),
            *_TURN_BACK_FIELDS,
            FieldSpec(
                name="cross_factor",
                label="Crosses away from crossings",
                kind="number",
                default=None,
                required=False,
                help=(
                    "For a pedestrian walked by CARLA's walker controller: the "
                    "probability (0 to 1) that its route may cross a road away "
                    "from a crossing -- 1 for a recorded walk that did, 0 keeps "
                    "it to crossings. Empty is CARLA's default (0)."
                ),
            ),
            FieldSpec(
                name="walking_speed",
                label="Walking speed",
                kind="number",
                default=1.4,
                required=False,
                unit="m/s",
                help="How fast a pedestrian walks the route. A vehicle ignores it.",
            ),
        ),
        description=(
            "Send a vehicle or pedestrian to a goal, through waypoints, on "
            "time. The route is a goal lanelet with via lanelets, or "
            "waypoints written out -- map poses, lanelets, poses relative to "
            "an entity's lane or to the scenario's route -- each departed on "
            "its time, on a waypoint condition, or on arrival. Driver mode "
            "hands the route to the entity's own driver and sets its speed to "
            "meet the times (an Autoware ego takes the route only); Follow and "
            "Position move the entity along straight legs themselves. The "
            "ego's own destination is set on the ego, beside its spawn: a "
            "card is for changing it mid-run, or for another road user."
        ),
    )
)

register_action_spec(
    ActionSpec(
        type_id="traffic_signal",
        title="Set Traffic Signal",
        category="Environment",
        builder="build_traffic_signal_action",
        target="..actions:TrafficSignalAction",
        choices=(
            ArgumentChoice(
                kwarg="lanelet2_traffic_light_ids",
                discriminator="target",
                cases=(
                    ArgumentCase("all", constant="..actions:TrafficLightTarget.ALL"),
                    ArgumentCase("ids", field="lanelet2_traffic_light_ids"),
                ),
            ),
        ),
        scope="environment",
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

register_action_spec(
    ActionSpec(
        type_id="traffic_signal_controller",
        title="Set Junction Phase",
        category="Environment",
        builder="build_traffic_signal_controller_action",
        target="..actions:TrafficSignalControllerAction",
        scope="environment",
        visual_kind="instant",
        default_phase="init",
        # Named `signal_phase` in the document, `phase` on the class.  The
        # action inspector posts its own `phase` field -- the tick phase every
        # action has -- into the same flat form namespace, so a card param
        # called `phase` is a second input of that name: the tick phase would
        # land in the params and the signal phase would be offered to
        # `as_action_phase`, which rejects it silently.  The constructor has no
        # such clash, so only the document-side name moves.
        argmap=(("signal_phase", "phase"),),
        fields=(
            FieldSpec(
                name="controller",
                label="Controller",
                kind="signal_controller",
                default="",
                help=(
                    "A controller this scenario declares under "
                    "map.traffic_signal_controllers.  Its phases, and how long "
                    "each holds, are declared there too -- the timing is this "
                    "scenario's, while map.signal_groups says which signals "
                    "move together."
                ),
            ),
            FieldSpec(
                name="signal_phase",
                label="Signal phase",
                kind="signal_phase",
                default="",
                help=(
                    "Which of that controller's phases to show.  The cycle "
                    "carries on from it, so this starts the junction at a "
                    "point rather than holding it there.  Not the tick phase "
                    "below, which is when this card runs."
                ),
            ),
        ),
        description=(
            "Jump a junction's controller to one of the phases this scenario "
            "declares.  A phase names every signal the controller drives, so "
            "the junction goes to a consistent state rather than one light at "
            "a time."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="all",
        title="ALL",
        category="Composition",
        builder="build_and_condition",
        target="..conditions:AndCondition",
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
        target="..conditions:OrCondition",
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
        target="..conditions:NotCondition",
        kind="wrapper",
        visual=ConditionVisual(metric="NOT"),
        fields=(
            FieldSpec(
                name="holds_when_pending",
                label="Holds while the child does not",
                kind="bool",
                default=False,
                required=False,
                help=(
                    "A logical NOT: hold on every tick the child has not "
                    "fired, rather than flip the verdict of a child that has. "
                    "Under ANY of lanelet positions, it says the entity is on "
                    "none of them -- off its lanes, or off the road."
                ),
            ),
        ),
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
        target="..conditions:StickyCondition",
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
        target="..conditions:PersistentCondition",
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
        target="..conditions:EntityDistanceCondition",
        argmap=(("distance", "value"), ("measure", "coordinate_system")),
        visual=ConditionVisual(
            metric="Distance",
            subject="source",
            target="target",
            rule="rule",
            value="distance",
            unit="m",
            # Both of these change what the threshold means, so a card that
            # left them out drew a longitudinal edge-to-edge gap and a
            # Euclidean centre distance identically.
            details=("measure", "distance_type", "edge_to_edge"),
        ),
        fields=(
            _entity_field("source", "Subject"),
            _entity_field("target", "Target"),
            _rule_field(),
            FieldSpec(
                name="distance", label="Distance", kind="number", default=20.0, unit="m"
            ),
            _coordinate_system_field(),
            FieldSpec(
                name="distance_type",
                label="Component",
                kind="select",
                default="EUCLIDEAN",
                options=(
                    SelectOption("EUCLIDEAN", "Straight line"),
                    SelectOption("LONGITUDINAL", "Along the subject"),
                    SelectOption("LATERAL", "Across the subject"),
                ),
                required=False,
                help=(
                    "Which part of the straight-line separation is meant.  A "
                    "car in the next lane is 20 m away in a straight line and "
                    "2 m away longitudinally, and following-distance and "
                    "cut-in scenarios mean the second.  Along the road is "
                    "one-dimensional, so this is only read when Measured is "
                    "a straight line."
                ),
            ),
            _edge_to_edge_field(),
            FieldSpec(
                name="signed",
                label="Ahead (+) or behind (-)",
                kind="bool",
                default=False,
                required=False,
                help=(
                    "Along the road only: how far the target is ahead of the "
                    "subject, the way the subject faces, negative behind it -- "
                    "so -30 with at most reads at least 30 m behind."
                ),
            ),
        ),
        description=(
            "Distance from the subject to the target.  Say which frame and "
            "component are meant and whether the vehicles' extents count; "
            "the defaults are a straight line between centres."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="distance",
        title="Distance to a place",
        category="Relative",
        builder="build_distance_condition",
        target="..conditions:EntityPositionDistanceCondition",
        argmap=(("entity", "entity_name"), ("distance", "value")),
        builds=(
            BuiltArgument(
                kwarg="position",
                target="..coordinate:Lanelet2Pose",
                parts=(BuiltPart(args=(("lanelet_id", "lanelet_id"), ("s", "s"))),),
            ),
        ),
        visual=ConditionVisual(
            metric="Distance",
            subject="entity",
            target="lanelet_id",
            target_prefix="Lanelet",
            rule="rule",
            value="distance",
            unit="m",
            details=("s",),
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="lanelet_id",
                label="Lanelet",
                kind="lanelet",
                default=0,
                help=(
                    "The lanelet holding the place to measure to.  Unlike "
                    "Position (Lanelet2), which asks whether the entity is on "
                    "a lane, this measures to a point: say where along the "
                    "lanelet with s."
                ),
            ),
            FieldSpec(
                name="s",
                label="s",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="Along the lanelet from its start.",
            ),
            _rule_field(),
            FieldSpec(
                name="distance",
                label="Distance",
                kind="number",
                default=20.0,
                unit="m",
            ),
        ),
        description=(
            "Distance from an entity to a place on the map -- a stop line, a "
            "conflict point.  Use Distance for the gap between two vehicles."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="ttc",
        title="TTC",
        category="Relative",
        builder="build_ttc_condition",
        target="..conditions:TimeToCollisionCondition",
        argmap=(("seconds", "value"), ("measure", "coordinate_system")),
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
            _rule_field(),
            FieldSpec(
                name="seconds", label="TTC", kind="number", default=4.0, unit="s"
            ),
            _coordinate_system_field(),
            _edge_to_edge_field(),
        ),
        description=(
            "Time to collision from the subject to the target.  Measured "
            "along the road, both the gap and the speed closing it are "
            "resolved onto the road, so a vehicle's cornering is not counted "
            "as approach."
        ),
    )
)


# A separate condition rather than a target that may be either, for the reason
# the two position conditions are separate: a field accepting an entity name or
# a lanelet id is how a bare id comes to mean two things.
register_condition_spec(
    ConditionSpec(
        type_id="ttc_to_position",
        title="TTC to a place",
        category="Relative",
        builder="build_ttc_to_position_condition",
        target="..conditions:TimeToCollisionCondition",
        argmap=(("entity", "source"), ("seconds", "value")),
        builds=(
            BuiltArgument(
                kwarg="position",
                target="..coordinate:Lanelet2Pose",
                parts=(BuiltPart(args=(("lanelet_id", "lanelet_id"), ("s", "s"))),),
            ),
        ),
        visual=ConditionVisual(
            metric="TTC",
            subject="entity",
            target="lanelet_id",
            target_prefix="Lanelet",
            rule="rule",
            value="seconds",
            unit="s",
            details=("s",),
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="lanelet_id",
                label="Lanelet",
                kind="lanelet",
                default=0,
                help="The lanelet holding the place the subject is closing on.",
            ),
            FieldSpec(
                name="s",
                label="s",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="Along the lanelet from its start.",
            ),
            _rule_field(),
            FieldSpec(
                name="seconds", label="TTC", kind="number", default=4.0, unit="s"
            ),
            _edge_to_edge_field(),
        ),
        description=(
            "Time until the subject reaches a place on the map -- a stop "
            "line, a conflict point.  A place does not move, so this is the "
            "subject's own speed along the line of sight."
        ),
    )
)


# ---------------------------------------------------------------------------
# Built-in conditions -- single entity
# ---------------------------------------------------------------------------

register_condition_spec(
    ConditionSpec(
        type_id="time_headway",
        title="Time headway",
        category="Relative",
        builder="build_time_headway_condition",
        target="..conditions:TimeHeadwayCondition",
        argmap=(
            ("entity", "source"),
            ("seconds", "value"),
            ("measure", "coordinate_system"),
        ),
        visual=ConditionVisual(
            metric="Headway",
            subject="entity",
            target="target",
            rule="rule",
            value="seconds",
            unit="s",
        ),
        fields=(
            _entity_field("entity", "Follower"),
            _entity_field("target", "Vehicle ahead"),
            _rule_field(),
            FieldSpec(
                name="seconds", label="Headway", kind="number", default=2.0, unit="s"
            ),
            _coordinate_system_field(),
        ),
        description=(
            "How long the follower would take to reach where the vehicle "
            "ahead is now, at its own current speed.  Unlike TTC this is "
            "defined even when the gap is steady, which is what makes it the "
            "following-distance measure; it is undefined, and never fires, "
            "while the follower is stopped.  Measured in a straight line by "
            "default, which on a curve under-reads and past a quarter turn "
            "stops firing entirely; measure along the road to avoid that."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="speed",
        title="Speed",
        category="Entity",
        builder="build_speed_condition",
        target="..conditions:SpeedCondition",
        argmap=(("entity", "entity_name"),),
        visual=ConditionVisual(
            metric="Speed",
            subject="entity",
            rule="rule",
            value="value",
            unit="m/s",
        ),
        fields=(
            _entity_field("entity", "Subject"),
            _rule_field("greater_than"),
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
        type_id="route_progress",
        title="Route progress",
        category="Entity",
        builder="build_route_progress_condition",
        target="..conditions:RouteProgressCondition",
        argmap=(("entity", "entity_name"),),
        visual=ConditionVisual(
            metric="Route progress",
            subject="entity",
            rule="rule",
            value="value",
            unit="m",
            details=("anchor",),
        ),
        fields=(
            _entity_field("entity", "Subject", required=False),
            _rule_field("greater_than_or_equal"),
            FieldSpec(
                name="value",
                label="Distance",
                kind="number",
                default=0.0,
                unit="m",
                help="Past the anchor along the route; negative is before it.",
            ),
            FieldSpec(
                name="anchor",
                label="From",
                kind="text",
                default="start",
                required=False,
                help=(
                    "start, end, segment:K:start, segment:K:end, "
                    "junction:K:entry or junction:K:exit (K from 0)."
                ),
            ),
        ),
        description=(
            "How far an entity (the ego when none is named) has come along the "
            "scenario's route, measured on the route whichever lane it is in. "
            "Needs the scenario's route search.  Usable as a trigger and as a "
            "waypoint condition."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="acceleration",
        title="Acceleration",
        category="Entity",
        builder="build_acceleration_condition",
        target="..conditions:AccelerationCondition",
        argmap=(("entity", "entity_name"),),
        visual=ConditionVisual(
            metric="Acceleration",
            subject="entity",
            rule="rule",
            value="value",
            unit="m/s²",
        ),
        fields=(
            _entity_field("entity", "Subject"),
            _rule_field(),
            FieldSpec(
                name="value",
                label="Acceleration",
                kind="number",
                default=-3.0,
                unit="m/s²",
            ),
            FieldSpec(
                name="direction",
                label="Component",
                kind="select",
                default="LONGITUDINAL",
                options=(
                    SelectOption("MAGNITUDE", "Magnitude"),
                    SelectOption("LONGITUDINAL", "Longitudinal"),
                    SelectOption("LATERAL", "Lateral"),
                ),
                required=False,
                help=(
                    "Longitudinal is signed in the entity's own frame: "
                    "negative is braking, which is what a harsh-braking "
                    "threshold is about.  Magnitude is never negative, so a "
                    '"less than -3" rule on it can never fire.'
                ),
            ),
        ),
        description=(
            "Acceleration of a single entity.  Read from CARLA directly, not "
            "differentiated from speed."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="relative_speed",
        title="Relative speed",
        category="Relative",
        builder="build_relative_speed_condition",
        target="..conditions:RelativeSpeedCondition",
        argmap=(
            ("entity", "entity_name"),
            ("reference", "reference_entity_name"),
        ),
        visual=ConditionVisual(
            metric="Relative speed",
            subject="entity",
            target="reference",
            rule="rule",
            value="value",
            unit="m/s",
        ),
        fields=(
            _entity_field("entity", "Subject"),
            _entity_field("reference", "Relative to"),
            _rule_field(),
            FieldSpec(
                name="value",
                label="Relative speed",
                kind="number",
                default=-5.0,
                unit="m/s",
            ),
            FieldSpec(
                name="direction",
                label="Component",
                kind="select",
                default="LONGITUDINAL",
                options=(
                    SelectOption("MAGNITUDE", "Magnitude"),
                    SelectOption("LONGITUDINAL", "Longitudinal"),
                    SelectOption("LATERAL", "Lateral"),
                ),
                required=False,
                help=(
                    "All three are signed, and negative means the subject is "
                    "the slower of the two.  Magnitude is the difference of "
                    "the two speeds; longitudinal and lateral are components "
                    "of the velocity difference in the reference entity's "
                    "frame."
                ),
            ),
        ),
        description=(
            "Speed of one entity relative to another: the subject's velocity "
            "minus the reference's."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="standstill",
        title="Standstill",
        category="Entity",
        builder="build_standstill_condition",
        target="..conditions:StandstillCondition",
        argmap=(("entity", "entity_name"),),
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
        type_id="turn_indicator",
        title="Turn Indicator",
        category="Entity",
        builder="build_turn_indicator_condition",
        target="..conditions:TurnIndicatorCondition",
        argmap=(("entity", "entity_name"),),
        visual=ConditionVisual(
            metric="Turn indicator", subject="entity", value="state"
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="state",
                label="Indicators",
                kind="select",
                default="LEFT",
                options=_TURN_INDICATOR_STATES,
            ),
        ),
        description=(
            "The vehicle shows these turn indicators -- however they were lit, "
            "a driver policy's command included."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="lane_change_settled",
        title="Lane Change Settled",
        category="Entity",
        builder="build_lane_change_settled_condition",
        target="..conditions:LaneChangeSettledCondition",
        argmap=(("entity", "entity_name"),),
        visual=ConditionVisual(metric="Lane change settled", subject="entity"),
        fields=(_entity_field("entity", "Subject"),),
        description=(
            "The entity has settled onto the lane it was sent to.  This is how "
            "a Lane Change card knows it is over, and an author can wait on "
            "the same thing."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="entity_lane_position",
        title="Position (Lanelet2)",
        category="Entity",
        builder="build_entity_lane_position_condition",
        target="..conditions:EntityLanePositionCondition",
        argmap=(("entity", "entity_name"),),
        builds=(
            BuiltArgument(
                kwarg="position",
                target="..coordinate:Lanelet2Pose",
                parts=(
                    BuiltPart(
                        args=(("lanelet_id", "lanelet_id"),), constants=(("s", "0.0"),)
                    ),
                ),
            ),
            # The editor offers a *range* because that is what an author means
            # by "over this stretch"; the runtime takes one-sided rules, so each
            # bound that is set becomes one and an unset bound becomes nothing.
            BuiltArgument(
                kwarg="rules",
                target="..conditions.comparison:ScalarComparisonRule",
                parts=(
                    BuiltPart(
                        constants=(
                            ("field", '"s"'),
                            (
                                "rule",
                                "..conditions.comparison:"
                                "ComparisonRule.GREATER_THAN_OR_EQUAL",
                            ),
                        ),
                        args=(("value", "s_min"),),
                        when_present="s_min",
                    ),
                    BuiltPart(
                        constants=(
                            ("field", '"s"'),
                            (
                                "rule",
                                "..conditions.comparison:"
                                "ComparisonRule.LESS_THAN_OR_EQUAL",
                            ),
                        ),
                        args=(("value", "s_max"),),
                        when_present="s_max",
                    ),
                    BuiltPart(
                        constants=(
                            ("field", '"t"'),
                            (
                                "rule",
                                "..conditions.comparison:"
                                "ComparisonRule.GREATER_THAN_OR_EQUAL",
                            ),
                        ),
                        args=(("value", "t_min"),),
                        when_present="t_min",
                    ),
                    BuiltPart(
                        constants=(
                            ("field", '"t"'),
                            (
                                "rule",
                                "..conditions.comparison:"
                                "ComparisonRule.LESS_THAN_OR_EQUAL",
                            ),
                        ),
                        args=(("value", "t_max"),),
                        when_present="t_max",
                    ),
                ),
            ),
        ),
        visual=ConditionVisual(
            metric="Position",
            subject="entity",
            target="lanelet_id",
            target_prefix="Lanelet",
            details=("s_min", "s_max", "t_min", "t_max"),
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
                    "The lanelet the entity has to be on, and the frame its "
                    "s and t are read in. A lanelet already names one lane, so "
                    "there is nothing further to pin -- use Position (OpenDRIVE) "
                    "to address a road directly."
                ),
            ),
            *_extent_fields("lanelet"),
        ),
        description=(
            "The entity is on the lane a Lanelet2 lanelet describes, optionally "
            "only over a stretch of it.  Pick this when you are thinking in "
            "lanelets; everything else on the canvas already is."
        ),
    )
)

#: The lanes :class:`EntityLaneOfCondition` can mean.  Spelled out rather than
#: imported -- the conditions package pulls in CARLA -- and checked against the
#: runtime's own ``LANE_RELATIONS`` by ``test_authoring_registry``.
_LANE_RELATIONS: tuple[SelectOption, ...] = (
    SelectOption("same", "Same lane"),
    SelectOption("left", "Lane to its left"),
    SelectOption("right", "Lane to its right"),
    SelectOption("any", "Any lane of its road"),
)

register_condition_spec(
    ConditionSpec(
        type_id="entity_lane_of",
        title="Lane of a position",
        category="Entity",
        builder="build_entity_lane_of_condition",
        target="..conditions:EntityLaneOfCondition",
        argmap=(("entity", "entity_name"),),
        builds=(
            BuiltArgument(
                kwarg="position",
                target="..coordinate:Lanelet2Pose",
                parts=(BuiltPart(args=(("lanelet_id", "lanelet_id"), ("s", "s"))),),
            ),
        ),
        visual=ConditionVisual(
            metric="Lane of",
            subject="entity",
            target="lanelet_id",
            target_prefix="Lanelet",
            details=("s", "lane"),
            value_label="on",
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="lanelet_id",
                label="Lanelet",
                kind="lanelet",
                default=0,
                help="Where the reference position is -- a spawn, a junction's start.",
            ),
            FieldSpec(
                name="s",
                label="At s",
                kind="number",
                default=0.0,
                unit="m",
                help="Along the lanelet, from its start.",
            ),
            FieldSpec(
                name="lane",
                label="Lane",
                kind="select",
                default="same",
                options=_LANE_RELATIONS,
            ),
        ),
        description=(
            "The entity is on the OpenDRIVE lane under a Lanelet2 position, the "
            "lane beside it, or any lane of its road -- anywhere along the "
            "road, not only within the lanelet.  Beside a spawn is where a lane "
            "change from it ends."
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
        target="..conditions:EntityLanePositionCondition",
        argmap=(("entity", "entity_name"),),
        builds=(
            BuiltArgument(
                kwarg="position",
                target="..coordinate:OpenDrivePose",
                parts=(
                    BuiltPart(
                        args=(("road_id", "road_id"), ("lane_id", "lane_id")),
                        constants=(("s", "0.0"),),
                    ),
                ),
            ),
            # The editor offers a *range* because that is what an author means
            # by "over this stretch"; the runtime takes one-sided rules, so each
            # bound that is set becomes one and an unset bound becomes nothing.
            BuiltArgument(
                kwarg="rules",
                target="..conditions.comparison:ScalarComparisonRule",
                parts=(
                    BuiltPart(
                        constants=(
                            ("field", '"s"'),
                            (
                                "rule",
                                "..conditions.comparison:"
                                "ComparisonRule.GREATER_THAN_OR_EQUAL",
                            ),
                        ),
                        args=(("value", "s_min"),),
                        when_present="s_min",
                    ),
                    BuiltPart(
                        constants=(
                            ("field", '"s"'),
                            (
                                "rule",
                                "..conditions.comparison:"
                                "ComparisonRule.LESS_THAN_OR_EQUAL",
                            ),
                        ),
                        args=(("value", "s_max"),),
                        when_present="s_max",
                    ),
                    BuiltPart(
                        constants=(
                            ("field", '"t"'),
                            (
                                "rule",
                                "..conditions.comparison:"
                                "ComparisonRule.GREATER_THAN_OR_EQUAL",
                            ),
                        ),
                        args=(("value", "t_min"),),
                        when_present="t_min",
                    ),
                    BuiltPart(
                        constants=(
                            ("field", '"t"'),
                            (
                                "rule",
                                "..conditions.comparison:"
                                "ComparisonRule.LESS_THAN_OR_EQUAL",
                            ),
                        ),
                        args=(("value", "t_max"),),
                        when_present="t_max",
                    ),
                ),
            ),
        ),
        visual=ConditionVisual(
            metric="Position",
            subject="entity",
            target="road_id",
            target_prefix="Road",
            details=("lane_id", "s_min", "s_max", "t_min", "t_max"),
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
                default=0,
                help=(
                    "Required: in OpenDRIVE a road and a lane together are what "
                    "name a place, so a road on its own does not determine one. "
                    "The runtime keeps a road-only condition for the case where "
                    "the lane genuinely does not matter, but it is a different "
                    "constructor on purpose -- so that 'I could not work out "
                    "the lane' cannot be written the same way."
                ),
            ),
            *_extent_fields("road"),
        ),
        description=(
            "The entity is on an OpenDRIVE road, optionally on one of its lanes "
            "and only over a stretch of it.  Road and lane together are what "
            "address a lane uniquely."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="entity_existence",
        title="Exists",
        category="Entity",
        builder="build_entity_existence_condition",
        target="..conditions:EntityExistenceCondition",
        argmap=(("entity", "entity_name"),),
        visual=ConditionVisual(
            metric="Existence", subject="entity", value_label="missing"
        ),
        fields=(_entity_field("entity", "Subject"),),
        description="Fires while the entity is absent from the world.",
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="off_road_fraction",
        title="Off the road",
        category="Entity",
        builder="build_off_road_fraction_condition",
        target="..conditions:OffRoadFractionCondition",
        argmap=(("entity", "entity_name"),),
        visual=ConditionVisual(
            metric="Off the road",
            subject="entity",
            value="fraction",
            value_label="of the box",
        ),
        fields=(
            _entity_field("entity", "Subject", required=False),
            FieldSpec(
                name="fraction",
                label="Fraction",
                kind="number",
                default=0.3,
                help=(
                    "Of the 15 points of a 5 x 3 grid over the bounding box, "
                    "edges and corners included (not of its area), in (0, 1]: "
                    "1/15 fails on any one point, 1 only when the whole box is "
                    "off."
                ),
            ),
        ),
        description=(
            "Enough of an entity's box (the ego's when none is named) is on no "
            "driving lane.  The pass/fail counterpart of the offroad metric, "
            "which fires on a single corner -- a corner over a lane gap at a "
            "junction entry included.  Usually a fail condition."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="lane_ends",
        title="Lane ends ahead",
        category="Entity",
        builder="build_waypoint_condition",
        target="..conditions:WaypointCondition",
        argmap=(("entity", "entity_name"),),
        visual=ConditionVisual(
            metric="Lane ahead", subject="entity", value_label="none"
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
        description=(
            "No lane continues the given distance ahead -- a dead end, the edge "
            "of the map, or a lane that merges away. It follows the entity's own "
            "lane and never considers a lane change, so the *road* may well go "
            "on without it."
        ),
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
        target="..conditions:ElapsedTimeCondition",
        visual=ConditionVisual(
            metric="Elapsed time", rule="rule", value="duration_seconds", unit="s"
        ),
        fields=(
            _rule_field("greater_than_or_equal"),
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
        type_id="place_clear",
        title="Place clear",
        category="World",
        builder="build_place_clear_condition",
        target="..conditions:PlaceClearCondition",
        builds=(
            BuiltArgument(
                kwarg="position",
                target="..coordinate:Lanelet2Pose",
                parts=(
                    BuiltPart(
                        args=(
                            ("lanelet_id", "lanelet_id"),
                            ("s", "s"),
                            ("t", "t"),
                            ("heading", "heading"),
                        )
                    ),
                ),
            ),
        ),
        visual=ConditionVisual(
            metric="Place clear",
            target="lanelet_id",
            target_prefix="Lanelet",
            details=("s",),
        ),
        fields=(
            FieldSpec(
                name="lanelet_id",
                label="Lanelet",
                kind="lanelet",
                default=0,
                help="The lanelet holding the place's centre.",
            ),
            FieldSpec(
                name="s",
                label="s",
                kind="number",
                default=0.0,
                unit="m",
                help="Along the lanelet from its start.",
            ),
            FieldSpec(
                name="t",
                label="t",
                kind="number",
                default=0.0,
                required=False,
                unit="m",
                help="Off the centreline, positive to the left.",
            ),
            FieldSpec(
                name="heading",
                label="Heading",
                kind="number",
                default=0.0,
                required=False,
                unit="rad",
                help="Turned from the lane's direction, counter-clockwise.",
            ),
            FieldSpec(
                name="length",
                label="Length",
                kind="number",
                default=4.6,
                unit="m",
                help="Along its heading: a car's, a walker's, whatever is to stand there.",
            ),
            FieldSpec(
                name="width",
                label="Width",
                kind="number",
                default=2.0,
                unit="m",
            ),
            FieldSpec(
                name="margin",
                label="Margin",
                kind="number",
                default=0.3,
                required=False,
                unit="m",
                help="How near another road user's outline may come.",
            ),
        ),
        description=(
            "No vehicle's or walker's bounding box overlaps a rectangle at a "
            "place on the map (a margin added) -- where an entity is to "
            "appear, so it is not brought in on top of another road user."
        ),
    )
)

#: The metrics a Metric card can watch.
METRIC_OPTIONS: tuple[SelectOption, ...] = tuple(
    SelectOption(metric.key, metric.title) for metric in BUILT_IN_METRICS.values()
)

register_condition_spec(
    ConditionSpec(
        type_id="metric",
        title="Metric",
        category="World",
        builder="build_metric_condition",
        target="..conditions:MetricCondition",
        visual=ConditionVisual(
            metric="Metric", rule="rule", value="value", details=("metric",)
        ),
        fields=(
            FieldSpec(
                name="metric",
                label="Metric",
                kind="select",
                default="collision_at_fault",
                options=METRIC_OPTIONS,
            ),
            _rule_field("greater_than_or_equal"),
            FieldSpec(
                name="value",
                label="Value",
                kind="number",
                default=1.0,
                help=(
                    "In the metric's unit: metres for a distance, 1 (it "
                    "happens) or 0 for an event."
                ),
            ),
        ),
        description=(
            "How the ego drives, read off the world: a collision, leaving the "
            "road, driving against the lane, how close it comes to other road "
            "users or the lane edge (AlpaSim's metrics).  Usually a fail "
            "condition."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="trajectory_time",
        title="Trajectory time",
        category="World",
        builder="build_trajectory_time_condition",
        target="..conditions:TrajectoryTimeCondition",
        visual=ConditionVisual(metric="Trajectory clock", value="time", unit="s"),
        fields=(
            FieldSpec(
                name="time",
                label="Time",
                kind="number",
                default=0.0,
                unit="s",
                help=(
                    "On the trajectory's clock: the card's time reference "
                    "(scale, offset, from the action or the scenario start) "
                    "maps it onto the scenario's."
                ),
            ),
        ),
        description=(
            "A vertex's time: the entity departs the vertex when the "
            "trajectory's clock reaches it.  Only as a Set Goal "
            "card's whole waypoint condition."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="timeout",
        title="Timeout",
        category="World",
        builder="build_timeout_condition",
        target="..conditions:TimeoutCondition",
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
        target="..conditions:CollisionCondition",
        visual=ConditionVisual(
            metric="Collision",
            target="target",
            value_label="occurred",
            details=("target_type",),
        ),
        fields=(
            FieldSpec(
                name="min_impulse",
                label="Minimum impulse",
                kind="number",
                default=0.0,
                unit="N s",
                required=False,
            ),
            _entity_field("target", "With entity", required=False),
            FieldSpec(
                name="target_type",
                label="With any",
                kind="select",
                default="ANY",
                options=(
                    SelectOption("ANY", "Anything"),
                    SelectOption("VEHICLE", "Vehicle"),
                    SelectOption("PEDESTRIAN", "Pedestrian"),
                    SelectOption("STATIC", "Static object"),
                ),
                required=False,
                help=(
                    "A class of object instead of one named entity.  Set the "
                    "entity or this, never both: they are two ways of saying "
                    "what was hit, and a condition that said both would be "
                    "asserting one thing twice."
                ),
            ),
        ),
        description=(
            "The ego vehicle collided.  Unrestricted it fires on hitting "
            "anything, which in a scenario with several actors cannot tell "
            "the collision under test from clipping a kerb -- name the entity "
            "or the class of object the scenario is about."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="traffic_signal",
        title="Traffic signal state",
        category="World",
        builder="build_traffic_signal_condition",
        target="..conditions:TrafficSignalCondition",
        argmap=(("state", "expected_state"),),
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
        type_id="traffic_signal_controller",
        title="Junction phase",
        category="World",
        builder="build_traffic_signal_controller_condition",
        target="..conditions:TrafficSignalControllerCondition",
        visual=ConditionVisual(
            metric="Junction phase",
            target="controller",
            target_prefix="Controller",
            value="signal_phase",
            value_label="shows",
        ),
        # Spelled the same way as the action's, so one word means one thing
        # across the pair; see the note there for why it is not just `phase`.
        argmap=(("signal_phase", "phase"),),
        fields=(
            FieldSpec(
                name="controller",
                label="Controller",
                kind="signal_controller",
                default="",
                help=(
                    "A controller this scenario declares under "
                    "map.traffic_signal_controllers."
                ),
            ),
            FieldSpec(
                name="signal_phase",
                label="Signal phase",
                kind="signal_phase",
                default="",
                help=(
                    "Passes while that phase is the one showing.  Read off the "
                    'controller rather than off the lights, so it means "this '
                    'phase is running" and not "something set these colours".'
                ),
            ),
        ),
        description=("The junction's controller is showing a named phase."),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="action_state",
        title="Action state",
        category="World",
        builder="build_action_state_condition",
        target="..conditions:ActionStateCondition",
        argmap=(("action", "action_id"),),
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
        target="..conditions:TemporaryStopCondition",
        argmap=(("entity", "entity_name"),),
        # The editor offers the *start* of each lanelet, which the condition's
        # own `s_margin` widens; the runtime takes poses, so each id becomes
        # one.
        builds=(
            BuiltArgument(
                kwarg="stop_positions",
                target="..coordinate:Lanelet2Pose",
                over="stop_lanelets",
                item_kwarg="lanelet_id",
                parts=(BuiltPart(constants=(("s", "0.0"),)),),
            ),
        ),
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
            *_temporary_stop_fields("a lanelet's start"),
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
        type_id="temporary_stop_at_stop_lines",
        title="Temporary stop at stop lines",
        category="Entity",
        builder="build_temporary_stop_at_stop_lines_condition",
        target="..conditions:TemporaryStopCondition",
        argmap=(("entity", "entity_name"),),
        builds=(
            BuiltArgument(
                kwarg="stop_positions",
                target="..coordinate:get_stop_line_poses_with_following",
                parts=(
                    BuiltPart(
                        args=(("lanelet_id", "from_lanelet"), ("depth", "depth"))
                    ),
                ),
            ),
        ),
        visual=ConditionVisual(
            metric="Stops at",
            subject="entity",
            target="from_lanelet",
            target_prefix="Stop lines from",
            value="stop_duration",
            unit="s",
        ),
        fields=(
            _entity_field("entity", "Subject"),
            FieldSpec(
                name="from_lanelet",
                label="Search from",
                kind="lanelet",
                default=0,
                help="The stop lines on this lanelet and the ones after it.",
            ),
            FieldSpec(
                name="depth",
                label="Steps on",
                kind="int",
                default=1,
                help="How many steps of following lanelets to search.",
            ),
            *_temporary_stop_fields("a stop line"),
        ),
        description=(
            "The entity came to a stop at one of the map's stop lines, found "
            "from a lanelet onwards.  Temporary stop, with the positions read "
            "off the map rather than given."
        ),
    )
)

register_condition_spec(
    ConditionSpec(
        type_id="always_true",
        title="Always",
        category="World",
        builder="build_always_true_condition",
        target="..conditions:AlwaysTrueCondition",
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
        type_id="turn_direction",
        title="Turn direction",
        category="Topology",
        fields=(
            FieldSpec(
                name="value",
                label="Direction",
                kind="select",
                default="left",
                options=_TURN_DIRECTIONS,
            ),
        ),
        description=(
            "A junction lanelet whose turn_direction tag is this: the "
            "movements of a junction, on any map."
        ),
    )
)
register_constraint_spec(
    ConstraintSpec(
        type_id="lanelet_length",
        title="Lanelet length",
        category="Geometry",
        fields=(
            _rule_field("greater_than_or_equal"),
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
register_constraint_spec(
    ConstraintSpec(
        type_id="on_route",
        title="On route",
        category="Identity",
        fields=(
            FieldSpec(
                name="lanelet_ids",
                label="Route lanelets",
                kind="lanelet_list",
                default=[],
                help="The lanelets of the route. Empty: no route, every lanelet matches.",
            ),
        ),
        description="Match the lanelets of a route; with none given, every lanelet.",
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

register_binding_spec(
    BindingSpec(
        type_id="matched",
        title="The matched lanelet",
        produces="lanelet",
        description=(
            "The lanelet the search picked -- before a spawn's offset walks "
            "back off it."
        ),
    )
)
register_binding_spec(
    BindingSpec(
        type_id="adjacent",
        title="Beside the matched lanelet",
        produces="lanelet",
        fields=(
            FieldSpec(
                name="side",
                label="Side",
                kind="select",
                default="left",
                options=_DIRECTIONS,
            ),
        ),
        description=(
            "The lane beside the pick a vehicle may change into from it.  Pair "
            "it with Has adjacent lane on the search, or picks without one are "
            "dropped."
        ),
    )
)
register_binding_spec(
    BindingSpec(
        type_id="route_through",
        title="Along the route from the matched lanelet",
        produces="lanelet",
        fields=(
            FieldSpec(
                name="depth",
                label="Steps on",
                kind="int",
                default=1,
                help="1 is the lanelet after the pick; where the road forks, "
                "the lowest id.",
            ),
        ),
        # A lanelet slot takes one id: the route's last lanelet.
        fixed=(("last_only", True),),
        description="The lanelet that many steps on from the pick.",
    )
)
register_binding_spec(
    BindingSpec(
        type_id="stop_line_approach",
        title="Lanelet before the stop line",
        produces="lanelet",
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
            "The lanelet a spawn that far before the pick's stop line is on: "
            "the pick, or the predecessor Before stop line walks back to."
        ),
    )
)
