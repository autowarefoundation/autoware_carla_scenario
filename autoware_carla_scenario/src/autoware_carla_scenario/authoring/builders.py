"""Factories that turn a compiled plan into the framework's runtime objects.

Every builder returns an object that already existed in
``autoware_carla_scenario`` -- ``LaneChangeAction``, ``AndCondition``,
``StickyCondition`` and friends.  The editor contributes no runtime of its own;
it only decides which of these to build and with what arguments.

Most builders are not written here at all.  A spec that names a ``target`` in
:mod:`autoware_carla_scenario.authoring.registry` has its builder *generated*
from that class's constructor signature into
:mod:`autoware_carla_scenario.authoring._builders_generated`, because renaming a
field to a keyword and lifting a select into an enum is work the signature
already describes.  What stays here is what a signature cannot describe: a
builder that assembles a pose out of several fields, or chooses between two
constructors.  :func:`_resolve` looks in both modules, so a spec does not have
to say which kind it is.

Module-level imports are kept free of CARLA and lanelet2 so that
:mod:`autoware_carla_scenario.authoring` can be imported in the editor process.
The heavy imports happen inside the builders, which only run inside
:meth:`~autoware_carla_scenario.declarative.DeclarativeScenario.setup` -- i.e.
against a live CARLA world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..actions import BaseAction
    from ..conditions import BaseCondition
    from .compiler import BuildContext, CompiledAction, CompiledCondition

__all__ = [
    "instantiate_action",
    "instantiate_condition",
]


# ---------------------------------------------------------------------------
# Hand-written builders
#
# A builder belongs here only when its runtime object is not a plain call with
# one argument per field.  Anything else must name a ``target`` instead and be
# generated, so that the generator's checks -- exhaustive select cases, fields
# that reach no argument, options that have drifted from their enum -- apply.
# ---------------------------------------------------------------------------


def _scalar_bounds(params: dict[str, Any]) -> "list[Any]":
    """Turn the editor's ``s``/``t`` from-to fields into comparison rules.

    The editor offers a *range* because that is what an author means by "over
    this stretch of the lane"; the runtime takes a list of one-sided rules, so
    each bound that is set becomes one.  An empty bound is simply absent, which
    is how "the whole length" is said.
    """
    from ..conditions.comparison import ComparisonRule  # noqa: PLC0415
    from ..conditions.comparison import ScalarComparisonRule  # noqa: PLC0415

    bounds = (
        ("s", "s_min", ComparisonRule.GREATER_THAN_OR_EQUAL),
        ("s", "s_max", ComparisonRule.LESS_THAN_OR_EQUAL),
        ("t", "t_min", ComparisonRule.GREATER_THAN_OR_EQUAL),
        ("t", "t_max", ComparisonRule.LESS_THAN_OR_EQUAL),
    )
    return [
        ScalarComparisonRule(field=field, rule=rule, value=float(params[name]))
        for field, name, rule in bounds
        if params.get(name) is not None
    ]


def build_entity_lane_position_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityLanePositionCondition` from a Lanelet2 reference.

    The lanelet goes in whole rather than being reduced to a road and a lane
    here.  A lanelet names one lane, and the condition resolves it -- which is
    the only resolution left, so there is nothing for a second one to drift
    from.  The pose carries ``s=0.0`` because it is an address: *where along*
    the lane is said with the ``s``/``t`` bounds below.
    """
    from ..conditions import EntityLanePositionCondition  # noqa: PLC0415
    from ..coordinate import Lanelet2Pose  # noqa: PLC0415

    params = compiled.params
    return EntityLanePositionCondition(
        params["entity"],
        Lanelet2Pose(lanelet_id=int(params["lanelet_id"]), s=0.0),
        rules=_scalar_bounds(params),
        label=compiled.label,
    )


def build_entity_road_position_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityLanePositionCondition` from an OpenDRIVE address.

    This is the frame the runtime already speaks, so nothing is resolved.  A
    lane the author left empty means the road alone is being asserted, which
    the condition has its own constructor for.
    """
    from ..conditions import EntityLanePositionCondition  # noqa: PLC0415
    from ..coordinate import OpenDrivePose  # noqa: PLC0415

    params = compiled.params
    rules = _scalar_bounds(params)
    road_id = str(params["road_id"])
    lane_id = params.get("lane_id")
    if lane_id is None:
        return EntityLanePositionCondition.anywhere_on_road(
            params["entity"], road_id, rules=rules, label=compiled.label
        )
    return EntityLanePositionCondition(
        params["entity"],
        OpenDrivePose(road_id=road_id, lane_id=int(lane_id), s=0.0),
        rules=rules,
        label=compiled.label,
    )


def build_temporary_stop_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TemporaryStopCondition` from a set of lanelets.

    The runtime condition accepts any pose; the editor offers the *start* of a
    lanelet, which the condition's own ``s_margin`` then widens.  That is the
    same reduction the rest of the canvas makes -- authors think in lanelets --
    and it is why the margin is an editable field rather than a constant.
    """
    from ..conditions import TemporaryStopCondition  # noqa: PLC0415
    from ..coordinate import Lanelet2Pose  # noqa: PLC0415

    params = compiled.params
    return TemporaryStopCondition(
        entity_name=params["entity"],
        stop_positions=[
            Lanelet2Pose(lanelet_id=int(lanelet_id), s=0.0)
            for lanelet_id in params["stop_lanelets"]
        ],
        s_margin=params["s_margin"],
        speed_threshold=params["speed_threshold"],
        stop_duration=params["stop_duration"],
        label=compiled.label,
    )


# ---------------------------------------------------------------------------
# Action builders
# ---------------------------------------------------------------------------


def build_routing_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`RoutingAction`.

    Hand-written because the goal is one pose assembled from two fields, which
    a constructor signature cannot describe.  Everything else is the ordinary
    shape: the action names its entity and looks it up when it runs.
    """
    from ..actions import RoutingAction  # noqa: PLC0415
    from ..coordinate import Lanelet2Pose  # noqa: PLC0415

    del ctx  # The entity is named, not handed over.
    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return RoutingAction(
        compiled.actor_role,
        Lanelet2Pose(
            lanelet_id=int(params["goal_lanelet_id"]),
            s=float(params.get("goal_s") or 0.0),
        ),
        condition,
        timing,
        label=compiled.label,
        once=compiled.node.once,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _resolve(builder_name: str) -> Callable[..., Any]:
    """Return the builder function a spec names.

    The generated module is searched first and this one second, so a spec never
    has to say whether its builder was written or rendered.

    Raises:
        LookupError: If neither module defines it, which means a primitive was
            registered with neither a ``target`` nor a hand-written builder.
    """
    from . import _builders_generated  # noqa: PLC0415

    for namespace in (vars(_builders_generated), globals()):
        builder = namespace.get(builder_name)
        if callable(builder):
            return builder
    raise LookupError(
        f"No builder named {builder_name!r} in "
        f"autoware_carla_scenario.authoring.builders or its generated module."
    )


def instantiate_condition(
    compiled: "CompiledCondition", ctx: "BuildContext"
) -> "BaseCondition":
    """Recursively build the runtime condition tree for *compiled*."""
    children = [instantiate_condition(c, ctx) for c in compiled.children]
    return _resolve(compiled.spec.builder)(compiled, children, ctx)


def instantiate_action(compiled: "CompiledAction", ctx: "BuildContext") -> "BaseAction":
    """Build the runtime action for *compiled*, including its trigger."""
    from ..actions import TickTiming  # noqa: PLC0415

    condition = (
        instantiate_condition(compiled.trigger, ctx)
        if compiled.trigger is not None
        else None
    )
    timing = TickTiming(compiled.node.timing)
    return _resolve(compiled.spec.builder)(compiled, condition, timing, ctx)
