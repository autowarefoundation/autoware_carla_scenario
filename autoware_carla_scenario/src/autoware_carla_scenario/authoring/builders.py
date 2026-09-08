"""Factories that turn a compiled plan into the framework's runtime objects.

Every builder returns an object that already existed in
``autoware_carla_scenario`` -- ``LaneChangeAction``, ``AndCondition``,
``StickyCondition`` and friends.  The editor contributes no runtime of its own;
it only decides which of these to build and with what arguments.

No builder is written here.  Every spec names a ``target`` in
:mod:`autoware_carla_scenario.authoring.registry` and has its builder
*generated* from that class's constructor signature into
:mod:`autoware_carla_scenario.authoring._builders_generated`.  What a signature
cannot describe -- a pose assembled out of several fields, a list of comparison
rules built from four bounds -- the spec says with ``builds``, so that those
builders are generated and checked like the rest rather than being exceptions
the generator's checks never see.  :func:`_resolve` still looks in both modules,
so a hand-written builder remains possible; there is simply none to write.

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

    from .models import TICK_PHASES  # noqa: PLC0415

    condition = (
        instantiate_condition(compiled.trigger, ctx)
        if compiled.trigger is not None
        else None
    )
    # An init action is performed once by `run_init`, which ticks it directly,
    # so its tick position is never consulted; `TickTiming` has no member for a
    # phase that is not a tick, and inventing one would put "init" in front of
    # the loop's own dispatch.
    phase = compiled.node.phase
    timing = TickTiming(phase) if phase in TICK_PHASES else TickTiming.PRE_TICK
    return _resolve(compiled.spec.builder)(compiled, condition, timing, ctx)
