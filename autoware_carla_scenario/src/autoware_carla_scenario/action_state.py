"""The lifecycle state of a scenario action.

Kept out of :mod:`autoware_carla_scenario.actions` on purpose: conditions are
imported *by* that package, so a condition on an action's state could not import
the enum from there without a cycle.  This mirrors :mod:`entity_role`, which is
top-level for the same reason.
"""

from __future__ import annotations

import enum


class ActionState(enum.Enum):
    """Lifecycle of one action, following OpenSCENARIO's storyboard element states.

    The vocabulary and the transitions are ASAM OpenSCENARIO 1.2's
    ``StoryboardElementState`` applied to a single action, so a scenario author
    who knows OpenSCENARIO can predict what each value means:

    ``STANDBY``
        Instantiated and waiting for its start trigger.
    ``START_TRANSITION``
        The start trigger fired and :meth:`BaseAction.execute` has run.  Held
        for exactly the tick it happens on, which is what makes it usable as a
        trigger for something else.
    ``RUNNING``
        The work is under way.  A forced lane change sits here for the whole
        manoeuvre, because ``force_lane_change`` returns long before the vehicle
        has moved.
    ``END_TRANSITION``
        The completion criteria have been met.  Also held for one tick.
    ``COMPLETE``
        Finished.  A repeating action (``once=False``) returns to ``STANDBY``
        instead, matching OpenSCENARIO's behaviour for an element with a
        maximum execution count above one.

    There is deliberately no failure state: OpenSCENARIO has none either.  An
    action whose completion criteria are never met stays ``RUNNING``, and it is
    the scenario's own timeout -- already a FAIL condition here -- that ends the
    run.  Inventing an action-level timeout would mean ``COMPLETE`` could be
    reached by a manoeuvre that never happened, which is exactly what a
    condition waiting on completion must not accept.
    """

    STANDBY = "standbyState"
    START_TRANSITION = "startTransition"
    RUNNING = "runningState"
    END_TRANSITION = "endTransition"
    COMPLETE = "completeState"
