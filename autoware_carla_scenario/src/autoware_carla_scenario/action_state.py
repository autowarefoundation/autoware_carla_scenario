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
        The work is under way.  A lane change sits here for the whole
        manoeuvre, because the request to the traffic backend returns long
        before the vehicle has moved.
    ``END_TRANSITION``
        The completion criteria have been met.  Also held for one tick --
        but only by an action that runs on (see below).
    ``COMPLETE``
        Finished.  A repeating action (``once=False``) returns to ``STANDBY``
        instead, matching OpenSCENARIO's behaviour for an element with a
        maximum execution count above one.

    ``RUNNING`` and ``END_TRANSITION`` are only seen on an action that *runs
    on*: one given an ``until`` condition, or one that reissues its command
    every tick (:attr:`BaseAction.reissues_while_running`).  Such an action
    sits in ``RUNNING`` on every tick its ``until`` has not yet held -- none,
    if that is the tick after it fired, or if it reissues with no ``until`` --
    and then holds ``END_TRANSITION`` for one tick.  An action with neither is
    over the moment it is commanded: on the tick after ``START_TRANSITION`` it
    goes straight to ``COMPLETE`` (``STANDBY`` when ``once=False``, from which
    it may fire again on that same tick), and is never seen in ``RUNNING`` or
    ``END_TRANSITION``.  A condition waiting on either of those for such an
    action never holds; wait on ``COMPLETE`` (or ``START_TRANSITION``)
    instead.  A repeating one whose trigger still holds does fire again on
    that tick, so it stays ``START_TRANSITION`` from tick to tick and is not
    seen in ``STANDBY`` until its trigger stops holding.

    An action stopped by another built with ``overrides=True`` goes straight
    to ``COMPLETE`` (:meth:`BaseAction._stop_overridden`, OpenSCENARIO's stop
    transition), even with ``once=False``, and never holds
    ``END_TRANSITION``.

    An action fired in the init phase never holds ``END_TRANSITION`` either:
    it is settled there at once (:meth:`BaseAction._settle_start`), to
    ``COMPLETE`` (``STANDBY`` when ``once=False``) if it does not reissue its
    command and either has no ``until`` or its ``until`` already holds at
    ``elapsed=0``.  One that reissues, or whose ``until`` does not yet hold,
    is left in ``RUNNING``, and as the init phase never ticks it again it
    stays there for good -- unless it was put on the tick loop instead:
    :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_init`
    does that for every action whose :attr:`BaseAction.needs_tick_loop` is
    true.  An action can opt out, as the background-traffic source and sink
    (``TrafficSourceAction``, ``TrafficSinkAction``) do (a
    ``ConstructionZoneAction`` with a later start or an end time does not):
    registered for initialization, one reissues by default
    (``REISSUES_BY_DEFAULT = True``) with ``needs_tick_loop`` false, so it
    holds ``RUNNING`` for the rest of the run, even with an ``until`` that
    already holds.

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
