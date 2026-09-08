"""Condition on the lifecycle state of a named action.

This is OpenSCENARIO's ``StoryboardElementStateCondition`` narrowed to the one
storyboard element type this framework has: an action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from ..action_state import ActionState
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import carla

    from ..actions import BaseAction


class ActionStateCondition(BaseCondition):
    """Satisfied while the referenced action is in the expected lifecycle state.

    This is what lets one actor react to another *finishing* a manoeuvre rather
    than to a state that merely coincides with it.  "Once NPC1's lane change has
    completed" is ``completeState`` on that action; "once NPC1 is inside lanelet
    183" is a position check that is also true if NPC1 spawned there and never
    moved.

    Waiting on ``completeState`` is waiting on the action's own completion
    criteria, not on the command having been issued -- a forced lane change
    stays in ``runningState`` until the vehicle has settled onto the next lane,
    and never leaves it if the manoeuvre does not happen.

    The two transition states are held for a single tick each, so a condition
    watching for ``startTransition`` sees it on exactly the tick the action
    fired.

    Args:
        action_id: Key of the action to watch.
        state: The :class:`~autoware_carla_scenario.actions.base.ActionState`
            to wait for, or its OpenSCENARIO name (e.g. ``"completeState"``).
        actions: Mapping of action id to live action, resolved on every check
            rather than at construction time.  A trigger may name an action
            built after it -- an ego reaction referring to an NPC's manoeuvre,
            say -- and every action exists before the first tick, so late
            lookup is what removes the ordering constraint.
        label: Human-readable identifier for logs and result summaries.
    """

    def __init__(
        self,
        action_id: str,
        state: Union[ActionState, str],
        actions: Mapping[str, "BaseAction"],
        *,
        label: str = "action_state",
    ) -> None:
        super().__init__(label=label)
        self._action_id = action_id
        self._state = state if isinstance(state, ActionState) else ActionState(state)
        self._actions = actions

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a passing result while the action is in the expected state."""
        action = self._actions.get(self._action_id)
        if action is None or action.state is not self._state:
            return None
        return ScenarioResult(
            passed=True,
            message=f"Action '{self._action_id}' is in {self._state.value}",
            elapsed_seconds=elapsed,
        )

    def get_details(self) -> dict[str, Any]:
        """Expose the watched action and state for structured logging."""
        return {"action_id": self._action_id, "state": self._state.value}
