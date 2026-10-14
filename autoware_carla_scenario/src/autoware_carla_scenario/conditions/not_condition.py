"""NOT condition that inverts the satisfied result of a child condition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


class NotCondition(BaseCondition):
    """Condition that inverts the result of a child condition.

    By default it inverts the child's *verdict*: when the child returns a
    satisfied result (``passed=True``), this condition flips it to
    ``passed=False`` and vice versa.  When the child returns ``None`` (not yet
    determined), this also returns ``None``.

    That is not a logical NOT of whether the child holds -- a condition that
    has not fired returns ``None``, so "the entity is *not* on any of these
    lanelets" cannot be said with it.  ``holds_when_pending=True`` gives the
    logical one: the NOT holds (``passed=True``) on every check the child
    returns ``None``, and is pending (``None``) while the child has fired,
    whatever its verdict.  Wrapped in a :class:`PersistentCondition` it waits
    out a single tick's noise::

        # Fail once the ego has been on none of these lanelets for 0.2 s:
        # changed into a lane beside it, turned off, or left the road.
        PersistentCondition(
            NotCondition(
                OrCondition([
                    EntityLanePositionCondition(
                        "ego", Lanelet2Pose(lanelet_id=i, s=0.0), label=f"on_{i}"
                    )
                    for i in lanelets_ahead(spawn_lanelet_id, 300.0)
                ]),
                holds_when_pending=True,
            ),
            0.2,
            label="ego_left_its_lane",
        )

    Args:
        condition: The child condition whose result is inverted.
        holds_when_pending: Invert whether the child holds rather than its
            verdict, as above.  ``False`` (the default) keeps the
            verdict-flipping behaviour.
    """

    # Declared for the static check (docs/typecheck.md).
    _condition: BaseCondition
    _holds_when_pending: bool

    def __init__(
        self,
        condition: BaseCondition,
        *,
        label: str | None = None,
        holds_when_pending: bool = False,
    ) -> None:
        super().__init__(label=label if label is not None else condition.label)
        self._condition = condition
        self._holds_when_pending = holds_when_pending

    def get_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "operator": "NOT",
            "child": self._condition.to_summary_dict(),
        }
        if self._holds_when_pending:
            details["holds_when_pending"] = True
        return details

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Check the child condition and invert the result.

        Args:
            world: The CARLA world instance.
            elapsed: Elapsed time in seconds since the scenario started.

        Returns:
            :class:`ScenarioResult` with inverted ``passed`` flag, or
            ``None`` if the child is still pending.  With
            ``holds_when_pending``: ``passed=True`` while the child is
            pending, ``None`` once it has fired.
        """
        result = self._condition.check(world, elapsed)
        if self._holds_when_pending:
            if result is not None:
                return None
            return ScenarioResult(
                passed=True,
                message=(
                    f"NOT({self._condition.label}): not satisfied at {elapsed:.2f}s"
                ),
                elapsed_seconds=elapsed,
            )
        if result is None:
            return None
        return ScenarioResult(
            passed=not result.passed,
            message=f"NOT({result.message})",
            elapsed_seconds=result.elapsed_seconds,
        )
