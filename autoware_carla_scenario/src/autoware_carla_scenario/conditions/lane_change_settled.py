"""Condition satisfied once a commanded lane change has settled."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ..entity_role import EntityRole
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..traffic.driven import BackendDriven


class LaneChangeSettledCondition(BaseCondition):
    """Satisfied once *entity_name* is centred on the lane it was sent to.

    Asked of the entity rather than measured here: what "settled" means depends
    on what performed the change.  A TrafficManager-driven vehicle has settled
    once it is centred on the lane it was aimed at; a stack that plans its own
    manoeuvres would answer from its own state.

    This is the end of :class:`~autoware_carla_scenario.actions.lane_change.LaneChangeAction`'s
    run, and saying it as a condition rather than as a predicate on the action
    is what lets it be read -- and reused -- beside the trigger that started it.

    A manoeuvre that never happens simply never settles, and an action waiting
    on this stays :attr:`~autoware_carla_scenario.action_state.ActionState.RUNNING`.
    That is OpenSCENARIO's behaviour, and it keeps ``completeState`` from being
    reached by a lane change that did not happen; ending the run on a timer is
    the scenario timeout's job, not this condition's.

    Args:
        entity_name: ``role_name`` of the vehicle whose manoeuvre is watched.
        label: Human-readable identifier for logs and result summaries.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        *,
        label: str = "lane_change_settled",
    ) -> None:
        super().__init__(label=label)
        self._entity_name = str(entity_name)

    def get_details(self) -> dict[str, Any]:
        return {"entity_name": str(self._entity_name)}

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a passing result once the entity reports the change settled.

        Args:
            world: The CARLA world instance.
            elapsed: Simulated seconds since the run began.

        Returns:
            :class:`ScenarioResult` once settled, otherwise ``None`` -- an
            entity that is absent, or that cannot be asked, has not settled.
        """
        # Imported here rather than at module scope: the entity package reaches
        # back into `conditions.base`, so naming it at import time would close
        # a cycle.
        from ..entity.registry import find_vehicle_entity  # noqa: PLC0415

        entity = find_vehicle_entity(self._entity_name)
        if entity is None:
            return None
        # An entity of no known kind is found here too; it may not be able to
        # tell.  `hasattr` is decided on the static type in Codon, so on the
        # vehicle rather than the Optional.
        vehicle: BackendDriven = entity
        if not hasattr(vehicle, "lane_change_finished"):
            return None
        if not vehicle.lane_change_finished(world):
            return None
        return ScenarioResult(
            passed=True,
            message=(f"Entity '{self._entity_name}' has settled onto its new lane"),
            elapsed_seconds=elapsed,
        )
