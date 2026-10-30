"""Traffic signal condition: verify traffic light state by Lanelet2 regulatory element ID."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..coordinate.traffic_light import (
    get_signal_ids_for_controller,
    lanelet2_traffic_light_id_to_opendrive_controller_id,
)
from ..utils.traffic_light import traffic_light_state_name
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


class TrafficSignalCondition(BaseCondition):
    """Check whether a traffic light matches an expected state.

    Resolves a Lanelet2 regulatory element ID to OpenDRIVE signal IDs on the
    first ``check()`` call and caches the result.  Subsequent calls reuse the
    cached signal IDs to find matching CARLA traffic light actors and compare
    their current state against the expected state.

    The condition fires (returns ``passed=True``) only while every matching
    light shows the expected state, and returns ``None`` otherwise -- as
    :class:`TrafficSignalControllerCondition` does.  It never returns a
    failing result: the runner and an action both treat any non-``None``
    result as the condition having fired, so a ``passed=False`` would pass the
    run, or start the action, while the light shows something else.  To fail
    a run on a light's state, register the condition for the state that
    should fail it.

    Args:
        lanelet2_regulatory_element_id: Lanelet2 regulatory element ID of
            the traffic light to monitor.
        expected_state: The :class:`carla.TrafficLightState` the traffic
            light is expected to be in (e.g. ``carla.TrafficLightState.Green``).
        label: Human-readable identifier for this condition.
    """

    # Declared for the static check (docs/typecheck.md).
    _lanelet2_id: int
    _expected_state: "carla.TrafficLightState"
    _expected_name: str
    _cached_signal_ids: Optional[set[str]]
    _last_mismatch: Optional[str]
    _warned_no_actors: bool

    def __init__(
        self,
        lanelet2_regulatory_element_id: int,
        expected_state: "carla.TrafficLightState",
        *,
        label: str,
    ) -> None:
        super().__init__(label=label)
        self._lanelet2_id = lanelet2_regulatory_element_id
        self._expected_state = expected_state
        self._expected_name = traffic_light_state_name(expected_state)
        self._cached_signal_ids: Optional[set[str]] = None
        self._last_mismatch: Optional[str] = None
        self._warned_no_actors = False

    # ------------------------------------------------------------------
    # Signal ID resolution (lazy + cached)
    # ------------------------------------------------------------------

    def _resolve_signal_ids(self) -> Optional[set[str]]:
        """Resolve the Lanelet2 ID to a set of OpenDRIVE signal IDs.

        Returns:
            A set of signal ID strings, or ``None`` if resolution fails.
        """
        controller_id = lanelet2_traffic_light_id_to_opendrive_controller_id(
            self._lanelet2_id,
        )
        if controller_id is None:
            logger.warning(
                "TrafficSignalCondition [%s]: no OpenDRIVE controller found "
                "for Lanelet2 regulatory element ID %d",
                self.label,
                self._lanelet2_id,
            )
            return None

        # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
        controller: int = controller_id
        signal_ids = get_signal_ids_for_controller(controller)
        if not signal_ids:
            logger.warning(
                "TrafficSignalCondition [%s]: controller %d has no signal IDs",
                self.label,
                controller_id,
            )
            return None

        return set(signal_ids)

    # ------------------------------------------------------------------
    # BaseCondition interface
    # ------------------------------------------------------------------

    def check(
        self,
        world: "carla.World",
        elapsed: float,
    ) -> Optional[ScenarioResult]:
        """Check whether matching traffic lights are in the expected state.

        ``None`` covers three situations:

        * **Signal ID resolution fails.**  Warned, and retried next tick.
        * **No matching actor is found.**  Warned once, because a condition
          that never fires for a whole run should say why.
        * **A matching light shows another state.**  The ordinary answer,
          logged at debug with the state each mismatching light *is*
          showing, which :meth:`get_details` also reports.

        None of them returns a failing result: an action treats any
        non-``None`` result as its trigger having fired, so a
        ``passed=False`` here would start the action it is meant to hold
        back, and the runner would end the run on it.

        Returns:
            ``ScenarioResult(passed=True)`` if all matching actors have the
            expected state, ``None`` otherwise.
        """
        # Lazy resolution with caching
        if self._cached_signal_ids is None:
            resolved = self._resolve_signal_ids()
            if resolved is None:
                return None
            self._cached_signal_ids = resolved
        # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
        signal_ids: set[str] = self._cached_signal_ids

        # Single-pass: find matching actors and collect mismatches
        match_count = 0
        mismatches: list[tuple[str, int]] = []
        for actor in world.get_actors().filter("traffic.traffic_light*"):
            if actor.get_opendrive_id() in signal_ids:
                match_count += 1
                state = actor.get_state()
                if state != self._expected_state:
                    mismatches.append((actor.get_opendrive_id(), state))

        if match_count == 0:
            self._last_mismatch = None
            if not self._warned_no_actors:
                self._warned_no_actors = True
                logger.warning(
                    "TrafficSignalCondition [%s]: no traffic light actor "
                    "matches signal IDs %s, so this condition cannot fire "
                    "until one does.",
                    self.label,
                    sorted(signal_ids),
                )
            return None

        if mismatches:
            mismatch_details = ", ".join(
                f"{sig_id}={traffic_light_state_name(state)}"
                for sig_id, state in mismatches
            )
            self._last_mismatch = mismatch_details
            logger.debug(
                "TrafficSignalCondition [%s]: expected %s, got %s",
                self.label,
                self._expected_name,
                mismatch_details,
            )
            return None

        self._last_mismatch = None

        return ScenarioResult(
            passed=True,
            message=(
                f"TrafficSignalCondition [{self.label}]: "
                f"all {match_count} actor(s) in expected state "
                f"{self._expected_name}"
            ),
            elapsed_seconds=elapsed,
        )

    def get_details(self) -> dict[str, Any]:
        """Return structured details about this condition's configuration."""
        return {
            "lanelet2_regulatory_element_id": self._lanelet2_id,
            "expected_state": self._expected_name,
            "cached_signal_ids": (
                sorted(self._cached_signal_ids)
                if self._cached_signal_ids is not None
                else None
            ),
            "last_mismatch": self._last_mismatch,
        }
