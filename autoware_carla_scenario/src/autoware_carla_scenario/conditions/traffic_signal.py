"""Traffic signal condition: verify traffic light state by Lanelet2 regulatory element ID."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..coordinate.traffic_light import find_traffic_lights_for_lanelet2_id
from ..utils.traffic_light import traffic_light_state_name
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


class TrafficSignalCondition(BaseCondition):
    """Check whether a traffic light matches an expected state.

    Resolves a Lanelet2 regulatory element ID to CARLA traffic light actors on
    the first ``check()`` that finds any, the way
    :class:`~autoware_carla_scenario.actions.TrafficSignalAction` and the
    signal controllers do
    (:func:`~autoware_carla_scenario.coordinate.traffic_light.find_traffic_lights_for_lanelet2_id`:
    by OpenDRIVE controller, or by stop line on a map where no controller is
    named after the light), and caches their actor ids.  Subsequent calls read
    those actors' current state and compare it against the expected state.

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
    _cached_actor_ids: Optional[set[int]]
    _cached_signal_ids: Optional[set[str]]
    _last_mismatch: Optional[str]
    _warned_unresolved: bool
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
        self._cached_actor_ids: Optional[set[int]] = None
        self._cached_signal_ids: Optional[set[str]] = None
        self._last_mismatch: Optional[str] = None
        self._warned_unresolved = False
        self._warned_no_actors = False

    # ------------------------------------------------------------------
    # Light resolution (lazy + cached)
    # ------------------------------------------------------------------

    def _resolve_actor_ids(self, world: "carla.World") -> Optional[set[int]]:
        """Resolve the Lanelet2 ID to the actor ids of its CARLA lights.

        Warns once when the id resolves to no light, not on every tick it
        is retried.

        Returns:
            A set of actor ids, or ``None`` if resolution fails.
        """
        lights = find_traffic_lights_for_lanelet2_id(world, self._lanelet2_id)
        if not lights:
            if not self._warned_unresolved:
                self._warned_unresolved = True
                logger.warning(
                    "TrafficSignalCondition [%s]: no traffic light actor found "
                    "for Lanelet2 regulatory element ID %d (neither by OpenDRIVE "
                    "controller nor by stop line), so this condition cannot "
                    "fire until one is.",
                    self.label,
                    self._lanelet2_id,
                )
            return None
        self._cached_signal_ids = {light.get_opendrive_id() for light in lights}
        return {int(light.id) for light in lights}

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

        * **The id resolves to no light.**  Warned once, because a
          condition that never fires for a whole run should say why, and
          retried next tick.
        * **The resolved lights are gone.**  They are resolved again at
          once, since a reloaded world numbers its actors anew; when that
          finds none either, warned once, likewise.
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
        fresh = False
        if self._cached_actor_ids is None:
            resolved = self._resolve_actor_ids(world)
            if resolved is None:
                self._last_mismatch = None
                return None
            self._cached_actor_ids = resolved
            fresh = True
        # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
        actor_ids: set[int] = self._cached_actor_ids
        match_count, mismatches = self._match(world, actor_ids)

        if match_count == 0 and not fresh:
            # The cached actors are gone, as after a world reload under a
            # condition the scenario's ``__init__`` registered, which outlives
            # a run: actor ids are the episode's, unlike the signal ids that
            # were cached before (#901).  Resolve the lights again.
            resolved = self._resolve_actor_ids(world)
            if resolved is None:
                self._cached_actor_ids = None
                self._last_mismatch = None
                return None
            # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
            reresolved: set[int] = resolved
            self._cached_actor_ids = reresolved
            actor_ids = reresolved
            match_count, mismatches = self._match(world, actor_ids)

        if match_count == 0:
            # Resolved anew next tick.
            self._cached_actor_ids = None
            self._last_mismatch = None
            if not self._warned_no_actors:
                self._warned_no_actors = True
                logger.warning(
                    "TrafficSignalCondition [%s]: no traffic light actor "
                    "matches actor IDs %s any more, so this condition cannot "
                    "fire until one does.",
                    self.label,
                    sorted(actor_ids),
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

    def _match(
        self, world: "carla.World", actor_ids: set[int]
    ) -> tuple[int, list[tuple[str, int]]]:
        """Count the lights among *actor_ids* that *world* still has, and
        collect ``(signal id, state)`` of those not in the expected state."""
        match_count = 0
        mismatches: list[tuple[str, int]] = []
        for actor in world.get_actors().filter("traffic.traffic_light*"):
            if int(actor.id) in actor_ids:
                match_count += 1
                state = actor.get_state()
                if state != self._expected_state:
                    mismatches.append((actor.get_opendrive_id(), state))
        return match_count, mismatches

    def get_details(self) -> dict[str, Any]:
        """Return structured details about this condition's configuration."""
        return {
            "lanelet2_regulatory_element_id": self._lanelet2_id,
            "expected_state": self._expected_name,
            "cached_actor_ids": (
                sorted(self._cached_actor_ids)
                if self._cached_actor_ids is not None
                else None
            ),
            "cached_signal_ids": (
                sorted(self._cached_signal_ids)
                if self._cached_signal_ids is not None
                else None
            ),
            "last_mismatch": self._last_mismatch,
        }
