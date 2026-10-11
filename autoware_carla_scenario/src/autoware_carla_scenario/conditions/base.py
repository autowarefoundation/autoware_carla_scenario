"""Base classes and helpers for scenario pass/fail conditions."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

from ..entity_role import EntityRole

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


def _role_name(actor: "carla.Actor") -> Optional[str]:
    """The ``role_name`` attribute of *actor*, or ``None`` if it has none."""
    # Not attributes.get(): Codon's dict.get() requires a default of the
    # value type, and no string means "absent".
    attributes = actor.attributes
    return attributes["role_name"] if "role_name" in attributes else None


def find_actor_in_list(
    actors: Union[carla.ActorList, list[carla.Actor]], role_name: Union[EntityRole, str]
) -> Optional["carla.Actor"]:
    """Find a single actor by its ``role_name`` in a pre-fetched actor list.

    Args:
        actors: Pre-fetched actors (e.g. ``world.get_actors()``, or a list).
        role_name: The ``role_name`` attribute value to search for.
            Accepts both :class:`EntityRole` and plain ``str``.

    Returns:
        The matching actor, or ``None`` if no actor with that role name exists.
    """
    name = str(role_name)
    return next(
        (a for a in actors if _role_name(a) == name),
        None,
    )


def find_actor_pair(
    actors: Union[carla.ActorList, list[carla.Actor]],
    source: Union[EntityRole, str],
    target: Union[EntityRole, str],
) -> "tuple[Optional[carla.Actor], Optional[carla.Actor]]":
    """Find two actors by ``role_name`` in one pass over *actors*.

    Every pairwise condition wants both ends of the same measurement, and two
    :func:`find_actor_in_list` calls scan the list twice for it.

    Args:
        actors: Pre-fetched actors (e.g. ``world.get_actors()``, or a list).
        source: The ``role_name`` of the measurement's first end.
        target: The ``role_name`` of its second end.

    Returns:
        ``(source actor, target actor)``, either of which may be ``None``.
    """
    source_name, target_name = str(source), str(target)
    found: "dict[str, carla.Actor]" = {}
    for actor in actors:
        role = _role_name(actor)
        if (
            role is not None
            and role in (source_name, target_name)
            and role not in found
        ):
            found[role] = actor
            if len(found) == 2:
                break
    return (
        found[source_name] if source_name in found else None,
        found[target_name] if target_name in found else None,
    )


def find_actor_by_role_name(
    world: "carla.World", role_name: Union[EntityRole, str]
) -> Optional["carla.Actor"]:
    """Find a single actor by its ``role_name`` attribute.

    Args:
        world: The CARLA world instance.
        role_name: The ``role_name`` attribute value to search for.
            Accepts both :class:`EntityRole` and plain ``str``.

    Returns:
        The matching actor, or ``None`` if no actor with that role name exists.
    """
    return find_actor_in_list(world.get_actors(), role_name)


@dataclass
class ConditionStatus:
    """Snapshot of a single condition's state when the scenario ended.

    Attributes:
        label: Human-readable identifier (e.g. ``"pass[0](elapsed_time_30s)"``).
        satisfied: Whether the condition was satisfied at snapshot time.
        message: Human-readable description of the state.
        condition_type: Class name of the condition (e.g. ``"TimeoutCondition"``).
        role: Whether this is a ``"pass"`` or ``"fail"`` condition.
        details: Structured key/value pairs specific to the condition type.
    """

    label: str
    satisfied: bool
    message: str
    condition_type: str = ""
    role: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable nested dict."""
        return {
            "label": self.label,
            "satisfied": self.satisfied,
            "message": self.message,
            "condition_type": self.condition_type,
            "role": self.role,
            "details": self.details,
        }


@dataclass
class ScenarioResult:
    """Result of a scenario execution."""

    passed: bool
    message: str
    elapsed_seconds: float
    condition_statuses: list[ConditionStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable nested dict of the full result."""
        return {
            "passed": self.passed,
            "message": self.message,
            "elapsed_seconds": self.elapsed_seconds,
            "condition_statuses": [cs.to_dict() for cs in self.condition_statuses],
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to a JSON string.

        Args:
            **kwargs: Forwarded to :func:`json.dumps`
                (e.g. ``indent=2``, ``ensure_ascii=False``).
        """
        # Imported here: Codon has no json, and the library check leaves this
        # method out (typecheck/library.py), so it never compiles the import.
        import json  # noqa: PLC0415

        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)


class BaseCondition(ABC):
    """Abstract base class for scenario pass/fail conditions."""

    label: str
    _last_satisfied: bool
    _last_message: str

    def __init__(self, label: str) -> None:
        if not label:
            raise ValueError(
                f"{type(self).__name__}: label must not be empty. "
                "Provide a non-empty string to identify this condition."
            )
        self.label = label
        self._last_satisfied: bool = False
        self._last_message: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-wrap ``check()`` to track the latest result for UI display."""
        super().__init_subclass__(**kwargs)
        if "check" in cls.__dict__:
            original = cls.__dict__["check"]

            @functools.wraps(original)
            def _tracking_check(
                self: "BaseCondition",
                world: "carla.World",
                elapsed: float,
                _orig: Any = original,
            ) -> Optional[ScenarioResult]:
                result = _orig(self, world, elapsed)
                if result is not None:
                    self._last_satisfied = result.passed
                    self._last_message = result.message
                elif getattr(self, "_last_satisfied", False):
                    self._last_satisfied = False
                    self._last_message = ""
                return result

            cls.check = _tracking_check  # type: ignore[method-assign]

    @abstractmethod
    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Check the condition.

        Args:
            world: The CARLA world instance.
            elapsed: Simulated seconds since the run began -- the world's own
                clock, not the wall clock.  A scenario's durations have to be
                the same on a fast host and a slow one, and only the simulated
                clock advances at a rate the scenario controls.

        Returns:
            A ScenarioResult if the condition is met, None otherwise.
        """
        ...

    def get_details(self) -> dict[str, Any]:
        """Return structured details about this condition's configuration.

        Subclasses should override this to expose condition-specific
        parameters (thresholds, targets, etc.) for structured JSON logging.
        """
        return {}

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a summary dict identifying this condition and its details.

        Combines ``condition_type``, ``label``, runtime status, and
        :meth:`get_details` into a single dict suitable for nesting
        inside parent conditions.
        """
        # update() rather than `**` in the display: Codon's parser fails on
        # that, even in a function it does not compile.
        summary: dict[str, Any] = {
            "condition_type": type(self).__name__,
            "label": self.label,
            "satisfied": self._last_satisfied,
            "message": self._last_message,
        }
        summary.update(self.get_details())
        return summary
