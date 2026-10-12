"""Binding the set-level constraints wherever they sit in a constraint tree.

``find_matching_lanelets`` caches what each ``previous_of`` / ``following_of``
/ ``has_adjacent`` matches before it evaluates the tree.  The constraints of
the module say what they bind (``Constraint._bind_set_level``); a constraint
that only has an ``evaluate()`` -- not derived from ``Constraint`` -- is still
walked through its ``constraints`` / ``constraint``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from autoware_carla_scenario.sweeper.constraints import (
    AndConstraint,
    Constraint,
    EqualsConstraint,
    FollowingOfConstraint,
    HasAdjacentConstraint,
    NotConstraint,
    OrConstraint,
    PreviousOfConstraint,
    find_matching_lanelets,
)


class _Layer:
    """A lanelet layer: iterated over, and looked up by id."""

    def __init__(self, ids: list[int]) -> None:
        self._lanelets = {i: SimpleNamespace(id=i) for i in ids}

    def __iter__(self) -> Any:
        return iter(self._lanelets.values())

    def __getitem__(self, lanelet_id: int) -> SimpleNamespace:
        return self._lanelets[lanelet_id]


class _Graph:
    """A routing graph of the chain 1 -> 2 -> 3 -> 4, with 5 left of 3."""

    _following: ClassVar[dict[int, list[int]]] = {1: [2], 2: [3], 3: [4]}
    _previous: ClassVar[dict[int, list[int]]] = {2: [1], 3: [2], 4: [3]}

    def following(self, lanelet: SimpleNamespace) -> list[SimpleNamespace]:
        return [SimpleNamespace(id=i) for i in self._following.get(lanelet.id, [])]

    def previous(self, lanelet: SimpleNamespace) -> list[SimpleNamespace]:
        return [SimpleNamespace(id=i) for i in self._previous.get(lanelet.id, [])]

    def left(self, lanelet: SimpleNamespace) -> SimpleNamespace | None:
        return SimpleNamespace(id=5) if lanelet.id == 3 else None

    def right(self, lanelet: SimpleNamespace) -> SimpleNamespace | None:
        return None


# Stand-ins for lanelet2's map and routing graph, so typed `Any`.
_MAP: Any = SimpleNamespace(laneletLayer=_Layer([1, 2, 3, 4, 5]))
_GRAPH: Any = _Graph()


class _DuckIs:
    """A constraint that only has evaluate(): no Constraint base.

    Built through :func:`_duck_is`, typed ``Any``: a duck-typed constraint
    works at run time but is no ``Constraint`` to a type checker.
    """

    def __init__(self, *ids: int) -> None:
        self.ids = ids

    def evaluate(self, lanelet: Any) -> bool:
        return bool(lanelet.id in self.ids)


def _duck_is(*ids: int) -> Any:
    return _DuckIs(*ids)


def _lanelet(lanelet_id: int) -> Any:
    """A stand-in for a lanelet2 lanelet, so typed ``Any``."""
    return SimpleNamespace(id=lanelet_id)


class _DuckAll:
    """A duck-typed composite holding its children as ``constraints``."""

    def __init__(self, *constraints: Any) -> None:
        self.constraints = constraints

    def evaluate(self, lanelet: Any) -> bool:
        return all(c.evaluate(lanelet) for c in self.constraints)


class _DuckNot:
    """A duck-typed composite holding its child as ``constraint``."""

    def __init__(self, constraint: Any) -> None:
        self.constraint = constraint

    def evaluate(self, lanelet: Any) -> bool:
        return not self.constraint.evaluate(lanelet)


class _AnyOf(Constraint):
    """A Constraint of someone else's, with children but no _bind_set_level."""

    def __init__(self, *constraints: Any) -> None:
        self.constraints = constraints

    def evaluate(self, lanelet: Any) -> bool:
        return any(c.evaluate(lanelet) for c in self.constraints)


def _match(*constraints: Any) -> list[int]:
    return find_matching_lanelets(list(constraints), _MAP, _GRAPH)


def test_set_level_constraints_are_bound_inside_the_combinators() -> None:
    previous = PreviousOfConstraint((EqualsConstraint(3),))
    following = FollowingOfConstraint((EqualsConstraint(1),))
    adjacent = HasAdjacentConstraint("left")
    assert _match(AndConstraint((previous, NotConstraint(adjacent)))) == [2]
    assert _match(OrConstraint((following, adjacent))) == [2, 3]


def test_a_duck_typed_constraint_still_matches() -> None:
    assert _match(_duck_is(2, 3)) == [2, 3]
    assert _match(AndConstraint((_duck_is(2, 3), HasAdjacentConstraint("left")))) == [3]
    assert _match(PreviousOfConstraint((_duck_is(4),))) == [3]


def test_a_duck_typed_composite_has_its_children_bound() -> None:
    assert _match(_DuckAll(PreviousOfConstraint((EqualsConstraint(3),)))) == [2]
    assert _match(_DuckNot(FollowingOfConstraint((EqualsConstraint(1),)))) == [
        1,
        3,
        4,
        5,
    ]


def test_a_derived_constraint_with_children_has_them_bound() -> None:
    assert _match(_AnyOf(PreviousOfConstraint((EqualsConstraint(3),)))) == [2]


def test_an_unbound_set_level_constraint_has_no_cached_ids() -> None:
    previous = PreviousOfConstraint((EqualsConstraint(3),))
    with pytest.raises(AttributeError):
        previous.evaluate(_lanelet(2))
    # The cache is no part of the comparison, the hash or the repr.
    _match(previous)
    assert previous == PreviousOfConstraint((EqualsConstraint(3),))
    assert hash(previous) == hash(PreviousOfConstraint((EqualsConstraint(3),)))
    assert "_cached_ids" not in repr(previous)


def test_the_base_class_is_not_a_constraint_itself() -> None:
    with pytest.raises(TypeError):
        Constraint()  # type: ignore[abstract]
