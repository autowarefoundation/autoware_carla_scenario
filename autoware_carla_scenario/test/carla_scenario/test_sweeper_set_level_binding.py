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
    InSetConstraint,
    LaneletLengthConstraint,
    NotConstraint,
    OnRouteConstraint,
    OrConstraint,
    PreviousOfConstraint,
    find_matching_lanelets,
    parse_constraint,
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


def test_an_id_of_non_ascii_digits_stays_a_string() -> None:
    """#1612: ``"²".isdigit()`` is true but ``int("²")`` raises."""
    assert EqualsConstraint("12").value == 12
    assert EqualsConstraint("²").value == "²"


def test_quoted_ids_name_their_lanelets_in_in_set() -> None:
    """#1670: a digit string is an id, as ``equals`` reads one."""
    constraint = parse_constraint({"type": "in_set", "values": ["2", 4, "9"]})
    assert constraint == InSetConstraint((2, 4, 9))
    assert _match(constraint) == [2, 4]


def test_quoted_ids_name_their_lanelets_in_on_route() -> None:
    """#1670: a quoted route is that route, not one matching nothing."""
    constraint = parse_constraint({"type": "on_route", "lanelet_ids": ["1", "3"]})
    assert constraint == OnRouteConstraint((1, 3))
    assert _match(constraint) == [1, 3]


@pytest.mark.parametrize("cls", [InSetConstraint, OnRouteConstraint])
def test_valid_ids_match_as_before(cls: Any) -> None:
    """Ints, and whole floats (``equals`` matches ``2.0`` too), are kept."""
    assert _match(cls((2, 4))) == [2, 4]
    assert _match(cls((2.0, 4))) == [2, 4]
    assert _match(EqualsConstraint(2.0)) == [2]  # type: ignore[arg-type]
    assert _match(cls()) == ([] if cls is InSetConstraint else [1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    ("type_id", "key"), [("in_set", "values"), ("on_route", "lanelet_ids")]
)
@pytest.mark.parametrize("bad", [True, 2.5, "two", "", " 2", "-2", "²", None])
def test_an_invalid_id_is_refused_by_field(type_id: str, key: str, bad: Any) -> None:
    with pytest.raises(ValueError, match=rf"{type_id}\.{key}"):
        parse_constraint({"type": type_id, key: [1, bad]})


@pytest.mark.parametrize(
    ("type_id", "key"), [("in_set", "values"), ("on_route", "lanelet_ids")]
)
def test_a_string_is_no_list_of_ids(type_id: str, key: str) -> None:
    with pytest.raises(ValueError, match=rf"{type_id}\.{key}"):
        parse_constraint({"type": type_id, key: "12"})


@pytest.mark.parametrize(
    ("type_id", "key"), [("in_set", "values"), ("on_route", "lanelet_ids")]
)
def test_an_unresolved_reference_still_parses(type_id: str, key: str) -> None:
    """The editor and an export parse before Hydra resolves ``${...}``."""
    constraint = parse_constraint({"type": type_id, key: "${map.ids}"})
    assert _match(constraint) == []


@pytest.mark.parametrize("bad", ["10", True, float("nan"), float("inf"), None])
def test_lanelet_length_refuses_a_value_that_is_no_finite_number(bad: Any) -> None:
    """#1670: refused by key when parsed, not a bare TypeError when matched."""
    with pytest.raises(ValueError, match=r"lanelet_length\.value"):
        parse_constraint({"type": "lanelet_length", "value": bad})


def test_lanelet_length_keeps_a_valid_value() -> None:
    assert LaneletLengthConstraint(value=10).value == 10
    assert LaneletLengthConstraint("less_than", 5.5).value == 5.5
