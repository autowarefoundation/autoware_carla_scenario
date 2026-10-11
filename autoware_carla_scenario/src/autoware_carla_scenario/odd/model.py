"""An ODD: the attributes it is described in, and the modules that bound it.

The model follows ASAM OpenODD 1.0 (its chapters 6 and 7), and so do the
names:

* An **attribute** is a taxonomy concept the world can be measured on, such
  as the speed limit or the rain level.  Here it also has a **probe**, which
  reads its value from the world, and **buckets**, which make it a cover
  item: the conditions it can be in, so coverage can say which of them the
  runs reached.
* A **module** is a named rule.  It holds when its ``INCLUDE`` section holds
  and its ``EXCLUDE`` section does not (OpenODD: ``MODULE === INCLUDE AND
  (NOT EXCLUDE)``).  A module has at most one include section (``AND`` or
  ``OR``) and at most one exclude section.  A condition tests one attribute
  (``speed_limit.between(0, 60)``), groups others (:func:`all_of`,
  :func:`any_of`), or refers to another module or a label
  (:func:`module_holds`).  A label holds when any module declaring it holds.
* An **inactive** module is ignored: a condition referring to it drops out
  of its section, as if it were not written (OpenODD: "it is ignored").  A
  section left with nothing in it is absent: an include holds, an exclude
  does not.
* The **roots** are the entry points.  Each root candidate (by default,
  every module) that no other module refers to, by name or through a label
  it declares, is a root.  The ODD holds when its active roots hold.

Missing values follow OpenODD's *missing-value semantics*: a value the probe
could not read (``None``) does not by itself put a situation outside the ODD.
This is open-world semantics.  A condition that needs the value says so with
:meth:`OddAttribute.is_unknown` (``x: unknown`` in YAML), for example in an
exclude section.  Internally, conditions are evaluated three-valued.  A
verdict that is unknown only because values are missing counts as inside, and
is flagged as resting on missing values.

The same evaluation decides which buckets lie outside the ODD.  The bucket
stands for the attribute's value, and every other attribute is left open.
A bucket is outside when the ODD fails for every value in it, whatever the
other attributes are.

Modules follow ISO 34503's "default" definition mode, as OpenODD requires:
whatever no module rules out is inside the ODD.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

from ..coverage.items import (
    CoverGroup,
    HOLDS,
    CoverItem,
    SamplingEvent,
    _check_criteria,
    duplicates,
    value_label,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OddAttribute",
    "OddCondition",
    "OddDefinition",
    "OddModule",
    "OddVerdict",
    "UNDECIDED",
    "all_of",
    "any_of",
    "module_holds",
    "read_probe",
]

Probe = Callable[[Any], Any]

#: What a module's or label's entry in the evaluation's truth table holds.
#: ``INACTIVE`` marks an inactive module (or a label only inactive modules
#: declare): a condition referring to it drops out of its section.
INACTIVE = "inactive"
Truth = Mapping[str, Union[Optional[bool], str]]


def read_probe(probe: Probe, world: Any) -> Any:
    """What *probe* reads from *world*; ``None`` (missing) when it raises."""
    try:
        return probe(world)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Three-valued logic, used internally
# ---------------------------------------------------------------------------


def _and(values: Iterable[Any]) -> Optional[bool]:
    unknown = False
    for v in values:
        if v == INACTIVE:
            continue
        if v is False:
            return False
        if v is None:
            unknown = True
    return None if unknown else True


def _or(values: Iterable[Any]) -> Optional[bool]:
    unknown = False
    for v in values:
        if v == INACTIVE:
            continue
        if v is True:
            return True
        if v is None:
            unknown = True
    return None if unknown else False


def _not(value: Optional[bool]) -> Optional[bool]:
    return None if value is None else not value


# ---------------------------------------------------------------------------
# Abstract values: what deciding the buckets outside the ODD evaluates
# ---------------------------------------------------------------------------


class _Open:
    """The type of :data:`UNDECIDED`."""

    def __repr__(self) -> str:
        return "UNDECIDED"

    def __reduce__(self) -> str:
        return "UNDECIDED"  # pickled and copied as the one instance


#: An attribute whose value is open: present, but could be anything.
_OPEN = _Open()

#: What a map-only probe (a probe's ``on_lanelet``, see
#: :mod:`~autoware_carla_scenario.odd.route`) returns for a value the map
#: cannot decide, because only the run knows it: CARLA's speed limit on a
#: lanelet with no ``speed_limit`` tag, say.  It is not missing (``None``):
#: the run will read a value, which could be anything.  Conditions on it are
#: unknown.
UNDECIDED: Any = _OPEN


@dataclass(frozen=True)
class _Bucket:
    """Every value of one bucket of a cover item."""

    item: CoverItem
    index: int


# ---------------------------------------------------------------------------
# Predicates on one value
# ---------------------------------------------------------------------------


def _number(value: Any) -> Optional[float]:
    """*value* as a float when it is a number (not a bool), else ``None``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _missing_value(value: Any) -> bool:
    """Whether *value* says nothing: ``None``, or a NaN (say, 0/0)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


@dataclass(frozen=True)
class _InSet:
    labels: frozenset[str]
    #: The numbers among the values, compared as numbers (30 is 30.0).
    numbers: frozenset[float] = frozenset()

    def holds(self, value: Any) -> bool:
        number = _number(value)
        if number is not None and number in self.numbers:
            return True
        return value_label(value) in self.labels

    def describe(self) -> str:
        return "in [" + ", ".join(sorted(self.labels)) + "]"


@dataclass(frozen=True)
class _Interval:
    low: float = -math.inf
    high: float = math.inf
    include_low: bool = True
    include_high: bool = True

    def holds(self, value: Any) -> bool:
        x = float(value)
        if math.isnan(x):
            return False
        low_ok = x > self.low or (self.include_low and x == self.low)
        high_ok = x < self.high or (self.include_high and x == self.high)
        return low_ok and high_ok

    def on_bucket(self, low: float, high: float, closed: bool) -> Optional[bool]:
        """Whether ``[low, high)`` (``]`` when *closed*) lies within: ``None``, partly."""
        if self.holds(low) and (self.holds(high) if closed else high <= self.high):
            return True
        below = high < self.low or (
            high == self.low and (not closed or not self.include_low)
        )
        above = low > self.high or (low == self.high and not self.include_high)
        return False if below or above else None

    def describe(self) -> str:
        if self.low == -math.inf:
            return f"{'<=' if self.include_high else '<'} {self.high:g}"
        if self.high == math.inf:
            return f"{'>=' if self.include_low else '>'} {self.low:g}"
        return (
            f"in {'[' if self.include_low else '('}{self.low:g} .. "
            f"{self.high:g}{']' if self.include_high else ')'}"
        )


@dataclass(frozen=True)
class _Equal:
    value: Any

    def holds(self, value: Any) -> bool:
        expected = _number(self.value)
        if expected is not None:
            actual = _number(value)
            return actual == expected if actual is not None else False
        return value_label(value) == value_label(self.value)

    def describe(self) -> str:
        return f"== {value_label(self.value)}"


_Predicate = Union[_InSet, _Interval, _Equal]


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


class OddCondition:
    """A condition of a module.  Build one from an :class:`OddAttribute`."""

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Optional[bool]:
        """Whether the condition holds for *values* (by attribute name).

        *truth* holds the verdict on every module and label evaluated so far.
        ``None`` is unknown: a value it needs is missing.
        """
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def _references(self) -> set[str]:
        """Names of the modules and labels this condition refers to."""
        return set()

    def _attributes(self) -> list["OddAttribute"]:
        return []

    def __repr__(self) -> str:
        return f"<OddCondition {self.describe()}>"


class _Leaf(OddCondition):
    def __init__(self, attribute: "OddAttribute", predicate: _Predicate) -> None:
        self.attribute = attribute
        self.predicate = predicate

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Optional[bool]:
        value = values.get(self.attribute.name)
        if value is _OPEN or _missing_value(value):
            return None
        if isinstance(value, _Bucket):
            return self._on_bucket(value)
        try:
            return self.predicate.holds(value)
        except (TypeError, ValueError):
            return False  # present, but not a value the condition can hold for

    def _on_bucket(self, bucket: _Bucket) -> Optional[bool]:
        item, i = bucket.item, bucket.index
        if not item.numeric:
            values = item.values
            assert isinstance(values, list)  # CoverItem keeps its values as a list
            try:
                return self.predicate.holds(values[i])
            except (TypeError, ValueError):
                return None
        last = i == len(item.labels) - 1
        predicate = self.predicate
        if isinstance(predicate, _Equal) and _number(predicate.value) is not None:
            point = float(predicate.value)
            predicate = _Interval(point, point)
        if isinstance(predicate, _Interval):
            return predicate.on_bucket(item.edges[i], item.edges[i + 1], last)
        return None

    def describe(self) -> str:
        return f"{self.attribute.name} {self.predicate.describe()}"

    def _attributes(self) -> list["OddAttribute"]:
        return [self.attribute]


class _Missing(OddCondition):
    """OpenODD's ``x: unknown``: true exactly when the value is missing."""

    def __init__(self, attribute: "OddAttribute") -> None:
        self.attribute = attribute

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Optional[bool]:
        value = values.get(self.attribute.name)
        return None if value is _OPEN else _missing_value(value)

    def describe(self) -> str:
        return f"{self.attribute.name} is unknown"

    def _attributes(self) -> list["OddAttribute"]:
        return [self.attribute]


class _Group(OddCondition):
    def __init__(self, op: str, children: Sequence[OddCondition]) -> None:
        if not children:
            raise ValueError(f"{op}_of() needs at least one condition")
        self.op = op
        self.children = list(children)

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Any:
        """The group's verdict; ``INACTIVE`` when every condition dropped out."""
        results = [c.evaluate(values, truth) for c in self.children]
        if all(r == INACTIVE for r in results):
            return INACTIVE
        return _and(results) if self.op == "all" else _or(results)

    def describe(self) -> str:
        joiner = " and " if self.op == "all" else " or "
        return "(" + joiner.join(c.describe() for c in self.children) + ")"

    def _references(self) -> set[str]:
        return set().union(*(c._references() for c in self.children))

    def _attributes(self) -> list["OddAttribute"]:
        return [a for c in self.children for a in c._attributes()]


class _ModuleRef(OddCondition):
    def __init__(self, name: str, holds: bool) -> None:
        self.name = name
        self.holds = holds

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Any:
        verdict = truth.get(self.name)
        if verdict == INACTIVE:
            return INACTIVE  # OpenODD: an inactive module is ignored
        if verdict is None:
            return None
        return verdict == self.holds

    def describe(self) -> str:
        return f"{self.name} is {'true' if self.holds else 'false'}"

    def _references(self) -> set[str]:
        return {self.name}


def all_of(conditions: Sequence[OddCondition]) -> OddCondition:
    """A condition that holds when every one of *conditions* does (``AND``)."""
    return _Group("all", conditions)


def any_of(conditions: Sequence[OddCondition]) -> OddCondition:
    """A condition that holds when one of *conditions* does (``OR``)."""
    return _Group("any", conditions)


def module_holds(name: str, holds: bool = True) -> OddCondition:
    """A condition on another module or a label: that it holds (or not).

    A label holds when any active module that declares it holds.  A condition
    on an inactive module drops out of its section, whatever *holds* says.
    """
    if not name:
        raise ValueError("module_holds(): name must not be empty")
    return _ModuleRef(name, holds)


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


class OddAttribute:
    """A measurable taxonomy concept.

    Args:
        name: Its name, e.g. ``"scenery.speed_limit"``.  Its cover item is
            named ``"odd." + name``.
        probe: Reads the value from the world on every tick.  ``None`` means
            the value is missing.
        unit: The unit the probe returns.  OpenODD conditions written in
            another unit are converted into it.
        range: With *every*, equal buckets (see
            :meth:`BaseScenario.register_cover`).
        every: The bucket width over *range*.
        buckets: Explicit bucket edges.
        values: One bucket per value.
        text: A description for the report.
        target: What a bucket needs to count as covered, in *cover_by*.
        cover_by: What *target* counts: ``"hits"`` (ticks, the default),
            ``"seconds"``, ``"meters"`` the ego drove, or ``"entries"``.
        min_stay: Stays in a bucket shorter than this many seconds do not
            count towards *target*.

    Give at most one of *values*, *buckets* and *range*.  An attribute with
    none of them is monitored (conditions may test it) but not covered.
    """

    def __init__(
        self,
        name: str,
        probe: Probe,
        *,
        unit: str = "",
        range: Optional[tuple[float, float]] = None,
        every: Optional[float] = None,
        buckets: Optional[Sequence[float]] = None,
        values: Optional[Iterable[Any]] = None,
        text: str = "",
        target: float = 1,
        cover_by: str = "hits",
        min_stay: Optional[float] = None,
    ) -> None:
        if not name:
            raise ValueError("OddAttribute: name must not be empty")
        unit = _measure_unit(name, probe, unit)
        self.name = name
        self.probe = probe
        self.unit = unit
        self.text = text
        self.item: Optional[CoverItem] = None
        _check_criteria(
            f"OddAttribute({name})", target, cover_by, min_stay, SamplingEvent.TICK
        )
        self.target = target
        self.cover_by = cover_by
        self.min_stay = min_stay
        self._set_buckets(range=range, every=every, buckets=buckets, values=values)

    def _set_buckets(self, **buckets: Any) -> None:
        """Make the attribute's cover item from ``range``/``every``/``buckets``/``values``."""
        if all(buckets.get(k) is None for k in ("values", "buckets", "range")):
            if buckets.get("every") is not None:
                raise ValueError(f"OddAttribute({self.name}): every needs a range")
            return
        self.item = CoverItem(
            name=f"odd.{self.name}",
            expression=self.probe,
            unit=self.unit,
            event=SamplingEvent.TICK,
            text=self.text,
            group=CoverGroup.ODD,
            target=self.target,
            cover_by=self.cover_by,
            min_stay=self.min_stay,
            **buckets,
        )

    # -- conditions ------------------------------------------------------

    def is_in(self, values: Iterable[Any]) -> OddCondition:
        """The value is one of *values* (an OpenODD list expression)."""
        values = list(values)
        if not values:
            raise ValueError(f"{self.name}.is_in(): no values")
        numbers = frozenset(n for n in map(_number, values) if n is not None)
        return _Leaf(self, _InSet(frozenset(map(value_label, values)), numbers))

    def equals(self, value: Any) -> OddCondition:
        """The value equals *value* (an OpenODD equal expression)."""
        return _Leaf(self, _Equal(value))

    def between(self, low: float, high: float) -> OddCondition:
        """``low <= value <= high``: OpenODD's ``"[low .. high]"``."""
        if not low <= high:
            raise ValueError(f"{self.name}.between(): low must not exceed high")
        return _Leaf(self, _Interval(float(low), float(high)))

    def at_least(self, bound: float) -> OddCondition:
        """``value >= bound``."""
        return _Leaf(self, _Interval(low=float(bound)))

    def at_most(self, bound: float) -> OddCondition:
        """``value <= bound``."""
        return _Leaf(self, _Interval(high=float(bound)))

    def greater_than(self, bound: float) -> OddCondition:
        """``value > bound``."""
        return _Leaf(self, _Interval(low=float(bound), include_low=False))

    def less_than(self, bound: float) -> OddCondition:
        """``value < bound``."""
        return _Leaf(self, _Interval(high=float(bound), include_high=False))

    def is_unknown(self) -> OddCondition:
        """The value is missing (OpenODD's ``unknown`` keyword).

        In an exclude section, this makes the value required.  Without it, a
        missing value does not put a situation outside the ODD.
        """
        return _Missing(self)

    def __repr__(self) -> str:
        return f"OddAttribute({self.name!r})"


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def _measure_unit(name: str, probe: Any, unit: str) -> str:
    """The unit of an attribute mapped onto a built-in scenario measure: its.

    A measure is read in its own unit, so the attribute's conditions and
    buckets are in it too; a unit given that is not the measure's would put
    every value in the wrong bucket.
    """
    from .scenario_measure import ScenarioMeasure  # noqa: PLC0415
    from .units import normalize_unit  # noqa: PLC0415

    if not isinstance(probe, ScenarioMeasure) or not probe.unit:
        return unit
    if unit and normalize_unit(unit) != normalize_unit(probe.unit):
        raise ValueError(
            f"OddAttribute({name}): measure {probe.key} is in {probe.unit!r}, "
            f"not {unit!r}"
        )
    return probe.unit


def _never_called(world: Any) -> Any:
    """A situation's expression: the collector fills it from the ODD's verdict."""
    return None


class OddModule:
    """A named rule: it holds when its include section holds and its exclude does not.

    Args:
        name: Unique within the ODD.
        include_and: All of these must hold (``INCLUDE_AND``).
        include_or: One of these must hold (``INCLUDE_OR``).
        exclude_and: The module fails when all of these hold (``EXCLUDE_AND``).
        exclude_or: The module fails when one of these holds (``EXCLUDE_OR``).
        labels: Labels the module declares.  A label holds when any active
            module declaring it holds.
        active: An inactive module is ignored: conditions referring to it drop
            out of their sections.
        text: A description for the report (OpenODD's ``TITLE``).
        situation: Cover the module as a situation: record how long, how far
            and how often it held (``odd.situation.<name>``).  A situation
            is not a root candidate by default: it is a combination the runs
            should drive, not a bound of the ODD.
        target: With *situation*: what covering it takes, in *cover_by*.
        cover_by: With *situation*: ``"hits"`` (ticks, the default),
            ``"seconds"``, ``"meters"`` or ``"entries"``.
        min_stay: With *situation*: stays shorter than this many seconds do
            not count.

    As in OpenODD, a module has at most one include section and at most one
    exclude section.
    """

    def __init__(
        self,
        name: str,
        *,
        include_and: Optional[Sequence[OddCondition]] = None,
        include_or: Optional[Sequence[OddCondition]] = None,
        exclude_and: Optional[Sequence[OddCondition]] = None,
        exclude_or: Optional[Sequence[OddCondition]] = None,
        labels: Optional[Sequence[str]] = None,
        active: bool = True,
        text: str = "",
        situation: bool = False,
        target: float = 1,
        cover_by: str = "hits",
        min_stay: Optional[float] = None,
    ) -> None:
        if not name:
            raise ValueError("OddModule: name must not be empty")
        if include_and and include_or:
            raise ValueError(
                f"OddModule({name}): one include section, INCLUDE_AND or INCLUDE_OR"
            )
        if exclude_and and exclude_or:
            raise ValueError(
                f"OddModule({name}): one exclude section, EXCLUDE_AND or EXCLUDE_OR"
            )
        self.name = name
        #: The sections, as groups: ``all`` for ``*_AND``, ``any`` for ``*_OR``.
        self.include: Optional[_Group] = (
            _Group("all", include_and)
            if include_and
            else _Group("any", include_or)
            if include_or
            else None
        )
        self.exclude: Optional[_Group] = (
            _Group("all", exclude_and)
            if exclude_and
            else _Group("any", exclude_or)
            if exclude_or
            else None
        )
        self.labels = list(labels or ())
        self.active = active
        self.text = text
        #: The module's cover item when it is covered as a situation.
        self.item: Optional[CoverItem] = None
        if situation:
            self.as_situation(target=target, cover_by=cover_by, min_stay=min_stay)
        elif (target, cover_by, min_stay) != (1, "hits", None):
            raise ValueError(
                f"OddModule({name}): target, cover_by and min_stay need situation=True"
            )

    @property
    def situation(self) -> bool:
        """Whether the module is covered as a situation."""
        return self.item is not None

    def as_situation(
        self,
        *,
        target: float = 1,
        cover_by: str = "hits",
        min_stay: Optional[float] = None,
    ) -> None:
        """Cover the module as a situation, with these criteria.

        Call it before the module is given to an :class:`OddDefinition`,
        which works its roots out when it is built.
        """
        self.item = CoverItem(
            name=f"odd.situation.{self.name}",
            expression=_never_called,
            values=[HOLDS],
            event=SamplingEvent.TICK,
            text=self.text,
            target=target,
            group=CoverGroup.SITUATION,
            cover_by=cover_by,
            min_stay=min_stay,
        )

    def _conditions(self) -> list[OddCondition]:
        return [c for s in (self.include, self.exclude) if s for c in s.children]

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Optional[bool]:
        """Whether the module holds for *values*, given *truth* of the others."""
        include = self.include.evaluate(values, truth) if self.include else True
        exclude = self.exclude.evaluate(values, truth) if self.exclude else False
        if include == INACTIVE:
            include = True  # an include with nothing left in it is absent
        if exclude == INACTIVE:
            exclude = False
        return _and([include, _not(exclude)])

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "text": self.text,
            "active": self.active,
        }
        for kind, section in (("include", self.include), ("exclude", self.exclude)):
            if section is not None:
                key = f"{kind}_{'and' if section.op == 'all' else 'or'}"
                out[key] = [c.describe() for c in section.children]
        if self.labels:
            out["labels"] = list(self.labels)
        if self.item is not None:
            out["situation"] = {
                "target": self.item.target,
                "cover_by": self.item.cover_by,
                "min_stay": self.item.min_stay,
            }
        return out


# ---------------------------------------------------------------------------
# The ODD
# ---------------------------------------------------------------------------


@dataclass
class OddVerdict:
    """The ODD's verdict on one set of values."""

    #: Whether the values are inside the ODD (missing values do not put them out).
    inside: bool
    #: Inside only because values were missing: the known values alone could
    #: not settle it.
    assumed: bool = False
    #: Every module's verdict, by name: ``True``, ``False``, ``None`` when it
    #: rests on missing values, or ``"inactive"``.
    modules: dict[str, Union[Optional[bool], str]] = field(default_factory=dict)


class OddDefinition:
    """An ODD: its attributes and the modules that bound it.

    Args:
        name: The ODD's name, written to coverage files and reports.
        attributes: The attributes, measured on every tick.
        modules: The rules.  With none, every situation is inside the ODD.
        roots: The root candidates: those no other module refers to are the
            roots (all of them, if every one is referred to).  ``None``
            takes every module.  An inactive root is ignored.
        text: A description for the report.
    """

    def __init__(
        self,
        name: str,
        attributes: Sequence[OddAttribute],
        modules: Optional[Sequence[OddModule]] = None,
        *,
        roots: Optional[Sequence[str]] = None,
        text: str = "",
    ) -> None:
        if not name:
            raise ValueError("OddDefinition: name must not be empty")
        self.name = name
        self.text = text
        self.attributes = list(attributes)
        self.modules = list(modules or ())
        #: The git sources an OpenODD ODD was read from, with their commits.
        self.sources: list[dict[str, str]] = []

        names = [a.name for a in self.attributes]
        if duplicates(names):
            raise ValueError(
                f"ODD {name}: attributes named more than once: {duplicates(names)}"
            )
        module_names = [m.name for m in self.modules]
        if duplicates(module_names):
            raise ValueError(
                f"ODD {name}: modules named more than once: {duplicates(module_names)}"
            )
        self._by_module = {m.name: m for m in self.modules}
        self._label_modules: dict[str, list[str]] = {}
        for m in self.modules:
            for label in m.labels:
                if label in self._by_module or label in names:
                    raise ValueError(
                        f"ODD {name}: label {label!r} is also a module's or an "
                        "attribute's name"
                    )
                self._label_modules.setdefault(label, []).append(m.name)
        unknown_roots = [r for r in roots or () if r not in self._by_module]
        if unknown_roots:
            raise ValueError(f"ODD {name}: no root module {unknown_roots}")

        known = set(names)
        for m in self.modules:
            for condition in m._conditions():
                for attribute in condition._attributes():
                    if attribute.name not in known:
                        raise ValueError(
                            f"ODD {name}: module {m.name} tests {attribute.name}, "
                            "which is not one of the ODD's attributes"
                        )
                for ref in condition._references():
                    if ref not in self._by_module and ref not in self._label_modules:
                        raise ValueError(
                            f"ODD {name}: module {m.name} refers to {ref!r}, "
                            "which is neither a module nor a label"
                        )
        self._order = self._evaluation_order()
        # Each label is settled after the last module declaring it.
        last: dict[str, str] = {}
        for module in self._order:
            for label in module.labels:
                last[label] = module.name
        self._labels_after: dict[str, list[str]] = {}
        for label, module_name in last.items():
            self._labels_after.setdefault(module_name, []).append(label)

        # What a bound refers to is not a root.  A situation's references do
        # not count: it is what to drive, and a bound it tests must still
        # bound the ODD.
        referenced: set[str] = set()
        for module in self.modules:
            if not module.situation:
                referenced |= self._depends_on(module)
        if roots:
            self.roots = [r for r in roots if r not in referenced] or list(roots)
        else:
            # A graph without cycles always has a module nothing refers to.
            # Situations are what to drive, not bounds of the ODD.
            self.roots = [
                m.name
                for m in self.modules
                if m.name not in referenced and not m.situation
            ]
            if self.modules and not self.roots:
                logger.warning(
                    "ODD %s: every module is a situation, so nothing bounds the "
                    "ODD; refer to one from a bound, or name it in roots",
                    name,
                )
        self._plain = [a for a in self.attributes if not _derived(a.probe)]
        self._derived = [a for a in self.attributes if _derived(a.probe)]
        self._outside = self._outside_buckets()

    # -- structure -------------------------------------------------------

    def _depends_on(self, module: OddModule) -> set[str]:
        """The modules *module* refers to, by name or through a label."""
        out: set[str] = set()
        for condition in module._conditions():
            for ref in condition._references():
                out.update(self._label_modules.get(ref, [ref]))
        return out

    def _evaluation_order(self) -> list[OddModule]:
        order: list[OddModule] = []
        state: dict[str, int] = {}  # 1 visiting, 2 done

        def visit(name: str, path: list[str]) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                cycle = " -> ".join([*path[path.index(name) :], name])
                raise ValueError(
                    f"ODD {self.name}: modules refer to each other: {cycle}"
                )
            state[name] = 1
            module = self._by_module[name]
            for dep in sorted(self._depends_on(module)):
                visit(dep, [*path, name])
            state[name] = 2
            order.append(module)

        for m in self.modules:
            visit(m.name, [])
        return order

    # -- evaluation ------------------------------------------------------

    def evaluate(self, values: Mapping[str, Any]) -> OddVerdict:
        """The ODD's verdict on *values*, by attribute name (``None``: missing)."""
        truth: dict[str, Union[Optional[bool], str]] = {}
        # The order is topological over labels too (a reference to a label
        # depends on every module declaring it), so a label is settled before
        # any module that refers to it.
        for module in self._order:
            truth[module.name] = (
                module.evaluate(values, truth) if module.active else INACTIVE
            )
            for label in self._labels_after.get(module.name, ()):
                verdicts = [
                    truth[m] for m in self._label_modules[label] if truth[m] != INACTIVE
                ]
                truth[label] = _or(verdicts) if verdicts else INACTIVE  # type: ignore[arg-type]
        verdict = _and(truth[r] for r in self.roots)  # type: ignore[misc]
        return OddVerdict(
            inside=verdict is not False,
            assumed=verdict is None,
            modules={m.name: truth[m.name] for m in self.modules},
        )

    def sample(self, world: Any) -> dict[str, Any]:
        """Every attribute's value in *world*, by name; ``None`` when missing.

        Each probe runs once.  An attribute derived from others (an OpenODD
        categorical defined by expressions) is worked out from their values.
        """
        values = {a.name: read_probe(a.probe, world) for a in self._plain}
        for attribute in self._derived:
            values[attribute.name] = attribute.probe.from_values(values)  # type: ignore[attr-defined]
        return values

    # -- buckets inside and outside the ODD --------------------------------

    def _outside_buckets(self) -> dict[str, list[str]]:
        """Attribute name -> the labels of its buckets the ODD rules out.

        A bucket is ruled out when the ODD fails for every value in it,
        whatever the other attributes are: the bucket is evaluated in place
        of the attribute's value, with every other attribute left open.
        """
        open_values: dict[str, Any] = {a.name: _OPEN for a in self.attributes}
        outside: dict[str, list[str]] = {}
        for attribute in self.attributes:
            item = attribute.item
            if item is None:
                continue
            labels = [
                label
                for index, label in enumerate(item.labels)
                if not self.evaluate(
                    self._with_derived(
                        {**open_values, attribute.name: _Bucket(item, index)},
                        attribute,
                    )
                ).inside
            ]
            if labels:
                outside[attribute.name] = labels
        return outside

    def _with_derived(
        self, values: dict[str, Any], given: OddAttribute
    ) -> dict[str, Any]:
        """*values* with every derived attribute but *given* worked out, if it can be."""
        for attribute in self._derived:
            if attribute is not given:
                value = attribute.probe.from_values(values)  # type: ignore[attr-defined]
                values[attribute.name] = _OPEN if value is None else value
        return values

    def outside_buckets(self, attribute: OddAttribute) -> list[str]:
        """The labels of *attribute*'s buckets that lie outside the ODD."""
        return list(self._outside.get(attribute.name, ()))

    def cover_items(self) -> list[CoverItem]:
        """The cover items of the attributes that have buckets."""
        return [a.item for a in self.attributes if a.item is not None]

    def situations(self) -> list[OddModule]:
        """The modules covered as situations."""
        return [m for m in self.modules if m.item is not None]

    def unmeasured(self) -> list[str]:
        """Attributes with no buckets: monitored, but not covered."""
        return [a.name for a in self.attributes if a.item is None]

    def describe(self) -> dict[str, Any]:
        """The ODD's definition, as written to a coverage file."""
        return {
            "name": self.name,
            "text": self.text,
            "roots": list(self.roots),
            "modules": [m.describe() for m in self.modules],
            "unmeasured": self.unmeasured(),
            **({"sources": list(self.sources)} if self.sources else {}),
        }

    def __repr__(self) -> str:
        return f"OddDefinition({self.name!r})"


def _derived(probe: Any) -> bool:
    """Whether *probe* derives its value from the other attributes' values."""
    return callable(getattr(probe, "from_values", None))
