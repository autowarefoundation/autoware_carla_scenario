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
* An **inactive** module is ignored: every condition referring to it is
  satisfied.
* The **root** modules are the entry points.  They are the one given as
  ``root``, otherwise every active module no other module refers to, either
  by name or through a label it declares.  The ODD holds when its roots
  hold.

Missing values follow OpenODD's *missing-value semantics*: a value the probe
could not read (``None``) does not by itself put a situation outside the ODD.
This is open-world semantics.  A condition that needs the value says so with
:meth:`OddAttribute.is_unknown` (``x: unknown`` in YAML), for example in an
exclude section.  Internally, conditions are evaluated three-valued.  A
verdict that is unknown only because values are missing counts as inside, and
is flagged as resting on missing values.

Modules follow ISO 34503's "default" definition mode, as OpenODD requires:
whatever no module rules out is inside the ODD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

from ..coverage.items import CoverGroup, CoverItem, SamplingEvent, value_label

__all__ = [
    "OddAttribute",
    "OddCondition",
    "OddDefinition",
    "OddModule",
    "OddVerdict",
    "all_of",
    "any_of",
    "module_holds",
]

Probe = Callable[[Any], Any]

#: What a module's or label's entry in the evaluation's truth table holds.
#: ``INACTIVE`` marks an inactive module, which conditions referring to it
#: take as satisfied.
INACTIVE = "inactive"
Truth = Mapping[str, Union[Optional[bool], str]]


# ---------------------------------------------------------------------------
# Three-valued logic, used internally
# ---------------------------------------------------------------------------


def _and(values: Iterable[Optional[bool]]) -> Optional[bool]:
    unknown = False
    for v in values:
        if v is False:
            return False
        if v is None:
            unknown = True
    return None if unknown else True


def _or(values: Iterable[Optional[bool]]) -> Optional[bool]:
    unknown = False
    for v in values:
        if v is True:
            return True
        if v is None:
            unknown = True
    return None if unknown else False


def _not(value: Optional[bool]) -> Optional[bool]:
    return None if value is None else not value


# ---------------------------------------------------------------------------
# Predicates on one value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _InSet:
    labels: frozenset[str]

    def holds(self, value: Any) -> bool:
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

    def contains_bucket(self, low: float, high: float, closed: bool) -> bool:
        """Whether every value in ``[low, high)`` (``]`` when *closed*) holds."""
        low_ok = low > self.low or (self.include_low and low == self.low)
        if closed:
            high_ok = high < self.high or (self.include_high and high == self.high)
        else:
            high_ok = high <= self.high
        return low_ok and high_ok

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
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            try:
                return float(value) == float(self.value)
            except (TypeError, ValueError):
                return False
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
        if value is None:
            return None
        try:
            return self.predicate.holds(value)
        except (TypeError, ValueError):
            return None

    def describe(self) -> str:
        return f"{self.attribute.name} {self.predicate.describe()}"

    def _attributes(self) -> list["OddAttribute"]:
        return [self.attribute]

    def contains_bucket(self, item: CoverItem, index: int) -> bool:
        """Whether every value of bucket *index* of *item* satisfies this leaf."""
        if item.numeric:
            if not isinstance(self.predicate, _Interval):
                return False
            last = index == len(item.labels) - 1
            return self.predicate.contains_bucket(
                item.edges[index], item.edges[index + 1], closed=last
            )
        assert item.values is not None
        value = list(item.values)[index]
        try:
            return self.predicate.holds(value)
        except (TypeError, ValueError):
            return False


class _Missing(OddCondition):
    """OpenODD's ``x: unknown``: true exactly when the value is missing."""

    def __init__(self, attribute: "OddAttribute") -> None:
        self.attribute = attribute

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Optional[bool]:
        return values.get(self.attribute.name) is None

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

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Optional[bool]:
        results = (c.evaluate(values, truth) for c in self.children)
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

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Optional[bool]:
        verdict = truth.get(self.name)
        if verdict == INACTIVE:
            return True  # OpenODD: a condition on an inactive module is satisfied
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
    on an inactive module is satisfied whatever *holds* says.
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
    ) -> None:
        if not name:
            raise ValueError("OddAttribute: name must not be empty")
        self.name = name
        self.probe = probe
        self.unit = unit
        self.text = text
        self.item: Optional[CoverItem] = None
        if values is not None or buckets is not None or range is not None:
            self.item = CoverItem(
                name=f"odd.{name}",
                expression=probe,
                unit=unit,
                range=range,
                every=every,
                buckets=buckets,
                values=values,
                event=SamplingEvent.TICK,
                text=text,
                group=CoverGroup.ODD,
            )
        elif every is not None:
            raise ValueError(f"OddAttribute({name}): every needs a range")

    # -- conditions ------------------------------------------------------

    def is_in(self, values: Iterable[Any]) -> OddCondition:
        """The value is one of *values* (an OpenODD list expression)."""
        labels = frozenset(value_label(v) for v in values)
        if not labels:
            raise ValueError(f"{self.name}.is_in(): no values")
        return _Leaf(self, _InSet(labels))

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
        active: An inactive module is ignored: conditions referring to it are
            satisfied.
        text: A description for the report (OpenODD's ``TITLE``).

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
        self.include_and = list(include_and or ())
        self.include_or = list(include_or or ())
        self.exclude_and = list(exclude_and or ())
        self.exclude_or = list(exclude_or or ())
        self.labels = list(labels or ())
        self.active = active
        self.text = text

    def _conditions(self) -> list[OddCondition]:
        return [
            *self.include_and,
            *self.include_or,
            *self.exclude_and,
            *self.exclude_or,
        ]

    def evaluate(self, values: Mapping[str, Any], truth: Truth) -> Optional[bool]:
        """Whether the module holds for *values*, given *truth* of the others."""
        if self.include_or:
            include = _or(c.evaluate(values, truth) for c in self.include_or)
        else:
            include = _and(c.evaluate(values, truth) for c in self.include_and)
        if self.exclude_and:
            exclude = _and(c.evaluate(values, truth) for c in self.exclude_and)
        else:
            exclude = _or(c.evaluate(values, truth) for c in self.exclude_or)
        return _and([include, _not(exclude)])

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "text": self.text,
            "active": self.active,
        }
        for key in ("include_and", "include_or", "exclude_and", "exclude_or"):
            conditions = getattr(self, key)
            if conditions:
                out[key] = [c.describe() for c in conditions]
        if self.labels:
            out["labels"] = list(self.labels)
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
        root: The module whose verdict is the ODD's.  ``None`` takes every
            active module that no other module refers to.
        text: A description for the report.
    """

    def __init__(
        self,
        name: str,
        attributes: Sequence[OddAttribute],
        modules: Optional[Sequence[OddModule]] = None,
        *,
        root: Optional[str] = None,
        text: str = "",
    ) -> None:
        if not name:
            raise ValueError("OddDefinition: name must not be empty")
        self.name = name
        self.text = text
        self.attributes = list(attributes)
        self.modules = list(modules or ())

        names = [a.name for a in self.attributes]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"ODD {name}: attributes named more than once: {duplicates}"
            )
        module_names = [m.name for m in self.modules]
        duplicates = sorted({n for n in module_names if module_names.count(n) > 1})
        if duplicates:
            raise ValueError(f"ODD {name}: modules named more than once: {duplicates}")
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
        if root is not None and root not in self._by_module:
            raise ValueError(f"ODD {name}: root module {root!r} does not exist")

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
        self.roots = [root] if root is not None else self._unreferenced()
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

    def _unreferenced(self) -> list[str]:
        """Active modules no other module refers to: the roots, by default."""
        referenced: set[str] = set()
        for module in self.modules:
            referenced |= self._depends_on(module)
        return [m.name for m in self.modules if m.active and m.name not in referenced]

    def _use_roots(self, roots: Sequence[str]) -> None:
        """Make *roots* the entry points (an OpenODD document's ``ODD`` modules)."""
        unknown = [r for r in roots if r not in self._by_module]
        if unknown:
            raise ValueError(f"ODD {self.name}: no root module {unknown}")
        self.roots = list(roots)
        self._outside = self._outside_buckets()

    # -- evaluation ------------------------------------------------------

    def evaluate(self, values: Mapping[str, Any]) -> OddVerdict:
        """The ODD's verdict on *values*, by attribute name (``None``: missing)."""
        truth: dict[str, Union[Optional[bool], str]] = {}
        # The order is topological over labels too (a reference to a label
        # depends on every module declaring it), so each label is complete
        # before any module that refers to it is evaluated.
        for module in self._order:
            truth[module.name] = (
                module.evaluate(values, truth) if module.active else INACTIVE
            )
            for label in module.labels:
                verdicts = [
                    truth[m]
                    for m in self._label_modules[label]
                    if m in truth and truth[m] != INACTIVE
                ]
                truth[label] = _or(verdicts) if verdicts else INACTIVE  # type: ignore[arg-type]
        verdict = _and(truth[r] for r in self.roots)  # type: ignore[misc]
        return OddVerdict(
            inside=verdict is not False,
            assumed=verdict is None,
            modules={m.name: truth[m.name] for m in self.modules},
        )

    def sample(self, world: Any) -> dict[str, Any]:
        """Every attribute's value in *world*; a probe that raises gives ``None``."""
        values: dict[str, Any] = {}
        for attribute in self.attributes:
            try:
                values[attribute.name] = attribute.probe(world)
            except Exception:
                values[attribute.name] = None
        return values

    # -- buckets inside and outside the ODD --------------------------------

    def _members(self, name: str) -> list[str]:
        """The modules a module or label name stands for."""
        return self._label_modules.get(name, [name] if name in self._by_module else [])

    def _requirements(self) -> tuple[list[_Leaf], list[_Leaf]]:
        """Single-attribute conditions that must hold, and that must not.

        Walks from the roots through module references: a module the ODD
        needs to hold contributes its include-AND leaves (and a lone
        include-OR leaf) as conditions that must hold, and its exclude leaves
        as ones that must not.  A module it needs *not* to hold contributes
        the leaves that alone make it hold.
        """
        must_hold: list[_Leaf] = []
        must_not: list[_Leaf] = []
        seen: set[tuple[str, bool]] = set()
        pending = [(r, True) for r in self.roots]
        while pending:
            name, holds = pending.pop()
            if (name, holds) in seen:
                continue
            seen.add((name, holds))
            module = self._by_module[name]
            if not module.active:
                continue
            include = module.include_and or module.include_or
            single_include = len(module.include_or) == 1 or bool(module.include_and)
            if holds:
                if single_include:
                    must_hold += [c for c in include if isinstance(c, _Leaf)]
                excludes = module.exclude_or or (
                    module.exclude_and if len(module.exclude_and) == 1 else []
                )
                must_not += [c for c in excludes if isinstance(c, _Leaf)]
                for c in module.include_and:
                    if isinstance(c, _ModuleRef):
                        pending += [(m, c.holds) for m in self._members(c.name)]
                for c in module.exclude_or:
                    if isinstance(c, _ModuleRef):
                        pending += [(m, not c.holds) for m in self._members(c.name)]
            elif not module.exclude_and and not module.exclude_or:
                # Without an exclude section the module holds whenever its
                # include section does, so what makes that hold must not.
                if module.include_or:
                    must_not += [c for c in module.include_or if isinstance(c, _Leaf)]
                elif len(module.include_and) == 1 and isinstance(
                    module.include_and[0], _Leaf
                ):
                    must_not.append(module.include_and[0])
        return must_hold, must_not

    def _outside_buckets(self) -> dict[str, list[str]]:
        """Attribute name -> the labels of its buckets the ODD rules out.

        A bucket is ruled out when every value in it fails a condition that
        must hold, or meets one that must not (:meth:`_requirements`).  A
        condition that ties attributes together can rule out a combination,
        never a bucket on its own, so it leaves the bucket in.
        """
        must_hold, must_not = self._requirements()
        outside: dict[str, list[str]] = {}
        for attribute in self.attributes:
            item = attribute.item
            if item is None:
                continue
            labels = [
                label
                for index, label in enumerate(item.labels)
                if any(
                    leaf.attribute is attribute
                    and not leaf.contains_bucket(item, index)
                    for leaf in must_hold
                )
                or any(
                    leaf.attribute is attribute and leaf.contains_bucket(item, index)
                    for leaf in must_not
                )
            ]
            if labels:
                outside[attribute.name] = labels
        return outside

    def outside_buckets(self, attribute: OddAttribute) -> list[str]:
        """The labels of *attribute*'s buckets that lie outside the ODD."""
        return list(self._outside.get(attribute.name, ()))

    def cover_items(self) -> list[CoverItem]:
        """The cover items of the attributes that have buckets."""
        items = []
        for attribute in self.attributes:
            if attribute.item is not None:
                attribute.item.outside = self.outside_buckets(attribute)
                items.append(attribute.item)
        return items

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
        }

    def __repr__(self) -> str:
        return f"OddDefinition({self.name!r})"
