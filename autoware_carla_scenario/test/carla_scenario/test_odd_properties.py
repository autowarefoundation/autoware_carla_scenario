"""Property tests: random ODDs, generated from OpenODD's grammar, against its semantics.

Hypothesis builds random ODDs within the rules of ASAM OpenODD 1.0:

* a taxonomy of numeric and categorical concepts;
* modules with one include and one exclude section, nested one level with
  the other operator;
* bounds, ranges, literal lists and ``unknown``;
* references to earlier modules and to labels, and inactive modules.

Each ODD is checked against an independent reference written straight from
the standard's text (chapter 6.4: ``MODULE === INCLUDE AND (NOT EXCLUDE)``,
labels, inactive modules, roots).  It is also written out as OpenODD YAML,
read back with :func:`load_openodd`, and checked to judge every situation as
the Python ODD does.

The number of examples is bounded and the run is derandomized, so the suite
stays a few seconds long and gives the same verdict on every run.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from autoware_carla_scenario.odd import (
    OddAttribute,
    OddCondition,
    OddDefinition,
    OddModule,
    all_of,
    any_of,
    load_openodd,
    module_holds,
)

_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)

#: Numeric concepts take integer thresholds in [0, 10]; values are drawn
#: around and on them.
_NUMBERS = [-1.0, 0.0, 0.5, 2.0, 3.0, 4.5, 5.0, 7.0, 10.0, 11.0]


# ---------------------------------------------------------------------------
# A plain description of an ODD: what the strategies draw
# ---------------------------------------------------------------------------


@dataclass
class Concept:
    name: str
    literals: Optional[list[str]] = None  # None: numeric

    @property
    def numeric(self) -> bool:
        return self.literals is None


@dataclass
class Leaf:
    concept: Concept
    op: str  # "<", "<=", ">", ">=", "range", "in", "eq", "unknown"
    args: tuple[Any, ...] = ()


@dataclass
class Ref:
    name: str
    holds: bool


@dataclass
class Group:
    op: str  # "AND" or "OR"
    children: list[Any]


@dataclass
class Module:
    name: str
    include: Optional[Group] = None
    exclude: Optional[Group] = None
    labels: list[str] = field(default_factory=list)
    active: bool = True


@dataclass
class Odd:
    concepts: list[Concept]
    modules: list[Module]


@st.composite
def _leaf(draw: Any, concepts: list[Concept]) -> Leaf:
    concept = draw(st.sampled_from(concepts))
    if draw(st.integers(0, 9)) == 0:
        return Leaf(concept, "unknown")
    if concept.numeric:
        op = draw(st.sampled_from(["<", "<=", ">", ">=", "range", "eq"]))
        a, b = sorted(draw(st.lists(st.integers(0, 10), min_size=2, max_size=2)))
        return Leaf(concept, op, (a, b) if op == "range" else (a,))
    assert concept.literals is not None
    if draw(st.booleans()):
        chosen = draw(
            st.lists(st.sampled_from(concept.literals), min_size=1, unique=True)
        )
        return Leaf(concept, "in", tuple(chosen))
    return Leaf(concept, "eq", (draw(st.sampled_from(concept.literals)),))


@st.composite
def _section(draw: Any, op: str, concepts: list[Concept], refs: list[str]) -> Group:
    def item(nested: bool) -> Any:
        if refs and draw(st.integers(0, 3)) == 0:
            return Ref(draw(st.sampled_from(refs)), draw(st.booleans()))
        if not nested and draw(st.integers(0, 4)) == 0:
            other = "OR" if op == "AND" else "AND"
            return Group(
                other, _unique([item(True) for _ in range(draw(st.integers(1, 3)))])
            )
        return draw(_leaf(concepts))

    return Group(op, _unique([item(False) for _ in range(draw(st.integers(1, 3)))]))


def _unique(children: list[Any]) -> list[Any]:
    """One entry per key: a YAML section's keys must be unique."""
    seen: set[str] = set()
    out = []
    for child in children:
        key = _key(child)
        if key not in seen:
            seen.add(key)
            out.append(child)
    return out


def _key(child: Any) -> str:
    if isinstance(child, Leaf):
        return child.concept.name
    if isinstance(child, Ref):
        return child.name
    return child.op


@st.composite
def odds(draw: Any) -> Odd:
    concepts: list[Concept] = []
    for i in range(draw(st.integers(1, 3))):
        if draw(st.booleans()):
            concepts.append(Concept(f"num_{i}"))
        else:
            n = draw(st.integers(2, 4))
            concepts.append(Concept(f"cat_{i}", [f"c{i}_{j}" for j in range(n)]))
    modules: list[Module] = []
    labels: list[str] = []
    for i in range(draw(st.integers(1, 4))):
        module = Module(f"m{i}", active=draw(st.integers(0, 4)) != 0)
        if draw(st.integers(0, 2)) == 0:
            module.labels.append(draw(st.sampled_from(["lab_a", "lab_b"])))
        # Earlier modules and their labels, but not a label this module
        # declares: a module referring to itself is a cycle, refused.
        refs = [m.name for m in modules] + [
            label for label in labels if label not in module.labels
        ]
        # OpenODD requires a section; every module gets an include one.
        module.include = draw(
            _section(draw(st.sampled_from(["AND", "OR"])), concepts, refs)
        )
        if draw(st.booleans()):
            module.exclude = draw(
                _section(draw(st.sampled_from(["AND", "OR"])), concepts, refs)
            )
        for label in module.labels:
            if label not in labels:
                labels.append(label)
        modules.append(module)
    return Odd(concepts, modules)


@st.composite
def situations(draw: Any, odd: Odd, missing: bool = False) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for concept in odd.concepts:
        if missing and draw(st.booleans()):
            values[concept.name] = None
        elif concept.numeric:
            values[concept.name] = draw(st.sampled_from(_NUMBERS))
        else:
            assert concept.literals is not None
            values[concept.name] = draw(st.sampled_from(concept.literals))
    return values


# ---------------------------------------------------------------------------
# The ODD under test, built from the description
# ---------------------------------------------------------------------------


def build(odd: Odd) -> OddDefinition:
    attributes = {
        c.name: OddAttribute(
            c.name,
            lambda world: None,
            buckets=[0, 2.5, 5, 10] if c.numeric else None,
            values=c.literals,
        )
        for c in odd.concepts
    }

    def condition(child: Any) -> OddCondition:
        if isinstance(child, Ref):
            return module_holds(child.name, child.holds)
        if isinstance(child, Group):
            parts = [condition(c) for c in child.children]
            return all_of(parts) if child.op == "AND" else any_of(parts)
        a = attributes[child.concept.name]
        op, args = child.op, child.args
        if op == "unknown":
            return a.is_unknown()
        if op == "range":
            return a.between(*args)
        if op == "in":
            return a.is_in(list(args))
        if op == "eq":
            return a.equals(float(args[0]) if child.concept.numeric else args[0])
        return {
            "<": a.less_than,
            "<=": a.at_most,
            ">": a.greater_than,
            ">=": a.at_least,
        }[op](float(args[0]))

    def section(group: Optional[Group], kind: str) -> dict[str, Any]:
        if group is None:
            return {}
        key = f"{kind}_{'and' if group.op == 'AND' else 'or'}"
        return {key: [condition(c) for c in group.children]}

    modules = [
        OddModule(
            m.name,
            **section(m.include, "include"),
            **section(m.exclude, "exclude"),
            labels=m.labels,
            active=m.active,
        )
        for m in odd.modules
    ]
    return OddDefinition("random", list(attributes.values()), modules)


def to_yaml(odd: Odd) -> str:
    """The ODD as an OpenODD 1.0 YAML document (chapter 10)."""
    lines = ["TAXONOMY:"]
    for c in odd.concepts:
        if c.numeric:
            lines.append(f"    {c.name}: float length")
        else:
            lines.append(f"    {c.name}: [{', '.join(c.literals or [])}]")
    lines.append("MODULES:")

    def expression(leaf: Leaf) -> str:
        op, args = leaf.op, leaf.args
        if op == "unknown":
            return "unknown"
        if op == "range":
            return f'"[{args[0]} .. {args[1]}] m"'
        if op == "in":
            return "[" + ", ".join(args) + "]"
        if op == "eq":
            return f'"{args[0]} m"' if leaf.concept.numeric else str(args[0])
        return f'"{op} {args[0]} m"'

    def emit(children: list[Any], indent: int) -> None:
        pad = " " * indent
        for child in children:
            if isinstance(child, Ref):
                lines.append(f"{pad}{child.name}: {'true' if child.holds else 'false'}")
            elif isinstance(child, Group):
                lines.append(f"{pad}{child.op}:")
                emit(child.children, indent + 4)
            else:
                lines.append(f"{pad}{child.concept.name}: {expression(child)}")

    for m in odd.modules:
        lines.append(f"    {m.name}:")
        if not m.active:
            lines.append("        ACTIVE: false")
        if m.labels:
            lines.append(f"        LABELS: [{', '.join(m.labels)}]")
        for kind, group in (("INCLUDE", m.include), ("EXCLUDE", m.exclude)):
            if group is not None:
                lines.append(f"        {kind}_{group.op}:")
                emit(group.children, 12)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The reference: OpenODD's semantics, two-valued, from the standard's text
# ---------------------------------------------------------------------------


def reference(odd: Odd, values: dict[str, Any]) -> bool:
    """Whether *values* (none missing) are inside *odd*, per OpenODD 6.4."""
    truth: dict[str, Any] = {}
    by_name = {m.name: m for m in odd.modules}
    label_members: dict[str, list[str]] = {}
    for m in odd.modules:
        for label in m.labels:
            label_members.setdefault(label, []).append(m.name)

    def leaf(child: Leaf) -> bool:
        value = values[child.concept.name]
        op, args = child.op, child.args
        if op == "unknown":
            return value is None
        if op == "range":
            return args[0] <= value <= args[1]
        if op == "in":
            return value in args
        if op == "eq":
            return value == (float(args[0]) if child.concept.numeric else args[0])
        x = float(args[0])
        return {
            "<": value < x,
            "<=": value <= x,
            ">": value > x,
            ">=": value >= x,
        }[op]

    ignored = object()  # a condition on an inactive module: "it is ignored"

    def ref(child: Ref) -> Any:
        members = label_members.get(child.name, [child.name])
        active = [n for n in members if by_name[n].active]
        if not active:
            return ignored
        return any(module(n) for n in active) == child.holds

    def holds(child: Any) -> Any:
        if isinstance(child, Ref):
            return ref(child)
        if isinstance(child, Group):
            results = [r for r in map(holds, child.children) if r is not ignored]
            if not results:
                return ignored
            return all(results) if child.op == "AND" else any(results)
        return leaf(child)

    def module(name: str) -> bool:
        if name not in truth:
            m = by_name[name]
            include = holds(m.include) if m.include else True
            exclude = holds(m.exclude) if m.exclude else False
            truth[name] = (include is ignored or include) and not (
                exclude is not ignored and exclude
            )
        return truth[name]

    referenced: set[str] = set()

    def walk(child: Any) -> None:
        if isinstance(child, Ref):
            referenced.update(label_members.get(child.name, [child.name]))
        elif isinstance(child, Group):
            for c in child.children:
                walk(c)

    for m in odd.modules:
        for group in (m.include, m.exclude):
            if group is not None:
                walk(group)
    roots = [m.name for m in odd.modules if m.name not in referenced]
    return all(module(r) for r in roots if by_name[r].active)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@_SETTINGS
@given(st.data())
def test_the_odd_judges_as_the_standard_says(data: Any) -> None:
    odd = data.draw(odds())
    definition = build(odd)
    for _ in range(5):
        values = data.draw(situations(odd))
        assert definition.evaluate(values).inside is reference(odd, values), (
            to_yaml(odd),
            values,
        )


@_SETTINGS
@given(st.data())
def test_the_yaml_odd_judges_as_the_python_one(data: Any) -> None:
    odd = data.draw(odds())
    python = build(odd)
    yaml_odd = load_openodd(to_yaml(odd), name="random")
    assert yaml_odd.roots == python.roots
    for _ in range(5):
        values = data.draw(situations(odd, missing=True))
        assert yaml_odd.evaluate(values) == python.evaluate(values), (
            to_yaml(odd),
            values,
        )


def _completions(odd: Odd, values: dict[str, Any]) -> list[dict[str, Any]]:
    """Every way to fill the missing values (from a small domain)."""
    choices = [
        [values[c.name]]
        if values[c.name] is not None
        else (_NUMBERS if c.numeric else list(c.literals or []))
        for c in odd.concepts
    ]
    return [
        {c.name: v for c, v in zip(odd.concepts, combo)}
        for combo in itertools.product(*choices)
    ]


@_SETTINGS
@given(st.data())
def test_missing_values_never_rule_a_situation_out_alone(data: Any) -> None:
    # Missing-value semantics: outside only when no completion is inside.
    odd = data.draw(odds())
    if any(
        isinstance(c, Leaf) and c.op == "unknown"
        for m in odd.modules
        for g in (m.include, m.exclude)
        if g is not None
        for c in _flatten(g)
    ):
        return  # `unknown` tests the missing value itself
    definition = build(odd)
    values = data.draw(situations(odd, missing=True))
    verdict = definition.evaluate(values)
    completions = _completions(odd, values)
    if any(reference(odd, c) for c in completions):
        assert verdict.inside, (to_yaml(odd), values)
    if all(v is not None for v in values.values()):
        assert verdict.assumed is False


def _flatten(group: Group) -> list[Any]:
    out: list[Any] = []
    for child in group.children:
        out += _flatten(child) if isinstance(child, Group) else [child]
    return out


@_SETTINGS
@given(st.data())
def test_a_bucket_outside_the_odd_has_no_value_inside(data: Any) -> None:
    odd = data.draw(odds())
    definition = build(odd)
    for attribute, concept in zip(definition.attributes, odd.concepts):
        outside = definition.outside_buckets(attribute)
        if not outside:
            continue
        item = attribute.item
        assert item is not None
        for label in outside:
            points: list[Any]
            if concept.numeric:
                i = item.labels.index(label)
                low, high = item.edges[i], item.edges[i + 1]
                points = [low, (low + high) / 2] + (
                    [high] if i == len(item.labels) - 1 else []
                )
            else:
                points = [label]
            for point in points:
                for other in _completions(
                    odd, {c.name: None for c in odd.concepts} | {concept.name: point}
                )[:50]:
                    assert not reference(odd, other), (to_yaml(odd), label, other)


def test_the_generator_covers_the_grammar() -> None:
    # A sanity check that the strategies reach every construct at all.
    ops: set[str] = set()
    kinds: set[str] = set()

    @settings(max_examples=60, deadline=None, derandomize=True, database=None)
    @given(odds())
    def collect(odd: Odd) -> None:
        for m in odd.modules:
            if not m.active:
                kinds.add("inactive")
            if m.labels:
                kinds.add("label")
            for g in (m.include, m.exclude):
                if g is None:
                    continue
                kinds.add(g.op)
                for c in g.children:
                    if isinstance(c, Group):
                        kinds.add("nested")
                    if isinstance(c, Ref):
                        kinds.add("ref")
                for c in _flatten(g):
                    if isinstance(c, Leaf):
                        ops.add(c.op)

    collect()
    assert ops == {"<", "<=", ">", ">=", "range", "in", "eq", "unknown"}
    assert kinds >= {"AND", "OR", "nested", "ref", "label", "inactive"}
