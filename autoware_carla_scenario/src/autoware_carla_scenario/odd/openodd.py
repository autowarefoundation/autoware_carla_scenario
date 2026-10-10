"""Read an ODD written in ASAM OpenODD 1.0 YAML (chapter 10 of the standard).

The reader builds the same :class:`OddDefinition` a Python ODD builds, so a
YAML ODD is measured, reported and judged exactly like one written in code.

An OpenODD document holds what an ODD *is*.  How this framework *measures*
it lives in a separate **binding file** (:func:`load_odd_binding`): which
probe reads each concept, and its buckets.  The standard does not allow
constructs of its own in an OpenODD document, so they are kept apart.

What is read
============

``IMPORT``: other OpenODD files, relative to the importing one (a cycle is
refused).

``TAXONOMY``: the concepts.  Nested mappings are records and containers.  A
leaf is one of:

* ``<primitive> <unit type>``, with primitive ``integer``, ``long``,
  ``float`` or ``double``: a numeric concept;
* ``boolean``;
* a list: a categorical;
* a mapping of literals to expressions on other concepts: a categorical
  defined by them (``rainfall_level: {no_rain: {rainfall_rate: "< 0.1
  mm/h"}, ...}``).  Its value is the literal whose expressions hold;
* the id of a categorical: a categorical with the same literals.

A reference to a record (a user-defined type), and a ``shapefile``, are not
followed.

``MODULES`` and ``ODD``: the modules.  Those under ``ODD`` are the root
candidates, the entry points.  When there is no ``ODD`` section, every module
no other module refers to is a root.  The ODD holds when all its roots hold.

A module has the following keys:

* ``TITLE`` and ``DESCRIPTION``;
* ``ACTIVE``;
* ``LABEL`` / ``LABELS`` (a name or a list of names);
* ``METADATA`` (ignored);
* at most one of ``INCLUDE_AND`` / ``INCLUDE_OR``;
* at most one of ``EXCLUDE_AND`` / ``EXCLUDE_OR``.

A section may nest one level of the other operator (``AND`` in an ``OR``
section, ``OR`` in an ``AND`` one).

A section maps either a concept to an expression, or a module or label to
``true`` / ``false``.  Concepts are named by their id (``rainfall_rate``),
or with as much of their path as makes them unique
(``wind.speed``).  The expressions are:

* ``value``: equal, for example ``low``, ``3``, ``true``, ``"0 mm/h"``;
* ``[a, b]``: one of the literals;
* ``"> x unit"``, ``">="``, ``"<"``, ``"<="``: a bound;
* ``"[low .. high] unit"`` (or ``[low, high]``): an inclusive range;
* ``"< heavy_rain"`` / ``"[light .. heavy]"`` on a categorical whose
  literals are defined by ranges: a bound or range in their order;
* ``unknown`` (or ``none``, ``null``, ``undefined``): the value is missing.

A number's unit is checked against the concept's unit type and converted into
the probe's unit.  Units come from :mod:`.units`, plus those a document adds
with ``conversion:``.

Not read: ``COD`` / ``OD`` records, numeric terms (``1.75*ego_width``),
``$`` parameters and condition-level metadata.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import yaml

from . import probes
from .model import (
    OddAttribute,
    OddCondition,
    OddDefinition,
    OddModule,
    _Equal,
    _Group,
    _Interval,
    _Leaf,
    all_of,
    any_of,
    module_holds,
    read_probe,
)
from .registry import import_callable
from .units import Units, UnitError, normalize_unit

__all__ = [
    "OpenOddError",
    "load_odd_binding",
    "load_odd_file",
    "load_openodd",
]

logger = logging.getLogger(__name__)


class OpenOddError(ValueError):
    """An OpenODD document this reader cannot make an ODD of."""


#: Built-in probe name -> (probe, the unit it returns).
_PROBES: dict[str, tuple[Callable[[Any], Any], str]] = {
    "ego_speed_kph": (probes.ego_speed_kph, "km/h"),
    "speed_limit_kph": (probes.speed_limit_kph, "km/h"),
    "lanelet_speed_limit_kph": (probes.lanelet_speed_limit_kph, "km/h"),
    **{
        name: (getattr(probes, name), "")
        for name in (
            "in_junction",
            "lane_count",
            "lanelet_location",
            "lanelet_subtype",
            "illumination",
            "rain",
            "fog",
            "traffic_density",
            "pedestrian_nearby",
        )
    },
}


def _probe(spec: str) -> tuple[Callable[[Any], Any], str]:
    """A probe and its unit, by built-in name or ``package.module:function``."""
    if spec in _PROBES:
        return _PROBES[spec]
    if ":" not in spec:
        raise OpenOddError(
            f"unknown probe {spec!r}; built-in: {', '.join(sorted(_PROBES))}, "
            "or package.module:function"
        )
    try:
        return import_callable(spec), ""
    except ValueError as exc:
        raise OpenOddError(f"probe: {exc}") from exc


def _missing(world: Any) -> None:
    """The probe of a concept nothing measures: always missing."""
    return None


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_KNOWN_KEYS = {"IMPORT", "TAXONOMY", "MODULES", "ODD", "COD", "OD", "conversion"}


@dataclass
class _Documents:
    taxonomy: dict[str, Any] = field(default_factory=dict)
    modules: dict[str, Any] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)
    conversion: dict[str, Any] = field(default_factory=dict)
    first_stem: Optional[str] = None


def _merge(into: dict[str, Any], doc: Mapping[str, Any], where: str) -> None:
    for key, value in doc.items():
        if isinstance(value, Mapping) and isinstance(into.get(key), dict):
            _merge(into[key], value, f"{where}.{key}")
        elif key in into and into[key] != value:
            raise OpenOddError(f"{where}.{key} is defined twice, differently")
        else:
            into[key] = dict(value) if isinstance(value, Mapping) else value


def _read(
    source: Union[str, Path],
    docs: _Documents,
    stack: tuple[Path, ...],
    loaded: Optional[list[Any]] = None,
) -> None:
    """Read *source* (a path or YAML text) into *docs*; *loaded* if parsed already."""
    if isinstance(source, Path) or (
        "\n" not in source and source.strip().endswith((".yaml", ".yml"))
    ):
        path = Path(source).resolve()
        if path in stack:
            chain = " -> ".join(p.name for p in (*stack, path))
            raise OpenOddError(f"IMPORT cycle: {chain}")
        base, stack = path.parent, (*stack, path)
        docs.first_stem = docs.first_stem or path.stem
    else:
        path, base = None, Path.cwd()
    if loaded is None:
        loaded = _parse(path if path is not None else str(source))
    for doc in loaded:
        if not isinstance(doc, Mapping):
            raise OpenOddError("an OpenODD document must be a mapping")
        unknown = sorted(str(k) for k in doc if k not in _KNOWN_KEYS)
        if unknown:
            hint = (
                "; probes and buckets belong in a binding file (load_odd_binding)"
                if "COVERAGE" in unknown
                else ""
            )
            raise OpenOddError(f"not OpenODD YAML keys: {unknown}{hint}")
        for imported in _as_list(doc.get("IMPORT")):
            _read(base / str(imported), docs, stack)
        _merge(docs.taxonomy, doc.get("TAXONOMY") or {}, "TAXONOMY")
        _merge(docs.conversion, doc.get("conversion") or {}, "conversion")
        for section in ("MODULES", "ODD"):
            for name, mdef in (doc.get(section) or {}).items():
                name = str(name)
                if name in docs.modules:
                    raise OpenOddError(f"module {name} is defined twice")
                docs.modules[name] = mdef
                if section == "ODD":
                    docs.roots.append(name)
        if doc.get("COD") or doc.get("OD"):
            logger.info("OpenODD: COD/OD records are not read")


def _parse(source: Union[Path, str]) -> list[Any]:
    """The YAML documents in a file (a path) or in text."""
    try:
        text = (
            source.read_text(encoding="utf-8") if isinstance(source, Path) else source
        )
        return [d for d in yaml.safe_load_all(text) if d is not None]
    except OSError as exc:
        raise OpenOddError(f"cannot read {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise OpenOddError(f"not YAML: {exc}") from exc


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

_PRIMITIVES = {"integer", "long", "float", "double"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UNKNOWN = {"unknown", "none", "null", "undefined"}


@dataclass
class _Concept:
    path: tuple[str, ...]
    kind: str  # "number", "boolean", "categorical" ("reference" until resolved)
    unit_type: str = ""
    literals: list[str] = field(default_factory=list)
    #: literal -> the expressions defining it (a derived categorical)
    definitions: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    attribute: Optional[OddAttribute] = None

    @property
    def name(self) -> str:
        return ".".join(self.path)

    @property
    def ordered(self) -> bool:
        """Literals defined by ranges are ordered (in the order written)."""
        return bool(self.definitions)


def _literal(key: Any) -> str:
    if isinstance(key, bool):
        return "true" if key else "false"
    return str(key)


def _is_type_spec(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    return value.strip().split()[0] in _PRIMITIVES | {"boolean", "shapefile"}


def _is_expression(value: Any) -> bool:
    """A value a literal's definition may map a concept to."""
    return not isinstance(value, Mapping) and not _is_type_spec(value)


def _keys(node: Any, out: Counter[str]) -> Counter[str]:
    if isinstance(node, Mapping):
        for key, child in node.items():
            out[str(key)] += 1
            _keys(child, out)
    return out


def _defines_literals(node: Mapping[str, Any], ids: Counter[str]) -> bool:
    """Whether *node* maps literals to expressions on concepts defined elsewhere.

    A record whose fields are all categoricals looks the same, except that
    its keys name its own fields: a literal's expressions name concepts
    outside it.
    """
    own = _keys(node, Counter())
    return bool(node) and all(
        isinstance(v, Mapping)
        and v
        and all(ids[str(k)] > own[str(k)] for k in v)
        and all(_is_expression(x) for x in v.values())
        for v in node.values()
    )


def _flatten(
    node: Any, prefix: tuple[str, ...], ids: Counter[str], out: dict[str, _Concept]
) -> None:
    path = ".".join(prefix)
    if isinstance(node, list):
        out[path] = _Concept(
            prefix, "categorical", literals=[_literal(v) for v in node]
        )
        return
    if isinstance(node, Mapping):
        if _defines_literals(node, ids):
            concept = _Concept(prefix, "categorical")
            concept.literals = [_literal(k) for k in node]
            concept.definitions = {_literal(k): v for k, v in node.items()}
            out[path] = concept
            return
        for key, child in node.items():
            _flatten(child, (*prefix, str(key)), ids, out)
        return
    if not isinstance(node, str):
        raise OpenOddError(f"TAXONOMY {path}: cannot read {node!r}")
    tokens = node.strip().split()
    head = tokens[0] if tokens else ""
    if head in _PRIMITIVES:
        out[path] = _Concept(prefix, "number", unit_type=" ".join(tokens[1:]))
    elif head == "boolean" and len(tokens) == 1:
        out[path] = _Concept(prefix, "boolean")
    elif head == "shapefile":
        logger.info("OpenODD: %s is a shapefile; not read", path)
    elif len(tokens) == 1 and _IDENTIFIER.match(head):
        out[path] = _Concept(prefix, "reference", unit_type=head)
    else:
        raise OpenOddError(
            f"TAXONOMY {path}: cannot read the type {node!r} (OpenODD types: "
            "boolean, integer, long, float, double, a list, or a type's id)"
        )


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(
    r"^\[\s*(?P<low>[^\],]+?)\s*(?:\.\.|,)\s*(?P<high>[^\]]+?)\s*\]\s*(?P<unit>\S+)?$"
)
_BOUND_RE = re.compile(r"^(?P<op><=|>=|<|>)\s*(?P<rest>.+)$")
_QUANTITY_RE = re.compile(rf"^(?P<value>{_NUMBER})\s*(?P<unit>\S+)?$")


class _Reader:
    def __init__(
        self,
        docs: _Documents,
        bindings: Mapping[str, Any],
        name: str,
        text: str,
    ) -> None:
        self.docs = docs
        self.name = name
        self.text = text
        self.units = Units()
        self.units.add_conversions(docs.conversion)
        ids = _keys(docs.taxonomy, Counter())
        self.concepts: dict[str, _Concept] = {}
        _flatten(docs.taxonomy, (), ids, self.concepts)
        self._resolve_references()
        self.module_names = set(docs.modules)
        self.labels: set[str] = set()
        for mdef in docs.modules.values():
            if isinstance(mdef, Mapping):
                self.labels.update(_labels_of(mdef))
        self.bindings = {
            self.concept(str(k), "binding").name: dict(v or {})
            for k, v in bindings.items()
        }

    # -- names -----------------------------------------------------------

    def concept(self, ref: str, where: str) -> _Concept:
        """The concept *ref* names: its id, or a dotted tail of its path."""
        found = [
            c
            for c in self.concepts.values()
            if c.name == ref or c.name.endswith("." + ref)
        ]
        if not found:
            raise OpenOddError(
                f"{where}: {ref} is neither a concept nor a module or label"
            )
        if len(found) > 1:
            names = ", ".join(sorted(c.name for c in found))
            raise OpenOddError(
                f"{where}: {ref} is ambiguous ({names}); give more of its path"
            )
        return found[0]

    def _resolve_references(self) -> None:
        """A concept typed by a categorical's id takes its literals."""
        for concept in list(self.concepts.values()):
            if concept.kind != "reference":
                continue
            target = next(
                (
                    c
                    for c in self.concepts.values()
                    if c.path[-1] == concept.unit_type and c.kind == "categorical"
                ),
                None,
            )
            if target is None:
                logger.info(
                    "OpenODD: %s is of type %s, which is not followed",
                    concept.name,
                    concept.unit_type,
                )
                del self.concepts[concept.name]
                continue
            concept.kind = "categorical"
            concept.literals = list(target.literals)
            concept.unit_type = ""

    # -- units -----------------------------------------------------------

    def unit_of(self, concept: _Concept) -> str:
        binding = self.bindings.get(concept.name, {})
        if "unit" in binding:
            return normalize_unit(str(binding["unit"]))
        if "probe" in binding:
            return normalize_unit(_probe(str(binding["probe"]))[1])
        return ""

    def number(
        self, token: str, unit: Optional[str], concept: _Concept, where: str
    ) -> float:
        try:
            value = float(token)
        except ValueError as exc:
            raise OpenOddError(
                f"{where}: {concept.name}: {token!r} is not a number "
                "(numeric terms and $parameters are not read)"
            ) from exc
        if (
            unit
            and self.units.unit_type(unit) is None
            and unit != self.unit_of(concept)
        ):
            raise OpenOddError(
                f"{where}: {concept.name}: unknown unit {unit!r} "
                "(numeric terms and $parameters are not read)"
            )
        try:
            if unit and concept.unit_type:
                self.units.check_type(unit, concept.unit_type)
            return self.units.convert(value, unit or "", self.unit_of(concept))
        except UnitError as exc:
            raise OpenOddError(f"{where}: {concept.name}: {exc}") from exc

    # -- attributes --------------------------------------------------------

    def build_attributes(self) -> list[OddAttribute]:
        # Plain concepts first: a derived categorical's probe reads them.
        ordered = sorted(self.concepts.values(), key=lambda c: bool(c.definitions))
        for concept in ordered:
            concept.attribute = self.attribute_of(concept)
        return [c.attribute for c in ordered if c.attribute is not None]

    def attribute_of(self, concept: _Concept) -> OddAttribute:
        binding = self.bindings.get(concept.name, {})
        probe: Callable[[Any], Any]
        if "probe" in binding:
            probe = _probe(str(binding["probe"]))[0]
        elif concept.definitions:
            probe = self.derived_probe(concept) or _missing
        else:
            probe = _missing
        buckets: dict[str, Any] = {}
        if "values" in binding:
            buckets["values"] = list(binding["values"])
        elif "buckets" in binding:
            buckets["buckets"] = [float(x) for x in binding["buckets"]]
        elif "range" in binding:
            low, high = binding["range"]
            buckets["range"] = (float(low), float(high))
            buckets["every"] = float(binding.get("every", 0)) or None
        elif probe is not _missing:
            if concept.kind == "categorical":
                buckets["values"] = list(concept.literals)
            elif concept.kind == "boolean":
                buckets["values"] = [False, True]
        try:
            return OddAttribute(
                concept.name,
                probe,
                unit=self.unit_of(concept),
                text=str(binding.get("text", "")),
                **buckets,
            )
        except ValueError as exc:
            raise OpenOddError(f"binding {concept.name}: {exc}") from exc

    def derived_probe(self, concept: _Concept) -> Optional["_DerivedProbe"]:
        """A categorical defined by expressions; ``None`` when nothing it reads is measured."""
        rules: list[tuple[str, OddCondition]] = []
        for literal, definition in concept.definitions.items():
            where = f"TAXONOMY {concept.name}.{literal}"
            rules.append(
                (literal, all_of(self.conditions(definition, where, nested=True)))
            )
        sources = {a.name: a for _, c in rules for a in c._attributes()}
        if all(a.probe is _missing for a in sources.values()):
            return None
        return _DerivedProbe(rules, list(sources.values()))

    def threshold_buckets(self, modules: list[OddModule]) -> None:
        """Give a measured number without buckets buckets at the thresholds tested.

        Every bound and range the modules and the derived categoricals test a
        number against becomes a bucket edge, so each boundary the ODD draws
        is covered from both sides.
        """
        conditions: list[OddCondition] = [c for m in modules for c in m._conditions()]
        for concept in self.concepts.values():
            if isinstance(concept.attribute, OddAttribute) and isinstance(
                concept.attribute.probe, _DerivedProbe
            ):
                conditions += [c for _, c in concept.attribute.probe.rules]
        edges: dict[str, set[float]] = {}
        while conditions:
            condition = conditions.pop()
            if isinstance(condition, _Group):
                conditions += condition.children
            elif isinstance(condition, _Leaf):
                predicate = condition.predicate
                found = edges.setdefault(condition.attribute.name, set())
                if isinstance(predicate, _Interval):
                    found |= {
                        x for x in (predicate.low, predicate.high) if math.isfinite(x)
                    }
                elif (
                    isinstance(predicate, _Equal)
                    and isinstance(predicate.value, (int, float))
                    and not isinstance(predicate.value, bool)
                ):
                    found.add(float(predicate.value))
        for concept in self.concepts.values():
            attribute = concept.attribute
            if (
                attribute is None
                or attribute.item is not None
                or attribute.probe is _missing
                or concept.kind != "number"
                or not edges.get(attribute.name)
            ):
                continue
            attribute._set_buckets(
                buckets=[-math.inf, *sorted(edges[attribute.name]), math.inf]
            )

    # -- conditions --------------------------------------------------------

    def conditions(
        self, section: Any, where: str, *, op: str = "", nested: bool = False
    ) -> list[OddCondition]:
        if not isinstance(section, Mapping) or not section:
            raise OpenOddError(f"{where}: a section maps concepts to expressions")
        out: list[OddCondition] = []
        for key, value in section.items():
            key = str(key)
            if key in ("AND", "OR"):
                if nested:
                    raise OpenOddError(f"{where}: sections nest one level only")
                if key == op:
                    other = "OR" if op == "AND" else "AND"
                    raise OpenOddError(
                        f"{where}: an {op} section nests {other}, not {key}"
                    )
                children = self.conditions(value, where, op=key, nested=True)
                out.append(all_of(children) if key == "AND" else any_of(children))
            elif key in self.module_names or key in self.labels:
                if not isinstance(value, bool):
                    raise OpenOddError(
                        f"{where}: {key} is a module or label: true or false"
                    )
                out.append(module_holds(key, value))
            else:
                out.append(self.leaf(key, value, where))
        return out

    def leaf(self, key: str, value: Any, where: str) -> OddCondition:
        concept = self.concept(key, where)
        attribute = concept.attribute
        if attribute is None:
            raise OpenOddError(f"{where}: {concept.name} is defined by itself")
        if value is None or (
            isinstance(value, str) and value.strip().lower() in _UNKNOWN
        ):
            return attribute.is_unknown()
        if isinstance(value, list):
            literals = [_literal(v) for v in value]
            self.require_literals(concept, literals, where)
            return attribute.is_in(literals)
        if isinstance(value, bool):
            if concept.kind == "categorical":
                self.require_literals(concept, [_literal(value)], where)
                return attribute.is_in([_literal(value)])
            return attribute.equals(value)
        if isinstance(value, (int, float)):
            return attribute.equals(self.number(str(value), None, concept, where))
        text = str(value).strip()
        if concept.kind == "number":
            return self.numeric(concept, attribute, text, where)
        return self.categorical(concept, attribute, text, where)

    def numeric(
        self, concept: _Concept, attribute: OddAttribute, text: str, where: str
    ) -> OddCondition:
        m = _RANGE_RE.match(text)
        if m:
            low = self.number(m["low"], m["unit"], concept, where)
            high = self.number(m["high"], m["unit"], concept, where)
            return attribute.between(low, high)
        m = _BOUND_RE.match(text)
        if m:
            q = _QUANTITY_RE.match(m["rest"].strip())
            if q is None:
                raise OpenOddError(f"{where}: {concept.name}: cannot read {text!r}")
            x = self.number(q["value"], q["unit"], concept, where)
            bound = {
                ">=": attribute.at_least,
                ">": attribute.greater_than,
                "<=": attribute.at_most,
                "<": attribute.less_than,
            }[m["op"]]
            return bound(x)
        q = _QUANTITY_RE.match(text)
        if q:
            return attribute.equals(self.number(q["value"], q["unit"], concept, where))
        raise OpenOddError(f"{where}: {concept.name}: cannot read {text!r}")

    def categorical(
        self, concept: _Concept, attribute: OddAttribute, text: str, where: str
    ) -> OddCondition:
        m = _RANGE_RE.match(text)
        b = _BOUND_RE.match(text)
        if m or b:
            if not concept.ordered:
                raise OpenOddError(
                    f"{where}: {concept.name}: {text!r} needs literals defined by ranges"
                )
            order = concept.literals
            if m:
                low, high = m["low"].strip(), m["high"].strip()
                self.require_literals(concept, [low, high], where)
                i, j = sorted((order.index(low), order.index(high)))
                return attribute.is_in(order[i : j + 1])
            assert b is not None
            ref = b["rest"].strip()
            self.require_literals(concept, [ref], where)
            i = order.index(ref)
            chosen = {
                "<": order[:i],
                "<=": order[: i + 1],
                ">": order[i + 1 :],
                ">=": order[i:],
            }[b["op"]]
            if not chosen:
                raise OpenOddError(
                    f"{where}: {concept.name} {text!r} holds for no literal"
                )
            return attribute.is_in(chosen)
        if concept.kind == "categorical":
            self.require_literals(concept, [text], where)
        return attribute.equals(text)

    @staticmethod
    def require_literals(
        concept: _Concept, literals: Sequence[str], where: str
    ) -> None:
        if concept.kind != "categorical":
            return
        unknown = [x for x in literals if x not in concept.literals]
        if unknown:
            raise OpenOddError(
                f"{where}: {concept.name} has no literal {', '.join(unknown)} "
                f"(literals: {', '.join(concept.literals)})"
            )

    # -- modules ---------------------------------------------------------

    _MODULE_KEYS = {
        "TITLE",
        "DESCRIPTION",
        "ACTIVE",
        "LABEL",
        "LABELS",
        "METADATA",
        "INCLUDE_AND",
        "INCLUDE_OR",
        "EXCLUDE_AND",
        "EXCLUDE_OR",
    }

    def build_modules(self) -> list[OddModule]:
        modules = []
        for name, mdef in self.docs.modules.items():
            where = f"module {name}"
            if not isinstance(mdef, Mapping):
                raise OpenOddError(f"{where}: expected a mapping")
            if "unknown" in name.lower():
                raise OpenOddError(f"{where}: a module's id must not contain 'unknown'")
            extra = sorted(str(k) for k in mdef if k not in self._MODULE_KEYS)
            if extra:
                raise OpenOddError(f"{where}: unknown keys {extra}")
            includes = [k for k in ("INCLUDE_AND", "INCLUDE_OR") if k in mdef]
            excludes = [k for k in ("EXCLUDE_AND", "EXCLUDE_OR") if k in mdef]
            if len(includes) > 1 or len(excludes) > 1:
                raise OpenOddError(
                    f"{where}: at most one INCLUDE_* and one EXCLUDE_* section"
                )
            if not includes and not excludes:
                raise OpenOddError(f"{where}: needs an INCLUDE_* or EXCLUDE_* section")
            sections = {
                key.lower(): self.conditions(
                    mdef[key], f"{where} {key}", op=key.rsplit("_", 1)[1]
                )
                for key in (*includes, *excludes)
            }
            labels = _labels_of(mdef)
            if any("unknown" in label.lower() for label in labels):
                raise OpenOddError(f"{where}: a label must not contain 'unknown'")
            text = " -- ".join(
                str(mdef[k]) for k in ("TITLE", "DESCRIPTION") if mdef.get(k)
            )
            modules.append(
                OddModule(
                    name,
                    **sections,
                    labels=labels,
                    active=_flag(mdef.get("ACTIVE", True), f"{where} ACTIVE"),
                    text=text,
                )
            )
        return modules

    def build(self) -> OddDefinition:
        attributes = self.build_attributes()
        modules = self.build_modules()
        self.threshold_buckets(modules)
        missing = [
            c.name
            for c in self.concepts.values()
            if c.attribute is not None and c.attribute.probe is _missing
        ]
        if missing:
            logger.info(
                "OpenODD %s: no probe for %s (always missing)", self.name, missing
            )
        clash = sorted(self.labels & {c.path[-1] for c in self.concepts.values()})
        if clash:
            raise OpenOddError(f"labels named like concepts: {clash}")
        try:
            # Modules under ODD are the root candidates; the standard's own
            # examples put modules they refer to there too, which the
            # candidates rule leaves out.
            return OddDefinition(
                self.name,
                attributes,
                modules,
                roots=self.docs.roots or None,
                text=self.text,
            )
        except ValueError as exc:
            raise OpenOddError(str(exc)) from exc


class _DerivedProbe:
    """The value of a categorical defined by expressions: the literal that holds.

    It is worked out from the tick's other values
    (:meth:`OddDefinition.sample`).  Where two literals' ranges share an
    endpoint, the one written first takes it.
    """

    def __init__(
        self, rules: list[tuple[str, OddCondition]], sources: list[OddAttribute]
    ) -> None:
        self.rules = rules
        self.sources = sources

    def from_values(self, values: Mapping[str, Any]) -> Optional[str]:
        for literal, condition in self.rules:
            verdict = condition.evaluate(values, {})
            if verdict is None:
                return None
            if verdict:
                return literal
        return None

    def __call__(self, world: Any) -> Optional[str]:
        return self.from_values(
            {a.name: read_probe(a.probe, world) for a in self.sources}
        )


def _labels_of(mdef: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ("LABEL", "LABELS"):
        labels += [str(x) for x in _as_list(mdef.get(key))]
    return labels


def _flag(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "yes"):
        return True
    if text in ("false", "no"):
        return False
    raise OpenOddError(f"{where}: expected true or false, not {value!r}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def load_openodd(
    *sources: Union[str, Path],
    bindings: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    text: str = "",
) -> OddDefinition:
    """Read an ODD from OpenODD YAML files or text.

    Args:
        sources: Paths, or YAML text.  Several are read together, as if one
            imported the others.  A file's ``IMPORT`` entries are read
            relative to it.
        bindings: Concept -> how to measure it:
            ``{"probe": ..., "unit": ..., "values" | "buckets" | "range" +
            "every": ..., "text": ...}`` (see :func:`load_odd_binding`).  A
            concept with no probe is always missing.
        name: The ODD's name; the first file's stem by default.
        text: A description for the report.

    Raises:
        OpenOddError: when the documents do not make an ODD.
    """
    if not sources:
        raise OpenOddError("load_openodd(): no sources")
    docs = _Documents()
    for source in sources:
        _read(source, docs, ())
    reader = _Reader(docs, bindings or {}, name or docs.first_stem or "openodd", text)
    return reader.build()


def load_odd_binding(path: Union[str, Path]) -> OddDefinition:
    """Read an ODD from a binding file: OpenODD files, and how to measure them.

    A binding file is this framework's, not OpenODD's::

        openodd: [taxonomy.yaml, odd.yaml]   # relative to this file
        name: urban                          # default: this file's stem
        text: Urban roads, fair weather
        probes:
          speed_limit:                       # a concept, as conditions name it
            probe: speed_limit_kph           # built-in, or package.module:function
            unit: km/h                       # what the probe returns
            buckets: [0, 30, 60, 90]         # or values, or range + every
          rainfall_level: {probe: rain}

    A numeric concept with a probe but no buckets gets buckets at the
    thresholds the ODD tests it against; a categorical gets one per literal.
    """
    path = Path(path)
    docs = _parse(path)
    doc = docs[0] if len(docs) == 1 else None
    if not isinstance(doc, Mapping) or "openodd" not in doc:
        raise OpenOddError(
            f"{path}: a binding file names its OpenODD files under 'openodd'"
        )
    return _load_binding(path, doc)


def _load_binding(path: Path, doc: Mapping[str, Any]) -> OddDefinition:
    extra = sorted(
        str(k) for k in doc if k not in ("openodd", "name", "text", "probes")
    )
    if extra:
        raise OpenOddError(f"{path}: unknown keys {extra}")
    sources = [path.parent / str(p) for p in _as_list(doc["openodd"])]
    return load_openodd(
        *sources,
        bindings=doc.get("probes") or {},
        name=str(doc.get("name") or path.stem),
        text=str(doc.get("text", "")),
    )


def load_odd_file(path: Union[str, Path]) -> OddDefinition:
    """Read a ``.yaml`` ODD file: a binding file, or an OpenODD document on its own.

    An OpenODD document on its own is read with no probes, so nothing is
    measured; a binding file names the probes (:func:`load_odd_binding`).
    """
    path = Path(path)
    docs = _parse(path)
    if len(docs) == 1 and isinstance(docs[0], Mapping) and "openodd" in docs[0]:
        return _load_binding(path, docs[0])
    od = _Documents()
    _read(path, od, (), loaded=docs)
    return _Reader(od, {}, od.first_stem or path.stem, "").build()
