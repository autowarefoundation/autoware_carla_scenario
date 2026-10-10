"""Read an ODD written in ASAM OpenODD YAML.

The reader builds the same :class:`OddDefinition` a Python ODD builds, so a
YAML ODD is measured, reported and judged exactly like one written in code.

What is read
============

``TAXONOMY``: the attribute tree.  A leaf is one of:

* ``float <unit type>`` / ``int <unit type>``: a number;
* ``boolean``;
* ``string``;
* a list: an enumeration of literals;
* a mapping of literals, each to the conditions under which it holds (a
  derived enumeration, e.g. a fog severity defined by visibility ranges).  Its
  value is the first literal whose conditions all hold.

A string that names another record (``"vehicle/pose"``) is a reference, which
this reader does not follow.

``MODULES``: named rules.  Each has ``INCLUDE_AND`` / ``INCLUDE_OR`` /
``EXCLUDE_AND`` / ``EXCLUDE_OR`` sections, ``TITLE``, ``DESCRIPTION``,
``LABEL`` / ``LABELS`` and ``ACTIVE``.  A section maps a taxonomy path to an
expression, or a module or label name to ``true`` / ``false``, or ``AND`` /
``OR`` to a nested section.  The expressions are:

* ``[low .. high] unit``: inclusive range;
* ``>= x unit``, ``> x``, ``<= x``, ``< x``, ``== x``;
* a list of literals: one of them;
* a bare literal, number or boolean: equal to it.

Two sections extend OpenODD with what measuring needs:

``COVERAGE`` binds a taxonomy path to a probe and gives its buckets::

    COVERAGE:
      road.speed_limit:
        probe: speed_limit_kph   # a built-in probe, or package.module:function
        unit: km/h               # the probe's unit (built-in probes know theirs)
        buckets: [0, 30, 60, 90] # or range: [0, 120] + every: 10, or values: [...]

A numeric attribute with a probe but no buckets gets buckets at the
thresholds the modules test it against.  Then every side of every boundary
the ODD draws is a coverage target.  An attribute with no probe is monitored
as unknown.  It never decides a tick, and the report lists it as
unmeasured.

``ODD`` names the ODD and its root module::

    ODD:
      name: urban_l4
      root: root_odd
      text: Urban driving up to 60 km/h

Without a ``root``, the ODD holds when every active module does.

OpenODD 1.0's normative YAML mapping was not available while this was
written.  The format follows the public examples of the standard's YAML
mapping.  Comparisons are strict: ``>`` excludes its bound and ``>=``
includes it.
"""

from __future__ import annotations

import importlib
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import yaml

from . import probes
from .model import (
    OddAttribute,
    OddCondition,
    OddDefinition,
    OddModule,
    all_of,
    any_of,
    module_holds,
)
from .units import UnitError, convert, normalize_unit

__all__ = [
    "BUILTIN_PROBES",
    "OpenOddError",
    "load_openodd",
    "register_odd_probe",
]

logger = logging.getLogger(__name__)


class OpenOddError(ValueError):
    """An OpenODD document this reader cannot make an ODD of."""


#: Probe name -> (probe, the unit it returns).
BUILTIN_PROBES: dict[str, tuple[Callable[[Any], Any], str]] = {
    "ego_speed_kph": (probes.ego_speed_kph, "km/h"),
    "speed_limit_kph": (probes.speed_limit_kph, "km/h"),
    "lanelet_speed_limit_kph": (probes.lanelet_speed_limit_kph, "km/h"),
    "in_junction": (probes.in_junction, ""),
    "lane_count": (probes.lane_count, ""),
    "lanelet_location": (probes.lanelet_location, ""),
    "lanelet_subtype": (probes.lanelet_subtype, ""),
    "illumination": (probes.illumination, ""),
    "rain": (probes.rain, ""),
    "fog": (probes.fog, ""),
    "traffic_density": (probes.traffic_density, ""),
    "pedestrian_nearby": (probes.pedestrian_nearby, ""),
}

_PROBES: dict[str, tuple[Callable[[Any], Any], str]] = dict(BUILTIN_PROBES)


def register_odd_probe(name: str, probe: Callable[[Any], Any], unit: str = "") -> None:
    """Make *probe* (returning values in *unit*) bindable by *name* in ``COVERAGE``."""
    if not name:
        raise ValueError("register_odd_probe(): name must not be empty")
    _PROBES[name] = (probe, unit)


def _probe(spec: str) -> tuple[Callable[[Any], Any], str]:
    if spec in _PROBES:
        return _PROBES[spec]
    if ":" in spec:
        module, _, attr = spec.partition(":")
        try:
            return getattr(importlib.import_module(module), attr), ""
        except (ImportError, AttributeError) as exc:
            raise OpenOddError(f"probe {spec!r}: {exc}") from exc
    raise OpenOddError(
        f"unknown probe {spec!r}; built-in: {', '.join(sorted(_PROBES))}, "
        "or package.module:function"
    )


def _unknown(world: Any) -> None:
    """The probe of an attribute nothing measures: always unknown."""
    return None


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RANGE_RE = re.compile(
    rf"^\[\s*(?P<low>{_NUMBER})\s*\.\.\s*(?P<high>{_NUMBER})\s*\]\s*(?P<unit>\S+)?$"
)
_COMPARE_RE = re.compile(
    rf"^(?P<op><=|>=|==|=|<|>)\s*(?P<value>{_NUMBER})\s*(?P<unit>\S+)?$"
)


class _Leaf:
    """A taxonomy leaf being built: what it is, and what tests it."""

    def __init__(self, path: str, kind: str, unit_type: str = "") -> None:
        self.path = path
        self.kind = kind  # "number", "boolean", "string", "enum", "derived"
        self.unit_type = unit_type
        self.literals: list[str] = []
        self.derived: dict[str, Mapping[str, Any]] = {}
        self.thresholds: set[float] = set()
        self.attribute: Optional[OddAttribute] = None


def _flatten(node: Any, prefix: tuple[str, ...], out: dict[str, _Leaf]) -> None:
    if isinstance(node, list):
        leaf = _Leaf(".".join(prefix), "enum")
        leaf.literals = [str(v) for v in node]
        out[leaf.path] = leaf
        return
    if isinstance(node, Mapping):
        if node and all(
            isinstance(v, Mapping) and v and all("." in str(k) for k in v)
            for v in node.values()
        ):
            leaf = _Leaf(".".join(prefix), "derived")
            leaf.literals = [str(k) for k in node]
            leaf.derived = {str(k): v for k, v in node.items()}
            out[leaf.path] = leaf
            return
        for key, child in node.items():
            _flatten(child, (*prefix, str(key)), out)
        return
    if node is None:
        return
    spec = str(node).strip().split()
    head = spec[0].lower() if spec else ""
    path = ".".join(prefix)
    if head in ("float", "double", "int", "integer"):
        out[path] = _Leaf(path, "number", spec[1] if len(spec) > 1 else "")
    elif head in ("bool", "boolean"):
        out[path] = _Leaf(path, "boolean")
    elif head in ("string", "str"):
        out[path] = _Leaf(path, "string")
    elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z0-9_]+)+$", str(node).strip()):
        logger.info("OpenODD: %s refers to %s; references are not followed", path, node)
    else:
        raise OpenOddError(f"TAXONOMY {path}: cannot read the type {node!r}")


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


class _Reader:
    def __init__(self, doc: Mapping[str, Any], name: str) -> None:
        self.doc = doc
        self.leaves: dict[str, _Leaf] = {}
        _flatten(doc.get("TAXONOMY") or {}, (), self.leaves)
        self.coverage: Mapping[str, Any] = doc.get("COVERAGE") or {}
        self.modules_doc: Mapping[str, Any] = doc.get("MODULES") or {}
        meta = doc.get("ODD") or {}
        self.name = str(meta.get("name") or name)
        self.root = meta.get("root")
        self.text = str(meta.get("text", ""))
        self.module_names = {str(k) for k in self.modules_doc}
        self.labels: set[str] = set()
        for mdef in self.modules_doc.values():
            if isinstance(mdef, Mapping):
                self.labels.update(_labels_of(mdef))
        unknown = sorted(set(self.coverage) - set(self.leaves))
        if unknown:
            raise OpenOddError(f"COVERAGE names paths not in TAXONOMY: {unknown}")

    # -- pass 1: thresholds, so numeric leaves without buckets get some ----

    def collect_thresholds(self) -> None:
        def walk(section: Any) -> None:
            if not isinstance(section, Mapping):
                return
            for key, value in section.items():
                key = str(key)
                if key in ("AND", "OR"):
                    walk(value)
                    continue
                leaf = self.leaves.get(key)
                if leaf is None or leaf.kind != "number" or isinstance(value, bool):
                    continue
                for number, unit in _numbers_in(value):
                    try:
                        leaf.thresholds.add(convert(number, unit, self.unit_of(leaf)))
                    except UnitError:
                        pass  # reported when the condition is built

        for mdef in self.modules_doc.values():
            if isinstance(mdef, Mapping):
                for key in ("INCLUDE_AND", "INCLUDE_OR", "EXCLUDE_AND", "EXCLUDE_OR"):
                    walk(mdef.get(key))
        for leaf in self.leaves.values():
            for conditions in leaf.derived.values():
                walk(conditions)

    def unit_of(self, leaf: _Leaf) -> str:
        binding = self.coverage.get(leaf.path) or {}
        if "unit" in binding:
            return normalize_unit(str(binding["unit"]))
        if "probe" in binding:
            return normalize_unit(_probe(str(binding["probe"]))[1])
        return ""

    # -- pass 2: attributes ------------------------------------------------

    def build_attributes(self) -> list[OddAttribute]:
        # Plain leaves first: a derived leaf's probe reads them.
        ordered = [leaf for leaf in self.leaves.values() if leaf.kind != "derived"]
        ordered += [leaf for leaf in self.leaves.values() if leaf.kind == "derived"]
        for leaf in ordered:
            leaf.attribute = self.attribute_of(leaf)
        return [leaf.attribute for leaf in ordered if leaf.attribute is not None]

    def attribute_of(self, leaf: _Leaf) -> OddAttribute:
        binding = dict(self.coverage.get(leaf.path) or {})
        unit = self.unit_of(leaf)
        probe: Callable[[Any], Any]
        if "probe" in binding:
            probe = _probe(str(binding["probe"]))[0]
        elif leaf.kind == "derived":
            probe = self.derived_probe(leaf)
        else:
            probe = _unknown
        bucket_args: dict[str, Any] = {}
        if "values" in binding:
            bucket_args["values"] = list(binding["values"])
        elif "buckets" in binding:
            bucket_args["buckets"] = [float(x) for x in binding["buckets"]]
        elif "range" in binding:
            low, high = binding["range"]
            bucket_args["range"] = (float(low), float(high))
            bucket_args["every"] = float(binding.get("every", 0)) or None
        elif probe is not _unknown:
            if leaf.kind in ("enum", "derived"):
                bucket_args["values"] = list(leaf.literals)
            elif leaf.kind == "boolean":
                bucket_args["values"] = [False, True]
            elif leaf.kind == "number" and leaf.thresholds:
                bucket_args["buckets"] = [-math.inf, *sorted(leaf.thresholds), math.inf]
        try:
            return OddAttribute(
                leaf.path,
                probe,
                unit=unit,
                text=str(binding.get("text", "")),
                **bucket_args,
            )
        except ValueError as exc:
            raise OpenOddError(f"COVERAGE {leaf.path}: {exc}") from exc

    def derived_probe(self, leaf: _Leaf) -> Callable[[Any], Any]:
        """The probe of a derived enumeration: its first literal that holds."""
        rules: list[tuple[str, OddCondition]] = []
        for literal, conditions in leaf.derived.items():
            rules.append(
                (literal, all_of(self.conditions(conditions, where=leaf.path)))
            )
        sources = sorted(
            {a.name: a for _, c in rules for a in c._attributes()}.values(),
            key=lambda a: a.name,
        )

        def probe(world: Any) -> Optional[str]:
            values = {}
            for attribute in sources:
                try:
                    values[attribute.name] = attribute.probe(world)
                except Exception:
                    values[attribute.name] = None
            for literal, condition in rules:
                verdict = condition.evaluate(values, {})
                if verdict is None:
                    return None
                if verdict:
                    return literal
            return None

        return probe

    # -- pass 3: modules ---------------------------------------------------

    def attribute(self, path: str, where: str) -> OddAttribute:
        leaf = self.leaves.get(path)
        if leaf is None:
            raise OpenOddError(f"{where}: {path} is not in TAXONOMY")
        if leaf.attribute is None:
            raise OpenOddError(f"{where}: {path} refers to itself")
        return leaf.attribute

    def conditions(self, section: Any, *, where: str) -> list[OddCondition]:
        if not isinstance(section, Mapping):
            raise OpenOddError(f"{where}: a section must map paths to expressions")
        out: list[OddCondition] = []
        for key, value in section.items():
            key = str(key)
            if key in ("AND", "OR"):
                children = self.conditions(value, where=where)
                out.append(all_of(children) if key == "AND" else any_of(children))
            elif isinstance(value, bool) and (
                key in self.module_names or key in self.labels
            ):
                out.append(module_holds(key, value))
            else:
                out.append(self.leaf_condition(key, value, where=where))
        return out

    def leaf_condition(self, path: str, value: Any, *, where: str) -> OddCondition:
        leaf = self.leaves.get(path)
        if leaf is None:
            raise OpenOddError(
                f"{where}: {path} is neither in TAXONOMY nor a module or label"
            )
        attribute = self.attribute(path, where)
        if isinstance(value, list):
            return attribute.is_in([str(v) for v in value])
        if isinstance(value, bool) or not isinstance(value, str):
            return attribute.equals(value)
        text = value.strip()
        unit = self.unit_of(leaf)
        m = _RANGE_RE.match(text)
        if m:
            low = self.number(m["low"], m["unit"], unit, where, path)
            high = self.number(m["high"], m["unit"], unit, where, path)
            return attribute.between(low, high)
        m = _COMPARE_RE.match(text)
        if m:
            x = self.number(m["value"], m["unit"], unit, where, path)
            op = m["op"]
            if op == ">=":
                return attribute.at_least(x)
            if op == ">":
                return attribute.greater_than(x)
            if op == "<=":
                return attribute.at_most(x)
            if op == "<":
                return attribute.less_than(x)
            return attribute.equals(x)
        if leaf.kind == "number":
            raise OpenOddError(f"{where}: {path}: cannot read {value!r}")
        return attribute.equals(text)

    @staticmethod
    def number(
        token: str, unit: Optional[str], to: str, where: str, path: str
    ) -> float:
        try:
            return convert(float(token), unit or "", to)
        except UnitError as exc:
            raise OpenOddError(f"{where}: {path}: {exc}") from exc

    def build_modules(self) -> list[OddModule]:
        modules = []
        for name, mdef in self.modules_doc.items():
            name = str(name)
            if not isinstance(mdef, Mapping):
                raise OpenOddError(f"MODULES {name}: expected a mapping")
            where = f"MODULES {name}"
            sections = {
                key.lower(): self.conditions(mdef[key], where=f"{where} {key}")
                for key in ("INCLUDE_AND", "INCLUDE_OR", "EXCLUDE_AND", "EXCLUDE_OR")
                if key in mdef
            }
            text = " -- ".join(
                str(mdef[k]) for k in ("TITLE", "DESCRIPTION") if mdef.get(k)
            )
            modules.append(
                OddModule(
                    name,
                    **sections,
                    labels=_labels_of(mdef),
                    active=bool(mdef.get("ACTIVE", True)),
                    text=text,
                )
            )
        return modules

    def build(self) -> OddDefinition:
        self.collect_thresholds()
        attributes = self.build_attributes()
        modules = self.build_modules()
        unbound = [
            leaf.path
            for leaf in self.leaves.values()
            if leaf.attribute is not None and leaf.attribute.probe is _unknown
        ]
        if unbound:
            logger.info("OpenODD %s: no probe for %s (unknown)", self.name, unbound)
        try:
            return OddDefinition(
                self.name,
                attributes,
                modules,
                root=str(self.root) if self.root is not None else None,
                text=self.text,
            )
        except ValueError as exc:
            raise OpenOddError(str(exc)) from exc


def _labels_of(mdef: Mapping[str, Any]) -> list[str]:
    labels = [str(x) for x in mdef.get("LABELS") or ()]
    if mdef.get("LABEL"):
        labels.insert(0, str(mdef["LABEL"]))
    return labels


def _numbers_in(value: Any) -> list[tuple[float, str]]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [(float(value), "")]
    if not isinstance(value, str):
        return []
    text = value.strip()
    m = _RANGE_RE.match(text)
    if m:
        return [(float(m["low"]), m["unit"] or ""), (float(m["high"]), m["unit"] or "")]
    m = _COMPARE_RE.match(text)
    if m:
        return [(float(m["value"]), m["unit"] or "")]
    return []


def _merge(into: dict[str, Any], doc: Mapping[str, Any]) -> None:
    for key, value in doc.items():
        if isinstance(value, Mapping) and isinstance(into.get(key), dict):
            _merge(into[key], value)
        else:
            into[key] = dict(value) if isinstance(value, Mapping) else value


def load_openodd(
    *sources: Union[str, Path], name: Optional[str] = None
) -> OddDefinition:
    """Read an ODD from OpenODD YAML files or text.

    Args:
        sources: Paths, or YAML text.  Several are merged in order, so a
            taxonomy, the modules and the ``COVERAGE`` bindings may each live
            in a file of their own; a file may also hold several documents.
        name: The ODD's name when no ``ODD: name`` gives one; else the first
            file's stem.

    Raises:
        OpenOddError: when the documents do not make an ODD.
    """
    if not sources:
        raise OpenOddError("load_openodd(): no sources")
    merged: dict[str, Any] = {}
    default_name = name
    for source in sources:
        if isinstance(source, Path) or (
            isinstance(source, str)
            and "\n" not in source
            and source.endswith((".yaml", ".yml"))
        ):
            path = Path(source)
            text = path.read_text(encoding="utf-8")
            default_name = default_name or path.stem
        else:
            text = str(source)
        try:
            docs = [d for d in yaml.safe_load_all(text) if d is not None]
        except yaml.YAMLError as exc:
            raise OpenOddError(f"not YAML: {exc}") from exc
        for doc in docs:
            if not isinstance(doc, Mapping):
                raise OpenOddError("an OpenODD document must be a mapping")
            _merge(merged, doc)
    return _Reader(merged, default_name or "openodd").build()
