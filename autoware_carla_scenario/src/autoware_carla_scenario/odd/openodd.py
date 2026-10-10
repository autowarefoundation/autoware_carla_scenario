"""Read an ODD written in ASAM OpenODD 1.0 YAML (chapter 10 of the standard).

The reader builds the same :class:`OddDefinition` a Python ODD builds, so a
YAML ODD is measured, reported and judged exactly like one written in code.

An OpenODD document holds what an ODD *is*.  How this framework *measures*
it lives in a separate **binding file** (:func:`load_odd_binding`): which
probe reads each concept, and its buckets.  The standard does not allow
constructs of its own in an OpenODD document, so they are kept apart.

What is read
============

``IMPORT``: other OpenODD files, relative to the importing one, or else in
the directories of the other sources (a cycle is refused).  Sources can be
files in git repositories at a revision (:mod:`.sources`).

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

from .sources import GitSource, GitSourceError
from .sources import checkout as git_checkout
from .sources import parse_entry as parse_git_entry

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


def _missing_on_lanelet(lanelet: Any, lanelet_map: Any, routing_graph: Any) -> None:
    """On a planned route too (:mod:`.route`), the value is missing."""
    return None


_missing.on_lanelet = _missing_on_lanelet  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_KNOWN_KEYS = {"IMPORT", "TAXONOMY", "MODULES", "ODD", "COD", "OD", "conversion"}


class _Loader(yaml.SafeLoader):
    """PyYAML's safe loader with YAML 1.2 booleans, refusing duplicate keys.

    YAML 1.1 reads ``off``, ``no`` and ``NO`` as booleans, which turns
    literals such as ``wipers: [off, on]`` or a country code into ``false``.
    A key written twice would otherwise silently drop the first.
    """

    def construct_mapping(self, node: Any, deep: bool = False) -> Any:
        keys = [self.construct_object(k, deep=deep) for k, _ in node.value]
        seen: set[Any] = set()
        for key, (key_node, _) in zip(keys, node.value):
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"{key!r} is written twice", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


_Loader.yaml_implicit_resolvers = {
    first: [
        (tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"
    ]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


@dataclass
class _Documents:
    taxonomy: dict[str, Any] = field(default_factory=dict)
    modules: dict[str, Any] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)
    conversion: dict[str, Any] = field(default_factory=dict)
    first_stem: Optional[str] = None
    #: Files read already: one imported twice is read once.
    seen: set[Path] = field(default_factory=set)
    #: File name -> the file read under it (names are unique in a transmission).
    names: dict[str, Path] = field(default_factory=dict)
    #: Taxonomy concept -> the file defining it.
    origins: dict[str, str] = field(default_factory=dict)
    #: Where an ``IMPORT`` not next to its importer is looked for: the
    #: directories of the sources, and the roots of their git checkouts.
    search: list[Path] = field(default_factory=list)
    #: The git sources read, with the commits their revisions named.
    provenance: list[dict[str, str]] = field(default_factory=list)
    #: The roots of their checkouts: a file in one imports only from the
    #: sources, never from elsewhere on this machine.
    checkouts: list[Path] = field(default_factory=list)

    def locate(self, base: Path, name: str) -> Path:
        """The file an ``IMPORT`` of *name* in a file under *base* names.

        Next to the importer first.  Otherwise in the other sources: the
        standard makes file names unique within one transmission, so a
        module file may import a taxonomy kept in another repository.
        """
        here = base / name
        if not here.exists():
            found = {
                (root / name).resolve()
                for root in self.search
                if (root / name).is_file()
            }
            if len(found) > 1:
                raise OpenOddError(
                    f"IMPORT {name} is in several sources: "
                    + ", ".join(sorted(map(str, found)))
                )
            here = found.pop() if found else here
        if _under(base, self.checkouts) and not _under(here, self.search):
            raise OpenOddError(
                f"IMPORT {name} in {base}: a file from git imports only from "
                "the sources"
            )
        return here


def _under(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in roots)


def _merge(
    into: dict[str, Any],
    doc: Mapping[str, Any],
    where: str,
    origin: str,
    origins: Optional[dict[str, str]] = None,
) -> None:
    """Merge *doc*, read from *origin*, into *into*.

    Containers merge, so a file can add concepts to a taxonomy another file
    started.  With *origins* (concept -> the file defining it), a concept
    defined in two files is refused even when both say the same: OpenODD
    makes ids unique within a transmission.  Without, only a contradiction
    is.
    """
    for key, value in doc.items():
        name = f"{where}.{key}"
        if isinstance(value, Mapping) and isinstance(into.get(key), dict):
            _merge(into[key], value, name, origin, origins)
        elif key in into and (origins is not None or into[key] != value):
            first = f" (in {origins[name]} and {origin})" if origins else ""
            same = "" if origins is not None else ", differently"
            raise OpenOddError(f"{name} is defined twice{same}{first}")
        else:
            into[key] = dict(value) if isinstance(value, Mapping) else value
            if origins is not None:
                _record(origins, name, value, origin)


def _record(origins: dict[str, str], name: str, value: Any, origin: str) -> None:
    origins[name] = origin
    if isinstance(value, Mapping):
        for key, child in value.items():
            _record(origins, f"{name}.{key}", child, origin)


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
        if path in docs.seen:
            return
        other = docs.names.get(path.name)
        if other is not None:
            raise OpenOddError(
                f"two files named {path.name} ({other} and {path}): OpenODD "
                "makes file names unique within a transmission"
            )
        docs.seen.add(path)
        docs.names[path.name] = path
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
            _read(docs.locate(base, str(imported)), docs, stack)
        origin = path.name if path is not None else "<text>"
        _merge(
            docs.taxonomy, doc.get("TAXONOMY") or {}, "TAXONOMY", origin, docs.origins
        )
        _merge(docs.conversion, doc.get("conversion") or {}, "conversion", origin)
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
        return [d for d in yaml.load_all(text, Loader=_Loader) if d is not None]
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

#: What a binding file may say about how a concept is measured.
_BINDING_KEYS = {
    "probe",
    "unit",
    "values",
    "buckets",
    "range",
    "every",
    "text",
    "target",
    "cover_by",
    "min_stay",
}

_PRIMITIVES = {"integer", "long", "float", "double"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
#: The standard's keyword for a missing value, and its accepted replacements.
_UNKNOWN = {"unknown", "none", "null", "undefined"}


@dataclass
class _Concept:
    path: tuple[str, ...]
    kind: str  # "number", "boolean", "categorical" ("reference" until resolved)
    unit_type: str = ""
    literals: list[str] = field(default_factory=list)
    #: literal -> the expressions defining it (a derived categorical)
    definitions: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Literals defined by bounds and ranges on one number are ordered, in
    #: the order written.
    ordered: bool = False
    attribute: Optional[OddAttribute] = None

    @property
    def name(self) -> str:
        return ".".join(self.path)


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
    # One literal makes no categorical: a single key is a record's field.
    return len(node) >= 2 and all(
        isinstance(v, Mapping)
        and v
        and all(ids[_tail(k)] > own[_tail(k)] for k in v if str(k) != "METADATA")
        and all(_is_expression(x) for x in v.values())
        for v in node.values()
    )


def _tail(key: Any) -> str:
    return str(key).rsplit(".", 1)[-1]


def _flatten(
    node: Any,
    prefix: tuple[str, ...],
    ids: Counter[str],
    out: dict[str, _Concept],
    records: frozenset[tuple[str, ...]] = frozenset(),
) -> None:
    path = ".".join(prefix)
    if isinstance(node, list):
        out[path] = _Concept(
            prefix, "categorical", literals=[_literal(v) for v in node]
        )
        return
    if isinstance(node, Mapping):
        if prefix not in records and _defines_literals(node, ids):
            concept = _Concept(prefix, "categorical")
            concept.literals = [_literal(k) for k in node]
            concept.definitions = {_literal(k): v for k, v in node.items()}
            out[path] = concept
            return
        for key, child in node.items():
            _flatten(child, (*prefix, str(key)), ids, out, records)
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
        # A mapping of literals to expressions looks like a record of
        # categoricals: one whose "expressions" do not fit the concepts they
        # name is read again as a record.
        records: frozenset[tuple[str, ...]] = frozenset()
        while True:
            self.concepts: dict[str, _Concept] = {}
            _flatten(docs.taxonomy, (), ids, self.concepts, records)
            self._resolve_references()
            misread = frozenset(
                c.path
                for c in self.concepts.values()
                if c.definitions and not self._fits(c)
            )
            if not misread:
                break
            records |= misread
        for concept in self.concepts.values():
            concept.ordered = concept.ordered or self._defined_by_ranges(concept)
        self.module_names = set(docs.modules)
        self.labels: set[str] = set()
        for mdef in docs.modules.values():
            if isinstance(mdef, Mapping):
                self.labels.update(_labels_of(mdef))
        self.bindings = {
            self.concept(str(k), "binding").name: dict(v or {})
            for k, v in bindings.items()
        }
        #: Concept -> the unit its numbers are read in, when no probe says.
        self.assumed_units: dict[str, str] = {}

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
        """A concept typed by another concept's id takes its type.

        A categorical's literals (and their order), a number's unit type, a
        boolean.  A record type is not followed.
        """
        for concept in list(self.concepts.values()):
            if concept.kind != "reference":
                continue
            target = next(
                (
                    c
                    for c in self.concepts.values()
                    if c.path[-1] == concept.unit_type and c.kind != "reference"
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
            concept.kind = target.kind
            concept.literals = list(target.literals)
            concept.unit_type = target.unit_type
            concept.ordered = target.ordered or self._defined_by_ranges(target)

    def _outside(self, ref: str, concept: _Concept) -> Optional[_Concept]:
        """The concept *ref* names, outside *concept*; ``None`` if none or several."""
        found = [
            c
            for c in self.concepts.values()
            if (c.name == ref or c.name.endswith("." + ref))
            and c.path[: len(concept.path)] != concept.path
        ]
        return found[0] if len(found) == 1 else None

    def _fits(self, concept: _Concept) -> bool:
        """Whether each literal's expressions fit the concepts they name."""
        for definition in concept.definitions.values():
            for key, value in definition.items():
                if str(key) == "METADATA":
                    continue
                target = self._outside(str(key), concept)
                if target is None or target is concept:
                    return False
                if target.kind == "number":
                    if not isinstance(value, (int, float, str)) or isinstance(
                        value, bool
                    ):
                        return False
                    if isinstance(value, str) and not (
                        _RANGE_RE.match(value.strip())
                        or _BOUND_RE.match(value.strip())
                        or _QUANTITY_RE.match(value.strip())
                    ):
                        return False
                elif target.kind == "boolean":
                    if not isinstance(value, bool):
                        return False
                elif target.kind == "categorical":
                    literals = (
                        [_literal(v) for v in value]
                        if isinstance(value, list)
                        else [_literal(value)]
                    )
                    if not target.definitions and any(
                        x not in target.literals for x in literals
                    ):
                        return False
        return True

    def _defined_by_ranges(self, concept: _Concept) -> bool:
        """Whether every literal is a bound or range on one and the same number."""
        if not concept.definitions:
            return False
        targets = set()
        for definition in concept.definitions.values():
            keys = [k for k in definition if str(k) != "METADATA"]
            if len(keys) != 1:
                return False
            value = definition[keys[0]]
            target = self._outside(str(keys[0]), concept)
            if (
                target is None
                or target.kind != "number"
                or not isinstance(value, str)
                or not (
                    _RANGE_RE.match(value.strip()) or _BOUND_RE.match(value.strip())
                )
            ):
                return False
            targets.add(target.name)
        return len(targets) == 1

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
        to = self.unit_of(concept)
        if unit and self.units.unit_type(unit) is None and unit != to:
            raise OpenOddError(
                f"{where}: {concept.name}: unknown unit {unit!r} "
                "(numeric terms and $parameters are not read)"
            )
        if unit and not to:
            # The probe's unit is unknown: numbers on this concept are read in
            # the first unit written, and another unit is converted into it.
            # A probe that measures needs its unit to compare at all.
            if (
                concept.attribute is not None
                and concept.attribute.probe is not _missing
            ):
                raise OpenOddError(
                    f"{where}: {concept.name}: {unit!r} needs the probe's unit "
                    "(binding 'unit:')"
                )
            to = self.assumed_units.setdefault(concept.name, normalize_unit(unit))
        try:
            if unit and concept.unit_type:
                self.units.check_type(unit, concept.unit_type)
            return self.units.convert(value, unit or "", to)
        except UnitError as exc:
            raise OpenOddError(f"{where}: {concept.name}: {exc}") from exc

    # -- attributes --------------------------------------------------------

    def build_attributes(self) -> list[OddAttribute]:
        # A derived categorical after every concept its expressions read.
        built: list[_Concept] = []
        pending = list(self.concepts.values())
        while pending:
            ready = [
                c
                for c in pending
                if all(
                    (target := self._outside(str(k), c)) is not None
                    and target.attribute is not None
                    for d in c.definitions.values()
                    for k in d
                    if str(k) != "METADATA"
                )
            ]
            if not ready:
                names = ", ".join(c.name for c in pending)
                raise OpenOddError(f"TAXONOMY: {names} are defined by each other")
            for concept in ready:
                concept.attribute = self.attribute_of(concept)
                built.append(concept)
                pending.remove(concept)
        return [c.attribute for c in built if c.attribute is not None]

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
        extra = sorted(str(k) for k in binding if k not in _BINDING_KEYS)
        if extra:
            raise OpenOddError(f"probes.{concept.name}: unknown keys {extra}")
        criteria: dict[str, Any] = {}
        try:
            if "target" in binding:
                criteria["target"] = float(binding["target"])
            if "cover_by" in binding:
                criteria["cover_by"] = str(binding["cover_by"])
            if binding.get("min_stay") is not None:
                criteria["min_stay"] = float(binding["min_stay"])
        except (TypeError, ValueError) as exc:
            raise OpenOddError(
                f"probes.{concept.name}: target and min_stay must be numbers ({exc})"
            ) from exc
        try:
            return OddAttribute(
                concept.name,
                probe,
                unit=self.unit_of(concept),
                text=str(binding.get("text", "")),
                **criteria,
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
        if not [k for k in section if str(k) != "METADATA"]:
            raise OpenOddError(f"{where}: a section maps concepts to expressions")
        for key, value in section.items():
            key = str(key)
            if key == "METADATA":
                continue  # carries no semantics
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
            isinstance(value, str)
            and value.strip().lower() in _UNKNOWN
            and value.strip() not in concept.literals  # a literal named "none"
        ):
            return attribute.is_unknown()
        if concept.kind == "boolean":
            if not isinstance(value, bool):
                raise OpenOddError(
                    f"{where}: {concept.name} is a boolean: true or false, not {value!r}"
                )
            return attribute.equals(value)
        if isinstance(value, list):
            if concept.kind == "number":
                raise OpenOddError(
                    f"{where}: {concept.name} is a number: a list is for "
                    'literals; a range is written "[low .. high] unit"'
                )
            literals = [_literal(v) for v in value]
            self.require_literals(concept, literals, where)
            return attribute.is_in(literals)
        if concept.kind == "categorical" and not isinstance(value, str):
            self.require_literals(concept, [_literal(value)], where)
            return attribute.is_in([_literal(value)])
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
            if low > high:
                raise OpenOddError(
                    f"{where}: {concept.name}: {text!r} ends before it starts"
                )
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
            if any(c.path[-1] == name for c in self.concepts.values()):
                raise OpenOddError(f"{where}: a module's id must not be a concept's")
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
    *sources: Union[str, Path, GitSource],
    bindings: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    text: str = "",
) -> OddDefinition:
    """Read an ODD from OpenODD YAML files or text.

    Args:
        sources: Paths, YAML text, or files in git repositories
            (:class:`GitSource`).  Several are read together, as if one
            imported the others.  A file's ``IMPORT`` entries are read
            relative to it, or else from the other sources' directories.
        bindings: Concept -> how to measure it:
            ``{"probe": ..., "unit": ..., "values" | "buckets" | "range" +
            "every": ..., "text": ..., "target": ..., "cover_by": ...,
            "min_stay": ...}`` (see :func:`load_odd_binding`).  A
            concept with no probe is always missing.
        name: The ODD's name; the first file's stem by default.
        text: A description for the report.

    Raises:
        OpenOddError: when the documents do not make an ODD.
    """
    if not sources:
        raise OpenOddError("load_openodd(): no sources")
    docs = _Documents()
    paths: list[Union[str, Path]] = []
    for source in sources:
        if isinstance(source, GitSource):
            try:
                path, checkout = git_checkout(source)
            except GitSourceError as exc:
                raise OpenOddError(str(exc)) from exc
            docs.search += [path.parent, checkout.root]
            docs.checkouts.append(checkout.root)
            docs.provenance.append(
                {
                    "git": source.url,
                    "rev": source.rev,
                    "commit": checkout.commit,
                    "path": source.path,
                }
            )
            paths.append(path)
        else:
            if isinstance(source, Path):
                docs.search.append(source.resolve().parent)
            paths.append(source)
    for item in paths:
        _read(item, docs, ())
    reader = _Reader(docs, bindings or {}, name or docs.first_stem or "openodd", text)
    odd = reader.build()
    odd.sources = docs.provenance
    return odd


def load_odd_binding(path: Union[str, Path]) -> OddDefinition:
    """Read an ODD from a binding file: OpenODD files, and how to measure them.

    A binding file is this framework's, not OpenODD's::

        openodd:                             # relative to this file,
          - odd.yaml
          - git: https://example.com/odd/taxonomy.git
            rev: v1.2.0                      # or in git (:mod:`.sources`)
            path: taxonomy.yaml
        name: urban                          # default: this file's stem
        text: Urban roads, fair weather
        probes:
          speed_limit:                       # a concept, as conditions name it
            probe: speed_limit_kph           # built-in, or package.module:function
            unit: km/h                       # what the probe returns
            buckets: [0, 30, 60, 90]         # or values, or range + every
            cover_by: meters                 # optional: hits, seconds, meters, entries
            target: 200                      # optional: 200 m in each bucket
            min_stay: 2                      # optional: stays under 2 s do not count
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
    sources: list[Union[Path, GitSource]] = []
    for entry in _as_list(doc["openodd"]):
        if isinstance(entry, Mapping):
            try:
                sources += parse_git_entry(entry, str(path))
            except GitSourceError as exc:
                raise OpenOddError(str(exc)) from exc
        else:
            sources.append(path.parent / str(entry))
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
