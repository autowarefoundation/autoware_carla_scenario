"""Sample cover items during a run and write what they hit (openodd's collector).

The scenario runner drives openodd's :class:`~openodd.CoverageCollector`:
``start`` when the clock starts, ``tick`` after every world tick, ``end`` when
the tick loop is over, each given the ``carla.World``.  :class:`CoverageCollector`
here is that collector with this framework's defaults: the CARLA backend
(:data:`~autoware_carla_scenario.odd.backend.CARLA_BACKEND`), which says where
the ego is, and the framework named in the file's ``generator``.

What a coverage file holds -- hits and exposure per bucket, the ODD's ticks,
cells and samples, and the ODD's archive beside it -- is openodd's; see its
:mod:`openodd.collector` and ``docs/coverage.md``.  This framework adds what
its scenario measures mean:

* the cover item of an ODD attribute mapped onto a measure whose definition is
  past its first version is named for that version
  (``odd.dynamic.vehicle_ahead_gap@v2``), so openodd's merge, which keys an
  item by its name, buckets, event and criteria -- not its text -- never adds
  hits of one definition to another's (docs/odd.md, "Measure versions").
  ``@v<n>``, not ``#<n>``: ``#<n>`` is openodd's own suffix for an item it
  meets defined differently, and its sampler pools those;
  :func:`current_measure_coverage` is the coverage a sampler reads instead;
* ``measures``: each measure the ODD maps, with its version, unit and text,
  and for the vehicle-ahead measures how many frames each definition read
  them (:func:`~autoware_carla_scenario.measures.vehicle_ahead_definitions`).
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
from types import SimpleNamespace
from typing import Any, Optional

from openodd import Backend
from openodd.collector import COVERAGE_SCHEMA, CROSS_SEPARATOR, ODD_ARCHIVE_SUFFIX
from openodd.collector import CoverageCollector as _CoverageCollector
from openodd.collector import generator as _generator

__all__ = [
    "COVERAGE_SCHEMA",
    "CROSS_SEPARATOR",
    "FRAMEWORK_DISTRIBUTION",
    "ODD_ARCHIVE_SUFFIX",
    "CoverageCollector",
    "current_measure_coverage",
    "generator",
    "prepare_measures",
    "versioned_item_name",
]

#: This framework's distribution, named in a coverage file's ``generator``.
FRAMEWORK_DISTRIBUTION = "autoware-carla-scenario"


def _framework() -> dict[str, Optional[str]]:
    try:
        version: Optional[str] = metadata.version(FRAMEWORK_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        version = None
    return {"framework": f"{FRAMEWORK_DISTRIBUTION}=={version}" if version else None}


def generator() -> dict[str, Optional[str]]:
    """What writes the coverage files: openodd's and this framework's versions
    (``None`` when one cannot be told)."""
    return _generator(_framework())


#: What a cover item's name is suffixed with for its measure's version.
VERSION_MARK = "@v"


def versioned_item_name(name: str, version: int) -> str:
    """The name a cover item of a measure of *version* is written under:
    *name* itself for the first version, ``name@v<version>`` after it.

    Not ``name#<version>``: ``#<n>`` is the suffix openodd's merge gives the
    n-th differing definition of an item, and a v1 item whose buckets changed
    would take ``#2`` too; openodd's sampler, besides, pools an item's ``#<n>``
    definitions.  :func:`current_measure_coverage` hands a sampler only the
    current version's."""
    if version <= 1:
        return name
    return f"{name}{VERSION_MARK}{version}"


def _split_versioned(name: str) -> tuple[str, int, str]:
    """``(base, version, rest)`` of a merged item's name: ``name@v2#3`` is
    ``("name", 2, "#3")``, ``name#3`` is ``("name", 1, "#3")``."""
    rest = ""
    hash_at = name.find("#")
    if hash_at >= 0:
        name, rest = name[:hash_at], name[hash_at:]
    mark_at = name.rfind(VERSION_MARK)
    if mark_at >= 0 and name[mark_at + len(VERSION_MARK) :].isdigit():
        return name[:mark_at], int(name[mark_at + len(VERSION_MARK) :]), rest
    return name, 1, rest


class _Renamed:
    """A merged entry read under another name."""

    def __init__(self, entry: Any, name: str) -> None:
        self.name = name
        self.buckets = list(entry.buckets)
        self.target = entry.target
        self.cover_by = getattr(entry, "cover_by", "hits")
        self._entry = entry

    def amount(self, bucket: str) -> float:
        return float(self._entry.amount(bucket))


def current_measure_coverage(coverage: Any, odd: Any) -> Any:
    """*coverage* (a merged report) as a sampler of *odd* should read it.

    The items of attributes mapped onto a measure past its first version
    (:func:`versioned_item_name`) are kept only as the measure's current
    version reads them, under the attribute's own item name, so a
    coverage-driven sweep aims at what the current definition has not
    covered -- never at hits of an older definition, nor both added up.
    Any other item is kept as it is; *coverage* itself when nothing is
    versioned."""
    from ..measures import measure_version  # noqa: PLC0415

    if coverage is None or odd is None:
        return coverage
    current = {
        name: measure_version(key)
        for name, key in _measured_items(odd).items()
        if measure_version(key) > 1
    }
    if not current:
        return coverage
    entries: list[Any] = []
    for entry in coverage.entries:
        base, version, rest = _split_versioned(str(entry.name))
        if base not in current:
            entries.append(entry)
        elif version == current[base]:
            entries.append(_Renamed(entry, base + rest))
    return SimpleNamespace(entries=entries)


def prepare_measures(odd: Any) -> None:
    """Build what the measures *odd* maps read before the run's first tick:
    the routing graph, for the vehicle-ahead measures
    (:func:`~autoware_carla_scenario.measures.prepare_vehicle_ahead`)."""
    from ..measures import (  # noqa: PLC0415
        VEHICLE_AHEAD_GAP_M,
        VEHICLE_AHEAD_RELATIVE_SPEED_KPH,
        prepare_vehicle_ahead,
    )

    keys = set(_measured_items(odd).values()) if odd is not None else set()
    if keys & {VEHICLE_AHEAD_GAP_M, VEHICLE_AHEAD_RELATIVE_SPEED_KPH}:
        prepare_vehicle_ahead()


def _measured_items(odd: Any) -> dict[str, str]:
    """ODD cover item -> the measure key its attribute reads."""
    from ..odd.scenario_measure import ScenarioMeasure  # noqa: PLC0415

    found: dict[str, str] = {}
    for attribute in getattr(odd, "attributes", ()):
        item = getattr(attribute, "item", None)
        probe = getattr(attribute, "probe", None)
        if item is not None and isinstance(probe, ScenarioMeasure):
            found[item.name] = probe.key
    return found


class CoverageCollector(_CoverageCollector):
    """openodd's collector, measuring with the CARLA backend by default, and
    writing what the scenario measures it reads mean (the module docstring)."""

    def __init__(
        self,
        items: list[Any],
        crosses: Optional[list[Any]] = None,
        odd: Any = None,
        *,
        backend: Optional[Backend] = None,
        generator: Optional[Mapping[str, Optional[str]]] = None,
    ) -> None:
        from ..odd.backend import CARLA_BACKEND  # noqa: PLC0415 - odd imports coverage

        super().__init__(
            items,
            crosses,
            odd,
            backend=backend or CARLA_BACKEND,
            generator=_framework() if generator is None else generator,
        )
        self._measured_items = _measured_items(odd) if odd is not None else {}
        self._definitions: dict[str, int] = {}

    def end(self, context: Any, elapsed: float) -> None:
        """Sample the items on the end, and keep which definitions the
        vehicle-ahead measures read with (the runner forgets them as the run
        ends, before the file is written)."""
        from ..measures import vehicle_ahead_definitions  # noqa: PLC0415

        super().end(context, elapsed)
        self._definitions = vehicle_ahead_definitions()

    def to_dict(self, scenario: str) -> dict[str, Any]:
        """openodd's coverage document, with measure items versioned and the
        measures the ODD reads described."""
        from ..measures import (  # noqa: PLC0415
            BUILT_IN_MEASURES,
            VEHICLE_AHEAD_GAP_M,
            VEHICLE_AHEAD_RELATIVE_SPEED_KPH,
            measure_version,
            vehicle_ahead_definitions,
        )

        doc = super().to_dict(scenario)
        renamed = {
            name: versioned_item_name(name, measure_version(key))
            for name, key in self._measured_items.items()
        }
        for item in doc.get("items") or ():
            if item.get("name") in renamed:
                item["name"] = renamed[item["name"]]
        for cross in doc.get("crosses") or ():
            if isinstance(cross.get("items"), list):
                cross["items"] = [renamed.get(n, n) for n in cross["items"]]
        definitions = self._definitions or vehicle_ahead_definitions()
        measures: dict[str, Any] = {}
        for key in sorted(set(self._measured_items.values())):
            entry: dict[str, Any] = {"version": measure_version(key)}
            if key in BUILT_IN_MEASURES:
                entry["unit"] = BUILT_IN_MEASURES[key].unit
                entry["text"] = BUILT_IN_MEASURES[key].text
            if key in (VEHICLE_AHEAD_GAP_M, VEHICLE_AHEAD_RELATIVE_SPEED_KPH):
                entry["definitions"] = dict(definitions)
            measures[key] = entry
        if measures:
            doc["measures"] = measures
        return doc
