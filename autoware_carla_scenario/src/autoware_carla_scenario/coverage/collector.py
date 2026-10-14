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
  (``odd.dynamic.vehicle_ahead_gap#2``), so openodd's merge, which keys an
  item by its name, buckets, event and criteria -- not its text -- never adds
  hits of one definition to another's (docs/odd.md, "Measure versions");
* ``measures``: each measure the ODD maps, with its version, unit and text,
  and for the vehicle-ahead measures how many frames each definition read
  them (:func:`~autoware_carla_scenario.measures.vehicle_ahead_definitions`).
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
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
    "generator",
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


def versioned_item_name(name: str, version: int) -> str:
    """The name a cover item of a measure of *version* is written under:
    *name* itself for the first version, ``name#<version>`` after it.

    ``#<n>`` is the suffix openodd's merge gives the n-th definition of an
    item it meets, and the one its sampler reads past to the ODD's attribute,
    so coverage of the current definition still steers a coverage-driven
    sweep."""
    if version <= 1:
        return name
    return f"{name}#{version}"


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
