"""Sample cover items during a run and write what they hit (openodd's collector).

The scenario runner drives openodd's :class:`~openodd.CoverageCollector`:
``start`` when the clock starts, ``tick`` after every world tick, ``end`` when
the tick loop is over, each given the ``carla.World``.  :class:`CoverageCollector`
here is that collector with this framework's defaults: the CARLA backend
(:data:`~autoware_carla_scenario.odd.backend.CARLA_BACKEND`), which says where
the ego is, and the framework named in the file's ``generator``.

What a coverage file holds -- hits and exposure per bucket, the ODD's ticks,
cells and samples, and the ODD's archive beside it -- is openodd's; see its
:mod:`openodd.collector` and ``docs/coverage.md``.
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


class CoverageCollector(_CoverageCollector):
    """openodd's collector, measuring with the CARLA backend by default."""

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
