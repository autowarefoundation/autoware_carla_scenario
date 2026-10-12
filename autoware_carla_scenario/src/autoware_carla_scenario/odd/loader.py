"""Read an ODD from ASAM OpenODD 1.0 YAML and a binding file -- with openodd_py.

openodd_py reads the YAML (taxonomy, modules, ``IMPORT``, units, files in
git) and builds the ODD; the binding file names the probes that measure each
concept.  These functions read with :data:`~.backend.CARLA_BACKEND`, so the
probe names a binding gives (``probe: speed_limit_kph``, ``measure:
vehicle_ahead_gap_m``) are the framework's.  See openodd_py's
``load_odd_binding`` for the binding file's format.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional, Union

import openodd
from openodd import GitSource, GitSourceError, OddDefinition, OpenOddError

from .backend import CARLA_BACKEND

__all__ = [
    "GitSource",
    "GitSourceError",
    "OpenOddError",
    "load_odd_binding",
    "load_odd_file",
    "load_openodd",
]

PathLike = Union[str, "os.PathLike[str]"]


def load_odd_binding(path: PathLike) -> OddDefinition:
    """Read an ODD from a binding file: OpenODD files, and the probes measuring them."""
    return openodd.load_odd_binding(path, backend=CARLA_BACKEND)


def load_odd_file(path: PathLike) -> OddDefinition:
    """Read a ``.yaml`` ODD file: a binding file, or an OpenODD document on its
    own (read with no probes, so nothing is measured)."""
    return openodd.load_odd_file(path, backend=CARLA_BACKEND)


def load_openodd(
    *sources: Union[PathLike, GitSource],
    bindings: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    text: str = "",
    situations: Optional[Mapping[str, Any]] = None,
) -> OddDefinition:
    """Read an ODD from OpenODD YAML files or text (paths, text, or :class:`GitSource`).

    *bindings* map a concept to how it is measured (a binding file's
    ``probes``); *situations* a module to its criteria.
    """
    return openodd.load_openodd(
        *sources,
        bindings=bindings,
        name=name,
        text=text,
        situations=situations,
        backend=CARLA_BACKEND,
    )
