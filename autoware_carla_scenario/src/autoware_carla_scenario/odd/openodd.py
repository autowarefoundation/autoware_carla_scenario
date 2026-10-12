"""Read an ODD written in ASAM OpenODD 1.0 YAML, with a binding file (openodd's).

openodd reads the documents; this framework's backend (:mod:`.backend`)
names the probes a binding file may use (``probe: speed_limit_kph``,
``measure: vehicle_ahead_gap_m``, ``package.module:function``). See
``docs/odd.md``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional, Union

import openodd
from openodd import OpenOddError

from openodd.model import _missing  # noqa: F401 - the probe of a concept nothing measures

from .backend import sync_cache_env
from .model import OddDefinition

__all__ = ["OpenOddError", "load_odd_binding", "load_odd_file", "load_openodd"]


def load_openodd(
    *sources: Union[str, "os.PathLike[str]", openodd.GitSource],
    bindings: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    text: str = "",
    situations: Optional[Mapping[str, Any]] = None,
) -> OddDefinition:
    """Read an ODD from OpenODD YAML files or text (see :func:`openodd.load_openodd`)."""
    sync_cache_env()
    return openodd.load_openodd(
        *sources, bindings=bindings, name=name, text=text, situations=situations
    )


def load_odd_binding(path: Union[str, "os.PathLike[str]"]) -> OddDefinition:
    """Read an ODD from a binding file (see :func:`openodd.load_odd_binding`)."""
    sync_cache_env()
    return openodd.load_odd_binding(path)


def load_odd_file(path: Union[str, "os.PathLike[str]"]) -> OddDefinition:
    """Read a ``.yaml`` ODD file: a binding file, or an OpenODD document on its own."""
    sync_cache_env()
    return openodd.load_odd_file(path)
