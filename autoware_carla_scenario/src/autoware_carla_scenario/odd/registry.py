"""The built-in ODD, ODDs registered by name, and how a run picks one.

A run's ODD is named by a string (the ``odd`` key of the CLI config, or
``ScenarioQueue(odd=...)``):

* ``"default"`` (or nothing): :func:`default_odd`, every attribute the
  framework can measure and no modules, so every condition is inside it;
* a name registered with :func:`register_odd`, in code or by a package through
  the ``autoware_carla_scenario.odds`` entry point group (the entry point
  names a function that takes nothing and calls :func:`register_odd`, as the
  scenario and traffic-backend groups do);
* a path to a ``.yaml`` / ``.yml`` file: a binding file (OpenODD files and
  the probes that measure them, :func:`~autoware_carla_scenario.odd.load_odd_binding`),
  or an OpenODD document on its own
  (:func:`~autoware_carla_scenario.odd.load_openodd`; nothing is measured);
* ``"package.module:function"``: a builder to import and call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

from openodd import OddDefinition, bundled_odd, import_callable

from .backend import CARLA_BACKEND

__all__ = [
    "DEFAULT_ODD",
    "ENTRY_POINT_GROUP",
    "default_odd",
    "import_callable",
    "odd_builder",
    "odd_names",
    "register_odd",
    "resolve_odd",
]

logger = logging.getLogger(__name__)

#: The name of the built-in ODD.
DEFAULT_ODD = "default"

#: Entry point group a package registers ODD builders under.
ENTRY_POINT_GROUP = "autoware_carla_scenario.odds"

OddBuilder = Callable[[], OddDefinition]

_REGISTRY: dict[str, OddBuilder] = {}
_PLUGINS_LOADED = False


def default_odd() -> OddDefinition:
    """Every attribute the framework measures, after ISO 34503, and no modules.

    openodd_py's bundled ``default`` ODD (``openodd.bundled_odd()``), measured
    by :data:`~.backend.CARLA_BACKEND`: scenery (junction, speed limit, lane
    count, and the Lanelet2 ``location`` and ``subtype`` tags), environmental
    conditions (illumination, rain, fog) and dynamic elements (ego speed,
    traffic density, pedestrians nearby, and -- mapped onto the running
    scenario's measures -- the vehicle ahead and a pedestrian crossing ahead).

    A bucket a run only passes through is not covered: ego speed counts a
    stay of 3 s, the road attributes one of 2 s, the vehicle ahead one of
    1 s.  The speed buckets are centred on the multiples of 10 km/h, so an
    ego holding a speed limit stays inside one rather than flickering across
    an edge.
    """
    return bundled_odd(DEFAULT_ODD, backend=CARLA_BACKEND)


def register_odd(name: str, builder: OddBuilder) -> None:
    """Register *builder* (taking nothing, returning an ODD) under *name*."""
    if not name:
        raise ValueError("register_odd(): name must not be empty")
    if name == DEFAULT_ODD:
        raise ValueError(f"register_odd(): {DEFAULT_ODD!r} is the built-in ODD")
    _REGISTRY[name] = builder


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    from ..registry import load_entry_point_plugins  # noqa: PLC0415

    load_entry_point_plugins(ENTRY_POINT_GROUP, "ODD")


def odd_names() -> list[str]:
    """The names an ODD can be picked by: the built-in one and the registered ones."""
    _load_plugins()
    return [DEFAULT_ODD, *sorted(_REGISTRY)]


def _is_yaml(text: str) -> bool:
    return text.lower().endswith((".yaml", ".yml"))


def odd_builder(spec: Union[str, Path, OddDefinition, None]) -> Optional[OddBuilder]:
    """The Python builder *spec* names, or ``None`` for the built-in ODD or YAML."""
    if isinstance(spec, Path) and not _is_yaml(str(spec)):
        raise ValueError(f"{spec}: an ODD file is .yaml or .yml")
    if spec is None or isinstance(spec, (OddDefinition, Path)):
        return None
    text = str(spec).strip()
    if not text or text == DEFAULT_ODD or _is_yaml(text):
        return None
    _load_plugins()
    if text in _REGISTRY:
        return _REGISTRY[text]
    if ":" in text:
        return import_callable(text)
    raise ValueError(
        f"no ODD named {text!r}; known: {', '.join(odd_names())}, "
        "a .yaml file, or package.module:function"
    )


def resolve_odd(spec: Union[str, Path, OddDefinition, None] = None) -> OddDefinition:
    """The ODD *spec* names (see the module docstring).

    Raises:
        ValueError: if *spec* names nothing, or its builder does not return an
            :class:`OddDefinition`.
    """
    if isinstance(spec, OddDefinition):
        return spec
    builder = odd_builder(spec)
    if builder is not None:
        odd = builder()
        if not isinstance(odd, OddDefinition):
            raise ValueError(f"ODD builder {spec!r} returned {type(odd).__name__}")
        return odd
    text = str(spec).strip() if spec is not None else ""
    if _is_yaml(text):
        from .loader import load_odd_file  # noqa: PLC0415

        return load_odd_file(text)
    return default_odd()
