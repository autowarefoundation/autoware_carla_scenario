"""The built-in ODD, ODDs registered by name, and how a run picks one.

A run's ODD is named by a string (the ``odd`` key of the CLI config, or
``ScenarioQueue(odd=...)``):

* ``"default"`` (or nothing): :func:`default_odd`, openodd's bundled default
  ODD -- every attribute the framework can measure and no modules, so every
  condition is inside it;
* a name registered with :func:`register_odd`, in code or by a package through
  the ``autoware_carla_scenario.odds`` entry point group (the entry point
  names a function that takes nothing and calls :func:`register_odd`, as the
  scenario and traffic-backend groups do), or openodd's own ``openodd.odds``
  group;
* a path to a ``.yaml`` / ``.yml`` file: a binding file (OpenODD files and
  the probes that measure them, :func:`~autoware_carla_scenario.odd.load_odd_binding`),
  or an OpenODD document on its own
  (:func:`~autoware_carla_scenario.odd.load_openodd`; nothing is measured);
* ``"package.module:function"``: a builder to import and call.

The registry and the reading of the string are openodd's
(:mod:`openodd.registry`): an ODD registered here is registered there, and the
other way round. This module adds this framework's entry point group, and
reads files with this framework's backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from openodd import registry as _registry
from openodd.backend import import_callable
from openodd.registry import DEFAULT_ODD as _DEFAULT_ODD
from openodd.registry import OddBuilder, odd_names, register_odd

from .model import OddDefinition

__all__ = [
    "DEFAULT_ODD",
    "ENTRY_POINT_GROUP",
    "OddBuilder",
    "default_odd",
    "import_callable",
    "odd_builder",
    "odd_names",
    "register_odd",
    "resolve_odd",
]

#: The name of the built-in ODD.
DEFAULT_ODD = _DEFAULT_ODD

#: Entry point group a package registers ODD builders under (openodd's
#: ``openodd.odds`` is read too).
ENTRY_POINT_GROUP = "autoware_carla_scenario.odds"

_registry.add_entry_point_group(ENTRY_POINT_GROUP)


def default_odd() -> OddDefinition:
    """The built-in ODD: openodd's bundled ``default``, measured in CARLA.

    Every attribute the framework measures, after ISO 34503, and no modules:
    scenery (junction, speed limit, lane count, the Lanelet2 ``location`` and
    ``subtype`` tags), environmental conditions (illumination, rain, fog) and
    dynamic elements (ego speed, traffic density, pedestrians nearby, and --
    mapped onto the running scenario's measures -- the vehicle ahead and a
    pedestrian crossing ahead). Its files ship with openodd, so its revision
    changes only when they do. See ``docs/odd.md``.
    """
    from openodd import bundled_odd  # noqa: PLC0415

    from .backend import CARLA_BACKEND  # noqa: PLC0415

    return bundled_odd(DEFAULT_ODD, backend=CARLA_BACKEND)


def odd_builder(spec: Union[str, Path, OddDefinition, None]) -> Optional[OddBuilder]:
    """The Python builder *spec* names, or ``None`` for the built-in ODD or YAML."""
    return _registry.odd_builder(spec)


def resolve_odd(spec: Union[str, Path, OddDefinition, None] = None) -> OddDefinition:
    """The ODD *spec* names (see the module docstring), files read in CARLA.

    Raises:
        ValueError: if *spec* names nothing, or its builder does not return an
            :class:`OddDefinition`.
    """
    from .backend import CARLA_BACKEND, sync_cache_env  # noqa: PLC0415

    sync_cache_env()
    return _registry.resolve_odd(spec, backend=CARLA_BACKEND)
