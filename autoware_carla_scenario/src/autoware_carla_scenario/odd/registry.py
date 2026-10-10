"""The built-in ODD, ODDs registered by name, and how a run picks one.

A run's ODD is named by a string (the ``odd`` key of the CLI config, or
``ScenarioQueue(odd=...)``):

* ``"default"`` (or nothing): :func:`default_odd`, every attribute the
  framework can measure and no modules, so every condition is inside it;
* a name registered with :func:`register_odd`, in code or by a package through
  the ``autoware_carla_scenario.odds`` entry point group (the entry point
  names the builder: a function that takes nothing and returns an
  :class:`OddDefinition`);
* a path to an OpenODD YAML file (``.yaml`` / ``.yml``), read with
  :func:`~autoware_carla_scenario.odd.load_openodd`;
* ``"package.module:function"``: a builder to import and call.
"""

from __future__ import annotations

import importlib
import logging
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable, Optional, Union

from . import probes
from .model import OddAttribute, OddDefinition

__all__ = [
    "DEFAULT_ODD",
    "ENTRY_POINT_GROUP",
    "default_odd",
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

    Scenery (junction, speed limit, lane count, and the Lanelet2 ``location``
    and ``subtype`` tags), environmental conditions (illumination, rain, fog)
    and dynamic elements (ego speed, traffic density, pedestrians nearby).
    """
    return OddDefinition(
        DEFAULT_ODD,
        attributes=[
            # Scenery
            OddAttribute(
                "scenery.junction",
                probes.in_junction,
                values=[False, True],
                text="Ego inside a junction",
            ),
            OddAttribute(
                "scenery.location",
                probes.lanelet_location,
                values=["urban", "nonurban", "private"],
                text="Lanelet2 location tag of the ego's lanelet",
            ),
            OddAttribute(
                "scenery.road_type",
                probes.lanelet_subtype,
                values=["road", "highway", "road_shoulder", "play_street", "parking"],
                text="Lanelet2 subtype tag of the ego's lanelet",
            ),
            OddAttribute(
                "scenery.speed_limit",
                probes.speed_limit_kph,
                unit="km/h",
                buckets=[0, 30, 40, 50, 60, 80, 100, 130],
                text="Speed limit for the ego's lane (Lanelet2, else CARLA)",
            ),
            OddAttribute(
                "scenery.lane_count",
                probes.lane_count,
                values=[1, 2, 3, 4],
                text="Driving lanes in the ego's direction (outside junctions)",
            ),
            # Environmental conditions
            OddAttribute(
                "environment.illumination",
                probes.illumination,
                values=probes.ILLUMINATION_LEVELS,
                text="Illumination, from the sun's altitude",
            ),
            OddAttribute(
                "environment.rain",
                probes.rain,
                values=probes.INTENSITY_LEVELS,
                text="Rain, from CARLA's 0-100 precipitation",
            ),
            OddAttribute(
                "environment.fog",
                probes.fog,
                values=probes.INTENSITY_LEVELS,
                text="Fog, from CARLA's 0-100 fog density",
            ),
            # Dynamic elements
            OddAttribute(
                "dynamic.ego_speed",
                probes.ego_speed_kph,
                unit="km/h",
                range=(0.0, 120.0),
                every=10.0,
                text="Ego speed",
            ),
            OddAttribute(
                "dynamic.traffic_density",
                probes.traffic_density,
                values=probes.TRAFFIC_DENSITY_LEVELS,
                text=f"Other vehicles within {probes.NEARBY_RADIUS_M:g} m of the ego",
            ),
            OddAttribute(
                "dynamic.pedestrian_nearby",
                probes.pedestrian_nearby,
                values=[False, True],
                text=f"A pedestrian within {probes.NEARBY_RADIUS_M:g} m of the ego",
            ),
        ],
        text="Every attribute the framework measures; no restrictions.",
    )


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
    for entry_point in importlib_metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            _REGISTRY.setdefault(entry_point.name, entry_point.load())
        except Exception:  # noqa: BLE001 -- one bad plugin must not kill the CLI
            logger.exception(
                "Failed to load ODD plugin %r; skipping.", entry_point.name
            )


def odd_names() -> list[str]:
    """The names an ODD can be picked by: the built-in one and the registered ones."""
    _load_plugins()
    return [DEFAULT_ODD, *sorted(_REGISTRY)]


def odd_builder(spec: Union[str, Path, OddDefinition, None]) -> Optional[OddBuilder]:
    """The Python builder *spec* names, or ``None`` for the built-in ODD or YAML."""
    if spec is None or isinstance(spec, (OddDefinition, Path)):
        return None
    text = str(spec).strip()
    if not text or text == DEFAULT_ODD or text.endswith((".yaml", ".yml")):
        return None
    _load_plugins()
    if text in _REGISTRY:
        return _REGISTRY[text]
    if ":" in text:
        module_name, _, attr = text.partition(":")
        builder: OddBuilder = getattr(importlib.import_module(module_name), attr)
        return builder
    raise ValueError(
        f"no ODD named {text!r}; known: {', '.join(odd_names())}, "
        "an OpenODD .yaml file, or package.module:function"
    )


def resolve_odd(spec: Union[str, Path, OddDefinition, None] = None) -> OddDefinition:
    """The ODD *spec* names (see the module docstring).

    Raises:
        ValueError: if *spec* names nothing, or its builder does not return an
            :class:`OddDefinition`.
    """
    if isinstance(spec, OddDefinition):
        return spec
    text = str(spec).strip() if spec is not None else ""
    if not text or text == DEFAULT_ODD:
        return default_odd()
    if text.endswith((".yaml", ".yml")):
        from .openodd import load_openodd  # noqa: PLC0415

        return load_openodd(Path(text))
    builder = odd_builder(text)
    assert builder is not None
    odd = builder()
    if not isinstance(odd, OddDefinition):
        raise ValueError(f"ODD builder {text!r} returned {type(odd).__name__}")
    return odd
