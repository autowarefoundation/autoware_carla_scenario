"""The program the checker compiles: build a scenario and call its setup().

Codon only checks a function it compiles, and it compiles a function only
when something calls it.  The driver therefore does what the runner does with
a scenario registered by :func:`~autoware_carla_scenario.register_scenario`::

    scenario_cls(ego, config=config_cls(**scenario_dict),
                 spawn_pose=spawn_pose, ground_projection=ground_projection)
    scenario.setup()
    scenario.is_done()

with the config built from the scenario's own YAML values, so a value of the
wrong type (``timeout_seconds: fast``) or a key the config class does not have
is caught as well.  When every field of the config class has a default (the
usual case for a Hydra-driven config), the driver builds the default config
and assigns each value on a line of its own, so Codon reports a bad value at
that line, and :attr:`Driver.config_lines` maps it back to the
``scenario.<key>`` it came from; otherwise it passes the values to the
constructor, as the runner does.

A nested mapping is rendered as the dataclass its field is annotated with only
where the config the runner builds holds one there (its ``__post_init__``
converts it): ``config_cls(**scenario_dict)`` leaves a mapping it does not
convert a ``dict``, and so does the driver, so Codon refuses the value at its
``scenario.<key>`` line instead of passing code that reads the field's
attributes.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import math
import types
import typing
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DRIVER_MODULE", "Driver", "render_driver"]

#: File name of the generated program in the check workspace.
DRIVER_MODULE = "_acs_check_main.py"


@dataclass
class Driver:
    """A generated driver program."""

    source: str
    #: Driver line number -> the ``scenario.<key>`` whose value is on it.
    config_lines: dict[int, str] = field(default_factory=dict)


class _Imports:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def name(self, cls: type) -> str:
        """The name *cls* is imported as in the driver."""
        line = f"from {cls.__module__} import {cls.__qualname__.split('.')[0]}"
        if line not in self.lines:
            self.lines.append(line)
        return cls.__qualname__


def _hint_args(hint: Any) -> tuple[Any, tuple[Any, ...]]:
    origin = typing.get_origin(hint)
    return origin, typing.get_args(hint)


#: The runtime value is not known (the config did not build in Python).
_UNKNOWN: Any = object()


def _at(actual: Any, key: Any) -> Any:
    """The part of the runtime value *actual* that holds *key*."""
    if actual is _UNKNOWN:
        return _UNKNOWN
    try:
        if dataclasses.is_dataclass(actual) and not isinstance(actual, type):
            return getattr(actual, key)
        return actual[key]
    except (AttributeError, IndexError, KeyError, TypeError):
        return _UNKNOWN


def _render(value: Any, hint: Any, imports: _Imports, actual: Any = _UNKNOWN) -> str:
    """*value* as a Codon expression of the type *hint* names (Any: its own).

    *actual* is what the config built in Python holds for *value*: a mapping
    becomes the dataclass *hint* names only if that holds one.
    """
    origin, args = _hint_args(hint)
    if origin is typing.Union or (
        hasattr(types, "UnionType") and isinstance(hint, types.UnionType)
    ):
        rest = [a for a in args if a is not type(None)]
        if value is None:
            return "None"
        return _render(value, rest[0] if len(rest) == 1 else Any, imports, actual)
    if isinstance(value, enum.Enum):
        return f"{imports.name(type(value))}.{value.name}"
    if value is None:
        return "None"
    if isinstance(value, bool):
        return repr(value)
    if hint is float and isinstance(value, int):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return 'float("nan")'
        if math.isinf(value):
            return 'float("inf")' if value > 0 else 'float("-inf")'
        return repr(value)
    if isinstance(value, (int, str)):
        return repr(value)
    if isinstance(value, dict):
        if (
            isinstance(hint, type)
            and dataclasses.is_dataclass(hint)
            and (actual is _UNKNOWN or isinstance(actual, hint))
        ):
            return _render_dataclass(hint, value, imports, actual)
        key_hint, value_hint = (args + (Any, Any))[:2] if origin is dict else (Any, Any)
        items = ", ".join(
            f"{_render(k, key_hint, imports)}: "
            f"{_render(v, value_hint, imports, _at(actual, k))}"
            for k, v in value.items()
        )
        return "{" + items + "}"
    if isinstance(value, (list, tuple)):
        item_hint = args[0] if origin in (list, tuple) and args else Any
        items = ", ".join(
            _render(v, item_hint, imports, _at(actual, i)) for i, v in enumerate(value)
        )
        return f"({items},)" if isinstance(value, tuple) and value else f"[{items}]"
    return repr(value)


def _field_hints(cls: type) -> dict[str, Any]:
    try:
        return typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001 - an unresolvable hint renders as Any
        return {}


def _render_dataclass(
    cls: type, values: dict[str, Any], imports: _Imports, actual: Any = _UNKNOWN
) -> str:
    hints = _field_hints(cls)
    args = ", ".join(
        f"{key}={_render(value, hints.get(key, Any), imports, _at(actual, key))}"
        for key, value in values.items()
    )
    return f"{imports.name(cls)}({args})"


def _all_fields_have_defaults(cls: type) -> bool:
    if not dataclasses.is_dataclass(cls):
        return False
    return all(
        f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING
        for f in dataclasses.fields(cls)
        if f.init
    )


def render_driver(
    scenario_cls: type,
    config_cls: type,
    scenario_dict: dict[str, Any],
) -> Driver:
    """The driver program for *scenario_cls* built with *scenario_dict*."""
    imports = _Imports()
    scenario_name = imports.name(scenario_cls)
    config_name = imports.name(config_cls)
    hints = _field_hints(config_cls)
    try:
        # What the runner builds, to render each value as the config holds it.
        built: Any = config_cls(**copy.deepcopy(scenario_dict))
    except Exception:  # noqa: BLE001 - the driver's constructor call reports it
        built = _UNKNOWN

    by_assignment = _all_fields_have_defaults(config_cls)
    body = [
        "def _acs_expect_scenario(scenario: BaseScenario):",
        "    pass",
        "",
        "",
        "def _acs_check_scenario():",
        f"    config = {config_name}()"
        if by_assignment
        else f"    config = {config_name}(",
    ]
    config_args: list[tuple[str, str]] = []
    for key, value in scenario_dict.items():
        rendered = _render(value, hints.get(key, Any), imports, _at(built, key))
        line = (
            f"    config.{key} = {rendered}"
            if by_assignment
            else f"        {key}={rendered},"
        )
        config_args.append((key, line))
    tail = [
        *([] if by_assignment else ["    )"]),
        f"    scenario = {scenario_name}(",
        "        EgoConfig(SpawnTransform(carla.Transform())),",
        "        config=config,",
        "        spawn_pose=Lanelet2Pose(0, 0.0),",
        "        ground_projection=GroundProjectionConfig(),",
        "    )",
        "    _acs_expect_scenario(scenario)",
        "    scenario.setup()",
        "    done: bool = scenario.is_done()",
        "",
        "",
        "_acs_check_scenario()",
    ]
    head = [
        "import carla",
        "from autoware_carla_scenario import (BaseScenario, EgoConfig, GroundProjectionConfig,",
        "                                     Lanelet2Pose, SpawnTransform)",
        *imports.lines,
        "",
        "",
    ]
    lines = head + body
    config_lines: dict[int, str] = {}
    for key, line in config_args:
        lines.append(line)
        config_lines[len(lines)] = f"scenario.{key}"
    lines += tail
    return Driver("\n".join(lines) + "\n", config_lines)
