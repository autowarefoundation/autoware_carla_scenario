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

Codon types the program statically, so the values of a sweep's cases do not
change the verdict, only their types do: :attr:`Driver.shape` is the program
with each value replaced by its type, and the check's key
(:func:`~.check._cache_key`), so a sweep compiles once per shape of its
config, not once per case.

After that, the driver calls the function appended to each of the scenario's
own modules (:func:`render_module_checks`), which calls every function and
method of the module and assigns each class-level default, so what the
calls above do not reach is compiled too.  Last, it checks what the
scenario's ``setup()`` and ``is_done()`` return, strictly: ``None`` and a
``bool`` (#238).
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

__all__ = [
    "DRIVER_MODULE",
    "Driver",
    "render_driver",
    "render_module_checks",
    "render_odd_driver",
]

#: The driver's function checking the results of the scenario's hooks.
HOOK_CHECK = "_acs_check_scenario_hooks"
#: File name of the generated program in the check workspace.
DRIVER_MODULE = "_acs_check_main.py"


@dataclass
class Driver:
    """A generated driver program."""

    source: str
    #: Driver line number -> the ``scenario.<key>`` whose value is on it.
    config_lines: dict[int, str] = field(default_factory=dict)
    #: What the driver's own calls are, as a warning names them: a function
    #: the appended calls leave out is compiled only where these reach it.
    reaches: str = "the scenario's constructor, setup() and is_done()"
    #: What runs after the module checks (:func:`render_module_checks`).
    trailer: str = ""
    #: :attr:`source` with each config value replaced by its type, as
    #: :func:`render_driver` renders it in a shape (:class:`_Imports`): what
    #: the verdict depends on, so what the check is keyed by.  Empty when
    #: the program builds no values: the key is :attr:`source` itself.
    shape: str = ""
    #: The config values the program builds, each key's ``repr()``: what
    #: tells one case of a sweep from another in a log.
    case: dict[str, str] = field(default_factory=dict)
    #: What building ``config_cls(**scenario_dict)`` in Python raised, as
    #: ``"TypeError: ..."``: the runner builds the config the same way, so the
    #: scenario cannot run.  Not part of the shape: the verdict is cached
    #: for the types, and this depends on the values.
    build_error: str = ""


class _Imports:
    def __init__(self, *, shape: bool = False) -> None:
        self.lines: list[str] = []
        #: Render each value as a token of its type (:func:`_literal`).
        self.shape = shape
        #: ``(statement, config path)``: a scalar field of a nested dataclass,
        #: passed to a function taking that type first, so that Codon's
        #: error for it is at a line of its own (it reports a call's
        #: argument at the call): the path names the field (#531).
        self.checks: list[tuple[str, str]] = []

    def name(self, cls: type) -> str:
        """The name *cls* is imported as in the driver."""
        line = f"from {cls.__module__} import {cls.__qualname__.split('.')[0]}"
        if line not in self.lines:
            self.lines.append(line)
        return cls.__qualname__


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


#: The ints a Codon literal holds (``int`` is 64-bit): one beyond is an error.
_INT_MIN, _INT_MAX = -(2**63), 2**63 - 1


def _literal(value: Any, rendered: str, imports: _Imports) -> str:
    """*rendered*, or in a shape (:attr:`_Imports.shape`), the type of *value*.

    Only a value whose rendering is a literal Codon accepts whatever the value
    is turns into its type: a bool, a float (``float("nan")`` too), an int
    that fits in 64 bits and a printable ASCII string.  Codon types the
    program statically and reads no value at compile time, so another such
    value of the same type is the same program to it.  Anything else (an int
    too large, a string with escapes, a ``repr()`` of some other object) is
    kept as it is, so a value that could change the verdict changes the key.
    """
    if not imports.shape:
        return rendered
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, float):
        return "<float>"
    if isinstance(value, int) and _INT_MIN <= value <= _INT_MAX:
        return "<int>"
    if (
        isinstance(value, str)
        and value.isascii()
        and value.isprintable()
        and "\\" not in rendered
    ):
        return "<str>"
    return rendered


#: Marks a line of a rendered value with the config path it holds (#531).
_PATH_MARK = "  #@acs-path:"


def _render(
    value: Any,
    hint: Any,
    imports: _Imports,
    actual: Any = _UNKNOWN,
    path: str = "",
) -> str:
    """*value* as a Codon expression of the type *hint* names (Any: its own).

    *actual* is what the config built in Python holds for *value*: a mapping
    becomes the dataclass *hint* names only if that holds one, and a string
    or list the config's ``__post_init__`` turned into an Enum member or a
    tuple is rendered as that (#531).  *path* is the config path of *value*
    (``scenario.legs[1]``): a nested dataclass is rendered one field to a
    line, each marked with its path (:data:`_PATH_MARK`).
    """
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Union or isinstance(hint, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        return _render(value, rest[0] if len(rest) == 1 else Any, imports, actual, path)
    if isinstance(actual, enum.Enum) and not isinstance(value, enum.Enum):
        value = actual
    elif isinstance(actual, tuple) and isinstance(value, list):
        value = actual
    if isinstance(value, enum.Enum):
        cls = imports.name(type(value))
        return f"<{cls}>" if imports.shape else f"{cls}.{value.name}"
    if value is None:
        return "None"
    if isinstance(value, bool):
        return _literal(value, repr(value), imports)
    if hint is float and isinstance(value, int):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return _literal(value, 'float("nan")', imports)
        if math.isinf(value):
            inf = 'float("inf")' if value > 0 else 'float("-inf")'
            return _literal(value, inf, imports)
        return _literal(value, repr(value), imports)
    if isinstance(value, (int, str)):
        return _literal(value, repr(value), imports)
    if isinstance(value, dict):
        if (
            isinstance(hint, type)
            and dataclasses.is_dataclass(hint)
            and (actual is _UNKNOWN or isinstance(actual, hint))
        ):
            return _render_dataclass(hint, value, imports, actual, path)
        key_hint, value_hint = (args + (Any, Any))[:2] if origin is dict else (Any, Any)
        items = ", ".join(
            f"{_render(k, key_hint, imports)}: "
            f"{_render(v, value_hint, imports, _at(actual, k), f'{path}[{k!r}]')}"
            for k, v in value.items()
        )
        return "{" + items + "}"
    if isinstance(value, (list, tuple)):
        item_hint = args[0] if origin in (list, tuple) and args else Any
        items = ", ".join(
            _render(v, item_hint, imports, _at(actual, i), f"{path}[{i}]")
            for i, v in enumerate(value)
        )
        return f"({items},)" if isinstance(value, tuple) and value else f"[{items}]"
    return repr(value)


def _field_hints(cls: type) -> dict[str, Any]:
    try:
        return typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001 - an unresolvable hint renders as Any
        return {}


def _render_dataclass(
    cls: type,
    values: dict[str, Any],
    imports: _Imports,
    actual: Any = _UNKNOWN,
    path: str = "",
) -> str:
    hints = _field_hints(cls)
    if not path:
        args = ", ".join(
            f"{key}={_render(value, hints.get(key, Any), imports, _at(actual, key))}"
            for key, value in values.items()
        )
        return f"{imports.name(cls)}({args})"
    # One field to a line, so that Codon's error names the field (#531).
    pieces = []
    for key, value in values.items():
        hint = hints.get(key, Any)
        rendered = _render(value, hint, imports, _at(actual, key), f"{path}.{key}")
        if hint in (bool, int, float, str) and (
            type(value) is hint or type(value) in (int, float, str)
        ):
            imports.checks.append(
                (f"_acs_as_{hint.__name__}({rendered})", f"{path}.{key}")
            )
        pieces.append(f"\n{key}={rendered},{_PATH_MARK}{path}.{key}")
    fields = "".join(pieces)
    return f"{imports.name(cls)}({_PATH_MARK}{path}{fields}\n)"


def _split_marks(text: str, path: str) -> list[tuple[str, str]]:
    """The lines of rendered *text*, each with the config path it holds."""
    out: list[tuple[str, str]] = []
    for line in text.split("\n"):
        code, mark, where = line.partition(_PATH_MARK)
        out.append((code, where if mark else path))
    return out


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
    build_error = ""
    try:
        # What the runner builds, to render each value as the config holds it.
        built: Any = config_cls(**copy.deepcopy(scenario_dict))
    except Exception as exc:  # noqa: BLE001 - reported with the verdict (#531)
        built = _UNKNOWN
        build_error = f"{type(exc).__name__}: {exc}"

    by_assignment = _all_fields_have_defaults(config_cls)
    # The same program with each value a token of its type: the key.
    shape_imports = _Imports(shape=True)
    config_args: list[tuple[str, str, str]] = []  # (key, line, shaped line)
    for key, value in scenario_dict.items():
        line, shaped = (
            f"    config.{key} = {rendered}"
            if by_assignment
            else f"        {key}={rendered},"
            for rendered in (
                _render(
                    value,
                    hints.get(key, Any),
                    imports_,
                    _at(built, key),
                    f"scenario.{key}",
                )
                for imports_ in (imports, shape_imports)
            )
        )
        config_args.append((key, line, shaped))
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
        "    scenario.is_done()",
        "",
        "",
        # The hooks' results, checked strictly (#238): `done: bool =
        # scenario.is_done()` passed `is_done() -> str`, as Codon converts
        # any value with __bool__ in an annotated assignment.  Called in the
        # trailer, after the module checks, which report an override's
        # result at its own `def` line first.
        f"def {HOOK_CHECK}():",
        f"    scenario = _acs_value({scenario_name})",
        *(
            f'    BaseScenario._acs_hook_returns("{hook}", scenario.{hook}())'
            for hook in ("setup", "is_done")
        ),
        "",
        "",
        "_acs_check_scenario()",
    ]
    head = [
        "import typesafe_carla.carla as carla",
        "from autoware_carla_scenario import (BaseScenario, EgoConfig, GroundProjectionConfig,",
        "                                     Lanelet2Pose, SpawnTransform)",
        "from autoware_carla_scenario._tc import _acs_value",
        *imports.lines,
        "",
        "",
        *(
            f"def _acs_as_{name}(value: {name}): pass"
            for name in ("bool", "int", "float", "str")
            if any(c.startswith(f"_acs_as_{name}(") for c, _ in imports.checks)
        ),
    ]
    # Rendering a value imports the same classes (its enum, its dataclass)
    # whatever the value is, so the shape keeps the program's imports.
    config_lines: dict[int, str] = {}
    opening = body.index("def _acs_check_scenario():") + 1
    lines = head + body[:opening]
    shape = head + body[:opening]
    for code, where in imports.checks:
        lines.append(f"    {code}")
        config_lines[len(lines)] = where
    shape.extend(f"    {code}" for code, _ in shape_imports.checks)
    lines += body[opening:]
    shape += body[opening:]
    for key, line, shaped in config_args:
        parts = _split_marks(line, f"scenario.{key}")
        for code, where in parts:
            lines.append(code)
            config_lines[len(lines)] = where
        shape.extend(code for code, _ in _split_marks(shaped, ""))
    lines += tail
    shape += tail
    return Driver(
        "\n".join(lines) + "\n",
        config_lines,
        trailer=f"{HOOK_CHECK}()\n",
        shape="\n".join(shape) + "\n",
        case={key: _abbreviated(repr(value)) for key, value in scenario_dict.items()},
        build_error=build_error,
    )


def _abbreviated(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_module_checks(modules: list[str]) -> str:
    """The lines that run each of *modules*' appended calls, after the driver's.

    Each of the scenario's own modules ends with the library check's
    function (:func:`.library_driver.render_library_checks`), which calls
    every function and method of the module, so that Codon compiles what
    the driver's own calls do not reach.  These lines go after the driver's
    program, so no line of it moves (:attr:`Driver.config_lines`).
    """
    from .library_driver import LIBRARY_CHECK  # noqa: PLC0415 - imports this module

    lines = ["", ""]
    for i, module in enumerate(modules):
        lines.append(f"from {module} import {LIBRARY_CHECK} as _acs_module_check_{i}")
        lines.append(f"_acs_module_check_{i}()")
    return "\n".join(lines) + "\n"


def render_odd_driver(module: str, function: str) -> Driver:
    """The program that checks an ODD: call its builder, keep an ODD."""
    lines = [
        "from autoware_carla_scenario import OddDefinition",
        f"from {module} import {function}",
        "",
        "",
        "def _acs_check_odd():",
        f"    odd: OddDefinition = {function}()",
        "",
        "",
        "_acs_check_odd()",
    ]
    return Driver("\n".join(lines) + "\n", reaches="the ODD's builder")
