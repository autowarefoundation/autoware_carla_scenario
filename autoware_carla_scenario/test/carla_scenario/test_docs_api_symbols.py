"""``docs/api.md``'s tables name symbols where they import from (#2255).

A section whose header names a module (``## Conditions
(`autoware_carla_scenario.conditions`)``) lists symbols of that module.  Every
code span in the first cell of a table row that looks like a Python name
(``Name``, ``Name(args)``, ``module.Name``) must import from the header's
module, or from a module the row itself names: a full
``autoware_carla_scenario.x`` path anywhere in the row, a span that is a
module relative to the package (``entity.bridged_ego``), the module part of a
dotted span (``route.frame`` of ``route.frame.RouteFrame``), or a
``module:`` prefix (```utils.stop_line`: `get_stop_line_linestrings` ``),
which alone is then searched.  A top-level re-export does not count: the
header claims the symbol for its module.

Where a span spells a signature (``export_cod(doc, out_dir, stem)``), each
parameter it names must be one of the callable's, unless the callable takes
``**kwargs``: in the first cell, and in the description too
(``NotCondition(condition, holds_when_pending=True)``) when the name resolves
in the row's modules or the top-level package -- there an unresolved name is
prose, not a symbol.  The
check is conservative: a span that is not a plain name (an expression, a CLI
flag, prose) is skipped, and so is an argument that is not a plain name.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

_PACKAGE = "autoware_carla_scenario"
_API_DOC = Path(__file__).resolve().parents[2] / "docs" / "api.md"
_HEADER = re.compile(rf"^#{{2,3}} .*\(`({_PACKAGE}(?:\.\w+)*)`\)\s*$")
_SPAN = re.compile(r"`([^`]+)`")
_SYMBOL = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\((.*)\))?$")
_FULL_MODULE = re.compile(rf"\b({_PACKAGE}(?:\.\w+)*)")
_DOTTED = re.compile(r"^\w+(?:\.\w+)*$")
_MISSING = object()  # a name that is absent (a constant may be None)


@functools.lru_cache(maxsize=None)
def _module(name: str) -> Optional[ModuleType]:
    """The module ``name``, or None when there is none.

    Only a missing ``name`` (or a missing parent package of it) means "no
    module": a module that exists but fails to import, say for want of a
    third-party dependency, raises, rather than reading as a symbol the doc
    misplaced.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name and (name == error.name or name.startswith(f"{error.name}.")):
            return None
        raise


def _as_module(name: str) -> Optional[ModuleType]:
    """``name`` as a full or package-relative module path."""
    if name.startswith(_PACKAGE):
        return _module(name)
    return _module(f"{_PACKAGE}.{name}")


def _resolve(base: ModuleType, dotted: str) -> Any:
    """``dotted`` looked up on ``base``, through submodules; _MISSING if absent."""
    obj: Any = base
    for part in dotted.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif inspect.ismodule(obj) and _module(f"{obj.__name__}.{part}"):
            obj = _module(f"{obj.__name__}.{part}")
        else:
            return _MISSING
    return obj


def _module_prefix(dotted: str) -> Optional[ModuleType]:
    """The longest leading part of ``dotted`` that is a package module."""
    parts = dotted.split(".")
    for end in range(len(parts) - 1, 0, -1):
        module = _as_module(".".join(parts[:end]))
        if module is not None:
            return module
    return None


def _rows() -> list[tuple[int, str, str]]:
    """(line, header module, row) for every table row under a module header."""
    rows = []
    section: Optional[str] = None
    lines = _API_DOC.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if line.startswith(("## ", "### ")):
            match = _HEADER.match(line)
            section = match[1] if match else None
        elif section and line.startswith("| ") and not line.startswith("| Symbol"):
            rows.append((number, section, line))
    assert rows, "docs/api.md has no tables under module headers"
    return rows


def _parameter_problems(
    number: int, span: str, name: str, found: Any, arguments: str
) -> list[str]:
    """The parameters the signature span ``span`` names that ``found`` lacks."""
    if not callable(found) or inspect.ismodule(found):
        return []
    try:
        parameters = inspect.signature(found).parameters
    except (TypeError, ValueError):
        return []
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return []
    problems = []
    for argument in arguments.split(","):
        parameter = argument.split("=", 1)[0].strip().lstrip("*")
        if re.fullmatch(r"[A-Za-z_]\w*", parameter) and parameter not in parameters:
            problems.append(
                f"line {number}: `{span}` names `{parameter}`, which"
                f" {name} does not take ({list(parameters)})"
            )
    return problems


def _problems() -> list[str]:
    problems = []
    for number, section, row in _rows():
        first, _, description = row[2:].partition(" | ")
        spans = _SPAN.findall(first)
        modules = [section, *_FULL_MODULE.findall(row)]
        modules += [span for span in _SPAN.findall(row) if _DOTTED.match(span)]
        candidates = [m for m in map(_as_module, modules) if m is not None]
        candidates += [
            prefix
            for span in spans
            if _DOTTED.match(span) and (prefix := _module_prefix(span)) is not None
        ]
        prefixed: Optional[list[ModuleType]] = None
        for span in spans:
            if f"`{span}`:" in first and _as_module(span) is not None:
                prefixed = [_as_module(span)]  # type: ignore[list-item]
                continue
            match = _SYMBOL.match(span)
            if not match or _as_module(match[1]) is not None:
                continue
            found: Any = _MISSING
            for base in prefixed or candidates:
                found = _resolve(base, match[1])
                if found is not _MISSING:
                    break
            if found is _MISSING and "." in match[1] and prefixed is None:
                found = _resolve(_module(_PACKAGE), match[1])  # type: ignore[arg-type]
                if inspect.ismodule(found):
                    found = _MISSING
            if found is _MISSING:
                where = [m.__name__ for m in prefixed or candidates]
                problems.append(
                    f"line {number}: `{span}` is not in {where}; name the module"
                    " it lives in on the row (a full path, or a `module:`"
                    " prefix), or move it out of the first cell"
                )
                continue
            if match[2] is not None:
                problems += _parameter_problems(number, span, match[1], found, match[2])
        for span in _SPAN.findall(description):
            match = _SYMBOL.match(span)
            if not match or match[2] is None:
                continue
            for base in [*candidates, importlib.import_module(_PACKAGE)]:
                found = _resolve(base, match[1])
                if found is not _MISSING:
                    problems += _parameter_problems(
                        number, span, match[1], found, match[2]
                    )
                    break
    return problems


def test_api_table_symbols_import_from_their_section() -> None:
    problems = _problems()
    assert not problems, "docs/api.md:\n" + "\n".join(problems)
