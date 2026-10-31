"""What docs/typecheck.md says of the library check, generated (#789).

"What is checked of the framework" gives how many modules are ``CHECKED``,
``EXCLUDED`` and ``UNCALLED`` and lists them.  Written by hand, every pull
request that moved a module into ``CHECKED`` edited the same sentences, and
two open at once conflicted on them.  The page holds placeholders instead,
``{{ library.checked_modules }}`` and the like, which the docs build fills in
from :mod:`.library` (``mkdocs_hooks/typecheck_library.py``) with
:func:`render_library_docs`.  A module moved into ``CHECKED`` is then one
line of :mod:`.library` and nothing on the page: the list shows its
docstring's first paragraph.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import Path

from .library import CHECKED, EXCLUDED, UNCALLED, package_modules

__all__ = ["PLACEHOLDER", "library_docs", "module_summary", "render_library_docs"]

#: A placeholder: ``{{ library.<name> }}``.  The ``library.`` prefix keeps it
#: apart from the ``{{`` of the code the page quotes (``f"{{x}}"``).
PLACEHOLDER = re.compile(r"\{\{\s*library\.(\w+)\s*\}\}")

_PACKAGE = "autoware_carla_scenario"

#: A Sphinx cross-reference, ``:class:`~.model.Trajectory```.
_ROLE = re.compile(r":\w+:`([^`]+)`")


def _short(name: str) -> str:
    """``autoware_carla_scenario.coordinate.frames`` -> ``coordinate.frames``."""
    return name.removeprefix(f"{_PACKAGE}.")


def _hundreds(lines: int) -> str:
    """``69308`` -> ``"69,300"``: the page says "about" so many lines."""
    return f"{round(lines, -2):,}"


def _is_protected(name: str) -> bool:
    function = name.rsplit(".", 1)[-1]
    return function.startswith("_") and not function.startswith("__")


def _role(match: re.Match[str]) -> str:
    target = match.group(1)
    if "<" in target:  # :mod:`text <target>`
        return f"`{target.split('<', 1)[0].strip()}`"
    if target.startswith("~"):
        target = target[1:].rsplit(".", 1)[-1]
    return f"`{target.lstrip('.')}`"


def module_summary(name: str, path: Path) -> str:
    """The first paragraph of a module's docstring, as one line of Markdown.

    reST's ````code```` and cross-references become code spans, and a
    leading ``name -- `` (the module's own name) is dropped.  An empty
    string where the module has no docstring.
    """
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    if not docstring:
        return ""
    text = " ".join(docstring.strip().split("\n\n", 1)[0].split())
    text = _ROLE.sub(_role, text.replace("``", "`"))
    own = re.escape(name.rsplit(".", 1)[-1])
    return re.sub(rf"^{own}\s+(?:-{{1,2}}|–)\s+", "", text)


#: A code span, or a run of text between two.
_CODE_SPAN = re.compile(r"(`[^`]*`)")


def _plain(text: str) -> str:
    """*text* with ``_``, ``*`` and ``<`` outside its code spans escaped.

    The reasons and docstrings are prose with code spans, but name
    ``__init__`` or ``cls.__dict__`` bare, which Markdown would print as
    bold ``init``.
    """
    parts = _CODE_SPAN.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"([_*])", r"\\\1", parts[i]).replace("<", "&lt;")
    return "".join(parts)


def _collapsible(title: str, lines: list[str]) -> str:
    body = "\n".join(f"    {line}" if line else "" for line in lines)
    return f'??? note "{title}"\n\n{body}\n'


def _checked_table(modules: Mapping[str, Path]) -> str:
    rows = ["| Module | What it is |", "| --- | --- |"]
    for name in sorted(CHECKED):
        summary = _plain(module_summary(name, modules[name])).replace("|", "\\|")
        rows.append(f"| `{_short(name)}` | {summary or '-'} |")
    return _collapsible(f"The {len(CHECKED)} CHECKED modules", rows)


def _reasons(title: str, entries: Mapping[str, str]) -> str:
    items = [f"- `{_short(name)}`: {_plain(entries[name])}" for name in sorted(entries)]
    return _collapsible(title, items)


def library_docs() -> dict[str, str]:
    """Every placeholder's name, with what the page prints for it.

    A module's lines are its source file's physical lines; the package is
    every module :func:`.library.package_modules` lists (all but
    ``typecheck/``).
    """
    modules = package_modules()
    lines = {
        name: len(path.read_text(encoding="utf-8").splitlines())
        for name, path in modules.items()
    }
    return {
        "checked_modules": str(len(CHECKED)),
        "package_modules": str(len(modules)),
        "checked_percent": str(round(100 * len(CHECKED) / len(modules))),
        "checked_lines": _hundreds(sum(lines[name] for name in CHECKED)),
        "package_lines": _hundreds(sum(lines.values())),
        "uncalled": str(len(UNCALLED)),
        "uncalled_protected": str(sum(1 for name in UNCALLED if _is_protected(name))),
        "excluded": str(len(EXCLUDED)),
        "checked_table": _checked_table(modules),
        "uncalled_list": _reasons(
            f"The {len(UNCALLED)} UNCALLED functions and methods, and why", UNCALLED
        ),
        "excluded_list": _reasons(
            f"The {len(EXCLUDED)} EXCLUDED modules, and why", EXCLUDED
        ),
    }


def render_library_docs(markdown: str, values: Mapping[str, str]) -> str:
    """``markdown`` with every placeholder replaced by its value.

    Raises:
        ValueError: a placeholder names no value, so the build fails rather
            than print it.
    """
    unknown = sorted({m.group(1) for m in PLACEHOLDER.finditer(markdown)} - set(values))
    if unknown:
        raise ValueError(
            f"unknown placeholder(s) {', '.join(f'library.{n}' for n in unknown)}; "
            f"known: {', '.join(sorted(values))}"
        )
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], markdown)
