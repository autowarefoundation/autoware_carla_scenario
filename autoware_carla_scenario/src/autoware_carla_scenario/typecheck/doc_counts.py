"""The counts docs/typecheck.md gives of the library check (#789).

"What is checked of the framework" says how many modules are ``CHECKED``,
``EXCLUDED`` and ``UNCALLED``.  Written by hand, every pull request that moved
a module into ``CHECKED`` edited the same sentence, and two open at once
conflicted on it.  The page holds placeholders instead,
``{{ counts.checked_modules }}`` and the like, which the docs build fills in
from :mod:`.library` (``mkdocs_hooks/typecheck_counts.py``) with
:func:`render_counts`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .library import CHECKED, EXCLUDED, UNCALLED, package_modules

__all__ = ["PLACEHOLDER", "doc_counts", "render_counts"]

#: A placeholder: ``{{ counts.<name> }}``.  The ``counts.`` prefix keeps it
#: apart from the ``{{`` of the code the page quotes (``f"{{x}}"``).
PLACEHOLDER = re.compile(r"\{\{\s*counts\.(\w+)\s*\}\}")


def _hundreds(lines: int) -> str:
    """``69308`` -> ``"69,300"``: the page says "about" so many lines."""
    return f"{round(lines, -2):,}"


def _is_protected(name: str) -> bool:
    function = name.rsplit(".", 1)[-1]
    return function.startswith("_") and not function.startswith("__")


def doc_counts() -> dict[str, str]:
    """Every placeholder's name, with its value as the page prints it.

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
    }


def render_counts(markdown: str, counts: Mapping[str, str]) -> str:
    """``markdown`` with every placeholder replaced by its count.

    Raises:
        ValueError: a placeholder names no count, so the build fails rather
            than print it.
    """
    unknown = sorted({m.group(1) for m in PLACEHOLDER.finditer(markdown)} - set(counts))
    if unknown:
        raise ValueError(
            f"unknown placeholder(s) {', '.join(f'counts.{n}' for n in unknown)}; "
            f"known: {', '.join(sorted(counts))}"
        )
    return PLACEHOLDER.sub(lambda m: counts[m.group(1)], markdown)
