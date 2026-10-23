"""Hints for two modules of a scenario package that import each other (#507).

Codon resolves an import cycle only with the names already defined, and a
class-body annotation (``peer: Optional["B"]``) is evaluated when the class
is defined, so a class cannot hold an attribute typed by a class of a module
that imports its own.  Codon refuses even that within one module, for a class
defined further down.  Nothing can rewrite it (the check requires the
declaration), so the check names the pattern that compiles.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Collection
from pathlib import Path

_CANNOT_IMPORT = re.compile(r"^cannot import name '(\w+)' from '([\w.]+)'$")
_UNNAMED = re.compile(r"^(expected type expression|name '(\w+)' is not defined)$")

_PATTERN = (
    "a module cycle cannot hold a class attribute typed by the other "
    "module's class: Codon 0.19 resolves a cycle only with the names already "
    "defined, and an attribute declaration is read when the class is defined. "
    "Declare the attribute in one module only; in the other, name the class "
    "in signatures only (the check drops those: keep the `if TYPE_CHECKING:` "
    "import at the end of that module, and import that module first) or "
    "import it inside the function that uses it; or put both classes in one "
    "module"
)


def _in_type_checking(tree: ast.Module, line: int) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and "TYPE_CHECKING" in ast.unparse(node.test)
            and node.lineno < line <= (node.end_lineno or node.lineno)
        ):
            return True
    return False


def _imported(tree: ast.Module) -> set[str]:
    """The names the module imports at module level (also under an ``if``)."""
    names: set[str] = set()
    for stmt in tree.body:
        for node in ast.walk(stmt) if isinstance(stmt, ast.If) else [stmt]:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def cycle_hint(
    message: str, path: str | None, line: int | None, modules: Collection[str]
) -> str | None:
    """What to do about a refusal that comes from a module cycle, or ``None``.

    *modules* are the scenario package's own modules (dotted names).
    """
    if path is None or line is None:
        return None
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    cannot = _CANNOT_IMPORT.match(message)
    if cannot is not None:
        if cannot[2] not in modules:
            return None
        where = (
            "the check reads TYPE_CHECKING as true, so this import made for "
            "annotations is a cycle: "
            if _in_type_checking(tree, line)
            else "the two modules import each other: "
        )
        return where + _PATTERN
    if _UNNAMED.match(message) is None:
        return None
    imported = _imported(tree)
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for stmt in cls.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and stmt.lineno == line
                and imported & set(re.findall(r"[A-Za-z_]\w*", ast.unparse(stmt)))
            ):
                return _PATTERN
    return None
