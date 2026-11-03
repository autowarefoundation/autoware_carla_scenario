"""The docs' code examples call the framework with arguments it accepts (#1245).

Every Python code block of ``docs/**/*.md`` is parsed, the names it takes
from ``autoware_carla_scenario`` are resolved (its imports, and the package's
public names it uses unimported), and each call of one of them (``Name(...)``,
or ``Name.attr(...)`` for a class- or static method) must bind
against ``inspect.signature``: no required argument missing (a keyword-only
``label``, say), no unknown keyword, no positional too many. A copied example
that fails to bind raises a ``TypeError`` before it runs.

The check is conservative. It skips a call it cannot judge: a name it cannot
resolve to the package, a ``*args``/``**kwargs`` splat, or
``Name(...)``, whose ``...`` elides every argument. A ``...`` among other
arguments stands for one value and is bound like any. Calls that leave a
required argument out on purpose are listed in ``_PLACEHOLDERS``.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import re
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_CODE_BLOCK = re.compile(r"^```py(?:thon)?\n(.*?)^```", re.S | re.M)

# (doc, call as ``ast.unparse`` writes it): calls that are shorthand, not code
# to copy. Each entry must still be found, so a stale one fails.
_PLACEHOLDERS: dict[tuple[str, str], str] = {
    (
        "architecture.md",
        "EntityLanePositionCondition.anywhere_on_road('ego', '5', ...)",
    ): "sketches composition; the elided arguments include the label",
    (
        "architecture.md",
        "EntityLanePositionCondition.anywhere_on_road('ego', '12', ...)",
    ): "sketches composition; the elided arguments include the label",
    (
        "architecture.md",
        "EntityLanePositionCondition.anywhere_on_road('ego', '8', ...)",
    ): "sketches composition; the elided arguments include the label",
}


def _code_blocks() -> Iterator[tuple[str, int, ast.Module]]:
    """(doc, line of the block, parsed block) for every Python block."""
    for doc in sorted(_DOCS.rglob("*.md")):
        text = doc.read_text(encoding="utf-8")
        for match in _CODE_BLOCK.finditer(text):
            try:
                tree = ast.parse(textwrap.dedent(match.group(1)))
            except SyntaxError:
                continue  # a fragment, not a program
            line = text.count("\n", 0, match.start()) + 2
            yield str(doc.relative_to(_DOCS)), line, tree


def _package_names(tree: ast.Module) -> dict[str, object]:
    """The names a block takes from ``autoware_carla_scenario``.

    What its ``from autoware_carla_scenario... import`` binds, over the
    package's public names, which a snippet often uses without importing;
    a name the block binds itself is not the package's.
    """
    package = importlib.import_module("autoware_carla_scenario")
    names: dict[str, object] = {name: package for name in package.__all__}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.pop(node.name, None)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.pop(node.id, None)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.pop(alias.asname or alias.name.split(".")[0], None)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "autoware_carla_scenario"
        ):
            continue
        try:
            module = importlib.import_module(node.module)
        except ImportError:
            continue
        for alias in node.names:
            if hasattr(module, alias.name):
                names[alias.asname or alias.name] = getattr(module, alias.name)
    # The package's names resolve lazily: only those the block calls load.
    return {
        name: getattr(value, name) if value is package else value
        for name, value in names.items()
        if value is not package or _called(tree, name)
    }


def _called(tree: ast.Module, name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == name
            )
        )
        for node in ast.walk(tree)
    )


def _callee(call: ast.Call, names: dict[str, object]) -> Callable[..., object] | None:
    """The function a call reaches, or None when the block does not say."""
    func = call.func
    if isinstance(func, ast.Name):
        target = names.get(func.id)
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        owner = names.get(func.value.id)
        if owner is None or not hasattr(owner, func.attr):
            return None
        if inspect.isclass(owner) and inspect.isfunction(
            inspect.getattr_static(owner, func.attr)
        ):
            return None  # an instance method called through its class
        target = getattr(owner, func.attr)
    else:
        return None
    return target if callable(target) else None


def _is_ellipsis(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


@functools.cache
def _binding_errors() -> list[tuple[str, str, str]]:
    """(doc, call, where and why) for each call that does not bind."""
    return list(_find_binding_errors())


def _find_binding_errors() -> Iterator[tuple[str, str, str]]:
    for doc, line, tree in _code_blocks():
        names = _package_names(tree)
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            keywords = {k.arg: k.value for k in call.keywords if k.arg is not None}
            if any(isinstance(a, ast.Starred) for a in call.args) or len(
                keywords
            ) != len(call.keywords):
                continue  # a *args or **kwargs splat
            if len(call.args) == 1 and not call.keywords and _is_ellipsis(call.args[0]):
                continue
            callee = _callee(call, names)
            if callee is None:
                continue
            try:
                signature = inspect.signature(callee)
            except (TypeError, ValueError):
                continue
            try:
                signature.bind(*call.args, **keywords)
            except TypeError as error:
                where = f"docs/{doc}:{line + call.lineno - 1}"
                yield doc, ast.unparse(call), f"{where}: {error}"


def test_doc_examples_bind_their_calls() -> None:
    errors = _binding_errors()
    failures = [
        f"{why}\n    {source}"
        for doc, source, why in errors
        if (doc, source) not in _PLACEHOLDERS
    ]
    assert not failures, (
        "These docs examples call the framework with arguments it does not "
        "accept (add the argument, or list a deliberate placeholder in "
        "_PLACEHOLDERS):\n" + "\n".join(failures)
    )


@pytest.mark.parametrize(("doc", "source"), sorted(_PLACEHOLDERS))
def test_placeholder_is_still_in_the_docs(doc: str, source: str) -> None:
    found = {(d, s) for d, s, _ in _binding_errors()}
    assert (
        (doc, source) in found
    ), f"{source} no longer fails to bind in docs/{doc}; drop it from _PLACEHOLDERS"
