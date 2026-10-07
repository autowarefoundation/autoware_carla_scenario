"""The rewrites a scenario module needs to *run* under Codon, beyond the check's.

:func:`~..typecheck.transform.transform_source` makes a module compile; a
module that compiles can still miscompile. :func:`transform_for_runtime` adds
the rewrites for what Codon 0.19 compiles but runs wrongly, each line for line
(so a build error still points at the line the author wrote):

* ``a or b`` becomes ``_acs_or(a, lambda: b)``. Codon types ``a or b`` as a
  Union of the two operand types, and an ``Optional[T]`` on the left -- the
  ``config or MyConfig()`` every scenario's ``__init__`` writes -- yields a
  Union it then fails to read a ``T`` back from, at run time ("invalid union
  getter"), whether or not ``a`` is ``None``. ``_acs_or`` (``_lists.codon``)
  returns ``a`` when it is truthy and calls the thunk otherwise, so ``b`` is
  still evaluated only when Python would evaluate it.
"""

from __future__ import annotations

import ast

from ..typecheck.transform import PRELUDE, _Edits, transform_source

__all__ = ["RUNTIME_PRELUDE", "rewrite_or", "transform_for_runtime"]

#: :data:`..typecheck.transform.PRELUDE`, also importing ``_acs_or``; one line,
#: so a line reported by Codon maps back the same way.
RUNTIME_PRELUDE = PRELUDE.rstrip("\n") + ", _acs_or\n"


def _is_or(node: ast.AST) -> bool:
    return isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)


def _innermost_ors(tree: ast.AST) -> list[ast.BoolOp]:
    """The ``or`` expressions with no other ``or`` inside, outside f-strings."""
    found: list[ast.BoolOp] = []

    def visit(node: ast.AST) -> bool:
        """Visit *node*; whether it contains an ``or``."""
        if isinstance(node, ast.JoinedStr):
            return False  # f-string internals: left alone
        contains = False
        for child in ast.iter_child_nodes(node):
            contains = visit(child) or contains
        if _is_or(node):
            if not contains:
                found.append(node)  # type: ignore[arg-type]
            return True
        return contains

    visit(tree)
    return found


def rewrite_or(source: str) -> str:
    """Every ``a or b [or c ...]`` in *source*, as nested ``_acs_or`` calls."""
    while True:
        tree = ast.parse(source)
        targets = _innermost_ors(tree)
        if not targets:
            return source
        edits = _Edits(source.encode().splitlines(keepends=True))
        for node in targets:
            parts = [edits.text(*edits.node_span(v)) for v in node.values]
            text = parts[-1]
            for part in reversed(parts[:-1]):
                text = f"_acs_or({part}, lambda: {text})"
            edits.replace(*edits.node_span(node), f"({text})")
        source = edits.apply()


def transform_for_runtime(source: str) -> str:
    """:func:`transform_source`, plus the runtime rewrites; same line numbers."""
    checked = transform_source(source)
    if not checked.startswith(PRELUDE):
        raise AssertionError("transform_source() no longer starts with PRELUDE")
    return RUNTIME_PRELUDE + rewrite_or(checked[len(PRELUDE) :])
