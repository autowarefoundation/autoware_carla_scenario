"""Valid Python that Codon 0.19 refuses or crashes on, rewritten (#376).

A pass of its own, run by :func:`.transform.transform_source_lines` before the
other rewrites.  Each rewrite is on the same lines as what it replaces, and
only the checked copy of a module changes: what runs is the code as written,
so a rewrite has to keep the *types*, not the values, of what it replaces.

* ``class X: ...`` becomes ``class X: pass`` (an ``...`` statement of a class
  body is "unexpected expression in class definition" to Codon).
* A dict display with ``**`` (``{**a, "k": v, **b}``, which crashes Codon
  with no diagnostic) becomes a ``|`` of its parts (``(a | {"k": v} | b)``);
  ``{**a}`` alone, a copy, becomes ``dict(a)``.
* ``"...".format(...)`` on a string literal becomes an f-string: auto-numbered,
  indexed and named fields, attribute and index lookups, ``!r`` / ``!s`` /
  ``!a`` and format specs.  Arguments that are plain names, constants or
  attributes are written into the f-string; any other becomes the argument of
  an immediately called lambda, so it is evaluated once.  A literal brace is a
  separate string joined with ``+`` (Codon's f-strings have no ``{{``).  A
  template the rewrite cannot express (``*args`` / ``**kwargs``, a field
  inside a format spec, a missing argument) is refused.
* ``zip(..., strict=...)`` becomes ``zip(...)``.
* A top-level class that defines ``__lt__`` (what ``sorted()``, ``.sort()``
  and ``heapq`` need in Python) is given the ``__gt__`` and ``__ge__`` Codon's
  timsort calls, with ``@extend``; ``@functools.total_ordering`` is removed
  with its import (Codon has neither) and the class given every comparison
  ``total_ordering`` would add, from whichever one it defines.
* ``lambda x, k=k: ...``, a default that is not a literal (Codon looks the name
  up in the lambda: "name '%_k_930' is not defined"), becomes
  ``(lambda k: lambda x: ...)(k)``: the default is bound by a call, as Python
  binds it once.  The captured parameter can then no longer be passed.
"""

from __future__ import annotations

import ast
import re
import string

from .transform import _dataclass_options, _defined_methods, _dotted, _Edits

__all__ = ["REWRITE_HINTS", "python_rewrites", "python_rewrite_refusals"]

#: Codon's refusals of what this pass leaves alone, and what to write.
REWRITE_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^'(?:str|'.*')' object has no attribute 'format'"),
        "Codon's str has no .format(): the check rewrites it only on a string "
        "literal with plain arguments; write an f-string instead",
    ),
    (
        # A lambda rewritten for a non-literal default no longer takes it.
        re.compile(
            r"%_lambda_\d+\(\) takes \d+ arguments|"
            r"is an invalid keyword argument for %_lambda_\d+\(\)"
        ),
        "if this lambda has a default that is not a literal (`k=k`), the check "
        "binds that default once and the lambda no longer takes it: do not "
        "pass it, or use functools.partial",
    ),
    (
        re.compile(r"unsupported operand type\(s\) for (?:<|>=): '(\w+)' and '\1'"),
        "this class has no ordering, which sorted(), .sort() and min() need: "
        "define __lt__ (the check adds the __gt__ and __ge__ Codon's sort "
        "calls), use @dataclass(order=True), or sort with key=",
    ),
)

_ORDERING = ("__lt__", "__le__", "__gt__", "__ge__")
_MAX_PASSES = 20
_FORMAT_REFERENCE = re.compile(r"(?:\.\w+|\[[^\]\[]+\])*")
_ACCESS = re.compile(r"\.(\w+)|\[([^\]\[]+)\]")


def _literal(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp):
        node = node.operand
    return isinstance(node, ast.Constant)


def _plain(node: ast.AST) -> bool:
    """A name, a constant or an attribute chain: it can be written twice."""
    if isinstance(node, ast.Constant):
        return not isinstance(node.value, (str, bytes))
    if isinstance(node, ast.Attribute):
        return _plain(node.value) and not isinstance(node.value, ast.Constant)
    return isinstance(node, ast.Name)


def _escaped(text: str) -> str:
    out = []
    for char in text:
        if char in '\\"':
            out.append("\\" + char)
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif ord(char) < 32 or ord(char) == 127:
            out.append(f"\\x{ord(char):02x}")
        else:
            out.append(char)
    return "".join(out)


class _Pass(ast.NodeVisitor):
    def __init__(self, edits: _Edits, total_ordering: set[str]) -> None:
        self.edits = edits
        self.total_ordering = total_ordering
        self.refused: list[tuple[int, str]] = []
        self._depth = 0

    def _text(self, node: ast.AST) -> str:
        return self.edits.text(*self.edits.node_span(node))

    def _refuse(self, node: ast.AST, message: str) -> None:
        self.refused.append((node.lineno, message))  # type: ignore[attr-defined]

    # -- class X: ... ------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            if _dotted(decorator) in self.total_ordering:
                if self._depth:
                    self._refuse(
                        decorator,
                        "@total_ordering on a class that is not at module level "
                        "is not supported by the check: define the class at "
                        "module level, or write the comparisons",
                    )
                else:
                    start, end = self.edits.node_span(decorator)
                    self.edits.replace(start - 1, end, "")  # with its "@"
            else:
                self.visit(decorator)
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is Ellipsis
            ):
                self.edits.replace(*self.edits.node_span(stmt), "pass")
        for base in (*node.bases, *node.keywords):
            self.visit(base)
        self._depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._depth -= 1

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_FunctionDef = _function  # noqa: N815
    visit_AsyncFunctionDef = _function  # noqa: N815

    # -- from functools import total_ordering -----------------------------

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "functools" or node.level:
            return
        kept = [a for a in node.names if a.name != "total_ordering"]
        if len(kept) == len(node.names):
            return
        text = "pass"
        if kept:
            text = ast.unparse(ast.ImportFrom(module="functools", names=kept, level=0))
        self.edits.replace(*self.edits.node_span(node), text)

    # -- {**a, **b} ---------------------------------------------------------

    def visit_Dict(self, node: ast.Dict) -> None:
        if all(key is not None for key in node.keys):
            self.generic_visit(node)
            return
        parts: list[str] = []
        shown: list[str] = []
        for key, value in zip(node.keys, node.values):
            if key is None:
                if shown:
                    parts.append("{" + ", ".join(shown) + "}")
                    shown = []
                text = self._text(value)
                simple = isinstance(
                    value, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)
                )
                parts.append(text if simple else f"({text})")
            else:
                shown.append(f"{self._text(key)}: {self._text(value)}")
        if shown:
            parts.append("{" + ", ".join(shown) + "}")
        if len(parts) == 1:
            text = f"dict({parts[0]})"
        else:
            text = "(" + " | ".join(parts) + ")"
        self.edits.replace(*self.edits.node_span(node), text)

    # -- zip(strict=), "...".format(...) ------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "zip"
            and any(k.arg == "strict" for k in node.keywords)
        ):
            # Remove the keyword with the comma before it, in place, so a call
            # spread over lines keeps its line breaks inside its brackets.
            for keyword in [k for k in node.keywords if k.arg == "strict"]:
                start, end = self.edits.node_span(keyword)
                before = [
                    self.edits.node_span(other)[1]
                    for other in (*node.args, *node.keywords)
                    if self.edits.node_span(other)[1] <= start
                ]
                if before:
                    start = max(before)
                else:
                    tail = re.match(rb"\s*,?\s*", self.edits.data[end:])
                    assert tail is not None
                    end += tail.end()
                self.edits.replace(start, end, "")
            return
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
        ):
            text = self._format(node, node.func.value.value)
            if text is not None:
                self.edits.replace(*self.edits.node_span(node), text)
                return
        self.generic_visit(node)

    def _format(self, node: ast.Call, template: str) -> str | None:
        """The f-string for ``template.format(...)``, or ``None`` if refused."""
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            self._refuse(
                node,
                "str.format with *args or **kwargs cannot be checked: write an "
                "f-string, or list the arguments",
            )
            return None
        positional = list(node.args)
        named = {k.arg: k.value for k in node.keywords if k.arg}
        params: dict[str, ast.expr] = {}
        for index, value in enumerate(positional):
            params[f"_acs_f{index}"] = value
        for name, value in named.items():
            params[f"_acs_k_{name}"] = value
        inline = all(_plain(v) for v in params.values())
        parts: list[str] = []  # plain strings and f-string chunks, to join
        chunk = ""  # the f-string being built
        auto = 0
        numbering: str | None = None

        def flush() -> None:
            nonlocal chunk
            if chunk:
                parts.append(f'f"{chunk}"' if "{" in chunk else f'"{chunk}"')
                chunk = ""

        try:
            fields = list(string.Formatter().parse(template))
        except ValueError as error:
            self._refuse(node, f"the format string is invalid: {error}")
            return None
        for literal, field, spec, conversion in fields:
            for char in re.split(r"([{}])", literal):
                if char in "{}" and char:
                    flush()
                    parts.append(f'"{char}"')
                else:
                    chunk += _escaped(char)
            if field is None:
                continue
            head = re.match(r"[^.\[]*", field)
            assert head is not None
            first, rest = head.group(0), field[head.end() :]
            if not _FORMAT_REFERENCE.fullmatch(rest):
                self._refuse(node, f"the format field {field!r} cannot be rewritten")
                return None
            if first == "" or first.isdigit():
                kind = "auto" if first == "" else "manual"
                if numbering not in (None, kind):
                    self._refuse(
                        node,
                        "the format string mixes automatic and manual field "
                        "numbering (a ValueError in Python)",
                    )
                    return None
                numbering = kind
                index = auto if first == "" else int(first)
                auto += first == ""
                if index >= len(positional):
                    self._refuse(
                        node,
                        f"the format string uses positional argument {index} but "
                        f"only {len(positional)} given (an IndexError in Python)",
                    )
                    return None
                key = f"_acs_f{index}"
            else:
                if first not in named:
                    self._refuse(
                        node,
                        f"the format string uses keyword {first!r}, which is not "
                        "given (a KeyError in Python)",
                    )
                    return None
                key = f"_acs_k_{first}"
            if conversion not in (None, "r", "s", "a"):
                self._refuse(node, f"unknown conversion !{conversion} in the format")
                return None
            if any(c in (spec or "") for c in "{}\"'\\\n"):
                self._refuse(
                    node,
                    "a format spec with a field inside it (`{:>{}}`) cannot be "
                    "rewritten: Codon's f-strings have none; format the width "
                    "into the spec first, or write the pieces separately",
                )
                return None
            access = ""
            for attribute, subscript in _ACCESS.findall(rest):
                if attribute:
                    access += "." + attribute
                elif subscript.isdigit():
                    access += f"[{subscript}]"
                elif any(c in subscript for c in "\"'\\"):
                    self._refuse(node, f"the format field {field!r} has odd quoting")
                    return None
                else:
                    access += f"['{subscript}']"
            value = params[key]
            if inline:
                expression = self._text(value)
            else:
                expression = key
            if "\n" in expression:
                expression = "(" + expression + ")"
            body = expression + access
            if conversion:
                body += "!" + conversion
            if spec:
                body += ":" + spec
            chunk += "{" + body + "}"
        flush()
        if not parts:
            return '""'
        text = "(" + " + ".join(parts) + ")" if len(parts) > 1 else parts[0]
        if inline:
            return text
        names = list(params)
        values = [self._text(v) for v in params.values()]
        return f"(lambda {', '.join(names)}: {text})({', '.join(values)})"

    # -- lambda x, k=k: ... -------------------------------------------------

    def visit_Lambda(self, node: ast.Lambda) -> None:
        args = node.args
        positional = [*args.posonlyargs, *args.args]
        first = len(positional) - len(args.defaults)
        moved: list[tuple[str, ast.expr]] = []
        keep_defaults = list(args.defaults)
        for index, default in enumerate(args.defaults):
            if not _literal(default):
                cut = index
                moved = [
                    (positional[first + i].arg, d)
                    for i, d in enumerate(args.defaults)
                    if i >= cut
                ]
                keep_defaults = args.defaults[:cut]
                break
        inner = ast.arguments(
            posonlyargs=[a for a in args.posonlyargs if a.arg not in dict(moved)],
            args=[a for a in args.args if a.arg not in dict(moved)],
            vararg=args.vararg,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=args.kwarg,
            defaults=keep_defaults,
        )
        for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
            if kw_default is not None and not _literal(kw_default):
                moved.append((arg.arg, kw_default))
            else:
                inner.kwonlyargs.append(arg)
                inner.kw_defaults.append(kw_default)
        if not moved:
            self.generic_visit(node)
            return
        names = ", ".join(name for name, _ in moved)
        values = ", ".join(self._text(default) for _, default in moved)
        params = ast.unparse(inner)
        head = f"lambda {params}" if params else "lambda"
        self.edits.replace(
            *self.edits.node_span(node),
            f"(lambda {names}: {head}: {self._text(node.body)})({values})",
        )


def _total_ordering_names(tree: ast.Module) -> set[str]:
    """How ``functools.total_ordering`` is spelled in a decorator."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "functools":
                    names.add(f"{alias.asname or 'functools'}.total_ordering")
        elif isinstance(node, ast.ImportFrom) and node.module == "functools":
            for alias in node.names:
                if alias.name == "total_ordering" and not node.level:
                    names.add(alias.asname or alias.name)
    return names


def _ordering_blocks(
    tree: ast.Module, total_ordering: set[str], refused: list[tuple[int, str]]
) -> list[tuple[str, str]]:
    """The comparisons Codon lacks, per class, to append with ``@extend``.

    A top-level class defining ``__lt__`` gets ``__gt__`` (as Python reflects
    it: ``other < self``) and ``__ge__`` (``not self < other``, which Codon's
    timsort calls).  A class under ``@total_ordering`` gets every comparison
    the decorator adds, from its first of ``__lt__``, ``__le__``, ``__gt__``,
    ``__ge__``.  Each is (the class, the lines).
    """
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    def defines(
        node: ast.ClassDef, name: str, seen: frozenset[str] = frozenset()
    ) -> bool:
        options = _dataclass_options(node)
        if name in _defined_methods(node) or (
            options is not None and options["order"] and name in _ORDERING
        ):
            return True
        return any(
            isinstance(base, ast.Name)
            and base.id in classes
            and base.id not in seen
            and defines(classes[base.id], name, seen | {node.name})
            for base in node.bases
        )

    out: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        ordered = any(_dotted(d) in total_ordering for d in node.decorator_list)
        own = _defined_methods(node)
        options = _dataclass_options(node)
        has_eq = (
            "__eq__" in own
            or (options is not None and options["eq"])
            or any(
                _dotted(base) not in ("object", "ABC", "abc.ABC") for base in node.bases
            )
        )
        equal = "self == other" if has_eq else "self is other"
        if ordered and not any(defines(node, n) for n in _ORDERING):
            refused.append(
                (
                    node.lineno,
                    "@total_ordering needs the class to define one of __lt__, "
                    "__le__, __gt__ or __ge__ (a ValueError in Python)",
                )
            )
            continue
        bodies: dict[str, str] = {}
        if ordered:
            # Python's root is the greatest of the comparisons the class has,
            # inherited ones included.
            root = next(n for n in _ORDERING if defines(node, n))
            table = {
                "__lt__": {
                    "__gt__": f"not (self < other or {equal})",
                    "__le__": f"self < other or {equal}",
                    "__ge__": "not (self < other)",
                },
                "__le__": {
                    "__lt__": f"self <= other and not ({equal})",
                    "__gt__": "not (self <= other)",
                    "__ge__": f"not (self <= other) or {equal}",
                },
                "__gt__": {
                    "__lt__": f"not (self > other or {equal})",
                    "__le__": "not (self > other)",
                    "__ge__": f"self > other or {equal}",
                },
                "__ge__": {
                    "__lt__": "not (self >= other)",
                    "__le__": f"not (self >= other) or {equal}",
                    "__gt__": f"self >= other and not ({equal})",
                },
            }
            bodies = table[root]
        elif "__lt__" in own and not (options is not None and options["order"]):
            bodies = {
                "__gt__": "other.__lt__(self)",
                "__ge__": "not self.__lt__(other)",
            }
        lines = [
            f"    def {name}(self, other: {node.name}) -> bool:\n"
            f"        return {body}\n"
            for name, body in bodies.items()
            if not defines(node, name)
        ]
        if lines:
            out.append(
                (node.name, f"\n\n@extend\nclass {node.name}:\n" + "".join(lines))
            )
    return out


def _run(source: str) -> tuple[str, list[tuple[str, str]], list[tuple[int, str]]]:
    if not source.endswith("\n"):
        source += "\n"
    blocks: list[tuple[str, str]] = []
    refused: list[tuple[int, str]] = []
    for step in range(_MAX_PASSES):
        tree = ast.parse(source)
        names = _total_ordering_names(tree)
        if step == 0:
            blocks = _ordering_blocks(tree, names, refused)
        edits = _Edits(source.encode().splitlines(keepends=True))
        rewriter = _Pass(edits, names)
        rewriter.visit(tree)
        for problem in rewriter.refused:
            if problem not in refused:
                refused.append(problem)
        if not edits.edits:
            break
        source = edits.apply()
    return source, blocks, sorted(refused)


def python_rewrites(source: str) -> tuple[str, list[tuple[str, str]]]:
    """*source* rewritten (every statement on its line), and the ordering
    methods to append after the module, each as (the class, the lines)."""
    text, blocks, _ = _run(source)
    return text, blocks


def python_rewrite_refusals(source: str) -> list[tuple[int, str]]:
    """What the rewrites cannot express, as (line, message), in line order."""
    return _run(source)[2]
