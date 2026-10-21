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
* Builtins and str methods Codon lacks (#439), each with the type of what it
  replaces: ``s.startswith((a, b))`` (also ``endswith``, and with a ``start``
  / ``end``) becomes ``(s.startswith(a) or s.startswith(b))``, the receiver and
  the items bound once by a lambda unless they are names, attributes or
  constants; ``format(x, spec)`` becomes ``x.__format__(spec)`` and
  ``format(x)`` ``str(x)``; ``ascii(x)`` becomes ``repr(x)`` (the same type: it
  differs from ``ascii`` only in the characters it escapes, and only the
  checked copy changes).  A name the module binds itself (``def format``) is
  left alone.
* A top-level class that defines ``__lt__`` (what ``sorted()``, ``.sort()``
  and ``heapq`` need in Python) is given the ``__gt__`` and ``__ge__`` Codon's
  timsort calls, with ``@extend``; ``@functools.total_ordering`` is removed
  with its import (Codon has neither) and the class given every comparison
  ``total_ordering`` would add, from whichever one it defines.
* ``lambda x, k=k: ...``, a default that is not a literal (Codon looks the name
  up in the lambda: "name '%_k_930' is not defined"), becomes
  ``(lambda k: lambda x: ...)(k)``: the default is bound by a call, as Python
  binds it once.  The captured parameter can then no longer be passed.
* A class constant read by a later statement of its own class body (#433):
  ``B = A * 2``, a dataclass field default ``x: int = A``, a method
  parameter default or decorator.  Python evaluates those in the class scope;
  Codon in the module's ("name 'A' is not defined"), so the name becomes
  ``Class.A``.  Only what Python resolves that way is rewritten: a name bound
  earlier in that class body by an assignment, read where the class scope is
  visible.  A method body, a lambda body and a comprehension (but for its
  first iterable) do not see the class scope, so a bare name there is a module
  global and is left alone; a nested class sees only its own body.  In an
  Enum a member read by a later statement of the body is its value (an alias
  in Python), so the literal is written in its place.

Refused with a hint, as no cheap faithful rewrite exists (#379, #433, #439): an
``except*`` handler (Codon has no exception groups and reports the line after
it as "unexpected dedent"), a
class constant read in the body of a class defined in a function, or assigned
again later in the class body (Codon reads the last value), or an Enum member
without a literal value; a ``match`` class, mapping (with a value pattern or ``**rest``), star or ``as`` pattern;
and Codon's own refusals of ``f(*list)`` / ``f(**dict)``.  (A class passed
where a ``Protocol`` is expected is hinted in ``check.py``, which knows the
scenario's classes.)
"""

from __future__ import annotations

import ast
import re
import string
from collections.abc import Iterator

from .transform import (
    _ENUM_BASES,
    _is_named_tuple,
    _dataclass_options,
    _defined_methods,
    _dotted,
    _Edits,
)

__all__ = ["REWRITE_HINTS", "python_rewrites", "python_rewrite_refusals"]

#: Codon's refusals of what this pass leaves alone, and what to write.
REWRITE_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^'(?:str|'.*')' object has no attribute 'format'"),
        "Codon's str has no .format(): the check rewrites it only on a string "
        "literal with plain arguments; write an f-string instead",
    ),
    (
        re.compile(r"'Tuple\[str(?:,str)*\]' does not match expected type 'str'"),
        "if this is str.startswith / str.endswith with a tuple held in a name, "
        "Codon takes one str: write any(s.startswith(p) for p in prefixes) (the "
        "check rewrites a tuple written in the call)",
    ),
    (
        re.compile(r"no function 'Dict\.update' with arguments"),
        "Codon's dict.update takes a dict or pairs, not keywords: if it is "
        'd.update(k=v), write d.update({"k": v})',
    ),
    (
        re.compile(r"'int' object has no attribute '(?:from_bytes|to_bytes)'"),
        "Codon's int has no from_bytes / to_bytes (and its bytes is a str): "
        "build the value with shifts and masks (n = (n << 8) | byte), or keep "
        "it out of the checked code",
    ),
    (
        re.compile(r"object has no attribute 'format_map'"),
        "Codon's str has no format_map(): write an f-string, or "
        '"...".format(**d) spelled out ("{a}".format(a=d["a"]))',
    ),
    (
        re.compile(r"'str' object has no attribute '(?:decode|hex)'"),
        "Codon's bytes is a str, so it has no decode() / hex(): a bytes value "
        'is already text there; format with f"{b:02x}" per byte if needed',
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
    (
        re.compile(r"argument after \* must be a tuple, not '(?:List|Set)\["),
        "Codon unpacks only a tuple with *: pass the elements "
        "(f(xs[0], xs[1])), or take the list as a parameter",
    ),
    (
        re.compile(r"argument after \*\* must be a named tuple, not 'Dict\["),
        "Codon unpacks only a named tuple with **: pass the keywords "
        "(f(a=d['a'], b=d['b'])), or take the dict as a parameter",
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


#: The classes whose bare pattern (``case int():``) Codon compiles.
_BUILTIN_PATTERNS = frozenset(
    {"int", "float", "str", "bool", "bytes", "list", "dict", "tuple", "set"}
)


def _match_refusal(pattern: ast.AST) -> str | None:
    """Why Codon 0.19 cannot compile *pattern*, a ``match`` pattern: it binds
    no name inside a class, mapping or star pattern, and compiles a class
    pattern's arguments as a call of the class (#379)."""
    if isinstance(pattern, ast.MatchStar):
        return (
            "a star pattern (`case [a, *rest]`) is not supported by the check "
            "(Codon: unexpected star expression): match the length with "
            "`case [a, b]`, or test len(xs) and slice"
        )
    if isinstance(pattern, ast.MatchClass) and (
        pattern.patterns
        or pattern.kwd_patterns
        or not (
            isinstance(pattern.cls, ast.Name) and pattern.cls.id in _BUILTIN_PATTERNS
        )
    ):
        return (
            "a class pattern with arguments (`case int(n)`, `case Pt(x=x)`), or "
            "of a class of your own (`case Pt()`), is not supported by the "
            "check (Codon binds none of its names, and reads the pattern as a "
            "call of the class): write isinstance(v, Pt), then read the "
            "attributes (v.x); `case int():` and the like compile"
        )
    if isinstance(pattern, ast.MatchMapping) and (
        pattern.rest is not None
        or any(
            isinstance(sub, ast.MatchAs)
            for value in pattern.patterns
            for sub in ast.walk(value)
        )
    ):
        return (
            "a mapping pattern that binds or ignores a value (`case {'k': v}`) "
            "is not supported by the check (Codon binds none of its names): "
            "test `'k' in d` and read d['k'], or match literal values only "
            "(`case {'k': 1}`)"
        )
    if isinstance(pattern, ast.MatchAs) and pattern.pattern is not None:
        return (
            "an `as` pattern (`case [a, b] as whole`) is not supported by the "
            "check (Codon: unexpected indentation): match the subject "
            "unnamed, and read the name it already has"
        )
    return None


def _scope_names(node: ast.AST) -> Iterator[ast.Name]:
    """The names *node* loads in the scope it is written in.

    A lambda body and the elements and conditions of a comprehension run in a
    scope of their own, which Python does not give the class's names; the
    defaults of a lambda and the first iterable of a comprehension run in the
    enclosing one.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Name):
            if isinstance(current.ctx, ast.Load):
                yield current
        elif isinstance(current, ast.Lambda):
            stack.extend(current.args.defaults)
            stack.extend(d for d in current.args.kw_defaults if d is not None)
        elif isinstance(
            current, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            stack.append(current.generators[0].iter)
        else:
            stack.extend(ast.iter_child_nodes(current))


def _stores(stmt: ast.stmt) -> set[str]:
    """Every name *stmt* binds in its own scope, nested statements included."""
    names: set[str] = set()
    stack: list[ast.AST] = [stmt]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Name) and not isinstance(current.ctx, ast.Load):
            names.add(current.id)
        elif isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(current.name)
            continue  # its body is a scope of its own
        elif isinstance(current, (ast.Import, ast.ImportFrom)):
            names.update(
                (a.asname or a.name).partition(".")[0]
                for a in current.names
                if a.name != "*"
            )
        elif isinstance(current, ast.ExceptHandler) and current.name:
            names.add(current.name)
        elif isinstance(current, (ast.Lambda, ast.ListComp, ast.SetComp)):
            continue
        stack.extend(ast.iter_child_nodes(current))
    return names


def _assigned(stmt: ast.stmt) -> set[str] | None:
    """The names a plain ``A = ...`` / ``A: T = ...`` binds, else ``None``."""
    if isinstance(stmt, ast.Assign) and all(
        isinstance(t, ast.Name) for t in stmt.targets
    ):
        return {t.id for t in stmt.targets}  # type: ignore[attr-defined]
    if (
        isinstance(stmt, ast.AnnAssign)
        and stmt.value is not None
        and isinstance(stmt.target, ast.Name)
    ):
        return {stmt.target.id}
    return None


def _bare_final(annotation: ast.expr) -> bool:
    """``Final`` alone, which the check drops: the name stays a class attribute."""
    return (_dotted(annotation) or "").removeprefix("typing.") == "Final"


def _class_scope_exprs(stmt: ast.stmt) -> list[ast.AST]:
    """The parts of *stmt* Python evaluates in the class scope, as it runs."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = stmt.args
        return [
            *stmt.decorator_list,
            *args.defaults,
            *(d for d in args.kw_defaults if d is not None),
        ]
    if isinstance(stmt, ast.ClassDef):
        return [*stmt.decorator_list, *stmt.bases, *(k.value for k in stmt.keywords)]
    if isinstance(stmt, (ast.If, ast.While)):
        return [stmt.test]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [item.context_expr for item in stmt.items]
    if isinstance(stmt, ast.Try):
        return [h.type for h in stmt.handlers if h.type is not None]
    if isinstance(stmt, ast.AnnAssign):  # the annotation is no value
        return [n for n in (stmt.value, stmt.target) if n is not None]
    if isinstance(stmt, ast.Match):
        return [stmt.subject]
    return [stmt]


def _nested_bodies(stmt: ast.stmt) -> list[list[ast.stmt]]:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return []
    if isinstance(stmt, ast.Try):
        return [
            stmt.body,
            *(h.body for h in stmt.handlers),
            stmt.orelse,
            stmt.finalbody,
        ]
    if isinstance(stmt, ast.Match):
        return [case.body for case in stmt.cases]
    return [
        value
        for name in ("body", "orelse")
        if isinstance(value := getattr(stmt, name, None), list)
    ]


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


def _rebound_names(tree: ast.AST) -> set[str]:
    """Every name the module binds anywhere (a variable, parameter, function,
    class or import), so a builtin of the same name is not taken for it."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


class _Pass(ast.NodeVisitor):
    def __init__(self, edits: _Edits, total_ordering: set[str]) -> None:
        self.edits = edits
        self.total_ordering = total_ordering
        self.refused: list[tuple[int, str]] = []
        self._depth = 0
        self._rebound: set[str] = set()
        #: Where each enclosing scope's class can be named from outside: the
        #: dotted path of a class, ``None`` for a function (a class defined in
        #: one has no name Codon can resolve there).
        self._paths: list[str | None] = []

    def _text(self, node: ast.AST) -> str:
        return self.edits.text(*self.edits.node_span(node))

    def _refuse(self, node: ast.AST, message: str) -> None:
        self.refused.append((node.lineno, message))  # type: ignore[attr-defined]

    # -- class X: ... ------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        path = (
            node.name
            if not self._paths
            else f"{self._paths[-1]}.{node.name}"
            if self._paths[-1] is not None
            else None
        )
        self._class_constants(node, path)
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
        self._paths.append(path)
        for stmt in node.body:
            self.visit(stmt)
        self._paths.pop()
        self._depth -= 1

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._depth += 1
        self._paths.append(None)
        self.generic_visit(node)
        self._paths.pop()
        self._depth -= 1

    visit_FunctionDef = _function  # noqa: N815
    visit_AsyncFunctionDef = _function  # noqa: N815

    # -- class A: B = A_CONST * 2 ------------------------------------------

    def _refuse_rebound(self, name: ast.Name, node: ast.ClassDef) -> None:
        self._refuse(
            name,
            f"`{name.id}` is assigned again later in the body of class "
            f"{node.name}: Codon reads a class attribute's last value, not "
            "the one at this point; give the second assignment another name",
        )

    def _class_constants(self, node: ast.ClassDef, path: str | None) -> None:
        """Write ``Class.X`` for a class constant read in the class scope (#433).

        The names Python resolves in the class scope are those a statement of
        the body, a method default or a decorator loads after an assignment
        in the same body bound them.  Names bound by a ``def``, ``class``,
        import, loop or ``if`` are left alone (a property setter decorator
        ``@x.setter`` names the property, not an attribute to read).
        """
        enum = not node.keywords and any(
            (_dotted(b) or "").rsplit(".", 1)[-1] in _ENUM_BASES for b in node.bases
        )
        later = [_stores(stmt) for stmt in node.body]
        # name -> its value, for the names assigned once at the body's level
        constants: dict[str, ast.expr | None] = {}
        other: set[str] = set()  # bound some other way: not a constant to read
        # name -> its value, for the annotated names: typed fields in Codon
        fields: dict[str, ast.expr] = {}

        def check(expr: ast.AST, index: int, field_default: bool = False) -> None:
            for name in _scope_names(expr):
                if (name.id not in constants and name.id not in fields) or (
                    name.id in other
                ):
                    continue
                span = self.edits.node_span(name)
                if self.edits.text(*span) != name.id:
                    continue  # a position inside an f-string, before 3.12
                if name.id in fields:
                    fvalue = fields[name.id]
                    text = self._text(fvalue)
                    if not _literal(fvalue) or "\n" in text:
                        self._refuse(
                            name,
                            f"`{name.id}` is annotated in the body of class "
                            f"{node.name}, which Codon makes a field of the "
                            "instance, not a class attribute to read here, "
                            "and its value is not a literal the check can "
                            "write in its place: write "
                            f"`{name.id}: Final = ...` (or no annotation) for "
                            "a constant, or use a module-level constant",
                        )
                        continue
                    # Python reads the value the body bound; a literal can be
                    # written where the name is.
                    if any(name.id in names for names in later[index + 1 :]):
                        self._refuse_rebound(name, node)
                        continue
                    wrap = isinstance(fvalue, ast.UnaryOp) and not (
                        expr is name or getattr(expr, "value", None) is name
                    )
                    self.edits.replace(*span, f"({text})" if wrap else text)
                    continue
                if any(name.id in names for names in later[index + 1 :]):
                    self._refuse_rebound(name, node)
                    continue
                if field_default:
                    self._refuse(
                        name,
                        f"`{name.id}` is a class constant read by a field "
                        f"default of the NamedTuple {node.name}, which Codon "
                        "cannot compile (no module named "
                        f"'{node.name}'): write the value, or use a "
                        "module-level constant",
                    )
                    continue
                value = constants[name.id]
                if enum and value is not None:
                    if not _literal(value):
                        self._refuse(
                            name,
                            f"`{name.id}` is an Enum member read in the body of "
                            f"{node.name}, where Python gives its value, which "
                            "the check can write only for a literal: write "
                            "the value",
                        )
                        continue
                    text = self._text(value)
                    if "\n" in text:
                        self._refuse(
                            name,
                            f"`{name.id}` is an Enum member whose value spans "
                            "lines: write the value",
                        )
                        continue
                    wrap = isinstance(value, ast.UnaryOp) and not (
                        expr is name or getattr(expr, "value", None) is name
                    )
                    self.edits.replace(*span, f"({text})" if wrap else text)
                elif path is None:
                    self._refuse(
                        name,
                        f"`{name.id}` is a class constant read in the body of "
                        f"class {node.name}, which is defined in a function: "
                        "Codon cannot name the class from there, so make the "
                        "class module-level or use a module-level constant",
                    )
                else:
                    self.edits.replace(*span, f"{path}.{name.id}")

        def walk(stmts: list[ast.stmt], index: int | None, level: bool) -> None:
            for i, stmt in enumerate(stmts):
                at = i if index is None else index
                default = isinstance(stmt, ast.AnnAssign) and _is_named_tuple(node)
                for expr in _class_scope_exprs(stmt):
                    check(expr, at, default)
                for body in _nested_bodies(stmt):
                    walk(body, at, False)
                assigned = _assigned(stmt)
                if assigned is None or not level:
                    other.update(_stores(stmt))
                    continue
                if isinstance(stmt, ast.AnnAssign) and not _bare_final(stmt.annotation):
                    for name in assigned - other:
                        constants.pop(name, None)
                        fields[name] = stmt.value  # type: ignore[assignment]
                    continue
                member = enum and isinstance(stmt, ast.Assign)
                value = stmt.value if isinstance(stmt, ast.Assign) else None
                for name in assigned - other:
                    fields.pop(name, None)
                    # The last assignment before a read is the one it sees.
                    constants[name] = (
                        value if member and not name.startswith("_") else None
                    )

        walk(node.body, None, True)

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

    # -- match ... case ... -------------------------------------------------

    def visit_Match(self, node: ast.Match) -> None:
        for case in node.cases:
            for sub in ast.walk(case.pattern):
                message = _match_refusal(sub)
                if message is not None:
                    self._refuse(sub, message)
                    break
        self.generic_visit(node)

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

    def visit_Module(self, node: ast.Module) -> None:
        self._rebound = _rebound_names(node)
        self.generic_visit(node)

    # -- except* -------------------------------------------------------------

    def visit_TryStar(self, node: ast.AST) -> None:  # ast.TryStar, Python 3.11+
        for handler in node.handlers:  # type: ignore[attr-defined]
            self._refuse(
                handler,
                "exception groups (`except*`) are not supported by the check "
                "(Codon has no syntax for them and reports the line after the "
                "handler as 'unexpected dedent'): use `except`, naming the "
                "exception types one handler each",
            )
        self.generic_visit(node)

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
        text = self._missing_builtin(node)
        if text is not None:
            self.edits.replace(*self.edits.node_span(node), text)
            return
        self.generic_visit(node)

    # -- startswith((a, b)), format(), ascii() (#439) ------------------------

    def _wrapped(self, node: ast.AST, inner: str) -> str:
        """``(inner)``, with the line breaks of *node* kept inside the brackets
        (a break after them could end the statement)."""
        lost = self._text(node).count("\n") - inner.count("\n")
        return "(" + inner + "\n" * max(lost, 0) + ")"

    def _missing_builtin(self, node: ast.Call) -> str | None:
        """What Codon 0.19 lacks, written with what it has; ``None`` if *node*
        is none of them."""
        func = node.func
        plain_call = (
            not any(isinstance(a, ast.Starred) for a in node.args) and not node.keywords
        )
        if isinstance(func, ast.Name) and func.id not in self._rebound:
            if func.id == "format" and plain_call and len(node.args) in (1, 2):
                value = self._operand(node.args[0])
                if len(node.args) == 1:
                    return f"str({self._text(node.args[0])})"
                return f"{value}.__format__({self._text(node.args[1])})"
            if func.id == "ascii" and plain_call and len(node.args) == 1:
                return f"repr({self._text(node.args[0])})"
            return None
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("startswith", "endswith")
            and node.args
            and isinstance(node.args[0], ast.Tuple)
            and node.args[0].elts
            and plain_call
            and len(node.args) <= 3
            and not any(isinstance(e, ast.Starred) for e in node.args[0].elts)
        ):
            return self._affix_tuple(node, func)
        return None

    def _operand(self, node: ast.expr) -> str:
        """The text of *node*, parenthesised where a ``.attr`` needs it."""
        text = self._text(node)
        if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
            return text
        return f"({text})"

    def _affix_tuple(self, node: ast.Call, func: ast.Attribute) -> str:
        """``(s.startswith(a) or s.startswith(b))`` for ``s.startswith((a, b))``,
        the receiver, items and ``start`` / ``end`` evaluated once, in order."""
        items = list(node.args[0].elts)  # type: ignore[attr-defined]
        extra = list(node.args[1:])
        parts = [func.value, *items, *extra]

        def inline(part: ast.expr) -> bool:
            return isinstance(part, ast.Constant) or _plain(part)

        if all(inline(p) for p in parts):
            receiver = self._operand(func.value)
            names = {id(p): self._text(p) for p in parts}
            call = ", ".join(names[id(e)] for e in extra)
            tails = [
                f"{receiver}.{func.attr}({', '.join([names[id(i)]] + ([call] if call else []))})"
                for i in items
            ]
            return self._wrapped(node, " or ".join(tails))
        params = {id(p): f"_acs_a{n}" for n, p in enumerate(parts)}
        tail = "".join(f", {params[id(e)]}" for e in extra)
        body = " or ".join(
            f"{params[id(func.value)]}.{func.attr}({params[id(i)]}{tail})"
            for i in items
        )
        lambda_params = ", ".join(params[id(p)] for p in parts)
        values = ", ".join(self._text(p) for p in parts)
        return self._wrapped(node, f"(lambda {lambda_params}: {body})({values})")

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
