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
* ``def __exit__(self, exc_type, exc, tb)``, which Codon calls with *no*
  arguments whatever the block did (so it is "missing 1 required positional
  argument"), is given a ``None`` default for each parameter without one, and
  loses their annotations (``type[BaseException] | None`` and the like, which
  Codon cannot express): the parameters are ``None`` in the check, so a
  branch ``if exc_type is not None:`` is not compiled (#435).  ``*args`` and
  parameters with a default of their own are left alone.
* A list or dict display of functions or lambdas assigned to a name annotated
  with ``Callable[[A], R]`` elements (``fs: list[Callable[[int], int]] =
  [inc, dbl]``, a dispatch table) gets each element wrapped in Codon's
  ``Callable[[A], R](...)``: Codon gives every function a type of its own, so
  two different ones in a container do not unify otherwise (#436).  So does a
  ``return`` of such a display from a function annotated so.

Refused with a hint, as no cheap faithful rewrite exists (#379): a ``match``
class, mapping (with a value pattern or ``**rest``), star or ``as`` pattern;
and Codon's own refusals of ``f(*list)`` / ``f(**dict)``.  (A class passed
where a ``Protocol`` is expected is hinted in ``check.py``, which knows the
scenario's classes.)
"""

from __future__ import annotations

import ast
import re
import string

from .transform import (
    _dataclass_options,
    _defined_methods,
    _dotted,
    _Edits,
    _is_none,
    callable_annotation,
    codon_annotation,
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
        # Codon gives every function and lambda a type of its own.
        re.compile(
            r"'(?:%_lambda_\d+(?:\[[^']*\]|\([^']*\))?|\w+\([^']*\.\.\.\))' "
            r"does not match expected type "
            r"'(?:%_lambda_\d+(?:\[[^']*\]|\([^']*\))?|\w+\([^']*\.\.\.\))'"
        ),
        "Codon gives every function and lambda a type of its own, so a list "
        "or dict holding different ones (a dispatch table) does not compile "
        "unless it is annotated with the shared signature where it is "
        "assigned or returned (`fs: list[Callable[[int], int]] = [inc, dbl]`; "
        "a class attribute, a computed display, a bound method "
        "or a lambda using a local variable is not rewritten): write an "
        "if/elif chain, or a `match` on the key, instead",
    ),
    (
        re.compile(r"no function 'Callable\.__new__' with arguments"),
        "a dispatch table can hold functions and lambdas that capture "
        "nothing: a bound method (`self.run`) or a lambda using a local "
        "variable cannot be converted to Codon's Callable; use a "
        "module-level function that takes the object, or an if/elif chain "
        "(a dispatch table of different functions)",
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

_SEQUENCES = frozenset({"list", "List", "set", "Set", "Sequence", "MutableSequence"})
_MAPPINGS = frozenset({"dict", "Dict", "Mapping", "MutableMapping"})

_DISPLAYS = (ast.List, ast.Set, ast.Dict)
_ORDERING = ("__lt__", "__le__", "__gt__", "__ge__")
_MAX_PASSES = 20
_FORMAT_REFERENCE = re.compile(r"(?:\.\w+|\[[^\]\[]+\])*")
_ACCESS = re.compile(r"\.(\w+)|\[([^\]\[]+)\]")


def _literal(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp):
        node = node.operand
    return isinstance(node, ast.Constant)


def _callable_type(annotation: ast.AST) -> str | None:
    """Codon's spelling of a ``Callable[[A, B], R]`` annotation, or ``None``.

    The argument types are a ``Tuple[A, B]``, not the list the annotation
    has: the transform turns a display of different types (``[float, str]``)
    into an ``_acs_list(...)`` call.
    """
    spelled = callable_annotation(annotation)
    if spelled is None:
        return None
    if isinstance(annotation, ast.Constant):
        annotation = ast.parse(str(annotation.value), mode="eval").body
    assert isinstance(annotation, ast.Subscript)
    assert isinstance(annotation.slice, ast.Tuple)
    params, result = annotation.slice.elts
    assert isinstance(params, ast.List)
    if not params.elts:
        return spelled
    args = [codon_annotation(p, class_level=True) for p in params.elts]
    ret = codon_annotation(result, class_level=True)
    return f"Callable[Tuple[{', '.join(str(a) for a in args)}], {ret}]"


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
        #: The functions being visited, innermost last, with their local names.
        self._functions: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]]
        self._functions = []
        self._class_body = False
        #: The names ``import x`` / ``import x.y as z`` bind: ``x.f`` is a function.
        self._modules: set[str] = set()

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
            elif isinstance(stmt, ast.FunctionDef) and stmt.name == "__exit__":
                self._exit_parameters(stmt)
        for base in (*node.bases, *node.keywords):
            self.visit(base)
        self._depth += 1
        outer, self._class_body = self._class_body, True
        for stmt in node.body:
            self.visit(stmt)
        self._class_body = outer
        self._depth -= 1

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._depth += 1
        outer, self._class_body = self._class_body, False
        names = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
        names |= {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        self._functions.append((node, names))
        self.generic_visit(node)
        self._functions.pop()
        self._class_body = outer
        self._depth -= 1

    visit_FunctionDef = _function  # noqa: N815
    visit_AsyncFunctionDef = _function  # noqa: N815

    # -- def __exit__(self, exc_type, exc, tb) ------------------------------

    def _exit_parameters(self, node: ast.FunctionDef) -> None:
        """Give the parameters Codon passes nothing for a ``None`` default.

        Codon calls ``__exit__()`` with no arguments, on an exception as well
        (#435).  The annotations of those parameters go: ``type[BaseException]
        | None`` and the like cannot be expressed, and the parameter is a
        ``None`` in the check, whatever it is in Python.
        """
        args = node.args
        positional = [*args.posonlyargs, *args.args]
        first = len(positional) - len(args.defaults)
        for index, arg in enumerate(positional):
            if index == 0:
                continue  # self
            default = args.defaults[index - first] if index >= first else None
            if default is None:
                self.edits.replace(*self.edits.node_span(arg), f"{arg.arg}=None")
            elif arg.annotation is not None and _is_none(default):
                self.edits.replace(*self.edits.node_span(arg), arg.arg)

    # -- fs: list[Callable[[int], int]] = [inc, dbl] -------------------------

    def _captures(self, node: ast.Lambda) -> bool:
        """Whether *node* uses a local variable of a function around it, which
        Codon cannot convert it to a ``Callable`` for."""
        own = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
        used = {
            n.id
            for n in ast.walk(node.body)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        return any((used - own) & names for _, names in self._functions)

    def _callables(self, annotation: ast.AST, value: ast.AST) -> None:
        """Wrap the functions of the display *value* in the ``Callable`` that
        *annotation* names for them (#436): ``[inc, dbl]`` under
        ``list[Callable[[int], int]]`` becomes ``[Callable[[int], int](inc),
        Callable[[int], int](dbl)]``, so they have one type."""
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                annotation = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                return
        spelled = _callable_type(annotation)
        if spelled is not None:
            root = (_dotted(value) or "").partition(".")[0]
            if (
                isinstance(value, ast.Name)
                or (isinstance(value, ast.Attribute) and root in self._modules)
                or (isinstance(value, ast.Lambda) and not self._captures(value))
            ):
                self.edits.replace(
                    *self.edits.node_span(value), f"{spelled}({self._text(value)})"
                )
            return
        if not isinstance(annotation, ast.Subscript):
            return
        name = (_dotted(annotation.value) or "").rpartition(".")[2]
        parts = (
            annotation.slice.elts
            if isinstance(annotation.slice, ast.Tuple)
            else [annotation.slice]
        )
        if name in _SEQUENCES and len(parts) == 1:
            if isinstance(value, (ast.List, ast.Set)):
                for element in value.elts:
                    self._callables(parts[0], element)
        elif name in _MAPPINGS and len(parts) == 2 and isinstance(value, ast.Dict):
            for key, element in zip(value.keys, value.values):
                if key is not None:
                    self._callables(parts[1], element)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # A class attribute is left alone: Codon crashes on a class-level
        # `Callable` list.
        if isinstance(node.value, _DISPLAYS) and not self._class_body:
            self._callables(node.annotation, node.value)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if isinstance(node.value, _DISPLAYS) and self._functions:
            returns = self._functions[-1][0].returns
            if returns is not None:
                self._callables(returns, node.value)
        self.generic_visit(node)

    # -- from functools import total_ordering -----------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._modules.add(alias.asname or alias.name.partition(".")[0])

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
