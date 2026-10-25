"""Diagnostics for a call or a ``return`` that say what the author got wrong,
in the author's terms, not Codon's (#527, #528, #530).

Codon reports a call it cannot match as ``no function 'X.__init__' with
arguments (X, str, str)``: ``self`` is an argument, a keyword argument is a
bare positional one, and neither the parameter nor the signature is named.
It reports a ``return`` of the wrong type with the two types swapped.  Both
are rewritten here from the author's own source (the call or ``return``
statement, found by its position) and the real Python signature of the
callee, which the check compiles the model of and
``test_typecheck_model.py`` compares with.

Everything here is a hint: a message that cannot be rewritten with certainty
stays as it is, apart from the generic form of the call message.
"""

from __future__ import annotations

import ast
import difflib
import enum
import inspect
import re
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable

#: ``no function 'X.__init__' with arguments (X, str, str)``
_NO_FUNCTION = re.compile(
    r"^no function '(?P<name>[\w.]+)' with arguments \((?P<args>[^()]*)\)$"
)
#: ``register_pass_condition() takes 2 arguments (3 given)``
_TAKES = re.compile(r"^(?P<name>\w+)\(\) takes \d+ arguments? \(\d+ given\)$")
#: ``'float' does not match expected type 'str'``
_MISMATCH = re.compile(
    r"^'(?P<declared>[^']+)' does not match expected type '(?P<got>[^']+)'$"
)
#: Codon prints a static string type inside the quotes the check adds.
_LITERAL_TYPE = re.compile(r"(?<=[\s(,\[])''[^']*''(?=[\s),\]:]|$)")
#: Frames of the model's own helpers (``_keyword_only``, ``_required_str``,
#: ``_acs_list`` ...) and of a ``__init__`` that encodes keyword-only
#: parameters as ``*_kw``: Codon realizing the model, not the author's code.
_INTERNAL_FRAME = re.compile(
    r"^(?:_keyword_only|_required_\w+|_acs_\w+)\(|\*_kw\b|\[[^\]\[]*\.codon:\d+\]$"
)

Resolver = Callable[[str], Any]


# ---------------------------------------------------------------------------
# Codon's own words in any message (#530)
# ---------------------------------------------------------------------------


def plain_literal_types(text: str) -> str:
    """*text* with Codon's static string types (``''3''``) read as ``'str'``."""
    return _LITERAL_TYPE.sub("'str'", text)


def is_internal_frame(frame: str) -> bool:
    """Whether a "during the realization of" frame is of the model, not of
    the author's code."""
    body = frame.removeprefix("during the realization of ")
    return _INTERNAL_FRAME.search(body) is not None


def codon_type(text: str) -> str:
    """A Codon type as the author writes it: ``Optional[NoneType]`` is
    ``None``, ``List[int]`` is ``list[int]``."""
    text = text.strip()
    found = re.fullmatch(r"Optional\[(.*)\]", text)
    if found:
        inner = codon_type(found[1])
        return inner if inner == "None" else f"{inner} | None"
    text = text.replace("NoneType", "None")
    return re.sub(
        r"\b(List|Dict|Tuple|Set|FrozenSet)\[",
        lambda m: m[1].lower() + "[",
        text,
    )


def _union_members(node: ast.expr) -> tuple[list[ast.expr], bool]:
    """The members of an annotation written ``X | Y``, ``Optional[X]`` or
    ``Union[X, Y]`` (``typing.`` prefix or not), and whether ``None`` is one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _union_members(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return [node], False
    if isinstance(node, ast.Constant) and node.value is None:
        return [], True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, left_none = _union_members(node.left)
        right, right_none = _union_members(node.right)
        return left + right, left_none or right_none
    if isinstance(node, ast.Subscript):
        head = node.value.attr if isinstance(node.value, ast.Attribute) else None
        if isinstance(node.value, ast.Name):
            head = node.value.id
        if head in ("Optional", "Union"):
            args = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            members: list[ast.expr] = []
            has_none = head == "Optional"
            for arg in args:
                inner, inner_none = _union_members(arg)
                members += inner
                has_none = has_none or inner_none
            return members, has_none
    return [node], False


def canonical_annotation(annotation: ast.expr | str) -> str:
    """An annotation, or a Codon type, in one spelling: ``X | None`` for
    ``Optional[X]``, ``Union[X, None]`` and ``None | X``, ``list[int]`` for
    ``List[int]``, no ``typing.`` prefix, no space after a comma."""
    if isinstance(annotation, str):
        try:
            node = ast.parse(annotation.strip(), mode="eval").body
        except SyntaxError:
            return annotation.strip()
    else:
        node = annotation
    members, has_none = _union_members(node)
    texts = []
    for member in members:
        text = re.sub(r"\b(?:\w+\.)+(\w+)", r"\1", ast.unparse(member))
        text = codon_type(text.strip("'\""))
        texts.append(re.sub(r",\s+", ",", text))
    if has_none:
        texts.append("None")
    return " | ".join(texts)


# ---------------------------------------------------------------------------
# Source lookup
# ---------------------------------------------------------------------------


class _Files:
    """The parse trees of the author's files, read once."""

    def __init__(self) -> None:
        self._trees: dict[str, tuple[ast.Module, list[str]] | None] = {}

    def get(self, path: str | None) -> tuple[ast.Module, list[str]] | None:
        if path is None:
            return None
        if path not in self._trees:
            try:
                text = Path(path).read_text()
                self._trees[path] = (ast.parse(text), text.splitlines())
            except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
                self._trees[path] = None
        return self._trees[path]


def _column(lines: list[str], line: int, offset: int) -> int:
    """The 1-based column of the UTF-8 byte *offset* of source line *line*."""
    try:
        text = lines[line - 1].encode()[:offset].decode()
    except (IndexError, UnicodeDecodeError):
        return offset + 1
    return len(text) + 1


def _callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _find_call(
    tree: ast.Module,
    lines: list[str],
    line: int,
    column: int | None,
    accept: Callable[[ast.Call], bool],
) -> ast.Call | None:
    """The call Codon points at: the one that starts at (*line*, *column*),
    else the only acceptable call of the line."""
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and n.lineno == line and accept(n)
    ]
    if column is not None:
        at = [n for n in calls if _column(lines, line, n.col_offset) == column]
        if len(at) == 1:
            return at[0]
    return calls[0] if len(calls) == 1 else None


def _builds_subclass(call: ast.Call, owner: str, resolve_class: Resolver) -> bool:
    """Whether *call* builds a subclass of *owner* (Codon names the class the
    inherited ``__init__`` is defined in)."""
    built = _callee_name(call)
    base = resolve_class(owner)
    cls = resolve_class(built) if built else None
    return isinstance(base, type) and isinstance(cls, type) and issubclass(cls, base)


def _enclosing_class(tree: ast.Module, line: int) -> str | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.lineno <= line <= (
            node.end_lineno or node.lineno
        ):
            if best is None or node.lineno > best.lineno:
                best = node
    return best.name if best is not None else None


# ---------------------------------------------------------------------------
# Types of the real signature
# ---------------------------------------------------------------------------


def _hints(target: Any) -> dict[str, Any]:
    """The annotations of *target*, the strings of ``from __future__ import
    annotations`` evaluated where they can be."""
    try:
        return typing.get_type_hints(target)
    except Exception:  # noqa: BLE001 - a name that only a type checker sees
        pass
    raw = getattr(target, "__annotations__", {}) or {}
    scope = dict(vars(inspect.getmodule(target) or types))
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            try:
                value = eval(value, scope)  # noqa: S307 - the author's annotation
            except Exception:  # noqa: BLE001
                pass
        out[key] = value
    return out


def _members(tp: Any) -> list[Any]:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        out: list[Any] = []
        for arg in typing.get_args(tp):
            out += _members(arg)
        return out
    return [tp]


def type_text(tp: Any) -> str:
    """An annotation as the author writes it."""
    if tp is None or tp is type(None):
        return "None"
    if isinstance(tp, str):
        return tp
    if tp is inspect.Parameter.empty:
        return "Any"
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        return " | ".join(type_text(m) for m in _members(tp))
    if isinstance(tp, type) and not typing.get_args(tp):
        return tp.__name__
    return re.sub(r"\b(?:\w+\.)+(\w+)", r"\1", str(tp)).replace("NoneType", "None")


def _accepts(got: str, tp: Any, resolve_class: Resolver) -> bool:
    """Whether a value Codon types *got* can be passed where *tp* is
    declared; ``True`` whenever that is not certain."""
    base = got.split("[")[0]
    inner = re.fullmatch(r"Optional\[(.*)\]", got)
    if inner is not None:
        return _accepts(inner[1], tp, resolve_class)
    for member in _members(tp):
        if isinstance(member, (str, typing.TypeVar, typing.ForwardRef)):
            return True
        if member is typing.Any or member is object:
            return True
        if member is None or member is type(None):
            if base == "NoneType":
                return True
            continue
        origin = typing.get_origin(member) or member
        if not isinstance(origin, type):
            return True  # Literal, Callable, a TypeVar ...: not judged
        if origin.__module__ in ("collections.abc", "typing") or getattr(
            origin, "_is_protocol", False
        ):
            return True
        if base == origin.__name__ or base.lower() == origin.__name__:
            return True
        if (base, origin) in (("int", float), ("int", complex), ("float", complex)):
            return True
        if base == "bool" and origin in (int, float):
            return True
        cls = resolve_class(base)
        if isinstance(cls, type) and issubclass(cls, origin):
            return True
    return False


def _default_text(value: Any) -> str:
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str) and len(value) <= 20:
        return repr(value)
    return "..."


def signature_text(sig: inspect.Signature, hints: dict[str, Any]) -> str:
    """``a: int, b: str = 'x', *, label: str``."""
    parts: list[str] = []
    star = False
    for p in sig.parameters.values():
        if p.kind is p.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        text = p.name
        if p.kind is p.VAR_POSITIONAL:
            text = "*" + text
        elif p.kind is p.VAR_KEYWORD:
            text = "**" + text
        if p.kind is p.VAR_POSITIONAL:
            star = True
        if p.name in hints:
            text += f": {type_text(hints[p.name])}"
        if p.default is not p.empty:
            text += f" = {_default_text(p.default)}"
        parts.append(text)
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Binding a call
# ---------------------------------------------------------------------------


@dataclass
class _Problem:
    message: str
    node: ast.expr | ast.keyword | None  # where to point; the call itself when ``None``


def _split_types(text: str) -> list[str]:
    out, depth, current = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            out.append(current.strip())
            current = ""
            continue
        depth += ch in "[("
        depth -= ch in "])"
        current += ch
    if current.strip():
        out.append(current.strip())
    return out


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _quoted(names: list[str]) -> str:
    quoted = [f"'{n}'" for n in names]
    return (
        quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]
    )


def _bind(
    label: str,
    call: ast.Call,
    sig: inspect.Signature,
    hints: dict[str, Any],
    got: list[str] | None,
    resolve_class: Resolver,
    shown: str,
) -> _Problem | None:
    """What is wrong with *call* against *sig*, or ``None`` when it binds and
    every argument has a type its parameter accepts.

    *got* is the Codon type of each argument (positional ones, then keyword
    ones, in call order), or ``None`` when Codon did not say.  *shown* is
    the signature as shown with a problem of the call's shape.
    """
    params = list(sig.parameters.values())
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    var_positional = any(p.kind is p.VAR_POSITIONAL for p in params)
    var_keyword = any(p.kind is p.VAR_KEYWORD for p in params)
    bound: dict[str, ast.expr] = {}
    types_of: dict[str, str] = {}

    for index, node in enumerate(call.args):
        if index < len(positional):
            bound[positional[index].name] = node
            if got is not None:
                types_of[positional[index].name] = got[index]
        elif not var_positional:
            only_by_name = [
                p.name
                for p in params
                if p.kind is p.KEYWORD_ONLY and p.name not in bound
            ]
            allowed = _plural(len(positional), "positional argument")
            extra = (
                f"; {_quoted(only_by_name)} can only be passed by name "
                f"(`{only_by_name[0]}=...`)"
                if only_by_name
                else ""
            )
            return _Problem(
                f"{label}: takes at most {allowed}, got {len(call.args)}{extra}; "
                f"{shown}",
                node,
            )
    for offset, keyword in enumerate(call.keywords):
        name = keyword.arg
        assert name is not None
        param = sig.parameters.get(name)
        if param is not None and param.kind not in (
            param.POSITIONAL_ONLY,
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            if name in bound:
                return _Problem(f"{label}: got multiple values for '{name}'", keyword)
            bound[name] = keyword.value
            if got is not None:
                types_of[name] = got[len(call.args) + offset]
        elif not var_keyword:
            free = [
                p.name
                for p in params
                if p.name not in bound
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
            close = difflib.get_close_matches(name, free, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            return _Problem(
                f"{label}: '{name}' is an invalid keyword argument{hint}; {shown}",
                keyword,
            )

    missing = [
        p.name
        for p in params
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and p.name not in bound
    ]
    if missing:
        what = "argument" if len(missing) == 1 else "arguments"
        return _Problem(
            f"{label}: missing required {what} {_quoted(missing)}; {shown}", None
        )

    for name, node in bound.items():
        if name in types_of and name in hints:
            if not _accepts(types_of[name], hints[name], resolve_class):
                return _Problem(
                    f"{label}: {name}: expected {type_text(hints[name])}, got "
                    f"{codon_type(types_of[name])}",
                    node,
                )
    return None


# ---------------------------------------------------------------------------
# The rewrites
# ---------------------------------------------------------------------------


def _target(
    name: str,
    call: ast.Call,
    tree: ast.Module,
    resolve_class: Resolver,
    resolve_callable: Resolver,
) -> tuple[str, inspect.Signature, dict[str, Any]] | None:
    """(the callee as the author writes it, its signature without ``self``,
    its parameter types), or ``None`` when it cannot be told."""
    owner, _, member = name.rpartition(".")
    if owner:
        cls = resolve_class(owner)
        if not isinstance(cls, type):
            return None
        if member == "__init__":
            built = resolve_class(_callee_name(call) or "")
            if not (isinstance(built, type) and issubclass(built, cls)):
                return None
            obj: Any = built
            label = built.__name__
            hints = _hints(built.__init__)
        else:
            if _callee_name(call) != member:
                return None
            obj = getattr(cls, member, None)
            if obj is None:
                return None
            label = _method_label(obj, member)
            hints = _hints(obj)
        sig = inspect.signature(obj)
        if member != "__init__" and not isinstance(
            inspect.getattr_static(cls, member), (classmethod, staticmethod)
        ):
            sig = sig.replace(parameters=list(sig.parameters.values())[1:])  # self
        return label, sig, hints
    if _callee_name(call) != name:
        return None
    obj = resolve_callable(name)
    if obj is None:
        return None
    return name, inspect.signature(obj), _hints(obj)


def _method_label(obj: Any, member: str) -> str:
    """``XScenario.h`` for a method the author wrote, the bare name for one of
    the framework's (``register_pass_condition``)."""
    func = getattr(obj, "__func__", obj)
    qualname = getattr(func, "__qualname__", "")
    module = getattr(func, "__module__", "") or ""
    if (
        qualname.endswith("." + member)
        and "<locals>" not in qualname
        and module.partition(".")[0] != "autoware_carla_scenario"
    ):
        return qualname
    return member


def call_message(
    message: str,
    path: str | None,
    line: int | None,
    column: int | None,
    files: _Files,
    resolve_class: Resolver,
    resolve_callable: Resolver,
) -> tuple[str, int | None, int | None] | None:
    """The author's message (and its line and column) for Codon's refusal of
    a call, or ``None`` when *message* is not one."""
    found = _NO_FUNCTION.match(message)
    takes = _TAKES.match(message)
    if (found is None and takes is None) or line is None:
        return None
    parsed = files.get(path)
    name = (found or takes)["name"]  # type: ignore[index]
    short = name.rpartition(".")[2]
    types_ = _split_types(found["args"]) if found else None
    if found and name.endswith(".__init__"):
        short = name.rpartition(".")[0]
    # self comes first for a constructor and a method.
    got = types_[1:] if types_ is not None and "." in name else types_
    call = None
    if parsed is not None:
        tree, lines = parsed
        call = _find_call(
            tree,
            lines,
            line,
            column,
            lambda c: _callee_name(c) in (short, name.rpartition(".")[2])
            or _builds_subclass(c, short, resolve_class),
        )
    if (
        call is None
        or any(isinstance(a, ast.Starred) for a in call.args)
        or any(k.arg is None for k in call.keywords)
    ):
        return _generic(short, got, None, line, column)
    assert parsed is not None
    if got is not None and len(got) != len(call.args) + len(call.keywords):
        got = None
    if takes is not None:
        owner = _enclosing_class(tree, line)
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and owner is not None
        ):
            name = f"{owner}.{short}"
        else:
            name = short
    resolved = None
    try:
        resolved = _target(name, call, tree, resolve_class, resolve_callable)
    except Exception:  # noqa: BLE001 - only a hint: the callee may not inspect
        pass
    if resolved is None:
        return _generic(short, got, call, line, column)
    label, sig, hints = resolved
    params = signature_text(sig, hints)
    problem = _bind(
        f"{label}()", call, sig, hints, got, resolve_class, f"{label} takes ({params})"
    )
    if problem is None:
        if takes is not None:
            return None  # Codon's own message was right
        return _generic(label, got, call, line, column, params)
    return _located(problem.message, problem.node, line, column, lines)


def _located(
    message: str,
    node: ast.expr | ast.keyword | None,
    line: int,
    column: int | None,
    lines: list[str],
) -> tuple[str, int | None, int | None]:
    if node is None:
        return message, line, column
    return message, node.lineno, _column(lines, node.lineno, node.col_offset)


def _generic(
    name: str,
    got: list[str] | None,
    call: ast.Call | None,
    line: int,
    column: int | None,
    params: str | None = None,
) -> tuple[str, int | None, int | None] | None:
    """``X() takes (a: float, ...); got (EntityRole, str, label=str)``, with
    ``self`` dropped and a keyword argument marked."""
    if got is None:
        return None
    shown = [codon_type(t) for t in got]
    if call is not None and len(got) == len(call.args) + len(call.keywords):
        for offset, keyword in enumerate(call.keywords):
            index = len(call.args) + offset
            shown[index] = f"{keyword.arg}={shown[index]}"
    given = f"got ({', '.join(shown)})"
    if params is None:
        return f"{name}() cannot be called with ({', '.join(shown)})", line, column
    return f"{name}() takes ({params}); {given}", line, column


# ---------------------------------------------------------------------------
# A wrong argument type, or an unknown keyword, in a call to the author's own
# callable (#540)
# ---------------------------------------------------------------------------

#: ``'c' is an invalid keyword argument for __init__()``
_INVALID_KEYWORD = re.compile(
    r"^'(?P<keyword>\w+)' is an invalid keyword argument for (?P<name>\w+)\(\)$"
)
_LITERAL_TYPES = {
    str: "str",
    bool: "bool",
    int: "int",
    float: "float",
    type(None): "None",
}


def _enclosing_function(
    tree: ast.Module, line: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.lineno <= line <= (node.end_lineno or node.lineno)
        ):
            if best is None or node.lineno > best.lineno:
                best = node
    return best


def _binds(scope: ast.AST, name: str) -> int:
    """How many times *scope* (a function) binds *name*: a parameter, an
    assignment target, a nested ``def`` or ``class``, an import, an ``except``."""
    count = 0
    for n in ast.walk(scope):
        if isinstance(n, ast.arg) and n.arg == name:
            count += 1
        elif (
            isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load)
        ):
            count += 1
        elif (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n is not scope
            and n.name == name
        ):
            count += 1
        elif isinstance(n, ast.ExceptHandler) and n.name == name:
            count += 1
        elif isinstance(n, (ast.Import, ast.ImportFrom)) and any(
            (a.asname or a.name.partition(".")[0]) == name for a in n.names
        ):
            count += 1
    return count


def _enclosing_functions(
    tree: ast.Module, line: int
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.lineno <= line <= (n.end_lineno or n.lineno)
    ]


def _certain_argument_type(
    node: ast.expr, function: ast.FunctionDef | ast.AsyncFunctionDef | None
) -> str | None:
    """The type Codon gives the argument *node*, when the syntax settles it:
    a literal, an f-string, or a parameter of the enclosing function that is
    annotated and never assigned.  ``None`` when it is not certain."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _certain_argument_type(node.operand, function)
        return inner if inner in ("int", "float") else None
    if isinstance(node, ast.Constant):
        return _LITERAL_TYPES.get(type(node.value))
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.Name) and function is not None:
        args = function.args
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        found = [a for a in every if a.arg == node.id]
        if len(found) != 1 or found[0].annotation is None:
            return None
        # the parameter itself is the only binding of the name (no lambda or
        # nested function parameter of that name, no assignment)
        if _binds(function, node.id) != 1:
            return None
        return canonical_annotation(found[0].annotation)
    return None


def _own_callee(
    call: ast.Call,
    tree: ast.Module,
    resolve_class: Resolver,
    resolve_callable: Resolver,
) -> tuple[str, inspect.Signature, dict[str, Any]] | None:
    """The function, method, static method or constructor *call* calls, when
    it can be told from the author's source alone."""
    func = call.func
    if isinstance(func, ast.Name):
        if any(_binds(f, func.id) for f in _enclosing_functions(tree, call.lineno)):
            return None  # a local of that name shadows the callable
        if isinstance(resolve_class(func.id), type):
            return _target(
                f"{func.id}.__init__", call, tree, resolve_class, resolve_callable
            )
        return _target(func.id, call, tree, resolve_class, resolve_callable)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        owner: str | None = None
        if func.value.id == "self":
            owner = _enclosing_class(tree, call.lineno)
        elif isinstance(resolve_class(func.value.id), type):
            owner = func.value.id
        if owner is not None:
            return _target(
                f"{owner}.{func.attr}", call, tree, resolve_class, resolve_callable
            )
    return None


def _plain_call(call: ast.Call) -> bool:
    return not any(isinstance(a, ast.Starred) for a in call.args) and all(
        k.arg is not None for k in call.keywords
    )


def argument_mismatch(
    message: str,
    path: str | None,
    line: int | None,
    column: int | None,
    files: _Files,
    resolve_class: Resolver,
    resolve_callable: Resolver,
) -> tuple[str, int | None, int | None] | None:
    """``f(): b: expected int, got str``, at the argument, for Codon's
    ``'str' does not match expected type 'int'`` at the start of a call.

    The argument is named only when it is the one: exactly one argument whose
    parameter is declared the type Codon expects has a type that is certain
    from the syntax and is the type Codon got, and no other argument of such
    a parameter could be it.  Anything else keeps Codon's message."""
    found = _MISMATCH.match(message)
    parsed = files.get(path)
    if found is None or parsed is None or line is None or column is None:
        return None
    tree, lines = parsed
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and n.lineno == line
        and _column(lines, line, n.col_offset) == column
    ]
    if len(calls) != 1 or not _plain_call(calls[0]):
        return None
    call = calls[0]
    callee = _own_callee(call, tree, resolve_class, resolve_callable)
    if callee is None:
        return None
    label, sig, hints = callee
    try:
        bound = sig.bind(*call.args, **{k.arg: k.value for k in call.keywords if k.arg})
    except TypeError:
        return None
    # Here the first type is the argument's and the second the parameter's.
    got = codon_type(found["declared"])
    want = canonical_annotation(found["got"])
    function = _enclosing_function(tree, line)
    culprits: list[tuple[str, ast.expr]] = []
    for name, node in bound.arguments.items():
        if name not in hints:
            continue
        if canonical_annotation(type_text(hints[name])) != want:
            continue
        if sig.parameters[name].kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return None
        assert isinstance(node, ast.expr)
        certain = _certain_argument_type(node, function)
        if certain is None:
            return None  # this argument may be the one: do not guess
        if canonical_annotation(certain) == canonical_annotation(got):
            culprits.append((name, node))
    if len(culprits) != 1:
        return None
    name, node = culprits[0]
    return _located(
        f"{label}(): {name}: expected {type_text(hints[name])}, got {got}",
        node,
        line,
        column,
        lines,
    )


def invalid_keyword(
    message: str,
    path: str | None,
    line: int | None,
    column: int | None,
    files: _Files,
    resolve_class: Resolver,
    resolve_callable: Resolver,
) -> tuple[str, int | None, int | None] | None:
    """``P(): 'c' is an invalid keyword argument; P takes (a: int)`` for
    Codon's ``'c' is an invalid keyword argument for __init__()``."""
    found = _INVALID_KEYWORD.match(message)
    parsed = files.get(path)
    if found is None or parsed is None or line is None:
        return None
    tree, lines = parsed
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and _plain_call(n)
        and any(k.arg == found["keyword"] and k.lineno == line for k in n.keywords)
    ]
    if len(calls) != 1:
        return None
    call = calls[0]
    callee = _own_callee(call, tree, resolve_class, resolve_callable)
    if callee is None:
        return None
    label, sig, hints = callee
    problem = _bind(
        f"{label}()",
        call,
        sig,
        hints,
        None,
        resolve_class,
        f"{label} takes ({signature_text(sig, hints)})",
    )
    if (
        problem is None
        or not isinstance(problem.node, ast.keyword)
        or problem.node.arg != found["keyword"]
    ):
        return None
    return _located(problem.message, problem.node, line, column, lines)


# ---------------------------------------------------------------------------
# A return of the wrong type (#528)
# ---------------------------------------------------------------------------


def _returning_function(
    tree: ast.Module, lines: list[str], line: int, column: int
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Return] | None:
    """The annotated function whose own ``return`` statement starts at
    (*line*, *column*), and that statement; ``None`` when that ``return``
    belongs to an unannotated nested function, or the function is a
    generator."""

    def search(
        node: ast.AST, owner: ast.FunctionDef | ast.AsyncFunctionDef | None
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, ast.Return] | None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found = search(child, child)
            elif isinstance(child, ast.Lambda):
                found = search(child, None)
            else:
                if (
                    isinstance(child, ast.Return)
                    and child.lineno == line
                    and _column(lines, line, child.col_offset) == column
                ):
                    return (owner, child)
                found = search(child, owner)
            if found is not None:
                return found
        return None

    hit = search(tree, None)
    if hit is None or hit[0] is None or hit[0].returns is None:
        return None
    function = hit[0]
    if any(
        isinstance(n, (ast.Yield, ast.YieldFrom))
        for stmt in function.body
        for n in _own_nodes(stmt)
    ):
        return None
    return function, hit[1]


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """*node* and its descendants outside nested functions and lambdas."""
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            yield from _own_nodes(child)


def qualified_name(tree: ast.Module, function: ast.AST) -> str:
    """``XScenario.f`` for a method, ``f`` for any other function."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and function in node.body:
            assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            return f"{node.name}.{function.name}"
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    return function.name


def return_message(
    message: str,
    path: str | None,
    line: int | None,
    column: int | None,
    files: _Files,
) -> str | None:
    """``val() returns str, but the function is declared -> float`` for Codon's
    ``'float' does not match expected type 'str'`` at a ``return`` statement."""
    found = _MISMATCH.match(message)
    none_returned = message == "cannot typecheck 'return None'"
    parsed = files.get(path)
    if not (found or none_returned) or parsed is None or line is None or column is None:
        return None
    tree, lines = parsed
    hit = _returning_function(tree, lines, line, column)
    if hit is None:
        return None
    function, statement = hit
    assert function.returns is not None
    name = qualified_name(tree, function)
    declared = ast.unparse(function.returns).strip("'\"")
    if found is None:
        return f"{name}() returns None, but the function is declared -> {declared}"
    # Codon's message is the other way round: the first type is the declared
    # one (here ``Optional[float]`` for ``-> float | None``, whichever way the
    # author spelled it), the second the type of the returned value.
    want = canonical_annotation(codon_type(found["declared"]))
    have = canonical_annotation(function.returns)
    if want.lower() != have.lower():
        head = want.split("[")[0]
        if "|" in want or head.lower() not in have.lower():
            return None  # not the declared type: another mismatch on this line
    shown = canonical_annotation(codon_type(found["got"]))
    if statement.value is not None and shown.endswith(" | None"):
        # Codon wraps the value in the declared Optional: say what it is.
        certain = _certain_argument_type(
            statement.value, _enclosing_function(tree, line)
        )
        if certain is not None and certain != "None":
            shown = certain
    return f"{name}() returns {shown}, but the function is declared -> {declared}"


# ---------------------------------------------------------------------------
# An annotated assignment of the wrong type (#564)
# ---------------------------------------------------------------------------


def annotated_message(
    message: str,
    path: str | None,
    line: int | None,
    column: int | None,
    files: _Files,
) -> str | None:
    """``speed is assigned a JSONValue, but is declared float`` for Codon's
    ``'float' does not match expected type 'JSONValue'`` at ``speed: float =
    doc["speed"]``: Codon names the declared type first, as for a ``return``."""
    found = _MISMATCH.match(message)
    parsed = files.get(path)
    if found is None or parsed is None or line is None or column is None:
        return None
    tree, lines = parsed
    statements = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AnnAssign)
        and n.value is not None
        and n.lineno == line
        and _column(lines, line, n.col_offset) == column
    ]
    if len(statements) != 1:
        return None
    statement = statements[0]
    want = canonical_annotation(codon_type(found["declared"]))
    have = canonical_annotation(statement.annotation)
    if want.lower() != have.lower():
        return None  # not the declared type: another mismatch on this line
    shown = canonical_annotation(codon_type(found["got"]))
    target = ast.unparse(statement.target)
    article = "an" if shown[:1] in "aeiouAEIOU" else "a"
    value = "None" if shown == "None" else f"{article} {shown}"
    return f"{target} is assigned {value}, but is declared {have}"


#: What to write instead of a use of a ``JSONValue`` the check refuses (#564).
_JSON_CONVERT = (
    "a value read from a JSON document has no type the check knows "
    '(JSONValue): convert it where you use it, `float(doc["speed"])`, '
    "`int(v)`, `str(v)`, `[float(x) for x in v]`"
)
_JSON_READ_ONLY = (
    "a JSON document is read-only here: copy what you need into a typed local"
)
_JSON_ATTRIBUTE = (
    'a JSON object has no attributes: index it, `v["x"]`, and convert what '
    "you take (JSONValue is none of the types the check knows)"
)


def json_hint(message: str) -> str:
    """*message* with a hint when it is about a ``JSONValue``."""
    if "JSONValue" not in message or "(JSONValue)" in message:
        return message
    if "'__setitem__'" in message or "'__delitem__'" in message:
        return f"{message}: {_JSON_READ_ONLY}"
    if "object has no attribute" in message:
        return f"{message}: {_JSON_ATTRIBUTE}"
    return f"{message}: {_JSON_CONVERT}"


# ---------------------------------------------------------------------------
# A forgotten ``self`` (#542)
# ---------------------------------------------------------------------------

#: ``'flag(XScenario)' object has no attribute '__bool__'``
_BOUND_METHOD_TRUTH = re.compile(
    r"^'(?P<name>\w+)\((?P<cls>\w+)\)' object has no attribute '__bool__'$"
)
_NOT_DEFINED = re.compile(r"^name '(?P<name>\w+)' is not defined$")


def _method_receiver(tree: ast.Module, line: int) -> tuple[str, str] | None:
    """(the class, the name of ``self``) of the method that holds *line*."""
    owner = _enclosing_class(tree, line)
    function = _enclosing_function(tree, line)
    if owner is None or function is None or not function.args.args:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == owner:
            if function in node.body:
                break
    else:
        return None
    if any(
        isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod")
        for d in function.decorator_list
    ):
        return None
    return owner, function.args.args[0].arg


def forgotten_self(
    message: str,
    path: str | None,
    line: int | None,
    column: int | None,
    files: _Files,
    resolve_class: Resolver,
) -> tuple[str, int | None, int | None] | None:
    """The hint for a bare ``flag()`` of a method (or attribute) of the
    enclosing class, and for ``if self.flag:`` where ``flag`` is a method."""
    parsed = files.get(path)
    if parsed is None or line is None:
        return None
    tree, lines = parsed
    undefined = _NOT_DEFINED.match(message)
    if undefined is not None:
        name = undefined["name"]
        receiver = _method_receiver(tree, line)
        if receiver is None:
            return None
        owner, self_name = receiver
        names = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Name)
            and n.id == name
            and n.lineno == line
            and (column is None or _column(lines, line, n.col_offset) == column)
        ]
        cls = resolve_class(owner)
        if len(names) != 1 or not isinstance(cls, type):
            return None
        called = any(
            isinstance(c, ast.Call) and c.func is names[0] for c in ast.walk(tree)
        )
        member = inspect.getattr_static(cls, name, None)
        suffix = "()" if called else ""
        if isinstance(member, (staticmethod, classmethod)) or inspect.isfunction(
            member
        ):
            what = f"{owner} has a method {name}()"
        elif name in getattr(cls, "__annotations__", {}) or member is not None:
            what = f"{owner} has an attribute {name}"
        else:
            return None
        fix = f"{self_name}.{name}{suffix}"
        return f"{message}: {what}, did you mean `{fix}`?", line, column
    bound = _BOUND_METHOD_TRUTH.match(message)
    if bound is None:
        return None
    cls = resolve_class(bound["cls"])
    if not isinstance(cls, type) or not inspect.isfunction(
        inspect.getattr_static(cls, bound["name"], None)
    ):
        return None
    callee_ids = {id(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
    uses = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and n.attr == bound["name"]
        and n.lineno == line
        and id(n) not in callee_ids
    ]
    if len(uses) != 1:
        return None
    use = uses[0]
    text = ast.unparse(use)
    return (
        f"`{text}` is a method, not its result (a method is always true): "
        f"call it, `{text}()`",
        use.lineno,
        _column(lines, use.lineno, use.col_offset),
    )


@dataclass
class Refiner:
    """Applies the rewrites above to a list of diagnostics."""

    #: (name, path of the file the diagnostic is in) -> the Python object
    resolve_class: Callable[[str, str | None], Any]
    resolve_callable: Callable[[str, str | None], Any]

    def __post_init__(self) -> None:
        self.files = _Files()

    def refine(
        self, message: str, path: str | None, line: int | None, column: int | None
    ) -> tuple[str, int | None, int | None]:
        try:
            changed = call_message(
                message,
                path,
                line,
                column,
                self.files,
                lambda name: self.resolve_class(name, path),
                lambda name: self.resolve_callable(name, path),
            )
            if changed is not None:
                return changed
            resolvers = (
                lambda name: self.resolve_class(name, path),
                lambda name: self.resolve_callable(name, path),
            )
            for rewrite in (argument_mismatch, invalid_keyword):
                changed = rewrite(message, path, line, column, self.files, *resolvers)
                if changed is not None:
                    return changed
            returned = return_message(message, path, line, column, self.files)
            if returned is None:
                returned = annotated_message(message, path, line, column, self.files)
            if returned is not None:
                return json_hint(returned), line, column
            changed = forgotten_self(
                message, path, line, column, self.files, resolvers[0]
            )
            if changed is not None:
                return changed
        except Exception:  # noqa: BLE001 - only a hint: never lose Codon's error
            pass
        return json_hint(message), line, column
