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
            label = member
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
# A return of the wrong type (#528)
# ---------------------------------------------------------------------------


def _returning_function(
    tree: ast.Module, lines: list[str], line: int, column: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The annotated function whose own ``return`` statement starts at
    (*line*, *column*); ``None`` when that ``return`` belongs to an
    unannotated nested function, or the function is a generator."""

    def search(
        node: ast.AST, owner: ast.FunctionDef | ast.AsyncFunctionDef | None
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None] | None:
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
                    return (owner,)
                found = search(child, owner)
            if found is not None:
                return found
        return None

    hit = search(tree, None)
    function = hit[0] if hit is not None else None
    if function is None or function.returns is None:
        return None
    if any(
        isinstance(n, (ast.Yield, ast.YieldFrom))
        for stmt in function.body
        for n in _own_nodes(stmt)
    ):
        return None
    return function


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """*node* and its descendants outside nested functions and lambdas."""
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            yield from _own_nodes(child)


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
    function = _returning_function(tree, lines, line, column)
    if function is None:
        return None
    declared = ast.unparse(function.returns).strip("'\"")  # type: ignore[arg-type]
    if found is None:
        return (
            f"{function.name}() returns None, but the function is declared "
            f"-> {declared}"
        )
    head = found["declared"].split("[")[0].replace("NoneType", "None")
    if head.lower() not in declared.lower():
        return None  # not the declared type: another mismatch on this line
    return (
        f"{function.name}() returns {codon_type(found['got'])}, but the function "
        f"is declared -> {declared}"
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
            returned = return_message(message, path, line, column, self.files)
            if returned is not None:
                return returned, line, column
        except Exception:  # noqa: BLE001 - only a hint: never lose Codon's error
            pass
        return message, line, column
