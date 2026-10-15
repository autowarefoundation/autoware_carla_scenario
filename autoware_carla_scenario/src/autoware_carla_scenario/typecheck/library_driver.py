"""The calls the library check compiles: every function, with typed values.

Codon only checks a function it compiles, and it compiles a function only
when something calls it.  To check a framework module on its own, the library
check (:func:`.check.typecheck_library`) appends a function to the module
that calls every function and every method of every class, each
with a value of every parameter's annotated type::

    def _acs_library_check():
        normalize_angle_deg(_acs_value(float))
        Vector3(_acs_value(float), _acs_value(float), _acs_value(float))
        _acs_value(Vector3).dot(_acs_value(Vector3))
        Vector3.zero()

``_acs_value(T)`` (the model's ``_tc.codon``) is a ``T`` read through a null
pointer: Codon types it as a ``T``, and nothing compiled is ever run.  The
calls are written in the module's own namespace, so an annotation means what
it means in the module.  A parameter annotated with a union
(``Lanelet2Pose | OpenDrivePose``, or ``Optional[Union[A, B]]``) is called
once with each member.  A function with a defaulted parameter is called
once more with only the arguments that have none (a method through its
class), so its defaults are compiled too (:meth:`_Renderer._defaults_call`),
and a dataclass with a defaulted field is constructed once more with only
the fields that have none (:meth:`_Renderer._required_fields_call`).

Protected (``_``) functions and methods are called too: they are the hooks a
subclass or another module uses (``BaseScenario._setup_ego_spawn``), and
Codon would compile one only if a public function called it.  A module's
dunders (PEP 562's ``__getattr__``) are Python's hooks, not the module's
code, and are left out; a class's (``__eq__``, ``__enter__``) are called.
The methods of a ``_`` class are called as a public class's are: what uses
such a helper may not compile it (a constructor that is ``UNCALLED``, a call
behind a ``hasattr`` that Codon decides statically, a dataclass's
``__post_init__``, which Codon's constructor does not call).

The class-level defaults of a class with a constructor (``STABILIZE_TICKS:
int = 5``) are assigned to typed locals (:meth:`_Renderer._class_defaults`):
Codon makes such a member a field and drops its value, so nothing else
compiles it::

        _acs_cls_default_12: int = 5

Some annotations give the check nothing to call with (``Any``, ``object``,
``Callable[..., R]``, none at all): each is reported as a problem of the
module, to be fixed there with a type Codon can check.  ``object`` is
accepted for the other operand of ``__eq__`` and ``__ne__``, which Python
requires; it is given a value of the class itself.  A ``Callable[[A], R]``
with types Codon can express is given a value of Codon's ``Callable`` type
(:func:`callable_annotation`); one with a union among its argument types,
which Codon 0.19 crashes on, a callable of as many arguments, of any type,
returning an ``R`` (:func:`union_callable_annotation`).
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from .driver import Driver
from .transform import (
    PRELUDE,
    _ENUM_BASES,
    _TYPE_VAR_NAMES,
    _dotted,
    _is_none,
    _members_or_self,
    _names_any,
    _union_members,
    callable_annotation,
    codon_annotation,
    uncalled_nodes,
    union_callable_annotation,
)

__all__ = [
    "LIBRARY_CHECK",
    "LibraryChecks",
    "callable_annotation",
    "render_library_checks",
    "union_callable_annotation",
    "render_library_driver",
    "uncalled_nodes",
    "workspace_module",
]

#: The function appended to every checked module.
LIBRARY_CHECK = "_acs_library_check"
#: The prefix of the local a class-level default is assigned to.
CLASS_DEFAULT = "_acs_cls_default"
#: Where the checked modules live in the check's workspace.
LIBRARY_PACKAGE = "_acs_lib"
#: The most calls made for one function: past it, a union parameter is
#: varied one at a time, the others taking their first member.
MAX_CALLS = 16

_DATACLASS = {"dataclass", "dataclasses.dataclass"}
_FIELD_FUNCTIONS = {"field", "dataclasses.field"}
#: OmegaConf's mandatory-value marker: a field "defaulted" to it is required.
_MANDATORY = {"MISSING"}
_SKIPPED_DECORATORS = {"overload", "typing.overload"}
_PROPERTIES = {"property", "functools.cached_property", "cached_property"}
_CLASS_VAR = {"ClassVar", "typing.ClassVar"}
_OBJECT_OPERAND = {"__eq__", "__ne__"}


def workspace_module(module: str) -> str:
    """The name a checked framework module is compiled under.

    Codon names a file only by its base name in its errors, and a package
    has many ``frames.py`` and ``__init__.py``: each checked module is one
    file of :data:`LIBRARY_PACKAGE` named by its whole dotted name
    (``autoware_carla_scenario.coordinate.frames`` ->
    ``_acs_lib.autoware_carla_scenario__coordinate__frames``).
    """
    return f"{LIBRARY_PACKAGE}.{module.replace('.', '__')}"


@dataclass
class LibraryChecks:
    """The function appended to one checked module."""

    source: str
    #: Line of :attr:`source` (1-based) -> the call on it, as the author reads it.
    labels: dict[int, str] = field(default_factory=dict)
    #: (line of the module, message): a parameter the check cannot call with.
    problems: list[tuple[int, str]] = field(default_factory=list)


def _called_vararg(vararg: ast.arg | None) -> ast.arg | None:
    """``*args: T``, which the check passes one T; ``None`` for any other.

    A ``*args`` with no annotation, or of ``object`` or ``Any``
    (``__exit__(self, *args: object)``), is given nothing, as before: there
    is no value of it to give.
    """
    if vararg is None or vararg.annotation is None:
        return None
    if _dotted(vararg.annotation) in ("object", "Any", "typing.Any"):
        return None
    return vararg


class _Renderer:
    def __init__(
        self,
        uncalled: Collection[str] = (),
        classes: Mapping[str, ast.ClassDef] | None = None,
        *,
        constructed: Collection[str] = (),
        type_vars: Collection[str] = (),
    ) -> None:
        self.uncalled = set(uncalled)
        #: Classes by name, for the base classes of a dataclass and the
        #: annotation a class-level default overrides.
        self.classes: dict[str, ast.ClassDef] = dict(classes or {})
        #: Classes the module's top level constructs with no argument.
        self.constructed = set(constructed)
        #: The module's ``TypeVar``s, which Codon cannot type a local with.
        self.type_vars = set(type_vars)
        # `_acs_Returning` comes with the transform's PRELUDE, which a
        # parameter's default may name above this function; imported again
        # here, Codon 0.19 cannot type the module ("cannot typecheck").
        self.lines = [
            "",
            "",
            "from autoware_carla_scenario._tc import _acs_value, _expect_bool",
            "",
            "",
            f"def {LIBRARY_CHECK}():",
        ]
        self.labels: dict[int, str] = {}
        self.problems: list[tuple[int, str]] = []

    def emit(self, text: str, label: str) -> None:
        self.lines.append(f"    {text}")
        self.labels[len(self.lines)] = label

    def types(self, arg: ast.arg, owner: str | None, where: str) -> list[str] | None:
        """The Codon types to call with for *arg*; ``None`` after a problem."""
        annotation = arg.annotation
        if annotation is None:
            self.problems.append(
                (arg.lineno, f"{where}: parameter `{arg.arg}` has no annotation")
            )
            return None
        text = (
            codon_annotation(annotation)
            or callable_annotation(annotation)
            or union_callable_annotation(annotation)
        )
        if text is not None:
            return [text]
        members = _union_members(annotation)
        if (
            members is None
            and isinstance(annotation, ast.Subscript)
            and _dotted(annotation.value) in ("Optional", "typing.Optional")
        ):
            # Optional[Callable[...]] or Optional[Union[A, B]], which
            # codon_annotation drops: called with the callable, or with each
            # member, as `Callable[...] | None` and `A | B | None` are.
            members = _members_or_self(annotation)
        if members is not None:
            texts = [
                codon_annotation(m, class_level=True)
                or callable_annotation(m)
                or union_callable_annotation(m)
                for m in members
                if not _is_none(m)
            ]
        else:
            texts = [codon_annotation(annotation, class_level=True)]
        if texts and all(t is not None for t in texts):
            return [t for t in texts if t is not None]
        method = where.rsplit(".", 1)[-1]
        if owner is not None and method in _OBJECT_OPERAND:
            if _dotted(annotation) == "object":
                return [owner]
        self.problems.append(
            (
                arg.lineno,
                f"{where}: Codon cannot express `{ast.unparse(annotation)}`, the "
                f"annotation of parameter `{arg.arg}`; annotate it with a type "
                "Codon can check",
            )
        )
        return None

    def calls(
        self,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        head: str,
        where: str,
        *,
        drop_first: bool,
        owner: str | None = None,
        receiver: str | None = None,
        class_head: str | None = None,
    ) -> None:
        """Emit the calls of *func*, and one leaving its defaults out.

        *class_head* (``Class.method``) is what a method called on a value
        (*head*, ``_acs_value(Class).method``) is called through to leave its
        defaults out: its dispatch thunk fills in none.
        """
        args = func.args
        positional = [*args.posonlyargs, *args.args][1 if drop_first else 0 :]
        vararg = _called_vararg(args.vararg)
        if vararg is not None:
            # `*args: T`, given one T, so the body is compiled for one.
            positional.append(vararg)
        every = [*positional, *args.kwonlyargs]
        choices = [self.types(arg, owner, where) for arg in every]
        if any(c is None for c in choices):
            return
        typed = [c for c in choices if c]
        self._emit_calls(
            head,
            where,
            positional,
            args.kwonlyargs,
            typed,
            receiver=receiver,
        )
        self._defaults_call(
            func,
            dict(zip((a.arg for a in every), typed)),
            where,
            positional,
            head=class_head or head,
            receiver=receiver if class_head is None else f"_acs_value({owner})",
        )

    def _defaults_call(
        self,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        typed: dict[str, list[str]],
        where: str,
        positional: list[ast.arg],
        *,
        head: str,
        receiver: str | None,
    ) -> None:
        """One call more, leaving out every argument that has a default.

        The calls of :meth:`_emit_calls` pass every argument, so no default
        is compiled: this one leaves them out (each parameter given its
        first type), so Codon compiles each default against the body, as a
        caller that leaves it out has it compiled.  A method is called
        through its class (*head*), as a dispatch thunk fills in no default.
        A ``*args: T`` is still given one ``T``, so the positional
        parameters before it are all passed.  Nothing is emitted for a
        function without a default.
        """
        args = func.args
        every = [*args.posonlyargs, *args.args]
        defaulted = {a.arg for a in every[len(every) - len(args.defaults) :]}
        defaulted |= {
            a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None
        }
        vararg = _called_vararg(args.vararg)
        passed_positional = [
            a for a in positional if vararg is not None or a.arg not in defaulted
        ]
        passed_keyword = [a for a in args.kwonlyargs if a.arg not in defaulted]
        left_out = [
            a.arg
            for a in [*positional, *args.kwonlyargs]
            if a not in passed_positional and a not in passed_keyword
        ]
        if not left_out:
            return
        values = [receiver] if receiver is not None else []
        values += [f"_acs_value({typed[a.arg][0]})" for a in passed_positional]
        values += [f"{a.arg}=_acs_value({typed[a.arg][0]})" for a in passed_keyword]
        signature = ", ".join(
            f"{a.arg}: {typed[a.arg][0]}" for a in [*passed_positional, *passed_keyword]
        )
        self.emit(
            f"{head}({', '.join(values)})",
            f"{where}({signature}), with the default of "
            + ", ".join(f"`{name}`" for name in left_out),
        )

    def _emit_calls(
        self,
        head: str,
        where: str,
        positional: list[ast.arg],
        keyword: list[ast.arg],
        choices: list[list[str]],
        *,
        receiver: str | None = None,
    ) -> None:
        for combo in _combinations(choices):
            values = [receiver] if receiver is not None else []
            values += [f"_acs_value({t})" for t in combo[: len(positional)]]
            values += [
                f"{a.arg}=_acs_value({t})"
                for a, t in zip(keyword, combo[len(positional) :])
            ]
            signature = ", ".join(
                f"{a.arg}: {t}" for a, t in zip([*positional, *keyword], combo)
            )
            self.emit(f"{head}({', '.join(values)})", f"{where}({signature})")

    def function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name not in self.uncalled:
            self.calls(node, node.name, node.name, drop_first=False)

    def cls(self, node: ast.ClassDef) -> None:
        name = node.name
        bases = [(_dotted(b) or "").rsplit(".", 1)[-1] for b in node.bases]
        is_enum = any(b in _ENUM_BASES for b in bases)
        decorators = {
            _dotted(d.func if isinstance(d, ast.Call) else d)
            for d in node.decorator_list
        }
        methods = [
            m
            for m in node.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        init = next((m for m in methods if m.name == "__init__"), None)
        if f"{name}.__init__" in self.uncalled:
            pass
        elif init is not None and not is_enum:
            self.calls(init, name, f"{name}.__init__", drop_first=True, owner=name)
        elif decorators & _DATACLASS and not is_enum:
            self._dataclass_init(node)
        if not is_enum and not decorators & _DATACLASS:
            self._class_defaults(node)
        for method in methods:
            if method.name == "__init__":
                continue
            kinds = {_dotted(d) or "" for d in method.decorator_list}
            where = f"{name}.{method.name}"
            if where in self.uncalled:
                continue
            if kinds & _SKIPPED_DECORATORS or any(
                k.endswith((".setter", ".deleter")) for k in kinds
            ):
                continue
            if kinds & _PROPERTIES:
                self.emit(f"_acs_value({name}).{method.name}", where)
            elif "staticmethod" in kinds:
                self.calls(method, f"{name}.{method.name}", where, drop_first=False)
            elif "classmethod" in kinds:
                self.calls(method, f"{name}.{method.name}", where, drop_first=True)
            elif method.args.kwonlyargs or method.args.kwarg is not None:
                # Through the class: Codon 0.19 calls a method a subclass may
                # override through a dispatch thunk, which takes neither a
                # keyword argument nor **kwargs.
                self.calls(
                    method,
                    f"{name}.{method.name}",
                    where,
                    drop_first=True,
                    owner=name,
                    receiver=f"_acs_value({name})",
                )
            else:
                self.calls(
                    method,
                    f"_acs_value({name}).{method.name}",
                    where,
                    drop_first=True,
                    owner=name,
                    class_head=f"{name}.{method.name}",
                )

    def _class_defaults(self, node: ast.ClassDef) -> None:
        """A typed local for each class-level default of *node* (#190).

        On a class with a constructor, its own or a base's, Codon makes
        ``x: T = v`` a field and drops ``v``, so no call compiles it.  Each is
        assigned to a local of its type instead, ``_acs_cls_default_N: T =
        v``, which compiles ``v`` against ``T``.  A default written without
        an annotation (``REISSUES_BY_DEFAULT = True``) overrides one a base
        declares, and is given the base's annotation, a ``ClassVar[T]``'s
        ``T``.

        Left out, as compiled already or as having no type to check:

        - a ``ClassVar``, which Codon keeps as a class variable, with its value;
        - every default of a class with no constructor that the module's top
          level constructs with no argument (``_CACHE = _Cache()``): Codon's
          generated constructor takes each field with its default, and that
          call compiles them;
        - a default without an annotation whose name no base declares (a base
          not checked, or not a class of a checked module), or a dunder
          (``__slots__``);
        - an annotation Codon cannot express, or naming a ``TypeVar``;
        - a value or annotation naming another name the class body binds (a
          member, unpacked ones included, a method, a nested class), which
          means something else at the function's scope.

        A dataclass's defaults are compiled through its constructor
        (:meth:`_required_fields_call`), and an enum's members are values.
        """
        if node.name in self.constructed and not self._defines_init(node, set()):
            return  # Codon's generated constructor takes each default
        members = _class_body_names(node)
        for stmt in node.body:
            annotation: ast.expr | None
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.value is None or _is_class_var(stmt.annotation):
                    continue
                name, annotation = stmt.target.id, stmt.annotation
                assigned: ast.expr = stmt.value
            elif (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and not _is_dunder(stmt.targets[0].id)
            ):
                name, assigned = stmt.targets[0].id, stmt.value
                inherited = self._inherited_annotation(node, name, set())
                if inherited is None:
                    continue
                annotation = _class_var_type(inherited)
                if annotation is None:
                    continue  # a bare `ClassVar`: no type to check against
            else:
                continue
            if _names(annotation) & self.type_vars or (
                (_names(annotation) | _names(assigned)) & members
            ):
                continue
            typed = codon_annotation(annotation, class_level=True)
            if typed is None:
                continue
            value = ast.unparse(assigned)
            if typed == "bool":
                # Codon converts any value assigned to a `bool` with its
                # `__bool__` (`x: bool = "x"` compiles); a call does not.
                value = f"_expect_bool({value})"
            self.emit(
                f"{CLASS_DEFAULT}_{len(self.lines) + 1}: {typed} = {value}",
                f"class-level default of {node.name}.{name}",
            )

    def _defines_init(self, node: ast.ClassDef, seen: set[str]) -> bool:
        """Whether *node* or a base defines ``__init__``; a base not found may."""
        if any(
            isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            and m.name == "__init__"
            for m in node.body
        ):
            return True
        seen = seen | {node.name}
        for base_node in node.bases:
            if _dotted(base_node) in ("object", "builtins.object"):
                continue
            base = self.classes.get((_dotted(base_node) or "").rsplit(".", 1)[-1])
            if base is None or base.name in seen or self._defines_init(base, seen):
                return True
        return False

    def _inherited_annotation(
        self, node: ast.ClassDef, name: str, seen: set[str]
    ) -> ast.expr | None:
        """The annotation a base of *node* declares *name* with, or ``None``.

        The bases are searched depth first, left to right, as Python's method
        resolution order finds a member in a single inheritance chain.
        """
        seen = seen | {node.name}
        for base_node in node.bases:
            base = self.classes.get((_dotted(base_node) or "").rsplit(".", 1)[-1])
            if base is None or base.name in seen:
                continue
            for stmt in base.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == name
                ):
                    return stmt.annotation
            found = self._inherited_annotation(base, name, seen)
            if found is not None:
                return found
        return None

    def _dataclass_init(self, node: ast.ClassDef) -> None:
        """The generated ``__init__`` of a dataclass: one value per field.

        A field annotated with a union (``A | B``) is given each member in
        turn, as a union parameter is.  A field with a default whose
        annotation names ``Any`` (``dict[str, Any]``, a free-form Hydra
        mapping) is left out of the call, as the model leaves it out of its
        class, and the fields after it are passed by keyword; the checked
        class keeps it, with a default of its type (``_field_default``).

        The fields of a dataclass base class come first, as in Python's
        constructor and Codon's, a field redefined keeping its place.  Of
        several bases with fields, Python takes the last one's first and
        Codon 0.19 the first one's: such a constructor is a problem.
        """
        fields = self._fields(node, set())
        if fields is None:
            self.problems.append(
                (
                    node.lineno,
                    f"{node.name}: Codon 0.19 orders the fields of several "
                    "dataclass bases otherwise than Python; list "
                    f"`{node.name}.__init__` as UNCALLED",
                )
            )
            return
        positional: list[ast.arg] = []
        keyword: list[ast.arg] = []
        choices: list[list[str]] = []
        left_out = False
        for arg, _ in fields.values():
            if arg.arg == _LEFT_OUT:
                left_out = True
                continue
            texts = self._field_types(arg.annotation)
            if texts is None:
                self.problems.append(
                    (
                        arg.lineno,
                        f"{node.name}: Codon cannot express the annotation of "
                        f"field `{arg.arg}`; annotate it with a type Codon can check",
                    )
                )
                return
            choices.append(texts)
            # Past a field left out, the fields are passed by keyword.
            (keyword if left_out else positional).append(arg)
        self._emit_calls(
            node.name, f"{node.name}.__init__", positional, keyword, choices
        )
        self._required_fields_call(
            node.name,
            fields,
            dict(zip((a.arg for a in [*positional, *keyword]), choices)),
        )

    def _required_fields_call(
        self,
        name: str,
        fields: dict[str, tuple[ast.arg, bool]],
        typed: dict[str, list[str]],
    ) -> None:
        """One constructor call more, passing only the fields with no default.

        The calls of :meth:`_dataclass_init` pass every field, so no field
        default is compiled: this one leaves them out (each field given its
        first type), so Codon compiles each default against the field's
        type, as a caller that leaves it out has it compiled.  A required
        field after a defaulted one (a ``kw_only`` field) is passed by
        keyword.  Nothing is emitted for a dataclass without a default.
        """
        passed_positional: list[str] = []
        passed_keyword: list[str] = []
        left_out: list[str] = []
        seen_default = False
        for field_name, (arg, has_default) in fields.items():
            if has_default:
                seen_default = True
                if arg.arg != _LEFT_OUT:
                    left_out.append(field_name)
                continue
            (passed_keyword if seen_default else passed_positional).append(field_name)
        if not left_out:
            return

        values = [f"_acs_value({typed[f][0]})" for f in passed_positional]
        values += [f"{f}=_acs_value({typed[f][0]})" for f in passed_keyword]
        signature = ", ".join(
            f"{f}: {typed[f][0]}" for f in [*passed_positional, *passed_keyword]
        )
        self.emit(
            f"{name}({', '.join(values)})",
            f"{name}.__init__({signature}), with the default of "
            + ", ".join(f"`{f}`" for f in left_out),
        )

    def _fields(
        self, node: ast.ClassDef, seen: set[str]
    ) -> dict[str, tuple[ast.arg, bool]] | None:
        """The fields of the dataclass *node*, its bases' first, by name.

        Each with whether it has a default.  A field the call leaves out is
        an ``ast.arg`` named :data:`_LEFT_OUT`.  ``None`` when more than one
        base brings fields (Python and Codon 0.19 order them differently).
        """
        seen = seen | {node.name}
        fields: dict[str, tuple[ast.arg, bool]] = {}
        inherited: list[dict[str, tuple[ast.arg, bool]]] = []
        for base in [b for n in node.bases for b in self._dataclasses_of(n, seen)]:
            base_fields = self._fields(base, seen)
            if base_fields is None:
                return None
            if base_fields:
                inherited.append(base_fields)
        if len(inherited) > 1:
            return None
        for base_fields in inherited:
            fields.update(base_fields)
        for stmt in node.body:
            if not (
                isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ):
                continue
            if (_dotted(stmt.annotation) or "").endswith("KW_ONLY"):
                break  # keyword-only fields: not modelled, so not called
            if isinstance(stmt.annotation, ast.Subscript) and (
                _dotted(stmt.annotation.value) or ""
            ).endswith("ClassVar"):
                continue
            value = stmt.value
            if isinstance(value, ast.Call) and any(
                kw.arg == "init"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in value.keywords
            ):
                # Not in the constructor, even redefining a base's field.
                fields.pop(stmt.target.id, None)
                continue
            name = stmt.target.id
            if value is not None and _names_any(stmt.annotation):
                fields[name] = (ast.arg(_LEFT_OUT, stmt.annotation), True)
                continue
            arg = ast.arg(name, stmt.annotation)
            arg.lineno = stmt.lineno
            fields[name] = (arg, _has_default(value))
        return fields

    def _dataclasses_of(
        self, base_node: ast.expr, seen: set[str]
    ) -> list[ast.ClassDef]:
        """The dataclasses whose fields the base class *base_node* brings.

        The base itself if it is a dataclass; for a plain class, the
        dataclasses it derives from, as Python finds the fields through it.
        """
        base = self.classes.get((_dotted(base_node) or "").rsplit(".", 1)[-1])
        if base is None or base.name in seen:
            return []
        if _is_dataclass(base):
            return [base]
        return [
            b for n in base.bases for b in self._dataclasses_of(n, seen | {base.name})
        ]

    @staticmethod
    def _field_types(annotation: ast.expr | None) -> list[str] | None:
        """The Codon types to construct a dataclass field with, or ``None``."""
        if annotation is None:
            return None
        text = codon_annotation(annotation, class_level=True) or callable_annotation(
            annotation
        )
        if text is not None:
            return [text]
        members = _union_members(annotation)
        if members is None:
            return None
        texts = [
            codon_annotation(m, class_level=True) or callable_annotation(m)
            for m in members
            if not _is_none(m)
        ]
        if not texts or any(t is None for t in texts):
            return None
        return [t for t in texts if t is not None]


#: The name of a dataclass field the constructor call leaves out.
_LEFT_OUT = "_acs_left_out"


def _has_default(value: ast.expr | None) -> bool:
    """Whether a dataclass field assigned *value* has a default.

    OmegaConf's mandatory-value marker (``spawn_lanelet_id: int = MISSING``,
    the string ``"???"``) is none: it is a placeholder for a value the config
    must supply, not one of the field's type, so the call passes the field
    and the marker is not compiled.  Python does not enforce it: a dataclass
    built without the field (``NpcVehicleConfig()``, or from
    ``OmegaConf.to_container()``) holds the string, unchecked.
    """
    if value is None or (_dotted(value) or "").rsplit(".", 1)[-1] in _MANDATORY:
        return False
    if isinstance(value, ast.Call) and _dotted(value.func) in _FIELD_FUNCTIONS:
        return any(kw.arg in ("default", "default_factory") for kw in value.keywords)
    return True


def _is_dataclass(node: ast.ClassDef) -> bool:
    return bool(
        {_dotted(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list}
        & _DATACLASS
    )


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _is_class_var(annotation: ast.expr) -> bool:
    """Whether *annotation* is ``ClassVar[T]`` or a bare ``ClassVar``."""
    if isinstance(annotation, ast.Subscript):
        annotation = annotation.value
    return _dotted(annotation) in _CLASS_VAR


def _class_var_type(annotation: ast.expr) -> ast.expr | None:
    """The type of a member declared *annotation*: ``T`` of ``ClassVar[T]``.

    ``None`` for a bare ``ClassVar``, which names no type.
    """
    if not _is_class_var(annotation):
        return annotation
    return annotation.slice if isinstance(annotation, ast.Subscript) else None


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _class_body_names(node: ast.ClassDef) -> set[str]:
    """Every name *node*'s body binds: what its class-level values may name.

    Assigned names, unpacked ones included (``LOW, HIGH = 0.0, 1.0``), and
    the methods, nested classes and imports of the body; not the variables of
    a comprehension or a lambda, which are their own scope.
    """
    out: set[str] = set()
    stack: list[ast.AST] = list(node.body)
    while stack:
        sub = stack.pop()
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(sub.name)
            stack.extend(sub.decorator_list)
            continue
        if isinstance(sub, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in sub.names)
            continue
        if isinstance(
            sub, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            continue
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            out.add(sub.id)
        stack.extend(ast.iter_child_nodes(sub))
    return out


def _module_level_nodes(tree: ast.Module) -> list[ast.AST]:
    """Every node the module runs when imported, outside a function or class."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _combinations(choices: list[list[str]]) -> list[tuple[str, ...]]:
    total = 1
    for c in choices:
        total *= len(c)
    if total <= MAX_CALLS:
        return list(itertools.product(*choices))
    first = tuple(c[0] for c in choices)
    out = [first]
    for i, c in enumerate(choices):
        for alternative in c[1:]:
            out.append(first[:i] + (alternative,) + first[i + 1 :])
    return out


def render_library_checks(
    tree: ast.Module,
    uncalled: Collection[str] = (),
    classes: Mapping[str, ast.ClassDef] | None = None,
) -> LibraryChecks:
    """The function to append to the module *tree* is the source of.

    *uncalled* names the functions the check leaves out (``function`` or
    ``Class.method``; ``Class.__init__`` for the constructor, generated or
    not), as :data:`.library.UNCALLED` lists them.  *classes* are the classes
    of the other checked modules, by name: a dataclass derived from one of
    them is constructed with its fields too.
    """
    own = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    constructed = {
        node.func.id
        for node in _module_level_nodes(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.args
        and not node.keywords
    }
    type_vars = {
        target.id
        for stmt in tree.body
        if isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and _dotted(stmt.value.func) in _TYPE_VAR_NAMES
        for target in stmt.targets
        if isinstance(target, ast.Name)
    }
    renderer = _Renderer(
        uncalled,
        {**(classes or {}), **own},
        constructed=constructed,
        type_vars=type_vars,
    )
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_dunder(node.name):
                renderer.function(node)
        elif isinstance(node, ast.ClassDef):
            renderer.cls(node)
    renderer.lines.append("    pass")
    return LibraryChecks(
        "\n".join(renderer.lines) + "\n", renderer.labels, renderer.problems
    )


def render_library_driver(modules: list[str]) -> Driver:
    """The program that runs the appended function of each module in *modules*.

    It imports the model through the transform's prelude first, so the model
    is read before any checked module whichever module comes first.  The
    model reaches a checked module through a replaced model module
    (``_poses.codon`` re-exports ``coordinate.frames``); read from the
    checked module instead, the prelude's import of the model would come back
    to that module before it defines anything (a run of
    ``coordinate.poses`` alone, which starts at ``coordinate.frames``).
    """
    lines: list[str] = PRELUDE.splitlines()
    calls: list[str] = []
    for i, module in enumerate(modules):
        lines.append(
            f"from {workspace_module(module)} import {LIBRARY_CHECK} as _acs_check_{i}"
        )
        calls.append(f"_acs_check_{i}()")
    return Driver("\n".join([*lines, "", "", *calls]) + "\n")
