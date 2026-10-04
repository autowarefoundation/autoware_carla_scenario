"""The Codon model declares the public API as the Python package defines it.

The static check compiles scenarios against ``typecheck/codon/`` instead of
the Python package, so a parameter renamed in Python and not in the model
would reject a correct scenario (or accept a wrong one).  These tests read
the model as Python syntax -- it is written to parse as Python -- and compare
every class and function ``autoware_carla_scenario/__init__.codon`` exports
with the object of the same name: its parameters (names, order, which are
keyword-only, which have defaults), the methods and properties it declares,
its enum members.  They need no Codon.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import typing
from pathlib import Path
from typing import Any

import pytest

import autoware_carla_scenario as acs
from autoware_carla_scenario.typecheck import model_dir

_MODEL = model_dir() / "autoware_carla_scenario"

#: Python members the model leaves out on purpose, by class.  Each is a
#: free-form mapping (dict[str, Any]), which has no Codon type.
_OMITTED_PARAMETERS: dict[str, set[str]] = {
    "SweepConfig": {"constraints", "bindings"},
}

#: (kind, name, has default): kind is "pos", "kw", "*args" or "**kwargs".
Param = tuple[str, str, bool]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _exports() -> dict[str, str]:
    """Name -> model module (stem) for every name ``__init__.codon`` imports."""
    out: dict[str, str] = {}
    for node in _parse(_MODEL / "__init__.codon").body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = node.module
    return out


def _definitions(module: str) -> dict[str, list[ast.stmt]]:
    """Top-level definitions of a model module (a function may be overloaded)."""
    out: dict[str, list[ast.stmt]] = {}
    for node in _parse(_MODEL / f"{module}.codon").body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            out.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for target in targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(node)
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                out.setdefault(alias.asname or alias.name, []).extend(
                    _definitions(node.module).get(alias.name, [])
                )
    return out


def _is_required_marker(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_Required"
    )


def _model_params(fn: ast.FunctionDef, *, drop_first: bool) -> list[Param]:
    args = fn.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    out: list[Param] = [
        ("pos", a.arg, d is not None and not _is_required_marker(d))
        for a, d in zip(positional, defaults)
    ]
    if drop_first:
        out = out[1:]
    if args.vararg is not None and args.vararg.arg != "_kw":
        out.append(("*args", args.vararg.arg, False))
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        out.append(("kw", a.arg, d is not None and not _is_required_marker(d)))
    if args.kwarg is not None:
        out.append(("**kwargs", args.kwarg.arg, False))
    return out


def _model_fields(cls: ast.ClassDef) -> list[Param]:
    """The generated __init__ of a model class without one: its fields."""
    out: list[Param] = []
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if "ClassVar" in ast.unparse(stmt.annotation):
                continue
            out.append(("pos", stmt.target.id, stmt.value is not None))
    return out


_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: "pos",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "pos",
    inspect.Parameter.KEYWORD_ONLY: "kw",
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
}


def _python_params(obj: Any, *, drop_first: bool) -> list[Param]:
    params = list(inspect.signature(obj).parameters.values())
    if drop_first:
        params = params[1:]
    return [
        (_KINDS[p.kind], p.name, p.default is not inspect.Parameter.empty)
        for p in params
    ]


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    out: dict[str, ast.FunctionDef] = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            out.setdefault(stmt.name, stmt)  # a property's getter, not its setter
    return out


def _decorators(fn: ast.FunctionDef) -> set[str]:
    return {ast.unparse(d) for d in fn.decorator_list}


def _class_vars(cls: ast.ClassDef) -> set[str]:
    return {
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and "ClassVar" in ast.unparse(stmt.annotation)
    }


_EXPORTS = _exports()


def test_the_model_exports_only_names_the_package_exports() -> None:
    unknown = sorted(set(_EXPORTS) - set(acs.__all__) - set(acs._LAZY_IMPORTS))
    assert not unknown, f"the model exports names the package does not: {unknown}"


@pytest.mark.parametrize("name", sorted(_EXPORTS))
def test_an_exported_name_is_declared_as_in_python(name: str) -> None:
    definitions = _definitions(_EXPORTS[name]).get(name)
    assert definitions, f"{name} is imported by __init__.codon but not defined"
    python = getattr(acs, name)
    node = definitions[0]

    if isinstance(node, ast.FunctionDef):
        assert inspect.isfunction(python), f"{name} is a function in the model only"
        expected = _python_params(python, drop_first=False)
        for overload in definitions:
            assert isinstance(overload, ast.FunctionDef)
            assert _model_params(overload, drop_first=False) == expected, name
        return

    if not isinstance(node, ast.ClassDef):
        return  # a constant (EGO_ROLE_NAME): its type is the class it is built from

    if typing.get_origin(python) is typing.Union:
        # A Union alias (SpawnLocation): the base class of its members here.
        for member in typing.get_args(python):
            classes = _definitions(_EXPORTS[member.__name__]).get(member.__name__, [])
            assert classes and isinstance(classes[0], ast.ClassDef)
            assert name in {
                ast.unparse(b) for b in classes[0].bases
            }, f"{member.__name__} does not derive from {name} in the model"
        return

    assert inspect.isclass(python), f"{name} is a class in the model only"
    if issubclass(python, enum.Enum):
        assert _class_vars(node) == {
            m.name for m in python
        }, f"{name}: the model's members differ from the enum's"
        return

    methods = _methods(node)
    if "__init__" in methods:
        model_init = _model_params(methods["__init__"], drop_first=True)
    else:
        model_init = _model_fields(node)
    omitted = _OMITTED_PARAMETERS.get(name, set())
    python_init = [
        p for p in _python_params(python, drop_first=False) if p[1] not in omitted
    ]
    assert model_init == python_init, f"{name}(): the model's parameters differ"

    if dataclasses.is_dataclass(python) and "__init__" not in methods:
        python_fields = {f.name for f in dataclasses.fields(python)}
        declared = {p[1] for p in _model_fields(node)}
        assert declared <= python_fields, f"{name}: fields the dataclass does not have"

    for attr in _class_vars(node):
        assert hasattr(python, attr), f"{name}.{attr} is in the model only"

    for method_name, method in methods.items():
        if method_name.startswith("__") or method_name.startswith("_acs_"):
            continue
        assert hasattr(
            python, method_name
        ), f"{name}.{method_name} is in the model only"
        python_attr = inspect.getattr_static(python, method_name)
        if "property" in _decorators(method):
            assert isinstance(
                python_attr, property
            ), f"{name}.{method_name} is a property"
            continue
        if "staticmethod" in _decorators(method):
            # A classmethod in Python (EntityRole.ego): bound, it takes the same.
            assert isinstance(
                python_attr, (staticmethod, classmethod)
            ), f"{name}.{method_name} is not a static or class method in Python"
            assert _model_params(method, drop_first=False) == _python_params(
                getattr(python, method_name), drop_first=False
            ), f"{name}.{method_name}(): the model's parameters differ"
            continue
        assert _model_params(method, drop_first=True) == _python_params(
            getattr(python, method_name), drop_first=True
        ), f"{name}.{method_name}(): the model's parameters differ"


def test_every_model_file_parses_as_python() -> None:
    # The tests above read the model with ast; keep it valid Python syntax.
    for path in sorted(model_dir().rglob("*.codon")):
        _parse(path)
