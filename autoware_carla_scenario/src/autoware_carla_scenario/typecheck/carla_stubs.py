"""Type stubs of ``typesafe_carla.carla`` for the None check (#267).

typesafe_carla's CPython package is a prebuilt extension with no ``.pyi`` and
no ``py.typed``, so mypy reads ``typesafe_carla.carla`` as ``Any``: an
unguarded ``world.get_map().get_waypoint(loc, project_to_road=False)
.transform`` passed both the compile (Codon unwraps an ``Optional``
implicitly) and the None check.  This module writes a stub package for it
from typesafe_carla's own inventory of its Codon library
(``typesafe_carla.pycarla.Inventory``, which the CPython package is generated
from), and :func:`.none_check.run_none_check` puts it on ``MYPYPATH``.

The stubs give mypy what the None check needs and no more:

* every public class, with its bases, and every public method and property
  with its return type, so a chain of calls keeps its types and an
  ``Optional[...]`` result is one to mypy (``Map.get_waypoint``,
  ``ActorList.find``, ``World.get_actor``, ``Actor.parent``, ...);
* parameters as ``*args: Any, **kwargs: Any``: the arguments are Codon's to
  check, and the library's parameter types (``i32``, a ``Static[str]``, a
  Vector3D accepted for a Location) are not Python's;
* a return type mypy cannot be given (a ``Ptr``, a type parameter, an
  overloaded method whose overloads return different types) as ``Any``,
  and every name not written (the enumerations, ``carla.command``, a
  member nobody wrote) as ``Any`` through a ``__getattr__``;
* ``Map.get_waypoint()`` as a ``Waypoint`` where it projects onto the road
  (its default), ``Optional`` where it may not (:data:`_HAND_WRITTEN`).

So the only errors the stubs add are of an ``Optional`` the library returns,
and the None check keeps only those (:mod:`.none_check`).  The package is
written once per library: under the user's cache directory, named by a
digest of the library's Codon sources and of this module.
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = ["carla_stub_root", "render_carla_stub"]

#: Codon's types that are a Python type.
_SCALARS = {
    "int": "int",
    "float": "float",
    "str": "str",
    "bool": "bool",
    "float32": "float",
    "byte": "int",
    "i8": "int",
    "i16": "int",
    "i32": "int",
    "i64": "int",
    "u8": "int",
    "u16": "int",
    "u32": "int",
    "u64": "int",
}
_GENERICS = {
    "List": "list",
    "Dict": "dict",
    "Set": "set",
    "Generator": "Iterator",
    "Iterator": "Iterator",
}
#: The dunders written: Python's own protocols (arithmetic, comparison,
#: containers, iteration), not Codon's (``__to_py__``, ``__raw__``, ...).
_DUNDERS = {
    f"__{name}__"
    for name in (
        "abs add sub mul truediv floordiv mod neg pos radd rsub rmul rtruediv "
        "iadd isub imul itruediv eq ne lt le gt ge hash bool len iter next "
        "getitem setitem contains enter exit str repr float int"
    ).split()
}
#: Members written by hand: ``Map.get_waypoint()`` projected onto the road
#: (``project_to_road=True``, its default) is ``None`` only on a map with no
#: lane of ``lane_type`` at all, so it is a ``Waypoint`` to mypy, and
#: ``Optional`` only where it may not project (``project_to_road=False``, or
#: a flag mypy cannot read as ``True``).  The framework itself uses the
#: projected one unguarded.
_HAND_WRITTEN = {
    ("Map", "get_waypoint"): [
        "@overload",
        "def get_waypoint(self, location: Any, project_to_road: Literal[True] = ...,"
        " lane_type: Any = ...) -> Waypoint: ...",
        "@overload",
        "def get_waypoint(self, location: Any, project_to_road: bool = ...,"
        " lane_type: Any = ...) -> Optional[Waypoint]: ...",
    ],
}


def _stub_type(node: ast.expr | None, known: set[str]) -> str:
    """The stub's spelling of the Codon annotation *node*: ``Any`` if none."""
    if node is None:
        return "Any"
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    if isinstance(node, ast.Name):
        if node.id in _SCALARS:
            return _SCALARS[node.id]
        if node.id in known:
            return node.id
        if node.id == "NoneType":
            return "None"
        return "Any"
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        head = node.value.id
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if head == "Optional" and len(items) == 1:
            return f"Optional[{_stub_type(items[0], known)}]"
        if head == "Tuple":
            return f"tuple[{', '.join(_stub_type(i, known) for i in items)}]"
        if head in _GENERICS:
            return (
                f"{_GENERICS[head]}[{', '.join(_stub_type(i, known) for i in items)}]"
            )
    return "Any"


def _base_name(base: str) -> str:
    """``Static[B]`` / ``B[T]`` -> ``B``: the name of a base class."""
    if base.startswith("Static[") and base.endswith("]"):
        base = base[len("Static[") : -1]
    return base.split("[", 1)[0]


def _parse(text: str | None) -> ast.expr | None:
    if text is None:
        return None
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return None


def _returns(funcs: list[Any], known: set[str]) -> str:
    """The return type of a member with the definitions *funcs*.

    Overloads that return different types return ``Any``: which one a call
    picks is the arguments', which the stubs do not type.
    """
    types = {_stub_type(_parse(f.ret), known) for f in funcs}
    return types.pop() if len(types) == 1 else "Any"


def _class_lines(name: str, inventory: Any, known: set[str]) -> list[str]:
    cls = inventory.classes[name]
    bases = [b for b in map(_base_name, cls.bases) if b in known and b != name]
    lines = [f"class {name}({', '.join(bases)}):" if bases else f"class {name}:"]
    body = [
        "def __init__(self, *args: Any, **kwargs: Any) -> None: ...",
        "def __getattr__(self, name: str) -> Any: ...",
        "def __setattr__(self, name: str, value: Any) -> None: ...",
    ]
    for field, annotation in cls.fields:
        if not field.startswith("_"):
            body.append(f"{field}: {_stub_type(_parse(annotation), known)}")
    for constant in cls.classvars:
        if not constant.startswith("_"):
            body.append(f"{constant}: ClassVar[Any]")
    for member, funcs in sorted(cls.members.items()):
        if member.endswith(".setter"):
            continue
        if member.startswith("_") and member not in _DUNDERS:
            continue
        if (name, member) in _HAND_WRITTEN:
            body += _HAND_WRITTEN[name, member]
            continue
        returns = _returns(funcs, known)
        kind = funcs[0].kind
        if kind == "property":
            if f"{member}.setter" in cls.members:
                body.append(f"{member}: {returns}")
            else:
                body += ["@property", f"def {member}(self) -> {returns}: ..."]
        elif kind == "static":
            body += [
                "@staticmethod",
                f"def {member}(*args: Any, **kwargs: Any) -> {returns}: ...",
            ]
        else:
            body.append(
                f"def {member}(self, *args: Any, **kwargs: Any) -> {returns}: ..."
            )
    return lines + [f"    {line}" for line in body]


def _ordered(names: Iterable[str], inventory: Any, known: set[str]) -> list[str]:
    """*names*, each class after the classes it derives from."""
    out: list[str] = []

    def add(name: str) -> None:
        if name in out:
            return
        for base in map(_base_name, inventory.classes[name].bases):
            if base in known and base != name:
                add(base)
        out.append(name)

    for name in sorted(names):
        add(name)
    return out


def render_carla_stub(inventory: Any) -> str:
    """The ``typesafe_carla/carla/__init__.pyi`` of the library *inventory*.

    *inventory* is a ``typesafe_carla.pycarla.Inventory``: the library's
    classes, functions and aliases, read from its Codon sources.
    """
    public = [inventory.resolve(n) for n in inventory.public]
    known = {
        n
        for n in public
        if n in inventory.classes
        and not n.startswith("_")
        and not inventory.is_typed_enum(n)
    }
    # The classes the public ones derive from and return are the package's
    # too (Waypoint's Junction, Lane, Road).
    pending = list(known)
    while pending:
        cls = inventory.classes[pending.pop()]
        names = [_base_name(b) for b in cls.bases]
        for member, funcs in cls.members.items():
            if not member.startswith("_") or member in _DUNDERS:
                for f in funcs:
                    tree = _parse(f.ret)
                    if tree is not None:
                        names += [
                            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
                        ]
        for name in names:
            if (
                name in inventory.classes
                and name not in known
                and not name.startswith("_")
                and not inventory.is_typed_enum(name)
            ):
                known.add(name)
                pending.append(name)
    lines = [
        "# Generated by autoware_carla_scenario.typecheck.carla_stubs from",
        "# typesafe_carla's Codon library, for the scenario check's None check.",
        "from typing import Any, ClassVar, Iterator, Literal, Optional, overload",
        "",
        "def __getattr__(name: str) -> Any: ...",
        "",
    ]
    for name in _ordered(known, inventory, known):
        lines += _class_lines(name, inventory, known)
        lines.append("")
    for alias in sorted(inventory.public):
        target = inventory.resolve(alias)
        if alias != target and target in known and alias not in known:
            lines.append(f"{alias} = {target}")
    for name, funcs in sorted(inventory.functions.items()):
        if name in inventory.public and name not in known:
            returns = _returns(funcs, known)
            lines.append(f"def {name}(*args: Any, **kwargs: Any) -> {returns}: ...")
    return "\n".join(lines) + "\n"


def _library_dir() -> Path:
    from typesafe_carla import paths

    return Path(paths.codon_modules_dir()) / "typesafe_carla"


def _digest(library: Path) -> str:
    """This module, typesafe_carla's version (its ``Inventory`` reads the
    library) and the library's Codon sources."""
    from importlib.metadata import version

    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(f"\0typesafe-carla {version('typesafe-carla')}\0".encode())
    for path in sorted(library.glob("*.codon")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _write(root: Path) -> None:
    from typesafe_carla.pycarla import Inventory

    package = root / "typesafe_carla"
    (package / "carla").mkdir(parents=True, exist_ok=True)
    (package / "__init__.pyi").write_text(
        "from typing import Any\n\ndef __getattr__(name: str) -> Any: ...\n"
    )
    (package / "carla" / "__init__.pyi").write_text(render_carla_stub(Inventory()))


def carla_stub_root(scratch: Path | None = None) -> Path | None:
    """A directory holding the stub package ``typesafe_carla``, for ``MYPYPATH``.

    Written once per library under the user's cache directory; where that
    cannot be written, into *scratch* (a temporary directory of the caller's).
    ``None`` where typesafe_carla's library or inventory cannot be read: the
    None check then runs with CARLA untyped, as before.
    """
    try:
        library = _library_dir()
        key = _digest(library)
    except Exception:  # an incomplete installation: no stubs
        return None
    try:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        cached = Path(base) / "autoware_carla_scenario" / "carla-stubs" / key
        if (cached / "typesafe_carla" / "carla" / "__init__.pyi").is_file():
            return cached
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=cached.parent, prefix=f".{key}-"))
        try:
            _write(tmp)
            try:
                tmp.rename(cached)
            except OSError:  # a concurrent check wrote it first
                pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if (cached / "typesafe_carla" / "carla" / "__init__.pyi").is_file():
            return cached
    except (OSError, RuntimeError):  # RuntimeError: no home directory
        pass
    except Exception:  # the inventory could not be read: no stubs
        return None
    if scratch is None:
        return None
    try:
        _write(scratch)
    except Exception:
        return None
    return scratch
