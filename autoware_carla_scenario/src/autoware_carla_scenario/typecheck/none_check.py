"""The None check: an Optional used without a test for None (#239).

Codon 0.19 unwraps an ``Optional[T]`` implicitly wherever a ``T`` is needed
(arithmetic, an attribute, an argument), so the compile accepts
``cfg.min_speed_kmh / 3.6`` for a ``float | None`` field, and Python raises
``TypeError`` when the field is ``None``.  Codon has no strict-optional mode,
so the scenario check runs mypy over the scenario's own modules, the ones it
compiles, and keeps only what mypy reports of a value that may be ``None``
used where it may not be: an attribute of it (``union-attr``), an operator
(``operator``), an argument (``arg-type``), an assignment (``assignment``),
a return (``return-value``), an index (``index``), an overloaded call
(``call-overload``) or an item of a list or dict display (``list-item``,
``dict-item``; a set display's is an ``arg-type``, #308), each only where
``None`` is the offending type
(:func:`is_none_misuse`): an operand or an item that is ``None`` or a union
holding it, or a value whose type holds ``None`` where the type expected
does not.  A message that merely mentions ``None`` (``"Callable[[Any], str |
None]"; expected "Callable[[Any], float | None]"``, a ``str`` for a
``float``) is not one (#267).  Every other mypy error
is Codon's to report, or no business of the check's, and so is an
``assignment`` that only declares: a ``None`` default or value for a
non-Optional annotation (``speed: float = None``, which Codon reads as an
``Optional[float]``), or a value assigned to an attribute mypy inferred as
``None`` from its first assignment.  Neither uses a value that may be
``None``; a use of it later is reported at the use.

The same pass keeps a second kind of finding (#309): Codon holds a union of
the framework's classes the model declares as a ``Union``
(``TrajectoryVertex.position``, a :data:`TrajectoryPosition`) and converts
it to whichever member a use needs at run time, so an attribute of one
member, or the union passed where one member is expected, compiles without
the ``isinstance`` test Python needs.  mypy reports both (``union-attr``,
``arg-type``), and they are kept where the union mypy names is made of the
members of one of the model's unions (:func:`framework_unions`, read from
the model, so it follows the model), as :data:`UNION` findings with their
own advice.  mypy reports an attribute missing on several members once per
member; the check reports it once, naming them all.

A third kind (#567): a field of one of the modules' own ``@dataclass(frozen=True)``
classes assigned (``p.x = 2.0``, ``p.x += 1.0``, ``self.x = ...`` in a method of
the class).  Codon compiles a frozen dataclass as an ordinary mutable class, so
the assignment compiles, and Python raises ``FrozenInstanceError``.  mypy knows
the type of the receiver and reports ``Property "x" defined in "Pt" is
read-only`` (``misc``); it is kept (:data:`FROZEN`) where ``Pt`` is a frozen
dataclass of the given modules (:func:`frozen_dataclasses`).  A receiver of an
unknown type (``Any``) is not reported, and ``object.__setattr__(self, ...)``,
the idiom of a frozen ``__post_init__``, is not an assignment to mypy.

mypy runs as its own process (``python -m mypy``), with no configuration file
(the scenario package's own mypy settings do not apply), following imports
silently: it reads the framework and the scenario package's other modules for
their types and reports only in the modules given.  typesafe_carla's CPython
package has no type information, so mypy is given stubs of it written from
typesafe_carla's Codon library (:mod:`.carla_stubs`, on ``MYPYPATH``): a
CARLA call returning an ``Optional`` (``Map.get_waypoint``, ``ActorList.find``)
returns one to mypy too (#267).  The bodies of functions
without annotations are checked too (``--check-untyped-defs``).  Its cache is kept in the
user's cache directory, so only the first check reads the framework; where
that cannot be written (a read-only home in a container), mypy runs without
a cache.

mypy takes the directory above a module's top package for a source root, and
for an installed package (not editable) that is ``site-packages``, whose
top-level modules (``typing_extensions.py``, ``mypy_extensions.py``) then
shadow the library ones and stop mypy before it checks anything (#281).  So
a module under ``site-packages`` is given to mypy through a link to its top
package in a directory of its own (:func:`_mypy_paths`), and reported at its
own path.

When mypy gives no verdict (it is not installed, it ran out of time, it
crashed), :func:`run_none_check` raises :class:`NoneCheckSkipped`, which the
check reports as a warning.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Iterable
from pathlib import Path

from .carla_stubs import carla_stub_root

__all__ = [
    "NoneCheckSkipped",
    "NoneFinding",
    "NONE",
    "UNION",
    "FROZEN",
    "FrameworkUnion",
    "find_none_misuses",
    "frozen_dataclasses",
    "framework_unions",
    "is_none_misuse",
    "is_union_misuse",
    "run_none_check",
]

#: The mypy error codes of a value that may be None used where it may not.
NONE_CODES = frozenset(
    {
        "union-attr",
        "operator",
        "arg-type",
        "assignment",
        "return-value",
        "index",
        "call-overload",
        # An item of a list or dict display (#308); a set display's item is
        # an argument to mypy (``Argument 2 to <set> ...``, ``arg-type``).
        "list-item",
        "dict-item",
    }
)

#: The mypy error codes of a framework union used as one of its members (#309).
UNION_CODES = frozenset({"union-attr", "arg-type"})

#: :attr:`NoneFinding.kind`: a value that may be None, used where it may not.
NONE = "none"
#: :attr:`NoneFinding.kind`: a framework union used as one of its members.
UNION = "union"
#: :attr:`NoneFinding.kind`: a field of a frozen dataclass assigned (#567).
FROZEN = "frozen"

#: ``Property "x" defined in "Pt" is read-only`` (``misc``): an assignment to
#: a field of a frozen dataclass (or to a property with no setter).
_READ_ONLY = re.compile(
    r'^Property "(?P<attr>\w+)" defined in "(?P<cls>[\w.]+)" is read-only$'
)

#: An ``assignment`` that only declares (see the module's docstring): a
#: ``None`` default or value for a non-Optional annotation, or a value for a
#: variable mypy inferred as ``None``.
_DECLARATION = re.compile(
    r'\((?:expression|default) has type "None",|variable has type "None"\)'
)

#: A value of type A where B is expected: ``Argument 1 to "f" has
#: incompatible type "A"; expected "B"``, ``(expression has type "A",
#: variable has type "B")``, ``(expression has type "A", target has type
#: "B")`` (an item assigned, ``d["k"] = a``), ``(got "A", expected "B")``,
#: ``Invalid index type "A" for "C"; expected type "B"``.
_EXPECTED = re.compile(
    r'(?:incompatible type|expression has type|got|Invalid index type) "(?P<actual>.*?)"'
    r'(?: for ".*?")?[;,] (?:expected|variable has type|target has type|expected type) '
    r'"(?P<expected>.*?)"'
)
#: ``Dict entry 0 has incompatible type "K": "V"; expected "K": "V"``
#: (``dict-item``): the key's and the value's types, of each side.
_DICT_ENTRY = re.compile(
    r'incompatible type "(?P<key>.*?)": "(?P<value>.*?)"; '
    r'expected "(?P<expected_key>.*?)": "(?P<expected_value>.*?)"'
)
#: ``Item "None" of "X | None" has no attribute "y"`` (``union-attr``).
_NONE_ITEM = re.compile(r'\bItem "None" of "')
#: ``Item "A" of "A | B" has no attribute "y"`` (``union-attr``).
_ITEM = re.compile(
    r'^Item "(?P<item>.*?)" of "(?P<union>.*?)" has no attribute "(?P<attr>.*?)"$'
)
_QUOTED = re.compile(r'"(.*?)"')

_LINE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?:(?P<col>\d+):)? "
    r"(?P<severity>error|note): (?P<message>.*?)(?:  \[(?P<code>[a-z-]+)\])?$"
)


def _members(type_text: str) -> list[str]:
    """The members of the union mypy spells *type_text*, at its top level.

    ``"float | None"`` -> ``["float", "None"]``; ``"Optional[X]"`` and
    ``"Union[X, None]"`` likewise; ``"Callable[[Any], str | None]"`` is one
    member, the ``None`` inside it being another type's.
    """
    text = type_text.strip()
    for head in ("Optional[", "Union["):
        if text.startswith(head) and text.endswith("]"):
            inner = _split(text[len(head) : -1], ",")
            return [*inner, "None"] if head == "Optional[" else inner
    return _split(text, "|")


def _split(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(text):
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _holds_none(type_text: str) -> bool:
    """Whether the type mypy spells *type_text* is ``None`` or a union holding it."""
    return "None" in _members(type_text)


def _offends(actual: str, expected: str) -> bool:
    """Whether *actual* holds ``None`` and *expected* does not."""
    return _holds_none(actual) and not _holds_none(expected)


def none_given(message: str) -> str | None:
    """The type expected, when mypy's *message* is of ``None`` itself (a
    ``None`` literal or a name always bound to it) given where another type
    is expected, and not of a value that may be ``None`` (#530)."""
    expected = _EXPECTED.search(message)
    if expected is None or expected["actual"] != "None":
        return None
    return expected["expected"]


def is_none_misuse(code: str, message: str) -> bool:
    """Whether mypy's error *message* (of *code*) is of a value that may be None.

    It is when ``None`` is the offending type: the item of a union the
    attribute is missing on (``Item "None" of "X | None"``); the type given
    where another is expected holding ``None`` and the type expected not
    (``incompatible type "float | None"; expected "float"``); or, for an
    operator, an index or an overloaded call, an operand, the value indexed
    or an argument being ``None`` or a union holding it.  A message that
    only mentions ``None`` inside another type, or in the type expected, is
    not.
    """
    if code not in NONE_CODES:
        return False
    if code == "union-attr":
        return _NONE_ITEM.search(message) is not None
    if code == "dict-item":
        entry = _DICT_ENTRY.search(message)
        return entry is not None and (
            _offends(entry["key"], entry["expected_key"])
            or _offends(entry["value"], entry["expected_value"])
        )
    expected = _EXPECTED.search(message)
    if expected is not None:
        return _offends(expected["actual"], expected["expected"])
    if code in ("operator", "index", "call-overload"):
        return any(_holds_none(t) for t in _QUOTED.findall(message))
    return False


#: The model of the framework the check compiles against.
_MODEL_PACKAGE = Path(__file__).resolve().parent / "codon" / "autoware_carla_scenario"

#: ``Name = Union[A, B]`` at the top level of a model file, or a method's
#: ``-> Union[A, B]`` (a property: ``TrajectoryVertex.position``).
_MODEL_UNION = re.compile(
    r"^(?:(?P<alias>[A-Za-z_]\w*) = |[ \t]+def (?P<attribute>\w+)\(self\) -> )"
    r"Union\[(?P<members>[^\[\]]*)\]",
    re.MULTILINE,
)
_MODEL_CLASS = re.compile(r"^class (?P<name>\w+)\b", re.MULTILINE)


@dataclass(frozen=True)
class FrameworkUnion:
    """A union of the framework's classes that the model holds as a Codon ``Union``.

    *module* is the framework module the model file stands for
    (``autoware_carla_scenario.trajectory``), *name* the alias
    (``TrajectoryPosition``), or ``Class.attribute`` for a property whose
    type the model spells out.
    """

    module: str
    name: str
    members: frozenset[str]


@lru_cache(maxsize=1)
def framework_unions() -> tuple[FrameworkUnion, ...]:
    """The unions of classes the model of the framework holds as a ``Union``.

    Read from the model (``typecheck/codon/autoware_carla_scenario``): each
    module-level ``Name = Union[...]`` alias and each property declared to
    return a ``Union[...]``, with no ``None`` among its members (an
    ``Optional``, which is the None check's).  So the list follows the
    model; ``test_typecheck_none_check`` checks that each is the union
    Python's module of the framework declares.
    """
    out: list[FrameworkUnion] = []
    for path in sorted(_MODEL_PACKAGE.rglob("*.codon")):
        relative = path.relative_to(_MODEL_PACKAGE).with_suffix("")
        parts = [p for p in relative.parts if p != "__init__"]
        module = ".".join(["autoware_carla_scenario", *parts])
        text = path.read_text(encoding="utf-8")
        classes = [(m.start(), m["name"]) for m in _MODEL_CLASS.finditer(text)]
        for match in _MODEL_UNION.finditer(text):
            members = frozenset(
                m.strip() for m in match["members"].replace("\n", " ").split(",")
            )
            if len(members) < 2 or "None" in members:
                continue
            if not all(re.fullmatch(r"[A-Za-z_]\w*", m) for m in members):
                continue
            if match["alias"]:
                name = match["alias"]
            else:
                owner = [c for start, c in classes if start < match.start()]
                if not owner:
                    continue
                name = f"{owner[-1]}.{match['attribute']}"
            out.append(FrameworkUnion(module, name, members))
    return tuple(out)


def _short(member: str) -> str:
    """*member* without its module (mypy qualifies a name it finds ambiguous)."""
    if member == "..." or not re.fullmatch(r"[\w.]+", member):
        return member
    return member.rpartition(".")[2]


def framework_union_of(type_text: str) -> FrameworkUnion | None:
    """The framework union (:func:`framework_unions`) *type_text* is made of.

    *type_text* is a union as mypy spells it, of two members or more besides
    ``None`` (and the ``...`` mypy may shorten a long one with), all members
    of one of the framework's unions: an alias first, so a property of the
    same members is named by it.
    """
    members = {_short(m) for m in _members(type_text)} - {"None", "..."}
    if len(members) < 2:
        return None
    found = [u for u in framework_unions() if members <= u.members]
    found.sort(key=lambda u: ("." in u.name, len(u.members)))
    return found[0] if found else None


def is_union_misuse(code: str, message: str) -> bool:
    """Whether mypy's error *message* (of *code*) uses a framework union as one member.

    It is when the union an attribute is missing on (``union-attr``), or the
    type given where another is expected (``arg-type``), is made of the
    members of one of the model's unions (:func:`framework_union_of`), and
    the message is not the None check's (:func:`is_none_misuse`).
    """
    if code not in UNION_CODES or is_none_misuse(code, message):
        return False
    if code == "union-attr":
        item = _ITEM.match(message)
        return item is not None and framework_union_of(item["union"]) is not None
    expected = _EXPECTED.search(message)
    return expected is not None and framework_union_of(expected["actual"]) is not None


@dataclass(frozen=True)
class NoneFinding:
    """What mypy reports of a value that may be None, at its place, or of a
    framework union used as one of its members (*kind* :data:`UNION`)."""

    path: str
    line: int
    column: int | None
    message: str
    code: str
    notes: tuple[str, ...] = ()
    kind: str = NONE

    @property
    def read_only(self) -> tuple[str, str] | None:
        """The (class, field) a :data:`FROZEN` finding is of."""
        match = _READ_ONLY.match(self.message) if self.kind == FROZEN else None
        return (match["cls"].rpartition(".")[2], match["attr"]) if match else None

    @property
    def union(self) -> FrameworkUnion | None:
        """The framework union a :data:`UNION` finding is of."""
        if self.kind != UNION:
            return None
        item = _ITEM.match(self.message) or _ITEMS.match(self.message)
        if item is not None:
            return framework_union_of(item["union"])
        expected = _EXPECTED.search(self.message)
        return framework_union_of(expected["actual"]) if expected else None


#: The findings :func:`_merged_items` makes of several ``Item "A" of "U"``.
_ITEMS = re.compile(
    r'^Items (?P<items>".*?") of "(?P<union>.*?)" have no attribute "(?P<attr>.*?)"$'
)


def _merged_items(findings: list[NoneFinding]) -> list[NoneFinding]:
    """*findings*, with the ``union-attr`` :data:`UNION` findings mypy makes of
    one attribute, one per member of the union lacking it, made one."""
    out: list[NoneFinding] = []
    merged: dict[tuple[str, int, int | None, str, str], int] = {}
    for finding in findings:
        item = _ITEM.match(finding.message) if finding.kind == UNION else None
        if item is None:
            out.append(finding)
            continue
        key = (finding.path, finding.line, finding.column, item["union"], item["attr"])
        if key not in merged:
            merged[key] = len(out)
            out.append(finding)
            continue
        first = out[merged[key]]
        items = [
            *(
                _QUOTED.findall(m["items"])
                if (m := _ITEMS.match(first.message))
                else [_ITEM.match(first.message)["item"]]  # type: ignore[index]
            ),
            item["item"],
        ]
        message = (
            "Items "
            + ", ".join(f'"{i}"' for i in items)
            + f' of "{item["union"]}" have no attribute "{item["attr"]}"'
        )
        out[merged[key]] = dataclasses.replace(
            first, message=message, notes=(*first.notes, *finding.notes)
        )
    return out


def frozen_dataclasses(trees: Iterable[ast.Module]) -> frozenset[str]:
    """The names of the classes of *trees* decorated ``@dataclass(frozen=True)``
    (``@dataclasses.dataclass(frozen=True)``), at any depth."""
    names: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for deco in node.decorator_list:
                if not isinstance(deco, ast.Call):
                    continue
                func = deco.func
                called = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if called == "dataclass" and any(
                    k.arg == "frozen"
                    and isinstance(k.value, ast.Constant)
                    and k.value.value is True
                    for k in deco.keywords
                ):
                    names.add(node.name)
    return frozenset(names)


def _parsed(paths: list[Path]) -> list[ast.Module]:
    trees: list[ast.Module] = []
    for path in paths:
        try:
            trees.append(ast.parse(path.read_text(encoding="utf-8")))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue  # the compile reports it
    return trees


class NoneCheckSkipped(Exception):
    """mypy gave no verdict on the modules; the message says why."""


def _cache_dir() -> str:
    """mypy's cache directory, or ``os.devnull`` (no cache) where it cannot
    be written: mypy fails outright on a cache directory it cannot create."""
    try:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        version = f"py{sys.version_info.major}{sys.version_info.minor}"
        path = Path(base) / "autoware_carla_scenario" / "mypy" / version
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
    except (OSError, RuntimeError):  # RuntimeError: no home directory
        return os.devnull
    return str(path)


def search_path(variable: str) -> list[str]:
    """The directories of ``$variable`` (``MYPYPATH``, ``PYTHONPATH``), each
    made absolute against the directory the check runs from: mypy runs in a
    directory of its own (:func:`run_none_check`), where a relative entry
    would name nothing."""
    entries = os.environ.get(variable, "").split(os.pathsep)
    return [os.path.abspath(entry) for entry in entries if entry]


def _environment(stubs: Path | None) -> dict[str, str]:
    """mypy's environment: the CARLA stubs (:mod:`.carla_stubs`) first on
    ``MYPYPATH``, where they could be written; ``MYPYPATH`` and
    ``PYTHONPATH`` as :func:`search_path` gives them."""
    env = dict(os.environ)
    mypypath = ([str(stubs)] if stubs is not None else []) + search_path("MYPYPATH")
    for variable, entries in (
        ("MYPYPATH", mypypath),
        ("PYTHONPATH", search_path("PYTHONPATH")),
    ):
        if entries:
            env[variable] = os.pathsep.join(entries)
        else:
            env.pop(variable, None)
    return env


def mypy_available() -> bool:
    """Whether mypy can be run (a dependency of the framework)."""
    try:
        return importlib.util.find_spec("mypy") is not None
    except (ImportError, ValueError):
        return False


def _own_path(reported: str, paths: list[Path]) -> str:
    """The module of *paths* mypy reports an error of at *reported*.

    mypy replays the errors of a module its cache holds unchanged with the
    path the module had when it was cached, so a module of the same name and
    contents elsewhere (a test's temporary package, another checkout) is
    reported at that path: the module given whose path ends the same way,
    the longest way, is meant.
    """
    if any(str(p) == reported for p in paths):
        return reported
    parts = Path(reported).parts

    def shared(path: Path) -> int:
        n = 0
        for mine, theirs in zip(reversed(path.parts), reversed(parts)):
            if mine != theirs:
                break
            n += 1
        return n

    best = max(paths, key=shared)
    return str(best) if shared(best) > 0 else reported


#: The names of the directories installed packages are in.
_SITE_DIRECTORIES = frozenset({"site-packages", "dist-packages"})


def _source_root(path: Path) -> Path:
    """The directory mypy takes for the source root of the module *path*: the
    one above its top package (the last directory up with an ``__init__.py``)."""
    root = path.parent
    while (root / "__init__.py").is_file() and root.parent != root:
        root = root.parent
    return root


def _mypy_paths(paths: list[Path], scratch: Path) -> dict[str, Path]:
    """The path to give mypy for each module of *paths*, to the module.

    A module whose source root is a ``site-packages`` directory is given
    through a link in *scratch* to its top package (or to the module, where
    it has none), so the source root mypy takes is *scratch*, not
    ``site-packages`` (#281).  Any other module is given as it is.

    Raises:
        NoneCheckSkipped: The link cannot be made.
    """
    given: dict[str, Path] = {}
    for path in paths:
        root = _source_root(path)
        if root.name not in _SITE_DIRECTORIES:
            # Absolute: mypy runs in a directory of its own (run_none_check).
            given[str(path.absolute())] = path
            continue
        relative = path.relative_to(root)
        link = scratch / relative.parts[0]
        try:
            if not link.is_symlink():
                # Absolute: a relative target is read from the link's directory.
                link.symlink_to((root / relative.parts[0]).absolute())
        except OSError as exc:
            # Given as it is, mypy would fail on the shadowed modules instead.
            raise NoneCheckSkipped(f"could not link {path} for mypy: {exc}") from None
        given[str(scratch / relative)] = path
    return given


def find_none_misuses(
    paths: list[Path], timeout: float = 600.0
) -> list[NoneFinding] | None:
    """What mypy reports of a ``None`` misused in the modules *paths*.

    ``None`` when there is no verdict (:func:`run_none_check` says why).
    """
    try:
        return run_none_check(paths, timeout)
    except NoneCheckSkipped:
        return None


def run_none_check(paths: list[Path], timeout: float = 600.0) -> list[NoneFinding]:
    """What mypy reports of a ``None`` misused in the modules *paths*.

    Raises:
        NoneCheckSkipped: There is no verdict: no module, mypy is not
            installed, it ran out of time, or it failed without a report (a
            crash, a syntax error the compile reports anyway).
    """
    if not paths:
        raise NoneCheckSkipped("no module to check")
    if not mypy_available():
        raise NoneCheckSkipped("mypy is not installed")
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file=",
        "--follow-imports=silent",
        "--ignore-missing-imports",
        "--strict-optional",
        # The body of a function without annotations too: the check compiles
        # every function of the scenario's modules.
        "--check-untyped-defs",
        "--show-column-numbers",
        "--show-error-codes",
        "--no-pretty",
        "--no-error-summary",
        "--hide-error-context",
        "--no-color-output",
        f"--cache-dir={_cache_dir()}",
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="acs-carla-stubs-") as scratch:
            sources = Path(scratch) / "sources"
            sources.mkdir()
            given = _mypy_paths(paths, sources)
            # An empty working directory: mypy searches the one it runs in,
            # and what happens to be in the caller's would reach a verdict
            # that check.py keeps beyond the process under a key without it.
            cwd = Path(scratch) / "cwd"
            cwd.mkdir()
            proc = subprocess.run(
                [*command, *given],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=cwd,
                env=_environment(carla_stub_root(Path(scratch))),
            )
    except subprocess.TimeoutExpired:
        raise NoneCheckSkipped(f"mypy did not finish within {timeout:.0f}s") from None
    except OSError as exc:
        raise NoneCheckSkipped(f"mypy could not be run: {exc}") from None
    if proc.returncode not in (0, 1):
        # 2: mypy could not check them (a crash, bad arguments)
        output = (proc.stdout + proc.stderr).splitlines()
        errors = [line for line in output if "error" in line.lower()] or output
        raise NoneCheckSkipped(
            f"mypy failed (exit status {proc.returncode})"
            + (f": {errors[0].strip()}" if errors else "")
        )
    findings: list[NoneFinding] = []
    current: NoneFinding | None = None
    reported = ""
    frozen = frozen_dataclasses(_parsed(paths))
    for raw in proc.stdout.splitlines():
        match = _LINE.match(raw.strip())
        if match is None:
            continue
        if match["severity"] == "note":
            if current is not None and match["path"] == reported:
                current = dataclasses.replace(
                    current, notes=(*current.notes, match["message"])
                )
                findings[-1] = current
            continue
        current = None
        code = match["code"] or ""
        if is_none_misuse(code, match["message"]):
            kind = NONE
        elif is_union_misuse(code, match["message"]):
            kind = UNION
        elif (
            code == "misc"
            and (ro := _READ_ONLY.match(match["message"])) is not None
            and ro["cls"].rpartition(".")[2] in frozen
        ):
            kind = FROZEN
        else:
            continue
        if code == "assignment" and _DECLARATION.search(match["message"]):
            continue
        reported = match["path"]
        own = given.get(reported)
        current = NoneFinding(
            str(own) if own is not None else _own_path(reported, paths),
            int(match["line"]),
            int(match["col"]) if match["col"] else None,
            match["message"],
            code,
            kind=kind,
        )
        findings.append(current)
    return _merged_items(findings)
