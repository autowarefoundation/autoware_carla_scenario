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

A fourth kind (#574): ``bytes`` and ``str`` mixed.  Codon's ``bytes`` is a
``str``, so ``b"x" + "y"``, ``"sent " + name.encode()``, a ``bytes`` returned
for a ``str`` or passed to a ``str`` parameter, ``"x" in raw`` and
``raw == "ab"`` all compile, and Python raises ``TypeError`` (or the
comparison is always ``False``).  mypy reports them (``operator``,
``arg-type``, ``return-value``, ``assignment``, ``list-item``, ``dict-item``,
``index`` and ``comparison-overlap``); they are kept (:data:`BYTES_STR`) only
where the offending pair is exactly ``{bytes, str}`` (:func:`is_bytes_str_mix`),
so no other incompatible type is admitted and a ``str | bytes`` union, which
mypy names as a union, is left alone.

A fifth (#575): a comparison that is always ``False``, which mypy reports with
``--strict-equality`` as ``comparison-overlap``: an ``Enum`` member compared
with its raw value (``Color.RED == "red"``, ``mode == 1``, ``mode in (1, 2)``)
or with a member of another enum.  Python evaluates these to ``False``; Codon
compiles them.  Only the ``comparison-overlap`` findings with an enum as an
operand are kept (:data:`ENUM_EQ`), not all of them: mypy also reports a
comparison of two different literals, which is a deliberate switch
(``MODE: Final = 1`` and ``if MODE == 2``), so ``list[int] == list[str]`` and
the like are left alone.  mypy skips the classes that define ``__eq__`` and
the mixins that do overlap (``IntEnum``, ``(str, Enum)``, ``StrEnum``).  mypy compares
the member *literals* of those by identity (``Level.LOW == 1`` is reported
for an ``IntEnum``, though Python's is ``True``), so a comparison of a member
literal is kept only for an enum of the modules whose bases are only ``Enum``
or ``Flag``, and not for two members of one enum (:func:`_is_doubtful_enum_literal`).
The framework and the template produce none.  ``--strict-equality`` adds no other error.

A sixth (#577): a contract the author declared with a type and then broke, which
Python mostly does not enforce and Codon does not read.  Four kinds are kept,
each matched on mypy's code *and* message and no other ``misc``, ``override``,
``index`` or ``attr-defined`` error (Codon reports attribute errors itself):
a ``Final`` name or attribute assigned again (:data:`FINAL`), a ``ClassVar``
assigned through an instance (:data:`CLASS_VAR`), a method overriding the
superclass's with an incompatible signature (:data:`OVERRIDE`; not the dunder
methods, whose narrowed ``__eq__(self, other: Pt)`` is common and benign, and
mypy's own ``# type: ignore[override]`` is honoured), and a ``Mapping``,
``Sequence`` or ``AbstractSet`` mutated (:data:`READ_ONLY_ABC`; ``m[k] = v``,
``del m[k]`` and the mutating methods only, so a stub gap or a typo is not one).

A seventh (#589): a string formatting that fails at run time, which mypy
reports as ``str-format`` for a ``str.format`` call and a ``%`` with a literal
format string: a conversion the value does not take (``"{:d}".format(speed)``
for a ``float``, ``"%d" % name`` for a ``str``), too few or too many
arguments, a missing replacement field or mapping key, a malformed or unknown
specifier or conversion.  Python raises ``ValueError``, ``TypeError``,
``IndexError`` or ``KeyError`` at the call; Codon formats with its own rules
and compiles it.  Only the messages that are a run-time failure are kept
(:func:`is_str_format_failure`, :data:`STR_FORMAT`), and one naming the
value's type only for a built-in type that fails for every value
(``str``, ``float``, ``None``, ...): mypy does not count ``__index__``
(``"%x" % obj`` runs) nor a ``__getitem__`` without ``keys`` (``"%(k)s"``),
and refuses ``"{:.2f}"`` of a ``complex`` or a ``Fraction``, which run.
``Not all arguments converted`` is kept of a ``%`` whose right operand is a
tuple display or a constant: ``"{}".format(a, b)`` ignores the extra one, and
``"abc" % d`` formats no key of a dict (or any operand with ``__getitem__``).
``Unrecognized format specification`` is kept where the Python running the
check refuses the spec (``"{:z.1f}"`` runs on 3.11).  Left alone, because
mypy reports valid code with them: ``Numeric flags are only allowed for
numeric types`` (``"{:05}".format(name)`` and ``"{:<05}"`` run, though
``"{:+}"`` does not, and the message does not say which flag), ``Syntax error
in format specifier`` and the field-accessor messages (``"{0[my key]}"`` and
``"{0['k']}"`` index a dict with ``"my key"`` and ``"'k'"``).  mypy does not
check the spec of an f-string field (``f"{speed:d}"``): it is not reported.

Not kept, on purpose (#576): ``possibly-undefined`` (a name bound on one path
only, ``UnboundLocalError`` in Python).  mypy's flow analysis cannot see that
the branch that binds a name and the branch that reads it are the same
condition (``if x is not None: v = ...`` and later ``if x is not None: use(v)``),
nor that a loop over a non-empty constant runs; run over the framework's 313
modules it reports 15 times, 13 of them false.  Fixing a false report means
restructuring correct code, so the code is not enabled.

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
    "BYTES_STR",
    "ENUM_EQ",
    "FINAL",
    "CLASS_VAR",
    "OVERRIDE",
    "READ_ONLY_ABC",
    "STR_FORMAT",
    "FrameworkUnion",
    "find_none_misuses",
    "frozen_dataclasses",
    "framework_unions",
    "is_none_misuse",
    "is_union_misuse",
    "is_bytes_str_mix",
    "is_equality_overlap",
    "contract_kind",
    "is_str_format_failure",
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

#: :attr:`NoneFinding.kind`: ``bytes`` and ``str`` mixed (#574).
BYTES_STR = "bytes-str"
#: :attr:`NoneFinding.kind`: an enum compared with a type it never equals (#575).
ENUM_EQ = "enum-eq"

#: :attr:`NoneFinding.kind`: a ``Final`` name or attribute assigned again (#577).
FINAL = "final"
#: :attr:`NoneFinding.kind`: a ``ClassVar`` assigned through an instance (#577).
CLASS_VAR = "class-var"
#: :attr:`NoneFinding.kind`: a method incompatible with the one it overrides (#577).
OVERRIDE = "override"
#: :attr:`NoneFinding.kind`: a ``Mapping`` / ``Sequence`` / ``AbstractSet`` mutated (#577).
READ_ONLY_ABC = "read-only-abc"
#: :attr:`NoneFinding.kind`: a string formatting that fails at run time (#589).
STR_FORMAT = "str-format"

#: The ``str-format`` messages of a formatting Python fails on (#589); the
#: others mypy also reports of valid code (see the module's docstring).  A
#: message naming the value's type is kept only where :data:`_UNFORMATTABLE`
#: names it, ``Not all arguments converted`` only of a ``%`` whose right
#: operand is a tuple display or a constant (:func:`_too_many_fail`), and an
#: unrecognized spec only where Python refuses it (:func:`_spec_refused`).
_STR_FORMAT_FAILURE = re.compile(
    r"^(?:Not enough arguments for format string$"
    r"|Not all arguments converted during string formatting$"
    r"|Cannot find replacement for (?:positional|named) format specifier "
    r"|Invalid conversion specifier in format string"
    r"|Cannot combine automatic field numbering and manual field specification$"
    r'|Invalid conversion type ".*", must be one of'
    r"|Formatting nesting must be at most two levels deep$"
    r'|Key ".*" not found in mapping$'
    r'|Unsupported format character "'
    r'|"[%:]c" requires (?:int or char|an integer in range\(256\) or a single byte)$'
    r"|\* wants int$"
    r'|Format character "b" is only supported on bytes patterns$'
    r"|String interpolation (?:contains both stars and mapping keys"
    r"|mixes specifier with and without mapping keys)$"
    r"|On Python 3 b'%s' requires bytes, not string$"
    r"|Dictionary keys in bytes formatting must be bytes, not strings$)"
)
#: A ``str-format`` message naming the type of the value formatted.
_STR_FORMAT_TYPED = re.compile(
    r"^(?:Incompatible types in string interpolation"
    r"|Format requires a mapping"
    r'|"[%:]c" requires (?:int or char|an integer in range\(256\) or a single byte))'
    r' \(expression has type "(?P<actual>[^"]*)"'
    r'(?:, placeholder has type "(?P<placeholder>[^"]*)")?'
)
#: The types whose value no conversion mypy refuses it for formats in Python,
#: nor a ``%(key)s`` takes as a mapping: a class of the scenario's may define
#: ``__index__`` (``"%x"``, ``"%c"``) or ``__getitem__`` (``"%(k)s"``), which
#: mypy does not count, and Python's ``Fraction`` formats with ``"{:.2f}"``.
_UNFORMATTABLE = re.compile(
    r"^(?:str|float|bytes|bytearray|None|int|bool|complex"
    r"|(?:list|dict|set|frozenset|tuple)\[.*\])$"
)
#: ``Unrecognized format specification "spec"``.
_UNRECOGNIZED_SPEC = re.compile(r'^Unrecognized format specification "(?P<spec>.*)"$')
#: ``Not all arguments converted during string formatting``.
_TOO_MANY = "Not all arguments converted during string formatting"

#: ``Cannot assign to final name "X"`` / ``... final attribute "X"`` (``misc``).
_FINAL_ASSIGNED = re.compile(
    r'^Cannot assign to final (?:name|attribute) "(?P<name>\w+)"$'
)
#: ``Cannot assign to class variable "X" via instance`` (``misc``).
_CLASS_VAR_VIA_INSTANCE = re.compile(
    r'^Cannot assign to class variable "(?P<name>\w+)" via instance$'
)
#: What mypy says of a method that does not fit the superclass's (``override``).
#: ``(?P<method>...)`` is the method; a dunder one is not kept (see the module's docstring).
_OVERRIDE = re.compile(
    r'^(?:Argument \d+ of "(?P<m1>\w+)" is incompatible with supertype "(?P<s1>[\w.]+)"'
    r'|Return type ".*" of "(?P<m2>\w+)" incompatible with return type ".*" in supertype "(?P<s2>[\w.]+)"'
    r'|Signature of "(?P<m3>\w+)" incompatible with supertype "(?P<s3>[\w.]+)")'
)
#: The read-only ABCs of the contract and what mutates each.
_READ_ONLY_TARGET = re.compile(
    r'^Unsupported target for indexed assignment \("(?:Mapping|Sequence)\[.*\]"\)$'
)
_MUTATORS = (
    "append|extend|insert|pop|remove|sort|reverse|clear|update|setdefault|popitem"
    "|add|discard|difference_update|intersection_update|symmetric_difference_update"
    "|__delitem__"
)
_READ_ONLY_MUTATED = re.compile(
    r'^"(?P<abc>Mapping|Sequence|AbstractSet)\[.*\]" has no attribute '
    rf'"(?P<attr>{_MUTATORS})"(?:;|$)'
)

#: The mypy error codes ``bytes`` and ``str`` mixed is reported with (#574).
BYTES_STR_CODES = frozenset(
    {
        "operator",
        "arg-type",
        "return-value",
        "assignment",
        "list-item",
        "dict-item",
        "index",
        "comparison-overlap",
    }
)

#: The mypy error code of a comparison that is always False (``--strict-equality``).
OVERLAP_CODE = "comparison-overlap"

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


#: An operand type mypy spells ``Literal[...]``: ``Literal['ab']`` is a ``str``,
#: ``Literal[b'ab']`` a ``bytes``; an enum member is ``Literal[Color.RED]``.
_LITERAL = re.compile(r"^Literal\[(?P<value>.*)\]$")
_ENUM_LITERAL = re.compile(r"^Literal\[[A-Za-z_][\w.]*\.[A-Za-z_]\w*\]$")


def _plain_type(type_text: str) -> str:
    """*type_text* with a ``Literal['x']`` / ``Literal[b'x']`` as ``str`` / ``bytes``."""
    text = type_text.strip()
    literal = _LITERAL.match(text)
    if literal is None:
        return text
    value = literal["value"]
    if value[:2] in ("b'", 'b"'):
        return "bytes"
    if value[:1] in ("'", '"'):
        return "str"
    return text


def _is_bytes_str(first: str, second: str) -> bool:
    return {_plain_type(first), _plain_type(second)} == {"bytes", "str"}


def _overlap_operands(message: str) -> list[str]:
    """The two operand types of a ``comparison-overlap`` *message*, as mypy
    spells them (``left operand type: "A", right operand type: "B"``,
    ``left "A", right "B"``, ``element type: "A", container item type: "B"``)."""
    return _QUOTED.findall(message)


def is_equality_overlap(code: str, message: str) -> bool:
    """Whether mypy's error *message* (of *code*) is of a comparison that is
    always False (``--strict-equality``, #575)."""
    return code == OVERLAP_CODE and len(_overlap_operands(message)) == 2


def is_bytes_str_mix(code: str, message: str) -> bool:
    """Whether mypy's error *message* (of *code*) is of ``bytes`` and ``str``
    mixed (#574): the offending pair is exactly ``{bytes, str}``.

    A ``str | bytes`` union, ``bytearray`` and every other incompatible type
    are not.  mypy's message gives the pair as the type given and the type
    expected, as the two operands of an operator, or as the two sides of a
    comparison.
    """
    if code not in BYTES_STR_CODES:
        return False
    if code == "comparison-overlap":
        operands = _overlap_operands(message)
        return len(operands) == 2 and _is_bytes_str(*operands)
    if code == "dict-item":
        entry = _DICT_ENTRY.search(message)
        return entry is not None and (
            _is_bytes_str(entry["key"], entry["expected_key"])
            or _is_bytes_str(entry["value"], entry["expected_value"])
        )
    expected = _EXPECTED.search(message)
    if expected is not None:
        return _is_bytes_str(expected["actual"], expected["expected"])
    if code == "operator":
        operands = _QUOTED.findall(message)
        return len(operands) == 2 and _is_bytes_str(*operands)
    return False


def _enum_classes(trees: Iterable[ast.Module]) -> tuple[frozenset[str], frozenset[str]]:
    """The enums of *trees*: (plain, all) by class name.

    An enum is a class with a base named ``...Enum`` or ``...Flag``
    (``enum.Enum``, ``IntEnum``, ``StrEnum``, ``IntFlag``); it is *plain* where
    its bases are only ``Enum`` / ``Flag`` (no ``int`` or ``str`` mixin, no
    ``IntEnum``), and every class of that name is.
    """
    plain: set[str] = set()
    mixed: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            names = [
                (
                    base.id
                    if isinstance(base, ast.Name)
                    else base.attr
                    if isinstance(base, ast.Attribute)
                    else ""
                )
                for base in node.bases
            ]
            if not any(n.endswith(("Enum", "Flag")) for n in names):
                continue
            (plain if set(names) <= {"Enum", "Flag"} else mixed).add(node.name)
    return frozenset(plain - mixed), frozenset(plain | mixed)


def _enum_members(message: str) -> list[str | None]:
    """The enum class of each operand of a ``comparison-overlap`` *message*
    that is an enum member literal (``Literal[Color.RED]``), else ``None``."""
    out: list[str | None] = []
    for operand in _overlap_operands(message):
        if _ENUM_LITERAL.match(operand):
            out.append(operand[len("Literal[") : -1].split(".")[-2])
        else:
            out.append(None)
    return out


def _is_doubtful_enum_literal(message: str, plain: frozenset[str]) -> bool:
    """Whether a ``comparison-overlap`` *message* is of a comparison mypy reads
    wrongly or that is no confusion of an enum and a raw value.

    mypy compares enum member literals by identity, so for ``Level.LOW == 1``
    of an ``IntEnum`` or ``Mode.FAST == "fast"`` of a ``(str, Enum)`` it
    reports non-overlap although Python's is ``True``; and two members of one
    enum (``Color.RED != Color.BLUE``) are a comparison of members, not of a
    member with a raw value.  Neither is kept; nor is a literal of an enum the
    modules do not define (its bases are unknown).
    """
    members = _enum_members(message)
    classes = [m for m in members if m is not None]
    if not classes:
        return False
    if any(c not in plain for c in classes):
        return True
    return len(classes) == 2 and classes[0] == classes[1]


def _involves_enum(message: str, enums: frozenset[str]) -> bool:
    """Whether an operand of a ``comparison-overlap`` *message* is an enum:
    an enum member literal, or a class of *enums*."""
    for operand in _overlap_operands(message):
        if _ENUM_LITERAL.match(operand) or operand.rpartition(".")[2] in enums:
            return True
    return False


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


def contract_kind(code: str, message: str) -> str | None:
    """The kind of contract violation (#577) mypy's error *message* (of *code*)
    reports, or ``None`` when it is none of them.

    Both the code and the message must fit, so no other ``misc``,
    ``override``, ``index`` or ``attr-defined`` error is admitted.
    """
    if code == "misc":
        if _FINAL_ASSIGNED.match(message):
            return FINAL
        if _CLASS_VAR_VIA_INSTANCE.match(message):
            return CLASS_VAR
    elif code == "override":
        found = _OVERRIDE.match(message)
        if found is not None:
            method = found["m1"] or found["m2"] or found["m3"]
            supertype = found["s1"] or found["s2"] or found["s3"]
            if not (method.startswith("__") and method.endswith("__")) and (
                supertype.rpartition(".")[2] != "object"
            ):
                return OVERRIDE
    elif code == "index":
        if _READ_ONLY_TARGET.match(message):
            return READ_ONLY_ABC
    elif code == "attr-defined" and _READ_ONLY_MUTATED.match(message):
        return READ_ONLY_ABC
    return None


def is_str_format_failure(code: str, message: str) -> bool:
    """Whether mypy's error *message* (of *code*) is of a string formatting
    Python fails on at run time (#589), for every value of the types it names.

    ``Not all arguments converted during string formatting`` is one only of a
    ``%`` (not of ``str.format``, which ignores the extra arguments), which
    :func:`run_none_check` decides at the place mypy reports.
    """
    if code != "str-format":
        return False
    typed = _STR_FORMAT_TYPED.match(message)
    if typed is not None:
        actual = typed["actual"]
        if _UNFORMATTABLE.match(actual) is None:
            return False
        if actual in ("int", "bool"):
            # Formatted as a number everywhere but by b"%b" / b"%s" and as a
            # mapping, which mypy reports with no placeholder.
            return typed["placeholder"] in (None, "bytes")
        # A complex takes the float specs of str.format ("{:.2f}"), the only
        # placeholder mypy writes "int | float" for.
        return not (actual == "complex" and typed["placeholder"] == "int | float")
    spec = _UNRECOGNIZED_SPEC.match(message)
    if spec is not None:
        return _spec_refused(spec["spec"])
    return _STR_FORMAT_FAILURE.match(message) is not None


def _spec_refused(spec: str) -> bool:
    """Whether this Python refuses the format *spec* for an int, a float, a
    complex and a str alike: mypy refuses ``"{:z.1f}"`` (Python 3.11)."""
    for value in (0, 0.0, 0j, ""):
        try:
            format(value, spec)
        except (ValueError, TypeError):
            continue
        return False
    return True


def _too_many_fail(path: str, line: int, column: int | None) -> bool:
    """Whether the ``Not all arguments converted`` mypy reports at *line* and
    *column* of *path* fails in Python: one of a ``%`` whose right operand,
    where mypy reports it, is a tuple display or a constant other than bytes.
    ``str.format`` ignores extra arguments, and a ``%`` takes any other
    operand with ``__getitem__`` (a dict, a list, bytes) as a mapping and
    formats no key of it (#589)."""
    if column is None:
        return False
    try:
        source = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return False
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)):
            continue
        right = node.right
        if right.lineno != line or line > len(lines):
            continue
        prefix = lines[line - 1].encode("utf-8")[: right.col_offset]
        if len(prefix.decode("utf-8", errors="replace")) + 1 != column:
            continue
        if isinstance(right, ast.Tuple):
            return bool(right.elts) and not any(
                isinstance(e, ast.Starred) for e in right.elts
            )
        return isinstance(right, ast.Constant) and not isinstance(right.value, bytes)
    return False


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


def _dataclass_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """The names *tree* binds to ``dataclasses.dataclass`` (``from dataclasses
    import dataclass as dc``) and to the ``dataclasses`` module."""
    functions = {"dataclass"}
    modules = {"dataclasses"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
            functions |= {
                a.asname for a in node.names if a.name == "dataclass" and a.asname
            }
        elif isinstance(node, ast.Import):
            modules |= {
                a.asname for a in node.names if a.name == "dataclasses" and a.asname
            }
    return functions, modules


def _is_frozen(node: ast.ClassDef, functions: set[str], modules: set[str]) -> bool:
    for deco in node.decorator_list:
        if not isinstance(deco, ast.Call):
            continue
        func = deco.func
        if not (
            (isinstance(func, ast.Name) and func.id in functions)
            or (
                isinstance(func, ast.Attribute)
                and func.attr == "dataclass"
                and isinstance(func.value, ast.Name)
                and func.value.id in modules
            )
        ):
            continue
        if any(
            k.arg == "frozen"
            and isinstance(k.value, ast.Constant)
            and k.value.value is True
            for k in deco.keywords
        ):
            return True
    return False


def frozen_dataclasses(trees: Iterable[ast.Module]) -> frozenset[str]:
    """The names of the classes of *trees* decorated ``@dataclass(frozen=True)``
    (``@dataclasses.dataclass(frozen=True)``, an aliased import), at any depth.

    mypy names the class that defines a read-only property, not its module, so
    a name is kept only where every class of that name in *trees* is frozen:
    a read-only property of a same-named plain class is not a frozen field.
    """
    frozen: set[str] = set()
    other: set[str] = set()
    for tree in trees:
        functions, modules = _dataclass_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                (frozen if _is_frozen(node, functions, modules) else other).add(
                    node.name
                )
    return frozenset(frozen - other)


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
        # A comparison that is always False (an Enum member and its raw
        # value, bytes and str): ``comparison-overlap``, the only error it adds.
        "--strict-equality",
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
    trees = _parsed(paths)
    frozen = frozen_dataclasses(trees)
    plain_enums, enums = _enum_classes(trees)
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
        elif is_bytes_str_mix(code, match["message"]):
            kind = BYTES_STR
        elif is_equality_overlap(code, match["message"]):
            if _is_doubtful_enum_literal(match["message"], plain_enums):
                continue
            if not _involves_enum(match["message"], enums):
                continue
            kind = ENUM_EQ
        elif (
            code == "misc"
            and (ro := _READ_ONLY.match(match["message"])) is not None
            and ro["cls"].rpartition(".")[2] in frozen
        ):
            kind = FROZEN
        elif (contract := contract_kind(code, match["message"])) is not None:
            kind = contract
        elif is_str_format_failure(code, match["message"]):
            kind = STR_FORMAT
        else:
            continue
        if code == "assignment" and _DECLARATION.search(match["message"]):
            continue
        reported = match["path"]
        own = given.get(reported)
        own_path = str(own) if own is not None else _own_path(reported, paths)
        line = int(match["line"])
        column = int(match["col"]) if match["col"] else None
        if (
            kind == STR_FORMAT
            and match["message"] == _TOO_MANY
            and not _too_many_fail(own_path, line, column)
        ):
            continue
        current = NoneFinding(
            own_path,
            line,
            column,
            match["message"],
            code,
            kind=kind,
        )
        # mypy reports an override once per superclass that defines the
        # method; the nearest one (reported first) is the one to fix.
        if kind == OVERRIDE and any(
            f.kind == OVERRIDE
            and (f.path, f.line, f.column)
            == (current.path, current.line, current.column)
            for f in findings
        ):
            current = None  # nor its notes, which repeat the first one's
            continue
        findings.append(current)
    return _merged_items(findings)
