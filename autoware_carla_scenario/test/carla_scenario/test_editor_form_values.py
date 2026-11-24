"""The editor's list-of-integers and row-count parsers (``editor.forms``).

The ``_old_*`` oracles below are ``editor/forms.py``'s ``_LIST_SEPARATORS``,
``_MAX_RECORDS``, ``_parse_list_int()``, ``parse_int_list()`` and
``parse_records()`` as they were before their pure parts moved into the
library check (#2135), copied verbatim but for their names (``_old_`` in
front, and in the calls between them).  What the editor has now must do the
same: return the same value, of the same types, or raise the same exception
with the same message, for whatever a form or a document can post -- digit
grouping (``1_000``), Unicode digits and spaces, signs, integers too long for
``int()``, empty items, every separator and random text built from them.

One change is deliberate (#2235): an interpolation is kept only where the
validator accepts one, the exact ``${map.no_3d_model_lanelet_ids}`` of an
``int_list_or_ref`` field, and refused everywhere else
(:func:`_expected_int_list`).
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Mapping

import pytest

from autoware_carla_scenario.authoring.registry import (
    REFERENCE_PATTERN,
    FieldKind,
    FieldSpec,
)
from autoware_carla_scenario.authoring.hydra_config import MAP_EXCLUSION_REF
from autoware_carla_scenario.editor import forms

# -- the oracles: editor/forms.py as it was ------------------------------------

#: More records than any list is written with, so a forged row count is
#: refused rather than looped over.
_OLD_MAX_RECORDS = 1000

#: Separators accepted in a list-of-integers field: commas, spaces, newlines.
_OLD_LIST_SEPARATORS = re.compile(r"[,\s]+")

_ROWS = "__rows"
_OP = "__op"


def _old_parse_list_int(part: str) -> int:
    """Return one item of an id list as an int.

    Raises:
        ValueError: If *part* is not a whole number.
    """
    try:
        return int(part)
    except ValueError:
        raise ValueError(f"must be whole numbers, got {part!r}") from None


def _old_parse_int_list(raw: Any) -> Any:
    """Return *raw* as a list of ints, or verbatim when it is an interpolation.

    Raises:
        ValueError: If an item is not a whole number, naming that item.
    """
    if isinstance(raw, (list, tuple)):
        return [_old_parse_list_int(str(v).strip()) for v in raw if str(v).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    if REFERENCE_PATTERN.match(text):
        return text
    return [
        _old_parse_list_int(part) for part in _OLD_LIST_SEPARATORS.split(text) if part
    ]


def _old_parse_records(
    spec: FieldSpec, form: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return the records a ``record_list`` field's rows post, with the row
    button pressed (if any) applied.

    Raises:
        ValueError: If a record's field cannot be parsed, or the row count or
            the button is malformed.
    """
    count = int(str(form[f"{spec.name}.{_ROWS}"]))
    if not 0 <= count <= _OLD_MAX_RECORDS:
        raise ValueError(f"{count} rows")
    # Each row's fields, by row, in one pass over the form.
    posted: list[dict[str, Any]] = [{} for _ in range(count)]
    prefix = f"{spec.name}."
    for key, value in form.items():
        if not key.startswith(prefix):
            continue
        slot, _, sub = key[len(prefix) :].partition(".")
        if sub and slot.isascii() and slot.isdigit() and int(slot) < count:
            posted[int(slot)][sub] = value
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(posted):
        try:
            rows.append(forms.parse_params(spec.item_fields, row))
        except ValueError as exc:
            raise ValueError(f"row {index + 1}: {exc}") from exc
    op = str(form.get(f"{spec.name}.{_OP}") or "")
    if op:
        forms._apply_row_op(spec, rows, op)
    return rows


def _expected_int_list(raw: Any, reference: str | None = None) -> Any:
    """:func:`_old_parse_int_list`, but for the interpolations the old one
    kept verbatim: only *reference*, exactly, is kept now (#2235)."""
    text = raw.strip() if isinstance(raw, str) else None
    if text is not None and text != reference and REFERENCE_PATTERN.fullmatch(text):
        if reference is not None:
            raise ValueError(f"must be whole numbers or {reference}, got {text!r}")
        raise ValueError(f"must be whole numbers, got {text!r}")
    return _old_parse_int_list(raw)


# -- the comparison -------------------------------------------------------------


def _outcome(fn: Callable[..., Any], *args: Any) -> tuple[str, Any]:
    """What *fn* does with *args*: its value with the types inside, or the
    exception's type and message."""
    try:
        value = fn(*args)
    except Exception as exc:  # noqa: BLE001 -- compared, not handled
        return ("raise", (type(exc), str(exc)))
    return ("return", _typed(value))


def _typed(value: Any) -> Any:
    if isinstance(value, list):
        return ("list", [_typed(v) for v in value])
    if isinstance(value, dict):
        return ("dict", sorted((k, _typed(v)) for k, v in value.items()))
    return (type(value), value)


#: Items a list field's text is made of.
_ITEMS = (
    "0",
    "1",
    "42",
    "-7",
    "+5",
    "-0",
    "007",
    "1_000",
    "1__000",
    "_1",
    "1_",
    "١٢٣",  # Arabic-Indic 123
    "１２",  # fullwidth 12
    "१",  # Devanagari 1
    "²",  # superscript two: isdigit(), not int()
    "Ⅷ",  # Roman numeral eight
    "1.0",
    "1e3",
    "0x10",
    "0b1",
    "inf",
    "nan",
    "abc",
    "${a}",
    "${",
    "}",
    "-",
    "+",
    "9" * 4300,
    "9" * 4301,
    "-" + "1" * 5000,
    " 1",
    "1 ",
    "​1",
    "1;2",
    "\ud800",
)

#: Separators and spaces between them.
_SEPARATORS = (
    ",",
    " ",
    "\n",
    "\t",
    "\r\n",
    ",,",
    ", ",
    " , ",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x1f",
    "\x85",
    " ",
    " ",
    "　",
    "​",
    ";",
    "",
)

_TEXTS = (
    "",
    " ",
    ",",
    ",,,",
    " , \n ",
    "1,2,3",
    "1 2\n3",
    " 1_000 ",
    "1,,2",
    ",1,",
    "${lanelets}",
    " ${lanelets} ",
    "${a} ${b}",
    "${a},1",
    "1,${a}",
    "${}",
    "$ {a}",
    "　 1 　",
    "  ",
    "1 2",
    "1 2",
)

_NON_TEXT: tuple[Any, ...] = (
    None,
    0,
    5,
    -3,
    0.0,
    2.5,
    False,
    True,
    b"12",
    [],
    (),
    ["1", " 2 ", "", "  ", None, 3, -4, True, 5.0],
    ("1_000", "١", " +7 "),
    [None],
    ["${a}"],
    ["1,2"],
    [b"1"],
    [" "],
    [[1]],
    {"a": 1},
)


def _random_texts(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        parts = []
        for _ in range(rng.randint(0, 6)):
            parts.append(rng.choice(_ITEMS[:-4]) if rng.random() < 0.8 else "")
            parts.append(rng.choice(_SEPARATORS))
        texts.append("".join(parts))
    return texts


_ALL_TEXTS = (
    *_TEXTS,
    *_ITEMS,
    *(f"1{sep}2" for sep in _SEPARATORS),
    *(f"{sep}1{sep}" for sep in _SEPARATORS),
    *_random_texts(2135, 600),
)


@pytest.mark.parametrize("part", [*_ITEMS, *_TEXTS, *(s.strip() for s in _ITEMS)])
def test_one_item_parses_as_before(part: str) -> None:
    assert _outcome(forms._parse_list_int, part) == _outcome(_old_parse_list_int, part)


def test_every_text_parses_as_before() -> None:
    for text in _ALL_TEXTS:
        assert _outcome(forms.parse_int_list, text) == _outcome(
            _expected_int_list, text
        ), text


@pytest.mark.parametrize("raw", _NON_TEXT, ids=repr)
def test_every_non_text_value_parses_as_before(raw: Any) -> None:
    assert _outcome(forms.parse_int_list, raw) == _outcome(_expected_int_list, raw)


def test_lists_of_random_items_parse_as_before() -> None:
    rng = random.Random(42)
    pool: list[Any] = [*_ITEMS[:-4], *_SEPARATORS, None, 1, -2, 3.5, True]
    for _ in range(400):
        raw: Any = [rng.choice(pool) for _ in range(rng.randint(0, 5))]
        if rng.random() < 0.3:
            raw = tuple(raw)
        assert _outcome(forms.parse_int_list, raw) == _outcome(
            _expected_int_list, raw
        ), raw


@pytest.mark.parametrize("kind", ["int_list", "lanelet_list", "int_list_or_ref"])
def test_a_list_field_parses_as_before(kind: FieldKind) -> None:
    spec = FieldSpec("ids", "Ids", kind, (), required=False)
    reference = MAP_EXCLUSION_REF if kind == "int_list_or_ref" else None
    for text in (*_TEXTS, MAP_EXCLUSION_REF, *_random_texts(7, 100)):
        expected = _outcome(_expected_int_list, text.strip(), reference)
        got = _outcome(lambda t: forms.parse_value(spec, t, present=True), text)
        if text.strip():
            assert got == expected, text
        else:
            assert got == ("return", (type(None), None)), text


_RECORDS = FieldSpec(
    "rows",
    "Rows",
    "record_list",
    (),
    item_fields=(FieldSpec("n", "N", "int", 1, required=False),),
)

_COUNTS: tuple[Any, ...] = (
    "0",
    "1",
    "3",
    " 2 ",
    "+2",
    "-0",
    "-1",
    "1000",
    "1001",
    "1_000",
    "1_001",
    "٣",
    "２",
    "²",
    "2.0",
    "1e2",
    "",
    " ",
    "x",
    "9" * 4300,
    "9" * 4301,
    "-" + "9" * 30,
    0,
    2,
    -5,
    1001,
    True,
    None,
    2.0,
)


@pytest.mark.parametrize("count", _COUNTS, ids=repr)
@pytest.mark.parametrize("op", [None, "add", "delete:0", "up:1", "down:0", "bad:0"])
def test_a_row_count_parses_as_before(count: Any, op: Any) -> None:
    form: dict[str, Any] = {"rows.__rows": count, "rows.0.n": "4", "rows.1.n": "x"}
    if op is not None:
        form["rows.__op"] = op
    good = dict(form, **{"rows.1.n": "5"})
    for posted in (form, good):
        assert _outcome(forms.parse_records, _RECORDS, posted) == _outcome(
            _old_parse_records, _RECORDS, posted
        )


def test_forms_re_exports_the_moved_names() -> None:
    from autoware_carla_scenario.editor import form_values

    assert forms._parse_list_int is form_values.parse_list_int
    assert forms._parse_int_text is form_values.parse_int_text
    assert forms._row_count is form_values.row_count
    assert forms._LIST_SEPARATORS is form_values.LIST_SEPARATORS
    assert forms._MAX_RECORDS is form_values.MAX_RECORDS


@pytest.mark.parametrize("text", [*_ALL_TEXTS[:200]])
def test_the_checked_split_is_the_old_one(text: str) -> None:
    from autoware_carla_scenario.editor import form_values

    def old_split(t: str) -> list[int]:
        return [
            _old_parse_list_int(part) for part in _OLD_LIST_SEPARATORS.split(t) if part
        ]

    assert _outcome(form_values.parse_int_text, text) == _outcome(old_split, text)


class TestReferences:
    """An interpolation only where the validator takes one (#2235)."""

    def test_the_exclusion_list_is_kept_for_a_field_that_takes_it(self) -> None:
        spec = FieldSpec("ids", "Ids", "int_list_or_ref", (), required=False)
        assert (
            forms.parse_value(spec, f" {MAP_EXCLUSION_REF} ", present=True)
            == MAP_EXCLUSION_REF
        )

    @pytest.mark.parametrize(
        "text", ["${map.other}", "${ map.no_3d_model_lanelet_ids }", "${a}"]
    )
    def test_another_reference_is_refused_there(self, text: str) -> None:
        spec = FieldSpec("ids", "Ids", "int_list_or_ref", (), required=False)
        with pytest.raises(ValueError, match="whole numbers or"):
            forms.parse_value(spec, text, present=True)

    @pytest.mark.parametrize("kind", ["int_list", "lanelet_list"])
    def test_no_reference_is_taken_by_a_plain_list(self, kind: FieldKind) -> None:
        spec = FieldSpec("ids", "Ids", kind, (), required=False)
        with pytest.raises(ValueError, match="must be whole numbers"):
            forms.parse_value(spec, MAP_EXCLUSION_REF, present=True)


# -- the service's number fields (#2386) ---------------------------------------
#
# editor/service.py's _as_float(), as_int() and as_port() as they were before
# their parsing and messages moved into the checked editor.form_values
# (float_field(), int_field()), copied verbatim but for their names.


def _old_as_float(raw: Any, label: str, fallback: float) -> float:
    import math  # noqa: PLC0415

    from autoware_carla_scenario.editor.service import EditorError  # noqa: PLC0415

    text = str(raw).strip()
    if not text:
        return fallback
    try:
        number = float(text)
    except ValueError as exc:
        raise EditorError(f"{label} must be a number, got {raw!r}.") from exc
    if not math.isfinite(number):
        raise EditorError(f"{label} must be a finite number, got {raw!r}.")
    return number


def _old_as_int(raw: Any, label: str, fallback: int) -> int:
    from autoware_carla_scenario.editor.service import EditorError  # noqa: PLC0415

    text = str(raw).strip()
    if not text:
        return fallback
    try:
        return int(text)
    except ValueError as exc:
        raise EditorError(f"{label} must be a whole number, got {raw!r}.") from exc


def _old_as_port(raw: Any, label: str, fallback: int) -> int:
    from autoware_carla_scenario.editor.service import EditorError  # noqa: PLC0415
    from autoware_carla_scenario.serve_bind import check_port  # noqa: PLC0415

    number = _old_as_int(raw, label, fallback)
    try:
        return check_port(number)
    except ValueError as exc:
        raise EditorError(f"{label} {exc}.") from exc


#: Numbers as a form posts them, and what else reaches the fields.
_NUMBERS = (
    *_ITEMS,
    "",
    " ",
    "　",
    "1.5",
    " -2.25 ",
    "+.5",
    "5.",
    ".",
    "1e308",
    "1e309",
    "-1e309",
    "9" * 400,
    "-inf",
    "Infinity",
    "NaN",
    "1_0.5",
    "1__0.5",
    "١.٥",
    "0",
    "65535",
    "65536",
    "-1",
    "80",
    " 8080 ",
    "1" + "0" * 40,
)
_RAW: tuple[Any, ...] = (
    *_NUMBERS,
    None,
    0,
    5,
    -3,
    0.0,
    2.5,
    float("inf"),
    float("nan"),
    10**400,
    False,
    True,
    b"12",
    [],
    [1],
    {"a": 1},
)


def _random_numbers(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    pieces = ("1", "0", "9", "_", ".", "e", "-", "+", " ", "\t", "١", "１", "x", "inf")
    out = []
    for _ in range(count):
        out.append("".join(rng.choice(pieces) for _ in range(rng.randint(0, 7))))
        out.append(f"{rng.uniform(-1e6, 1e6)!r}")
        out.append(str(rng.randrange(-70000, 70000)))
    return out


@pytest.mark.parametrize(
    ("new_name", "old"),
    [("_as_float", _old_as_float), ("as_int", _old_as_int), ("as_port", _old_as_port)],
)
def test_the_service_s_number_fields_read_as_before(
    new_name: str, old: Callable[..., Any]
) -> None:
    from autoware_carla_scenario.editor import service

    new = getattr(service, new_name)
    labels = ("Timeout", "", "CARLA port", "ü {x}")
    fallbacks: tuple[Any, ...] = (60.0, 2000, 0, -1, None, "x", 70000)
    rng = random.Random(2386)
    for raw in (*_RAW, *_random_numbers(2386, 400)):
        label = rng.choice(labels)
        fallback = rng.choice(fallbacks)
        assert _outcome(new, raw, label, fallback) == _outcome(
            old, raw, label, fallback
        ), (raw, label, fallback)


def test_the_checked_number_fields_take_the_stripped_text() -> None:
    from autoware_carla_scenario.editor import form_values

    assert form_values.float_field("", "''", "A", 1.5) == 1.5
    assert form_values.int_field("", "''", "A", 7) == 7
    with pytest.raises(ValueError, match=r"^A must be a number, got 'x '\.$"):
        form_values.float_field("x", "'x '", "A", 0.0)
    with pytest.raises(ValueError, match=r"^A must be a finite number, got 'inf'\.$"):
        form_values.float_field("inf", "'inf'", "A", 0.0)
    with pytest.raises(ValueError, match=r"^A must be a whole number, got '1\.5'\.$"):
        form_values.int_field("1.5", "'1.5'", "A", 0)
