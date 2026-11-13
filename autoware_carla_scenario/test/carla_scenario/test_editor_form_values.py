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
            _old_parse_int_list, text
        ), text


@pytest.mark.parametrize("raw", _NON_TEXT, ids=repr)
def test_every_non_text_value_parses_as_before(raw: Any) -> None:
    assert _outcome(forms.parse_int_list, raw) == _outcome(_old_parse_int_list, raw)


def test_lists_of_random_items_parse_as_before() -> None:
    rng = random.Random(42)
    pool: list[Any] = [*_ITEMS[:-4], *_SEPARATORS, None, 1, -2, 3.5, True]
    for _ in range(400):
        raw: Any = [rng.choice(pool) for _ in range(rng.randint(0, 5))]
        if rng.random() < 0.3:
            raw = tuple(raw)
        assert _outcome(forms.parse_int_list, raw) == _outcome(
            _old_parse_int_list, raw
        ), raw


@pytest.mark.parametrize("kind", ["int_list", "lanelet_list", "int_list_or_ref"])
def test_a_list_field_parses_as_before(kind: FieldKind) -> None:
    spec = FieldSpec("ids", "Ids", kind, (), required=False)
    for text in (*_TEXTS, *_random_texts(7, 100)):
        expected = _outcome(_old_parse_int_list, text.strip())
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
