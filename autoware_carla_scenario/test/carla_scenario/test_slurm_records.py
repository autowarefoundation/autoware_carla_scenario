"""slurm.records: a JSON object held as the text of each of its members.

The members must give back, byte for byte, what ``slurm.manifest.write_json``
writes for the object, and read back as ``json.loads()`` reads it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from autoware_carla_scenario.slurm.manifest import write_json
from autoware_carla_scenario.slurm.records import JsonMembers, parse_object

#: Bounded and derandomized, as test_odd_properties.py's: the same examples,
#: and so the same verdict, on every run, in a few seconds.
_SETTINGS = settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _same(a: Any, b: Any) -> bool:
    """Whether *a* and *b* are the same JSON value: of the same type (a bool
    is no int), NaN equal to NaN, -0.0 not 0.0, and objects in the same key
    order."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    return bool(a == b)


def _written(tmp_path: Path, data: Any) -> bytes:
    path = tmp_path / "doc.json"
    path.unlink(missing_ok=True)
    write_json(path, data)
    return path.read_bytes()


def _dumped(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _check_members(members: JsonMembers, expected: dict[str, Any]) -> None:
    """*members* hold *expected* (as ``json.loads()`` read it): its keys in
    order, and each value."""
    assert members.keys() == list(expected)
    assert len(members) == len(expected)
    for key, value in expected.items():
        assert members.has(key)
        assert _same(members.value(key), value), key
        assert members.value_text(key) == json.dumps(value, ensure_ascii=False)
        assert (
            members.member_text(key)
            == (json.dumps({key: value}, indent=2, ensure_ascii=False)[2:-2])
        )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

#: Any character a str may hold, lone surrogates and NUL included.
_ANY_CHAR = st.characters(codec=None, exclude_categories=())
#: Text a UTF-8 file can hold (no surrogates).
_TEXT = st.text(st.characters(exclude_categories=["Cs"]), max_size=12)
_ANY_TEXT = st.text(_ANY_CHAR, max_size=12)

_SPECIAL_FLOATS = st.sampled_from(
    [
        math.nan,
        math.inf,
        -math.inf,
        -0.0,
        0.0,
        1e308 * 10,
        5e-324,
        0.1 + 0.2,
        1.7976931348623157e308,
        2.0**53 + 1,
        123456789.12345679,
    ]
)
_NUMBERS = st.one_of(
    st.integers(),
    st.integers(min_value=-(10**1000), max_value=10**1000),
    st.floats(allow_nan=True, allow_infinity=True),
    _SPECIAL_FLOATS,
)


def _values(text: st.SearchStrategy[str]) -> st.SearchStrategy[Any]:
    scalars = st.one_of(st.none(), st.booleans(), _NUMBERS, text)
    return st.recursive(
        scalars,
        lambda inner: st.one_of(
            st.lists(inner, max_size=5),
            st.dictionaries(text, inner, max_size=5),
        ),
        max_leaves=25,
    )


_KEYS = st.one_of(_TEXT, st.sampled_from(["", "a", "é", "\x00", "日本語", '"\\']))
_OBJECTS = st.dictionaries(_KEYS, _values(_TEXT), max_size=8)
_ANY_OBJECTS = st.dictionaries(_ANY_TEXT, _values(_ANY_TEXT), max_size=6)

#: Object texts with keys named more than once, laid out every way JSON allows.
_SMALL_KEYS = st.sampled_from(["a", "b", "c", "é", ""])
_SPACE = st.sampled_from(["", " ", "\n", "\t", "\r\n  "])


@st.composite
def _object_texts(draw: st.DrawFn) -> str:
    pairs = draw(st.lists(st.tuples(_SMALL_KEYS, _values(_TEXT)), max_size=8))
    ascii_ = draw(st.booleans())
    indent = draw(st.sampled_from([None, 0, 2, 4]))
    sp = draw(_SPACE)
    members = [
        sp
        + json.dumps(key, ensure_ascii=ascii_)
        + sp
        + ":"
        + sp
        + json.dumps(value, ensure_ascii=ascii_, indent=indent)
        + sp
        for key, value in pairs
    ]
    return draw(_SPACE) + "{" + ",".join(members) + "}" + draw(_SPACE)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@_SETTINGS
@given(obj=_OBJECTS)
def test_from_value_writes_what_write_json_writes(
    tmp_path_factory: pytest.TempPathFactory, obj: dict[str, Any]
) -> None:
    members = JsonMembers.from_value(obj)
    written = _written(tmp_path_factory.mktemp("w"), obj)
    assert members.object_text().encode("utf-8") == written
    _check_members(members, json.loads(written))


@_SETTINGS
@given(obj=_ANY_OBJECTS)
def test_from_value_of_any_str_is_what_json_dumps_gives(
    tmp_path_factory: pytest.TempPathFactory, obj: dict[str, Any]
) -> None:
    # Lone surrogates included: write_json cannot encode them, and neither can
    # the members' text.
    members = JsonMembers.from_value(obj)
    text = members.object_text()
    assert text == _dumped(obj)
    tmp = tmp_path_factory.mktemp("w")
    try:
        written = _written(tmp, obj)
    except UnicodeEncodeError:
        with pytest.raises(UnicodeEncodeError):
            text.encode("utf-8")
    else:
        assert text.encode("utf-8") == written
    _check_members(members, json.loads(text))


@_SETTINGS
@given(text=_object_texts())
def test_parse_object_reads_what_json_loads_reads(text: str) -> None:
    members = parse_object(text)
    assert members is not None
    data = json.loads(text)
    assert members.object_text() == _dumped(data)
    _check_members(members, data)


@_SETTINGS
@given(obj=_ANY_OBJECTS)
def test_parse_object_of_written_text_gives_it_back(obj: dict[str, Any]) -> None:
    for ascii_ in (False, True):
        text = json.dumps(obj, ensure_ascii=ascii_)
        members = parse_object(text)
        assert members is not None
        assert members.object_text() == _dumped(json.loads(text))
        assert members.object_text() == JsonMembers.from_value(obj).object_text()


@_SETTINGS
@given(obj=_OBJECTS, key=_KEYS, value=_values(_TEXT))
def test_set_text_sets_a_member_as_a_dict_does(
    obj: dict[str, Any], key: str, value: Any
) -> None:
    members = JsonMembers.from_value(obj)
    members.set_text(key, json.dumps(value))
    expected = dict(obj)
    expected[key] = value
    assert members.object_text() == _dumped(expected)
    _check_members(members, json.loads(_dumped(expected)))


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

#: The document the pins of write_json write (test_slurm_manifest.py): a
#: value of each kind JSON writes, and the keys json turns into strings.
_DOCUMENT: dict[Any, Any] = {
    "b": 1,
    "a": [1.5, -0.0, math.nan, math.inf, -math.inf, 10**30, 0, -7],
    "é": 'ü \x00"\\/\t',
    "nested": {"t": True, "f": False, "n": None, "empty": {}, "list": []},
    "tuple": (1, "x"),
    1: "int key",
    2.5: "float key",
    False: "bool key",
    None: "none key",
}


def test_from_value_writes_the_pinned_document(tmp_path: Path) -> None:
    members = JsonMembers.from_value(_DOCUMENT)
    written = _written(tmp_path, _DOCUMENT)
    assert members.object_text().encode() == written
    assert members.keys() == [
        "b",
        "a",
        "é",
        "nested",
        "tuple",
        "1",
        "2.5",
        "false",
        "null",
    ]
    assert members.member_text("tuple") == '  "tuple": [\n    1,\n    "x"\n  ]'
    assert members.value_text("tuple") == '[1, "x"]'
    assert members.value("tuple") == [1, "x"]
    assert members.value("false") == "bool key"
    _check_members(members, json.loads(written))


@pytest.mark.parametrize(
    ("text", "keys"),
    [
        ('{"a": 1, "b": 2, "a": 3}', ["a", "b"]),
        ('{"a": {"x": 1}, "a": [], "a": null}', ["a"]),
        ('{"b": 0, "a": 1, "b": NaN}', ["b", "a"]),
        ('{"\\u00e9": 1, "é": 2}', ["é"]),
        ('{"": 1, "": -0.0}', [""]),
    ],
)
def test_parse_object_keeps_a_repeated_key_first_with_its_last_value(
    text: str, keys: list[str]
) -> None:
    members = parse_object(text)
    assert members is not None
    assert members.keys() == keys
    data = json.loads(text)
    assert members.object_text() == _dumped(data)
    _check_members(members, data)


@pytest.mark.parametrize(
    ("text", "value", "value_text"),
    [
        ('{"a": NaN}', math.nan, "NaN"),
        ('{"a": Infinity}', math.inf, "Infinity"),
        ('{"a": -Infinity}', -math.inf, "-Infinity"),
        ('{"a": 1e400}', math.inf, "Infinity"),
        ('{"a": -1e400}', -math.inf, "-Infinity"),
        ('{"a": -0.0}', -0.0, "-0.0"),
        ('{"a": -0}', 0, "0"),
        ('{"a": 1E2}', 100.0, "100.0"),
        ('{"a": 0.30000000000000004}', 0.1 + 0.2, "0.30000000000000004"),
        ('{"a": 1e-400}', 0.0, "0.0"),
        ('{"a": 12345678901234567890123}', 12345678901234567890123, None),
        ('{"a": true}', True, "true"),
        ('{"a": 1}', 1, "1"),
        ('{"a": 1.0}', 1.0, "1.0"),
        ('{"a": "\\u0000"}', "\x00", '"\\u0000"'),
        ('{"a": "\\ud800"}', "\ud800", '"\ud800"'),
        ('{"a": "\\u65e5"}', "日", '"日"'),
        ('{"a": {}}', {}, "{}"),
        ('{"a": []}', [], "[]"),
        ('{"a": [{}]}', [{}], "[{}]"),
        ('{"a": {"b": [1, {"c": 2}]}}', {"b": [1, {"c": 2}]}, '{"b": [1, {"c": 2}]}'),
    ],
)
def test_value_reads_as_json_loads_reads(
    text: str, value: Any, value_text: str | None
) -> None:
    members = parse_object(text)
    assert members is not None
    assert _same(members.value("a"), value)
    assert _same(members.value("a"), json.loads(text)["a"])
    if value_text is not None:
        assert members.value_text("a") == value_text


@pytest.mark.parametrize(
    "text",
    ["[]", "[{}]", "null", "true", "1", "-0.0", "NaN", '"{}"', ' "x" '],
)
def test_parse_object_of_json_that_is_no_object_is_none(text: str) -> None:
    assert parse_object(text) is None


@pytest.mark.parametrize(
    "text",
    ["", " ", "{", '{"a"}', '{"a": 1,}', "{'a': 1}", '{"a": 1} x', "nan", "{}{}"],
)
def test_parse_object_raises_what_json_loads_raises(text: str) -> None:
    with pytest.raises(json.JSONDecodeError) as expected:
        json.loads(text)
    with pytest.raises(json.JSONDecodeError) as got:
        parse_object(text)
    assert str(got.value) == str(expected.value)


def test_an_empty_object(tmp_path: Path) -> None:
    for members in (JsonMembers(), JsonMembers.from_value({}), parse_object("{}")):
        assert members is not None
        assert members.object_text() == "{}\n"
        assert members.object_text().encode() == _written(tmp_path, {})
        assert members.keys() == []
        assert len(members) == 0
        assert not members.has("")


def test_a_missing_member_is_a_key_error() -> None:
    members = parse_object('{"a": 1}')
    assert members is not None
    assert not members.has("b")
    for read in (members.value, members.value_text, members.member_text):
        with pytest.raises(KeyError):
            read("b")


def test_set_text_builds_an_object_in_order(tmp_path: Path) -> None:
    members = JsonMembers()
    members.set_text("job_id", "null")
    members.set_text("submitted_at", "1700000000.25")
    members.set_text("cases", "[0, 1, 2]")
    members.set_text("job_id", '"123"')
    data = {"job_id": "123", "submitted_at": 1700000000.25, "cases": [0, 1, 2]}
    assert members.object_text().encode() == _written(tmp_path, data)


def test_set_text_raises_for_text_that_is_not_json() -> None:
    members = JsonMembers()
    with pytest.raises(json.JSONDecodeError):
        members.set_text("a", "{")
    assert not members.has("a")


def test_from_value_refuses_what_is_no_dict() -> None:
    values: list[Any] = [[], None, "x", 1, ({"a": 1},)]
    for value in values:
        with pytest.raises(TypeError, match="a JSON object's members are a dict's"):
            JsonMembers.from_value(value)


@pytest.mark.parametrize(
    "data",
    [
        {1: "a", "1": "b"},
        {"null": 0, None: 1},
        {True: 0, "true": 1},
        {2.5: 0, "2.5": 1},
    ],
)
def test_from_value_refuses_keys_json_writes_alike(data: dict[Any, Any]) -> None:
    with pytest.raises(ValueError, match="names the member .* twice"):
        JsonMembers.from_value(data)


@pytest.mark.parametrize(
    "data",
    [
        {"a": object()},
        {"a": {1, 2}},
        {(1, 2): 1},
        {"ok": 1, "a": b"bytes"},
    ],
)
def test_from_value_raises_what_write_json_raises(
    tmp_path: Path, data: dict[Any, Any]
) -> None:
    with pytest.raises(TypeError) as expected:
        write_json(tmp_path / "doc.json", data)
    with pytest.raises(TypeError) as got:
        JsonMembers.from_value(data)
    assert str(got.value) == str(expected.value)


def test_from_value_refuses_a_circular_object(tmp_path: Path) -> None:
    looped: dict[str, Any] = {}
    looped["self"] = looped
    # str(), not match=: from 3.14.10 json adds a note ("when serializing
    # dict item 'self'"), which match= reads as part of the message.
    with pytest.raises(ValueError) as written:
        write_json(tmp_path / "doc.json", looped)
    assert str(written.value) == "Circular reference detected"
    with pytest.raises(ValueError) as read:
        JsonMembers.from_value(looped)
    assert str(read.value) == "Circular reference detected"


@pytest.mark.skipif(
    not hasattr(sys, "set_int_max_str_digits"),
    reason="Python 3.10 writes an int of any length",
)
def test_from_value_raises_for_an_int_python_will_not_write(tmp_path: Path) -> None:
    data = {"a": 10**5000}
    with pytest.raises(ValueError) as expected:
        write_json(tmp_path / "doc.json", data)
    with pytest.raises(ValueError) as got:
        JsonMembers.from_value(data)
    assert str(got.value) == str(expected.value)


@pytest.mark.skipif(
    hasattr(sys, "set_int_max_str_digits"),
    reason="Python 3.11+ refuses an int this long",
)
def test_from_value_writes_an_int_of_any_length_on_310(tmp_path: Path) -> None:
    data = {"a": 10**5000}
    members = JsonMembers.from_value(data)
    assert members.object_text().encode() == _written(tmp_path, data)
    assert members.value("a") == 10**5000


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def _nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def _loads_limit() -> int:
    """The deepest array ``json.loads()`` reads here, called as deep in the
    stack as this function is."""
    low, high = 1, 50_000
    while low < high:
        middle = (low + high + 1) // 2
        try:
            json.loads(_nested(middle))
        except RecursionError:
            high = middle - 1
        else:
            low = middle
    return low


def test_parse_object_reads_what_json_loads_reads_nested_however_deep() -> None:
    # A value deeper than json's encoder written in Python lays out (a
    # thousand levels): reading holds it as the C encoder writes it.
    limit = _loads_limit()
    for depth in sorted({min(limit, 5000), limit}):
        text = '{"a": 1, "deep": ' + _nested(depth - 1) + "}"
        members = parse_object(text)
        assert members is not None
        assert members.keys() == ["a", "deep"]
        assert members.value_text("deep") == _nested(depth - 1)
        assert members.value("deep") == json.loads(text)["deep"]
        members.set_text("a", members.value_text("deep"))
        assert members.value("a") == members.value("deep")


def test_parse_object_raises_what_json_loads_raises_nested_too_deep() -> None:
    text = '{"deep": ' + _nested(2 * _loads_limit()) + "}"
    with pytest.raises(RecursionError):
        json.loads(text)
    with pytest.raises(RecursionError):
        parse_object(text)


@pytest.mark.parametrize("depth", [500, 5000])
def test_a_member_too_deep_to_lay_out_raises_as_write_json_does(
    tmp_path: Path, depth: int
) -> None:
    # Laid out as write_json lays it out: json.dump() takes json's encoder
    # written in Python on every version, which lays out five hundred levels
    # but not five thousand (which json.loads() reads on 3.12). json.dumps()
    # takes the C encoder from 3.13 on, which on 3.14 lays out both, so the
    # members must not lay out with it.
    text = '{"a": 1, "deep": ' + _nested(depth) + "}"
    if depth >= _loads_limit():
        pytest.skip("json.loads() reads no text this deep here")
    members = parse_object(text)
    assert members is not None
    data = json.loads(text)
    try:
        written = _written(tmp_path, data)
    except RecursionError:
        with pytest.raises(RecursionError):
            members.object_text()
        with pytest.raises(RecursionError):
            members.member_text("deep")
        assert members.member_text("a") == '  "a": 1'
    else:
        assert members.object_text().encode() == written


def test_from_value_lays_out_nested_keys_json_writes_alike(tmp_path: Path) -> None:
    # write_json writes both keys of the nested dict; json.loads() reads the
    # last.
    data = {"a": {1: "int", "1": "str"}, "b": [{None: 0, "null": 1}]}
    members = JsonMembers.from_value(data)
    written = _written(tmp_path, data)
    assert members.object_text().encode() == written
    assert members.member_text("a") == '  "a": {\n    "1": "int",\n    "1": "str"\n  }'
    assert members.value("a") == {"1": "str"}
    assert members.value("b") == json.loads(written)["b"]
    members.set_text("a", members.value_text("a"))
    assert members.member_text("a") == '  "a": {\n    "1": "str"\n  }'
