"""Failure.from_dict() reads back what to_dict() wrote as it always did.

The library check compiles from_dict() through the checked readers that call
it with a document json.loads() read (a ``JSONValue`` to Codon), so its body
is written for Codon; these pin what it returns and raises against a copy of
the reader as it was, for every JSON type in each field, and at the same
depth of the stack for values nested deep enough to raise RecursionError.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any, Optional

import pytest

from autoware_carla_scenario.batch_resume import KeptResult, SummaryEntry
from autoware_carla_scenario.failure import Failure, FailureCategory
from autoware_carla_scenario.slurm.collect import RecordedResult
from autoware_carla_scenario.slurm.manifest import readable_result
from autoware_carla_scenario.slurm.records import JsonMembers


def _old_from_dict(data: Mapping[str, Any]) -> Failure:
    """Failure.from_dict() as it was, verbatim."""
    detail = data.get("detail")
    try:
        category = FailureCategory[str(data.get("category", "")).upper()]
    except KeyError:
        category = FailureCategory.OTHER
    return Failure(
        condition=str(data.get("condition", "")),
        category=category,
        at=float(data.get("at") or 0.0),
        message=str(data.get("message", "")),
        detail=None if detail is None else str(detail),
        also_triggered=[str(label) for label in data.get("also_triggered") or []],
    )


def _canon(failure: Failure) -> tuple[Any, ...]:
    """Every field of *failure*, with the type of each value and a float as
    its repr (``nan``, ``-0.0``)."""
    return (
        type(failure.condition).__name__,
        failure.condition,
        failure.category,
        type(failure.at).__name__,
        repr(failure.at),
        type(failure.message).__name__,
        failure.message,
        None if failure.detail is None else type(failure.detail).__name__,
        failure.detail,
        type(failure.also_triggered).__name__,
        [(type(label).__name__, label) for label in failure.also_triggered],
    )


def _outcome(read: Callable[[Any], Failure], data: Any) -> tuple[Any, ...]:
    """What *read* returns for *data*, or the type and message it raises."""
    try:
        return ("returns", _canon(read(data)))
    except Exception as exc:  # noqa: BLE001 - the exception is the outcome
        return ("raises", type(exc).__name__, str(exc))


def _at_depth(frames: int, read: Callable[[Any], Failure], data: Any) -> Any:
    """:func:`_outcome` of *read* called *frames* calls deeper in the stack."""
    if frames > 0:
        return _at_depth(frames - 1, read, data)
    return _outcome(read, data)


_VALUES: list[Any] = [
    None,
    True,
    False,
    0,
    1,
    -1,
    10**400,
    -(10**400),
    0.0,
    -0.0,
    1.5,
    -2.25,
    1e308,
    math.nan,
    math.inf,
    -math.inf,
    "",
    "abc",
    "1.5",
    " 2 ",
    "nan",
    "-inf",
    "1e500",
    "collision",
    "COLLISION",
    "Traffic_Rule",
    "ß",
    "日本",
    [],
    [1],
    ["a", 2, None, True, 1.5],
    [[1, [2]], {"k": "v"}],
    {},
    {"a": 1},
    {"category": "collision"},
]

_FIELDS = ["condition", "category", "at", "message", "detail", "also_triggered"]

_FULL: dict[str, Any] = {
    "condition": "fail[0](hit)",
    "category": "collision",
    "at": 12.5,
    "message": "crash",
    "detail": None,
    "also_triggered": ["fail[1](late)"],
}


def _documents() -> list[Any]:
    documents: list[Any] = [{}, dict(_FULL), {**_FULL, "extra": [1, 2]}]
    for field in _FIELDS:
        documents.append({k: v for k, v in _FULL.items() if k != field})
        for value in _VALUES:
            documents.append({**_FULL, field: value})
            documents.append({field: value})
    for category in FailureCategory:
        documents.append({"category": category.name.lower()})
        documents.append({"category": category.name})
        documents.append({"category": category.name.title()})
    return documents


@pytest.mark.parametrize("data", _documents(), ids=repr)
def test_every_json_value_of_every_field_reads_as_before(data: Any) -> None:
    assert _outcome(Failure.from_dict, data) == _outcome(_old_from_dict, data)


@pytest.mark.parametrize("data", _documents(), ids=repr)
def test_every_document_reads_back_from_its_json_text_as_before(data: Any) -> None:
    text = json.dumps(data)
    assert _outcome(Failure.from_dict, json.loads(text)) == _outcome(
        _old_from_dict, json.loads(text)
    )


@pytest.mark.parametrize("data", [[], [1], "x", 5, None, ["category"]], ids=repr)
def test_a_document_that_is_no_mapping_raises_as_before(data: Any) -> None:
    new = _outcome(Failure.from_dict, data)
    assert new == _outcome(_old_from_dict, data)
    assert new[:2] == ("raises", "AttributeError")


def test_what_it_returns_and_raises_is_pinned() -> None:
    assert _canon(Failure.from_dict(_FULL)) == _canon(
        Failure(
            "fail[0](hit)",
            FailureCategory.COLLISION,
            12.5,
            "crash",
            None,
            ["fail[1](late)"],
        )
    )
    empty = Failure.from_dict({})
    assert _canon(empty) == _canon(
        Failure("", FailureCategory.OTHER, 0.0, "", None, [])
    )
    # A false `at` is 0.0, -0.0 and "" among them; a true one is converted.
    false_values: list[Any] = [None, False, 0, -0.0, "", [], {}]
    for at in false_values:
        assert repr(Failure.from_dict({"at": at}).at) == "0.0"
    assert Failure.from_dict({"at": True}).at == 1.0
    assert Failure.from_dict({"at": "1e500"}).at == math.inf
    assert math.isnan(Failure.from_dict({"at": "nan"}).at)
    with pytest.raises(OverflowError, match="int too large to convert to float"):
        Failure.from_dict({"at": 10**400})
    with pytest.raises(ValueError, match="could not convert string to float: 'abc'"):
        Failure.from_dict({"at": "abc"})
    with pytest.raises(TypeError, match="float\\(\\) argument must be"):
        Failure.from_dict({"at": [1]})
    # A category by its name in any case; anything else is OTHER.
    assert Failure.from_dict({"category": "Traffic_Rule"}).category == (
        FailureCategory.TRAFFIC_RULE
    )
    for category in (None, 1, True, ["collision"], {"collision": 1}, "nope"):
        assert Failure.from_dict({"category": category}).category == (
            FailureCategory.OTHER
        )
    # Each value is str() of what JSON held; also_triggered iterates any
    # true value (a string by its characters, an object by its keys).
    failure = Failure.from_dict(
        {"condition": None, "message": 1.5, "detail": False, "also_triggered": "ab"}
    )
    assert (failure.condition, failure.message, failure.detail) == (
        "None",
        "1.5",
        "False",
    )
    assert failure.also_triggered == ["a", "b"]
    assert Failure.from_dict({"also_triggered": {"x": 1, "y": 2}}).also_triggered == [
        "x",
        "y",
    ]
    assert Failure.from_dict({"also_triggered": 0}).also_triggered == []
    with pytest.raises(TypeError, match="'int' object is not iterable"):
        Failure.from_dict({"also_triggered": 5})
    with pytest.raises(AttributeError, match="'list' object has no attribute 'get'"):
        Failure.from_dict([])  # type: ignore[arg-type]


def _nested_list(depth: int) -> Any:
    value: Any = []
    for _ in range(depth):
        value = [value]
    return value


def _nested_dict(depth: int) -> Any:
    value: Any = {}
    for _ in range(depth):
        value = {"k": value}
    return value


#: Where a value nested deep is put: each field str() reads, and a label.
_DEEP_PLACES: list[tuple[str, Callable[[int], Any]]] = [
    ("condition", lambda d: {"condition": _nested_list(d)}),
    ("category", lambda d: {"category": _nested_list(d)}),
    ("message", lambda d: {"message": _nested_dict(d)}),
    ("detail", lambda d: {"detail": _nested_list(d)}),
    ("label", lambda d: {"also_triggered": [_nested_list(d)]}),
]


def _threshold(build: Callable[[int], Any]) -> int:
    """The least nesting at which the old reader raises RecursionError here."""
    low, high = 1, 1
    while _at_depth(0, _old_from_dict, build(high))[0] == "returns":
        low, high = high, high * 2
        if high > 1 << 22:
            pytest.fail("no RecursionError found")
    while low < high:
        mid = (low + high) // 2
        if _at_depth(0, _old_from_dict, build(mid))[0] == "returns":
            low = mid + 1
        else:
            high = mid
    return low


@pytest.mark.parametrize("place", [p for p, _ in _DEEP_PLACES])
def test_a_value_nested_deep_raises_recursion_error_at_the_same_depth(
    place: str,
) -> None:
    build = dict(_DEEP_PLACES)[place]
    # Found a call deeper than the test reads: the range straddles it here.
    threshold = _threshold(build)
    kinds = set()
    for depth in range(max(1, threshold - 4), threshold + 5):
        data = build(depth)
        for frames in (0, 1, 2, 5, 40):
            old = _at_depth(frames, _old_from_dict, data)
            assert _at_depth(frames, Failure.from_dict, data) == old, (
                place,
                depth,
                frames,
            )
            if frames == 0:
                kinds.add(old[0] if old[0] == "returns" else old[1])
    assert kinds == {"returns", "RecursionError"}


# -- The checked readers that call it ----------------------------------------------


def _old_kept_failure(members: dict[str, str]) -> Optional[Failure]:
    failure = json.loads(members.get("failure", "null"))
    if not isinstance(failure, dict):
        return None
    return _old_from_dict(failure)


def _old_recorded_failure(members: JsonMembers) -> Optional[Failure]:
    failure_json = members.value_text("failure") if members.has("failure") else "null"
    if failure_json == "null":
        return None
    return _old_from_dict(json.loads(failure_json))


def _result_members(failure: Any) -> JsonMembers:
    members = JsonMembers()
    members.set_text("passed", "false")
    members.set_text("message", '"m"')
    members.set_text("elapsed_seconds", "1.0")
    members.set_text("condition_statuses", "[]")
    if failure is not _MISSING:
        members.set_text("failure", json.dumps(failure))
    return members


class _Missing:
    """No ``failure`` member: by name in a test's id, the same in each worker."""

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()

_CALLER_FAILURES: list[Any] = [
    _MISSING,
    None,
    [],
    "x",
    3,
    *[d for d in _documents() if d is not None],
]


def _maybe(read: Callable[[], Optional[Failure]]) -> tuple[Any, ...]:
    try:
        failure = read()
    except Exception as exc:  # noqa: BLE001 - the exception is the outcome
        return ("raises", type(exc).__name__, str(exc))
    return ("returns", None if failure is None else _canon(failure))


@pytest.mark.parametrize("failure", _CALLER_FAILURES, ids=repr)
def test_the_readers_read_a_failure_as_before(failure: Any) -> None:
    members = _result_members(failure)
    texts = {k: members.value_text(k) for k in members.keys()}
    kept = KeptResult(SummaryEntry("s", False, texts))
    assert _maybe(lambda: kept.failure) == _maybe(lambda: _old_kept_failure(texts))
    recorded = RecordedResult(members)
    assert _maybe(lambda: recorded.failure) == _maybe(
        lambda: _old_recorded_failure(members)
    )
    old_readable = not isinstance(failure, (list, str, int)) or failure is None
    if isinstance(failure, dict):
        try:
            _old_from_dict(failure)
        except (TypeError, ValueError, AttributeError, OverflowError):
            old_readable = False
    assert readable_result(members) is old_readable


def _too_deep_for_loads() -> int:
    """A nesting json.loads() does not read, called as deep in the stack as
    this function is: some ten thousand levels on Python 3.12, but from 3.14
    its limit is the thread's stack, which reads a hundred thousand on some
    hosts."""
    depth = 100_000
    while True:
        try:
            json.loads("[" * depth + "]" * depth)
        except RecursionError:
            return depth
        if depth >= 10_000_000:
            pytest.skip("json.loads() reads ten million levels here")
        depth *= 10


def test_the_readers_raise_recursion_error_for_a_failure_nested_deep() -> None:
    # Deeper than json.loads() reads here, written by hand: json.dumps()
    # cannot write it.  The readers call json.loads() deeper in the stack.
    depth = _too_deep_for_loads()
    members = JsonMembers()
    members.put("failure", '{"also_triggered": [' + "[" * depth + "]" * depth + "]}")
    texts = {"failure": members.value_text("failure")}
    for read in (
        lambda: KeptResult(SummaryEntry("s", False, texts)).failure,
        lambda: RecordedResult(members).failure,
    ):
        with pytest.raises(RecursionError):
            read()
