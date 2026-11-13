"""Pins what rerun's conversion record and run summary do (#2124).

``RerunConversion``, ``latest()``, ``report()`` and ``_draw_radius_argument()``
are compared, through the names :mod:`autoware_carla_scenario.rerun` has for
them, with verbatim copies of them as they were before they moved into a
module of their own (the ``_Old*`` / ``_old_*`` oracles).
"""

from __future__ import annotations

import dataclasses
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from autoware_carla_scenario import rerun


@dataclass(frozen=True)
class _OldRerunConversion:
    label: str
    trajectory: Path
    recording: Path
    error: Optional[str] = None
    stats: Optional[dict[str, Any]] = field(default=None, compare=False)
    skipped: bool = False

    @property
    def ok(self) -> bool:
        """Whether the recording was written."""
        return self.error is None


def _old_latest(conversions: list[Any]) -> list[Any]:
    last: dict[str, Any] = {}
    for conversion in conversions:
        last[str(conversion.recording)] = conversion
    return list(last.values())


def _old_report(conversions: list[Any]) -> str:
    conversions = _old_latest(conversions)
    failed = [c for c in conversions if not c.ok]
    lines = [
        f"Rerun recordings: {len(conversions) - len(failed)}/{len(conversions)} "
        "written (open one with: rerun <file.rrd>)"
    ]
    lines.extend(
        f"  [{'SKIPPED' if c.skipped else 'FAILED'}] {c.label}: {c.error}"
        for c in failed
    )
    return "\n".join(lines)


def _old_draw_radius_argument(draw_radius_m: Optional[float]) -> dict[str, float]:
    return {} if draw_radius_m is None else {"draw_radius_m": draw_radius_m}


_LABELS = ["A", "B", "CutInScenario-3", "", "ü", "x:y", "{}"]
_ERRORS: list[Optional[str]] = [None, None, "E: x", "", "ValueError: {0}", "a\nb"]


def _random_conversions(make: Any, rng: random.Random) -> list[Any]:
    out = []
    for _ in range(rng.randrange(0, 12)):
        label = rng.choice(_LABELS)
        name = rng.choice(["A", "B", "C", "dir/A", "./A"])
        out.append(
            make(
                label,
                Path(f"{name}.trajectory.jsonl"),
                Path(f"{name}.rrd") if rng.random() < 0.9 else f"{name}.rrd",
                error=rng.choice(_ERRORS),
                stats=rng.choice([None, {}, {"ticks": rng.randrange(9)}]),
                skipped=rng.choice([False, False, True]),
            )
        )
    return out


def _pairs(seed: int) -> tuple[list[Any], list[Any]]:
    old = _random_conversions(_OldRerunConversion, random.Random(seed))
    new = _random_conversions(rerun.RerunConversion, random.Random(seed))
    return old, new


def _fields(conversion: Any) -> tuple[Any, ...]:
    return tuple(getattr(conversion, f.name) for f in dataclasses.fields(conversion))


@pytest.mark.parametrize("seed", range(300))
def test_latest_and_report_are_as_they_were(seed: int) -> None:
    old, new = _pairs(seed)
    assert [_fields(c) for c in rerun.latest(new)] == [
        _fields(c) for c in _old_latest(old)
    ]
    assert rerun.report(new) == _old_report(old)
    # latest() keeps the very objects, and the input list as it was.
    before = list(new)
    kept = rerun.latest(new)
    assert new == before and all(any(k is c for c in new) for k in kept)
    assert kept is not new


def test_latest_and_report_take_any_iterable_and_duck_typed_records() -> None:
    # Python never checked the element type: report() reads label, error,
    # skipped, ok and recording of whatever it is given.
    class Duck:
        def __init__(self, recording: object, ok: bool) -> None:
            self.recording = recording
            self.ok = ok
            self.label = "duck"
            self.error = "e"
            self.skipped = False

    ducks = [Duck(1, True), Duck("1", False), Duck(2, False)]
    assert rerun.report(ducks) == _old_report(ducks)  # type: ignore[arg-type]
    assert rerun.latest(iter(ducks)) == _old_latest(iter(ducks))  # type: ignore[arg-type]
    assert rerun.report([]) == _old_report([])
    for bad in (None, 3):
        with pytest.raises(TypeError) as new_error:
            rerun.report(bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError) as old_error:
            _old_report(bad)  # type: ignore[arg-type]
        assert str(new_error.value) == str(old_error.value)


@pytest.mark.parametrize(
    "value", [None, 0, 0.0, -1.5, 100, 150.5, float("inf"), float("nan"), True, "x"]
)
def test_draw_radius_argument_is_as_it_was(value: Any) -> None:
    new = rerun._draw_radius_argument(value)
    old = _old_draw_radius_argument(value)
    assert new == old or (new.keys() == old.keys() and value != value)
    assert [type(v) for v in new.values()] == [type(v) for v in old.values()]
    assert all(v is value for v in new.values())
    # A new dict each call: the caller may change it.
    assert rerun._draw_radius_argument(value) is not new


def test_the_conversion_record_is_as_it_was() -> None:
    new_cls = rerun.RerunConversion
    old_fields = [
        (f.name, f.type, f.default, f.compare, f.init, f.repr, f.hash)
        for f in dataclasses.fields(_OldRerunConversion)
    ]
    new_fields = [
        (f.name, f.type, f.default, f.compare, f.init, f.repr, f.hash)
        for f in dataclasses.fields(new_cls)
    ]
    assert new_fields == old_fields
    assert new_cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
    assert new_cls.__name__ == "RerunConversion"
    assert new_cls.__qualname__ == "RerunConversion"
    assert new_cls.__doc__ is not None and "How converting one trajectory" in (
        new_cls.__doc__
    )
    args = ("A", Path("A.trajectory.jsonl"), Path("A.rrd"))
    keywords: list[dict[str, Any]] = [
        {},
        {"error": "E: x"},
        {"stats": {"ticks": 3}},
        {"error": "x", "stats": {}, "skipped": True},
    ]
    for kwargs in keywords:
        new = new_cls(*args, **kwargs)
        old = _OldRerunConversion(*args, **kwargs)
        assert _fields(new) == _fields(old)
        assert new.ok == old.ok
        assert repr(new) == repr(old).replace("_OldRerunConversion", "RerunConversion")
        assert hash(new) == hash(old)
        assert new == new_cls(*args, **{**kwargs, "stats": {"other": 1}})
        with pytest.raises(dataclasses.FrozenInstanceError):
            new.error = "y"  # type: ignore[misc]
        assert pickle.loads(pickle.dumps(new)) == new
    with pytest.raises(TypeError):
        new_cls("A", Path("A"))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        new_cls("A", Path("A"), Path("B"), None, None, False, 1)  # type: ignore[call-arg]


def test_rerun_keeps_its_names() -> None:
    assert {"RerunConversion", "latest", "report"} <= set(rerun.__all__)
    for name in ("RerunConversion", "latest", "report", "_draw_radius_argument"):
        assert hasattr(rerun, name), name
