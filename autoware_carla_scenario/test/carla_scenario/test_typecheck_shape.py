"""A sweep compiles once per shape of its config, not once per case.

Codon types the driver statically, so cases that differ only in their values
are one program to it: the check is keyed by the driver with each value
replaced by its type (``Driver.shape``).  No Codon needed: the compile is
replaced by a counter.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from autoware_carla_scenario.typecheck import check
from autoware_carla_scenario.typecheck.check import (
    CACHE_DIR_ENV,
    Diagnostic,
    TypeCheckResult,
    _Module,
    _Sources,
    typecheck_scenario,
)
from autoware_carla_scenario.typecheck.driver import render_driver
from autoware_carla_scenario.typecheck.none_check import NoneCheckSkipped


class _Turn(enum.Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclasses.dataclass
class _Npc:
    spawn_lanelet_id: int = 0
    spawn_s: float = 0.0


@dataclasses.dataclass
class _Config:
    timeout_seconds: float = 5.0
    min_speed_kmh: Optional[float] = None
    stop_line: Any = None
    label: str = "case"
    enabled: bool = True
    ids: list[int] = dataclasses.field(default_factory=list)
    npcs: list[_Npc] = dataclasses.field(default_factory=list)
    turn: _Turn = _Turn.LEFT

    def __post_init__(self) -> None:
        self.npcs = [_Npc(**n) if isinstance(n, dict) else n for n in self.npcs]


class _Scenario:
    pass


def _shape(**values: Any) -> str:
    return render_driver(_Scenario, _Config, values).shape


_CASE: dict[str, Any] = {
    "timeout_seconds": 5.0,
    "min_speed_kmh": 3,
    "stop_line": 1.5,
    "label": "a",
    "enabled": True,
    "ids": [460, 265],
    "npcs": [{"spawn_lanelet_id": 1, "spawn_s": 2}],
    "turn": _Turn.LEFT,
}


def test_cases_that_differ_only_in_their_values_have_one_shape() -> None:
    other = {
        "timeout_seconds": 12.5,
        "min_speed_kmh": 7.25,
        "stop_line": float("nan"),
        "label": "another label",
        "enabled": False,
        "ids": [1, 2],
        "npcs": [{"spawn_lanelet_id": 9, "spawn_s": 0.5}],
        "turn": _Turn.RIGHT,
    }
    assert _shape(**_CASE) == _shape(**other)
    driver = render_driver(_Scenario, _Config, _CASE)
    assert driver.source != render_driver(_Scenario, _Config, other).source
    # The shape is the program, each value its type: line for line.
    assert len(driver.shape.splitlines()) == len(driver.source.splitlines())
    lines = driver.shape.splitlines()
    by_key = {key: lines[line - 1].strip() for line, key in driver.config_lines.items()}
    assert by_key == {
        "scenario.timeout_seconds": "config.timeout_seconds = <float>",
        "scenario.min_speed_kmh": "config.min_speed_kmh = <float>",
        "scenario.stop_line": "config.stop_line = <float>",
        "scenario.label": "config.label = <str>",
        "scenario.enabled": "config.enabled = <bool>",
        "scenario.ids": "config.ids = [<int>, <int>]",
        # A nested dataclass is one field to a line: Codon's error names it (#531).
        "scenario.npcs[0]": "config.npcs = [_Npc(",
        "scenario.npcs[0].spawn_lanelet_id": "spawn_lanelet_id=<int>,",
        "scenario.npcs[0].spawn_s": "spawn_s=<float>,",
        "scenario.npcs": ")]",
        "scenario.turn": "config.turn = <_Turn>",
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        # An int where the field takes Any stays an int: another type.
        ("stop_line", 1),
        # None, not a value of the Optional.
        ("min_speed_kmh", None),
        # A str where the config takes a float: Codon refuses this one.
        ("timeout_seconds", "fast"),
        # Another length of a list, an empty one (its type is inferred).
        ("ids", [460]),
        ("ids", []),
        # A key the nested dataclass does not have.
        ("npcs", [{"spawn_lanelet_id": 1, "spawn_s": 2, "extra": 1}]),
        # An int Codon's 64-bit literal cannot hold.
        ("ids", [460, 2**64]),
        # A string with an escape is kept as it is.
        ("label", "a\nb"),
    ],
)
def test_another_type_is_another_shape(key: str, value: Any) -> None:
    assert _shape(**{**_CASE, key: value}) != _shape(**_CASE)


def test_another_key_or_order_is_another_shape() -> None:
    assert _shape(**_CASE, wrong_key=1) != _shape(**_CASE)
    without = {k: v for k, v in _CASE.items() if k != "label"}
    assert _shape(**without) != _shape(**_CASE)
    reordered = dict(reversed(list(_CASE.items())))
    assert _shape(**reordered) != _shape(**_CASE)


def test_the_case_names_its_values() -> None:
    driver = render_driver(_Scenario, _Config, {"timeout_seconds": 5, "label": "a"})
    assert driver.case == {"timeout_seconds": "5", "label": "'a'"}


@pytest.fixture
def compiles(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Each program compiled, by its driver source; no Codon and no mypy."""
    monkeypatch.setenv(CACHE_DIR_ENV, "")
    monkeypatch.setattr(check, "_CACHE", {})
    monkeypatch.setattr(check, "_CHECKED_FOR", {})
    monkeypatch.setattr(check, "_IN_FLIGHT", {})
    monkeypatch.setattr(check, "codon_path_dir", lambda: Path("/codon"))
    monkeypatch.setattr(check, "_module_file", lambda name: Path("/run/scenario.py"))
    monkeypatch.setattr(check, "_toolchain_digest", lambda tc: "tc")
    monkeypatch.setattr(check, "_framework_digest", lambda codon_path: "fw")
    text = "x = 1\n"
    module = _Module(Path("/run/scenario.py"), False, text, ast.parse(text))
    monkeypatch.setattr(
        check, "_collect", lambda roots, tc, codon_path: _Sources({"scenario": module})
    )

    def no_mypy(paths: Any, timeout: float) -> Any:
        raise NoneCheckSkipped("mypy is not installed")

    monkeypatch.setattr(check, "run_none_check", no_mypy)
    compiled: list[str] = []

    def compile_(name: str, sources: Any, driver: Any, *args: Any) -> TypeCheckResult:
        compiled.append(driver.source)
        bad = "'fast'" in driver.source or "'slow'" in driver.source
        return TypeCheckResult(
            name,
            ok=not bad,
            diagnostics=[
                Diagnostic(
                    "'str' does not match expected type 'float'",
                    "scenario.timeout_seconds",
                )
            ]
            if bad
            else [],
        )

    monkeypatch.setattr(check, "_compile", compile_)
    return compiled


def _check(**values: Any) -> TypeCheckResult:
    return typecheck_scenario(
        _Scenario,
        _Config,
        values,
        toolchain=SimpleNamespace(),  # type: ignore[arg-type]
    )


def test_a_sweep_compiles_once_per_shape(
    compiles: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=check.__name__)
    first = _check(timeout_seconds=5.0, label="a")
    # mypy is missing, so the verdict carries a warning and is never written
    # to disk; the process still keeps it for the sweep's other cases.
    assert first.ok and first.warnings
    for timeout, label in ((7.5, "b"), (10.0, "a"), (12.5, "a")):
        assert _check(timeout_seconds=timeout, label=label) is first
    assert len(compiles) == 1
    assert "config.timeout_seconds = 5.0" in compiles[0]
    reused = [r.getMessage() for r in caplog.records if "reusing" in r.getMessage()]
    assert len(reused) == 3
    # Only what differs from the case it was compiled for.
    assert "made for timeout_seconds=5.0, label='a' for " in reused[0]
    assert "for timeout_seconds=7.5, label='b':" in reused[0]
    assert "made for timeout_seconds=5.0 for timeout_seconds=10.0:" in reused[1]
    # Another shape is another program.
    _check(timeout_seconds=5, label="a", enabled=False)
    assert len(compiles) == 2


def test_a_failed_check_refuses_every_case_of_its_shape(
    compiles: list[str],
) -> None:
    from autoware_carla_scenario.typecheck.mode import _enforce

    results = [_check(timeout_seconds=v) for v in ("fast", "slow")]
    assert len(compiles) == 1
    for result in results:
        assert not result.ok
        assert result.diagnostics[0].path == "scenario.timeout_seconds"
        with pytest.raises(check.ScenarioTypeError):
            _enforce(result, "auto")


def test_a_check_of_a_shape_being_compiled_waits_for_it(
    compiles: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """scenario-check runs configs in threads: the second of a shape waits for
    the first one's compile instead of starting its own."""
    compile_ = check._compile
    waiting = threading.Event()

    class Watched(Future[TypeCheckResult]):
        def result(self, timeout: Optional[float] = None) -> TypeCheckResult:
            waiting.set()
            return super().result(timeout)

    def blocked(*args: Any) -> Optional[TypeCheckResult]:
        assert waiting.wait(30), "the second check never waited"
        return compile_(*args)

    monkeypatch.setattr(check, "Future", Watched)
    monkeypatch.setattr(check, "_compile", blocked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_check, timeout_seconds=5.0)
        second = pool.submit(_check, timeout_seconds=7.5)
        assert first.result(60) is second.result(60)
    assert len(compiles) == 1
    assert check._IN_FLIGHT == {}


def test_a_failed_compile_fails_the_checks_waiting_for_it(
    compiles: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    owned: Future[Optional[TypeCheckResult]] = Future()
    owned.set_exception(RuntimeError("codon died"))

    def spy(key: str, sources: Any) -> Optional[TypeCheckResult]:
        check._IN_FLIGHT[key] = owned  # another thread is compiling it
        return None

    monkeypatch.setattr(check, "_cached", spy)
    with pytest.raises(RuntimeError, match="codon died"):
        _check(timeout_seconds=5.0)
    assert compiles == []


def test_a_check_waiting_for_a_compile_that_timed_out_compiles_itself(
    compiles: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out compile is no verdict: the check that waited for it gets
    its own, under its own timeout."""
    keys: list[str] = []

    class TimedOut(Future[Optional[TypeCheckResult]]):
        """Another thread's compile, which timed out and is then gone."""

        def result(self, timeout: Optional[float] = None) -> None:
            del check._IN_FLIGHT[keys[0]]
            return None

    def spy(key: str, sources: Any) -> Optional[TypeCheckResult]:
        if not keys:
            keys.append(key)
            check._IN_FLIGHT[key] = TimedOut()
        return None

    monkeypatch.setattr(check, "_cached", spy)
    result = _check(timeout_seconds=5.0)
    assert result.ok and len(compiles) == 1
    assert check._IN_FLIGHT == {}


def test_a_timed_out_check_is_not_reused(
    compiles: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def timed_out(*args: Any) -> None:
        calls.append(1)

    monkeypatch.setattr(check, "_compile", timed_out)
    assert _check(timeout_seconds=1.0).crashed
    assert _check(timeout_seconds=2.0).crashed
    assert len(calls) == 2


def test_the_disk_cache_is_keyed_by_the_shape(
    compiles: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(check, "run_none_check", lambda paths, timeout: [])
    assert _check(timeout_seconds=5.0).ok
    assert len(list(tmp_path.glob("*.json"))) == 1
    check._CACHE.clear()  # as a fresh process has it
    assert _check(timeout_seconds=6.0).ok
    assert len(compiles) == 1
