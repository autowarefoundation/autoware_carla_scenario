"""Verdicts kept beyond the process (``$AUTOWARE_CARLA_SCENARIO_TYPECHECK_CACHE``).

No Codon needed: these store and load results directly.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from autoware_carla_scenario.typecheck import check
from autoware_carla_scenario.typecheck.check import (
    CACHE_DIR_ENV,
    Diagnostic,
    TypeCheckResult,
    _cache_key,
    _cached,
    _Module,
    _Sources,
    _store,
    _toolchain_digest,
)
from autoware_carla_scenario.typecheck.toolchain import Toolchain


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(check, "_CACHE", {})
    return tmp_path / "cache"


def _sources(directory: str, text: str = "x = 1\n") -> _Sources:
    path = Path(directory) / "scenario.py"
    return _Sources({"scenario": _Module(path, False, text, ast.parse(text))})


def _failed(directory: str = "/run1") -> TypeCheckResult:
    return TypeCheckResult(
        "pkg.MyScenario",
        ok=False,
        diagnostics=[
            Diagnostic(
                f"bad, see {directory}/scenario.py:3",
                f"{directory}/scenario.py",
                3,
                7,
                ("realizing f [x.py:1]",),
            )
        ],
        output="out",
        seconds=1.5,
        warnings=[Diagnostic("left out", severity="warning")],
    )


def test_a_stored_verdict_is_loaded_by_another_process(cache_dir: Path) -> None:
    result = _failed()
    _store("k", result, _sources("/run1"))
    check._CACHE.clear()  # as a fresh process has it
    assert _cached("k", _sources("/run1")) == result
    assert (cache_dir / "k.json").is_file()


def test_a_verdict_names_the_files_of_the_sources_it_is_loaded_for(
    cache_dir: Path,
) -> None:
    _store("k", _failed("/run1"), _sources("/run1"))
    check._CACHE.clear()
    assert _cached("k", _sources("/run2")) == _failed("/run2")


def test_the_key_of_a_program_does_not_depend_on_where_its_modules_are(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(check, "_toolchain_digest", lambda tc: "tc")
    monkeypatch.setattr(check, "_framework_digest", lambda codon_path: "fw")

    def key(directory: str, text: str = "x = 1\n") -> str:
        sources = _sources(directory, text)
        driver = f"__file__: str = '{directory}/scenario.py'\n"
        return _cache_key(None, Path("/codon"), sources, driver)  # type: ignore[arg-type]

    assert key("/run1") == key("/run2")
    assert key("/run1") != key("/run1", "x = 2\n")


def test_a_read_entry_has_its_times_touched(cache_dir: Path) -> None:
    _store("k", _failed(), _sources("/run1"))
    os.utime(cache_dir / "k.json", (0, 0))
    check._CACHE.clear()
    assert _cached("k", _sources("/run1")) is not None
    assert (cache_dir / "k.json").stat().st_mtime > 0


def test_a_crash_is_kept_for_the_process_only(cache_dir: Path) -> None:
    crashed = TypeCheckResult("pkg.MyScenario", ok=False, crashed=True)
    _store("k", crashed, _sources("/run1"))
    assert _cached("k", _sources("/run1")) is crashed
    assert not (cache_dir / "k.json").exists()


def test_an_unreadable_entry_is_a_miss(cache_dir: Path) -> None:
    cache_dir.mkdir()
    (cache_dir / "k.json").write_text("{not json", encoding="utf-8")
    assert _cached("k", _sources("/run1")) is None
    (cache_dir / "k.json").write_text('{"scenario": "x"}', encoding="utf-8")
    assert _cached("k", _sources("/run1")) is None


def test_without_a_directory_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(check, "_CACHE", {})
    monkeypatch.chdir(tmp_path)
    _store("k", _failed(), _sources("/run1"))
    assert _cached("k", _sources("/run1")) is not None
    assert not list(tmp_path.iterdir())


def test_another_compiler_at_the_same_path_is_another_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdlib = tmp_path / "lib" / "codon"
    stdlib.mkdir(parents=True)
    (stdlib / "internal.codon").write_text("a", encoding="utf-8")
    version = "0.19.3"
    monkeypatch.setattr(Toolchain, "version", lambda self: version)

    def digest() -> str:
        _toolchain_digest.cache_clear()
        return _toolchain_digest(Toolchain(tmp_path / "bin" / "codon", tmp_path, "x"))

    before = digest()
    version = "0.19.4"
    upgraded = digest()
    (stdlib / "internal.codon").write_text("b", encoding="utf-8")
    patched = digest()
    _toolchain_digest.cache_clear()
    assert len({before, upgraded, patched}) == 3
