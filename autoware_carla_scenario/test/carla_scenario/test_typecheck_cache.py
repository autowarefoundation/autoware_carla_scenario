"""Verdicts kept beyond the process (``$AUTOWARE_CARLA_SCENARIO_TYPECHECK_CACHE``).

No Codon needed: these store and load results directly.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoware_carla_scenario.typecheck import check, typecheck_library
from autoware_carla_scenario.typecheck.check import (
    CACHE_DIR_ENV,
    Diagnostic,
    TypeCheckResult,
    _cache_key,
    _cached,
    _distributions_digest,
    _editable_roots,
    _package_digest,
    _Module,
    _source_tree_digest,
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


def test_a_hit_in_the_same_process_names_the_files_of_its_sources(
    cache_dir: Path,
) -> None:
    """#2193: the verdict kept in the process is path-free, as the one on disk
    is, so a second check of the same program at another path does not name
    the first check's files (possibly in a deleted ``tmp_path``)."""
    _store("k", _failed("/run1"), _sources("/run1"))
    assert _cached("k", _sources("/run2")) == _failed("/run2")
    assert _cached("k", _sources("/run1")) == _failed("/run1")


def test_a_process_only_hit_names_the_files_of_its_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2193: without a cache directory, the process's copy is all there is."""
    monkeypatch.setenv(CACHE_DIR_ENV, "")
    monkeypatch.setattr(check, "_CACHE", {})
    _store("k", _failed("/run1"), _sources("/run1"))
    assert _cached("k", _sources("/run2")) == _failed("/run2")
    crashed = dataclasses.replace(_failed("/run1"), crashed=True)
    _store("c", crashed, _sources("/run1"))
    assert _cached("c", _sources("/run2")) == dataclasses.replace(
        _failed("/run2"), crashed=True
    )


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


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    """The key of one program, with a fixed compiler and model (#268)."""
    monkeypatch.setattr(check, "_toolchain_digest", lambda tc: "tc")
    monkeypatch.setattr(check, "_framework_digest", lambda codon_path: "fw")

    def key(directory: str = "/run1", problem: str | None = None) -> str:
        sources = _sources(directory)
        if problem is not None:
            sources.problems.append(Diagnostic(problem, f"{directory}/scenario.py", 2))
        driver = f"__file__: str = '{directory}/scenario.py'\n"
        return _cache_key(None, Path("/codon"), sources, driver)  # type: ignore[arg-type]

    return key


def test_a_change_to_the_frameworks_python_alone_is_another_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: Callable[..., str]
) -> None:
    """#268: mypy reads the framework's Python, so a verdict on a scenario
    calling ``lanelet_length`` is stale once it may return None, though
    the model is unchanged."""
    package = Path(check.__file__).resolve().parent.parent
    copy = tmp_path / package.name
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(check, "__file__", str(copy / "typecheck" / "check.py"))

    def digest() -> str:
        _package_digest.cache_clear()
        return key()

    try:
        before = digest()
        assert digest() == before
        transform = copy / "coordinate" / "transform.py"
        text = transform.read_text(encoding="utf-8")
        old = "def lanelet_length(lanelet_id: int) -> float:"
        assert old in text
        transform.write_text(
            text.replace(
                old, 'def lanelet_length(lanelet_id: int) -> "Optional[float]":'
            ),
            encoding="utf-8",
        )
        after = digest()
        assert after != before
        # A stub, which mypy reads in place of its module, counts too.
        (stub,) = copy.rglob("autoware_bridge_pb2.pyi")
        stub.write_text(stub.read_text(encoding="utf-8") + "\nx: int\n")
        assert digest() != after
    finally:
        _package_digest.cache_clear()


def test_another_mypy_is_another_key(
    monkeypatch: pytest.MonkeyPatch, key: Callable[..., str]
) -> None:
    monkeypatch.setattr(check, "_mypy_version", lambda: "1.19.0")
    before = key()
    monkeypatch.setattr(check, "_mypy_version", lambda: "1.20.0")
    assert key() != before
    monkeypatch.setattr(check, "_mypy_version", lambda: "")  # not installed
    assert len({before, key()}) == 2


def test_another_installed_distribution_is_another_key(
    monkeypatch: pytest.MonkeyPatch, key: Callable[..., str]
) -> None:
    """mypy follows a scenario's imports into the installed distributions
    (numpy, lanelet2, typesafe_carla), whose types decide what may be None;
    the framework's own version is not one of them (a release alone keeps
    the verdicts)."""

    def installed(**versions: str) -> str:
        dists = [
            SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in versions.items()
        ]
        monkeypatch.setattr(check.importlib.metadata, "distributions", lambda: dists)
        _distributions_digest.cache_clear()
        return key()

    try:
        before = installed(numpy="2.2.0", autoware_carla_scenario="14.2.1")
        assert installed(numpy="2.2.0", autoware_carla_scenario="14.2.2") == before
        assert installed(numpy="2.3.0", autoware_carla_scenario="14.2.1") != before
        assert (
            installed(
                numpy="2.2.0", lanelet2="1.0", **{"autoware-carla-scenario": "14.2.1"}
            )
            != before
        )
    finally:
        _distributions_digest.cache_clear()


def _editable(monkeypatch: pytest.MonkeyPatch, name: str, checkout: Path) -> None:
    """*checkout* as the one distribution installed, editable."""
    direct_url = json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}})
    dist = SimpleNamespace(
        metadata={"Name": name},
        version="0.1.0",
        read_text=lambda file: direct_url if file == "direct_url.json" else None,
    )
    monkeypatch.setattr(check.importlib.metadata, "distributions", lambda: [dist])


@pytest.fixture
def fresh_digests() -> Iterator[None]:
    """Digests taken once a process, taken again for the test."""
    caches = (_distributions_digest, _editable_roots, _source_tree_digest)
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


def _write(path: Path, text: str) -> None:
    """Write *text*, with a modification time no earlier write has had."""
    path.parent.mkdir(parents=True, exist_ok=True)
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(before + 10**9, before + 10**9))


def test_an_edit_to_an_editable_distribution_is_another_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: Callable[..., str],
    fresh_digests: None,
) -> None:
    """Its version stays the same however its source changes; what is not its
    source (a virtualenv inside the checkout, bytecode) does not count."""
    checkout = tmp_path / "helpers"
    module = checkout / "helpers" / "geometry.py"
    _write(module, "def length() -> float: ...\n")
    _editable(monkeypatch, "helpers", checkout)

    def digest() -> str:
        _source_tree_digest.cache_clear()
        return key()

    before = digest()
    _write(checkout / ".venv" / "site.py", "x = 1\n")
    _write(checkout / "venv" / "site.py", "x = 1\n")
    _write(checkout / "helpers" / "__pycache__" / "x.py", "")
    assert digest() == before
    _write(module, "def length() -> float | None: ...\n")
    edited = digest()
    assert edited != before
    _write(checkout / "helpers" / "py.typed", "")
    assert digest() != edited


def test_the_package_being_checked_counts_by_its_stubs_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_digests: None
) -> None:
    """A scenario package installed editable: its modules are in the key by
    their text, so an edit to another scenario of it is not -- but a stub mypy
    prefers to a module is, and so is another package of the checkout."""
    monkeypatch.setattr(check, "_toolchain_digest", lambda tc: "tc")
    monkeypatch.setattr(check, "_framework_digest", lambda codon_path: "fw")
    checkout = tmp_path / "checkout"
    package = checkout / "my_scenarios"
    _write(package / "__init__.py", "")
    cut_in = package / "cut_in.py"
    _write(cut_in, "x = 1\n")
    _write(package / "merge.py", "y = 1\n")
    _write(checkout / "helpers" / "__init__.py", "")
    _editable(monkeypatch, "my-scenarios", checkout)
    text = cut_in.read_text(encoding="utf-8")
    sources = _Sources(
        {"my_scenarios.cut_in": _Module(cut_in, False, text, ast.parse(text))}
    )

    def key() -> str:
        _source_tree_digest.cache_clear()
        return _cache_key(None, Path("/codon"), sources, "")  # type: ignore[arg-type]

    before = key()
    _write(package / "merge.py", "y = 2\n")
    assert key() == before
    _write(package / "merge.pyi", "y: int | None\n")
    with_stub = key()
    assert with_stub != before
    _write(checkout / "helpers" / "__init__.py", "z = 1\n")
    assert key() != with_stub


def test_only_the_top_of_a_tree_is_left_out_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_digests: None
) -> None:
    """``build/`` beside the package is a build's output; ``mypkg/log/`` is a
    subpackage. A virtualenv is left out wherever it is, by its pyvenv.cfg."""
    checkout = tmp_path / "mypkg"
    _write(checkout / "build" / "lib" / "x.py", "")
    _write(checkout / "mypkg" / "log" / "__init__.py", "")
    _write(checkout / "mypkg" / "tools" / "env" / "pyvenv.cfg", "")

    def digest() -> str:
        _source_tree_digest.cache_clear()
        return _source_tree_digest(str(checkout))

    before = digest()
    _write(checkout / "build" / "lib" / "x.py", "x = 1\n")
    _write(checkout / "mypkg" / "tools" / "env" / "site.py", "x = 1\n")
    assert digest() == before
    _write(checkout / "mypkg" / "log" / "__init__.py", "x = 1\n")
    assert digest() != before


def test_a_symlinked_directory_is_followed(tmp_path: Path, fresh_digests: None) -> None:
    """``colcon build --symlink-install``: the source is behind a link."""
    source = tmp_path / "src" / "pkg"
    _write(source / "__init__.py", "")
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / "pkg").symlink_to(source, target_is_directory=True)
    (site / "loop").symlink_to(site, target_is_directory=True)  # and no loop

    def digest() -> str:
        _source_tree_digest.cache_clear()
        return _source_tree_digest(str(site))

    before = digest()
    _write(source / "__init__.py", "x = 1\n")
    assert digest() != before


def test_a_relative_search_path_entry_is_the_directory_the_check_runs_from(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: Callable[..., str],
    fresh_digests: None,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MYPYPATH", "stubs")
    stub = tmp_path / "stubs" / "helperlib.pyi"
    _write(stub, "def get() -> int: ...\n")
    before = key()
    _source_tree_digest.cache_clear()
    _write(stub, "def get() -> int | None: ...\n")
    assert key() != before


@pytest.mark.parametrize("variable", ["MYPYPATH", "PYTHONPATH"])
def test_a_directory_mypy_searches_is_in_the_key(
    variable: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: Callable[..., str],
    fresh_digests: None,
) -> None:
    """Which directories, and what is in them: a stub edited in place counts."""
    monkeypatch.delenv("MYPYPATH", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    without = key()
    stubs = tmp_path / "stubs"
    stub = stubs / "openodd" / "__init__.pyi"
    _write(stub, "def f() -> int: ...\n")
    monkeypatch.setenv(variable, str(stubs))
    with_stubs = key()
    assert with_stubs != without
    _source_tree_digest.cache_clear()
    _write(stub, "def f() -> int | None: ...\n")
    assert key() != with_stubs


def test_another_python_is_another_key(
    monkeypatch: pytest.MonkeyPatch, key: Callable[..., str]
) -> None:
    before = key()
    with monkeypatch.context() as patch:
        patch.setattr(check, "sys", SimpleNamespace(version_info=(3, 99, 0)))
        other = key()
    assert other != before
    assert key() == before


def test_what_python_found_wrong_is_in_the_key(key: Callable[..., str]) -> None:
    """#268: a hit returns before the compile reads ``sources.problems``."""
    assert key() != key(problem="foo is missing")
    assert key(problem="foo is missing") != key(problem="bar is missing")
    # Where the modules are still does not matter.
    assert key("/run1", "foo is missing") == key("/run2", "foo is missing")


def test_a_stale_verdict_is_not_returned_once_python_finds_a_problem(
    cache_dir: Path, key: Callable[..., str]
) -> None:
    ok = TypeCheckResult("pkg.MyScenario", ok=True)
    _store(key(), ok, _sources("/run1"))
    check._CACHE.clear()
    assert _cached(key(), _sources("/run1")) == ok
    assert _cached(key(problem="foo is missing"), _sources("/run1")) is None


def test_a_read_entry_has_its_times_touched(cache_dir: Path) -> None:
    _store("k", _failed(), _sources("/run1"))
    os.utime(cache_dir / "k.json", (0, 0))
    check._CACHE.clear()
    assert _cached("k", _sources("/run1")) is not None
    assert (cache_dir / "k.json").stat().st_mtime > 0


def test_an_entry_whose_times_cannot_be_touched_is_still_a_hit(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A read-only cache (restored by CI) serves its verdicts (#2249).
    _store("k", _failed(), _sources("/run1"))
    check._CACHE.clear()

    def read_only(*args: object, **kwargs: object) -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(check.os, "utime", read_only)
    assert _cached("k", _sources("/run1")) == _failed()


def test_a_crash_is_kept_for_the_process_only(cache_dir: Path) -> None:
    crashed = TypeCheckResult("pkg.MyScenario", ok=False, crashed=True)
    _store("k", crashed, _sources("/run1"))
    assert _cached("k", _sources("/run1")) == crashed
    assert not (cache_dir / "k.json").exists()


def test_a_killed_build_is_kept_nowhere(cache_dir: Path) -> None:
    # SIGKILL (the out-of-memory killer) is no verdict, like a timeout (#694).
    killed = TypeCheckResult("pkg.MyScenario", ok=False, crashed=True, killed=True)
    assert _store("k", killed, _sources("/run1")) is killed
    assert _cached("k", _sources("/run1")) is None
    assert not (cache_dir / "k.json").exists()


def test_an_unreadable_entry_is_a_miss(cache_dir: Path) -> None:
    cache_dir.mkdir()
    (cache_dir / "k.json").write_text("{not json", encoding="utf-8")
    assert _cached("k", _sources("/run1")) is None
    (cache_dir / "k.json").write_text('{"scenario": "x"}', encoding="utf-8")
    assert _cached("k", _sources("/run1")) is None


def test_an_empty_directory_keeps_verdicts_for_the_process_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV, "")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(check, "_CACHE", {})
    monkeypatch.chdir(tmp_path)
    _store("k", _failed(), _sources("/run1"))
    assert _cached("k", _sources("/run1")) is not None
    assert not list(tmp_path.iterdir())


def test_unset_the_verdicts_are_kept_in_the_users_cache_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(check, "_CACHE", {})
    _store("k", _failed(), _sources("/run1"))
    assert (tmp_path / "autoware_carla_scenario" / "typecheck" / "k.json").is_file()
    monkeypatch.setattr(check, "_CACHE", {})
    assert _cached("k", _sources("/run2")) is not None


@pytest.mark.parametrize(
    ("xdg", "home", "expected"),
    [
        ("/xdg", "/home/u", "/xdg/autoware_carla_scenario/typecheck"),
        ("", "/home/u", "/home/u/.cache/autoware_carla_scenario/typecheck"),
        # The XDG spec: a relative XDG_CACHE_HOME is ignored.
        ("relative", "/home/u", "/home/u/.cache/autoware_carla_scenario/typecheck"),
        # HOME="": Path.home() is "." -- the directory the check runs from.
        ("", "", None),
    ],
)
def test_the_default_directory(
    xdg: str, home: str, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    monkeypatch.setattr(check.Path, "home", lambda: Path(home))
    found = check._cache_dir()
    assert (str(found) if found is not None else None) == expected


def test_a_tilde_in_xdg_cache_home_is_expanded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default is utils.cache_dir.user_cache_dir()'s (#2187, #2249).
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("HOME", "/home/u")
    monkeypatch.setenv("XDG_CACHE_HOME", "~/xdg")
    assert check._cache_dir() == Path("/home/u/xdg/autoware_carla_scenario/typecheck")


def test_a_tilde_in_the_configured_directory_is_expanded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/home/u")
    monkeypatch.setenv(CACHE_DIR_ENV, "~/verdicts")
    assert check._cache_dir() == Path("/home/u/verdicts")


def test_a_relative_configured_directory_is_ignored_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # It would put the cache wherever the check runs from (#2249).
    monkeypatch.setenv(CACHE_DIR_ENV, "relative/cache")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    check._warn_relative_cache_dir.cache_clear()
    with caplog.at_level("WARNING", logger=check.logger.name):
        found = check._cache_dir()
    assert found == tmp_path / "autoware_carla_scenario" / "typecheck"
    assert "relative/cache" in caplog.text
    assert "not an absolute path" in caplog.text


def test_without_a_home_directory_verdicts_stay_in_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(check.Path, "home", no_home)
    assert check._cache_dir() is None


def test_an_unwritable_directory_keeps_verdicts_in_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("", encoding="utf-8")  # a file where the directory goes
    monkeypatch.setenv(CACHE_DIR_ENV, str(blocked / "cache"))
    monkeypatch.setattr(check, "_CACHE", {})
    _store("k", _failed(), _sources("/run1"))
    assert _cached("k", _sources("/run1")) is not None


def _fail_replace(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> list[Path]:
    """Make the rename of the scratch file over the entry raise *error*;
    the scratch files it was asked to rename are listed in the result."""
    renamed: list[Path] = []

    def replace(src: str, dst: str) -> None:
        renamed.append(Path(src))
        raise error

    monkeypatch.setattr(os, "replace", replace)
    return renamed


def test_a_failed_write_leaves_no_scratch_file(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A full disk, say: the verdict stays in the process, and the shared
    # directory gains no scratch file (#1859).
    renamed = _fail_replace(monkeypatch, OSError(28, "No space left on device"))
    result = _failed()
    assert _store("k", result, _sources("/run1")) is result
    assert renamed and renamed[0].parent == cache_dir
    assert list(cache_dir.iterdir()) == []
    assert _cached("k", _sources("/run1")) == result


def test_an_interrupted_write_leaves_no_scratch_file(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renamed = _fail_replace(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _store("k", _failed(), _sources("/run1"))
    assert renamed and renamed[0].parent == cache_dir
    assert list(cache_dir.iterdir()) == []


@pytest.mark.usefixtures("umask_022")
def test_a_stored_entry_is_the_json_of_the_verdict(cache_dir: Path) -> None:
    _store("k", _failed(), _sources("/run1"))
    entry = cache_dir / "k.json"
    assert [p.name for p in cache_dir.iterdir()] == ["k.json"]
    assert entry.stat().st_mode & 0o777 == 0o644
    stored = json.loads(entry.read_text(encoding="utf-8"))
    assert stored["scenario"] == "pkg.MyScenario"
    check._CACHE.clear()
    assert _cached("k", _sources("/run1")) == _failed()


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


def test_a_library_cache_hit_reports_this_runs_modules_and_time(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#652: modules named differently that share a closure are one program,
    one key; a hit says what this run checked and how long it took, not what
    the run that compiled it did."""
    monkeypatch.setattr(check, "codon_path_dir", lambda: Path("/codon"))
    monkeypatch.setattr(check, "_toolchain_digest", lambda tc: "tc")
    monkeypatch.setattr(check, "_framework_digest", lambda codon_path: "fw")
    monkeypatch.setattr(check, "_library_imports", lambda *args: None)
    monkeypatch.setattr(
        check, "run_none_check", lambda paths, timeout, started=None: []
    )
    compiled: list[str] = []

    def compile_(name: str, *args: object, **kwargs: object) -> TypeCheckResult:
        compiled.append(name)
        return TypeCheckResult(name, ok=True, seconds=13.2)

    monkeypatch.setattr(check, "_compile", compile_)
    package = "autoware_carla_scenario.coordinate"

    def run(*modules: str) -> TypeCheckResult:
        return typecheck_library(
            [f"{package}.{m}" for m in modules],
            toolchain=SimpleNamespace(),  # type: ignore[arg-type]
        )

    first = run("frames", "poses")
    assert first.ok and not first.cached
    assert "(2 checked modules" in first.scenario
    assert first.format().endswith("static check passed (13.2s)")
    check._CACHE.clear()  # as a fresh process has it
    second = run("frames")
    assert len(compiled) == 1, "the same program is compiled once"
    assert len(list(cache_dir.glob("*.json"))) == 1
    assert second.ok and second.cached
    assert "(1 checked modules" in second.scenario
    assert second.seconds < 13.2
    assert second.format().endswith(f"passed ({second.seconds:.1f}s, cached)")
    # The kept verdict is not changed by the hit.
    assert check._CACHE[next(iter(check._CACHE))].scenario == first.scenario
    third = run("frames", "poses")  # a hit in the same process
    assert third.cached and third.scenario == first.scenario


def test_a_reused_scenario_verdict_is_named_for_the_check_it_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#652: a verdict kept under a key another check shares carries that
    check's name; one for the same name is handed back as it is."""
    monkeypatch.setattr(check, "_CHECKED_FOR", {})
    kept = TypeCheckResult("pkg.First", ok=True, seconds=4.0)
    assert check._reused("pkg.First", "k", {}, kept) is kept
    other = check._reused("pkg.Second", "k", {}, kept)
    assert other.scenario == "pkg.Second" and other.seconds == 4.0
    assert kept.scenario == "pkg.First"


def _digest_of(root: Path) -> str:
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    check._tree_digest(digest, list(root.rglob("*.codon")), root)
    return digest.hexdigest()


def test_a_tree_digest_covers_paths_as_well_as_bytes(tmp_path: Path) -> None:
    # Three trees of the same bytes in the same order: three digests (#2249).
    a = tmp_path / "a"
    (a / "pkg").mkdir(parents=True)
    (a / "pkg" / "x.codon").write_text("x = 1\n", encoding="utf-8")
    moved = tmp_path / "moved"
    (moved / "other").mkdir(parents=True)
    (moved / "other" / "x.codon").write_text("x = 1\n", encoding="utf-8")
    renamed = tmp_path / "renamed"
    (renamed / "pkg").mkdir(parents=True)
    (renamed / "pkg" / "y.codon").write_text("x = 1\n", encoding="utf-8")
    added = tmp_path / "added"
    shutil.copytree(a, added)
    (added / "pkg" / "__init__.codon").write_text("", encoding="utf-8")
    digests = {_digest_of(t) for t in (a, moved, renamed, added)}
    assert len(digests) == 4


def test_a_tree_digest_does_not_depend_on_where_the_tree_is(tmp_path: Path) -> None:
    a = tmp_path / "a"
    (a / "pkg").mkdir(parents=True)
    (a / "pkg" / "x.codon").write_text("x = 1\n", encoding="utf-8")
    b = tmp_path / "elsewhere" / "b"
    shutil.copytree(a, b)
    assert _digest_of(a) == _digest_of(b)


def test_a_split_between_files_is_another_digest(tmp_path: Path) -> None:
    one = tmp_path / "one"
    one.mkdir()
    (one / "a.codon").write_text("ab", encoding="utf-8")
    (one / "b.codon").write_text("", encoding="utf-8")
    two = tmp_path / "two"
    two.mkdir()
    (two / "a.codon").write_text("a", encoding="utf-8")
    (two / "b.codon").write_text("b", encoding="utf-8")
    assert _digest_of(one) != _digest_of(two)
