"""Verdicts kept beyond the process (``$AUTOWARE_CARLA_SCENARIO_TYPECHECK_CACHE``).

No Codon needed: these store and load results directly.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoware_carla_scenario.typecheck import check
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


def test_a_namespace_packages_root_is_in_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_digests: None
) -> None:
    """mypy is given the root of a namespace scenario package to search: what
    else is there counts, and the package's other modules do not."""
    monkeypatch.setattr(check, "_toolchain_digest", lambda tc: "tc")
    monkeypatch.setattr(check, "_framework_digest", lambda codon_path: "fw")
    cut_in = tmp_path / "nsscen" / "cut_in.py"
    _write(cut_in, "x = 1\n")
    _write(tmp_path / "nsscen" / "merge.py", "y = 1\n")
    _write(tmp_path / "helpers.py", "z = 1\n")
    text = cut_in.read_text(encoding="utf-8")
    sources = _Sources({"nsscen.cut_in": _Module(cut_in, False, text, ast.parse(text))})
    assert check._namespace_roots(sources) == (str(tmp_path),)

    def key() -> str:
        _source_tree_digest.cache_clear()
        return _cache_key(None, Path("/codon"), sources, "")  # type: ignore[arg-type]

    before = key()
    _write(tmp_path / "nsscen" / "merge.py", "y = 2\n")
    assert key() == before
    _write(tmp_path / "helpers.py", "z = 2\n")
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
