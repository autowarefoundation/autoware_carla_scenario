"""The viewer's launcher and the ``scenario`` CLI resolve a pattern alike (#895).

Both go through :func:`autoware_carla_scenario.scenario_glob.resolve_scenario_glob`:
``*`` stays within one path component, the names are sorted as strings, and a
pattern the CLI rejects matches nothing in the viewer either.
"""

from __future__ import annotations

import importlib.metadata
import shutil
from pathlib import Path
from typing import Any

import pytest

from fastapi.testclient import TestClient

import autoware_carla_scenario
import autoware_carla_scenario.registry as registry
from autoware_carla_scenario import scenario_glob
from autoware_carla_scenario.examples import compose_errors, run
from autoware_carla_scenario.odd import cli as odd_cli
from autoware_carla_scenario.scenario_glob import (
    is_scenario_glob,
    resolve_scenario_glob,
    scenario_names,
)
from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import runner, scanner

_BUILTIN_LEFT = (
    Path(autoware_carla_scenario.__file__).resolve().parent
    / "examples"
    / "conf"
    / "scenario"
    / "lane_change"
    / "left.yaml"
)

# The package's scenarios, relative to its ``conf/scenario``: three levels
# deep, and names with ``-``, which sorts before ``/``.
_PACKAGE_SCENARIOS = (
    "glob_pkg/left.yaml",
    "glob_pkg/a-b.yaml",
    "glob_pkg/a/b.yaml",
    "glob_pkg/a/left-b.yaml",
    "glob_pkg-x/left.yaml",
    "cut_in_pkg/one.yaml",
)

_PATTERNS = (
    "*",
    "*left*",
    "cut_in*",
    "*/*",
    "*/left",
    "glob_pkg/*",
    "glob_pkg/**/*",
    "glob_pkg/**",
    "**/*",
    "**",
    "lane_change/left",
    "glob_pkg/a/b",
    "*/a-*",
    "no_such_*",
)


@pytest.fixture
def glob_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An installed scenario package whose ``conf/scenario`` holds
    :data:`_PACKAGE_SCENARIOS`; the registry's conf dirs and its loaded-once
    latch are restored after the test."""
    conf_dir = tmp_path / "glob_conf"
    for name in _PACKAGE_SCENARIOS:
        path = conf_dir / "scenario" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_BUILTIN_LEFT, path)

    def register() -> None:
        registry.register_conf_dir(conf_dir)

    entry_point = importlib.metadata.EntryPoint(
        name="glob_pkg",
        value="glob_pkg:register",
        group=registry.SCENARIO_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(importlib.metadata.EntryPoint, "load", lambda self: register)

    def fake_entry_points(**kwargs: Any) -> list[importlib.metadata.EntryPoint]:
        if kwargs.get("group") == registry.SCENARIO_ENTRY_POINT_GROUP:
            return [entry_point]
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    monkeypatch.setattr(registry, "_CONF_DIRS", list(registry.get_conf_dirs()))
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    # As the CLI's main() does before it resolves a pattern.
    registry.load_scenario_plugins()
    return conf_dir.resolve()


def _cli(pattern: str) -> list[str]:
    """The CLI's names for *pattern*, ``[]`` where it exits for no match."""
    try:
        return run._resolve_scenario_glob(pattern)
    except SystemExit as exc:
        assert exc.code == 1
        return []


@pytest.mark.parametrize("pattern", _PATTERNS)
def test_viewer_and_cli_resolve_alike(glob_package: Path, pattern: str) -> None:
    assert app_module._resolve_scenario_names(pattern) == _cli(pattern)


def test_star_stays_within_one_component(glob_package: Path) -> None:
    # No top-level config: '*' and '*left*' match nothing, as in the CLI.
    assert app_module._resolve_scenario_names("*") == []
    assert app_module._resolve_scenario_names("*left*") == []
    assert app_module._resolve_scenario_names("cut_in*") == []
    assert app_module._resolve_scenario_names("glob_pkg/*") == [
        "glob_pkg/a-b",
        "glob_pkg/left",
    ]
    assert "glob_pkg-x/left" in app_module._resolve_scenario_names("*/left")
    assert "lane_change/left" in app_module._resolve_scenario_names("*/left")


def test_names_sort_as_strings(glob_package: Path) -> None:
    names = app_module._resolve_scenario_names("glob_pkg/**/*")
    assert names == [
        "glob_pkg/a-b",
        "glob_pkg/a/b",
        "glob_pkg/a/left-b",
        "glob_pkg/left",
    ]


def test_a_name_without_glob_characters_is_kept(glob_package: Path) -> None:
    assert app_module._resolve_scenario_names("lane_change/left") == [
        "lane_change/left"
    ]
    assert _cli("lane_change/left") == ["lane_change/left"]


def test_cli_exits_on_no_match(
    glob_package: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run._resolve_scenario_glob("no_such_*")
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "no scenario configs match pattern 'no_such_*'" in out
    assert str(glob_package / "scenario") in out


def test_each_name_once_across_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for conf_dir in (first, second):
        (conf_dir / "scenario" / "s").mkdir(parents=True)
        (conf_dir / "scenario" / "s" / "x.yaml").write_text("{}\n")
    (tmp_path / "empty").mkdir()
    dirs = [first, tmp_path / "empty", second]
    assert resolve_scenario_glob("s/*", dirs) == ["s/x"]
    assert resolve_scenario_glob("t/*", dirs) == []


def test_scenario_names_lists_each_name_once_by_path_component(
    tmp_path: Path,
) -> None:
    """The viewer's scenario list (#898): every YAML file under ``scenario/``,
    each name once across the directories, ordered by path component (``a/z``
    before ``a-b/x``, which plain string order puts first)."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    for conf_dir in (first, second):
        (conf_dir / "scenario" / "a" / "deep").mkdir(parents=True)
        (conf_dir / "scenario" / "a" / "z.yaml").write_text("{}\n")
    (first / "scenario" / "a-b").mkdir()
    (first / "scenario" / "a-b" / "x.yaml").write_text("{}\n")
    (second / "scenario" / "a" / "deep" / "y.yaml").write_text("{}\n")
    (second / "scenario" / "a" / "note.txt").write_text("not a config\n")
    (tmp_path / "empty").mkdir()
    dirs = [first, tmp_path / "empty", second]
    assert scenario_names(dirs) == ["a/deep/y", "a/z", "a-b/x"]
    assert scenario_names([]) == []


def test_the_viewer_lists_scenario_names_of_its_config_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ui.scanner.list_scenario_configs`` is :func:`scenario_names` of the
    viewer's config directories (#898)."""
    (tmp_path / "scenario" / "s").mkdir(parents=True)
    (tmp_path / "scenario" / "s" / "x.yaml").write_text("{}\n")
    monkeypatch.setattr(scanner, "scenario_conf_dirs", lambda: [tmp_path])
    assert scanner.list_scenario_configs() == ["s/x"]
    assert scanner.scenario_names is scenario_names


@pytest.mark.parametrize(
    ("value", "is_glob"),
    [
        ("lane_change/left", False),
        ("lane_change/*", True),
        ("lane_change/lef?", True),
        ("lane_change/[lr]*", True),
        ("a-b", False),
    ],
)
def test_viewer_and_cli_tell_a_pattern_alike(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    is_glob: bool,
) -> None:
    """The viewer resolves *value* as a pattern exactly when the CLI does."""
    assert is_scenario_glob(value) is is_glob
    assert run._is_glob_pattern(value) is is_glob
    resolved: list[str] = []

    def spy(pattern: str, conf_dirs: list[Path]) -> list[str]:
        resolved.append(pattern)
        return []

    monkeypatch.setattr(app_module, "resolve_scenario_glob", spy)
    app_module._resolve_scenario_names(value)
    assert resolved == ([value] if is_glob else [])


# A tree under ``scenario/`` for the ``**`` tests (#1001): configs one, two
# and three levels deep, and one at the top.
_STAR_STAR_TREE = (
    "top.yaml",
    "x/a.yaml",
    "x/sub/b.yaml",
    "x/sub/deep/c.yaml",
    "y/d.yaml",
)
_EVERY_X = ["x/a", "x/sub/b", "x/sub/deep/c"]


@pytest.fixture
def star_star_conf(tmp_path: Path) -> Path:
    for name in _STAR_STAR_TREE:
        path = tmp_path / "scenario" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    (tmp_path / "scenario" / "x" / "note.txt").write_text("not a config\n")
    return tmp_path


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("x/**", _EVERY_X),
        ("**", ["top", *_EVERY_X, "y/d"]),
        ("x/sub/**", ["x/sub/b", "x/sub/deep/c"]),
        ("x/**/*", _EVERY_X),
        ("**/*", ["top", *_EVERY_X, "y/d"]),
        ("x/**/c", ["x/sub/deep/c"]),
        ("x/*", ["x/a"]),
    ],
)
def test_double_star_matches_everything_below(
    star_star_conf: Path, pattern: str, expected: list[str]
) -> None:
    """A trailing ``**`` is ``**/*`` on every Python (#1001): 3.10-3.12
    would raise and 3.13+ would not recurse."""
    assert resolve_scenario_glob(pattern, [star_star_conf]) == expected


@pytest.mark.parametrize(
    ("pattern", "glob"),
    [
        ("x/**", "x/**/*"),
        ("**", "**/*"),
        ("x/**/*", "x/**/*"),
        ("x/**/c", "x/**/c"),
        ("x/*", "x/*"),
        ("lane_change/left", "lane_change/left"),
    ],
)
def test_only_a_trailing_double_star_is_rewritten(pattern: str, glob: str) -> None:
    assert scenario_glob._glob_pattern(pattern) == glob


_MALFORMED = ("x/**a", "a**", "x/b**/c", "**x/*")


@pytest.mark.parametrize("pattern", _MALFORMED)
def test_a_double_star_inside_a_component_is_refused(
    star_star_conf: Path, pattern: str
) -> None:
    with pytest.raises(ValueError, match="invalid scenario pattern") as excinfo:
        resolve_scenario_glob(pattern, [star_star_conf])
    assert repr(pattern) in str(excinfo.value)


@pytest.mark.parametrize("pattern", _MALFORMED)
def test_the_cli_reports_a_malformed_pattern(
    pattern: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run._resolve_scenario_glob(pattern)
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert out.startswith("Error: invalid scenario pattern")
    assert repr(pattern) in out


def test_the_odd_cli_reports_a_malformed_pattern() -> None:
    with pytest.raises(ValueError, match="invalid scenario pattern 'x/\\*\\*a'"):
        odd_cli._scenario_configs(["scenario=x/**a"])


def test_the_odd_cli_reports_no_match() -> None:
    with pytest.raises(ValueError, match="no scenario configs match 'no_such_\\*'"):
        odd_cli._scenario_configs(["scenario=no_such_*"])


def test_compose_errors_reports_a_malformed_pattern(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert compose_errors.resolve_scenarios("x/**a") is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("Error: invalid scenario pattern 'x/**a'")


def test_the_viewer_resolver_raises_value_error(star_star_conf: Path) -> None:
    with pytest.raises(ValueError, match="invalid scenario pattern"):
        app_module._resolve_scenario_names("x/**a")


@pytest.mark.parametrize("sweeper", ["", "lanelet"])
def test_the_viewer_answers_a_malformed_pattern_with_an_error(
    star_star_conf: Path, monkeypatch: pytest.MonkeyPatch, sweeper: str
) -> None:
    """``/api/run`` answers an invalid pattern with an error, not a 500, and
    starts nothing."""

    def no_run(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("nothing should run")

    monkeypatch.setattr(runner, "start_run", no_run)
    monkeypatch.setattr(scanner, "scenario_conf_dirs", lambda: [star_star_conf])
    client = TestClient(app_module.app)
    response = client.post(
        "/api/run",
        json={"scenario": "x/**a", "extra_overrides": [], "sweeper": sweeper},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert "invalid scenario pattern 'x/**a'" in body["message"]


def test_the_viewer_runs_every_config_below_a_double_star(
    star_star_conf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[list[str]]] = []

    def fake_start_run(overrides_list: list[list[str]], **kwargs: Any) -> bool:
        seen.append(overrides_list)
        return True

    monkeypatch.setattr(runner, "start_run", fake_start_run)
    monkeypatch.setattr(scanner, "scenario_conf_dirs", lambda: [star_star_conf])
    client = TestClient(app_module.app)
    response = client.post("/api/run", json={"scenario": "x/**"})
    assert response.json() == {"status": "started"}
    assert seen == [[[f"scenario={name}"] for name in _EVERY_X]]


# Patterns that would reach outside ``scenario/`` (#1040): an absolute one made
# ``Path.glob`` raise NotImplementedError (a 500 in the viewer, a traceback in
# the CLIs), and a ``..`` component matched configs beside ``scenario/``.
_OUTSIDE = ("/x/*", "/*", "/**", "../*", "x/../*", "x/..", "../scenario/x/*")


@pytest.mark.parametrize("pattern", _OUTSIDE)
def test_a_pattern_outside_scenario_is_refused(
    star_star_conf: Path, pattern: str
) -> None:
    (star_star_conf / "secret.yaml").write_text("{}\n")
    with pytest.raises(ValueError, match="invalid scenario pattern") as excinfo:
        resolve_scenario_glob(pattern, [star_star_conf])
    assert repr(pattern) in str(excinfo.value)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("x/?", ["x/a"]),
        ("[xy]/*", ["x/a", "y/d"]),
        ("x/**/[bc]", ["x/sub/b", "x/sub/deep/c"]),
        ("x/..a", []),
        ("x/a..", []),
    ],
)
def test_legitimate_patterns_still_match(
    star_star_conf: Path, pattern: str, expected: list[str]
) -> None:
    assert resolve_scenario_glob(pattern, [star_star_conf]) == expected


@pytest.mark.parametrize("pattern", _OUTSIDE)
def test_the_cli_reports_a_pattern_outside_scenario(
    pattern: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run._resolve_scenario_glob(pattern)
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert out.startswith("Error: invalid scenario pattern")
    assert repr(pattern) in out


@pytest.mark.parametrize("pattern", _OUTSIDE)
def test_compose_errors_reports_a_pattern_outside_scenario(
    pattern: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """``scenario-check`` and ``scenario-expand`` exit 2 on ``None``."""
    assert compose_errors.resolve_scenarios(pattern) is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"Error: invalid scenario pattern {pattern!r}")


def test_the_odd_cli_reports_a_pattern_outside_scenario() -> None:
    with pytest.raises(ValueError, match="invalid scenario pattern '/x/\\*'"):
        odd_cli._scenario_configs(["scenario=/x/*"])


@pytest.mark.parametrize("pattern", ["/x/*", "../*"])
def test_the_viewer_answers_a_pattern_outside_scenario_with_an_error(
    star_star_conf: Path, monkeypatch: pytest.MonkeyPatch, pattern: str
) -> None:
    def no_run(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("nothing should run")

    monkeypatch.setattr(runner, "start_run", no_run)
    monkeypatch.setattr(scanner, "scenario_conf_dirs", lambda: [star_star_conf])
    client = TestClient(app_module.app)
    response = client.post("/api/run", json={"scenario": pattern})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert f"invalid scenario pattern {pattern!r}" in body["message"]
