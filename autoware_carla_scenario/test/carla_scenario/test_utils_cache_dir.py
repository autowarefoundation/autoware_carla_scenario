"""The user's cache directory, by the XDG rules, and the caches under it (#2187)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoware_carla_scenario.commonroad.map import default_cache_dir
from autoware_carla_scenario.ui import web_viewer_paths
from autoware_carla_scenario.utils.cache_dir import user_cache_dir


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home of its own, run from another directory, with no XDG_CACHE_HOME."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("VIEWER_CACHE_DIR", raising=False)
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    return home


def test_an_absolute_xdg_cache_home_is_the_cache(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert user_cache_dir() == tmp_path / "xdg"


def test_an_absolute_xdg_cache_home_needs_no_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    assert user_cache_dir() == tmp_path / "xdg"


@pytest.mark.parametrize(
    "xdg", [None, "", "relative", "./cache", "~no-such-user-2187/cache"]
)
def test_an_unset_empty_or_relative_xdg_cache_home_is_the_home_cache(
    home: Path, monkeypatch: pytest.MonkeyPatch, xdg: str | None
) -> None:
    # A relative one would put the cache under the current directory.
    if xdg is not None:
        monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    assert user_cache_dir() == home / ".cache"


@pytest.mark.parametrize("xdg", ["", "relative"])
def test_the_caches_ignore_an_empty_or_relative_xdg_cache_home(
    home: Path, monkeypatch: pytest.MonkeyPatch, xdg: str
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    cache = home / ".cache" / "autoware_carla_scenario"
    assert web_viewer_paths._cache_root() == cache / "viewer"
    assert default_cache_dir() == cache / "commonroad"


def test_a_leading_tilde_in_xdg_cache_home_is_the_home(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As the viewer's and the CommonRoad maps' caches always took it: an
    # existing cache under a quoted `~/x` stays where it is.
    monkeypatch.setenv("XDG_CACHE_HOME", "~/x")
    assert user_cache_dir() == home / "x"
    cache = home / "x" / "autoware_carla_scenario"
    assert web_viewer_paths._cache_root() == cache / "viewer"
    assert default_cache_dir() == cache / "commonroad"
