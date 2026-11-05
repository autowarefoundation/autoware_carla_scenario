"""The web viewer's selection and the path rules of ``ui.web_viewer_paths``,
and that ``ui.recordings`` still offers (and routes through) every name moved
there (#1672)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from autoware_carla_scenario.ui import recordings, web_viewer_paths
from autoware_carla_scenario.ui.web_viewer_paths import (
    WebViewer,
    cdn_url,
    configured_viewer,
    local_package,
    recording_path,
    web_viewer_cache_dir,
)

_MOVED = (
    "DEFAULT_WEB_VIEWER_VERSION",
    "LOCAL_ROUTE",
    "_WEB_VIEWER_FILES",
    "_ENV_WEB_VIEWER",
    "_ENV_CACHE",
    "_TRAJECTORY",
    "WebViewer",
    "_cache_root",
    "_complete",
    "recording_path",
)


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("VIEWER_CACHE_DIR", "VIEWER_RERUN_WEB_VIEWER", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    # No converter: the version is DEFAULT_WEB_VIEWER_VERSION.
    monkeypatch.setitem(sys.modules, "autoware_carla_scenario_rerun", None)


def _package(directory: Path, version: str = "0.38.1", omit: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("index.js", "re_viewer.js", "re_viewer_bg.wasm"):
        if name != omit:
            (directory / name).write_text("x", encoding="utf-8")
    if omit != "package.json":
        (directory / "package.json").write_text(
            json.dumps({"version": version}), encoding="utf-8"
        )
    return directory


@pytest.mark.parametrize("name", _MOVED)
def test_recordings_still_offers_every_moved_name(name: str) -> None:
    assert getattr(recordings, name) is getattr(web_viewer_paths, name)


def test_the_cdn_url_is_the_one_cdn_url_formats() -> None:
    assert cdn_url("0.38.1") == recordings.CDN_URL.format(version="0.38.1")


def test_a_recording_is_named_after_its_log() -> None:
    assert recording_path(Path("/x/Case-3.trajectory.jsonl")) == Path("/x/Case-3.rrd")
    assert recording_path(Path("/x/odd.jsonl")) == Path("/x/odd.rrd")


# ── The cache directory ────────────────────────────────────────────────────


def test_the_cache_is_viewer_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert web_viewer_paths._cache_root() == tmp_path / "c"
    assert (
        web_viewer_cache_dir("0.38.1") == tmp_path / "c" / "rerun-web-viewer" / "0.38.1"
    )


@pytest.mark.parametrize("cache_setting", [None, ""])
def test_without_viewer_cache_dir_it_is_under_xdg_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_setting: str | None
) -> None:
    if cache_setting is not None:
        monkeypatch.setenv("VIEWER_CACHE_DIR", cache_setting)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert web_viewer_paths._cache_root() == (
        tmp_path / "xdg" / "autoware_carla_scenario" / "viewer"
    )


@pytest.mark.parametrize("xdg_setting", [None, ""])
def test_without_either_it_is_under_the_home_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xdg_setting: str | None
) -> None:
    if xdg_setting is not None:
        monkeypatch.setenv("XDG_CACHE_HOME", xdg_setting)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert web_viewer_paths._cache_root() == (
        tmp_path / ".cache" / "autoware_carla_scenario" / "viewer"
    )


def test_a_given_root_is_used_instead_of_the_cache(tmp_path: Path) -> None:
    assert (
        web_viewer_cache_dir("1.0.0", tmp_path)
        == tmp_path / "rerun-web-viewer" / "1.0.0"
    )


def test_a_patch_of_recordings_cache_root_reaches_the_web_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recordings, "_cache_root", lambda: tmp_path / "patched")
    version = recordings.DEFAULT_WEB_VIEWER_VERSION
    cached = tmp_path / "patched" / "rerun-web-viewer" / version
    assert recordings.web_viewer_cache_dir(version) == cached
    _package(cached, version)
    viewer = recordings.web_viewer()
    assert (viewer.source, viewer.directory) == ("local", cached)


# ── Which web viewer ───────────────────────────────────────────────────────


@pytest.mark.parametrize("setting", ["none", "off", "false", "0", "NONE", "Off"])
def test_a_setting_can_embed_no_viewer(setting: str) -> None:
    assert configured_viewer(setting, "0.38.1") == WebViewer("0.38.1", None, "none")


@pytest.mark.parametrize(
    ("setting", "base_url"),
    [
        ("https://mirror.example/wv/", "https://mirror.example/wv"),
        ("http://mirror.example/wv", "http://mirror.example/wv"),
        ("/no/such/directory/", "/no/such/directory"),
    ],
)
def test_a_url_setting_is_a_mirror(setting: str, base_url: str) -> None:
    assert configured_viewer(setting, "0.38.1") == WebViewer("0.38.1", base_url, "url")


def test_an_absolute_directory_is_no_url(tmp_path: Path) -> None:
    assert configured_viewer(str(tmp_path), "0.38.1") is None


@pytest.mark.parametrize("setting", ["", "relative/dir", "~/viewer"])
def test_other_settings_name_no_viewer_by_themselves(setting: str) -> None:
    assert configured_viewer(setting, "0.38.1") is None


def test_the_named_directory_comes_before_the_cache(tmp_path: Path) -> None:
    named = _package(tmp_path / "named")
    cache = _package(tmp_path / "cache")
    assert local_package(str(named), cache) == named


def test_a_package_directory_inside_is_looked_in(tmp_path: Path) -> None:
    inner = _package(tmp_path / "unpacked" / "package")
    assert local_package(str(tmp_path / "unpacked"), tmp_path / "cache") == inner


def test_an_incomplete_named_directory_falls_back_to_the_cache(tmp_path: Path) -> None:
    _package(tmp_path / "named", omit="re_viewer_bg.wasm")
    cache = _package(tmp_path / "cache" / "package")
    assert local_package(str(tmp_path / "named"), tmp_path / "cache") == cache


def test_without_a_setting_only_the_cache_is_looked_in(tmp_path: Path) -> None:
    cache = _package(tmp_path / "cache")
    assert local_package("", cache) == cache


@pytest.mark.parametrize(
    "omit", ["index.js", "re_viewer.js", "re_viewer_bg.wasm", "package.json"]
)
def test_without_every_file_there_is_no_local_package(
    tmp_path: Path, omit: str
) -> None:
    cache = _package(tmp_path / "cache", omit=omit)
    assert local_package("", cache) is None
    assert local_package("", tmp_path / "missing") is None
