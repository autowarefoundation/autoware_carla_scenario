"""``detect-no-3d-model`` resolves map files as the runner does (issue #992).

The tool used to read only ``xodr_path`` and ``lanelet2_path``, without ``~``,
and to require both -- so ``town10hd_opt.yaml``, which names its map with
``map.source`` alone, could not be run through the tool its own comment points
at.  It now goes through ``resolve_map_paths``, captures the OpenDRIVE from
the loaded world when the map carries none, and reports a bad config as one
line rather than a traceback.  None of this needs CARLA.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.maps import MapSource, ResolvedMap
from autoware_carla_scenario.maps.cache import CACHE_ROOT_ENV
from autoware_carla_scenario.maps.opendrive import map_asset_env_var
from autoware_carla_scenario.tools import detect_no_3d_model_lanelets as tool
from autoware_carla_scenario.tools.detect_no_3d_model_lanelets import (
    MapConfigError,
    _load_world,
    _opendrive_for,
    _resolve_map_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA = _REPO_ROOT / "data"
_OSM = _DATA / "nishishinjuku.osm"
_XODR = _DATA / "nishishinjuku_carla.xodr"
_TOWN10_YAML = (
    _REPO_ROOT
    / "autoware_carla_scenario/src/autoware_carla_scenario/examples/conf/map"
    / "town10hd_opt.yaml"
)
_OPENDRIVE_TEXT = "<OpenDRIVE><header/></OpenDRIVE>\n"


class _FakeMap:
    def to_opendrive(self) -> str:
        return _OPENDRIVE_TEXT


class _FakeWorld:
    def get_map(self) -> _FakeMap:
        return _FakeMap()


class _FakeClient:
    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.world = _FakeWorld()

    def load_world(self, name: str) -> _FakeWorld:
        self.loaded.append(name)
        return self.world


@pytest.fixture(autouse=True)
def _own_map_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "map_cache"
    monkeypatch.setenv(CACHE_ROOT_ENV, str(cache))
    return cache


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_tilde_paths_are_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / "maps").mkdir(parents=True)
    shutil.copy(_OSM, home / "maps" / _OSM.name)
    shutil.copy(_XODR, home / "maps" / _XODR.name)
    monkeypatch.setenv("HOME", str(home))
    config = _write(
        tmp_path / "map.yaml",
        "map:\n"
        "  name: NishishinjukuMap\n"
        f"  xodr_path: ~/maps/{_XODR.name}\n"
        f"  lanelet2_path: ~/maps/{_OSM.name}\n"
        "  projector_type: MGRS\n"
        "  no_3d_model_lanelet_ids: []\n",
    )

    paths = _resolve_map_config(config)

    assert paths.name == "NishishinjukuMap"
    assert paths.lanelet2_path == home / "maps" / _OSM.name
    assert paths.install_xodr == home / "maps" / _XODR.name
    assert paths.projector_type == "MGRS"
    assert not paths.overwrite_xodr
    assert _opendrive_for(paths, _FakeWorld()) == home / "maps" / _XODR.name  # type: ignore[arg-type]


def test_map_directory_built_from_data(tmp_path: Path) -> None:
    directory = tmp_path / "NishishinjukuMap"
    directory.mkdir()
    shutil.copy(_OSM, directory / _OSM.name)
    shutil.copy(_XODR, directory / _XODR.name)
    config = _write(
        tmp_path / "map.yaml",
        f"map:\n  directory: {directory}\n  overwrite_xodr: true\n",
    )

    paths = _resolve_map_config(config)

    # The name falls back to the directory's, as for the runner.
    assert paths.name == "NishishinjukuMap"
    assert paths.lanelet2_path == directory / _OSM.name
    assert paths.install_xodr == directory / _XODR.name
    assert paths.overwrite_xodr


def test_map_directory_without_opendrive_captures_it(tmp_path: Path) -> None:
    directory = tmp_path / "NishishinjukuMap"
    directory.mkdir()
    shutil.copy(_OSM, directory / _OSM.name)
    config = _write(tmp_path / "map.yaml", f"map:\n  directory: {directory}\n")

    paths = _resolve_map_config(config)

    assert paths.install_xodr is None
    destination = paths.opendrive_path
    assert destination is not None and not destination.exists()
    client = _FakeClient()
    world = _load_world(client, paths)  # type: ignore[arg-type]
    assert client.loaded == ["NishishinjukuMap"]
    assert _opendrive_for(paths, world) == destination
    assert destination.read_text(encoding="utf-8") == _OPENDRIVE_TEXT


def test_source_only_config_resolves_through_resolve_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shipped ``town10hd_opt.yaml``: ``map.source`` and nothing else."""
    checkout = tmp_path / "checkout" / "Town10HD_Opt"
    checkout.mkdir(parents=True)
    shutil.copy(_OSM, checkout / "lanelet2_map.osm")
    derived = tmp_path / "derived"
    seen: list[str] = []

    def fake_resolve_map(source: str, *, refresh: bool = False) -> ResolvedMap:
        seen.append(source)
        return ResolvedMap(
            source=MapSource.parse(source),
            name="Town10HD_Opt",
            directory=checkout,
            lanelet2_path=checkout / "lanelet2_map.osm",
            xodr_path=None,
            projector_info_path=None,
            derived=derived,
        )

    monkeypatch.setattr(
        "autoware_carla_scenario.maps.config.resolve_map", fake_resolve_map
    )
    config = tmp_path / "town10hd_opt.yaml"
    shutil.copy(_TOWN10_YAML, config)

    paths = _resolve_map_config(config)

    assert len(seen) == 1 and seen[0].startswith("git+https://")
    assert paths.name == "Town10HD_Opt"
    assert paths.lanelet2_path == checkout / "lanelet2_map.osm"
    assert paths.install_xodr is None
    assert paths.opendrive_path == derived / "Town10HD_Opt.xodr"

    # Loaded by name, with CARLA's own road network, which is then captured.
    monkeypatch.setenv(map_asset_env_var("Town10HD_Opt"), str(tmp_path / "nope"))
    client = _FakeClient()
    world = _load_world(client, paths)  # type: ignore[arg-type]
    assert client.loaded == ["Town10HD_Opt"]
    assert not (tmp_path / "nope").exists()
    captured = _opendrive_for(paths, world)
    assert captured == derived / "Town10HD_Opt.xodr"
    assert captured.read_text(encoding="utf-8") == _OPENDRIVE_TEXT
    # A second run reads the cached file instead of capturing again.
    captured.write_text("cached", encoding="utf-8")
    assert _opendrive_for(paths, world).read_text(encoding="utf-8") == "cached"


def test_overwrite_xodr_installs_the_opendrive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write(
        tmp_path / "map.yaml",
        "map:\n"
        "  name: NishishinjukuMap\n"
        f"  xodr_path: {_XODR}\n"
        f"  lanelet2_path: {_OSM}\n"
        "  overwrite_xodr: true\n",
    )
    paths = _resolve_map_config(config)
    destination = tmp_path / "installed.xodr"
    monkeypatch.setenv(map_asset_env_var("NishishinjukuMap"), str(destination))

    client = _FakeClient()
    _load_world(client, paths)  # type: ignore[arg-type]

    assert client.loaded == ["NishishinjukuMap"]
    assert destination.read_bytes() == _XODR.read_bytes()


def test_overwrite_xodr_without_destination_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write(
        tmp_path / "map.yaml",
        "map:\n"
        "  name: NishishinjukuMap\n"
        f"  xodr_path: {_XODR}\n"
        f"  lanelet2_path: {_OSM}\n"
        "  overwrite_xodr: true\n",
    )
    paths = _resolve_map_config(config)
    monkeypatch.delenv(map_asset_env_var("NishishinjukuMap"), raising=False)
    client = _FakeClient()

    with pytest.raises(MapConfigError, match="NISHISHINJUKU_MAP_PATH"):
        _load_world(client, paths)  # type: ignore[arg-type]
    assert client.loaded == []


def test_without_overwrite_xodr_nothing_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write(
        tmp_path / "map.yaml",
        f"name: NishishinjukuMap\nxodr_path: {_XODR}\nlanelet2_path: {_OSM}\n",
    )
    paths = _resolve_map_config(config)
    destination = tmp_path / "installed.xodr"
    monkeypatch.setenv(map_asset_env_var("NishishinjukuMap"), str(destination))

    _load_world(_FakeClient(), paths)  # type: ignore[arg-type]

    assert not destination.exists()


# ---------------------------------------------------------------------------
# Config errors are one line and a non-zero exit
# ---------------------------------------------------------------------------


def _run_main(path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as excinfo:
        tool.main([str(path)])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    lines = err.strip().splitlines()
    assert len(lines) == 1, err
    return lines[0]


def test_malformed_yaml_is_a_one_line_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write(tmp_path / "map.yaml", "map:\n  name: [unterminated\n")
    line = _run_main(config, capsys)
    assert line.startswith("detect-no-3d-model: error:")
    assert "not valid YAML" in line


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("- a\n- b\n", "expected a YAML mapping"),
        ("map: 3\n", "expected 'map' to be a mapping"),
        ("map:\n  xodr_path: a.xodr\n  lanelet2_path: a.osm\n", "map.name"),
        ("map:\n  name: M\n", "names no Lanelet2 map"),
        ("map:\n  name: M\n  lanelet2_path: /nonexistent/a.osm\n", "not found"),
        (f"map:\n  name: M\n  lanelet2_path: {_OSM}\n", "names no OpenDRIVE"),
        (
            f"map:\n  name: M\n  lanelet2_path: {_OSM}\n  xodr_path: /nonexistent.xodr\n",
            "XODR file not found",
        ),
        (
            "map:\n  name: M\n"
            "  lanelet2_path: ${oc.env:DETECT_NO_3D_TEST_UNSET_VARIABLE}\n",
            "DETECT_NO_3D_TEST_UNSET_VARIABLE",
        ),
        ("map:\n  name: M\n  source: not-a-uri\n", "map.yaml"),
        ("map:\n  name: M\n  directory: /nonexistent/dir\n", "does not exist"),
    ],
)
def test_config_errors_are_one_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    match: str,
) -> None:
    monkeypatch.delenv("DETECT_NO_3D_TEST_UNSET_VARIABLE", raising=False)
    config = _write(tmp_path / "map.yaml", text)
    with pytest.raises(MapConfigError, match=match) as excinfo:
        _resolve_map_config(config)
    assert "\n" not in str(excinfo.value)
    assert match in _run_main(config, capsys)


def test_missing_config_file_is_a_one_line_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert "YAML config not found" in _run_main(tmp_path / "absent.yaml", capsys)


def test_errors_stop_before_connecting_to_carla(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_client(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("connected to CARLA")

    monkeypatch.setattr(tool, "carla", SimpleNamespace(Client=no_client))
    config = _write(tmp_path / "map.yaml", "map:\n  name: M\n")
    assert "names no Lanelet2 map" in _run_main(config, capsys)
