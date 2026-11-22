"""pyxodr's roads, remembered per OpenDRIVE content (maps.road_network_cache)."""

from __future__ import annotations

import contextlib
import os
import pickle
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyxodr.road_objects.network import RoadNetwork

from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.maps import road_network_cache
from autoware_carla_scenario.maps.road_network_cache import (
    load_roads,
    open_road_network,
)
from autoware_carla_scenario.utils.traffic_light import get_signal_ids_for_controller

_DATA = Path(__file__).resolve().parents[3] / "data"


def _road(road_id: str, x: float, link: str, lane_link: str) -> str:
    return f"""
  <road id="{road_id}" length="10" junction="-1">
    <link>{link}</link>
    <planView>
      <geometry s="0" x="{x}" y="0" hdg="0" length="10"><line/></geometry>
    </planView>
    <elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
    <lanes>
      <laneOffset s="0" a="0" b="0" c="0" d="0"/>
      <laneSection s="0">
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false">
            <link>{lane_link}</link>
            <width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          </lane>
          <lane id="-2" type="sidewalk" level="false">
            <width sOffset="0" a="2" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>"""


#: Two roads, one after the other: small enough to build in a moment.
_XODR = (
    '<?xml version="1.0"?>\n<OpenDRIVE>\n  <header revMajor="1" revMinor="4"/>'
    + _road(
        "1",
        0.0,
        '<successor elementType="road" elementId="2" contactPoint="start"/>',
        '<successor id="-1"/>',
    )
    + _road(
        "2",
        10.0,
        '<predecessor elementType="road" elementId="1" contactPoint="end"/>',
        '<predecessor id="-1"/>',
    )
    + "\n</OpenDRIVE>\n"
)


@pytest.fixture
def xodr(tmp_path: Path) -> Path:
    path = tmp_path / "map.xodr"
    path.write_text(_XODR, encoding="utf-8")
    return path


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[RoadNetwork]:
    """Every network pyxodr builds roads for, in order."""
    built: list[RoadNetwork] = []
    get_roads = RoadNetwork.get_roads

    def counting(self: RoadNetwork, *args: Any, **kwargs: Any) -> Any:
        built.append(self)
        return get_roads(self, *args, **kwargs)

    monkeypatch.setattr(RoadNetwork, "get_roads", counting)
    return built


def _load(xodr: Path, cache: Path) -> RoadNetwork:
    network = open_road_network(xodr)
    load_roads(network, root=cache)
    return network


def test_miss_builds_and_stores(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork]
) -> None:
    network = _load(xodr, tmp_path / "cache")
    assert builds == [network]
    assert sorted(network.road_ids_to_object) == ["1", "2"]
    assert len(list((tmp_path / "cache").glob("*.pickle"))) == 1
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_hit_loads_without_building(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork]
) -> None:
    _load(xodr, tmp_path / "cache")
    network = _load(xodr, tmp_path / "cache")
    assert len(builds) == 1
    assert sorted(network.road_ids_to_object) == ["1", "2"]
    first, second = (network.road_ids_to_object[i] for i in ("1", "2"))
    assert first.successor_data[0] is second
    assert second.predecessor_data[0] is first
    lane = first.lane_sections[0].get_lane_from_id(-1)
    assert lane.successor_data[0][0] is second.lane_sections[0].get_lane_from_id(-1)


def test_changed_content_is_a_miss(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork]
) -> None:
    _load(xodr, tmp_path / "cache")
    xodr.write_text(_XODR + "<!-- edited -->\n", encoding="utf-8")
    _load(xodr, tmp_path / "cache")
    assert len(builds) == 2
    assert len(list((tmp_path / "cache").glob("*.pickle"))) == 2


@pytest.mark.parametrize(
    "option", [("resolution", 0.5), ("ignored_lane_types", {"sidewalk"})]
)
def test_changed_options_are_a_miss(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork], option: tuple[str, Any]
) -> None:
    _load(xodr, tmp_path / "cache")
    network = open_road_network(xodr)
    setattr(network, *option)
    load_roads(network, root=tmp_path / "cache")
    assert len(builds) == 2


@pytest.mark.parametrize("damage", ["garbage", "truncated", "other document"])
def test_damaged_entry_is_rebuilt(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork], damage: str
) -> None:
    _load(xodr, tmp_path / "cache")
    (entry,) = (tmp_path / "cache").glob("*.pickle")
    if damage == "garbage":
        entry.write_bytes(b"not a pickle")
    elif damage == "truncated":
        entry.write_bytes(entry.read_bytes()[:-100])
    else:
        entry.write_bytes(pickle.dumps((1, [], [], {})))
    network = _load(xodr, tmp_path / "cache")
    assert len(builds) == 2
    assert sorted(network.road_ids_to_object) == ["1", "2"]
    # ... and the entry is repaired.
    _load(xodr, tmp_path / "cache")
    assert len(builds) == 2


# An entry another user may have written is not read (#2325).


def _entry(cache: Path) -> Path:
    (entry,) = cache.glob("*.pickle")
    return entry


def _foreign(monkeypatch: pytest.MonkeyPatch, *paths: Path) -> int:
    """Make *paths* read as another user's, to stat() and fstat(); that uid."""
    real_stat, real_fstat = os.stat, os.fstat
    theirs = os.getuid() + 1

    def owned(info: os.stat_result) -> os.stat_result:
        for path in paths:
            with contextlib.suppress(OSError):
                if os.path.samestat(info, real_stat(path)):
                    fields = list(info)
                    fields[stat.ST_UID] = theirs
                    return os.stat_result(fields)
        return info

    monkeypatch.setattr(
        os, "stat", lambda path, *a, **k: owned(real_stat(path, *a, **k))
    )
    monkeypatch.setattr(os, "fstat", lambda fd: owned(real_fstat(fd)))
    return theirs


def test_a_foreign_owned_entry_is_a_miss(
    xodr: Path,
    tmp_path: Path,
    builds: list[RoadNetwork],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = tmp_path / "cache"
    _load(xodr, cache)
    theirs = _foreign(monkeypatch, _entry(cache))
    with caplog.at_level("WARNING", logger=road_network_cache.__name__):
        network = _load(xodr, cache)
    assert len(builds) == 2
    assert sorted(network.road_ids_to_object) == ["1", "2"]
    assert f"is owned by uid {theirs}, not {os.getuid()}" in caplog.text
    assert "Not reading the cached roads" in caplog.text


def test_another_users_directory_is_passed_over_for_ones_own(
    xodr: Path,
    tmp_path: Path,
    builds: list[RoadNetwork],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A map root shared with a group: its road_network is another user's,
    # with their entry in it.
    cache = tmp_path / "cache"
    _load(xodr, cache)
    _foreign(monkeypatch, cache)
    with caplog.at_level("WARNING", logger=road_network_cache.__name__):
        _load(xodr, cache)
        assert len(builds) == 2
        own = tmp_path / f"cache-{os.getuid()}"
        assert stat.S_IMODE(_entry(own).stat().st_mode) == 0o644
        # ... and read from there on.
        _load(xodr, cache)
    assert len(builds) == 2
    assert caplog.text == ""


def test_a_foreign_directory_is_never_stored_into(
    xodr: Path,
    tmp_path: Path,
    builds: list[RoadNetwork],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Its own sibling squatted too: refused, and not written again.
    cache = tmp_path / "cache"
    own = tmp_path / f"cache-{os.getuid()}"
    _load(xodr, own)
    entry = _entry(own)
    before = entry.stat().st_mtime_ns
    cache.mkdir()
    theirs = _foreign(monkeypatch, cache, own)
    with caplog.at_level("WARNING", logger=road_network_cache.__name__):
        _load(xodr, cache)
        _load(xodr, cache)
    assert len(builds) == 3
    assert f"its directory is owned by uid {theirs}" in caplog.text
    assert entry.stat().st_mtime_ns == before


@pytest.mark.parametrize("where", ["entry", "directory"])
def test_a_group_writable_entry_is_a_miss(
    xodr: Path,
    tmp_path: Path,
    builds: list[RoadNetwork],
    caplog: pytest.LogCaptureFixture,
    where: str,
) -> None:
    cache = tmp_path / "cache"
    _load(xodr, cache)
    entry = _entry(cache)
    if where == "entry":
        entry.chmod(0o664)
    else:
        cache.chmod(0o775)
    with caplog.at_level("WARNING", logger=road_network_cache.__name__):
        _load(xodr, cache)
    assert len(builds) == 2
    assert "is writable by others" in caplog.text
    # The directory is the user's: the entry is written again -- the
    # directory closed to the group first, as one made under a umask of 002
    # is -- and read from then on, not refused on every run.
    assert stat.S_IMODE(entry.stat().st_mode) == 0o644
    assert stat.S_IMODE(cache.stat().st_mode) == 0o755
    _load(xodr, cache)
    assert len(builds) == 2


def test_a_directory_made_under_umask_002_is_closed_and_read(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork]
) -> None:
    # As an earlier version left it: mkdir() of the default mode, 0o775.
    cache = tmp_path / "cache"
    previous = os.umask(0o002)
    try:
        cache.mkdir()
        assert stat.S_IMODE(cache.stat().st_mode) == 0o775
        _load(xodr, cache)
        _load(xodr, cache)
    finally:
        os.umask(previous)
    assert len(builds) == 1
    assert stat.S_IMODE(cache.stat().st_mode) == 0o755


def test_a_world_writable_directory_is_a_miss(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork]
) -> None:
    cache = tmp_path / "cache"
    _load(xodr, cache)
    cache.chmod(0o777)
    _load(xodr, cache)
    assert len(builds) == 2
    assert stat.S_IMODE(cache.stat().st_mode) == 0o755
    _load(xodr, cache)
    assert len(builds) == 2


def test_a_symlinked_entry_is_a_miss(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork]
) -> None:
    cache = tmp_path / "cache"
    _load(xodr, cache)
    entry = _entry(cache)
    elsewhere = tmp_path / "elsewhere.pickle"
    entry.rename(elsewhere)
    entry.symlink_to(elsewhere)
    _load(xodr, cache)
    assert len(builds) == 2


class _Planted:
    """Stands for anything an entry should never name."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (os.getcwd, ())


@pytest.mark.parametrize(
    "planted", [_Planted(), {"x": np.float64}, pytest.param(eval, id="builtin")]
)
def test_an_entry_naming_another_class_is_a_miss(
    xodr: Path,
    tmp_path: Path,
    builds: list[RoadNetwork],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    planted: Any,
) -> None:
    cache = tmp_path / "cache"
    _load(xodr, cache)
    entry = _entry(cache)
    count = len(list(open_road_network(xodr).root.iter()))
    entry.write_bytes(pickle.dumps((count, [], [], {"1": planted})))
    called: list[object] = []
    monkeypatch.setattr(os, "getcwd", lambda: called.append(None))
    with caplog.at_level("WARNING", logger=road_network_cache.__name__):
        network = _load(xodr, cache)
    assert called == []
    assert len(builds) == 2
    assert sorted(network.road_ids_to_object) == ["1", "2"]
    assert "which the cache never writes" in caplog.text
    # ... and the entry is repaired.
    _load(xodr, cache)
    assert len(builds) == 2


def test_an_entry_names_only_allowed_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the fixture map's entry names: every global is on the allowlist,
    and the allowlist holds nothing the fixture map's entry does not name but
    numpy's scalars (which no road of it holds)."""
    named: set[tuple[str, str]] = set()
    find_class = road_network_cache._Unpickler.find_class

    def recording(self: Any, module: str, name: str) -> Any:
        named.add((module, name))
        return find_class(self, module, name)

    monkeypatch.setattr(road_network_cache._Unpickler, "find_class", recording)
    xodr = _DATA / "nishishinjuku_carla.xodr"
    _load(xodr, tmp_path)
    assert _load(xodr, tmp_path).road_ids_to_object
    assert named <= road_network_cache._ALLOWED_GLOBALS
    assert road_network_cache._ALLOWED_GLOBALS - named == {
        ("numpy._core.multiarray", "scalar"),
        # numpy 1.x's names, which numpy 2 does not write.
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
    }


def test_unwritable_cache_still_builds(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork]
) -> None:
    # A file where the cache directory should be: mkdir fails for anyone.
    (tmp_path / "cache").write_text("", encoding="utf-8")
    assert sorted(_load(xodr, tmp_path / "cache").road_ids_to_object) == ["1", "2"]
    assert sorted(_load(xodr, tmp_path / "cache").road_ids_to_object) == ["1", "2"]
    assert len(builds) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_a_network_it_did_not_open_is_only_built(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork]
) -> None:
    network = RoadNetwork(str(xodr))
    load_roads(network, root=tmp_path / "cache")
    assert builds == [network]
    assert not (tmp_path / "cache").exists()


def test_loaded_roads_are_not_loaded_again(
    xodr: Path, tmp_path: Path, builds: list[RoadNetwork]
) -> None:
    network = _load(xodr, tmp_path / "cache")
    roads = network.road_ids_to_object
    load_roads(network, root=tmp_path / "cache")
    assert network.road_ids_to_object is roads
    assert builds == [network]


def test_the_default_location_is_under_the_map_root(xodr: Path) -> None:
    # test/conftest.py points the map cache at a temporary directory.
    load_roads(open_road_network(xodr))
    entries = road_network_cache.map_root() / ".derived" / "road_network"
    assert list(entries.glob("*.pickle"))


# A loaded network equals a built one, on the fixture map.


@pytest.fixture(scope="module")
def networks(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[RoadNetwork, RoadNetwork]:
    xodr = _DATA / "nishishinjuku_carla.xodr"
    cache = tmp_path_factory.mktemp("cache")
    built = _load(xodr, cache)
    loaded = _load(xodr, cache)
    assert loaded is not built
    return built, loaded


def _or_error(read: Any) -> Any:
    try:
        return read()
    except Exception as error:  # noqa: BLE001 -- compared below
        return type(error)


def _same(built: Any, loaded: Any) -> None:
    if isinstance(built, type):
        assert loaded is built
    else:
        assert np.allclose(built, loaded)


def _ids(data: list[tuple[Any, Any]]) -> list[str]:
    return sorted(f"{lane!r} {position}" for lane, position in data)


def test_a_loaded_network_equals_a_built_one(
    networks: tuple[RoadNetwork, RoadNetwork],
) -> None:
    built, loaded = networks
    assert built.road_ids_to_object.keys() == loaded.road_ids_to_object.keys()
    for road_id, road in built.road_ids_to_object.items():
        twin = loaded.road_ids_to_object[road_id]
        assert np.allclose(road.reference_line, twin.reference_line)
        assert np.allclose(road.z_coordinates, twin.z_coordinates)
        for link in ("successor_data", "predecessor_data"):
            assert repr(getattr(road, link)) == repr(getattr(twin, link))
        sections = _or_error(lambda r=road: r.lane_sections)
        twins = _or_error(lambda r=twin: r.lane_sections)
        if isinstance(sections, type):
            assert twins is sections
            continue
        assert len(sections) == len(twins)
        for section, twin_section in zip(sections, twins):
            assert repr(section.successor_data) == repr(twin_section.successor_data)
            assert [lane.id for lane in section.lanes] == [
                lane.id for lane in twin_section.lanes
            ]
            for lane, twin_lane in zip(section.lanes, twin_section.lanes):
                _same(
                    _or_error(lambda la=lane: la.centre_line),
                    _or_error(lambda la=twin_lane: la.centre_line),
                )
                assert _ids(lane.successor_data) == _ids(twin_lane.successor_data)
                assert _ids(lane.predecessor_data) == _ids(twin_lane.predecessor_data)


def test_a_loaded_network_holds_the_elements_of_its_tree(
    networks: tuple[RoadNetwork, RoadNetwork],
) -> None:
    _built, loaded = networks
    elements = {id(element) for element in loaded.root.iter()}
    for road in loaded.road_ids_to_object.values():
        assert id(road.road_xml) in elements
        assert road.road_xml.getparent() is loaded.root
        for section in road.__dict__.get("lane_sections", []):
            assert id(section.lane_section_xml) in elements
            for lane in section.lanes:
                assert lane.lane_xml.getparent().getparent() is not None


def test_slices_stay_views_of_their_road(
    networks: tuple[RoadNetwork, RoadNetwork],
) -> None:
    _built, loaded = networks
    sections = [
        (road, section)
        for road in loaded.road_ids_to_object.values()
        for section in road.__dict__.get("lane_sections", [])
        if len(section.lane_section_reference_line)
    ]
    assert sections
    for road, section in sections:
        assert np.shares_memory(
            section.lane_section_reference_line, road.reference_line
        )


def test_the_traffic_light_lookup_reads_the_loaded_root(
    networks: tuple[RoadNetwork, RoadNetwork],
) -> None:
    built, loaded = networks
    controllers = [c.get("id") for c in built.root.findall("controller")]
    assert controllers
    for controller in controllers:
        assert get_signal_ids_for_controller(
            loaded, int(controller)
        ) == get_signal_ids_for_controller(built, int(controller))


@pytest.fixture
def manager() -> Iterator[MapManager]:
    MapManager.reset()
    yield MapManager.get_instance()
    MapManager.reset()


def test_map_manager_reads_its_roads_through_the_cache(
    manager: MapManager, builds: list[RoadNetwork]
) -> None:
    manager.initialize(_DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm")
    roads = manager.road_network.road_ids_to_object
    assert roads
    assert manager.road_network.road_ids_to_object is roads
    assert len(builds) <= 1
    entries = road_network_cache.map_root() / ".derived" / "road_network"
    assert list(entries.glob("*.pickle"))
