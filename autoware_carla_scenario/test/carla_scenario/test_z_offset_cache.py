"""The Lanelet2 z offset, remembered per map content (maps.z_offset_cache)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import json

from autoware_carla_scenario.maps.z_offset_cache import (
    _entry_json,
    cached_z_offset,
)


class _Compute:
    """A stand-in for MapManager._compute_z_offset that counts its calls."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.value


@pytest.fixture
def maps(tmp_path: Path) -> tuple[Path, Path]:
    lanelet2 = tmp_path / "lanelet2_map.osm"
    xodr = tmp_path / "map.xodr"
    lanelet2.write_text("<osm/>", encoding="utf-8")
    xodr.write_text("<OpenDRIVE/>", encoding="utf-8")
    return lanelet2, xodr


def _cached(
    maps: tuple[Path, Path],
    root: Path,
    compute: _Compute,
    name: str = "Town",
    projection: str = "mgrs",
) -> float:
    return cached_z_offset(
        *maps, name, compute, projection=projection, root=root / "cache"
    )


def test_miss_computes_and_stores(maps: tuple[Path, Path], tmp_path: Path) -> None:
    compute = _Compute(-0.0605)
    assert _cached(maps, tmp_path, compute) == -0.0605
    assert compute.calls == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_hit_does_not_compute(maps: tuple[Path, Path], tmp_path: Path) -> None:
    _cached(maps, tmp_path, _Compute(-0.0605))
    compute = _Compute(99.0)
    assert _cached(maps, tmp_path, compute) == -0.0605
    assert compute.calls == 0


@pytest.mark.parametrize("change", ["lanelet2", "xodr", "name", "projection"])
def test_changed_input_is_a_miss(
    maps: tuple[Path, Path], tmp_path: Path, change: str
) -> None:
    _cached(maps, tmp_path, _Compute(1.0))
    name, projection = "Town", "mgrs"
    if change == "lanelet2":
        maps[0].write_text("<osm version='2'/>", encoding="utf-8")
    elif change == "xodr":
        maps[1].write_text("<OpenDRIVE version='2'/>", encoding="utf-8")
    elif change == "name":
        name = "OtherTown"
    else:
        projection = "utm"
    compute = _Compute(2.0)
    assert _cached(maps, tmp_path, compute, name, projection) == 2.0
    assert compute.calls == 1


@pytest.mark.parametrize("content", ["not json", "{}", '{"z_offset": "x"}', "[]"])
def test_corrupt_entry_is_recomputed(
    maps: tuple[Path, Path], tmp_path: Path, content: str
) -> None:
    _cached(maps, tmp_path, _Compute(1.0))
    (entry,) = (tmp_path / "cache").glob("*.json")
    entry.write_text(content, encoding="utf-8")
    compute = _Compute(2.0)
    assert _cached(maps, tmp_path, compute) == 2.0
    assert compute.calls == 1
    # ... and the entry is repaired.
    assert _cached(maps, tmp_path, _Compute(3.0)) == 2.0


def test_unwritable_cache_still_returns_the_value(
    maps: tuple[Path, Path], tmp_path: Path
) -> None:
    # A file where the cache directory should be: mkdir fails for anyone.
    (tmp_path / "cache").write_text("", encoding="utf-8")
    compute = _Compute(1.5)
    assert _cached(maps, tmp_path, compute) == 1.5
    assert _cached(maps, tmp_path, compute) == 1.5
    assert compute.calls == 2


def test_a_file_is_hashed_once_while_it_is_unchanged(
    maps: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queue re-initialises the map manager for every run: the maps' digests
    are taken again only once a file has changed."""
    import os

    from autoware_carla_scenario.maps import z_offset_cache

    monkeypatch.setattr(z_offset_cache, "_DIGESTS", {})
    opened: list[str] = []
    real_open = Path.open

    def counted(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self in maps:
            opened.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    for _ in range(3):
        _cached(maps, tmp_path, _Compute(1.0))
    assert sorted(opened) == ["lanelet2_map.osm", "map.xodr"]
    stat = maps[1].stat()
    maps[1].write_text("<OpenDRIVE version='2'/>", encoding="utf-8")
    os.utime(maps[1], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    opened.clear()  # write_text() opens it too
    assert _cached(maps, tmp_path, _Compute(2.0)) == 2.0
    assert opened == ["map.xodr"]


def test_the_entry_is_a_json_object_with_z_offset_a_number(
    maps: tuple[Path, Path], tmp_path: Path
) -> None:
    _cached(maps, tmp_path, _Compute(-0.0605))
    (entry,) = (tmp_path / "cache").glob("*.json")
    document = json.loads(entry.read_text(encoding="utf-8"))
    assert document["z_offset"] == -0.0605
    assert type(document["z_offset"]) is float
    assert list(document) == ["z_offset", "lanelet2", "xodr", "carla_map", "projection"]
    assert document["carla_map"] == "Town"
    assert document["projection"] == "mgrs"


@pytest.mark.parametrize("value", [-0.0605, 0.0, 1.5, 1e-07, 3.0])
@pytest.mark.parametrize("projection", ["", "mgrs", 'a "quoted" \u00e9 name'])
def test_the_entry_is_what_json_dumps_of_the_whole_mapping_writes(
    value: float, projection: str
) -> None:
    lanelet2, xodr = Path("/maps/a.osm"), Path("/maps/b.xodr")
    assert _entry_json(value, lanelet2, xodr, "Town", projection) == json.dumps(
        {
            "z_offset": value,
            "lanelet2": str(lanelet2),
            "xodr": str(xodr),
            "carla_map": "Town",
            "projection": projection,
        }
    )


def test_an_entry_written_before_still_loads(
    maps: tuple[Path, Path], tmp_path: Path
) -> None:
    from autoware_carla_scenario.maps.z_offset_cache import z_offset_key

    key = z_offset_key(*maps, "Town", "mgrs")
    root = tmp_path / "cache"
    root.mkdir()
    # Byte for byte what json.dumps wrote before the entry was built by hand.
    (root / f"{key}.json").write_text(
        '{"z_offset": -0.0605, "lanelet2": "l.osm", "xodr": "m.xodr", '
        '"carla_map": "Town", "projection": "mgrs"}',
        encoding="utf-8",
    )
    compute = _Compute(99.0)
    assert _cached(maps, tmp_path, compute) == -0.0605
    assert compute.calls == 0
