"""The Lanelet2 z offset, remembered per map content (maps.z_offset_cache)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoware_carla_scenario.maps.z_offset_cache import cached_z_offset


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
