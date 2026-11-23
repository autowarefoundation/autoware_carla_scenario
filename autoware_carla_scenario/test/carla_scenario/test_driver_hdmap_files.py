"""driver.hdmap_files: a map set's file names and map ids (#1728).

Needs no roadgen, unlike ``test_driver_hdmap.py``: these are what
``driver.hdmap_export`` re-exports from the checked module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pytest

from autoware_carla_scenario.driver import hdmap_export, hdmap_files
from autoware_carla_scenario.driver.hdmap_files import map_id_for


@pytest.mark.parametrize(
    ("map_name", "opendrive", "provided", "expected"),
    [
        ("Carla/Maps/Town 10", "<OpenDRIVE/>", None, "Town_10-255e3acfbbbf"),
        ("Carla/Maps/Town 10", "<OpenDRIVE/>", {}, "Town_10-255e3acfbbbf"),
        (
            "Carla/Maps/Town 10",
            "<OpenDRIVE/>",
            {"t4": b"\x01" * 32, "lanelet2": bytes(range(32))},
            "Town_10-7f78b558602e",
        ),
        ("", "x", {}, "map-2d711642b726"),
    ],
)
def test_map_id_is_pinned(
    map_name: str,
    opendrive: str,
    provided: Optional[Mapping[str, bytes]],
    expected: str,
) -> None:
    """Map ids name sets already on disk: moving the function changes none."""
    assert map_id_for(map_name, opendrive, provided) == expected


def test_hdmap_export_re_exports_the_same_objects() -> None:
    for name in hdmap_files.__all__:
        assert getattr(hdmap_export, name) is getattr(hdmap_files, name), name
    assert hdmap_export.__all__ == [
        "MANIFEST_FILE",
        "MAP_FORMATS",
        "export_map",
        "map_id_for",
    ]


_OSM = (
    "<?xml version='1.0'?>\n<osm version='0.6'><node id='1' lat='0' lon='0'/></osm>\n"
)


@pytest.fixture
def given_osm(tmp_path: Path) -> Path:
    path = tmp_path / "given" / "lanelet2_map.osm"
    path.parent.mkdir()
    path.write_text(_OSM, encoding="utf-8")
    return path


def test_a_given_map_without_projector_keeps_its_id(
    given_osm: Path, tmp_path: Path
) -> None:
    """Sets already on disk: a given map without a projector keeps its id."""
    map_id = hdmap_export.export_map(
        "<OpenDRIVE/>", "Town", tmp_path / "maps", formats=(), lanelet2_path=given_osm
    )
    assert map_id == "Town-c1e43f649d0f"


def test_the_projector_of_a_given_map_is_in_its_id(
    given_osm: Path, tmp_path: Path
) -> None:
    """#2378: editing only the projector used to reuse the stale set."""
    maps = tmp_path / "maps"
    projector = given_osm.parent / "map_projector_info.yaml"
    ids = []
    for grid in ("54SUE", "53SQU"):
        text = f"projector_type: MGRS\nvertical_datum: WGS84\nmgrs_grid: {grid}\n"
        projector.write_text(text, encoding="utf-8")
        map_id = hdmap_export.export_map(
            "<OpenDRIVE/>", "Town", maps, formats=(), lanelet2_path=given_osm
        )
        assert (maps / map_id / "map_projector_info.yaml").read_text() == text
        ids.append(map_id)
    assert ids[0] != ids[1]
    assert "Town-c1e43f649d0f" not in ids
