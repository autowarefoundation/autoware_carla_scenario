"""driver.hdmap_files: a map set's file names and map ids (#1728).

Needs no roadgen, unlike ``test_driver_hdmap.py``: these are what
``driver.hdmap_export`` re-exports from the checked module.
"""

from __future__ import annotations

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
