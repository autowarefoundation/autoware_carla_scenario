"""Regenerate this directory's two-way street with roadgen.

    uv run --with 'roadgen>=0.3.2' python generate.py

A straight 100 m road, one lane each way, as roadgen writes it: the backward
lane runs against the OpenDRIVE road's reference line, which is the case the
tests of frame-dependent positions need -- the converter's own fixture maps
are one-way roads throughout. The CARLA package's OpenDRIVE (its geoReference
is what CARLA and the MapManager anchor on) and the Lanelet2 map are kept;
the rest of the package is not.
"""

import shutil
import tempfile
from pathlib import Path

import roadgen

HERE = Path(__file__).resolve().parent

m = roadgen.Map(name="two_way_street", origin=(35.68, 139.69, 0.0))
m.add_road(
    start=(0.0, 0.0, 0.0),
    end=(100.0, 0.0, 0.0),
    lanes=[
        roadgen.Lane(width=3.5, direction="forward"),
        roadgen.Lane(width=3.5, direction="backward"),
    ],
    name="street",
)
with tempfile.TemporaryDirectory() as tmp:
    written = m.export_carla(tmp, name="two_way_street")
    shutil.copy(written["xodr"], HERE / "two_way_street.xodr")
m.export_lanelet2(str(HERE / "lanelet2_map.osm"))
(HERE / "map_projector_info.yaml").write_text("projector_type: Local\n")
