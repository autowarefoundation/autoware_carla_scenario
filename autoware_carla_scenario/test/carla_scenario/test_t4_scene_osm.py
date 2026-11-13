"""The map resource's OSM bytes, byte for byte (``t4_scene.map_resource``).

Pins what ``local_meters_osm()`` writes and ``resource_digest()`` hashes, and
the refusals of a map the ``LOCAL_METERS`` profile cannot hold, on small maps
built here and on the fixture map, so that moving the helpers into a module
the library check compiles (#2116) changes nothing.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
from typing import Any

import lanelet2
import lanelet2.core
import lanelet2.io
import pytest

from autoware_carla_scenario.coordinate.projection import load_lanelet2, map_origin
from autoware_carla_scenario.t4_scene.map_resource import (
    CENTERLINE_INVERTED_TAG,
    local_meters_osm,
    map_resource,
    resource_digest,
)

DATA = Path(__file__).resolve().parents[3] / "data"
OSM = DATA / "nishishinjuku.osm"
XODR = DATA / "nishishinjuku_carla.xodr"


def _point(id_: int, x: float, y: float, z: float = 0.0) -> Any:
    return lanelet2.core.Point3d(id_, x, y, z)


def _lanelet_map(centerline: Any = None) -> Any:
    """One lanelet (20) between two bounds (10, 11), with *centerline* as
    its custom centerline (added to the map too) when given."""
    left = lanelet2.core.LineString3d(
        10, [_point(1, 0.0, 0.0, 0.5), _point(2, 10.0, 0.0)]
    )
    right = lanelet2.core.LineString3d(
        11, [_point(3, 0.0, 3.5), _point(4, 10.0, 3.5, 1.25)]
    )
    lanelet = lanelet2.core.Lanelet(20, left, right)
    lanelet_map = lanelet2.core.LaneletMap()
    if centerline is not None:
        lanelet.centerline = centerline
        lanelet_map.add(centerline)
    lanelet_map.add(lanelet)
    return lanelet_map


def _centerline(*points: Any) -> Any:
    return lanelet2.core.LineString3d(12, list(points))


def _plain() -> Any:
    return _lanelet_map()


def _custom() -> Any:
    return _lanelet_map(_centerline(_point(5, 0.0, 1.75), _point(6, 10.0, 1.75)))


def _custom_reversed() -> Any:
    """A custom centerline whose way runs backwards: still read forwards."""
    return _lanelet_map(_centerline(_point(6, 10.0, 1.75), _point(5, 0.0, 1.75)))


def _inverted() -> Any:
    """A custom centerline the lanelet reads backwards (the way's invert())."""
    line = _centerline(_point(5, 0.0, 1.75), _point(6, 10.0, 1.75))
    lanelet_map = _lanelet_map(line)
    lanelet_map.laneletLayer[20].centerline = line.invert()
    return lanelet_map


def _tagged() -> Any:
    """Points that carry ``ele`` or ``local_x`` tags of their own, and a
    tagged lanelet."""
    lanelet_map = _lanelet_map()
    lanelet_map.laneletLayer[20].attributes["subtype"] = "road"
    lanelet_map.pointLayer[1].attributes["ele"] = "7"
    lanelet_map.pointLayer[2].attributes["local_x"] = "99"
    return lanelet_map


def _fixture() -> Any:
    """The nishishinjuku map as a run reads it (test_t4_scene's fixture)."""
    (lat, lon, alt), _ = map_origin(OSM, XODR)
    lanelet_map, _projector, _ = load_lanelet2(OSM, lanelet2.io.Origin(lat, lon, alt))
    return lanelet_map


_HEAD = (
    b"<?xml version='1.0' encoding='utf-8'?>\n"
    b'<osm version="0.6" upload="false" generator="lanelet2">\n'
)
_NODES = (
    b'  <node id="1" visible="true" version="1" lat="0.0" lon="0.0">\n'
    b'    <tag k="ele" v="0.5" />\n'
    b'  <tag k="local_x" v="0.0" /><tag k="local_y" v="0.0" /></node>\n'
    b'  <node id="2" visible="true" version="1" lat="0.0"'
    b' lon="8.983152841195213e-05"><tag k="local_x" v="10.0" />'
    b'<tag k="local_y" v="0.0" /><tag k="ele" v="0.0" /></node>\n'
    b'  <node id="3" visible="true" version="1" lat="3.1441034934687195e-05"'
    b' lon="0.0"><tag k="local_x" v="0.0" /><tag k="local_y" v="3.5" />'
    b'<tag k="ele" v="0.0" /></node>\n'
    b'  <node id="4" visible="true" version="1" lat="3.1441034934687195e-05"'
    b' lon="8.983152841195213e-05">\n'
    b'    <tag k="ele" v="1.25" />\n'
    b'  <tag k="local_x" v="10.0" /><tag k="local_y" v="3.5" /></node>\n'
)
_CENTER_NODES = (
    b'  <node id="5" visible="true" version="1" lat="1.5720517481554452e-05"'
    b' lon="0.0"><tag k="local_x" v="0.0" /><tag k="local_y" v="1.75" />'
    b'<tag k="ele" v="0.0" /></node>\n'
    b'  <node id="6" visible="true" version="1" lat="1.5720517481554452e-05"'
    b' lon="8.983152841195213e-05"><tag k="local_x" v="10.0" />'
    b'<tag k="local_y" v="1.75" /><tag k="ele" v="0.0" /></node>\n'
)
_BOUNDS = (
    b'  <way id="10" visible="true" version="1">\n'
    b'    <nd ref="1" />\n'
    b'    <nd ref="2" />\n'
    b"  </way>\n"
    b'  <way id="11" visible="true" version="1">\n'
    b'    <nd ref="3" />\n'
    b'    <nd ref="4" />\n'
    b"  </way>\n"
)


def _centerline_way(*refs: int) -> bytes:
    nds = b"".join(b'    <nd ref="%d" />\n' % ref for ref in refs)
    return b'  <way id="12" visible="true" version="1">\n' + nds + b"  </way>\n"


def _relation(inverted: str) -> bytes:
    return (
        b'  <relation id="20" visible="true" version="1">\n'
        b'    <member type="way" ref="10" role="left" />\n'
        b'    <member type="way" ref="11" role="right" />\n'
        b'    <member type="way" ref="12" role="centerline" />\n'
        b'    <tag k="type" v="lanelet" />\n'
        b'  <tag k="devkit:centerline_inverted" v="' + inverted.encode() + b'" />'
        b"</relation>\n"
        b"</osm>"
    )


_EXPECTED = {
    "plain": _HEAD
    + _NODES
    + _BOUNDS
    + b'  <relation id="20" visible="true" version="1">\n'
    b'    <member type="way" ref="10" role="left" />\n'
    b'    <member type="way" ref="11" role="right" />\n'
    b'    <tag k="type" v="lanelet" />\n'
    b"  </relation>\n"
    b"</osm>",
    "custom": _HEAD
    + _NODES
    + _CENTER_NODES
    + _BOUNDS
    + _centerline_way(5, 6)
    + _relation("no"),
    "custom_reversed": _HEAD
    + _NODES
    + _CENTER_NODES
    + _BOUNDS
    + _centerline_way(6, 5)
    + _relation("no"),
    "inverted": _HEAD
    + _NODES
    + _CENTER_NODES
    + _BOUNDS
    + _centerline_way(5, 6)
    + _relation("yes"),
    "tagged": _HEAD
    + b'  <node id="1" visible="true" version="1" lat="0.0" lon="0.0">\n'
    b'    <tag k="ele" v="0.5" />\n'
    b'    <tag k="ele" v="0.5" />\n'
    b'  <tag k="local_x" v="0.0" /><tag k="local_y" v="0.0" /></node>\n'
    b'  <node id="2" visible="true" version="1" lat="0.0"'
    b' lon="8.983152841195213e-05">\n'
    b'    <tag k="local_x" v="10.0" />\n'
    b'  <tag k="local_y" v="0.0" /><tag k="ele" v="0.0" /></node>\n'
    + _NODES[_NODES.index(b'  <node id="3"') :]
    + _BOUNDS
    + b'  <relation id="20" visible="true" version="1">\n'
    b'    <member type="way" ref="10" role="left" />\n'
    b'    <member type="way" ref="11" role="right" />\n'
    b'    <tag k="subtype" v="road" />\n'
    b'    <tag k="type" v="lanelet" />\n'
    b"  </relation>\n"
    b"</osm>",
}

_BUILDERS = {
    "plain": _plain,
    "custom": _custom,
    "custom_reversed": _custom_reversed,
    "inverted": _inverted,
    "tagged": _tagged,
}


@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_local_meters_osm_writes_these_bytes(name: str) -> None:
    lanelet_map = _BUILDERS[name]()
    data = local_meters_osm(lanelet_map)
    assert type(data) is bytes
    assert data == _EXPECTED[name]
    # The same bytes again: the map is not changed by writing it.
    assert local_meters_osm(lanelet_map) == data


def test_writing_leaves_the_map_as_it_was() -> None:
    lanelet_map = _tagged()
    before = [(p.id, p.x, p.y, p.z, dict(p.attributes)) for p in lanelet_map.pointLayer]
    local_meters_osm(lanelet_map)
    after = [(p.id, p.x, p.y, p.z, dict(p.attributes)) for p in lanelet_map.pointLayer]
    assert after == before
    assert dict(lanelet_map.laneletLayer[20].attributes) == {"subtype": "road"}
    assert CENTERLINE_INVERTED_TAG not in lanelet_map.laneletLayer[20].attributes


#: The fixture map's resource: its length and SHA-256.
_FIXTURE_LENGTH = 12545566
_FIXTURE_SHA256 = "d0b87b29079de2ba1940eccecad310b99080f5ba4d41a6267a855bebf43760ce"


def test_the_fixture_maps_bytes_are_these() -> None:
    data = local_meters_osm(_fixture())
    assert data.startswith(_HEAD)
    assert len(data) == _FIXTURE_LENGTH
    assert hashlib.sha256(data).hexdigest() == _FIXTURE_SHA256


def test_the_resource_digest_hashes_the_profile_then_the_bytes() -> None:
    assert (
        resource_digest(b"")
        == "7565ab2cd4d6c28fa5c48fa1475a99ae640d8cba5255011b3da6c6dfe4a6f32f"
    )
    assert (
        resource_digest(b"abc")
        == "2597b8064e008c89e9741b45e576f22dfcd697f620fb959d924bff0569fe67d4"
    )
    payload = _EXPECTED["custom"]
    assert (
        resource_digest(payload)
        == hashlib.sha256(b"lanelet2-osm\0local-meters\0" + payload).hexdigest()
    )


def test_map_resource_carries_the_bytes_and_their_digest() -> None:
    resource = map_resource(_custom())
    assert resource.data == _EXPECTED["custom"]
    assert resource.sha256 == resource_digest(_EXPECTED["custom"])


@pytest.mark.parametrize(
    ("x", "y", "z", "message"),
    [
        (
            math.nan,
            0.0,
            0.0,
            "the map has a point at (nan, 0.0, 0.0), which is not finite",
        ),
        (
            0.0,
            math.inf,
            0.0,
            "the map has a point at (0.0, inf, 0.0), which is not finite",
        ),
        (
            0.0,
            0.0,
            -math.inf,
            "the map has a point at (0.0, 0.0, -inf), which is not finite",
        ),
        (
            0.0,
            1e9,
            0.0,
            "the map's point (0.0, 1000000000.0) is too far out for the"
            " LOCAL_METERS profile",
        ),
    ],
    ids=["nan-x", "inf-y", "-inf-z", "too-far-out"],
)
def test_a_point_the_profile_cannot_hold_is_refused(
    x: float, y: float, z: float, message: str
) -> None:
    lanelet_map = lanelet2.core.LaneletMap()
    lanelet_map.add(_point(1, x, y, z))
    with pytest.raises(ValueError) as raised:
        local_meters_osm(lanelet_map)
    assert str(raised.value) == message
    assert type(raised.value) is ValueError


@pytest.mark.parametrize(
    ("x", "y", "length"),
    [(1e9, 0.0, 271), (0.0, 3e7, 269), (0.0, 2e7, 269)],
)
def test_a_far_point_mercator_takes_back_is_written(
    x: float, y: float, length: int
) -> None:
    lanelet_map = lanelet2.core.LaneletMap()
    lanelet_map.add(_point(1, x, y))
    assert len(local_meters_osm(lanelet_map)) == length


def test_a_map_lanelet2_cannot_write_is_refused_with_its_first_errors() -> None:
    # A custom centerline that is not in the map: the relation names a way
    # Lanelet2 does not write.
    lanelet_map = _lanelet_map()
    lanelet_map.laneletLayer[20].centerline = _centerline(
        _point(5, 0.0, 1.75), _point(6, 10.0, 1.75)
    )
    with pytest.raises(ValueError) as raised:
        local_meters_osm(lanelet_map)
    assert str(raised.value) == (
        "Lanelet2 could not write the map: ['Errors ocurred while writing"
        " Lanelet Map:', '\\t- Error writing primitive 20: Relation has a member"
        " with id 12 that is not in the map!']"
    )


def test_the_module_exports_these_names() -> None:
    # t4_scene re-exports a map_resource() function under the module's name.
    module = importlib.import_module("autoware_carla_scenario.t4_scene.map_resource")
    assert module.__all__ == [
        "CENTERLINE_INVERTED_TAG",
        "local_meters_osm",
        "map_resource",
        "resource_digest",
    ]
    assert CENTERLINE_INVERTED_TAG == "devkit:centerline_inverted"


def test_map_resource_re_exports_the_checked_osm_helpers() -> None:
    # The helpers live in t4_scene.osm, which the library check compiles
    # (#2116); t4_scene.map_resource re-exports the same objects.
    osm = importlib.import_module("autoware_carla_scenario.t4_scene.osm")
    module = importlib.import_module("autoware_carla_scenario.t4_scene.map_resource")
    assert osm.__all__ == [
        "CENTERLINE_INVERTED_TAG",
        "local_meters_osm",
        "resource_digest",
    ]
    for name in osm.__all__:
        assert getattr(module, name) is getattr(osm, name), name
