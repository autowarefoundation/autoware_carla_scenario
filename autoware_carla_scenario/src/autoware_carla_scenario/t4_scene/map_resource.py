"""A loaded Lanelet2 map as the T4Scene contract's ``MapResource``.

A T4Scene carries no map bytes: its ``MapReference`` names a ``MapResource``
by SHA-256, and the resource travels on its own (docs/t4_scene.md).  The
contract has one profile, ``LANELET2_OSM`` in ``LOCAL_METERS``: the map as
Lanelet2 OSM XML whose nodes carry their exact coordinates, in the frame the
map was loaded in, as ``local_x``/``local_y``/``ele`` tags, with ``lat``/``lon``
the same coordinates through a Mercator projector at (0, 0) -- auxiliary
values that keep Lanelet2's boundary alignment right when the file is read
back, not a geolocation.  This is the devkit's own encoder
(``t4_e2e_devkit.maps.io.osm_bytes``), rewritten on Lanelet2 and the standard
library so the export needs neither the devkit nor its torch.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import lanelet2
import lanelet2.core
import lanelet2.io
import lanelet2.projection

from ._proto import t4_scene_pb2 as pb

__all__ = [
    "CENTERLINE_INVERTED_TAG",
    "local_meters_osm",
    "map_resource",
    "resource_digest",
]

#: The transport tag the contract reserves on a lanelet with a custom
#: centerline: ``yes`` when the lanelet reads its centerline way backwards.
CENTERLINE_INVERTED_TAG = "devkit:centerline_inverted"


def resource_digest(payload: bytes) -> str:
    """The resource identity: SHA-256 over the encoding, the projection and the OSM bytes."""
    return hashlib.sha256(b"lanelet2-osm\0local-meters\0" + payload).hexdigest()


def _auxiliary_gps(
    projector: lanelet2.projection.MercatorProjector, x: float, y: float, z: float
) -> lanelet2.core.GPSPoint:
    """A point's auxiliary lat/lon, which must project back within a micrometre."""
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError(f"the map has a point at ({x}, {y}, {z}), which is not finite")
    gps = projector.reverse(lanelet2.core.BasicPoint3d(x, y, z))
    back = projector.forward(gps)
    if abs(back.x - x) > 1e-6 or abs(back.y - y) > 1e-6:
        raise ValueError(
            f"the map's point ({x}, {y}) is too far out for the LOCAL_METERS profile"
        )
    return gps


def local_meters_osm(lanelet_map: lanelet2.core.LaneletMap) -> bytes:
    """*lanelet_map* as OSM XML in the ``LOCAL_METERS`` profile.

    The map is written as it is, with its own IDs; *lanelet_map* itself is not
    changed.

    Raises:
        ValueError: If a point is not finite or too far out for the profile,
            Lanelet2 could not write the map, or it wrote a custom centerline
            with other points than the lanelet's.
    """
    projector = lanelet2.projection.MercatorProjector(lanelet2.io.Origin(0, 0))
    gps = {
        point.id: _auxiliary_gps(projector, point.x, point.y, point.z)
        for point in lanelet_map.pointLayer
    }
    with tempfile.TemporaryDirectory(prefix="t4scene-osm-") as directory:
        path = Path(directory) / "map.osm"
        errors = lanelet2.io.writeRobust(str(path), lanelet_map, projector)
        if errors:
            raise ValueError(f"Lanelet2 could not write the map: {errors[:3]}")
        root = ET.fromstring(path.read_bytes())
    for node in root.findall("node"):
        point = lanelet_map.pointLayer[int(node.attrib["id"])]
        node.set("lat", repr(gps[point.id].lat))
        node.set("lon", repr(gps[point.id].lon))
        tags = {tag.attrib["k"]: tag for tag in node.findall("tag")}
        for name, value in (
            ("local_x", point.x),
            ("local_y", point.y),
            ("ele", point.z),
        ):
            tag = tags.get(name)
            if tag is None:
                tag = ET.SubElement(node, "tag", k=name)
            tag.set("v", repr(float(value)))
    ways = {int(way.attrib["id"]): way for way in root.findall("way")}
    for relation in root.findall("relation"):
        member = relation.find("member[@role='centerline']")
        if member is None:
            continue
        line = lanelet_map.laneletLayer[int(relation.attrib["id"])].centerline
        stored = [
            int(nd.attrib["ref"])
            for nd in ways[int(member.attrib["ref"])].findall("nd")
        ]
        desired = [point.id for point in line]
        if desired not in (stored, stored[::-1]):
            raise ValueError("Lanelet2 wrote a custom centerline with other points")
        tag = relation.find(f"tag[@k='{CENTERLINE_INVERTED_TAG}']")
        if tag is None:
            tag = ET.SubElement(relation, "tag", k=CENTERLINE_INVERTED_TAG)
        tag.set("v", "yes" if desired != stored else "no")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def map_resource(lanelet_map: lanelet2.core.LaneletMap) -> pb.MapResource:
    """*lanelet_map* as a ``MapResource`` a T4Scene's ``MapReference`` names."""
    payload = local_meters_osm(lanelet_map)
    return pb.MapResource(
        sha256=resource_digest(payload),
        encoding=pb.MapResource.LANELET2_OSM,
        projection=pb.MapResource.LOCAL_METERS,
        data=payload,
    )
