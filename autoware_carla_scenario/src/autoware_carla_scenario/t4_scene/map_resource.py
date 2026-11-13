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
library so the export needs neither the devkit nor its torch.  The OSM bytes
and their digest are t4_scene.osm's (:func:`local_meters_osm`,
:func:`resource_digest`), which Codon checks (#2116), re-exported here.
"""

from __future__ import annotations

import lanelet2
import lanelet2.core

from ._proto import t4_scene_pb2 as pb
from .osm import CENTERLINE_INVERTED_TAG, local_meters_osm, resource_digest

__all__ = [
    "CENTERLINE_INVERTED_TAG",
    "local_meters_osm",
    "map_resource",
    "resource_digest",
]


def map_resource(lanelet_map: lanelet2.core.LaneletMap) -> pb.MapResource:
    """*lanelet_map* as a ``MapResource`` a T4Scene's ``MapReference`` names."""
    payload = local_meters_osm(lanelet_map)
    return pb.MapResource(
        sha256=resource_digest(payload),
        encoding=pb.MapResource.LANELET2_OSM,
        projection=pb.MapResource.LOCAL_METERS,
        data=payload,
    )
