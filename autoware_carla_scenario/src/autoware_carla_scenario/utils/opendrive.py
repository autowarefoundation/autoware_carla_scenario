"""CARLA's OpenDRIVE, made acceptable to roadgen.

Shared by the two places that convert a world's map with roadgen: the SUMO
backend (:mod:`..traffic.sumo.network`) and the map files written for a driver
policy (:mod:`..driver.hdmap_export`).  This module imports nothing but the
standard library.
"""

from __future__ import annotations

import re

__all__ = ["sanitize_opendrive"]


def sanitize_opendrive(text: str) -> str:
    """Make CARLA's OpenDRIVE acceptable to roadgen's OpenDRIVE 1.7 parser.

    CARLA's maps (made with RoadRunner) bend the schema in places the strict
    parser refuses the whole document for, none of which carries road geometry:
    ``<userData>`` without a ``code``, ``<roadMark>`` without a ``color``,
    objects of type ``-1`` and outline corners (``<cornerLocal>``,
    ``<cornerRoad>``) without a ``height``.
    """
    text = re.sub(r"<userData\b[^>]*/>|<userData\b.*?</userData>", "", text, flags=re.S)
    text = re.sub(r'(<object [^>]*?)type="-1"', r'\1type="none"', text)
    text = re.sub(r"<roadMark (?![^>]*color=)", '<roadMark color="standard" ', text)
    text = re.sub(r"<cornerLocal (?![^>]*height=)", '<cornerLocal height="0" ', text)
    return re.sub(r"<cornerRoad (?![^>]*height=)", '<cornerRoad height="0" ', text)
