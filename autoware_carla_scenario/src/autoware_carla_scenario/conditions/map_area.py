"""Areas of the map chosen by what the ground is: crosswalks, walkways, the lanes
through a junction, Autoware's intersection areas, the raised and grooved parts
of a road's surface.

A :class:`MapArea` names ground on the Lanelet2 map by its attributes rather
than by drawing it: every lanelet of a subtype (``crosswalk``, ``walkway``,
``road_shoulder``, ``road``), the lanelets through a junction (those tagged
``turn_direction``), or every polygon of a type in the map's polygon layer
(Autoware's ``intersection_area``, ``detection_area``, ``no_stopping_area``,
``no_parking_area``, ``parking_lot``).  It narrows that further by picked
lanelets and by ``key=value`` attributes, so one crosswalk, the left turns
through a junction, or the lanelets of a private road are all one area::

    MapArea(AreaGround.CROSSWALK, lanelet_ids=(587,))
    MapArea(AreaGround.JUNCTION, attributes=(("turn_direction", "left"),))
    MapArea(AreaGround.INTERSECTION_AREA)

Its shapes are the outlines of what it matches (:func:`area_shapes`), in the
Lanelet2 map's frame, read from any loaded Lanelet2 map: the run reads them
from its own map, and the scenario editor from the map it previews, so the
outline the editor draws is the ground the run measures to.

The surface grounds are not on the Lanelet2 map at all: the parts of a road's
surface that stand at least :attr:`MapArea.relief_m` above its design
(``SURFACE_RAISED``: a speed hump, a frost heave) or below it
(``SURFACE_GROOVED``: ruts, a pothole), or either (``SURFACE_RELIEF``), read
from the ASAM OpenCRG grids the OpenDRIVE map's roads refer to
(:mod:`~autoware_carla_scenario.maps.opencrg`, :func:`surface_regions`).
They are outlines in a road's ``(s, t)``, placed in CARLA's world through the
OpenDRIVE map: the two maps stay apart, as everywhere in the framework::

    MapArea(AreaGround.SURFACE_RAISED, relief_m=0.02)
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from ..lanelet_junction import is_junction_lanelet
from ..maps.opencrg import DEFAULT_RELIEF_M, ReliefRegion, relief_regions

if TYPE_CHECKING:
    from lanelet2.core import AttributeMap, ConstLanelet, Lanelet, LaneletMap, Polygon3d

__all__ = [
    "AreaShape",
    "AreaGround",
    "MapArea",
    "area_lanelet_ids",
    "area_shapes",
    "map_area_from_fields",
    "parse_attributes",
    "surface_regions",
]


class AreaGround(enum.Enum):
    """What the ground of an area is, and so which layer of the map holds it.

    Attributes:
        CROSSWALK: Lanelets of subtype ``crosswalk``.
        WALKWAY: Lanelets of subtype ``walkway``.
        ROAD_SHOULDER: Lanelets of subtype ``road_shoulder``.
        ROAD: Lanelets of subtype ``road``.
        JUNCTION: Lanelets through a junction: those tagged
            ``turn_direction``.
        LANELET: Any lanelet; narrow it with picked lanelets or attributes.
        INTERSECTION_AREA: Polygons of type ``intersection_area``.
        DETECTION_AREA: Polygons of type ``detection_area``.
        NO_STOPPING_AREA: Polygons of type ``no_stopping_area``.
        NO_PARKING_AREA: Polygons of type ``no_parking_area``.
        PARKING_LOT: Polygons of type ``parking_lot``.
        POLYGON: Any polygon of the polygon layer; narrow it with attributes.
        SURFACE_RAISED: The parts of a road's surface standing at least
            ``relief_m`` above its design, from the OpenDRIVE map's OpenCRG
            grids.
        SURFACE_GROOVED: The parts standing at least ``relief_m`` below it.
        SURFACE_RELIEF: Either.
    """

    CROSSWALK = "crosswalk"
    WALKWAY = "walkway"
    ROAD_SHOULDER = "road_shoulder"
    ROAD = "road"
    JUNCTION = "junction"
    LANELET = "lanelet"
    INTERSECTION_AREA = "intersection_area"
    DETECTION_AREA = "detection_area"
    NO_STOPPING_AREA = "no_stopping_area"
    NO_PARKING_AREA = "no_parking_area"
    PARKING_LOT = "parking_lot"
    POLYGON = "polygon"
    SURFACE_RAISED = "surface_raised"
    SURFACE_GROOVED = "surface_grooved"
    SURFACE_RELIEF = "surface_relief"

    @property
    def on_lanelets(self) -> bool:
        """Whether the ground is lanelets (otherwise polygons of the polygon
        layer, or a road's surface)."""
        return self in _LANELET_GROUNDS

    @property
    def on_surface(self) -> bool:
        """Whether the ground is the relief of a road's surface, read from the
        OpenDRIVE map's OpenCRG grids rather than from the Lanelet2 map."""
        return self in _SURFACE_GROUNDS


_LANELET_GROUNDS = (
    AreaGround.CROSSWALK,
    AreaGround.WALKWAY,
    AreaGround.ROAD_SHOULDER,
    AreaGround.ROAD,
    AreaGround.JUNCTION,
    AreaGround.LANELET,
)

_SURFACE_GROUNDS = (
    AreaGround.SURFACE_RAISED,
    AreaGround.SURFACE_GROOVED,
    AreaGround.SURFACE_RELIEF,
)

#: What the scenario editor and a description call each ground.
_GROUND_TEXT = {
    AreaGround.CROSSWALK: "crosswalks",
    AreaGround.WALKWAY: "walkways",
    AreaGround.ROAD_SHOULDER: "road shoulders",
    AreaGround.ROAD: "roads",
    AreaGround.JUNCTION: "lanes through a junction",
    AreaGround.LANELET: "lanelets",
    AreaGround.INTERSECTION_AREA: "intersection areas",
    AreaGround.DETECTION_AREA: "detection areas",
    AreaGround.NO_STOPPING_AREA: "no-stopping areas",
    AreaGround.NO_PARKING_AREA: "no-parking areas",
    AreaGround.PARKING_LOT: "parking lots",
    AreaGround.POLYGON: "polygons",
    AreaGround.SURFACE_RAISED: "raised road surface",
    AreaGround.SURFACE_GROOVED: "grooved road surface",
    AreaGround.SURFACE_RELIEF: "raised or grooved road surface",
}


@dataclass(frozen=True)
class MapArea:
    """Ground on the map, chosen by what it is.

    An element of the map is in the area when it is of the *ground*, is one of
    *lanelet_ids* (when any are given), and has every one of *attributes*.  A
    surface ground is the relief of the roads' surface instead, by
    *relief_m*.

    Attributes:
        ground: What the ground is, and so which layer holds it.
        lanelet_ids: Lanelets it is narrowed to; empty: every lanelet of the
            ground.  Only for a ground of lanelets.
        attributes: ``(key, value)`` pairs every element must have, as the
            map tags them (``("turn_direction", "left")``); an empty value
            asks only that the key is there.
        relief_m: For a surface ground, how far (m) off the design the
            surface has to stand to count: 1 cm by default, above a road's
            ordinary unevenness and below any hump or rut a wheel feels.
    """

    ground: AreaGround = AreaGround.CROSSWALK
    lanelet_ids: tuple[int, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()
    relief_m: float = DEFAULT_RELIEF_M

    def validate(self) -> None:
        """Refuse an area no map could hold.

        Raises:
            ValueError: Picked lanelets with a ground that is not lanelets, a
                lanelet id that is not positive, an attribute with no key or
                on a surface ground (the relief has no tags), or a
                *relief_m* that is not positive.
        """
        if not math.isfinite(self.relief_m) or self.relief_m <= 0.0:
            raise ValueError(f"relief_m must be positive, not {self.relief_m}")
        if self.ground.on_surface:
            if self.lanelet_ids:
                raise ValueError(
                    f"{_GROUND_TEXT[self.ground]} is read from the OpenDRIVE "
                    "map's OpenCRG grids, not from lanelets: pick no lanelets"
                )
            if self.attributes:
                raise ValueError(
                    f"{_GROUND_TEXT[self.ground]} has no tags to filter by: "
                    "give no attributes"
                )
        if self.lanelet_ids and not self.ground.on_lanelets:
            raise ValueError(
                f"{_GROUND_TEXT[self.ground]} are polygons, not lanelets: "
                "narrow them by attributes, not by picked lanelets"
            )
        for lanelet_id in self.lanelet_ids:
            if lanelet_id <= 0:
                raise ValueError(f"lanelet ids are positive, not {lanelet_id}")
        for key, _value in self.attributes:
            if not key:
                raise ValueError("an attribute needs a key")

    def describe(self) -> str:
        """The area in words: ``crosswalks (lanelets 587, 612) where
        participant:pedestrian=yes``."""
        text = _GROUND_TEXT[self.ground]
        if self.ground.on_surface:
            return f"{text} ({self.relief_m:g} m and more)"
        if self.lanelet_ids:
            ids = ", ".join(str(i) for i in self.lanelet_ids)
            text += f" (lanelet{'s' if len(self.lanelet_ids) > 1 else ''} {ids})"
        if self.attributes:
            text += " where " + ", ".join(
                key if not value else f"{key}={value}" for key, value in self.attributes
            )
        return text


def parse_attributes(text: str) -> tuple[tuple[str, str], ...]:
    """``key=value`` pairs apart by commas or spaces (``turn_direction=left,
    location=urban``); a bare ``key`` asks only that it is there.

    Raises:
        ValueError: A pair with no key (``=left``) or an ``=`` with no value
            (``turn_direction=``), which is a value left out rather than one
            that may be anything: write the bare key for that.
    """
    pairs: list[tuple[str, str]] = []
    for part in text.replace(",", " ").split():
        key, sep, value = part.partition("=")
        if not key:
            raise ValueError(f"attribute {part!r} has no key")
        if sep and not value:
            raise ValueError(
                f"attribute {part!r} has no value: write {key!r} alone for any value"
            )
        pairs.append((key, value))
    return tuple(pairs)


def map_area_from_fields(
    ground: str = "CROSSWALK",
    lanelet_ids: Optional[Sequence[int]] = None,
    attributes: str = "",
    relief_m: Optional[float] = None,
) -> MapArea:
    """A :class:`MapArea` from a scenario document's fields: *ground* a
    :class:`AreaGround` by name, *attributes* one text of ``key=value`` pairs
    (:func:`parse_attributes`), no *relief_m* 1 cm.

    Raises:
        ValueError: An unknown *ground*, or an area no map could hold
            (:meth:`MapArea.validate`).
    """
    found: Optional[AreaGround] = None
    for member in AreaGround:
        if member.name == ground.upper():
            found = member
    if found is None:
        known = ", ".join(member.name for member in AreaGround)
        raise ValueError(f"unknown ground {ground!r}; known: {known}")
    lanelets: list[int] = []
    if lanelet_ids is not None:
        for lanelet in lanelet_ids:
            lanelets.append(int(lanelet))
    threshold = DEFAULT_RELIEF_M
    if relief_m is not None:
        threshold = float(relief_m)
    area = MapArea(
        ground=found,
        lanelet_ids=tuple(lanelets),
        attributes=parse_attributes(attributes),
        relief_m=threshold,
    )
    area.validate()
    return area


# ---------------------------------------------------------------------------
# Matching on a Lanelet2 map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AreaShape:
    """One element of the map an area matches, as an outline.

    Attributes:
        layer: ``"lanelet"`` or ``"polygon"``: which layer it is from.
        element_id: Its id in that layer.
        points: Its outline, ``(x, y)`` in the Lanelet2 map's frame.
    """

    layer: str
    element_id: int
    points: list[tuple[float, float]]


def _tag(attributes: "AttributeMap", key: str) -> Optional[str]:
    if key not in attributes:
        return None
    return str(attributes[key])


def _has_attributes(
    attributes: "AttributeMap", wanted: tuple[tuple[str, str], ...]
) -> bool:
    for key, value in wanted:
        held = _tag(attributes, key)
        if held is None or (value and held != value):
            return False
    return True


def _of_ground(lanelet: "ConstLanelet", ground: AreaGround) -> bool:
    attributes = lanelet.attributes
    if ground is AreaGround.JUNCTION:
        return is_junction_lanelet(lanelet)
    return ground is AreaGround.LANELET or _tag(attributes, "subtype") == ground.value


def _polygon_matches(polygon: "Polygon3d", area: MapArea) -> bool:
    attributes = polygon.attributes
    if area.ground is not AreaGround.POLYGON and _tag(attributes, "type") != (
        area.ground.value
    ):
        return False
    return _has_attributes(attributes, area.attributes)


def area_shapes(lanelet_map: "LaneletMap", area: MapArea) -> list[AreaShape]:
    """The outlines of every element of *lanelet_map* in *area*, in the order
    the map holds them (picked lanelets, in the order picked), in the map's
    own frame.  None for a surface ground, which is not on the Lanelet2 map
    (:func:`surface_regions`).

    Raises:
        ValueError: A picked lanelet that is not on the map, or not of the
            area's ground -- a road picked for a crosswalk is a mistake to
            say, not a lanelet to leave out quietly.
    """
    shapes: list[AreaShape] = []
    if area.ground.on_surface:
        return shapes
    if area.ground.on_lanelets:
        layer = lanelet_map.laneletLayer
        if area.lanelet_ids:
            candidates: list["Lanelet"] = []
            for lanelet_id in area.lanelet_ids:
                if not layer.exists(lanelet_id):
                    raise ValueError(f"lanelet {lanelet_id} is not on the map")
                picked = layer[lanelet_id]
                if not _of_ground(picked, area.ground):
                    raise ValueError(
                        f"lanelet {lanelet_id} is not one of the "
                        f"{_GROUND_TEXT[area.ground]}"
                    )
                candidates.append(picked)
        else:
            candidates = [
                lanelet for lanelet in layer if _of_ground(lanelet, area.ground)
            ]
        for lanelet in candidates:
            if _has_attributes(lanelet.attributes, area.attributes):
                points = [(float(p.x), float(p.y)) for p in lanelet.polygon2d()]
                shapes.append(AreaShape("lanelet", int(lanelet.id), points))
        return shapes
    for polygon in lanelet_map.polygonLayer:
        if _polygon_matches(polygon, area):
            points = [(float(p.x), float(p.y)) for p in polygon]
            shapes.append(AreaShape("polygon", int(polygon.id), points))
    return shapes


def area_lanelet_ids(lanelet_map: "LaneletMap", area: MapArea) -> list[int]:
    """The ids of the lanelets of *lanelet_map* in *area* (none for a ground
    of polygons): the lanelet layer of :func:`area_shapes`, which is what the
    scenario editor outlines on its map.  A shorthand for asking which
    lanelets an area takes; the framework itself calls :func:`area_shapes`."""
    return [
        s.element_id for s in area_shapes(lanelet_map, area) if s.layer == "lanelet"
    ]


# ---------------------------------------------------------------------------
# The relief of the roads' surface, on the OpenDRIVE map
# ---------------------------------------------------------------------------


def surface_regions(
    xodr_path: Path, area: MapArea
) -> tuple[list[ReliefRegion], list[str]]:
    """The regions of a surface ground on the OpenDRIVE map at *xodr_path*,
    outlines in their roads' ``(s, t)``, and what of the map's OpenCRG grids
    could not be read (:func:`~autoware_carla_scenario.maps.opencrg.relief_regions`).
    None for a ground that is not a surface's.

    Raises:
        OSError: The OpenDRIVE file cannot be read.
    """
    if not area.ground.on_surface:
        return ([], [])
    return relief_regions(
        xodr_path,
        raised=area.ground is not AreaGround.SURFACE_GROOVED,
        grooved=area.ground is not AreaGround.SURFACE_RAISED,
        threshold_m=area.relief_m,
    )
