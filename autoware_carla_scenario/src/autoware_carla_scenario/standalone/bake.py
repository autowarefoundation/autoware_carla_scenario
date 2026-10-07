"""Write the map data a standalone scenario binary reads at run time.

A scenario binary (docs/standalone.md) has neither Lanelet2 nor pyxodr: the
coordinate functions it runs (``to_carla_world``, ``to_opendrive``,
``on_lanelet``, ...) work on the geometry baked here, at build time, from the
same files :class:`~autoware_carla_scenario.coordinate.MapManager` loads:

* each lanelet's centerline (3D, in the Lanelet2 map frame) and its outline
  (the 2D polygon ``lanelet.polygon2d()`` is, used for ``inside`` and
  ``findNearest``);
* each OpenDRIVE road's reference line and the elevation along it, as pyxodr
  samples them;
* the MGRS offset between the two frames, and the z offset MapManager falls
  back on when it has no CARLA world (the binary measures the real one from
  the world's spawn points, as MapManager does when it has one).

The file is plain text, one record per line, every float written with
``repr`` so it reads back to the same double::

    acsmap 1
    mgrs_offset <x> <y>
    z_offset <z>
    lanelet <id> <n> <x> <y> <z> ...      # centerline, n points
    outline <id> <m> <x> <y> ...          # polygon2d, m points
    road <id> <k> <x> <y> <z> ...         # reference line and elevation
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["BakeStats", "FORMAT_VERSION", "bake_map"]

#: Bumped whenever the record layout changes; the runtime refuses another.
FORMAT_VERSION = 1


@dataclass(frozen=True)
class BakeStats:
    """What :func:`bake_map` wrote."""

    path: Path
    lanelets: int
    roads: int
    bytes: int


def _floats(values: Iterable[float]) -> str:
    return " ".join(repr(float(v)) for v in values)


def bake_map(
    xodr_path: Path,
    lanelet2_path: Path,
    out_path: Path,
    *,
    projector_type: str | None = None,
) -> BakeStats:
    """Bake the map *xodr_path* / *lanelet2_path* into *out_path*.

    Loads the map with a :class:`MapManager` of its own (the process-wide one
    is reset first and after), so the geometry is exactly what the Python
    runner would compute with.
    """
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415

    MapManager.reset()
    try:
        mm = MapManager.get_instance()
        mm.initialize(xodr_path, lanelet2_path, projector_type=projector_type)
        offset_x, offset_y = mm.mgrs_offset
        lines = [
            f"acsmap {FORMAT_VERSION}",
            f"mgrs_offset {_floats((offset_x, offset_y))}",
            f"z_offset {_floats((mm.z_offset,))}",
        ]
        lanelets = 0
        for lanelet in mm.lanelet_map.laneletLayer:
            centerline = [c for p in lanelet.centerline for c in (p.x, p.y, p.z)]
            outline = [c for p in lanelet.polygon2d() for c in (p.x, p.y)]
            lines.append(
                f"lanelet {lanelet.id} {len(centerline) // 3} {_floats(centerline)}"
            )
            lines.append(f"outline {lanelet.id} {len(outline) // 2} {_floats(outline)}")
            lanelets += 1
        roads = 0
        for road_id, road in mm.road_network.road_ids_to_object.items():
            ref_line = road.reference_line
            z_coords = road.z_coordinates
            points = [
                c
                for (x, y), z in zip(ref_line, z_coords)
                for c in (float(x), float(y), float(z))
            ]
            if " " in road_id:
                raise ValueError(f"OpenDRIVE road id {road_id!r} contains a space")
            lines.append(f"road {road_id} {len(points) // 3} {_floats(points)}")
            roads += 1
    finally:
        MapManager.reset()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return BakeStats(out_path, lanelets, roads, len(text.encode()))
