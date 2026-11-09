"""One judgment of whether a lanelet is in a junction, for every reader.

Where a sweep places a scenario (its ``is_junction`` constraint) and what a
run measures (the ODD's junction) must never disagree, so both -- and the
route search and the bindings -- ask :func:`is_junction_lanelet`.  On the
synthetic crossroads of :mod:`._route_maps`, every reader gives its answer
for every lanelet, and no module but :mod:`lanelet_junction` reads the tag
for itself.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

from autoware_carla_scenario.lanelet_junction import (
    TURN_DIRECTION_TAG,
    is_junction_lanelet,
)
from autoware_carla_scenario.odd import probes
from autoware_carla_scenario.route.geometry import MapFeatures
from autoware_carla_scenario.sweeper.constraints import IsJunctionConstraint

from . import _route_maps as rm

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "autoware_carla_scenario"


def test_every_reader_gives_the_same_answer() -> None:
    lanelet_map = rm.crossroads()
    graph = rm.routing_graph(lanelet_map)
    features = MapFeatures(lanelet_map, graph)
    constraint = IsJunctionConstraint()
    seen = set()
    for each in lanelet_map.laneletLayer:
        lanelet = cast("ConstLanelet", each)
        inside = is_junction_lanelet(lanelet)
        seen.add(inside)
        assert constraint.evaluate(lanelet) == inside
        assert probes.lanelet_in_junction(lanelet) == inside
        assert MapFeatures.is_junction(lanelet) == inside
        # The ODD's junction, read off the map as a planned route reads it.
        assert getattr(probes.junction_type, "on_lanelet")(
            lanelet, lanelet_map, graph
        ) == ("intersection" if inside else None)
        assert (
            getattr(probes.in_junction, "on_lanelet")(lanelet, lanelet_map, graph)
            == inside
        )
    # The crossroads has lanelets both in its junction and on its roads.
    assert seen == {True, False}
    assert features.lanelet(rm.J[("W", "E")]) is not None


def test_no_module_reads_the_tag_for_itself() -> None:
    """A second copy of the judgment would drift: every module asks
    :func:`is_junction_lanelet` (or names :data:`TURN_DIRECTION_TAG`)."""
    pattern = re.compile(
        r"""["']turn_direction["']\s+(not\s+)?in\b|\[\s*["']turn_direction["']\s*\]"""
    )
    found = [
        f"{path.relative_to(PACKAGE)}:{n}"
        for path in PACKAGE.rglob("*.py")
        if path.name != "lanelet_junction.py"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert found == []
    assert TURN_DIRECTION_TAG == "turn_direction"
