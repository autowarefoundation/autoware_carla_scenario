"""Every swept case starts the full ``stop_line_offset`` before its line (#2392).

A ``stop_line_offset`` binding walks a spawn back through predecessor lanelets
where the line is nearer its lanelet's start than the offset. Where the road
behind the line runs out first, the case would start short of the line, so the
binding refuses and the sweep drops the pick. The shipped sweeps that bind
``ego.spawn_s`` that way are expanded on the Nishishinjuku fixture map, as
``scenario-expand`` expands them, and each case's distance from the spawn to
the line it is about is measured along the lanes.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import lanelet2.core
import lanelet2.geometry
import pytest
import yaml
from omegaconf import OmegaConf

import autoware_carla_scenario.examples as examples_package
from autoware_carla_scenario.sweeper.bindings import StopLineOffsetBinding
from autoware_carla_scenario.sweeper.constraints import create_routing_graph
from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map
from autoware_carla_scenario.utils.stop_line import (
    _collect_stop_lines_from_reg_elems,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OSM = _REPO_ROOT / "data" / "nishishinjuku.osm"
_XODR = _REPO_ROOT / "data" / "nishishinjuku_carla.xodr"
_SCENARIO_DIR = Path(examples_package.__file__).parent / "conf" / "scenario"

#: How far a temporary_stop case's spawn may be from its 15 m, in metres:
#: the walk-back and this test measure the same centerlines.
_TOLERANCE_M = 0.05

#: How much short of its offset any swept case may start, in metres. The
#: walk-back measures back along the routing graph's first predecessor and
#: this test forward along every following lane, whose centerlines can differ
#: by a few centimetres where lanes meet; the case of #2392 was over 5 m short.
_SHORT_M = 0.5


def _offset_sweeps() -> dict[str, float]:
    """The example scenarios whose sweep binds ``ego.spawn_s`` with
    ``stop_line_offset``, and that offset."""
    found: dict[str, float] = {}
    for path in sorted(_SCENARIO_DIR.rglob("*.yaml")):
        sweep = (yaml.safe_load(path.read_text()) or {}).get("sweep") or {}
        binding = (sweep.get("bindings") or {}).get("ego.spawn_s") or {}
        if sweep.get("constraints") and binding.get("type") == "stop_line_offset":
            name = path.relative_to(_SCENARIO_DIR).with_suffix("").as_posix()
            found[name] = float(binding.get("offset", 15.0))
    return found


_SWEEPS = _offset_sweeps()


@pytest.fixture(scope="module")
def nishishinjuku() -> tuple[Any, Any]:
    lanelet_map = load_lanelet2_map(_OSM, _XODR)
    return lanelet_map, create_routing_graph(lanelet_map)


def _stop_line_s(lanelet: Any) -> float | None:
    """Arc length of *lanelet*'s stop line along its centerline, if it owns one."""
    lines = _collect_stop_lines_from_reg_elems(lanelet.regulatoryElements, set())
    if not lines:
        return None
    points = list(lines[0])
    centroid = lanelet2.core.BasicPoint2d(
        sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points)
    )
    centerline = lanelet2.geometry.to2D(lanelet.centerline)
    return float(lanelet2.geometry.toArcCoordinates(centerline, centroid).length)


def _distances_to_stop_lines(
    lanelet_map: Any, graph: Any, spawn_id: int, spawn_s: float
) -> list[float]:
    """The distance along the lanes from the spawn to the first stop line on
    each way forward, up to four lanelets on."""
    spawn = lanelet_map.laneletLayer[spawn_id]
    found: list[float] = []
    # (lanelet, distance from the spawn to the lanelet's start, steps)
    queue: deque[tuple[Any, float, int]] = deque([(spawn, -spawn_s, 0)])
    while queue:
        lanelet, start, steps = queue.popleft()
        line_s = _stop_line_s(lanelet)
        if line_s is not None and start + line_s >= 0.0:
            found.append(start + line_s)
            continue
        if steps < 4:
            end = start + lanelet2.geometry.length2d(lanelet)
            queue.extend((after, end, steps + 1) for after in graph.following(lanelet))
    return found


def _case_spawn(case: list[str]) -> tuple[int, float]:
    values = dict(override.split("=", 1) for override in case)
    return int(values["ego.spawn_lanelet_id"]), float(values["ego.spawn_s"])


def _expand(scenario: str) -> list[list[str]]:
    from autoware_carla_scenario.examples.compose_errors import compose_or_report
    from autoware_carla_scenario.registry import load_scenario_plugins
    from autoware_carla_scenario.sweeper.expand import expand_config

    load_scenario_plugins()
    cfg = compose_or_report(scenario, [])
    assert cfg is not None, f"{scenario} does not compose"
    assert OmegaConf.select(cfg, "map.lanelet2_path")
    return expand_config(cfg)


def test_the_offset_sweeps_are_found() -> None:
    assert _SWEEPS.get("temporary_stop/temporary_stop") == 15.0
    assert "intersection_passing/right_turn" in _SWEEPS


@pytest.mark.usefixtures("at_repo_root")
@pytest.mark.parametrize("scenario", sorted(_SWEEPS))
def test_no_case_starts_short_of_its_offset(
    scenario: str, nishishinjuku: tuple[Any, Any], caplog: pytest.LogCaptureFixture
) -> None:
    # The picks a binding fails on are dropped with a warning each.
    caplog.set_level(logging.ERROR)
    lanelet_map, graph = nishishinjuku
    offset = _SWEEPS[scenario]
    cases = _expand(scenario)
    assert cases and cases != [[]]
    # An exact offset for temporary_stop, the scenario of #2392; for the
    # others, none short of it.
    exact = scenario == "temporary_stop/temporary_stop"
    wrong = []
    for case in cases:
        spawn_id, spawn_s = _case_spawn(case)
        nearest = min(
            _distances_to_stop_lines(lanelet_map, graph, spawn_id, spawn_s),
            default=None,
        )
        if nearest is None or (
            abs(nearest - offset) > _TOLERANCE_M
            if exact
            else nearest < offset - _SHORT_M
        ):
            wrong.append((spawn_id, spawn_s, nearest))
    assert not wrong, (
        f"{scenario}: cases not {offset} m before a stop line "
        f"(spawn lanelet, spawn_s, distance): {wrong}"
    )


@pytest.mark.usefixtures("at_repo_root")
def test_the_temporary_stop_case_with_too_little_road_is_dropped() -> None:
    """The stop line of lanelet 580 is 1.19 m in, and the only road before it
    is lanelet 124260, 8.5 m long with nothing before it: the case used to
    start there at s=0, under 10 m short of the line."""
    spawns = [_case_spawn(case) for case in _expand("temporary_stop/temporary_stop")]
    assert spawns
    assert all(lanelet_id != 124260 for lanelet_id, _ in spawns)


def test_a_walk_back_that_runs_out_of_road_is_refused(
    nishishinjuku: tuple[Any, Any],
) -> None:
    lanelet_map, graph = nishishinjuku
    binding = StopLineOffsetBinding("ego.spawn_s", 15.0)
    with pytest.raises(ValueError, match="lanelet 124260 has no lanelet before it"):
        binding.resolve(580, lanelet_map, graph)
    # Where the road behind is long enough, the same walk-back lands 15 m short.
    result = binding.resolve(551, lanelet_map, graph)
    assert result.lanelet_id_override == 2
    assert result.value == pytest.approx(8.58, abs=0.01)


def test_at_a_fork_the_nearest_stop_line_wins_not_the_fewest_lanelets() -> None:
    """A spawn lanelet forks: straight on, one 100 m lanelet whose stop line is
    90 m in; to the right, a 5.7 m lanelet and then one whose stop line is
    1 m in.  The right branch's line is two lanelets on but under 7 m ahead,
    and the spawn is 15 m before it, not clamped to the spawn lanelet's end
    by the straight branch's line one lanelet on (#2539)."""
    from lanelet2.core import (
        AttributeMap,
        Lanelet,
        LaneletMap,
        LineString3d,
        Point3d,
        TrafficLight,
        getId,
    )

    points: dict[tuple[float, float], Any] = {}

    def point(x: float, y: float) -> Any:
        if (x, y) not in points:
            points[x, y] = Point3d(getId(), x, y, 0.0)
        return points[x, y]

    def line(*xys: tuple[float, float], kind: str = "line_thin") -> Any:
        return LineString3d(
            getId(), [point(x, y) for x, y in xys], AttributeMap({"type": kind})
        )

    def lanelet(left: Any, right: Any) -> Any:
        return Lanelet(
            getId(),
            left,
            right,
            AttributeMap(
                {
                    "type": "lanelet",
                    "subtype": "road",
                    "location": "urban",
                    "one_way": "yes",
                }
            ),
        )

    def traffic_light(on: Any, stop_line: Any, x: float, y: float) -> None:
        light = LineString3d(
            getId(),
            [Point3d(getId(), x, y, 5.0), Point3d(getId(), x + 1.0, y, 5.0)],
            AttributeMap({"type": "traffic_light"}),
        )
        on.addRegulatoryElement(
            TrafficLight(getId(), AttributeMap(), [light], stop_line)
        )

    spawn = lanelet(line((0, 1.5), (20, 1.5)), line((0, -1.5), (20, -1.5)))
    straight = lanelet(line((20, 1.5), (120, 1.5)), line((20, -1.5), (120, -1.5)))
    traffic_light(straight, line((110, 1.5), (110, -1.5), kind="stop_line"), 115, 3)
    right = lanelet(line((20, 1.5), (25, -2.0)), line((20, -1.5), (24, -5.0)))
    after_right = lanelet(line((25, -2.0), (40, -12)), line((24, -5.0), (38, -15)))
    traffic_light(
        after_right, line((25.8, -2.6), (24.8, -5.7), kind="stop_line"), 30, -20
    )
    lanelet_map = LaneletMap()
    for each in (spawn, straight, right, after_right):
        lanelet_map.add(each)
    graph = create_routing_graph(lanelet_map)
    assert {each.id for each in graph.following(spawn)} == {straight.id, right.id}

    result = StopLineOffsetBinding("ego.spawn_s", 15.0).resolve(
        spawn.id, lanelet_map, graph
    )
    assert result.lanelet_id_override is None
    # 20 m of spawn lanelet, 5.7 m of the right branch, the line ~1 m on.
    assert result.value == pytest.approx(11.74, abs=0.05)
