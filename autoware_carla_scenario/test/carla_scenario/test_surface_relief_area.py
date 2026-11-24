"""The raised and grooved parts of a road's surface as an area: OpenCRG grids
read off the OpenDRIVE map (maps.opencrg), and EntityAreaDistanceCondition
measuring to them."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import AreaGround, EntityAreaDistanceCondition, MapArea
from autoware_carla_scenario.conditions import map_area_from_fields
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import OpenDrivePose
from autoware_carla_scenario.coordinate.transform import to_carla_world
from autoware_carla_scenario.maps.opencrg import read_crg, relief_regions

_DATA = Path(__file__).resolve().parents[3] / "data"

#: A road of the fixture map, 83.7 m long.
_ROAD = "80"


def _crg_text(
    s_start: float,
    s_end: float,
    v_right: float,
    v_left: float,
    value: Callable[[float, float], float],
    step: float = 0.05,
    data_format: str = "LRFI",
) -> str:
    """An OpenCRG ASCII file as roadgen writes it: a value per grid point,
    ten characters wide, eight to a record."""
    rows = round((s_end - s_start) / step) + 1
    columns = round((v_left - v_right) / step) + 1
    lines = [
        "$CT",
        "test surface",
        "$",
        "$ROAD_CRG",
        f"REFERENCE_LINE_START_U   = {s_start:.6f}",
        f"REFERENCE_LINE_END_U     = {s_end:.6f}",
        f"REFERENCE_LINE_INCREMENT = {step:.6f}",
        f"LONG_SECTION_V_RIGHT     = {v_right:.6f}",
        f"LONG_SECTION_V_LEFT      = {v_left:.6f}",
        f"LONG_SECTION_V_INCREMENT = {step:.6f}",
        "$",
        "$KD_Definition",
        f"#:{data_format}",
        f"U:reference line u,m,{s_start:.6f},{step:.6f}",
        *[f"D:long section {k + 1},m" for k in range(columns)],
        "$",
        "$$$$$$$$10$$$$$$$$20$$$$$$$$30$$$$$$$$40$$$$$$$$50$$$$$$$$60$$$$$$$$70$$$$$$$$80",
    ]
    for row in range(rows):
        u = s_start + row * step
        record = ""
        for column in range(columns):
            record += f"{value(u, v_right + column * step):10.5f}"
            if (column + 1) % 8 == 0 or column + 1 == columns:
                lines.append(record)
                record = ""
    return "\n".join(lines) + "\n"


def _hump_and_groove(u: float, v: float) -> float:
    """A 6 cm hump across s 20..24, t -1..1, and a 3 cm groove at s 40..42,
    t 1..2."""
    if 20.0 <= u <= 24.0 and -1.0 <= v <= 1.0:
        return 0.06
    if 40.0 <= u <= 42.0 and 1.0 <= v <= 2.0:
        return -0.03
    return 0.0


def _xodr(folder: Path, crg: str, **attributes: str) -> Path:
    """A minimal OpenDRIVE file whose road refers to *crg*."""
    attrs = {
        "file": crg,
        "sStart": "15",
        "sEnd": "45",
        "orientation": "same",
        "mode": "attached",
        "purpose": "elevation",
        **attributes,
    }
    text = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    path = folder / "map.xodr"
    path.write_text(
        f'<OpenDRIVE><road id="{_ROAD}" length="83.7"><surface><CRG {text}/>'
        "</surface></road></OpenDRIVE>"
    )
    return path


@pytest.fixture
def surface(tmp_path: Path) -> Path:
    (tmp_path / "road.crg").write_text(
        _crg_text(15.0, 45.0, -2.0, 2.5, _hump_and_groove)
    )
    return _xodr(tmp_path, "road.crg")


def _bounds(outline: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    s = [p[0] for p in outline]
    t = [p[1] for p in outline]
    return (min(s), max(s), min(t), max(t))


class TestReading:
    def test_the_grid_is_read_as_written(self, tmp_path: Path) -> None:
        path = tmp_path / "g.crg"
        path.write_text(_crg_text(15.0, 45.0, -2.0, 2.5, _hump_and_groove))
        grid = read_crg(path)
        assert len(grid.u) == 601 and len(grid.v) == 91
        assert grid.u[0] == pytest.approx(15.0)
        assert grid.v[0] == pytest.approx(-2.0)
        assert grid.values[round(22.0 / 0.05) - 300][40] == pytest.approx(0.06)

    def test_a_raised_part_and_a_grooved_part_are_two_regions(
        self, surface: Path
    ) -> None:
        raised, problems = relief_regions(surface, grooved=False)
        assert problems == []
        (hump,) = raised
        assert hump.road_id == _ROAD and hump.kind == "raised"
        assert hump.extreme_m == pytest.approx(0.06)
        s0, s1, t0, t1 = _bounds(hump.outline)
        assert (s0, s1) == (pytest.approx(19.975), pytest.approx(24.025))
        assert (t0, t1) == (pytest.approx(-1.025), pytest.approx(1.025))
        (groove,) = relief_regions(surface, raised=False)[0]
        assert groove.kind == "grooved" and groove.extreme_m == pytest.approx(-0.03)
        assert len(relief_regions(surface)[0]) == 2

    def test_what_stands_off_less_than_the_threshold_does_not_count(
        self, surface: Path
    ) -> None:
        assert [r.kind for r in relief_regions(surface, threshold_m=0.05)[0]] == [
            "raised"
        ]
        assert relief_regions(surface, threshold_m=0.1)[0] == []

    def test_an_opposite_grid_runs_back_from_the_end_mirrored(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "road.crg").write_text(
            _crg_text(15.0, 45.0, -2.0, 2.5, _hump_and_groove)
        )
        xodr = _xodr(tmp_path, "road.crg", orientation="opposite")
        (hump,) = relief_regions(xodr, grooved=False)[0]
        s0, s1, t0, t1 = _bounds(hump.outline)
        # u 20..24 from the grid's start is s 45 - (5..9) on the road.
        assert (s0, s1) == (pytest.approx(35.975), pytest.approx(40.025))
        assert (t0, t1) == (pytest.approx(-1.025), pytest.approx(1.025))

    def test_a_region_crossed_twice_is_outlined_across_its_gap(
        self, tmp_path: Path
    ) -> None:
        def u_shape(u: float, v: float) -> float:
            sides = 20.0 <= u <= 24.0 and (-1.0 <= v <= -0.5 or 0.5 <= v <= 1.0)
            bottom = 20.0 <= u <= 20.5 and -1.0 <= v <= 1.0
            return 0.05 if sides or bottom else 0.0

        (tmp_path / "road.crg").write_text(_crg_text(15.0, 45.0, -2.0, 2.5, u_shape))
        (region,) = relief_regions(_xodr(tmp_path, "road.crg"))[0]
        _s0, _s1, t0, t1 = _bounds(region.outline)
        assert (t0, t1) == (pytest.approx(-1.025), pytest.approx(1.025))

    def test_a_friction_grid_says_nothing_about_height(self, tmp_path: Path) -> None:
        (tmp_path / "road.crg").write_text(
            _crg_text(15.0, 45.0, -2.0, 2.5, _hump_and_groove)
        )
        xodr = _xodr(tmp_path, "road.crg", purpose="friction")
        assert relief_regions(xodr) == ([], [])

    @pytest.mark.parametrize(
        ("attributes", "said"),
        [
            ({"mode": "genuine"}, "genuine mode"),
            ({"file": "missing.crg"}, "missing.crg"),
        ],
    )
    def test_what_cannot_be_read_is_said_and_passed_over(
        self, tmp_path: Path, attributes: dict[str, str], said: str
    ) -> None:
        (tmp_path / "road.crg").write_text(
            _crg_text(15.0, 45.0, -2.0, 2.5, _hump_and_groove)
        )
        regions, problems = relief_regions(_xodr(tmp_path, "road.crg", **attributes))
        assert regions == []
        assert len(problems) == 1 and said in problems[0]

    def test_a_binary_grid_is_said(self, tmp_path: Path) -> None:
        text = _crg_text(15.0, 16.0, -1.0, 1.0, _hump_and_groove)
        (tmp_path / "road.crg").write_text(text.replace("#:LRFI", "#:KRBI"))
        _regions, (problem,) = relief_regions(_xodr(tmp_path, "road.crg"))
        assert "'KRBI' is not an OpenCRG ASCII format" in problem


class TestMapArea:
    def test_a_surface_ground_takes_a_threshold(self) -> None:
        area = map_area_from_fields("SURFACE_RAISED", relief_m=0.02)
        assert area == MapArea(AreaGround.SURFACE_RAISED, relief_m=0.02)
        assert area.describe() == "raised road surface (0.02 m and more)"

    @pytest.mark.parametrize(
        ("area", "said"),
        [
            (MapArea(AreaGround.SURFACE_RELIEF, lanelet_ids=(12,)), "pick no lanelets"),
            (
                MapArea(AreaGround.SURFACE_GROOVED, attributes=(("a", "b"),)),
                "no tags",
            ),
            (MapArea(AreaGround.SURFACE_RAISED, relief_m=0.0), "must be positive"),
        ],
    )
    def test_what_no_map_could_hold_is_refused(self, area: MapArea, said: str) -> None:
        with pytest.raises(ValueError, match=said):
            area.validate()


# ---------------------------------------------------------------------------
# On the fixture map, with a surface written onto one of its roads
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def surfaced_map(tmp_path_factory: pytest.TempPathFactory) -> Iterator[MapManager]:
    """The fixture map, its road 80 given a CRG grid with a hump and a groove."""
    folder = tmp_path_factory.mktemp("surfaced")
    (folder / "road_80.crg").write_text(
        _crg_text(15.0, 45.0, -2.0, 2.5, _hump_and_groove)
    )
    text = (_DATA / "nishishinjuku_carla.xodr").read_text()
    start = re.search(r'<road [^>]*id="80"[^>]*>', text)
    assert start is not None
    end = text.index("</road>", start.end())
    surface = (
        '<surface><CRG file="road_80.crg" sStart="15" sEnd="45" '
        'orientation="same" mode="attached" purpose="elevation"/></surface>'
    )
    (folder / "map.xodr").write_text(text[:end] + surface + text[end:])
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(folder / "map.xodr", _DATA / "nishishinjuku.osm")
    yield manager
    MapManager._instance = saved


@pytest.fixture
def run_map(surfaced_map: MapManager) -> Iterator[MapManager]:
    saved = MapManager._instance
    MapManager._instance = surfaced_map
    yield surfaced_map
    MapManager._instance = saved


def _ego_on_road(s: float) -> MagicMock:
    """A 4 m by 2 m vehicle on road 80's reference line at *s*, along it."""
    pose = to_carla_world(OpenDrivePose(_ROAD, 0, s, 0.0))
    actor = MagicMock()
    actor.attributes = {"role_name": "Ego"}
    transform = carla.Transform(
        carla.Location(pose.x, pose.y, pose.z), carla.Rotation(yaw=pose.yaw)
    )
    actor.get_transform.return_value = transform
    actor.get_location.return_value = transform.location
    actor.bounding_box = carla.BoundingBox(
        carla.Location(0.0, 0.0, 0.0), carla.Vector3D(2.0, 1.0, 0.8)
    )
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


class TestOnTheMap:
    def test_it_measures_to_the_hump_along_the_road(self, run_map: MapManager) -> None:
        condition = EntityAreaDistanceCondition(
            entity_name="Ego",
            area=MapArea(AreaGround.SURFACE_RAISED),
            value=1000.0,
            label="near the hump",
        )
        assert condition.problem == ""
        result = condition.check(_world(_ego_on_road(10.0)), 1.0)
        assert result is not None
        # The body's front is at s 12, the hump's cells begin at s 19.975; the
        # road bends a little, so the plan distance is within a few cm of it.
        assert condition.closest_m == pytest.approx(7.975, abs=0.1)
        details = condition.get_details()
        assert details["closest"] == "raised 0.060 m on road 80"
        assert "closest_surface" not in details
        assert details["area"] == "raised road surface (0.01 m and more)"

    def test_over_the_hump_is_zero_and_the_groove_is_another_ground(
        self, run_map: MapManager
    ) -> None:
        raised = EntityAreaDistanceCondition(
            entity_name="Ego",
            area=MapArea(AreaGround.SURFACE_RAISED),
            value=0.1,
            label="on the hump",
        )
        grooved = EntityAreaDistanceCondition(
            entity_name="Ego",
            area=MapArea(AreaGround.SURFACE_GROOVED),
            value=0.1,
            label="in the groove",
        )
        world = _world(_ego_on_road(22.0))
        assert raised.check(world, 1.0) is not None
        assert raised.closest_m == 0.0
        assert grooved.check(world, 1.0) is None
        assert grooved.closest_m is not None and grooved.closest_m > 15.0
        assert math.isfinite(grooved.closest_m)

    def test_a_map_with_no_surface_is_said_and_never_holds(self) -> None:
        saved = MapManager._instance
        MapManager.reset()
        manager = MapManager.get_instance()
        manager.initialize(
            _DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm"
        )
        try:
            condition = EntityAreaDistanceCondition(
                entity_name="Ego",
                area=MapArea(AreaGround.SURFACE_RELIEF),
                value=1.0,
                label="relief",
            )
            assert condition.problem == (
                "no raised or grooved road surface (0.01 m and more) on the map"
            )
        finally:
            MapManager._instance = saved
