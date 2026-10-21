"""TeraSim's construction zone: its layout, and the action that places it (#402).

The action runs on the synthetic crossroads of :mod:`._route_maps`, standing
in the MapManager with no offset (CARLA's frame is the map's with y flipped):
the eastbound inner lane is W_IN1 (x -110..-60) then W_IN2 (x -60..-10), at
y = -1.75, 3.5 m wide.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from collections.abc import Iterator

import pytest

from autoware_carla_scenario import ConstructionSettings, ConstructionZoneAction
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions import prop as prop_module
from autoware_carla_scenario.adversity.construction import (
    BARRIER,
    CONE,
    SIGN,
    construction_layout,
    construction_zones,
)

from . import _route_maps as rm
from .test_adversity_feasible_condition import crossroads as crossroads  # noqa: F401

SHORT = ConstructionSettings(
    spacing=10.0,
    taper_length_in=10.0,
    taper_length_out=10.0,
    warning_zone_length=20.0,
    warning_zone_spacing=10.0,
    buffer_zone_length=10.0,
    termination_zone_length=10.0,
)


# -- the layout -------------------------------------------------------------------


def test_the_zones_are_terasims() -> None:
    zones = construction_zones(200.0, 400.0)
    assert zones == [
        ("warning", 100.0, 200.0),
        ("taper_in", 200.0, 260.0),
        ("buffer", 260.0, 270.0),
        ("work", 270.0, 370.0),
        ("taper_out", 370.0, 400.0),
        ("termination", 400.0, 430.0),
    ]
    # No room for a work zone: none; a warning zone stops at the lane's start.
    short = construction_zones(20.0, 60.0)
    assert [z for z, _, _ in short] == [
        "warning",
        "taper_in",
        "buffer",
        "taper_out",
        "termination",
    ]
    assert short[0] == ("warning", 0.0, 20.0)


def test_the_layout_steps_through_each_zone() -> None:
    objects = construction_layout(20.0, 70.0, 3.5, SHORT)
    by_zone = [(o.zone, o.kind, o.s) for o in objects]
    assert by_zone == [
        ("warning", SIGN, 0.0),
        ("warning", SIGN, 10.0),
        ("taper_in", CONE, 20.0),
        ("taper_in", CONE, 27.0),  # 0.7 of the spacing in a taper
        ("buffer", CONE, 30.0),
        ("buffer", CONE, 38.0),  # 0.8 in the buffer
        ("work", CONE, 40.0),
        ("work", CONE, 50.0),
        ("taper_out", CONE, 60.0),
        ("taper_out", CONE, 67.0),
        ("termination", SIGN, 70.0),
    ]


def test_the_offsets_taper_from_the_edge_to_the_work_zone() -> None:
    objects = construction_layout(20.0, 70.0, 3.5, SHORT)
    t = {(o.zone, o.s): o.t for o in objects}
    edge = -(3.5 / 2 - 0.3)  # 0.3 m inside the right edge
    assert t[("taper_in", 20.0)] == pytest.approx(edge)
    assert t[("taper_in", 27.0)] == pytest.approx(edge * 0.3)
    assert t[("work", 40.0)] == 0.0 and t[("buffer", 30.0)] == 0.0
    assert t[("taper_out", 67.0)] == pytest.approx(edge * 0.7)
    # Signs on the shoulder of the closed side, 2.5 m past the edge.
    assert t[("warning", 0.0)] == pytest.approx(-(1.75 + 2.5))


def test_the_left_side_a_curved_taper_and_a_clamped_offset() -> None:
    left = ConstructionSettings(
        **{**SHORT.__dict__, "side": "left", "taper_type": "curved"}
    )
    objects = construction_layout(20.0, 70.0, 3.5, left)
    t = {(o.zone, o.s): o.t for o in objects}
    edge = 3.5 / 2 - 0.3
    p = 0.7
    assert t[("taper_in", 27.0)] == pytest.approx(edge * (1 - (3 * p * p - 2 * p**3)))
    assert t[("warning", 0.0)] == pytest.approx(1.75 + 2.5)
    wide = ConstructionSettings(**{**SHORT.__dict__, "work_zone_offset": 9.0})
    work = [o.t for o in construction_layout(20.0, 70.0, 3.5, wide) if o.zone == "work"]
    assert work == [pytest.approx(edge)] * 2


def test_mixed_work_zones_and_dynamic_spacing() -> None:
    mixed = ConstructionSettings(**{**SHORT.__dict__, "construction_type": "mixed"})
    kinds = [
        o.kind for o in construction_layout(0.0, 100.0, 3.5, mixed) if o.zone == "work"
    ]
    assert kinds[:4] == [CONE, CONE, BARRIER, CONE]
    mutcd = ConstructionSettings(use_dynamic_spacing=True, speed_limit_mph=40.0)
    taper = [
        o.s
        for o in construction_layout(100.0, 300.0, 3.5, mutcd)
        if o.zone == "taper_in"
    ]
    assert taper[1] - taper[0] == pytest.approx(40 * 0.3048)


def test_what_the_layout_refuses() -> None:
    with pytest.raises(ValueError):
        construction_layout(50.0, 50.0, 3.5)
    with pytest.raises(ValueError):
        construction_layout(0.0, 100.0, 3.5, ConstructionSettings(spacing=0.0))


# -- the action -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_props_standing() -> Iterator[None]:
    prop_module.clear_props()
    yield
    prop_module.clear_props()


class _Prop:
    def __init__(self, blueprint: str, transform: Any) -> None:
        self.blueprint, self.transform = blueprint, transform
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class _World:
    def __init__(
        self,
        props: tuple[str, ...] = (
            "constructioncone",
            "streetbarrier",
            "warningconstruction",
        ),
    ) -> None:
        self.props = props
        self.spawned: list[_Prop] = []

    def get_blueprint_library(self) -> Any:
        def find(name: str) -> list[str]:
            return [name] if any(name.endswith(p) for p in self.props) else []

        return SimpleNamespace(filter=find)

    def get_map(self) -> Any:
        ground = SimpleNamespace(
            transform=SimpleNamespace(location=SimpleNamespace(z=0.0))
        )
        return SimpleNamespace(get_waypoint=lambda *a, **k: ground)

    def try_spawn_actor(self, blueprint: str, transform: Any) -> Optional[_Prop]:
        prop = _Prop(blueprint, transform)
        self.spawned.append(prop)
        return prop


class _Backend:
    def __init__(self) -> None:
        self.vehicles = {"in_zone": (-80.0, 1.75), "elsewhere": (-80.0, -1.75)}
        self.removed: list[str] = []
        self.closed: list[Any] = []
        self.reopened: list[str] = []

    def background_vehicles(self, world: Any) -> dict[str, tuple[float, float]]:
        return dict(self.vehicles)

    def remove_background(self, world: Any, handle: str) -> None:
        self.removed.append(handle)

    def close_lane(self, world: Any, transform: Any) -> Optional[str]:
        self.closed.append(transform)
        return "lane"

    def reopen_lane(self, world: Any, handle: str) -> None:
        self.reopened.append(handle)


def _run(action: ConstructionZoneAction, world: _World, times: list[float]) -> None:
    for t in times:
        action.tick(world, t)  # type: ignore[arg-type]


def test_the_zone_appears_at_its_start_and_is_cleared_away_at_its_end() -> None:
    world, backend = _World(), _Backend()
    action = ConstructionZoneAction(
        rm.W_IN1, 20.0, 70.0, SHORT, start_time=1.0, end_time=3.0
    )
    action.set_traffic_backend(backend)  # type: ignore[arg-type]
    _run(action, world, [0.0, 0.5])
    assert world.spawned == []
    _run(action, world, [1.0])
    assert len(world.spawned) == 11 and action.prop_count == 11
    # The first cone of the entry taper: 20 m along W_IN1, at the right edge.
    cone = world.spawned[2]
    assert cone.blueprint == "static.prop.constructioncone"
    location = cone.transform.location
    assert (location.x, location.y) == pytest.approx((-90.0, 1.75 + 1.45), abs=0.05)
    # Past W_IN1's end (50 m), on along W_IN2.
    assert world.spawned[7].transform.location.x == pytest.approx(-60.0, abs=0.05)
    # The background vehicle in the zone is gone; the lane closed once per
    # lanelet the cones stand on (W_IN1, W_IN2), on its centreline.
    assert backend.removed == ["in_zone"]
    assert [t.location.y for t in backend.closed] == pytest.approx([1.75, 1.75])
    assert backend.reopened == []
    _run(action, world, [2.0, 3.0, 3.1])
    assert all(p.destroyed for p in world.spawned) and action.prop_count == 0
    assert backend.reopened == ["lane"]
    assert action.cleared
    _run(action, world, [3.2, 3.3])
    assert action.state is ActionState.COMPLETE


def test_a_zone_for_good_is_cleared_away_when_the_run_ends() -> None:
    world = _World()
    action = ConstructionZoneAction(
        rm.W_IN1, 20.0, 70.0, SHORT, close_lane=False, clear_traffic=False
    )
    _run(action, world, [0.0, 10.0, 100.0])
    assert action.prop_count == 11 and not action.cleared
    # Its props are PropAction's, named by the zone; the runner takes them
    # away when the run ends, after closing the action.
    assert len(prop_module.standing_props()) == 11
    assert "construction_zone/taper_in/2" in prop_module.standing_props()
    action.close()
    assert action.prop_count == 0
    prop_module.clear_props()
    assert all(p.destroyed for p in world.spawned)


def test_a_missing_prop_is_skipped() -> None:
    world = _World(props=("constructioncone",))
    action = ConstructionZoneAction(rm.W_IN1, 20.0, 70.0, SHORT)
    _run(action, world, [0.0])
    assert {p.blueprint for p in world.spawned} == {"static.prop.constructioncone"}


def test_what_the_action_refuses() -> None:
    with pytest.raises(ValueError):
        ConstructionZoneAction(rm.W_IN1, 70.0, 20.0)
    with pytest.raises(ValueError):
        ConstructionZoneAction(rm.W_IN1, 20.0, 70.0, start_time=-1.0)
    with pytest.raises(ValueError):
        ConstructionZoneAction(rm.W_IN1, 20.0, 70.0, start_time=5.0, end_time=5.0)
