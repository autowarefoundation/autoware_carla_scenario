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

from autoware_carla_scenario import (
    URBAN_CONSTRUCTION,
    ConstructionSettings,
    ConstructionZoneAction,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions import spawn as prop_module
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


# -- the urban layout (#445) -------------------------------------------------------


def _urban(**changes: Any) -> ConstructionSettings:
    return ConstructionSettings(**{**URBAN_CONSTRUCTION.__dict__, **changes})


def test_the_urban_preset_is_terasims() -> None:
    assert URBAN_CONSTRUCTION.layout == "urban"
    assert URBAN_CONSTRUCTION.spacing == 10.0
    assert URBAN_CONSTRUCTION.warning_zone_length == 200.0
    assert URBAN_CONSTRUCTION.warning_sign_offset == 2.0
    assert URBAN_CONSTRUCTION.merge_distance == 150.0
    assert URBAN_CONSTRUCTION.merge_signs == 2


def test_the_urban_zones_end_in_a_merge_taper_at_the_intersection() -> None:
    # A 300 m approach: a warning zone of 0.3 of it (not 200 m), the work zone
    # to the merge taper, the merge signs 100 m before it, no taper in,
    # buffer or termination zone.
    assert construction_zones(100.0, 400.0, URBAN_CONSTRUCTION) == [
        ("warning", 100.0, 190.0),
        ("work", 190.0, 250.0),
        ("merge_signs", 150.0, 250.0),
        ("merge_taper", 250.0, 400.0),
    ]
    # A long approach: the warning zone is capped at its length.
    long = construction_zones(0.0, 1000.0, URBAN_CONSTRUCTION)
    assert long[0] == ("warning", 0.0, 200.0)
    # No merge taper: the work zone runs to 0.9 of the approach.
    assert construction_zones(0.0, 100.0, _urban(merge_distance=0.0)) == [
        ("warning", 0.0, 30.0),
        ("work", 30.0, 90.0),
    ]
    # A merge taper longer than the approach starts at its start, and the
    # merge signs do not stand before it.
    short = construction_zones(0.0, 100.0, URBAN_CONSTRUCTION)
    assert [z for z, _, _ in short] == ["warning", "merge_signs", "merge_taper"]
    assert short[-1] == ("merge_taper", 0.0, 100.0)


def test_the_urban_layout_steps_through_each_zone() -> None:
    objects = construction_layout(0.0, 300.0, 3.5, URBAN_CONSTRUCTION)
    signs = [(o.zone, o.s) for o in objects if o.kind == SIGN]
    # At most three warning signs, 30 m apart; the merge signs 50 m apart
    # before the taper, the nearest first.
    assert signs == [
        ("warning", 0.0),
        ("warning", 30.0),
        ("warning", 60.0),
        ("merge_signs", 100.0),
        ("merge_signs", 50.0),
    ]
    work = [(o.kind, o.s) for o in objects if o.zone == "work"]
    # Every 10 m from the warning zone's end, short of the taper's start; a
    # barrier every fifth, from the first.
    assert work == [
        (BARRIER, 90.0),
        (CONE, 100.0),
        (CONE, 110.0),
        (CONE, 120.0),
        (CONE, 130.0),
        (BARRIER, 140.0),
    ]
    taper = [o.s for o in objects if o.zone == "merge_taper"]
    # Every 5 m, both ends included: the last cone at the intersection.
    assert taper == pytest.approx([150.0 + 5.0 * i for i in range(31)])
    assert {o.kind for o in objects if o.zone == "merge_taper"} == {CONE}


def test_the_urban_offsets_follow_the_curb_then_cross_the_lane() -> None:
    objects = construction_layout(0.0, 300.0, 3.5, URBAN_CONSTRUCTION)
    edge = 3.5 / 2 - 0.3
    work = [o.t for o in objects if o.zone == "work"]
    # 0.5 m inside the right edge.
    assert work == [pytest.approx(-(3.5 / 2 - 0.5))] * len(work) and work
    taper = [o.t for o in objects if o.zone == "merge_taper"]
    # From 0.3 m inside the right edge to 0.3 m inside the left: shut.
    assert taper[0] == pytest.approx(-edge)
    assert taper[15] == pytest.approx(0.0)
    assert taper[-1] == pytest.approx(edge)
    assert all(b > a for a, b in zip(taper, taper[1:]))
    # Signs 2 m out on the shoulder of the closed side.
    signs = [o.t for o in objects if o.kind == SIGN]
    assert signs == [pytest.approx(-(1.75 + 2.0))] * len(signs) and signs
    # From the left, mirrored, and a curved taper an s-curve.
    left = construction_layout(
        0.0, 300.0, 3.5, _urban(side="left", taper_type="curved")
    )
    curved = [o.t for o in left if o.zone == "merge_taper"]
    p = 3 / 30
    assert curved[0] == pytest.approx(edge) and curved[-1] == pytest.approx(-edge)
    assert curved[3] == pytest.approx(edge - 2 * edge * (3 * p * p - 2 * p**3))
    assert {o.t for o in left if o.zone == "work"} == {1.25}


def test_the_urban_work_zone_kinds_and_spacing() -> None:
    def kinds(settings: ConstructionSettings) -> list[str]:
        layout = construction_layout(0.0, 1000.0, 3.5, settings)
        return [o.kind for o in layout if o.zone == "work"][:6]

    assert kinds(_urban(construction_type="barrier")) == [BARRIER] * 6
    assert kinds(_urban(construction_type="mixed")) == [CONE, CONE, BARRIER] * 2
    taper = [
        o.s
        for o in construction_layout(0.0, 300.0, 3.5, _urban(spacing=8.0))
        if o.zone == "merge_taper"
    ]
    # Half the spacing; as TeraSim's, the last cone a whole step on, short of
    # the intersection when the taper is no whole number of steps (150 / 4).
    assert taper[1] - taper[0] == pytest.approx(4.0)
    assert taper[-1] == pytest.approx(298.0) and len(taper) == 38
    # Merge signs clamped to the approach's start stand there once.
    clamped = construction_layout(0.0, 200.0, 3.5, _urban(merge_signs=3))
    assert [o.s for o in clamped if o.zone == "merge_signs"] == [0.0]


def test_an_urban_work_zone_with_no_merge_taper_includes_its_end() -> None:
    # As TeraSim's: with nothing standing at 0.9 of the approach, the work
    # zone's last object does.
    objects = construction_layout(0.0, 100.0, 3.5, _urban(merge_distance=0.0))
    assert [o.s for o in objects if o.zone == "work"] == pytest.approx(
        [30.0 + 10.0 * i for i in range(7)]
    )
    assert {o.zone for o in objects} == {"warning", "work"}


def test_an_urban_zone_with_no_warning_zone_has_no_warning_signs() -> None:
    settings = _urban(warning_zone_length=0.0)
    zones = construction_zones(0.0, 300.0, settings)
    assert zones[0] == ("work", 0.0, 150.0)
    objects = construction_layout(0.0, 300.0, 3.5, settings)
    assert not [o for o in objects if o.zone == "warning"]


def test_the_urban_editor_action_starts_from_the_preset() -> None:
    # The editor's urban_construction_zone, at its defaults, is the preset.
    from autoware_carla_scenario.actions.base import TickTiming
    from autoware_carla_scenario.authoring import registry
    from autoware_carla_scenario.authoring._builders_generated import (
        build_urban_construction_zone_action,
    )

    spec = registry.get_action_spec("urban_construction_zone")
    assert spec is not None
    compiled = SimpleNamespace(
        params=registry.default_params(spec.fields), label="urban"
    )
    action = build_urban_construction_zone_action(
        compiled,  # type: ignore[arg-type]
        None,
        TickTiming.PRE_TICK,
        None,  # type: ignore[arg-type]
    )
    assert isinstance(action, ConstructionZoneAction)
    assert action._settings == URBAN_CONSTRUCTION


def test_what_the_urban_layout_refuses() -> None:
    with pytest.raises(ValueError):
        construction_layout(0.0, 100.0, 3.5, _urban(spacing=0.0))
    with pytest.raises(ValueError):
        construction_layout(0.0, 300.0, 3.5, _urban(warning_zone_spacing=0.0))
    with pytest.raises(ValueError):
        construction_layout(0.0, 100.0, 3.5, _urban(layout="rural"))
    with pytest.raises(ValueError):
        construction_zones(0.0, 100.0, _urban(layout="rural"))


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


def test_whole_tick_times_are_met_on_their_tick_on_a_summed_clock() -> None:
    """CARLA's clock is a running sum of the 0.05 s tick, so tick 10 reads a
    hair below 0.5 and tick 60 a hair below 3.0; the zone still appears and
    is cleared away on those ticks, not one late (#819)."""
    clock, now = [], 0.0
    for _ in range(70):
        clock.append(now)
        now += 0.05
    assert clock[10] < 0.5 and clock[60] < 3.0
    world = _World()
    action = ConstructionZoneAction(
        rm.W_IN1,
        20.0,
        70.0,
        SHORT,
        start_time=0.5,
        end_time=3.0,
        close_lane=False,
        clear_traffic=False,
    )
    placed: Optional[int] = None
    cleared: Optional[int] = None
    for tick, t in enumerate(clock):
        action.tick(world, t)  # type: ignore[arg-type]
        if placed is None and action.prop_count > 0:
            placed = tick
        if cleared is None and action.cleared:
            cleared = tick
    assert (placed, cleared) == (10, 60)


def test_a_zone_for_good_is_cleared_away_when_the_run_ends() -> None:
    world = _World()
    action = ConstructionZoneAction(
        rm.W_IN1, 20.0, 70.0, SHORT, close_lane=False, clear_traffic=False
    )
    _run(action, world, [0.0, 10.0, 100.0])
    assert action.prop_count == 11 and not action.cleared
    # Its props are SpawnAction's, named by the zone; the runner takes them
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


def test_an_urban_zone_is_placed_by_the_same_action() -> None:
    world, backend = _World(), _Backend()
    urban = _urban(merge_distance=40.0, warning_zone_length=20.0)
    action = ConstructionZoneAction(rm.W_IN1, 0.0, 100.0, urban)
    action.set_traffic_backend(backend)  # type: ignore[arg-type]
    _run(action, world, [0.0])
    expected = construction_layout(0.0, 100.0, 3.5, urban)
    assert len(world.spawned) == len(expected) == action.prop_count
    # The last cone of the merge taper, at W_IN2's end, 0.3 m inside the
    # left edge: the lane is shut before the intersection.
    last = world.spawned[-1].transform.location
    assert (last.x, last.y) == pytest.approx((-10.0, 1.75 - 1.45), abs=0.05)
    assert "construction_zone/merge_taper/" + str(len(expected) - 1) in (
        prop_module.standing_props()
    )
    assert backend.removed == ["in_zone"] and len(backend.closed) == 2


def test_what_the_action_refuses() -> None:
    with pytest.raises(ValueError):
        ConstructionZoneAction(rm.W_IN1, 70.0, 20.0)
    with pytest.raises(ValueError):
        ConstructionZoneAction(rm.W_IN1, 20.0, 70.0, start_time=-1.0)
    with pytest.raises(ValueError):
        ConstructionZoneAction(rm.W_IN1, 20.0, 70.0, start_time=5.0, end_time=5.0)
