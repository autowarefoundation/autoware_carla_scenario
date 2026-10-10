"""Scenario measures (:mod:`autoware_carla_scenario.measures`) and the ODD's mapping onto them."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from autoware_carla_scenario import BaseScenario, EgoConfig
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.entity._spawn import SpawnPointIndex
from autoware_carla_scenario.measures import (
    BUILT_IN_MEASURES,
    CROSSING_PEDESTRIAN_GAP_M,
    CROSSING_PEDESTRIAN_SPEED_MS,
    VEHICLE_AHEAD_GAP_M,
    VEHICLE_AHEAD_RELATIVE_SPEED_KPH,
    measured_scenario,
    read_measure,
    set_measured_scenario,
)
from autoware_carla_scenario.odd import (
    OddAttribute,
    ScenarioMeasure,
    default_odd,
    load_odd_binding,
    scenario_measure,
)


class _Vec(SimpleNamespace):
    pass


class _Placed:
    """A road user with a heading: where it is, which way it faces, how fast."""

    def __init__(
        self,
        actor_id: int,
        type_id: str,
        at: tuple[float, float],
        velocity: tuple[float, float] = (0.0, 0.0),
        yaw: float = 0.0,
        role: str = "",
    ) -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role}
        self._at = _Vec(x=at[0], y=at[1], z=0.0)
        self._v = _Vec(x=velocity[0], y=velocity[1], z=0.0)
        self._yaw = yaw

    def get_location(self) -> _Vec:
        return self._at

    def get_velocity(self) -> _Vec:
        return self._v

    def get_transform(self) -> Any:
        return SimpleNamespace(
            location=self._at, rotation=SimpleNamespace(yaw=self._yaw)
        )


class _World:
    def __init__(self, actors: list[_Placed], frame: int = 1) -> None:
        self._actors = actors
        self.frame = frame

    def get_snapshot(self) -> Any:
        return SimpleNamespace(frame=self.frame)

    def get_actors(self) -> list[_Placed]:
        return self._actors


_FRAMES = iter(range(1, 1_000_000))


def _world_of(*others: _Placed, yaw: float = 0.0, ego_speed: float = 10.0) -> Any:
    heading = math.radians(yaw)
    ego = _Placed(
        1,
        "vehicle.ego",
        (0.0, 0.0),
        (ego_speed * math.cos(heading), ego_speed * math.sin(heading)),
        yaw=yaw,
        role=str(EGO_ROLE_NAME),
    )
    # A frame of its own, so what an earlier world measured is not reused.
    return _World([ego, *others], frame=next(_FRAMES))


@pytest.fixture(autouse=True)
def _no_scenario() -> Iterator[None]:
    set_measured_scenario(None)
    yield
    set_measured_scenario(None)


class _Scenario(BaseScenario):
    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return True


def _scenario() -> _Scenario:
    return _Scenario(
        EgoConfig(vehicle_type="vehicle.x", spawn_location=SpawnPointIndex(0))
    )


# ---------------------------------------------------------------------------
# The built-in measures, in the ego's frame
# ---------------------------------------------------------------------------


def test_the_vehicle_ahead_in_the_next_lane_is_measured() -> None:
    world = _world_of(
        _Placed(2, "vehicle.npc", (12.0, 3.5), (12.0, 0.0)),  # next lane, 12 m ahead
        _Placed(3, "vehicle.far", (40.0, 0.0)),  # own lane, further
        _Placed(4, "vehicle.behind", (-5.0, 0.0)),
        _Placed(5, "vehicle.two_lanes_over", (8.0, 7.5)),
    )

    assert read_measure(VEHICLE_AHEAD_GAP_M, world) == pytest.approx(12.0)
    # 12 m/s against the ego's 10: 2 m/s faster.
    assert read_measure(VEHICLE_AHEAD_RELATIVE_SPEED_KPH, world) == pytest.approx(7.2)


def test_ahead_follows_the_ego_heading() -> None:
    # Facing +y: a vehicle at (0, 20) is ahead, one at (20, 0) is beside.
    world = _world_of(
        _Placed(2, "vehicle.ahead", (0.0, 20.0)),
        _Placed(3, "vehicle.beside", (20.0, 0.0)),
        yaw=90.0,
    )

    assert read_measure(VEHICLE_AHEAD_GAP_M, world) == pytest.approx(20.0)


def test_a_pedestrian_counts_once_it_sets_off() -> None:
    standing = _world_of(_Placed(2, "walker.pedestrian.1", (20.0, -3.0)))
    running = _world_of(_Placed(2, "walker.pedestrian.1", (20.0, -3.0), (0.0, 2.0)))

    assert read_measure(CROSSING_PEDESTRIAN_GAP_M, standing) is None
    assert read_measure(CROSSING_PEDESTRIAN_GAP_M, running) == pytest.approx(20.0)
    assert read_measure(CROSSING_PEDESTRIAN_SPEED_MS, running) == pytest.approx(2.0)


def test_nothing_there_is_missing() -> None:
    world = _world_of()

    for key in BUILT_IN_MEASURES:
        assert read_measure(key, world) is None
    assert read_measure("no_such_measure", world) is None


# ---------------------------------------------------------------------------
# A scenario's measures
# ---------------------------------------------------------------------------


def test_every_scenario_has_the_built_in_measures() -> None:
    assert set(_scenario().measures) == set(BUILT_IN_MEASURES)


def test_a_scenario_replaces_a_measure_and_the_odd_reads_its_own() -> None:
    """The scenario knows which actor is its cut-in vehicle; the ODD need not."""
    scenario = _scenario()
    scenario.register_measure(VEHICLE_AHEAD_GAP_M, lambda world: 7.5)
    scenario.register_measure("cut_in_after_s", lambda world: 2.0, unit="s")
    world = _world_of(_Placed(2, "vehicle.npc", (12.0, 3.5)))
    probe = scenario_measure(VEHICLE_AHEAD_GAP_M)

    assert probe(world) == pytest.approx(12.0)  # no scenario running: built-in
    set_measured_scenario(scenario)
    assert measured_scenario() is scenario
    assert probe(world) == 7.5
    assert scenario_measure("cut_in_after_s")(world) == 2.0
    assert scenario.measures[VEHICLE_AHEAD_GAP_M].unit == "m"  # the built-in's


def test_a_replacement_keeps_the_built_in_unit() -> None:
    with pytest.raises(ValueError, match="'m', not 'ft'"):
        _scenario().register_measure(VEHICLE_AHEAD_GAP_M, lambda w: 1.0, unit="ft")


def test_a_measure_that_raises_reads_as_missing() -> None:
    scenario = _scenario()

    def broken(world: Any) -> float:
        raise RuntimeError("boom")

    scenario.register_measure("broken", broken)
    set_measured_scenario(scenario)

    assert scenario_measure("broken")(_world_of()) is None


# ---------------------------------------------------------------------------
# The ODD's mapping onto measures
# ---------------------------------------------------------------------------


def test_the_default_odd_maps_its_dynamic_elements_onto_measures() -> None:
    probes = {a.name: a.probe for a in default_odd().attributes}

    assert probes["dynamic.vehicle_ahead_gap"] == scenario_measure(VEHICLE_AHEAD_GAP_M)
    assert isinstance(probes["dynamic.crossing_pedestrian_speed"], ScenarioMeasure)
    assert scenario_measure(VEHICLE_AHEAD_GAP_M).unit == "m"
    assert scenario_measure("its_own").unit == ""


def test_a_binding_file_maps_a_taxonomy_concept_onto_a_measure(tmp_path: Path) -> None:
    (tmp_path / "taxonomy.yml").write_text(
        "TAXONOMY:\n" "  dynamic_elements:\n" "    headway: float length\n"
    )
    binding = tmp_path / "gap.binding.yaml"
    binding.write_text(
        "openodd: [taxonomy.yml]\n"
        "name: gap\n"
        "probes:\n"
        "  headway: {measure: vehicle_ahead_gap_m, buckets: [0, 10, 20]}\n"
    )

    odd = load_odd_binding(binding)

    (attribute,) = [a for a in odd.attributes if a.name.endswith("headway")]
    assert isinstance(attribute, OddAttribute)
    assert attribute.probe == scenario_measure(VEHICLE_AHEAD_GAP_M)
    assert attribute.unit == "m"
