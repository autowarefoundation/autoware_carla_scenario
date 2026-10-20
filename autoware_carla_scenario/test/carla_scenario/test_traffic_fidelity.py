"""Tests for level of detail in traffic backends: the entity's flag, the
policy that moves vehicles between levels, and the mesoscopic queue model.

None of them needs a simulator; the SUMO backend's use of all three is in
``test_traffic_sumo.py``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pytest

from autoware_carla_scenario.entity import EgoVehicle, VehicleEntity
from autoware_carla_scenario.entity._spawn import SpawnPointIndex
from autoware_carla_scenario.entity.vehicle_entity import VehicleEntityConfig
from autoware_carla_scenario.traffic import (
    DrivenEntity,
    NullTrafficBackend,
    SimulationFidelity,
    TrafficBackend,
)
from autoware_carla_scenario.traffic.fidelity import (
    FidelityCandidate,
    MesoscopicConfig,
    plan_fidelity,
)
from autoware_carla_scenario.traffic.mesoscopic import MesoLink, MesoscopicTraffic
from autoware_carla_scenario.traffic.sumo.config import SumoBackendConfig

MICRO = SimulationFidelity.MICROSCOPIC
MESO = SimulationFidelity.MESOSCOPIC


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


def test_a_vehicle_is_microscopic_until_a_backend_says_otherwise() -> None:
    entity = DrivenEntity()
    backend = NullTrafficBackend()
    assert entity.simulation_fidelity is MICRO
    assert backend.fidelity_of(entity) is MICRO
    assert backend.require_microscopic(entity, world=None)  # type: ignore[arg-type]
    assert backend.fidelity_census() == {}

    backend._mark_fidelity(entity, MESO)
    assert backend.fidelity_of(entity) is MESO
    # A backend with one level cannot bring it back, and says so.
    assert not TrafficBackend.require_microscopic(backend, entity, None)  # type: ignore[arg-type]
    # The flag is the entity's own, not the class's.
    assert DrivenEntity().simulation_fidelity is MICRO


def test_an_npc_lets_a_backend_take_its_actor_and_an_ego_does_not() -> None:
    npc = VehicleEntity(VehicleEntityConfig("npc", SpawnPointIndex(0)))
    npc._vehicle = actor = object()  # type: ignore[assignment]
    npc._lane_change_target = (1, -1)
    assert npc._replace_actor(None)
    assert npc.actor is None
    assert npc._lane_change_target is None  # the old actor's manoeuvre
    assert npc._replace_actor(actor)  # type: ignore[arg-type]
    assert npc.actor is actor

    ego = EgoVehicle()
    assert not ego._replace_actor(None)
    assert not DrivenEntity()._replace_actor(None)


def test_an_entity_asks_its_backend_to_be_microscopic() -> None:
    asked: list[Any] = []

    class Backend(TrafficBackend):
        name = "fake"

        def require_microscopic(self, entity: DrivenEntity, world: Any) -> bool:
            asked.append(entity)
            return True

    npc = VehicleEntity(VehicleEntityConfig("npc", SpawnPointIndex(0)))
    npc.set_traffic_backend(Backend())
    assert npc.require_microscopic(None)  # type: ignore[arg-type]
    assert asked == [npc]


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def test_the_policy_changes_level_with_hysteresis() -> None:
    config = MesoscopicConfig(enabled=True, micro_radius_m=100.0, meso_radius_m=150.0)
    assert config.decide(MICRO, 120.0) is MICRO  # in the band: kept
    assert config.decide(MICRO, 151.0) is MESO
    assert config.decide(MESO, 120.0) is MESO  # in the band: kept
    assert config.decide(MESO, 100.0) is MICRO
    assert MesoscopicConfig(micro_radius_m=80.0).demote_radius_m == 100.0


@pytest.mark.parametrize(
    "options",
    [
        {"micro_radius_m": 0.0},
        {"micro_radius_m": 100.0, "meso_radius_m": 50.0},
        {"update_interval_s": -1.0},
        {"headway_s": 0.0},
        {"jam_spacing_m": -1.0},
    ],
)
def test_the_policy_refuses_impossible_options(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MesoscopicConfig(**options)


def test_the_policy_is_an_option_of_the_sumo_backend() -> None:
    config = SumoBackendConfig.from_mapping(
        {"mesoscopic": {"enabled": True, "micro_radius_m": 50.0}}
    )
    assert config.mesoscopic == MesoscopicConfig(enabled=True, micro_radius_m=50.0)
    assert not SumoBackendConfig().mesoscopic.enabled
    with pytest.raises((TypeError, ValueError)):
        SumoBackendConfig.from_mapping({"mesoscopic": {"radius": 1.0}})


def test_the_plan_changes_the_ones_that_matter_most_first() -> None:
    config = MesoscopicConfig(enabled=True, micro_radius_m=10.0, meso_radius_m=20.0)
    candidates = [
        FidelityCandidate("near_micro", MICRO, 5.0, 0.0),
        FidelityCandidate("far_micro", MICRO, 100.0, 0.0),
        FidelityCandidate("farther_micro", MICRO, 0.0, -300.0),
        FidelityCandidate("near_meso", MESO, 9.0, 0.0),
        FidelityCandidate("nearer_meso", MESO, 0.0, 1.0),
        FidelityCandidate("far_meso", MESO, 500.0, 0.0),
    ]
    to_meso, to_micro = plan_fidelity(config, candidates, [(0.0, 0.0), (1000.0, 0.0)])
    assert to_meso == ["farther_micro", "far_micro"]
    assert to_micro == ["nearer_meso", "near_meso"]
    # Nothing to measure from, or switched off: nothing changes.
    assert plan_fidelity(config, candidates, []) == ([], [])
    config.enabled = False
    assert plan_fidelity(config, candidates, [(0.0, 0.0)]) == ([], [])


# ---------------------------------------------------------------------------
# The queue model
# ---------------------------------------------------------------------------


def _model(
    *, lanes: int = 1, gate: Optional[Any] = None, length: float = 100.0
) -> MesoscopicTraffic:
    links = {name: MesoLink(name, length, lanes, 10.0) for name in ("a", "b", "c")}
    return MesoscopicTraffic(links, headway_s=2.0, jam_spacing_m=10.0, gate=gate)


def _run(model: MesoscopicTraffic, until: float, dt: float = 0.1) -> list[str]:
    arrived = []
    t = 0.0
    while t <= until + 1e-9:
        arrived += [v.id for v in model.step(t)]
        t += dt
    return arrived


def test_a_free_vehicle_drives_each_road_at_its_speed() -> None:
    model = _model()
    model.add("v", ["a", "b"], now=0.0, offset_m=20.0)
    assert model.position("v", 5.0) == ("a", pytest.approx(70.0))
    assert model.speed("v", 5.0) == pytest.approx(10.0)
    _run(model, 8.5)
    assert model.vehicle("v").link == "b"  # 80 m at 10 m/s: on b at 8 s
    assert model.position("v", 13.0) == ("b", pytest.approx(50.0, abs=1.0))
    assert _run(model, 20.0) == ["v"]
    assert "v" not in model and len(model) == 0


def test_a_vehicle_drives_no_faster_than_itself_or_its_cap() -> None:
    model = _model()
    model.add("slow", ["a"], now=0.0, max_speed=5.0)
    assert model.vehicle("slow").exit_s == pytest.approx(20.0)
    model.set_speed_cap("slow", 0.0, now=10.0)
    assert model.position("slow", 30.0) == ("a", pytest.approx(50.0))
    assert model.speed("slow", 30.0) == 0.0
    assert _run(model, 60.0) == []  # told to stand, it stands
    model.set_speed_cap("slow", None, now=60.0)
    assert model.vehicle("slow").exit_s == pytest.approx(70.0)


def test_a_road_lets_one_vehicle_out_per_headway_per_lane() -> None:
    for lanes, gap in ((1, 2.0), (2, 1.0)):
        model = _model(lanes=lanes)
        for i in range(3):
            model.add(f"v{i}", ["a", "b", "c"], now=0.0, offset_m=95.0)
        entered: dict[str, float] = {}
        t = 0.0
        while len(entered) < 3:
            model.step(t)
            for i in range(3):
                if model.vehicle(f"v{i}").link == "b":
                    entered.setdefault(f"v{i}", t)
            t = round(t + 0.1, 6)
        # First in, first out, a headway per lane apart.
        assert entered["v0"] < entered["v1"] < entered["v2"]
        assert entered["v1"] - entered["v0"] == pytest.approx(gap, abs=0.11)


def test_a_full_road_holds_the_queue_behind_it() -> None:
    # 30 m at 10 m of jam spacing: three vehicles fill b.
    links = {
        "a": MesoLink("a", 100.0, 1, 10.0),
        "b": MesoLink("b", 30.0, 1, 10.0),
    }
    model = MesoscopicTraffic(links, headway_s=0.1, jam_spacing_m=10.0)
    for i in range(3):
        model.add(f"b{i}", ["b"], now=0.0, offset_m=30.0, speed_cap=0.0)
    model.add("v", ["a", "b"], now=0.0, offset_m=90.0)
    _run(model, 5.0)
    assert model.vehicle("v").link == "a"
    assert model.position("v", 5.0) == ("a", pytest.approx(100.0))
    assert model.speed("v", 5.0) == 0.0
    # Its queue backs up behind it: the second is a jam spacing short.
    model.add("w", ["a", "b"], now=5.0, offset_m=0.0)
    assert model.position("w", 30.0) == ("a", pytest.approx(90.0))
    model.remove("b0")
    _run(model, 6.0)
    assert model.vehicle("v").link == "b"


def test_a_red_light_holds_a_vehicle_at_the_end_of_its_road() -> None:
    green = {"value": False}
    model = _model(gate=lambda a, b: green["value"])
    model.add("v", ["a", "b"], now=0.0, offset_m=90.0)
    _run(model, 10.0)
    assert model.vehicle("v").link == "a"
    green["value"] = True
    model.step(10.0)
    assert model.vehicle("v").link == "b"


def test_a_vehicle_turns_where_it_is_told_to() -> None:
    model = _model()
    model.add("v", ["a", "b"], now=0.0)
    model.set_route("v", ["a", "c"])
    assert model.vehicle("v").remaining_route == ["a", "c"]
    with pytest.raises(ValueError):
        model.set_route("v", ["b", "c"])
    _run(model, 11.0)
    assert model.vehicle("v").link == "c"


def test_a_route_leaving_the_model_arrives_where_it_leaves() -> None:
    model = _model()
    model.add("v", ["a", "elsewhere"], now=0.0, offset_m=95.0)
    assert [v.id for v in model.step(1.0)] == ["v"]


def test_the_model_refuses_what_it_cannot_hold() -> None:
    model = _model()
    with pytest.raises(ValueError):
        model.add("v", [], now=0.0)
    with pytest.raises(ValueError):
        model.add("v", ["nowhere"], now=0.0)
    model.add("v", ["a"], now=0.0)
    with pytest.raises(ValueError):
        model.add("v", ["a"], now=0.0)
    assert model.remove("w") is None
    assert model.remove("v") is not None
    assert list(model) == []
    assert _run(model, 1.0) == []
    model.add("x", ["a"], now=0.0, max_speed=0.0)
    assert math.isinf(model.vehicle("x").exit_s)
