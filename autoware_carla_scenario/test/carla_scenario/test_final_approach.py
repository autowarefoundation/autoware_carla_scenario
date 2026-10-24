"""A vehicle whose given path ends before a dead end drives its last metres
itself (traffic.final_approach), instead of being held short of its goal."""

from __future__ import annotations

import logging
from typing import Any, List

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.traffic.final_approach import FinalApproach

from .test_routing_trajectory_action import _DT
from .test_routing_trajectory_action import _Actor as _Vehicle
from .test_traffic_backend import _Actor, _on_autopilot, _road, _stand


def test_it_drives_the_path_to_its_end_and_stops_there() -> None:
    vehicle = _Vehicle(speed=6.0)
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path)  # type: ignore[arg-type]
    assert approach.length == pytest.approx(20.0)
    elapsed = 0.0
    while elapsed < 30.0:
        approach.step(_DT, elapsed)
        vehicle.step(_DT, driven=True)
        elapsed += _DT
    assert approach.done
    assert vehicle.x == pytest.approx(20.0, abs=2.0)
    assert vehicle.speed < 0.5


class _Approach:
    """Stands in for FinalApproach: counts its steps."""

    made: List["_Approach"] = []

    def __init__(self, actor: Any, start: Any, points: List[Any]) -> None:
        self.points = points
        self.steps = 0
        self.length = 10.0
        _Approach.made.append(self)

    def step(self, dt: float, elapsed: float) -> None:
        self.steps += 1


def test_a_vehicle_on_a_path_ending_before_a_dead_end_drives_on_to_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import autoware_carla_scenario.traffic.traffic_manager as tm_module

    _Approach.made = []
    monkeypatch.setattr(tm_module, "FinalApproach", _Approach)
    _road(monkeypatch, 20.0, dead=True)
    ego = _Actor(1)
    ego.attributes = {"role_name": "Ego"}
    ego.location = carla.Location(0.0, 0.0, 0.0)
    backend, _ = _on_autopilot(ego)
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15)]
    backend._remember_path(ego, path)  # type: ignore[arg-type]
    with caplog.at_level(logging.INFO):
        _stand(backend, 5.0)
    # Off autopilot (so never destroyed), not braked: driven by the approach.
    assert ego.autopilot == [(True, 8123), (False, 8123)]
    assert ego.controls == []
    (approach,) = _Approach.made
    assert approach.points == path[1:]
    assert approach.steps > 0
    said = [r for r in caplog.records if "drives the last" in r.getMessage()]
    assert len(said) == 1 and said[0].getMessage().startswith("Ego:")


def test_a_roaming_vehicle_is_still_held(monkeypatch: pytest.MonkeyPatch) -> None:
    import autoware_carla_scenario.traffic.traffic_manager as tm_module

    _Approach.made = []
    monkeypatch.setattr(tm_module, "FinalApproach", _Approach)
    _road(monkeypatch, 20.0, dead=True)
    npc = _Actor(2)
    backend, _ = _on_autopilot(npc)
    _stand(backend, 5.0)
    assert _Approach.made == []
    assert [(c.throttle, c.brake) for c in npc.controls] == [(0.0, 1.0)]


def test_a_path_given_before_the_vehicle_is_kept_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ego is given its route at its init, before the backend starts and
    keeps it: that path is the one its final approach drives."""
    import autoware_carla_scenario.traffic.traffic_manager as tm_module
    from autoware_carla_scenario.traffic.config import TrafficManagerBackendConfig
    from autoware_carla_scenario.traffic.traffic_manager import TrafficManagerBackend

    from .test_traffic_backend import _Client, _World

    _Approach.made = []
    monkeypatch.setattr(tm_module, "FinalApproach", _Approach)
    _road(monkeypatch, 20.0, dead=True)
    ego = _Actor(1)
    ego.location = carla.Location(0.0, 0.0, 0.0)
    backend = TrafficManagerBackend(
        TrafficManagerBackendConfig(port=8123), client=_Client()
    )
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15)]
    backend._remember_path(ego, path)  # type: ignore[arg-type]
    backend.start(_World([ego]))
    _stand(backend, 5.0)
    (approach,) = _Approach.made
    assert approach.points == path[1:]
    assert ego.controls == []


def test_a_lead_out_does_not_stop_at_its_end() -> None:
    """Driving a vehicle out of a junction it hands it on at speed."""
    vehicle = _Vehicle(speed=4.0)
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path, stop=False)  # type: ignore[arg-type]
    elapsed = 0.0
    while not approach.done and elapsed < 30.0:
        approach.step(_DT, elapsed)
        vehicle.step(_DT, driven=True)
        elapsed += _DT
    assert approach.done
    assert vehicle.x == pytest.approx(19.0, abs=2.0)
    assert vehicle.speed > 2.0


class _DoneApproach(_Approach):
    def __init__(self, actor: Any, start: Any, points: List[Any], stop: bool = True) -> None:
        super().__init__(actor, start, points)
        self.stop = stop
        self.done = False

    def step(self, dt: float, elapsed: float) -> None:
        super().step(dt, elapsed)
        self.done = self.steps >= 3


def test_a_vehicle_led_out_of_its_junction_is_handed_back_with_its_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off autopilot while it is driven out of the junction's mouth, then on
    autopilot again with the rest of its path."""
    import autoware_carla_scenario.traffic.traffic_manager as tm_module

    _Approach.made = []
    monkeypatch.setattr(tm_module, "FinalApproach", _DoneApproach)
    _road(monkeypatch, 500.0, dead=False)
    npc = _Actor(4)
    npc.attributes = {"role_name": "npc4"}
    npc.location = carla.Location(0.0, 0.0, 0.0)
    backend, client = _on_autopilot(npc)
    route = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20, 25)]
    path = route[2:]
    backend._remember_path(npc, path)  # type: ignore[arg-type]
    backend._lead_out(npc, carla.Location(0.0, 0.0, 0.0), route, path)  # type: ignore[arg-type]
    (approach,) = _Approach.made
    assert approach.points == route[:3] and not approach.stop  # type: ignore[attr-defined]
    assert npc.autopilot[-1] == (False, 8123)
    npc.location = carla.Location(15.0, 0.0, 0.0)
    _stand(backend, 2.0)
    assert npc.autopilot[-1] == (True, 8123)
    assert backend._kept[4].approach is None
    assert client.tm.paths[-1] == (4, route[3:])
