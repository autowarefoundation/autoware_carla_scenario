"""Tests for far traffic in SUMO's own mesoscopic simulation
(``mesoscopic.engine: sumo``), against a real ``sumo --mesosim``.

The backend's use of it is in ``test_traffic_sumo.py`` (the tests taking
``meso_engine``); these drive the model alone, on a grid of two-lane roads
200 m apart with a signal at every junction.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Mapping, Optional

import pytest

from autoware_carla_scenario.traffic.fidelity import MesoscopicConfig
from autoware_carla_scenario.traffic.mesoscopic import MesoLink

if TYPE_CHECKING:
    from autoware_carla_scenario.traffic.sumo.mesosim import SumoMesoscopicTraffic

_HAS_SUMO = all(
    importlib.util.find_spec(m) is not None for m in ("libsumo", "sumolib", "traci")
)
pytestmark = pytest.mark.skipif(
    not _HAS_SUMO, reason="the `sumo` extra is not installed"
)

STEP = 0.1


@pytest.fixture(scope="module")
def grid(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 3 x 3 grid of junctions, each signalled: edges ``A0B0``, ``B0C0``..."""
    from autoware_carla_scenario.traffic.sumo.network import sumo_home

    net = tmp_path_factory.mktemp("meso") / "grid.net.xml"
    subprocess.run(
        [
            str(sumo_home() / "bin" / "netgenerate"),
            "--grid",
            "--grid.number",
            "3",
            "--grid.length",
            "200",
            "--default.lanenumber",
            "2",
            "--tls.guess",
            "true",
            "--tls.guess.threshold",
            "0",
            "-o",
            str(net),
        ],
        check=True,
        capture_output=True,
    )
    return net


def _links(net: Path) -> dict[str, MesoLink]:
    import sumolib

    links = {}
    for edge in sumolib.net.readNet(str(net)).getEdges():
        if edge.getFunction() == "":
            links[edge.getID()] = MesoLink(
                edge.getID(),
                float(edge.getLength()),
                edge.getLaneNumber(),
                float(edge.getSpeed()),
            )
    return links


@pytest.fixture
def make(
    grid: Path, tmp_path: Path
) -> Iterator[Callable[..., "SumoMesoscopicTraffic"]]:
    """A started model on the grid; closed at the end of the test."""
    from autoware_carla_scenario.traffic.sumo.mesosim import SumoMesoscopicTraffic
    from autoware_carla_scenario.traffic.sumo.network import sumo_home

    started: list[SumoMesoscopicTraffic] = []

    def build(
        signal_states: Optional[Callable[[], Mapping[str, str]]] = None,
        junction_control: bool = True,
    ) -> SumoMesoscopicTraffic:
        model = SumoMesoscopicTraffic(
            _links(grid),
            net_file=grid,
            binary=str(sumo_home() / "bin" / "sumo"),
            step_length=STEP,
            seed=7,
            junction_control=junction_control,
            signal_states=signal_states,
            log_file=tmp_path / "meso.log",
        )
        started.append(model)
        return model

    yield build
    for model in started:
        model.close()


class _Clock:
    """Steps a model and keeps the time."""

    def __init__(self, model: "SumoMesoscopicTraffic") -> None:
        self.model = model
        self.now = 0.0
        self.arrived: list[str] = []

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / STEP)):
            self.now = round(self.now + STEP, 6)
            self.arrived += [v.id for v in self.model.step(self.now)]


def test_a_vehicle_drives_on_smoothly_between_segments(
    make: Callable[..., "SumoMesoscopicTraffic"],
) -> None:
    model = make()
    clock = _Clock(model)
    model.add("v", ["A0B0", "B0C0"], now=0.0, offset_m=10.0)
    assert "v" in model and len(model) == 1
    clock.run(1.0)
    last = model.position("v", clock.now)[1]
    assert last >= 10.0
    jumps = []
    for _ in range(30):
        clock.run(STEP)
        road, offset = model.position("v", clock.now)
        if road == "A0B0":
            jumps.append(offset - last)
            last = offset
    speed = model.speed("v", clock.now)
    assert speed > 5.0
    # Moved on every step, never by more than a step's driving: no segment
    # sized jumps.
    assert jumps and all(-1e-6 <= j <= speed * STEP + 0.5 for j in jumps)
    assert sum(jumps) > 10.0


def test_a_vehicle_at_the_end_of_its_route_arrives_and_leaves(
    make: Callable[..., "SumoMesoscopicTraffic"],
) -> None:
    model = make(junction_control=False)
    clock = _Clock(model)
    model.add("v", ["A0B0", "B0C0"], now=0.0)
    for _ in range(60):
        clock.run(1.0)
        if clock.arrived:
            break
    assert clock.arrived == ["v"]
    assert "v" not in model and len(model) == 0


def test_a_vehicle_told_to_stand_stands_and_drives_on_from_there(
    make: Callable[..., "SumoMesoscopicTraffic"],
) -> None:
    model = make()
    clock = _Clock(model)
    model.add("v", ["A0B0", "B0C0"], now=0.0)
    clock.run(3.0)
    model.set_speed_cap("v", 0.0, clock.now)
    assert model.held("v")
    stood = model.position("v", clock.now)
    assert stood[1] > 10.0
    clock.run(10.0)
    assert model.position("v", clock.now) == stood
    assert model.speed("v", clock.now) == 0.0
    assert model.vehicle("v").speed_cap == 0.0

    model.set_speed_cap("v", None, clock.now)
    assert not model.held("v")
    clock.run(2.0)
    road, offset = model.position("v", clock.now)
    # From where it stood (up to a segment back, where SUMO puts it in), on.
    assert road != "A0B0" or offset >= stood[1] - 21.0
    clock.run(3.0)
    road, offset = model.position("v", clock.now)
    assert road != "A0B0" or offset > stood[1]


def test_a_slower_speed_holds_from_where_it_is(
    make: Callable[..., "SumoMesoscopicTraffic"],
) -> None:
    model = make()
    clock = _Clock(model)
    model.add("v", ["A0B0", "B0C0"], now=0.0)
    clock.run(2.0)
    model.set_speed_cap("v", 3.0, clock.now)
    clock.run(3.0)
    assert model.speed("v", clock.now) <= 3.0 + 1e-6
    model.set_max_speed("v", 2.0, clock.now)
    clock.run(3.0)
    assert model.speed("v", clock.now) <= 2.0 + 1e-6
    assert model.vehicle("v").max_speed == 2.0


def test_a_red_signal_holds_a_vehicle_until_it_is_green(
    make: Callable[..., "SumoMesoscopicTraffic"],
) -> None:
    states: dict[str, str] = {}
    model = make(signal_states=lambda: states)
    conn = model._conn
    for tls in conn.trafficlight.getIDList():
        states[tls] = "r" * len(conn.trafficlight.getRedYellowGreenState(tls))
    clock = _Clock(model)
    model.add("v", ["A0B0", "B0C0", "C0C1"], now=0.0)
    clock.run(40.0)
    road, offset = model.position("v", clock.now)
    assert road == "A0B0" and offset > 100.0
    assert model.speed("v", clock.now) == 0.0

    for tls in states:
        states[tls] = "G" * len(states[tls])
    clock.run(10.0)
    assert model.vehicle("v").link != "A0B0"


def test_a_vehicle_turns_where_it_is_told_to(
    make: Callable[..., "SumoMesoscopicTraffic"],
) -> None:
    model = make(junction_control=False)
    clock = _Clock(model)
    model.add("v", ["A0B0", "B0C0"], now=0.0)
    clock.run(1.0)
    model.set_route("v", ["A0B0", "B0B1"])
    seen = set()
    for _ in range(40):
        clock.run(1.0)
        if "v" not in model:
            break
        seen.add(model.vehicle("v").link)
    assert "B0B1" in seen and "B0C0" not in seen
    with pytest.raises(ValueError, match="must start on"):
        model.add("w", ["A0B0"], now=clock.now)
        model.set_route("w", ["B0C0"])


def test_a_vehicle_taken_out_may_go_back_in_at_once(
    make: Callable[..., "SumoMesoscopicTraffic"],
) -> None:
    model = make()
    clock = _Clock(model)
    model.add("v", ["A0B0", "B0C0"], now=0.0, length=7.5)
    clock.run(2.0)
    taken = model.remove("v")
    assert taken is not None and taken.length == 7.5
    assert model.remove("v") is None
    # The same id, in the same step: SUMO must not refuse it.
    model.add("v", ["A0B0", "B0C0"], now=clock.now, offset_m=30.0, length=7.5)
    clock.run(1.0)
    assert model.speed("v", clock.now) > 0.0


def test_the_model_refuses_what_it_cannot_hold(
    make: Callable[..., "SumoMesoscopicTraffic"],
) -> None:
    model = make()
    with pytest.raises(ValueError, match="no mesoscopic road"):
        model.add("v", [], now=0.0)
    with pytest.raises(ValueError, match="no mesoscopic road"):
        model.add("v", ["nowhere"], now=0.0)
    model.add("v", ["A0B0"], now=0.0)
    with pytest.raises(ValueError, match="already"):
        model.add("v", ["A0B0"], now=0.0)


def test_closing_stops_sumo(make: Callable[..., "SumoMesoscopicTraffic"]) -> None:
    model = make()
    process = model._process
    assert process is not None and process.poll() is None
    model.close()
    assert process.poll() is not None
    model.close()  # twice is harmless


def test_segment_count_is_sumos() -> None:
    from autoware_carla_scenario.traffic.sumo.mesosim import segment_count

    assert segment_count(200.0, 20.0) == 10
    assert segment_count(29.0, 20.0) == 1
    assert segment_count(31.0, 20.0) == 2
    assert segment_count(5.0, 20.0) == 1


def test_the_engine_is_sumos_by_default_and_checked() -> None:
    assert MesoscopicConfig().engine == "sumo"
    assert MesoscopicConfig(engine="queue").engine == "queue"
    with pytest.raises(
        ValueError, match="mesoscopic.engine must be one of sumo, queue"
    ):
        MesoscopicConfig(engine="vissim")
    with pytest.raises(ValueError, match="segment_length_m must be positive"):
        MesoscopicConfig(segment_length_m=0.0)
    with pytest.raises(ValueError, match="segment_length_m must be positive"):
        MesoscopicConfig.from_mapping({"segment_length_m": True})
