"""A background vehicle's ``speed_kmh`` means the same on every backend (#967).

``speed_kmh`` is the most a background vehicle drives at.  SUMO used to leave a
vehicle asked for 0 uncapped (it drove at the lane's limit) while the
TrafficManager stood it at 1 km/h, and the two floored small speeds
differently (1 m/s against 1 km/h).  Now a speed below 1 km/h, or one that is
not finite, is refused wherever it enters, and an accepted one is the cap on
both backends exactly.  Both backends are driven with fakes: nothing talks to
CARLA or SUMO.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    BaseScenario,
    EgoConfig,
    SpawnTransform,
    TrafficSourceAction,
)
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.constants import MIN_BACKGROUND_SPEED_KMH
from autoware_carla_scenario.traffic.base import (
    TrafficBackend,
    check_background_speed,
)
from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
from autoware_carla_scenario.traffic.traffic_manager import TrafficManagerBackend

_NOT_JUNCTION = [{"type": "not", "constraint": {"type": "is_junction"}}]
#: What neither backend may be asked for: no motion, backwards, not a number.
_REFUSED = (0.0, -5.0, 0.5, math.nan, math.inf, -math.inf)
#: Speeds both backends cap a vehicle at exactly, km/h.
_ACCEPTED = (MIN_BACKGROUND_SPEED_KMH, 2.0, 30.0)


def _transform() -> Any:
    return carla.Transform(
        carla.Location(x=10.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0)
    )


# ---------------------------------------------------------------------------
# TrafficManager, with a fake TrafficManager and world
# ---------------------------------------------------------------------------


class _FakeTM:
    def __init__(self) -> None:
        self.desired: dict[int, float] = {}

    def set_desired_speed(self, actor: Any, kmh: float) -> None:
        self.desired[actor.id] = kmh


class _FakeActor:
    id = 7

    def __init__(self) -> None:
        self.velocity: Optional[tuple[float, float]] = None

    def set_target_velocity(self, velocity: Any) -> None:
        self.velocity = (velocity.x, velocity.y)


class _FakeBlueprint:
    def has_attribute(self, name: str) -> bool:
        return False


class _FakeWorld:
    """A world with room everywhere."""

    def __init__(self) -> None:
        self.actor = _FakeActor()
        self.spawned = 0

    def get_blueprint_library(self) -> Any:
        return SimpleNamespace(filter=lambda pattern: [_FakeBlueprint()])

    def try_spawn_actor(self, blueprint: Any, transform: Any) -> _FakeActor:
        self.spawned += 1
        return self.actor

    def get_map(self) -> Any:
        """A driving lane under every spot (#976 refuses one off the road)."""
        return SimpleNamespace(
            get_waypoint=lambda location, **kwargs: SimpleNamespace(
                transform=carla.Transform(location, carla.Rotation())
            )
        )


def _tm_backend() -> tuple[TrafficManagerBackend, _FakeTM]:
    tm = _FakeTM()
    backend = TrafficManagerBackend()
    backend._client = SimpleNamespace(get_trafficmanager=lambda port: tm)
    return backend, tm


def _tm_cap_ms(speed_kmh: float) -> tuple[float, float]:
    """The TrafficManager's cap on a vehicle asked for *speed_kmh*, and the
    speed it starts at, both in m/s."""
    backend, tm = _tm_backend()
    world = _FakeWorld()
    assert backend.spawn_background(world, _transform(), speed_kmh=speed_kmh)
    assert world.actor.velocity is not None
    return tm.desired[7] / 3.6, math.hypot(*world.actor.velocity)


# ---------------------------------------------------------------------------
# SUMO, with a fake TraCI
# ---------------------------------------------------------------------------


def _sumo_backend() -> tuple[SumoTrafficBackend, list[tuple[Any, ...]]]:
    calls: list[tuple[Any, ...]] = []
    here = [0.0, 0.0]
    traci = SimpleNamespace(
        simulation=SimpleNamespace(convert2D=lambda edge, pos, lane: tuple(here)),
        route=SimpleNamespace(add=lambda *args, **kwargs: None),
        vehicle=SimpleNamespace(
            add=lambda vid, rid, **kw: calls.append(("add", vid, kw["departSpeed"])),
            setMaxSpeed=lambda vid, v: calls.append(("setMaxSpeed", vid, v)),
        ),
    )
    backend = SumoTrafficBackend.__new__(SumoTrafficBackend)
    backend.__dict__.update(
        _running=True,
        _traci=traci,
        _background_count=0,
        _background=set(),
        _blueprints={},
        _blueprint_offsets={},
        _offset=(0.0, 0.0),
        _config=SimpleNamespace(blueprint="vehicle.x"),
    )

    def road_position(pose: Any, sumo_id: str) -> tuple[str, float, int]:
        here[:] = [pose.x, pose.y]
        return ("edge0", 5.0, 0)

    backend.__dict__["_road_position"] = road_position
    backend.__dict__["_wander"] = lambda edge, *args: [edge]
    return backend, calls


def _sumo_cap_ms(speed_kmh: float) -> tuple[float, str]:
    """SUMO's cap on a vehicle asked for *speed_kmh*, m/s, and its departSpeed."""
    backend, calls = _sumo_backend()
    handle = backend.spawn_background(None, _transform(), speed_kmh=speed_kmh)
    assert handle is not None
    (cap,) = [c[2] for c in calls if c[0] == "setMaxSpeed" and c[1] == handle]
    (depart,) = [c[2] for c in calls if c[0] == "add" and c[1] == handle]
    return cap, depart


# ---------------------------------------------------------------------------
# One meaning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("speed_kmh", _ACCEPTED)
def test_both_backends_cap_a_vehicle_at_exactly_the_speed_asked(
    speed_kmh: float,
) -> None:
    tm_cap, tm_start = _tm_cap_ms(speed_kmh)
    sumo_cap, depart = _sumo_cap_ms(speed_kmh)
    assert tm_cap == pytest.approx(speed_kmh / 3.6)
    assert sumo_cap == pytest.approx(speed_kmh / 3.6)
    # It enters at no more than the cap: the TrafficManager's at it, SUMO's
    # at the fastest it safely can under it.
    assert tm_start == pytest.approx(speed_kmh / 3.6, rel=1e-5)
    assert depart == "max"


@pytest.mark.parametrize("speed_kmh", _REFUSED)
def test_both_backends_refuse_a_speed_no_vehicle_can_be_capped_at(
    speed_kmh: float,
) -> None:
    tm_backend, tm = _tm_backend()
    world = _FakeWorld()
    with pytest.raises(ValueError, match="speed_kmh"):
        tm_backend.spawn_background(world, _transform(), speed_kmh=speed_kmh)
    assert world.spawned == 0 and tm.desired == {}

    sumo_backend, calls = _sumo_backend()
    with pytest.raises(ValueError, match="speed_kmh"):
        sumo_backend.spawn_background(None, _transform(), speed_kmh=speed_kmh)
    assert calls == []


def test_the_default_backend_refuses_it_too() -> None:
    class _Plain(TrafficBackend):
        name = "plain"

    with pytest.raises(ValueError, match="speed_kmh"):
        _Plain().spawn_background(None, _transform(), speed_kmh=0.0)  # type: ignore[arg-type]


def test_check_background_speed() -> None:
    assert check_background_speed(30) == 30.0
    assert check_background_speed(MIN_BACKGROUND_SPEED_KMH) == MIN_BACKGROUND_SPEED_KMH
    for bad in _REFUSED:
        with pytest.raises(ValueError):
            check_background_speed(bad)


# ---------------------------------------------------------------------------
# Where it enters: the action, the config, the editor's validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("speed_kmh", _REFUSED)
def test_traffic_source_action_refuses_it(speed_kmh: float) -> None:
    with pytest.raises(ValueError, match="speed_kmh"):
        TrafficSourceAction(_NOT_JUNCTION, speed_kmh=speed_kmh)


def test_traffic_source_action_takes_a_small_positive_speed() -> None:
    TrafficSourceAction(_NOT_JUNCTION, speed_kmh=2.0)


def _scenario() -> BaseScenario:
    class _Scenario(BaseScenario):
        def setup(self) -> None: ...

        def is_done(self) -> bool:
            return True

    return _Scenario(
        EgoConfig(
            spawn_location=SpawnTransform(
                carla.Transform(carla.Location(x=0, y=0, z=0))
            ),
            vehicle_type="vehicle.mini.cooper",
        )
    )


def _with_background(speed: str) -> Any:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from autoware_carla_scenario.examples import conf as conf_package

    conf_dir = str(Path(conf_package.__file__).parent.resolve())
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        return compose(
            config_name="config",
            overrides=[
                "background_traffic.enabled=true",
                f"background_traffic.source.speed_kmh={speed}",
            ],
        )


@pytest.mark.parametrize("speed", ["0", "-10.0", "0.5", "nan", "inf", "fast"])
def test_the_config_refuses_it_naming_the_key(speed: str) -> None:
    from autoware_carla_scenario.examples.run import add_background_traffic

    with pytest.raises(ValueError, match=r"background_traffic\.source\.speed_kmh"):
        add_background_traffic(_with_background(speed), _scenario())


def test_the_config_takes_a_small_positive_speed() -> None:
    from autoware_carla_scenario.examples.run import add_background_traffic

    scenario = _scenario()
    add_background_traffic(_with_background("2.0"), scenario)
    (initial,) = scenario._init_actions
    assert isinstance(initial, TrafficSourceAction)
    assert initial._speed_kmh == 2.0


def _speed_errors(speed: Any) -> list[str]:
    document = ScenarioDocument(
        actions=[
            ActionNode(
                id="source",
                type="traffic_source",
                phase="init",
                params={"speed_kmh": speed},
            )
        ],
    )
    return [
        issue.message
        for issue in validate_document(document).errors
        if issue.path.endswith("speed_kmh")
    ]


@pytest.mark.parametrize("speed", [0, -3.0, 0.5, "nan", "inf"])
def test_the_validator_refuses_it(speed: Any) -> None:
    assert _speed_errors(speed)


@pytest.mark.parametrize("speed", [1.0, 2, 30.0])
def test_the_validator_takes_a_speed_of_at_least_1_kmh(speed: Any) -> None:
    assert _speed_errors(speed) == []


def test_sumo_holding_to_sight_never_raises_a_vehicle_above_its_cap() -> None:
    # Fog holds SUMO's traffic to its sight speed, but never below 1 m/s; a
    # background vehicle capped at 2 km/h (0.56 m/s) keeps its cap.
    caps: dict[str, float] = {"bg1": 2.0 / 3.6}
    backend = SumoTrafficBackend.__new__(SumoTrafficBackend)
    backend.__dict__.update(
        _max_speeds={},
        _traci=SimpleNamespace(
            vehicle=SimpleNamespace(
                getMaxSpeed=lambda vid: caps[vid],
                getDecel=lambda vid: 4.5,
                getTau=lambda vid: 1.0,
                setMaxSpeed=caps.__setitem__,
            )
        ),
    )
    backend._hold_to_sight("bg1", 0.1)
    assert caps["bg1"] == pytest.approx(2.0 / 3.6)
    backend._hold_to_sight("bg1", None)
    assert caps["bg1"] == pytest.approx(2.0 / 3.6)
