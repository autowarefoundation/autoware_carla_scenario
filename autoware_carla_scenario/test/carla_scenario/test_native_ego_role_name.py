"""An ego spawned for CARLA's native Autoware interface (role ``ego``, or
``hero``) is the ego everywhere the framework reads actor role names (#2044):
the post-run metric frames, the NPC collision monitor, the leftover-actor
sweep and the run-config check of ``ego.native_autoware_interface``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest

from autoware_carla_scenario import npc_collisions
from autoware_carla_scenario.metric_frames import FrameRecorder
from autoware_carla_scenario.npc_collisions import NpcCollisionMonitor
from autoware_carla_scenario.scenario_runner import _destroy_all_dynamic_actors

_CONF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
)

#: Every name the ego answers to: the framework's own and the native ones.
_EGO_ROLES = ["Ego", "ego", "hero", "HERO"]


def _vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Any:
    return SimpleNamespace(x=x, y=y, z=z)


class _Vehicle:
    """A vehicle with the steer and light state a recorder reads of the ego."""

    def __init__(
        self, actor_id: int, role: str, type_id: str = "vehicle.tesla.model3"
    ) -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes: Dict[str, str] = {"role_name": role, "base_type": "car"}
        self.bounding_box = SimpleNamespace(
            extent=_vector(2.0, 1.0, 0.8),
            location=_vector(),
            rotation=SimpleNamespace(yaw=0.0),
        )
        self.is_alive = True
        self.parent: Optional[Any] = None
        self.destroyed = False

    def get_transform(self) -> Any:
        return SimpleNamespace(
            location=_vector(float(self.id), 0.0, 0.0),
            rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0),
        )

    def get_velocity(self) -> Any:
        return _vector(1.0)

    def get_control(self) -> Any:
        return SimpleNamespace(steer=0.3)

    def get_light_state(self) -> int:
        return 4

    def destroy(self) -> bool:
        self.destroyed = True
        return True


class _Actors(list):  # type: ignore[type-arg]
    def filter(self, pattern: str) -> "_Actors":
        prefix = pattern.rstrip("*")
        return _Actors(a for a in self if str(a.type_id).startswith(prefix))


class _Sensor:
    def __init__(self, parent: int) -> None:
        self.parent = parent
        self.callback: Optional[Callable[[Any], None]] = None

    def listen(self, callback: Callable[[Any], None]) -> None:
        self.callback = callback

    def stop(self) -> None:
        pass

    def destroy(self) -> bool:
        return True


class _World:
    def __init__(self, actors: List[Any]) -> None:
        self._actors = _Actors(actors)
        self.sensors: List[_Sensor] = []

    def get_actors(self) -> _Actors:
        return self._actors

    def get_snapshot(self) -> Any:
        return SimpleNamespace(frame=7)

    def tick(self) -> int:
        return 7

    def get_blueprint_library(self) -> Any:
        return SimpleNamespace(find=lambda _name: object())

    def spawn_actor(self, _blueprint: Any, _transform: Any, attach_to: Any) -> Any:
        sensor = _Sensor(int(attach_to.id))
        self.sensors.append(sensor)
        return sensor


# ---------------------------------------------------------------------------
# The metric frames
# ---------------------------------------------------------------------------


class TestFrameRecorder:
    @pytest.mark.parametrize("role", _EGO_ROLES)
    def test_the_ego_s_steer_and_light_state_are_recorded(self, role: str) -> None:
        recorder = FrameRecorder()
        recorder.record(_World([_Vehicle(1, role)]), 0.0, None)  # type: ignore[arg-type]

        (ego,) = recorder.frames[0].actors
        assert ego.steer == pytest.approx(0.3)
        assert ego.light_state == 4

    @pytest.mark.parametrize("role", _EGO_ROLES)
    def test_the_ego_s_role_is_recorded_as_ego(self, role: str) -> None:
        recorder = FrameRecorder()
        recorder.record(_World([_Vehicle(1, role)]), 0.0, None)  # type: ignore[arg-type]

        assert recorder.frames[0].actors[0].role == "Ego"

    def test_an_npc_has_neither(self) -> None:
        recorder = FrameRecorder()
        recorder.record(_World([_Vehicle(2, "car1")]), 0.0, None)  # type: ignore[arg-type]

        (npc,) = recorder.frames[0].actors
        assert npc.role == "car1"
        assert npc.steer is None and npc.light_state is None


# ---------------------------------------------------------------------------
# The NPC collision monitor
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor() -> Any:
    made = NpcCollisionMonitor(despawn_fallen_after_s=None)
    yield made
    made.close()


def _event(other: Any) -> Any:
    return SimpleNamespace(frame=7, other_actor=other, normal_impulse=_vector(300.0))


class TestNpcCollisionMonitor:
    @pytest.mark.parametrize("role", _EGO_ROLES)
    def test_no_sensor_is_attached_to_the_ego(self, monitor: Any, role: str) -> None:
        world = _World([_Vehicle(1, role), _Vehicle(2, "car1")])
        monitor._scan(world)

        assert [sensor.parent for sensor in world.sensors] == [2]

    @pytest.mark.parametrize("role", _EGO_ROLES)
    def test_an_npc_hitting_the_ego_is_marked_as_the_ego_s(
        self, monitor: Any, role: str
    ) -> None:
        ego, npc = _Vehicle(1, role), _Vehicle(2, "car1")
        world = _World([ego, npc])
        monitor._scan(world)
        monitor._on_event(npc.id, _event(ego))

        (record,) = monitor._collisions(1.0)
        assert record.ego is True
        assert record.actors == ["car1", "Ego"]

    @pytest.mark.parametrize("role", ["ego", "hero"])
    def test_an_explicit_native_ego_role_is_read_as_the_ego(self, role: str) -> None:
        made = NpcCollisionMonitor(ego_role=role, despawn_fallen_after_s=None)
        try:
            world = _World([_Vehicle(1, "Ego"), _Vehicle(2, "car1")])
            made._scan(world)
            assert [sensor.parent for sensor in world.sensors] == [2]
        finally:
            made.close()

    def test_two_npcs_are_not_the_ego_s(self, monitor: Any) -> None:
        ego = _Vehicle(1, "ego")
        first, second = _Vehicle(2, "car1"), _Vehicle(3, "car2")
        monitor._scan(_World([ego, first, second]))
        monitor._on_event(first.id, _event(second))

        (record,) = monitor._collisions(1.0)
        assert record.ego is False
        assert record.actors == ["car1", "car2"]

    @pytest.mark.parametrize("role", ["ego", "hero"])
    def test_name_of_reads_the_native_ego_as_ego(self, role: str) -> None:
        assert npc_collisions._name_of(_Vehicle(1, role)) == "Ego"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The leftover-actor sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", _EGO_ROLES)
def test_an_attached_native_ego_survives_the_sweep(role: str) -> None:
    ego, npc = _Vehicle(1, role), _Vehicle(2, "car1")
    _destroy_all_dynamic_actors(
        _World([ego, npc]),  # type: ignore[arg-type]
        "native",
        keep_role_names=frozenset({"Ego"}),
    )

    assert not ego.destroyed
    assert npc.destroyed


# ---------------------------------------------------------------------------
# The run-config check of ego.native_autoware_interface
# ---------------------------------------------------------------------------


def _composed(overrides: List[str]) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(_CONF), version_base=None):
        return compose(config_name="config", overrides=overrides)


class TestTheRunConfigCheck:
    @pytest.mark.parametrize("value", ["yes", "1", "'true'", "null", "[true]"])
    def test_a_value_that_is_not_a_boolean_is_refused(self, value: str) -> None:
        from autoware_carla_scenario.examples import run

        cfg = _composed(
            [
                "ego.entity=autoware",
                f"ego.native_autoware_interface={value}",
                "autoware.launcher.type=none",
            ]
        )
        with pytest.raises(
            run.RunConfigError,
            match=r"^ego\.native_autoware_interface must be true or false",
        ):
            run.check_run_config(cfg)

    @pytest.mark.parametrize("entity", ["autopilot", "carla_driver"])
    def test_true_with_another_entity_is_refused(self, entity: str) -> None:
        from autoware_carla_scenario.examples import run

        cfg = _composed([f"ego.entity={entity}", "ego.native_autoware_interface=true"])
        with pytest.raises(
            run.RunConfigError,
            match=(
                r"^ego\.native_autoware_interface=true needs ego\.entity=autoware "
                rf"\(got ego\.entity={entity}\)"
            ),
        ):
            run.check_run_config(cfg)
        with pytest.raises(ValueError, match="needs ego.entity=autoware"):
            run.build_ego_entity(cfg)

    def test_a_batch_names_the_case_refused(self) -> None:
        from autoware_carla_scenario.examples import run

        good = _composed(["ego.entity=autoware", "autoware.launcher.type=none"])
        bad = _composed(["ego.native_autoware_interface=true"])
        with pytest.raises(
            run.RunConfigError, match=r"^second: ego\.native_autoware_interface"
        ):
            run.check_batch_configs([good, bad], ["first", "second"])

    @pytest.mark.parametrize("value", ["true", "false"])
    def test_a_boolean_with_the_autoware_entity_passes(self, value: str) -> None:
        from autoware_carla_scenario.examples import run

        run.check_run_config(
            _composed(
                [
                    "ego.entity=autoware",
                    f"ego.native_autoware_interface={value}",
                    "autoware.launcher.type=none",
                ]
            )
        )

    def test_false_with_another_entity_passes(self) -> None:
        from autoware_carla_scenario.examples import run

        run.check_run_config(
            _composed(["ego.entity=autopilot", "ego.native_autoware_interface=false"])
        )
