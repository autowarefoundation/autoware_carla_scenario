"""Closing conditions at the run's end: a listening sensor is released through
its own handle.

LibCarla unsubscribes a sensor's stream on the client only through the handle
that called ``listen()``. A sensor destroyed through another handle -- one from
``world.get_actors()``, as the leftovers' sweep before a reload takes it --
leaves the client's stream behind, and a batch's process eventually aborted on
it (``close: Bad file descriptor``). So every condition is closed at the end of
the run, and a ``CollisionCondition`` stops and destroys its sensor itself.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import typesafe_carla.carla as carla
import pytest

from autoware_carla_scenario import (
    AndCondition,
    BaseScenario,
    CollisionCondition,
    EgoConfig,
    NotCondition,
    OrCondition,
    PersistentCondition,
    ScenarioResult,
    SpawnTransform,
    StickyCondition,
)
from autoware_carla_scenario.actions.base import BaseAction
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.conditions.base import BaseCondition
from autoware_carla_scenario.conditions.composition.base import CompositionCondition


class _FakeSensor:
    """A collision sensor's client handle, recording what is asked of it."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.callback: Any = None
        self.stop_error: Optional[Exception] = None
        self.destroy_error: Optional[Exception] = None

    def listen(self, callback: Any) -> None:
        self.calls.append("listen")
        self.callback = callback

    def stop(self) -> None:
        self.calls.append("stop")
        if self.stop_error is not None:
            raise self.stop_error

    def destroy(self) -> bool:
        self.calls.append("destroy")
        if self.destroy_error is not None:
            raise self.destroy_error
        return True


def _world_with_ego(sensor: _FakeSensor) -> MagicMock:
    ego = MagicMock()
    ego.attributes = {"role_name": str(EGO_ROLE_NAME)}
    world = MagicMock()
    world.get_actors.return_value.filter.return_value = [ego]
    world.spawn_actor.return_value = sensor
    return world


def _attached(calls: list[str]) -> tuple[CollisionCondition, _FakeSensor, MagicMock]:
    sensor = _FakeSensor(calls)
    world = _world_with_ego(sensor)
    condition = CollisionCondition(label="collision")
    assert condition.check(world, 0.0) is None
    assert calls == ["listen"]
    return condition, sensor, world


def _collide(sensor: _FakeSensor) -> None:
    event = MagicMock()
    event.other_actor = None
    event.normal_impulse = carla.Vector3D(100.0, 0.0, 0.0)
    sensor.callback(event)


class TestCollisionConditionClose:
    def test_stops_then_destroys_the_sensor_it_listens_to(self) -> None:
        calls: list[str] = []
        condition, _sensor, _world = _attached(calls)
        condition.close()
        assert calls == ["listen", "stop", "destroy"]

    def test_closing_twice_releases_once(self) -> None:
        calls: list[str] = []
        condition, _sensor, _world = _attached(calls)
        condition.close()
        condition.close()
        assert calls == ["listen", "stop", "destroy"]

    def test_a_sensor_stopped_on_the_collision_is_not_stopped_again(self) -> None:
        calls: list[str] = []
        condition, sensor, world = _attached(calls)
        _collide(sensor)
        result = condition.check(world, 1.0)
        assert isinstance(result, ScenarioResult) and not result.passed
        assert calls == ["listen", "stop"]
        condition.close()
        # LibCarla warns on a stop of a sensor that does not listen.
        assert calls == ["listen", "stop", "destroy"]

    def test_no_sensor_attached_is_nothing_to_close(self) -> None:
        condition = CollisionCondition(label="collision")
        condition.close()  # no ego was ever found: nothing to release

    def test_a_closed_condition_attaches_no_sensor(self) -> None:
        calls: list[str] = []
        sensor = _FakeSensor(calls)
        world = _world_with_ego(sensor)
        condition = CollisionCondition(label="collision")
        condition.close()
        assert condition.check(world, 0.0) is None
        world.spawn_actor.assert_not_called()
        assert calls == []

    @pytest.mark.parametrize("failing", ["stop", "destroy"])
    def test_a_sensor_already_gone_is_let_go(self, failing: str) -> None:
        calls: list[str] = []
        condition, sensor, _world = _attached(calls)
        error = RuntimeError("trying to operate on a destroyed actor")
        if failing == "stop":
            sensor.stop_error = error
        else:
            sensor.destroy_error = error
        condition.close()
        assert calls == ["listen", "stop", "destroy"]
        condition.close()
        assert calls == ["listen", "stop", "destroy"]


class _Closable(BaseCondition):
    def __init__(self, label: str, closed: list[str]) -> None:
        super().__init__(label=label)
        self._closed = closed

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        return None

    def close(self) -> None:
        self._closed.append(self.label)


class _Composed(CompositionCondition):
    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        return None


class TestCompositeConditionsCloseTheirChildren:
    def test_and_or(self) -> None:
        closed: list[str] = []
        AndCondition([_Closable("a", closed), _Closable("b", closed)]).close()
        OrCondition([_Closable("c", closed), _Closable("d", closed)]).close()
        assert closed == ["a", "b", "c", "d"]

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda child: NotCondition(child),
            lambda child: StickyCondition(child),
            lambda child: PersistentCondition(child, duration=1.0),
            lambda child: _Composed(child=child, entity_name="npc1", label="composed"),
        ],
        ids=["not", "sticky", "persistent", "composition"],
    )
    def test_wrappers(self, wrap: Any) -> None:
        closed: list[str] = []
        wrap(_Closable("child", closed)).close()
        assert closed == ["child"]

    def test_a_collision_nested_in_a_composite_releases_its_sensor(self) -> None:
        calls: list[str] = []
        collision, _sensor, _world = _attached(calls)
        StickyCondition(OrCondition([collision, _Closable("x", [])])).close()
        assert calls == ["listen", "stop", "destroy"]

    def test_a_leaf_condition_closes_as_a_no_op(self) -> None:
        _Composed(label="leaf").close()


class _Action(BaseAction):
    def execute(self, world: "carla.World") -> None:
        pass


class _Scenario(BaseScenario):
    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return True


def _scenario() -> _Scenario:
    return _Scenario(
        EgoConfig(
            spawn_location=SpawnTransform(
                carla.Transform(carla.Location(x=0, y=0, z=0))
            ),
            vehicle_type="vehicle.mini.cooper",
        )
    )


class TestScenarioClosesItsConditions:
    def test_an_action_closes_its_trigger_and_its_until(self) -> None:
        closed: list[str] = []
        _Action(
            label="act",
            condition=_Closable("trigger", closed),
            until=_Closable("until", closed),
        ).close()
        assert closed == ["trigger", "until"]

    def test_every_condition_of_the_run_is_closed(self) -> None:
        closed: list[str] = []
        scenario = _scenario()
        scenario.register_pass_condition(_Closable("pass", closed))
        scenario.register_fail_condition(_Closable("fail", closed))
        scenario.register_init(
            _Action(label="init", condition=_Closable("init_trigger", closed))
        )
        scenario.register_pre_tick(
            _Action(label="pre", condition=_Closable("pre_trigger", closed))
        )
        scenario.register_post_tick(
            _Action(
                label="post",
                condition=_Closable("post_trigger", closed),
                until=_Closable("post_until", closed),
            )
        )
        scenario.close_conditions()
        assert closed == [
            "pass",
            "fail",
            "init_trigger",
            "pre_trigger",
            "post_trigger",
            "post_until",
        ]

    def test_one_failing_close_does_not_keep_the_others_open(self) -> None:
        closed: list[str] = []

        class _Failing(_Closable):
            def close(self) -> None:
                raise RuntimeError("boom")

        scenario = _scenario()
        scenario.register_fail_condition(_Failing("bad", closed))
        scenario.register_fail_condition(_Closable("good", closed))
        scenario.close_conditions()
        assert closed == ["good"]
