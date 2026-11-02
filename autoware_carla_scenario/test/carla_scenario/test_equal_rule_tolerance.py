"""An "=" rule on a continuous card compares within a tolerance it can set (#1091).

Cards measuring a speed, a distance or a time offered "=" with no tolerance
field, so the runtime's 1e-6 applied and "Speed = 10 m/s" practically never
fired.  Each such card now has an optional ``tolerance``, passed through to
the runtime; a document without one keeps the old 1e-6 and is warned about.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from autoware_carla_scenario.authoring import _builders_generated, codegen, registry
from autoware_carla_scenario.authoring.builders import instantiate_condition
from autoware_carla_scenario.authoring.compiler import (
    EGO_ROLE,
    BuildContext,
    compile_document,
)
from autoware_carla_scenario.authoring.models import ScenarioDocument
from autoware_carla_scenario.authoring.validator import validate_document

#: The cards measuring a continuous quantity whose runtime takes a tolerance.
_TOLERANT = (
    "entity_distance",
    "distance",
    "ttc",
    "ttc_to_position",
    "time_headway",
    "speed",
    "route_progress",
    "acceleration",
    "relative_speed",
    "metric",
)


def _document(trigger: dict[str, Any]) -> ScenarioDocument:
    return ScenarioDocument.model_validate(
        {
            "entities": [
                {"id": "ego", "kind": "ego", "spawn": {"lanelet_id": 242}},
            ],
            "assertions": {
                "pass": [trigger],
                "fail": [{"type": "timeout", "params": {"timeout_seconds": 30.0}}],
            },
        }
    )


def _speed(tolerance: Optional[float] = None, rule: str = "equal_to") -> dict:
    params: dict[str, Any] = {"entity": "ego", "rule": rule, "value": 10.0}
    if tolerance is not None:
        params["tolerance"] = tolerance
    return {"type": "speed", "params": params}


def _world(speed: float) -> MagicMock:
    actor = MagicMock()
    actor.attributes = {"role_name": EGO_ROLE}
    actor.type_id = "vehicle.x"
    location = SimpleNamespace(x=0.0, y=0.0, z=0.5)
    actor.get_location.return_value = location
    actor.get_transform.return_value = SimpleNamespace(
        location=location, rotation=SimpleNamespace(yaw=0.0, pitch=0.0, roll=0.0)
    )
    actor.get_velocity.return_value = SimpleNamespace(x=speed, y=0.0, z=0.0)
    world = MagicMock()
    world.get_actors.return_value = [actor]
    return world


def _built(trigger: dict[str, Any]) -> Any:
    compiled = compile_document(_document(trigger)).pass_conditions[0]
    return instantiate_condition(compiled, BuildContext(scenario=None))


@pytest.mark.parametrize("card", _TOLERANT)
def test_every_continuous_card_has_a_tolerance_it_passes_on(card: str) -> None:
    spec = registry.get_condition_spec(card)
    assert spec is not None
    field = next(f for f in spec.fields if f.name == "tolerance")
    assert not field.required
    assert field.default == registry.EQUAL_TO_DEFAULT_TOLERANCE
    source = inspect.getsource(getattr(_builders_generated, spec.builder))
    assert 'tolerance=params["tolerance"]' in source


def test_the_generated_builders_are_up_to_date() -> None:
    assert codegen.main(["--check"]) == 0


def test_a_speed_equal_to_with_a_tolerance_fires_near_the_value() -> None:
    condition = _built(_speed(tolerance=0.05))
    assert condition.check(_world(9.97), 0.0)
    assert condition.check(_world(10.04), 0.05)
    assert not condition.check(_world(9.9), 0.1)
    assert not condition.check(_world(10.3), 0.15)


def test_a_document_without_a_tolerance_still_loads_and_keeps_1e_6() -> None:
    document = _document(_speed())
    report = validate_document(document)
    assert report.ok, report.summary()
    condition = _built(_speed())
    assert not condition.check(_world(9.97), 0.0)
    assert condition.check(_world(10.0), 0.05)


def test_an_equal_to_left_at_1e_6_is_warned_about() -> None:
    report = validate_document(_document(_speed()))
    messages = [i.message for i in report.warnings if i.path.endswith(".tolerance")]
    assert len(messages) == 1
    assert "m/s" in messages[0]


@pytest.mark.parametrize(
    "trigger",
    [
        _speed(tolerance=0.1),
        _speed(rule="greater_than"),
        # A metric's tolerance has no unit: a collision "= 1" is exact.
        {"type": "metric", "params": {"rule": "equal_to", "value": 1.0}},
    ],
    ids=["tolerance-given", "not-equal-to", "metric"],
)
def test_no_warning_when_the_rule_can_be_met(trigger: dict[str, Any]) -> None:
    report = validate_document(_document(trigger))
    assert report.ok, report.summary()
    assert not [i for i in report.warnings if i.path.endswith(".tolerance")]


def test_a_negative_tolerance_is_an_error() -> None:
    report = validate_document(_document(_speed(tolerance=-0.1)))
    assert any(i.path.endswith(".tolerance") for i in report.errors)


def test_a_lanelet_length_equal_to_is_warned_about() -> None:
    """The sweeper's length constraint has no tolerance to give it."""
    choice = {
        "mode": "constraint_search",
        "constraints": [
            {"type": "lanelet_length", "params": {"rule": "equal_to", "value": 10.0}}
        ],
    }
    document = _document(
        {
            "type": "distance",
            "params": {"entity": "ego", "lanelet_id": 0},
            "searches": {"lanelet_id": choice},
        }
    )
    report = validate_document(document)
    assert report.ok, report.summary()
    assert any("lanelet length" in i.message for i in report.warnings)
