"""Every field of an authored action reaches the run, or is refused (#966).

The sibling of ``test_entity_field_parity``, for :class:`ActionNode`.  The
model holds ``once`` and ``overrides`` for every action type, but a builder
passes each only when its runtime constructor takes it.  A value set where
nothing reads it would be dropped in silence: the validator accepted it, the
editor showed it, and the run did something else.  So each field, on each
action type, is either carried -- it changes the action the builder makes --
or refused by the validator with an error on that field.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest

from autoware_carla_scenario.authoring import registry
from autoware_carla_scenario.authoring._builders_generated import (
    ONCE_ACTIONS,
    OVERRIDING_ACTIONS,
)
from autoware_carla_scenario.authoring.builders import instantiate_action
from autoware_carla_scenario.authoring.compiler import BuildContext, compile_document
from autoware_carla_scenario.authoring.models import (
    TICK_PHASES,
    ActionNode,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.validator import validate_document

#: The entity of each kind in :func:`_document`, by kind.
ACTORS = {"ego": "ego", "vehicle": "car", "pedestrian": "walker"}

#: Parameters a card needs beyond its defaults to be valid and buildable.
NEEDED_PARAMS: dict[str, dict[str, Any]] = {
    **{
        type_id: {"location": "any"}
        for type_id in (
            "adversarial_brake",
            "adversarial_cut_in",
            "adversarial_merge",
            "head_on",
            "ignore_leader",
            "jaywalk",
            "lane_change_abort",
            "pedestrian_run_red_light",
            "run_traffic_rule",
            "stop_crossing",
        )
    },
    "adversarial_traffic": {
        "adversities": [
            {"kind": "brake_check", "location": "any", "probability": 0.001}
        ]
    },
    "traffic_signal_controller": {"controller": "crossing", "signal_phase": "ns_green"},
    "play_motion": {"motion_path": "wave.npz"},
    "smart_pedestrian": {
        "prompts": [{"time_s": 0.0, "text": "walk"}],
        "waypoints": [{"x": 1.0, "y": 2.0}],
    },
    "traffic_parameters": {"speed_factor_mean": 1.0},
    "vehicle_parameters": {"speed_factor_mean": 1.0},
    "routing": {"goal_lanelet_id": 300},
}

#: Fields that are not flags the built action is configured with.
NOT_RUNTIME = {
    "id": "the action's identity; it names the action only when it has no title",
    "type": "chooses which action is built; every case below is run per type",
    "actor": "which entity performs it; checked against the spec's actor kinds",
    "params": "a container; its fields are the spec's, checked by the registry",
    "trigger": "a condition, built and checked by the condition builders",
    "searches": "how a lanelet parameter is chosen, resolved before the build",
    "advance_conditions": "waypoint conditions, checked by the trajectory tests",
}

#: Every other field, with the values it may take other than its default.
CHANGED: dict[str, tuple[Any, ...]] = {
    "title": ("Renamed",),
    # The initialization phase is dispatched by the runner from the node, not
    # by the built action, and is refused with a trigger (test_authoring_document).
    "phase": tuple(sorted(TICK_PHASES - {ActionNode.model_fields["phase"].default})),
    "once": (False,),
    "overrides": (True,),
}

TYPES = [spec.type_id for spec in registry.action_specs()]


def _document(type_id: str, **fields: Any) -> ScenarioDocument:
    spec = registry.get_action_spec(type_id)
    assert spec is not None
    params = registry.default_params(spec.fields)
    params.update(NEEDED_PARAMS.get(type_id, {}))
    action: dict[str, Any] = {
        "id": "a",
        "type": type_id,
        "params": params,
        "trigger": {"type": "elapsed_time", "params": {"duration_seconds": 1.0}},
        **fields,
    }
    if spec.scope == "actor":
        action["actor"] = ACTORS[spec.actor_kinds[0]]
    return ScenarioDocument.model_validate(
        {
            "entities": [
                {"id": "ego", "kind": "ego", "spawn": {"lanelet_id": 242}},
                {"id": "car", "kind": "vehicle", "spawn": {"lanelet_id": 243}},
                {"id": "walker", "kind": "pedestrian", "spawn": {"lanelet_id": 244}},
            ],
            "map": {
                "traffic_signal_controllers": [
                    {
                        "name": "crossing",
                        "phases": [
                            {
                                "name": "ns_green",
                                "duration_seconds": 5.0,
                                "states": [
                                    {
                                        "lanelet2_regulatory_element_id": 1001,
                                        "state": "green",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            "actions": [action],
            "assertions": {
                "pass": [{"type": "elapsed_time", "params": {"duration_seconds": 5.0}}],
                "fail": [{"type": "timeout", "params": {"timeout_seconds": 30.0}}],
            },
        }
    )


@pytest.fixture(autouse=True)
def _no_motion_file() -> Iterator[None]:
    """A motion is read off disk when the action is built; its content is
    beside the point here."""
    with mock.patch(
        "autoware_carla_scenario.actions.play_motion.load_soma_motion",
        return_value=object(),
    ):
        yield


def _snapshot(document: ScenarioDocument) -> dict[str, Any]:
    """The plain values the built action holds: what a run is configured with."""
    action = instantiate_action(
        compile_document(document).actions[0], BuildContext(scenario=None)
    )
    return {
        name: value
        for name, value in vars(action).items()
        if value is None or isinstance(value, (bool, int, float, str, enum.Enum))
    }


def test_every_field_is_classified() -> None:
    """A new field must say how it reaches the run, or why it does not."""
    assert set(ActionNode.model_fields) == set(NOT_RUNTIME) | set(CHANGED)


@pytest.mark.parametrize("type_id", TYPES)
def test_the_base_document_is_valid(type_id: str) -> None:
    report = validate_document(_document(type_id))
    assert report.ok, [f"{i.path}: {i.message}" for i in report.errors]


@pytest.mark.parametrize(
    ("type_id", "field", "value"),
    [
        (type_id, field, value)
        for type_id in TYPES
        for field, values in CHANGED.items()
        for value in values
    ],
)
def test_every_field_is_carried_or_refused(
    type_id: str, field: str, value: Any
) -> None:
    changed = _document(type_id, **{field: value})
    path = f"actions[0].{field}"
    refused = [i for i in validate_document(changed).errors if i.path == path]
    if refused:
        return
    assert _snapshot(changed) != _snapshot(
        _document(type_id)
    ), f"{field}={value!r} on {type_id} is accepted but does not change the action"


@pytest.mark.parametrize("type_id", TYPES)
def test_once_false_is_refused_exactly_where_it_is_dropped(type_id: str) -> None:
    errors = validate_document(_document(type_id, once=False)).errors
    refused = [i for i in errors if i.path == "actions[0].once"]
    assert bool(refused) == (type_id not in ONCE_ACTIONS)
    if refused:
        spec = registry.get_action_spec(type_id)
        assert spec is not None
        # The message names the card, so the author knows which one to fix.
        assert spec.title in refused[0].message
        assert type_id in refused[0].message


def test_the_actions_that_cannot_repeat() -> None:
    """The ones #966 found: their constructors take no ``once``."""
    assert {
        "stalled_object",
        "construction_zone",
        "urban_construction_zone",
        "adversarial_traffic",
    } == set(TYPES) - ONCE_ACTIONS
    # Both flags name action types the registry knows.
    assert ONCE_ACTIONS <= set(TYPES)
    assert OVERRIDING_ACTIONS <= set(TYPES)
