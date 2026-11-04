"""What the validator accepts, the builder builds (#1021).

A card the validator passes but whose action or condition refuses its
arguments fails only in ``DeclarativeScenario.setup``, against a live CARLA
world, after the document has saved, compiled and exported.  The checks here
keep the two in step: the Spawn and Despawn cards' actor-or-prop choice, a
prop's place, and -- for every card type in the registry, over a set of
representative field combinations -- "accepted implies builds".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest

from autoware_carla_scenario.actions import SpawnAction
from autoware_carla_scenario.authoring import compiler, registry
from autoware_carla_scenario.authoring.builders import (
    instantiate_action,
    instantiate_condition,
)
from autoware_carla_scenario.authoring.compiler import BuildContext, compile_document
from autoware_carla_scenario.authoring.models import (
    BindingRef,
    LaneletChoice,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.registry import FieldSpec
from autoware_carla_scenario.authoring.validator import (
    ValidationReport,
    _check_binding,
    _Collector,
    _traffic_mean_bounds,
    validate_document,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import Lanelet2Pose
from autoware_carla_scenario.sweeper.bindings import parse_binding

from .test_play_motion import _write_motion

_DATA = Path(__file__).resolve().parents[3] / "data"
#: A road lanelet of the fixture map.
_LANELET = 243
#: Entity ids by the kind a card's actor may be.
_ENTITIES = {"ego": "ego", "vehicle": "car", "pedestrian": "ped"}


@pytest.fixture(scope="module")
def loaded_map() -> Iterator[MapManager]:
    """Load the fixture map once: conditions resolve their places on it."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(_DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm")
    yield manager
    MapManager._instance = saved


@pytest.fixture(autouse=True)
def _module_map(loaded_map: MapManager) -> None:
    """Keep this module's map in place for each test (conftest restores it)."""
    MapManager._instance = loaded_map


def _document(
    action: Optional[dict[str, Any]] = None,
    trigger: Optional[dict[str, Any]] = None,
) -> ScenarioDocument:
    return ScenarioDocument.model_validate(
        {
            "entities": [
                {"id": "ego", "kind": "ego", "spawn": {"lanelet_id": 242}},
                {
                    "id": "car",
                    "kind": "vehicle",
                    "spawn": {"lanelet_id": _LANELET, "hidden": True},
                },
                {"id": "ped", "kind": "pedestrian", "spawn": {"lanelet_id": _LANELET}},
            ],
            "actions": [action] if action is not None else [],
            "assertions": {
                "pass": [
                    trigger
                    or {"type": "elapsed_time", "params": {"duration_seconds": 5.0}}
                ],
                "fail": [{"type": "timeout", "params": {"timeout_seconds": 30.0}}],
            },
        }
    )


def _card(type_id: str, actor: Optional[str], **params: Any) -> ScenarioDocument:
    spec = registry.get_action_spec(type_id)
    assert spec is not None
    values = registry.default_params(spec.fields)
    values.update(params)
    action: dict[str, Any] = {
        "id": "a",
        "type": type_id,
        "params": values,
        "trigger": {"type": "elapsed_time", "params": {"duration_seconds": 1.0}},
    }
    if actor is not None:
        action["actor"] = actor
    return _document(action=action)


def _errors(report: ValidationReport) -> list[str]:
    return [f"{issue.path}: {issue.message}" for issue in report.errors]


def _build(document: ScenarioDocument) -> object:
    return instantiate_action(
        compile_document(document).actions[0], BuildContext(scenario=None)
    )


@pytest.fixture
def unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile without the validator, which :func:`compile_document` runs
    first: what a refused card does when built is then the action's or the
    condition's own answer, not the validator's again."""
    monkeypatch.setattr(compiler, "validate_document", lambda _: ValidationReport())


# -- Spawn and Despawn: an entity or a prop ----------------------------------

_TARGETS = [
    # (actor, name, accepted)
    ("car", None, True),
    (None, "cone1", True),
    ("car", "cone1", False),
    (None, None, False),
    (None, "  ", False),  # a blank name is no name
    ("car", "  ", True),
]


@pytest.mark.parametrize("type_id", ["spawn", "despawn"])
@pytest.mark.parametrize(("actor", "name", "accepted"), _TARGETS)
def test_spawn_and_despawn_take_an_entity_or_a_prop(
    type_id: str,
    actor: Optional[str],
    name: Optional[str],
    accepted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params: dict[str, Any] = {"name": name}
    if type_id == "spawn":
        params["lanelet_id"] = _LANELET
    document = _card(type_id, actor, **params)
    report = validate_document(document)
    assert report.ok is accepted, _errors(report)
    if accepted:
        _build(document)
    else:
        assert any(
            "entity" in issue.message and "prop" in issue.message
            for issue in report.errors
        ), _errors(report)
        monkeypatch.setattr(compiler, "validate_document", lambda _: ValidationReport())
        with pytest.raises(ValueError, match="one of the two"):
            _build(document)


@pytest.mark.parametrize(
    ("lanelet_id", "accepted"),
    [(0, False), (-1, False), (None, False), (_LANELET, True)],
)
def test_a_prop_spawn_card_needs_a_lanelet(
    lanelet_id: Optional[int], accepted: bool
) -> None:
    document = _card("spawn", None, name="cone1", lanelet_id=lanelet_id)
    report = validate_document(document)
    assert report.ok is accepted, _errors(report)
    if not accepted:
        assert [i.path for i in report.errors] == ["actions[0].lanelet_id"]


@pytest.mark.usefixtures("unchecked")
@pytest.mark.parametrize("lanelet_id", [0, -1, None])
def test_a_prop_spawn_card_on_no_lanelet_does_not_build(
    lanelet_id: Optional[int],
) -> None:
    with pytest.raises(ValueError, match="needs a lanelet"):
        _build(_card("spawn", None, name="cone1", lanelet_id=lanelet_id))


@pytest.mark.parametrize("type_id", ["spawn", "distance"])
def test_a_searched_lanelet_is_a_place(type_id: str) -> None:
    """The sweep fills a searched lanelet in: no default is a warning, as for
    every other searched lanelet, not a missing place."""
    choice = {"mode": "constraint_search", "constraints": [{"type": "is_junction"}]}
    search = {"lanelet_id": choice}
    if type_id == "spawn":
        document = _card("spawn", None, name="cone1", lanelet_id=0)
        document.actions[0].searches = {
            "lanelet_id": LaneletChoice.model_validate(choice)
        }
    else:
        document = _document(
            trigger={
                "type": "distance",
                "params": {"entity": "car", "lanelet_id": 0},
                "searches": search,
            }
        )
    report = validate_document(document)
    assert report.ok, _errors(report)
    assert any("No default lanelet" in i.message for i in report.warnings)


def test_a_prop_spawn_card_may_be_placed_by_a_route_pose_instead() -> None:
    document = _card("spawn", None, name="cone1", route_pose="lane offset=10")
    report = validate_document(document)
    assert "actions[0].lanelet_id" not in [i.path for i in report.errors]


def test_an_entity_spawn_card_may_leave_its_lanelet_empty() -> None:
    report = validate_document(_card("spawn", "car"))
    assert report.ok, _errors(report)


@pytest.mark.parametrize("lanelet_id", [0, -3])
def test_a_prop_on_no_lanelet_is_refused_at_construction(lanelet_id: int) -> None:
    with pytest.raises(ValueError, match="needs a lanelet"):
        SpawnAction(name="cone1", position=Lanelet2Pose(lanelet_id, 5.0))


def test_an_entity_on_no_lanelet_still_waits_where_it_was_declared() -> None:
    SpawnAction("npc1", position=Lanelet2Pose(0, 0.0))


# -- the other mismatches the generic check found -----------------------------


@pytest.mark.usefixtures("unchecked")
@pytest.mark.parametrize(
    ("type_id", "actor", "params", "path"),
    [
        (
            "construction_zone",
            None,
            {"lanelet_id": _LANELET, "start_s": 50.0, "end_s": 50.0},
            "actions[0].end_s",
        ),
        (
            "urban_construction_zone",
            None,
            {"lanelet_id": _LANELET, "start_s": 60.0, "end_s": 10.0},
            "actions[0].end_s",
        ),
        (
            "construction_zone",
            None,
            {"lanelet_id": _LANELET, "start_time": 5.0, "end_time": 5.0},
            "actions[0].end_time",
        ),
        (
            "stalled_object",
            "car",
            {"lanelet_id": _LANELET, "start_time": 3.0, "end_time": 1.0},
            "actions[0].end_time",
        ),
        (
            "stalled_object",
            "car",
            {"lanelet_id": _LANELET, "end_time": 2.0e9},
            "actions[0].end_time",
        ),
        ("traffic_parameters", None, {}, "actions[0]"),
        ("vehicle_parameters", "car", {}, "actions[0]"),
        (
            "traffic_parameters",
            None,
            {"speed_factor_std": 0.1, "lane_changes": True},
            "actions[0].speed_factor_std",
        ),
        (
            "traffic_source",
            None,
            {"constraints": "not a list"},
            "actions[0].constraints",
        ),
        (
            "traffic_sink",
            None,
            {"constraints": "[{type: no_such_constraint}]"},
            "actions[0].constraints",
        ),
    ],
)
def test_validator_refuses_what_the_action_refuses(
    type_id: str, actor: Optional[str], params: dict[str, Any], path: str
) -> None:
    document = _card(type_id, actor, **params)
    report = validate_document(document)
    assert path in [i.path for i in report.errors], _errors(report)
    with pytest.raises(ValueError):
        _build(document)


@pytest.mark.parametrize(
    ("type_id", "actor"),
    [
        ("construction_zone", None),
        ("urban_construction_zone", None),
        ("stalled_object", "car"),
    ],
)
def test_validator_refuses_an_action_on_lanelet_0(
    type_id: str, actor: Optional[str], loaded_map: MapManager
) -> None:
    """These build on lanelet 0 and fail only when the run looks it up."""
    report = validate_document(_card(type_id, actor, lanelet_id=0))
    assert "a.lanelet_id" in [i.path for i in report.errors], _errors(report)
    with pytest.raises(Exception, match="InvalId"):
        loaded_map.lanelet_map.laneletLayer[0]


@pytest.mark.parametrize(
    ("condition", "field"),
    [
        (
            {"type": "distance", "params": {"entity": "car", "lanelet_id": 0}},
            "lanelet_id",
        ),
        (
            {
                "type": "entity_distance",
                "params": {
                    "source": "car",
                    "target": "ped",
                    "measure": "entity",
                    "signed": True,
                },
            },
            "signed",
        ),
        (
            {
                "type": "temporary_stop",
                "params": {"entity": "car", "stop_lanelets": []},
            },
            "stop_lanelets",
        ),
        # Along the road is one-dimensional, between centres (#1097).
        (
            {
                "type": "entity_distance",
                "params": {
                    "source": "car",
                    "target": "ped",
                    "measure": "lane",
                    "distance_type": "LATERAL",
                },
            },
            "distance_type",
        ),
        (
            {
                "type": "entity_distance",
                "params": {
                    "source": "car",
                    "target": "ped",
                    "measure": "lane",
                    "edge_to_edge": True,
                },
            },
            "edge_to_edge",
        ),
        (
            {
                "type": "ttc",
                "params": {
                    "source": "car",
                    "target": "ped",
                    "measure": "lane",
                    "edge_to_edge": True,
                },
            },
            "edge_to_edge",
        ),
    ],
)
@pytest.mark.usefixtures("unchecked")
def test_validator_refuses_what_the_condition_refuses(
    condition: dict[str, Any], field: str
) -> None:
    document = _document(trigger=condition)
    report = validate_document(document)
    paths = [i.path for i in report.errors]
    assert any(p.endswith(f".{field}") for p in paths), _errors(report)
    with pytest.raises((ValueError, RuntimeError)):
        compiled = compile_document(document.model_copy(deep=True))
        instantiate_condition(compiled.pass_conditions[0], BuildContext(scenario=None))


# -- accepted implies builds, for every card ---------------------------------

#: A representative value for a field of each kind.
_SAMPLES: dict[str, Any] = {
    "lanelet": _LANELET,
    "lanelet_list": [_LANELET],
    "number": 1.0,
    "int": 1,
    "bool": True,
    "text": "x",
    "entity": "car",
    "int_list": [1],
}


@pytest.fixture(scope="module")
def samples(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], Any]:
    """Values for fields a generic sample does not suit: files that have to
    exist on disk, which the run reads (relative to its working directory)
    and the validator does not."""
    folder = tmp_path_factory.mktemp("files")
    motion = _write_motion(folder / "m.npz", [{}])
    keyframes = folder / "keyframes.json"
    keyframes.write_text(
        json.dumps(
            {"keyframes": [{"time_s": 0.5, "joints": {"Head": [0.0, 0.0, 1.7]}}]}
        )
    )
    return {
        ("play_motion", "motion_path"): str(motion),
        ("smart_pedestrian", "poses"): str(keyframes),
    }


def _sample(type_id: str, spec: FieldSpec, samples: dict[tuple[str, str], Any]) -> Any:
    if (type_id, spec.name) in samples:
        return samples[(type_id, spec.name)]
    if spec.kind == "record_list":
        record = registry.default_params(spec.item_fields)
        for item in spec.item_fields:
            if item.kind in _SAMPLES and record.get(item.name) in (None, "", []):
                record[item.name] = _SAMPLES[item.kind]
        return [record]
    return _SAMPLES.get(spec.kind)


def _variants(
    type_id: str, fields: tuple[FieldSpec, ...], samples: dict[tuple[str, str], Any]
) -> Iterator[tuple[str, dict[str, Any]]]:
    """The defaults; each field alone set on them; every field filled; and,
    on the filled card, each field sampled and each select option."""
    base = registry.default_params(fields)
    yield "defaults", dict(base)
    for spec in fields:
        value = _sample(type_id, spec, samples)
        if value is not None and base.get(spec.name) != value:
            yield f"only {spec.name}", {**base, spec.name: value}
    filled = dict(base)
    for spec in fields:
        value = _sample(type_id, spec, samples)
        if value is not None and (
            spec.required or filled.get(spec.name) in (None, "", [])
        ):
            filled[spec.name] = value
        elif spec.kind == "select" and spec.options and not filled.get(spec.name):
            filled[spec.name] = spec.options[0].value
    yield "filled", filled
    for spec in fields:
        if spec.kind == "select":
            for option in spec.options:
                yield f"{spec.name}={option.value}", {**filled, spec.name: option.value}
        else:
            value = _sample(type_id, spec, samples)
            if value is not None and base.get(spec.name) != value:
                yield f"{spec.name} sampled", {**filled, spec.name: value}


def _built_or_failure(build: Any) -> Optional[str]:
    try:
        build()
    except Exception as exc:  # noqa: BLE001 -- any refusal is the mismatch
        return f"{type(exc).__name__}: {exc}"
    return None


_ACTIONS = [spec.type_id for spec in registry.action_specs()]
_CONDITIONS = [spec.type_id for spec in registry.condition_specs()]


@pytest.mark.parametrize("card", _ACTIONS)
def test_every_accepted_action_card_builds(
    card: str, samples: dict[tuple[str, str], Any]
) -> None:
    spec = registry.get_action_spec(card)
    assert spec is not None
    actors: list[Optional[str]] = [None]
    if spec.scope != "environment":
        actors = [_ENTITIES[k] for k in spec.actor_kinds if k in _ENTITIES]
        if spec.actor_optional:
            actors.append(None)
    failures = []
    logging.disable(logging.CRITICAL)
    try:
        for actor in actors:
            for label, params in _variants(card, spec.fields, samples):
                document = _card(card, actor, **params)
                if not validate_document(document).ok:
                    continue
                failure = _built_or_failure(lambda: _build(document))
                if failure is not None:
                    failures.append(f"actor={actor} {label}: {failure}")
    finally:
        logging.disable(logging.NOTSET)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("card", _CONDITIONS)
def test_every_accepted_condition_card_builds(
    card: str, samples: dict[tuple[str, str], Any]
) -> None:
    spec = registry.get_condition_spec(card)
    assert spec is not None
    child = {"type": "elapsed_time", "params": {"duration_seconds": 1.0}}
    failures = []
    logging.disable(logging.CRITICAL)
    try:
        for label, params in _variants(card, spec.fields, samples):
            node: dict[str, Any] = {"type": card, "params": params}
            if spec.min_children:
                node["children"] = [dict(child) for _ in range(spec.min_children)]
            document = _document(trigger=node)
            if not validate_document(document).ok:
                continue
            compiled = compile_document(document).pass_conditions[0]
            failure = _built_or_failure(
                lambda: instantiate_condition(compiled, BuildContext(scenario=None))
            )
            if failure is not None:
                failures.append(f"{label}: {failure}")
    finally:
        logging.disable(logging.NOTSET)
    assert not failures, "\n".join(failures)


# -- a field's bounds are the runtime's (#1111) -------------------------------


def _bounded(fields: tuple[FieldSpec, ...]) -> Iterator[FieldSpec]:
    for spec in fields:
        if spec.kind == "record_list":
            yield from _bounded(spec.item_fields)
        elif (spec.minimum, spec.exclusive_minimum, spec.maximum) != (None,) * 3:
            yield spec


def _outside(spec: FieldSpec) -> list[float]:
    """Values just outside *spec*'s bounds."""
    values: list[float] = []
    if spec.exclusive_minimum is not None:
        values += [spec.exclusive_minimum, spec.exclusive_minimum - 1.0]
    if spec.minimum is not None:
        values.append(spec.minimum - 1.0)
    if spec.maximum is not None:
        values.append(spec.maximum + 1.0)
    return values


def _inside(spec: FieldSpec, value: Any) -> bool:
    return (
        (spec.exclusive_minimum is None or value > spec.exclusive_minimum)
        and (spec.minimum is None or value >= spec.minimum)
        and (spec.maximum is None or value <= spec.maximum)
    )


_ALL_SPECS: list[Any] = [*registry.action_specs(), *registry.condition_specs()]


@pytest.mark.parametrize("card", _ALL_SPECS, ids=lambda s: s.type_id)
def test_a_bounded_fields_default_is_within_its_bounds(card: Any) -> None:
    for spec in _bounded(card.fields):
        assert spec.kind in ("number", "int"), spec.name
        if spec.default is not None:
            assert _inside(spec, spec.default), (card.type_id, spec.name)


def _issue_paths(
    type_id: str, fields: tuple[FieldSpec, ...], name: str, value: float
) -> list[str]:
    params = registry.default_params(fields)
    for spec in fields:
        if spec.kind == "record_list" and any(i.name == name for i in spec.item_fields):
            params[spec.name] = [
                {**registry.default_params(spec.item_fields), name: value}
            ]
    if name in params:
        params[name] = value
    if registry.get_action_spec(type_id) is not None:
        action_spec = registry.get_action_spec(type_id)
        assert action_spec is not None
        actor = None
        if action_spec.scope != "environment":
            actor = _ENTITIES[action_spec.actor_kinds[0]]
        document = _card(type_id, actor, **params)
    else:
        node: dict[str, Any] = {"type": type_id, "params": params}
        condition_spec = registry.get_condition_spec(type_id)
        assert condition_spec is not None
        if condition_spec.min_children:
            node["children"] = [
                {"type": "elapsed_time", "params": {"duration_seconds": 1.0}}
            ] * condition_spec.min_children
        document = _document(trigger=node)
    return [i.path for i in validate_document(document).errors]


@pytest.mark.parametrize(
    ("card", "name", "value"),
    [
        (card, spec.name, value)
        for card in _ALL_SPECS
        for spec in _bounded(card.fields)
        for value in _outside(spec)
    ],
    ids=lambda v: getattr(v, "type_id", str(v)),
)
def test_the_validator_refuses_a_value_outside_a_fields_bounds(
    card: Any, name: str, value: float
) -> None:
    paths = _issue_paths(card.type_id, card.fields, name, value)
    assert any(p.endswith(f".{name}") for p in paths), paths


def test_the_bound_messages_say_what_the_runtime_says() -> None:
    document = _document(
        trigger={
            "type": "standstill",
            "params": {"entity": "car", "duration": 0.0, "speed_threshold": -1.0},
        }
    )
    messages = {i.path.rsplit(".", 1)[-1]: i.message for i in document_errors(document)}
    assert messages["duration"] == "Duration must be positive."
    assert messages["speed_threshold"].endswith("must not be negative.")
    document = _document(
        trigger={"type": "off_road_fraction", "params": {"fraction": 1.5}}
    )
    messages = {i.path.rsplit(".", 1)[-1]: i.message for i in document_errors(document)}
    assert messages["fraction"] == "Fraction must be in (0, 1]."


def document_errors(document: ScenarioDocument) -> list[Any]:
    return list(validate_document(document).errors)


@pytest.mark.usefixtures("unchecked")
@pytest.mark.parametrize(
    ("condition", "field"),
    [
        (
            {"type": "elapsed_time", "params": {"duration_seconds": 0.0}},
            "duration_seconds",
        ),
        (
            {"type": "elapsed_time", "params": {"duration_seconds": -2.0}},
            "duration_seconds",
        ),
        (
            {"type": "standstill", "params": {"entity": "car", "duration": 0.0}},
            "duration",
        ),
        (
            {"type": "standstill", "params": {"entity": "car", "duration": -1.0}},
            "duration",
        ),
        (
            {
                "type": "standstill",
                "params": {"entity": "car", "speed_threshold": -0.1},
            },
            "speed_threshold",
        ),
        (
            {
                "type": "persistent",
                "params": {"duration": 0.0},
                "children": [
                    {"type": "elapsed_time", "params": {"duration_seconds": 1.0}}
                ],
            },
            "duration",
        ),
        (
            {
                "type": "persistent",
                "params": {"duration": -1.0},
                "children": [
                    {"type": "elapsed_time", "params": {"duration_seconds": 1.0}}
                ],
            },
            "duration",
        ),
        (
            {
                "type": "temporary_stop",
                "params": {
                    "entity": "car",
                    "stop_lanelets": [_LANELET],
                    "stop_duration": 0.0,
                },
            },
            "stop_duration",
        ),
        (
            {
                "type": "temporary_stop",
                "params": {
                    "entity": "car",
                    "stop_lanelets": [_LANELET],
                    "stop_duration": -1.0,
                },
            },
            "stop_duration",
        ),
        (
            {
                "type": "temporary_stop",
                "params": {
                    "entity": "car",
                    "stop_lanelets": [_LANELET],
                    "s_margin": 0.0,
                },
            },
            "s_margin",
        ),
        ({"type": "timeout", "params": {"timeout_seconds": 0.0}}, "timeout_seconds"),
        (
            {"type": "lane_ends", "params": {"entity": "car", "distance": 0.0}},
            "distance",
        ),
        ({"type": "off_road_fraction", "params": {"fraction": 0.0}}, "fraction"),
        ({"type": "off_road_fraction", "params": {"fraction": 1.5}}, "fraction"),
        (
            {"type": "place_clear", "params": {"lanelet_id": _LANELET, "width": 0.0}},
            "width",
        ),
        (
            {"type": "place_clear", "params": {"lanelet_id": _LANELET, "margin": -1.0}},
            "margin",
        ),
        ({"type": "entity_count", "params": {"count": -1}}, "count"),
        (
            {"type": "entity_count", "params": {"min_distance_m": -1.0}},
            "min_distance_m",
        ),
        (
            {
                "type": "entity_count",
                "params": {"min_distance_m": 30.0, "max_distance_m": 10.0},
            },
            "min_distance_m",
        ),
        (
            {
                "type": "entity_count",
                "params": {"min_ahead_m": 5.0, "max_ahead_m": -5.0},
            },
            "min_ahead_m",
        ),
    ],
)
def test_validator_refuses_a_bound_the_condition_refuses(
    condition: dict[str, Any], field: str
) -> None:
    spec = registry.get_condition_spec(condition["type"])
    assert spec is not None
    condition = {
        **condition,
        "params": {**registry.default_params(spec.fields), **condition["params"]},
    }
    document = _document(trigger=condition)
    report = validate_document(document)
    paths = [i.path for i in report.errors]
    assert any(p.endswith(f".{field}") for p in paths), _errors(report)
    with pytest.raises(ValueError):
        compiled = compile_document(document.model_copy(deep=True))
        instantiate_condition(compiled.pass_conditions[0], BuildContext(scenario=None))


@pytest.mark.usefixtures("unchecked")
@pytest.mark.parametrize(
    ("type_id", "actor", "params", "field"),
    [
        ("communication_loss", "ego", {"duration": 0.0}, "duration"),
        ("walk_straight", "ped", {"speed_ms": -1.0}, "speed_ms"),
        ("walk_straight", "ped", {"turn_back_within_m": -1.0}, "turn_back_within_m"),
        ("walk_straight", "ped", {"turn_back_ttc_s": -1.0}, "turn_back_ttc_s"),
        ("smart_pedestrian", "ped", {"speed_ms": 0.0}, "speed_ms"),
        ("smart_pedestrian", "ped", {"duration_s": 0.0}, "duration_s"),
        ("smart_pedestrian", "ped", {"prompts": [{"time_s": -1.0}]}, "time_s"),
        ("set_speed", "car", {"target_speed_kmh": -1.0}, "target_speed_kmh"),
        ("set_speed", "car", {"rate_kmh_s": 0.0}, "rate_kmh_s"),
        ("spawn", "car", {"length": -1.0}, "length"),
        (
            "stalled_object",
            "car",
            {"lanelet_id": _LANELET, "start_time": -1.0},
            "start_time",
        ),
        (
            "construction_zone",
            None,
            {"lanelet_id": _LANELET, "start_time": -1.0},
            "start_time",
        ),
        (
            "urban_construction_zone",
            None,
            {"lanelet_id": _LANELET, "start_time": -1.0},
            "start_time",
        ),
        ("traffic_source", None, {"vehicles_per_minute": -1.0}, "vehicles_per_minute"),
        ("traffic_source", None, {"initial_vehicles": -1}, "initial_vehicles"),
        (
            "routing",
            "car",
            {"goal_lanelet_id": _LANELET, "walking_speed": 0.0},
            "walking_speed",
        ),
        (
            "routing",
            "car",
            {"goal_lanelet_id": _LANELET, "initial_distance_offset": -1.0},
            "initial_distance_offset",
        ),
        ("traffic_parameters", None, {"speed_factor_mean": 0.0}, "speed_factor_mean"),
        ("traffic_parameters", None, {"ignore_lights_mean": 1.5}, "ignore_lights_mean"),
        (
            "traffic_parameters",
            None,
            {"speed_factor_mean": 1.0, "speed_factor_std": -0.1},
            "speed_factor_std",
        ),
        (
            "traffic_parameters",
            None,
            {"pedestrian_min_gap_m_mean": -1.0},
            "pedestrian_min_gap_m_mean",
        ),
        (
            "traffic_parameters",
            None,
            {"bicycle_max_speed_mps_mean": 0.1},
            "bicycle_max_speed_mps_mean",
        ),
        ("vehicle_parameters", "car", {"decel_mean": 0.0}, "decel_mean"),
        (
            "adversarial_traffic",
            None,
            {"adversities": [{"kind": "brake_check", "probability": 1.5}]},
            "probability",
        ),
    ],
)
def test_validator_refuses_a_bound_the_action_refuses(
    type_id: str,
    actor: Optional[str],
    params: dict[str, Any],
    field: str,
    samples: dict[tuple[str, str], Any],
) -> None:
    spec = registry.get_action_spec(type_id)
    assert spec is not None
    for key, value in samples.items():
        if key[0] == type_id:
            params = {key[1]: value, **params}
    for record in spec.fields:
        if record.kind == "record_list" and record.name in params:
            defaults = registry.default_params(record.item_fields)
            params[record.name] = [{**defaults, **row} for row in params[record.name]]
    document = _card(type_id, actor, **params)
    report = validate_document(document)
    paths = [i.path for i in report.errors]
    assert any(p.endswith(f".{field}") for p in paths), _errors(report)
    with pytest.raises(ValueError):
        _build(document)


def test_every_traffic_mean_field_has_the_runtimes_bounds() -> None:
    bounds = _traffic_mean_bounds()
    for type_id in ("traffic_parameters", "vehicle_parameters"):
        spec = registry.get_action_spec(type_id)
        assert spec is not None
        means = {f.name for f in spec.fields if f.name.endswith("_mean")}
        assert means <= set(bounds), means - set(bounds)


def _on_the_bounds(spec: FieldSpec) -> list[float]:
    """Values on or just inside *spec*'s bounds, which the runtime takes."""
    values: list[float] = []
    if spec.exclusive_minimum is not None:
        values.append(spec.exclusive_minimum + 1e-3)
    if spec.minimum is not None:
        values.append(spec.minimum)
    if spec.maximum is not None:
        values.append(spec.maximum)
    return values


#: What a card needs beyond its defaults to build, so a value on a bound is
#: built and not only validated.
_BOUNDS_BASE: dict[str, dict[str, Any]] = {
    "construction_zone": {"start_s": 0.0, "end_s": 50.0},
    "urban_construction_zone": {"start_s": 0.0, "end_s": 50.0},
    "traffic_source": {"constraints": "[{type: is_junction}]"},
    "routing": {"goal_lanelet_id": _LANELET},
    "spawn": {"name": None},
}


def _with_value(
    type_id: str,
    fields: tuple[FieldSpec, ...],
    name: str,
    value: float,
    samples: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    """The card's defaults, its blank required fields sampled, with *name* (a
    field or a record's) set."""
    params = registry.default_params(fields)
    for spec in fields:
        blank = params.get(spec.name) in (None, "", [], ())
        if spec.kind == "lanelet" and not params.get(spec.name):
            blank = True
        if spec.required and blank:
            params[spec.name] = _sample(type_id, spec, samples)
    params.update(_BOUNDS_BASE.get(type_id, {}))
    if name.endswith("_std"):
        # A std is given with its mean: the least mean the runtime takes.
        mean = name[: -len("_std")] + "_mean"
        params[mean] = _traffic_mean_bounds()[mean][0]
    for spec in fields:
        if spec.kind == "record_list" and any(i.name == name for i in spec.item_fields):
            rows = params.get(spec.name) or [registry.default_params(spec.item_fields)]
            params[spec.name] = [{**row, name: value} for row in rows]
    if name in params:
        params[name] = value
    return params


@pytest.mark.parametrize(
    ("card", "name", "value"),
    [
        (card, spec.name, value)
        for card in _ALL_SPECS
        for spec in _bounded(card.fields)
        for value in _on_the_bounds(spec)
    ],
    ids=lambda v: getattr(v, "type_id", str(v)),
)
def test_a_value_on_a_fields_bounds_is_accepted_and_builds(
    card: Any, name: str, value: float, samples: dict[tuple[str, str], Any]
) -> None:
    """The validator is no stricter than the runtime: a value on a bound (or
    just inside an exclusive one) is no error, and the card builds."""
    params = _with_value(card.type_id, card.fields, name, value, samples)
    logging.disable(logging.CRITICAL)
    try:
        report, build = _validated(card, params)
        assert not [i for i in report.errors if i.path.endswith(f".{name}")], _errors(
            report
        )
        if report.ok:
            assert _built_or_failure(build) is None
    finally:
        logging.disable(logging.NOTSET)


def _validated(card: Any, params: dict[str, Any]) -> tuple[ValidationReport, Any]:
    """The report on *card* carrying *params*, and how to build it."""
    action_spec = registry.get_action_spec(card.type_id)
    if action_spec is not None:
        actor = None
        if action_spec.scope != "environment":
            actor = _ENTITIES[action_spec.actor_kinds[0]]
        document = _card(card.type_id, actor, **params)
        return validate_document(document), lambda: _build(document)
    node: dict[str, Any] = {"type": card.type_id, "params": params}
    if card.min_children:
        node["children"] = [
            {"type": "elapsed_time", "params": {"duration_seconds": 1.0}}
        ] * card.min_children
    document = _document(trigger=node)

    def build() -> object:
        compiled = compile_document(document).pass_conditions[0]
        return instantiate_condition(compiled, BuildContext(scenario=None))

    return validate_document(document), build


# -- a field with no bound takes what the runtime takes (#1381) ---------------

#: Values outside the usual domain of a field: below 0, just below 0, above a
#: percentage, and far too large.  A field with no bound must build with each
#: one the validator lets through, or it is missing the runtime's bound.
_OUT_OF_DOMAIN: tuple[float, ...] = (-1.0, -0.001, 101.0, 1e12)

#: Unbounded fields whose refusal of a probe is the map's, not a bound's:
#: (card, field) -> why.
_NOT_A_BOUND: dict[tuple[str, str], str] = {
    ("temporary_stop_at_stop_lines", "depth"): (
        "a depth of 0 or less searches only the lanelet itself, which works "
        "on a lanelet with a stop line of its own; the fixture's has none"
    ),
}


def _unbounded(fields: tuple[FieldSpec, ...]) -> Iterator[FieldSpec]:
    for spec in fields:
        if spec.kind == "record_list":
            yield from _unbounded(spec.item_fields)
        elif (
            spec.kind in ("number", "int")
            and (
                spec.minimum,
                spec.exclusive_minimum,
                spec.maximum,
            )
            == (None,) * 3
        ):
            yield spec


def _probes(spec: FieldSpec) -> list[float]:
    """The out-of-domain values for *spec*: whole ones for a whole number."""
    if spec.kind == "int":
        return [float(int(v)) for v in _OUT_OF_DOMAIN if v == int(v)]
    return list(_OUT_OF_DOMAIN)


def _typed(spec: FieldSpec, value: float) -> Any:
    return int(value) if spec.kind == "int" else value


@pytest.mark.parametrize("card", _ALL_SPECS, ids=lambda s: s.type_id)
def test_an_accepted_value_of_an_unbounded_field_builds(
    card: Any, samples: dict[tuple[str, str], Any]
) -> None:
    """A field with no bound is one the runtime takes any finite value for:
    a value well outside its usual domain that the validator accepts builds."""
    failures = []
    logging.disable(logging.CRITICAL)
    try:
        for spec in _unbounded(card.fields):
            if (card.type_id, spec.name) in _NOT_A_BOUND:
                continue
            for value in _probes(spec):
                typed = _typed(spec, value)
                params = _with_value(
                    card.type_id, card.fields, spec.name, typed, samples
                )
                report, build = _validated(card, params)
                if not report.ok:
                    continue
                failure = _built_or_failure(build)
                if failure is not None:
                    failures.append(f"{spec.name}={typed!r}: {failure}")
    finally:
        logging.disable(logging.NOTSET)
    assert not failures, "\n".join(failures)


def _binding_values(spec: FieldSpec) -> list[float]:
    if (spec.minimum, spec.exclusive_minimum, spec.maximum) == (None,) * 3:
        return _probes(spec)
    return [*_on_the_bounds(spec), *_outside(spec)]


@pytest.mark.parametrize(
    ("binding", "name", "value"),
    [
        (binding, spec.name, _typed(spec, value))
        for binding in registry.binding_specs()
        for spec in binding.fields
        if spec.kind in ("number", "int")
        for value in _binding_values(spec)
    ],
    ids=lambda v: getattr(v, "type_id", str(v)),
)
def test_an_accepted_binding_parses(binding: Any, name: str, value: Any) -> None:
    """A binding the validator accepts is one the sweeper can parse, and a
    value outside a binding field's bounds is refused."""
    ref = BindingRef(
        type=binding.type_id,
        params={**registry.default_params(binding.fields), name: value},
    )
    out = _Collector()
    _check_binding(out, "binding", ref, binding.produces, "x")
    errors = [i for i in out.issues if i.severity == "error"]
    spec = next(s for s in binding.fields if s.name == name)
    if not _inside(spec, value):
        assert errors, (name, value)
    if not errors:
        assert (
            _built_or_failure(lambda: parse_binding("x", ref.to_sweep_dict())) is None
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (name, value)
        for name, (low, high) in sorted(_traffic_mean_bounds().items())
        for value in ([low] if high == float("inf") else [low, high])
    ],
)
def test_a_traffic_mean_on_its_bounds_is_accepted_and_builds(
    name: str, value: float
) -> None:
    document = _card("traffic_parameters", None, **{name: value})
    report = validate_document(document)
    assert report.ok, _errors(report)
    _build(document)


def test_a_whole_number_too_large_for_a_float_does_not_break_the_validator() -> None:
    document = _document(
        trigger={"type": "entity_count", "params": {"count": "9" * 400}}
    )
    validate_document(document)
