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
from autoware_carla_scenario.authoring.models import LaneletChoice, ScenarioDocument
from autoware_carla_scenario.authoring.registry import FieldSpec
from autoware_carla_scenario.authoring.validator import (
    ValidationReport,
    validate_document,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import Lanelet2Pose

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
