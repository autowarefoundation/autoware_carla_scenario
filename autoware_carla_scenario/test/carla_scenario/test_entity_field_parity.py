"""Every field of an authored entity reaches the run, or is refused (#960).

The model holds every field for every kind of entity, but not every kind reads
every field.  A value set where nothing reads it would be dropped in silence:
the validator accepted it, the editor showed it, and the run did something
else.  So each field, on each kind, is either carried -- it changes the
exported Hydra config or the entity the runtime builds -- or refused by the
validator with an error on that field.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from omegaconf import DictConfig, OmegaConf

from autoware_carla_scenario.authoring.compiler import compile_document
from autoware_carla_scenario.authoring.hydra_config import build_scenario_config
from autoware_carla_scenario.authoring.models import (
    AUTO_MODEL,
    BindingRef,
    ConstraintNode,
    Entity,
    EntitySize,
    GoalSpec,
    ScenarioDocument,
    SpawnSpec,
    SValue,
)
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.examples.run import build_ego_and_spawn

import autoware_carla_scenario.examples as examples_package

CONF = Path(examples_package.__file__).parent / "conf"

#: Where each kind sits in :func:`_base`'s entity list.
INDEX = {"ego": 0, "vehicle": 1, "pedestrian": 2}

#: Fields that are not values the run is configured with.
NOT_RUNTIME = {
    "id": "the entity's identity, required and with no default",
    "kind": "chooses which entity is built; every case below is run per kind",
    "title": "a display name for the editor and reports",
    "spawn": "a container; its own fields are checked one by one",
}


def _base() -> ScenarioDocument:
    return ScenarioDocument.model_validate(
        {
            "entities": [
                {"id": "ego", "kind": "ego", "spawn": {"lanelet_id": 242}},
                {"id": "car", "kind": "vehicle", "spawn": {"lanelet_id": 243}},
                {"id": "walker", "kind": "pedestrian", "spawn": {"lanelet_id": 244}},
            ],
            "assertions": {
                "pass": [{"type": "elapsed_time", "params": {"duration_seconds": 5.0}}],
                "fail": [{"type": "timeout", "params": {"timeout_seconds": 30.0}}],
            },
        }
    )


def _fake_world(pose: Lanelet2Pose) -> CarlaWorldPose:
    """A map conversion that keeps every part of *pose* visible."""
    return CarlaWorldPose(
        x=float(pose.lanelet_id), y=pose.s, z=pose.t, yaw=math.degrees(pose.heading)
    )


def _ego_spawn(export: dict[str, Any]) -> Lanelet2Pose:
    """The ego's spawn as the runner builds it from the exported keys."""
    defaults = OmegaConf.load(CONF / "ego" / "default.yaml")
    entity = OmegaConf.load(CONF / "entity" / "default.yaml")
    cfg = OmegaConf.merge(defaults, entity, {"ego": export.get("ego", {})})
    _, spawn, _ = build_ego_and_spawn(cfg)  # type: ignore[arg-type]
    return spawn


def _built(document: ScenarioDocument) -> dict[str, Any]:
    """What the runtime builds for every entity other than the ego."""
    from autoware_carla_scenario import declarative
    from autoware_carla_scenario.entity import fit

    scenario = object.__new__(declarative.DeclarativeScenario)
    scenario._document = document
    scenario._compiled = compile_document(document)
    scenario._ground_projection = None  # type: ignore[assignment]
    scenario.ego_config = SimpleNamespace(  # type: ignore[assignment]
        spawn_retry_max_count=0, spawn_retry_t_step=0.0, spawn_retry_z_step=0.0
    )
    built: dict[str, Any] = {}
    with (
        mock.patch(
            "autoware_carla_scenario.coordinate.transform.to_carla_world",
            side_effect=_fake_world,
        ),
        mock.patch(
            "autoware_carla_scenario.coordinate.transform.to_opendrive",
            side_effect=repr,
        ),
        mock.patch.object(
            declarative,
            "snap_to_carla_road",
            side_effect=lambda pose, world, **_: _fake_world(pose),
        ),
        mock.patch.object(
            fit,
            "fitted_vehicle",
            side_effect=lambda world, length, width: f"vehicle.fit_{length}x{width}",
        ),
    ):
        for entity in scenario._compiled.npcs:
            if entity.kind == "pedestrian":
                made: Any = scenario._build_pedestrian(entity, object())
            else:
                made = scenario._build_npc(entity, object())
            built[entity.id] = {
                f.name: repr(getattr(made._config, f.name))
                for f in dataclasses.fields(made._config)
            }
    return built


def _runtime(document: ScenarioDocument) -> dict[str, Any]:
    """Everything of *document* that reaches a run."""
    export = build_scenario_config(document)
    return {"export": export, "ego": _ego_spawn(export), "built": _built(document)}


# ---------------------------------------------------------------------------
# The cases: (kind, field, outcome, setup, change)
# ---------------------------------------------------------------------------

Mutation = Callable[[Entity, ScenarioDocument], None]


def _nothing(entity: Entity, document: ScenarioDocument) -> None:
    del entity, document


def _set(**values: Any) -> Mutation:
    def apply(entity: Entity, document: ScenarioDocument) -> None:
        del document
        for name, value in values.items():
            target: Any = entity
            *path, last = name.split("__")
            for part in path:
                target = getattr(target, part)
            setattr(target, last, value)

    return apply


def _both(*mutations: Mutation) -> Mutation:
    def apply(entity: Entity, document: ScenarioDocument) -> None:
        for mutation in mutations:
            mutation(entity, document)

    return apply


def _searched(value: str = "243") -> Mutation:
    return _set(
        spawn__mode="constraint_search",
        spawn__constraints=[ConstraintNode(type="equals", params={"value": value})],
    )


def _derived_offset(offset: float) -> Mutation:
    return _set(
        spawn__s=SValue(
            mode="derived",
            value=1.0,
            binding=BindingRef(type="stop_line_offset", params={"offset": offset}),
        )
    )


def _derived_lanelet(side: str) -> Mutation:
    """Derive *entity*'s lanelet from a search on another entity's spawn."""

    def apply(entity: Entity, document: ScenarioDocument) -> None:
        other = document.entities[1 if entity.kind == "ego" else 0]
        _searched()(other, document)
        entity.spawn.mode = "derived"
        entity.spawn.binding = BindingRef(type="adjacent", params={"side": side})

    return apply


def _cases() -> Iterator[tuple[str, str, str, Mutation, Mutation]]:
    model = {"ego": "vehicle.tesla.model3", "vehicle": "vehicle.tesla.model3"}
    for kind in INDEX:
        walker = kind == "pedestrian"
        ego = kind == "ego"
        yield (
            kind,
            "vehicle_type",
            "carried",
            _nothing,
            _set(vehicle_type="walker.pedestrian.0002" if walker else model[kind]),
        )
        yield (
            kind,
            "size",
            "refused",
            _nothing,
            _set(size=EntitySize(length=4.5, width=1.8)),
        )
        yield (
            kind,
            "initial_speed_kmh",
            "refused" if walker else "carried",
            _nothing,
            _set(initial_speed_kmh=5.0),
        )
        goal = _set(goal=GoalSpec(lanelet_id=265, s=1.0))
        yield (
            kind,
            "driven_by",
            "carried" if ego else "refused",
            goal if ego else _nothing,
            _set(driven_by="autoware"),
        )
        yield (kind, "goal", "carried" if ego else "refused", _nothing, goal)
        yield (
            kind,
            "spawn.lanelet_id",
            "carried",
            _nothing,
            _set(spawn__lanelet_id=250),
        )
        yield (kind, "spawn.t", "carried", _nothing, _set(spawn__t=1.5))
        yield (kind, "spawn.heading", "carried", _nothing, _set(spawn__heading=0.5))
        yield (
            kind,
            "spawn.hidden",
            "refused" if ego else "carried",
            _nothing,
            _set(spawn__hidden=True),
        )
        yield (kind, "spawn.mode", "carried", _nothing, _searched())
        yield (kind, "spawn.constraints", "carried", _searched(), _searched("250"))
        yield (
            kind,
            "spawn.binding",
            "carried",
            _derived_lanelet("right"),
            _derived_lanelet("left"),
        )
        yield (kind, "spawn.s.value", "carried", _nothing, _set(spawn__s__value=7.5))
        yield (kind, "spawn.s.mode", "carried", _searched(), _derived_offset(10.0))
        yield (
            kind,
            "spawn.s.binding",
            "carried",
            _both(_searched(), _derived_offset(10.0)),
            _derived_offset(15.0),
        )
    # A size is read to choose an ``auto`` vehicle, and for nothing else.
    auto = _set(vehicle_type=AUTO_MODEL, size=EntitySize(length=4.5, width=1.8))
    yield (
        "vehicle",
        "size",
        "carried",
        auto,
        _set(size=EntitySize(length=12.0, width=2.5)),
    )


CASES = list(_cases())


def _prepared(kind: str, setup: Mutation, change: Mutation | None) -> ScenarioDocument:
    document = _base()
    entity = document.entities[INDEX[kind]]
    setup(entity, document)
    if change is not None:
        change(entity, document)
    return document


@pytest.mark.parametrize(
    ("kind", "field", "outcome", "setup", "change"),
    CASES,
    ids=[f"{kind}-{field}-{outcome}" for kind, field, outcome, _, _ in CASES],
)
def test_a_field_reaches_the_run_or_is_refused(
    kind: str, field: str, outcome: str, setup: Mutation, change: Mutation
) -> None:
    before = _prepared(kind, setup, None)
    after = _prepared(kind, setup, change)
    path = f"entities[{INDEX[kind]}].{field.split('.')[0]}"
    errors = validate_document(after).errors
    on_field = [e for e in errors if e.path.startswith(path)]
    if outcome == "refused":
        assert on_field, f"{kind}.{field} is neither carried nor refused"
        return
    assert validate_document(before).errors == ()
    assert errors == (), [(e.path, e.message) for e in errors]
    assert _runtime(after) != _runtime(
        before
    ), f"{kind}.{field} is accepted but never reaches the run"


def test_every_field_of_every_kind_has_a_case() -> None:
    """A field added to the model fails here until it is carried or refused."""
    wanted = (
        {name for name in Entity.model_fields if name not in NOT_RUNTIME}
        | {f"spawn.{name}" for name in SpawnSpec.model_fields if name != "s"}
        | {f"spawn.s.{name}" for name in SValue.model_fields}
    )
    covered: dict[str, set[str]] = {kind: set() for kind in INDEX}
    for kind, field, _, _, _ in CASES:
        covered[kind].add(field)
    for kind in INDEX:
        assert covered[kind] >= wanted, (kind, wanted - covered[kind])
        assert covered[kind] <= wanted, (kind, covered[kind] - wanted)


# ---------------------------------------------------------------------------
# The ego's spawn offset and heading
# ---------------------------------------------------------------------------


def _with_ego_spawn(t: float, heading: float) -> ScenarioDocument:
    document = _base()
    ego = document.ego
    assert ego is not None
    ego.spawn = SpawnSpec(lanelet_id=242, s=SValue(value=5.0), t=t, heading=heading)
    return document


class TestTheEgoSpawn:
    def test_offset_and_heading_are_exported(self) -> None:
        ego = build_scenario_config(_with_ego_spawn(1.5, 0.5))["ego"]
        assert ego["spawn_t"] == 1.5
        assert ego["spawn_heading"] == 0.5

    def test_the_runner_builds_the_pose_from_them(self) -> None:
        export = build_scenario_config(_with_ego_spawn(-0.75, math.pi))
        assert _ego_spawn(export) == Lanelet2Pose(
            lanelet_id=242, s=5.0, t=-0.75, heading=math.pi
        )

    def test_a_config_without_them_spawns_on_the_centreline(self) -> None:
        cfg = OmegaConf.merge(
            OmegaConf.load(CONF / "entity" / "default.yaml"),
            {
                "ego": {
                    "vehicle_type": "vehicle.mini.cooper",
                    "initial_speed_kmh": 0.0,
                    "spawn_lanelet_id": 7,
                    "spawn_s": 2.0,
                }
            },
        )
        _, spawn, _ = build_ego_and_spawn(cfg)  # type: ignore[arg-type]
        assert spawn == Lanelet2Pose(lanelet_id=7, s=2.0, t=0.0, heading=0.0)

    @pytest.mark.parametrize(
        "path", [CONF / "ego" / "default.yaml", CONF / "config.yaml"]
    )
    def test_the_keys_are_declared(self, path: Path) -> None:
        """Hydra's struct mode accepts only a key a config has declared."""
        config = OmegaConf.load(path)
        assert isinstance(config, DictConfig)
        ego = config["ego"]
        assert ego["spawn_t"] == 0.0
        assert ego["spawn_heading"] == 0.0

    def test_the_schema_declares_them(self) -> None:
        from autoware_carla_scenario.scenario_config import EgoVehicleConfig

        schema = OmegaConf.structured(EgoVehicleConfig)
        merged = OmegaConf.merge(schema, {"spawn_t": 1.0, "spawn_heading": 0.2})
        assert merged.spawn_t == 1.0
        assert merged.spawn_heading == 0.2


# ---------------------------------------------------------------------------
# The refusals, one by one
# ---------------------------------------------------------------------------


def _errors_at(document: ScenarioDocument, path: str) -> list[str]:
    return [e.message for e in validate_document(document).errors if e.path == path]


class TestRefused:
    def test_a_pedestrian_has_no_initial_speed(self) -> None:
        document = _base()
        document.entities[2].initial_speed_kmh = 5.0
        (message,) = _errors_at(document, "entities[2].initial_speed_kmh")
        assert "walk" in message

    def test_a_standing_pedestrian_is_valid(self) -> None:
        assert validate_document(_base()).errors == ()

    def test_only_the_ego_is_driven_by_what_the_document_names(self) -> None:
        document = _base()
        document.entities[1].driven_by = "autoware"
        (message,) = _errors_at(document, "entities[1].driven_by")
        assert "Only the ego" in message

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_a_size_beside_a_named_blueprint(self, index: int) -> None:
        document = _base()
        document.entities[index].size = EntitySize(length=4.5, width=1.8)
        (message,) = _errors_at(document, f"entities[{index}].size")
        assert "ignored" in message

    def test_a_size_with_auto_is_read(self) -> None:
        document = _base()
        document.entities[1].vehicle_type = AUTO_MODEL
        document.entities[1].size = EntitySize(length=4.5, width=1.8)
        assert validate_document(document).errors == ()
