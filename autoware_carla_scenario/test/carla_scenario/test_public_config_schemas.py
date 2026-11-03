"""The public config classes hold the shipped YAMLs and the docs' defaults (#1289).

``docs/tutorial.md`` tells a scenario package to reuse the public config
classes as its structured schema, yet ``MapConfig`` refused
``map/nishishinjuku.yaml``'s ``overwrite_xodr`` and the ``no_3d_model_lanelet_ids``
every map sets, and ``DriverConfig`` refused ``driver/default.yaml``'s
``bridge`` and ``t4_scene*`` keys.  Every block of a shipped config that a
public class describes must merge into it, the default groups must set the
class's defaults, and the docs' parameter tables must name fields with their
defaults.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf

from autoware_carla_scenario import (
    DriverConfig,
    EgoVehicleConfig,
    EntityConfig,
    MapConfig,
    ServerConfig,
    TrafficManagerBackendConfig,
)
from autoware_carla_scenario.driver.base import DriverClientConfig

_PACKAGE = Path(__file__).resolve().parents[2]
_CONF = _PACKAGE / "src" / "autoware_carla_scenario" / "examples" / "conf"
_DOCS = _PACKAGE / "docs"

#: The public class each top-level config key is read with.
_SCHEMAS: dict[str, type] = {
    "server": ServerConfig,
    "map": MapConfig,
    "ego": EgoVehicleConfig,
    "entity": EntityConfig,
    "driver": DriverConfig,
    "traffic_manager": TrafficManagerBackendConfig,
}

#: The shipped files whose blocks are the defaults of their classes.
_DEFAULT_FILES = {
    "config.yaml",
    "driver/default.yaml",
    "ego/default.yaml",
    "entity/default.yaml",
    "server/localhost.yaml",
    "traffic_manager/default.yaml",
}

#: The docs whose parameter tables document these keys.
_DOCUMENTED = ("usage.md", "t4_scene.md", "t4_dataset.md")


def _shipped_blocks() -> list[tuple[str, str, dict[str, Any]]]:
    """(file relative to conf/, key, block) for every block a class describes."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for path in sorted(_CONF.rglob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key, block in document.items():
            if key in _SCHEMAS and isinstance(block, dict):
                out.append((path.relative_to(_CONF).as_posix(), key, block))
    return out


_BLOCKS = _shipped_blocks()


def _documented() -> list[tuple[str, str, str, Any]]:
    """(doc, group, key, default) for every row of the docs' tables."""
    row = re.compile(
        r"^\| `(" + "|".join(_SCHEMAS) + r")\.(\w+)` \| (?:[^|\\]|\\\|)+ \| "
        r"(`[^`]*`|\*\*required\*\*) \|",
        flags=re.MULTILINE,
    )
    out: list[tuple[str, str, str, Any]] = []
    for doc in _DOCUMENTED:
        for group, key, cell in row.findall((_DOCS / doc).read_text("utf-8")):
            value = "???" if cell == "**required**" else yaml.safe_load(cell[1:-1])
            out.append((doc, group, key, value))
    return out


_ROWS = _documented()


def test_every_group_with_a_public_class_is_covered() -> None:
    assert {key for _, key, _ in _BLOCKS} == set(_SCHEMAS)
    assert {"map/nishishinjuku.yaml", "map/town10hd_opt.yaml"} <= {
        name for name, key, _ in _BLOCKS if key == "map"
    }
    assert _DEFAULT_FILES <= {name for name, _, _ in _BLOCKS}


@pytest.mark.parametrize(
    ("name", "key", "block"),
    _BLOCKS,
    ids=[f"{name}:{key}" for name, key, _ in _BLOCKS],
)
def test_the_shipped_block_merges_into_its_public_class(
    name: str, key: str, block: dict[str, Any]
) -> None:
    schema = OmegaConf.structured(_SCHEMAS[key])
    merged = OmegaConf.merge(schema, block)
    if name in _DEFAULT_FILES:
        assert OmegaConf.to_container(merged) == OmegaConf.to_container(
            schema
        ), f"{name} sets a {key} value other than {_SCHEMAS[key].__name__}'s default"


def test_the_docs_tables_are_parsed() -> None:
    documented = {(group, key) for _, group, key, _ in _ROWS}
    assert {
        ("map", "name"),
        ("map", "projector_type"),
        ("map", "overwrite_xodr"),
        ("map", "no_3d_model_lanelet_ids"),
        ("driver", "t4_scene_map"),
    } <= documented


@pytest.mark.parametrize(
    ("doc", "group", "key", "value"),
    _ROWS,
    ids=[f"{doc}:{group}.{key}" for doc, group, key, _ in _ROWS],
)
def test_the_docs_tables_document_the_defaults(
    doc: str, group: str, key: str, value: Any
) -> None:
    defaults = OmegaConf.to_container(OmegaConf.structured(_SCHEMAS[group]))
    assert isinstance(defaults, dict)
    assert key in defaults, f"docs/{doc} documents {group}.{key}, no field"
    assert defaults[key] == value, f"docs/{doc}: {group}.{key}"


def test_every_map_field_is_documented() -> None:
    documented = {key for _, group, key, _ in _ROWS if group == "map"}
    assert documented == {f.name for f in dataclasses.fields(MapConfig)}


def _defaults(cls: type) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.default_factory is not dataclasses.MISSING:
            out[f.name] = f.default_factory()
        else:
            out[f.name] = f.default
    return out


def test_the_driver_schema_has_the_runtime_driver_fields() -> None:
    # DriverClientConfig is what the run builds from the `driver` block; the
    # controller (`control`) and the bridge (`bridge`) are read on their own.
    schema = _defaults(DriverConfig)
    runtime = _defaults(DriverClientConfig)
    assert set(schema) - {"control", "bridge"} == set(runtime)
    for key, value in runtime.items():
        if key in ("cameras", "lidars"):
            continue  # Specs here, DriverCameraConfig / DriverLidarConfig there.
        if isinstance(value, tuple):
            value = list(value)
        assert schema[key] == value, f"driver.{key}"


def test_the_driver_schema_takes_a_bridge_mapping() -> None:
    # driver/default.yaml's comment: a mapping takes AutowareBridgeConfig's fields.
    bridge = {"address": "0.0.0.0:50052", "ready_timeout_ticks": 1200}
    merged = OmegaConf.merge(OmegaConf.structured(DriverConfig), {"bridge": bridge})
    assert OmegaConf.to_container(merged.bridge) == bridge
    assert OmegaConf.structured(DriverConfig).bridge is None
