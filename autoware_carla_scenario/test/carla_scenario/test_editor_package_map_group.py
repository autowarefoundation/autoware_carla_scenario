"""The editor's map preview sees scenario packages' map groups (#2035).

The ``scenario-editor`` process never ran ``load_scenario_plugins()``, so a
``map`` group a package ships (its ``conf/map/<group>.yaml``) was not found and
its ``no_3d_model_lanelet_ids`` read as ``[]``: the preview matched lanelets a
run of the package excludes.  The lookup now loads the packages itself when
the framework has no such group.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import pytest

import autoware_carla_scenario.registry as registry
from autoware_carla_scenario.authoring.hydra_config import (
    MAP_EXCLUSION_REF,
    effective_no_3d_model_lanelet_ids,
    map_group_no_3d_model_lanelet_ids,
)
from autoware_carla_scenario.authoring.models import ConstraintNode
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.editor import map_preview


@pytest.fixture
def plugin_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """An installed scenario package whose ``conf/map/plugmap.yaml`` excludes
    lanelets 11 and 22, not yet loaded.

    Its entry point is the only one installed; the registry's conf dirs and
    its loaded-once latch are restored after the test.  Returns the names of
    the entry points loaded, in order.
    """
    conf_dir = tmp_path / "plug_conf"
    (conf_dir / "map").mkdir(parents=True)
    (conf_dir / "map" / "plugmap.yaml").write_text(
        "# @package _global_\nmap:\n  name: Plug\n  no_3d_model_lanelet_ids: [11, 22]\n",
        encoding="utf-8",
    )
    loads: list[str] = []

    def register() -> None:
        loads.append("plug_pkg")
        registry.register_conf_dir(conf_dir)

    entry_point = importlib.metadata.EntryPoint(
        name="plug_pkg",
        value="plug_pkg:register",
        group=registry.SCENARIO_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(importlib.metadata.EntryPoint, "load", lambda self: register)

    def fake_entry_points(**kwargs: Any) -> list[importlib.metadata.EntryPoint]:
        if kwargs.get("group") == registry.SCENARIO_ENTRY_POINT_GROUP:
            return [entry_point]
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    monkeypatch.setattr(registry, "_CONF_DIRS", list(registry.get_conf_dirs()))
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    return loads


def test_package_map_group_is_found_without_loading_plugins_first(
    plugin_loads: list[str],
) -> None:
    assert map_group_no_3d_model_lanelet_ids("plugmap") == [11, 22]
    assert plugin_loads == ["plug_pkg"]
    # Once per process.
    assert map_group_no_3d_model_lanelet_ids("plugmap") == [11, 22]
    assert plugin_loads == ["plug_pkg"]


def test_builtin_map_group_loads_no_plugin(plugin_loads: list[str]) -> None:
    assert map_group_no_3d_model_lanelet_ids("nishishinjuku")
    assert plugin_loads == []


def test_unknown_map_group_is_none(plugin_loads: list[str]) -> None:
    assert map_group_no_3d_model_lanelet_ids("no_such_group") is None
    assert plugin_loads == ["plug_pkg"]


def test_editor_preview_honours_the_package_groups_exclusions(
    plugin_loads: list[str],
) -> None:
    document = new_document()
    document.map.group = "plugmap"
    document.map.no_3d_model_lanelet_ids = []
    assert effective_no_3d_model_lanelet_ids(document) == [11, 22]

    nodes = [
        ConstraintNode(
            type="not",
            constraints=[
                ConstraintNode(type="in_set", params={"values": MAP_EXCLUSION_REF})
            ],
        )
    ]
    (tree,) = map_preview.materialize_constraints(nodes, document)
    assert MAP_EXCLUSION_REF not in repr(tree)
    assert "[11, 22]" in repr(tree)
