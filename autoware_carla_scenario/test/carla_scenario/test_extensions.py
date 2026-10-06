"""Vocabulary a separate package adds through the extensions entry point."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pytest

from autoware_carla_scenario import extensions
from autoware_carla_scenario.authoring import registry
from autoware_carla_scenario.sweeper import (
    bindings,
    constraints,
    parse_binding,
    parse_constraint,
    register_binding,
    register_constraint,
)
from autoware_carla_scenario.sweeper.bindings import BindingResult


@dataclass(frozen=True)
class _Everything:
    def evaluate(self, lanelet: Any) -> bool:
        return True


@dataclass
class _Same:
    target_key: str

    def resolve(self, lanelet_id: int, lanelet_map: Any, routing_graph: Any = None):
        return BindingResult(value=lanelet_id)


def _register() -> None:
    register_constraint("ext_everything", _Everything)
    register_binding("ext_same", _Same)
    registry.register_constraint_spec(
        registry.ConstraintSpec(
            type_id="ext_everything", title="Everything", category="Extension"
        )
    )


class _EntryPoint:
    name = "test_extension"
    value = "tests:_register"

    def load(self) -> Any:
        return _register


@pytest.fixture
def an_installed_extension(monkeypatch: pytest.MonkeyPatch):
    def entry_points(*, group: str) -> list[Any]:
        return (
            [_EntryPoint()] if group == extensions.EXTENSION_ENTRY_POINT_GROUP else []
        )

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    monkeypatch.setattr(extensions, "_loaded", False)
    yield
    constraints._LEAF_REGISTRY.pop("ext_everything", None)
    bindings._BINDING_REGISTRY.pop("ext_same", None)
    registry._CONSTRAINT_SPECS.pop("ext_everything", None)


@pytest.mark.usefixtures("an_installed_extension")
class TestAnInstalledExtension:
    def test_its_constraint_is_parsed_on_first_use(self) -> None:
        assert isinstance(parse_constraint({"type": "ext_everything"}), _Everything)

    def test_its_binding_is_parsed_on_first_use(self) -> None:
        assert isinstance(parse_binding("k", {"type": "ext_same"}), _Same)

    def test_its_spec_is_offered_to_the_editor(self) -> None:
        assert registry.get_constraint_spec("ext_everything") is not None
        assert "ext_everything" in {s.type_id for s in registry.constraint_specs()}


def test_an_unknown_type_is_still_refused() -> None:
    with pytest.raises(ValueError, match="Unknown constraint type"):
        parse_constraint({"type": "no_such_constraint"})
