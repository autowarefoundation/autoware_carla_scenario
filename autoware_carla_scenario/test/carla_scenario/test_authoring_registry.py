"""The metadata registry is the contract between the runtime and the GUI.

These tests pin the parts a template or a builder relies on: that every
registered primitive has a builder, that a condition's visual actually names
fields the primitive has, and that the option lists spelled out in the registry
still match the runtime enums they mirror.
"""

from __future__ import annotations

import pytest

from autoware_carla_scenario.authoring import builders, registry
from autoware_carla_scenario.conditions.comparison import ComparisonRule


class TestSpecCoverage:
    def test_every_action_has_a_builder(self) -> None:
        for spec in registry.action_specs():
            assert hasattr(builders, spec.builder), (
                f"action {spec.type_id!r} names builder {spec.builder!r}, "
                "which does not exist"
            )

    def test_every_condition_has_a_builder(self) -> None:
        for spec in registry.condition_specs():
            assert hasattr(builders, spec.builder), (
                f"condition {spec.type_id!r} names builder {spec.builder!r}, "
                "which does not exist"
            )

    def test_specs_are_unique_and_titled(self) -> None:
        for specs in (
            registry.action_specs(),
            registry.condition_specs(),
            registry.constraint_specs(),
            registry.binding_specs(),
        ):
            ids = [s.type_id for s in specs]
            assert len(ids) == len(set(ids))
            assert all(s.title for s in specs)


class TestConditionVisuals:
    """A visual that names a field the primitive lacks renders as blank."""

    @pytest.mark.parametrize(
        "spec", registry.condition_specs(), ids=lambda s: s.type_id
    )
    def test_visual_names_real_fields(self, spec: registry.ConditionSpec) -> None:
        names = {f.name for f in spec.fields}
        visual = spec.visual
        for attribute in ("subject", "target", "rule", "value"):
            referenced = getattr(visual, attribute)
            if referenced is not None:
                assert referenced in names, (
                    f"{spec.type_id}.visual.{attribute} names {referenced!r}, "
                    f"which is not one of its fields {sorted(names)}"
                )

    def test_every_condition_has_a_metric(self) -> None:
        for spec in registry.condition_specs():
            assert spec.visual.metric


class TestOptionSets:
    """The registry spells out enums it cannot import; keep them in step."""

    def test_comparison_rules_match_the_runtime_enum(self) -> None:
        assert {o.value for o in registry.COMPARISON_RULES} == {
            rule.name.lower() for rule in ComparisonRule
        }

    def test_traffic_light_states_match_carla(self) -> None:
        carla = pytest.importorskip("carla")
        declared = {o.value for o in registry.TRAFFIC_LIGHT_STATES}
        available = {
            name
            for name in dir(carla.TrafficLightState)
            if not name.startswith("_") and name[0].isupper()
        }
        assert declared <= available, declared - available


class TestSelectFieldDefaults:
    def test_select_defaults_are_valid_options(self) -> None:
        for specs in (
            registry.action_specs(),
            registry.condition_specs(),
            registry.constraint_specs(),
            registry.binding_specs(),
        ):
            for spec in specs:
                for field_spec in spec.fields:
                    if field_spec.kind != "select":
                        continue
                    allowed = {o.value for o in field_spec.options}
                    assert field_spec.default in allowed, (
                        f"{spec.type_id}.{field_spec.name} defaults to "
                        f"{field_spec.default!r}, not one of {sorted(allowed)}"
                    )

    def test_default_params_covers_every_field(self) -> None:
        for spec in registry.condition_specs():
            params = registry.default_params(spec.fields)
            assert set(params) == {f.name for f in spec.fields}


class TestConstraintVocabulary:
    """The GUI must not invent constraints the sweeper cannot parse."""

    def test_every_constraint_is_known_to_the_sweeper(self) -> None:
        from autoware_carla_scenario.sweeper.constraints import _LEAF_REGISTRY

        composites = {"and", "or", "not", "previous_of", "following_of"}
        parseable = set(_LEAF_REGISTRY) | composites
        registered = {s.type_id for s in registry.constraint_specs()}
        assert registered <= parseable, registered - parseable

    def test_every_binding_is_known_to_the_sweeper(self) -> None:
        from autoware_carla_scenario.sweeper.bindings import _BINDING_REGISTRY

        registered = {s.type_id for s in registry.binding_specs()}
        assert registered <= set(_BINDING_REGISTRY)
