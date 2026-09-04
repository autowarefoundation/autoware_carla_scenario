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

    def test_a_non_entity_target_says_what_frame_it_is_in(self) -> None:
        """A bare id on a card belongs to no coordinate system on sight.

        This framework speaks both Lanelet2 and OpenDRIVE, and their ids are
        written the same way, so any target that is not an actor has to name
        what it is.
        """
        for spec in registry.condition_specs():
            target = spec.visual.target
            if target is None:
                continue
            field = next((f for f in spec.fields if f.name == target), None)
            if field is None or field.kind == "entity":
                continue
            assert spec.visual.target_prefix, (
                f"{spec.type_id} targets {target!r}, which is not an entity, "
                "without saying what it is"
            )

    def test_a_visual_detail_names_a_real_field(self) -> None:
        for spec in registry.condition_specs():
            if spec.visual.detail is None:
                continue
            assert any(f.name == spec.visual.detail for f in spec.fields), (
                f"{spec.type_id} shows a detail named {spec.visual.detail!r} "
                "that it has no field for"
            )

    def test_the_two_position_conditions_do_not_mix_frames(self) -> None:
        """Each frame gets its own condition, so neither can contradict itself.

        A lanelet already names one lane; an OpenDRIVE road does not, and needs
        a lane beside it. Offering all three fields at once let an author pin a
        lanelet and an unrelated lane on the same condition.
        """
        lanelet = registry.get_condition_spec("entity_lane_position")
        opendrive = registry.get_condition_spec("entity_road_position")
        assert lanelet is not None and opendrive is not None
        assert {f.name for f in lanelet.fields} == {"entity", "lanelet_id"}
        assert {f.name for f in opendrive.fields} == {"entity", "road_id", "lane_id"}

    def test_every_lanelet2_id_is_picked_off_the_map(self) -> None:
        """Nothing that names a Lanelet2 primitive may be a plain text box.

        Ids are not memorable, so typing one is guesswork; the map answers it
        exactly.  This test is the management: a new field naming a lanelet
        either declares a picker kind or is listed here with a reason, and
        adding one without doing either fails.
        """
        # Fields whose value is *not only* a Lanelet2 id, so a picker cannot
        # express the whole domain.  Each needs its escape spelled out before
        # it can be converted.
        allowed_text = {
            # "any" matches every lanelet -- a sentinel, not an id, and the
            # sweeper parses that exact string.
            ("equals", "value"),
            # Accepts ${map.no_3d_model_lanelet_ids}: a reference resolved at
            # sweep time, which is not a set of ids the map could highlight.
            ("in_set", "values"),
            # Measured, not assumed: the viewer's `regulatory` layer reports
            # the id of the linestring that draws a sign
            # ("linestring 1357 · traffic_sign/unknown"), not of the regulatory
            # element.  A picker there would save a confidently wrong number,
            # which is worse than a text box.
            ("traffic_signal", "lanelet2_regulatory_element_id"),
            ("traffic_signal", "lanelet2_traffic_light_ids"),
        }

        offenders = []
        for kind, specs in (
            ("condition", registry.condition_specs()),
            ("action", registry.action_specs()),
            ("constraint", registry.constraint_specs()),
            ("binding", registry.binding_specs()),
        ):
            for spec in specs:
                for field in spec.fields:
                    blob = f"{field.name} {field.label} {field.help}".lower()
                    if "lanelet" not in blob:
                        continue
                    if field.kind in ("lanelet", "lanelet_list"):
                        continue
                    if (spec.type_id, field.name) in allowed_text:
                        continue
                    offenders.append(f"{kind} {spec.type_id}.{field.name}")
        assert not offenders, (
            "these name a Lanelet2 primitive but cannot be picked off the map: "
            + ", ".join(offenders)
        )

    def test_a_picker_only_accepts_layers_that_report_a_lanelet(self) -> None:
        """A Lanelet2 map draws more than lanelets, and the layers overlap.

        A click a hair off the lane lands on a ``bound``, which reports the id
        of a linestring.  Saving that as a lanelet id would compile and then
        fail at scenario setup, so which layers count is part of the field's
        declaration.
        """
        # `bound` is deliberately absent: it reports a linestring id, not the
        # id of either lanelet it separates.
        lanelet_owned = {"lanelet_fill", "centerline", "direction"}
        for specs in (
            registry.condition_specs(),
            registry.action_specs(),
            registry.constraint_specs(),
            registry.binding_specs(),
        ):
            for spec in specs:
                for field in spec.fields:
                    if field.kind not in ("lanelet", "lanelet_list"):
                        continue
                    assert field.picks, f"{spec.type_id}.{field.name} picks nothing"
                    assert set(field.picks) <= lanelet_owned, (
                        f"{spec.type_id}.{field.name} accepts {field.picks!r}, "
                        "which includes a layer whose id is not a lanelet's"
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
