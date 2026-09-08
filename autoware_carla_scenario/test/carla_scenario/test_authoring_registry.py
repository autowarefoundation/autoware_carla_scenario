"""The metadata registry is the contract between the runtime and the GUI.

These tests pin the parts a template or a builder relies on: that every
registered primitive has a builder, that a condition's visual actually names
fields the primitive has, and that the option lists spelled out in the registry
still match the runtime enums they mirror.

They also guard the generated half.  Most builders are rendered from the
constructor a spec targets, so the committed module has to stay what the
generator would write -- the same guarantee ``test_proto_generated`` gives the
compiled protobufs, for the same reason: a stale generated file is a file that
disagrees with its source and nothing else notices.
"""

from __future__ import annotations

import pytest

from autoware_carla_scenario.authoring import builders, registry
from autoware_carla_scenario.conditions.comparison import ComparisonRule

#: Every primitive that has a builder, actions and conditions alike.  The two
#: spec types share no base class, so the annotation is what lets a test read
#: `spec.choices` off the joined list.
_ALL_SPECS: list[registry.ActionSpec | registry.ConditionSpec] = [
    *registry.action_specs(),
    *registry.condition_specs(),
]


class TestSpecCoverage:
    def test_every_action_has_a_builder(self) -> None:
        for spec in registry.action_specs():
            # Through `_resolve`, because a builder may be hand-written or
            # generated and the spec deliberately does not say which.
            builders._resolve(spec.builder)

    def test_every_runtime_condition_is_reachable_from_the_editor(self) -> None:
        """A new condition class must be exposed, or say why it is not.

        The existing checks only look one way -- every spec has a builder --
        so a condition added to the runtime and never registered simply never
        appears in the editor, and nothing says so.  A document naming it
        cannot be validated, compiled or exported: `validate_document` reports
        "Unknown condition type" and the inspector shows it as unknown.

        This is the other direction.  Adding a class and forgetting the spec
        fails here, with the two ways to make it pass spelled out.
        """
        from autoware_carla_scenario import conditions
        from autoware_carla_scenario.conditions.base import BaseCondition

        # Empty on purpose, as for actions: every condition the runtime has is
        # authorable.  The last exception, `EntityInAreaCondition`, was removed
        # rather than excused -- its polygon lived in absolute world
        # coordinates, so it could not survive a constraint sweep and had no
        # place in an abstract scenario.
        self._assert_every_class_is_built(conditions, BaseCondition, set(), "condition")

    def test_every_runtime_action_is_reachable_from_the_editor(self) -> None:
        """The same guarantee for actions."""
        from autoware_carla_scenario import actions
        from autoware_carla_scenario.actions.base import BaseAction

        # Empty on purpose: every action the runtime has is authorable.
        self._assert_every_class_is_built(actions, BaseAction, set(), "action")

    @staticmethod
    def _assert_every_class_is_built(
        module: object, base: type, runtime_only: set[str], noun: str
    ) -> None:
        """Fail unless every exported subclass of *base* is built or excused.

        Membership is read from the builders' own imports rather than from a
        second list, so there is nothing extra to keep in step: a builder that
        constructs a class necessarily imports it by name.
        """
        import ast
        import inspect
        import pathlib

        from autoware_carla_scenario.authoring import _builders_generated, builders

        built = {
            alias.name
            for source in (builders, _builders_generated)
            for node in ast.walk(
                ast.parse(pathlib.Path(str(source.__file__)).read_text())
            )
            if isinstance(node, ast.ImportFrom) and node.level == 2
            for alias in node.names
        }
        exported = {
            name
            for name, obj in vars(module).items()
            if inspect.isclass(obj) and issubclass(obj, base) and obj is not base
        }

        # An excuse for a class that no longer exists silently weakens the
        # guard: it would go on excusing a *future* class that happens to be
        # given the same name.  Deleting the sensor attach actions left exactly
        # this behind.
        stale = sorted(runtime_only - exported)
        assert not stale, (
            f"these {noun}s are excused but no longer exist: {stale}. "
            f"Remove them from `runtime_only`."
        )

        missing = sorted(exported - built - runtime_only)
        assert not missing, (
            f"these {noun}s exist in the runtime but no builder constructs them, "
            f"so a document naming one cannot be edited, validated or exported: "
            f"{missing}. Register a {noun} spec and a builder, or add the class "
            f"to `runtime_only` in this test with the reason it is not editable."
        )

    def test_every_condition_has_a_builder(self) -> None:
        for spec in registry.condition_specs():
            # Through `_resolve`, because a builder may be hand-written or
            # generated and the spec deliberately does not say which.
            builders._resolve(spec.builder)

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
            names = {f.name for f in spec.fields}
            unknown = sorted(set(spec.visual.details) - names)
            assert (
                not unknown
            ), f"{spec.type_id} shows details {unknown} that it has no fields for"

    def test_the_two_position_conditions_do_not_mix_frames(self) -> None:
        """Each frame gets its own condition, so neither can contradict itself.

        A lanelet already names one lane; an OpenDRIVE road does not, and needs
        a lane beside it. Offering all three fields at once let an author pin a
        lanelet and an unrelated lane on the same condition.
        """
        lanelet = registry.get_condition_spec("entity_lane_position")
        opendrive = registry.get_condition_spec("entity_road_position")
        assert lanelet is not None and opendrive is not None
        lanelet_fields = {f.name for f in lanelet.fields}
        opendrive_fields = {f.name for f in opendrive.fields}

        # Each addresses a place in one frame only.  Asserted as "neither names
        # the other's address" rather than as an exact field list, so a field
        # both frames share -- the `s`/`t` bounds, say -- does not read as a
        # regression.
        assert "lanelet_id" in lanelet_fields
        assert not {"road_id", "lane_id"} & lanelet_fields
        assert {"road_id", "lane_id"} <= opendrive_fields
        assert "lanelet_id" not in opendrive_fields

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
                    # Name and label only.  Prose mentions lanelets without
                    # holding one -- a margin measured from a lanelet's start
                    # is a distance, not an id -- and matching on help text
                    # made the guard fire on exactly that.
                    named = f"{field.name} {field.label}".lower()
                    if "lanelet" not in named:
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

    def test_only_lanelet_owned_layers_count_as_a_lanelet_pick(self) -> None:
        """A Lanelet2 map draws more than lanelets, and the layers overlap.

        A click a hair off the lane lands on a ``bound``, which reports the id
        of a linestring.  Saving that as a lanelet id would compile and then
        fail at scenario setup, so the accepted layers are pinned here --
        ``bound`` deliberately absent.
        """
        from autoware_carla_scenario.editor.app import LANELET_PICK_LAYERS

        assert set(LANELET_PICK_LAYERS) == {
            "lanelet_fill",
            "centerline",
            "direction",
        }

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


class TestGeneratedBuilders:
    """The committed builders must stay what the generator would write."""

    def test_committed_module_is_not_stale(self) -> None:
        """A spec change that is not regenerated is a spec change with no effect.

        The diff is printed rather than a bare "out of date" because the useful
        question on a failure is *which* primitive moved.
        """
        codegen = pytest.importorskip(
            "autoware_carla_scenario.authoring.codegen",
            reason="the generator introspects the runtime, which needs CARLA",
        )
        stale = codegen.diff_against_disk(codegen.generate())
        assert not stale, (
            "_builders_generated.py disagrees with the registry. Regenerate it:\n"
            "    uv run python -m autoware_carla_scenario.authoring.codegen\n\n"
            f"{stale}"
        )

    @pytest.mark.parametrize(
        "spec",
        _ALL_SPECS,
        ids=lambda s: s.type_id,
    )
    def test_a_spec_is_generated_or_hand_written_but_not_both(
        self, spec: registry.ActionSpec | registry.ConditionSpec
    ) -> None:
        """A ``target`` and a hand-written builder would both claim the name.

        ``_resolve`` searches the generated module first, so a hand-written
        builder shadowed by a generated one of the same name would simply never
        run -- and would go on looking like the code that does.
        """
        from autoware_carla_scenario.authoring import builders as hand_written

        hand = getattr(hand_written, spec.builder, None)
        assert bool(spec.target) != callable(hand), (
            f"{spec.type_id!r} must either name a target (and be generated) or "
            f"define {spec.builder!r} by hand, not both and not neither"
        )

    @pytest.mark.parametrize(
        "spec",
        [spec for spec in _ALL_SPECS if spec.choices],
        ids=lambda s: s.type_id,
    )
    def test_every_choice_covers_every_option(
        self, spec: registry.ActionSpec | registry.ConditionSpec
    ) -> None:
        """Guards the check itself, not just its result.

        Exhaustiveness is enforced inside the generator, where a gap raises
        rather than falling through to a wrong value.  Asserting it here too
        means a future option added without a case fails on the spec that is
        wrong, not on a diff in a generated file.
        """
        fields = {field.name: field for field in spec.fields}
        for choice in spec.choices:
            options = {o.value for o in fields[choice.discriminator].options}
            cased = {case.when for case in choice.cases}
            assert cased == options, (
                f"{spec.type_id!r}: {choice.kwarg!r} has cases for {sorted(cased)} "
                f"but {choice.discriminator!r} offers {sorted(options)}"
            )


class TestBuiltArguments:
    """The `builds` a spec declares are the part a signature cannot describe.

    They used to be hand-written builders, exempt from every check in this
    file. Now that the generator renders them, the semantics they encode --
    which bound becomes which comparison, which pose an address is -- are worth
    stating once here, because nothing else says them.
    """

    @staticmethod
    def _build(type_id: str, kwarg: str) -> registry.BuiltArgument:
        spec = registry.get_condition_spec(type_id) or registry.get_action_spec(type_id)
        assert spec is not None, type_id
        return next(b for b in spec.builds if b.kwarg == kwarg)

    @pytest.mark.parametrize(
        "type_id", ["entity_lane_position", "entity_road_position"]
    )
    def test_every_bound_becomes_the_rule_it_reads_as(self, type_id: str) -> None:
        build = self._build(type_id, "rules")

        seen = []
        for part in build.parts:
            constants = dict(part.constants)
            seen.append(
                (
                    part.when_present,
                    constants["field"].strip('"'),
                    constants["rule"].rsplit(".", 1)[-1],
                )
            )

        assert seen == [
            ("s_min", "s", "GREATER_THAN_OR_EQUAL"),
            ("s_max", "s", "LESS_THAN_OR_EQUAL"),
            ("t_min", "t", "GREATER_THAN_OR_EQUAL"),
            ("t_max", "t", "LESS_THAN_OR_EQUAL"),
        ]

    def test_a_lanelet_position_is_an_address_not_a_place_along_one(self) -> None:
        """`s` on the pose is 0.0: where along the lane is said by the bounds."""
        build = self._build("entity_lane_position", "position")

        assert build.target.endswith(":Lanelet2Pose")
        assert dict(build.parts[0].constants)["s"] == "0.0"
        assert dict(build.parts[0].args) == {"lanelet_id": "lanelet_id"}

    def test_an_opendrive_position_names_a_road_and_a_lane(self) -> None:
        """Neither alone determines a pose, so the spec asks for both."""
        build = self._build("entity_road_position", "position")
        spec = registry.get_condition_spec("entity_road_position")
        assert spec is not None

        assert dict(build.parts[0].args) == {
            "road_id": "road_id",
            "lane_id": "lane_id",
        }
        lane = next(f for f in spec.fields if f.name == "lane_id")
        assert lane.required, "a road without a lane does not name a place"


class TestNothingIsHandWrittenAnyMore:
    """Every spec is generated, and the exemption should not come back quietly.

    A hand-written builder is exempt from every check in this file: exhaustive
    select cases, fields that reach no argument, options drifted from their
    enum. That is the cost of the exemption, and it is why `builds` exists --
    what a signature cannot describe, the spec says instead.
    """

    def test_every_spec_names_a_target(self) -> None:
        specs: list[registry.ActionSpec | registry.ConditionSpec] = [
            *registry.condition_specs(),
            *registry.action_specs(),
        ]
        without = sorted(spec.type_id for spec in specs if not spec.target)
        assert not without, (
            f"these specs are hand-written: {without}. Describe what the "
            f"signature cannot with `builds`, or say here why this one is "
            f"genuinely beyond it."
        )

    def test_the_hand_written_module_defines_no_builder(self) -> None:
        from autoware_carla_scenario.authoring import builders

        defined = [name for name in vars(builders) if name.startswith("build_")]
        assert not defined, f"hand-written builders are back: {defined}"
