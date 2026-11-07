"""A hand-written document's unquoted ``Off`` / ``off`` / ``OFF`` is text (#1946).

PyYAML's safe loader is YAML 1.1, which reads a plain ``yes`` / ``no`` /
``on`` / ``off`` as a boolean -- and several documented options are exactly
those words.  A document is read with YAML 1.2's booleans instead: only
``true`` / ``false``.  A ``bool`` field still reads the other spellings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from autoware_carla_scenario.authoring.compiler import compile_document
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    JunctionSegment,
    LaneSegment,
    ScenarioDocument,
    SignalControllerRef,
    SignalGroupRef,
    SignalPhaseRef,
    SignalStateRef,
)
from autoware_carla_scenario.authoring.persistence import (
    DraftStore,
    dump_document_yaml,
    load_document,
    load_yaml,
)
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.editor.service import parse_route_yaml


def _hand_written(document: ScenarioDocument, *words: str) -> str:
    """*document*'s YAML with each of *words* written unquoted, as by hand."""
    text = dump_document_yaml(document)
    for word in words:
        quoted = f"'{word}'"
        assert quoted in text, f"the dumper no longer quotes {word!r}"
        text = text.replace(quoted, word)
    return text


def _load(tmp_path: Path, text: str) -> ScenarioDocument:
    path = tmp_path / "document.yaml"
    path.write_text(text, encoding="utf-8")
    return load_document(path)


def _errors(document: ScenarioDocument) -> list[str]:
    return [f"{i.path}: {i.message}" for i in validate_document(document).errors]


class TestLoader:
    def test_only_true_and_false_are_booleans(self) -> None:
        assert load_yaml(
            "[Off, off, OFF, On, yes, No, y, n, true, True, TRUE, false, FALSE]"
        ) == [
            "Off",
            "off",
            "OFF",
            "On",
            "yes",
            "No",
            "y",
            "n",
            True,
            True,
            True,
            False,
            False,
        ]

    def test_everything_else_resolves_as_before(self) -> None:
        text = "{a: 1, b: 1.5, c: null, d: ~, e: !!bool yes, f: 0x10, g: text}"
        assert load_yaml(text) == yaml.safe_load(text)

    def test_pyyaml_s_own_loader_is_untouched(self) -> None:
        load_yaml("a: Off")
        assert yaml.safe_load("a: Off") == {"a": False}


class TestDocumentOptions:
    def test_an_unquoted_traffic_signal_off_loads_and_validates(
        self, tmp_path: Path
    ) -> None:
        document = new_document()
        document.actions.append(
            ActionNode(
                id="lights",
                type="traffic_signal",
                params={"state": "Off", "target": "all"},
            )
        )
        text = _hand_written(document, "Off")
        assert "state: Off" in text
        loaded = _load(tmp_path, text)
        action = next(a for a in loaded.actions if a.id == "lights")
        assert action.params["state"] == "Off"
        assert _errors(loaded) == []

    def test_an_unquoted_turn_indicator_off_loads_and_validates(
        self, tmp_path: Path
    ) -> None:
        document = new_document()
        document.actions.append(
            ActionNode(
                id="indicators",
                type="turn_indicator",
                actor="npc1",
                params={"state": "OFF"},
            )
        )
        loaded = _load(tmp_path, _hand_written(document, "OFF"))
        action = next(a for a in loaded.actions if a.id == "indicators")
        assert action.params["state"] == "OFF"
        assert _errors(loaded) == []

    def test_an_unquoted_phase_state_off_loads(self, tmp_path: Path) -> None:
        document = new_document()
        document.map.signal_groups = [
            SignalGroupRef(name="ns", lanelet2_regulatory_element_ids=[1001])
        ]
        document.map.traffic_signal_controllers = [
            SignalControllerRef(
                name="crossing",
                phases=[
                    SignalPhaseRef(
                        name="dark",
                        duration_seconds=5.0,
                        states=[SignalStateRef(group="ns", state="off")],
                    )
                ],
            )
        ]
        text = _hand_written(document, "off")
        assert "state: off" in text
        loaded = _load(tmp_path, text)
        phase = loaded.map.traffic_signal_controllers[0].phases[0]
        assert phase.states[0].state == "off"
        assert _errors(loaded) == []

    def test_a_flow_phase_as_the_docs_write_it_loads(self) -> None:
        raw = load_yaml("{group: ns, state: off}")
        assert SignalStateRef.model_validate(raw).state == "off"

    def test_a_draft_reads_an_unquoted_off_too(self, tmp_path: Path) -> None:
        store = DraftStore(tmp_path)
        document = new_document()
        document.actions.append(
            ActionNode(
                id="lights",
                type="traffic_signal",
                params={"state": "Off", "target": "all"},
            )
        )
        draft = store.create(document, title="dark")
        path = store.path_for(draft.id)
        path.write_text(
            path.read_text(encoding="utf-8").replace("'Off'", "Off"),
            encoding="utf-8",
        )
        loaded = store.get(draft.id)
        assert loaded is not None
        action = next(a for a in loaded.document.actions if a.id == "lights")
        assert action.params["state"] == "Off"

    def test_saving_round_trips(self, tmp_path: Path) -> None:
        document = new_document()
        document.actions.append(
            ActionNode(
                id="lights",
                type="traffic_signal",
                params={"state": "Off", "target": "all"},
            )
        )
        loaded = _load(tmp_path, _hand_written(document, "Off"))
        again = _load(tmp_path, dump_document_yaml(loaded))
        assert again.model_dump() == loaded.model_dump()
        # What is written still reads the same under YAML 1.1 (OmegaConf).
        assert "state: 'Off'" in dump_document_yaml(again)


class TestBoolFields:
    @pytest.mark.parametrize("word", ["yes", "on", "true", "Yes", "ON", "True"])
    def test_a_bool_param_spelt_any_way_reads_true(
        self, tmp_path: Path, word: str
    ) -> None:
        document = new_document()
        document.actions.append(
            ActionNode(
                id="lights",
                type="traffic_signal",
                params={"state": "Red", "target": "all", "freeze": "FREEZE"},
            )
        )
        text = dump_document_yaml(document).replace("freeze: FREEZE", f"freeze: {word}")
        loaded = _load(tmp_path, text)
        assert _errors(loaded) == []
        compiled = compile_document(loaded)
        action = next(a for a in compiled.actions if a.node.id == "lights")
        assert action.params["freeze"] is True

    @pytest.mark.parametrize(
        ("word", "expected"),
        [("yes", True), ("on", True), ("true", True)]
        + [("no", False), ("off", False), ("false", False)],
    )
    def test_a_bool_model_field_spelt_any_way_reads_as_it_says(
        self, tmp_path: Path, word: str, expected: bool
    ) -> None:
        document = new_document()
        document.actions[0].overrides = True
        text = dump_document_yaml(document)
        assert "overrides: true" in text
        loaded = _load(tmp_path, text.replace("overrides: true", f"overrides: {word}"))
        assert loaded.actions[0].overrides is expected

    def test_a_route_segment_s_yes_and_no_read_as_written(self) -> None:
        route = parse_route_yaml(
            "segments:\n"
            "  - {kind: lane, opposite_lane: yes, stop_line: no}\n"
            "  - {kind: junction, traffic_light: Yes, crossing_from_left: OFF}\n"
        )
        assert route is not None
        lane, junction = route.segments
        assert isinstance(lane, LaneSegment)
        assert isinstance(junction, JunctionSegment)
        assert (lane.opposite_lane, lane.stop_line) == ("yes", "no")
        # The spellings YAML 1.1 read as a boolean still mean what they did.
        assert (junction.traffic_light, junction.crossing_from_left) == ("yes", "no")
