"""Logical scenarios from CodSceneClassifier scene labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.authoring.cod_import import (
    CodImportOptions,
    import_cod_output,
    main,
    scene_to_document,
)
from autoware_carla_scenario.authoring.hydra_config import build_scenario_config
from autoware_carla_scenario.authoring.models import ScenarioDocument
from autoware_carla_scenario.authoring.persistence import load_document
from autoware_carla_scenario.authoring.validator import validate_document

_NS = 1_000_000_000


def _scenery(signal: str | None = None, junction: str | None = None) -> dict[str, Any]:
    signal_element = (
        {
            "road_auxiliary_object_type": "signal",
            "signal_shape": "three_aspect",
            "signal_type": "vehicle_signal",
            "signal_color": signal,
        }
        if signal
        else None
    )
    return {
        "scenery_elements": [
            {
                "road_infrastructure": {
                    "road_auxiliary_objects": {"signal": signal_element}
                },
                "drivable_area_geometry": {
                    "horizontal_plane": {
                        "horizontal_plane_type": "straight",
                        "lane_curvature": 0.0,
                    },
                    "junctions": {
                        "intersection": {
                            "junction_type": junction,
                            "intersection_type": None,
                        }
                    },
                },
            }
        ]
    }


def _scene(
    lateral: str = "follow_lane",
    longitudinal: str = "driving_forward_keeping_speed",
    *,
    entities: list[dict[str, Any]] | None = None,
    signal: str | None = None,
    junction: str | None = None,
    seconds: float = 8.0,
    event: tuple[str, ...] = ("cruising",),
) -> dict[str, Any]:
    start = 1_788_854_180 * _NS
    return {
        "scene_id": "scene_003",
        "key_time": str(start),
        "start_time": str(start),
        "end_time": str(start + int(seconds * _NS)),
        "driving_decisions": {"lateral": lateral, "longitudinal": longitudinal},
        "event": list(event),
        "dynamic_entities": entities or [],
        "scenery": _scenery(signal, junction),
        "curved_ratio": 0.0,
    }


def _vehicle(
    x: float,
    lateral_position: str = "in_the_same_lane_as_the_subject_vehicle(s)",
    *,
    direction: str = "similar_as_the_subject_vehicle(s)",
    lateral: str = "follow_lane",
    longitudinal: str = "driving_forward_keeping_speed",
    context: str | None = None,
    velocity: float = 10.0,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "type": "vehicle",
        "position": {"x": x, "y": 0.0},
        "velocity": velocity,
        "behavior": {"lateral_action": lateral, "longitudinal_action": longitudinal},
        "role": "leading",
        "state_or_initial_state": {
            "initial_longitudinal_position": "in_front_of_the_subject_vehicle(s)",
            "initial_lateral_position": lateral_position,
            "initial_direction": direction,
        },
    }
    if context:
        raw["context"] = context
    return raw


def _errors(document: ScenarioDocument) -> list[str]:
    return [issue.message for issue in validate_document(document).errors]


def _sweep(document: ScenarioDocument) -> dict[str, Any]:
    return build_scenario_config(document)["sweep"]


class TestEgo:
    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_a_turn_searches_that_turn_and_starts_before_its_stop_line(
        self, direction: str
    ) -> None:
        document = scene_to_document(_scene(f"turn_{direction}")).document
        assert _errors(document) == []
        sweep = _sweep(document)
        (root,) = sweep["constraints"]["ego.spawn_lanelet_id"]
        assert {"type": "turn_direction", "value": direction} in root["constraints"]
        assert sweep["bindings"]["ego.spawn_s"]["type"] == "stop_line_offset"
        assert sweep["bindings"]["ego.goal_lanelet_id"]["type"] == "route_through"

    def test_a_turn_passes_once_the_ego_has_been_on_the_turn(self) -> None:
        document = scene_to_document(_scene("turn_left")).document
        (sticky,) = document.assertions.pass_conditions
        (on_turn,) = sticky.children
        assert on_turn.type == "entity_lane_position"
        binding = on_turn.searches["lanelet_id"].binding
        assert binding is not None and binding.type == "matched"

    def test_a_lane_change_is_sent_to_the_lane_beside(self) -> None:
        document = scene_to_document(_scene("change_lane_right")).document
        assert _errors(document) == []
        sweep = _sweep(document)
        (root,) = sweep["constraints"]["ego.spawn_lanelet_id"]
        assert {"type": "has_adjacent", "value": "right"} in root["constraints"]
        assert sweep["bindings"]["ego.goal_lanelet_id"] == {
            "type": "adjacent",
            "side": "right",
        }

    def test_stopping_at_a_red_light_searches_a_signalled_stop_line(self) -> None:
        document = scene_to_document(
            _scene(longitudinal="standing_still", signal="red")
        ).document
        assert _errors(document) == []
        (root,) = _sweep(document)["constraints"]["ego.spawn_lanelet_id"]
        assert {"type": "has_traffic_light_stop_line"} in root["constraints"]
        assert document.ego is not None and document.ego.initial_speed_kmh == 0.0
        (signal,) = [a for a in document.actions if a.type == "traffic_signal"]
        assert signal.params["state"] == "Red"
        (sticky,) = document.assertions.pass_conditions
        assert sticky.children[0].type == "standstill"

    def test_lane_following_passes_after_the_scene_length(self) -> None:
        document = scene_to_document(_scene(seconds=12.0)).document
        (elapsed,) = document.assertions.pass_conditions
        assert elapsed.type == "elapsed_time"
        assert elapsed.params["duration_seconds"] == 12.0
        assert document.timeout_seconds == 34.0

    def test_a_decision_without_an_equivalent_is_noted(self) -> None:
        imported = scene_to_document(_scene("pulling_over"))
        assert any("pulling_over" in note for note in imported.notes)
        assert "pulling_over" in imported.document.description

    def test_the_ego_is_driven_by_the_option(self) -> None:
        document = scene_to_document(
            _scene(), options=CodImportOptions(driven_by="autopilot")
        ).document
        assert document.ego is not None and document.ego.driven_by == "autopilot"


class TestVehicles:
    def test_a_leading_vehicle_is_placed_ahead_in_the_ego_lane(self) -> None:
        document = scene_to_document(_scene(entities=[_vehicle(20.0)])).document
        assert _errors(document) == []
        bindings = _sweep(document)["bindings"]
        # The ego starts 5 m into the pick, so 20 m ahead of it is 25 m in.
        assert bindings["scenario.spawn_overrides.npc1.lanelet_id"] == {
            "type": "route_offset",
            "distance": 25.0,
            "side": "same",
        }
        assert bindings["scenario.spawn_overrides.npc1.s"]["type"] == "route_offset_s"
        npc = document.entity("npc1")
        assert npc is not None and npc.initial_speed_kmh == 36.0

    def test_a_side_lane_is_required_of_the_search(self) -> None:
        document = scene_to_document(
            _scene(entities=[_vehicle(5.0, "to_the_left_of_the_subject_vehicle(s)")])
        ).document
        (root,) = _sweep(document)["constraints"]["ego.spawn_lanelet_id"]
        assert {"type": "has_adjacent", "value": "left"} in root["constraints"]

    def test_a_cut_in_changes_lane_towards_the_ego(self) -> None:
        document = scene_to_document(
            _scene(
                entities=[
                    _vehicle(
                        15.0,
                        "to_the_right_of_the_subject_vehicle(s)",
                        context="cutting_in",
                    )
                ]
            )
        ).document
        assert _errors(document) == []
        (lane_change,) = [a for a in document.actions if a.type == "lane_change"]
        assert lane_change.actor == "npc1"
        assert lane_change.params["direction"] == "left"
        assert lane_change.trigger is not None

    def test_an_oncoming_vehicle_is_placed_in_the_opposite_lane(self) -> None:
        document = scene_to_document(
            _scene(entities=[_vehicle(30.0, direction="oncoming")])
        ).document
        npc = document.entity("npc1")
        assert npc is not None and npc.spawn.binding is not None
        assert npc.spawn.binding.params["side"] == "opposite"

    def test_a_crossing_vehicle_is_left_out_and_noted(self) -> None:
        imported = scene_to_document(
            _scene(entities=[_vehicle(30.0, direction="crossing_from_left")])
        )
        assert imported.document.entity("npc1") is None
        assert any("crossing" in note for note in imported.notes)

    @pytest.mark.parametrize(
        ("longitudinal", "target"),
        [
            ("standing_still", 0.0),
            ("driving_forward_accelerating", 51.0),
            ("driving_forward_decelerating", 21.0),
            ("driving_forward_keeping_speed", 36.0),
        ],
    )
    def test_its_speed_follows_its_label(
        self, longitudinal: str, target: float
    ) -> None:
        document = scene_to_document(
            _scene(entities=[_vehicle(20.0, longitudinal=longitudinal)])
        ).document
        (set_speed,) = [a for a in document.actions if a.type == "set_speed"]
        assert set_speed.params["target_speed_kmh"] == target


class TestPedestrians:
    def _pedestrian(self, **extra: Any) -> dict[str, Any]:
        raw = {
            "type": "pedestrian",
            "position": {"x": 12.0, "y": 3.0},
            "velocity": 1.2,
            "context": "crossing_road",
            "state_or_initial_state": {
                "initial_longitudinal_position": "in_front_of_the_subject_vehicle(s)",
                "initial_lateral_position": "left_of_the_subject_vehicle(s)",
                "initial_direction": "crossing_from_left",
            },
        }
        raw.update(extra)
        return raw

    def test_a_crossing_walker_faces_across_and_walks(self) -> None:
        document = scene_to_document(_scene(entities=[self._pedestrian()])).document
        assert _errors(document) == []
        walker = document.entity("ped1")
        assert walker is not None and walker.kind == "pedestrian"
        assert walker.spawn.t == 3.0
        assert walker.spawn.heading == pytest.approx(-math.pi / 2)
        (walk,) = [a for a in document.actions if a.type == "walk_straight"]
        assert walk.actor == "ped1"

    def test_a_group_is_capped(self) -> None:
        imported = scene_to_document(
            _scene(entities=[self._pedestrian(pedestrian_count=7)])
        )
        walkers = [e for e in imported.document.entities if e.kind == "pedestrian"]
        assert [w.id for w in walkers] == ["ped1_1", "ped1_2", "ped1_3"]
        assert any("group of 7" in note for note in imported.notes)

    def test_a_standing_walker_does_not_walk(self) -> None:
        document = scene_to_document(
            _scene(entities=[self._pedestrian(context="standing_on_sidewalk")])
        ).document
        assert not [a for a in document.actions if a.type == "walk_straight"]


class TestOutput:
    def _output(self) -> dict[str, Any]:
        blacklisted = _scene()
        blacklisted.pop("driving_decisions")
        blacklisted["justification"] = "Pulled over"
        return {
            "date": "2026-09-14",
            "vehicle_id": "",
            "taxonomy_version": "0.1.1",
            "time_series": {
                "09-12-33": {
                    "name": "09-12-33",
                    "whitelist_scenes": [_scene("turn_left"), _scene()],
                    "blacklist_scenes": [blacklisted],
                }
            },
            "metadata": {},
        }

    def test_one_document_per_whitelisted_scene(self) -> None:
        imported = import_cod_output(self._output())
        assert len(imported) == 2
        assert imported[0].document.id == "cod_2026_09_14_09_12_33_scene_003"
        assert imported[0].session == "09-12-33"

    def test_blacklisted_scenes_on_request(self) -> None:
        imported = import_cod_output(
            self._output(), CodImportOptions(include_blacklist=True)
        )
        assert [i.blacklisted for i in imported] == [False, False, True]
        assert "Pulled over" in imported[2].document.title

    def test_the_cli_writes_documents_and_a_report(self, tmp_path: Path) -> None:
        source = tmp_path / "2026-09-14.json"
        source.write_text(json.dumps(self._output()), encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(source), "-o", str(out)]) == 0
        report = json.loads((out / "import_report.json").read_text(encoding="utf-8"))
        assert len(report) == 2
        document = load_document(Path(report[0]["document"]))
        assert _errors(document) == []
