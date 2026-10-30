"""ui.drive_metrics, split out of ui.results so the library check compiles
it (#741): ui.results still exports every name it did, and case_metrics reads
the same metrics, to the last bit, off every log it read them off before.

``data/ui_drive_metrics/golden.json`` holds what ui.results read before the
split, off the logs below; regenerate it only for a change meant to alter
them::

    uv run python autoware_carla_scenario/test/carla_scenario/test_ui_drive_metrics.py

(It imports ui.drive_metrics only in the tests that need it, so the golden
file can be captured with master's ui.results as well.)
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.ui import results

_GOLDEN = Path(__file__).resolve().parent / "data" / "ui_drive_metrics" / "golden.json"


def _actor(actor_id: int, role: str | None, extent: Any = None) -> dict[str, Any]:
    record: dict[str, Any] = {"type": "actor", "id": actor_id}
    if role is not None:
        record["role"] = role
    if extent is not None:
        record["extent"] = extent
    return record


def _state(
    actor_id: int,
    x: Any,
    y: Any,
    yaw: Any = 0.0,
    vx: Any = 0.0,
    vy: Any = 0.0,
) -> list[Any]:
    return [actor_id, x, y, 0.0, 0.0, 0.0, yaw, vx, vy, 0.0]


def _tick(t: Any, *states: list[Any], control: Any = None) -> dict[str, Any]:
    record: dict[str, Any] = {"type": "tick", "frame": 0, "t": t}
    record["actors"] = list(states)
    if control is not None:
        record["control"] = control
    return record


_WORLD = {"type": "world", "schema_version": 2}
_EGO = _actor(1, "Ego", [2.0, 1.0, 0.7])
_NPC = _actor(2, "npc1", [2.0, 1.0, 0.7])


def _drive(seed: int, ticks: int, others: int) -> list[Any]:
    """A seeded drive: the ego speeding up, braking hard, stopping and going
    again over uneven ticks, among *others* road users turned every way."""
    rng = random.Random(seed)
    records: list[Any] = [_WORLD, _EGO]
    for n in range(others):
        records.append(_actor(10 + n, f"npc{n}", [rng.uniform(0.2, 6.0), 1.1, 1.0]))
    t, x, y, speed = 0.0, 0.0, 0.0, 0.0
    for step in range(ticks):
        t += rng.choice((0.05, 0.05, 0.1, 0.0333, 0.2))
        phase = step / ticks
        if phase < 0.3:
            speed += rng.uniform(0.0, 0.8)
        elif phase < 0.5:
            speed = max(0.0, speed - rng.uniform(0.5, 2.5))
        elif phase < 0.6:
            speed = 0.0
        else:
            speed += rng.uniform(-0.3, 0.6)
        heading = rng.uniform(-3.0, 3.0)
        vx = speed * math.cos(math.radians(heading))
        vy = speed * math.sin(math.radians(heading))
        x += vx * 0.05
        y += vy * 0.05
        states = [_state(1, x, y, heading, vx, vy)]
        for n in range(others):
            states.append(
                _state(
                    10 + n,
                    x + rng.uniform(-15.0, 15.0),
                    y + rng.uniform(-6.0, 6.0),
                    rng.uniform(-180.0, 180.0),
                )
            )
        records.append(_tick(round(t, 4), *states, control={"1": [0.0, 0.0, 0.0]}))
    return records


def _approach(stop_short_of: float) -> list[Any]:
    """test_ui_results' log: 10 m/s, braking from t=1 to stop *stop_short_of*
    m behind an NPC standing at x=50."""
    records: list[Any] = [_WORLD, _EGO, _NPC]
    stop_at = 50.0 - 4.0 - stop_short_of
    decel = 10.0**2 / (2 * (stop_at - 10.0))
    for step in range(201):
        t = step * 0.05
        tau = max(0.0, t - 1.0)
        speed = max(0.0, 10.0 - decel * tau)
        tau = min(tau, 10.0 / decel)
        x = min(t, 1.0) * 10.0 + 10.0 * tau - decel * tau * tau / 2
        records.append(
            _tick(
                round(t, 2),
                _state(1, x, 0.0, 0.0, speed),
                _state(2, 50.0, 0.0),
                control={"1": [0.0, 0.0, 0.0, 1]},
            )
        )
    return records


#: name -> the log's lines: records (written as JSON) or raw text.
_LOGS: dict[str, list[Any]] = {
    "empty": [],
    "world_only": [_WORLD],
    "actors_but_no_ticks": [_WORLD, _EGO, _NPC],
    "ticks_without_an_ego": [
        _WORLD,
        _actor(2, "npc1", [2.0, 1.0, 0.7]),
        _tick(0.0, _state(2, 0.0, 0.0)),
        _tick(0.1, _state(2, 1.0, 0.0)),
    ],
    "ego_absent_from_its_ticks": [
        _WORLD,
        _EGO,
        _tick(0.0, _state(2, 0.0, 0.0)),
        _tick(0.1, _state(3, 1.0, 0.0)),
    ],
    "a_single_tick": [
        _WORLD,
        _EGO,
        _NPC,
        _tick(0.25, _state(1, 3.0, 1.0, 10.0, 4.0, 0.5), _state(2, 12.0, 4.0, 95.0)),
    ],
    "a_single_tick_alone": [
        _WORLD,
        _EGO,
        _tick(0.25, _state(1, 3.0, 1.0, 10.0, 4.0, 0.5)),
    ],
    "overlapping_boxes": [
        _WORLD,
        _EGO,
        _NPC,
        _tick(0.0, _state(1, 0.0, 0.0, 0.0, 3.0), _state(2, 3.0, 0.5, 30.0)),
        _tick(0.6, _state(1, 1.8, 0.0, 0.0, 1.0), _state(2, 3.0, 0.5, 30.0)),
    ],
    "touching_boxes": [
        _WORLD,
        _EGO,
        _NPC,
        _tick(0.0, _state(1, 0.0, 0.0), _state(2, 4.0, 0.0)),
    ],
    "separated_boxes": [
        _WORLD,
        _EGO,
        _NPC,
        _actor(3, "pedestrian", [0.3, 0.3, 0.9]),
        _tick(
            0.0,
            _state(1, 0.0, 0.0, 0.0, 5.0),
            _state(2, 0.0, 3.0, 180.0),
            _state(3, 7.3, -2.1, 45.0),
        ),
        _tick(
            0.5,
            _state(1, 2.5, 0.0, 0.0, 5.0),
            _state(2, 10.0, 0.0, 90.0),
            _state(3, 7.3, -2.1, 45.0),
        ),
        _tick(
            1.0,
            _state(1, 5.0, 0.2, 2.0, 4.9, 0.2),
            _state(2, 10.0, 3.7, 133.7),
            _state(3, 7.3, -2.1, -17.0),
        ),
    ],
    "the_ego_from_its_control_and_missing_extents": [
        _WORLD,
        _actor(1, None),
        _actor(2, "npc1", None),
        _tick(0.0, _state(1, 0.0, 0.0, 0.0, 6.0), control={"1": [0.0, 0.0, 0.0]}),
        _tick(
            0.5,
            _state(1, 3.0, 0.0, 0.0, 6.0),
            _state(2, 6.0, 1.0, 20.0),
            _state(4, -5.0, -1.0, 70.0),
            control={"1": [0.0]},
        ),
    ],
    "role_in_capitals_and_a_null_extent": [
        _WORLD,
        _actor(7, "EGO", None),
        {"type": "actor", "id": 8, "role": "npc", "extent": None},
        _tick(0.0, _state(7, 1.0, 1.0, 0.0, 2.0), _state(8, 1.5, 1.2)),
        _tick(1.0, _state(7, 3.0, 1.0, 0.0, 2.0), _state(8, 9.0, 1.2)),
    ],
    "integer_coordinates": [
        _WORLD,
        _EGO,
        _NPC,
        _tick(0, _state(1, 0, 0, 0, 10, 0), _state(2, 30, 3, 90), control={"1": [1]}),
        _tick(1, _state(1, 10, 0, 0, 8, 1), _state(2, 30, 3, 90)),
        _tick(2, _state(1, 18, 1, 5, 0, 0), _state(2, 30, 3, 90)),
        _tick(3, _state(1, 18, 1, 5, 0, 0), _state(2, 30, 3, 90)),
    ],
    "null_actors_and_control": [
        _WORLD,
        _EGO,
        {"type": "tick", "t": 0.0, "actors": None, "control": None},
        _tick(0.1, _state(1, 0.0, 0.0, 0.0, 1.0)),
    ],
    "records_naming_tick_and_actor": [
        _WORLD,
        _EGO,
        _NPC,
        {"type": "spawn", "t": 0.0, "label": "actor", "entity": "tick", "id": 2},
        _tick(0.0, _state(1, 0.0, 0.0, 0.0, 9.0), _state(2, 20.0, 0.0)),
        {"type": "collision", "t": 0.1, "actors": ["tick", "actor"], "ids": [2]},
        _tick(0.7, _state(1, 6.0, 0.0, 0.0, 0.1), _state(2, 20.0, 0.0)),
        "not json, and no record type the reader looks for",
    ],
    "stop_then_go_again": [
        _WORLD,
        _EGO,
        _tick(0.0, _state(1, 0.0, 0.0, 0.0, 3.0)),
        _tick(0.5, _state(1, 1.5, 0.0, 0.0, 0.05)),
        _tick(1.0, _state(1, 1.5, 0.0, 0.0, 0.0)),
        _tick(1.5, _state(1, 2.5, 0.0, 0.0, 4.0)),
        _tick(2.0, _state(1, 4.5, 0.0, 0.0, 0.01)),
    ],
    "time_running_backwards": [
        _WORLD,
        _EGO,
        _tick(2.0, _state(1, 0.0, 0.0, 0.0, 10.0)),
        _tick(1.0, _state(1, 5.0, 0.0, 0.0, 2.0)),
        _tick(3.0, _state(1, 9.0, 0.0, 0.0, 0.0)),
        _tick(2.2, _state(1, 9.0, 0.0, 0.0, 6.0)),
    ],
    "non_finite_values": [
        _WORLD,
        _EGO,
        _NPC,
        _tick(0.0, _state(1, 0.0, 0.0, 0.0, 1.0), _state(2, 5.0, 0.0)),
        _tick(0.6, _state(1, 1e300, -1e300, 0.0, 1.0), _state(2, 5.0, 0.0)),
        _tick(1.2, _state(1, 0.0, 0.0, 0.0, float("nan")), _state(2, 5.0, 0.0)),
    ],
    "two_actors_with_the_ego_role": [
        _WORLD,
        _EGO,
        _actor(2, "ego", [1.0, 0.5, 0.7]),
        _tick(0.0, _state(1, 0.0, 0.0, 0.0, 2.0), _state(2, 5.0, 0.0, 0.0, 8.0)),
        _tick(0.5, _state(1, 1.0, 0.0, 0.0, 2.0), _state(2, 9.0, 0.0, 0.0, 7.0)),
    ],
    "ticks_before_the_actor_records": [
        _WORLD,
        _tick(
            0.0, _state(1, 0.0, 0.0, 0.0, 6.0), _state(2, 8.0, 0.0), control={"1": []}
        ),
        _EGO,
        _NPC,
        _tick(0.5, _state(1, 3.0, 0.0, 0.0, 6.0), _state(2, 8.0, 0.0)),
    ],
    "velocities_given_as_text": [
        _WORLD,
        _EGO,
        _tick(0.0, _state(1, 0.0, 0.0, 0.0, "3.0", "-4")),
        _tick(0.5, _state(1, 1.5, -2.0, 0.0, "0.1", "0")),
    ],
    "an_approach_3m_short": _approach(3.0),
    "an_approach_0m_short": _approach(0.0),
    "a_seeded_drive": _drive(7, 400, 3),
    "a_seeded_crowd": _drive(41, 120, 9),
    # What the reader gives up on: no metrics at all.
    "a_tick_without_t": [
        _WORLD,
        _EGO,
        _tick(0.0, _state(1, 0.0, 0.0)),
        {"type": "tick", "actors": [_state(1, 1.0, 0.0)]},
    ],
    "a_position_given_as_text": [
        _WORLD,
        _EGO,
        _tick(0.0, _state(1, "1.0", 0.0)),
    ],
    "an_other_s_position_given_as_null": [
        _WORLD,
        _EGO,
        _tick(0.0, _state(1, 1.0, 0.0), _state(2, None, 0.0)),
    ],
    "a_short_state": [
        _WORLD,
        _EGO,
        _tick(0.0, _state(1, 0.0, 0.0)),
        _tick(0.1, [1, 1.0, 0.0]),
    ],
    "an_actor_without_an_id": [_WORLD, {"type": "actor", "role": "ego"}],
    "a_broken_tick_line": [_WORLD, _EGO, '{"type": "tick", "t": '],
}

#: Boxes (x, y, yaw, extent) whose distance the golden file holds.
_BOXES = [
    (0.0, 0.0, 0.0, (2.0, 1.0)),
    (3.0, 0.5, 30.0, (2.0, 1.0)),
    (0.0, 3.0, 180.0, (2.0, 1.0)),
    (10.0, 0.0, 90.0, (2.0, 1.0)),
    (-4.2, 7.7, -63.0, (0.3, 0.3)),
    (1.0, -2.0, 359.0, (2.4, 1.0)),
    (4.0, 0.0, 0.0, (2.0, 1.0)),
    (0.0, 0.0, 45.0, (0.0, 0.0)),
]


def _line(record: Any) -> str:
    return record if isinstance(record, str) else json.dumps(record)


def _render(directory: Path) -> dict[str, Any]:
    """Everything the golden file holds, read by ui.results."""
    out: dict[str, Any] = {}
    for name, records in _LOGS.items():
        log = directory / f"{name}.trajectory.jsonl"
        log.write_text("".join(_line(r) + "\n" for r in records), encoding="utf-8")
        out[name] = results.case_metrics(log)
    out["_box"] = [results._box(*b) for b in _BOXES]
    out["_box_distance"] = [
        results._box_distance(results._box(*a), results._box(*b))
        for a in _BOXES
        for b in _BOXES
    ]
    # The text JSON writes: every float to the last bit, NaN and the
    # infinities included, and the order case_metrics adds the keys in.
    return {key: json.dumps(value) for key, value in out.items()}


def _golden() -> dict[str, Any]:
    if not _GOLDEN.exists():  # being regenerated
        return {}
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _render(tmp_path_factory.mktemp("logs"))


@pytest.mark.parametrize("key", sorted(_golden()))
def test_case_metrics_reads_what_it_read_before_the_split(
    key: str, rendered: dict[str, Any]
) -> None:
    assert rendered[key] == _golden()[key]


def test_the_golden_file_covers_every_log() -> None:
    assert set(_golden()) == {*_LOGS, "_box", "_box_distance"}
    # The edge cases are what they claim to be.
    golden = {k: json.loads(v) for k, v in _golden().items()}
    assert golden["actors_but_no_ticks"] == {}
    assert golden["a_tick_without_t"] == {}
    assert set(golden["a_single_tick"]) == {
        "distance_m",
        "max_speed_kmh",
        "final_speed_kmh",
        "max_decel_ms2",
        "min_gap_m",
    }
    assert "min_gap_m" not in golden["a_single_tick_alone"]
    assert golden["overlapping_boxes"]["min_gap_m"] == 0.0
    assert golden["separated_boxes"]["min_gap_m"] > 0.0
    assert "stop_time_s" in golden["stop_then_go_again"]


@pytest.mark.parametrize(
    ("name", "moved"),
    [
        ("_DECEL_WINDOW_S", "DECEL_WINDOW_S"),
        ("_box", "box"),
        ("_box_distance", "box_distance"),
        ("_point_segment", "_point_segment"),
        ("_separated", "_separated"),
    ],
)
def test_results_re_exports_the_drive_metrics(name: str, moved: str) -> None:
    from autoware_carla_scenario.ui import drive_metrics  # noqa: PLC0415

    assert getattr(results, name) is getattr(drive_metrics, moved)


def test_results_still_exports_every_name() -> None:
    for name in (*results.__all__, "METRICS", "_metrics_cache"):
        assert hasattr(results, name), name


def test_no_tick_gives_no_metrics() -> None:
    from autoware_carla_scenario.ui import drive_metrics  # noqa: PLC0415

    metrics = drive_metrics.DriveMetrics()
    metrics.gap(1.0)
    assert metrics.to_dict() == {}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        rendered_now = _render(Path(tmp))
    _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN.write_text(
        json.dumps(rendered_now, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(f"wrote {_GOLDEN}\n")
