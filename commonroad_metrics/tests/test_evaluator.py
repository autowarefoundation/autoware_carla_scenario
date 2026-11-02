"""The measures on a frame of known answer.

``data/straight.xml`` is roadgen's CommonRoad export of one straight road from
``(0, 0)`` to ``(240, 180)``: two lanes forward, to the right of the reference
line, and one backward.  It heads 36.87 degrees off +x, so a measure that read
the speeds against the lane's own heading (as CriMe does unless the frame is
turned) would be off by its cosine, 0.8.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

from commonroad_metrics import CRIME_MEASURES, MEASURES, Actor, Evaluator, EvaluatorSettings
from commonroad_metrics.worker import serve

MAP = Path(__file__).parent / "data" / "straight.xml"
COS, SIN = 0.8, 0.6
HEADING = math.atan2(SIN, COS)


def on_road(distance: float, right: float, speed: float, actor_id: int) -> Actor:
    """A car *distance* metres along the road, *right* metres to the right of
    its reference line, going *speed* m/s forward."""
    return Actor(
        actor_id,
        "car",
        distance * COS + right * SIN,
        distance * SIN - right * COS,
        HEADING,
        vx=speed * COS,
        vy=speed * SIN,
        length=4.5,
        width=1.8,
    )


@pytest.fixture(scope="module")
def evaluator() -> Evaluator:
    return Evaluator(str(MAP))


def test_following_a_slower_car(evaluator: Evaluator) -> None:
    # 30 m apart centre to centre, a 25.5 m gap; closing at 10 m/s.
    ego = on_road(50, 1.75, 15.0, 1)
    lead = on_road(80, 1.75, 5.0, 2)
    values, errors = evaluator.evaluate(
        ego, [lead], ["crime_ttc_s", "crime_hw_m", "crime_a_long_req_mps2", "crime_thw_s"]
    )
    assert errors == {}
    assert values["crime_hw_m"] == pytest.approx(25.5, abs=0.1)
    assert values["crime_ttc_s"] == pytest.approx(2.55, abs=0.05)
    # Relative speed to zero over the gap: 10^2 / (2 * 25.5).
    assert values["crime_a_long_req_mps2"] == pytest.approx(-1.96, abs=0.05)
    assert values["crime_thw_s"] == pytest.approx(1.8, abs=0.15)


def test_a_car_in_another_lane_is_no_conflict(evaluator: Evaluator) -> None:
    ego = on_road(50, 1.75, 15.0, 1)
    beside = on_road(80, 5.25, 5.0, 2)
    values, _ = evaluator.evaluate(ego, [beside], ["crime_ttc_s"])
    assert values["crime_ttc_s"] == math.inf


def test_the_reachable_set_shrinks_behind_a_stopped_car(evaluator: Evaluator) -> None:
    ego = on_road(50, 1.75, 15.0, 1)
    free, _ = evaluator.evaluate(ego, [], ["reach_drivable_area_m2"])
    blocked, _ = evaluator.evaluate(
        ego,
        [on_road(70, 1.75, 0.0, 2), on_road(70, 5.25, 0.0, 3)],
        ["reach_drivable_area_m2"],
    )
    assert free["reach_drivable_area_m2"] > 0.0
    assert blocked["reach_drivable_area_m2"] < free["reach_drivable_area_m2"]


def test_off_the_map_reads_nothing(evaluator: Evaluator) -> None:
    ego = Actor(1, "car", -500.0, -500.0, 0.0, vx=10.0)
    values, errors = evaluator.evaluate(ego, [], MEASURES)
    assert errors == {}
    assert values == {key: None for key in MEASURES}


def test_an_unknown_measure_is_an_error(evaluator: Evaluator) -> None:
    values, errors = evaluator.evaluate(on_road(50, 1.75, 10.0, 1), [], ["nope"])
    assert values == {}
    assert errors == {"nope": "unknown measure"}


def test_the_worker_answers_line_by_line() -> None:
    requests = [
        {"op": "evaluate", "measures": ["crime_ttc_s"], "ego": {}},
        {"op": "init", "map": str(MAP), "settings": {"horizon_steps": 20}},
        {
            "op": "evaluate",
            "measures": ["crime_ttc_s"],
            "ego": on_road(50, 1.75, 15.0, 1).__dict__,
            "others": [on_road(80, 1.75, 5.0, 2).__dict__],
        },
        {"op": "close"},
        {"op": "init", "map": str(MAP)},  # after close: not read
    ]
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    serve(stdin, stdout)
    answers = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(answers) == 4
    assert answers[0] == {"ok": False, "error": "evaluate before init"}
    assert answers[1]["ok"] and set(answers[1]["measures"]) == set(MEASURES)
    assert answers[2]["ok"]
    assert answers[2]["values"]["crime_ttc_s"] == pytest.approx(2.55, abs=0.05)
    assert answers[3] == {"ok": True}


def test_infinity_crosses_the_wire() -> None:
    requests = [
        {"op": "init", "map": str(MAP)},
        {"op": "evaluate", "measures": ["crime_ttc_s"], "ego": on_road(50, 1.75, 15.0, 1).__dict__},
    ]
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    serve(stdin, stdout)
    last = stdout.getvalue().splitlines()[-1]
    # Alone on the road: no one to measure from.
    assert json.loads(last)["values"]["crime_ttc_s"] is None


def test_sharing_the_frame_changes_no_value() -> None:
    """CriMe's measures read the same values sharing a frame's scenario as
    copying it for each, on frames of road users ahead, beside, behind,
    braking and slower."""
    shared = Evaluator(str(MAP))
    copied = Evaluator(str(MAP), EvaluatorSettings(share_frame=False))
    crime = list(CRIME_MEASURES)
    frames = [
        (on_road(50, 1.75, 15.0, 1), [on_road(80, 1.75, 5.0, 2)]),
        (on_road(40, 5.25, 12.0, 1), [on_road(55, 5.25, 12.0, 2), on_road(60, 1.75, 2.0, 3)]),
        (
            on_road(100, 1.75, 20.0, 1),
            [on_road(130, 1.75, 10.0, 2), on_road(90, 1.75, 25.0, 3), on_road(120, 5.25, 0.0, 4)],
        ),
    ]
    braking = Actor(**{**on_road(70, 1.75, 10.0, 5).__dict__, "ax": -3.0 * COS, "ay": -3.0 * SIN})
    frames.append((on_road(50, 1.75, 14.0, 1), [braking]))
    for ego, others in frames:
        assert shared.evaluate(ego, others, crime) == copied.evaluate(ego, others, crime)
