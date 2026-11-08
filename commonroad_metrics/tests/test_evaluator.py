"""The measures on a frame of known answer.

``data/straight.xml`` is roadgen's CommonRoad export of one straight road from
``(0, 0)`` to ``(240, 180)``: two lanes forward, to the right of the reference
line, and one backward.  It heads 36.87 degrees off +x, so a measure that read
the speeds against the lane's own heading (as CriMe does unless the frame is
turned) would be off by its cosine, 0.8.
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
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
    frames.append((on_road(50, 1.75, 14.0, 1), [braking(on_road(70, 1.75, 10.0, 5), 3.0)]))
    oncoming = replace(on_road(90, 1.75, -10.0, 6), yaw=HEADING + math.pi)
    frames.append((on_road(50, 1.75, 10.0, 1), [oncoming]))
    for ego, others in frames:
        assert shared.evaluate(ego, others, crime) == copied.evaluate(ego, others, crime)


def braking(actor: Actor, deceleration: float) -> Actor:
    """*actor* braking at *deceleration* m/s^2 along the road."""
    return replace(actor, ax=-deceleration * COS, ay=-deceleration * SIN)


def test_a_braking_lead_closes_the_gap() -> None:
    """CriMe 0.4 reads the lead's deceleration without its sign, as speeding
    up (TTC ``inf``, ALongReq ``0``); read with it, both copying and sharing
    the frame."""
    ego = on_road(50, 1.75, 15.0, 1)
    # A 20 m gap centre to centre less the two half lengths, the same speed.
    lead = braking(on_road(74.5, 1.75, 15.0, 2), 5.0)
    keys = ["crime_hw_m", "crime_ttc_s", "crime_a_long_req_mps2", "crime_btn"]
    for share_frame in (True, False):
        values, errors = Evaluator(str(MAP), EvaluatorSettings(share_frame=share_frame)).evaluate(
            ego, [lead], keys
        )
        assert errors == {}
        assert values["crime_hw_m"] == pytest.approx(20.0, abs=0.1)
        # 20 = 0.5 * 5 * t^2, before the lead stops (at 3 s).
        assert values["crime_ttc_s"] == pytest.approx(math.sqrt(8.0), abs=0.02)
        # The lead stops 22.5 m on: the ego stops behind it braking at
        # 15^2 / (2 * 42.5), less than the lead's own 5 m/s^2 (which CriMe's
        # constant acceleration, the lead never stopping, would ask for).
        assert values["crime_a_long_req_mps2"] == pytest.approx(-225.0 / 85.0, abs=0.01)
        assert values["crime_btn"] == pytest.approx(2.65 / 11.5, abs=1e-3)


def test_a_lead_stopping_before_the_ego_reaches_it(evaluator: Evaluator) -> None:
    """A lead braking to a stop stays where it stops rather than reversing,
    as the worker predicts every road user."""
    ego = on_road(50, 1.75, 5.0, 1)
    # A 10 m gap; the lead stops after 0.5 s and 0.5 m.
    lead = braking(on_road(64.5, 1.75, 2.0, 2), 4.0)
    values, errors = evaluator.evaluate(
        ego, [lead], ["crime_hw_m", "crime_ttc_s", "crime_a_long_req_mps2"]
    )
    assert errors == {}
    assert values["crime_hw_m"] == pytest.approx(10.0, abs=0.1)
    # The ego reaches where the lead stopped, 10.5 m on: not the 1.61 s of a
    # lead reversing.
    assert values["crime_ttc_s"] == pytest.approx(2.1, abs=0.02)
    assert values["crime_a_long_req_mps2"] == pytest.approx(-25.0 / 21.0, abs=0.01)


def test_a_queue_creeping_to_a_halt_is_no_conflict(evaluator: Evaluator) -> None:
    """A stopped ego 2 m behind a car creeping to a halt (4 cm on): CriMe's
    constant acceleration would bring the car back into the ego."""
    ego = on_road(50, 1.75, 0.0, 1)
    lead = braking(on_road(56.5, 1.75, 0.5, 2), 3.0)
    keys = ["crime_ttc_s", "crime_a_long_req_mps2", "crime_btn", "crime_a_lat_req_mps2"]
    values, errors = evaluator.evaluate(ego, [lead], keys)
    assert errors == {}
    assert values == {
        "crime_ttc_s": math.inf,
        "crime_a_long_req_mps2": 0.0,
        "crime_btn": 0.0,
        "crime_a_lat_req_mps2": 0.0,
    }


def test_a_braking_car_behind_needs_no_braking(evaluator: Evaluator) -> None:
    ego = on_road(50, 1.75, 10.0, 1)
    follower = braking(on_road(30, 1.75, 10.0, 2), 4.0)
    values, errors = evaluator.evaluate(
        ego, [follower], ["crime_ttc_s", "crime_a_long_req_mps2", "crime_btn"]
    )
    assert errors == {}
    assert values == {"crime_ttc_s": math.inf, "crime_a_long_req_mps2": 0.0, "crime_btn": 0.0}


def test_the_contact_time_stops_both_rather_than_reversing() -> None:
    from commonroad_metrics.evaluator import _contact_time

    # Constant speeds: the gap over the closing speed.
    assert _contact_time(20.0, 15.0, 0.0, 5.0, 0.0) == pytest.approx(2.0)
    # The ego stops 10 m on, short of a stopped car 12 m ahead.
    assert _contact_time(12.0, 10.0, -5.0, 0.0, 0.0) == math.inf
    # The lead stops 5 m on (at 1 s); the ego, braking too, reaches 20 m when
    # 10 t - t^2 = 20.
    assert _contact_time(15.0, 10.0, -2.0, 10.0, -10.0) == pytest.approx(5.0 - math.sqrt(5.0))
    # A car standing with its acceleration backwards stays (never reverses).
    assert _contact_time(10.0, 5.0, 0.0, 0.0, -3.0) == pytest.approx(2.0)
    # Already touching and closing.
    assert _contact_time(0.0, 5.0, 0.0, 0.0, 0.0) == 0.0
    # Opening.
    assert _contact_time(5.0, 5.0, 0.0, 10.0, 0.0) == math.inf


def test_an_oncoming_car_in_the_ego_lane_closes_the_gap() -> None:
    """A car coming the other way in the ego's lane reads as coming, not as
    going the ego's way."""
    ego = on_road(50, 1.75, 10.0, 1)
    oncoming = replace(on_road(94.5, 1.75, -10.0, 2), yaw=HEADING + math.pi)
    keys = ["crime_hw_m", "crime_ttc_s", "crime_a_long_req_mps2", "crime_btn"]
    for share_frame in (True, False):
        values, errors = Evaluator(str(MAP), EvaluatorSettings(share_frame=share_frame)).evaluate(
            ego, [oncoming], keys
        )
        assert errors == {}
        assert values["crime_hw_m"] == pytest.approx(40.0, abs=0.1)
        # Closing at 20 m/s.
        assert values["crime_ttc_s"] == pytest.approx(2.0, abs=0.02)
        # It keeps coming: the ego must stand far enough back that it is not
        # reached within the 3 s horizon, 40 - 3 * 10 = 10 m on: 10^2 / (2 * 10).
        assert values["crime_a_long_req_mps2"] == pytest.approx(-5.0, abs=0.02)
        assert values["crime_btn"] == pytest.approx(5.0 / 11.5, abs=1e-3)
    # Braking at 2 m/s^2 itself, it would stop 25 m on.  The ego must stop
    # before it reaches the ego: it does at 3.47 s, after the horizon, as
    # the ego stops braking at A, 40 A^2 - 150 A + 100 = 0.  (Not CriMe's
    # speed matching -3, which leaves the ego moving as they meet.)
    braking_oncoming = replace(oncoming, ax=2.0 * COS, ay=2.0 * SIN)
    values, errors = Evaluator(str(MAP)).evaluate(ego, [braking_oncoming], keys)
    assert errors == {}
    assert values["crime_a_long_req_mps2"] == pytest.approx(
        -(150.0 + math.sqrt(6500.0)) / 80.0, abs=0.01
    )
    # 5 m apart, 20 m/s closing: it reaches the ego within the horizon
    # however hard it brakes.
    close = replace(oncoming, x=ego.x + 9.5 * COS, y=ego.y + 9.5 * SIN)
    values, errors = Evaluator(str(MAP)).evaluate(ego, [close], keys)
    assert errors == {}
    assert values["crime_a_long_req_mps2"] == -math.inf
    assert values["crime_btn"] == math.inf


def test_a_slow_approach_beyond_the_horizon_needs_no_more_than_a_stop() -> None:
    """A road user that would reach a standing ego only long after the
    horizon is not one braking must avoid: a pedestrian crossing the lane a
    little towards the ego, a car rolling back."""
    evaluator = Evaluator(str(MAP))
    keys = ["crime_a_long_req_mps2", "crime_btn"]
    ego = on_road(50, 1.75, 10.0, 1)
    values = {}
    for degrees in (90.0, 95.0):
        yaw = HEADING + math.radians(degrees)
        pedestrian = replace(
            on_road(110, 1.75, 0.0, 2),
            kind="pedestrian",
            yaw=yaw,
            vx=1.4 * math.cos(yaw),
            vy=1.4 * math.sin(yaw),
            length=0.5,
            width=0.5,
        )
        values[degrees], errors = evaluator.evaluate(ego, [pedestrian], keys)
        assert errors == {}
    # Stopping short of it, 57.5 m ahead: 10^2 / (2 * 57.5); and about the
    # same with it coming 0.12 m/s towards the ego.
    assert values[90.0]["crime_a_long_req_mps2"] == pytest.approx(-0.87, abs=0.01)
    assert values[95.0]["crime_a_long_req_mps2"] == pytest.approx(-0.89, abs=0.01)
    assert values[95.0]["crime_btn"] < 0.1
    # A stopped ego, a car 40 m ahead rolling back at 0.2 m/s (TTC 200 s).
    rolling = replace(on_road(94.5, 1.75, 0.0, 2), vx=-0.2 * COS, vy=-0.2 * SIN)
    values_rolling, errors = evaluator.evaluate(on_road(50, 1.75, 0.0, 1), [rolling], keys)
    assert errors == {}
    assert values_rolling == {"crime_a_long_req_mps2": 0.0, "crime_btn": 0.0}


def test_the_required_deceleration_is_the_least_that_avoids_contact() -> None:
    """ALongReq in the model TTC reads: each holding its acceleration until it
    stops.  Continuous across a lead's acceleration of 0 on an opening gap,
    where CriMe's formula reads -v_rel^2 / (2 gap) for a lead that never comes
    back."""
    from commonroad_metrics.evaluator import _contact_time, _required_deceleration

    # A stopped ego, a lead pulling away at 10 m/s 2 m ahead.
    for a_other in (0.0, 0.01, -0.01):
        assert _required_deceleration(2.0, 0.0, 10.0, a_other, 1) == 0.0
    # The ego at 10 m/s, a lead pulling away at 15 m/s 20 m ahead.
    for a_other in (0.0, 0.01):
        assert _required_deceleration(20.0, 10.0, 15.0, a_other, 1) == 0.0
    # ... braking slightly: it stops 15^2 / 0.02 = 11250 m on, and the ego
    # must stop behind that; next to nothing.
    assert _required_deceleration(20.0, 10.0, 15.0, -0.01, 1) == pytest.approx(
        -100.0 / (2 * 11270.0), rel=1e-6
    )
    # A closing gap, a lead at constant speed: CriMe's v_rel^2 / (2 gap).
    assert _required_deceleration(25.5, 15.0, 5.0, 0.0, 1) == pytest.approx(-100.0 / 51.0)
    # Cars coming the other way, braking to a stop where the ego must stop
    # short of: 40 - 25 m, 2 - 0.9 m, 20 - 10 m ahead.
    cases = [
        (40.0, 10.0, -10.0, 2.0, -100.0 / 30.0),
        (2.0, 20.0, -3.0, 5.0, -400.0 / 2.2),
        (20.0, 20.0, -10.0, 5.0, -20.0),
    ]
    for gap, v_ego, v_other, a_other, expected in cases:
        a_req = _required_deceleration(gap, v_ego, v_other, a_other, 1)
        assert a_req == pytest.approx(expected, rel=1e-6)
        assert _contact_time(gap, v_ego, a_req, v_other, a_other) == math.inf
        assert _contact_time(gap, v_ego, a_req * 0.99, v_other, a_other) < math.inf
    # One that reaches the ego even stopped.
    assert _required_deceleration(5.0, 10.0, -10.0, 1.0, 1) == -math.inf
    # Within a horizon: the contacts of the cases above come within 3 s, but
    # for the first's, which comes at 3.47 s, the ego standing by then.
    assert _required_deceleration(40.0, 10.0, -10.0, 2.0, 1, 3.0) == pytest.approx(
        -(150.0 + math.sqrt(6500.0)) / 80.0, rel=1e-6
    )
    for gap, v_ego, v_other, a_other, expected in cases[1:]:
        assert _required_deceleration(gap, v_ego, v_other, a_other, 1, 3.0) == pytest.approx(
            expected, rel=1e-6
        )
    assert _required_deceleration(25.5, 15.0, 5.0, 0.0, 1, 3.0) == pytest.approx(-100.0 / 51.0)
    # A car coming at a standing ego, reaching it only after the horizon.
    assert _required_deceleration(40.0, 0.0, -0.2, 0.0, 1, 3.0) == 0.0
    assert _required_deceleration(5.0, 10.0, -10.0, 0.0, 1, 3.0) == -math.inf
    # One behind.
    assert _required_deceleration(math.inf, 10.0, 10.0, -4.0, 1) == 0.0


_LONG_STRAIGHT = """<?xml version="1.0" encoding="UTF-8"?>
<commonRoad timeStepSize="0.1" commonRoadVersion="2020a" author="test" affiliation=""
  source="test" benchmarkID="ZAM_Long-1_1_T-1" date="2026-01-01">
  <scenarioTags><simulated/></scenarioTags>
  <lanelet id="1">
    <leftBound>
      <point><x>0.0</x><y>3.5</y></point>
      <point><x>400.0</x><y>3.5</y></point>
    </leftBound>
    <rightBound>
      <point><x>0.0</x><y>0.0</y></point>
      <point><x>400.0</x><y>0.0</y></point>
    </rightBound>
    <laneletType>urban</laneletType>
  </lanelet>
</commonRoad>
"""


def test_a_long_straight_lanelet_is_kept_from_its_middle(tmp_path: Path) -> None:
    """A straight lanelet is just its two end points: 200 m from both, the
    ego is still on it."""
    path = tmp_path / "long.xml"
    path.write_text(_LONG_STRAIGHT)
    evaluator = Evaluator(str(path), EvaluatorSettings(radius_m=100.0))
    ego = Actor(1, "car", 200.0, 1.75, 0.0, vx=15.0)
    lead = Actor(2, "car", 230.0, 1.75, 0.0, vx=5.0)
    keys = ["crime_ttc_s", "crime_hw_m", "reach_drivable_area_m2"]
    values, errors = evaluator.evaluate(ego, [lead], keys)
    assert errors == {}
    assert values["crime_hw_m"] == pytest.approx(25.5, abs=0.1)
    assert values["crime_ttc_s"] == pytest.approx(2.55, abs=0.05)
    assert values["reach_drivable_area_m2"] > 0.0


def test_the_reachable_set_reads_the_ego_footprint(evaluator: Evaluator) -> None:
    """Reach inflates the other road users by the ego's half width (its
    inscribed circle) and approximates the ego by discs of its own box, not
    its default vehicle's."""
    from commonroad_metrics.evaluator import _set_ego_footprint

    config = copy.deepcopy(evaluator._reach_base_config())
    _set_ego_footprint(config, 5.0, 2.1)
    assert config.vehicle.ego.radius_inflation == pytest.approx(1.05)
    assert config.vehicle.ego.radius_disc == pytest.approx(math.hypot(5.0 / 6, 1.05))
    assert config.vehicle.ego.circle_distance == pytest.approx(5.0 * 2 / 3)

    def area(width: float) -> float:
        ego = replace(on_road(50, 1.75, 15.0, 1), width=width)
        values, errors = evaluator.evaluate(
            ego,
            [on_road(70, 1.75, 0.0, 2), on_road(70, 5.25, 0.0, 3)],
            ["reach_drivable_area_m2"],
        )
        assert errors == {}
        return values["reach_drivable_area_m2"]

    assert area(2.1) < area(1.6)


def test_a_reversing_ego_reaches_backwards(evaluator: Evaluator) -> None:
    forward, _ = evaluator.evaluate(on_road(50, 1.75, 3.0, 1), [], ["reach_drivable_area_m2"])
    backward, errors = evaluator.evaluate(
        on_road(50, 1.75, -3.0, 1), [], ["reach_drivable_area_m2"]
    )
    assert errors == {}
    assert backward["reach_drivable_area_m2"] > 0.0
    assert backward["reach_drivable_area_m2"] != forward["reach_drivable_area_m2"]
    # Faster backwards than Reach's default vehicle reverses (13.9 m/s).
    fast, errors = evaluator.evaluate(on_road(50, 1.75, -20.0, 1), [], ["reach_drivable_area_m2"])
    assert errors == {}
    assert fast["reach_drivable_area_m2"] > 0.0


def test_the_threat_numbers_are_the_required_accelerations_over_the_most(
    evaluator: Evaluator,
) -> None:
    ego = on_road(50, 1.75, 15.0, 1)
    lead = on_road(80, 1.75, 5.0, 2)
    values, errors = evaluator.evaluate(
        ego,
        [lead],
        ["crime_a_long_req_mps2", "crime_a_lat_req_mps2", "crime_btn", "crime_stn"],
    )
    assert errors == {}
    # CriMe's most a car can brake and steer: 11.5 and 8 m/s^2.
    assert values["crime_btn"] == pytest.approx(-values["crime_a_long_req_mps2"] / 11.5, abs=1e-3)
    assert values["crime_stn"] == pytest.approx(abs(values["crime_a_lat_req_mps2"]) / 8.0, abs=1e-3)


def test_the_worst_case_time_to_react(evaluator: Evaluator) -> None:
    ego = on_road(50, 1.75, 15.0, 1)
    # Closing at 10 m/s on a 25.5 m gap, the next lane free: time to react,
    # but less than until they would meet.
    values, errors = evaluator.evaluate(
        ego, [on_road(80, 1.75, 5.0, 2)], ["crime_wttr_s", "crime_ttc_s"]
    )
    assert errors == {}
    assert 0.0 < values["crime_wttr_s"] < values["crime_ttc_s"]
    # Both lanes blocked 40 m ahead: less time than with a lane to swerve into
    # and a car going away.
    blocked, _ = evaluator.evaluate(
        ego, [on_road(90, 1.75, 0.0, 2), on_road(90, 5.25, 0.0, 3)], ["crime_wttr_s"]
    )
    assert 0.0 < blocked["crime_wttr_s"] < values["crime_wttr_s"]
    # Both lanes blocked 12 m ahead at 15 m/s: no way is left.
    trapped, _ = evaluator.evaluate(
        ego, [on_road(62, 1.75, 0.0, 2), on_road(62, 5.25, 0.0, 3)], ["crime_wttr_s"]
    )
    assert trapped["crime_wttr_s"] == 0.0


def test_no_collision_ahead_leaves_all_the_time_to_react(evaluator: Evaluator) -> None:
    ego = on_road(50, 1.75, 15.0, 1)
    alone, _ = evaluator.evaluate(ego, [], ["crime_wttr_s"])
    beside, _ = evaluator.evaluate(ego, [on_road(80, 5.25, 5.0, 2)], ["crime_wttr_s"])
    assert alone["crime_wttr_s"] == math.inf
    assert beside["crime_wttr_s"] == math.inf


def test_the_time_to_react(evaluator: Evaluator) -> None:
    ego = on_road(50, 1.75, 15.0, 1)
    keys = ["crime_ttr_s", "crime_wttr_s", "crime_ttc_s"]
    values, errors = evaluator.evaluate(ego, [on_road(80, 1.75, 5.0, 2)], keys)
    assert errors == {}
    # Simulated manoeuvres under-approximate what the reachable set leaves.
    assert 0.0 < values["crime_ttr_s"] <= values["crime_wttr_s"] < values["crime_ttc_s"]
    # No manoeuvre is left: 0, not CriMe's -inf.
    trapped, _ = evaluator.evaluate(
        ego, [on_road(62, 1.75, 0.0, 2), on_road(62, 5.25, 0.0, 3)], ["crime_ttr_s"]
    )
    assert trapped["crime_ttr_s"] == 0.0


def test_workers_started_together_import_reach_one_at_a_time(tmp_path: Path) -> None:
    """CommonRoad-Reach compiles a Cython module into ~/.pyxbld on its first
    import: workers that start together must not load each other's
    half-built module (unlocked, about 2 in 5 of 8 such processes failed)."""
    code = (
        "from commonroad_metrics.evaluator import _import_reach\n"
        "_import_reach()\n"
        "import commonroad_crime.measure\n"
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(6)
    ]
    outputs = [worker.communicate(timeout=600)[0] for worker in workers]
    assert [worker.returncode for worker in workers] == [0] * len(workers), outputs


#: Runs the worker's main() with a handler that, before answering, writes to
#: descriptor 1 as C or C++ code would, and to sys.stdout as print() does.
_NOISY_MAIN = """
import os, sys
from commonroad_metrics import worker

def handle(self, request):
    os.write(1, b"from C++ on descriptor 1\\n")
    print("from print()")
    return {"ok": True, "op": request.get("op")}

worker.Worker.handle = handle
raise SystemExit(worker.main([]))
"""


def test_writes_to_descriptor_1_do_not_break_into_the_answers() -> None:
    """#1954: the answers go to a duplicate of the output, and descriptor 1
    to the standard error, so a C++ write to it cannot corrupt an answer."""
    requests = [{"op": "evaluate"}, {"op": "close"}]
    done = subprocess.run(
        [sys.executable, "-c", _NOISY_MAIN],
        input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    answers = [json.loads(line) for line in done.stdout.splitlines()]
    assert answers == [{"ok": True, "op": "evaluate"}, {"ok": True, "op": "close"}]
    assert done.stderr.count("from C++ on descriptor 1") == 2
    assert done.stderr.count("from print()") == 2
