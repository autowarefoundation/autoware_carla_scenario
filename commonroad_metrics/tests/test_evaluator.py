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
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from commonroad_metrics import CRIME_MEASURES, MEASURES, Actor, Evaluator, EvaluatorSettings
from commonroad_metrics.evaluator import NonFiniteError
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
    ego = on_road(50, 1.75, 15.0, 1).__dict__
    requests = [
        {"op": "init", "map": str(MAP)},
        # A car in the next lane: no conflict, TTC never comes.
        {
            "op": "evaluate",
            "measures": ["crime_ttc_s"],
            "ego": ego,
            "others": [on_road(80, 5.25, 5.0, 2).__dict__],
        },
        # Alone on the road: no one to measure from.
        {"op": "evaluate", "measures": ["crime_ttc_s"], "ego": ego},
    ]
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    serve(stdin, stdout)
    _, beside, alone = stdout.getvalue().splitlines()
    assert '"crime_ttc_s": Infinity' in beside
    assert json.loads(beside)["values"]["crime_ttc_s"] == math.inf
    assert '"crime_ttc_s": null' in alone
    assert json.loads(alone)["values"]["crime_ttc_s"] is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "name", ["x", "y", "yaw", "vx", "vy", "ax", "ay", "yaw_rate", "length", "width"]
)
def test_a_road_user_with_a_non_finite_number_is_refused(name: str, bad: float) -> None:
    """#2342: a NaN speed read as TTC inf, a NaN position as every measure
    null; refused instead, naming the field."""
    fields = {**on_road(50, 1.75, 15.0, 1).__dict__, name: bad}
    with pytest.raises(ValueError, match=rf"^{name} must be finite"):
        Actor.from_mapping(fields)


@pytest.mark.parametrize("bad", ["15", None, True, 10**400])
def test_a_road_user_with_a_number_that_is_no_float_is_refused(bad: Any) -> None:
    fields = {**on_road(50, 1.75, 15.0, 1).__dict__, "vx": bad}
    with pytest.raises(ValueError, match=r"^vx must be "):
        Actor.from_mapping(fields)


def test_a_non_finite_number_is_told_apart_from_one_that_is_no_number() -> None:
    """The worker fails a frame's measures on the first, refuses the request
    on the second."""
    fields = on_road(50, 1.75, 15.0, 1).__dict__
    for bad in (math.nan, math.inf, 10**400):
        with pytest.raises(NonFiniteError):
            Actor.from_mapping({**fields, "vx": bad})
    for bad in ("15", None, True):
        with pytest.raises(ValueError) as refused:
            Actor.from_mapping({**fields, "vx": bad})
        assert not isinstance(refused.value, NonFiniteError)


def test_the_worker_fails_the_measures_of_a_non_finite_frame_and_reads_on() -> None:
    """#2342: a NaN in one road user of one frame fails that frame's measures
    -- ``ok: true``, each ``null`` with the message -- rather than the
    request, which made the framework give up every CommonRoad metric of the
    run; the next frame is evaluated as ever."""
    ego: dict[str, Any] = on_road(50, 1.75, 15.0, 1).__dict__
    lead: dict[str, Any] = on_road(80, 1.75, 5.0, 2).__dict__
    measures = ["crime_ttc_s", "reach_drivable_area_m2", "crime_ttc_s"]
    evaluate: dict[str, Any] = {"op": "evaluate", "measures": measures, "ego": ego}
    requests = [
        {"op": "init", "map": str(MAP)},
        {**evaluate, "others": [{**lead, "vx": math.nan}]},  # json writes NaN
        {**evaluate, "ego": {**ego, "x": math.inf}, "others": [lead]},
        {**evaluate, "measures": ["crime_ttc_s", "nope"], "others": [{**lead, "y": -math.inf}]},
        {**evaluate, "measures": ["crime_ttc_s"], "others": [lead]},
    ]
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    serve(stdin, stdout)
    answers = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(answers) == 5
    nan = "vx must be finite, got nan"
    assert answers[1] == {
        "ok": True,
        "values": {"crime_ttc_s": None, "reach_drivable_area_m2": None},
        "errors": {"crime_ttc_s": nan, "reach_drivable_area_m2": nan},
    }
    inf = "x must be finite, got inf"
    assert answers[2] == {
        "ok": True,
        "values": {"crime_ttc_s": None, "reach_drivable_area_m2": None},
        "errors": {"crime_ttc_s": inf, "reach_drivable_area_m2": inf},
    }
    # An unknown measure is unknown still.
    assert answers[3] == {
        "ok": True,
        "values": {"crime_ttc_s": None},
        "errors": {"crime_ttc_s": "y must be finite, got -inf", "nope": "unknown measure"},
    }
    assert answers[4]["ok"] and answers[4]["errors"] == {}
    assert answers[4]["values"]["crime_ttc_s"] == pytest.approx(2.55, abs=0.05)


@pytest.mark.parametrize(
    ("ego", "error"),
    [
        ({"id": 1, "kind": "car", "x": 50.0, "y": 1.75}, "TypeError: "),  # no yaw
        ({**on_road(50, 1.75, 15.0, 1).__dict__, "vx": "15"}, "ValueError: vx must be a number"),
        ({**on_road(50, 1.75, 15.0, 1).__dict__, "vx": None}, "ValueError: vx must be a number"),
    ],
)
def test_the_worker_refuses_a_malformed_road_user(ego: dict[str, Any], error: str) -> None:
    """A field missing or of the wrong type is out of protocol: ``ok: false``."""
    requests = [
        {"op": "init", "map": str(MAP)},
        {"op": "evaluate", "measures": ["crime_ttc_s"], "ego": ego},
    ]
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    serve(stdin, stdout)
    answers = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert answers[1]["ok"] is False
    assert answers[1]["error"].startswith(error)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dt": 0.0}, "dt must be positive"),
        ({"dt": -0.1}, "dt must be positive"),
        ({"dt": math.nan}, "dt must be finite"),
        ({"dt": math.inf}, "dt must be finite"),
        ({"radius_m": 0.0}, "radius_m must be positive"),
        ({"radius_m": -5.0}, "radius_m must be positive"),
        ({"radius_m": math.nan}, "radius_m must be finite"),
        ({"radius_m": math.inf}, "radius_m must be finite"),
        ({"radius_m": "100"}, "radius_m must be a number"),
        ({"horizon_steps": 0}, "horizon_steps must be an int of at least 1"),
        ({"horizon_steps": -3}, "horizon_steps must be an int of at least 1"),
        ({"horizon_steps": 30.0}, "horizon_steps must be an int of at least 1"),
        ({"horizon_steps": True}, "horizon_steps must be an int of at least 1"),
    ],
)
def test_settings_that_would_break_every_frame_are_refused(
    changes: dict[str, Any], message: str
) -> None:
    """#2342: refused when made, not by every frame after init."""
    with pytest.raises(ValueError, match=f"^{message}"):
        EvaluatorSettings.from_mapping(changes)


def test_the_smallest_settings_are_accepted() -> None:
    settings = EvaluatorSettings.from_mapping({"dt": 1e-3, "horizon_steps": 1, "radius_m": 1})
    assert (settings.dt, settings.horizon_steps, settings.radius_m) == (1e-3, 1, 1)


def test_the_worker_refuses_bad_settings_at_init() -> None:
    requests = [
        {"op": "init", "map": str(MAP), "settings": {"horizon_steps": 0}},
        {"op": "evaluate", "measures": ["crime_ttc_s"], "ego": {}},
    ]
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    serve(stdin, stdout)
    answers = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert answers == [
        {"ok": False, "error": "ValueError: horizon_steps must be an int of at least 1, got 0"},
        {"ok": False, "error": "evaluate before init"},
    ]


def test_a_line_that_is_not_utf8_is_answered_and_the_worker_reads_on() -> None:
    """#2342: the line was decoded as it was read, outside the request's try,
    and its UnicodeDecodeError ended the worker."""
    lines = [
        json.dumps({"op": "evaluate", "measures": [], "ego": {}}).encode(),
        b'{"op": "\xff\xfe"}',
        b"",  # a blank line: no request, no answer
        b"   ",
        json.dumps({"op": "close"}).encode(),
    ]
    stdin = io.BytesIO(b"".join(line + b"\n" for line in lines))
    stdout = io.StringIO()
    serve(stdin, stdout)
    answers = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(answers) == 3
    assert answers[0] == {"ok": False, "error": "evaluate before init"}
    assert answers[1]["ok"] is False
    assert answers[1]["error"].startswith("UnicodeDecodeError: 'utf-8' codec can't decode")
    assert answers[2] == {"ok": True}


def test_the_worker_process_survives_a_line_that_is_not_utf8() -> None:
    """The same through main(): its standard input is read as bytes."""
    answered = subprocess.run(
        [sys.executable, "-m", "commonroad_metrics"],
        input=b'\xff\n{"op": "nope"}\n{"op": "close"}\n',
        capture_output=True,
        timeout=120,
        check=True,
    )
    answers = [json.loads(line) for line in answered.stdout.decode().splitlines()]
    assert [a["ok"] for a in answers] == [False, False, True]
    assert answers[0]["error"].startswith("UnicodeDecodeError: ")
    assert answers[1] == {"ok": False, "error": "unknown op 'nope'"}


@pytest.mark.parametrize("failure", ["none", "raises"])
def test_the_ego_lane_heading_missing_reads_nothing(
    evaluator: Evaluator, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """#2342: CriMe reads no lane heading at the ego -- (None, None) out of the
    projection's domain, ValueError for a lanelet it cannot project on.  TTC
    read NaN, ALatReq failed on math.sin(None) and ALongReq read the least
    critical 0; every one reads NaN (null, no error) now."""
    import commonroad_crime.utility.solver as utils_sol

    def no_heading(lanelet: Any, position: Any) -> Any:
        if failure == "raises":
            raise ValueError("Reference path must have at least 3 points")
        return None, None

    monkeypatch.setattr(utils_sol, "compute_lanelet_width_orientation", no_heading)
    measures = [
        "crime_ttc_s",
        "crime_a_long_req_mps2",
        "crime_btn",
        "crime_a_lat_req_mps2",
        "crime_stn",
    ]
    ego = on_road(50, 1.75, 15.0, 1)
    values, errors = evaluator.evaluate(ego, [on_road(80, 1.75, 5.0, 2)], measures)
    assert errors == {}
    assert values == {key: None for key in measures}


def test_a_degenerate_ego_box_is_floored_as_an_obstacle_is(
    evaluator: Evaluator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2342: the obstacles' boxes were floored at 0.1 m, Reach's ego's not."""
    import commonroad_metrics.evaluator as module

    footprints: list[tuple[float, float]] = []
    original = module._set_ego_footprint

    def recorded(config: Any, length: float, width: float) -> None:
        footprints.append((length, width))
        original(config, length, width)

    monkeypatch.setattr(module, "_set_ego_footprint", recorded)
    ego = replace(on_road(50, 1.75, 15.0, 1), length=0.0, width=0.05)
    values, errors = evaluator.evaluate(ego, [], ["reach_drivable_area_m2"])
    assert errors == {}
    assert footprints and set(footprints) == {(0.1, 0.1)}
    assert values["reach_drivable_area_m2"] is not None
    assert values["reach_drivable_area_m2"] > 0.0


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
        # It keeps coming: the ego, braking at A, stops 10 / A s and 50 / A m
        # on, and must not be reached for the 3 s horizon after: the car,
        # 40 - 50 / A m off then, takes (40 - 50 / A) / 10 > 3 s, so A > 15.
        assert values["crime_a_long_req_mps2"] == pytest.approx(-15.0, abs=0.02)
        assert values["crime_btn"] == pytest.approx(15.0 / 11.5, abs=1e-3)
    # Braking at 2 m/s^2 itself, it stops 25 m on, at 5 s: within the horizon
    # after the ego stops, so the ego must stop short of that point, 15 m on.
    # (Not CriMe's speed matching -3, which leaves the ego moving as they meet.)
    braking_oncoming = replace(oncoming, ax=2.0 * COS, ay=2.0 * SIN)
    values, errors = Evaluator(str(MAP)).evaluate(ego, [braking_oncoming], keys)
    assert errors == {}
    assert values["crime_a_long_req_mps2"] == pytest.approx(-100.0 / 30.0, abs=0.01)
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
    assert values[95.0]["crime_a_long_req_mps2"] == pytest.approx(-0.90, abs=0.005)
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
    # Within a horizon: the cars above stop within 3 s of the ego stopping.
    for gap, v_ego, v_other, a_other, expected in cases:
        assert _required_deceleration(gap, v_ego, v_other, a_other, 1, 3.0) == pytest.approx(
            expected, rel=1e-6
        )
    assert _required_deceleration(25.5, 15.0, 5.0, 0.0, 1, 3.0) == pytest.approx(-100.0 / 51.0)
    # A car coming at a standing ego, reaching it only after the horizon.
    assert _required_deceleration(40.0, 0.0, -0.2, 0.0, 1, 3.0) == 0.0
    assert _required_deceleration(5.0, 10.0, -10.0, 0.0, 1, 3.0) == -math.inf
    # Cars coming at constant speed: the ego, braking at A from v, stops
    # v / A s and v^2 / (2 A) m on, and must not be reached for 3 s after,
    # gap - v^2 / (2 A) > u (v / A + 3).
    for gap, v_ego, u in [(200.0, 20.0, 10.0), (120.0, 20.0, 10.0), (100.0, 25.0, 5.0)]:
        expected = -(v_ego**2 / 2 + u * v_ego) / (gap - 3.0 * u)
        a_req = _required_deceleration(gap, v_ego, -u, 0.0, 1, 3.0)
        assert a_req == pytest.approx(expected, rel=1e-6)
    assert _required_deceleration(200.0, 20.0, -10.0, 0.0, 1, 3.0) == pytest.approx(
        -2.353, abs=1e-3
    )
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


OVERLAP_MEASURES = [
    "crime_hw_m",
    "crime_thw_s",
    "crime_ttc_s",
    "crime_a_long_req_mps2",
    "crime_btn",
    "crime_a_lat_req_mps2",
    "crime_stn",
]


@pytest.mark.parametrize("centre_gap", [4.0, 2.0, 0.0], ids=["4m", "2m", "level"])
def test_a_car_overlapping_the_ego_from_ahead_is_a_collision(
    evaluator: Evaluator, centre_gap: float
) -> None:
    """#2290: CriMe read the negative headway of a car whose box overlaps the
    ego's as one behind (TTC inf, ALongReq 0): the most critical frames as
    the least.  Overlapping from ahead, it is touching: the gap is closed."""
    ego = on_road(50, 1.75, 15.0, 1)
    lead = on_road(50 + centre_gap, 1.75, 5.0, 2)
    values, errors = evaluator.evaluate(ego, [lead], OVERLAP_MEASURES)
    assert errors == {}
    assert values == {
        "crime_hw_m": 0.0,
        "crime_thw_s": 0.0,
        "crime_ttc_s": 0.0,
        "crime_a_long_req_mps2": -math.inf,
        "crime_btn": math.inf,
        "crime_a_lat_req_mps2": math.inf,
        "crime_stn": math.inf,
    }


def test_a_car_touching_the_ego_from_ahead_is_a_collision(evaluator: Evaluator) -> None:
    """#2290: bumper to bumper (4.5 m long, 4.5 m apart) read as one behind."""
    ego = on_road(50, 1.75, 15.0, 1)
    values, errors = evaluator.evaluate(ego, [on_road(54.5, 1.75, 5.0, 2)], OVERLAP_MEASURES)
    assert errors == {}
    assert values["crime_hw_m"] == pytest.approx(0.0, abs=0.01)
    assert values["crime_ttc_s"] == 0.0
    assert values["crime_a_long_req_mps2"] == -math.inf
    assert values["crime_btn"] == math.inf
    # Just short of touching, closing at 10 m/s: about to.
    near, errors = evaluator.evaluate(ego, [on_road(54.6, 1.75, 5.0, 2)], ["crime_ttc_s"])
    assert errors == {}
    assert 0.0 < near["crime_ttc_s"] <= 0.02


def test_a_car_overlapping_the_ego_and_pulling_away(evaluator: Evaluator) -> None:
    """#2290: an overlapping lead faster than the ego is touching, but the gap
    opens: no contact to come, no braking needed."""
    ego = on_road(50, 1.75, 5.0, 1)
    values, errors = evaluator.evaluate(
        ego,
        [on_road(54, 1.75, 15.0, 2)],
        ["crime_hw_m", "crime_ttc_s", "crime_a_long_req_mps2", "crime_btn"],
    )
    assert errors == {}
    assert values == {
        "crime_hw_m": 0.0,
        "crime_ttc_s": math.inf,
        "crime_a_long_req_mps2": 0.0,
        "crime_btn": 0.0,
    }


def test_a_car_overlapping_the_ego_from_ahead_and_aside_is_a_collision(
    evaluator: Evaluator,
) -> None:
    """#2290: boxes overlapping by a corner, 1 m apart sideways."""
    ego = on_road(50, 1.75, 15.0, 1)
    values, errors = evaluator.evaluate(ego, [on_road(52, 2.75, 5.0, 2)], OVERLAP_MEASURES)
    assert errors == {}
    assert values["crime_hw_m"] == 0.0
    assert values["crime_ttc_s"] == 0.0
    assert values["crime_btn"] == math.inf


@pytest.mark.parametrize(
    ("ego_right", "other_distance", "other_right"),
    [(1.75, 52.0, 3.7), (3.0, 51.0, 5.25), (3.3, 50.0, 5.25)],
    ids=["straddling-the-line", "ego-changing-lanes", "level"],
)
def test_a_car_beside_the_ego_in_a_shared_lanelet_is_no_collision(
    evaluator: Evaluator, ego_right: float, other_distance: float, other_right: float
) -> None:
    """#2290 review: CriMe takes a road user in the ego's lane by any lanelet
    their boxes share, so a car beside the ego, its centre ahead and the
    boxes apart sideways, has a negative headway too: no contact."""
    ego = on_road(50, ego_right, 15.0, 1)
    other = on_road(other_distance, other_right, 5.0, 2)
    values, errors = evaluator.evaluate(ego, [other], OVERLAP_MEASURES)
    assert errors == {}
    assert values == {
        "crime_hw_m": math.inf,
        "crime_thw_s": math.inf,
        "crime_ttc_s": math.inf,
        "crime_a_long_req_mps2": 0.0,
        "crime_btn": 0.0,
        "crime_a_lat_req_mps2": 0.0,
        "crime_stn": 0.0,
    }


@pytest.mark.parametrize("centre_gap", [-2.0, -4.0, -20.0], ids=["overlap", "deep", "far"])
def test_a_car_behind_the_ego_is_still_no_conflict(evaluator: Evaluator, centre_gap: float) -> None:
    """#2290: only a road user whose centre is behind the ego's -- one hitting
    it from behind too -- reads HW inf: no braking of the ego's avoids it."""
    ego = on_road(50, 1.75, 5.0, 1)
    follower = on_road(50 + centre_gap, 1.75, 15.0, 2)
    values, errors = evaluator.evaluate(ego, [follower], OVERLAP_MEASURES)
    assert errors == {}
    assert values == {
        "crime_hw_m": math.inf,
        "crime_thw_s": math.inf,
        "crime_ttc_s": math.inf,
        "crime_a_long_req_mps2": 0.0,
        "crime_btn": 0.0,
        "crime_a_lat_req_mps2": 0.0,
        "crime_stn": 0.0,
    }


@pytest.mark.parametrize(
    ("reach_settings", "message"),
    [
        # The example the README and the docstring gave (#2290).
        (
            {"planning": {"size_grid": 0.3}},
            "reach_settings['planning']['size_grid']: CommonRoad-Reach's 'planning' "
            "section has no key 'size_grid'",
        ),
        (
            {"reachable_set": {"size_gird": 0.3}},
            "reach_settings['reachable_set']['size_gird']:",
        ),
        (
            {"planing": {"steps_computation": 5}},
            "reach_settings['planing']: CommonRoad-Reach's configuration has no section",
        ),
        (
            {"config_omega": {}},
            "reach_settings['config_omega']: CommonRoad-Reach's configuration has no section",
        ),
        (
            {"vehicle": {"ego": 1.0}},
            "reach_settings['vehicle']['ego']: vehicle.ego is a section",
        ),
        (
            {"reachable_set": 0.3},
            "reach_settings['reachable_set'] must be a mapping",
        ),
    ],
)
def test_reach_settings_refuse_what_reach_does_not_have(
    reach_settings: dict[str, Any], message: str
) -> None:
    """#2290: a misspelled or misplaced key was set where nothing read it."""
    with pytest.raises(ValueError, match=re.escape(message)):
        Evaluator(str(MAP), EvaluatorSettings(reach_settings=reach_settings))


def test_reach_settings_set_reach_configuration() -> None:
    evaluator = Evaluator(
        str(MAP), EvaluatorSettings(reach_settings={"reachable_set": {"size_grid": 0.5}})
    )
    assert evaluator._reach_base_config().reachable_set.size_grid == 0.5


def test_the_worker_refuses_unknown_reach_settings_at_init() -> None:
    """#2290: refused by init, not by every frame's Reach measures after it."""
    requests = [
        {
            "op": "init",
            "map": str(MAP),
            "settings": {"reach_settings": {"planning": {"size_grid": 0.3}}},
        },
        {"op": "evaluate", "measures": ["reach_drivable_area_m2"], "ego": {}},
    ]
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    serve(stdin, stdout)
    answers = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert answers[0]["ok"] is False
    assert answers[0]["error"].startswith("ValueError: reach_settings['planning']['size_grid']: ")
    assert answers[1] == {"ok": False, "error": "evaluate before init"}


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


def test_the_time_to_brake(evaluator: Evaluator) -> None:
    ego = on_road(50, 1.75, 15.0, 1)
    keys = ["crime_ttb_s", "crime_ttc_s"]
    values, errors = evaluator.evaluate(ego, [on_road(80, 1.75, 5.0, 2)], keys)
    assert errors == {}
    assert 0.0 < values["crime_ttb_s"] < values["crime_ttc_s"]
    # Braking at full no longer avoids a car standing 12 m ahead at 15 m/s:
    # 0, not CriMe's -inf.
    trapped, errors = evaluator.evaluate(ego, [on_road(62, 1.75, 0.0, 2)], ["crime_ttb_s"])
    assert errors == {}
    assert trapped["crime_ttb_s"] == 0.0
    # In contact already (CriMe's TTC* 0): 0 too.
    touching, errors = evaluator.evaluate(ego, [on_road(53, 1.75, 0.0, 2)], ["crime_ttb_s"])
    assert errors == {}
    assert touching["crime_ttb_s"] == 0.0


def test_no_one_ahead_in_the_lane_reads_none_is(evaluator: Evaluator) -> None:
    """With a road user beside the ego but none ahead in its lane, the lane
    measures read that none is (inf, or 0); with no road user at all, the
    road-user measures read nothing."""
    ego = on_road(50, 1.75, 15.0, 1)
    keys = [
        "crime_ttc_s",
        "crime_thw_s",
        "crime_hw_m",
        "crime_a_long_req_mps2",
        "crime_a_lat_req_mps2",
        "crime_btn",
        "crime_stn",
    ]
    beside, errors = evaluator.evaluate(ego, [on_road(80, 5.25, 15.0, 2)], keys)
    assert errors == {}
    assert beside == {
        "crime_ttc_s": math.inf,
        "crime_thw_s": math.inf,
        "crime_hw_m": math.inf,
        "crime_a_long_req_mps2": 0.0,
        "crime_a_lat_req_mps2": 0.0,
        "crime_btn": 0.0,
        "crime_stn": 0.0,
    }
    alone, _ = evaluator.evaluate(ego, [], keys)
    assert alone == dict.fromkeys(keys)


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


# ----------------------------------------------------------------------
# Curved lanes
# ----------------------------------------------------------------------


Line = list[tuple[float, float]]
Bends = dict[tuple[float, int], tuple[Evaluator, Line]]


def _bend(radius: float, sign: int, heading: float = 0.3, n: int = 301) -> Line:
    """The centre line of a lane bending left (*sign* +1) or right (-1) at
    *radius* metres, a vertex every metre, from the origin heading *heading*."""
    cx, cy = -sign * radius * math.sin(heading), sign * radius * math.cos(heading)
    start = heading - sign * math.pi / 2
    angles = [start + sign * s / radius for s in range(n)]
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


def _write_lane(path: Path, centre: Line) -> None:
    """One 3.5 m lanelet around *centre*, as a CommonRoad scenario."""
    import numpy as np
    from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
    from commonroad.planning.planning_problem import PlanningProblemSet
    from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType
    from commonroad.scenario.scenario import Scenario, ScenarioID

    c = np.asarray(centre)
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    normal = np.stack([-d[:, 1], d[:, 0]], axis=1)
    lanelet = Lanelet(c + 1.75 * normal, c, c - 1.75 * normal, 1, lanelet_type={LaneletType.URBAN})
    network = LaneletNetwork()
    network.add_lanelet(lanelet)
    scenario = Scenario(0.1, ScenarioID(country_id="ZAM", map_name="Bend", map_id=1))
    scenario.add_objects(network)
    writer = CommonRoadFileWriter(scenario, PlanningProblemSet(), "test", "test", "test", set())
    writer.write_to_file(str(path), OverwriteExistingFile.ALWAYS)


def _on_line(centre: Line, at: float, speed: float, actor_id: int) -> Actor:
    """A car on the centre line *at* vertices along it (a fraction is that far
    into the segment), heading along that segment at *speed* m/s."""
    i = int(at)
    (ax, ay), (bx, by) = centre[i], centre[i + 1]
    yaw = math.atan2(by - ay, bx - ax)
    f = at - i
    return Actor(
        actor_id,
        "car",
        ax + f * (bx - ax),
        ay + f * (by - ay),
        yaw,
        vx=speed * math.cos(yaw),
        vy=speed * math.sin(yaw),
        length=4.5,
        width=1.8,
    )


@pytest.fixture(scope="module")
def bends(tmp_path_factory: pytest.TempPathFactory) -> Bends:
    made: Bends = {}
    for radius in (100.0, 300.0):
        for sign in (1, -1):
            centre = _bend(radius, sign)
            path = tmp_path_factory.mktemp("bend") / "bend.xml"
            _write_lane(path, centre)
            made[radius, sign] = Evaluator(str(path)), centre
    return made


@pytest.mark.parametrize(
    ("radius", "sign", "into"),
    [
        # On a vertex, and inside a segment where CriMe's headings either
        # side of the ego straddle 0 and its interpolation read ~2pi off:
        # TTC 66.65 or inf, ALongReq and BTN 0, on the right bend at R=300;
        # TTC inf or 4.22 on the left bend at R=100.
        (300.0, -1, 0.0),
        (300.0, -1, 0.5),
        (300.0, -1, 0.75),
        (300.0, 1, 0.0),
        (300.0, 1, 0.5),
        (100.0, -1, 0.75),
        (100.0, 1, 0.0),
        (100.0, 1, 0.6),
        (100.0, 1, 0.75),
    ],
)
def test_following_a_slower_car_round_a_bend(
    bends: Bends,
    radius: float,
    sign: int,
    into: float,
) -> None:
    """The lead 30 m ahead along the lane less how far the ego is into its
    segment: on a gentle bend, nearly what the same gap reads on a straight."""
    evaluator, centre = bends[radius, sign]
    ego = _on_line(centre, 100 + into, 15.0, 1)
    lead = _on_line(centre, 130, 5.0, 2)
    keys = ["crime_ttc_s", "crime_a_long_req_mps2", "crime_btn"]
    values, errors = evaluator.evaluate(ego, [lead], keys)
    assert errors == {}
    gap = 25.5 - into  # bumper to bumper, along the lane
    assert values["crime_ttc_s"] == pytest.approx(gap / 10.0, abs=0.1)
    assert values["crime_a_long_req_mps2"] == pytest.approx(-(10.0**2) / (2 * gap), abs=0.15)
    assert values["crime_btn"] == pytest.approx(-values["crime_a_long_req_mps2"] / 11.5, abs=1e-3)
    assert values["crime_btn"] > 0.1


def test_a_lane_heading_is_read_across_zero() -> None:
    """Two segments either side of heading 0, as CriMe stores them in
    [0, 2pi): halfway between them is 0, not pi."""
    from commonroad_metrics.evaluator import _lane_heading

    path_length = [0.0, 1.0, 2.0]
    orientation = [2 * math.pi - 0.002, 0.002, 0.006]
    heading = _lane_heading(0.5, path_length, orientation)
    assert math.cos(heading) == pytest.approx(1.0, abs=1e-9)
    assert math.sin(heading) == pytest.approx(0.0, abs=1e-9)
    assert _lane_heading(1.5, path_length, orientation) == pytest.approx(0.004)
    # Off the ends the end headings hold.
    assert _lane_heading(-1.0, path_length, orientation) == pytest.approx(2 * math.pi - 0.002)
    assert _lane_heading(3.0, path_length, orientation) == pytest.approx(0.006)


def test_a_lane_heading_of_exactly_zero_is_a_heading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the ego's lane turned exactly onto +x, CriMe reads its heading as
    0.0: ALongReq still reads the closing lead rather than none."""
    import commonroad_metrics.evaluator as evaluator_module

    monkeypatch.setattr(evaluator_module, "_TILT", 0.0)
    path = tmp_path / "long.xml"
    path.write_text(_LONG_STRAIGHT)
    evaluator = Evaluator(str(path), EvaluatorSettings(radius_m=100.0))
    ego = Actor(1, "car", 200.0, 1.75, 0.0, vx=15.0)
    lead = Actor(2, "car", 230.0, 1.75, 0.0, vx=5.0)
    keys = ["crime_ttc_s", "crime_a_long_req_mps2", "crime_btn"]
    values, errors = evaluator.evaluate(ego, [lead], keys)
    assert errors == {}
    assert values["crime_ttc_s"] == pytest.approx(2.55, abs=0.05)
    assert values["crime_a_long_req_mps2"] == pytest.approx(-1.96, abs=0.05)
    assert values["crime_btn"] > 0.1


def test_crime_reads_the_lanelet_the_frame_was_turned_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where a turning lanelet overlaps the straight one the ego drives along
    (a junction), CriMe's shared curvilinear frame is built on the straight
    one, as the frame's turn is, whichever the lanelet assignment lists
    first."""
    from types import SimpleNamespace

    import commonroad_clcs.pycrccosy as pycrccosy
    import commonroad_crime.data_structure.base as base
    import commonroad_crime.utility.general as general
    import numpy as np

    from commonroad_metrics.evaluator import _best_lanelet, _shared_frame

    centres = {
        1: np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 20.0]]),  # turning left
        2: np.array([[0.0, 0.0], [20.0, 0.0]]),  # straight on, along +x
    }
    network = SimpleNamespace(
        find_lanelet_by_id=lambda i: SimpleNamespace(center_vertices=centres[i])
    )
    assert next(iter({1, 2})) == 1  # the turning one comes first
    assert _best_lanelet(network, {1, 2}, 0.0) == 2
    assert _best_lanelet(network, [2, 1], math.pi / 2) == 1

    monkeypatch.setattr(general, "generate_reference_path", lambda i, _network: i)
    monkeypatch.setattr(pycrccosy, "CurvilinearCoordinateSystem", lambda path: path)
    ego = SimpleNamespace(
        initial_state=SimpleNamespace(time_step=0, orientation=0.0),
        prediction=SimpleNamespace(center_lanelet_assignment={0: {1, 2}}),
    )
    updates: dict[str, object] = {}
    measure = SimpleNamespace(
        ego_vehicle=ego,
        sce=SimpleNamespace(lanelet_network=network),
        configuration=SimpleNamespace(update=lambda **kwargs: updates.update(kwargs)),
    )
    with _shared_frame():
        base.CriMeBase._update_clcs(measure)
    assert updates == {"CLCS": 2}


def test_a_lat_req_reads_the_lanelet_the_frame_was_turned_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ALatReq reads the lane headings on the lanelet _best_lanelet picks,
    not the first one found under the ego (#2211): with the turning lanelet
    listed first, the straight one's headings are the ones read."""
    from types import SimpleNamespace

    import commonroad_crime.utility.solver as utils_sol
    import numpy as np

    from commonroad_metrics.evaluator import _signed_crime_measures

    centres = {
        1: np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 20.0]]),  # turning left
        2: np.array([[0.0, 0.0], [20.0, 0.0]]),  # straight on, along +x
    }
    lanelets = {i: SimpleNamespace(id=i, center_vertices=c) for i, c in centres.items()}
    network = SimpleNamespace(
        find_lanelet_by_id=lambda i: lanelets[i],
        find_lanelet_by_position=lambda _positions: [[1, 2]],
    )
    read: list[int] = []

    def orientation(lanelet: Any, _position: Any) -> tuple[float, float]:
        read.append(lanelet.id)
        return 3.5, 0.0

    monkeypatch.setattr(utils_sol, "compute_lanelet_width_orientation", orientation)
    measure_cls = _signed_crime_measures()["ALatReq"]
    measure = measure_cls.__new__(measure_cls)
    state = SimpleNamespace(position=np.zeros(2), orientation=0.0)
    measure.validate_update_states_log = lambda *_args, **_kwargs: True
    measure._except_obstacle_in_same_lanelet = lambda **_kwargs: False
    measure._ttc_object = SimpleNamespace(compute=lambda *_args, **_kwargs: math.inf)
    measure.ego_vehicle = measure.other_vehicle = SimpleNamespace(state_at_time=lambda _step: state)
    measure.sce = SimpleNamespace(lanelet_network=network)
    assert measure.compute(7, 0, verbose=False) == 0.0
    assert read == [2, 2]
