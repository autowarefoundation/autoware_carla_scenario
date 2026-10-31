"""The viewer's results on the ODD and its comparison of sessions."""

from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import results, scanner

#: Four cases of a real cut_in sweep's coverage documents (16-09-52 on
#: blackwell2: #1 passed, #8, #9 and #17 failed).
_DOCS = json.loads(
    gzip.decompress(
        (
            Path(__file__).parent / "data" / "ui_results" / "cut_in_coverage.json.gz"
        ).read_bytes()
    )
)

GAP = "dynamic.vehicle_ahead_gap"
SPEED = "dynamic.vehicle_ahead_relative_speed"


def _tick(t: float, ego: tuple[float, float, float], npc: tuple[float, float]) -> str:
    """A tick: the ego at (x, 0) driving at speed (m/s) along x, the NPC at
    (x, y), still."""
    ex, ey, speed = ego
    return json.dumps(
        {
            "type": "tick",
            "frame": int(t * 20),
            "t": t,
            "actors": [
                [1, ex, ey, 0.0, 0.0, 0.0, 0.0, speed, 0.0, 0.0],
                [2, npc[0], npc[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "control": {"1": [0.0, 0.0, 0.0, 1]},
        }
    )


def _trajectory(path: Path, *, stop_short_of: float) -> None:
    """The ego drives at 10 m/s, brakes from t=1 and stops at
    *stop_short_of* m (bumper to bumper) behind an NPC standing at x=50."""
    lines = [
        json.dumps({"type": "world", "schema_version": 2}),
        json.dumps(
            {"type": "actor", "id": 1, "role": "Ego", "extent": [2.0, 1.0, 0.7]}
        ),
        json.dumps(
            {"type": "actor", "id": 2, "role": "npc1", "extent": [2.0, 1.0, 0.7]}
        ),
    ]
    stop_at = 50.0 - 4.0 - stop_short_of  # both half-lengths
    # 10 m/s until t=1 (at x=10), then a constant deceleration to stop there.
    decel = 10.0**2 / (2 * (stop_at - 10.0))
    for step in range(201):  # 10 s: it stops by about 7.6 s
        t = step * 0.05
        tau = max(0.0, t - 1.0)
        speed = max(0.0, 10.0 - decel * tau)
        tau = min(tau, 10.0 / decel)
        x = min(t, 1.0) * 10.0 + 10.0 * tau - decel * tau * tau / 2
        lines.append(_tick(round(t, 2), (x, 0.0, speed), (50.0, 0.0)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_metrics_read_off_a_trajectory_log(tmp_path: Path) -> None:
    log = tmp_path / "Case.trajectory.jsonl"
    _trajectory(log, stop_short_of=3.0)
    metrics = results.case_metrics(log)
    assert metrics["min_gap_m"] == pytest.approx(3.0, abs=0.05)
    assert metrics["max_speed_kmh"] == pytest.approx(36.0)
    assert metrics["final_speed_kmh"] == 0.0
    assert 6.0 < metrics["stop_time_s"] < 8.0
    assert metrics["max_decel_ms2"] > 1.0
    assert metrics["distance_m"] == pytest.approx(43.0, abs=0.1)


def test_a_road_user_parked_under_the_map_is_no_gap(tmp_path: Path) -> None:
    """A hidden road user is parked 500 m under the map, here right under the
    ego's path (#844): it is not 0 m away."""
    log = tmp_path / "Case.trajectory.jsonl"
    lines = [
        json.dumps({"type": "world", "schema_version": 2}),
        json.dumps(
            {"type": "actor", "id": 1, "role": "Ego", "extent": [2.0, 1.0, 0.7]}
        ),
        json.dumps(
            {"type": "actor", "id": 2, "role": "npc1", "extent": [2.0, 1.0, 0.7]}
        ),
    ]
    for step in range(41):
        tick = json.loads(
            _tick(round(step * 0.05, 2), (step * 0.5, 0.0, 10.0), (10.0, 0.0))
        )
        tick["actors"][1][3] = -500.0
        lines.append(json.dumps(tick))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics = results.case_metrics(log)
    assert metrics["distance_m"] == pytest.approx(20.0)
    assert "min_gap_m" not in metrics


def test_records_other_than_ticks_and_actors_are_skipped(tmp_path: Path) -> None:
    """A record that merely names "tick" or "actor" (a label, a role) is not
    read as a tick: a spawn or despawn line, or a collision's role list."""
    plain = tmp_path / "Plain.trajectory.jsonl"
    _trajectory(plain, stop_short_of=3.0)
    lines = plain.read_text(encoding="utf-8").splitlines()
    others = [
        {
            "type": "spawn",
            "frame": 5,
            "t": 0.1,
            "label": "actor",
            "entity": "tick",
            "id": 2,
        },
        {
            "type": "despawn",
            "frame": 9,
            "t": 0.3,
            "label": "actor",
            "entity": "tick",
            "id": 2,
            "reason": "action",
        },
        {
            "type": "collision",
            "frame": 9,
            "t": 0.3,
            "actors": ["tick", "actor"],
            "ids": [2, 3],
            "impulse": 1.0,
            "ego": False,
            "phase": "start",
            "duration": None,
        },
    ]
    mixed = tmp_path / "Mixed.trajectory.jsonl"
    mixed.write_text(
        "\n".join(lines[:5] + [json.dumps(o) for o in others] + lines[5:]) + "\n",
        encoding="utf-8",
    )
    assert results.case_metrics(mixed) == results.case_metrics(plain)


def test_boxes_that_overlap_are_no_distance_apart() -> None:
    a = results._box(0.0, 0.0, 0.0, (2.0, 1.0))
    assert results._box_distance(a, results._box(3.0, 0.5, 30.0, (2.0, 1.0))) == 0.0
    # Side by side, 1 m apart, one turned the other way round.
    b = results._box(0.0, 3.0, 180.0, (2.0, 1.0))
    assert results._box_distance(a, b) == pytest.approx(1.0)
    c = results._box(10.0, 0.0, 90.0, (2.0, 1.0))
    assert math.isclose(results._box_distance(a, c), 7.0)


def test_a_log_without_an_ego_has_no_metrics(tmp_path: Path) -> None:
    log = tmp_path / "Case.trajectory.jsonl"
    log.write_text(json.dumps({"type": "world", "schema_version": 2}) + "\n")
    assert results.case_metrics(log) == {}


# ── Sessions on disk ───────────────────────────────────────────────────────


def _session(
    base: Path,
    time: str,
    outcomes: list[bool | None],
    order: list[int] | None = None,
) -> str:
    """A batch of the fixture's cases (in *order*), each passing or failing
    as *outcomes* says (``None``: errored, no files)."""
    directory = base / "outputs" / "2026-10-18" / time
    directory.mkdir(parents=True)
    entries = []
    stem = 0
    for position, case in enumerate(order or range(len(outcomes))):
        doc = copy.deepcopy(_DOCS[case])
        passed = outcomes[case]
        overrides = ["ego.spawn_lanelet_id=324", *doc["case"]["overrides"]]
        # The sampler's encoded case names the position: not part of the key.
        overrides.append(f"++odd_case=position{position}")
        result = {
            "passed": bool(passed),
            "message": f"case {case} {'passed' if passed else 'failed'}",
            "elapsed_seconds": 5.0 + case,
            "condition_statuses": [],
        }
        if passed is None:
            entries.append(
                {
                    "scenario": f"cut_in#{position + 1}",
                    "overrides": overrides,
                    "error": True,
                }
                | result
            )
            continue
        name = "CutIn" if stem == 0 else f"CutIn-{stem}"
        stem += 1
        doc["outcome"]["passed"] = passed
        (directory / f"{name}_coverage.json").write_text(json.dumps(doc))
        (directory / f"{name}_result.json").write_text(json.dumps(result))
        _trajectory(directory / f"{name}.trajectory.jsonl", stop_short_of=1.0 + case)
        entries.append(
            {"scenario": f"cut_in#{position + 1}", "overrides": overrides} | result
        )
    (directory / "batch_results.json").write_text(json.dumps(entries))
    return f"batch/2026-10-18/{time}"


def test_a_session_s_cases_come_with_their_case_and_metrics(tmp_path: Path) -> None:
    _session(tmp_path, "10-00-00", [True, False, False, False])
    found = results.load_results(tmp_path, "batch", "2026-10-18", "10-00-00")
    assert [r.outcome for r in found] == ["pass", "fail", "fail", "fail"]
    assert found[1].buckets[SPEED] == "[5, 15)"
    assert found[1].metrics["elapsed_s"] == 6.0
    assert found[1].metrics["min_gap_m"] == pytest.approx(2.0, abs=0.05)
    # The most distinct buckets first: three of the speed, two of the gap.
    assert results.default_axes(found) == (SPEED, GAP)
    assert "odd_case" not in found[0].key


def test_the_cases_in_a_cell_are_those_that_spent_time_in_it(tmp_path: Path) -> None:
    _session(tmp_path, "10-00-00", [True, False, False, False])
    found = results.load_results(tmp_path, "batch", "2026-10-18", "10-00-00")
    near = results.cases_in_cell(found, [(GAP, "[0, 10)")])
    # The two cases drawn there, and none of those that never came that near.
    assert {r.index for r in near} >= {2, 3}
    assert results.cases_in_cell(found, []) == found
    assert results.cases_in_cell(found, [(GAP, "no such bucket")]) == []


def test_sessions_are_compared_case_by_case_not_by_position(tmp_path: Path) -> None:
    a = _session(tmp_path, "10-00-00", [True, False, False, False])
    # Run again in another order: case 0 now fails, case 1 passes, case 3
    # errored.
    b = _session(tmp_path, "11-00-00", [False, True, False, None], order=[3, 2, 1, 0])
    loaded = [(ref, results.load_results(tmp_path, *ref.split("/"))) for ref in (a, b)]
    comparison = results.compare(loaded, x=GAP, y=SPEED)
    changes = {c.first.message.split()[1]: c.change() for c in comparison.cases}
    assert changes == {"0": "regression", "1": "fixed", "2": "same", "3": "same"}
    assert comparison.count("regression") == 1
    # Regressions come first.
    assert comparison.cases[0].change() == "regression"
    assert comparison.totals == [(1, 4), (1, 4)]
    regression = comparison.cases[0]
    assert regression.delta("elapsed_s") == 0.0
    # The cell of cases 2 and 3: drawn near, closing in.
    cell = comparison.cells[("[0, 10)", "[-15, -5)")]
    assert cell.tallies == [(0, 2), (0, 2)]
    assert comparison.cells[("[20, 30)", "[-5, 5)")].delta == -1.0


def test_a_case_only_one_session_ran_is_new_or_gone(tmp_path: Path) -> None:
    a = _session(tmp_path, "10-00-00", [True, True, False, False], order=[0, 1])
    b = _session(tmp_path, "11-00-00", [True, True, False, False], order=[1, 2])
    loaded = [(ref, results.load_results(tmp_path, *ref.split("/"))) for ref in (a, b)]
    changes = sorted(c.change() for c in results.compare(loaded).cases)
    assert changes == ["gone", "new", "same"]


def test_a_change_is_between_the_first_and_last_sessions_that_ran_it(
    tmp_path: Path,
) -> None:
    a = _session(tmp_path, "10-00-00", [False, True, True, True], order=[0])
    b = _session(tmp_path, "11-00-00", [True, True, True, True], order=[0])
    c = _session(tmp_path, "12-00-00", [True, True, True, True], order=[1])
    loaded = [
        (ref, results.load_results(tmp_path, *ref.split("/"))) for ref in (a, b, c)
    ]
    by_case = {
        m.first.message.split()[1]: m.change() for m in results.compare(loaded).cases
    }
    # Case 0 is not run by the latest session, and was fixed all the same.
    assert by_case == {"0": "fixed", "1": "new"}


# ── The pages ──────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    scanner.clear_cache()
    return TestClient(app_module.app)


def test_the_odd_page_is_openodd_s_matrix_with_the_cases(
    client: TestClient, tmp_path: Path
) -> None:
    ref = _session(tmp_path, "10-00-00", [True, False, False, False])
    page = client.get(f"/session/{ref}/odd")
    assert page.status_code == 200, page.text
    text = page.text
    # openodd's component: drillable cells, linked back to this page.
    assert 'class="oo-cov"' in text and "data-oo-drill" in text
    assert f'data-oo-link="/session/{ref}/odd?{{conditions}}"' in text
    assert f'data-oo-current="{GAP}"' in text
    assert "All cases" in text
    assert text.count('data-outcome="fail"') == 3
    assert f'href="/session/{ref}/1#recording"' in text


def test_a_drilled_cell_lists_its_cases(client: TestClient, tmp_path: Path) -> None:
    ref = _session(tmp_path, "10-00-00", [True, False, False, False])
    page = client.get(
        f"/session/{ref}/odd",
        params=[("at", f"{GAP}=[20, 30)"), ("at", f"{SPEED}=[5, 15)")],
    )
    assert page.status_code == 200
    assert "Cases in" in page.text
    # The runs that spent time in the cell: case 1, drawn there, among them.
    rows = page.text.split('id="cell-cases"', 1)[1]
    assert 1 <= rows.count("data-outcome=") < 4
    assert 'data-outcome="fail"' in rows


def test_a_session_without_coverage_says_so(client: TestClient, tmp_path: Path) -> None:
    ref = _session(tmp_path, "10-00-00", [True, False, False, False])
    for path in (tmp_path / "outputs").rglob("*_coverage.json"):
        path.unlink()
    page = client.get(f"/session/{ref}/odd")
    assert page.status_code == 200
    assert "left a coverage file" in page.text


def test_the_odd_route_refuses_a_name_outside_the_base(client: TestClient) -> None:
    assert client.get("/session/batch/..%2F..%2Fetc/x/odd").status_code == 404
    assert client.get("/session/nope/2026-10-18/x/odd").status_code == 404


def test_the_compare_page(client: TestClient, tmp_path: Path) -> None:
    a = _session(tmp_path, "10-00-00", [True, False, False, False])
    b = _session(tmp_path, "11-00-00", [False, True, False, None], order=[3, 2, 1, 0])
    # Picked newest first: still compared oldest first.
    page = client.get("/compare", params=[("s", b), ("s", a)])
    assert page.status_code == 200, page.text
    text = page.text
    assert "1 regression(s)" in text and "1 fixed" in text
    summary = text.split('id="compare-summary"', 1)[1]
    assert summary.index(f"{a} (baseline)") < summary.index(b)
    assert 'data-change="regression"' in text
    assert 'id="cell-grid"' in text
    assert f'href="/session/{a}/0#recording"' in text
    # A side-by-side view of the regressed case's runs: #0 in a, #3 in b.
    assert f"/compare/runs?run={a}/0&amp;run={b}/3" in text
    # Both sessions are offered, and picked.
    assert text.count('name="s"') == 2
    assert text.count("checked") >= 2


def test_the_compare_page_with_fewer_than_two_sessions_offers_them(
    client: TestClient, tmp_path: Path
) -> None:
    _session(tmp_path, "10-00-00", [True, False, False, False])
    page = client.get("/compare")
    assert page.status_code == 200
    assert 'name="s"' in page.text
    assert 'id="case-table"' not in page.text


def test_runs_side_by_side(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "autoware_carla_scenario_rerun", None)
    a = _session(tmp_path, "10-00-00", [True, False, False, False])
    b = _session(tmp_path, "11-00-00", [False, True, False, None], order=[3, 2, 1, 0])
    for path in (tmp_path / "outputs").rglob("*.trajectory.jsonl"):
        path.with_name(path.name.replace(".trajectory.jsonl", ".rrd")).write_bytes(
            b"RRF2"
        )
    page = client.get("/compare/runs", params=[("run", f"{a}/0"), ("run", f"{b}/3")])
    assert page.status_code == 200
    assert page.text.count('class="rerun-pane') == 2
    assert f'data-make-url="/api/recording/{a}/0"' in page.text
    assert f'data-make-url="/api/recording/{b}/3"' in page.text
    assert "case 0 passed" in page.text and "case 0 failed" in page.text


def test_the_session_page_links_its_odd_and_comparison(
    client: TestClient, tmp_path: Path
) -> None:
    ref = _session(tmp_path, "10-00-00", [True, False, False, False])
    page = client.get(f"/session/{ref}")
    assert f'href="/session/{ref}/odd"' in page.text
    assert f'href="/compare?s={ref}"' in page.text
    # And the case pages are still the case pages.
    assert client.get(f"/session/{ref}/1").status_code == 200
