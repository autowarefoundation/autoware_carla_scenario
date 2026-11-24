"""The viewer's pages: a batch case's own overrides, a session with no case,
a missing session or case, and the Run panel's script (#2344)."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.slurm.manifest import MANIFEST_VERSION, BatchLayout
from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import scanner

DATE = "2026-01-01"
TIME = "00-00-00"
BATCH = f"batch/{DATE}/{TIME}"
HTML = {"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    scanner.clear_cache()
    yield
    scanner.clear_cache()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    return TestClient(app_module.app, base_url="http://127.0.0.1")


def _result(passed: bool) -> dict[str, Any]:
    return {
        "passed": passed,
        "message": "ok" if passed else "failed",
        "elapsed_seconds": 1.0,
        "condition_statuses": [],
    }


def _batch(base: Path, entries: list[dict[str, Any]]) -> Path:
    directory = base / "outputs" / DATE / TIME
    directory.mkdir(parents=True)
    (directory / "batch_results.json").write_text(json.dumps(entries))
    return directory


# ── 1. A batch case's own overrides ───────────────────────────────────────


def test_a_batch_case_shows_its_own_overrides_and_can_be_run_again(
    tmp_path: Path, client: TestClient
) -> None:
    """A batch directory has no ``.hydra/overrides.yaml``: each entry's
    ``overrides`` are its case's, with ``scenario=`` added from its label
    (less the ``#k`` a sweep numbers it with), so Re-run is offered."""
    _batch(
        tmp_path,
        [
            {"scenario": "lane_change/cut_in#1", "overrides": ["ego.speed=10"]}
            | _result(True),
            {"scenario": "lane_change/cut_in#2", "overrides": ["ego.speed=20"]}
            | _result(False),
            # Not expanded: no overrides of its own.
            {"scenario": "intersection_passing/left_turn"} | _result(True),
            # Names its scenario already: kept as it is.
            {"scenario": "x#1", "overrides": ["scenario=a/b", "k=1", 5]}
            | _result(True),
            # A label whose '#' numbers nothing is the scenario's name.
            {"scenario": "odd#name", "overrides": []} | _result(True),
            # Names no scenario: nothing to add.
            {"overrides": ["k=2"]} | _result(True),
        ],
    )

    items = scanner.load_session(tmp_path, "batch", DATE, TIME)

    want = [
        ["scenario=lane_change/cut_in", "ego.speed=10"],
        ["scenario=lane_change/cut_in", "ego.speed=20"],
        ["scenario=intersection_passing/left_turn"],
        ["scenario=a/b", "k=1", "5"],
        ["scenario=odd#name"],
        ["k=2"],
    ]
    assert [i.overrides for i in items] == want
    for index, overrides in enumerate(want):
        view = scanner.load_scenario(tmp_path, "batch", DATE, TIME, index)
        assert view is not None
        assert view.overrides == overrides

    page = client.get(f"/session/{BATCH}/1")
    assert page.status_code == 200, page.text
    assert "rerunScenario()" in page.text
    assert json.dumps(want[1]) in page.text


def test_a_batch_case_says_its_re_run_lacks_the_batch_command_line(
    tmp_path: Path, client: TestClient
) -> None:
    """A batch records each case's sweep overrides, not the overrides its
    command line gave every case: the Re-run is labelled as running with the
    listed overrides alone, and the page says why.  A multirun job's
    ``.hydra/overrides.yaml`` is its whole command line: plain Re-run."""
    _batch(tmp_path, [{"scenario": "a/b#1", "overrides": ["k=1"]} | _result(True)])
    job = tmp_path / "multirun" / DATE / TIME / "0"
    (job / ".hydra").mkdir(parents=True)
    (job / ".hydra" / "overrides.yaml").write_text("- scenario=a/b\n- k=1\n")
    (job / "b_result.json").write_text(json.dumps(_result(True)))

    batch = client.get(f"/session/{BATCH}/0").text
    multirun = client.get(f"/session/multirun/{DATE}/{TIME}/0").text

    assert 'id="rerun-case-overrides"' in batch
    assert "Re-run with these overrides" in batch
    assert 'id="batch-overrides-note"' in batch
    assert "rerunScenario()" in multirun
    assert 'id="rerun-case-overrides"' not in multirun
    assert 'id="batch-overrides-note"' not in multirun


def test_a_pending_batch_case_keeps_its_overrides_and_says_why(
    tmp_path: Path, client: TestClient
) -> None:
    """A case the batch has not run yet is served (200) with its overrides
    and the entry's message, not as a case that failed to start."""
    _batch(
        tmp_path,
        [
            {"scenario": "a/b#1", "overrides": ["k=1"]} | _result(True),
            {
                "scenario": "a/b#2",
                "overrides": ["k=2"],
                "passed": None,
                "message": "No result yet (the batch has not run this case)",
                "elapsed_seconds": None,
                "pending": True,
            },
        ],
    )

    view = scanner.load_scenario(tmp_path, "batch", DATE, TIME, 1)

    assert view is not None
    assert view.passed is None
    assert view.overrides == ["scenario=a/b", "k=2"]
    assert view.message == "No result yet (the batch has not run this case)"
    page = client.get(f"/session/{BATCH}/1")
    assert page.status_code == 200, page.text
    assert "rerunScenario()" in page.text


def test_a_slurm_batch_case_shows_its_case_overrides(
    tmp_path: Path, client: TestClient
) -> None:
    """A SLURM batch not collected yet reads its cases off the manifest
    (``case_overrides``), as a collected one's summary records them."""
    layout = BatchLayout(tmp_path / "outputs" / DATE / TIME)
    for path, document in (
        (
            layout.manifest,
            {
                "version": MANIFEST_VERSION,
                "cases": [
                    {
                        "index": 0,
                        "label": "a/b#1",
                        "scenario": "a/b",
                        "overrides": ["k=1", "launcher=slurm"],
                        "case_overrides": ["k=1"],
                    },
                    {
                        "index": 1,
                        "label": "a/b#2",
                        "scenario": "a/b",
                        "overrides": ["k=2", "launcher=slurm"],
                        "case_overrides": ["k=2"],
                    },
                ],
                "launcher": {},
            },
        ),
        (layout.submission(0), {"index": 0, "chunks": [[0, 1]]}),
        (layout.result(0), {"submission": 0, "result": _result(True)}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))

    items = scanner.load_session(tmp_path, "batch", DATE, TIME)

    assert [(i.passed, i.pending, i.overrides) for i in items] == [
        (True, False, ["scenario=a/b", "k=1"]),
        (None, True, ["scenario=a/b", "k=2"]),
    ]
    view = scanner.load_scenario(tmp_path, "batch", DATE, TIME, 1)
    assert view is not None
    assert view.overrides == ["scenario=a/b", "k=2"]
    assert client.get(f"/session/{BATCH}/1").status_code == 200


# ── 3. A session with no case ─────────────────────────────────────────────


def test_a_session_with_no_case_shows_no_results_yet(
    tmp_path: Path, client: TestClient
) -> None:
    """Never "0/0 All passed" in green, nor a progress bar's fill."""
    (tmp_path / "multirun" / DATE / TIME).mkdir(parents=True)

    page = client.get(f"/session/multirun/{DATE}/{TIME}")

    assert page.status_code == 200, page.text
    header = page.text[page.text.index('<div class="text-right">') :]
    header = header[: header.index("<!-- Result filter")]
    assert "No results yet" in header
    assert "All passed" not in header
    assert "text-green-600" not in header
    assert "bg-green-500" not in header


def test_a_session_with_cases_still_shows_its_pass_rate(
    tmp_path: Path, client: TestClient
) -> None:
    _batch(tmp_path, [{"scenario": "a"} | _result(True)])

    page = client.get(f"/session/{BATCH}")

    assert "All passed" in page.text
    assert "No results yet" not in page.text
    assert 'style="width: 100.0%"' in page.text


# ── 4. A missing session or case ──────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        f"/session/{BATCH}",
        f"/session/{BATCH}/0",
        f"/session/{BATCH}/odd",
        "/no/such/page",
    ],
)
def test_a_missing_page_is_a_page_for_a_browser(client: TestClient, path: str) -> None:
    page = client.get(path, headers=HTML)

    assert page.status_code == 404
    assert page.headers["content-type"].startswith("text/html")
    assert 'id="error-page"' in page.text
    assert 'href="/"' in page.text
    # The viewer's own template: its navigation bar and Run panel.
    assert "Scenario Viewer" in page.text
    # The guard's headers are on it too.
    assert page.headers["x-frame-options"] == "DENY"


def test_a_missing_page_elsewhere_is_not_called_a_session(client: TestClient) -> None:
    page = client.get("/static/no-such-file.js", headers=HTML)

    assert page.status_code == 404
    assert 'id="error-page"' in page.text
    assert "There is no such page here." in page.text
    assert "session or case" not in page.text


@pytest.mark.parametrize(
    "headers",
    [
        # A script, a stylesheet, a Rerun viewer's fetch: not a navigation.
        {"Accept": "*/*"},
        # An htmx request asks for HTML partials, but only of /api/.
        {"Accept": "*/*", "HX-Request": "true"},
    ],
)
def test_a_missing_static_file_fetched_by_a_page_is_json(
    client: TestClient, headers: dict[str, str]
) -> None:
    answer = client.get("/static/no-such-file.js", headers=headers)

    assert answer.status_code == 404
    assert answer.headers["content-type"] == "application/json"


@pytest.mark.parametrize(
    ("path", "headers"),
    [
        (f"/session/{BATCH}", {"Accept": "application/json"}),
        (f"/session/{BATCH}", {}),
        ("/api/slurm/2026-01-01/00-00-00/status", HTML),
        ("/api/no-such-route", HTML),
    ],
)
def test_the_api_and_clients_not_asking_for_html_get_json(
    client: TestClient, path: str, headers: dict[str, str]
) -> None:
    answer = client.get(path, headers=headers)

    assert answer.status_code == 404
    assert answer.headers["content-type"] == "application/json"
    assert "detail" in answer.json()


def test_a_case_the_session_does_not_have_is_404(
    tmp_path: Path, client: TestClient
) -> None:
    """A case index past a batch's cases, a multirun job that is not there,
    any but case 0 of a single run: 404, not 200 "No result data found"."""
    _batch(tmp_path, [{"scenario": "a"} | _result(True)])
    job = tmp_path / "multirun" / DATE / TIME / "0"
    job.mkdir(parents=True)
    single = tmp_path / "outputs" / DATE / "00-00-01"
    single.mkdir(parents=True)
    (single / "a_result.json").write_text(json.dumps({"scenario": "a"} | _result(True)))

    for path in (
        f"/session/{BATCH}/1",
        f"/session/multirun/{DATE}/{TIME}/1",
        f"/session/single/{DATE}/00-00-01/1",
    ):
        page = client.get(path, headers=HTML)
        assert page.status_code == 404, path
        assert "This session has no such case." in page.text
    for path in (
        f"/session/{BATCH}/0",
        # A job still running, with no result yet: there, and served.
        f"/session/multirun/{DATE}/{TIME}/0",
        f"/session/single/{DATE}/00-00-01/0",
    ):
        assert client.get(path).status_code == 200, path


def test_a_single_run_without_a_result_yet_is_still_served(
    tmp_path: Path, client: TestClient
) -> None:
    """A single run still running has no result file, and no item; its case
    0 is there all the same."""
    single = tmp_path / "outputs" / DATE / TIME
    single.mkdir(parents=True)
    (single / "scenario.log").write_text("starting\n")

    assert client.get(f"/session/single/{DATE}/{TIME}/0").status_code == 200
    assert client.get(f"/session/single/{DATE}/{TIME}/1").status_code == 404


# ── 5, 6. The Run panel's script ──────────────────────────────────────────


def _function(page: str, name: str) -> str:
    """The source of base.html's function *name*, as the page serves it."""
    start = page.index(f"    function {name}(")
    end = page.index("\n    }\n", start) + 7
    return page[start:end]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_run_panel_keeps_delete_overrides(
    client: TestClient, tmp_path: Path
) -> None:
    """``~key`` (Hydra's delete) has no ``=`` and was dropped (#2344)."""
    script = _function(client.get("/").text, "overrideLines") + (
        "console.log(JSON.stringify(overrideLines("
        + json.dumps(" a=1 \n~b\n\n~c=2\nnot an override\n  ~d  ")
        + ")));\n"
    )
    harness = tmp_path / "overrides.js"
    harness.write_text(script, encoding="utf-8")
    done = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["a=1", "~b", "~c=2", "~d"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_run_panel_sends_flag_lines_for_the_server_to_refuse(
    client: TestClient, tmp_path: Path
) -> None:
    """A line starting with '-' is sent, not dropped, so the server's 400
    says why in the preview and the run (#2440)."""
    script = _function(client.get("/").text, "overrideLines") + (
        "console.log(JSON.stringify(overrideLines("
        + json.dumps("a=1\n--multirun\n --cfg=job \nnot an override")
        + ")));\n"
    )
    harness = tmp_path / "overrides.js"
    harness.write_text(script, encoding="utf-8")
    done = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["a=1", "--multirun", "--cfg=job"]


def test_the_server_accepts_delete_overrides() -> None:
    body = app_module.parse_run_body({"scenario": "a/b", "extra_overrides": ["~k"]})
    assert body.extra_overrides == ["~k"]


#: Runs base.html's loadLaunchers against a fetch that fails, then one that
#: answers, with the elements it touches stubbed.
_LAUNCHERS_HARNESS = """
function element() {
  const classes = new Set(['hidden']);
  return {textContent: '', disabled: true, value: 'local', innerHTML: '',
    classList: {add: (c) => classes.add(c), remove: (c) => classes.delete(c),
      toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)),
      contains: (c) => classes.has(c)},
    appendChild() {}, querySelector: () => option};
}
const option = element();
const elements = {'run-launcher': element(), 'slurm-unavailable': element(),
  'slurm-partitions': element(), 'slurm-options': element()};
globalThis.document = {getElementById: (id) => elements[id],
  createElement: () => element()};
let answers = [];
globalThis.fetch = () => answers.shift()();
const settle = () => new Promise((r) => setImmediate(r));
let launchersLoaded = false;
__SCRIPT__
(async () => {
  const seen = [];
  const note = elements['slurm-unavailable'];
  for (const answer of [
    () => Promise.reject(new TypeError('Failed to fetch')),
    () => Promise.resolve({ok: false, status: 502, json: async () => ({})}),
    () => Promise.resolve({ok: true, status: 200, json: async () => (
      {local: {available: true}, slurm: {available: true, reason: '', partitions: []}})}),
  ]) {
    answers.push(answer);
    loadLaunchers();
    await settle(); await settle(); await settle();
    seen.push({loaded: launchersLoaded, disabled: option.disabled,
      note: note.textContent, hidden: note.classList.contains('hidden')});
  }
  console.log(JSON.stringify(seen));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_failed_launcher_check_says_why_slurm_is_disabled(
    client: TestClient, tmp_path: Path
) -> None:
    """No unhandled rejection: SLURM stays disabled and ``#slurm-unavailable``
    says why, and the check is made again (#2344)."""
    page = client.get("/").text
    script = _LAUNCHERS_HARNESS.replace(
        "__SCRIPT__",
        _function(page, "loadLaunchers") + _function(page, "showLauncherOptions"),
    )
    harness = tmp_path / "launchers.js"
    harness.write_text(script, encoding="utf-8")
    done = subprocess.run(
        ["node", "--unhandled-rejections=strict", str(harness)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    failed, refused, answered = json.loads(done.stdout)
    assert failed["loaded"] is False and failed["disabled"] is True
    assert failed["hidden"] is False
    assert "Failed to fetch" in failed["note"]
    assert refused["loaded"] is False and "HTTP 502" in refused["note"]
    assert answered == {
        "loaded": True,
        "disabled": False,
        "note": refused["note"],
        "hidden": True,
    }


#: Runs base.html's toggleRunPanel against a /api/scenarios that fails, then
#: one that answers.
_SCENARIOS_HARNESS = """
function element() {
  const classes = new Set(['hidden']);
  return {children: [], value: '',
    classList: {toggle: (c) => (classes.has(c) ? classes.delete(c) : classes.add(c)),
      contains: (c) => classes.has(c)},
    appendChild(child) { this.children.push(child); }};
}
const elements = {'run-panel': element(), 'scenario-list': element()};
globalThis.document = {getElementById: (id) => elements[id] || null,
  createElement: () => element()};
let answers = [];
globalThis.fetch = () => answers.shift()();
const settle = () => new Promise((r) => setImmediate(r));
let launchersLoaded = true;
function updateCmdPreview() {}
console.warn = () => {};
__SCRIPT__
(async () => {
  const seen = [];
  for (const answer of [
    () => Promise.resolve({ok: false, status: 500, json: async () => ({})}),
    () => Promise.resolve({ok: true, status: 200, json: async () => ['a/b']}),
  ]) {
    answers.push(answer);
    toggleRunPanel();  // open
    await settle(); await settle(); await settle();
    toggleRunPanel();  // close
    seen.push({loaded: scenarioListLoaded,
      options: elements['scenario-list'].children.map((o) => o.value)});
  }
  console.log(JSON.stringify(seen));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_failed_scenario_list_is_loaded_again(
    client: TestClient, tmp_path: Path
) -> None:
    page = client.get("/").text
    start = page.index("    let scenarioListLoaded = false;")
    end = page.index("\n    }\n", page.index("function toggleRunPanel(")) + 7
    script = _SCENARIOS_HARNESS.replace("__SCRIPT__", page[start:end])
    harness = tmp_path / "scenarios.js"
    harness.write_text(script, encoding="utf-8")
    done = subprocess.run(
        ["node", "--unhandled-rejections=strict", str(harness)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    failed, answered = json.loads(done.stdout)
    assert failed == {"loaded": False, "options": []}
    assert answered["loaded"] is True
    assert answered["options"][-1] == "a/b"
