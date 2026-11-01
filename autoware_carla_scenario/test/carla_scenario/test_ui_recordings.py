"""The viewer's Rerun recordings: which files are a case's, making a recording
on demand, and where the web viewer is loaded from."""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import cases, recordings, scanner

CONVERTER = "autoware_carla_scenario_rerun"


def _result(passed: bool, elapsed: float, message: str = "") -> dict[str, Any]:
    return {
        "passed": passed,
        "message": message or ("ok" if passed else "failed"),
        "elapsed_seconds": elapsed,
        "condition_statuses": [],
    }


def _write_case(
    directory: Path,
    stem: str,
    result: dict[str, Any] | None,
    case_overrides: list[str] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.trajectory.jsonl").write_text(
        json.dumps({"type": "world", "schema_version": 2}) + "\n", encoding="utf-8"
    )
    if result is not None:
        (directory / f"{stem}_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
    if case_overrides is not None:
        (directory / f"{stem}_coverage.json").write_text(
            json.dumps({"case": {"overrides": case_overrides}}), encoding="utf-8"
        )


def _batch(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    """A local batch of 12 cases of one class, where case 3 raised before it
    opened its log ("errored": no files), and two cases share a result."""
    directory = tmp_path / "outputs" / "2026-10-18" / "16-09-52"
    entries: list[dict[str, Any]] = []
    stem_index = 0
    for i in range(12):
        overrides = ["ego.spawn_lanelet_id=1", f"scenario.gap={i}"]
        if i == 3:
            entries.append(
                {"scenario": f"s#{i}", "overrides": overrides, "error": True}
                | _result(False, 0.0, "raised")
            )
            continue
        # Cases 1 and 2 pass in the same time with the same message.
        result = _result(i % 5 != 4, 5.7 if i in (1, 2) else 5.0 + i)
        stem = "Case" if stem_index == 0 else f"Case-{stem_index}"
        stem_index += 1
        _write_case(directory, stem, result, [f"scenario.gap={i}"])
        entries.append({"scenario": f"s#{i}", "overrides": overrides} | result)
    (directory / "batch_results.json").write_text(json.dumps(entries), encoding="utf-8")
    return directory, entries


# ── Which files are a case's ───────────────────────────────────────────────


def test_stems_come_in_the_order_the_runner_hands_them_out() -> None:
    stems = ["Case-10", "Case-2", "Case", "Case-1"]
    assert sorted(stems, key=cases.stem_order) == [
        "Case",
        "Case-1",
        "Case-2",
        "Case-10",
    ]


def test_a_batch_entry_takes_the_stem_of_its_own_result(tmp_path: Path) -> None:
    directory, entries = _batch(tmp_path)
    stems = cases.match_batch_stems(directory, entries)
    # The errored case has none, and the ones after it are not shifted.
    assert stems == [
        "Case",
        "Case-1",
        "Case-2",
        None,
        "Case-3",
        "Case-4",
        "Case-5",
        "Case-6",
        "Case-7",
        "Case-8",
        "Case-9",
        "Case-10",
    ]


def test_cases_with_the_same_result_are_told_apart_by_their_case(
    tmp_path: Path,
) -> None:
    directory, entries = _batch(tmp_path)
    # Swapped in the summary: each still finds the stem that ran its case.
    entries[1], entries[2] = entries[2], entries[1]
    stems = cases.match_batch_stems(directory, entries)
    assert stems[1:3] == ["Case-2", "Case-1"]


def test_a_slurm_batch_names_its_stems_in_its_manifest(tmp_path: Path) -> None:
    directory, entries = _batch(tmp_path)
    manifest = {
        "version": 1,
        "cases": [
            {"index": i, "stem": f"Other-{i}" if i else "Other"}
            for i in range(len(entries))
        ],
    }
    (directory / "slurm").mkdir()
    (directory / "slurm" / "manifest.json").write_text(json.dumps(manifest))
    files = cases.batch_case_files(directory)
    assert [f.stem if f else None for f in files][:3] == ["Other", "Other-1", "Other-2"]


@pytest.mark.parametrize(
    ("session_type", "date", "time"),
    [
        ("batch", "..", "x"),
        ("batch", "2026-10-18", "a/b"),
        ("nope", "2026-10-18", "16-09-52"),
    ],
)
def test_a_session_name_cannot_leave_the_base(
    tmp_path: Path, session_type: str, date: str, time: str
) -> None:
    assert cases.session_dir(tmp_path, session_type, date, time) is None
    assert cases.case_files(tmp_path, session_type, date, time, 0) is None


def test_the_scenario_page_names_the_case_s_own_log(tmp_path: Path) -> None:
    directory, _ = _batch(tmp_path)
    scanner.clear_cache()
    view = scanner.load_scenario(tmp_path, "batch", "2026-10-18", "16-09-52", 4)
    assert view is not None and view.trajectory_path is not None
    assert (
        Path(view.trajectory_path) == (directory / "Case-3.trajectory.jsonl").resolve()
    )


# ── Making a recording ─────────────────────────────────────────────────────


class _FakeConverter(types.ModuleType):
    """The converter's API (convert, RERUN_VERSION, ConversionError)."""

    def __init__(self, fail: bool = False) -> None:
        super().__init__(CONVERTER)
        self.RERUN_VERSION = "0.38.1"
        self.__version__ = "0.1.0"
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

        class ConversionError(Exception):
            pass

        self.ConversionError = ConversionError

    def convert(self, jsonl: str, out_rrd: str, **_: Any) -> dict[str, Any]:
        self.calls.append((jsonl, out_rrd))
        if self.fail:
            raise self.ConversionError(f"{jsonl}: line 1: not a record")
        Path(out_rrd).write_bytes(b"RRF2fake")
        return {"warnings": []}


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> _FakeConverter:
    fake = _FakeConverter()
    monkeypatch.setitem(sys.modules, CONVERTER, fake)
    return fake


@pytest.fixture
def no_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules: `import` raises ImportError, installed or not.
    monkeypatch.setitem(sys.modules, CONVERTER, None)


@pytest.fixture(autouse=True)
def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(cache))
    monkeypatch.delenv("VIEWER_RERUN_WEB_VIEWER", raising=False)
    monkeypatch.setenv("VIEWER_RERUN_MAP_DIR", str(tmp_path / "maps"))
    return cache


def test_the_recording_is_named_as_the_converter_names_it() -> None:
    log = Path("/x/CutInScenario-3.trajectory.jsonl")
    assert recordings.recording_path(log) == Path("/x/CutInScenario-3.rrd")


def test_a_recording_is_made_once_beside_the_log(
    tmp_path: Path, converter: _FakeConverter
) -> None:
    _write_case(tmp_path, "Case", None)
    log = tmp_path / "Case.trajectory.jsonl"
    made = recordings.ensure_recording(log)
    assert made == tmp_path / "Case.rrd"
    assert made.read_bytes() == b"RRF2fake"
    assert recordings.ensure_recording(log) == made
    assert len(converter.calls) == 1
    # Written under another name and renamed: nothing else is left behind.
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "cache") == [
        "Case.rrd",
        "Case.trajectory.jsonl",
    ]


def test_a_recording_older_than_its_log_is_made_again(
    tmp_path: Path, converter: _FakeConverter
) -> None:
    _write_case(tmp_path, "Case", None)
    log = tmp_path / "Case.trajectory.jsonl"
    rrd = tmp_path / "Case.rrd"
    rrd.write_bytes(b"old")
    os.utime(rrd, (1, 1))
    assert recordings.ensure_recording(log).read_bytes() == b"RRF2fake"
    # Made into the cache: the one beside the log is left as it was.
    assert rrd.read_bytes() == b"old"


def test_a_stale_recording_with_sensors_beside_the_log_is_never_replaced(
    tmp_path: Path, converter: _FakeConverter, _cache: Path
) -> None:
    # A run with rerun.lidar / rerun.cameras merged its sensors into Case.rrd;
    # a copy that does not keep the times (cp -r) left the log newer.
    _write_case(tmp_path, "Case", None)
    log = tmp_path / "Case.trajectory.jsonl"
    rrd = tmp_path / "Case.rrd"
    rrd.write_bytes(b"trajectory + LiDAR + cameras")
    now = time.time()
    os.utime(rrd, (now - 5, now - 5))
    os.utime(log, (now, now))
    made = recordings.ensure_recording(log)
    assert made.parent == _cache / "recordings"
    assert made.read_bytes() == b"RRF2fake"
    assert rrd.read_bytes() == b"trajectory + LiDAR + cameras"
    # Served from the cache from now on, without converting again.
    assert recordings.ensure_recording(log) == made
    assert len(converter.calls) == 1
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "cache") == [
        "Case.rrd",
        "Case.trajectory.jsonl",
    ]


def test_the_cache_keeps_one_recording_a_log(
    tmp_path: Path, converter: _FakeConverter, _cache: Path
) -> None:
    _write_case(tmp_path, "Case", None)
    log = tmp_path / "Case.trajectory.jsonl"
    rrd = tmp_path / "Case.rrd"
    rrd.write_bytes(b"made by the run")
    os.utime(rrd, (1, 1))
    first = recordings.ensure_recording(log)
    # The log changes again: the cached recording is made again, in place.
    os.utime(first, (2, 2))
    second = recordings.ensure_recording(log)
    assert second == first
    assert len(converter.calls) == 2
    assert [p.name for p in (_cache / "recordings").iterdir()] == [first.name]
    assert rrd.read_bytes() == b"made by the run"


def test_a_fresh_recording_beside_the_log_wins_over_the_cache(
    tmp_path: Path, converter: _FakeConverter
) -> None:
    _write_case(tmp_path, "Case", None)
    log = tmp_path / "Case.trajectory.jsonl"
    rrd = tmp_path / "Case.rrd"
    rrd.write_bytes(b"old")
    os.utime(rrd, (1, 1))
    recordings.ensure_recording(log)
    # Converted again beside the log (python -m autoware_carla_scenario.rerun).
    rrd.write_bytes(b"converted again, sensors kept")
    assert recordings.ensure_recording(log) == rrd
    assert len(converter.calls) == 1


def test_a_recording_already_there_needs_no_converter(
    tmp_path: Path, no_converter: None
) -> None:
    _write_case(tmp_path, "Case", None)
    (tmp_path / "Case.rrd").write_bytes(b"made by the run")
    made = recordings.ensure_recording(tmp_path / "Case.trajectory.jsonl")
    assert made.read_bytes() == b"made by the run"


def test_without_the_converter_it_says_what_to_install(
    tmp_path: Path, no_converter: None
) -> None:
    _write_case(tmp_path, "Case", None)
    assert not recordings.converter_available()
    with pytest.raises(recordings.ConverterMissingError) as error:
        recordings.ensure_recording(tmp_path / "Case.trajectory.jsonl")
    assert "pip install 'autoware-carla-scenario[rerun]'" in str(error.value)
    assert "autoware-carla-scenario-rerun" in str(error.value)


def test_a_failed_conversion_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, CONVERTER, _FakeConverter(fail=True))
    _write_case(tmp_path, "Case", None)
    with pytest.raises(recordings.RecordingError, match="not a record"):
        recordings.ensure_recording(tmp_path / "Case.trajectory.jsonl")
    assert not list(tmp_path.glob("*.rrd"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
def test_a_read_only_session_keeps_its_recording_in_the_cache(
    tmp_path: Path, converter: _FakeConverter, _cache: Path
) -> None:
    session = tmp_path / "ro"
    _write_case(session, "Case", None)
    session.chmod(0o555)
    try:
        made = recordings.ensure_recording(session / "Case.trajectory.jsonl")
    finally:
        session.chmod(0o755)
    assert made.parent == _cache / "recordings"
    assert not (session / "Case.rrd").exists()


# ── The web viewer ─────────────────────────────────────────────────────────


def test_the_web_viewer_is_the_recordings_version_from_the_cdn(
    converter: _FakeConverter,
) -> None:
    converter.RERUN_VERSION = "0.38.0"
    viewer = recordings.web_viewer()
    assert (viewer.source, viewer.version) == ("cdn", "0.38.0")
    assert viewer.base_url == "https://unpkg.com/@rerun-io/web-viewer@0.38.0"


def test_without_the_converter_the_web_viewer_is_the_default_version(
    no_converter: None,
) -> None:
    assert recordings.web_viewer().version == recordings.DEFAULT_WEB_VIEWER_VERSION


def _package_tarball(
    path: Path, version: str, wasm: bytes = b"\0asm", omit: str | None = None
) -> Path:
    files = {
        "package/package.json": json.dumps({"version": version}).encode(),
        "package/index.js": b"export class WebViewer {}",
        "package/re_viewer.js": b"export default null;",
        "package/re_viewer_bg.wasm": wasm,
        "package/README.md": b"readme",
    }
    files.pop(f"package/{omit}", None)
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_a_fetched_copy_is_served_by_the_app(
    tmp_path: Path, converter: _FakeConverter, _cache: Path
) -> None:
    tarball = _package_tarball(tmp_path / "web-viewer-0.38.1.tgz", "0.38.1")
    target = recordings.fetch_web_viewer(tarball=tarball)
    assert target == _cache / "rerun-web-viewer" / "0.38.1"
    viewer = recordings.web_viewer()
    assert (viewer.source, viewer.base_url) == ("local", "/rerun-web-viewer/0.38.1")
    client = TestClient(app_module.app)
    wasm = client.get("/rerun-web-viewer/0.38.1/re_viewer_bg.wasm")
    assert wasm.status_code == 200
    assert wasm.headers["content-type"] == "application/wasm"
    # index.js imports "./re_viewer", without the extension.
    module = client.get("/rerun-web-viewer/0.38.1/re_viewer")
    assert module.text == "export default null;"
    assert module.headers["content-type"].startswith("text/javascript")
    for name in ("README.md", "..%2Fpackage.json"):
        assert client.get(f"/rerun-web-viewer/0.38.1/{name}").status_code == 404
    assert client.get("/rerun-web-viewer/0.38.0/index.js").status_code == 404


def _cache_listing(cache: Path) -> dict[str, bytes]:
    root = cache / "rerun-web-viewer"
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_a_tarball_of_another_version_is_refused(
    tmp_path: Path, converter: _FakeConverter, _cache: Path
) -> None:
    good = recordings.fetch_web_viewer(
        tarball=_package_tarball(tmp_path / "good.tgz", "0.38.1")
    )
    before = _cache_listing(_cache)
    tarball = _package_tarball(tmp_path / "web-viewer.tgz", "0.37.0")
    with pytest.raises(recordings.RecordingError, match="0.37.0, not 0.38.1"):
        recordings.fetch_web_viewer(tarball=tarball)
    # The good copy is untouched and still served.
    assert _cache_listing(_cache) == before
    viewer = recordings.web_viewer()
    assert (viewer.version, viewer.source, viewer.directory) == (
        "0.38.1",
        "local",
        good,
    )


@pytest.mark.parametrize("bad", ["version", "missing", "corrupt"])
def test_a_refused_tarball_leaves_nothing_behind(
    tmp_path: Path, converter: _FakeConverter, _cache: Path, bad: str
) -> None:
    if bad == "version":
        tarball = _package_tarball(tmp_path / "bad.tgz", "0.37.0")
    elif bad == "missing":
        tarball = _package_tarball(tmp_path / "bad.tgz", "0.38.1", omit="index.js")
    else:
        tarball = tmp_path / "bad.tgz"
        tarball.write_bytes(b"not a tarball")
    with pytest.raises(recordings.RecordingError):
        recordings.fetch_web_viewer(tarball=tarball)
    root = _cache / "rerun-web-viewer"
    assert not root.exists() or list(root.iterdir()) == []


def test_a_fetch_replaces_the_old_copy(
    tmp_path: Path, converter: _FakeConverter, _cache: Path
) -> None:
    old = _package_tarball(tmp_path / "old.tgz", "0.38.1", wasm=b"\0asm old")
    recordings.fetch_web_viewer(tarball=old)
    target = _cache / "rerun-web-viewer" / "0.38.1"
    (target / "stale.txt").write_text("left over")
    new = _package_tarball(tmp_path / "new.tgz", "0.38.1", wasm=b"\0asm new")
    assert recordings.fetch_web_viewer(tarball=new) == target
    assert (target / "re_viewer_bg.wasm").read_bytes() == b"\0asm new"
    assert not (target / "stale.txt").exists()
    assert sorted(p.name for p in target.parent.iterdir()) == ["0.38.1"]


@pytest.mark.parametrize("failing", ["aside", "into place"])
def test_a_failed_swap_keeps_the_old_copy(
    tmp_path: Path,
    converter: _FakeConverter,
    _cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
) -> None:
    recordings.fetch_web_viewer(
        tarball=_package_tarball(tmp_path / "old.tgz", "0.38.1", wasm=b"\0asm old")
    )
    before = _cache_listing(_cache)
    replace = os.replace
    calls = []

    def flaky(src: Path, dst: Path) -> None:
        calls.append(src)
        if len(calls) == (1 if failing == "aside" else 2):
            raise OSError("no room")
        replace(src, dst)

    monkeypatch.setattr(recordings.os, "replace", flaky)
    new = _package_tarball(tmp_path / "new.tgz", "0.38.1", wasm=b"\0asm new")
    with pytest.raises(OSError, match="no room"):
        recordings.fetch_web_viewer(tarball=new)
    # The old copy is back in place, and nothing else is left in the cache.
    assert _cache_listing(_cache) == before
    assert sorted(p.name for p in (_cache / "rerun-web-viewer").iterdir()) == ["0.38.1"]


@pytest.mark.parametrize(
    ("setting", "source", "base_url"),
    [
        ("none", "none", None),
        (
            "https://mirror.example/web-viewer/",
            "url",
            "https://mirror.example/web-viewer",
        ),
    ],
)
def test_the_web_viewer_setting(
    monkeypatch: pytest.MonkeyPatch,
    converter: _FakeConverter,
    setting: str,
    source: str,
    base_url: str | None,
) -> None:
    monkeypatch.setenv("VIEWER_RERUN_WEB_VIEWER", setting)
    viewer = recordings.web_viewer()
    assert (viewer.source, viewer.base_url) == (source, base_url)


def test_an_unpacked_package_named_by_the_setting_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converter: _FakeConverter
) -> None:
    with tarfile.open(_package_tarball(tmp_path / "p.tgz", "0.38.1")) as archive:
        for member in archive.getmembers():
            source = archive.extractfile(member)
            assert source is not None
            target = tmp_path / "unpacked" / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
    monkeypatch.setenv("VIEWER_RERUN_WEB_VIEWER", str(tmp_path / "unpacked"))
    viewer = recordings.web_viewer()
    assert viewer.source == "local"
    assert viewer.directory == tmp_path / "unpacked" / "package"


# ── The pages and routes ───────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _batch(tmp_path)
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    scanner.clear_cache()
    return TestClient(app_module.app)


SESSION = "batch/2026-10-18/16-09-52"


def test_the_scenario_page_embeds_the_viewer(
    client: TestClient, converter: _FakeConverter
) -> None:
    page = client.get(f"/session/{SESSION}/4")
    assert page.status_code == 200
    assert 'id="rerun-viewer"' in page.text
    assert f'"/api/recording/{SESSION}/4"' in page.text
    assert '"https://unpkg.com/@rerun-io/web-viewer@0.38.1"' in page.text
    assert "Case-3.trajectory.jsonl" in page.text


def test_without_the_converter_the_page_says_what_to_install(
    client: TestClient, no_converter: None
) -> None:
    page = client.get(f"/session/{SESSION}/4")
    assert 'id="rerun-viewer"' not in page.text
    assert "The Rerun converter is not installed" in page.text
    assert "pip install &#39;autoware-carla-scenario[rerun]&#39;" in page.text
    made = client.post(f"/api/recording/{SESSION}/4")
    assert made.status_code == 503
    assert made.json()["missing_converter"] is True
    assert "autoware-carla-scenario-rerun" in made.json()["install_hint"]


def test_a_recording_is_made_and_served(
    client: TestClient, converter: _FakeConverter, tmp_path: Path
) -> None:
    made = client.post(f"/api/recording/{SESSION}/4")
    assert made.status_code == 200, made.text
    body = made.json()
    assert body["url"] == f"/recording/{SESSION}/4.rrd"
    assert body["path"].endswith("Case-3.rrd")
    served = client.get(body["url"])
    assert served.status_code == 200
    assert served.content == b"RRF2fake"
    assert 'filename="Case-3.rrd"' in served.headers["content-disposition"]
    # Made once, and the page now names it.
    client.get(body["url"])
    assert len(converter.calls) == 1
    assert "Case-3.rrd" in client.get(f"/session/{SESSION}/4").text


def test_a_case_without_a_log_has_no_recording(
    client: TestClient, converter: _FakeConverter
) -> None:
    # Case 3 raised before it opened its log.
    assert client.post(f"/api/recording/{SESSION}/3").status_code == 404
    assert client.get(f"/recording/{SESSION}/3.rrd").status_code == 404
    assert client.get(f"/recording/{SESSION}/99.rrd").status_code == 404
    assert 'id="recording"' not in client.get(f"/session/{SESSION}/3").text


def test_a_failed_conversion_is_reported(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, CONVERTER, _FakeConverter(fail=True))
    made = client.post(f"/api/recording/{SESSION}/0")
    assert made.status_code == 500
    assert "not a record" in made.json()["error"]
