"""How a SLURM batch's manifest (``slurm/manifest.json``) is read.

Pinned against the manifests a submitter (``slurm/submit.py``), a hand,
another version or outside corruption may have left: each key a submitter
writes (``version``, ``created``, ``batch_dir``, ``batch_dir_relative``,
``cwd``, ``python``, ``argv``, ``scenario_names``, ``overrides``, ``expand``,
``build_scenario_fn``, ``typechecked``, ``launcher``, ``env``,
``batch_settings``, ``cases``) and each key of a case (``index``, ``label``,
``scenario``, ``overrides``, ``case_overrides``, ``scenario_class``,
``stem``, ``config``, ``files``), missing or of every JSON type -- NaN,
infinities, a number no float holds, an integer no float holds exactly,
non-ASCII text -- in a manifest of version 1 and of version 2, manifests
older than ``stem``, ``batch_settings`` and ``batch_dir_relative``, cases
that are not objects, keys named twice, files that hold no manifest, and
manifests nested deep.

Each is compared with what the readers of a batch made of the manifest when
it was read as a ``dict`` (the ``_old_*`` oracles below, the code that read
it): what ``read_manifest`` gives for each key, as ``manifest[key]``,
``manifest.get(key)`` and ``key in manifest``; and, with the manifest read as
a ``dict`` and the oracles in place of its checked readers, what ``status``,
``collect``, ``resubmit --dry-run``, ``missing_cases``, ``unfinished_cases``,
``resubmit_target``, the gate, the scripts' inputs, a task's start
(``worker.run_task``, up to building its cases) and the viewer's batch status
print, log, write and return.
"""

from __future__ import annotations

import json
import logging
import operator
import shutil
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import pytest

import autoware_carla_scenario.slurm.manifest as manifest_mod
from autoware_carla_scenario.batch_report import print_summary
from autoware_carla_scenario.slurm import cli, collect as collect_mod, submit, worker
from autoware_carla_scenario.slurm import status as slurm_status
from autoware_carla_scenario.slurm.manifest import (
    BatchLayout,
    manifest_of_text,
    recorded_verdict,
    write_json,
)
from autoware_carla_scenario.slurm.settings import SlurmSettings

from . import test_slurm_follow
from .test_slurm_batch_readers import old_reading

# -- The corpus -------------------------------------------------------------------

#: A member left out of its object.
_ABSENT = "<absent>"

#: A value of every JSON type, as its text in the file.
_VALUES: list[str] = [
    _ABSENT,
    "null",
    "true",
    "false",
    "0",
    "1",
    "-1",
    "2",
    "5",
    "1.0",
    "1.5",
    "-0.0",
    "NaN",
    "Infinity",
    "-Infinity",
    "1e400",
    "123456789012345678901234567890",
    "1" + "0" * 400,
    '""',
    '"text"',
    '"1"',
    '" 1 "',
    '"\\u0661"',
    '"\\u00fc ber \\u2603"',
    '"\\ud800"',
    "[]",
    "[1]",
    '["a", "b"]',
    '["x=1", 2, null]',
    "{}",
    '{"1": 1}',
    '{"k": "v"}',
]


def _case(index: int) -> dict[str, str]:
    """Case *index* as a submitter writes it: each key's JSON text."""
    return {
        "index": str(index),
        "label": f'"s#{index + 1}"',
        "scenario": '"demo"',
        "overrides": f'["speed={index}"]',
        "case_overrides": f'["speed={index}"]',
        "scenario_class": '"Demo"',
        "stem": f'"stem{index}"',
        "config": '{"map": "/m/\\u00fc.osm", "speed": 1.5, "seed": [1, -0.0]}',
        "files": '{"/m/\\u00fc.osm": "abc"}',
    }


#: The keys of a case, in the order the submitter writes them.
_CASE_KEYS = tuple(_case(0))

#: What read_manifest holds as JSON text of a manifest and of each case: what
#: the checked readers read (the rest is read from the document json.loads()
#: read, through Manifest.get() and manifest[key]).
_HELD = ("version", "expand", "batch_dir", "batch_dir_relative")
_CASE_HELD = ("index", "label", "stem", "case_overrides")


def _object(members: dict[str, str]) -> str:
    """The text of the object of *members* (each a value's JSON text)."""
    return (
        "{"
        + ", ".join(
            f"{json.dumps(key)}: {value}"
            for key, value in members.items()
            if value != _ABSENT
        )
        + "}"
    )


def _cases(**values: str) -> str:
    """The three cases, case 1 with *values* in place of (or after) its own."""
    return (
        "["
        + ", ".join(
            _object({**_case(i), **(values if i == 1 else {})}) for i in range(3)
        )
        + "]"
    )


#: The manifest of version 1 (no ``remote_cwd``), as a submitter writes it.
_V1: dict[str, str] = {
    "version": "1",
    "created": '"2026-11-12T00:00:00"',
    "batch_dir": '"/submit/out/\\u00fc"',
    "cwd": '"/submit"',
    "python": '"/usr/bin/python3"',
    "argv": '["scenario", "launcher=slurm"]',
    "scenario_names": '["demo"]',
    "overrides": '["launcher=slurm"]',
    "expand": "true",
    "build_scenario_fn": "null",
    "typechecked": "false",
    "launcher": '{"wait": false, "partition": "p\\u00fc", "cases_per_task": 2}',
    "env": '{"HOME": "/home/u", "LANG": "C.UTF-8"}',
    "batch_settings": '{"first_case": 0, "rerun_cases": [], "canvas_case": null}',
    "cases": _cases(),
}

#: The manifest of version 2 (``remote_cwd``): ``batch_dir_relative`` after
#: ``batch_dir``.
_V2: dict[str, str] = {}
for _key, _value in _V1.items():
    _V2[_key] = "2" if _key == "version" else _value
    if _key == "batch_dir":
        _V2["batch_dir_relative"] = '"out/\\u00fc"'

#: The keys of a manifest, in the order the submitter writes them.
_KEYS = tuple(_V2)


def _manifest(base: dict[str, str], **values: str) -> str:
    return _object({**base, **values})


def _corpus() -> list[tuple[str, str]]:
    """(name, the text of the manifest)."""
    corpus: list[tuple[str, str]] = [("v1", _manifest(_V1)), ("v2", _manifest(_V2))]
    for key in _KEYS:
        if key == "cases":
            continue
        for value in _VALUES:
            corpus.append((f"v1-{key}-{value}", _manifest(_V1, **{key: value})))
    for key in ("version", "batch_dir", "batch_dir_relative", "cwd", "python", "env"):
        for value in _VALUES:
            corpus.append((f"v2-{key}-{value}", _manifest(_V2, **{key: value})))
    for key in _CASE_KEYS:
        for value in _VALUES:
            corpus.append(
                (f"case-{key}-{value}", _manifest(_V1, cases=_cases(**{key: value})))
            )
    # Case 1 with expand off: its case_overrides are not read.
    for value in ("[1]", "1", "null", '"ab"'):
        corpus.append(
            (
                f"no-expand-case_overrides-{value}",
                _manifest(_V1, expand="false", cases=_cases(case_overrides=value)),
            )
        )
    # The cases of every JSON type, and cases that are not objects.
    good = [_object(_case(i)) for i in range(3)]
    for value in [*_VALUES, '"abc"', '{"a": 1, "b": 2}']:
        corpus.append((f"cases-{value}", _manifest(_V1, cases=value)))
    for position in range(3):
        for value in ("1", "null", "true", "1.5", '"x"', "[]", '["index"]', "[0]"):
            cases = list(good)
            cases[position] = value
            corpus.append(
                (
                    f"case{position}-is-{value}",
                    _manifest(_V1, cases="[" + ", ".join(cases) + "]"),
                )
            )
    corpus += [
        ("cases-one", _manifest(_V1, cases="[" + good[0] + "]")),
        ("cases-reversed", _manifest(_V1, cases="[" + ", ".join(good[::-1]) + "]")),
        (
            "cases-twice",
            _manifest(_V1, cases="[" + ", ".join(good + good) + "]"),
        ),
        (
            "cases-out-of-range",
            _manifest(_V1, cases=_cases(index="7")),
        ),
        (
            "cases-negative",
            _manifest(_V1, cases=_cases(index="-1")),
        ),
    ]
    # Older manifests: no stem (before #1957), no batch_settings (before
    # #1225), no batch_dir_relative (version 1), and the least a manifest
    # can be read with.
    no_stem = (
        "[" + ", ".join(_object({**_case(i), "stem": _ABSENT}) for i in range(3)) + "]"
    )
    corpus += [
        ("old-no-stem", _manifest(_V1, cases=no_stem)),
        ("old-no-batch-settings", _manifest(_V1, batch_settings=_ABSENT)),
        (
            "old-no-stem-no-settings",
            _manifest(_V1, batch_settings=_ABSENT, cases=no_stem),
        ),
        ("v2-no-batch-settings", _manifest(_V2, batch_settings=_ABSENT)),
        (
            "minimal",
            '{"version": 1, "cases": [{"index": 0, "label": "a"}], "launcher": {}}',
        ),
        (
            "minimal-no-launcher",
            '{"version": 1, "cases": [{"index": 0, "label": "a"}]}',
        ),
        ("only-version", '{"version": 1}'),
        ("reversed", _object(dict(reversed(list(_V2.items()))))),
        (
            "extra-keys",
            _manifest(_V1, z='[1, {"y": NaN}]', **{"\\u00fc": "[Infinity, -0.0]"}),
        ),
    ]
    # Keys named twice: the first place and the last value.
    corpus += [
        ("twice-version", '{"version": 3, "cases": [], "version": 1}'),
        ("twice-version-bad", '{"version": 1, "cases": [], "version": 3}'),
        (
            "twice-cases",
            '{"version": 1, "cases": [{"index": 0, "label": "a"}], "launcher": {}, '
            '"cases": [{"index": 1, "label": "b"}]}',
        ),
        (
            "twice-in-a-case",
            '{"version": 1, "launcher": {}, "cases": [{"index": 0, "label": "a", '
            '"index": 2, "stem": "s", "label": "c"}]}',
        ),
        (
            "twice-batch-dir",
            _manifest(_V2)[:-1] + ', "batch_dir": "/again", "batch_dir_relative": ""}',
        ),
    ]
    # Files that hold no manifest.
    corpus += [
        ("not-json", "{not json"),
        ("truncated", _manifest(_V1)[:-7]),
        ("empty", ""),
        ("array", "[" + _manifest(_V1) + "]"),
        ("string", '"x"'),
        ("number", "1"),
        ("null", "null"),
        ("trailing", _manifest(_V1) + " x"),
        ("version-too-long", '{"version": ' + "1" * 5000 + ', "cases": []}'),
        ("index-too-long", _manifest(_V1, cases=_cases(index="1" * 5000))),
    ]
    return corpus


_CORPUS = _corpus()
_BYTES: dict[str, bytes] = {
    "bom": b"\xef\xbb\xbf" + _manifest(_V1).encode(),
    "not-utf-8": _manifest(_V1).encode().replace(b"demo", b"de\xffmo"),
    "latin-1": _manifest(_V1, created='"\\u00fc"')
    .replace("\\u00fc", "ü")
    .encode("latin-1"),
}


def _all_contents() -> list[tuple[str, bytes]]:
    return [(name, text.encode("utf-8")) for name, text in _CORPUS] + list(
        _BYTES.items()
    )


_CONTENTS = _all_contents()
_IDS = [name for name, _ in _CONTENTS]


# -- The manifest as it was read: a dict ----------------------------------------------


def _old_read_manifest(self: BatchLayout) -> dict[str, Any]:
    """BatchLayout.read_manifest(), reading the manifest as a dict."""
    if not self.manifest.is_file():
        raise FileNotFoundError(
            f"{self.batch_dir} holds no SLURM batch (no {self.manifest})"
        )
    manifest = manifest_mod.read_json(self.manifest)
    if not isinstance(manifest, dict):
        raise ValueError(
            f"{self.manifest} is not a SLURM batch manifest "
            "(it holds JSON that is not an object)"
        )
    self._check_version(json.dumps(manifest.get("version")))
    return manifest


def _old_resubmit_target(layout: BatchLayout, manifest: Any) -> str:
    if manifest.get("batch_dir_relative") and manifest.get("batch_dir"):
        return str(manifest["batch_dir"])
    return str(layout.batch_dir)


def _old_missing_cases(layout: BatchLayout) -> list[int]:
    manifest = layout.read_manifest()
    return [
        case["index"]
        for case in manifest["cases"]
        if layout.read_result(int(case["index"])) is None
    ]


def _old_unfinished_cases(layout: BatchLayout, *, failed: bool = False) -> list[int]:
    manifest = layout.read_manifest()
    chosen: list[int] = []
    for case in manifest["cases"]:
        index = int(case["index"])
        record = layout.read_result(index)
        verdict = recorded_verdict(record)
        if (
            record is None
            or verdict is None
            or record.get("errored")
            or (failed and not verdict)
        ):
            chosen.append(index)
    return chosen


def _old_collect(layout: BatchLayout, *, quiet_hint: bool = False) -> int:
    """collect(), reading the manifest as a dict."""
    manifest = layout.read_manifest()
    target = _old_resubmit_target(layout, manifest)
    labels: list[str] = []
    results: list[Any] = []
    errored: list[int] = []
    missing: list[int] = []
    advice: dict[str, set[int]] = {}
    unreadable: list[int] = []
    rerun_errors: dict[int, str] = {}
    for position, case in enumerate(manifest["cases"]):
        index = int(case["index"])
        labels.append(str(case["label"]))
        record = layout.read_result(index)
        if record is None:
            missing.append(position)
            message, kind, about = collect_mod._missing_message(layout, index, target)
            about_kind = advice.setdefault(kind, set())
            if about is not None:
                about_kind.add(about)
            results.append(collect_mod._error_result(message))
            continue
        rerun_error = record.rerun_error
        if rerun_error is not None:
            rerun_errors[position] = rerun_error
        result = record.result()
        if result is None or recorded_verdict(record) is None:
            unreadable.append(position)
            results.append(
                collect_mod._error_result(
                    collect_mod._no_verdict_message(layout, index, target)
                )
            )
            continue
        results.append(collect_mod.RecordedResult(result))
        if record.errored:
            errored.append(position)
    overrides = (
        [list(case["case_overrides"]) for case in manifest["cases"]]
        if manifest.get("expand")
        else None
    )
    named = [str(case.get("stem") or "") for case in manifest["cases"]]
    stems = named if all(named) else None
    all_passed = print_summary(
        labels,
        results,
        output_dir=Path(layout.batch_dir),
        overrides=overrides,
        errored=sorted({*errored, *missing, *unreadable}),
        rerun_errors=rerun_errors,
        stems=stems,
    )
    if unreadable and not quiet_hint:
        print(  # noqa: T201
            f"{len(unreadable)} case(s) have a result with no verdict: "
            + ", ".join(labels[i] for i in unreadable)
            + f"\nRun them again with: scenario-slurm resubmit {target}"
        )
    if missing and not quiet_hint:
        print(  # noqa: T201
            f"{len(missing)} case(s) have no result: "
            + ", ".join(labels[i] for i in missing)
            + "\n"
            + collect_mod._missing_hint(layout, target, advice)
        )
    write_json(
        layout.root / "collected.json",
        {
            "cases": len(labels),
            "passed": sum(1 for r in results if r.passed),
            "errored": [labels[i] for i in sorted({*errored, *unreadable})],
            "missing": [labels[i] for i in missing],
        },
    )
    return 0 if all_passed else 1


@contextmanager
def _old_reading(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every reader of the batch reading the manifest as a dict."""
    with monkeypatch.context() as patch:
        patch.setattr(BatchLayout, "read_manifest", _old_read_manifest)
        patch.setattr(manifest_mod, "resubmit_target", _old_resubmit_target)
        patch.setattr(collect_mod, "resubmit_target", _old_resubmit_target)
        patch.setattr(collect_mod, "missing_cases", _old_missing_cases)
        patch.setattr(collect_mod, "unfinished_cases", _old_unfinished_cases)
        patch.setattr(collect_mod, "collect", _old_collect)
        patch.setattr(cli, "unfinished_cases", _old_unfinished_cases)
        patch.setattr(cli, "collect", _old_collect)
        # resubmit's --cases picking, as it read the manifest then (with
        # the old unfinished_cases, collect and busy check set on cli).
        patch.setattr(cli, "_resubmit", test_slurm_follow._old_resubmit)
        # status, cancel, the busy check and the viewer's case states, as
        # they read the records then.
        with old_reading(patch):
            yield


# -- The batch ------------------------------------------------------------------


def _passing_record(index: int) -> str:
    return json.dumps(
        {
            "index": index,
            "label": f"s#{index + 1}",
            "stem": f"stem{index}",
            "result": {
                "passed": True,
                "message": "ok",
                "elapsed_seconds": 1.5,
                "condition_statuses": [],
            },
            "errored": False,
        }
    )


def _failing_record(index: int) -> str:
    return json.dumps(
        {
            "index": index,
            "label": f"s#{index + 1}",
            "result": {
                "passed": False,
                "message": "boom",
                "elapsed_seconds": 2.0,
                "condition_statuses": [
                    {"label": "c", "satisfied": False, "message": "no"}
                ],
            },
            "errored": True,
        }
    )


def _batch(tmp_path: Path, content: bytes) -> BatchLayout:
    """A batch whose manifest is *content*, whose case 0 passed, case 1 has
    no outcome, and case 2 errored; submission 0 gave cases 0 and 1 to task
    0, which is running, and case 2 to task 1."""
    layout = BatchLayout(tmp_path / "pristine" / "batch")
    layout.manifest.parent.mkdir(parents=True, exist_ok=True)
    layout.manifest.write_bytes(content)
    layout.result(0).parent.mkdir(parents=True, exist_ok=True)
    layout.result(0).write_text(_passing_record(0), encoding="utf-8")
    layout.result(2).write_text(_failing_record(2), encoding="utf-8")
    write_json(
        layout.submission(0),
        {
            "index": 0,
            "created": "2026-11-12T00:00:00",
            "submitted_at": 1.0,
            "cases": [0, 1, 2],
            "chunks": [[0, 1], [2]],
            "launcher": {},
            "job_id": "70",
            "collect_job_id": None,
        },
    )
    write_json(layout.task(0, 0), {"state": "running", "current": 1})
    return layout


def _outcome(call: Callable[[], Any]) -> Any:
    """What *call* returns, or the type and text of what it raises."""
    try:
        return ("returns", call())
    except Exception as exc:  # noqa: BLE001
        return ("raises", type(exc).__name__, str(exc))


def _text(value: Any) -> str:
    """*value* as JSON text (repr for what JSON cannot write): NaN equal to
    NaN and -0.0 told from 0.0."""
    try:
        return json.dumps(value, indent=1, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return repr(value)


def _same(actual: Any, expected: Any) -> None:
    assert _text(actual) == _text(expected)


def _logs(caplog: pytest.LogCaptureFixture) -> list[tuple[str, int, str]]:
    return [(r.name, r.levelno, r.getMessage()) for r in caplog.records]


def _view(layout: BatchLayout) -> Any:
    """What read_manifest gives of each key a submitter writes, and of one it
    never writes: ``manifest[key]``, ``manifest.get(key)``, ``key in
    manifest``; and of each case, ``case[key]`` and ``case.get(key)``."""
    manifest = layout.read_manifest()
    view: list[Any] = []
    for key in (*_KEYS, "z", "ü"):
        view.append(
            (
                key,
                _text(_outcome(partial(operator.getitem, manifest, key))),
                _text(_outcome(partial(manifest.get, key))),
                key in manifest,
            )
        )
    view.append(_text(_outcome(lambda: manifest.get("env", {}))))
    view.append(_text(_outcome(lambda: manifest.get("python", ""))))
    cases = _outcome(lambda: list(manifest["cases"]))
    if cases[0] == "returns":
        for case in cases[1]:
            view.append(repr(case))
            for key in (*_CASE_KEYS, "z"):
                view.append(
                    (
                        _text(_outcome(partial(operator.getitem, case, key))),
                        _text(_outcome(partial(case.get, key))),
                    )
                )
    return view


class _Stop(Exception):
    """Where the fake ``examples.run`` stops a task: before it builds."""


def _fake_run() -> types.ModuleType:
    run = types.ModuleType("autoware_carla_scenario.examples.run")

    def stop() -> None:
        raise _Stop("stopped before building")

    run._cap_malloc_arenas = stop  # type: ignore[attr-defined]
    return run


def _run_task(layout: BatchLayout) -> Any:
    """A task's start: what it returns, and what its status says."""
    status = worker.run_task(layout.manifest, 0, 0)
    record = json.loads(layout.task(0, 0).read_text(encoding="utf-8"))
    return status, {
        key: record.get(key) for key in ("state", "cases", "error", "current")
    }


def _script_inputs(layout: BatchLayout) -> list[str]:
    """What the scripts read of the manifest, whether they index ``python``
    and ``cwd`` or not."""

    def inputs(python: bool, cwd: bool) -> Any:
        return submit._script_inputs(
            layout.read_manifest(),
            python=python,
            cwd=cwd,
            env=layout.read_manifest().get("env", {}),
        )

    return [
        repr(_outcome(partial(inputs, python, cwd)))
        for python in (True, False)
        for cwd in (True, False)
    ]


def _readers(layout: BatchLayout) -> dict[str, Callable[[], Any]]:
    """Each reader of the batch's manifest, as a call."""
    from autoware_carla_scenario.gate.evaluate import _slurm_unrecorded
    from autoware_carla_scenario.ui import slurm_launch

    batch = str(layout.batch_dir)
    settings = SlurmSettings()
    remote = SlurmSettings(python=".venv/bin/python", remote_cwd="/remote")
    return {
        "view": lambda: _view(layout),
        "status": lambda: cli.main(["status", batch]),
        "collect": lambda: cli.main(["collect", batch]),
        "collect-quiet": lambda: collect_mod.collect(layout, quiet_hint=True),
        "resubmit-dry": lambda: cli.main(["resubmit", batch, "--dry-run"]),
        "resubmit-dry-failed": lambda: cli.main(
            ["resubmit", batch, "--dry-run", "--failed", "--force"]
        ),
        "resubmit-dry-cases": lambda: cli.main(
            ["resubmit", batch, "--dry-run", "--force", "--cases", "s#1,1"]
        ),
        "resubmit-dry-wait": lambda: cli.main(
            ["resubmit", batch, "--dry-run", "--force", "--wait", "--cases", "0"]
        ),
        "missing-cases": lambda: collect_mod.missing_cases(layout),
        "unfinished-cases": lambda: collect_mod.unfinished_cases(layout),
        "unfinished-failed": lambda: collect_mod.unfinished_cases(layout, failed=True),
        "resubmit-target": lambda: collect_mod.resubmit_target(
            layout, layout.read_manifest()
        ),
        "gate": lambda: _slurm_unrecorded(layout.batch_dir),
        "script-inputs": lambda: _script_inputs(layout),
        "job-script": lambda: submit.job_script(
            layout, layout.read_manifest(), settings, 0
        ),
        "job-script-remote": lambda: submit.job_script(
            layout, layout.read_manifest(), remote, 0
        ),
        "collect-script": lambda: submit.collect_script(
            layout.read_manifest(), settings, layout
        ),
        "collect-command": lambda: submit._collect_command(
            layout, layout.read_manifest(), remote, "70"
        ),
        "batch-status": lambda: _text(slurm_launch.batch_status(layout.batch_dir)),
        "run-task": lambda: _run_task(layout),
    }


def _run_readers(
    pristine: BatchLayout,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """What each reader returns or raises, prints, logs and writes, each on a
    copy of the batch *pristine*."""
    layout = BatchLayout(tmp_path / "work" / "batch")
    seen: dict[str, Any] = {}
    for name, call in _readers(layout).items():
        if names is not None and name not in names:
            continue
        shutil.rmtree(layout.batch_dir, ignore_errors=True)
        shutil.copytree(pristine.batch_dir, layout.batch_dir)
        capsys.readouterr()
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            outcome = _outcome(call)
        captured = capsys.readouterr()
        files = {
            str(path.relative_to(layout.batch_dir)): path.read_bytes()
            for path in sorted(layout.batch_dir.rglob("*"))
            if path.is_file() and path != layout.task(0, 0)
        }
        seen[name] = (
            _text(outcome),
            captured.out,
            captured.err,
            _logs(caplog),
            files,
        )
    return seen


@pytest.fixture
def quiet_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """squeue lists no task of any job, a task starts without CARLA, the
    scenarios or signal handlers, and stops before it builds its cases."""
    from autoware_carla_scenario.ui import slurm_launch

    for module in (slurm_status, submit, slurm_launch):
        monkeypatch.setattr(module, "squeue_states", lambda job: {})
    monkeypatch.setattr(slurm_status, "cancel_jobs", lambda jobs: True)
    package = types.ModuleType("autoware_carla_scenario.examples")
    package.__path__ = []  # type: ignore[attr-defined]
    run = _fake_run()
    package.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autoware_carla_scenario.examples", package)
    monkeypatch.setitem(sys.modules, "autoware_carla_scenario.examples.run", run)
    monkeypatch.setattr(
        worker,
        "signal",
        types.SimpleNamespace(
            signal=lambda *_args: None, SIGTERM=15, SIGUSR1=10, SIG_IGN=1
        ),
    )
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "AUTOWARE_CARLA_SCENARIO_TYPECHECK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")


# -- What every reader makes of each manifest ----------------------------------------


def test_the_corpus_covers_every_kind_of_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    outcomes = []
    for _, content in _CONTENTS:
        path.write_bytes(content)
        outcomes.append(_outcome(lambda: manifest_mod.read_json(path))[0])
    assert outcomes.count("raises") == 9
    assert len(_CONTENTS) > 750


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_what_every_reader_makes_of_each_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    quiet_slurm: None,
    name: str,
    content: bytes,
) -> None:
    pristine = _batch(tmp_path, content)
    actual = _run_readers(pristine, tmp_path, capsys, caplog)
    with _old_reading(monkeypatch):
        expected = _run_readers(pristine, tmp_path, capsys, caplog)
    assert actual == expected


def test_a_batch_without_a_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    quiet_slurm: None,
) -> None:
    pristine = _batch(tmp_path, b"{}")
    pristine.manifest.unlink()
    actual = _run_readers(pristine, tmp_path, capsys, caplog)
    with _old_reading(monkeypatch):
        expected = _run_readers(pristine, tmp_path, capsys, caplog)
    assert actual == expected
    assert "holds no SLURM batch" in actual["status"][2]


# -- Exact text -----------------------------------------------------------------------


def test_what_the_readers_make_of_a_manifest_as_a_submitter_writes_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    quiet_slurm: None,
) -> None:
    pristine = _batch(tmp_path, _manifest(_V2).encode())
    seen = _run_readers(pristine, tmp_path, capsys, caplog)
    batch = tmp_path / "work" / "batch"
    assert seen["missing-cases"][0] == _text(("returns", [1]))
    assert seen["unfinished-cases"][0] == _text(("returns", [1, 2]))
    assert seen["unfinished-failed"][0] == _text(("returns", [1, 2]))
    assert seen["resubmit-target"][0] == _text(("returns", "/submit/out/ü"))
    assert seen["gate"][0] == _text(("returns", [["s#2", "s#3"], 3]))
    assert seen["collect"][0] == _text(("returns", 1))
    assert json.loads(seen["collect"][4]["slurm/collected.json"]) == {
        "cases": 3,
        "passed": 1,
        "errored": ["s#3"],
        "missing": ["s#2"],
    }
    assert seen["collect"][4]["slurm/collected.json"].decode() == (
        '{\n  "cases": 3,\n  "passed": 1,\n  "errored": [\n    "s#3"\n  ],\n'
        '  "missing": [\n    "s#2"\n  ]\n}\n'
    )
    assert "  [ERROR] s#2 (0.0s)" in seen["collect"][1]
    assert "1 case(s) have no result: s#2" in seen["collect"][1]
    assert seen["resubmit-dry"][1] == "Resubmitting 2 case(s): s#2, s#3\n"
    assert seen["resubmit-dry-cases"][1] == "Resubmitting 2 case(s): s#1, s#2\n"
    assert seen["resubmit-dry-wait"][1] == "Resubmitting 1 case(s): s#1\n"
    assert seen["status"][1].startswith(f"Batch {batch}: 3 case(s)\n")
    assert seen["run-task"][0] == _text(
        (
            "returns",
            [
                1,
                {
                    "state": "failed",
                    "cases": [0, 1],
                    "error": "_Stop: stopped before building",
                    "current": None,
                },
            ],
        )
    )


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        ("[1]", "a case without an index (1)"),
        ('[{"label": "a"}]', "a case without an index ({'label': 'a'})"),
        (
            '[{"index": "x", "config": {"a": [1.0, NaN]}}]',
            "a case without an index ({'index': 'x', 'config': {'a': [1.0, nan]}})",
        ),
    ],
)
def test_the_gate_names_a_case_without_an_index_as_python_shows_it(
    tmp_path: Path, cases: str, message: str
) -> None:
    from autoware_carla_scenario.gate.evaluate import _slurm_unrecorded

    pristine = _batch(tmp_path, _manifest(_V1, cases=cases).encode())
    assert _slurm_unrecorded(pristine.batch_dir) == f"{pristine.manifest}: {message}"


# -- The typed accessors --------------------------------------------------------------


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_typed_accessors_of_each_manifest(
    tmp_path: Path, name: str, content: bytes
) -> None:
    """What the checked readers ask of a manifest, as they asked it of the
    manifest read as a dict."""
    layout = BatchLayout(tmp_path / "batch")
    layout.manifest.parent.mkdir(parents=True)
    layout.manifest.write_bytes(content)
    old = _outcome(lambda: _old_read_manifest(layout))
    new = _outcome(layout.read_manifest)
    if old[0] == "raises":
        assert new == old
        return
    document, manifest = old[1], new[1]
    # The members the checked readers read, and only those, held as text.
    assert manifest.keys() == list(document)
    assert manifest.members.keys() == [k for k in document if k in _HELD]
    for key in manifest.members.keys():
        assert manifest.members.value_text(key) == json.dumps(
            document[key], ensure_ascii=False
        )
    for key in document:
        assert key in manifest and manifest.has(key)
        _same(manifest[key], document[key])
        assert type(manifest[key]) is type(document[key])
    assert "no such key" not in manifest
    assert manifest.version_text() == json.dumps(
        document.get("version"), ensure_ascii=False
    )
    for key in ("batch_dir", "batch_dir_relative", "expand", "no such key"):
        assert manifest.is_true(key) is bool(document.get(key))
        _same(
            _outcome(partial(manifest.text, key)),
            _outcome(partial(lambda key: str(document[key]), key)),
        )
    _same(
        _outcome(lambda: [type(c).__name__ for c in manifest.cases()]),
        _outcome(lambda: ["ManifestCase" for _ in document["cases"]]),
    )
    if not isinstance(document.get("cases"), (list, dict, str)):
        return
    for case, raw in zip(manifest.cases(), document["cases"]):
        pairs: list[tuple[Callable[[], Any], Callable[[], Any]]] = [
            (case.index, partial(lambda raw: int(raw["index"]), raw)),
            (case.index_value, partial(lambda raw: raw["index"], raw)),
            (case.label, partial(lambda raw: str(raw["label"]), raw)),
            (case.stem, partial(lambda raw: str(raw.get("stem") or ""), raw)),
            (
                case.case_overrides,
                partial(lambda raw: list(raw["case_overrides"]), raw),
            ),
        ]
        if isinstance(raw, dict):
            assert case.members.keys() == [k for k in raw if k in _CASE_HELD]
            for key in case.members.keys():
                assert case.members.value_text(key) == json.dumps(
                    raw[key], ensure_ascii=False
                )
        for typed, read in pairs:
            _same(_outcome(typed), _outcome(read))


def test_a_member_not_held_as_text_is_refused_not_misread() -> None:
    """What the manifest holds as text is what the checked readers read; a
    typed accessor asked for another member it has refuses, rather than
    read it as missing."""
    manifest = manifest_of_text(
        '{"version": 1, "cwd": "/x", "typechecked": true, "batch_dir": "/b", '
        '"cases": [{"index": 0, "label": "a", "config": {"k": 1}}]}'
    )
    assert manifest is not None
    assert manifest.members.keys() == ["version", "batch_dir"]
    assert manifest.keys() == ["version", "cwd", "typechecked", "batch_dir", "cases"]
    for key in ("cwd", "typechecked", "cases"):
        with pytest.raises(ValueError, match=f"holds no '{key}' as JSON text"):
            manifest.is_true(key)
        with pytest.raises(ValueError, match=f"holds no '{key}' as JSON text"):
            manifest.text(key)
    assert manifest.is_true("expand") is False
    assert manifest.text("batch_dir") == "/b"
    assert manifest["cwd"] == "/x" and manifest.get("typechecked") is True
    (case,) = manifest.cases()
    assert case.members.keys() == ["index", "label"]
    with pytest.raises(ValueError, match="holds no 'config' as JSON text"):
        case._value("config")
    with pytest.raises(ValueError, match="holds no 'config' as JSON text"):
        case._get("config")
    assert manifest["cases"][0]["config"] == {"k": 1}


@pytest.mark.parametrize(
    ("text", "shape"),
    [
        ("1", "0"),
        ("-1.5", "0.0"),
        ("true", "false"),
        ("null", "null"),
        ('"x"', '""'),
        ("[1, [2]]", "[]"),
    ],
)
def test_a_case_that_is_no_object_raises_as_its_type_does(
    text: str, shape: str
) -> None:
    manifest = manifest_of_text(f'{{"version": 1, "cases": [{text}]}}')
    assert manifest is not None
    (case,) = manifest.cases()
    raw = json.loads(text)
    for typed, read in [
        (case.index, lambda: int(raw["index"])),
        (case.label, lambda: str(raw["label"])),
        (case.stem, lambda: str(raw.get("stem") or "")),
        (case.case_overrides, lambda: list(raw["case_overrides"])),
    ]:
        with pytest.raises(Exception) as typed_error:
            typed()
        with pytest.raises(Exception) as read_error:
            read()
        assert type(typed_error.value) is type(read_error.value)
        assert str(typed_error.value) == str(read_error.value)
    assert case.members.keys() == []


@pytest.mark.parametrize(
    ("cases", "error", "message"),
    [
        (None, KeyError, "'cases'"),
        ("1", TypeError, "'int' object is not iterable"),
        ("1.5", TypeError, "'float' object is not iterable"),
        ("true", TypeError, "'bool' object is not iterable"),
        ("null", TypeError, "'NoneType' object is not iterable"),
    ],
)
def test_cases_python_cannot_iterate_raise_as_they_did(
    cases: str | None, error: type[Exception], message: str
) -> None:
    text = '{"version": 1}' if cases is None else f'{{"version": 1, "cases": {cases}}}'
    manifest = manifest_of_text(text)
    assert manifest is not None
    with pytest.raises(error) as info:
        manifest.cases()
    assert type(info.value) is error
    assert str(info.value) == message


@pytest.mark.parametrize(
    ("cases", "count"), [('"abc"', 3), ('""', 0), ('{"a": 1, "b": 2}', 2), ("[]", 0)]
)
def test_cases_of_a_string_or_an_object_are_its_characters_or_keys(
    cases: str, count: int
) -> None:
    manifest = manifest_of_text(f'{{"version": 1, "cases": {cases}}}')
    assert manifest is not None
    assert len(manifest.cases()) == count
    for case in manifest.cases():
        with pytest.raises(TypeError, match="string indices must be integers"):
            case.index()


@pytest.mark.parametrize("text", ["[]", '"x"', "1", "null", "true"])
def test_manifest_of_text_is_none_for_json_that_is_no_object(text: str) -> None:
    assert manifest_of_text(text) is None


def test_collected_json_is_written_as_write_json_writes_it(tmp_path: Path) -> None:
    labels = ["s#1", "\u00fc \u2603", 'q"uote', "back\\slash"]
    cases = ", ".join(
        f'{{"index": {i}, "label": {json.dumps(label)}, "stem": "s{i}"}}'
        for i, label in enumerate(labels)
    )
    layout = BatchLayout(tmp_path / "batch")
    layout.manifest.parent.mkdir(parents=True)
    layout.manifest.write_text(
        f'{{"version": 1, "cases": [{cases}], "launcher": {{}}}}', encoding="utf-8"
    )
    layout.result(0).parent.mkdir(parents=True)
    layout.result(0).write_text(_passing_record(0), encoding="utf-8")
    layout.result(2).write_text(_failing_record(2), encoding="utf-8")
    assert collect_mod.collect(layout, quiet_hint=True) == 1
    expected = tmp_path / "expected.json"
    write_json(
        expected,
        {
            "cases": 4,
            "passed": 1,
            "errored": [labels[2]],
            "missing": [labels[1], labels[3]],
        },
    )
    written = layout.root / "collected.json"
    assert written.read_bytes() == expected.read_bytes()
    assert oct(written.stat().st_mode & 0o777) == oct(expected.stat().st_mode & 0o777)


# -- A manifest nested deep ---------------------------------------------------------


def _nested(depth: int) -> str:
    return "[" * depth + "]" * depth


@pytest.mark.parametrize(
    "where", ["extra", "env", "case-config", "case-extra", "case-label", "cases"]
)
@pytest.mark.parametrize("depth", [500, 5000, 100_000])
def test_a_manifest_nested_deep_is_read_as_json_loads_reads_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    quiet_slurm: None,
    where: str,
    depth: int,
) -> None:
    """A manifest nested deeper than ``json``'s encoder written in Python
    lays out is read wherever ``json.loads()`` reads it, and raises what it
    raises where it does not: a RecursionError, which reading it as a
    ``dict`` let through too."""
    nest = _nested(depth)
    if where == "extra":
        text = _manifest(_V1, z=nest)
    elif where == "env":
        text = _manifest(_V1, env=nest)
    elif where == "case-config":
        text = _manifest(_V1, cases=_cases(config=nest))
    elif where == "case-extra":
        text = _manifest(_V1, cases=_cases(z=nest))
    elif where == "case-label":
        text = _manifest(_V1, cases=_cases(label=nest))
    else:
        text = _manifest(_V1, cases="[" + _object(_case(0)) + ", " + nest + "]")
    pristine = _batch(tmp_path, text.encode())
    names = (
        "status",
        "collect-quiet",
        "missing-cases",
        "unfinished-failed",
        "resubmit-target",
        "resubmit-dry",
    )
    actual = _run_readers(pristine, tmp_path, capsys, caplog, names)
    with _old_reading(monkeypatch):
        expected = _run_readers(pristine, tmp_path, capsys, caplog, names)
    assert actual == expected
