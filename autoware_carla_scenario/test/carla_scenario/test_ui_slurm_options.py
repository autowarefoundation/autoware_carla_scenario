"""ui.slurm_options, split out of ui.slurm_launch so the library check
compiles it (#742): slurm_launch still exports every name it did, and the
options it reads off a request, the Hydra overrides they become and the
``scenario`` command line built with them are the ones it built before.

``data/ui_slurm_options/golden.json`` holds what slurm_launch built before
the split, for the cases below; regenerate it only for a change meant to
alter them::

    uv run python autoware_carla_scenario/test/carla_scenario/test_ui_slurm_options.py
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.ui import slurm_launch
from autoware_carla_scenario.ui.slurm_launch import SlurmOptions

_GOLDEN = Path(__file__).resolve().parent / "data" / "ui_slurm_options" / "golden.json"

#: What slurm_launch exported before the split (its ``__all__``) and the
#: module-level names it defined.
_LAUNCH_NAMES = {
    "SlurmOptions",
    "Submission",
    "availability",
    "batch_status",
    "build_command",
    "partitions",
    "run_cli",
    "scenario_command",
    "start_submit",
    "submission",
    "_AT_END",
    "_BEFORE_QUOTE",
    "_ENV_COMMAND",
    "_ENV_NAME",
    "_NODE_PATH",
    "_NOT_A_STRING",
    "_OUTPUTS_IN",
    "_SIMPLE",
    "_TIME",
    "_case_states",
    "_new_manifests",
    "_quote",
    "_quote_always",
    "_submit",
}

_TEXT_VALUES = [
    "",
    "main",
    "gpu-a100",
    "shard:1",
    "12G",
    "01:30:00",
    "a b",
    "a'b",
    'a"b',
    "a\\b",
    "C:\\dir\\",
    "\\\\\\",
    "it's \\\\'quoted\\' \\",
    "null",
    "NULL",
    "true",
    "False",
    "inf",
    "-inf",
    "nan",
    "+NaN",
    "1",
    "-1",
    "01",
    "1_000",
    "1__0",
    "1.",
    ".5",
    "1e5",
    "1.5E-3",
    "1_0.0_1e1_0",
    "0x10",
    "infinity",
    "nullx",
    "~/work",
    "ünïcödé 東京",
    "k=v",
    "${HOME}",
    "$HOME",
    "a,b",
    "[a]",
    "{a: 1}",
]

#: name -> SlurmOptions' keyword arguments (values of any type: what its
#: fields declare, and what a caller might hand it besides).
_OPTIONS: dict[str, dict[str, Any]] = {
    "defaults": {},
    "every-key": {
        "partition": "main",
        "array_parallelism": 2,
        "cases_per_task": 3,
        "time": "1-00:00:00",
        "mem": "12G",
        "gres": "gpu:1",
        "start_stagger_seconds": 30.0,
        "remote_cwd": "workspace/OnePlanner",
        "python": ".venv/bin/python",
        "setup": ["module load cuda", 'echo "it\'s $HOME"', "X=a\\b\\\\"],
        "env": {"B": "2", "A": "a b", "C": "null", "D": "C:\\dir\\"},
    },
    "numbers": {
        "array_parallelism": 0,
        "cases_per_task": 10**20,
        "start_stagger_seconds": 0.1 + 0.2,
    },
    "float-specials": {"start_stagger_seconds": math.inf},
    "float-nan": {"start_stagger_seconds": math.nan},
    "float-tiny": {"start_stagger_seconds": 1e-7},
    "float-huge": {"start_stagger_seconds": 1e22},
    "int-stagger": {"start_stagger_seconds": 5},
    "bools": {"array_parallelism": True, "start_stagger_seconds": False},
    "number-in-text-keys": {"partition": 5, "time": 1.5, "python": False},
    "text-in-number-keys": {
        "array_parallelism": "4",
        "cases_per_task": "two words",
        "start_stagger_seconds": "null",
    },
    "list-in-text-key": {"partition": ["a"]},
    "bytes-in-text-key": {"mem": b"12G"},
    "list-in-number-key": {"array_parallelism": [1]},
    "empty-text": {"partition": "", "gres": ""},
    "setup-only": {"setup": ["a"]},
    "setup-empty-line": {"setup": [""]},
    "env-only": {"env": {"Z": "", "Y": "'", "X": "\\"}},
}
for _n, _value in enumerate(_TEXT_VALUES):
    _OPTIONS[f"text-{_n}"] = {
        "partition": _value,
        "time": _value,
        "remote_cwd": _value,
        "setup": [_value, _value + "x"],
        "env": {"V": _value},
    }

#: name -> the run panel's fields, as the viewer's request body holds them.
_REQUESTS: dict[str, Any] = {
    "none": None,
    "empty": {},
    "falsy-fields": {"partition": 0, "time": False, "env": "", "setup": ""},
    "panel": {
        "partition": " main ",
        "array_parallelism": "2",
        "cases_per_task": "",
        "time": "01:30:00",
        "mem": "12G",
        "gres": "shard:1",
        "start_stagger_seconds": "30",
        "env": "ONEPLANNER_MAP_DIR=/maps/town 10\nFOO=a'b",
    },
    "node-paths": {
        "remote_cwd": "workspace/OnePlanner",
        "python": ".venv/bin/python",
        "setup": 'source ~/workspace/opt/env.sh\n\necho "it\'s $HOME"\n',
    },
    "tilde": {"remote_cwd": "~", "python": "~/x"},
    "tilde-user": {"remote_cwd": "~bob/work"},
    "tilde-twice": {"python": "~~"},
    "empty-node-path-after-strip": {"python": "  "},
    "node-path-space": {"python": "a b"},
    "numbers-as-json": {
        "array_parallelism": 3,
        "cases_per_task": 1,
        "start_stagger_seconds": 0,
    },
    "float-text": {"start_stagger_seconds": "1.5e1"},
    "float-inf": {"start_stagger_seconds": "inf"},
    "float-nan": {"start_stagger_seconds": "nan"},
    "float-negative": {"start_stagger_seconds": "-1"},
    "float-negative-zero": {"start_stagger_seconds": "-0"},
    "int-underscore": {"array_parallelism": "1_0"},
    "int-float": {"array_parallelism": "1.5"},
    "int-zero": {"cases_per_task": "0"},
    "int-negative": {"array_parallelism": "-3"},
    "int-word": {"cases_per_task": "two"},
    "int-bool": {"array_parallelism": True},
    "time-forms": {"time": "1-2:03:04"},
    "time-minutes": {"time": "90"},
    "time-bad": {"time": "tomorrow"},
    "time-seconds-only": {"time": "1:2"},
    "time-bad-tail": {"time": "01:00:00x"},
    "partition-bad": {"partition": "main; rm -rf"},
    "mem-bad": {"mem": "12 G"},
    "gres-bad": {"gres": "gpu:1,shard"},
    "gres-unicode": {"gres": "ä"},
    "env-mapping": {"env": {"A": "1", " B ": "x y", "C": 5, "D": None}},
    "env-text": {"env": "A=1\n  B = 2 \n\nC==3"},
    "env-text-no-value": {"env": "A=1\nNOVALUE"},
    "env-bad-name": {"env": "1BAD=x"},
    "env-bad-name-mapping": {"env": {"a-b": "x"}},
    "env-empty-name": {"env": "=x"},
    "env-interpolation": {"env": "X=${HOME}"},
    "env-interpolation-mapping": {"env": {"X": "a${oc.env:HOME}"}},
    "env-interpolation-list": {"env": {"X": ["${x}"]}},
    "env-dollar": {"env": {"X": "$HOME"}},
    "env-pairs": {"env": [["A", "1"], ["B", "2"]]},
    "setup-list": {"setup": [" a ", "", 3, "b"]},
    "setup-interpolation": {"setup": "echo ${HOME}"},
    "setup-backslash-end": {"setup": "module load a \\"},
    "setup-three-backslashes": {"setup": "echo \\\\\\"},
    "setup-two-backslashes": {"setup": "echo a\\\\"},
    "setup-inner-backslash": {"setup": "echo a\\b"},
    "setup-trailing-space-backslash": {"setup": "echo \\ "},
    "order-env-before-time": {"env": "1BAD=x", "time": "tomorrow"},
    "order-time-before-partition": {"time": "x", "partition": "a b"},
    "order-partition-before-mem": {"partition": "a b", "mem": "c d"},
    "order-text-before-paths": {"gres": "a b", "remote_cwd": "a b"},
    "order-paths-before-setup": {"python": "a b", "setup": "${X}"},
    "order-setup-before-numbers": {"setup": "${X}", "array_parallelism": "x"},
    "order-numbers": {"start_stagger_seconds": "x", "cases_per_task": "y"},
    "order-number-ranges": {"cases_per_task": "0", "array_parallelism": "0"},
    "everything": {
        "partition": "main",
        "array_parallelism": "4",
        "cases_per_task": "2",
        "time": "2-00",
        "mem": "64G",
        "gres": "gpu:a100:1",
        "start_stagger_seconds": "2.5",
        "remote_cwd": "~/ws",
        "python": "/opt/venv/bin/python",
        "setup": "module load cuda\nexport A='b c'\nY=\"d\" \\\\",
        "env": "Z=last\nA=first\nM=a\\b\\",
    },
}

#: name -> (VIEWER_SCENARIO_COMMAND or None, scenario, extra overrides,
#: sweeper, the options' case in _OPTIONS).
_COMMANDS: dict[str, tuple[str | None, str, list[str], str, str]] = {
    "default-command": (None, "cut_in/*", ["map=town10hd_opt"], "lc", "every-key"),
    "no-sweeper": (None, "x", [], "", "defaults"),
    "empty-command": ("   ", "x", [], "", "defaults"),
    "venv-command": (".venv/bin/scenario", "x", ["a=1", "b=2"], "", "numbers"),
    "quoted-command": (
        "uv run --project '/opt/my ws' scenario",
        "cut_in/*",
        [],
        "lanelet_constraint",
        "every-key",
    ),
    "text-values": (None, "x", [], "s", "text-6"),
}


def _outcome(build: Any) -> Any:
    try:
        return build()
    except (ValueError, KeyError) as exc:
        return f"{type(exc).__name__}: {exc}"
    except (TypeError, AttributeError) as exc:
        # Raised by CPython itself, whose wording changes between versions
        # (3.12 adds "got 'int'"): the type is what has to stay the same.
        return type(exc).__name__


def _read(request: Any) -> dict[str, Any]:
    options = SlurmOptions.from_request(request)
    return {"fields": dataclasses.asdict(options), "overrides": options.overrides()}


def _render(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Everything the golden file holds, built by ui.slurm_launch."""
    out: dict[str, Any] = {}
    out["overrides"] = {
        name: _outcome(lambda k=kwargs: SlurmOptions(**k).overrides())
        for name, kwargs in _OPTIONS.items()
    }
    out["from_request"] = {
        name: _outcome(lambda r=request: _read(r))
        for name, request in _REQUESTS.items()
    }
    commands = {}
    for name, (configured, scenario, extra, sweeper, case) in _COMMANDS.items():
        if configured is None:
            monkeypatch.delenv("VIEWER_SCENARIO_COMMAND", raising=False)
        else:
            monkeypatch.setenv("VIEWER_SCENARIO_COMMAND", configured)
        commands[name] = slurm_launch.build_command(
            scenario, list(extra), sweeper, SlurmOptions(**_OPTIONS[case])
        )
    out["build_command"] = commands
    out["quote"] = {v: slurm_launch._quote(v) for v in _TEXT_VALUES}
    out["quote_always"] = {v: slurm_launch._quote_always(v) for v in _TEXT_VALUES}
    # JSON's own types (nan and inf are written as JSON's NaN and Infinity).
    return json.loads(json.dumps(out))


def _golden() -> dict[str, Any]:
    if not _GOLDEN.exists():  # being regenerated
        return {}
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def _same(a: Any, b: Any) -> bool:
    """Equal, with nan equal to itself (a field read back as nan)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return bool(a == b) and type(a) is type(b)


@pytest.mark.parametrize("key", sorted(_golden()))
def test_slurm_launch_builds_what_it_built_before_the_split(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered, golden = _render(monkeypatch)[key], _golden()[key]
    assert rendered.keys() == golden.keys()
    wrong = [name for name in golden if not _same(rendered[name], golden[name])]
    assert not wrong, {name: (rendered[name], golden[name]) for name in wrong}


def test_the_golden_file_covers_every_case() -> None:
    golden = _golden()
    assert set(golden) == {
        "overrides",
        "from_request",
        "build_command",
        "quote",
        "quote_always",
    }
    assert set(golden["overrides"]) == set(_OPTIONS)
    assert set(golden["from_request"]) == set(_REQUESTS)
    assert set(golden["build_command"]) == set(_COMMANDS)


def test_slurm_launch_still_exports_every_name() -> None:
    for name in _LAUNCH_NAMES:
        assert hasattr(slurm_launch, name), name
    assert set(slurm_launch.__all__) == {
        name for name in _LAUNCH_NAMES if not name.startswith("_")
    }


# ── The quoting, against the regular expressions it was written with ───────


def _quote_before(value: str) -> str:
    """slurm_launch._quote as it was before the split (#743)."""
    if slurm_launch._SIMPLE.fullmatch(
        value
    ) and not slurm_launch._NOT_A_STRING.fullmatch(value):
        return value
    return _quote_always_before(value)


def _quote_always_before(value: str) -> str:
    """slurm_launch._quote_always as it was before the split: the trailing
    run of backslashes doubled by ``(\\\\+)\\Z``, which Codon's re (RE2) has
    no syntax for."""
    body = slurm_launch._BEFORE_QUOTE.sub(lambda m: m[1] * 2 + "\\'", value)
    return "'" + slurm_launch._AT_END.sub(lambda m: m[1] * 2, body) + "'"


#: Every character class the quoting tells apart: a backslash, both quotes,
#: digits, the letters of Hydra's keywords and exponents, signs, a point,
#: an underscore, a space and a newline (before which ``$`` also matches).
_ALPHABET = ["\\", "'", '"', "1", "0", "e", "n", "a", "-", ".", "_", " ", "\n"]


def test_the_quoting_is_the_regular_expressions_for_every_short_value() -> None:
    small = ["\\", "'", "a", "1", "\n"]
    for length in range(8):
        for chars in itertools.product(small, repeat=length):
            value = "".join(chars)
            assert slurm_launch._quote_always(value) == _quote_always_before(value)
            assert slurm_launch._quote(value) == _quote_before(value)


def test_the_quoting_is_the_regular_expressions_for_random_values() -> None:
    rng = random.Random(742)
    words = [*_ALPHABET, "\\\\", "\\'", "inf", "nan", "null", "true", "1_0", "é", "東"]
    for _ in range(20000):
        value = "".join(rng.choice(words) for _ in range(rng.randrange(12)))
        assert slurm_launch._quote_always(value) == _quote_always_before(value)
        assert slurm_launch._quote(value) == _quote_before(value)


def test_the_checked_patterns_are_ones_codon_s_re_compiles() -> None:
    # Codon's re is RE2, which has no \Z (and no lookaround, no
    # backreference): the checked module's patterns use none.
    from autoware_carla_scenario.ui import slurm_options

    patterns = [
        value.pattern
        for value in vars(slurm_options).values()
        if isinstance(value, re.Pattern)
    ]
    assert patterns
    for pattern in patterns:
        assert not re.search(r"\\Z|\(\?[=!<]|\\[1-9]", pattern), pattern


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, float("1e400")])
def test_check_at_least_refuses_a_number_that_is_not_finite(value: float) -> None:
    from autoware_carla_scenario.ui.slurm_options import check_at_least

    # The run panel wrote launcher.start_stagger_seconds=inf (or nan) (#784).
    with pytest.raises(ValueError, match="start_stagger_seconds must be a finite"):
        check_at_least("start_stagger_seconds", value, 0)


@pytest.mark.parametrize("text", ["inf", "-inf", "nan", "1e400", "Infinity"])
def test_run_panel_refuses_a_stagger_that_is_not_finite(text: str) -> None:
    with pytest.raises(ValueError, match="must be a finite number"):
        SlurmOptions.from_request({"start_stagger_seconds": text})


def test_check_at_least_takes_a_finite_number_from_the_minimum() -> None:
    from autoware_carla_scenario.ui.slurm_options import check_at_least

    check_at_least("start_stagger_seconds", 0.0, 0)
    check_at_least("start_stagger_seconds", 1e300, 0)
    check_at_least("cases_per_task", 1, 1)
    with pytest.raises(ValueError, match="at least 1"):
        check_at_least("cases_per_task", 0, 1)


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(
            json.dumps(_render(patch), indent=1, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
