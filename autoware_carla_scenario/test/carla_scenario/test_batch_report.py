"""``batch_results.json`` as batch_report lays it out (#560).

The module writes the file as text, a member at a time, so that the library
check compiles it.  These tests hold that layout to what
``json.dumps(entries, indent=2, ensure_ascii=False)`` gives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.batch_report import write_batch_result_json


class _Result:
    """A result as batch_report reads one; *recording* ``False``: no attribute."""

    def __init__(self, data: dict[str, Any], recording: Any = None) -> None:
        self._data = data
        self.passed = bool(data.get("passed", False))
        self.message = ""
        self.elapsed_seconds = 0.0
        self.condition_statuses: list[Any] = []
        if recording is not False:
            self.rerun_recording = recording

    def to_dict(self) -> dict[str, Any]:
        return self._data


_DETAILED = {
    "passed": False,
    "message": "line one\nline two",
    "details": {"gap": [1.5, {"to": "日本語 ü"}], "none": {}, "empty": []},
}

_EXPECTED = """[
  {
    "scenario": "cut_in",
    "overrides": [
      "speed=10",
      "name=ü"
    ],
    "passed": false,
    "message": "line one\\nline two",
    "details": {
      "gap": [
        1.5,
        {
          "to": "日本語 ü"
        }
      ],
      "none": {},
      "empty": []
    },
    "rerun_error": "conversion failed: ü"
  },
  {
    "scenario": "lane_change",
    "overrides": [],
    "error": true
  }
]"""


def test_the_file_is_laid_out_as_json_dumps_with_indent_2(tmp_path: Path) -> None:
    path = write_batch_result_json(
        ["cut_in", "lane_change"],
        [_Result(_DETAILED, recording="a.rrd"), _Result({}, recording=False)],
        tmp_path,
        overrides=[["speed=10", "name=ü"], []],
        errored=[1],
        rerun_errors={0: "conversion failed: ü"},
    )

    text = path.read_text(encoding="utf-8")
    assert text == _EXPECTED
    assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=False)


def test_an_entry_without_overrides_or_errors_has_the_result_members_only(
    tmp_path: Path,
) -> None:
    path = write_batch_result_json(
        ["stop"], [_Result({"passed": True, "elapsed_seconds": 2.0})], tmp_path
    )

    assert path.read_text(encoding="utf-8") == (
        '[\n  {\n    "scenario": "stop",\n    "passed": true,\n'
        '    "elapsed_seconds": 2.0\n  }\n]'
    )


def test_an_empty_batch_writes_an_empty_list(tmp_path: Path) -> None:
    path = write_batch_result_json([], [], tmp_path, overrides=[])

    assert path.name == "batch_results.json"
    assert path.read_text(encoding="utf-8") == "[]"


def _merged(
    names: list[str],
    results: list[_Result],
    overrides: list[list[str]] | None,
    errored: list[int],
    rerun_errors: dict[int, str],
) -> str:
    """The file as the dict merge before #560 wrote it."""
    return json.dumps(
        [
            {
                "scenario": name,
                **({"overrides": overrides[i]} if overrides is not None else {}),
                **result.to_dict(),
                **({"error": True} if i in errored else {}),
                **({"rerun_error": rerun_errors[i]} if i in rerun_errors else {}),
            }
            for i, (name, result) in enumerate(zip(names, results))
        ],
        indent=2,
        ensure_ascii=False,
    )


_VALUES: dict[str, Any] = {
    "passed": True,
    "none": None,
    "int": -3,
    "big": 10**20,
    "float": 0.1 + 0.2,
    "exp": 1e-7,
    "nan": float("nan"),
    "inf": float("inf"),
    "-inf": float("-inf"),
    "escapes": 'quote " backslash \\ tab \t ctrl \x01 sep   emoji \U0001f697',
    'key "with" \\ escapes\n': "ü",
    "nested": [[], {}, [1, [2, {"a": [None, False]}]], {"b": {"c": {}}}],
}

# Keys of the result's dict that the entry also sets: the merge wrote each
# once, at its first place, with its last value.
_COLLIDING: dict[str, Any] = {
    "error": "from the result",
    "passed": False,
    "scenario": "from the result",
    "rerun_error": ["from", "the", "result"],
    "overrides": {"from": "the result"},
}


@pytest.mark.parametrize(
    ("data", "overrides", "errored", "rerun_errors"),
    [
        (_VALUES, None, [], {}),
        (_VALUES, [["x=1"]], [0], {0: "why"}),
        (_COLLIDING, None, [], {}),
        (_COLLIDING, [["x=1"]], [], {}),
        (_COLLIDING, None, [0], {0: "why"}),
        (_COLLIDING, [["x=1"]], [0], {0: "why"}),
    ],
)
def test_the_file_is_what_the_dict_merge_wrote(
    tmp_path: Path,
    data: dict[str, Any],
    overrides: list[list[str]] | None,
    errored: list[int],
    rerun_errors: dict[int, str],
) -> None:
    results = [_Result(data)]
    path = write_batch_result_json(
        ["case"],
        results,
        tmp_path,
        overrides=overrides,
        errored=errored,
        rerun_errors=rerun_errors,
    )

    text = path.read_text(encoding="utf-8")
    assert text == _merged(["case"], results, overrides, errored, rerun_errors)
    assert len(json.loads(text)[0]) == text.count('\n    "')
