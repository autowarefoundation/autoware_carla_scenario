"""``batch_results.json`` as batch_report lays it out (#560).

The module writes the file as text, a member at a time, so that the library
check compiles it.  These tests hold that layout to what
``json.dumps(entries, indent=2, ensure_ascii=False)`` gives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
