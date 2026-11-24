"""``detect-no-3d-model`` writes the detected IDs into the map YAML without
corrupting it (issue #749).

The old line-based edit appended the key under whatever section came last,
glued it onto a last line without a newline, and left the items of an
indentless list behind.  The edit is still textual (comments and layout
survive), but where the key goes is read from the parsed document and the
result is parsed again before anything is written.
"""

from __future__ import annotations

import errno
import logging
import os
import random
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from autoware_carla_scenario.tools import detect_no_3d_model_lanelets as tool
from autoware_carla_scenario.tools import yaml_text
from autoware_carla_scenario.tools.detect_no_3d_model_lanelets import (
    NO_3D_MODEL_KEY,
    YamlUpdateError,
    _resolve_map_config,
    _update_yaml,
    _write_detected_ids,
    update_no_3d_model_lanelet_ids_text,
)

_IDS = [7, 8]


def _update(text: str, ids: list[int] = _IDS) -> tuple[str, object]:
    result = update_no_3d_model_lanelet_ids_text(text, ids)
    return result, yaml.safe_load(result)


# ---------------------------------------------------------------------------
# The four reproduced corruptions
# ---------------------------------------------------------------------------


def test_minimal_top_level_yaml_gets_a_top_level_key(tmp_path: Path) -> None:
    # The minimal YAML docs/usage.md documents: no map: section.
    (tmp_path / "a.xodr").write_text("")
    (tmp_path / "a.osm").write_text("")
    path = tmp_path / "map.yaml"
    path.write_text(
        f"name: M\nxodr_path: {tmp_path / 'a.xodr'}\n"
        f"lanelet2_path: {tmp_path / 'a.osm'}\n"
    )
    _update_yaml(path, _IDS)
    data = yaml.safe_load(path.read_text())
    assert data["no_3d_model_lanelet_ids"] == _IDS
    assert data["name"] == "M"
    # The tool still reads the file it wrote.
    assert _resolve_map_config(path).name == "M"


@pytest.mark.parametrize(
    "text",
    [
        "map:\n  name: M",
        "map:\n  name: M\n  no_3d_model_lanelet_ids: []",
        "name: M",
    ],
)
def test_last_line_without_newline(text: str) -> None:
    result, data = _update(text)
    assert isinstance(data, dict)
    section = data.get("map", data)
    assert section["no_3d_model_lanelet_ids"] == _IDS
    assert section["name"] == "M"
    assert "M" + "no_3d" not in result


def test_indentless_sequence_is_replaced_whole() -> None:
    # PyYAML's own dump style: the items at the key's indentation.
    text = "map:\n  name: M\n  no_3d_model_lanelet_ids:\n  - 3\n  - 4\n  other: 1\n"
    _, data = _update(text)
    assert data == {"map": {"name": "M", "no_3d_model_lanelet_ids": _IDS, "other": 1}}


def test_map_section_that_is_not_last() -> None:
    text = "map:\n  name: M\n  xodr_path: a.xodr\nother:\n  x: 1\n"
    result, data = _update(text)
    assert data == {
        "map": {"name": "M", "xodr_path": "a.xodr", "no_3d_model_lanelet_ids": _IDS},
        "other": {"x": 1},
    }
    assert result.startswith("map:\n  name: M\n  xodr_path: a.xodr\n")


def test_missing_key_goes_after_the_section_not_its_trailing_comments() -> None:
    text = "map:\n  name: M  # name\n# about other\n\nother: 1\n"
    result, _ = _update(text)
    assert result == (
        "map:\n  name: M  # name\n  no_3d_model_lanelet_ids:\n    - 7\n    - 8\n"
        "# about other\n\nother: 1\n"
    )


# ---------------------------------------------------------------------------
# Existing values in every style
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "\n    - 3\n    - 4",  # block list, indented
        "\n  - 3\n  - 4",  # block list, indentless
        " [3, 4]",  # flow list
        " [3,\n    4]",  # flow list over two lines
        "",  # empty value
        " null",
        " ~",
        " []",
        " 5",  # scalar
        " '5'",
        "\n    - [1, 2]\n    - {a: 1}",  # nested values
    ],
)
@pytest.mark.parametrize("ids", [_IDS, []])
def test_existing_value_in_any_style(value: str, ids: list[int]) -> None:
    text = (
        "# @package _global_\nmap:\n  name: M\n"
        f"  no_3d_model_lanelet_ids:{value}\n  lanelet2_path: a.osm\n"
        "other:\n  x: 1\n"
    )
    result, data = _update(text, ids)
    assert data == {
        "map": {"name": "M", "no_3d_model_lanelet_ids": ids, "lanelet2_path": "a.osm"},
        "other": {"x": 1},
    }
    assert result.startswith("# @package _global_\n")


def test_block_list_written_in_block_style() -> None:
    text = "map:\n  name: M\n  no_3d_model_lanelet_ids: null\n"
    result, _ = _update(text)
    assert result == "map:\n  name: M\n  no_3d_model_lanelet_ids:\n    - 7\n    - 8\n"


def test_flow_list_stays_a_flow_list() -> None:
    text = "map:\n  no_3d_model_lanelet_ids: [1, 2]  # detected\n  name: M\n"
    result, _ = _update(text)
    assert result == "map:\n  no_3d_model_lanelet_ids: [7, 8]  # detected\n  name: M\n"


def test_empty_id_list() -> None:
    text = "map:\n  name: M\n  no_3d_model_lanelet_ids:\n    - 1\n    - 2\n"
    result, data = _update(text, [])
    assert result == "map:\n  name: M\n  no_3d_model_lanelet_ids: []\n"
    result, data = _update("map:\n  name: M\n", [])
    assert data == {"map": {"name": "M", "no_3d_model_lanelet_ids": []}}


# ---------------------------------------------------------------------------
# Layout survives
# ---------------------------------------------------------------------------


def test_comments_and_interpolations_preserved() -> None:
    text = (
        "# @package _global_\n"
        "# Header comment.\n"
        "map:\n"
        "  name: M  # trailing\n"
        "  xodr_path: ${oc.env:X_PATH,data/x.xodr}\n"
        "  # Lanelet IDs to exclude.\n"
        "  no_3d_model_lanelet_ids:\n"
        "    - 1\n"
        "    - 2\n"
        "  # Between keys.\n"
        "  lanelet2_path: ${oc.env:L_PATH,data/x.osm}\n"
        "\n"
        "# Footer.\n"
    )
    result, _ = _update(text)
    assert result == text.replace("    - 1\n    - 2\n", "    - 7\n    - 8\n")


def test_crlf_line_endings_preserved(tmp_path: Path) -> None:
    path = tmp_path / "map.yaml"
    original = (
        b"# c\r\nmap:\r\n  name: M\r\n  no_3d_model_lanelet_ids: []\r\nother: 1\r\n"
    )
    path.write_bytes(original)
    _update_yaml(path, _IDS)
    written = path.read_bytes()
    assert b"\n" not in written.replace(b"\r\n", b"")
    assert written == original.replace(
        b"ids: []\r\n", b"ids:\r\n    - 7\r\n    - 8\r\n"
    )

    path.write_bytes(b"map:\r\n  name: M\r\nother: 1")
    _update_yaml(path, _IDS)
    written = path.read_bytes()
    assert b"\n" not in written.replace(b"\r\n", b"")
    assert yaml.safe_load(written)["map"]["no_3d_model_lanelet_ids"] == _IDS


def test_repository_map_config_round_trips() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src/autoware_carla_scenario/examples/conf/map/nishishinjuku.yaml"
    )
    text = path.read_text(encoding="utf-8")
    current = yaml.safe_load(text)["map"]["no_3d_model_lanelet_ids"]
    assert update_no_3d_model_lanelet_ids_text(text, current) == text


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("map: {name: M, xodr_path: a.xodr}\n", "flow mapping"),
        ("{name: M}\n", "flow mapping"),
        ("map: null\n", "not a mapping"),
        ("map:\n  - 1\n", "not a mapping"),
        ("- 1\n", "mapping at the top level"),
        ("", "mapping at the top level"),
        ("map: [\n", "not valid YAML"),
        (
            "map:\n  no_3d_model_lanelet_ids: []\n  no_3d_model_lanelet_ids: []\n",
            "duplicate key",
        ),
        ("a: &x [1]\nmap:\n  no_3d_model_lanelet_ids: *x\n", "alias"),
        # Rewriting an anchored value would break the alias that uses it.
        ("map:\n  no_3d_model_lanelet_ids: &x [1]\nb: *x\n", "invalid YAML"),
    ],
)
def test_refuses_what_it_cannot_edit_safely(text: str, match: str) -> None:
    with pytest.raises(YamlUpdateError, match=match):
        update_no_3d_model_lanelet_ids_text(text, _IDS)


def test_refused_write_leaves_file_byte_identical(tmp_path: Path) -> None:
    path = tmp_path / "map.yaml"
    original = b"# keep me\r\nmap: {name: M, xodr_path: a.xodr}  \n"
    path.write_bytes(original)
    with pytest.raises(YamlUpdateError, match="map.yaml"):
        _update_yaml(path, _IDS)
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["map.yaml"]


def test_write_is_atomic_and_keeps_mode(tmp_path: Path) -> None:
    path = tmp_path / "map.yaml"
    path.write_text("map:\n  name: M\n")
    path.chmod(0o640)
    _update_yaml(path, _IDS)
    assert yaml.safe_load(path.read_text())["map"]["no_3d_model_lanelet_ids"] == _IDS
    assert path.stat().st_mode & 0o777 == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["map.yaml"]


def test_symlinked_yaml_is_written_through(tmp_path: Path) -> None:
    # The old in-place write followed the link; os.replace on the link itself
    # would turn it into a regular file and leave the target stale.
    target_dir = tmp_path / "real"
    target_dir.mkdir()
    target = target_dir / "map.yaml"
    target.write_text("map:\n  name: M\n")
    link = tmp_path / "map.yaml"
    link.symlink_to(target)
    _update_yaml(link, _IDS)
    assert link.is_symlink()
    assert yaml.safe_load(target.read_text())["map"]["no_3d_model_lanelet_ids"] == _IDS
    assert sorted(p.name for p in target_dir.iterdir()) == ["map.yaml"]


# ---------------------------------------------------------------------------
# Block scalars end past their trailing line breaks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["|", ">", "|-"])
def test_block_scalar_last_in_section(style: str) -> None:
    text = f"map:\n  name: {style}\n    foo\n\n  # c\nother: 1\n"
    result, data = _update(text)
    assert data == {
        "map": {
            "name": yaml.safe_load(text)["map"]["name"],
            "no_3d_model_lanelet_ids": _IDS,
        },
        "other": 1,
    }
    assert result.startswith(
        f"map:\n  name: {style}\n    foo\n  no_3d_model_lanelet_ids:\n"
    )


@pytest.mark.parametrize(
    "text",
    [
        # Adding a line after it would give the value the line break it lacks.
        "map:\n  name: >\n    foo",
        # Its trailing blank lines are part of the value.
        "map:\n  name: |+\n    foo\n\nother: 1\n",
    ],
)
def test_block_scalar_whose_value_an_insert_would_change_is_refused(
    text: str,
) -> None:
    with pytest.raises(YamlUpdateError, match="more than that value"):
        update_no_3d_model_lanelet_ids_text(text, _IDS)


def test_existing_list_of_block_scalars_is_replaced() -> None:
    text = "map:\n  no_3d_model_lanelet_ids:\n    - |\n      foo\n\n  name: M\n"
    result, data = _update(text)
    assert data == {"map": {"no_3d_model_lanelet_ids": _IDS, "name": "M"}}
    assert result == "map:\n  no_3d_model_lanelet_ids:\n    - 7\n    - 8\n\n  name: M\n"


def test_unwritable_directory_falls_back_to_writing_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A writable file in a directory that allows no new files: the old code
    # could update it, so the temporary file must not be a requirement.
    def refuse(*args: object, **kwargs: object) -> tuple[int, str]:
        raise PermissionError("read-only directory")

    monkeypatch.setattr(tempfile, "mkstemp", refuse)
    path = tmp_path / "map.yaml"
    path.write_text("map:\n  name: M\n")
    _update_yaml(path, _IDS)
    assert yaml.safe_load(path.read_text())["map"]["no_3d_model_lanelet_ids"] == _IDS


# ---------------------------------------------------------------------------
# A file that cannot be read or written ends in the guidance, not a
# traceback (issue #1741)
# ---------------------------------------------------------------------------


def _assert_guidance(
    path: Path,
    ids: list[int],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    ending: str = "; the file was not changed",
) -> list[str]:
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit) as excinfo:
        _write_detected_ids(path, ids)
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err + captured.out
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 2, errors
    assert errors[0].count(str(path)) == 1, errors[0]
    assert errors[0].endswith(ending)
    assert errors[1] == f"Set {NO_3D_MODEL_KEY} by hand to: {ids}"
    return errors


@pytest.mark.parametrize("ids", [_IDS, []])
def test_read_only_filesystem_prints_the_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    ids: list[int],
) -> None:
    # EROFS is an OSError but not a PermissionError, so it skips the
    # write-in-place fallback.
    def refuse(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(tempfile, "mkstemp", refuse)
    path = tmp_path / "map.yaml"
    path.write_text("map:\n  name: M\n")
    errors = _assert_guidance(path, ids, caplog, capsys)
    assert "Read-only file system" in errors[0]
    assert path.read_text() == "map:\n  name: M\n"


def test_unwritable_directory_and_file_prints_the_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(*args: object, **kwargs: object) -> tuple[int, str]:
        raise PermissionError(errno.EACCES, "Permission denied")

    def refuse_open(path: str | os.PathLike[str], flags: int, *args: object) -> int:
        raise PermissionError(errno.EACCES, "Permission denied", os.fspath(path))

    monkeypatch.setattr(tempfile, "mkstemp", refuse)
    monkeypatch.setattr(os, "open", refuse_open)
    path = tmp_path / "map.yaml"
    path.write_text("map:\n  name: M\n")
    errors = _assert_guidance(path, _IDS, caplog, capsys)
    assert "Permission denied" in errors[0]
    assert path.read_text() == "map:\n  name: M\n"


def test_unreadable_yaml_prints_the_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse_read(self: Path) -> bytes:
        raise PermissionError(errno.EACCES, "Permission denied", str(self))

    path = tmp_path / "map.yaml"
    path.write_text("map:\n  name: M\n")
    monkeypatch.setattr(Path, "read_bytes", refuse_read)
    errors = _assert_guidance(path, _IDS, caplog, capsys)
    assert "Permission denied" in errors[0]


def test_refused_edit_still_prints_the_same_guidance(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "map.yaml"
    path.write_text("- a\n- b\n")
    _assert_guidance(path, [], caplog, capsys)
    assert path.read_text() == "- a\n- b\n"


def _refuse_temporary_file(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> tuple[int, str]:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(tempfile, "mkstemp", refuse)


def test_write_in_place_failing_after_the_open_says_it_may_be_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The fallback has emptied the file by the time the write fails, so it
    # must not claim the file was not changed.
    def full_disk(fd: int, data: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    _refuse_temporary_file(monkeypatch)
    path = tmp_path / "map.yaml"
    path.write_text("map:\n  name: M\n")
    monkeypatch.setattr(os, "write", full_disk)
    errors = _assert_guidance(
        path, _IDS, caplog, capsys, "; the file may have been left incomplete"
    )
    assert "No space left on device" in errors[0]


def test_write_in_place_keeps_mode_and_handles_short_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = os.write

    def one_byte(fd: int, data: bytes) -> int:
        return real_write(fd, bytes(data[:1]))

    _refuse_temporary_file(monkeypatch)
    path = tmp_path / "map.yaml"
    path.write_text("map:\n  name: M\n  comment: a much longer old value\n")
    path.chmod(0o640)
    monkeypatch.setattr(os, "write", one_byte)
    _update_yaml(path, _IDS)
    monkeypatch.undo()
    loaded = yaml.safe_load(path.read_text())
    assert loaded["map"]["no_3d_model_lanelet_ids"] == _IDS
    assert loaded["map"]["comment"] == "a much longer old value"
    assert path.stat().st_mode & 0o777 == 0o640


# ---------------------------------------------------------------------------
# What follows the key on its own line (#2261)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ids", "expected"),
    [
        (
            "map:\n  name: M\n  no_3d_model_lanelet_ids:  # keep me\n"
            "    - 1\n    - 2\n  other: 3\n",
            [7, 8],
            "map:\n  name: M\n  no_3d_model_lanelet_ids:  # keep me\n"
            "    - 7\n    - 8\n  other: 3\n",
        ),
        (
            "map:\n  name: M\n  no_3d_model_lanelet_ids:  # keep me\n"
            "    - 1\n  other: 3\n",
            [],
            "map:\n  name: M\n  no_3d_model_lanelet_ids: []  # keep me\n  other: 3\n",
        ),
        (
            "map:\n  name: M\n  no_3d_model_lanelet_ids: &ids  # keep me\n" "    - 1\n",
            [7, 8],
            "map:\n  name: M\n  no_3d_model_lanelet_ids: &ids  # keep me\n"
            "    - 7\n    - 8\n",
        ),
        (
            "map:\n  name: M\n  no_3d_model_lanelet_ids: &ids  # keep me\n" "    - 1\n",
            [],
            "map:\n  name: M\n  no_3d_model_lanelet_ids: &ids []  # keep me\n",
        ),
        (
            "map:\r\n  name: M\r\n  no_3d_model_lanelet_ids:  # keep me\r\n"
            "    [1, 2]\r\n",
            [7, 8],
            "map:\r\n  name: M\r\n  no_3d_model_lanelet_ids: [7, 8]  # keep me\r\n",
        ),
    ],
)
def test_a_comment_or_anchor_on_the_keys_line_is_kept(
    text: str, ids: list[int], expected: str
) -> None:
    """The value starts on a later line, so the replaced span began at the key
    and took the rest of its line -- a comment, an anchor -- with it."""
    assert update_no_3d_model_lanelet_ids_text(text, ids) == expected


def test_a_comment_after_a_value_on_the_keys_line_is_still_kept() -> None:
    text = "map:\n  name: M\n  no_3d_model_lanelet_ids: [1]  # keep me\n"
    assert update_no_3d_model_lanelet_ids_text(text, [7, 8]) == (
        "map:\n  name: M\n  no_3d_model_lanelet_ids: [7, 8]  # keep me\n"
    )


@pytest.mark.parametrize(
    ("value", "expected_tail"),
    [
        ("[]  # keep me", "  # keep me"),
        (" # keep me", "  # keep me"),
        ("null # keep me", " # keep me"),
        ("~ # keep me", " # keep me"),
    ],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_a_comment_after_a_value_on_the_keys_line_stays_there_for_a_block_list(
    value: str, expected_tail: str, newline: str
) -> None:
    """The block list replacing a value that starts on the key's line goes on
    later lines, so the comment stays on the key's line, not after the last
    item (#2412)."""
    nl = newline
    text = f"map:{nl}  name: M{nl}  no_3d_model_lanelet_ids: {value}{nl}  other: 3{nl}"
    assert update_no_3d_model_lanelet_ids_text(text, [3, 4]) == (
        f"map:{nl}  name: M{nl}  no_3d_model_lanelet_ids:{expected_tail}{nl}"
        f"    - 3{nl}    - 4{nl}  other: 3{nl}"
    )


def test_a_key_line_comment_stays_there_at_the_end_of_the_file() -> None:
    text = "no_3d_model_lanelet_ids: []  # keep me"
    assert update_no_3d_model_lanelet_ids_text(text, [3, 4]) == (
        "no_3d_model_lanelet_ids:  # keep me\n  - 3\n  - 4"
    )


# ---------------------------------------------------------------------------
# The YAML text rendering, moved into the checked tools.yaml_text (#2364)
# ---------------------------------------------------------------------------
#
# The ``_old_*`` functions are what tools/detect_no_3d_model_lanelets.py ran
# before #2364, verbatim but for the names: the comment split was a regex
# with a lookbehind, which Codon's re lacks, and the key line's tail was read
# off the YAML nodes.


def _old_key_line_tail(text: str, key_node: yaml.Node, value_node: yaml.Node) -> str:
    colon = text.find(":", int(key_node.end_mark.index))
    line_ends = [i for i in (text.find("\n", colon), text.find("\r", colon)) if i != -1]
    line_end = min(line_ends) if line_ends else len(text)
    if tool._content_start(value_node) <= line_end:
        return ""
    return text[colon + 1 : line_end]


def _old_key_line_tail_at(text: str, key_end: int, content_start: int) -> str:
    colon = text.find(":", key_end)
    line_ends = [i for i in (text.find("\n", colon), text.find("\r", colon)) if i != -1]
    line_end = min(line_ends) if line_ends else len(text)
    if content_start <= line_end:
        return ""
    return text[colon + 1 : line_end]


def _old_comment_start(tail: str) -> int:
    comment = re.search(r"\s*(?:^|(?<=\s))#", tail)
    return comment.start() if comment else len(tail)


def _old_format_value(
    lanelet_ids: Sequence[int],
    indent: str,
    newline: str,
    *,
    flow: bool,
    tail: str = "",
) -> str:
    if not lanelet_ids or flow:
        items = ", ".join(str(int(lid)) for lid in lanelet_ids)
        comment = re.search(r"\s*(?:^|(?<=\s))#", tail)
        split = comment.start() if comment else len(tail)
        properties = tail[:split].strip()
        after = tail[split:]
        if after.startswith("#"):
            after = " " + after
        head = (
            f"{NO_3D_MODEL_KEY}: {properties} "
            if properties
            else f"{NO_3D_MODEL_KEY}: "
        )
        return f"{head}[{items}]{after}"
    lines = [f"{NO_3D_MODEL_KEY}:{tail}"]
    lines.extend(f"{indent}  - {int(lid)}" for lid in lanelet_ids)
    return newline.join(lines)


def _old_insert_after_line(text: str, last_end: int, entry: str, newline: str) -> str:
    line_end = text.find("\n", last_end)
    if line_end == -1:
        prefix = text if text.endswith(("\n", "\r")) else text + newline
        result = prefix + entry
    else:
        result = text[: line_end + 1] + entry + text[line_end + 1 :]
    return result


#: Characters a key line's tail is made of: ASCII and Unicode whitespace
#: (str.isspace(), which ``\s`` matches), several ``#``, an anchor's and a
#: tag's characters, line breaks and text.
_TAIL_CHARS = (
    " ",
    "\t",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x85",
    "\xa0",
    " ",
    " ",
    " ",
    "　",
    "﻿",
    "#",
    "#",
    "#",
    "&",
    "!",
    "*",
    "a",
    "b",
    "é",
    "漢",
    "\n",
    "\r",
    ":",
)


def _random_tail(rng: random.Random) -> str:
    return "".join(rng.choice(_TAIL_CHARS) for _ in range(rng.randrange(0, 14)))


def test_the_comment_split_is_the_lookbehind_regex_s() -> None:
    rng = random.Random(2364)
    tails = [_random_tail(rng) for _ in range(20000)]
    tails += ["", "#", " #", "a#b #c", "&x#y\t#z", "　#", "##", "a # #"]
    for tail in tails:
        assert yaml_text.comment_start(tail) == _old_comment_start(tail), repr(tail)


def test_whitespace_is_what_the_regex_calls_whitespace() -> None:
    """``\\s`` of a ``str`` pattern and str.isspace() agree on every code
    point, so the loop backs up over the same characters."""
    for code in range(0x110000):
        char = chr(code)
        assert bool(re.fullmatch(r"\s", char)) == char.isspace(), hex(code)


def test_the_value_renders_as_before() -> None:
    rng = random.Random(749)
    for _ in range(5000):
        ids = [rng.randrange(-5, 10**6) for _ in range(rng.randrange(0, 4))]
        indent = " " * rng.randrange(0, 5)
        newline = rng.choice(("\n", "\r\n"))
        flow = rng.random() < 0.5
        tail = _random_tail(rng)
        assert yaml_text.format_value(
            ids, indent, newline, flow=flow, tail=tail
        ) == _old_format_value(ids, indent, newline, flow=flow, tail=tail)
        assert tool._format_value(ids, indent, newline, flow=flow) == _old_format_value(
            ids, indent, newline, flow=flow
        )


def test_the_key_line_tail_from_indices_is_as_before() -> None:
    rng = random.Random(321)
    for _ in range(5000):
        text = _random_tail(rng) + ":" + _random_tail(rng) + _random_tail(rng)
        key_end = rng.randrange(0, len(text) + 1)
        content_start = rng.randrange(0, len(text) + 2)
        assert yaml_text.key_line_tail(
            text, key_end, content_start
        ) == _old_key_line_tail_at(text, key_end, content_start)


def test_the_insert_after_a_line_is_as_before() -> None:
    rng = random.Random(461)
    for _ in range(5000):
        text = _random_tail(rng) + _random_tail(rng)
        last_end = rng.randrange(0, len(text) + 1)
        newline = rng.choice(("\n", "\r\n"))
        entry = "  key: [1]" + newline
        assert yaml_text.insert_after_line(
            text, last_end, entry, newline
        ) == _old_insert_after_line(text, last_end, entry, newline)


def _old_update(text: str, lanelet_ids: Sequence[int]) -> str:
    """update_no_3d_model_lanelet_ids_text before #2364, verbatim but for the
    helpers it called (the old ones above; the unchanged ones off the tool),
    and for the key-line comment #2412 keeps on the key's line."""
    import copy  # noqa: PLC0415

    ids = [int(lid) for lid in lanelet_ids]
    newline = "\r\n" if "\r\n" in text else "\n"

    try:
        original = yaml.safe_load(text)
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise YamlUpdateError(f"not valid YAML: {exc}") from exc
    if not isinstance(original, dict) or not isinstance(root, yaml.MappingNode):
        raise YamlUpdateError("expected a YAML mapping at the top level")

    path = tool._target_path(original)
    label = ".".join((*path, NO_3D_MODEL_KEY))
    section: yaml.Node = root
    for name in path:
        found = (
            tool._find_key(section, name)
            if isinstance(section, yaml.MappingNode)
            else None
        )
        if found is None:
            raise YamlUpdateError(f"cannot set {label}: {name!r} not found")
        section = found[1]
    if not isinstance(section, yaml.MappingNode):
        raise YamlUpdateError(
            f"cannot set {label}: {'.'.join(path)!r} is not a mapping"
        )
    if section.flow_style:
        where = repr(".".join(path)) if path else "the top level"
        raise YamlUpdateError(
            f"cannot set {label}: {where} is a flow mapping ({{...}}), which "
            "cannot be edited as text; rewrite it as a block mapping"
        )

    first_key = section.value[0][0]
    indent = " " * int(first_key.start_mark.column)
    existing = tool._find_key(section, NO_3D_MODEL_KEY)
    if existing is not None:
        key_node, value_node = existing
        start = int(key_node.start_mark.index)
        if value_node.start_mark.index < key_node.end_mark.index:
            raise YamlUpdateError(
                f"cannot set {label}: its value is an alias (*name); "
                "replace it with a list"
            )
        end = max(tool._content_end(value_node, text), int(key_node.end_mark.index))
        flow = bool(
            isinstance(value_node, yaml.SequenceNode)
            and value_node.flow_style
            and value_node.value
        )
        tail = _old_key_line_tail(text, key_node, value_node)
        if not tail and ids and not flow:
            # #2412: a comment after a value on the key's line stays there.
            rest = re.match(r"[^\r\n]*", text[end:]).group()  # type: ignore[union-attr]
            comment = rest[_old_comment_start(rest) :]
            if comment:
                tail = comment
                end += len(rest)
        replacement = _old_format_value(ids, indent, newline, flow=flow, tail=tail)
        result = text[:start] + replacement + text[end:]
    else:
        last_end = tool._content_end(section.value[-1][1], text)
        line_end = text.find("\n", last_end)
        entry = indent + _old_format_value(ids, indent, newline, flow=False) + newline
        if line_end == -1:
            prefix = text if text.endswith(("\n", "\r")) else text + newline
            result = prefix + entry
        else:
            result = text[: line_end + 1] + entry + text[line_end + 1 :]

    expected = copy.deepcopy(original)
    target = expected
    for name in path:
        target = target[name]
    target[NO_3D_MODEL_KEY] = ids
    try:
        reparsed = yaml.safe_load(result)
    except yaml.YAMLError as exc:
        raise YamlUpdateError(
            f"setting {label} as text would produce invalid YAML: {exc}"
        ) from exc
    if reparsed != expected:
        raise YamlUpdateError(
            f"setting {label} as text would change more than that value"
        )
    return result


def _random_document(rng: random.Random) -> str:
    newline = rng.choice(("\n", "\r\n"))
    comment = rng.choice(("", "  # c", " #x", "\t# a#b", "  #"))
    props = rng.choice(("", " &ids", " !!seq", " &a#b", " &ids  # keep"))
    values = [
        f" [1, 2]{comment}",
        f" []{comment}",
        f"{props}{comment}{newline}    - 1{newline}    - 2",
        f"{props}{comment}{newline}  - 3",
        f"{props}{newline}    [4, 5]{comment}",
        f" null{comment}",
        f"{comment}",
    ]
    lines = []
    in_map = rng.random() < 0.7
    if in_map:
        lines.append("map:" + rng.choice(("", "  # section")))
    pad = "  " if in_map else ""
    lines.append(f"{pad}name: M" + rng.choice(("", " # n")))
    if rng.random() < 0.7:
        value = rng.choice(values).replace(newline + "  ", newline + "  " + pad)
        lines.append(f"{pad}{NO_3D_MODEL_KEY}:{value}")
    if rng.random() < 0.4:
        lines.append(f"{pad}other: |{newline}{pad}  text{newline}")
    if rng.random() < 0.3:
        lines.append("# trailing" + rng.choice(("", "  #")))
    text = newline.join(lines)
    return text + newline if rng.random() < 0.7 else text


def test_the_whole_edit_is_as_before() -> None:
    """Over random documents, the text written (or the refusal) is the old
    edit's."""
    rng = random.Random(469)
    for _ in range(1500):
        text = _random_document(rng)
        ids = [rng.randrange(0, 100) for _ in range(rng.randrange(0, 3))]
        try:
            old: object = _old_update(text, ids)
        except YamlUpdateError as exc:
            old = ("refused", str(exc))
        try:
            new: object = update_no_3d_model_lanelet_ids_text(text, ids)
        except YamlUpdateError as exc:
            new = ("refused", str(exc))
        assert new == old, repr(text)


def test_the_tool_still_hands_out_the_moved_names() -> None:
    assert tool.NO_3D_MODEL_KEY is yaml_text.NO_3D_MODEL_KEY
    assert tool._format_value is yaml_text.format_value
    assert tool._insert_after_line is yaml_text.insert_after_line
