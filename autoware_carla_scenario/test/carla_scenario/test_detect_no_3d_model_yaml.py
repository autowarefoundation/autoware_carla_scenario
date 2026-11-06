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
import tempfile
from pathlib import Path

import pytest
import yaml

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
) -> list[str]:
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit) as excinfo:
        _write_detected_ids(path, ids)
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err + captured.out
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 2, errors
    assert str(path) in errors[0]
    assert errors[0].endswith("; the file was not changed")
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

    def refuse_write(self: Path, data: bytes) -> int:
        raise PermissionError(errno.EACCES, "Permission denied", str(self))

    monkeypatch.setattr(tempfile, "mkstemp", refuse)
    monkeypatch.setattr(Path, "write_bytes", refuse_write)
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
