"""rerun_sensor_mount, the typed core of rerun_lidar split out so the library
check compiles it (#1688): rerun_lidar still exports every name it did, as the
same objects, its ``_CHUNK_LINE`` stays patchable, and :func:`chunk_entities`
and :func:`extract_sensors` read ``rerun rrd print`` exactly as the parse they
replace did."""

from __future__ import annotations

import re
import types
from pathlib import Path
from typing import Any, Optional, Union

import pytest

from autoware_carla_scenario import rerun_lidar, rerun_sensor_mount
from autoware_carla_scenario.entity import carla_driver_entity
from autoware_carla_scenario.rerun_sensor_mount import chunk_entities

#: What rerun_lidar held before the split that now lives in rerun_sensor_mount.
_MOVED = (
    "CameraMount",
    "SENSOR_ENTITIES",
    "_COLOUR_LOW_M",
    "_COLOUR_HIGH_M",
    "_CHUNK_LINE",
    "_height_colours",
    "_is_sensor_entity",
    "chunk_entities",
)


@pytest.mark.parametrize("name", _MOVED)
def test_rerun_lidar_re_exports_the_moved_name(name: str) -> None:
    assert getattr(rerun_lidar, name) is getattr(rerun_sensor_mount, name)


def test_the_driver_entity_mounts_cameras_with_the_checked_class() -> None:
    assert carla_driver_entity.CameraMount is rerun_sensor_mount.CameraMount


#: The regular expression rerun_lidar read the entities with before the split.
_OLD_CHUNK_LINE = re.compile(
    r"^Chunk\([^)]*\) with [^-]* - (/.*?)(?: - data columns:.*)?$"
)

_HEADER = (
    "SetStoreInfo(store_id=Recording('S'), application_id='autoware_carla_scenario')\n"
)


def _chunk(entity: str, columns: bool = True) -> str:
    line = f"Chunk(17E2B8A3C0F1D4E5) with 3 rows (7 columns) - {entity}"
    return line + (" - data columns: [Points3D:positions]" if columns else "")


#: Sample ``rerun rrd print`` outputs, in the form the parse reads and not.
_SAMPLES = {
    "trajectory and sensors": _HEADER
    + "\n".join(
        [
            _chunk("/world/ego/pose"),
            _chunk("/world/lidar/top"),
            _chunk("/world/lidar/top"),
            _chunk("/world/ego/pose/cameras/front"),
            _chunk("/world/ego/pose/cameras/front", columns=False),
            _chunk("/"),
            _chunk("/world/lidar_extra/x"),
            _chunk("/world/lidar"),
            _chunk("/metrics/speed - kph"),
            "ArrowMsg(...)",
        ]
    )
    + "\n",
    "trajectory alone": _HEADER
    + "\n".join(
        [
            _chunk("/world/ego/pose"),
            _chunk("/world/map", columns=False),
            _chunk("/world/ego/pose/camera"),
        ]
    ),
    "sensors alone": "\r\n".join(
        [_chunk("/world/lidar/top"), _chunk("/world/ego/pose/cameras/rear")]
    ),
    "no chunks": _HEADER + "nothing\n",
    "empty": "",
    "another form": _HEADER + "Chunk<17E2B8A3C0F1D4E5 /world/lidar/top>\n",
    "one line unread": "\n".join(
        [
            _chunk("/world/lidar/top"),
            "Chunk(1) with 3 rows - world/no/slash",
            _chunk("/world/ego/pose"),
            "Chunk(2) without rows",
        ]
    ),
}


def _old_parse(printed: str) -> tuple[Optional[set[str]], Optional[str]]:
    """The parse of rerun_lidar.extract_sensors before the split: the entities,
    or ``None`` and the first chunk line it does not read (``None``: no chunks)."""
    chunks = [line for line in printed.splitlines() if line.startswith("Chunk(")]
    matches = [_OLD_CHUNK_LINE.match(line) for line in chunks]
    if not chunks or None in matches:
        unread = next((c for c, m in zip(chunks, matches) if m is None), None)
        return None, unread
    return {match.group(1) for match in matches if match is not None}, None


@pytest.mark.parametrize("sample", sorted(_SAMPLES))
def test_chunk_entities_reads_what_the_old_parse_did(sample: str) -> None:
    printed = _SAMPLES[sample]
    entities, unread = chunk_entities(printed)
    old_entities, old_unread = _old_parse(printed)
    if old_entities is None:
        # Refused: the first unread line, or "" and no entities for no chunks.
        assert unread == (old_unread or "")
        assert unread or not entities
    else:
        assert unread == ""
        assert set(entities) == old_entities
        assert entities  # some chunks
        chunks = [line for line in printed.splitlines() if line.startswith("Chunk(")]
        assert len(entities) == len(chunks)


def _old_extract(recording: Path, printed: str) -> Union[str, bool, list[str]]:
    """What rerun_lidar.extract_sensors did with *printed* before the split:
    the message it raised, ``False`` (no sensors) or the entities it dropped."""
    entities, unread = _old_parse(printed)
    if entities is None:
        return f"cannot read the entities of {recording} from `rerun rrd print`" + (
            f": {unread!r}" if unread is not None else ": no chunks"
        )
    if not any(rerun_lidar._is_sensor_entity(entity) for entity in entities):
        return False
    return sorted(e for e in entities if not rerun_lidar._is_sensor_entity(e))


@pytest.fixture
def printing_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The Rerun SDK's CLI, as a script: ``rrd print`` prints ``$SAMPLE``,
    ``rrd filter`` writes its arguments to ``$FILTER_ARGS`` and an empty
    output file.  Returns where the sample goes."""
    import importlib.util

    package = tmp_path / "cli" / "rerun_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    cli = package / "rerun"
    cli.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2" = "rrd print" ]; then cat "$SAMPLE"; exit 0; fi\n'
        '[ "$1 $2" = "rrd filter" ] || exit 2\n'
        "shift 2\n"
        'printf "%s\\n" "$@" > "$FILTER_ARGS"\n'
        'for arg; do out="$arg"; done\n'
        ': > "$out"\n'
    )
    cli.chmod(0o755)
    real = importlib.util.find_spec

    def find_spec(name: str, *args: Any) -> Any:
        if name == "rerun_cli":
            return types.SimpleNamespace(origin=str(package / "__init__.py"))
        return real(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    sample = tmp_path / "printed.txt"
    monkeypatch.setenv("SAMPLE", str(sample))
    monkeypatch.setenv("FILTER_ARGS", str(tmp_path / "filter_args.txt"))
    return sample


@pytest.mark.parametrize("sample", sorted(_SAMPLES))
def test_extract_sensors_does_what_it_did_before_the_split(
    sample: str, tmp_path: Path, printing_cli: Path
) -> None:
    printing_cli.write_bytes(_SAMPLES[sample].encode())
    recording = tmp_path / "S.rrd"
    recording.write_bytes(b"RRD")
    sensors = tmp_path / "S.sensors.rrd"
    expected = _old_extract(recording, _SAMPLES[sample])
    if isinstance(expected, str):
        with pytest.raises(RuntimeError) as raised:
            rerun_lidar.extract_sensors(recording, sensors)
        assert str(raised.value) == expected
        assert not sensors.exists()
    elif isinstance(expected, bool):
        assert expected is False
        assert rerun_lidar.extract_sensors(recording, sensors) is False
        assert not sensors.exists()
    else:
        assert rerun_lidar.extract_sensors(recording, sensors) is True
        assert sensors.exists()
        args = (tmp_path / "filter_args.txt").read_text().splitlines()
        drops = [arg for arg in expected for arg in ("--drop-entity", arg)]
        assert args == [*drops, str(recording), "-o", args[-1]]
    assert recording.read_bytes() == b"RRD"
    assert not list(tmp_path.glob(".*"))


def test_the_chunk_line_stays_patchable_on_rerun_lidar(
    tmp_path: Path, printing_cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    printing_cli.write_text(_SAMPLES["trajectory and sensors"])
    monkeypatch.setattr(rerun_lidar, "_CHUNK_LINE", re.compile(r"^Chunk<(.*)>$"))
    with pytest.raises(RuntimeError, match="cannot read the entities"):
        rerun_lidar.extract_sensors(tmp_path / "S.rrd", tmp_path / "x.sensors.rrd")
