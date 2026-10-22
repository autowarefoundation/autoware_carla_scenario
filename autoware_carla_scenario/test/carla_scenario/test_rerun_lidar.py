"""The ego's sensors in a run's Rerun recording (``rerun.lidar``,
``rerun.cameras``): written beside the run's own recording, under its id and on
its timelines, only while a run records, and merged into it."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.constants import FIXED_DELTA_SECONDS
from autoware_carla_scenario.examples.run import build_rerun_converter
from autoware_carla_scenario.rerun import RerunConverter
from autoware_carla_scenario.rerun_lidar import (
    APPLICATION_ID,
    INSTALL_HINT,
    LidarRecorder,
    LidarRerunUnavailableError,
    active_recorder,
    require_sdk,
)


class _Points3D:
    def __init__(self, positions: Any, *, colors: Any, radii: float) -> None:
        self.positions = np.asarray(positions)
        self.colors = np.asarray(colors)


class _Logged:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _Stream:
    def __init__(self, application_id: str, *, recording_id: str) -> None:
        self.ids = (application_id, recording_id)
        self.saved: str | None = None
        self.times: dict[str, Any] = {}
        self.logged: list[tuple[str, dict[str, Any], Any]] = []
        self.closed = False

    def save(self, path: str) -> None:
        self.saved = path

    def set_time(self, timeline: str, **value: Any) -> None:
        self.times[timeline] = value

    def log(self, entity: str, *archetypes: Any, static: bool = False) -> None:
        times = {} if static else dict(self.times)
        for archetype in archetypes:
            self.logged.append((entity, times, archetype))

    def flush(self) -> None:
        pass

    def disconnect(self) -> None:
        self.closed = True


class _Sdk:
    """Stands in for the ``rerun`` module."""

    Points3D = _Points3D

    class ViewCoordinates:
        FLU = "FLU"

    @staticmethod
    def Transform3D(**fields: Any) -> _Logged:  # noqa: N802
        return _Logged(kind="transform", **fields)

    @staticmethod
    def Pinhole(**fields: Any) -> _Logged:  # noqa: N802
        return _Logged(kind="pinhole", **fields)

    @staticmethod
    def EncodedImage(**fields: Any) -> _Logged:  # noqa: N802
        return _Logged(kind="image", **fields)

    def __init__(self) -> None:
        self.streams: list[_Stream] = []

    def RecordingStream(self, application_id: str, *, recording_id: str) -> _Stream:  # noqa: N802
        stream = _Stream(application_id, recording_id=recording_id)
        self.streams.append(stream)
        return stream


class _Converter:
    def convert(self, jsonl: str, out_rrd: str, *, strict: bool = False) -> dict:
        Path(out_rrd).write_bytes(b"RRD")
        return {"ok": True}


_IDENTITY = np.eye(3)
_ORIGIN = np.zeros(3)


def test_without_the_sdk_lidar_is_refused_saying_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(LidarRerunUnavailableError) as raised:
        require_sdk()
    assert INSTALL_HINT in str(raised.value)
    with pytest.raises(LidarRerunUnavailableError):
        RerunConverter(converter=_Converter(), lidar_max_points=0)
    # Not asked for: not needed.
    RerunConverter(converter=_Converter()).close()


def test_a_sweep_lands_in_the_world_under_the_runs_id_and_timelines(
    tmp_path: Path,
) -> None:
    sdk = _Sdk()
    recorder = LidarRecorder(tmp_path / "Run-3.sensors.rrd", "Run-3", 100, sdk=sdk)
    stream = sdk.streams[0]
    assert stream.ids == (APPLICATION_ID, "Run-3")
    assert stream.saved == str(tmp_path / "Run-3.sensors.rrd")

    # The rig turned a quarter left and 10 m east of the origin.
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    points = np.array([[1.0, 0.0, 0.5, 0.9], [0.0, 2.0, 0.0, 0.1]], np.float32)
    assert recorder.log_sweep("lidar_top", 104, points, turn, np.array([10.0, 0, 0]))

    entity, times, logged = stream.logged[0]
    assert entity == "world/lidar/lidar_top"
    assert times["frame"] == {"sequence": 104}
    assert times["sim_time"]["duration"] == pytest.approx(4 * FIXED_DELTA_SECONDS)
    np.testing.assert_allclose(logged.positions, [[10.0, 1.0, 0.5], [8.0, 0.0, 0.0]])
    assert logged.colors.shape == (2, 3)
    assert recorder.close() == tmp_path / "Run-3.sensors.rrd"
    assert stream.closed


def test_warm_up_sweeps_are_not_written_and_a_sweep_is_thinned_evenly(
    tmp_path: Path,
) -> None:
    sdk = _Sdk()
    recorder = LidarRecorder(tmp_path / "r.sensors.rrd", "r", 100, 10, sdk=sdk)
    sweep = np.arange(400, dtype=np.float32).reshape(100, 4)
    assert not recorder.log_sweep("top", 99, sweep, _IDENTITY, _ORIGIN)
    assert recorder.log_sweep("top", 100, sweep, _IDENTITY, _ORIGIN)
    ((_, _, logged),) = sdk.streams[0].logged
    assert len(logged.positions) == 10
    np.testing.assert_allclose(logged.positions[[0, -1]], sweep[[0, -1], :3])


def test_a_converter_records_lidar_only_between_start_and_stop(tmp_path: Path) -> None:
    sdk = _Sdk()
    converter = RerunConverter(
        converter=_Converter(), lidar_max_points=0, lidar_sdk=sdk
    )
    assert converter.records_sensors
    assert active_recorder() is None
    converter.start_sensors(tmp_path / "Run-1.trajectory.jsonl", "Run-1", 50)
    recorder = active_recorder()
    assert recorder is not None
    assert recorder.path == tmp_path / "Run-1.sensors.rrd"
    assert recorder.start_frame == 50
    converter.stop_sensors()
    assert active_recorder() is None
    assert sdk.streams[0].closed
    converter.close()

    plain = RerunConverter(converter=_Converter())
    plain.start_sensors(tmp_path / "Run-2.trajectory.jsonl", "Run-2", 0)
    assert active_recorder() is None
    plain.close()


def test_rerun_lidar_is_read_from_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []

    class _Recording:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

    monkeypatch.setattr(
        "autoware_carla_scenario.examples.run.RerunConverter", _Recording
    )

    def cfg(**rerun: Any) -> Any:
        return OmegaConf.create({"rerun": rerun})

    assert build_rerun_converter(cfg(enabled=False, lidar=True)) is None
    build_rerun_converter(cfg(enabled=True))
    build_rerun_converter(cfg(enabled=True, lidar=True, lidar_max_points=1000))
    build_rerun_converter(
        cfg(enabled=True, lidar=True, lidar_max_points=1000),
        cfg(enabled=True, lidar=True, lidar_max_points=0),
    )
    assert [b.get("lidar_max_points") for b in built] == [None, 1000, 0]
    built.clear()
    assert build_rerun_converter(cfg(enabled=False, cameras=True)) is None
    build_rerun_converter(cfg(enabled=True, cameras=True))
    build_rerun_converter(cfg(enabled=True, lidar=True, cameras=True))
    assert built == [
        {"strict": False, "cameras": True},
        {"strict": False, "lidar_max_points": 0, "cameras": True},
    ]


@pytest.fixture
def fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The Rerun SDK's CLI, as a script whose ``rrd merge`` concatenates its inputs."""
    import importlib.util
    import types

    package = tmp_path / "cli" / "rerun_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    cli = package / "rerun"
    cli.write_text(
        "#!/bin/sh\n" '[ "$1 $2" = "rrd merge" ] || exit 2\n' 'cat "$3" "$4" > "$6"\n'
    )
    cli.chmod(0o755)
    real = importlib.util.find_spec

    def find_spec(name: str, *args: Any) -> Any:
        if name == "rerun_cli":
            return types.SimpleNamespace(origin=str(package / "__init__.py"))
        return real(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    return cli


def test_the_sweeps_are_merged_into_the_runs_recording(
    tmp_path: Path, fake_cli: Path
) -> None:
    from autoware_carla_scenario.rerun_lidar import merge_into

    recording = tmp_path / "Run-1.rrd"
    lidar = tmp_path / "Run-1.sensors.rrd"
    recording.write_bytes(b"RUN")
    lidar.write_bytes(b"+LIDAR")
    assert merge_into(recording, lidar) == recording
    assert recording.read_bytes() == b"RUN+LIDAR"
    assert not lidar.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Run-1.rrd", "cli"]


def test_a_converter_merges_a_runs_sweeps_once_its_log_is_converted(
    tmp_path: Path, fake_cli: Path
) -> None:
    sdk = _Sdk()
    trajectory = tmp_path / "Run-2.trajectory.jsonl"
    trajectory.write_text("")
    (tmp_path / "Run-2.sensors.rrd").write_bytes(b"+LIDAR")
    with RerunConverter(
        converter=_Converter(), lidar_max_points=0, lidar_sdk=sdk
    ) as converter:
        recording = converter.submit(trajectory, "Run-2")
        (conversion,) = converter.wait()
    assert conversion.ok
    assert recording.read_bytes() == b"RRD+LIDAR"
    assert not (tmp_path / "Run-2.sensors.rrd").exists()


def test_a_failed_merge_keeps_the_sweeps_and_the_recording(
    tmp_path: Path, fake_cli: Path
) -> None:
    fake_cli.write_text("#!/bin/sh\nexit 1\n")
    trajectory = tmp_path / "Run-3.trajectory.jsonl"
    trajectory.write_text("")
    (tmp_path / "Run-3.sensors.rrd").write_bytes(b"+LIDAR")
    with RerunConverter(
        converter=_Converter(), lidar_max_points=0, lidar_sdk=_Sdk()
    ) as converter:
        recording = converter.submit(trajectory, "Run-3")
        (conversion,) = converter.wait()
    assert conversion.ok
    assert recording.read_bytes() == b"RRD"
    assert (tmp_path / "Run-3.sensors.rrd").read_bytes() == b"+LIDAR"


def test_a_camera_frame_is_drawn_in_its_frustum_on_the_ego(tmp_path: Path) -> None:
    from autoware_carla_scenario.rerun_lidar import CameraMount, SensorRecorder

    sdk = _Sdk()
    recorder = SensorRecorder(
        tmp_path / "Run-4.sensors.rrd", "Run-4", 100, sdk=sdk, lidar=False, cameras=True
    )
    # 1.5 m ahead and 0.3 m right of the actor, 1.6 m up, turned 90 degrees right.
    mount = CameraMount(960, 604, 120.0, (1.5, 0.3, 1.6), (0.0, 0.0, 90.0))
    assert not recorder.log_camera_frame("front", 99, b"JPEG0", mount)
    assert recorder.log_camera_frame("front", 101, b"JPEG1", mount)
    assert recorder.log_camera_frame("front", 102, b"JPEG2", mount)
    entities = {e for e, _, _ in sdk.streams[0].logged}
    assert entities == {"world/ego/pose/cameras/front"}
    kinds = [(a.kind, times) for _, times, a in sdk.streams[0].logged]
    # The mount and the pinhole once, static; then a frame per call.
    assert kinds[:2] == [("transform", {}), ("pinhole", {})]
    assert [k for k, _ in kinds[2:]] == ["image", "image"]
    _, times, image = sdk.streams[0].logged[2]
    assert times["frame"] == {"sequence": 101}
    assert image.contents == b"JPEG1" and image.media_type == "image/jpeg"
    transform = sdk.streams[0].logged[0][2]
    assert transform.translation == [1.5, -0.3, 1.6]
    # CARLA's yaw to the right is a turn to the right (negative) in the
    # right-handed frame: the camera's x looks along -y.
    np.testing.assert_allclose(
        transform.mat3x3 @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-9
    )
    pinhole = sdk.streams[0].logged[1][2]
    assert pinhole.width == 960 and pinhole.camera_xyz == "FLU"
    assert pinhole.focal_length == pytest.approx(960 / (2 * np.tan(np.radians(60.0))))
    assert recorder.frames == 2 and recorder.sweeps == 0
    recorder.close()


def test_a_converter_records_cameras_without_the_lidar(tmp_path: Path) -> None:
    sdk = _Sdk()
    converter = RerunConverter(converter=_Converter(), cameras=True, lidar_sdk=sdk)
    assert converter.records_sensors
    converter.start_sensors(tmp_path / "Run-5.trajectory.jsonl", "Run-5", 7)
    recorder = active_recorder()
    assert recorder is not None
    assert recorder.records_cameras and not recorder.records_lidar
    converter.close()
    assert active_recorder() is None
