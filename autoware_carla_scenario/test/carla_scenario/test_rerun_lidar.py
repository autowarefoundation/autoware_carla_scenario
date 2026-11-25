"""The ego's sensors in a run's Rerun recording (``rerun.lidar``,
``rerun.cameras``): written beside the run's own recording, under its id and on
its timelines, only while a run records, and merged into it."""

from __future__ import annotations

import re
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
        self.log_time = True

    def set_log_time_enabled(self, enabled: bool) -> None:
        self.log_time = enabled

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
        # Absent: the documented default (#2298).
        {"strict": False, "lidar_max_points": 65536, "cameras": True},
    ]


@pytest.mark.parametrize("bad", [-1, -1.0, 1.5, True, "-1", "1.5", "lots"])
def test_a_bad_rerun_lidar_max_points_is_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    built: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "autoware_carla_scenario.examples.run.RerunConverter",
        lambda **kwargs: built.append(kwargs),
    )

    def cfg(**rerun: Any) -> Any:
        return OmegaConf.create({"rerun": rerun})

    bad_case = cfg(enabled=True, lidar=True, lidar_max_points=bad)
    good_case = cfg(enabled=True, lidar=True, lidar_max_points=100)
    for cases in ((bad_case,), (bad_case, good_case), (good_case, bad_case)):
        with pytest.raises(ValueError, match=r"rerun\.lidar_max_points"):
            build_rerun_converter(*cases)
    assert built == []
    # Not read when the case records no LiDAR sweeps.
    build_rerun_converter(cfg(enabled=True, lidar=False, lidar_max_points=bad))
    assert build_rerun_converter(cfg(lidar=True, lidar_max_points=bad)) is None
    # null is the documented default, 65536 (#2298); only 0 keeps every point.
    build_rerun_converter(cfg(enabled=True, lidar=True, lidar_max_points=None))
    assert built == [{"strict": False}, {"strict": False, "lidar_max_points": 65536}]


def test_rerun_switches_given_as_text_are_read_as_what_they_spell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``${oc.env:VAR,false}`` and a quoted override give text: ``"false"`` is
    off, not on as ``bool("false")`` read it (#1753)."""
    built: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "autoware_carla_scenario.examples.run.RerunConverter",
        lambda **kwargs: built.append(kwargs),
    )

    def cfg(**rerun: Any) -> Any:
        return OmegaConf.create({"rerun": rerun})

    assert build_rerun_converter(cfg(enabled="false", lidar="true")) is None
    assert build_rerun_converter(cfg(enabled=0, cameras="yes")) is None
    build_rerun_converter(
        cfg(enabled="true", strict="false", lidar="no", cameras="off")
    )
    build_rerun_converter(cfg(enabled="yes", strict="1", lidar="on", cameras=1))
    assert built == [
        {"strict": False},
        {"strict": True, "lidar_max_points": 65536, "cameras": True},
    ]


@pytest.mark.parametrize("key", ["enabled", "strict", "lidar", "cameras"])
@pytest.mark.parametrize("bad", ["ture", "", 2, 0.5, [True]])
def test_a_rerun_switch_that_spells_neither_is_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch, key: str, bad: Any
) -> None:
    from autoware_carla_scenario.examples import run

    monkeypatch.setattr(
        "autoware_carla_scenario.examples.run.RerunConverter",
        lambda **kwargs: pytest.fail("built"),
    )
    rerun = {"enabled": True, key: bad}
    good = OmegaConf.create({"rerun": {"enabled": False}})
    bad_case = OmegaConf.create({"rerun": rerun})
    message = rf"^rerun\.{key} must be true or false, got {re.escape(repr(bad))}$"
    with pytest.raises(ValueError, match=message):
        build_rerun_converter(good, bad_case)
    # check_batch_configs refuses it in one keyed line, whichever case sets
    # it, before a run starts (no queue, no Rerun SDK).
    monkeypatch.setattr(run, "check_batch_map", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "check_scenario_maps", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match=message):
        run._check_batch_writer_settings([good, bad_case])
    with pytest.raises(run.RunConfigError, match=message):
        run.check_batch_configs([good, bad_case], ["first", "second"])


@pytest.mark.parametrize(
    ("given", "points"), [(100, 100), (1e5, 100000), ("100", 100), (0.0, 0)]
)
def test_an_integral_rerun_lidar_max_points_is_read_as_that_integer(
    monkeypatch: pytest.MonkeyPatch, given: Any, points: int
) -> None:
    """``1e5`` in YAML is a float and ``${oc.env:...}`` a string (#1226)."""
    built: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "autoware_carla_scenario.examples.run.RerunConverter",
        lambda **kwargs: built.append(kwargs),
    )
    build_rerun_converter(
        OmegaConf.create(
            {"rerun": {"enabled": True, "lidar": True, "lidar_max_points": given}}
        )
    )
    assert built == [{"strict": False, "lidar_max_points": points}]
    assert type(built[0]["lidar_max_points"]) is int


@pytest.mark.parametrize(
    ("rerun", "points"),
    [
        ({}, 65536),
        ({"lidar_max_points": None}, 65536),
        ({"lidar_max_points": 0}, 0),
        ({"lidar_max_points": 7}, 7),
    ],
)
def test_an_absent_or_null_rerun_lidar_max_points_is_the_default(
    rerun: dict[str, Any], points: int
) -> None:
    """Absent or null is the documented 65536, and only an explicit 0 keeps
    every point of a sweep (#2298: null kept every point)."""
    from autoware_carla_scenario.examples import run

    cfg = OmegaConf.create({"rerun": {"enabled": True, "lidar": True, **rerun}})
    assert run._rerun_lidar_max_points(cfg) == points
    assert run._DEFAULT_LIDAR_MAX_POINTS == 65536
    shipped = OmegaConf.load(
        Path(run.__file__).resolve().parent / "conf" / "config.yaml"
    )
    assert shipped.rerun.lidar_max_points == run._DEFAULT_LIDAR_MAX_POINTS


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


def test_a_failed_merge_names_the_command_and_its_stderr(
    tmp_path: Path, fake_cli: Path
) -> None:
    """``str(CalledProcessError)`` drops the captured stderr (#2182)."""
    import subprocess

    from autoware_carla_scenario.rerun_lidar import RerunCliError, merge_into

    fake_cli.write_text("#!/bin/sh\necho 'bad chunk' >&2\nexit 3\n")
    recording = tmp_path / "Run-5.rrd"
    lidar = tmp_path / "Run-5.sensors.rrd"
    recording.write_bytes(b"RUN")
    lidar.write_bytes(b"+LIDAR")
    with pytest.raises(subprocess.CalledProcessError) as raised:
        merge_into(recording, lidar)
    assert isinstance(raised.value, RerunCliError)
    message = str(raised.value)
    assert f"rrd merge {recording} {lidar}" in message
    assert "exited with status 3: bad chunk" in message
    assert recording.read_bytes() == b"RUN"
    assert lidar.read_bytes() == b"+LIDAR"


def test_a_hung_merge_times_out_with_its_stderr(
    tmp_path: Path, fake_cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung ``rerun rrd merge`` must not block the converter forever (#2182)."""
    import subprocess

    from autoware_carla_scenario import rerun_lidar

    fake_cli.write_text(
        "#!/bin/sh\necho 'merging' >&2\necho partial > \"$6\"\nexec sleep 30\n"
    )
    monkeypatch.setattr(rerun_lidar, "CLI_TIMEOUT_S", 0.5)
    recording = tmp_path / "Run-6.rrd"
    lidar = tmp_path / "Run-6.sensors.rrd"
    recording.write_bytes(b"RUN")
    lidar.write_bytes(b"+LIDAR")
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        rerun_lidar.merge_into(recording, lidar)
    assert isinstance(raised.value, rerun_lidar.RerunCliTimeoutError)
    message = str(raised.value)
    assert "rrd merge" in message
    assert "timed out after 0.5 s: merging" in message
    assert recording.read_bytes() == b"RUN"
    assert lidar.read_bytes() == b"+LIDAR"
    # The killed merge's partial output is not left behind.
    assert sorted(p.name for p in tmp_path.glob(".*")) == []


def test_the_cli_calls_are_capped() -> None:
    from autoware_carla_scenario import rerun_lidar

    assert rerun_lidar.CLI_TIMEOUT_S == 600.0


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


def test_a_lens_camera_is_drawn_without_a_pinhole(tmp_path: Path) -> None:
    """Rerun's camera model is a pinhole alone: an F-theta lens camera gets its
    mount and its frames, not a pinhole of its ``fov`` that would misplace its
    frustum and its image (#2452)."""
    from autoware_carla_scenario.rerun_lidar import CameraMount, SensorRecorder

    sdk = _Sdk()
    recorder = SensorRecorder(
        tmp_path / "Run-6.sensors.rrd", "Run-6", 100, sdk=sdk, lidar=False, cameras=True
    )
    mount = CameraMount(832, 480, 120.0, (1.5, 0.0, 1.6), pinhole=False)
    assert recorder.log_camera_frame("wide", 101, b"JPEG1", mount)
    assert recorder.log_camera_frame("wide", 102, b"JPEG2", mount)
    kinds = [a.kind for _, _, a in sdk.streams[0].logged]
    assert kinds == ["transform", "image", "image"]
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


# ---------------------------------------------------------------------------
# The SDK's threads, its timelines and the ego's tilt, read back (#2485)
# ---------------------------------------------------------------------------


def test_the_fake_stream_records_no_wall_clock(tmp_path: Path) -> None:
    sdk = _Sdk()
    LidarRecorder(tmp_path / "r.sensors.rrd", "r", 0, sdk=sdk).close()
    assert sdk.streams[0].log_time is False


#: Five recordings through the real SDK in a process of their own, printing
#: the stream threads alive after the first and after the last on a
#: ``threads:`` line.
_LEAK_PROBE = """
import sys
from pathlib import Path
import numpy as np
from rerun_bindings import rerun_bindings as bindings
from autoware_carla_scenario.rerun import RerunConverter
from autoware_carla_scenario.rerun_lidar import active_recorder

NAMES = ("ChunkBatcher", "RecordingStream", "file_writer")


def threads():
    count = 0
    for thread in Path("/proc/self/task").iterdir():
        try:
            name = (thread / "comm").read_text().strip()
        except OSError:
            continue
        count += name.startswith(NAMES)
    return count


class Converter:
    def convert(self, jsonl, out_rrd, *, strict=False):
        Path(out_rrd).write_bytes(b"RRD")
        return {"ok": True}


def one_run(out, i):
    converter = RerunConverter(converter=Converter(), lidar_max_points=0)
    converter.start_sensors(out / f"Run-{i}.trajectory.jsonl", f"Run-{i}", 0)
    recorder = active_recorder()
    assert recorder is not None
    recorder.log_sweep("top", 1, np.ones((4, 3), np.float32), np.eye(3), np.zeros(3))
    converter.stop_sensors()
    converter.close()
    assert bindings.check_for_rrd_footer(str(out / f"Run-{i}.sensors.rrd")), i


out = Path(sys.argv[1])
one_run(out, 0)
before = threads()
for i in range(1, 6):
    one_run(out, i)
print("threads:", before, threads(), flush=True)
"""


def test_sensor_recordings_leave_no_threads_behind(tmp_path: Path) -> None:
    """Rerun 0.38 never joins a closed stream's two threads unless told to
    clean up: a batch would grow by two a case.  Counted in a process of its
    own: under xdist the worker holds other tests' streams, whose threads come
    and go on their own (#2489, #2502)."""
    import subprocess

    pytest.importorskip("rerun")
    if not Path("/proc/self/task").is_dir():
        pytest.skip("needs /proc to name the process's threads")
    result = subprocess.run(
        [sys.executable, "-c", _LEAK_PROBE, str(tmp_path)],
        # Not the repository root: its autoware_carla_scenario/ directory is
        # on a ``-c`` script's path first.
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # The counts are on the probe's own line: the SDK or a logger may print
    # to stdout too, before it or at exit.
    counts = [
        line for line in result.stdout.splitlines() if line.startswith("threads:")
    ]
    assert counts, result.stdout
    before, after = (int(n) for n in counts[-1].split()[1:])
    # Still a whole recording each time (the probe asserts its footer), and no
    # stream left running: two threads a case would make it 10 more.
    assert after <= before, result.stdout


def test_closing_a_recording_keeps_the_global_one(tmp_path: Path) -> None:
    """The SDK's sweep of orphaned recordings unsets ``rr.init``'s global
    recording: a policy logging to it in the same process would lose
    everything it logged after the run's sensors were closed."""
    rr = pytest.importorskip("rerun")
    from rerun_bindings import rerun_bindings as bindings

    rr.init("global_test", recording_id="global_test", spawn=False)
    rr.save(str(tmp_path / "global.rrd"))
    try:
        held = bindings.get_global_data_recording()
        assert held is not None
        rr.log("before", rr.Points3D([[0.0, 0.0, 0.0]]))
        LidarRecorder(tmp_path / "r.sensors.rrd", "r", 0).close()
        assert bindings.get_global_data_recording() is not None
        rr.log("after", rr.Points3D([[1.0, 1.0, 1.0]]))
    finally:
        rr.disconnect()
        bindings.set_global_data_recording(None)
    assert bindings.check_for_rrd_footer(str(tmp_path / "global.rrd"))
    entities = {chunk.entity_path for chunk in _read(tmp_path / "global.rrd")}
    assert {"/before", "/after"} <= entities


def _read(recording: Path) -> list[Any]:
    """The chunks of *recording*'s one recording."""
    chunk = pytest.importorskip("rerun.chunk")
    reader = chunk.RrdReader(str(recording))
    (store,) = reader.recordings()
    return list(reader.stream(store=store).to_chunks())


def test_the_sensors_are_on_the_converters_timelines_alone(tmp_path: Path) -> None:
    """No wall-clock `log_time`: the converter's chunks have none, and the
    viewer would show the sensors alone on it."""
    pytest.importorskip("rerun")
    from autoware_carla_scenario.rerun_lidar import CameraMount

    recorder = LidarRecorder(tmp_path / "r.sensors.rrd", "r", 0, cameras=True)
    recorder.log_sweep("top", 3, np.ones((4, 3), np.float32), _IDENTITY, _ORIGIN)
    recorder.log_camera_frame("front", 3, b"\xff\xd8\xff\xd9", CameraMount(4, 4, 90.0))
    recorder.log_ego_tilt(3, 0.0, 5.0)
    recorder.close()
    timed = [c for c in _read(tmp_path / "r.sensors.rrd") if not c.is_static]
    assert timed
    for chunk in timed:
        assert set(chunk.timeline_names) == {"frame", "sim_time"}, chunk.entity_path


def test_the_cameras_are_drawn_with_the_egos_pitch_and_roll(tmp_path: Path) -> None:
    """The converter's ``world/ego/pose`` has the ego's yaw alone; the
    cameras' parent carries the rest of its rotation on every tick."""
    pytest.importorskip("rerun")
    from autoware_carla_scenario.rerun_lidar import CameraMount, ego_tilt

    recorder = LidarRecorder(
        tmp_path / "r.sensors.rrd", "r", 100, lidar=False, cameras=True
    )
    assert not recorder.log_ego_tilt(99, 0.0, 0.0)  # a warm-up tick
    # CARLA's pitch up and roll right, in degrees.
    assert recorder.log_ego_tilt(100, 2.0, 10.0)
    assert recorder.log_ego_tilt(101, 0.0, 0.0)
    recorder.log_camera_frame(
        "front", 100, b"\xff\xd8\xff\xd9", CameraMount(4, 4, 90.0)
    )
    recorder.close()

    rows: list[tuple[int, Any]] = []
    for chunk in _read(tmp_path / "r.sensors.rrd"):
        if chunk.entity_path != "/world/ego/pose/cameras" or chunk.is_static:
            continue
        batch = chunk.to_record_batch().to_pydict()
        [frames] = [v for k, v in batch.items() if k == "frame"]
        [matrices] = [v for k, v in batch.items() if k.endswith("mat3x3")]
        rows.extend(zip(frames, matrices))
    assert [frame for frame, _ in rows] == [100, 101]
    tilt = np.asarray(rows[0][1], dtype=np.float64).reshape(3, 3)
    # Rerun keeps a 3x3 column-major.
    np.testing.assert_allclose(tilt.T, ego_tilt(2.0, 10.0), atol=1e-6)
    # CARLA's pitch up lifts the ego's nose (+z) in the right-handed frame.
    nose = ego_tilt(0.0, 10.0) @ np.array([1.0, 0.0, 0.0])
    up = np.radians(10.0)
    np.testing.assert_allclose(nose, np.array([np.cos(up), 0.0, np.sin(up)]))
    np.testing.assert_allclose(
        np.asarray(rows[1][1]).reshape(3, 3), np.eye(3), atol=1e-9
    )


def test_no_tilt_is_written_without_the_cameras(tmp_path: Path) -> None:
    sdk = _Sdk()
    recorder = LidarRecorder(tmp_path / "r.sensors.rrd", "r", 0, sdk=sdk)
    assert not recorder.log_ego_tilt(5, 1.0, 1.0)
    assert sdk.streams[0].logged == []
    recorder.close()
