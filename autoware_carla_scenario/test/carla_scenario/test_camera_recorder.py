"""Unit tests for CameraRecorder, with a fake world and a fake ffmpeg."""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import pytest

from autoware_carla_scenario import camera_recorder
from autoware_carla_scenario.camera_recorder import CameraRecorder


class _Sensor:
    def __init__(self) -> None:
        self.listener: Optional[Callable[[Any], None]] = None
        self.stopped = False
        self.destroyed = False

    def listen(self, callback: Callable[[Any], None]) -> None:
        self.listener = callback

    def stop(self) -> None:
        self.stopped = True

    def destroy(self) -> None:
        self.destroyed = True


class _Stdin:
    """ffmpeg's stdin: what was written, in order with when it was closed.

    *accept* writes are taken (all of them when None), and the next raises
    BrokenPipeError, as a pipe ffmpeg no longer reads does.  A write after
    close() raises, as a closed file does.
    """

    def __init__(self, accept: Optional[int] = None) -> None:
        self._data = bytearray()
        self.events: list[str] = []
        self._accept = accept

    def write(self, data: Any) -> int:
        if "close" in self.events:
            raise ValueError("write to closed file")
        if self._accept is not None and self._accept <= 0:
            raise BrokenPipeError
        if self._accept is not None:
            self._accept -= 1
        self._data += bytes(data)
        self.events.append("write")
        return len(bytes(data))

    def close(self) -> None:
        self.events.append("close")

    def getvalue(self) -> bytes:
        return bytes(self._data)


class _StuckStdin(_Stdin):
    """A pipe ffmpeg, alive, no longer reads: a write blocks until it is killed."""

    def __init__(self) -> None:
        super().__init__()
        self.killed = threading.Event()

    def write(self, data: Any) -> int:
        self.killed.wait()
        raise BrokenPipeError


class _Process:
    def __init__(
        self, returncode: int, stderr: bytes, stdin: Optional[_Stdin] = None
    ) -> None:
        self.stdin = stdin if stdin is not None else _Stdin()
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.killed = False

    def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        if isinstance(self.stdin, _StuckStdin):
            self.stdin.killed.set()


class _Pipeline:
    """ffmpeg-python's fluent pipeline, recording what it was given."""

    def __init__(self, process: _Process) -> None:
        self.process = process
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _Pipeline:
        self.calls.append(("input", args, kwargs))
        return self

    def output(self, *args: Any, **kwargs: Any) -> _Pipeline:
        self.calls.append(("output", args, kwargs))
        return self

    def overwrite_output(self) -> _Pipeline:
        self.calls.append(("overwrite_output", (), {}))
        return self

    def run_async(self, **kwargs: Any) -> _Process:
        self.calls.append(("run_async", (), kwargs))
        return self.process


def _recorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process: _Process
) -> tuple[CameraRecorder, _Sensor, _Pipeline]:
    sensor = _Sensor()
    blueprint = SimpleNamespace(set_attribute=lambda name, value: None)
    library = SimpleNamespace(find=lambda name: blueprint)
    world = SimpleNamespace(
        get_blueprint_library=lambda: library,
        spawn_actor=lambda bp, transform, attach_to: sensor,
    )
    pipeline = _Pipeline(process)
    monkeypatch.setattr(camera_recorder.ffmpeg, "input", pipeline)
    recorder = CameraRecorder(
        world,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path / "video" / "run.mp4",
        image_width=2,
        image_height=1,
    )
    return recorder, sensor, pipeline


def test_a_frame_the_sensor_hands_over_is_written_as_bgr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _Process(0, b"")
    recorder, sensor, pipeline = _recorder(tmp_path, monkeypatch, process)
    assert (tmp_path / "video").is_dir()
    assert [name for name, _, _ in pipeline.calls] == [
        "input",
        "output",
        "overwrite_output",
        "run_async",
    ]
    assert pipeline.calls[1][1] == (str(tmp_path / "video" / "run.mp4"),)
    assert pipeline.calls[3][2] == {"pipe_stdin": True, "pipe_stderr": True}

    # CARLA's Python API hands the listener the image itself (no as_image()).
    bgra = bytes([1, 2, 3, 255, 4, 5, 6, 255])
    assert sensor.listener is not None
    sensor.listener(SimpleNamespace(raw_data=bgra, height=1, width=2))
    assert recorder.write_frame() is True

    # The writer thread writes it; stop() waits for every frame handed over.
    recorder.stop()
    assert process.stdin.getvalue() == bytes([1, 2, 3, 4, 5, 6])
    assert sensor.stopped and sensor.destroyed


def test_no_frame_is_written_when_the_queue_stays_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(camera_recorder, "_FRAME_TIMEOUT", 0.01)
    process = _Process(0, b"")
    recorder, _sensor, _pipeline = _recorder(tmp_path, monkeypatch, process)
    assert recorder.write_frame() is False
    assert process.stdin.getvalue() == b""
    recorder.stop()


def test_a_failed_ffmpeg_logs_its_decoded_error_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder, _sensor, _pipeline = _recorder(
        tmp_path, monkeypatch, _Process(1, b"lost\xffbroken" + b"x" * 493)
    )
    with caplog.at_level(logging.WARNING, logger=camera_recorder.__name__):
        recorder.stop()
    (record,) = [r for r in caplog.records if "ffmpeg exited" in r.getMessage()]
    # Decoded with the invalid byte replaced, then its last 500 characters.
    assert record.getMessage() == (
        "ffmpeg exited with code 1: \ufffdbroken" + "x" * 493
    )


def test_draw_caption_marks_only_the_bottom_of_the_frame() -> None:
    image = np.full((72, 128, 3), 128, dtype=np.uint8)
    camera_recorder.draw_caption(image, "Warm-up")
    changed = (image != 128).any(axis=2)
    assert changed[48:].any()  # the caption, along the bottom
    assert not changed[:24].any()  # the top is left as it was


def test_a_captioned_frame_is_written_with_its_caption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _Process(0, b"")
    recorder, sensor, _pipeline = _recorder(tmp_path, monkeypatch, process)
    assert sensor.listener is not None
    gray = bytes([128, 128, 128, 255]) * (72 * 128)
    sensor.listener(SimpleNamespace(raw_data=gray, height=72, width=128))
    assert recorder.write_frame("Warm-up") is True
    # Without a caption, the frame goes out as it came.
    sensor.listener(SimpleNamespace(raw_data=gray, height=72, width=128))
    assert recorder.write_frame() is True
    recorder.stop()

    size = 72 * 128 * 3
    written = np.frombuffer(process.stdin.getvalue(), dtype=np.uint8)
    assert written.size == 2 * size
    assert (written[:size] != 128).any()
    assert (written[size:] == 128).all()


def test_every_frame_handed_over_is_written_in_order_before_stop_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _Process(0, b"")
    recorder, sensor, _pipeline = _recorder(tmp_path, monkeypatch, process)
    assert sensor.listener is not None
    count = 50
    for value in range(count):
        sensor.listener(
            SimpleNamespace(raw_data=bytes([value, 0, 0, 255]) * 2, height=1, width=2)
        )
        assert recorder.write_frame() is True
    recorder.stop()

    written = np.frombuffer(process.stdin.getvalue(), dtype=np.uint8)
    assert list(written[::6]) == list(range(count))
    assert recorder._frame_count == count  # noqa: SLF001
    # ffmpeg's input is closed once, after the last frame was written.
    assert process.stdin.events == ["write"] * count + ["close"]


def test_a_broken_pipe_drops_the_rest_and_stop_still_waits_for_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    process = _Process(1, b"gone", _Stdin(accept=2))
    recorder, sensor, _pipeline = _recorder(tmp_path, monkeypatch, process)
    assert sensor.listener is not None
    with caplog.at_level(logging.WARNING, logger=camera_recorder.__name__):
        # More frames than the writer's queue holds: write_frame() must not
        # wait on a writer that stopped writing.
        for _ in range(3 * camera_recorder._PENDING_FRAMES):  # noqa: SLF001
            sensor.listener(SimpleNamespace(raw_data=bytes(8), height=1, width=2))
            recorder.write_frame()
        recorder.stop()
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count("CameraRecorder: ffmpeg pipe broken") == 1
    assert "ffmpeg exited with code 1: gone" in messages
    # The frames written before the pipe broke count; the rest are reported.
    assert recorder._frame_count == 2  # noqa: SLF001
    dropped = 3 * camera_recorder._PENDING_FRAMES - 2  # noqa: SLF001
    assert (
        f"CameraRecorder: {dropped} frame(s) were not written: "
        f"{tmp_path / 'video' / 'run.mp4'} is incomplete"
    ) in messages
    assert process.stdin.events == ["write", "write", "close"]
    # Nor is a frame handed over once the recorder stopped.
    sensor.listener(SimpleNamespace(raw_data=bytes(8), height=1, width=2))
    assert recorder.write_frame() is False


@pytest.mark.parametrize("size", [(0, 1080), (1920, -2), (961, 540), (960, 541)])
def test_a_video_size_libx264_cannot_encode_is_refused(
    size: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="positive and even"):
        camera_recorder.check_video_size(*size)


def test_stop_kills_an_ffmpeg_that_stopped_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(camera_recorder, "_STOP_TIMEOUT", 0.2)
    process = _Process(0, b"", _StuckStdin())
    recorder, sensor, _pipeline = _recorder(tmp_path, monkeypatch, process)
    assert sensor.listener is not None
    for _ in range(camera_recorder._PENDING_FRAMES + 1):  # noqa: SLF001
        sensor.listener(SimpleNamespace(raw_data=bytes(8), height=1, width=2))
        recorder.write_frame()
    with caplog.at_level(logging.WARNING, logger=camera_recorder.__name__):
        recorder.stop()  # returns: the stuck write ends with ffmpeg killed
    assert process.killed
    messages = [r.getMessage() for r in caplog.records]
    assert any("stopped reading frames" in m for m in messages)
    assert any("were not written" in m for m in messages)


def _world(sensor: _Sensor) -> SimpleNamespace:
    blueprint = SimpleNamespace(set_attribute=lambda name, value: None)
    library = SimpleNamespace(find=lambda name: blueprint)
    return SimpleNamespace(
        get_blueprint_library=lambda: library,
        spawn_actor=lambda bp, transform, attach_to: sensor,
    )


def test_a_constructor_that_fails_leaves_neither_ffmpeg_nor_the_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _DeafSensor(_Sensor):
        def listen(self, callback: Callable[[Any], None]) -> None:
            raise RuntimeError("listen failed")

    sensor = _DeafSensor()
    process = _Process(0, b"")
    monkeypatch.setattr(camera_recorder.ffmpeg, "input", _Pipeline(process))
    threads = threading.active_count()
    with pytest.raises(RuntimeError, match="listen failed"):
        CameraRecorder(
            _world(sensor),  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            tmp_path / "run.mp4",
        )
    assert process.killed
    assert sensor.stopped and sensor.destroyed
    assert threading.active_count() == threads  # no writer was started


def test_a_constructor_whose_ffmpeg_fails_destroys_the_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_ffmpeg(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("ffmpeg")

    sensor = _Sensor()
    monkeypatch.setattr(camera_recorder.ffmpeg, "input", no_ffmpeg)
    with pytest.raises(FileNotFoundError):
        CameraRecorder(
            _world(sensor),  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            tmp_path / "run.mp4",
        )
    assert sensor.destroyed
    assert sensor.listener is None
