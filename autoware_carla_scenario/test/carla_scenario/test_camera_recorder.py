"""Unit tests for CameraRecorder, with a fake world and a fake ffmpeg."""

from __future__ import annotations

import io
import logging
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


class _Process:
    def __init__(self, returncode: int, stderr: bytes) -> None:
        self.stdin = io.BytesIO()
        self.stdin.close = lambda: None  # type: ignore[method-assign]
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


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


class _BrokenStdin(io.BytesIO):
    def write(self, data: Any) -> int:
        raise BrokenPipeError


def test_a_broken_pipe_drops_the_rest_and_stop_still_waits_for_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    process = _Process(1, b"gone")
    process.stdin = _BrokenStdin()
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
    assert recorder._frame_count == 0  # noqa: SLF001
    # Nor is a frame handed over once the recorder stopped.
    sensor.listener(SimpleNamespace(raw_data=bytes(8), height=1, width=2))
    assert recorder.write_frame() is False


@pytest.mark.parametrize("size", [(0, 1080), (1920, -2), (961, 540), (960, 541)])
def test_a_video_size_libx264_cannot_encode_is_refused(
    size: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="positive and even"):
        camera_recorder.check_video_size(*size)
