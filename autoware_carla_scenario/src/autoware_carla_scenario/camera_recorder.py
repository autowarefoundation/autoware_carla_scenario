"""RGB camera recorder that attaches to a CARLA actor and writes MP4 video.

Frames are captured synchronously (one per ``world.tick()``) via a
:class:`queue.Queue` and streamed to an *ffmpeg* process through
*ffmpeg-python*, producing a browser-compatible H.264 MP4 file.  A writer
thread of the recorder's own converts each frame and pipes it to ffmpeg, so
encoding overlaps the ticks that render the next frames.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import typesafe_carla.carla as carla
import ffmpeg
import numpy as np

logger = logging.getLogger(__name__)

#: Default image width in pixels.
DEFAULT_IMAGE_WIDTH: int = 1920
#: Default image height in pixels.
DEFAULT_IMAGE_HEIGHT: int = 1080
#: Default camera horizontal field of view in degrees.
DEFAULT_FOV: float = 90.0
#: Default recording frame rate (matches CARLA synchronous mode at 20 Hz).
DEFAULT_FPS: float = 20.0

#: Timeout (seconds) for waiting on a single frame from the sensor queue.
#: Set slightly above the expected frame interval (50 ms at 20 Hz) to
#: avoid hanging on a dead sensor.
_FRAME_TIMEOUT: float = 1.0

#: How many frames may wait for the writer thread before write_frame() blocks:
#: enough to ride out a slow frame, few enough to bound the memory (a 1080p
#: frame is 8 MB).
_PENDING_FRAMES: int = 4

#: Seconds stop() waits for the writer to write what it was handed before it
#: takes ffmpeg for stuck (alive, but no longer reading) and kills it.
_STOP_TIMEOUT: float = 30.0


def check_video_size(width: int, height: int) -> None:
    """Refuse a video size the recorder cannot encode.

    H.264 in yuv420p, the pixel format browsers play, halves the chroma in
    both directions, so libx264 takes even sizes only.

    Raises:
        ValueError: when *width* or *height* is not a positive even number.
    """
    if width <= 0 or height <= 0 or width % 2 != 0 or height % 2 != 0:
        raise ValueError(
            f"The video size must be positive and even, got {width}x{height}"
        )


def draw_caption(
    image_bgr: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]], text: str
) -> None:
    """Draw *text* centred along the bottom of *image_bgr*, in place.

    White on a black outline, sized to the image's height, so it reads on any
    background.
    """
    import cv2  # noqa: PLC0415

    height, width = image_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = height / 720.0
    thickness = max(1, round(2 * scale))
    (text_width, _), baseline = cv2.getTextSize(text, font, scale, thickness)
    origin = ((width - text_width) // 2, height - baseline - round(24 * scale))
    cv2.putText(
        image_bgr, text, origin, font, scale, (0, 0, 0), thickness * 3, cv2.LINE_AA
    )
    cv2.putText(
        image_bgr, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


class CameraRecorder:
    """Attach an RGB camera sensor to a CARLA actor and record video to MP4.

    The camera is placed at the given offset relative to the actor's local
    frame, producing the same viewpoint as the spectator camera when
    configured with matching *offset_back*, *offset_up*, and *pitch*.

    Frames are retrieved **synchronously** — call :meth:`write_frame` after
    each ``world.tick()`` to pull the latest frame from the sensor queue and
    stream it to ffmpeg for H.264 encoding.  The frame is handed to a writer
    thread, which converts it, captions it and pipes it to ffmpeg in the
    order the frames came, while the caller ticks on; :meth:`stop` waits for every
    frame handed over to be written.

    Args:
        world: The CARLA world instance.
        actor: The actor to attach the camera to (typically the ego vehicle).
        output_path: Destination path for the output ``.mp4`` file.
        offset_back: Distance behind the actor in metres.
        offset_up: Height above the actor in metres.
        pitch: Camera pitch angle in degrees (negative = look down).
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        fov: Horizontal field of view in degrees.
        fps: Recording frame rate.
    """

    _output_path: Path
    _frame_count: int
    _frame_queue: queue.Queue[carla.Image]
    #: The frames write_frame() handed over, each with its caption, for the
    #: writer thread; None tells it to finish.
    _pending: queue.Queue[Optional[tuple[carla.Image, str]]]
    _writer: Optional[threading.Thread]
    #: Whether ffmpeg stopped taking frames: the writer drops the rest.
    _pipe_failed: bool
    #: Frames taken from the camera but not written, after a failure.
    _dropped: int
    _sensor: Optional[carla.Actor]
    _process: Optional[subprocess.Popen[bytes]]

    def __init__(
        self,
        world: "carla.World",
        actor: "carla.Actor",
        output_path: Path,
        *,
        offset_back: float = 8.0,
        offset_up: float = 5.0,
        pitch: float = -15.0,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        fov: float = DEFAULT_FOV,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self._output_path = output_path
        self._frame_count = 0
        # Bounds memory if write_frame() falls behind: room for the frames
        # a caller lets the ticks run ahead of it, and one more arriving.
        self._frame_queue = queue.Queue(maxsize=4)
        self._pending = queue.Queue(maxsize=_PENDING_FRAMES)
        self._writer = None
        self._pipe_failed = False
        self._dropped = 0
        self._process = None

        # Set up the RGB camera blueprint
        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(image_width))
        camera_bp.set_attribute("image_size_y", str(image_height))
        camera_bp.set_attribute("fov", str(fov))

        # Place the camera at the same relative position as the spectator.
        # In the actor's local frame: -x = behind, +z = above.
        transform = carla.Transform(
            carla.Location(x=-offset_back, y=0.0, z=offset_up),
            carla.Rotation(pitch=pitch),
        )
        self._sensor = world.spawn_actor(camera_bp, transform, attach_to=actor)

        try:
            self._start_ffmpeg(output_path, image_width, image_height, fps)
            # Enqueue frames instead of processing them in the callback thread.
            self._sensor.listen(self._on_sensor_data)
        except Exception:
            # Leave neither the camera nor ffmpeg behind.
            self._discard()
            raise
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        logger.info(
            "CameraRecorder started: %dx%d @ %.0f fps -> %s",
            image_width,
            image_height,
            fps,
            output_path,
        )

    def _start_ffmpeg(
        self, output_path: Path, image_width: int, image_height: int, fps: float
    ) -> None:
        """Start ffmpeg via ffmpeg-python for H.264 streaming encoding."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{image_width}x{image_height}",
                r=fps,
            )
            .output(
                str(output_path),
                vcodec="libx264",
                pix_fmt="yuv420p",
                preset="fast",
                crf=23,
                # Errors only: stderr is read once ffmpeg has exited, so the
                # progress it would print meanwhile could fill the pipe and
                # stall it on a long recording.
                loglevel="error",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True)
        )

    def _discard(self) -> None:
        """Destroy the camera and kill ffmpeg, for a constructor that failed."""
        sensor = self._sensor
        self._sensor = None
        if sensor is not None:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                logger.warning("CameraRecorder: destroying the camera failed")
        process = self._process
        self._process = None
        if process is not None:
            process.kill()
            process.wait()

    def _on_sensor_data(self, data: Union["carla.Image", "carla.SensorData"]) -> None:
        """The sensor's listener: queues the frame for :meth:`write_frame`.

        It runs on a CARLA thread, and blocks while the queue is full.  CARLA's
        Python API, and typesafe_carla's CPython package, call it with the
        ``carla.Image`` itself.  typesafe_carla's Codon library calls it with
        the ``carla.SensorData``, which ``as_image()`` turns into the image; the
        static check sees that branch, and Python, whose ``Image`` has no
        ``as_image``, never takes it.
        """
        if hasattr(data, "as_image"):
            self._frame_queue.put(data.as_image())
        else:
            self._frame_queue.put(data)

    def write_frame(self, caption: str = "") -> bool:
        """Pull one frame from the sensor queue and hand it to the writer.

        Call this once per ``world.tick()``: right after it, or a tick or two
        later, which leaves the frame time to arrive meanwhile (the sensor
        queue holds up to four).  The writer thread converts the frame and pipes it to ffmpeg;
        this blocks only while :data:`_PENDING_FRAMES` frames are still
        waiting for it.

        Args:
            caption: Text drawn along the frame's bottom (:func:`draw_caption`);
                none when empty.

        Returns:
            ``True`` if a frame was handed over, ``False`` on timeout or if
            the ffmpeg process is no longer accepting input.
        """
        try:
            image: "carla.Image" = self._frame_queue.get(timeout=_FRAME_TIMEOUT)
        except queue.Empty:
            logger.debug("CameraRecorder: frame queue timeout")
            return False
        if self._pipe_failed or self._writer is None:
            self._dropped += 1
            return False
        self._pending.put((image, caption))
        return True

    def _write_loop(self) -> None:
        """The writer thread: write the frames handed over until told to stop.

        After ffmpeg stops taking frames the rest are taken and dropped, so
        :meth:`write_frame` never waits on a writer that does not write.
        """
        while True:
            item = self._pending.get()
            if item is None:
                return
            if self._pipe_failed:
                self._dropped += 1
                continue
            image, caption = item
            try:
                self._write(image, caption)
            except BrokenPipeError:
                logger.warning("CameraRecorder: ffmpeg pipe broken")
                self._pipe_failed = True
                self._dropped += 1
            except Exception:
                logger.warning("CameraRecorder: writing a frame failed", exc_info=True)
                self._pipe_failed = True
                self._dropped += 1

    def _write(self, image: "carla.Image", caption: str) -> None:
        """Convert *image* from BGRA to BGR, caption it and pipe it to ffmpeg."""
        if TYPE_CHECKING:
            # typesafe_carla's Codon raw_data is a view, and Codon's frombuffer
            # takes the bytes (a str).
            array = np.frombuffer(image.raw_data.tobytes(), dtype=np.uint8)
        else:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        bgr = np.ascontiguousarray(array[:, :, :3])
        if caption:
            if TYPE_CHECKING:
                # draw_caption draws with cv2, which is outside the static check.
                pass
            else:
                draw_caption(bgr, caption)

        process = self._process
        if process is None or process.stdin is None:
            self._pipe_failed = True
            self._dropped += 1
            return
        process.stdin.write(bgr.data)
        self._frame_count += 1

    def stop(self) -> None:
        """Stop the sensor, finalise the ffmpeg process, and destroy the sensor actor.

        Every frame :meth:`write_frame` handed over is written before ffmpeg's
        input is closed.  An ffmpeg that is alive but stopped reading for
        :data:`_STOP_TIMEOUT` seconds is killed, which ends the write it
        blocks; frames that were not written are logged as an error.
        """
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None

        if self._writer is not None:
            writer: threading.Thread = self._writer
            self._finish_writer(writer)
            if writer.is_alive():
                logger.error(
                    "CameraRecorder: ffmpeg stopped reading frames for %.0f s: "
                    "killing it",
                    _STOP_TIMEOUT,
                )
                self._pipe_failed = True
                if self._process is not None:
                    self._process.kill()
                self._finish_writer(writer)
            self._writer = None

        if self._process is not None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
            except BrokenPipeError:
                pass
            self._process.wait()
            if self._process.returncode != 0:
                stderr = b""
                if self._process.stderr is not None:
                    stderr = self._process.stderr.read()
                if TYPE_CHECKING:
                    # The static check reads bytes as str, which has no decode().
                    detail = stderr
                else:
                    detail = stderr.decode(errors="replace")
                logger.warning(
                    "ffmpeg exited with code %d: %s",
                    self._process.returncode,
                    detail[-500:],
                )
            self._process = None

        if self._dropped > 0:
            logger.error(
                "CameraRecorder: %d frame(s) were not written: %s is incomplete",
                self._dropped,
                self._output_path,
            )
        logger.info(
            "CameraRecorder stopped: %d frames written to %s",
            self._frame_count,
            self._output_path,
        )

    def _finish_writer(self, writer: threading.Thread) -> None:
        """Tell *writer* to finish, and wait up to :data:`_STOP_TIMEOUT` for it."""
        try:
            self._pending.put(None, timeout=_STOP_TIMEOUT)
        except queue.Full:
            return  # stuck on a write: the caller kills ffmpeg and asks again
        writer.join(timeout=_STOP_TIMEOUT)
