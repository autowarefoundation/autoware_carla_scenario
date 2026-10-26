"""Concrete CARLA RGB camera sensor implementation.

Extends :class:`CameraSensorBase` with CARLA-specific attributes such as
post-processing effects (bloom, motion blur, lens flare) and exposure
control.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from .base import CameraSensorBase, CameraSensorConfig

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

#: Timeout (seconds) for waiting on a single frame from the sensor queue.
_FRAME_TIMEOUT: float = 1.0


@dataclass(frozen=True)
class CarlaCameraSensorConfig(CameraSensorConfig):
    """CARLA ``sensor.camera.rgb`` configuration.

    Inherits all fields from :class:`CameraSensorConfig` (resolution, FOV,
    extrinsics, etc.) and adds attributes specific to CARLA's RGB camera
    blueprint.

    See `CARLA sensor reference
    <https://carla.readthedocs.io/en/latest/ref_sensors/#rgb-camera>`_
    for full attribute documentation.
    """

    # -- CARLA blueprint selection --------------------------------------------
    sensor_type: str = "sensor.camera.rgb"
    """CARLA blueprint name for the camera sensor."""

    # -- Blueprint attributes ---------------------------------------------------
    # Each one defaults to ``None``: the attribute is left at the blueprint's own
    # default, which is the one tuned for the server's renderer. The defaults
    # differ between CARLA releases -- 0.10 (UE5) exposes for ``iso=300000``,
    # ``fstop=9.8`` and an auto-exposure range of 0..20, where 0.9.x (UE4) used
    # ``iso=100``, ``fstop=1.4`` and 7..9 -- so pinning one release's values
    # renders the other's images wrong (0.9.x values make 0.10 frames black).
    # Set a value only to override the server.

    # -- Post-processing effects ----------------------------------------------
    bloom_intensity: Optional[float] = None
    """Intensity of the bloom post-processing effect (0.0 to 1.0)."""

    fstop: Optional[float] = None
    """Simulated f-stop number controlling depth of field."""

    iso: Optional[float] = None
    """Simulated film ISO sensitivity."""

    gamma: Optional[float] = None
    """Gamma correction value applied to the output image."""

    lens_flare_intensity: Optional[float] = None
    """Intensity of the lens flare effect (0.0 to 1.0)."""

    motion_blur_intensity: Optional[float] = None
    """Intensity of motion blur (0.0 to 1.0)."""

    motion_blur_max_distortion: Optional[float] = None
    """Maximum distortion percentage from motion blur."""

    motion_blur_min_object_screen_size: Optional[float] = None
    """Minimum screen-space fraction for an object to trigger motion blur."""

    # -- Exposure control -----------------------------------------------------
    exposure_mode: Optional[str] = None
    """Exposure mode: ``"histogram"`` (auto) or ``"manual"``."""

    exposure_compensation: Optional[float] = None
    """Logarithmic exposure compensation (EV)."""

    exposure_min_bright: Optional[float] = None
    """Minimum brightness for auto-exposure (histogram mode)."""

    exposure_max_bright: Optional[float] = None
    """Maximum brightness for auto-exposure (histogram mode)."""

    exposure_speed_up: Optional[float] = None
    """Speed of adaptation when scene brightens."""

    exposure_speed_down: Optional[float] = None
    """Speed of adaptation when scene darkens."""

    # -- Chromatic aberration -------------------------------------------------
    chromatic_aberration_intensity: Optional[float] = None
    """Intensity of chromatic aberration fringing (0.0 = disabled)."""

    chromatic_aberration_offset: Optional[float] = None
    """Offset applied to chromatic aberration channels."""

    # -- Lens distortion ------------------------------------------------------
    lens_circle_falloff: Optional[float] = None
    """Vignette falloff factor (higher = sharper falloff)."""

    lens_circle_multiplier: Optional[float] = None
    """Vignette radius multiplier (0.0 = disabled)."""

    lens_k: Optional[float] = None
    """Radial distortion coefficient k (negative = barrel distortion)."""

    lens_kcube: Optional[float] = None
    """Cubic radial distortion coefficient."""

    lens_x_size: Optional[float] = None
    """Horizontal size of the lens distortion grid."""

    lens_y_size: Optional[float] = None
    """Vertical size of the lens distortion grid."""


#: The :class:`CarlaCameraSensorConfig` fields that are ``sensor.camera.rgb``
#: attributes of the same name, set on the blueprint when not ``None``.
BLUEPRINT_ATTRIBUTES: tuple[str, ...] = (
    "bloom_intensity",
    "fstop",
    "iso",
    "gamma",
    "lens_flare_intensity",
    "motion_blur_intensity",
    "motion_blur_max_distortion",
    "motion_blur_min_object_screen_size",
    "exposure_mode",
    "exposure_compensation",
    "exposure_min_bright",
    "exposure_max_bright",
    "exposure_speed_up",
    "exposure_speed_down",
    "chromatic_aberration_intensity",
    "chromatic_aberration_offset",
    "lens_circle_falloff",
    "lens_circle_multiplier",
    "lens_k",
    "lens_kcube",
    "lens_x_size",
    "lens_y_size",
)


class CarlaCameraSensor(CameraSensorBase):
    """CARLA ``sensor.camera.rgb`` sensor implementation.

    Spawns the CARLA RGB camera blueprint with attributes driven by a
    :class:`CarlaCameraSensorConfig`.  Frames are delivered asynchronously
    via ``sensor.listen()`` into a :class:`queue.Queue` and retrieved with
    :meth:`get_image`.

    Args:
        config: CARLA camera sensor configuration.
        on_frame: Called with each frame from the sensor's own thread, instead
            of queueing it: :meth:`get_image` then has nothing to return.
            ``None`` (the default) queues the frames.
    """

    _carla_config: CarlaCameraSensorConfig
    _sensor: Optional[carla.Actor]
    _frame_queue: queue.Queue[carla.Image]
    _on_frame: Optional[Callable[[carla.Image], None]]

    def __init__(
        self,
        config: CarlaCameraSensorConfig,
        on_frame: Optional[Callable[[carla.Image], None]] = None,
    ) -> None:
        super().__init__(config)
        self._carla_config = config
        self._on_frame = on_frame
        self._sensor = None
        self._frame_queue = queue.Queue(maxsize=2)

    # ------------------------------------------------------------------
    # CameraSensorBase interface
    # ------------------------------------------------------------------

    def attach(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Spawn and attach the CARLA RGB camera to *actor*.

        Args:
            world: The CARLA world instance.
            actor: The actor to attach the camera to.

        Raises:
            RuntimeError: If the sensor is already attached.
        """
        if self._attached:
            raise RuntimeError("CarlaCameraSensor is already attached")

        import typesafe_carla.carla as _carla

        cfg = self._carla_config
        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find(cfg.sensor_type)

        # -- Resolution & optics
        camera_bp.set_attribute("image_size_x", str(cfg.image_width))
        camera_bp.set_attribute("image_size_y", str(cfg.image_height))
        camera_bp.set_attribute("fov", str(cfg.fov))
        camera_bp.set_attribute("sensor_tick", str(1.0 / cfg.fps))

        # -- Post-processing, exposure, lens: only what the config overrides, and
        # only what this server's camera has (CARLA versions add and drop them)
        if TYPE_CHECKING:
            # Codon has no getattr() of a name known only at run time: the
            # BLUEPRINT_ATTRIBUTES, in their order, spelled out.
            self._override(camera_bp, "bloom_intensity", cfg.bloom_intensity)
            self._override(camera_bp, "fstop", cfg.fstop)
            self._override(camera_bp, "iso", cfg.iso)
            self._override(camera_bp, "gamma", cfg.gamma)
            self._override(camera_bp, "lens_flare_intensity", cfg.lens_flare_intensity)
            self._override(
                camera_bp, "motion_blur_intensity", cfg.motion_blur_intensity
            )
            self._override(
                camera_bp, "motion_blur_max_distortion", cfg.motion_blur_max_distortion
            )
            self._override(
                camera_bp,
                "motion_blur_min_object_screen_size",
                cfg.motion_blur_min_object_screen_size,
            )
            self._override(camera_bp, "exposure_mode", cfg.exposure_mode)
            self._override(
                camera_bp, "exposure_compensation", cfg.exposure_compensation
            )
            self._override(camera_bp, "exposure_min_bright", cfg.exposure_min_bright)
            self._override(camera_bp, "exposure_max_bright", cfg.exposure_max_bright)
            self._override(camera_bp, "exposure_speed_up", cfg.exposure_speed_up)
            self._override(camera_bp, "exposure_speed_down", cfg.exposure_speed_down)
            self._override(
                camera_bp,
                "chromatic_aberration_intensity",
                cfg.chromatic_aberration_intensity,
            )
            self._override(
                camera_bp,
                "chromatic_aberration_offset",
                cfg.chromatic_aberration_offset,
            )
            self._override(camera_bp, "lens_circle_falloff", cfg.lens_circle_falloff)
            self._override(
                camera_bp, "lens_circle_multiplier", cfg.lens_circle_multiplier
            )
            self._override(camera_bp, "lens_k", cfg.lens_k)
            self._override(camera_bp, "lens_kcube", cfg.lens_kcube)
            self._override(camera_bp, "lens_x_size", cfg.lens_x_size)
            self._override(camera_bp, "lens_y_size", cfg.lens_y_size)
        else:
            for name in BLUEPRINT_ATTRIBUTES:
                self._override(camera_bp, name, getattr(cfg, name))

        # -- Transform (base_link -> camera)
        transform = _carla.Transform(
            _carla.Location(x=cfg.position_x, y=cfg.position_y, z=cfg.position_z),
            _carla.Rotation(roll=cfg.roll, pitch=cfg.pitch, yaw=cfg.yaw),
        )

        self._sensor = world.spawn_actor(camera_bp, transform, attach_to=actor)
        self._sensor.listen(self._on_sensor_data)
        self._attached = True

        logger.info(
            "CarlaCameraSensor attached: %dx%d fov=%.1f @ %.0f fps -> %s",
            cfg.image_width,
            cfg.image_height,
            cfg.fov,
            cfg.fps,
            actor.type_id,
        )

    def _override(
        self, camera_bp: "carla.ActorBlueprint", name: str, value: Optional[float | str]
    ) -> None:
        """Set the blueprint attribute *name* to *value*, unless it is ``None``.

        An attribute this server's camera lacks is skipped, with a warning.
        """
        if value is None:
            return
        if not camera_bp.has_attribute(name):
            logger.warning(
                "%s has no attribute %r on this CARLA server; %s=%s is ignored",
                self._carla_config.sensor_type,
                name,
                name,
                value,
            )
            return
        camera_bp.set_attribute(name, str(value))

    def _on_sensor_data(self, data: Union["carla.Image", "carla.SensorData"]) -> None:
        """The sensor's listener: queues the frame for :meth:`get_image`.

        With an ``on_frame`` callback the frame goes to it instead, unqueued.

        It runs on a CARLA thread, which delivers every sensor's data, so it must
        never block: when the queue is full it drops the oldest frame.  Once the
        policy stops calling :meth:`get_image` (a finished scenario), a blocking
        ``put`` would hang that thread for good, and no sensor attached after it
        would get a frame.  CARLA's
        Python API, and typesafe_carla's CPython package, call it with the
        ``carla.Image`` itself.  typesafe_carla's Codon library calls it with
        the ``carla.SensorData``, which ``as_image()`` turns into the image; the
        static check sees that branch, and Python, whose ``Image`` has no
        ``as_image``, never takes it.
        """
        image = data.as_image() if hasattr(data, "as_image") else data
        on_frame = self._on_frame
        if on_frame is not None:
            on_frame(image)
            return
        # Never block CARLA's delivery thread: drop the oldest frame when full.
        while True:
            try:
                self._frame_queue.put_nowait(image)
                return
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

    def destroy(self) -> None:
        """Stop and destroy the CARLA sensor actor."""
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None
        self._attached = False
        logger.info("CarlaCameraSensor destroyed")

    def get_image(
        self,
    ) -> Optional[np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]:
        """Return the *newest* captured frame as an HxWx3 BGR NumPy array.

        Frames arrive asynchronously into a bounded FIFO, and at the default
        20 Hz capture against a 10 Hz policy two frames land per policy step.
        Taking only the oldest each time would leave the queue permanently full:
        the ``listen`` callback then blocks on ``put``, frames back up, and the
        caller is fed steadily staler images.  So the queue is drained to its most
        recent entry and the older frames are dropped rather than delivered late.

        Blocks for up to :data:`_FRAME_TIMEOUT` seconds for the first frame, then
        takes whatever else is already queued without waiting.  Returns ``None``
        on timeout.
        """
        try:
            image: carla.Image = self._frame_queue.get(timeout=_FRAME_TIMEOUT)
        except queue.Empty:
            return None

        while True:
            try:
                image = self._frame_queue.get_nowait()
            except queue.Empty:
                break

        if TYPE_CHECKING:
            # typesafe_carla's Codon raw_data is a view, and Codon's frombuffer
            # takes the bytes (a str).
            array = np.frombuffer(image.raw_data.tobytes(), dtype=np.uint8)
        else:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        return np.ascontiguousarray(array[:, :, :3])  # BGR

    # ------------------------------------------------------------------
    # CARLA-specific helpers
    # ------------------------------------------------------------------

    @property
    def sensor_actor(self) -> Optional["carla.Actor"]:
        """Return the underlying CARLA sensor actor, or ``None``."""
        return self._sensor
