"""Concrete CARLA RGB camera sensor implementation.

Extends :class:`CameraSensorBase` with CARLA-specific attributes such as
post-processing effects (bloom, motion blur, lens flare) and exposure
control.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

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
    """

    def __init__(self, config: CarlaCameraSensorConfig) -> None:
        super().__init__(config)
        self._carla_config = config
        self._sensor: Optional[carla.Actor] = None
        self._frame_queue: queue.Queue[carla.Image] = queue.Queue(maxsize=2)

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
        for name in BLUEPRINT_ATTRIBUTES:
            value = getattr(cfg, name)
            if value is None:
                continue
            if not camera_bp.has_attribute(name):
                logger.warning(
                    "%s has no attribute %r on this CARLA server; %s=%s is ignored",
                    cfg.sensor_type,
                    name,
                    name,
                    value,
                )
                continue
            camera_bp.set_attribute(name, str(value))

        # -- Transform (base_link -> camera)
        transform = _carla.Transform(
            _carla.Location(x=cfg.position_x, y=cfg.position_y, z=cfg.position_z),
            _carla.Rotation(roll=cfg.roll, pitch=cfg.pitch, yaw=cfg.yaw),
        )

        self._sensor = world.spawn_actor(camera_bp, transform, attach_to=actor)
        self._sensor.listen(self._frame_queue.put)
        self._attached = True

        logger.info(
            "CarlaCameraSensor attached: %dx%d fov=%.1f @ %.0f fps -> %s",
            cfg.image_width,
            cfg.image_height,
            cfg.fov,
            cfg.fps,
            actor.type_id,
        )

    def destroy(self) -> None:
        """Stop and destroy the CARLA sensor actor."""
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None
        self._attached = False
        logger.info("CarlaCameraSensor destroyed")

    def get_image(self) -> Optional[NDArray[np.uint8]]:
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
