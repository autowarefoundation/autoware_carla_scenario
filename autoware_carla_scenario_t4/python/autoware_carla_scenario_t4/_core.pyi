from os import PathLike
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

__version__: str

class T4Writer:
    def __init__(
        self,
        data_root: str | PathLike[str],
        *,
        scene_name: str,
        scene_description: str = "",
        logfile: str = "",
        vehicle: str = "",
        date_captured: str = "",
        location: str = "",
        token_seed: Optional[str] = None,
        map_offset: Sequence[float] = (0.0, 0.0, 0.0),
        rear_axle_offset_m: float = 0.0,
        max_distance_m: Optional[float] = None,
        jpeg_quality: int = 90,
        workers: int = 2,
        queue: int = 16,
    ) -> None: ...
    @property
    def data_root(self) -> Path: ...
    @property
    def frames(self) -> int: ...
    def add_lanelet2_map(self, path: str | PathLike[str]) -> None: ...
    def add_sensor(
        self,
        channel: str,
        modality: str,
        translation: Sequence[float],
        rotation: Sequence[float],
        *,
        camera_intrinsic: Optional[Sequence[Sequence[float]]] = None,
        camera_distortion: Optional[Sequence[float]] = None,
    ) -> None: ...
    def begin_frame(
        self,
        timestamp_us: int,
        translation: Sequence[float],
        rotation: Sequence[float],
        *,
        twist: Optional[Sequence[float]] = None,
        acceleration: Optional[Sequence[float]] = None,
    ) -> None: ...
    def add_annotation(
        self,
        instance_id: int,
        category: str,
        translation: Sequence[float],
        size: Sequence[float],
        rotation: Sequence[float],
        *,
        velocity: Optional[Sequence[float]] = None,
        acceleration: Optional[Sequence[float]] = None,
        attributes: Sequence[str] = ...,
        instance_name: Optional[str] = None,
    ) -> None: ...
    def register_actor(
        self,
        actor_id: int,
        category: str,
        center: Sequence[float],
        size: Sequence[float],
        name: Optional[str] = None,
    ) -> None: ...
    def knows_actor(self, actor_id: int) -> bool: ...
    def begin_frame_carla(
        self,
        timestamp_us: int,
        transform: Sequence[float],
        velocity: Sequence[float],
        angular_velocity: Sequence[float],
        acceleration: Sequence[float],
    ) -> None: ...
    def add_carla_boxes(
        self, ids: npt.NDArray[np.uint64], states: npt.NDArray[np.float64]
    ) -> int: ...
    def add_camera_jpeg(
        self, channel: str, jpeg: bytes, width: int, height: int
    ) -> None: ...
    def add_lidar(self, channel: str, points: npt.NDArray[np.float32]) -> None: ...
    def add_camera(
        self, channel: str, image: npt.NDArray[np.uint8], pixel_format: str = "bgra"
    ) -> None: ...
    def add_vehicle_state(
        self,
        *,
        accel_pedal: Optional[float] = None,
        brake_pedal: Optional[float] = None,
        steer_pedal: Optional[float] = None,
        steering_tire_angle: Optional[float] = None,
        steering_wheel_angle: Optional[float] = None,
        shift_state: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> None: ...
    def end_frame(self) -> None: ...
    def finish(self) -> dict[str, int | str]: ...
