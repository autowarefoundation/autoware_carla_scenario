"""A camera on the ROS 2 ego whose frames are published as they arrive.

Imported by :class:`~..entity.ros2_entity.Ros2EgoEntity` when it mounts a
camera in ``ros2.cameras``.  Outside the static check (docs/typecheck.md): its
listener hands each frame back to the entity from a CARLA sensor thread, which
Codon cannot type through ``CarlaCameraSensor``'s listen callback.
"""

from __future__ import annotations

from typing import Any

from ..sensor.carla_camera import CarlaCameraSensor, CarlaCameraSensorConfig

__all__ = ["PublishingCamera"]


class PublishingCamera(CarlaCameraSensor):
    """A CARLA RGB camera that hands each frame to its entity as it arrives.

    Nothing queues: the frame is published from the sensor's own thread, so a
    slow reader never holds frames back.

    Args:
        config: The camera.
        entity: The :class:`~..entity.ros2_entity.Ros2EgoEntity` publishing it.
        index: The camera's index in its ``ros2.cameras``.
    """

    def __init__(self, config: CarlaCameraSensorConfig, entity: Any, index: int) -> None:
        super().__init__(config)
        self._entity = entity
        self._index = index

    def _on_sensor_data(self, data: Any) -> None:
        # CARLA's Python API calls the listener with the carla.Image itself.
        self._entity._publish_frame(self._index, data)
