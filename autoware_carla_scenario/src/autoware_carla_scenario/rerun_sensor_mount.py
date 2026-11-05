"""The typed core of the sensors in a run's Rerun recording: a camera's mount
on the ego, the colours of a LiDAR sweep, and which entities of a recording are
the sensors'.

:class:`CameraMount` places a camera of ``driver.cameras`` in the recording's
right-handed frame and gives its focal length; :func:`_height_colours` colours
a sweep's points by height; :data:`SENSOR_ENTITIES` and
:func:`_is_sensor_entity` tell the sensors' entities from the others; and
:func:`chunk_entities` reads the entities of a recording's chunks from
``rerun rrd print``.  rerun_lidar re-exports every name and keeps what needs
the Rerun SDK (held as ``Any``) and its CLI; this module is pure numpy and
``re``, so the library check compiles it (``docs/typecheck.md``, "Checking
the framework itself", #1688).
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "SENSOR_ENTITIES",
    "CameraMount",
    "chunk_entities",
]

#: The entities the sensors are recorded under: the LiDARs and the cameras.
SENSOR_ENTITIES = ("world/lidar", "world/ego/pose/cameras")

#: Heights (m, above the ego's rig) the colours run over, blue to red.
_COLOUR_LOW_M = -2.5
_COLOUR_HIGH_M = 4.0

#: An entity of a chunk, as ``rerun rrd print`` names it.
_CHUNK_LINE = re.compile(r"^Chunk\([^)]*\) with [^-]* - (/.*?)(?: - data columns:.*)?$")


def _is_sensor_entity(entity: str) -> bool:
    path = entity.strip("/")
    return any(path == root or path.startswith(root + "/") for root in SENSOR_ENTITIES)


def chunk_entities(
    printed: str, line: re.Pattern[str] = _CHUNK_LINE
) -> tuple[list[str], str]:
    """The entities of the chunks ``rerun rrd print`` lists in *printed*, one
    per chunk line (a line that starts with ``Chunk(``), in order, read with
    *line* (its group 1), and the first chunk line *line* does not read.

    That line is ``""`` when every chunk line is read (a chunk line is never
    empty), and the list is empty when there is no chunk line: Codon refuses
    an ``Optional[str]`` inside a returned tuple.  The entities are read up to
    the first line that is not.
    """
    entities: list[str] = []
    for chunk in printed.splitlines():
        if not chunk.startswith("Chunk("):
            continue
        match = line.match(chunk)
        if match is None:
            return entities, chunk
        # str(): Codon's group() is an Optional[str]; group 1 always matched.
        entities.append(str(match.group(1)))
    return entities, ""


class CameraMount:
    """A camera's mount on the ego, as ``driver.cameras`` gives it (CARLA's
    convention: x forward, y right, z up, degrees), and its image."""

    width: int
    height: int
    fov: float
    position: tuple[float, float, float]
    roll: float
    pitch: float
    yaw: float

    def __init__(
        self,
        width: int,
        height: int,
        fov: float,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fov = float(fov)
        self.position = position
        self.roll, self.pitch, self.yaw = rotation_deg

    def translation(self) -> list[float]:
        """In the recording's right-handed frame: CARLA's ``y`` negated."""
        x, y, z = self.position
        return [float(x), -float(y), float(z)]

    def rotation(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """CARLA's ``yaw * pitch * roll``, rebuilt right-handed: yaw and pitch
        negated, roll kept (as the driver's camera extrinsics)."""
        yaw, pitch, roll = (
            np.radians(-self.yaw),
            np.radians(-self.pitch),
            np.radians(self.roll),
        )
        cz, sz, cy, sy, cx, sx = (
            np.cos(yaw),
            np.sin(yaw),
            np.cos(pitch),
            np.sin(pitch),
            np.cos(roll),
            np.sin(roll),
        )
        rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        return rz @ ry @ rx

    def focal_length_px(self) -> float:
        """CARLA's horizontal field of view as a focal length in pixels."""
        return self.width / (2.0 * float(np.tan(np.radians(self.fov) / 2.0)))


def _height_colours(
    heights: np.ndarray[tuple[int], np.dtype[np.float32]],
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """Blue (low) through green to red (high), over the ego's surroundings."""
    t = np.clip((heights - _COLOUR_LOW_M) / (_COLOUR_HIGH_M - _COLOUR_LOW_M), 0.0, 1.0)
    colours = np.empty((len(t), 3), dtype=np.uint8)
    colours[:, 0] = (255 * np.clip(2.0 * t - 1.0, 0.0, 1.0)).astype(np.uint8)
    colours[:, 1] = (255 * (1.0 - np.abs(2.0 * t - 1.0))).astype(np.uint8)
    colours[:, 2] = (255 * np.clip(1.0 - 2.0 * t, 0.0, 1.0)).astype(np.uint8)
    return colours
