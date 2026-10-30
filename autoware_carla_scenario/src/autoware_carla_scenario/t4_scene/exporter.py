"""Each run's trajectory log as a T4Scene, as the run ends (``t4scene.enabled``).

The runner hands every closed trajectory log to a :class:`T4SceneExporter`,
which writes ``{stem}.t4scene.pb`` and ``{stem}.t4scene.map.pb`` beside it
(docs/t4_scene.md).  The conversion reads only the log and the map files, so
it runs on the runner's thread once CARLA is done with the case; the map is
read and encoded once however many cases ran on it.

Like the trajectory log and the T4 dataset, it is best effort: a log that
cannot be converted is logged and left as it is, it never fails the run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .convert import write_t4_scene

__all__ = ["T4SceneExporter"]

logger = logging.getLogger(__name__)


class T4SceneExporter:
    """Writes each run's trajectory log as a T4Scene.

    Args:
        step_s: The scenes' time step (``t4scene.step_s``).
        max_distance_m: The road users within this distance of the ego are a
            scene's objects; ``None`` takes all of them
            (``t4scene.max_distance_m``).
    """

    def __init__(
        self, *, step_s: float = 0.1, max_distance_m: Optional[float] = 100.0
    ) -> None:
        self.step_s = step_s
        self.max_distance_m = max_distance_m

    def export(self, trajectory: Path) -> Optional[Path]:
        """Write *trajectory* as a T4Scene beside it; return the scene's path,
        or ``None`` when it could not be written (the reason is logged)."""
        try:
            scene, _ = write_t4_scene(
                trajectory, step_s=self.step_s, max_distance_m=self.max_distance_m
            )
        except Exception:  # noqa: BLE001 - best effort: the run's verdict stands
            logger.warning("Writing %s as a T4Scene failed", trajectory, exc_info=True)
            return None
        logger.info("T4Scene written to %s", scene)
        return scene
