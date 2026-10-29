"""Trajectory logs as T4Scenes, the exchange format of tier4/e2e-devkit.

:func:`trajectory_log_to_t4_scene` reads a run's ``{stem}.trajectory.jsonl``
-- no CARLA needed -- and returns it as a ``T4Scene`` message
(``t4_scene.proto``, contract 0.1.0, vendored) with the ``MapResource`` of the
Lanelet2 map the run was on; :func:`write_t4_scene` writes the two as
``{stem}.t4scene.pb`` and ``{stem}.t4scene.map.pb``.  From the shell::

    scenario-t4scene outputs/2026-10-29/10-00-00

See docs/t4_scene.md for what goes where, and how the devkit reads them.
"""

from __future__ import annotations

from .convert import (
    LOG_SUFFIX,
    MAP_SUFFIX,
    SCENE_SUFFIX,
    T4SceneExport,
    trajectory_log_to_t4_scene,
    write_t4_scene,
)
from .map_resource import map_resource

__all__ = [
    "LOG_SUFFIX",
    "MAP_SUFFIX",
    "SCENE_SUFFIX",
    "T4SceneExport",
    "map_resource",
    "trajectory_log_to_t4_scene",
    "write_t4_scene",
]
