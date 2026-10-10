"""T4 driving scenes, written down as autoware_carla_scenario scenarios.

* :mod:`.reader` -- read a converted T4 scene (``derived/meta.json``,
  ``scalars.npz``, ``frames.pack``) into the ego's and every road user's
  trajectory: :func:`read_t4_scene`, :class:`T4SceneTranscription`;
* :mod:`.document` -- the scenario document that replays one, for the Scenario
  Editor and the declarative runtime: :func:`transcription_to_document`;
* :mod:`.cli` -- the ``t4-scenario`` command that does both.

The replay itself is the framework's ``FollowTrajectoryAction``; this package
only reads the dataset, so the framework carries no dataset format of its own.
"""

from .reader import (
    T4_FRAME_RATE_HZ,
    T4Category,
    T4Detection,
    T4ObjectTrack,
    T4SceneTranscription,
    associate_tracks,
    read_t4_scene,
)

__all__ = [
    "T4Category",
    "T4Detection",
    "T4ObjectTrack",
    "T4SceneTranscription",
    "T4_FRAME_RATE_HZ",
    "associate_tracks",
    "read_t4_scene",
]
