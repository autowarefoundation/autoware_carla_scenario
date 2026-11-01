"""Posing a pedestrian's skeleton: the core the pedestrian-motion actions share.

- :mod:`~autoware_carla_scenario.pedestrian.skeleton`: the CARLA walker's
  bones a motion drives, how a frame of bone directions poses them
  (:class:`~autoware_carla_scenario.pedestrian.skeleton.WalkerRetarget`), and
  the directions of joints a generator places
  (:func:`~autoware_carla_scenario.pedestrian.skeleton.directions_from_joints`).
- :mod:`~autoware_carla_scenario.pedestrian.soma`: a SOMA ``.npz`` motion read
  as those directions.
- :mod:`~autoware_carla_scenario.pedestrian.poser`: a walker taken over by a
  motion (:class:`~autoware_carla_scenario.pedestrian.poser.WalkerPoser`):
  found by role, posed frame by frame, placed, and handed back to its
  animation.

``PlayMotionAction`` plays a file through them;
``SmartPedestrianControllerAction`` a motion generated as the run goes.

The package imports nothing itself, so its numpy-free modules stay checkable
by Codon.
"""
