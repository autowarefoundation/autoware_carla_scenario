"""T4 datasets of autoware_carla_scenario runs.

:class:`T4Writer` is the Rust writer: it keeps the tables of one scene as the
frames come, encodes and writes the sensor files (LiDAR sweeps as
``.pcd.bin``, camera frames as JPEG) on worker threads, and writes the tables
when it is finished.  The datasets it writes load with t4-devkit
(``t4_devkit.Tier4``).  The runner drives it through
``autoware_carla_scenario.t4`` (``t4.enabled=true``).
"""

from ._core import T4Writer, __version__

__all__ = ["T4Writer", "__version__"]
