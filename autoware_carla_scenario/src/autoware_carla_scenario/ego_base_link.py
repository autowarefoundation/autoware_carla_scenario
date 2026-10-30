"""Where the running scenario's ego has its ``base_link``, as the run uses it.

An ego that places its frame at a rear-axle offset of its own -- a driver
entity's rig origin (``driver.rear_axle_offset_m``), a ROS 2 ego's published
odometry (``rear_axle_offset_m``), the ``base_link`` a bridged stack localizes
at (``base_link_offset_m``) -- says so here from its ``on_scenario_start``
(:func:`set_ego_base_link`), and the run's records read it back
(:func:`ego_base_link`): the trajectory log's ``rear_axle`` record and the T4
dataset's ego poses then put ``base_link`` where the run put it, not where
:func:`~autoware_carla_scenario.driver.observation.rear_axle_offset` would
guess it from the wheels or the box.

Unlike the T4 dataset's driver offer (``t4.offer_driver_sensors``), which also
means "the driver feeds the dataset's sensors", this says nothing but where
``base_link`` is.  It is process-wide and per run: the ego clears it in its
``on_scenario_end`` (:func:`clear_ego_base_link`), and
:class:`~autoware_carla_scenario.ScenarioRunner` clears it before and after
each run, so a value one case set is never read as the next one's.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["clear_ego_base_link", "ego_base_link", "set_ego_base_link"]

#: Metres along the running ego's x axis from its actor's origin to its
#: ``base_link``; None when no ego said where it is.
_OFFSET_M: Optional[float] = None


def set_ego_base_link(offset_m: float) -> None:
    """The running ego has its ``base_link`` *offset_m* metres along its x axis
    from its actor's origin (negative: behind it)."""
    global _OFFSET_M
    _OFFSET_M = float(offset_m)


def ego_base_link() -> Optional[float]:
    """The offset :func:`set_ego_base_link` set for the running ego, or ``None``
    when no ego set one."""
    return _OFFSET_M


def clear_ego_base_link() -> None:
    """Forget the running ego's ``base_link``: its scenario has ended."""
    global _OFFSET_M
    _OFFSET_M = None
