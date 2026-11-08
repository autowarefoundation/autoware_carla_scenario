"""Whether a Lanelet2 lanelet lies in a junction: one judgment for every reader.

Autoware's Lanelet2 maps tag every lanelet inside an intersection with
``turn_direction`` (``straight``, ``left`` or ``right``), and only those.
Where the framework asks whether a lanelet is in a junction -- the ODD's
measure of the junction the ego is in (``odd.probes``), a sweep's
``is_junction`` constraint and its bindings (``sweeper``), a route search
(``route.geometry``) -- it asks :func:`is_junction_lanelet`, so where a
scenario is placed and what a run measures never disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

__all__ = ["TURN_DIRECTION_TAG", "is_junction_lanelet"]

#: The Lanelet2 tag every junction lanelet carries, and only those.
TURN_DIRECTION_TAG = "turn_direction"


def is_junction_lanelet(lanelet: ConstLanelet) -> bool:
    """Whether *lanelet* lies in a junction: it has a ``turn_direction`` tag."""
    return TURN_DIRECTION_TAG in lanelet.attributes
