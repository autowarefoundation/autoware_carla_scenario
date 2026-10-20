"""The lane chain around an agent: where each lanelet ahead and behind starts.

The location gate looks ahead for a signal, the observation for a leader and
a follower; both measure along the lanes, by the shortest chain of following
(or preceding) lanelets, without changing lanes -- what SUMO's ``getLeader``,
``getFollower`` and ``getNextTLS`` measure along its route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..route.geometry import MapFeatures

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

__all__ = ["LaneSpan", "lane_chain"]


@dataclass(frozen=True)
class LaneSpan:
    """A lanelet of the chain, and where it starts relative to the agent.

    Attributes:
        lanelet: The lanelet.
        start: Its start, in metres along the chain from the agent: negative
            for the agent's own lanelet (``-s``) and those behind it.
        length: Its centreline length (m).
    """

    lanelet: ConstLanelet
    start: float
    length: float

    @property
    def end(self) -> float:
        """Its end, in metres along the chain from the agent."""
        return self.start + self.length


def _nearest(frontier: list[LaneSpan], ahead: bool) -> LaneSpan:
    index = 0
    for i in range(1, len(frontier)):
        if ahead and frontier[i].start < frontier[index].start:
            index = i
        if not ahead and frontier[i].end > frontier[index].end:
            index = i
    return frontier.pop(index)


def lane_chain(
    features: MapFeatures,
    lanelet: ConstLanelet,
    s: float,
    ahead: float,
    behind: float = 0.0,
) -> dict[int, LaneSpan]:
    """The lanelets within *ahead* metres ahead of and *behind* metres behind
    an agent *s* metres along *lanelet*, by id.

    A lanelet ahead is in the chain when it starts within *ahead*; one behind
    when it ends less than *behind* behind (none with the default 0).  Every branch of a fork or a junction is
    followed, each by its shortest chain; the agent's own lanelet is always
    in it.  A lanelet both ahead and behind (a loop) keeps its place ahead.
    """
    own = LaneSpan(lanelet, -s, features.length(lanelet))
    chain: dict[int, LaneSpan] = {int(lanelet.id): own}
    frontier = [own]
    while frontier:
        current = _nearest(frontier, True)
        if current.end > ahead:
            continue
        for following in features.graph.following(current.lanelet):
            key = int(following.id)
            if key in chain and chain[key].start <= current.end:
                continue
            span = LaneSpan(following, current.end, features.length(following))
            chain[key] = span
            frontier.append(span)
    behind_spans: dict[int, LaneSpan] = {}
    frontier = []
    if behind > 0.0:
        frontier.append(own)
    while frontier:
        current = _nearest(frontier, False)
        if -current.start >= behind:
            continue
        for previous in features.graph.previous(current.lanelet):
            key = int(previous.id)
            if key in chain and key not in behind_spans:
                continue
            length = features.length(previous)
            if key in behind_spans and behind_spans[key].end >= current.start:
                continue
            span = LaneSpan(previous, current.start - length, length)
            behind_spans[key] = span
            chain[key] = span
            frontier.append(span)
    return chain
