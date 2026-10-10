"""Find the routes of a map that match a route search.

A match is a sequence of lanelets, each following the one before it in the
routing graph (no lane changes), cut into the search's segments in order:

* a **lane segment** covers one or more consecutive lanelets that are not
  junction lanelets; a **junction segment** one or more consecutive junction
  lanelets (up to :data:`~.model.MAX_JUNCTION_LANELETS`);
* segments meet at lanelet boundaries, with two exceptions that keep a
  segment's length within its range: a **first** lane segment may start
  part-way into its first lanelet, and a **last** lane segment may end
  part-way into its last one.

Which part of the road a match covers is fixed by these rules, so the same
road is found once, not once per lanelet it could start on:

* a first lane segment is grown *backwards* from where it ends as far as its
  ``length.max`` (or :data:`~.model.DEFAULT_SEGMENT_MAX_M`), and is cut there;
  shorter, it has to start where the road does -- after a junction, or where
  nothing precedes it -- because otherwise it would have grown further;
* a last lane segment is grown forwards the same way and ends at its maximum,
  at a junction, or where the road ends;
* every other lane segment covers whole lanelets, and its length is what they
  add up to;
* a first junction segment starts where its junction does, and a last one ends
  where it does.

Where the road forks, every branch is tried.  The search is a depth-first walk
from every lanelet that may start the first segment, by id, bounded by each
segment's maximum length and lanelet count and by
:data:`~.model.MAX_EXPANSIONS` steps in all; matches are sorted by their
lanelet ids, then by where they start.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Optional

from ..trajectory.relative_lane import _centreline, _heading_at
from .geometry import MapFeatures
from .model import (
    MAX_EXPANSIONS,
    MAX_JUNCTION_LANELETS,
    MAX_LANELETS_PER_SEGMENT,
    STRAIGHT_MAX_TURN_DEG,
    JunctionSegmentSpec,
    LaneSegmentSpec,
    RouteMatch,
    RouteSearchSpec,
    RouteSegmentMatch,
    tristate_holds,
)

__all__ = ["RouteSearchError", "find_route_matches", "junction_turn", "lane_shape"]

logger = logging.getLogger(__name__)


class RouteSearchError(ValueError):
    """A route search that cannot be run (its map lacks what it needs)."""


class _Exhausted(Exception):
    """The search ran out of steps."""


def junction_turn(features: MapFeatures, way: list[Any]) -> str:
    """The turn a run of junction lanelets makes.

    ``straight`` when every one is tagged straight, else the one non-straight
    tag they carry; ``mixed`` when they turn both ways (matches only ``any``).
    """
    turns = {features.turn(ll) for ll in way} - {"straight", ""}
    if not turns:
        return "straight"
    return turns.pop() if len(turns) == 1 else "mixed"


def lane_shape(heading_change_rad: float) -> str:
    """``straight``, ``curved_left`` or ``curved_right`` of a net heading change."""
    degrees = math.degrees(heading_change_rad)
    if abs(degrees) <= STRAIGHT_MAX_TURN_DEG:
        return "straight"
    return "curved_left" if degrees > 0.0 else "curved_right"


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class _Search:
    spec: RouteSearchSpec
    features: MapFeatures
    expansions: int = 0

    def __post_init__(self) -> None:
        self.found: dict[tuple[tuple[int, ...], float, float], RouteMatch] = {}
        self._junction_ok: dict[tuple[Any, ...], bool] = {}

    # -- the walk ------------------------------------------------------------

    def run(self, stop_after: Optional[int]) -> None:
        first = self.spec.segments[0]
        for lanelet in sorted(
            self.features.map.laneletLayer, key=lambda ll: int(ll.id)
        ):
            if not self.features.is_road(lanelet):
                continue
            junction = self.features.is_junction(lanelet)
            if first.kind == "lane" and junction:
                continue
            if (
                first.kind == "lane"
                and len(self.spec.segments) == 1
                and not self._starts_road(lanelet)
            ):
                # A lone lane segment starts where its road does.
                continue
            if first.kind == "junction":
                if not junction or any(
                    self.features.is_junction(p)
                    for p in self.features.previous(lanelet)
                ):
                    continue
            self._extend([lanelet], 0, [0])
            if stop_after is not None and len(self.found) >= stop_after:
                return

    def _step(self) -> None:
        self.expansions += 1
        if self.expansions > MAX_EXPANSIONS:
            raise _Exhausted

    def _extend(self, path: list[Any], k: int, starts: list[int]) -> None:
        """*path*'s last lanelet has just joined segment *k* (starting at ``starts[k]``)."""
        self._step()
        seg = self.spec.segments[k]
        if isinstance(seg, LaneSegmentSpec):
            self._extend_lane(path, k, starts, seg)
        else:
            assert isinstance(seg, JunctionSegmentSpec)
            self._extend_junction(path, k, starts, seg)

    def _successors(self, path: list[Any]) -> list[Any]:
        on = {int(ll.id) for ll in path}
        return [
            ll
            for ll in self.features.following(path[-1])
            if int(ll.id) not in on and self.features.is_road(ll)
        ]

    def _extend_lane(
        self, path: list[Any], k: int, starts: list[int], seg: LaneSegmentSpec
    ) -> None:
        f = self.features
        last_index = len(self.spec.segments) - 1
        lanelets = path[starts[k] :]
        total = sum(f.length(ll) for ll in lanelets)
        limit = seg.max_length
        minimum = seg.length.min or 0.0
        successors = self._successors(path)
        plain = [ll for ll in successors if not f.is_junction(ll)]

        if k == last_index:
            # Grown forwards: to its maximum, a junction, or the road's end.
            if total >= limit - 1e-9:
                end_s = f.length(path[-1]) - (total - limit)
                self._finish(path, starts, end_s)
                return
            ends_here = not plain or any(f.is_junction(ll) for ll in successors)
            if ends_here and total >= minimum - 1e-9:
                self._finish(path, starts, f.length(path[-1]))
            if len(lanelets) < MAX_LANELETS_PER_SEGMENT:
                for nxt in plain:
                    path.append(nxt)
                    self._extend(path, k, starts)
                    path.pop()
            return

        if k == 0:
            # Grown backwards from its end; past its maximum by more than its
            # first lanelet, it would start on a later one: found from there.
            if total - f.length(path[0]) >= limit - 1e-9:
                return
            fits = min(total, limit) >= minimum - 1e-9 and (
                total > limit - 1e-9 or self._starts_road(path[0])
            )
        else:
            if total > limit + 1e-9:
                return
            fits = total >= minimum - 1e-9
        nxt_seg = self.spec.segments[k + 1]
        openers = [
            nxt
            for nxt in successors
            if f.is_junction(nxt) == (nxt_seg.kind == "junction")
        ]
        if fits and openers and self._lane_holds(path, starts, k, seg, None):
            for nxt in openers:
                path.append(nxt)
                self._extend(path, k + 1, [*starts, len(path) - 1])
                path.pop()
        if len(lanelets) < MAX_LANELETS_PER_SEGMENT and (k == 0 or total < limit):
            for nxt in plain:
                path.append(nxt)
                self._extend(path, k, starts)
                path.pop()

    def _starts_road(self, lanelet: Any) -> bool:
        """Whether a lane segment cannot be grown back past *lanelet*."""
        f = self.features
        return all(f.is_junction(p) or not f.is_road(p) for p in f.previous(lanelet))

    def _extend_junction(
        self, path: list[Any], k: int, starts: list[int], seg: JunctionSegmentSpec
    ) -> None:
        f = self.features
        last_index = len(self.spec.segments) - 1
        way = path[starts[k] :]
        successors = self._successors(path)
        if k == last_index:
            if not any(f.is_junction(ll) for ll in successors):
                exit_ = f.straightest_following(path[-1])
                if self._junction_holds(path, starts, k, seg, exit_):
                    self._finish(path, starts, f.length(path[-1]))
        else:
            nxt_seg = self.spec.segments[k + 1]
            for nxt in successors:
                if f.is_junction(nxt) != (nxt_seg.kind == "junction"):
                    continue
                if not self._junction_holds(path, starts, k, seg, nxt):
                    continue
                path.append(nxt)
                self._extend(path, k + 1, [*starts, len(path) - 1])
                path.pop()
        if len(way) < MAX_JUNCTION_LANELETS:
            for nxt in successors:
                if f.is_junction(nxt):
                    path.append(nxt)
                    self._extend(path, k, starts)
                    path.pop()

    # -- segment properties --------------------------------------------------

    def _start_trim(self, path: list[Any], starts: list[int]) -> float:
        """Where on the first lanelet the route starts."""
        first = self.spec.segments[0]
        if not isinstance(first, LaneSegmentSpec) or len(self.spec.segments) == 1:
            return 0.0
        end = starts[1] if len(starts) > 1 else len(path)
        total = sum(self.features.length(ll) for ll in path[:end])
        return max(0.0, total - first.max_length)

    def _lane_holds(
        self,
        path: list[Any],
        starts: list[int],
        k: int,
        seg: LaneSegmentSpec,
        end_s: Optional[float],
    ) -> bool:
        """Whether lane segment *k* (ending at ``end_s`` on its last lanelet) fits."""
        f = self.features
        end = starts[k + 1] if k + 1 < len(starts) else len(path)
        lanelets = path[starts[k] : end]
        for ll in lanelets:
            if not seg.lanes_left.contains(f.lanes_beside(ll, "left")):
                return False
            if not seg.lanes_right.contains(f.lanes_beside(ll, "right")):
                return False
            if seg.opposite_lane != "any" and not tristate_holds(
                seg.opposite_lane, f.opposite(ll) is not None
            ):
                return False
        last = lanelets[-1]
        if not tristate_holds(seg.stop_line, f.has_stop_line(last)):
            return False
        if not tristate_holds(
            seg.traffic_light_stop_line, f.has_traffic_light_stop_line(last)
        ):
            return False
        if seg.shape != "any":
            start_s = self._start_trim(path, starts) if k == 0 else 0.0
            stop_s = f.length(last) if end_s is None else end_s
            before = _heading_at(_centreline(lanelets[0]), start_s)
            after = _heading_at(_centreline(last), max(0.0, stop_s - 1e-6))
            if lane_shape(_wrap(after - before)) != seg.shape:
                return False
        return True

    def _junction_holds(
        self,
        path: list[Any],
        starts: list[int],
        k: int,
        seg: JunctionSegmentSpec,
        exit_: Any,
    ) -> bool:
        """Whether junction segment *k*, left onto *exit_*, fits."""
        f = self.features
        end = starts[k + 1] if k + 1 < len(starts) else len(path)
        way = path[starts[k] : end]
        entry = path[starts[k] - 1] if starts[k] > 0 else f.straightest_previous(way[0])
        key = (
            k,
            tuple(int(ll.id) for ll in way),
            None if entry is None else int(entry.id),
            None if exit_ is None else int(exit_.id),
        )
        if key not in self._junction_ok:
            self._junction_ok[key] = self._junction_fits(seg, way, entry, exit_)
        return self._junction_ok[key]

    def _junction_fits(
        self, seg: JunctionSegmentSpec, way: list[Any], entry: Any, exit_: Any
    ) -> bool:
        f = self.features
        if seg.turn != "any" and junction_turn(f, way) != seg.turn:
            return False
        if seg.traffic_light != "any":
            lit = any(f.has_traffic_light(ll) for ll in way) or (
                entry is not None and f.has_traffic_light(entry)
            )
            if not tristate_holds(seg.traffic_light, lit):
                return False
        wanted = {a: seg.crossing(a) for a in ("left", "right", "opposite")}
        if any(v != "any" for v in wanted.values()):
            members = f.junction_members(way)
            for approach, want in wanted.items():
                present = any(m.conflicts and m.approach == approach for m in members)
                if not tristate_holds(want, present):
                    return False
        if seg.crosswalk_entry != "any" or seg.crosswalk_exit != "any":
            crosswalks = f.junction_crosswalks(way, entry, exit_)
            for leg, want in (
                ("entry", seg.crosswalk_entry),
                ("exit", seg.crosswalk_exit),
            ):
                if not tristate_holds(want, any(c.leg == leg for c in crosswalks)):
                    return False
        return True

    # -- a match ---------------------------------------------------------------

    def _finish(self, path: list[Any], starts: list[int], end_s: float) -> None:
        f = self.features
        if len(starts) != len(self.spec.segments):
            return
        last = self.spec.segments[-1]
        if isinstance(last, LaneSegmentSpec) and not self._lane_holds(
            path, starts, len(starts) - 1, last, end_s
        ):
            return
        start_s = self._start_trim(path, starts)
        offsets = [0.0]
        for ll in path:
            offsets.append(offsets[-1] + f.length(ll))
        segments: list[RouteSegmentMatch] = []
        for k, first in enumerate(starts):
            stop = starts[k + 1] if k + 1 < len(starts) else len(path)
            if k + 1 < len(starts):
                seg_end = offsets[stop] - start_s
            else:
                seg_end = offsets[stop - 1] + end_s - start_s
            seg_start = 0.0 if k == 0 else segments[-1].end
            lanelets = path[first:stop]
            kind = self.spec.segments[k].kind
            segments.append(
                RouteSegmentMatch(
                    kind=kind,
                    start=seg_start,
                    end=seg_end,
                    lanelet_ids=tuple(int(ll.id) for ll in lanelets),
                    turn=junction_turn(f, lanelets) if kind == "junction" else "",
                )
            )
        ids = tuple(int(ll.id) for ll in path)
        key = (ids, round(start_s, 6), round(end_s, 6))
        if key not in self.found:
            self.found[key] = RouteMatch(
                lanelet_ids=ids,
                start_s=start_s,
                end_s=end_s,
                segments=tuple(segments),
            )


def find_route_matches(
    spec: RouteSearchSpec,
    lanelet_map: Any,
    routing_graph: Any = None,
    *,
    features: Optional[MapFeatures] = None,
) -> list[RouteMatch]:
    """Every route of *lanelet_map* that matches *spec*, up to ``spec.max_matches``.

    Sorted by lanelet ids, then by start; with ``spec.seed`` shuffled by it
    first.  Each match's :attr:`~.model.RouteMatch.index` is its position in
    the list returned.  When the search runs out of steps
    (:data:`~.model.MAX_EXPANSIONS`) it returns what it found, and logs that.
    """
    if routing_graph is None:
        from ..sweeper.constraints import create_routing_graph  # noqa: PLC0415

        routing_graph = create_routing_graph(lanelet_map)
    search = _Search(spec, features or MapFeatures(lanelet_map, routing_graph))
    # Sorted by lanelet ids, the matches from one starting lanelet come before
    # those of any later one: unshuffled, the walk can stop once it has enough.
    stop_after = spec.max_matches if spec.seed is None else None
    try:
        search.run(stop_after)
    except _Exhausted:
        logger.warning(
            "Route search stopped after %d steps with %d match(es); the map may "
            "have more.",
            MAX_EXPANSIONS,
            len(search.found),
        )
    matches = sorted(search.found.values(), key=lambda m: (m.lanelet_ids, m.start_s))
    if spec.seed is not None:
        random.Random(spec.seed).shuffle(matches)
    matches = matches[: spec.max_matches]
    out = [
        RouteMatch(m.lanelet_ids, m.start_s, m.end_s, m.segments, index=i)
        for i, m in enumerate(matches)
    ]
    logger.info("Route search found %d match(es)", len(out))
    return out
