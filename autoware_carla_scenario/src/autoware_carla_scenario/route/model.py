"""A route search and its matches, as data: no map, no lanelet2, no CARLA.

A *logical* scenario describes the ego's drive as a pattern of road --
"60-120 m of two-lane road, a signalised left turn with traffic crossing from
the right, then 30 m out of it" -- instead of as lanelet ids, so it runs on any
map that has such a road.  This module holds the two halves of that as plain
data:

* the **search**: an ordered list of segment patterns (:class:`LaneSegmentSpec`,
  :class:`JunctionSegmentSpec`) with how the ego is placed on a match
  (:class:`RouteSearchSpec`), read from the ``sweep.route`` mapping a scenario
  document renders (:func:`parse_route_search`);
* a **match**: the concrete lanelet sequence a map answers with, and where each
  segment starts and ends along it (:class:`RouteMatch`), which is what an
  expanded scenario carries from ``scenario-expand`` to the run.

Distances along a match are *route s*: metres along the centrelines of its
lanelets, ``0`` where the route starts (which may be part-way into its first
lanelet).  The named points on it a scenario refers to are *anchors*
(:func:`parse_anchor`).

The search itself (:mod:`.search`) and everything that needs the map's geometry
live elsewhere; this module is imported by the editor, which has no simulator.
See ``docs/logical_scenarios.md`` for the semantics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

__all__ = [
    "ANCHOR_PATTERN",
    "APPROACHES",
    "CROSSWALK_LEGS",
    "DEFAULT_MAX_MATCHES",
    "DEFAULT_SEGMENT_MAX_M",
    "JunctionSegmentSpec",
    "LaneSegmentSpec",
    "MAX_MATCHES_LIMIT",
    "Range",
    "RouteMatch",
    "RouteSearchSpec",
    "RouteSegmentMatch",
    "SHAPES",
    "SIDES",
    "STRAIGHT_MAX_TURN_DEG",
    "TRISTATE",
    "TURNS",
    "anchor_problem",
    "parse_anchor",
    "parse_route_search",
]

# ---------------------------------------------------------------------------
# Thresholds and caps (documented in docs/logical_scenarios.md)
# ---------------------------------------------------------------------------

#: A lane segment is ``straight`` when its net heading change, from where it
#: starts to where it ends, is at most this many degrees either way; beyond it
#: it is ``curved_left`` (anticlockwise) or ``curved_right``.
STRAIGHT_MAX_TURN_DEG = 15.0
#: The longest a lane segment with no ``length.max`` is grown to (m).
DEFAULT_SEGMENT_MAX_M = 300.0
#: How many lanelets one lane segment may span.
MAX_LANELETS_PER_SEGMENT = 50
#: How many consecutive junction lanelets one junction segment may span.
MAX_JUNCTION_LANELETS = 4
#: How many DFS steps one search may take before it stops (and says so).
MAX_EXPANSIONS = 200_000
#: How many matches a search returns when the document does not say.
DEFAULT_MAX_MATCHES = 64
#: The most matches a document may ask for.
MAX_MATCHES_LIMIT = 1000

#: Values of a yes / no / don't-care property.
TRISTATE: tuple[str, ...] = ("any", "yes", "no")
#: Values of a lane segment's ``shape``.
SHAPES: tuple[str, ...] = ("any", "straight", "curved_left", "curved_right")
#: Values of a junction segment's ``turn``.
TURNS: tuple[str, ...] = ("any", "left", "right", "straight")
#: The approaches traffic can enter a junction from, seen from the ego's own.
APPROACHES: tuple[str, ...] = ("left", "right", "opposite")
#: The legs of a junction a crosswalk may cross.
CROSSWALK_LEGS: tuple[str, ...] = ("entry", "exit")
#: Sides of the road.
SIDES: tuple[str, ...] = ("left", "right")

Tristate = Literal["any", "yes", "no"]


def _tristate(name: str, value: Any) -> str:
    """*value* as one of :data:`TRISTATE`; YAML booleans read as yes / no."""
    if value is None:
        return "any"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text not in TRISTATE:
        raise ValueError(f"{name} must be one of {list(TRISTATE)}, got {value!r}")
    return text


def tristate_holds(wanted: str, actual: bool) -> bool:
    """Whether *actual* satisfies a :data:`TRISTATE` *wanted*."""
    if wanted == "any":
        return True
    return actual if wanted == "yes" else not actual


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Range:
    """An inclusive range; an unset bound is open."""

    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self) -> None:
        for bound in (self.min, self.max):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("a range bound must be finite")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"range min {self.min:g} is above its max {self.max:g}")

    def contains(self, value: float) -> bool:
        """Whether *value* lies in the range."""
        if self.min is not None and value < self.min - 1e-9:
            return False
        return self.max is None or value <= self.max + 1e-9

    @classmethod
    def parse(cls, name: str, value: Any, *, whole: bool = False) -> "Range":
        """A range from ``{min, max}``, a bare number (exactly that), or nothing."""
        if value is None:
            return cls()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(float(value), float(value))
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping {{min, max}}, got {value!r}")
        unknown = sorted(set(value) - {"min", "max"})
        if unknown:
            raise ValueError(f"{name} takes only min and max, not {unknown}")
        bounds: list[Optional[float]] = []
        for key in ("min", "max"):
            raw = value.get(key)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                bounds.append(None)
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}.{key} must be a number, got {raw!r}") from exc
            if whole and not number.is_integer():
                raise ValueError(f"{name}.{key} is a count, a whole number")
            bounds.append(number)
        try:
            return cls(bounds[0], bounds[1])
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc

    def as_dict(self) -> dict[str, float]:
        """The range as ``{min, max}``, unset bounds left out."""
        out: dict[str, float] = {}
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        return out


@dataclass(frozen=True)
class LaneSegmentSpec:
    """A stretch of road between junctions (no ``turn_direction`` lanelet).

    Attributes:
        length: Its length along the route (m).  Unset ``max`` grows it to
            :data:`DEFAULT_SEGMENT_MAX_M` at most.
        lanes_left: How many lanes of the same direction lie to its left --
            on every lanelet of the segment.
        lanes_right: The same, to its right.
        opposite_lane: Whether a lane of the opposite direction runs beside
            the road -- on every lanelet of the segment (``yes``), on none
            (``no``), or either.
        shape: ``straight``, ``curved_left`` or ``curved_right`` by its net
            heading change (:data:`STRAIGHT_MAX_TURN_DEG`); ``any``.
        stop_line: Whether its last lanelet has a stop line.
        traffic_light_stop_line: Whether its last lanelet's stop line belongs
            to a traffic light.
    """

    length: Range = field(default_factory=Range)
    lanes_left: Range = field(default_factory=Range)
    lanes_right: Range = field(default_factory=Range)
    opposite_lane: str = "any"
    shape: str = "any"
    stop_line: str = "any"
    traffic_light_stop_line: str = "any"
    kind: str = "lane"

    @property
    def max_length(self) -> float:
        """The longest the segment is grown to (m)."""
        return DEFAULT_SEGMENT_MAX_M if self.length.max is None else self.length.max


@dataclass(frozen=True)
class JunctionSegmentSpec:
    """The ego's way through a junction (consecutive ``turn_direction`` lanelets).

    Attributes:
        turn: ``left``, ``right``, ``straight`` -- the lanelets'
            ``turn_direction`` -- or ``any``.
        traffic_light: Whether the junction is signalised for the ego: a
            traffic light on one of its junction lanelets or on the lanelet
            entering it.
        crossing_from_left / crossing_from_right / crossing_from_opposite:
            Whether a lanelet entering the same junction from that approach
            conflicts with the ego's way through it.
        crosswalk_entry / crosswalk_exit: Whether a crosswalk crosses the
            ego's path on the way into (out of) the junction.
    """

    turn: str = "any"
    traffic_light: str = "any"
    crossing_from_left: str = "any"
    crossing_from_right: str = "any"
    crossing_from_opposite: str = "any"
    crosswalk_entry: str = "any"
    crosswalk_exit: str = "any"
    kind: str = "junction"

    def crossing(self, approach: str) -> str:
        """The wanted :data:`TRISTATE` for crossing traffic from *approach*."""
        return str(getattr(self, f"crossing_from_{approach}"))


SegmentSpec = "LaneSegmentSpec | JunctionSegmentSpec"

_LANE_KEYS = {
    "kind",
    "length",
    "lanes_left",
    "lanes_right",
    "opposite_lane",
    "shape",
    "stop_line",
    "traffic_light_stop_line",
}
_JUNCTION_KEYS = {
    "kind",
    "turn",
    "traffic_light",
    "crossing_from_left",
    "crossing_from_right",
    "crossing_from_opposite",
    "crosswalk_entry",
    "crosswalk_exit",
}


def _parse_segment(index: int, raw: Any) -> "LaneSegmentSpec | JunctionSegmentSpec":
    where = f"segments[{index}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping, got {raw!r}")
    kind = str(raw.get("kind", "")).strip()
    if kind == "lane":
        unknown = sorted(set(raw) - _LANE_KEYS)
        if unknown:
            raise ValueError(f"{where}: a lane segment does not take {unknown}")
        shape = str(raw.get("shape") or "any").strip()
        if shape not in SHAPES:
            raise ValueError(f"{where}.shape must be one of {list(SHAPES)}")
        length = Range.parse(f"{where}.length", raw.get("length"))
        if (length.min or 0.0) < 0.0:
            raise ValueError(f"{where}.length.min must not be negative")
        if length.max is not None and length.max <= 0.0:
            raise ValueError(f"{where}.length.max must be positive")
        return LaneSegmentSpec(
            length=length,
            lanes_left=Range.parse(
                f"{where}.lanes_left", raw.get("lanes_left"), whole=True
            ),
            lanes_right=Range.parse(
                f"{where}.lanes_right", raw.get("lanes_right"), whole=True
            ),
            opposite_lane=_tristate(f"{where}.opposite_lane", raw.get("opposite_lane")),
            shape=shape,
            stop_line=_tristate(f"{where}.stop_line", raw.get("stop_line")),
            traffic_light_stop_line=_tristate(
                f"{where}.traffic_light_stop_line", raw.get("traffic_light_stop_line")
            ),
        )
    if kind == "junction":
        unknown = sorted(set(raw) - _JUNCTION_KEYS)
        if unknown:
            raise ValueError(f"{where}: a junction segment does not take {unknown}")
        turn = str(raw.get("turn") or "any").strip()
        if turn not in TURNS:
            raise ValueError(f"{where}.turn must be one of {list(TURNS)}")
        values = {
            key: _tristate(f"{where}.{key}", raw.get(key))
            for key in _JUNCTION_KEYS - {"kind", "turn"}
        }
        return JunctionSegmentSpec(turn=turn, **values)
    raise ValueError(f"{where}.kind must be 'lane' or 'junction', got {kind!r}")


@dataclass(frozen=True)
class RouteSearchSpec:
    """A route search: the segment pattern, and how the ego is put on a match.

    Attributes:
        segments: The pattern, in driving order.
        ego_spawn_s: Where the ego spawns, in metres along the route from its
            start.
        ego_goal: Whether the ego's goal is the route's end.
        ego_goal_margin: How far before the route's end the goal is (m).
        match_index: Which match a run that was not expanded takes.
        max_matches: How many matches the search returns at most.
        seed: ``None`` keeps the matches in their sorted order; a number
            shuffles them with it (deterministically) before ``max_matches``
            cuts the list, for a sample spread over the map.
    """

    segments: tuple["LaneSegmentSpec | JunctionSegmentSpec", ...]
    ego_spawn_s: float = 0.0
    ego_goal: bool = True
    ego_goal_margin: float = 0.0
    match_index: int = 0
    max_matches: int = DEFAULT_MAX_MATCHES
    seed: Optional[int] = None

    @property
    def junction_count(self) -> int:
        """How many junction segments the pattern has."""
        return sum(1 for s in self.segments if s.kind == "junction")


_SEARCH_KEYS = {
    "segments",
    "ego_spawn_s",
    "ego_goal",
    "ego_goal_margin",
    "match_index",
    "max_matches",
    "seed",
}


def parse_route_search(raw: Mapping[str, Any]) -> RouteSearchSpec:
    """Read a route search from its ``sweep.route`` mapping.

    Raises:
        ValueError: On anything that is not a well-formed search, saying what.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"a route search must be a mapping, got {raw!r}")
    unknown = sorted(set(raw) - _SEARCH_KEYS)
    if unknown:
        raise ValueError(f"a route search does not take {unknown}")
    segments_raw = raw.get("segments") or []
    if not isinstance(segments_raw, Sequence) or isinstance(segments_raw, str):
        raise ValueError("route segments must be a list")
    if not segments_raw:
        raise ValueError("a route search needs at least one segment")
    segments = tuple(_parse_segment(i, s) for i, s in enumerate(segments_raw))
    spawn_s = float(raw.get("ego_spawn_s") or 0.0)
    margin = float(raw.get("ego_goal_margin") or 0.0)
    if spawn_s < 0.0:
        raise ValueError("ego_spawn_s must not be negative")
    if margin < 0.0:
        raise ValueError("ego_goal_margin must not be negative")
    match_index = int(raw.get("match_index") or 0)
    if match_index < 0:
        raise ValueError("match_index must not be negative")
    raw_max = raw.get("max_matches")
    max_matches = DEFAULT_MAX_MATCHES if raw_max is None else int(raw_max)
    if not 1 <= max_matches <= MAX_MATCHES_LIMIT:
        raise ValueError(f"max_matches must be between 1 and {MAX_MATCHES_LIMIT}")
    seed_raw = raw.get("seed")
    goal = raw.get("ego_goal", True)
    return RouteSearchSpec(
        segments=segments,
        ego_spawn_s=spawn_s,
        ego_goal=True if goal is None else bool(goal),
        ego_goal_margin=margin,
        match_index=match_index,
        max_matches=max_matches,
        seed=None if seed_raw is None else int(seed_raw),
    )


# ---------------------------------------------------------------------------
# A match
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteSegmentMatch:
    """Where one segment of the pattern lies on a match.

    Attributes:
        kind: ``lane`` or ``junction``.
        start: Route s it starts at (m).
        end: Route s it ends at (m).
        lanelet_ids: The lanelets it covers, in driving order.
        turn: A junction segment's ``turn_direction``; empty for a lane.
    """

    kind: str
    start: float
    end: float
    lanelet_ids: tuple[int, ...]
    turn: str = ""

    @property
    def length(self) -> float:
        """Its length along the route (m)."""
        return self.end - self.start


@dataclass(frozen=True)
class RouteMatch:
    """A concrete route on one map: the answer to a :class:`RouteSearchSpec`.

    Attributes:
        lanelet_ids: Consecutive lanelets (each follows the one before in the
            routing graph), in driving order.
        start_s: Where on the first lanelet the route starts (m along it).
        end_s: Where on the last lanelet it ends (m along it).
        segments: Each segment of the pattern, in order; together they cover
            the route.
        index: Which match of its search this is (informational).
    """

    lanelet_ids: tuple[int, ...]
    start_s: float
    end_s: float
    segments: tuple[RouteSegmentMatch, ...]
    index: int = 0

    @property
    def length(self) -> float:
        """The route's length (m): where its last segment ends."""
        return self.segments[-1].end if self.segments else 0.0

    @property
    def junctions(self) -> tuple[RouteSegmentMatch, ...]:
        """The junction segments, in order: what ``junction:K`` counts."""
        return tuple(s for s in self.segments if s.kind == "junction")

    def junction(self, index: int) -> RouteSegmentMatch:
        """Junction segment *index* (from 0).

        Raises:
            ValueError: If the route has no such junction.
        """
        junctions = self.junctions
        if not 0 <= index < len(junctions):
            raise ValueError(
                f"the route has {len(junctions)} junction(s); there is no junction "
                f"{index}"
            )
        return junctions[index]

    def anchor_s(self, anchor: Optional[str]) -> float:
        """Route s of a named point (:func:`parse_anchor`); ``None`` is ``start``.

        Raises:
            ValueError: On an anchor that names something the route lacks.
        """
        kind, index, end = parse_anchor(anchor or "start")
        if kind == "start":
            return 0.0
        if kind == "end":
            return self.length
        if kind == "segment":
            if not 0 <= index < len(self.segments):
                raise ValueError(
                    f"anchor {anchor!r}: the route has {len(self.segments)} "
                    "segment(s)"
                )
            segment = self.segments[index]
        else:
            segment = self.junction(index)
        return segment.start if end in ("start", "entry") else segment.end

    # -- carried through Hydra ---------------------------------------------

    def to_config(self) -> dict[str, Any]:
        """The match as the ``scenario.route`` keys an expanded run is given."""
        return {
            "index": self.index,
            "lanelet_ids": list(self.lanelet_ids),
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "segment_kinds": [s.kind for s in self.segments],
            "segment_ends": [round(s.end, 4) for s in self.segments],
            "segment_lanelet_counts": [len(s.lanelet_ids) for s in self.segments],
            "junction_turns": [s.turn for s in self.segments if s.kind == "junction"],
        }

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> Optional["RouteMatch"]:
        """The match ``scenario.route`` holds, or ``None`` when it holds none.

        Raises:
            ValueError: On keys that do not describe one route.
        """
        lanelet_ids = [int(v) for v in (raw.get("lanelet_ids") or [])]
        if not lanelet_ids:
            return None
        kinds = [str(v) for v in (raw.get("segment_kinds") or [])]
        ends = [float(v) for v in (raw.get("segment_ends") or [])]
        counts = [int(v) for v in (raw.get("segment_lanelet_counts") or [])]
        turns = [str(v) for v in (raw.get("junction_turns") or [])]
        if not (len(kinds) == len(ends) == len(counts)) or not kinds:
            raise ValueError(
                "scenario.route: segment_kinds, segment_ends and "
                "segment_lanelet_counts must be lists of one length"
            )
        if sum(counts) != len(lanelet_ids):
            raise ValueError(
                "scenario.route: segment_lanelet_counts must add up to the "
                f"{len(lanelet_ids)} lanelets of the route"
            )
        if len(turns) != kinds.count("junction"):
            raise ValueError("scenario.route: one junction_turns entry per junction")
        segments: list[RouteSegmentMatch] = []
        start, first, turn_iter = 0.0, 0, iter(turns)
        for kind, end, count in zip(kinds, ends, counts):
            segments.append(
                RouteSegmentMatch(
                    kind=kind,
                    start=start,
                    end=end,
                    lanelet_ids=tuple(lanelet_ids[first : first + count]),
                    turn=next(turn_iter) if kind == "junction" else "",
                )
            )
            start, first = end, first + count
        return cls(
            lanelet_ids=tuple(lanelet_ids),
            start_s=float(raw.get("start_s") or 0.0),
            end_s=float(raw.get("end_s") or 0.0),
            segments=tuple(segments),
            index=int(raw.get("index") or 0),
        )


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

#: ``start``, ``end``, ``segment:K:start|end`` or ``junction:K:entry|exit``.
ANCHOR_PATTERN = re.compile(
    r"^(?:(start|end)|(segment):(\d+):(start|end)|(junction):(\d+):(entry|exit))$"
)


def parse_anchor(anchor: str) -> tuple[str, int, str]:
    """``(kind, index, end)`` of an anchor; ``start``/``end`` give index ``-1``.

    Raises:
        ValueError: On text that is not an anchor.
    """
    match = ANCHOR_PATTERN.match(str(anchor).strip())
    if match is None:
        raise ValueError(
            f"{anchor!r} is not a route anchor: write start, end, "
            "segment:K:start, segment:K:end, junction:K:entry or junction:K:exit"
        )
    if match.group(1):
        return match.group(1), -1, ""
    if match.group(2):
        return "segment", int(match.group(3)), match.group(4)
    return "junction", int(match.group(6)), match.group(7)


def anchor_problem(
    anchor: Optional[str], segment_count: int, junction_count: int
) -> Optional[str]:
    """Why *anchor* names nothing on a pattern of that shape, or ``None``."""
    if anchor is None or not str(anchor).strip():
        return None
    try:
        kind, index, _end = parse_anchor(anchor)
    except ValueError as exc:
        return str(exc)
    if kind == "segment" and index >= segment_count:
        return f"anchor {anchor!r}: the route has {segment_count} segment(s)"
    if kind == "junction" and index >= junction_count:
        return f"anchor {anchor!r}: the route has {junction_count} junction(s)"
    return None
