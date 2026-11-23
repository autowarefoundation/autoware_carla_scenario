"""The check of a route-progress target that lies off the scenario's route.

Which bound of the route an anchor is pinned to, whatever the map
(:func:`route_bound`), and the warning a ``route_progress`` condition gets
when its target lies before the route's start or past its end
(:func:`route_progress_warning`, #1385, #1389).  Kept apart from
:mod:`authoring.validator`, which imports pydantic, so the static check
compiles these rules (#2386); the validator reads the anchor, the segment
kinds, the value, the rule and the tolerance off the document and reports
what these return.
"""

from __future__ import annotations

from typing import Optional

from ..route.model import parse_anchor

__all__ = ["route_bound", "route_progress_warning"]


def route_bound(anchor_text: str, segment_kinds: list[str]) -> Optional[str]:
    """``"start"`` when *anchor_text* is route s 0 on every match of a route
    whose segments are of *segment_kinds*, ``"end"`` when it is the route's
    length on every match, else ``None``.

    A blank *anchor_text* is the route's start.  The route's length is only
    known once the search runs on a map, but its first segment always starts
    at 0 and its last always ends at the length, so these anchors are pinned
    to a bound whatever the map.
    """
    text = "start" if not anchor_text.strip() else anchor_text
    try:
        kind, index, end = parse_anchor(text)
    except ValueError:
        return None
    if kind == "start" or kind == "end":
        return kind
    if kind == "junction":
        junction_at = [
            k
            for k, segment_kind in enumerate(segment_kinds)
            if segment_kind == "junction"
        ]
        if index >= len(junction_at):
            return None
        index = junction_at[index]
    if index == 0 and (end == "start" or end == "entry"):
        return "start"
    if index == len(segment_kinds) - 1 and (end == "end" or end == "exit"):
        return "end"
    return None


def route_progress_warning(
    value: float,
    bound: Optional[str],
    rule: str,
    tolerance: Optional[float],
    default_tolerance: float,
) -> Optional[str]:
    """The warning for a ``route_progress`` target *value* m from an anchor
    pinned to *bound* (:func:`route_bound`), compared by *rule*, or ``None``.

    An entity's route s is clamped to ``[0, length]``, so a target before the
    route's start or past its end is one the rule always or never meets: from
    ``start - 15 m``, ``>=`` holds from the first tick the entity is on the map
    (one not spawned yet, or hidden below the route, has no progress), even
    behind the start; to ``end + 5 m`` it never holds, and a waypoint waiting
    on it never departs.  A target exactly on a bound with a strict rule
    pointing off the route never holds either (#1389): ``<`` the start, ``>``
    the end.  The other rules there -- the default ``>=`` the start, and
    ``<=`` the end, ``>`` the start, ``<`` the end, ``=`` either -- can all
    hold.  ``=`` compares within *tolerance*, *default_tolerance* when it is
    ``None``; an unknown *rule* is the field check's, and gets no warning.
    """
    side = bound if bound is not None else ""
    hint = "Give a distance that puts the target on the route."
    where = ""
    below = False
    if side == "start" and value < 0:
        where = f"{-value:g} m before the route's start"
        below = True
    elif side == "end" and value > 0:
        where = f"{value:g} m past the route's end"
        below = False
    elif side == "start" and value == 0 and rule == "less_than":
        where = "the route's start"
        below = True
        hint = 'Use "less than or equal" (or a positive distance) instead.'
    elif side == "end" and value == 0 and rule == "greater_than":
        where = "the route's end"
        below = False
        hint = 'Use "greater than or equal" (or a negative distance) instead.'
    else:
        return None
    rising = rule == "greater_than" or rule == "greater_than_or_equal"
    falling = rule == "less_than" or rule == "less_than_or_equal"
    always = False
    if rule == "equal_to":
        within = default_tolerance if tolerance is None else tolerance
        if abs(value) <= within:
            return None
        always = False
    elif rising or falling:
        always = rising == below
    else:
        return None  # an unknown rule is the field check's
    reading = rule.replace("_", " ")
    verdict = ""
    if always:
        extreme = "behind the route's start" if below else "at the route's end"
        verdict = (
            f'"{reading}" holds from the first tick the entity is on the map, '
            f"even for one {extreme}"
        )
    else:
        verdict = (
            f'"{reading}" can never hold: an action waiting on it never '
            "fires, and a waypoint never departs"
        )
    return (
        f"Route progress: the target is {where}. Progress is clamped to the "
        f"route, from 0 to its length, so {verdict}. {hint}"
    )
