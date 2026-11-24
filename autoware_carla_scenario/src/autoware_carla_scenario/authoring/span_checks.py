"""The checks of an action's spans: along its lane, in time and over its path.

What :mod:`authoring.validator` checks of a construction zone's and a stalled
object's start and end (:func:`check_spans`, with a highway construction
zone's length against its tapers and buffer, :func:`check_highway_zone_length`)
and of a trajectory's vertex times (:func:`check_vertex_times`), over the
numbers the validator reads off the card.  Kept apart from the validator,
which imports pydantic, so the static check compiles these rules (#2386).
The validator turns the ``(severity, path, message)`` triples they return
back into its issues, in the order they are returned.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["check_highway_zone_length", "check_spans", "check_vertex_times"]


def check_spans(
    path: str,
    node_type: str,
    start_s: Optional[float],
    end_s: Optional[float],
    start_time: Optional[float],
    end_time: Optional[float],
    taper_length_in: Optional[float],
    buffer_zone_length: Optional[float],
    taper_length_out: Optional[float],
    forever_s: float,
) -> list[tuple[str, str, str]]:
    """A construction zone ends after it starts, along its lane and in time;
    a stalled object is taken away after it is put down, as the actions ask
    (an end of *forever_s* or more is "never", which the empty field says).

    Each number is ``None`` when the card leaves it blank with no default, or
    gives no finite number; *forever_s* is read for a ``stalled_object`` only.
    """
    found: list[tuple[str, str, str]] = []
    if node_type != "stalled_object":
        if start_s is not None and end_s is not None and not end_s > start_s:
            found.append(
                (
                    "error",
                    f"{path}.end_s",
                    "The zone's end (end_s) must be further along than its start "
                    "(start_s).",
                )
            )
        elif start_s is not None and end_s is not None:
            if node_type == "construction_zone":
                found.extend(
                    check_highway_zone_length(
                        path,
                        end_s - start_s,
                        taper_length_in,
                        buffer_zone_length,
                        taper_length_out,
                    )
                )
    if start_time is None or end_time is None:
        return found
    if not end_time > start_time:
        found.append(
            (
                "error",
                f"{path}.end_time",
                "The end time must be after the start time; leave it empty for never.",
            )
        )
    elif node_type == "stalled_object" and end_time > forever_s:
        found.append(
            (
                "error",
                f"{path}.end_time",
                f"An end time past {forever_s:g} s is never: leave it empty.",
            )
        )
    return found


def check_highway_zone_length(
    path: str,
    length: float,
    taper_length_in: Optional[float],
    buffer_zone_length: Optional[float],
    taper_length_out: Optional[float],
) -> list[tuple[str, str, str]]:
    """Warn of a construction zone shorter than its tapers and buffer: it has
    no work zone, and the three are shortened to fit it (#2263), as
    ``construction_zones`` does.  Nothing is said when one of the three is
    ``None``."""
    found: list[tuple[str, str, str]] = []
    if taper_length_in is None or buffer_zone_length is None:
        return found
    if taper_length_out is None:
        return found
    # Unwrapped into floats first: Codon types a tuple of an Optional as one.
    taper_in: float = taper_length_in
    buffer: float = buffer_zone_length
    taper_out: float = taper_length_out
    parts: list[tuple[str, float]] = [
        ("taper_length_in", taper_in),
        ("buffer_zone_length", buffer),
        ("taper_length_out", taper_out),
    ]
    # sum() starts from the int 0; so does this, as a float.
    needed = 0.0
    for _, value in parts:
        needed = needed + value
    if not length < needed:
        return found
    named = " + ".join([f"{key} {value:g} m" for key, value in parts])
    found.append(
        (
            "warning",
            f"{path}.end_s",
            f"The zone is {length:g} m long (start_s to end_s), shorter than "
            f"{named} = {needed:g} m: it has no work zone, and the tapers and "
            "the buffer are shortened in proportion to fit it.",
        )
    )
    return found


def check_vertex_times(
    path: str, count: int, gated: set[int], times: dict[int, Optional[float]]
) -> list[tuple[str, str, str]]:
    """The timing rules of a trajectory of *count* vertices.

    *gated* holds the vertices (from 1) with a waypoint condition, and
    *times* those with a time condition, each to its time: ``None`` when the
    time is not a finite number.  Once any vertex departs at a time, every
    vertex departs on a condition (a time or another), and the times do not
    decrease along the path.
    """
    found: list[tuple[str, str, str]] = []
    if len(times) == 0 or count < 2:
        return found
    bare = [vertex for vertex in range(1, count + 1) if vertex not in gated]
    if len(bare) > 0:
        found.append(
            (
                "error",
                f"{path}.advance_conditions",
                "Either every vertex has a time or none has (a vertex with another "
                f"waypoint condition counts as having one): vertex {bare[0]} has "
                "none.",
            )
        )
    ordered: list[float] = []
    for vertex in sorted(times.keys()):
        time = times[vertex]
        if vertex <= count and time is not None:
            ordered.append(time)
    decreasing = False
    for index in range(1, len(ordered)):
        if ordered[index] < ordered[index - 1]:
            decreasing = True
    if decreasing:
        found.append(
            (
                "error",
                f"{path}.advance_conditions",
                "The vertex times decrease along the path.",
            )
        )
    return found
