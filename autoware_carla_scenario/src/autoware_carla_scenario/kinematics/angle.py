"""Angle wrapping shared by anything that compares two headings."""

from __future__ import annotations

__all__ = ["normalize_angle_deg"]


def normalize_angle_deg(angle: float) -> float:
    """Return *angle* wrapped into ``[-180, 180)``.

    Written once because a heading difference is meaningless until it is
    wrapped: 359 degrees apart is one degree apart, and every place that
    compares two yaws needs the same answer.
    """
    return (angle + 180.0) % 360.0 - 180.0
