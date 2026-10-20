"""Angle wrapping shared by anything that compares two headings."""

from __future__ import annotations

import math

__all__ = ["normalize_angle_deg", "normalize_angle_rad"]


def normalize_angle_deg(angle: float) -> float:
    """Return *angle* wrapped into ``[-180, 180)``.

    Written once because a heading difference is meaningless until it is
    wrapped: 359 degrees apart is one degree apart, and every place that
    compares two yaws needs the same answer.
    """
    return (angle + 180.0) % 360.0 - 180.0


def normalize_angle_rad(angle: float) -> float:
    """Return *angle* (radians) wrapped into ``[-pi, pi)``.

    :func:`normalize_angle_deg` for radians.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
