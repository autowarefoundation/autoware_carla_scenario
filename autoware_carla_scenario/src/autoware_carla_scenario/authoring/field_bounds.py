"""The numeric bounds the Scenario Editor holds a field's value to.

Whether a number lies within the bounds a field declares
(``exclusive_minimum``, ``minimum``, ``maximum``), how the refusal states
them, and the ranges a Traffic/Vehicle Parameters card's means must lie in.
Kept apart from :mod:`authoring.validator`, which imports pydantic, so the
static check compiles these rules (#2364); the validator reads the bounds off
a field's spec and reports what these return.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["bound_text", "bounds_text", "traffic_mean_bounds", "within_bounds"]


def within_bounds(
    number: float,
    exclusive_minimum: Optional[float],
    minimum: Optional[float],
    maximum: Optional[float],
) -> bool:
    """Whether *number* is above *exclusive_minimum*, at least *minimum* and
    at most *maximum*, each that is not ``None``.

    *number* may be an ``int`` too large for a float: it is compared as it
    is, never converted.  A NaN is within no bound.
    """
    if exclusive_minimum is not None and not number > exclusive_minimum:
        return False
    if minimum is not None and not number >= minimum:
        return False
    if maximum is not None and not number <= maximum:
        return False
    return True


def bound_text(is_int: bool, bound: float) -> str:
    """*bound* as the message prints it: a whole number field's in full (a
    seed's ``2147483647``, not ``2.14748e+09``), any other's with ``:g``.

    Args:
        is_int: Whether the field is an ``int`` one.
        bound: The bound to print.
    """
    if is_int and _is_whole(float(bound)):
        return str(int(bound))
    return f"{bound:g}"


def _is_whole(value: float) -> bool:
    """``value.is_integer()``, which Codon's float lacks: a finite value with
    no fractional part."""
    return math.isfinite(value) and math.floor(value) == value


def bounds_text(
    is_int: bool,
    exclusive_minimum: Optional[float],
    minimum: Optional[float],
    maximum: Optional[float],
) -> str:
    """What :func:`within_bounds` asks of a value, as a predicate: the
    message of a value out of bounds ("must be positive" for an exclusive
    bound of 0, "must not be negative" for an inclusive one).

    Only called for a value out of bounds, so at least one bound is given.
    """
    low, inclusive, high = exclusive_minimum, minimum, maximum
    if high is not None and (low is not None or inclusive is not None):
        if low is not None:
            opening = f"({bound_text(is_int, low)}"
        else:
            assert inclusive is not None  # noqa: S101 -- by the test above
            opening = f"[{bound_text(is_int, inclusive)}"
        return f"must be in {opening}, {bound_text(is_int, high)}]"
    if high is not None:
        return f"must be at most {bound_text(is_int, high)}"
    if low is not None:
        return (
            "must be positive"
            if low == 0
            else f"must be above {bound_text(is_int, low)}"
        )
    assert inclusive is not None  # noqa: S101 -- only called out of bounds
    return (
        "must not be negative"
        if inclusive == 0
        else f"must be at least {bound_text(is_int, inclusive)}"
    )


def traffic_mean_bounds() -> dict[str, tuple[float, float]]:
    """``*_mean`` field of a Traffic/Vehicle Parameters card -> the
    ``(low, high)`` its parameter's mean must lie in.

    The bounds are the runtime's own tables (``VEHICLE_BOUNDS`` and the
    rest), which the parameter classes check every mean against when the
    action is built; the field names are what ``traffic.authoring`` maps
    onto them (a pedestrian's gap is ``pedestrian_min_gap_m``, every bicycle
    field ``bicycle_*``).  The tables are imported here, as the validator
    imported them, so importing this module does not import the traffic
    package.
    """
    from ..traffic.parameters import (  # noqa: PLC0415
        BICYCLE_BOUNDS,
        PEDESTRIAN_BOUNDS,
        VEHICLE_BOUNDS,
    )

    bounds = {f"{name}_mean": span for name, span in VEHICLE_BOUNDS.items()}
    for name, span in PEDESTRIAN_BOUNDS.items():
        field_name = "pedestrian_min_gap_m" if name == "min_gap_m" else name
        bounds[f"{field_name}_mean"] = span
    for name, span in BICYCLE_BOUNDS.items():
        bounds[f"bicycle_{name}_mean"] = span
    return bounds
