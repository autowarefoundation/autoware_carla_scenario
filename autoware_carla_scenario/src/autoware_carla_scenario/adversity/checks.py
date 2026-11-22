"""The checks an adversity's settings make of their numbers (#2318).

Shared by :mod:`.avoidance`, :mod:`.construction`, :mod:`.idm`, :mod:`.nade`
and :mod:`.triggers`, which refuse a field's value with these words.
"""

from __future__ import annotations

import math

__all__ = ["check_non_negative", "check_positive"]


def check_non_negative(name: str, value: float) -> None:
    """Refuse *value* for the field *name* unless it is finite and >= 0."""
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and not negative, not {value}")


def check_positive(name: str, value: float) -> None:
    """Refuse *value* for the field *name* unless it is finite and > 0."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, not {value}")
