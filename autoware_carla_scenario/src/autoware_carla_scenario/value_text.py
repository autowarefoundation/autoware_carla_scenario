"""A config value as an error message shows it, a huge integer included.

Python's integers have no size limit, so a Hydra override or an unquoted YAML
integer can hand a config validator ``10**400``: ``float()`` and
``math.isfinite()`` raise ``OverflowError`` on it, and ``repr()`` raises
``ValueError`` on one of more digits than ``sys.get_int_max_str_digits()``.
A validator that refuses such a value with its keyed ``ValueError`` shows it
through :func:`shown`, which never raises (#1652).
"""

from __future__ import annotations

from typing import Any

__all__ = ["shown"]


def shown(value: Any) -> str:
    """``repr(value)`` for an error message, or, for an integer too long for
    ``repr()`` or a value holding one, a description of it
    (``an integer of about 10**5000``)."""
    try:
        return repr(value)
    except ValueError:
        if isinstance(value, int):
            sign = "-" if value < 0 else ""
            digits = int(abs(value).bit_length() * 0.30103)
            return f"an integer of about {sign}10**{digits}"
        return f"a {type(value).__name__} holding an integer too long to show"
