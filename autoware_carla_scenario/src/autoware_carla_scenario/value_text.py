"""A config value as an error message shows it, a huge integer included.

Python's integers have no size limit, so a Hydra override or an unquoted YAML
integer can hand a config validator ``10**400``: ``float()`` and
``math.isfinite()`` raise ``OverflowError`` on it, and ``repr()`` raises
``ValueError`` on one of more digits than ``sys.get_int_max_str_digits()``.
A validator that refuses such a value with its keyed ``ValueError`` shows it
through :func:`shown`, which never raises (#1652); one that knows it holds
such an integer shows it through :func:`huge_integer` (``hils.config`` does).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["huge_integer", "shown"]


def shown(value: Any) -> str:
    """``repr(value)`` for an error message, or, for an integer too long for
    ``repr()`` or a value holding one, a description of it
    (``an integer of about 10**5000``)."""
    try:
        return repr(value)
    except ValueError:
        if isinstance(value, int):
            return huge_integer(value)
        return f"a {type(value).__name__} holding an integer too long to show"


def huge_integer(value: int) -> str:
    """How an error message shows an integer too long for ``repr()``, which
    refuses one of more digits than ``sys.get_int_max_str_digits()``:
    ``an integer of about 10**5000`` (or ``-10**5000``), its number of
    decimal digits worked out from its number of bits."""
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if TYPE_CHECKING:
        # Codon's int has no bit_length(): the check counts the bits.  Python
        # never runs this (a loop a bit at a time is quadratic in the size of
        # Python's integers).
        bits = 0
        while magnitude:
            magnitude >>= 1
            bits += 1
    else:
        bits = magnitude.bit_length()
    return f"an integer of about {sign}10**{int(bits * 0.30103)}"
