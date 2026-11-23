"""The numbers the editor's forms post: a list field's ids, a record list's row count, and a number field.

Kept apart from :mod:`.forms` (which reads :class:`FieldSpec` and values of
any type through authoring.registry) so the library check compiles it
(#2135); :mod:`.forms` re-exports what it uses. The service's number fields
(``_as_float``, ``as_int``) read their text and messages through
:func:`float_field` and :func:`int_field` (#2386).
"""

from __future__ import annotations

import math
import re

__all__ = [
    "LIST_SEPARATORS",
    "MAX_RECORDS",
    "float_field",
    "int_field",
    "parse_int_text",
    "parse_list_int",
    "row_count",
]

#: More records than any list is written with, so a forged row count is
#: refused rather than looped over.
MAX_RECORDS = 1000

#: Separators accepted in a list-of-integers field: commas, spaces, newlines.
LIST_SEPARATORS = re.compile(r"[,\s]+")


def parse_list_int(part: str) -> int:
    """Return one item of an id list as an int.

    Raises:
        ValueError: If *part* is not a whole number.
    """
    try:
        return int(part)
    except ValueError:
        raise ValueError(f"must be whole numbers, got {part!r}") from None


def parse_int_text(text: str) -> list[int]:
    """Return the ints of a list field's *text*, split at its separators.

    Raises:
        ValueError: If an item is not a whole number, naming that item.
    """
    return [parse_list_int(part) for part in LIST_SEPARATORS.split(text) if part]


def row_count(text: str) -> int:
    """Return the row count a record list posts.

    Raises:
        ValueError: If *text* is not a whole number, or is negative or more
            than :data:`MAX_RECORDS`.
    """
    count = int(text)
    if not 0 <= count <= MAX_RECORDS:
        raise ValueError(f"{count} rows")
    return count


def float_field(text: str, shown: str, label: str, fallback: float) -> float:
    """Return a number field's stripped *text* as a float; *fallback* when empty.

    *shown* is how the message quotes what was posted (its ``repr``).

    Raises:
        ValueError: If *text* is not a number, or is not finite, with the
            user-facing message.
    """
    if not text:
        return fallback
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{label} must be a number, got {shown}.") from None
    # "nan" and "inf" parse, but no field of the editor means them: a NaN
    # timeout never fires, a NaN offset places nothing.
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number, got {shown}.")
    return number


def int_field(text: str, shown: str, label: str, fallback: int) -> int:
    """Return a whole-number field's stripped *text* as an int; *fallback* when empty.

    *shown* is how the message quotes what was posted (its ``repr``).

    Raises:
        ValueError: If *text* is not a whole number, with the user-facing
            message.
    """
    if not text:
        return fallback
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{label} must be a whole number, got {shown}.") from None
