"""The whole numbers the editor's forms post: a list field's ids and a record list's row count.

Kept apart from :mod:`.forms` (which reads :class:`FieldSpec` and values of
any type through authoring.registry) so the library check compiles it
(#2135); :mod:`.forms` re-exports what it uses.
"""

from __future__ import annotations

import re

__all__ = [
    "LIST_SEPARATORS",
    "MAX_RECORDS",
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
