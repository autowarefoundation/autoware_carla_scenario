"""The names the editor gives the rows it adds.

Kept apart from :mod:`.service` (which imports yaml, zipfile and pydantic)
so the library check compiles it (#1727).
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["unused_name"]


def unused_name(stem: str, taken: Sequence[str]) -> str:
    """Return ``stem_1``, ``stem_2``... -- the first one not already in use.

    A new row needs a name it can be referred to by before anyone has typed
    one, and two rows sharing a name is a validation error the author did not
    make.
    """
    existing = set(taken)
    index = 1
    while f"{stem}_{index}" in existing:
        index += 1
    return f"{stem}_{index}"
