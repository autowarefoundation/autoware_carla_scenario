"""Whether an inspector form is current: the editor's revision check.

Every inspector form carries the revision of the draft it was rendered from
(docs/scenario_editor.md, "An edit, a swap, a revision").  What decides
whether a post is current -- the revision it names, who wrote the saves since
(:class:`WriterHistory`), and whether a row-addressed form still points at
its row -- is here; :mod:`.service` holds the lock and raises
``StaleDraftError``, and :mod:`.routes` reads the form.  Kept apart from
them (yaml, zipfile, pydantic, fastapi) so the library check compiles it
(#2355).
"""

from __future__ import annotations

import collections
from collections.abc import Iterator

__all__ = [
    "WRITER_DRAFTS",
    "WRITER_HISTORY",
    "WriterHistory",
    "form_revision",
    "is_current",
    "row_is_shown",
]

#: How many of a draft's latest saves the editor remembers the writer of: the
#: window in which a form one revision behind is still known to be behind only
#: its own tab's edits.  A tab has at most a request or two in flight, so this
#: is generous; a form older than the window is refused.
WRITER_HISTORY = 64

#: How many drafts' writer histories the editor keeps: the drafts most
#: recently saved.  A draft that drops out is checked as after a restart -- a
#: form one save behind is refused -- which only a tab left idle while this
#: many other drafts were edited can meet.
WRITER_DRAFTS = 256


def form_revision(text: str) -> int:
    """The revision a form's ``draft_revision`` field names, or ``-1`` when
    it names none (blank, or not a whole number)."""
    try:
        return int(text.strip())
    except ValueError:
        return -1


def is_current(
    base: int,
    revision: int,
    writers: list[tuple[int, str]],
    client: str,
    strict: bool,
) -> bool:
    """Whether a form rendered from revision *base* of a draft now at
    *revision* is current.

    It is when *base* is *revision*.  Unless *strict*, it also is when every
    save since *base* was made by *client* (the page posting the form, ``""``
    when unknown), as *writers* -- the draft's ``(revision, client)`` of its
    latest saves, oldest first -- records them: a revision *writers* does not
    hold, or holds as another page's, makes it stale.
    """
    if base == revision:
        return True
    if strict:
        return False
    if 0 <= base < revision and client:
        # Later saves of the same revision win, as dict(writers) has it.
        by_revision: dict[int, str] = {}
        for saved, writer in writers:
            by_revision[saved] = writer
        for r in range(base + 1, revision + 1):
            if r not in by_revision or by_revision[r] != client:
                return False
        return True
    return False


def row_is_shown(current: str, shown: str, renamed: str) -> bool:
    """Whether a row-addressed form still points at the row it shows.

    *current* names the row now at the form's position, *shown* the one the
    form was rendered for, and *renamed* what the form itself posts as the
    row's new name (``""`` when it renames nothing): a rename of this row
    from this form still on its way.
    """
    if current == shown:
        return True
    wanted = renamed.strip()
    return bool(wanted) and current == wanted


class WriterHistory:
    """Who wrote each of the latest revisions of the drafts saved last.

    Each draft keeps its last *history* saves (``(revision, client)``, oldest
    first), and only the *drafts* drafts saved most recently are kept: one
    saved again becomes the newest, and the oldest is forgotten first.
    Iterating yields the draft ids, least recently saved first.  Not
    thread-safe: :class:`~.service.EditorService` holds its lock around it.
    """

    _drafts: dict[str, collections.deque[tuple[int, str]]]
    _history: int

    def __init__(self, history: int = WRITER_HISTORY) -> None:
        self._drafts = {}
        self._history = history

    def record(
        self,
        draft_id: str,
        revision: int,
        client: str,
        drafts: int = WRITER_DRAFTS,
    ) -> None:
        """Remember that *client* saved *draft_id* as *revision*, making it the
        most recently saved, and forget the oldest drafts beyond *drafts*."""
        if draft_id in self._drafts:
            # Moved to the end: a dict keeps insertion order.
            writers = self._drafts.pop(draft_id)
        else:
            writers = collections.deque(maxlen=self._history)
        self._drafts[draft_id] = writers
        while len(self._drafts) > drafts:
            del self._drafts[next(iter(self._drafts))]
        writers.append((revision, client))

    def writers(self, draft_id: str) -> list[tuple[int, str]]:
        """*draft_id*'s remembered saves, oldest first (none when forgotten)."""
        if draft_id in self._drafts:
            return list(self._drafts[draft_id])
        return []

    def forget(self, draft_id: str) -> None:
        """Forget *draft_id*'s saves (the draft was deleted)."""
        if draft_id in self._drafts:
            del self._drafts[draft_id]

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._drafts))

    def __len__(self) -> int:
        return len(self._drafts)
