"""The editor's revision check, moved out of editor.service and editor.routes
so the library check compiles it (#2355): it decides as the code it replaced
did, which is copied here to compare with."""

from __future__ import annotations

import collections
import random
from typing import Any, Optional

import pytest

from autoware_carla_scenario.editor import service
from autoware_carla_scenario.editor.revisions import (
    WRITER_DRAFTS,
    WRITER_HISTORY,
    WriterHistory,
    form_revision,
    is_current,
    row_is_shown,
)


class _OldWriters:
    """EditorService's writer memory before #2355."""

    def __init__(self, history: int, drafts: int) -> None:
        self.history = history
        self.drafts = drafts
        self.writers: collections.OrderedDict[
            str, collections.deque[tuple[int, str]]
        ] = collections.OrderedDict()

    def save(self, draft_id: str, revision: int, client: str) -> None:
        writers = self.writers.get(draft_id)
        if writers is None:
            writers = collections.deque(maxlen=self.history)
            self.writers[draft_id] = writers
        self.writers.move_to_end(draft_id)
        while len(self.writers) > self.drafts:
            self.writers.popitem(last=False)
        writers.append((revision, client))

    def delete(self, draft_id: str) -> None:
        self.writers.pop(draft_id, None)

    def is_current(
        self, draft_id: str, current: int, revision: Any, client: str, strict: bool
    ) -> bool:
        try:
            base = int(str(revision).strip())
        except ValueError:
            base = -1
        if base == current:
            return True
        if strict:
            return False
        if 0 <= base < current and client:
            known = dict(self.writers.get(draft_id, ()))
            between = range(base + 1, current + 1)
            if all(known.get(r) == client for r in between):
                return True
        return False


def _old_row_is_shown(current: str, shown: str, renamed_by: Optional[str]) -> bool:
    if current == shown:
        return True
    renamed = renamed_by.strip() if renamed_by else ""
    return bool(renamed and current == renamed)


def test_the_service_keeps_the_limits() -> None:
    assert service._WRITER_HISTORY == WRITER_HISTORY == 64
    assert service._WRITER_DRAFTS == WRITER_DRAFTS == 256


@pytest.mark.parametrize("seed", range(20))
def test_the_history_keeps_and_evicts_as_before(seed: int) -> None:
    rng = random.Random(seed)
    history, drafts = rng.randint(1, 6), rng.randint(1, 5)
    old = _OldWriters(history, drafts)
    new = WriterHistory(history)
    ids = [f"d{i}" for i in range(drafts + 3)]
    clients = ["", "tab-a", "tab-b"]
    revisions = dict.fromkeys(ids, 0)
    for _ in range(400):
        draft_id = rng.choice(ids)
        if rng.random() < 0.05:
            old.delete(draft_id)
            new.forget(draft_id)
            revisions[draft_id] = 0
        else:
            revisions[draft_id] += 1
            client = rng.choice(clients)
            old.save(draft_id, revisions[draft_id], client)
            new.record(draft_id, revisions[draft_id], client, drafts)
        assert list(new) == list(old.writers)
        assert len(new) == len(old.writers)
        for other in ids:
            assert new.writers(other) == list(old.writers.get(other, ()))
        # And every form decides as it did.
        for other in ids:
            current = revisions[other]
            for base in (current, current - 1, current - 2, current - 5, -1):
                for client in clients:
                    for strict in (False, True):
                        assert is_current(
                            form_revision(str(base)),
                            current,
                            new.writers(other),
                            client,
                            strict,
                        ) == old.is_current(other, current, base, client, strict)


@pytest.mark.parametrize(
    "text", ["3", " 3 ", "+3", "-1", "", "x", "3.0", "1_0", "None", "\t7\n"]
)
def test_a_forms_revision_reads_as_before(text: str) -> None:
    try:
        expected = int(str(text).strip())
    except ValueError:
        expected = -1
    assert form_revision(text) == expected


def test_a_repeated_revision_reads_its_last_writer() -> None:
    writers = [(2, "tab-b"), (2, "tab-a")]
    assert is_current(1, 2, writers, "tab-a", False)
    assert not is_current(1, 2, writers, "tab-b", False)


@pytest.mark.parametrize("current", ["a", "b", " a "])
@pytest.mark.parametrize("shown", ["a", "b"])
@pytest.mark.parametrize("renamed", [None, "", "a", " a ", "b", "  "])
def test_a_row_is_shown_as_before(
    current: str, shown: str, renamed: Optional[str]
) -> None:
    assert row_is_shown(current, shown, renamed or "") == _old_row_is_shown(
        current, shown, renamed
    )
