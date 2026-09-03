"""Reading, writing and drafting :class:`ScenarioDocument` files.

Drafts are plain YAML files in a directory -- there is no database.  A draft
wraps the document with a little bookkeeping (title, timestamps) so the editor
can list them; the document itself is stored verbatim under a ``document`` key,
so a draft file and an exported ``scenario/document.yaml`` hold the same shape.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .models import ScenarioDocument, new_object_id

__all__ = [
    "Draft",
    "DraftStore",
    "default_draft_dir",
    "dump_document_yaml",
    "dump_yaml",
    "load_document",
    "save_document",
]

#: Environment variable that relocates the draft directory.
DRAFT_DIR_ENV = "SCENARIO_EDITOR_DRAFTS"

_DRAFT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def default_draft_dir() -> Path:
    """Return the directory drafts are stored in."""
    env = os.environ.get(DRAFT_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "scenario_drafts").resolve()


def _now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dump_yaml(data: Any) -> str:
    """Return *data* as YAML the way every file this package writes is written.

    Key order is the order the writer chose, because a document, a Hydra config
    and a manifest all read top-down; wrapping is wide enough that a path or an
    interpolation stays on one line.
    """
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def dump_document_yaml(document: ScenarioDocument) -> str:
    """Return *document* as a YAML string with stable key order."""
    return dump_yaml(document.to_yaml_dict())


def load_document(path: str | Path) -> ScenarioDocument:
    """Load a :class:`ScenarioDocument` from a YAML file.

    Accepts both a bare document and a draft wrapper (``{document: {...}}``),
    so an exported package can read the same file the editor wrote.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is empty or not a mapping.
    """
    file_path = Path(path)
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{file_path} does not contain a scenario document mapping.")
    if "document" in raw and isinstance(raw["document"], dict):
        raw = raw["document"]
    return ScenarioDocument.model_validate(raw)


def save_document(document: ScenarioDocument, path: str | Path) -> Path:
    """Write *document* to *path* as YAML, creating parent directories."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(dump_document_yaml(document), encoding="utf-8")
    return file_path


@dataclass
class Draft:
    """A scenario document plus the editor's bookkeeping."""

    id: str
    title: str
    created_at: str
    updated_at: str
    document: ScenarioDocument

    def to_dict(self) -> dict[str, Any]:
        """Return the on-disk representation."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "document": self.document.to_yaml_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Draft":
        """Rebuild a draft from its on-disk representation."""
        document = ScenarioDocument.model_validate(raw.get("document") or {})
        return cls(
            id=str(raw.get("id") or new_object_id("draft")),
            title=str(raw.get("title") or document.title),
            created_at=str(raw.get("created_at") or _now()),
            updated_at=str(raw.get("updated_at") or _now()),
            document=document,
        )


class DraftStore:
    """A directory of draft YAML files.

    Args:
        root: Directory holding the drafts.  Created on first write.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_draft_dir()

    # -- paths ----------------------------------------------------------

    def path_for(self, draft_id: str) -> Path:
        """Return the file backing *draft_id*.

        Raises:
            ValueError: If *draft_id* is not a safe file-name fragment.  Draft
                ids arrive from the URL, so this is the boundary that keeps a
                request from reaching outside the store.
        """
        if not _DRAFT_ID_PATTERN.match(draft_id):
            raise ValueError(f"Invalid draft id: {draft_id!r}")
        return self.root / f"{draft_id}.yaml"

    # -- reads ----------------------------------------------------------

    def exists(self, draft_id: str) -> bool:
        """Whether a draft with this id is stored."""
        try:
            return self.path_for(draft_id).is_file()
        except ValueError:
            return False

    def get(self, draft_id: str) -> Optional[Draft]:
        """Return the draft, or ``None`` when it does not exist."""
        path = self.path_for(draft_id)
        if not path.is_file():
            return None
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw.setdefault("id", draft_id)
        return Draft.from_dict(raw)

    def list(self) -> list[Draft]:
        """Return every stored draft, most recently updated first."""
        if not self.root.is_dir():
            return []
        drafts: list[Draft] = []
        for path in sorted(self.root.glob("*.yaml")):
            try:
                draft = self.get(path.stem)
            except (ValueError, yaml.YAMLError):
                continue
            if draft is not None:
                drafts.append(draft)
        return sorted(drafts, key=lambda d: d.updated_at, reverse=True)

    # -- writes ---------------------------------------------------------

    def create(self, document: ScenarioDocument, title: str | None = None) -> Draft:
        """Store *document* as a new draft and return it."""
        stamp = _now()
        draft = Draft(
            id=new_object_id("draft"),
            title=title or document.title,
            created_at=stamp,
            updated_at=stamp,
            document=document,
        )
        self._write(draft)
        return draft

    def save(self, draft: Draft) -> Draft:
        """Persist *draft*, refreshing its ``updated_at`` stamp."""
        draft.updated_at = _now()
        draft.title = draft.document.title or draft.title
        self._write(draft)
        return draft

    def delete(self, draft_id: str) -> bool:
        """Remove a draft.  Returns whether anything was deleted."""
        path = self.path_for(draft_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _write(self, draft: Draft) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path_for(draft.id).write_text(dump_yaml(draft.to_dict()), encoding="utf-8")
