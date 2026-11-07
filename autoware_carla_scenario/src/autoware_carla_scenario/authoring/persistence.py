"""Reading, writing and drafting :class:`ScenarioDocument` files.

Drafts are plain YAML files in a directory -- there is no database.  A draft
wraps the document with a little bookkeeping (title, timestamps) so the editor
can list them; the document itself is stored verbatim under a ``document`` key,
so a draft file and an exported ``scenario/document.yaml`` hold the same shape.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

try:  # The C parser when PyYAML was built with libyaml -- ~10x on a draft.
    from yaml import CSafeDumper as _SafeDumper
    from yaml import CSafeLoader as _SafeLoader
except ImportError:  # pragma: no cover -- depends on the installed PyYAML
    from yaml import SafeDumper as _SafeDumper  # type: ignore[assignment]
    from yaml import SafeLoader as _SafeLoader  # type: ignore[assignment]

from .models import ScenarioDocument, new_object_id

#: The tag PyYAML resolves a plain boolean scalar to.
_BOOL_TAG = "tag:yaml.org,2002:bool"


class DocumentLoader(_SafeLoader):  # type: ignore[misc,valid-type]
    """PyYAML's safe loader with YAML 1.2's booleans: only ``true`` / ``false``.

    YAML 1.1, which PyYAML implements, also reads a plain ``yes`` / ``no`` /
    ``on`` / ``off`` (in any of three casings) as a boolean, and several
    documented options are exactly those words: a ``traffic_signal`` state
    ``Off``, a ``turn_indicator`` ``OFF``, a signal phase's ``off``, a route
    segment's ``yes`` / ``no``.  Unquoted, they were loaded as ``False`` /
    ``True`` and refused (#1946).  Here they stay text, and a ``bool`` field
    still reads them: a parameter through
    :func:`~autoware_carla_scenario.authoring.registry.read_bool`, a model
    field through pydantic, both of which take ``yes`` / ``no`` / ``on`` /
    ``off``.  Everything else (nulls, numbers, timestamps) resolves as before,
    and an explicit ``!!bool`` tag still makes a boolean.

    Writing needs no counterpart: the dumper quotes a string YAML 1.1 would
    read as a boolean, so what this package writes reads the same here, in
    PyYAML's own loader and in OmegaConf's.
    """


DocumentLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != _BOOL_TAG]
    for first, resolvers in _SafeLoader.yaml_implicit_resolvers.items()
}
DocumentLoader.add_implicit_resolver(
    _BOOL_TAG, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def load_yaml(text: str) -> Any:
    """Return the YAML *text* as Python data, read as a document is read.

    See :class:`DocumentLoader`: only ``true`` / ``false`` are booleans.

    Raises:
        yaml.YAMLError: If *text* is not YAML, or holds a value Python cannot
            build: an integer longer than ``sys.get_int_max_str_digits()``
            allows, a date that is no date, nesting deeper than the recursion
            limit.  The constructors raise ``ValueError`` (or
            ``RecursionError``) for those, which every caller here -- the
            editor's forms, the draft store -- would otherwise let escape as a
            500 (#1963).
    """
    try:
        return yaml.load(text, Loader=DocumentLoader)  # noqa: S506 - a SafeLoader
    except (ValueError, RecursionError) as exc:
        raise yaml.YAMLError(f"cannot be read: {exc}") from exc


logger = logging.getLogger(__name__)

__all__ = [
    "DocumentLoader",
    "Draft",
    "DraftStore",
    "default_draft_dir",
    "dump_document_yaml",
    "dump_yaml",
    "load_document",
    "load_yaml",
    "save_document",
    "utc_timestamp",
]

#: Environment variable that relocates the draft directory.
DRAFT_DIR_ENV = "SCENARIO_EDITOR_DRAFTS"

#: Matched with ``fullmatch``: ``$`` would also match before a trailing newline.
_DRAFT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def default_draft_dir() -> Path:
    """Return the directory drafts are stored in."""
    env = os.environ.get(DRAFT_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "scenario_drafts").resolve()


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dump_yaml(data: Any) -> str:
    """Return *data* as YAML the way every file this package writes is written.

    Key order is the order the writer chose, because a document, a Hydra config
    and a manifest all read top-down; wrapping is wide enough that a path or an
    interpolation stays on one line.
    """
    return yaml.dump(
        data, Dumper=_SafeDumper, sort_keys=False, allow_unicode=True, width=100
    )


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
    raw = load_yaml(file_path.read_text(encoding="utf-8"))
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
            created_at=str(raw.get("created_at") or utc_timestamp()),
            updated_at=str(raw.get("updated_at") or utc_timestamp()),
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
        if not _DRAFT_ID_PATTERN.fullmatch(draft_id):
            raise ValueError(f"Invalid draft id: {draft_id!r}")
        return self.root / f"{draft_id}.yaml"

    # -- reads ----------------------------------------------------------

    def get(self, draft_id: str) -> Optional[Draft]:
        """Return the draft, or ``None`` when it does not exist.

        The draft's id is *draft_id*, the file's name, whatever id the file
        stores: a draft saves back to the file it was read from, and a stored
        id that differs (or is not a valid id at all) would otherwise send the
        save to another file, or fail it.

        Raises:
            ValueError: If the file cannot be read (no permission, not UTF-8,
                an I/O error), is not YAML, or is not a mapping with a
                ``document`` mapping (what :meth:`save` writes), so is not a
                draft (the drafts directory may hold other YAML; :meth:`list`
                skips such files).
        """
        path = self.path_for(draft_id)
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # A ValueError for the same reason as below: one unreadable file
            # (a permission the server lacks, a binary file) is that file's
            # problem, not a 500 for the request that named it (#1747).
            raise ValueError(f"{path}: cannot be read ({exc})") from exc
        try:
            raw = load_yaml(text) or {}
        except yaml.YAMLError as exc:
            # A ValueError like the rest, so the editor answers a draft id
            # naming a broken file with "not found" rather than a 500.
            raise ValueError(f"{path}: not a draft ({exc})") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("document"), dict):
            # Other YAML is mostly a mapping too (a config, notes): without a
            # document it is not a draft, so not the store's to show or delete.
            raise ValueError(f"{path}: not a draft")
        raw["id"] = draft_id
        return Draft.from_dict(raw)

    def list(self) -> list[Draft]:
        """Return every stored draft, most recently updated first.

        A file that is not a draft is skipped, and so is one that cannot be
        read -- with a warning, since that one may well be a draft the server
        lacks the permission to open -- so one bad file never hides the rest
        (#1747).
        """
        if not self.root.is_dir():
            return []
        drafts: list[Draft] = []
        for path in sorted(self.root.glob("*.yaml")):
            try:
                draft = self.get(path.stem)
            except (OSError, UnicodeDecodeError) as exc:
                # get() turns a failed read into a ValueError; is_file() and
                # anything below it can still raise for a file that vanishes
                # or turns unreadable mid-listing.
                logger.warning("Skipping %s: cannot be read (%s)", path, exc)
                continue
            except (ValueError, yaml.YAMLError) as exc:
                if isinstance(exc.__cause__, (OSError, UnicodeDecodeError)):
                    logger.warning("Skipping %s", exc)
                continue
            if draft is not None:
                drafts.append(draft)
        return sorted(drafts, key=lambda d: d.updated_at, reverse=True)

    # -- writes ---------------------------------------------------------

    def create(self, document: ScenarioDocument, title: str | None = None) -> Draft:
        """Store *document* as a new draft and return it."""
        stamp = utc_timestamp()
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
        draft.updated_at = utc_timestamp()
        draft.title = draft.document.title or draft.title
        self._write(draft)
        return draft

    def delete(self, draft_id: str) -> bool:
        """Remove a draft.  Returns whether anything was deleted.

        Raises:
            ValueError: If *draft_id* is not a valid id, or its file is not a
                draft (:meth:`get` refuses it): the drafts directory may hold
                other YAML, which is not the store's to delete.
        """
        if self.get(draft_id) is None:
            return False
        self.path_for(draft_id).unlink()
        return True

    def _write(self, draft: Draft) -> None:
        """Write *draft* whole or not at all.

        Written aside and renamed into place, so a request reading the draft
        while another saves it -- the editor serves both at once -- never
        parses half a file.  The temporary name does not end in ``.yaml``, so
        :meth:`list` never mistakes it for a draft.

        Raises:
            OSError: If the drafts directory cannot be created or written to
                (no permission, a read-only file system, a full disk).  The
                stored draft, if any, is untouched, and the temporary file is
                removed as far as the directory allows.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(draft.id)
        partial = self.root / f".{draft.id}.{uuid.uuid4().hex}.partial"
        try:
            with partial.open("x", encoding="utf-8") as stream:
                stream.write(dump_yaml(draft.to_dict()))
            os.replace(partial, path)
        finally:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                # Best effort: a directory that refused the write may refuse
                # this too, and the error worth raising is the write's.
                logger.warning("Could not remove %s", partial)
