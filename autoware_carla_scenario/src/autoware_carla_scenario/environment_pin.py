"""The requirement a run's lock pins a distribution to, from where it was
installed from.

The string half of :mod:`.environment`, which reads the installed
distributions (``importlib.metadata``, PEP 610's ``direct_url.json``, git) and
writes the environment and the lock beside a run's result: a distribution's
source as a typed record (:class:`PinSource`), the normalised form of its
name, and the requirement line -- with the URL fragment pip and uv read a hash
and a subdirectory from -- that installs it again.  Nothing here touches a
file, a process or the network, so the library check compiles it
(docs/typecheck.md); :mod:`.environment` uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["HASH_PREFERENCE", "PinSource", "fragment", "normalize", "pin"]


def normalize(name: str) -> str:
    """*name* as the environment keys a distribution: lower case, ``-`` for
    ``_`` and ``.``."""
    return name.lower().replace("_", "-").replace(".", "-")


#: Archive hashes in the order a lock prefers them, strongest first; pip and uv
#: both read any of them from a URL fragment.
HASH_PREFERENCE: Tuple[str, str, str, str, str, str] = (
    "sha512",
    "sha384",
    "sha256",
    "sha224",
    "sha1",
    "md5",
)


@dataclass(frozen=True)
class PinSource:
    """Where a distribution was installed from (PEP 610's ``direct_url.json``).

    ``kind`` is ``index`` for a registry release; ``vcs`` with the ``url``,
    the ``vcs``, the ``commit`` and the revision that was ``requested``;
    ``archive`` with the ``url`` and its ``hashes``; ``editable`` / ``local``
    with the directory (``path``), and its git ``commit`` and uncommitted
    changes (``dirty``) when it is a checkout.  A ``subdirectory`` -- the
    project inside a monorepo -- is kept where the URL does not name it.
    An empty field is one the source does not have.
    """

    kind: str
    url: str = ""
    vcs: str = ""
    commit: str = ""
    requested: Optional[str] = None
    path: Optional[str] = None
    dirty: bool = False
    subdirectory: str = ""
    hashes: Dict[str, str] = field(default_factory=dict)


def fragment(source: PinSource) -> str:
    """The URL fragment pip and uv read a hash and a subdirectory from."""
    parts: List[str] = []
    hashes = source.hashes
    for algorithm in HASH_PREFERENCE:
        if algorithm in hashes and hashes[algorithm]:
            parts.append(f"{algorithm}={hashes[algorithm]}")
            break
    if source.subdirectory:
        parts.append(f"subdirectory={source.subdirectory}")
    return "#" + "&".join(parts) if parts else ""


def pin(name: str, version: str, source: PinSource) -> str:
    """The requirement that installs this distribution again, as far as one can."""
    kind = source.kind
    if kind == "vcs":
        # The commit is what was installed; the requested revision (a branch
        # or tag, which can move) is only the fallback when none was recorded.
        revision = source.commit
        if not revision and source.requested is not None:
            revision = source.requested
        if revision:
            text = f"{name} @ {source.vcs}+{source.url}@{revision}{fragment(source)}"
            if not source.commit:
                text += f"  # requested {revision}, no commit recorded: may move"
            return text
    if kind == "archive":
        return f"{name} @ {source.url}{fragment(source)}"
    if kind == "editable" or kind == "local":
        state = ""
        if source.commit:
            state = f", git {source.commit}" + (
                " with uncommitted changes" if source.dirty else ""
            )
        return f"{name}=={version}  # {kind} from {source.path}{state}"
    return f"{name}=={version}"
