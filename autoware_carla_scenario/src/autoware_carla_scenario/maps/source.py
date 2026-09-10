"""Where an HD map comes from, written as one string.

A scenario names its map with a single URI so that the same value can be typed
into a form field, passed as a Hydra override (``map.source=...``), and stored
in a document without a nested structure that only one of those three can
express.

The canonical form is::

    git+https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps@splatsim#autoware_maps/Town10HD_Opt
    \\_________________ repository ________________________________/ \\_ ref _/ \\______ path ______/

Only the repository is required.  ``@ref`` selects a branch, tag or commit and
defaults to the repository's own default branch; ``#path`` selects the
directory holding the map and defaults to the repository root.

A ref that is a full commit hash *pins* the map: the same URI names the same
bytes forever, which is what makes a scenario's result reproducible.  A branch
does not -- it means "whatever is there now" -- so
:meth:`MapSource.pinned` is what the rest of this package asks before it takes
any shortcut that could hand back a different revision.

The URL a person actually has is the one in their browser's address bar, so
:meth:`MapSource.parse` also accepts the ``/tree/<ref>/<path>`` (and GitHub's
``/blob/``, and HuggingFace's ``/resolve/``) forms of both hosts and normalises
them.  Pasting the page you are looking at is the intended way in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

__all__ = ["MapSource", "MapSourceError"]


class MapSourceError(ValueError):
    """A map source URI could not be understood."""


#: The ``/<kind>/<ref>/<path...>`` infix a web UI puts between the repository
#: and the file being viewed.  ``tree`` and ``blob`` are GitHub's, ``resolve``
#: and ``raw`` are HuggingFace's download and source views.
_WEB_VIEW_SEGMENTS = ("tree", "blob", "resolve", "raw", "src", "-")

#: Repository path prefixes HuggingFace puts in front of an owner/name pair.
#: A model repository has no prefix at all, which is why this is a lookup
#: rather than a fixed number of segments to skip.
_HF_REPO_PREFIXES = ("datasets", "spaces")

_SCP_LIKE = re.compile(r"^[A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+:(?!/)")

#: A full git object name.  Abbreviated hashes are deliberately not pins: they
#: are not guaranteed to stay unambiguous as a repository grows.
_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


def _split_trailing_ref(url: str) -> tuple[str, str]:
    """Split ``...repo@ref`` into ``(repo, ref)``.

    Only an ``@`` *after* the last ``/`` is a ref, which is what keeps this from
    eating the userinfo in ``https://user@host/owner/repo`` and the user in an
    scp-like ``git@github.com:owner/repo``.
    """
    tail_start = url.rfind("/") + 1
    at = url.find("@", tail_start)
    if at == -1:
        return url, ""
    return url[:at], url[at + 1 :]


def _normalise_web_url(url: str) -> tuple[str, str, str]:
    """Return ``(repo_url, ref, path)`` for a browser URL, or the URL unchanged.

    A URL that carries no web-view segment is already a clone URL, so it comes
    back with an empty ref and path rather than being rejected: ``parse`` layers
    the explicit ``@ref#path`` on top of whatever this finds.
    """
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url, "", ""

    segments = [segment for segment in parts.path.split("/") if segment]
    # How many leading segments name the repository itself.  Two everywhere --
    # owner and name -- plus HuggingFace's optional repo-type prefix.
    repo_len = 2
    if (
        parts.netloc.endswith("huggingface.co")
        and segments[:1]
        and (segments[0] in _HF_REPO_PREFIXES)
    ):
        repo_len = 3

    if len(segments) <= repo_len or segments[repo_len] not in _WEB_VIEW_SEGMENTS:
        return url, "", ""

    repo_segments = segments[:repo_len]
    rest = segments[repo_len + 1 :]
    if not rest:
        return url, "", ""

    ref, path_segments = rest[0], rest[1:]
    repo_url = urlunsplit(
        (parts.scheme, parts.netloc, "/" + "/".join(repo_segments), "", "")
    )
    return repo_url, ref, "/".join(path_segments)


def _is_repository(url: str) -> bool:
    """Whether *url* is something git could clone.

    Deliberately as wide as git itself: an ``https://`` URL, an scp-like
    ``git@host:owner/name``, a ``file://`` URL, and a plain absolute path are
    all repositories a map can be published in -- the last two being how a team
    shares one over a filesystem, and how these tests build one.
    """
    if _SCP_LIKE.match(url) or url.startswith("/"):
        return True
    parts = urlsplit(url)
    if not parts.scheme:
        return False
    return bool(parts.netloc) or (parts.scheme == "file" and bool(parts.path))


@dataclass(frozen=True)
class MapSource:
    """A git repository, a ref inside it, and the directory holding the map.

    Attributes:
        repo_url: The URL :mod:`~autoware_carla_scenario.maps.cache` clones.
        ref: Branch, tag or commit.  Empty means the default branch.
        path: POSIX path of the map directory inside the repository.  Empty
            means the repository root.
    """

    repo_url: str
    ref: str = ""
    path: str = ""

    def __post_init__(self) -> None:
        if not self.repo_url:
            raise MapSourceError("A map source needs a repository URL.")

    # -- parsing --------------------------------------------------------

    @classmethod
    def parse(cls, uri: str) -> "MapSource":
        """Return the source *uri* names.

        Raises:
            MapSourceError: If *uri* is empty or names no repository.
        """
        text = (uri or "").strip()
        if not text:
            raise MapSourceError("A map source needs a repository URL.")
        if text.startswith("git+"):
            text = text[len("git+") :]

        path = ""
        if "#" in text:
            text, _, path = text.partition("#")

        # The explicit ref wins over one read out of a browser URL: a person who
        # wrote `@main` meant `@main`, whatever page the rest was copied from.
        text, ref = _split_trailing_ref(text)
        repo_url, web_ref, web_path = _normalise_web_url(text)
        ref = ref or web_ref
        path = path.strip("/") or web_path

        if not _is_repository(repo_url):
            raise MapSourceError(
                f"{uri!r} is not a repository URL. Expected something like "
                "https://huggingface.co/datasets/OWNER/NAME@REF#PATH"
            )

        return cls(repo_url=repo_url.rstrip("/"), ref=ref, path=path.strip("/"))

    # -- rendering ------------------------------------------------------

    @property
    def uri(self) -> str:
        """The canonical ``git+<repo>@<ref>#<path>`` form."""
        text = f"git+{self.repo_url}"
        if self.ref:
            text += f"@{self.ref}"
        if self.path:
            text += f"#{self.path}"
        return text

    @property
    def pinned(self) -> bool:
        """Whether this source names one immutable revision.

        True only for a full commit hash.  A branch or a tag can be moved, and a
        tag can be deleted and recreated, so neither fixes what a scenario runs
        against.
        """
        return bool(_FULL_COMMIT.match(self.ref))

    def at(self, ref: str) -> "MapSource":
        """Return this source pinned to *ref*."""
        return MapSource(repo_url=self.repo_url, ref=ref, path=self.path)

    @property
    def repo_name(self) -> str:
        """The repository's own name, without any owner or ``.git`` suffix."""
        tail = self.repo_url.rstrip("/").rsplit("/", 1)[-1]
        return tail[:-4] if tail.endswith(".git") else tail

    @property
    def name(self) -> str:
        """What to call this map: its directory, falling back to the repo."""
        return self.path.rsplit("/", 1)[-1] if self.path else self.repo_name

    def with_path(self, path: str) -> "MapSource":
        """Return this source pointed at a different directory in the repo."""
        return MapSource(repo_url=self.repo_url, ref=self.ref, path=path.strip("/"))

    @property
    def directory(self) -> str:
        """The repository-relative directory holding the map.

        A source may point at the ``.osm`` itself -- that is what copying a file
        URL out of a repository browser gives you -- and the directory is what
        has to be checked out either way, so that ``map_projector_info.yaml``
        comes with it.  The resolver and the catalogue both need this answer,
        and it is one answer.
        """
        if not self.path.endswith(".osm"):
            return self.path
        head, separator, _ = self.path.rpartition("/")
        return head if separator else ""

    def __str__(self) -> str:
        return self.uri
