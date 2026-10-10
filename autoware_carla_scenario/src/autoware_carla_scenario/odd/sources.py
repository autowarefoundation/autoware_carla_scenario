"""OpenODD files kept in git repositories, pinned to a revision.

A binding file names them next to local files::

    openodd:
      - git: https://example.com/odd/taxonomy.git
        rev: v1.2.0                 # a tag, a branch or a commit
        path: taxonomy.yml          # in the repository; a list for several
      - odd.yml                     # relative to the binding file

Each repository is fetched once into a cache, and each commit is checked out
once, read-only in spirit: the files of a commit never change, so a commit
already checked out is used without the network.  A tag or a branch is
fetched on every load, to learn which commit it names now.

The cache is ``$AUTOWARE_CARLA_SCENARIO_ODD_CACHE``, or
``~/.cache/autoware_carla_scenario/openodd``.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Union

CACHE_ENV = "AUTOWARE_CARLA_SCENARIO_ODD_CACHE"
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
#: Git refuses ``ext::`` by default; this keeps it refused whatever the
#: user's configuration says, so a URL never runs a command.
_GIT = ("git", "-c", "protocol.ext.allow=never")


class GitSourceError(ValueError):
    """A git source could not be fetched, or names a file it does not have."""


@dataclass(frozen=True)
class GitSource:
    """OpenODD files in a git repository at a revision.

    Args:
        url: Anything ``git fetch`` takes: an https or ssh URL, or a path.
        rev: A tag, a branch or a commit.  A full commit id is fetched once
            and then read from the cache.
        path: The file in the repository; its ``IMPORT`` entries are read
            relative to it.
    """

    url: str
    rev: str
    path: str

    def __post_init__(self) -> None:
        for key in ("url", "rev", "path"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise GitSourceError(f"git source: {key} must be a non-empty string")
            if value.startswith("-") or any(c in value for c in "\n\r\0"):
                raise GitSourceError(f"git source: bad {key} {value!r}")


@dataclass(frozen=True)
class Checkout:
    """A commit's files in the cache."""

    root: Path
    commit: str


def cache_dir() -> Path:
    """Where repositories and checkouts are cached."""
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "autoware_carla_scenario" / "openodd"


def _git(*args: Union[str, Path], cwd: Optional[Path] = None) -> str:
    try:
        done = subprocess.run(
            [*_GIT, *map(str, args)],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except FileNotFoundError as exc:
        raise GitSourceError("git source: git is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise GitSourceError(
            f"git {' '.join(map(str, args[:2]))}: {exc.stderr.strip()}"
        ) from exc
    return done.stdout.strip()


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on *path* (runs in parallel share the cache)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _resolve(repo: Path, url: str, rev: str) -> str:
    """Fetch *rev* from *url* into the bare *repo*; the commit it names."""
    try:
        _git("fetch", "--depth", "1", "--", url, rev, cwd=repo)
        return _git("rev-parse", "--verify", "FETCH_HEAD^{commit}", cwd=repo)
    except GitSourceError:
        pass  # an abbreviated commit id, or a server refusing to fetch one
    unshallow = ["--unshallow"] if (repo / "shallow").exists() else []
    _git(
        "fetch",
        *unshallow,
        "--",
        url,
        "+refs/heads/*:refs/remotes/origin/*",
        "+refs/tags/*:refs/tags/*",  # forced: a tag moved upstream moves here
        cwd=repo,
    )
    for candidate in (rev, f"origin/{rev}"):
        try:
            return _git("rev-parse", "--verify", f"{candidate}^{{commit}}", cwd=repo)
        except GitSourceError:
            continue
    raise GitSourceError(f"git source {url}: no revision {rev!r}")


def _extract(repo: Path, commit: str, dest: Path) -> None:
    """Write *commit*'s files to *dest*, atomically."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{commit[:12]}-", dir=dest.parent))
    try:
        data = subprocess.run(
            [*_GIT, "archive", "--format=tar", commit],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if data.strip(b"\0"):  # an empty tree archives to padding alone
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(tmp, filter="data")
                else:  # pragma: no cover - Pythons before the filter backport
                    tar.extractall(tmp)  # noqa: S202 - reads check containment
        try:
            tmp.rename(dest)
        except OSError:
            if not dest.exists():
                raise
    except (subprocess.CalledProcessError, tarfile.TarError) as exc:
        detail = getattr(exc, "stderr", b"") or str(exc).encode()
        raise GitSourceError(
            f"git source: cannot check out {commit}: "
            f"{detail.decode(errors='replace').strip()}"
        ) from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch(url: str, rev: str) -> Checkout:
    """The files of *url* at *rev*, fetched into the cache if need be."""
    GitSource(url, rev, ".")  # validates
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    root = cache_dir()
    trees = root / "trees" / key
    if _FULL_SHA.match(rev.lower()) and (trees / rev.lower()).is_dir():
        return Checkout(trees / rev.lower(), rev.lower())
    repo = root / "repos" / f"{key}.git"
    with _locked(root / "repos" / f"{key}.lock"):
        if not (repo / "HEAD").exists():
            repo.mkdir(parents=True, exist_ok=True)
            _git("init", "--quiet", "--bare", cwd=repo)
        commit = _resolve(repo, url, rev)
        _extract(repo, commit, trees / commit)
    return Checkout(trees / commit, commit)


def checkout(source: GitSource) -> tuple[Path, Checkout]:
    """The file *source* names, and the checkout it is in."""
    co = fetch(source.url, source.rev)
    path = (co.root / source.path).resolve()
    if not path.is_relative_to(co.root.resolve()):
        raise GitSourceError(
            f"git source {source.url}: {source.path} leaves the repository"
        )
    if not path.is_file():
        raise GitSourceError(
            f"git source {source.url}@{source.rev}: no file {source.path}"
        )
    return path, co


def parse_entry(entry: Any, where: str) -> list[GitSource]:
    """The git sources of one binding-file ``openodd`` entry (a mapping)."""
    extra = sorted(str(k) for k in entry if k not in ("git", "rev", "path"))
    if extra:
        raise GitSourceError(f"{where}: unknown keys {extra} in a git source")
    missing = [k for k in ("git", "rev", "path") if not entry.get(k)]
    if missing:
        raise GitSourceError(f"{where}: a git source needs {missing}")
    paths = entry["path"] if isinstance(entry["path"], list) else [entry["path"]]
    return [GitSource(str(entry["git"]), str(entry["rev"]), str(p)) for p in paths]
