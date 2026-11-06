"""Whether a string of a case's config that differs on a node names the same thing.

The worker refuses a case whose config no longer resolves to the one it was
submitted with; a path that differs still counts as the same when it is the
submitting directory's seen from the job's working directory, or a file with
the content it had where it was submitted (the manifest's ``files``, by
SHA-256).  Split out of :mod:`.worker` so the library check compiles it
(#1720); comparing the configs (values of any type) stays there.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .scripts import file_sha256

__all__ = ["SameFiles"]


class SameFiles:
    """Whether a string of a case's config that differs here names the same
    thing: the submitting directory's path seen from this job's working
    directory, or a file with the content it had where it was submitted.

    *files* maps a config key to the SHA-256 its file had where the case was
    submitted; *submit_cwd* is the submitting process's working directory.
    """

    files: dict[str, str]
    prefixes: list[tuple[str, str]]
    _digests: dict[str, str]

    def __init__(self, files: dict[str, str], submit_cwd: Optional[str]) -> None:
        self.files = files
        self.prefixes = []
        if submit_cwd:
            cwd = os.getcwd()
            pwd = os.environ.get("PWD", "")  # unset or empty: the cwd
            here = {cwd, pwd if pwd else cwd}
            for directory in sorted(here):
                if directory != submit_cwd:
                    self.prefixes.append((submit_cwd.rstrip("/") + "/", directory))
        self._digests = {}

    def why_not(self, key: str, expected: str, actual: str) -> Optional[str]:
        """``None`` when *actual* names what *expected* did; else why not
        (``""`` when there is nothing more to say than that they differ)."""
        for prefix, directory in self.prefixes:
            if expected.startswith(prefix):
                moved = os.path.join(directory, expected[len(prefix) :])
                if moved == actual:
                    return None
        digest = self.files.get(key)
        if digest is None:
            return None if expected == actual else ""
        if not Path(actual).is_file():
            return f"{expected} there, {actual} here: missing on the node"
        if actual not in self._digests:
            self._digests[actual] = file_sha256(Path(actual))
        if self._digests[actual] != digest:
            return f"{expected} there, {actual} here: content differs"
        return None
