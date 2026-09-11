"""Running ``uv`` on behalf of an export.

Both halves of an export shell out to uv -- the package half locks and syncs,
the wheelhouse half exports the lock and builds wheels from it -- so the rules
about *how* uv is invoked live here rather than in either half.  There are only
two, but both are easy to get wrong and silent when they are:

* uv has to actually be installed.  Nothing downstream can invent a resolution.
* the editor's own ``VIRTUAL_ENV`` has to be dropped, or uv operates on the
  environment the editor is running in instead of the package's.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Optional

__all__ = ["UvUnavailable", "run_uv", "uv_executable", "uv_version"]


class UvUnavailable(RuntimeError):
    """Raised when ``uv`` is needed and is not installed."""


def uv_executable() -> Optional[str]:
    """Return the path to ``uv``, or ``None`` when it is not installed."""
    return shutil.which("uv")


def uv_version() -> Optional[str]:
    """Return the exact uv version, or ``None`` when it cannot be determined.

    A version is never guessed: when uv cannot be interrogated the caller
    records the absence instead of inventing a plausible number.
    """
    uv = uv_executable()
    if uv is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [uv, "--version"], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
    return match.group(1) if match else None


def run_uv(
    cwd: Path,
    *args: str,
    timeout: int,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``uv`` inside *cwd* and return the completed process.

    Args:
        cwd: Working directory for the command.
        *args: Arguments after ``uv``.
        timeout: Seconds before the command is killed.
        env: Extra environment variables, layered over the inherited ones.

    Raises:
        UvUnavailable: If uv is not installed.
    """
    uv = uv_executable()
    if uv is None:
        raise UvUnavailable(
            "uv is not installed, so the package's dependencies can be neither "
            "resolved nor built. An export needs both."
        )
    environment = dict(os.environ)
    # A parent VIRTUAL_ENV would make uv operate on the editor's environment
    # instead of the package's own.
    environment.pop("VIRTUAL_ENV", None)
    environment.update(env or {})
    return subprocess.run(  # noqa: S603
        [uv, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=environment,
    )
