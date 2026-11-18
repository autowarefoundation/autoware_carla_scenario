"""The user's cache directory, by the XDG Base Directory rules.

One lookup for every cache the package keeps under it (the bridge node's
link, the viewer's, the CommonRoad maps', the None check's mypy cache and
CARLA stubs), so that they agree on what ``XDG_CACHE_HOME`` means (#2187).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["user_cache_dir"]


def user_cache_dir() -> Path:
    """``$XDG_CACHE_HOME``, or ``~/.cache`` when it is unset, empty or relative.

    A relative ``XDG_CACHE_HOME`` -- ``""`` among them -- is ignored, as the
    XDG spec says: it would put the cache in whatever directory the process
    runs from.  A leading ``~`` is expanded first, as the viewer's and the
    CommonRoad maps' caches always did, so that a quoted ``~/x`` does not
    move them.  The home directory is looked up only when it is needed, so a
    process without one (``Path.home()`` raises ``RuntimeError``) still finds
    an absolute ``XDG_CACHE_HOME``.
    """
    xdg = os.path.expanduser(os.environ.get("XDG_CACHE_HOME", ""))
    if os.path.isabs(xdg):
        return Path(xdg)
    return Path.home() / ".cache"
