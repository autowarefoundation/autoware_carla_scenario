"""OpenODD files kept in git repositories, pinned to a revision (openodd's).

The cache is ``$AUTOWARE_CARLA_SCENARIO_ODD_CACHE`` when set, else openodd's
(``$OPENODD_CACHE``, or ``~/.cache/openodd``).
"""

from __future__ import annotations

from openodd.sources import Checkout, GitSource, GitSourceError
from openodd.sources import fetch as _fetch

from .backend import CACHE_ENV, sync_cache_env

__all__ = ["CACHE_ENV", "Checkout", "GitSource", "GitSourceError", "fetch"]


def fetch(url: str, rev: str) -> Checkout:
    """The files of *url* at *rev*, fetched into the cache if need be."""
    sync_cache_env()
    return _fetch(url, rev)
