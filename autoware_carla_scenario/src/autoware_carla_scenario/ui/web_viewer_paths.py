"""Where the viewer keeps recordings and Rerun's web viewer, and which viewer it picks.

The path rules of the viewer's recordings and of its copy of Rerun's web viewer:
the viewer's cache directory (``VIEWER_CACHE_DIR``, else under
``XDG_CACHE_HOME`` or ``~/.cache``), where a fetched copy of a web viewer
version is kept there (:func:`web_viewer_cache_dir`), the recording beside a
trajectory log (:func:`recording_path`, ``run_outputs``'s rule); and the
choice of the web viewer ``VIEWER_RERUN_WEB_VIEWER`` makes: none, a URL
(:func:`configured_viewer`), or an unpacked package on this machine
(:func:`local_package`), with the CDN (:func:`cdn_url`) as the fallback.

Split out of ``ui.recordings``, which re-exports every name moved here and
keeps what reads the package's JSON, hashes a log's path, downloads and
unpacks the package and calls the converter, so that the library check
compiles it (``docs/typecheck.md``, "Checking the framework itself").
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# The recording beside a trajectory log is the checked run_outputs's rule
# (#1678), the one rerun names it with; ui.recordings re-exports both names.
from ..run_outputs import TRAJECTORY_SUFFIX
from ..run_outputs import recording_path as recording_path

__all__ = [
    "DEFAULT_WEB_VIEWER_VERSION",
    "LOCAL_ROUTE",
    "WebViewer",
    "cdn_url",
    "configured_viewer",
    "local_package",
    "recording_path",
    "web_viewer_cache_dir",
]

#: The Rerun version recordings are written with when the converter cannot be
#: asked (it is not installed): the line ``autoware-carla-scenario-rerun`` 0.2
#: pins (rerun-sdk >=0.38.1,<0.39).
DEFAULT_WEB_VIEWER_VERSION = "0.38.1"

#: The route this app serves a local copy under (``app.py``).
LOCAL_ROUTE = "/rerun-web-viewer"

#: The files of the npm package the viewer loads.
_WEB_VIEWER_FILES = ("index.js", "re_viewer.js", "re_viewer_bg.wasm", "package.json")

_ENV_WEB_VIEWER = "VIEWER_RERUN_WEB_VIEWER"
_ENV_CACHE = "VIEWER_CACHE_DIR"

_TRAJECTORY = TRAJECTORY_SUFFIX


@dataclass(frozen=True)
class WebViewer:
    """Where the page loads Rerun's web viewer from."""

    #: ``@rerun-io/web-viewer``'s version.
    version: str
    #: The URL of the package's directory (``index.js`` in it), or ``None``
    #: when no viewer is embedded (``VIEWER_RERUN_WEB_VIEWER=none``).
    base_url: Optional[str]
    #: ``local`` (this app serves it), ``url`` (a mirror), ``cdn`` or ``none``.
    source: str
    #: The directory this app serves it from, when ``local``.
    directory: Optional[Path] = None


def _cache_root() -> Path:
    env = os.environ.get(_ENV_CACHE)
    if env is not None and env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg is not None and xdg:
        base = Path(xdg).expanduser()
    else:
        base = Path.home() / ".cache"
    return base / "autoware_carla_scenario" / "viewer"


def web_viewer_cache_dir(version: str, root: Optional[Path] = None) -> Path:
    """Where ``viewer --fetch-rerun-web-viewer`` keeps a copy of *version*:
    under *root*, the viewer's cache directory when it is ``None``."""
    base = root if root is not None else _cache_root()
    return base / "rerun-web-viewer" / version


def _complete(directory: Path) -> bool:
    for name in _WEB_VIEWER_FILES:
        if not (directory / name).is_file():
            return False
    return True


def cdn_url(version: str) -> str:
    """Where *version* of the web viewer is loaded from when there is no copy
    on this machine: unpkg, which resolves the package's extension-less
    ``import("./re_viewer")`` (jsDelivr does not)."""
    return f"https://unpkg.com/@rerun-io/web-viewer@{version}"


def configured_viewer(env: str, version: str) -> Optional[WebViewer]:
    """The viewer the setting *env* (``VIEWER_RERUN_WEB_VIEWER``, stripped)
    names by itself: none (``none``, ``off``, ``false``, ``0``), or a URL (one
    that is not a directory here); ``None`` when it names neither."""
    if env.lower() in ("none", "off", "false", "0"):
        return WebViewer(version, None, "none")
    is_url = env.startswith("http://") or env.startswith("https://")
    if (is_url or env.startswith("/")) and not Path(env).is_dir():
        return WebViewer(version, env.rstrip("/"), "url")
    return None


def local_package(env: str, cache_dir: Path) -> Optional[Path]:
    """The first directory holding every file of the web viewer: the one the
    setting *env* names, if any, then *cache_dir* (a ``package`` directory in
    either is looked in instead, as a tarball unpacks); ``None`` without one."""
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(cache_dir)
    for directory in candidates:
        package = (
            directory / "package" if (directory / "package").is_dir() else directory
        )
        if _complete(package):
            return package
    return None
