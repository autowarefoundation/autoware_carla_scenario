"""Rerun recordings of a case's trajectory log, and the web viewer they open in.

A case's trajectory log (``X.trajectory.jsonl``) is made into a Rerun
recording (``X.rrd`` beside it) by the ``autoware_carla_scenario_rerun``
package, which this module imports only when it converts: the viewer runs
without it and says how to install it.  A recording already beside the log
(made by the run, ``rerun.enabled``, or by hand) is served as it is, and one
made here is kept there for the next time (or, where the session's directory
cannot be written, in the user's cache directory).  The conversion runs in Rust
with the GIL released, so the viewer converts in a worker thread
(``asyncio.to_thread``) and goes on serving.

The page shows the recording in Rerun's web viewer, the ``@rerun-io/web-viewer``
package of the same version as the Rerun SDK the converter writes with
(:func:`web_viewer_version`; a recording opens only in a viewer of its minor
version).  The viewer's files (a 51 MB wasm module) are served by this app when
there is a copy of them on this machine -- ``VIEWER_RERUN_WEB_VIEWER`` naming
an unpacked package, or the copy ``viewer --fetch-rerun-web-viewer`` keeps in
the cache directory -- and otherwise loaded from the unpkg CDN (as the page's htmx is), pinned to
that version.  ``VIEWER_RERUN_WEB_VIEWER`` may also be a URL (a mirror of the
package), or ``none`` to embed no viewer at all.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import tarfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CDN_URL",
    "DEFAULT_WEB_VIEWER_VERSION",
    "INSTALL_HINT",
    "ConverterMissingError",
    "RecordingError",
    "WebViewer",
    "converter_available",
    "converter_version",
    "ensure_recording",
    "fetch_web_viewer",
    "recording_path",
    "web_viewer",
    "web_viewer_cache_dir",
    "web_viewer_version",
]

logger = logging.getLogger(__name__)

#: The Rerun version recordings are written with when the converter cannot be
#: asked (it is not installed): the line ``autoware-carla-scenario-rerun`` 0.1
#: pins (rerun-sdk >=0.38.1,<0.39).
DEFAULT_WEB_VIEWER_VERSION = "0.38.1"

#: Where the web viewer is loaded from when there is no copy on this machine:
#: unpkg, which resolves the package's extension-less `import("./re_viewer")`
#: (jsDelivr does not).
CDN_URL = "https://unpkg.com/@rerun-io/web-viewer@{version}"

#: The npm registry's record of a version of the package (its tarball, and the
#: tarball's integrity hash).
_REGISTRY_URL = "https://registry.npmjs.org/@rerun-io/web-viewer/{version}"

#: What to install to convert trajectory logs.
INSTALL_HINT = (
    "pip install 'autoware-carla-scenario[rerun]' "
    "(or pip install autoware-carla-scenario-rerun)"
)

#: The files of the npm package the viewer loads.
_WEB_VIEWER_FILES = ("index.js", "re_viewer.js", "re_viewer_bg.wasm", "package.json")

_ENV_WEB_VIEWER = "VIEWER_RERUN_WEB_VIEWER"
_ENV_MAP_DIR = "VIEWER_RERUN_MAP_DIR"
_ENV_CACHE = "VIEWER_CACHE_DIR"

_TRAJECTORY = ".trajectory.jsonl"


class RecordingError(RuntimeError):
    """A trajectory log could not be made into a recording."""


class ConverterMissingError(RecordingError):
    """The converter is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "Making a Rerun recording of a trajectory log needs the "
            f"autoware_carla_scenario_rerun package: {INSTALL_HINT}"
        )


def _cache_root() -> Path:
    env = os.environ.get(_ENV_CACHE)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "autoware_carla_scenario" / "viewer"


def _converter() -> Any:
    try:
        import autoware_carla_scenario_rerun  # noqa: PLC0415
    except ImportError:
        return None
    return autoware_carla_scenario_rerun


def converter_available() -> bool:
    """Whether ``autoware_carla_scenario_rerun`` can be imported."""
    return _converter() is not None


def converter_version() -> str | None:
    """The converter's version, if it is installed."""
    module = _converter()
    return None if module is None else str(getattr(module, "__version__", "?"))


def web_viewer_version() -> str:
    """The Rerun version the recordings are written with: the converter's
    ``RERUN_VERSION``, or :data:`DEFAULT_WEB_VIEWER_VERSION` without it."""
    module = _converter()
    version = getattr(module, "RERUN_VERSION", None) if module is not None else None
    return str(version) if version else DEFAULT_WEB_VIEWER_VERSION


def recording_path(trajectory: Path) -> Path:
    """``X.rrd`` beside ``X.trajectory.jsonl``, where the converter writes it."""
    name = trajectory.name
    stem = name[: -len(_TRAJECTORY)] if name.endswith(_TRAJECTORY) else trajectory.stem
    return trajectory.with_name(f"{stem}.rrd")


def _cached_recording(trajectory: Path) -> Path:
    """Where a recording is kept when the log's directory cannot be written."""
    digest = hashlib.sha256(str(trajectory.resolve()).encode()).hexdigest()[:24]
    return _cache_root() / "recordings" / f"{digest}.rrd"


def _fresh(recording: Path, trajectory: Path) -> bool:
    try:
        return recording.stat().st_mtime >= trajectory.stat().st_mtime
    except OSError:
        return False


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(str(path), threading.Lock())


def _map_dir() -> str | None:
    """Where the converter looks a map up by name when the log's own path is
    not on this machine: ``VIEWER_RERUN_MAP_DIR``, else the framework's map
    cache (``~/autoware_data/maps``) when there is one."""
    env = os.environ.get(_ENV_MAP_DIR)
    if env:
        return env
    default = Path.home() / "autoware_data" / "maps"
    return str(default) if default.is_dir() else None


def _convert(trajectory: Path, out: Path) -> None:
    module = _converter()
    if module is None:
        raise ConverterMissingError
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written beside and renamed over: a request served meanwhile never reads
    # half a recording.
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.{threading.get_ident()}.tmp.rrd")
    # The run's X_result.json beside the log is found by the converter itself.
    kwargs: dict[str, Any] = {}
    map_dir = _map_dir()
    if map_dir:
        kwargs["map_dir"] = map_dir
    try:
        stats = module.convert(str(trajectory), str(tmp), **kwargs)
        os.replace(tmp, out)
    except Exception as exc:  # ConversionError and its kinds, or an OSError
        raise RecordingError(f"Converting {trajectory.name} failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    for warning in (stats or {}).get("warnings") or []:
        logger.info("%s: %s", trajectory.name, warning)


def ensure_recording(trajectory: Path) -> Path:
    """The recording of *trajectory*, made now if there is none (or an older
    one than the log).

    Blocking: call it in a worker thread.

    Raises:
        FileNotFoundError: There is no such trajectory log.
        ConverterMissingError: A recording has to be made, and the converter
            is not installed.
        RecordingError: The conversion failed.
    """
    if not trajectory.is_file():
        raise FileNotFoundError(f"No trajectory log {trajectory}")
    beside = recording_path(trajectory)
    if _fresh(beside, trajectory):
        return beside
    cached = _cached_recording(trajectory)
    if _fresh(cached, trajectory):
        return cached
    with _lock_for(trajectory):
        # Another request may have made it while this one waited.
        for made in (beside, cached):
            if _fresh(made, trajectory):
                return made
        if os.access(trajectory.parent, os.W_OK):
            _convert(trajectory, beside)
            return beside
        _convert(trajectory, cached)
        return cached


# ── The web viewer ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WebViewer:
    """Where the page loads Rerun's web viewer from."""

    #: ``@rerun-io/web-viewer``'s version.
    version: str
    #: The URL of the package's directory (``index.js`` in it), or ``None``
    #: when no viewer is embedded (``VIEWER_RERUN_WEB_VIEWER=none``).
    base_url: str | None
    #: ``local`` (this app serves it), ``url`` (a mirror), ``cdn`` or ``none``.
    source: str
    #: The directory this app serves it from, when ``local``.
    directory: Path | None = None


def web_viewer_cache_dir(version: str) -> Path:
    """Where ``viewer --fetch-rerun-web-viewer`` keeps a copy of *version*."""
    return _cache_root() / "rerun-web-viewer" / version


def _complete(directory: Path) -> bool:
    return all((directory / name).is_file() for name in _WEB_VIEWER_FILES)


def _package_version(directory: Path) -> str | None:
    try:
        data = json.loads((directory / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return str(version) if version else None


#: The route this app serves a local copy under (``app.py``).
LOCAL_ROUTE = "/rerun-web-viewer"


def web_viewer() -> WebViewer:
    """Where the web viewer is loaded from (see the module's docstring)."""
    version = web_viewer_version()
    env = os.environ.get(_ENV_WEB_VIEWER, "").strip()
    if env.lower() in ("none", "off", "false", "0"):
        return WebViewer(version, None, "none")
    if env.startswith(("http://", "https://", "/")) and not Path(env).is_dir():
        return WebViewer(version, env.rstrip("/"), "url")
    candidates = [Path(env).expanduser()] if env else []
    candidates.append(web_viewer_cache_dir(version))
    for directory in candidates:
        package = (
            directory / "package" if (directory / "package").is_dir() else directory
        )
        if _complete(package):
            found = _package_version(package) or version
            if found.split(".")[:2] != version.split(".")[:2]:
                logger.warning(
                    "%s holds web viewer %s; the recordings are Rerun %s, which "
                    "needs a viewer of the same minor version",
                    package,
                    found,
                    version,
                )
            return WebViewer(found, f"{LOCAL_ROUTE}/{found}", "local", package)
    return WebViewer(version, CDN_URL.format(version=version), "cdn")


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        return bytes(response.read())


def _check_integrity(data: bytes, integrity: str) -> None:
    algorithm, _, expected = integrity.partition("-")
    if algorithm not in ("sha512", "sha384", "sha256"):
        raise RecordingError(f"Unknown integrity algorithm {algorithm!r}")
    digest = base64.b64encode(hashlib.new(algorithm, data).digest()).decode()
    if digest != expected:
        raise RecordingError("The web viewer's tarball does not match its integrity")


def fetch_web_viewer(version: str | None = None, tarball: Path | None = None) -> Path:
    """Keep a copy of the web viewer in the cache directory, for the viewer
    to serve itself: downloaded from the npm registry (checked against the
    integrity hash it states), or unpacked from *tarball* (the package's
    ``.tgz``, for a machine without the network).  Returns the directory.

    Raises:
        RecordingError: The tarball does not match, or is not the package.
    """
    version = version or web_viewer_version()
    if tarball is not None:
        data = tarball.read_bytes()
    else:
        meta = json.loads(_download(_REGISTRY_URL.format(version=version)))
        dist = meta.get("dist") or {}
        data = _download(str(dist["tarball"]))
        if dist.get("integrity"):
            _check_integrity(data, str(dist["integrity"]))
    target = web_viewer_cache_dir(version)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        for name in _WEB_VIEWER_FILES:
            member = members.get(f"package/{name}")
            if member is None:
                raise RecordingError(f"The tarball has no package/{name}")
            source = archive.extractfile(member)
            assert source is not None  # noqa: S101 -- a regular file
            tmp = target / f".{name}.tmp"
            tmp.write_bytes(source.read())
            os.replace(tmp, target / name)
    found = _package_version(target)
    if found != version:
        raise RecordingError(f"The tarball is web viewer {found}, not {version}")
    return target
