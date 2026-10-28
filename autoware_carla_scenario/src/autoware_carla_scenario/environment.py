"""What a run ran with: this framework, the scenario's package, every dependency.

A result says what happened; this says with which tools. Every run writes,
next to its result:

* ``{Scenario}_requirements.lock`` -- the installed distributions pinned, in
  the requirements format (``pip install -r`` / ``uv pip install -r``), with a
  header naming the framework, Python and platform. A distribution installed
  from git is pinned to its commit; one installed editable (a checkout) names
  its directory, commit and whether it had uncommitted changes, which no lock
  can reproduce -- so the lock says so;
* ``{Scenario}_environment.json`` -- the same, structured: the framework and
  the scenario's own package with where they came from, Python, the platform,
  and every distribution with its version and source;

and its ``_result.json`` carries a summary (``environment``) with the lock's
**id** -- the SHA-256 of what it pins -- so runs on one environment can be
found by it, and a run on another stands out.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

__all__ = ["ENVIRONMENT_SUFFIX", "LOCK_SUFFIX", "RunEnvironment", "capture", "write"]

logger = logging.getLogger(__name__)

#: Beside a run's result: its environment, and its lock.
ENVIRONMENT_SUFFIX = "_environment.json"
LOCK_SUFFIX = "_requirements.lock"
#: Written to every environment file.
ENVIRONMENT_SCHEMA = "autoware_carla_scenario.environment/1"
#: The framework's own distribution.
FRAMEWORK = "autoware-carla-scenario"
#: Distributions named in a result's summary besides the framework and the
#: scenario's: what decides how an ODD is judged and what CARLA is driven with.
KEY_PACKAGES = (
    "openodd-py",
    "typesafe-carla",
    "simple-lanelet2",
    "hydra-core",
    "numpy",
)


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _git(path: Path, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


#: Archive hashes in the order a lock prefers them, strongest first; pip and uv
#: both read any of them from a URL fragment.
_HASH_PREFERENCE = ("sha512", "sha384", "sha256", "sha224", "sha1", "md5")


def _archive_hashes(info: dict[str, Any]) -> dict[str, str]:
    """``archive_info``'s hashes: ``hashes``, or the deprecated ``hash``."""
    hashes = {
        str(k).lower(): str(v)
        for k, v in (info.get("hashes") or {}).items()
        if isinstance(v, str) and v
    }
    legacy = info.get("hash")
    if isinstance(legacy, str) and "=" in legacy:
        algorithm, _, value = legacy.partition("=")
        hashes.setdefault(algorithm.lower(), value)
    return hashes


def _source(dist: metadata.Distribution) -> dict[str, Any]:
    """Where *dist* was installed from (PEP 610's ``direct_url.json``).

    ``{"kind": "index"}`` for a registry release; ``vcs`` with the URL, the
    commit and the revision that was asked for; ``archive`` with the URL and
    its hashes; ``editable`` / ``local`` with the directory, and its git commit
    and uncommitted changes when it is a checkout.  A ``subdirectory`` -- the
    project inside a monorepo, as this framework is inside its repository --
    is kept for every kind: without it, the URL names the repository root.
    """
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:  # noqa: BLE001 - an odd distribution is reported as from an index
        raw = None
    if not raw:
        return {"kind": "index"}
    try:
        direct = json.loads(raw)
    except ValueError:
        return {"kind": "index"}
    if not isinstance(direct, dict):
        return {"kind": "index"}
    url = str(direct.get("url", ""))
    subdirectory = direct.get("subdirectory") or None
    if "vcs_info" in direct:
        vcs = direct["vcs_info"] or {}
        source: dict[str, Any] = {
            "kind": "vcs",
            "url": url,
            "vcs": vcs.get("vcs", "git"),
            "commit": vcs.get("commit_id", ""),
            "requested": vcs.get("requested_revision"),
        }
        if subdirectory:
            source["subdirectory"] = subdirectory
        return source
    if "archive_info" in direct:
        source = {"kind": "archive", "url": url}
        hashes = _archive_hashes(direct["archive_info"] or {})
        if hashes:
            source["hashes"] = hashes
        if subdirectory:
            source["subdirectory"] = subdirectory
        return source
    path = Path(unquote(urlparse(url).path)) if url.startswith("file:") else None
    if path is not None and subdirectory:
        path = path / subdirectory
    out: dict[str, Any] = {
        "kind": "editable"
        if (direct.get("dir_info") or {}).get("editable")
        else "local",
        "path": str(path) if path else url,
    }
    if path is None and subdirectory:
        out["subdirectory"] = subdirectory
    if path is not None and path.is_dir():
        commit = _git(path, "rev-parse", "HEAD")
        if commit:
            out["commit"] = commit
            status = _git(path, "status", "--porcelain", "--untracked-files=no")
            out["dirty"] = bool(status)
    return out


def _fragment(source: dict[str, Any]) -> str:
    """The URL fragment pip and uv read a hash and a subdirectory from."""
    parts: list[str] = []
    hashes = source.get("hashes") or {}
    for algorithm in _HASH_PREFERENCE:
        if hashes.get(algorithm):
            parts.append(f"{algorithm}={hashes[algorithm]}")
            break
    if source.get("subdirectory"):
        parts.append(f"subdirectory={source['subdirectory']}")
    return "#" + "&".join(parts) if parts else ""


def _pin(name: str, version: str, source: dict[str, Any]) -> str:
    """The requirement that installs this distribution again, as far as one can."""
    kind = source.get("kind")
    if kind == "vcs":
        # The commit is what was installed; the requested revision (a branch
        # or tag, which can move) is only the fallback when none was recorded.
        revision = source.get("commit") or source.get("requested")
        if revision:
            return (
                f"{name} @ {source['vcs']}+{source['url']}@{revision}"
                f"{_fragment(source)}"
            )
    if kind == "archive":
        return f"{name} @ {source['url']}{_fragment(source)}"
    if kind in ("editable", "local"):
        state = ""
        if source.get("commit"):
            state = f", git {source['commit']}" + (
                " with uncommitted changes" if source.get("dirty") else ""
            )
        return f"{name}=={version}  # {kind} from {source.get('path')}{state}"
    return f"{name}=={version}"


class RunEnvironment:
    """The environment of this process: Python, the platform, the distributions."""

    def __init__(self, packages: list[dict[str, Any]]) -> None:
        self.packages = packages
        self.by_name = {p["name"]: p for p in packages}
        self.python = platform.python_version()
        self.implementation = platform.python_implementation()
        self.platform = platform.platform()
        self.machine = platform.machine()

    @property
    def pins(self) -> list[str]:
        return [_pin(p["name"], p["version"], p["source"]) for p in self.packages]

    @property
    def id(self) -> str:
        """The SHA-256 of what the lock pins (and the Python it pins for)."""
        text = "\n".join([f"python {self.implementation} {self.python}", *self.pins])
        return hashlib.sha256(text.encode()).hexdigest()

    def scenario_package(self, scenario: Any) -> Optional[dict[str, Any]]:
        """The distribution the scenario's class comes from, if any."""
        module = (
            getattr(type(scenario), "__module__", "") if scenario is not None else ""
        )
        top = module.split(".")[0]
        if not top:
            return None
        names = metadata.packages_distributions().get(top) or []
        for name in names:
            found = self.by_name.get(_normalize(name))
            if found is not None:
                return found
        return None

    def summary(self, scenario: Any = None) -> dict[str, Any]:
        """What a result says of its environment."""

        def brief(p: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
            return (
                None
                if p is None
                else {"name": p["name"], "version": p["version"], "source": p["source"]}
            )

        return {
            "id": self.id,
            "python": self.python,
            "platform": self.platform,
            "framework": brief(self.by_name.get(FRAMEWORK)),
            "scenario": brief(self.scenario_package(scenario)),
            "packages": {
                k: self.by_name[k]["version"] for k in KEY_PACKAGES if k in self.by_name
            },
        }

    def to_dict(self, scenario: Any = None) -> dict[str, Any]:
        return {
            "schema": ENVIRONMENT_SCHEMA,
            **self.summary(scenario),
            "implementation": self.implementation,
            "machine": self.machine,
            "executable_prefix": sys.prefix,
            "distributions": self.packages,
        }

    def lock(self, scenario: Any = None) -> str:
        """The lock: a requirements file of every distribution, pinned."""
        framework = self.by_name.get(FRAMEWORK)
        scenario_pkg = self.scenario_package(scenario)
        header = [
            "# The environment a scenario run ran in (autoware_carla_scenario).",
            f"# environment id: {self.id}",
            f"# python {self.implementation} {self.python} on {self.platform}",
        ]
        if framework is not None:
            header.append(
                f"# framework: {_pin(framework['name'], framework['version'], framework['source'])}"
            )
        if scenario_pkg is not None and scenario_pkg is not framework:
            header.append(
                f"# scenario: {_pin(scenario_pkg['name'], scenario_pkg['version'], scenario_pkg['source'])}"
            )
        return "\n".join([*header, *self.pins]) + "\n"


@lru_cache(maxsize=1)
def capture() -> RunEnvironment:
    """This process's environment (read once: it does not change during a run)."""
    packages: dict[str, dict[str, Any]] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        key = _normalize(name)
        if key in packages:  # shadowed on sys.path: the first is the one imported
            continue
        packages[key] = {"name": key, "version": dist.version, "source": _source(dist)}
    return RunEnvironment(sorted(packages.values(), key=lambda p: p["name"]))


def write(
    output_dir: Path, file_stem: str, scenario: Any = None
) -> Optional[dict[str, Any]]:
    """Write the run's environment and lock beside its result; return the summary.

    A failure is logged, never raised: a run's result matters more than this.
    """
    try:
        env = capture()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{file_stem}{ENVIRONMENT_SUFFIX}").write_text(
            json.dumps(env.to_dict(scenario), indent=2), encoding="utf-8"
        )
        (output_dir / f"{file_stem}{LOCK_SUFFIX}").write_text(
            env.lock(scenario), encoding="utf-8"
        )
        return env.summary(scenario)
    except Exception:
        logger.warning("could not write the run's environment", exc_info=True)
        return None
