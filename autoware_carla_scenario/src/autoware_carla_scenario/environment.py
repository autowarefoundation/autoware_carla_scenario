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

The library check compiles what reads the environment (docs/typecheck.md):
where each distribution was installed from (:func:`_source_members`, which
holds what its ``direct_url.json`` gives as JSON text, of whatever type it
is), the distributions themselves (:func:`_installed`), the id and the lock's
text.  What holds and writes them as ``dict[str, Any]`` records -- the
public :class:`RunEnvironment`, :func:`capture` and :func:`write` -- reads
those values back and is left out.
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
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import unquote, urlparse

from .atomic_file import write_text_atomically

# Re-exported under the names they had here before environment_pin (#825).
from .environment_pin import HASH_PREFERENCE as _HASH_PREFERENCE  # noqa: F401
from .environment_pin import PinSource
from .environment_pin import fragment as _fragment  # noqa: F401
from .environment_pin import normalize as _normalize
from .environment_pin import pin as _pin

# Beside a run's result: its environment, and its lock -- the checked
# run_outputs's suffixes (#1713), re-exported under the names they had here.
from .run_outputs import ENVIRONMENT_SUFFIX as ENVIRONMENT_SUFFIX
from .run_outputs import LOCK_SUFFIX as LOCK_SUFFIX

__all__ = ["ENVIRONMENT_SUFFIX", "LOCK_SUFFIX", "RunEnvironment", "capture", "write"]

logger = logging.getLogger(__name__)

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


def _git(path: Path, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def _archive_hashes(info_json: str) -> dict[str, str]:
    """``archive_info``'s hashes: ``hashes``, or the deprecated ``hash``.

    *info_json* is the JSON text of ``archive_info`` as ``direct_url.json``
    gives it; one that is not true has none.
    """
    info = json.loads(info_json)
    hashes: dict[str, str] = {}
    if not info:
        return hashes
    given = info.get("hashes")
    if given:
        for k, v in given.items():
            if isinstance(v, str) and v:
                hashes[str(k).lower()] = str(v)
    legacy = info.get("hash")
    if isinstance(legacy, str) and "=" in legacy:
        algorithm, _, value = str(legacy).partition("=")
        hashes.setdefault(algorithm.lower(), value)
    return hashes


def _direct_url(dist: metadata.Distribution) -> Optional[str]:
    """The text of *dist*'s ``direct_url.json`` (PEP 610), or ``None``: it
    has none, or it cannot be read."""
    try:
        return dist.read_text("direct_url.json")
    except Exception:  # noqa: BLE001 - an odd distribution is reported as from an index
        return None


def _source_members(raw: Optional[str]) -> list[tuple[str, str]]:
    """Where a distribution was installed from, as the environment file
    records it (:func:`_source_record` of :func:`_source`), from the text of
    its ``direct_url.json`` (*raw*; ``None``: it has none): the record's
    members in their order, each value as its JSON text (``json.dumps()``).

    ``index`` for a registry release, or for a file that is not a JSON
    object; ``vcs`` with the URL, the VCS, the commit and the revision that
    was asked for; ``archive`` with the URL and its hashes; ``editable`` /
    ``local`` with the directory, and its git commit and uncommitted changes
    when it is a checkout.  A ``subdirectory`` -- the project inside a
    monorepo, as this framework is inside its repository -- is kept for every
    kind: without it, the URL names the repository root.  A value the file
    gives is kept as it gives it, of whatever JSON type, which is why it is
    held as its text.
    """
    if not raw:
        return [("kind", json.dumps("index"))]
    try:
        direct = json.loads(raw)
    except ValueError:
        return [("kind", json.dumps("index"))]
    if not isinstance(direct, dict):
        return [("kind", json.dumps("index"))]
    url = str(direct.get("url", ""))
    # A subdirectory that is not true is none.
    subdirectory = direct.get("subdirectory")
    subdirectory_json = json.dumps(subdirectory) if subdirectory else ""
    members: list[tuple[str, str]] = []
    if "vcs_info" in direct:
        members.append(("kind", json.dumps("vcs")))
        members.append(("url", json.dumps(url)))
        vcs = direct["vcs_info"]
        # A vcs_info that is not true has no member: its defaults.
        if vcs:
            members.append(("vcs", json.dumps(vcs.get("vcs", "git"))))
            members.append(("commit", json.dumps(vcs.get("commit_id", ""))))
            members.append(("requested", json.dumps(vcs.get("requested_revision"))))
        else:
            members.append(("vcs", json.dumps("git")))
            members.append(("commit", json.dumps("")))
            members.append(("requested", "null"))
        if subdirectory_json:
            members.append(("subdirectory", subdirectory_json))
        return members
    if "archive_info" in direct:
        members.append(("kind", json.dumps("archive")))
        members.append(("url", json.dumps(url)))
        hashes = _archive_hashes(json.dumps(direct["archive_info"]))
        if len(hashes) > 0:
            members.append(("hashes", json.dumps(hashes)))
        if subdirectory_json:
            members.append(("subdirectory", subdirectory_json))
        return members
    path: Optional[Path] = None
    if url.startswith("file:"):
        path = Path(unquote(urlparse(url).path))
    if path is not None and subdirectory:
        if TYPE_CHECKING:
            path = path / str(subdirectory)
        else:
            # As the file gives it: a subdirectory that is no str raises.
            path = path / subdirectory
    commit = ""
    dirty = False
    if path is not None and path.is_dir():
        head = _git(path, "rev-parse", "HEAD")
        if head:
            commit = head
        if commit:
            status = _git(path, "status", "--porcelain", "--untracked-files=no")
            dirty = bool(status)
    kind = "local"
    dir_info = direct.get("dir_info")
    if dir_info:
        if dir_info.get("editable"):
            kind = "editable"
    members.append(("kind", json.dumps(kind)))
    members.append(("path", json.dumps(url if path is None else str(path))))
    # A directory's subdirectory is in its path already.
    if path is None and subdirectory_json:
        members.append(("subdirectory", subdirectory_json))
    if commit:
        members.append(("commit", json.dumps(commit)))
        members.append(("dirty", json.dumps(dirty)))
    return members


def _record(members: list[tuple[str, str]]) -> dict[str, Any]:
    """The record *members* hold (:func:`_source_members`), each value read
    back from its JSON text."""
    return {key: json.loads(text) for key, text in members}


def _source(dist: metadata.Distribution) -> PinSource:
    """Where *dist* was installed from (PEP 610's ``direct_url.json``).

    ``index`` for a registry release; ``vcs`` with the URL, the commit and
    the revision that was asked for; ``archive`` with the URL and its hashes;
    ``editable`` / ``local`` with the directory, and its git commit and
    uncommitted changes when it is a checkout.  A ``subdirectory`` -- the
    project inside a monorepo, as this framework is inside its repository --
    is kept for every kind: without it, the URL names the repository root.
    :func:`_source_record` is what the environment file writes of it, the
    record :func:`_source_members` reads; each field is that record's value,
    as the file gives it, or the field's default where it has none.
    """
    record = _record(_source_members(_direct_url(dist)))
    return PinSource(
        kind=record["kind"],
        url=record.get("url", ""),
        vcs=record.get("vcs", ""),
        commit=record.get("commit", ""),
        requested=record.get("requested"),
        path=record.get("path"),
        dirty=record.get("dirty", False),
        subdirectory=record.get("subdirectory", ""),
        hashes=record.get("hashes", {}),
    )


def _source_record(source: PinSource) -> dict[str, Any]:
    """*source* as the environment file writes it: only the keys its kind
    has (``{"kind": "index"}`` for a registry release)."""
    record: dict[str, Any] = {"kind": source.kind}
    if source.kind == "vcs":
        record.update(
            url=source.url,
            vcs=source.vcs,
            commit=source.commit,
            requested=source.requested,
        )
    elif source.kind == "archive":
        record["url"] = source.url
        if source.hashes:
            record["hashes"] = source.hashes
    elif source.kind in ("editable", "local"):
        record["path"] = source.path
    if source.subdirectory:
        record["subdirectory"] = source.subdirectory
    if source.kind in ("editable", "local") and source.commit:
        record["commit"] = source.commit
        record["dirty"] = source.dirty
    return record


def _pin_source(record: dict[str, Any]) -> PinSource:
    """The :class:`PinSource` an environment record (:func:`_source_record`)
    describes; a key it lacks is a field the source does not have."""
    return PinSource(
        kind=record.get("kind"),  # type: ignore[arg-type]
        url=record.get("url", ""),
        vcs=record.get("vcs", ""),
        commit=record.get("commit") or "",
        requested=record.get("requested"),
        path=record.get("path"),
        dirty=bool(record.get("dirty")),
        subdirectory=record.get("subdirectory") or "",
        hashes=record.get("hashes") or {},
    )


def _package_pin(package: dict[str, Any]) -> str:
    """The lock's line for a recorded distribution."""
    return _pin(package["name"], package["version"], _pin_source(package["source"]))


class _Package:
    """An installed distribution as :func:`capture` records it: its name,
    normalised, its version (``None`` for metadata without one) and where it
    was installed from (:func:`_source_members`)."""

    name: str
    version: Optional[str]
    source: list[tuple[str, str]]

    def __init__(
        self, name: str, version: Optional[str], source: list[tuple[str, str]]
    ) -> None:
        self.name = name
        self.version = version
        self.source = source


def _installed() -> list[_Package]:
    """Every installed distribution with a name, by its normalised name,
    sorted: of two that normalise alike (one shadows the other on
    ``sys.path``), the first, which is the one imported."""
    packages: dict[str, _Package] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        key = _normalize(name)
        if key in packages:
            continue
        packages[key] = _Package(key, dist.version, _source_members(_direct_url(dist)))
    return sorted(packages.values(), key=lambda p: p.name)


def _interpreter() -> tuple[str, str, str, str]:
    """This interpreter's version and implementation, and the platform and
    machine it runs on, as :mod:`platform` names them."""
    return (
        platform.python_version(),
        platform.python_implementation(),
        platform.platform(),
        platform.machine(),
    )


def _environment_id(implementation: str, python: str, pins: list[str]) -> str:
    """The SHA-256 of what a lock pins (*pins*), and the Python it pins for."""
    text = "\n".join([f"python {implementation} {python}", *pins])
    return hashlib.sha256(text.encode()).hexdigest()


def _lock_text(
    environment_id: str,
    implementation: str,
    python: str,
    platform_name: str,
    framework_pin: Optional[str],
    scenario_pin: Optional[str],
    pins: list[str],
) -> str:
    """The lock: a header naming the environment, its Python and platform,
    the framework and the scenario's package, if any, then *pins*."""
    header = [
        "# The environment a scenario run ran in (autoware_carla_scenario).",
        f"# environment id: {environment_id}",
        f"# python {implementation} {python} on {platform_name}",
    ]
    if framework_pin is not None:
        header.append(f"# framework: {framework_pin}")
    if scenario_pin is not None:
        header.append(f"# scenario: {scenario_pin}")
    return "\n".join([*header, *pins]) + "\n"


def _distribution_names(top: str) -> list[str]:
    """The distributions that provide the top-level package *top*."""
    names = metadata.packages_distributions().get(top)
    if not names:
        return []
    return names


class RunEnvironment:
    """The environment of this process: Python, the platform, the distributions."""

    packages: list[dict[str, Any]]
    by_name: dict[str, dict[str, Any]]
    python: str
    implementation: str
    platform: str
    machine: str

    def __init__(self, packages: list[dict[str, Any]]) -> None:
        self.packages = packages
        self.by_name = {p["name"]: p for p in packages}
        self.python, self.implementation, self.platform, self.machine = _interpreter()

    @property
    def pins(self) -> list[str]:
        return [_package_pin(p) for p in self.packages]

    @property
    def id(self) -> str:
        """The SHA-256 of what the lock pins (and the Python it pins for)."""
        return _environment_id(self.implementation, self.python, self.pins)

    def scenario_package(self, scenario: Any) -> Optional[dict[str, Any]]:
        """The distribution the scenario's class comes from, if any."""
        module = (
            getattr(type(scenario), "__module__", "") if scenario is not None else ""
        )
        top = module.split(".")[0]
        if not top:
            return None
        for name in _distribution_names(top):
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
        # Built, then updated: Codon parses no `**` in a dict display.
        document: dict[str, Any] = {"schema": ENVIRONMENT_SCHEMA}
        document.update(self.summary(scenario))
        document["implementation"] = self.implementation
        document["machine"] = self.machine
        document["executable_prefix"] = sys.prefix
        document["distributions"] = self.packages
        return document

    def lock(self, scenario: Any = None) -> str:
        """The lock: a requirements file of every distribution, pinned."""
        framework = self.by_name.get(FRAMEWORK)
        scenario_pkg = self.scenario_package(scenario)
        environment_id = self.id
        framework_pin = None if framework is None else _package_pin(framework)
        scenario_pin = None
        if scenario_pkg is not None and scenario_pkg is not framework:
            scenario_pin = _package_pin(scenario_pkg)
        return _lock_text(
            environment_id,
            self.implementation,
            self.python,
            self.platform,
            framework_pin,
            scenario_pin,
            self.pins,
        )


@lru_cache(maxsize=1)
def capture() -> RunEnvironment:
    """This process's environment (read once: it does not change during a run)."""
    return RunEnvironment(
        [
            {"name": p.name, "version": p.version, "source": _record(p.source)}
            for p in _installed()
        ]
    )


def write(
    output_dir: Path, file_stem: str, scenario: Any = None
) -> Optional[dict[str, Any]]:
    """Write the run's environment and lock beside its result; return the summary.

    A failure is logged, never raised: a run's result matters more than this.
    """
    try:
        env = capture()
        output_dir.mkdir(parents=True, exist_ok=True)
        # In one rename each, as the result beside them: the viewer reads
        # them while a sweep runs (#1956).
        write_text_atomically(
            output_dir / f"{file_stem}{ENVIRONMENT_SUFFIX}",
            json.dumps(env.to_dict(scenario), indent=2),
        )
        write_text_atomically(
            output_dir / f"{file_stem}{LOCK_SUFFIX}", env.lock(scenario)
        )
        return env.summary(scenario)
    except Exception:
        logger.warning("could not write the run's environment", exc_info=True)
        return None
