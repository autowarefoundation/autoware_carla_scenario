"""Turn an exported Scenario Package into a self-contained wheelhouse.

A Scenario Package is a uv project, and that is what makes it reproducible:
``[tool.uv.sources]`` points the framework at an exact commit and ``uv.lock``
pins everything under it.  It is also what makes it unusable where scenarios
actually run.  Autoware's ``scenario_bridge`` installs a scenario into a venv
built from ``python3-venv`` and ``python3-pip`` -- both rosdep-resolvable --
and has no ``uv``, no ``git`` and, on a vehicle, no network.  Handing that
environment a uv project asks it for all three; handing it the package's wheel
alone is no better, because ``[tool.uv.sources]`` is not written into wheel
metadata, so pip goes looking on PyPI for a framework and a CARLA client that
are not published there.

A wheelhouse is the same dependency graph with the resolution already done:
every wheel the lock names, in one directory, installable with nothing but pip::

    pip install --no-index --find-links <wheelhouse> <distribution>

The wheels are built here, where uv, git and the network are available, which
is the whole point -- none of them are needed again to install it.

Two consequences worth stating plainly, because they are properties of a
wheelhouse rather than of this code:

* it is built **for one platform and one Python**, the exporting machine's.
  Wheels are selected by the interpreter that resolves them, so a wheelhouse
  built on cp310/linux-x86_64 installs on cp310/linux-x86_64;
* it is **large** -- the CARLA client, OpenCV and the lanelet2 bindings alone
  are most of a hundred megabytes.  That is the cost of not needing a network.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..templating import code_environment
from .uv_tool import UvUnavailable, run_uv

logger = logging.getLogger(__name__)

__all__ = [
    "CARLA_EXTRA",
    "CARLA_WHEELS_ENV",
    "Wheelhouse",
    "WheelhouseError",
    "build_wheelhouse",
    "carla_wheels",
]

#: Directory holding the ``*.jinja`` templates for a generated package.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

#: The framework extra that provides the CARLA client an exported scenario runs
#: against.  ``carla==0.10.0`` is not published to PyPI, so the only copy that
#: exists is the wheel vendored in this repository.
CARLA_EXTRA = "carla"

#: Overrides where the vendored CARLA wheels are looked for.
CARLA_WHEELS_ENV = "SCENARIO_EXPORT_CARLA_WHEELS"

#: Name of the directory a package vendors its CARLA wheel into.  Relative, so
#: the package stays self-contained wherever it is copied.
VENDORED_WHEELS_DIR = "carla_wheels"

_EXPORT_TIMEOUT_SECONDS = 300
_BUILD_TIMEOUT_SECONDS = 900
_WHEEL_TIMEOUT_SECONDS = 3600


class WheelhouseError(RuntimeError):
    """Raised when a wheelhouse could not be built completely.

    Attributes:
        log: Captured tool output, when the failure came from a tool.
    """

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log


@dataclass
class Wheelhouse:
    """What a wheelhouse build produced.

    Attributes:
        root: The wheelhouse directory.
        distribution: Distribution name to install from it.
        version: Version of that distribution.
        wheels: Every wheel filename in the directory, sorted.
        python_tag: The interpreter the wheels were resolved for, e.g. ``3.10``.
        log: Combined output of the tools that ran.
    """

    root: Path
    distribution: str
    version: str
    wheels: list[str] = field(default_factory=list)
    python_tag: str = ""
    log: str = ""

    @property
    def size_bytes(self) -> int:
        """Total size of the wheels, which is what a download will cost."""
        return sum(
            (self.root / name).stat().st_size
            for name in self.wheels
            if (self.root / name).is_file()
        )


# ---------------------------------------------------------------------------
# The CARLA client
# ---------------------------------------------------------------------------


def _carla_pin() -> Optional[str]:
    """Return the version the framework's ``carla`` extra pins, e.g. ``0.10.0``.

    Read from the framework's own metadata rather than hard-coded here, so the
    two cannot disagree about which client a scenario runs against.
    """
    from importlib import metadata  # noqa: PLC0415

    try:
        requirements = metadata.requires("autoware-carla-scenario") or []
    except metadata.PackageNotFoundError:  # pragma: no cover - always installed
        return None
    for requirement in requirements:
        if not re.search(rf"extra\s*==\s*[\"']{CARLA_EXTRA}[\"']", requirement):
            continue
        match = re.match(r"\s*carla\s*==\s*([0-9][^\s;]*)", requirement)
        if match:
            return match.group(1)
    return None


def _wheel_search_root() -> Optional[Path]:
    """Return the directory the vendored CARLA wheels live in, if it exists."""
    override = os.environ.get(CARLA_WHEELS_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_dir() else None

    from .framework_pin import framework_source_root  # noqa: PLC0415

    # <repo>/autoware_carla_scenario -> <repo>/carla_wheels.  An installed
    # framework has no repository above it and so has no vendored wheels.
    candidate = framework_source_root().parent / VENDORED_WHEELS_DIR
    return candidate if candidate.is_dir() else None


def carla_wheels() -> list[Path]:
    """Return the vendored CARLA wheels matching the framework's ``carla`` extra.

    Only the pinned version is returned: the repository also vendors a wheel for
    the legacy 0.9.16 client, and shipping both would put two mutually exclusive
    CARLA clients in the same wheelhouse.

    Returns:
        The matching wheels, or an empty list when none can be found -- which is
        the normal case for a framework installed from a wheel rather than run
        out of its repository.
    """
    version = _carla_pin()
    root = _wheel_search_root()
    if version is None or root is None:
        return []
    return sorted(root.glob(f"carla-{version}-*.whl"))


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _venv_python(venv: Path) -> Path:
    """Return the interpreter inside *venv*."""
    posix = venv / "bin" / "python"
    return posix if posix.exists() else venv / "Scripts" / "python.exe"


def _strip_generated_header(text: str) -> str:
    """Return *text* without the leading comment block a tool wrote into it.

    ``uv export`` opens with a note naming the command that produced the file.
    The wheelhouse writes its own, so the two are not stacked.
    """
    lines = text.splitlines()
    start = 0
    while start < len(lines) and (
        lines[start].startswith("#") or not lines[start].strip()
    ):
        start += 1
    return "\n".join(lines[start:]).strip()


def _export_requirements(package_root: Path) -> tuple[str, str]:
    """Return the package's locked dependencies as a requirements file.

    The project itself is excluded: it is built from source here rather than
    resolved, and naming it in a file that is fed to ``pip wheel`` would send
    pip looking for it on an index.  Editable dependencies are exported as
    ordinary ones: a development export points at a checkout by path, and pip
    cannot build a wheel from an editable requirement -- nor would a wheelhouse
    want one, since a wheel is a snapshot and an editable install is the
    opposite of a snapshot.

    Returns:
        ``(requirements, log)``.

    Raises:
        WheelhouseError: If the lock could not be exported.
    """
    result = run_uv(
        package_root,
        "export",
        "--format",
        "requirements-txt",
        "--locked",
        "--no-hashes",
        "--no-editable",
        "--no-emit-project",
        timeout=_EXPORT_TIMEOUT_SECONDS,
    )
    log = (
        "$ uv export --format requirements-txt --locked --no-hashes "
        f"--no-editable --no-emit-project\n{result.stderr}"
    )
    if result.returncode != 0:
        raise WheelhouseError(
            "The package's lockfile could not be exported, so there is no "
            "pinned dependency set to build a wheelhouse from.",
            log=log,
        )
    return _strip_generated_header(result.stdout), log


def _build_project_wheel(package_root: Path, destination: Path) -> str:
    """Build the scenario package's own wheel into *destination*.

    Raises:
        WheelhouseError: If the build failed.
    """
    result = run_uv(
        package_root,
        "build",
        "--wheel",
        "--out-dir",
        str(destination),
        timeout=_BUILD_TIMEOUT_SECONDS,
    )
    log = f"$ uv build --wheel\n{result.stdout}{result.stderr}"
    if result.returncode != 0:
        raise WheelhouseError("The scenario package's own wheel failed to build.", log)
    return log


def _builder_environment(parent: Path, python: str) -> tuple[Path, str]:
    """Create the venv whose pip downloads and builds the dependency wheels.

    uv creates environments without pip, and pip is what fills a wheelhouse:
    it is the tool that resolves a wheel *and builds one from a source
    distribution or a git checkout* when no wheel is published, which is the
    case for the framework itself.

    Raises:
        WheelhouseError: If the environment could not be created.
    """
    venv = parent / "builder"
    result = run_uv(
        parent,
        "venv",
        str(venv),
        "--python",
        python,
        "--seed",
        timeout=_BUILD_TIMEOUT_SECONDS,
    )
    log = f"$ uv venv --python {python} --seed\n{result.stdout}{result.stderr}"
    if result.returncode != 0 or not _venv_python(venv).exists():
        raise WheelhouseError(
            f"No Python {python} environment could be created to build the "
            "wheelhouse with. A wheelhouse is only valid for the interpreter "
            "that resolved it, so it is not built with another one.",
            log,
        )
    return venv, log


def _download_wheels(
    venv: Path, requirements: Path, destination: Path, find_links: list[Path]
) -> str:
    """Fill *destination* with a wheel for every pinned requirement.

    ``--no-deps`` is not a shortcut: the requirements file is the whole locked
    graph already, so letting pip resolve again could only pull in something
    the lock does not name.

    Raises:
        WheelhouseError: If any wheel could not be produced.
    """
    command = [
        str(_venv_python(venv)),
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--requirement",
        str(requirements),
        "--wheel-dir",
        str(destination),
    ]
    for link in find_links:
        command += ["--find-links", str(link)]
    result = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=_WHEEL_TIMEOUT_SECONDS,
    )
    log = f"$ pip wheel --no-deps -r requirements.txt\n{result.stdout}{result.stderr}"
    if result.returncode != 0:
        raise WheelhouseError(
            "Not every dependency could be turned into a wheel, so the "
            "wheelhouse would not install offline. Nothing was written.",
            log,
        )
    return log


def build_wheelhouse(
    package_root: Path,
    destination: Path,
    *,
    distribution: str,
    version: str = "0.1.0",
    python: str = "",
    scenario_id: str = "",
    map_group: str = "",
) -> Wheelhouse:
    """Build a self-contained wheelhouse for the package at *package_root*.

    Args:
        package_root: A locked Scenario Package -- ``uv.lock`` must exist.
        destination: Directory to fill.  Created if missing; it must be empty
            or absent, since a stale wheel left in it would be installed.
        distribution: Distribution name a consumer installs from the wheelhouse.
        version: That distribution's version, recorded in ``requirements.txt``.
        python: Interpreter version to resolve the wheels for.  Defaults to the
            package's ``.python-version``.
        scenario_id: Name the installed ``scenario`` command selects the
            scenario by, for the directory's own README.
        map_group: Map group that scenario runs on, likewise.

    Returns:
        The :class:`Wheelhouse` describing what was built.

    Raises:
        WheelhouseError: If any step failed.  The destination is removed, so a
            partial wheelhouse is never left behind looking installable.
    """
    package_root = Path(package_root)
    destination = Path(destination)
    if not (package_root / "uv.lock").is_file():
        raise WheelhouseError(
            "A wheelhouse is the lockfile resolved into wheels, so it cannot "
            "be built for a package that was never locked."
        )
    if destination.exists() and any(destination.iterdir()):
        raise WheelhouseError(f"{destination} is not empty.")

    if not python:
        recorded = package_root / ".python-version"
        python = (
            recorded.read_text(encoding="utf-8").strip() if recorded.is_file() else ""
        )
    if not python:  # pragma: no cover - every generated package records one
        import platform  # noqa: PLC0415

        python = platform.python_version()

    destination.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="scenario-wheelhouse-"))
    log = ""
    try:
        requirements, export_log = _export_requirements(package_root)
        log += export_log

        pinned = scratch / "dependencies.txt"
        pinned.write_text(f"{requirements}\n", encoding="utf-8")

        log += _build_project_wheel(package_root, destination)

        venv, venv_log = _builder_environment(scratch, python)
        log += venv_log

        # The package vendors its CARLA wheel; pip has to be told where, since
        # `[tool.uv] find-links` means nothing to it.
        vendored = package_root / VENDORED_WHEELS_DIR
        log += _download_wheels(
            venv, pinned, destination, [vendored] if vendored.is_dir() else []
        )
    except UvUnavailable as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise WheelhouseError(str(exc), log) from exc
    except WheelhouseError as exc:
        exc.log = f"{log}\n{exc.log}" if exc.log else log
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    wheels = sorted(path.name for path in destination.glob("*.whl"))
    _write_install_files(
        destination,
        distribution=distribution,
        version=version,
        python=python,
        requirements=requirements,
        wheels=wheels,
        scenario_id=scenario_id,
        map_group=map_group,
    )
    logger.info("Built a wheelhouse of %d wheels at %s", len(wheels), destination)
    return Wheelhouse(
        root=destination,
        distribution=distribution,
        version=version,
        wheels=wheels,
        python_tag=python,
        log=log,
    )


def _write_install_files(
    destination: Path,
    *,
    distribution: str,
    version: str,
    python: str,
    requirements: str,
    wheels: list[str],
    scenario_id: str,
    map_group: str,
) -> None:
    """Write the two files that make the directory installable by hand.

    ``requirements.txt`` names the whole pinned set including the scenario
    itself, so a consumer can install exactly what was resolved rather than let
    pip pick from the directory; ``README.md`` says how, for the person who
    unzips it a month later.
    """
    (destination / "requirements.txt").write_text(
        "# Every distribution this scenario needs, pinned to what the package's\n"
        "# uv.lock resolved. Install it against this directory and nothing is\n"
        "# fetched from an index:\n"
        "#\n"
        "#     pip install --no-index --find-links . -r requirements.txt\n"
        f"{distribution}=={version}\n"
        f"{requirements}\n",
        encoding="utf-8",
    )
    environment = code_environment(TEMPLATES_DIR)
    (destination / "README.md").write_text(
        environment.get_template("wheelhouse_README.md.jinja").render(
            distribution=distribution,
            version=version,
            python=python,
            wheel_count=len(wheels),
            scenario_id=scenario_id,
            map_group=map_group,
        ),
        encoding="utf-8",
    )
