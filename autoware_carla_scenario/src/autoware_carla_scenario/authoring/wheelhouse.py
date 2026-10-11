"""Turn an exported Scenario Package into a self-contained wheelhouse.

A Scenario Package is a uv project, and that is what makes it reproducible:
``[tool.uv.sources]`` points the framework at an exact commit and ``uv.lock``
pins everything under it.  It is also what makes it unusable where scenarios
actually run.  Autoware's ``scenario_bridge`` installs a scenario into a venv
built from ``python3-venv`` and ``python3-pip`` -- both rosdep-resolvable --
and has no ``uv``, no ``git`` and, on a vehicle, no network.  Handing that
environment a uv project asks it for all three; handing it the package's wheel
alone is no better, because ``[tool.uv.sources]`` is not written into wheel
metadata, so pip goes looking on PyPI for a framework at a commit that is not
published there -- and, on a vehicle, cannot reach PyPI at all.

A wheelhouse is the same dependency graph with the resolution already done:
every wheel the lock names, in one directory, installable with nothing but pip::

    pip install --no-index --find-links <wheelhouse> <distribution>

The wheels are built here, where uv, git and the network are available, which
is the whole point -- none of them are needed again to install it.

Two consequences worth stating plainly, because they are properties of a
wheelhouse rather than of this code:

* it is built **for one platform**, the exporting machine's, but for *every*
  interpreter the package supports.  Wheels are selected by the interpreter
  that resolves them, so one pass is made per interpreter and the results are
  merged into the single directory: the pure-Python wheels are shared, and the
  compiled ones sit side by side with their own ``cp3xx`` tag.  That is what
  lets the same wheelhouse install under ROS 2 Humble's Python 3.10 and
  Jazzy's 3.12, which is the difference between a scenario that runs on a
  vehicle and one that needs a PPA first;
* it is **large** -- the CARLA client (``typesafe-carla`` and the Codon
  compiler it pins, ``typesafe-carla-toolchain``), OpenCV and the lanelet2
  bindings alone are well over a hundred megabytes.  That is the cost of not
  needing a network.

The CARLA client needs nothing beyond its wheels either: the released
``typesafe-carla`` wheel the wheelhouse holds carries typesafe_carla's CPython
package (``typesafe_carla.carla``) prebuilt, one build for every Python 3.10+,
so an offline vehicle install is ``pip install`` and nothing else -- no
``cc``, no build step.  Only where that prebuilt package does not match (a
``typesafe-carla-toolchain`` other than the one the wheel was built with, for
instance) does the first ``import typesafe_carla.carla`` build it (15 to 50
minutes, about 14 GB of RAM, ``cc``), still without a network: the compiler is
the toolchain wheel already in the directory.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jinja2 import TemplateError

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from ..templating import code_environment
from .uv_tool import UvUnavailable, run_uv

# tomllib landed in 3.11, and 3.10 is still the floor of the supported range.
# Branching on sys.version_info rather than catching ImportError keeps mypy from
# reading the fallback as a redefinition when it checks against 3.11+.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib

logger = logging.getLogger(__name__)

__all__ = [
    "TESTED_PYTHONS",
    "Wheelhouse",
    "WheelhouseError",
    "build_wheelhouse",
    "supported_pythons",
    "venv_python",
]

#: Directory holding the ``*.jinja`` templates for a generated package.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

#: The interpreters CI tests the framework under, oldest first.  A wheelhouse
#: is resolved for each of them the package's ``requires-python`` admits -- see
#: :func:`supported_pythons`.  Raise it together with CI's interpreter matrix
#: and the framework's ``requires-python`` (a test checks the two agree).
TESTED_PYTHONS: tuple[str, ...] = ("3.10", "3.11", "3.12", "3.13", "3.14")

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


@dataclass(frozen=True)
class Wheelhouse:
    """What a wheelhouse build produced.

    Frozen, and :attr:`size_bytes` is a recorded number rather than a property
    that stats :attr:`root`: the editor deletes the build tree as soon as it has
    zipped it, and a report rendered afterwards would otherwise say the
    wheelhouse it just handed over is empty.  Use :func:`dataclasses.replace` to
    say where the directory ended up.

    Attributes:
        root: The wheelhouse directory.
        distribution: Distribution name to install from it.
        version: Version of that distribution.
        wheels: Every wheel filename in the directory, sorted.
        size_bytes: What those wheels came to, which is what a download costs.
        python_tags: Every interpreter the wheels were resolved for, in
            ascending order, e.g. ``("3.10", "3.11", "3.12")``.  The directory
            installs under any of them.
        log: Combined output of the tools that ran.
    """

    root: Path
    distribution: str
    version: str
    wheels: tuple[str, ...] = ()
    size_bytes: int = 0
    python_tags: tuple[str, ...] = ()
    log: str = ""


# ---------------------------------------------------------------------------
# Interpreters
# ---------------------------------------------------------------------------


def _declared_requires_python(package_root: Path) -> Optional[str]:
    """Return the ``requires-python`` the package at *package_root* declares."""
    pyproject = Path(package_root) / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    value = data.get("project", {}).get("requires-python")
    return value if isinstance(value, str) and value.strip() else None


def supported_pythons(package_root: Path) -> list[str]:
    """Return every interpreter the package at *package_root* is built for.

    The interpreters of :data:`TESTED_PYTHONS` that the package's own
    ``requires-python`` admits.  A generated package inherits the framework's
    ``requires-python``, which is CI's tested range, so this is normally all
    of them -- and a wheelhouse resolved for 3.10, 3.11 and 3.12 is what lets
    one export serve both ROS 2 distributions.

    Intersected rather than taken from ``requires-python`` alone, because a
    specifier is not a list: ``>=3.10`` admits interpreters nobody has tested
    and that a builder environment may not even exist for.

    Returns:
        The versions in ascending order, e.g. ``["3.10", "3.11", "3.12"]``, or
        an empty list when the package declares no ``requires-python`` (or
        one that admits none of them) -- the caller then falls back to the
        interpreter the package recorded in ``.python-version``.
    """
    declared = _declared_requires_python(package_root)
    if declared is None:
        return []
    try:
        specifier = SpecifierSet(declared)
    except InvalidSpecifier:
        return []
    return [version for version in TESTED_PYTHONS if specifier.contains(version)]


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def venv_python(venv: Path) -> Path:
    """Return the interpreter inside *venv*."""
    posix = venv / "bin" / "python"
    return posix if posix.exists() else venv / "Scripts" / "python.exe"


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
    arguments = (
        "export",
        "--format",
        "requirements-txt",
        "--locked",
        "--no-hashes",
        "--no-editable",
        "--no-emit-project",
        # The wheelhouse writes its own header onto this; uv's would sit above
        # it saying the same thing about a file the user never asked uv for.
        "--no-header",
    )
    result = run_uv(package_root, *arguments, timeout=_EXPORT_TIMEOUT_SECONDS)
    log = f"$ uv {' '.join(arguments)}\n{result.stderr}"
    if result.returncode != 0:
        raise WheelhouseError(
            "The package's lockfile could not be exported, so there is no "
            "pinned dependency set to build a wheelhouse from.",
            log=log,
        )
    return result.stdout.strip(), log


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

    One per interpreter, named after it: a wheelhouse is filled by a pass per
    interpreter and the second pass must not find the first one's environment.

    Raises:
        WheelhouseError: If the environment could not be created.
    """
    venv = parent / f"builder-{python}"
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
    if result.returncode != 0 or not venv_python(venv).exists():
        raise WheelhouseError(
            f"No Python {python} environment could be created to build the "
            "wheelhouse with. A wheelhouse is only valid for the interpreter "
            "that resolved it, so it is not built with another one.",
            log,
        )
    return venv, log


def _download_wheels(venv: Path, requirements: Path, destination: Path) -> str:
    """Fill *destination* with a wheel for every pinned requirement.

    ``--no-deps`` is not a shortcut: the requirements file is the whole locked
    graph already, so letting pip resolve again could only pull in something
    the lock does not name.

    Raises:
        WheelhouseError: If any wheel could not be produced.
    """
    command = [
        str(venv_python(venv)),
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--requirement",
        str(requirements),
        "--wheel-dir",
        str(destination),
    ]
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
    version: str,
    run_command: str,
    pythons: Sequence[str] = (),
) -> Wheelhouse:
    """Build a self-contained wheelhouse for the package at *package_root*.

    Args:
        package_root: A locked Scenario Package -- ``uv.lock`` must exist.
        destination: Directory to fill.  Created if missing; it must be empty
            or absent, since a stale wheel left in it would be installed.
        distribution: Distribution name a consumer installs from the wheelhouse.
        version: That distribution's version.  It has to be the one the
            package's own ``pyproject.toml`` declares, or the
            ``requirements.txt`` written here names a wheel that is not in the
            directory.
        run_command: The command that runs the scenario once installed, for the
            directory's own README.
        pythons: Interpreter versions to resolve the wheels for, e.g.
            ``("3.10", "3.12")``.  Defaults to every tested interpreter the
            package's ``requires-python`` admits -- see
            :func:`supported_pythons` -- and falls back to the package's
            ``.python-version`` when that yields none.  One pass is made per
            interpreter into the same directory.

    Returns:
        The :class:`Wheelhouse` describing what was built.

    Raises:
        WheelhouseError: If any step failed -- including a tool timing out or
            failing to start.  The destination is removed, so a partial
            wheelhouse is never left behind looking installable, and the next
            attempt does not find a non-empty directory.
    """
    package_root = Path(package_root)
    destination = Path(destination)
    if not (package_root / "uv.lock").is_file():
        raise WheelhouseError(
            "A wheelhouse is the lockfile resolved into wheels, so it cannot "
            "be built for a package that was never locked."
        )
    if destination.exists():
        # Checked before iterating: `iterdir()` on a regular file raises
        # NotADirectoryError, which would leave this function through a path
        # that promises WheelhouseError for an unusable destination.
        if not destination.is_dir():
            raise WheelhouseError(f"{destination} exists and is not a directory.")
        if any(destination.iterdir()):
            raise WheelhouseError(f"{destination} is not empty.")

    targets = [str(interpreter) for interpreter in pythons]
    if not targets:
        targets = supported_pythons(package_root)
    if not targets:
        recorded = package_root / ".python-version"
        single = (
            recorded.read_text(encoding="utf-8").strip() if recorded.is_file() else ""
        )
        if not single:  # pragma: no cover - every generated package records one
            import platform  # noqa: PLC0415

            single = platform.python_version()
        targets = [single]

    destination.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="scenario-wheelhouse-"))
    log = ""
    try:
        requirements, export_log = _export_requirements(package_root)
        log += export_log

        pinned = scratch / "dependencies.txt"
        pinned.write_text(f"{requirements}\n", encoding="utf-8")

        log += _build_project_wheel(package_root, destination)

        # A pass per interpreter, all into the same directory. pip resolves the
        # requirements file against the interpreter that runs it -- markers and
        # wheel tags both -- so this is the only way to end up with a directory
        # that installs under more than one. The passes cannot collide: a wheel
        # two interpreters share is byte-identical and is simply rewritten.
        for interpreter in targets:
            venv, venv_log = _builder_environment(scratch, interpreter)
            log += venv_log
            log += _download_wheels(venv, pinned, destination)

        # Inside the guard: the last two files are small, but the disk they go
        # on has just taken 160 MB of wheels, and a wheelhouse missing its
        # requirements.txt must not be what a failed build leaves behind.
        wheels = sorted(destination.glob("*.whl"))
        built = Wheelhouse(
            root=destination,
            distribution=distribution,
            version=version,
            wheels=tuple(path.name for path in wheels),
            size_bytes=sum(path.stat().st_size for path in wheels),
            python_tags=tuple(targets),
            log=log,
        )
        _write_install_files(built, requirements=requirements, run_command=run_command)
        logger.info("Built a wheelhouse of %d wheels at %s", len(wheels), destination)
        return built
    except WheelhouseError as exc:
        exc.log = f"{log}\n{exc.log}" if exc.log else log
        shutil.rmtree(destination, ignore_errors=True)
        raise
    except (UvUnavailable, subprocess.SubprocessError, OSError, TemplateError) as exc:
        # A tool that times out or cannot be spawned raises straight past the
        # checks above, and the destination is half-filled by then. Leaving it
        # would break the promise made below *and* refuse the next attempt,
        # which finds a non-empty directory.
        #
        # TemplateError belongs here for the same reason: the README is
        # rendered with StrictUndefined, so adding a variable to the template
        # and forgetting it at the call site raises past every other clause and
        # strands a wheelhouse holding every wheel and no README.
        shutil.rmtree(destination, ignore_errors=True)
        raise WheelhouseError(
            f"The wheelhouse build did not finish: {exc}", log
        ) from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _canonical(name: str) -> str:
    """Return *name* in PEP 503 normalised form, for comparing distributions."""
    return re.sub(r"[-_.]+", "-", name).lower()


#: What a requirement line looks like when uv writes the source instead of a
#: version. A git source comes through as ``name @ <url>``; a path source comes
#: through as the bare URL, with no name on it at all.
_DIRECT_SCHEMES = ("file:", "git+", "http:", "https:", "./", "../", "/")


def _referenced_distribution(reference: str) -> str:
    """Return the distribution a bare direct reference names.

    There is no name in the line to read, so it comes from the location: the
    subdirectory when the reference has one, otherwise the last path segment.
    That is a guess, and the caller only acts on it when it matches a wheel it
    actually built.
    """
    head = reference.split(";", 1)[0].strip()
    head, _, fragment = head.partition("#")
    subdirectory = re.search(r"subdirectory=([^&]+)", fragment)
    if subdirectory:
        return subdirectory.group(1).rstrip("/").split("/")[-1]
    if head.startswith("git+"):
        head = re.sub(r"@[^/@]+$", "", head)
    return head.rstrip("/").split("/")[-1].removesuffix(".git")


def _pin_direct_references(requirements: str, wheels: tuple[str, ...]) -> str:
    """Rewrite requirements that name a source to the version built for them.

    ``uv export`` keeps a git or path source as a *direct reference*, and pip
    honours a direct reference however many ``--find-links`` it was given: it
    clones the repository, or reads a directory on the exporting machine.  Both
    are exactly what a wheelhouse exists to avoid, and neither is there on the
    target.  The wheel is already in the directory, so naming it by version is
    what makes ``-r requirements.txt`` an offline install.

    Two shapes, because uv writes two: ``name @ <url>`` for a git source, and
    the bare URL for a path one.  A requirement whose wheel is not in the
    directory is left alone rather than guessed at.
    """
    versions = {}
    for wheel in wheels:
        name, version = wheel.split("-")[:2]
        versions[_canonical(name)] = version

    lines = []
    for line in requirements.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            lines.append(line)
            continue

        if " @ " in stripped:
            name, _, rest = stripped.partition(" @ ")
        elif stripped.startswith(_DIRECT_SCHEMES):
            # Nothing in the line is a name, so write the canonical one rather
            # than the directory spelling the location happened to use.
            name, rest = _canonical(_referenced_distribution(stripped)), stripped
        else:
            lines.append(line)
            continue

        pinned = versions.get(_canonical(name.split("[")[0]))
        if pinned is None:
            lines.append(line)
            continue
        # Markers travel with the requirement; the source does not.
        marker = f" ;{rest.split(';', 1)[1]}" if ";" in rest else ""
        lines.append(f"{name}=={pinned}{marker}")
    return "\n".join(lines)


def _write_install_files(
    wheelhouse: Wheelhouse, *, requirements: str, run_command: str
) -> None:
    """Write the two files that make the directory installable by hand.

    ``requirements.txt`` names the whole pinned set including the scenario
    itself, so a consumer can install exactly what was resolved rather than let
    pip pick from the directory; ``README.md`` says how, for the person who
    unzips it a month later.
    """
    (wheelhouse.root / "requirements.txt").write_text(
        "# Every distribution this scenario needs, pinned to what the package's\n"
        "# uv.lock resolved. Install it against this directory and nothing is\n"
        "# fetched from an index:\n"
        "#\n"
        "#     pip install --no-index --find-links . -r requirements.txt\n"
        f"{wheelhouse.distribution}=={wheelhouse.version}\n"
        f"{_pin_direct_references(requirements, wheelhouse.wheels)}\n",
        encoding="utf-8",
    )
    environment = code_environment(TEMPLATES_DIR)
    (wheelhouse.root / "README.md").write_text(
        environment.get_template("wheelhouse_README.md.jinja").render(
            distribution=wheelhouse.distribution,
            version=wheelhouse.version,
            pythons=list(wheelhouse.python_tags),
            wheel_count=len(wheelhouse.wheels),
            run_command=run_command,
            # A wheelhouse installs on the platform it was built for and no
            # other -- the interpreter is the axis it covers several of -- so
            # the layout of the venv it tells the reader to make is this
            # platform's: `Scripts` on Windows, `bin` everywhere else.
            venv_bin="Scripts" if os.name == "nt" else "bin",
        ),
        encoding="utf-8",
    )
