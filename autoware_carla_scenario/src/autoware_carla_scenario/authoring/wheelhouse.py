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
  built on cp312/linux-x86_64 installs on cp312/linux-x86_64;
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jinja2 import TemplateError

from ..templating import code_environment
from .framework_pin import DISTRIBUTION, framework_source_root
from .uv_tool import UvUnavailable, run_uv

logger = logging.getLogger(__name__)

__all__ = [
    "CARLA_WHEELS_ENV",
    "DEFAULT_CARLA_EXTRA",
    "Wheelhouse",
    "WheelhouseError",
    "build_wheelhouse",
    "carla_extra",
    "carla_wheels",
    "unpinned_carla_client",
    "venv_python",
]

#: Directory holding the ``*.jinja`` templates for a generated package.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

#: The framework extra to fall back on when no CARLA client is installed to
#: read the answer off.  Neither client is published to PyPI, so the only copies
#: that exist are the wheels vendored in this repository.
DEFAULT_CARLA_EXTRA = "carla"

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
        python_tag: The interpreter the wheels were resolved for, e.g. ``3.10``.
        log: Combined output of the tools that ran.
    """

    root: Path
    distribution: str
    version: str
    wheels: tuple[str, ...] = ()
    size_bytes: int = 0
    python_tag: str = ""
    log: str = ""


# ---------------------------------------------------------------------------
# The CARLA client
# ---------------------------------------------------------------------------


def _carla_pins() -> dict[str, str]:
    """Return the client version each of the framework's CARLA extras pins.

    Read from the framework's own metadata rather than hard-coded here, so the
    two cannot disagree about which client a scenario runs against.
    """
    from importlib import metadata  # noqa: PLC0415

    try:
        requirements = metadata.requires(DISTRIBUTION) or []
    except metadata.PackageNotFoundError:  # pragma: no cover - always installed
        return {}
    pins: dict[str, str] = {}
    for requirement in requirements:
        pinned = re.match(r"\s*carla\s*==\s*([0-9][^\s;]*)", requirement)
        extra = re.search(r"extra\s*==\s*[\"']([^\"']+)[\"']", requirement)
        if pinned and extra:
            pins[extra.group(1)] = pinned.group(1)
    return pins


def _installed_carla() -> Optional[str]:
    """Return the CARLA client version installed here, or ``None``."""
    from importlib import metadata  # noqa: PLC0415

    try:
        return metadata.version("carla")
    except metadata.PackageNotFoundError:
        return None


def _release(version: str) -> str:
    """Return *version* without its local segment, e.g. ``0.10.0+build`` -> ``0.10.0``.

    A locally built client is the same client: ``0.10.0+custom`` is the release
    the ``carla`` extra pins, compiled somewhere else.
    """
    return version.split("+", 1)[0]


def carla_extra() -> str:
    """Return the framework extra that installs the client this export needs.

    The framework declares two mutually exclusive clients -- ``carla`` for
    0.10.0 and ``carla-0-9-16`` for the legacy one -- and a scenario was
    authored against whichever of them is installed here.  Naming a fixed one
    would hand somebody working on the legacy client an export that installs
    cleanly and cannot run, with nothing saying why.

    Falls back to :data:`DEFAULT_CARLA_EXTRA` when no client is installed to
    read the answer off, and when one is installed that no extra pins -- see
    :func:`unpinned_carla_client`, which is how the caller says so out loud.
    """
    installed = _installed_carla()
    if installed is None:
        return DEFAULT_CARLA_EXTRA
    for extra, pinned in _carla_pins().items():
        if pinned == _release(installed):
            return extra
    return DEFAULT_CARLA_EXTRA


def unpinned_carla_client() -> Optional[str]:
    """Return the installed client's version when no extra pins it.

    ``None`` covers both the cases there is nothing to say about: no client
    installed, or one that an extra names exactly.  Anything else -- 0.9.15,
    say -- means the export is about to request a *different* client from the
    one the scenario was authored and validated against, which is worth saying
    rather than defaulting quietly.
    """
    installed = _installed_carla()
    if installed is None or _release(installed) in set(_carla_pins().values()):
        return None
    return installed


def _wheel_search_root() -> Optional[Path]:
    """Return the directory the vendored CARLA wheels live in, if it exists."""
    override = os.environ.get(CARLA_WHEELS_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_dir() else None

    # <repo>/autoware_carla_scenario -> <repo>/carla_wheels.  An installed
    # framework has no repository above it and so has no vendored wheels.
    candidate = framework_source_root().parent / VENDORED_WHEELS_DIR
    return candidate if candidate.is_dir() else None


def carla_wheels(extra: str = "") -> list[Path]:
    """Return the vendored CARLA wheels for *extra*, defaulting to this export's.

    One version, every interpreter tag of it. Only the one version, because the
    framework's two client extras are mutually exclusive and shipping both
    would put two CARLA clients that cannot coexist in the same wheelhouse.

    Returns:
        The matching wheels, or an empty list when none can be found -- which is
        the normal case for a framework installed from a wheel rather than run
        out of its repository.
    """
    version = _carla_pins().get(extra or carla_extra())
    root = _wheel_search_root()
    if version is None or root is None:
        return []
    # Every interpreter's wheel, not just the running one. A package vendoring
    # only its own tag looks tidier and does not lock: the generated package
    # inherits the framework's whole `requires-python`, and `uv lock` resolves
    # the client across all of it. The wheelhouse built from the lock still
    # holds exactly one -- pip takes the tag it can install.
    return sorted(root.glob(f"carla-{version}-*.whl"))


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
    if result.returncode != 0 or not venv_python(venv).exists():
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
    version: str,
    run_command: str,
    python: str = "",
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
        python: Interpreter version to resolve the wheels for.  Defaults to the
            package's ``.python-version``.

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
            python_tag=python,
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
            python=wheelhouse.python_tag,
            wheel_count=len(wheelhouse.wheels),
            run_command=run_command,
            # A wheelhouse installs on the platform it was built for and no
            # other, so the layout of the venv it tells the reader to make is
            # this platform's -- `Scripts` on Windows, `bin` everywhere else.
            venv_bin="Scripts" if os.name == "nt" else "bin",
        ),
        encoding="utf-8",
    )
