"""Export an authored scenario as a reproducible Scenario Package.

The editor's output is not a YAML file -- it is a directory that another
machine can copy, sync and run:

1. validate the document;
2. render the Hydra config;
3. render the package source;
4. pin the framework to an exact version or commit (never a branch);
5. record the exporting Python's exact patch version and the uv version;
6. generate ``uv.lock``;
7. build a wheelhouse: the whole locked graph, as wheels;
8. write a machine-readable manifest;
9. verify the result with ``uv sync --locked`` and the package's own tests.

Step 7 is what makes the export *usable* rather than merely reproducible.  The
directory itself is a uv project and needs uv, git and a network to install;
the wheelhouse built from it needs pip and none of those, which is what the
environments scenarios actually run in have.  See :mod:`.wheelhouse`.

Steps 6, 7 and 9 can fail for reasons outside the scenario (no network, an
unpushed commit).  When they do, the export **fails**: a package whose
dependencies never resolved is not a successful export, so the half-built
directory is discarded rather than left behind looking finished.  Everything is
built in a temporary directory and moved into place only once the checks pass.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from ..templating import code_environment, toml_string
from .framework_pin import Pin, PinResolutionError, resolve_framework_pin
from .hydra_config import dump_scenario_config
from .models import ScenarioDocument
from .persistence import dump_document_yaml, dump_yaml, utc_timestamp
from .uv_tool import UvUnavailable, run_uv
from .uv_tool import uv_version as _uv_version
from .validator import validate_document
from .wheelhouse import (
    VENDORED_WHEELS_DIR,
    Wheelhouse,
    WheelhouseError,
    build_wheelhouse,
    carla_extra,
    carla_wheels,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExportResult",
    "MANIFEST_FORMAT_VERSION",
    "PackageExportError",
    "export_package",
    "package_names",
]

#: Bumped when the manifest's own shape changes.  Version 2 moved the document
#: and the Hydra config inside the package module, so that they reach the wheel,
#: and added the ``wheelhouse`` section.
MANIFEST_FORMAT_VERSION = 2

#: Directory holding the ``*.jinja`` templates for a generated package.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

#: Version a generated package declares.  The wheelhouse writes it into its
#: ``requirements.txt``, so the two are the same literal or that file names a
#: wheel the directory does not hold.
PACKAGE_VERSION = "0.1.0"

_LOCK_TIMEOUT_SECONDS = 900
_TEST_TIMEOUT_SECONDS = 900


class PackageExportError(RuntimeError):
    """Raised when a package could not be exported completely.

    Attributes:
        log: Captured tool output, when the failure came from ``uv``.
    """

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log


@dataclass
class ExportResult:
    """What an export produced.

    Attributes:
        root: The generated package directory.
        manifest: The manifest that was written.
        pin: How the framework was pinned.
        locked: Whether ``uv.lock`` was generated.
        verified: Whether ``uv sync --locked`` succeeded against it.
        tested: Whether the package's own tests were run and passed.
        wheelhouse: The pip-installable wheelhouse built from the lock, when
            one was asked for and could be built.
        warnings: Reproducibility caveats worth showing the user.
        log: Combined output of the tools that ran.
    """

    root: Path
    manifest: dict[str, Any]
    pin: Pin
    locked: bool = False
    verified: bool = False
    tested: bool = False
    wheelhouse: Optional[Wheelhouse] = None
    warnings: list[str] = field(default_factory=list)
    log: str = ""


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _sanitize_identifier(raw: str) -> str:
    """Return *raw* reduced to a valid lower_snake_case Python identifier."""
    snake = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_").lower()
    snake = re.sub(r"__+", "_", snake)
    if not snake or snake[0].isdigit():
        snake = f"scenario_{snake}" if snake else "scenario"
    return snake


def package_names(document: ScenarioDocument) -> dict[str, str]:
    """Return the naming variants a generated package needs."""
    scenario_id = _sanitize_identifier(document.id)
    package_name = (
        scenario_id if scenario_id.endswith("_scenario") else f"{scenario_id}_scenario"
    )
    return {
        "scenario_id": scenario_id,
        "package_name": package_name,
        "distribution_name": package_name.replace("_", "-"),
        "document_path_env": f"{package_name.upper()}_DOCUMENT_PATH",
    }


# ---------------------------------------------------------------------------
# Environment probing
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _requires_python() -> str:
    """Return the framework's own ``requires-python``, or a safe fallback.

    Installed metadata cannot change while the process runs, so this is read
    from disk once however many packages get exported.
    """
    from importlib import metadata  # noqa: PLC0415

    try:
        value = metadata.metadata("autoware-carla-scenario")["Requires-Python"]
    except (metadata.PackageNotFoundError, KeyError):  # pragma: no cover
        value = None
    if value:
        return str(value)
    major, minor = sys.version_info[:2]
    return f">={major}.{minor},<{major}.{minor + 1}"


def _python_version() -> str:
    """Return the exporting interpreter's exact version, e.g. ``3.10.20``."""
    import platform  # noqa: PLC0415

    return platform.python_version()


def _editor_version() -> Optional[str]:
    """Return the framework version that generated the package."""
    from importlib import metadata  # noqa: PLC0415

    try:
        return metadata.version("autoware-carla-scenario")
    except metadata.PackageNotFoundError:  # pragma: no cover - always installed
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_source_entry(pin: Pin) -> str:
    """Return the inline ``[tool.uv.sources]`` body for *pin*, or an empty string."""
    source = pin.uv_source()
    if source is None:
        return ""
    parts = []
    for key, value in source.items():
        if isinstance(value, bool):
            parts.append(f"{key} = {str(value).lower()}")
        else:
            parts.append(f"{key} = {toml_string(value)}")
    return ", ".join(parts)


def _pins(pin: Pin) -> list[Pin]:
    """Return every distribution the exported package must pin.

    The framework imports the converter at module scope without declaring it,
    so a package that depended on the framework alone would fail to import it.
    Both are pinned the same way rather than leaving the second to chance.
    """
    return [pin, pin.companion()]


def _pin_note(pin: Pin) -> str:
    """Return the comment written above the framework dependency."""
    if pin.kind == "version":
        return (
            "Pinned to an exact release: a scenario package is only "
            "reproducible if the framework underneath it cannot move."
        )
    if pin.kind == "git":
        return (
            "Pinned to an exact commit (see [tool.uv.sources]) rather than a "
            "branch, which would move under the package."
        )
    return (
        "Development export: a local path, which does NOT resolve on another "
        "machine. Re-export without development mode before sharing."
    )


def _pin_summary(pin: Pin) -> str:
    """Return a one-line, human-readable description of *pin* for the README."""
    if pin.kind == "version":
        return f"version `{pin.version}` (exact)"
    if pin.kind == "git":
        return f"commit `{pin.commit}` of `{pin.repository}`"
    return f"local path `{pin.path}` -- **not portable**"


def _vendor_carla_wheels(root: Path, extra: str, warnings: list[str]) -> Optional[str]:
    """Copy the CARLA client wheels into the package.  Returns their directory.

    The package always *asks* for the client through the framework's extra; the
    question here is only where the client comes from.  0.10.0 is published to
    no index, so the wheel has to travel with the package: copying it in and
    pointing uv at it with a relative ``find-links`` keeps the package
    self-contained wherever it is copied.  0.9.16 is on PyPI, so nothing needs
    vendoring and the resolver finds it.

    Args:
        root: The package being written.
        extra: The framework extra whose client to vendor.
        warnings: Appended to when no local wheel was found.

    Returns:
        The relative directory the wheels were copied into, or ``None`` when
        there was no local wheel to copy -- either because the client is on an
        index, or because the framework is installed rather than run out of its
        repository.
    """
    wheels = carla_wheels(extra)
    if not wheels:
        warnings.append(
            f"No local wheel was vendored for the '{extra}' CARLA client, so "
            "it has to resolve from an index. If that client is not published "
            "there, locking will fail -- point SCENARIO_EXPORT_CARLA_WHEELS at "
            "a directory holding its wheel."
        )
        return None
    destination = root / VENDORED_WHEELS_DIR
    destination.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        shutil.copy2(wheel, destination / wheel.name)
    return VENDORED_WHEELS_DIR


def _write_package_tree(
    root: Path,
    document: ScenarioDocument,
    names: dict[str, str],
    pin: Pin,
    uv_version: Optional[str],
    vendored_wheels: Optional[str],
    warnings: list[str],
) -> dict[str, str]:
    """Render every file of the package under *root*.  Returns the file map."""
    env = code_environment(TEMPLATES_DIR)
    package_name = names["package_name"]
    scenario_id = names["scenario_id"]
    generated_at = utc_timestamp()
    pins = _pins(pin)
    sources = [(p.distribution, _render_source_entry(p)) for p in pins]

    context: dict[str, Any] = {
        **names,
        "description": document.description
        or f"{document.title} scenario, authored with the Scenario Editor.",
        "package_version": PACKAGE_VERSION,
        "requires_python": _requires_python(),
        "requirements": [p.requirement() for p in pins],
        "sources": [(name, body) for name, body in sources if body],
        "pin_summaries": [(p.distribution, _pin_summary(p)) for p in pins],
        "pin_note": _pin_note(pin),
        "uv_required_version": uv_version,
        "vendored_wheels": vendored_wheels,
        "uv_pin_summary": (
            f"`{uv_version}` (`tool.uv.required-version`)"
            if uv_version
            else "not recorded -- uv version could not be determined at export time"
        ),
        "python_version": _python_version(),
        "map_group": document.map.group,
        "generated_at": generated_at,
        "action_count": len(document.actions),
        "pass_count": len(document.assertions.pass_conditions),
        "fail_count": len(document.assertions.fail_conditions),
        "warnings": warnings,
    }

    files = {
        "pyproject.toml.jinja": "pyproject.toml",
        "README.md.jinja": "README.md",
        "package_init.py.jinja": f"src/{package_name}/__init__.py",
        "package_scenario.py.jinja": f"src/{package_name}/scenario.py",
        "package_test.py.jinja": "tests/test_scenario.py",
    }
    for template_name, relative in files.items():
        out_path = root / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            env.get_template(template_name).render(**context), encoding="utf-8"
        )

    (root / ".python-version").write_text(f"{_python_version()}\n", encoding="utf-8")
    (root / "src" / package_name / "py.typed").write_text("", encoding="utf-8")

    # Inside the module, not beside it: everything under ``src/<package>/``
    # reaches the wheel, and an installed scenario has no project directory
    # left to read a document out of.
    document_rel = f"src/{package_name}/document.yaml"
    hydra_rel = f"src/{package_name}/conf/scenario/{scenario_id}.yaml"

    document_path = root / document_rel
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(dump_document_yaml(document), encoding="utf-8")

    hydra_path = root / hydra_rel
    hydra_path.parent.mkdir(parents=True, exist_ok=True)
    hydra_path.write_text(dump_scenario_config(document), encoding="utf-8")

    (root / ".gitignore").write_text(
        "# uv.lock is intentionally tracked: it is what makes this package\n"
        "# reproducible. Everything below is build output.\n"
        ".venv/\n"
        "__pycache__/\n"
        "*.egg-info/\n"
        "dist/\n"
        "outputs/\n"
        "multirun/\n",
        encoding="utf-8",
    )

    return {
        "document": document_rel,
        "hydra_config": hydra_rel,
        "manifest": "scenario/manifest.yaml",
        "lockfile": "uv.lock",
        "python_version": ".python-version",
    }


def _build_manifest(
    document: ScenarioDocument,
    names: dict[str, str],
    pin: Pin,
    uv_version: Optional[str],
    files: dict[str, str],
    wheelhouse: Optional[Wheelhouse],
    warnings: list[str],
) -> dict[str, Any]:
    """Return the machine-readable manifest for the exported package.

    Every value here was actually observed at export time.  Anything that could
    not be determined is recorded as ``null`` with a note, never guessed.
    """
    runtime: dict[str, Any] = {
        "python": _python_version(),
        "uv": uv_version,
        "requires_python": _requires_python(),
    }
    notes = list(warnings)
    if uv_version is None:
        notes.append(
            "The uv version was not recorded: uv could not be interrogated at "
            "export time, so no required-version was written."
        )

    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "scenario": {
            "id": names["scenario_id"],
            "title": document.title,
            "document_version": document.version,
            "package": names["distribution_name"],
        },
        "runtime": runtime,
        "wheelhouse": (
            {
                "directory": wheelhouse.root.name,
                "install": (
                    f"pip install --no-index --find-links {wheelhouse.root.name} "
                    f"{wheelhouse.distribution}"
                ),
                "wheels": len(wheelhouse.wheels),
                "python": wheelhouse.python_tag,
            }
            if wheelhouse is not None
            else None
        ),
        "autoware_carla_scenario": pin.manifest(),
        "dependencies": {p.distribution: p.manifest() for p in _pins(pin)},
        "generated_by": {
            "editor_version": _editor_version(),
            "generated_at": utc_timestamp(),
        },
        "files": files,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# uv steps
# ---------------------------------------------------------------------------


def _run_uv(root: Path, *args: str, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run ``uv`` inside *root* and return the completed process.

    Raises:
        PackageExportError: If uv is not installed.
    """
    try:
        return run_uv(root, *args, timeout=timeout)
    except UvUnavailable as exc:
        raise PackageExportError(str(exc)) from exc


def _lock(root: Path) -> str:
    """Generate ``uv.lock``.

    Raises:
        PackageExportError: If locking fails or produces no lockfile.
    """
    result = _run_uv(root, "lock", timeout=_LOCK_TIMEOUT_SECONDS)
    log = f"$ uv lock\n{result.stdout}{result.stderr}\n"
    if result.returncode != 0:
        raise PackageExportError(
            "Dependency locking failed, so the exported package would not be "
            "reproducible. No package was written.",
            log=log,
        )
    if not (root / "uv.lock").is_file():
        raise PackageExportError(
            "uv lock reported success but produced no uv.lock.", log=log
        )
    return log


def _check_lock(root: Path) -> str:
    """Assert that ``uv.lock`` still matches ``pyproject.toml`` at *root*.

    Run after the package is moved into its final location: a path dependency
    is recorded in the lockfile *relative to the package*, so a move that
    changed the package's depth would silently invalidate a lockfile that was
    perfectly good where it was generated.

    Raises:
        PackageExportError: If the lockfile no longer matches.
    """
    result = _run_uv(root, "lock", "--check", timeout=_LOCK_TIMEOUT_SECONDS)
    log = f"$ uv lock --check\n{result.stdout}{result.stderr}\n"
    if result.returncode != 0:
        raise PackageExportError(
            "uv.lock does not match pyproject.toml in the exported package, so "
            "'uv sync --locked' would fail. No package was written.",
            log=log,
        )
    return log


def _verify_sync(root: Path) -> str:
    """Check that ``uv sync --locked`` succeeds against the generated lockfile.

    Raises:
        PackageExportError: If the sync fails.
    """
    result = _run_uv(root, "sync", "--locked", timeout=_LOCK_TIMEOUT_SECONDS)
    log = f"$ uv sync --locked\n{result.stdout}{result.stderr}\n"
    if result.returncode != 0:
        raise PackageExportError(
            "'uv sync --locked' failed in the exported package, so it is not "
            "reproducible. No package was written.",
            log=log,
        )
    return log


def _run_tests(root: Path) -> tuple[bool, str]:
    """Run the generated package's own tests.  Returns ``(passed, log)``."""
    result = _run_uv(
        root, "run", "--locked", "pytest", "-q", timeout=_TEST_TIMEOUT_SECONDS
    )
    log = f"$ uv run --locked pytest -q\n{result.stdout}{result.stderr}\n"
    return result.returncode == 0, log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


#: Machine-specific build output that verification leaves behind and that must
#: not travel with the package.
_BUILD_OUTPUT_DIRS = (".venv", ".pytest_cache", ".ruff_cache", "__pycache__")


def _strip_build_output(root: Path) -> None:
    """Delete everything the lock/verify/test steps generated inside *root*.

    The walk is materialised before anything is deleted: ``rglob`` is lazy, so
    removing a directory while iterating would have it try to descend into a
    path that no longer exists.
    """
    doomed = [
        path
        for path in list(root.rglob("*"))
        if path.is_dir()
        and (path.name in _BUILD_OUTPUT_DIRS or path.suffix == ".egg-info")
    ]
    for path in doomed:
        shutil.rmtree(path, ignore_errors=True)


def _self_check(
    staging: Path,
    *,
    lock: bool,
    verify: bool,
    run_tests: bool,
    warnings: list[str],
) -> tuple[tuple[bool, bool, bool], str]:
    """Lock, sync and test the staged package.

    Each step gates the next: there is nothing to sync without a lockfile, and
    nothing to test without a synced environment.  Returns
    ``((locked, verified, tested), log)`` and appends to *warnings* whenever a
    step is skipped or a test run fails -- a package that is not reproducible
    must not come back looking finished.
    """
    if not lock:
        warnings.append(
            "Dependency locking was skipped: this package has no uv.lock "
            "and is not reproducible."
        )
        return (False, False, False), ""

    log = _lock(staging)
    if not verify:
        return (True, False, False), log

    log += _verify_sync(staging)
    if not run_tests:
        return (True, True, False), log

    passed, test_log = _run_tests(staging)
    log += test_log
    if not passed:
        warnings.append("The generated package's own tests failed; see the export log.")
    return (True, True, passed), log


def _build_wheelhouse(
    staging: Path,
    destination: Path,
    document: ScenarioDocument,
    names: dict[str, str],
) -> Wheelhouse:
    """Build the package's wheelhouse.

    Raises:
        PackageExportError: If it could not be built.  A scenario that cannot
            be installed where it runs is not an export anyone can use, so this
            fails the whole thing rather than coming back half-done.
    """
    run = f"scenario scenario={names['scenario_id']} map={document.map.group}"
    try:
        return build_wheelhouse(
            staging,
            destination,
            distribution=names["distribution_name"],
            version=PACKAGE_VERSION,
            run_command=run,
        )
    except WheelhouseError as exc:
        raise PackageExportError(
            f"The wheelhouse could not be built: {exc}", log=exc.log
        ) from exc


def export_package(
    document: ScenarioDocument,
    destination: str | Path,
    *,
    dev_mode: bool = False,
    lock: bool = True,
    verify: bool = True,
    run_tests: bool = True,
    pin_uv_version: bool = True,
    force: bool = False,
) -> ExportResult:
    """Export *document* as a reproducible Scenario Package under *destination*.

    Args:
        document: The scenario to export.
        destination: Parent directory.  Two directories are created inside it,
            both named after the scenario: the package, and -- whenever the
            package was locked -- the wheelhouse built from it.  The wheelhouse
            is a peer rather than a member: the package is meant to be
            committed and 160 MB of wheels are not.
        dev_mode: Allow a local-path dependency on the framework.  The result is
            not portable and says so in its manifest.
        lock: Generate ``uv.lock``.  Turning this off produces a package that is
            explicitly *not* reproducible and is only useful for tests.
        verify: Run ``uv sync --locked`` against the generated lockfile.
        run_tests: Run the generated package's own tests after syncing.  A test
            failure is reported as a warning, not an export failure -- the
            dependency graph is what an export guarantees.
        pin_uv_version: Write ``[tool.uv] required-version`` when the uv version
            could be determined.
        force: Replace existing directories of the same names.

    Returns:
        An :class:`ExportResult` describing what was produced.

    Raises:
        PackageExportError: If the document is invalid, the framework cannot be
            pinned immutably, the destination is occupied, or locking or
            verification failed.  Nothing is left at *destination* in that case.
    """
    report = validate_document(document)
    if not report.ok:
        detail = "; ".join(f"{i.path}: {i.message}" for i in report.errors)
        raise PackageExportError(f"Cannot export an invalid scenario: {detail}")

    names = package_names(document)
    parent = Path(destination).expanduser().resolve()
    target = parent / names["package_name"]
    wheelhouse_target = parent / f"{names['package_name']}_wheelhouse"
    for occupied in (target, wheelhouse_target):
        if not occupied.exists():
            continue
        if not force:
            raise PackageExportError(
                f"{occupied} already exists. Choose another destination or "
                "export with force to replace it."
            )
        if not occupied.is_dir():
            raise PackageExportError(f"{occupied} exists and is not a directory.")

    try:
        pin = resolve_framework_pin(dev_mode=dev_mode)
    except PinResolutionError as exc:
        raise PackageExportError(str(exc)) from exc

    warnings = list(pin.warnings)
    warnings.extend(f"{i.path}: {i.message}" for i in report.warnings)

    uv_version = _uv_version() if pin_uv_version else None

    parent.mkdir(parents=True, exist_ok=True)
    # The staging directory is a *sibling* of the target, not a directory
    # inside one: a path dependency is locked relative to the package, so
    # staging one level deeper and then moving would leave a lockfile whose
    # relative paths no longer resolve.
    staging = Path(
        tempfile.mkdtemp(prefix=f".{names['package_name']}.export-", dir=str(parent))
    )
    wheelhouse_staging = Path(
        tempfile.mkdtemp(
            prefix=f".{names['package_name']}_wheelhouse.-", dir=str(parent)
        )
    )

    log = ""
    built: Optional[Wheelhouse] = None
    try:
        # The extra is requested whichever way the client is obtained: it is
        # what puts `carla` in the wheelhouse, and a scenario that cannot
        # import it cannot run. Vendoring is the separate question of whether
        # a copy has to travel with the package -- 0.10.0 is on no index, so
        # it does; 0.9.16 is on PyPI, so it does not.
        extra = carla_extra()
        pin = replace(pin, extras=(extra,))
        vendored = _vendor_carla_wheels(staging, extra, warnings)

        files = _write_package_tree(
            staging, document, names, pin, uv_version, vendored, warnings
        )

        checks, check_log = _self_check(
            staging, lock=lock, verify=verify, run_tests=run_tests, warnings=warnings
        )
        locked, verified, tested = checks
        log += check_log

        # A wheelhouse *is* the lockfile resolved into wheels, so there is one
        # exactly when there is a lock -- which is the only reason an export
        # ever comes back without one.
        if locked:
            built = _build_wheelhouse(staging, wheelhouse_staging, document, names)
            log += built.log
            # Named for where it is about to be moved: the manifest below
            # records the directory a reader will actually find, not the
            # temporary one it was assembled in.
            built = replace(built, root=wheelhouse_target)
        else:
            warnings.append(
                "No wheelhouse was built: it is the lockfile resolved into "
                "wheels, and this package was exported without a lock."
            )

        manifest = _build_manifest(
            document, names, pin, uv_version, files, built, warnings
        )
        manifest_path = staging / files["manifest"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            dump_yaml(manifest),
            encoding="utf-8",
        )

        _strip_build_output(staging)

        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staging), str(target))
        if built is not None:
            if wheelhouse_target.exists():
                shutil.rmtree(wheelhouse_target)
            shutil.move(str(wheelhouse_staging), str(wheelhouse_target))
        elif wheelhouse_target.exists():
            # An earlier export left one and this one has none to put there.
            # Leaving it would sit stale wheels next to a fresh manifest that
            # says the package has no wheelhouse at all.
            shutil.rmtree(wheelhouse_target)

        if locked:
            try:
                log += _check_lock(target)
            except PackageExportError:
                # Both are already in place; an export that cannot be synced
                # must not be left behind looking finished -- and a wheelhouse
                # left there would also block the next export without `force`.
                shutil.rmtree(target, ignore_errors=True)
                shutil.rmtree(wheelhouse_target, ignore_errors=True)
                raise
    except PackageExportError as exc:
        exc.log = f"{exc.log}\n{log}" if exc.log else log
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(wheelhouse_staging, ignore_errors=True)

    logger.info("Exported scenario package to %s", target)
    return ExportResult(
        root=target,
        manifest=manifest,
        pin=pin,
        locked=locked,
        verified=verified,
        tested=tested,
        wheelhouse=built,
        warnings=warnings,
        log=log,
    )
