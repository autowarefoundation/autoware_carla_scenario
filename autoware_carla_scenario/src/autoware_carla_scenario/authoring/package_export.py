"""Export an authored scenario as a reproducible Scenario Package.

The editor's output is not a YAML file -- it is a directory that another
machine can copy, sync and run:

1. validate the document;
2. render the Hydra config;
3. render the package source;
4. pin the framework to an exact version or commit (never a branch);
5. record the exporting Python's exact patch version and the uv version;
6. generate ``uv.lock``;
7. write a machine-readable manifest;
8. verify the result with ``uv sync --locked`` and the package's own tests.

Steps 6 and 8 can fail for reasons outside the scenario (no network, an
unpushed commit).  When they do, the export **fails**: a package whose
dependencies never resolved is not a successful export, so the half-built
directory is discarded rather than left behind looking finished.  Everything is
built in a temporary directory and moved into place only once the checks pass.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from ..templating import code_environment
from .framework_pin import Pin, PinResolutionError, resolve_framework_pin
from .hydra_config import dump_scenario_config
from .models import ScenarioDocument
from .persistence import dump_document_yaml, dump_yaml
from .validator import validate_document

logger = logging.getLogger(__name__)

__all__ = [
    "ExportResult",
    "MANIFEST_FORMAT_VERSION",
    "PackageExportError",
    "export_package",
    "package_names",
]

#: Bumped when the manifest's own shape changes.
MANIFEST_FORMAT_VERSION = 1

#: Directory holding the ``*.jinja`` templates for a generated package.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

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
        warnings: Reproducibility caveats worth showing the user.
        log: Combined output of the tools that ran.
    """

    root: Path
    manifest: dict[str, Any]
    pin: Pin
    locked: bool = False
    verified: bool = False
    tested: bool = False
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
        "package_dir_name": package_name,
        "distribution_name": package_name.replace("_", "-"),
        "document_path_env": f"{package_name.upper()}_DOCUMENT_PATH",
    }


# ---------------------------------------------------------------------------
# Environment probing
# ---------------------------------------------------------------------------


def _uv_executable() -> Optional[str]:
    """Return the path to ``uv``, or ``None`` when it is not installed."""
    return shutil.which("uv")


def _uv_version() -> Optional[str]:
    """Return the exact uv version, or ``None`` when it cannot be determined.

    A version is never guessed: when uv cannot be interrogated the manifest
    records the absence instead of inventing a plausible number.
    """
    uv = _uv_executable()
    if uv is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [uv, "--version"], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
    return match.group(1) if match else None


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
            parts.append(f'{key} = "{value}"')
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


def _write_package_tree(
    root: Path,
    document: ScenarioDocument,
    names: dict[str, str],
    pin: Pin,
    uv_version: Optional[str],
    warnings: list[str],
) -> dict[str, str]:
    """Render every file of the package under *root*.  Returns the file map."""
    env = code_environment(TEMPLATES_DIR)
    package_name = names["package_name"]
    scenario_id = names["scenario_id"]
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pins = _pins(pin)
    sources = [(p.distribution, _render_source_entry(p)) for p in pins]

    context: dict[str, Any] = {
        **names,
        "description": document.description
        or f"{document.title} scenario, authored with the Scenario Editor.",
        "package_version": "0.1.0",
        "requires_python": _requires_python(),
        "requirements": [p.requirement() for p in pins],
        "sources": [(name, body) for name, body in sources if body],
        "pin_summaries": [(p.distribution, _pin_summary(p)) for p in pins],
        "pin_note": _pin_note(pin),
        "uv_required_version": uv_version,
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

    document_rel = "scenario/document.yaml"
    hydra_rel = f"conf/scenario/{scenario_id}.yaml"

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
        "autoware_carla_scenario": pin.manifest(),
        "dependencies": {p.distribution: p.manifest() for p in _pins(pin)},
        "generated_by": {
            "editor_version": _editor_version(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    uv = _uv_executable()
    if uv is None:
        raise PackageExportError(
            "uv is not installed, so the package's dependencies cannot be "
            "locked. An unlocked package is not reproducible."
        )
    env = dict(os.environ)
    # A parent VIRTUAL_ENV would make uv operate on the editor's environment
    # instead of the package's own.
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run(  # noqa: S603
        [uv, *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )


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
    for name in _BUILD_OUTPUT_DIRS:
        shutil.rmtree(root / name, ignore_errors=True)
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
        destination: Parent directory; the package is created inside it, named
            after the scenario.
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
        force: Replace an existing package directory of the same name.

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
    target = parent / names["package_dir_name"]
    if target.exists():
        if not force:
            raise PackageExportError(
                f"{target} already exists. Choose another destination or export "
                "with force to replace it."
            )
        if not target.is_dir():
            raise PackageExportError(f"{target} exists and is not a directory.")

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
        tempfile.mkdtemp(
            prefix=f".{names['package_dir_name']}.export-", dir=str(parent)
        )
    )

    log = ""
    try:
        files = _write_package_tree(staging, document, names, pin, uv_version, warnings)

        checks, check_log = _self_check(
            staging, lock=lock, verify=verify, run_tests=run_tests, warnings=warnings
        )
        locked, verified, tested = checks
        log += check_log

        manifest = _build_manifest(document, names, pin, uv_version, files, warnings)
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

        if locked:
            try:
                log += _check_lock(target)
            except PackageExportError:
                # The package is already in place; an export that cannot be
                # synced must not be left behind looking finished.
                shutil.rmtree(target, ignore_errors=True)
                raise
    except PackageExportError as exc:
        exc.log = f"{exc.log}\n{log}" if exc.log else log
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info("Exported scenario package to %s", target)
    return ExportResult(
        root=target,
        manifest=manifest,
        pin=pin,
        locked=locked,
        verified=verified,
        tested=tested,
        warnings=warnings,
        log=log,
    )
