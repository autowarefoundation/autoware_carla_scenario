"""Locates the Codon compiler and typesafe_carla's Codon library.

Both come from typesafe_carla (https://github.com/hakuturu583/typesafe_carla),
the framework's CARLA client (a dependency on Linux x86_64 and aarch64): the compiler is found by
``typesafe_carla.toolchain.find_codon``, so one Codon setup serves both
projects, in its order:

1. ``TYPESAFE_CODON``: path to a ``codon`` executable.
2. The ``typesafe-carla-toolchain`` package (the pinned Codon typesafe-carla
   depends on).
3. ``CODON_DIR``: a Codon installation directory (``$CODON_DIR/bin/codon``).
4. ``~/.codon/bin/codon`` (the official installer's location).
5. ``codon`` on ``PATH``.

and the library is ``typesafe_carla.paths.codon_path_dir()``. Without
typesafe-carla installed there is neither, and :func:`find_codon` says so.

:func:`codon_environment` is the environment typesafe_carla's launcher runs
Codon in (``CODON_DIR``, and ``LD_LIBRARY_PATH`` for the bundled runtime).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ENV_CODON",
    "SUPPORTED_CODON_SERIES",
    "Toolchain",
    "ToolchainError",
    "VERSION_TIMEOUT_S",
    "codon_environment",
    "codon_path_dir",
    "find_codon",
    "is_supported_version",
]

#: The Codon release series the source rewrite and the model are written for.
#: The check's own pin: typesafe_carla moving to another series does not move
#: the model with it.
SUPPORTED_CODON_SERIES = "0.19"

#: typesafe_carla's variable naming a ``codon`` executable; kept here so it can
#: be read without typesafe_carla installed.
ENV_CODON = "TYPESAFE_CODON"

_NOT_INSTALLED = (
    "typesafe-carla is not installed: it is a dependency of autoware-carla-scenario "
    "on Linux x86_64 and aarch64, the only platforms it ships for"
)


#: How long ``codon --version`` may take before the executable counts as no
#: Codon at all (#2249): it runs before every check, and before every
#: ``scenario`` run under ``typecheck: auto``.
VERSION_TIMEOUT_S = 60.0


class ToolchainError(RuntimeError):
    pass


@dataclass(frozen=True)
class Toolchain:
    executable: Path
    codon_dir: Path  # installation root: bin/, lib/codon/
    source: str  # how it was found

    def version(self) -> str:
        """What ``codon --version`` prints.

        Raises:
            ToolchainError: The executable cannot be run (not executable,
                gone) or does not answer within :data:`VERSION_TIMEOUT_S`:
                no Codon, as the check treats any other lookup failure.
        """
        try:
            out = subprocess.run(  # noqa: S603 - the compiler the lookup found
                [str(self.executable), "--version"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=VERSION_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolchainError(
                f"{self.source}: {self.executable} --version did not answer "
                f"within {VERSION_TIMEOUT_S:g}s"
            ) from exc
        except OSError as exc:
            raise ToolchainError(
                f"{self.source}: {self.executable} cannot be run: {exc}"
            ) from exc
        return out.stdout.strip() or out.stderr.strip()

    def library_dirs(self) -> list[Path]:
        return [
            d
            for d in (self.codon_dir / "lib" / "codon", self.codon_dir / "lib")
            if d.is_dir()
        ]


def find_codon() -> Toolchain:
    """The Codon compiler, found as typesafe_carla finds it."""
    try:
        from typesafe_carla import toolchain as tsc_toolchain  # noqa: PLC0415
    except ImportError as exc:
        raise ToolchainError(_NOT_INSTALLED) from exc
    try:
        tc = tsc_toolchain.find_codon()
    except (tsc_toolchain.ToolchainError, RuntimeError) as exc:
        raise ToolchainError(str(exc)) from exc
    return Toolchain(Path(tc.executable), Path(tc.codon_dir), tc.source)


def codon_path_dir() -> Path:
    """typesafe_carla's ``CODON_PATH`` directory, as its launcher uses it.

    It holds the ``typesafe_carla`` library and the compile-time switches the
    library reads (``_tsc_build_config.codon``), outside strict mode: Python-API
    compatibility shortcuts compile, as they do for the launcher.
    """
    try:
        from typesafe_carla import paths as tsc_paths  # noqa: PLC0415
    except ImportError as exc:
        raise ToolchainError(_NOT_INSTALLED) from exc
    try:
        return tsc_paths.codon_path_dir(False)
    except tsc_paths.PathError as exc:
        raise ToolchainError(str(exc)) from exc


def is_supported_version(version: str) -> bool:
    version = version.strip()
    return version == SUPPORTED_CODON_SERIES or version.startswith(
        SUPPORTED_CODON_SERIES + "."
    )


def _prepend(value: str, existing: str | None) -> str:
    """*value* before *existing*, as a search path; an empty one is left out.

    An empty entry of ``LD_LIBRARY_PATH`` is the current directory to the
    loader, so neither may add one (#2511).
    """
    return os.pathsep.join(part for part in (value, existing) if part)


def codon_environment(tc: Toolchain, codon_path: Path) -> dict[str, str]:
    """The environment to run *tc* in, with *codon_path* as ``CODON_PATH``.

    As typesafe_carla's launcher builds it (``cli.build_environment``), less
    the native library the check never loads.
    """
    env = os.environ.copy()
    env["CODON_DIR"] = str(tc.codon_dir)
    env["CODON_PATH"] = str(codon_path)  # one directory: Codon reads no more
    ld = [str(d) for d in tc.library_dirs()]
    library_path = _prepend(os.pathsep.join(ld), env.get("LD_LIBRARY_PATH"))
    if library_path:
        env["LD_LIBRARY_PATH"] = library_path
    return env
