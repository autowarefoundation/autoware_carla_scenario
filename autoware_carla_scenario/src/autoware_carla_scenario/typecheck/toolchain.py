"""Locates the Codon compiler and typesafe_carla's Codon library.

Both come from typesafe_carla (https://github.com/hakuturu583/typesafe_carla),
installed with the ``codon`` extra: the compiler is found by
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
    "typesafe-carla is not installed: install the `codon` extra "
    "(autoware-carla-scenario[codon], Linux x86_64), or `uv sync --dev`"
)


class ToolchainError(RuntimeError):
    pass


@dataclass(frozen=True)
class Toolchain:
    executable: Path
    codon_dir: Path  # installation root: bin/, lib/codon/
    source: str  # how it was found

    def version(self) -> str:
        out = subprocess.run(
            [str(self.executable), "--version"], capture_output=True, text=True
        )
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
    return value if not existing else value + os.pathsep + existing


def codon_environment(tc: Toolchain, codon_path: Path) -> dict[str, str]:
    """The environment to run *tc* in, with *codon_path* as ``CODON_PATH``.

    As typesafe_carla's launcher builds it (``cli.build_environment``), less
    the native library the check never loads.
    """
    env = os.environ.copy()
    env["CODON_DIR"] = str(tc.codon_dir)
    env["CODON_PATH"] = str(codon_path)  # one directory: Codon reads no more
    ld = [str(d) for d in tc.library_dirs()]
    env["LD_LIBRARY_PATH"] = _prepend(os.pathsep.join(ld), env.get("LD_LIBRARY_PATH"))
    return env
