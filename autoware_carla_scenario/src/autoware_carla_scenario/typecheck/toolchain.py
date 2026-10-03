"""Locates the Codon compiler, as typesafe_carla does.

This is typesafe_carla's ``python/typesafe_carla/toolchain.py``
(https://github.com/hakuturu583/typesafe_carla, commit a5463c0), with the same
search order and environment, so one Codon setup serves both:

1. ``TYPESAFE_CODON``: path to a ``codon`` executable.
2. The ``typesafe-carla-toolchain`` package (a pinned, bundled Codon; the
   ``toolchain/`` workspace member), once it is installed. It must provide
   ``typesafe_carla_toolchain.codon_executable()``.
3. ``CODON_DIR``: a Codon installation directory (``$CODON_DIR/bin/codon``).
4. ``~/.codon/bin/codon`` (the official installer's location).
5. ``codon`` on ``PATH``.

:func:`codon_environment` is the environment typesafe_carla's launcher runs
Codon in (``CODON_DIR``, and ``LD_LIBRARY_PATH`` for the bundled runtime).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ENV_CODON",
    "SUPPORTED_CODON_SERIES",
    "Toolchain",
    "ToolchainError",
    "codon_environment",
    "find_codon",
    "is_supported_version",
]

# The Codon release series typesafe_carla is tested with.
SUPPORTED_CODON_SERIES = "0.19"

ENV_CODON = "TYPESAFE_CODON"


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


def _from_executable(exe: Path, source: str) -> Toolchain:
    exe = exe.expanduser().resolve()
    if not exe.is_file():
        raise ToolchainError(f"{source}: {exe} is not a file")
    return Toolchain(exe, exe.parent.parent, source)


def _bundled() -> Toolchain | None:
    try:
        import typesafe_carla_toolchain  # noqa: PLC0415
    except ImportError:
        return None
    try:
        exe = typesafe_carla_toolchain.codon_executable()
    except RuntimeError as e:  # installed without its Codon bundle
        raise ToolchainError(f"typesafe-carla-toolchain: {e}") from e
    return _from_executable(Path(exe), "typesafe-carla-toolchain")


def find_codon() -> Toolchain:
    explicit = os.environ.get(ENV_CODON)
    if explicit:
        return _from_executable(Path(explicit), ENV_CODON)
    bundled = _bundled()
    if bundled is not None:
        return bundled
    codon_dir = os.environ.get("CODON_DIR")
    if codon_dir and (Path(codon_dir) / "bin" / "codon").is_file():
        return _from_executable(Path(codon_dir) / "bin" / "codon", "CODON_DIR")
    home = Path.home() / ".codon" / "bin" / "codon"
    if home.is_file():
        return _from_executable(home, "~/.codon")
    on_path = shutil.which("codon")
    if on_path:
        return _from_executable(Path(on_path), "PATH")
    raise ToolchainError(
        "Codon compiler not found. Install typesafe-carla-toolchain, or Codon "
        f"{SUPPORTED_CODON_SERIES}.x (https://github.com/exaloop/codon/releases), and point "
        f"{ENV_CODON} at the codon executable if it is not in ~/.codon or on PATH."
    )


def is_supported_version(version: str) -> bool:
    return (
        version.strip().startswith(SUPPORTED_CODON_SERIES + ".")
        or version.strip() == SUPPORTED_CODON_SERIES
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
