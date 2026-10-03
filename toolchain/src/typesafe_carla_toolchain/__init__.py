"""A pinned Codon compiler, bundled for typesafe-carla.

    >>> import typesafe_carla_toolchain
    >>> typesafe_carla_toolchain.codon_dir()   # contains bin/codon and lib/codon/
"""

from __future__ import annotations

from pathlib import Path

CODON_VERSION = "0.19.3"

_ROOT = Path(__file__).resolve().parent / "codon"


def codon_dir() -> Path:
    """Root of the bundled Codon installation (``bin/codon``, ``lib/codon/``)."""
    if not (_ROOT / "bin" / "codon").is_file():
        raise RuntimeError(
            "typesafe-carla-toolchain is installed without its Codon bundle; "
            "it must be installed from a wheel, not from an unbuilt source tree")
    return _ROOT


def codon_executable() -> Path:
    return codon_dir() / "bin" / "codon"
