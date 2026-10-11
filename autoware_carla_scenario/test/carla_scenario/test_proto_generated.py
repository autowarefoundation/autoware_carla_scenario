"""The generated ``AutowareBridge`` protobuf modules match their proto.

The alpasim ``egodriver`` contract is checked by carla-driver-interface, which
vendors and generates it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_COMPILE_SCRIPT = _PACKAGE_ROOT / "scripts" / "compile_protos.py"


def _load_compiler():
    """Import ``scripts/compile_protos.py`` as a module."""
    spec = importlib.util.spec_from_file_location("_compile_protos", _COMPILE_SCRIPT)
    assert spec is not None and spec.loader is not None  # noqa: S101
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_modules_are_up_to_date() -> None:
    """Regenerating from the protos must not change the committed output."""
    pytest.importorskip(
        "grpc_tools", reason="grpcio-tools is only installed with the dev dependencies"
    )
    compiler = _load_compiler()
    assert compiler._check() == 0, (  # noqa: SLF001
        "Generated protobuf modules are stale. Run: "
        "uv run python autoware_carla_scenario/scripts/compile_protos.py"
    )
