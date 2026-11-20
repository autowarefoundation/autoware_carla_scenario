"""``codon --version``: a Codon that cannot answer is no Codon (#2249).

None of these runs Codon: the executables are shell scripts, or not
executable at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoware_carla_scenario.typecheck import check, toolchain
from autoware_carla_scenario.typecheck.toolchain import Toolchain, ToolchainError


def _script(tmp_path: Path, body: str, mode: int = 0o755) -> Toolchain:
    codon = tmp_path / "codon"
    codon.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    codon.chmod(mode)
    return Toolchain(codon, tmp_path, "TYPESAFE_CODON")


def test_the_version_is_what_codon_prints(tmp_path: Path) -> None:
    assert _script(tmp_path, "echo 0.19.3").version() == "0.19.3"


def test_a_codon_that_is_not_executable_is_a_toolchain_error(tmp_path: Path) -> None:
    tc = _script(tmp_path, "echo 0.19.3", mode=0o644)
    with pytest.raises(ToolchainError, match="cannot be run"):
        tc.version()


def test_a_codon_that_is_gone_is_a_toolchain_error(tmp_path: Path) -> None:
    tc = Toolchain(tmp_path / "missing", tmp_path, "TYPESAFE_CODON")
    with pytest.raises(ToolchainError, match="cannot be run"):
        tc.version()


def test_a_codon_that_hangs_is_a_toolchain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(toolchain, "VERSION_TIMEOUT_S", 0.5)
    tc = _script(tmp_path, "exec sleep 30")
    with pytest.raises(ToolchainError, match="did not answer within 0.5s"):
        tc.version()


def test_a_codon_that_cannot_answer_is_no_codon_to_the_checker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What `typecheck: auto` asks before a run: no Codon, so it warns.
    tc = _script(tmp_path, "echo 0.19.3", mode=0o644)
    monkeypatch.setattr(check, "find_codon", lambda: tc)
    assert check.available_toolchain() is None
    with pytest.raises(ToolchainError):
        check.find_supported_codon()


def test_a_codon_whose_version_fails_later_skips_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cache key reads the version again: a Codon that stopped answering
    # since it was found is no Codon, not a traceback.
    tc = _script(tmp_path, "echo 0.19.3", mode=0o644)
    monkeypatch.setattr(check, "codon_path_dir", lambda: tmp_path)
    result = check.typecheck_library(
        ["autoware_carla_scenario.kinematics.vector"], toolchain=tc
    )
    assert result.skipped is not None
    assert result.skipped.startswith("no Codon compiler: ")
