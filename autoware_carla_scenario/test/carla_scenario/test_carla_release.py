"""The CARLA release the framework targets, and the client it selects for it."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from autoware_carla_scenario import carla_release
from autoware_carla_scenario.carla_release import (
    CARLA_REF_ENV,
    client_server_mismatch,
    version_family,
)


@pytest.mark.parametrize(
    ("version", "family"),
    [
        ("1.0", "1"),
        ("1.0-82c52a7c2", "1"),
        ("1.0-82c52a7c2-dirty", "1"),
        ("1.1", "1"),
        ("0.10.0", "0.10"),
        ("0.10.0-185-g0a5ce0d5b", "0.10"),
        ("0.9.16", "0.9"),
        (" 1.0 ", "1"),
    ],
)
def test_version_family(version: str, family: str) -> None:
    assert version_family(version) == family


def test_one_family_matches() -> None:
    assert client_server_mismatch("1.0", "1.0-82c52a7c2") is None
    assert client_server_mismatch("0.10.0", "0.10.0-185-g0a5ce0d5b") is None


def test_a_mismatch_names_the_client_to_select(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CARLA_REF_ENV, "1.0")
    problem = client_server_mismatch("1.0", "0.10.0")
    assert problem is not None
    assert f"{CARLA_REF_ENV}=1.0" in problem  # what is selected
    assert f"{CARLA_REF_ENV}=ue5-dev" in problem  # what the server wants
    problem = client_server_mismatch("0.10.0", "1.0")
    assert problem is not None
    assert f"{CARLA_REF_ENV}=1.0" in problem


def test_an_unknown_family_asks_for_the_servers_branch() -> None:
    problem = client_server_mismatch("1.0", "2.0")
    assert problem is not None
    assert "branch the server was built from" in problem


def test_the_default_is_left_to_a_choice_made(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CARLA_REF_ENV, "ue58-dev")
    carla_release.select_default_client()
    assert os.environ[CARLA_REF_ENV] == "ue58-dev"
    monkeypatch.delenv(CARLA_REF_ENV)
    carla_release.select_default_client()
    assert os.environ[CARLA_REF_ENV] == carla_release.DEFAULT_CARLA_REF


def test_importing_the_package_selects_the_10_client() -> None:
    """Before typesafe_carla is imported, so its LibCarla is the 1.0 one."""
    env = {k: v for k, v in os.environ.items() if k != CARLA_REF_ENV}
    code = (
        "import os, sys, autoware_carla_scenario.server;"
        f"print(os.environ['{CARLA_REF_ENV}'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip().splitlines()[-1] == "1.0"
