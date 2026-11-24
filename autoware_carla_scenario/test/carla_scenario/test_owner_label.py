"""utils.owner_label: the ``<host>:<pid>`` label on the containers a run
starts, and whether the process it names still runs."""

from __future__ import annotations

import os
import socket

import pytest

from autoware_carla_scenario.utils.owner_label import (
    owner_alive,
    owner_value,
    process_alive,
)


def test_this_process_is_alive() -> None:
    assert owner_alive(owner_value()) is True
    assert process_alive(os.getpid())


@pytest.mark.parametrize("owner", ["", "1234", ":1234", "host:", "host:12a", "host:-1"])
def test_a_malformed_label_cannot_be_told(owner: str) -> None:
    assert owner_alive(owner) is None


def test_another_hosts_label_cannot_be_told() -> None:
    assert owner_alive(f"not-{socket.gethostname()}:{os.getpid()}") is None


@pytest.mark.parametrize("pid", ["0", "00"])
def test_pid_zero_is_no_process(pid: str) -> None:
    # #2251: os.kill(0, 0) signals the caller's process group, read as alive.
    assert owner_alive(f"{socket.gethostname()}:{pid}") is None


@pytest.mark.parametrize("pid", [str(2**31), str(2**63), "9" * 40])
def test_a_pid_too_large_cannot_be_told(pid: str) -> None:
    # #2251: os.kill raised OverflowError, aborting the leftover cleanup.
    assert owner_alive(f"{socket.gethostname()}:{pid}") is None
