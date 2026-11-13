"""What ``launcher.*`` (slurm.settings) and the run panel's SLURM fields
(ui.slurm_launch) refuse, and the values they read as before (#2133).

A falsy value is not an unset one: JSON's 0 is a count of 0, ``type: false``
is no launcher, ``env: []`` no mapping; a null or blank list item no
``sbatch`` argument; and the port range check counts the ports a slot uses.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from autoware_carla_scenario.slurm.port_slots import SlotPorts, slot_order
from autoware_carla_scenario.slurm.settings import SlurmSettings, launcher_type
from autoware_carla_scenario.ui.slurm_launch import SlurmOptions

# ── The run panel's numbers ────────────────────────────────────────────────


@pytest.mark.parametrize("key", ["array_parallelism", "cases_per_task"])
@pytest.mark.parametrize("value", [0, "0", " 0 ", -1, "-1"])
def test_the_run_panel_refuses_a_count_below_one(key: str, value: Any) -> None:
    # JSON's 0 read as an unset field (`data.get(key) or ""`).
    with pytest.raises(ValueError) as caught:
        SlurmOptions.from_request({key: value})
    assert str(caught.value) == f"{key} must be at least 1"


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (2.5, "2.5"),
        (0.0, "0.0"),
        ("2.5", "2.5"),
        ("1e3", "1e3"),
        ("two", "two"),
        (True, "True"),
        (False, "False"),
        ([2], "[2]"),
    ],
)
def test_the_run_panel_asks_for_a_whole_number(value: Any, shown: str) -> None:
    with pytest.raises(ValueError) as caught:
        SlurmOptions.from_request({"array_parallelism": value})
    assert str(caught.value) == (
        f"array_parallelism must be a whole number, not {shown!r}"
    )


def test_the_run_panel_asks_seconds_for_a_number() -> None:
    with pytest.raises(ValueError) as caught:
        SlurmOptions.from_request({"start_stagger_seconds": "soon"})
    assert str(caught.value) == "start_stagger_seconds must be a number, not 'soon'"


@pytest.mark.parametrize("unset", [None, "", "  "])
@pytest.mark.parametrize(
    "key", ["array_parallelism", "cases_per_task", "start_stagger_seconds"]
)
def test_the_run_panel_leaves_a_missing_null_or_blank_number_unset(
    key: str, unset: Any
) -> None:
    assert getattr(SlurmOptions.from_request({key: unset}), key) is None
    assert getattr(SlurmOptions.from_request({}), key) is None


def test_the_run_panel_reads_numbers_as_before() -> None:
    options = SlurmOptions.from_request(
        {
            "array_parallelism": 3,
            "cases_per_task": " 2 ",
            "start_stagger_seconds": "1.5e1",
        }
    )
    assert (
        options.array_parallelism,
        options.cases_per_task,
        options.start_stagger_seconds,
    ) == (3, 2, 15.0)
    # JSON's 0 seconds is 0 seconds, as "0" is.
    assert (
        SlurmOptions.from_request({"start_stagger_seconds": 0}).start_stagger_seconds
        == 0.0
    )
    assert (
        SlurmOptions.from_request({"start_stagger_seconds": "0"}).start_stagger_seconds
        == 0.0
    )


# ── launcher.type ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "shown"),
    [(False, "False"), (0, "0"), ("", "''"), (True, "True"), ("pbs", "'pbs'")],
)
def test_a_falsy_launcher_type_is_no_launcher(value: Any, shown: str) -> None:
    with pytest.raises(ValueError) as caught:
        launcher_type({"launcher": {"type": value}})
    assert str(caught.value) == (
        f"Unknown launcher.type: {shown}. Expected one of: 'local', 'slurm'."
    )


def test_a_missing_or_null_launcher_type_is_local() -> None:
    assert launcher_type(None) == "local"
    assert launcher_type({}) == "local"
    assert launcher_type({"launcher": None}) == "local"
    assert launcher_type({"launcher": {}}) == "local"
    assert launcher_type({"launcher": {"type": None}}) == "local"
    assert launcher_type({"launcher": {"type": "local"}}) == "local"
    assert launcher_type({"launcher": {"type": "slurm"}}) == "slurm"


# ── launcher.env ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "shown"),
    [([], "[]"), (0, "0"), ("", "''"), (False, "False"), (["A=1"], "['A=1']")],
)
def test_launcher_env_must_be_a_mapping(value: Any, shown: str) -> None:
    with pytest.raises(ValueError) as caught:
        SlurmSettings.from_mapping({"env": value})
    assert str(caught.value) == f"launcher.env must be a mapping, got {shown}"


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (None, "None"),
        (True, "True"),
        (False, "False"),
        (["x"], "['x']"),
        ({"a": 1}, "{'a': 1}"),
    ],
)
def test_a_launcher_env_value_must_be_text(value: Any, shown: str) -> None:
    with pytest.raises(ValueError) as caught:
        SlurmSettings.from_mapping({"env": {"A": "1", "B": value}})
    assert str(caught.value) == f"launcher.env.B must be text, got {shown}"


def test_launcher_env_reads_as_before() -> None:
    assert SlurmSettings.from_mapping({"env": None}).env == {}
    assert SlurmSettings.from_mapping({"env": {}}).env == {}
    assert SlurmSettings.from_mapping({}).env == {}
    env = SlurmSettings.from_mapping(
        {"env": {"A": "x y", "B": 5, "C": 1.5, "D": "", "E": "$HOME"}}
    ).env
    assert env == {"A": "x y", "B": "5", "C": "1.5", "D": "", "E": "$HOME"}


# ── The list settings ──────────────────────────────────────────────────────


@pytest.mark.parametrize("key", ["extra_sbatch_args", "env_passthrough", "setup"])
@pytest.mark.parametrize(
    ("value", "item"),
    [
        (["--nice=10", None], None),
        (["--nice=10", ""], ""),
        (["  "], "  "),
        ("", ""),
    ],
)
def test_a_null_or_blank_list_item_is_refused(key: str, value: Any, item: Any) -> None:
    # It became "", an empty sbatch argument (or line).
    with pytest.raises(ValueError) as caught:
        SlurmSettings.from_mapping({key: value})
    assert str(caught.value) == (
        f"launcher.{key} must be a list of non-blank text, got the item {item!r}"
    )


def test_a_list_item_of_another_type_is_still_refused_as_before() -> None:
    with pytest.raises(ValueError) as caught:
        SlurmSettings.from_mapping({"extra_sbatch_args": [True]})
    assert str(caught.value) == "launcher.extra_sbatch_args must be text, got True"


def test_list_settings_read_as_before() -> None:
    settings = SlurmSettings.from_mapping(
        {
            "extra_sbatch_args": ["--nice=10", 5, " --x "],
            "env_passthrough": "CARLA_*",
            "setup": [],
        }
    )
    assert settings.extra_sbatch_args == ("--nice=10", "5", " --x ")
    assert settings.env_passthrough == ("CARLA_*",)
    assert settings.setup == ()
    assert SlurmSettings.from_mapping({"setup": None}).setup == ()
    assert SlurmSettings.from_mapping({"env_passthrough": []}).env_passthrough == ()


# ── The port range ─────────────────────────────────────────────────────────


def _highest_port(base: int, slots: int) -> int:
    """The highest port any slot of the range uses, slot by slot."""
    return max(
        port
        for index, first in slot_order(base, slots, 0)
        for port in SlotPorts(index, first).ports()
    )


@pytest.mark.parametrize(
    ("base", "slots"), [(65530, 1), (20000, 4554), (65520, 2), (1030, 6451)]
)
def test_a_range_whose_last_port_is_65535_is_taken(base: int, slots: int) -> None:
    assert _highest_port(base, slots) == 65535
    settings = SlurmSettings.from_mapping({"port_base": base, "port_slots": slots})
    assert (settings.port_base, settings.port_slots) == (base, slots)


@pytest.mark.parametrize(
    ("base", "slots"), [(65531, 1), (20000, 4555), (65521, 2), (1030, 6452)]
)
def test_a_range_past_65535_is_refused(base: int, slots: int) -> None:
    highest = _highest_port(base, slots)
    assert highest > 65535
    with pytest.raises(ValueError) as caught:
        SlurmSettings.from_mapping({"port_base": base, "port_slots": slots})
    assert str(caught.value) == (
        "launcher.port_base/port_slots must keep every task's ports within "
        f"1024-65535 (got base {base}, {slots} slots: ports up to {highest})"
    )


def test_the_default_port_range_is_taken() -> None:
    assert _highest_port(20000, 1000) <= 65535
    assert SlurmSettings.from_mapping({"port_base": 20000, "port_slots": 1000}) == (
        SlurmSettings()
    )


def test_the_message_names_the_keys() -> None:
    with pytest.raises(ValueError, match=re.escape("launcher.port_base/port_slots")):
        SlurmSettings.from_mapping({"port_slots": 100000})
