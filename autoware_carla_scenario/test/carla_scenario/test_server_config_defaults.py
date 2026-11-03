"""``ServerConfig``'s defaults are the shipped server config's and the docs' (#1267).

``ServerConfig.cooldown_seconds`` defaulted to 3.0 while
``examples/conf/server/localhost.yaml`` set 2.0 and ``docs/usage.md``
documented 2.0. Each key the dataclass shares with the shipped YAML, and each
``server.*`` row of the docs' "Server Configuration" table, must name the
dataclass's default.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from autoware_carla_scenario import ServerConfig

_PACKAGE = Path(__file__).resolve().parents[2]
_LOCALHOST = (
    _PACKAGE
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
    / "server"
    / "localhost.yaml"
)
_USAGE = _PACKAGE / "docs" / "usage.md"

_DEFAULTS: dict[str, Any] = {
    f.name: f.default for f in dataclasses.fields(ServerConfig)
}


def _shipped_server() -> dict[str, Any]:
    document = yaml.safe_load(_LOCALHOST.read_text(encoding="utf-8"))
    server = document["server"]
    assert isinstance(server, dict)
    return server


def _documented_server() -> dict[str, Any]:
    """The ``server.*`` rows of the docs' table: key -> default, parsed."""
    rows = re.findall(
        r"^\| `server\.(\w+)` \| \w+ \| `([^`]*)` \|",
        _USAGE.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return {key: yaml.safe_load(value) for key, value in rows}


def test_every_default_is_a_plain_value() -> None:
    # A default_factory field would read as MISSING here and match nothing.
    assert all(
        f.default is not dataclasses.MISSING for f in dataclasses.fields(ServerConfig)
    )


def test_the_cooldown_default_is_the_shipped_one() -> None:
    assert ServerConfig().cooldown_seconds == _shipped_server()["cooldown_seconds"]


@pytest.mark.parametrize("key", sorted(_shipped_server()))
def test_the_shipped_server_config_sets_the_defaults(key: str) -> None:
    assert key in _DEFAULTS, f"server.{key} is not a ServerConfig field"
    assert _shipped_server()[key] == _DEFAULTS[key]


def test_the_docs_table_lists_the_cooldown() -> None:
    assert "cooldown_seconds" in _documented_server()


@pytest.mark.parametrize("key", sorted(_documented_server()))
def test_the_docs_table_documents_the_defaults(key: str) -> None:
    assert key in _DEFAULTS, f"docs/usage.md documents server.{key}, no field"
    assert _documented_server()[key] == _DEFAULTS[key]
