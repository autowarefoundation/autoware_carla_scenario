"""The docs' inventories list what the code has (#2304).

A docs audit found lists that had fallen behind the code: ``docs/usage.md``'s
troubleshooting entry named four of the seven built-in scenarios, its traffic
backend rows left out ``hybrid``, its Ego Vehicle Configuration table left out
four ``EgoVehicleConfig`` fields, and ``docs/quickstart.md`` said the package
installs ``carla-driver-interface`` 1.x while ``pyproject.toml`` requires 2.x.
Each of these is checked against its source here.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import pytest

from autoware_carla_scenario import EgoVehicleConfig
from autoware_carla_scenario.scaffold.names import BUILTIN_SCENARIO_NAMES

# tomllib landed in 3.11, and 3.10 is still the floor of the supported range.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _PACKAGE_ROOT / "docs"
_USAGE = _DOCS / "usage.md"
_TRAFFIC_CONF = (
    _PACKAGE_ROOT / "src" / "autoware_carla_scenario" / "examples" / "conf" / "traffic"
)


def _usage() -> str:
    return _USAGE.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body of the Markdown section *heading* (up to the next heading).

    Fenced code blocks are dropped first, so a ``# comment`` in one is not
    taken for a heading.
    """
    text = re.sub(r"^```.*?^```[^\n]*$", "", text, flags=re.MULTILINE | re.DOTALL)
    match = re.search(
        rf"^#+ {re.escape(heading)}\n(.*?)(?=^#+ |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"docs/usage.md has no section {heading!r}"
    return match.group(1)


def test_troubleshooting_lists_every_builtin_scenario() -> None:
    body = _section(_usage(), '"Unknown scenario name" error')
    line = next(
        (ln for ln in body.splitlines() if "Supported scenario names" in ln), None
    )
    assert line, "the troubleshooting entry no longer lists the scenario names"
    assert sorted(re.findall(r"`(\w+)`", line)) == sorted(BUILTIN_SCENARIO_NAMES)


def _traffic_backends() -> list[str]:
    stems = sorted(path.stem for path in _TRAFFIC_CONF.glob("*.yaml"))
    assert "traffic_manager" in stems
    return stems


def test_the_traffic_default_row_lists_every_backend() -> None:
    row = re.search(r"^\| `traffic` \| .*$", _usage(), re.MULTILINE)
    assert row, "the Root Configuration table has no `traffic` row"
    description = row.group(0).split("|")[3]
    for stem in _traffic_backends():
        assert f"`{stem}`" in description, f"the `traffic` row misses {stem}"


def test_the_config_tree_lists_every_backend() -> None:
    line = re.search(r"^├── traffic/ +# Traffic backend: (.*)$", _usage(), re.MULTILINE)
    assert line, "the config tree has no traffic/ line"
    listed = [name.strip() for name in line.group(1).split(",")]
    assert sorted(listed) == _traffic_backends()


def test_the_ego_table_has_a_row_per_field() -> None:
    body = _section(_usage(), "Ego Vehicle Configuration")
    documented = re.findall(r"^\| `ego\.(\w+)` \|", body, re.MULTILINE)
    fields = [f.name for f in dataclasses.fields(EgoVehicleConfig)]
    assert sorted(documented) == sorted(fields)


def _driver_interface_requirement() -> str:
    pyproject = tomllib.loads(
        (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    for requirement in pyproject["project"]["dependencies"]:
        if requirement.startswith("carla-driver-interface"):
            return str(requirement)
    raise AssertionError("pyproject.toml no longer requires carla-driver-interface")


def _driver_interface_major() -> int:
    upper = re.search(r",<(\d+)$", _driver_interface_requirement())
    assert upper, "carla-driver-interface's requirement has no major upper bound"
    return int(upper.group(1)) - 1


_QUOTED_REQUIREMENT = re.compile(r"carla-driver-interface>=[^`\s]+")


def _docs_quoting_the_driver_interface_requirement() -> list[str]:
    return sorted(
        p.name
        for p in _DOCS.glob("*.md")
        if _QUOTED_REQUIREMENT.search(p.read_text(encoding="utf-8"))
    )


def test_some_docs_quote_the_driver_interface_requirement() -> None:
    """Without this, the test below would quietly have no cases."""
    assert _docs_quoting_the_driver_interface_requirement()


@pytest.mark.parametrize("doc", _docs_quoting_the_driver_interface_requirement())
def test_the_docs_quote_the_driver_interface_requirement(doc: str) -> None:
    text = (_DOCS / doc).read_text(encoding="utf-8")
    for quoted in _QUOTED_REQUIREMENT.findall(text):
        assert quoted == _driver_interface_requirement(), f"docs/{doc}"


@pytest.mark.parametrize("doc", ["quickstart.md", "driver_interface.md"])
def test_the_docs_name_the_driver_interface_major(doc: str) -> None:
    text = (_DOCS / doc).read_text(encoding="utf-8")
    # "[`carla-driver-interface`](...carla_driver_interface)\n2.x" or "(2.x)".
    named = re.search(r"/carla_driver_interface\)\s*\(?(\d+)\.x", text)
    assert named, f"docs/{doc} no longer names the version it installs"
    assert int(named.group(1)) == _driver_interface_major()
