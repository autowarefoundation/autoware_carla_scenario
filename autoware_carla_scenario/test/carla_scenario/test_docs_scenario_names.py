"""Every ``scenario=`` a doc or an example's docstring names composes (#1194).

The user-facing docs (``docs/**/*.md``), the READMEs (the repository's, the
package's, ``stl_driver``'s and the scenario package template's) and the
module docstrings of ``examples/*.py`` show run commands; a ``scenario=`` value
there that no shipped config matches is a command that fails as soon as it is
copied.  Each value (each item of a Hydra ``scenario=a,b`` list) is resolved as
the runner resolves it (a glob into the configs it matches) and each config is
composed as a run composes it.

Values that name a scenario of the reader's own -- a package or a tutorial
scenario that is not shipped -- or that stand for any name are listed in
``_NOT_SHIPPED``, each with the reason.

The examples' docstrings also point at the tests that exercise them; each
``test/...py`` they name exists (#1315).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from autoware_carla_scenario.examples import run
from autoware_carla_scenario.scenario_glob import resolve_scenario_glob

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _PACKAGE_ROOT / "docs"
_REPOSITORY_ROOT = _PACKAGE_ROOT.parent
_READMES = (
    _REPOSITORY_ROOT / "README.md",
    _PACKAGE_ROOT / "README.md",
    _REPOSITORY_ROOT / "stl_driver" / "README.md",
    _REPOSITORY_ROOT / "examples" / "scenario_package_template" / "README.md",
)
_EXAMPLES = _PACKAGE_ROOT / "src" / "autoware_carla_scenario" / "examples"

# A value starts like a config name or a glob (``'*/*'``); ``<name>``, ``...``
# or ``{name}`` stand for one and are not taken. It ends at whitespace, a
# quote, a backtick or a closing bracket; a comma separates the items of
# Hydra's ``scenario=a,b`` list, each checked on its own.
_SCENARIO = re.compile(r"(?<![\w.+])scenario=['\"]?([A-Za-z0-9_*?][^\s'\"`)]*)")

_NOT_SHIPPED: dict[str, str] = {
    "my_scenario/default": "the tutorial's own scenario (docs/tutorial.md, docs/docker.md)",
    "my_scenario/*": "the tutorial's own scenarios, as a batch",
    "reach_goal/default": "examples/scenario_package_template's scenario",
    "reach_goal/*": "examples/scenario_package_template's scenarios",
    "my_package/*": "any package author's scenarios (docs/typecheck.md)",
    "cut_in": "the package docs/scenario_editor.md exports (conf/scenario/cut_in.yaml)",
    "*": "scenario-check's every config (typecheck/cli.py), not a runner glob",
    "X": "any scenario, in a diagram (docs/architecture.md)",
    "a": "any scenario, in a Hydra list (docs/metrics.md: scenario=a,b)",
    "b": "any scenario, in a Hydra list (docs/metrics.md: scenario=a,b)",
}


def _sources() -> list[tuple[str, str]]:
    """(where, text): every doc page, README and example module's docstring."""
    sources = [
        (str(path.relative_to(_REPOSITORY_ROOT)), path.read_text(encoding="utf-8"))
        for path in (*sorted(_DOCS.rglob("*.md")), *_READMES)
    ]
    for path in sorted(_EXAMPLES.glob("*.py")):
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        if docstring:
            sources.append((str(path.relative_to(_REPOSITORY_ROOT)), docstring))
    return sources


def _documented() -> dict[str, list[str]]:
    """Each documented ``scenario=`` value, with the places that name it."""
    found: dict[str, list[str]] = {}
    for where, text in _sources():
        for number, line in enumerate(text.splitlines(), start=1):
            for match in _SCENARIO.finditer(line):
                for item in match.group(1).split(","):
                    value = item.rstrip(".:;")
                    if value:
                        found.setdefault(value, []).append(f"{where}:{number}")
    return found


_DOCUMENTED = _documented()


def test_the_docs_name_scenarios() -> None:
    """The extraction finds the commands it is meant to check."""
    assert "cut_in/left" in _DOCUMENTED
    assert "intersection_passing/straight" in _DOCUMENTED
    assert "*/*" in _DOCUMENTED  # a glob that starts with one
    assert "b" in _DOCUMENTED  # the second item of a Hydra list
    assert any(where.startswith("stl_driver/") for where in _DOCUMENTED["cut_in/left"])


def test_every_placeholder_is_still_documented() -> None:
    """An entry of the allowlist that no doc names any more is removed."""
    assert sorted(set(_NOT_SHIPPED) - set(_DOCUMENTED)) == []


@pytest.mark.parametrize(
    "value", sorted(v for v in _DOCUMENTED if v not in _NOT_SHIPPED)
)
def test_a_documented_scenario_composes(value: str) -> None:
    where = ", ".join(_DOCUMENTED[value])
    # The built-in config directory only, as ``_compose_config`` composes from
    # it: not the registry's list, which other tests and plugins extend.
    names = resolve_scenario_glob(value, [run._CONF_DIR])
    assert names, f"scenario={value} ({where}) matches no shipped config"
    for name in names:
        try:
            run._compose_config(name, [])
        except Exception as exc:  # noqa: BLE001 - reported with where it is named
            pytest.fail(f"scenario={value} ({where}): {name} does not compose: {exc}")


_TEST_PATH = re.compile(r"``(test/[\w/]+\.py)``")


def test_every_test_an_example_names_exists() -> None:
    """A ``test/...py`` an example's docstring points at is in the package."""
    named = [
        (where, match.group(1))
        for where, text in _sources()
        if where.startswith("autoware_carla_scenario/src/")
        for match in _TEST_PATH.finditer(text)
    ]
    assert named  # the extraction finds the pointers it is meant to check
    missing = [
        f"{where}: {path}"
        for where, path in named
        if not (_PACKAGE_ROOT / path).is_file()
    ]
    assert missing == []
