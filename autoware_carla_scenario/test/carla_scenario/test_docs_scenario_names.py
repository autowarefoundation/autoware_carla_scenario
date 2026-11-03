"""Every ``scenario=`` a doc or an example's docstring names composes (#1194).

The user-facing docs (``docs/**/*.md``) and the module docstrings of
``examples/*.py`` show run commands; a ``scenario=`` value there that no
shipped config matches is a command that fails as soon as it is copied.  Each
value is resolved as the runner resolves it (a glob into the configs it
matches) and each config is composed as a run composes it.

Values that name a scenario of the reader's own -- a package or a tutorial
scenario that is not shipped -- or that stand for any name are listed in
``_NOT_SHIPPED``, each with the reason.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from autoware_carla_scenario.examples import run
from autoware_carla_scenario.registry import get_conf_dirs
from autoware_carla_scenario.scenario_glob import resolve_scenario_glob

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _PACKAGE_ROOT / "docs"
_EXAMPLES = Path(run.__file__).resolve().parent

# A value starts like a config name; ``<name>``, ``...`` or ``{name}`` stand for
# one and are not taken. It ends at whitespace, a quote, a backtick, a closing
# bracket or a comma (Hydra's ``scenario=a,b`` list).
_SCENARIO = re.compile(r"(?<![\w.+])scenario=['\"]?([A-Za-z0-9_][^\s'\"`),]*)")

_NOT_SHIPPED: dict[str, str] = {
    "my_scenario/default": "the tutorial's own scenario (docs/tutorial.md, docs/docker.md)",
    "my_scenario/*": "the tutorial's own scenarios, as a batch",
    "reach_goal/default": "examples/scenario_package_template's scenario",
    "reach_goal/*": "examples/scenario_package_template's scenarios (docs/tutorial.md)",
    "my_package/*": "any package author's scenarios (docs/typecheck.md)",
    "cut_in": "the package docs/scenario_editor.md exports (conf/scenario/cut_in.yaml)",
    "X": "any scenario, in a diagram (docs/architecture.md)",
    "a": "any scenarios, in a Hydra list (docs/metrics.md: scenario=a,b)",
}


def _sources() -> list[tuple[str, str]]:
    """(where, text): every doc page and every example module's docstring."""
    sources = [
        (str(path.relative_to(_PACKAGE_ROOT)), path.read_text(encoding="utf-8"))
        for path in sorted(_DOCS.rglob("*.md"))
    ]
    for path in sorted(_EXAMPLES.glob("*.py")):
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        if docstring:
            sources.append((str(path.relative_to(_PACKAGE_ROOT)), docstring))
    return sources


def _documented() -> dict[str, list[str]]:
    """Each documented ``scenario=`` value, with the places that name it."""
    found: dict[str, list[str]] = {}
    for where, text in _sources():
        for number, line in enumerate(text.splitlines(), start=1):
            for match in _SCENARIO.finditer(line):
                value = match.group(1).rstrip(".:;")
                found.setdefault(value, []).append(f"{where}:{number}")
    return found


_DOCUMENTED = _documented()


def test_the_docs_name_scenarios() -> None:
    """The extraction finds the commands it is meant to check."""
    assert "cut_in/left" in _DOCUMENTED
    assert "intersection_passing/straight" in _DOCUMENTED


def test_every_placeholder_is_still_documented() -> None:
    """An entry of the allowlist that no doc names any more is removed."""
    assert sorted(set(_NOT_SHIPPED) - set(_DOCUMENTED)) == []


@pytest.mark.parametrize(
    "value", sorted(v for v in _DOCUMENTED if v not in _NOT_SHIPPED)
)
def test_a_documented_scenario_composes(value: str) -> None:
    where = ", ".join(_DOCUMENTED[value])
    names = resolve_scenario_glob(value, get_conf_dirs())
    assert names, f"scenario={value} ({where}) matches no shipped config"
    for name in names:
        try:
            run._compose_config(name, [])
        except Exception as exc:  # noqa: BLE001 - reported with where it is named
            pytest.fail(f"scenario={value} ({where}): {name} does not compose: {exc}")
