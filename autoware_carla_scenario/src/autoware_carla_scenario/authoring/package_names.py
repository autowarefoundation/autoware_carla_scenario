"""The names an exported Scenario Package takes from its document's id.

How :mod:`authoring.package_export` turns a document id into the import,
distribution and directory names of the package it writes, which ids it
refuses, and the lines of the package's README and ``pyproject.toml`` that
describe the framework pin.  Kept apart from the exporter, which imports
pydantic, yaml and jinja2, so the static check compiles these rules (#1695);
the exporter re-exports them.

The id itself is restricted by the document model to ``^[a-z][a-z0-9_]*$``.
On top of that:

an id whose package would be the framework's own (``autoware_carla_scenario``:
the ids ``autoware_carla`` and ``autoware_carla_scenario``) is refused, by the
rule ``scenario-new`` applies (:func:`scaffold.names._is_framework_name`).

Ids that differ only by a ``_scenario`` suffix or by repeated or trailing
underscores (``cut_in``, ``cut_in_scenario``, ``cut__in``) share one package
name, and so one directory.  That is documented, not warned about: the
validator cannot tell whether the other id's package exists.
"""

from __future__ import annotations

import re
from typing import Optional

from ..scaffold.names import _FRAMEWORK_DISTRIBUTION, _is_framework_name
from .framework_pin import Pin

__all__ = [
    "PACKAGE_SUFFIX",
    "framework_name_problem",
    "names_for_id",
]

#: What every exported package name ends in.
PACKAGE_SUFFIX = "_scenario"


def _sanitize_identifier(raw: str) -> str:
    """Return *raw* reduced to a valid lower_snake_case Python identifier."""
    snake = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_").lower()
    snake = re.sub(r"__+", "_", snake)
    if not snake or snake[0].isdigit():
        snake = f"scenario_{snake}" if snake else "scenario"
    return snake


def _package_name(scenario_id: str) -> str:
    """Return the Python package name the id *scenario_id* exports as."""
    module_id = _sanitize_identifier(scenario_id)
    if module_id.endswith(PACKAGE_SUFFIX):
        return module_id
    return f"{module_id}{PACKAGE_SUFFIX}"


def framework_name_problem(scenario_id: str) -> Optional[str]:
    """Return why *scenario_id* cannot be exported, or ``None`` if it can.

    The id is refused when its package would take the framework's own name: a
    package that depends on itself, shadows the framework once installed, and
    only fails much later, in ``uv lock``.
    """
    package_name = _package_name(scenario_id)
    if not _is_framework_name(package_name):
        return None
    return (
        f"The id {scenario_id!r} exports the package {package_name!r}, which is "
        f"the framework's own ({_FRAMEWORK_DISTRIBUTION!r}): it would depend on "
        f"itself and replace the installed framework. Choose a different id."
    )


def names_for_id(scenario_id: str) -> dict[str, str]:
    """Return the naming variants a package exported from *scenario_id* needs.

    ``scenario_id`` is the id unchanged: it is the registered builder name,
    the conf file name and the conf's ``scenario.name`` alike, and they must
    agree for ``scenario=<id>`` to select the scenario.  It is only ever used
    as a string, never as a Python identifier, so a keyword such as ``class``
    is harmless.  Only the Python package and distribution names are
    sanitised; the package name always ends in ``_scenario``, so it is never a
    keyword.

    Raises:
        ValueError: If the package would take the framework's own name
            (:func:`framework_name_problem`).
    """
    problem = framework_name_problem(scenario_id)
    if problem is not None:
        raise ValueError(problem)
    package_name = _package_name(scenario_id)
    return {
        "scenario_id": scenario_id,
        "package_name": package_name,
        "distribution_name": package_name.replace("_", "-"),
        "document_path_env": f"{package_name.upper()}_DOCUMENT_PATH",
    }


def _pin_note(pin: Pin) -> str:
    """Return the comment written above the framework dependency."""
    if pin.kind == "version":
        return (
            "Pinned to an exact release: a scenario package is only "
            "reproducible if the framework underneath it cannot move."
        )
    if pin.kind == "git":
        return (
            "Pinned to an exact commit (see [tool.uv.sources]) rather than a "
            "branch, which would move under the package."
        )
    return (
        "Development export: a local path, which does NOT resolve on another "
        "machine. Re-export without development mode before sharing."
    )


def _pin_summary(pin: Pin) -> str:
    """Return a one-line, human-readable description of *pin* for the README."""
    if pin.kind == "version":
        return f"version `{pin.version}` (exact)"
    if pin.kind == "git":
        return f"commit `{pin.commit}` of `{pin.repository}`"
    return f"local path `{pin.path}` -- **not portable**"
