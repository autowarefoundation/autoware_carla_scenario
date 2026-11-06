"""The names an exported Scenario Package takes from its document's id.

How :mod:`authoring.package_export` turns a document id into the import,
distribution and directory names of the package it writes, which ids it
refuses, and the lines of the package's README and ``pyproject.toml`` that
describe the framework pin.  Kept apart from the exporter, which imports
pydantic, yaml and jinja2, so the static check compiles these rules (#1695);
the exporter re-exports them.

The id itself is restricted by the document model to ``^[a-z][a-z0-9_]*$``
(:data:`scaffold.names.SCENARIO_NAME_PATTERN`, the rule ``scenario-new`` holds
a scenario name to, #1858).
On top of that:

an id whose package would be the framework's own (``autoware_carla_scenario``:
the id ``autoware_carla``, or ``autoware__carla``) is refused, by the
rule ``scenario-new`` applies (:func:`scaffold.names._is_framework_name`).
So is an id that names a built-in scenario
(:data:`scaffold.names.BUILTIN_SCENARIO_NAMES`: ``cut_in``, ``lane_change``,
...), as ``scenario-new`` refuses it (#1808): the id is the name the package
registers its builder under, so once installed it would replace the built-in
builder while the built-in configs still select it.

The package is always ``<id>_scenario``: ``_scenario`` is added even to an
id that already ends in it, so ``my_cut_in`` and ``my_cut_in_scenario`` export
as ``my_cut_in_scenario`` and ``my_cut_in_scenario_scenario`` rather than as
one package (#1358).  Ids with repeated or trailing underscores
(``my_cut__in``, ``my_cut_in_``) would still share the package of the id
without them, as PEP 503 folds every run of ``_`` when it compares
distribution names; :func:`scenario_id_problem` flags them, the validator
warns about them (they still run) and the export refuses them.
"""

from __future__ import annotations

import re
from typing import Optional

from ..scaffold.names import (
    _FRAMEWORK_DISTRIBUTION,
    BUILTIN_SCENARIO_NAMES,
    _is_framework_name,
)
from .framework_pin import Pin

__all__ = [
    "PACKAGE_SUFFIX",
    "framework_name_problem",
    "names_for_id",
    "scenario_id_problem",
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
    return f"{_sanitize_identifier(scenario_id)}{PACKAGE_SUFFIX}"


def framework_name_problem(scenario_id: str) -> Optional[str]:
    """Return why *scenario_id* cannot be exported, or ``None`` if it can.

    The id is refused when its package would take the framework's own name: a
    package that depends on itself, shadows the framework once installed, and
    only fails much later, in ``uv lock``.  It is refused too when it names a
    built-in scenario (:data:`scaffold.names.BUILTIN_SCENARIO_NAMES`, #1808):
    the package registers its builder under the id, replacing the built-in one
    once installed, while the built-in configs (``scenario=<id>/...``) still
    select it and hand it the built-in's config keys.  Worded as
    ``scenario-new`` words the same refusal.
    """
    if scenario_id in BUILTIN_SCENARIO_NAMES:
        return (
            f"The id {scenario_id!r} is reserved: it names a built-in scenario, "
            f"which the installed package would replace while the built-in "
            f"configs (scenario={scenario_id}/...) still select it (built-in "
            f"scenarios: {', '.join(BUILTIN_SCENARIO_NAMES)}). Choose a "
            f"different id."
        )
    package_name = _package_name(scenario_id)
    if not _is_framework_name(package_name):
        return None
    return (
        f"The id {scenario_id!r} exports the package {package_name!r}, which is "
        f"the framework's own ({_FRAMEWORK_DISTRIBUTION!r}): it would depend on "
        f"itself and replace the installed framework. Choose a different id."
    )


def scenario_id_problem(scenario_id: str) -> Optional[str]:
    """Return why *scenario_id* would share another id's package, or ``None``.

    The exported package is ``<id>_scenario`` (distribution ``<id>-scenario``),
    and installers compare distribution names with every run of ``_`` folded
    into one ``-`` (PEP 503).  So ``__`` and a trailing ``_``, which the
    model's pattern accepts, would give an id the package of the id without
    them, and installing one would replace the other (#1358).  They are refused
    by :func:`~autoware_carla_scenario.authoring.package_export.export_package`
    alone: :func:`~autoware_carla_scenario.authoring.validator.validate_document`
    only warns, because it also gates every run, and a package exported with
    such an id before #1358 has to keep running.  The model's pattern still
    accepts them, so a draft that uses one opens and can be renamed.  An id
    whose package would be the framework's own is :func:`framework_name_problem`'s.
    """
    stem = _sanitize_identifier(scenario_id)
    package_name = _package_name(scenario_id)
    if "__" in scenario_id:
        return (
            f"Scenario id {scenario_id!r} contains `__`, so it would be "
            f"exported as `{package_name}`, the same package as the id "
            f"{stem!r}. Use a single `_` between words."
        )
    if scenario_id.endswith("_"):
        return (
            f"Scenario id {scenario_id!r} ends in `_`, so it would be exported "
            f"as `{package_name}`, the same package as the id {stem!r}. Drop "
            "the trailing `_`."
        )
    return None


def names_for_id(scenario_id: str) -> dict[str, str]:
    """Return the naming variants a package exported from *scenario_id* needs.

    ``scenario_id`` is the id unchanged: it is the registered builder name,
    the conf file name and the conf's ``scenario.name`` alike, and they must
    agree for ``scenario=<id>`` to select the scenario.  It is only ever used
    as a string, never as a Python identifier, so a keyword such as ``class``
    is harmless.  Only the Python package and distribution names are
    sanitised; the package name always ends in ``_scenario``, so it is never a
    keyword.  The mapping is one-to-one over the ids :func:`scenario_id_problem`
    accepts; this does not refuse the others, which
    :func:`~autoware_carla_scenario.authoring.package_export.export_package`
    does before writing anything.

    Raises:
        ValueError: If the package would take the framework's own name, or
            the id names a built-in scenario (:func:`framework_name_problem`).
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
