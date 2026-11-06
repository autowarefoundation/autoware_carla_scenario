"""The naming rules of ``scenario-new``: package, scenario and class names.

How :mod:`scaffold.generator` turns the raw name it is given into the import,
distribution, scenario and class names of the package it writes, and what it
refuses as an identifier.  Kept apart from the generator, which imports
argparse, ast, importlib.metadata and jinja2, so the static check compiles
these rules (#1654); the generator re-exports them.  The rule that a package
must not take the framework's own name is shared with the Scenario Editor's
export (:mod:`authoring.package_names`, #1695), so the two paths agree.
"""

from __future__ import annotations

import keyword
import re


def _split_camel_case(raw: str) -> str:
    """Return *raw* with ``_`` between an ASCII lower-case letter or digit and
    the ASCII upper-case letter that follows it (``fooBar9Baz`` ->
    ``foo_Bar9_Baz``).

    What ``re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw)`` did, which Codon's re
    (RE2) has no lookbehind or lookahead for.  The ranges are ASCII-only, as
    the pattern's were: ``EteScenario`` gets a ``_`` before ``S``, but
    ``ÉtéScenario`` none, ``é`` being no ASCII lower-case letter.
    """
    parts: list[str] = []
    previous = ""
    for char in raw:
        if ("a" <= previous <= "z" or "0" <= previous <= "9") and "A" <= char <= "Z":
            parts.append("_")
        parts.append(char)
        previous = char
    return "".join(parts)


def _to_snake(raw: str) -> str:
    """Normalise *raw* to a ``snake_case`` Python identifier fragment."""
    # Split camelCase / PascalCase boundaries, then unify separators.
    spaced = _split_camel_case(raw)
    snake = re.sub(r"[\s\-.]+", "_", spaced).strip("_").lower()
    return re.sub(r"__+", "_", snake)


def _to_pascal(snake: str) -> str:
    """Convert a ``snake_case`` name to ``PascalCase``."""
    return "".join(part.capitalize() for part in snake.split("_") if part)


def _to_hyphen(snake: str) -> str:
    """Convert a ``snake_case`` name to a ``hyphen-case`` distribution name."""
    return snake.replace("_", "-")


def _normalise_distribution(name: str) -> str:
    """Return *name* normalised as PEP 503 does (``A_b.c`` -> ``a-b-c``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


#: The scenarios the framework registers itself (``examples.run`` does, at
#: import), which a generated package may not take (#1802).  A plugin that
#: registers one of these names replaces the built-in builder, while Hydra
#: still finds the built-in's configs first: ``scenario=lane_change/left``
#: then hands ``LaneChangeConfig``'s keys to the plugin's scenario.  A plain
#: constant, so ``scenario-new`` need not import the runner (and CARLA) to
#: read it; ``test_scaffold.py`` checks it against the registry the runner
#: fills, so it cannot go stale.
BUILTIN_SCENARIO_NAMES = (
    "adversity",
    "cut_in",
    "intersection_passing",
    "lane_change",
    "pedestrian_dart_out",
    "temporary_stop",
    "traffic_light_compliance",
)


#: The framework's distribution name, which a generated package depends on.
_FRAMEWORK_DISTRIBUTION = "autoware-carla-scenario"


def _is_framework_name(name: str) -> bool:
    """Return whether *name* is the framework's own package or distribution name.

    Compared as PEP 503 normalises names, so ``autoware_carla_scenario``,
    ``autoware-carla-scenario`` and ``Autoware.Carla_Scenario`` all count.  A
    package of that name depends on itself and replaces the installed
    framework (#1451, #1695).
    """
    return _normalise_distribution(name) == _normalise_distribution(
        _FRAMEWORK_DISTRIBUTION
    )


def _validate_identifier(name: str, *, kind: str) -> str:
    """Return *name* if it is a valid, non-keyword Python identifier, else raise."""
    if not name:
        msg = f"{kind} is empty after normalisation; please pass an explicit value."
        raise ValueError(msg)
    if not name.isidentifier() or keyword.iskeyword(name):
        msg = f"{kind} {name!r} is not a valid Python identifier."
        raise ValueError(msg)
    return name


#: The scenario names ``scenario-new`` accepts, and the ids the Scenario
#: Editor gives a scenario and its objects (:mod:`authoring.models`, #1858):
#: a lower-case ASCII letter, then lower-case ASCII letters, digits and
#: ``_``.  Hydra's override grammar allows only ASCII letters in an id, so
#: ``scenario=café/default`` fails to parse; a Python identifier
#: (``str.isidentifier``) may be any Unicode word.  :func:`_is_scenario_name`
#: is the same rule without :mod:`re`.
SCENARIO_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"


def _is_scenario_name(name: str) -> bool:
    """Return whether *name* matches :data:`SCENARIO_NAME_PATTERN`.

    A character loop rather than :mod:`re`, so a trailing newline, which
    ``$`` would let through, is refused too.
    """
    if not name or not "a" <= name[0] <= "z":
        return False
    for char in name:
        if not ("a" <= char <= "z" or "0" <= char <= "9" or char == "_"):
            return False
    return True


def _validate_scenario_name(name: str) -> str:
    """Return *name* if Hydra's override grammar can select it, else raise.

    *name* is a scenario name after :func:`_to_snake` (and
    :func:`_validate_identifier`); see :data:`SCENARIO_NAME_PATTERN`.
    """
    if not _is_scenario_name(name):
        msg = (
            f"scenario name {name!r} must match {SCENARIO_NAME_PATTERN} (a "
            f"lower-case ASCII letter, then lower-case ASCII letters, digits "
            f"and '_'): Hydra's override grammar could not select it as "
            f"scenario={name}/default. Pass a different --scenario."
        )
        raise ValueError(msg)
    return name


def _default_scenario_name(package_name: str) -> str:
    """Return the scenario name ``scenario-new`` derives from *package_name*.

    A trailing package-ish suffix is stripped, so ``my_scenario_package``
    yields the scenario ``my_scenario``; a name that is nothing but the
    suffix is kept whole.
    """
    stripped = re.sub(r"_(package|pkg|scenarios?)$", "", package_name)
    return stripped or package_name
