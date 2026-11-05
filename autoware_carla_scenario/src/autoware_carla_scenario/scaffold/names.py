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


def _default_scenario_name(package_name: str) -> str:
    """Return the scenario name ``scenario-new`` derives from *package_name*.

    A trailing package-ish suffix is stripped, so ``my_scenario_package``
    yields the scenario ``my_scenario``; a name that is nothing but the
    suffix is kept whole.
    """
    stripped = re.sub(r"_(package|pkg|scenarios?)$", "", package_name)
    return stripped or package_name
