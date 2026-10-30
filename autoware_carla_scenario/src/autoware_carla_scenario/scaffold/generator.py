"""Generate a standalone scenario package from Jinja2 templates.

Exposes both a programmatic function (:func:`create_scenario_package`) and a
CLI entry point (:func:`main`, wired to the ``scenario-new`` console script).

The generated package is a normal installable Python package that depends on
``autoware-carla-scenario`` and plugs into the framework via the
``autoware_carla_scenario.scenarios`` entry point -- see
:mod:`autoware_carla_scenario.registry` for how discovery works.

This module has no CARLA/lanelet2 dependency, so it can be run in any
environment (the *generated* scenario needs CARLA at run time, not at
generation time).
"""

from __future__ import annotations

import argparse
import ast
import functools
import keyword
import re
import sys
from pathlib import Path
from typing import NamedTuple

from ..templating import code_environment


class ScaffoldResult(NamedTuple):
    """Result of :func:`create_scenario_package`."""

    #: The created package root (``output_dir / package_name``).
    root: Path
    #: The resolved template variables (see :func:`resolve_names`).
    names: dict[str, str]


#: Directory holding the ``*.jinja`` templates shipped with this package.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

#: Mapping of ``template name -> output path template``.  Output paths are
#: formatted with the resolved names (``package_name`` / ``scenario_name``).
_FILE_MANIFEST: dict[str, str] = {
    "pyproject.toml.jinja": "pyproject.toml",
    "README.md.jinja": "README.md",
    "package__init__.py.jinja": "src/{package_name}/__init__.py",
    "scenario.py.jinja": "src/{package_name}/{scenario_name}.py",
    "configs.py.jinja": "src/{package_name}/configs.py",
    "test_typecheck.py.jinja": "tests/test_typecheck.py",
    "default.yaml.jinja": (
        "src/{package_name}/conf/scenario/{scenario_name}/default.yaml"
    ),
}


#: The template of the scenario module in :data:`_FILE_MANIFEST`.
_SCENARIO_TEMPLATE = "scenario.py.jinja"

#: The template of the import package's ``__init__.py``.
_PACKAGE_INIT_TEMPLATE = "package__init__.py.jinja"

#: The prefix of every manifest entry inside the generated import package.
_PACKAGE_DIR = "src/{package_name}/"

#: Stand-in template variables, to render a template for its shape alone.
_PLACEHOLDER_NAMES: dict[str, str] = {
    "package_name": "placeholder_pkg",
    "distribution_name": "placeholder-pkg",
    "scenario_name": "placeholder",
    "scenario_class": "PlaceholderScenario",
    "config_class": "PlaceholderConfig",
    "requires_python": ">=3.10",
    "description": "Placeholder",
}


def _package_init_globals() -> set[str]:
    """Return the names the generated ``__init__.py`` binds at module level.

    Importing the scenario module (``register()`` does) binds it as an
    attribute of the package, that is, as a global of ``__init__.py``: a
    scenario called ``register_scenario`` replaced the function ``register()``
    calls next with a module.  ``__future__`` imports are compiler directives
    whose binding nothing reads, so they are left out.
    """
    source = (
        code_environment(TEMPLATES_DIR)
        .get_template(_PACKAGE_INIT_TEMPLATE)
        .render(**_PLACEHOLDER_NAMES)
    )
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                (alias.asname or alias.name).split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(
                    sub.id for sub in ast.walk(target) if isinstance(sub, ast.Name)
                )
    return names


@functools.cache
def _reserved_scenario_names() -> frozenset[str]:
    """Return the names the scenario module may not take.

    Every other entry :data:`_FILE_MANIFEST` writes inside the import package
    claims its top-level name there: ``configs.py`` claims ``configs``,
    ``__init__.py`` claims ``__init__`` and ``conf/...`` claims ``conf``.  A
    scenario of that name would make two templates render to the same path
    (one silently overwriting the other) or put a module beside a directory
    of the same name.  The globals of the generated ``__init__.py``
    (``register``, ``register_scenario``, ...) are claimed too, see
    :func:`_package_init_globals`.  Derived from the templates so it follows
    them.
    """
    reserved: set[str] = set()
    for template_name, out_template in _FILE_MANIFEST.items():
        if template_name == _SCENARIO_TEMPLATE or not out_template.startswith(
            _PACKAGE_DIR
        ):
            continue
        top = out_template[len(_PACKAGE_DIR) :].split("/", 1)[0]
        if "{" in top:
            # Named after the scenario itself, so it cannot collide.
            continue
        reserved.add(top.removesuffix(".py"))
    reserved |= _package_init_globals()
    # A scenario name is lower case once normalised (``CONF_DIR`` and the
    # config class cannot clash), so only lower-case names are listed.
    return frozenset(name for name in reserved if name == name.lower())


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------


def _to_snake(raw: str) -> str:
    """Normalise *raw* to a ``snake_case`` Python identifier fragment."""
    # Split camelCase / PascalCase boundaries, then unify separators.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw)
    snake = re.sub(r"[\s\-.]+", "_", spaced).strip("_").lower()
    return re.sub(r"__+", "_", snake)


def _to_pascal(snake: str) -> str:
    """Convert a ``snake_case`` name to ``PascalCase``."""
    return "".join(part.capitalize() for part in snake.split("_") if part)


def _to_hyphen(snake: str) -> str:
    """Convert a ``snake_case`` name to a ``hyphen-case`` distribution name."""
    return snake.replace("_", "-")


def _validate_identifier(name: str, *, kind: str) -> str:
    """Return *name* if it is a valid, non-keyword Python identifier, else raise."""
    if not name:
        msg = f"{kind} is empty after normalisation; please pass an explicit value."
        raise ValueError(msg)
    if not name.isidentifier() or keyword.iskeyword(name):
        msg = f"{kind} {name!r} is not a valid Python identifier."
        raise ValueError(msg)
    return name


def resolve_names(
    name: str,
    scenario_name: str | None = None,
    description: str | None = None,
) -> dict[str, str]:
    """Resolve all template variables from the raw package *name*.

    Returns a dict with ``package_name`` (import/snake), ``distribution_name``
    (hyphen), ``scenario_name`` (snake), ``scenario_class`` /  ``config_class``
    (Pascal), ``description``, and ``requires_python``.
    """
    from ..authoring.package_export import _requires_python  # noqa: PLC0415

    package_name = _validate_identifier(_to_snake(name), kind="package name")
    if not package_name.isascii():
        # The distribution name (PEP 508) and the entry-point key in
        # ``pyproject.toml`` (a bare TOML key) are ASCII-only, although a
        # Python identifier need not be.
        msg = f"package name {package_name!r} must be ASCII."
        raise ValueError(msg)

    if scenario_name is None:
        # Default: strip a trailing package-ish suffix so
        # ``my_scenario_package`` yields the scenario ``my_scenario``.
        stripped = re.sub(r"_(package|pkg|scenarios?)$", "", package_name)
        scenario = stripped or package_name
    else:
        scenario = _to_snake(scenario_name)
    scenario = _validate_identifier(scenario, kind="scenario name")
    reserved = _reserved_scenario_names()
    if scenario in reserved:
        msg = (
            f"scenario name {scenario!r} is reserved: the generated package "
            f"already has a {scenario!r} module, directory or global (reserved "
            f"names: "
            f"{', '.join(sorted(reserved))}). Pass a different --scenario."
        )
        raise ValueError(msg)

    pascal = _to_pascal(scenario)
    # Avoid awkward doubling when the name already ends in "scenario".
    scenario_class = pascal if pascal.endswith("Scenario") else f"{pascal}Scenario"
    return {
        "package_name": package_name,
        "distribution_name": _to_hyphen(package_name),
        "scenario_name": scenario,
        "scenario_class": scenario_class,
        "config_class": f"{pascal}Config",
        # Read off the installed framework rather than written out here: a
        # literal drifted the last time the supported range moved, and a
        # scaffolded package that refuses to install on the interpreter that
        # scaffolded it is a bad first five minutes.
        "requires_python": _requires_python(),
        "description": description
        or f"{pascal} scenario package for autoware_carla_scenario",
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def create_scenario_package(
    name: str,
    *,
    scenario_name: str | None = None,
    output_dir: str | Path = ".",
    description: str | None = None,
    force: bool = False,
) -> ScaffoldResult:
    """Render a new scenario package under *output_dir*.

    Parameters
    ----------
    name:
        Package name.  Accepts ``snake_case``, ``hyphen-case``, or ``PascalCase``
        -- it is normalised to a valid import name (e.g. ``my-pkg`` ->
        ``my_pkg``).
    scenario_name:
        Name of the scenario to scaffold.  Defaults to *name* with a trailing
        ``_package`` / ``_pkg`` / ``_scenario(s)`` suffix removed.
    output_dir:
        Parent directory in which the package directory is created.
    description:
        Optional package description.
    force:
        Overwrite existing files if the target directory already exists.

    Returns
    -------
    ScaffoldResult
        The created package ``root`` and the resolved ``names``.
    """
    names = resolve_names(name, scenario_name, description)
    root = Path(output_dir) / names["package_name"]

    if root.exists() and not force:
        msg = f"Target directory already exists: {root} (use force=True to overwrite)."
        raise FileExistsError(msg)

    env = code_environment(TEMPLATES_DIR)
    for template_name, out_template in _FILE_MANIFEST.items():
        out_path = root / out_template.format(**names)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = env.get_template(template_name).render(**names)
        out_path.write_text(rendered, encoding="utf-8")

    return ScaffoldResult(root=root, names=names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scenario-new",
        description="Generate a standalone autoware_carla_scenario package.",
    )
    parser.add_argument(
        "name",
        help="Package name (snake_case, hyphen-case, or PascalCase).",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        dest="scenario_name",
        default=None,
        help="Scenario name (default: derived from the package name).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory to create the package in (default: current directory).",
    )
    parser.add_argument(
        "-d",
        "--description",
        default=None,
        help="Package description.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite files if the target directory already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the ``scenario-new`` console script."""
    args = _build_parser().parse_args(argv)
    try:
        root, names = create_scenario_package(
            args.name,
            scenario_name=args.scenario_name,
            output_dir=args.output_dir,
            description=args.description,
            force=args.force,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        return 1

    print(f"Created scenario package at {root}")  # noqa: T201
    print("\nNext steps:")  # noqa: T201
    print(f"  uv pip install -e {root}")  # noqa: T201
    print(  # noqa: T201
        f"  uv run scenario scenario={names['scenario_name']}/default map=nishishinjuku"
    )
    print(  # noqa: T201
        f"  uv run scenario-check scenario={names['scenario_name']}/default"
        "   # static check, before CARLA starts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
