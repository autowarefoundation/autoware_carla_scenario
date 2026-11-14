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
import importlib.metadata
import shutil
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

from ..templating import code_environment
from .names import (
    BUILTIN_SCENARIO_NAMES,
    _FRAMEWORK_DISTRIBUTION,
    _default_scenario_name,
    _is_framework_name,
    _normalise_distribution,
    _to_hyphen,
    _to_pascal,
    _to_snake,
    _validate_identifier,
    _validate_scenario_name,
)

# Putting a rendered package in place, which the library check compiles from
# its own module; every name of it is re-exported here, where it has always
# been imported from (the same objects).
from .placement import _RENAME_ONTO_OCCUPIED as _RENAME_ONTO_OCCUPIED
from .placement import PackageExistsError as PackageExistsError
from .placement import _move_into_place as _move_into_place
from .placement import _occupied as _occupied


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


def _template_globals(template_name: str) -> set[str]:
    """Return the names a template binds at module level once rendered.

    Rendered with :data:`_PLACEHOLDER_NAMES`, so a name the template takes
    from the scenario (its classes) comes out as a placeholder.
    ``__future__`` imports are compiler directives whose binding nothing
    reads, so they are left out.
    """
    source = (
        code_environment(TEMPLATES_DIR)
        .get_template(template_name)
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


def _package_init_globals() -> set[str]:
    """Return the names the generated ``__init__.py`` binds at module level.

    Importing the scenario module (``register()`` does) binds it as an
    attribute of the package, that is, as a global of ``__init__.py``: a
    scenario called ``register_scenario`` replaced the function ``register()``
    calls next with a module.
    """
    return _template_globals(_PACKAGE_INIT_TEMPLATE)


@functools.cache
def _scenario_module_globals() -> frozenset[str]:
    """Return the names the scenario module binds besides its own classes.

    ``scenario.py`` imports ``EgoConfig``, ``BaseScenario`` and the rest from
    the framework, then binds ``{config_class}`` (from ``.configs``) and
    ``{scenario_class}``.  A generated class of the same name as one of those
    imports replaces it: a scenario called ``ego`` made ``EgoConfig`` in
    ``scenario.py`` the package's own dataclass rather than the framework's.
    The classes named after the scenario are left out, being placeholders.
    """
    own = {_PLACEHOLDER_NAMES["scenario_class"], _PLACEHOLDER_NAMES["config_class"]}
    return frozenset(_template_globals(_SCENARIO_TEMPLATE) - own)


#: The framework's own scenario config groups, ``conf/scenario/<group>/``,
#: which Hydra's search path puts ahead of an installed package's.
_BUILTIN_SCENARIO_CONF = (
    Path(__file__).resolve().parent.parent / "examples" / "conf" / "scenario"
)


@functools.cache
def _builtin_scenario_groups() -> tuple[str, ...]:
    """Return the names of the framework's own scenario config groups, sorted.

    Every directory of the packaged ``examples/conf/scenario/`` -- builders'
    names (``lane_change``) and groups of configs that reuse another builder
    (``lane_change_fail``) alike.  A generated package named after one shares
    that Hydra group with the built-in (#1976): ``scenario=<group>/*`` also
    runs the built-in configs, and a built-in config shadows the package's
    config of the same name.  Hidden and private directories (``.x``,
    ``__pycache__``) are no groups.  Read from the conf tree the package ships (the
    wheel carries it, Hydra needs it), so it follows that tree.
    """
    try:
        entries = list(_BUILTIN_SCENARIO_CONF.iterdir())
    except OSError:  # pragma: no cover - the conf tree ships with the package
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in entries
            if entry.is_dir() and not entry.name.startswith((".", "_"))
        )
    )


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


def _imported_modules(source: str) -> set[str]:
    """Return the top-level modules *source* imports absolutely, anywhere in it.

    By the first component of every ``import x.y`` and ``from x.y import z``
    (a function body's imports too); relative imports and ``__future__`` are
    left out.
    """
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".", 1)[0])
    modules.discard("__future__")
    return modules


@functools.cache
def _template_imported_modules() -> frozenset[str]:
    """Return the top-level modules the generated import package imports.

    Every absolute import of the Python templates :data:`_FILE_MANIFEST`
    writes inside the import package -- ``math`` and ``dataclasses``
    (``configs.py``), ``logging`` (``scenario.py``),
    ``autoware_carla_scenario``, ... -- and of the prelude the static check
    puts above every module it compiles (:data:`typecheck.transform.PRELUDE`:
    ``enum``, ...).  Codon resolves an import to a module beside the importing
    one first, so a scenario module named after one of these
    (``<pkg>/math.py``, ``<pkg>/enum.py``) replaces it for every other module
    of the package, and the package fails its own static check (#2031).  Read
    off the rendered templates and the prelude, so it follows both.
    """
    # The checker's rewriter, imported here: scenario-new need not load it to
    # write a package, only to know what the check will import.
    from ..typecheck.transform import PRELUDE  # noqa: PLC0415

    modules = _imported_modules(PRELUDE)
    for template_name, out_template in _FILE_MANIFEST.items():
        if not (out_template.startswith(_PACKAGE_DIR) and out_template.endswith(".py")):
            continue
        modules |= _imported_modules(
            code_environment(TEMPLATES_DIR)
            .get_template(template_name)
            .render(**_PLACEHOLDER_NAMES)
        )
    return frozenset(modules)


#: The top-level modules the framework's wheel ships besides
#: ``autoware_carla_scenario`` (``[tool.uv.build-backend] module-name`` in its
#: ``pyproject.toml``; ``test_scaffold.py`` checks the two agree).
#: ``hydra_plugins`` is a PEP 420 namespace package the framework shares with
#: Hydra: a generated package of that name is a regular package, which Python
#: prefers over every namespace portion, so installing it hides the
#: framework's searchpath plugin and sweeper (#2031).  A plain constant, as an
#: editable install maps ``hydra_plugins`` to no distribution.
_FRAMEWORK_TOP_LEVEL_MODULES: tuple[str, ...] = ("hydra_plugins",)


#: Import names of the framework's direct run-time dependencies, kept as a
#: floor under what :func:`_dependency_import_names` reads off the installed
#: metadata (which is absent when the framework runs from a bare source tree).
#: A scaffolded package of one of these names would shadow it once installed.
_DEPENDENCY_IMPORT_NAMES: frozenset[str] = frozenset(
    {
        "autoware_lanelet2_extension_python",  # simple-lanelet2
        "carla_driver_interface",
        "cv2",  # opencv-python-headless
        "dotenv",  # python-dotenv
        "fastapi",
        "google",  # protobuf
        "grpc",  # grpcio
        "hydra",  # hydra-core
        "jinja2",
        "lanelet2",  # simple-lanelet2
        "multipart",  # python-multipart
        "mypy",
        "numpy",
        "omegaconf",
        "openodd",  # openodd-py
        "orjson",
        "packaging",
        "pydantic",
        "pytest",
        "pyxodr",
        "tomli",
        "tqdm",
        "typesafe_carla",
        "uvicorn",
        "yaml",  # pyyaml
    }
)


@functools.cache
def _dependency_import_names() -> frozenset[str]:
    """Return the import names of the framework's direct run-time dependencies.

    Read off the installed metadata (the framework's ``Requires-Dist`` without
    the extras, mapped to import names through
    :func:`importlib.metadata.packages_distributions`) and joined with
    :data:`_DEPENDENCY_IMPORT_NAMES`, so a dependency added later is covered
    without editing the list and a bare source tree still refuses the known
    ones.
    """
    from packaging.requirements import Requirement  # noqa: PLC0415

    names = set(_DEPENDENCY_IMPORT_NAMES)
    try:
        requires = importlib.metadata.requires(_FRAMEWORK_DISTRIBUTION) or []
    except importlib.metadata.PackageNotFoundError:
        return frozenset(names)
    dependencies: set[str] = set()
    for spec in requires:
        requirement = Requirement(spec)
        if requirement.marker is not None and "extra" in str(requirement.marker):
            # An optional extra is not installed with the framework.
            continue
        dependencies.add(_normalise_distribution(requirement.name))
    for (
        import_name,
        distributions,
    ) in importlib.metadata.packages_distributions().items():
        if import_name.isidentifier() and any(
            _normalise_distribution(dist) in dependencies for dist in distributions
        ):
            names.add(import_name)
    return frozenset(names)


def _check_package_name(package_name: str) -> None:
    """Refuse a package name that would clash once the package is installed.

    A package named after the framework depends on itself and replaces the
    installed framework on ``uv pip install -e``; one named after another
    top-level module the framework ships (``hydra_plugins``, #2031), a stdlib
    module or a dependency of the framework shadows it (#1451).
    """
    framework_import = _FRAMEWORK_DISTRIBUTION.replace("-", "_")
    if _is_framework_name(package_name):
        msg = (
            f"package name {package_name!r} is reserved: it is the framework's "
            f"own ({framework_import!r} / {_FRAMEWORK_DISTRIBUTION!r}), so the "
            f"package would depend on itself and replace the installed "
            f"framework. Pass a different name."
        )
        raise ValueError(msg)
    if package_name in _FRAMEWORK_TOP_LEVEL_MODULES:
        msg = (
            f"package name {package_name!r} is reserved: {framework_import} "
            f"ships a {package_name!r} namespace package, which a package of "
            f"that name would hide once installed (framework top-level "
            f"modules: {', '.join(_FRAMEWORK_TOP_LEVEL_MODULES)}). Pass a "
            f"different name."
        )
        raise ValueError(msg)
    if package_name in sys.stdlib_module_names:
        msg = (
            f"package name {package_name!r} is reserved: it would shadow the "
            f"standard library module {package_name!r}. Pass a different name."
        )
        raise ValueError(msg)
    if package_name in _dependency_import_names():
        msg = (
            f"package name {package_name!r} is reserved: it would shadow "
            f"{package_name!r}, which {framework_import} depends on. Pass a "
            f"different name."
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Name handling (the naming rules themselves are scaffold.names')
# ---------------------------------------------------------------------------


def resolve_names(
    name: str,
    scenario_name: str | None = None,
    description: str | None = None,
) -> dict[str, str]:
    """Resolve all template variables from the raw package *name*.

    Returns a dict with ``package_name`` (import/snake), ``distribution_name``
    (hyphen), ``scenario_name`` (snake), ``scenario_class`` /  ``config_class``
    (Pascal), ``description``, and ``requires_python``.

    The package name is the only source of the import and distribution names,
    so every package ``scenario-new`` writes goes through the refusal of a
    name that would clash once installed (see :func:`_check_package_name`).
    """
    from ..authoring.package_export import _requires_python  # noqa: PLC0415

    package_name = _validate_identifier(_to_snake(name), kind="package name")
    if not package_name.isascii():
        # The distribution name (PEP 508) and the entry-point key in
        # ``pyproject.toml`` (a bare TOML key) are ASCII-only, although a
        # Python identifier need not be.
        msg = f"package name {package_name!r} must be ASCII."
        raise ValueError(msg)
    _check_package_name(package_name)

    if scenario_name is None:
        # Default: strip a trailing package-ish suffix so
        # ``my_scenario_package`` yields the scenario ``my_scenario``.
        scenario = _default_scenario_name(package_name)
    else:
        scenario = _to_snake(scenario_name)
    scenario = _validate_identifier(scenario, kind="scenario name")
    # Hydra's override grammar takes ASCII ids only, which a Python
    # identifier need not be (#1858).
    scenario = _validate_scenario_name(scenario)
    reserved = _reserved_scenario_names()
    if scenario in reserved:
        msg = (
            f"scenario name {scenario!r} is reserved: the generated package "
            f"already has a {scenario!r} module, directory or global (reserved "
            f"names: "
            f"{', '.join(sorted(reserved))}). Pass a different --scenario."
        )
        raise ValueError(msg)
    imported = _template_imported_modules()
    if scenario in imported:
        msg = (
            f"scenario name {scenario!r} is reserved: the generated package "
            f"imports the module {scenario!r}, which a scenario module "
            f"{scenario}.py beside it would replace in the static check "
            f"(imported modules: {', '.join(sorted(imported))}). Pass a "
            f"different --scenario."
        )
        raise ValueError(msg)
    if scenario in BUILTIN_SCENARIO_NAMES:
        msg = (
            f"scenario name {scenario!r} is reserved: it names a built-in "
            f"scenario, which the installed package would replace while the "
            f"built-in configs (scenario={scenario}/...) still select it "
            f"(built-in scenarios: {', '.join(BUILTIN_SCENARIO_NAMES)}). Pass a "
            f"different --scenario."
        )
        raise ValueError(msg)
    groups = _builtin_scenario_groups()
    if scenario in groups:
        msg = (
            f"scenario name {scenario!r} is reserved: it names a built-in "
            f"scenario config group (conf/scenario/{scenario}/), which the "
            f"generated package's configs would share: scenario={scenario}/* "
            f"would also run the built-in configs, and a built-in config would "
            f"shadow the package's config of the same name (built-in groups: "
            f"{', '.join(groups)}). Pass a different --scenario."
        )
        raise ValueError(msg)

    pascal = _to_pascal(scenario)
    # Avoid awkward doubling when the name already ends in "scenario".
    scenario_class = pascal if pascal.endswith("Scenario") else f"{pascal}Scenario"
    config_class = f"{pascal}Config"
    taken = _scenario_module_globals()
    for generated in (scenario_class, config_class):
        if generated in taken:
            msg = (
                f"scenario name {scenario!r} is reserved: its class "
                f"{generated!r} would replace the {generated!r} the scenario "
                f"module already binds (an import from the framework). Pass a "
                f"different --scenario."
            )
            raise ValueError(msg)
    return {
        "package_name": package_name,
        "distribution_name": _to_hyphen(package_name),
        "scenario_name": scenario,
        "scenario_class": scenario_class,
        "config_class": config_class,
        # Read off the installed framework rather than written out here: a
        # literal drifted the last time the supported range moved, and a
        # scaffolded package that refuses to install on the interpreter that
        # scaffolded it is a bad first five minutes.
        "requires_python": _requires_python(),
        # A blank description (whitespace and line breaks alone, which the
        # pyproject's one-line description would fold to nothing) gets the
        # default too.
        "description": description
        if description and description.strip()
        else f"{pascal} scenario package for autoware_carla_scenario",
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
        Optional package description.  ``pyproject.toml`` gets it on one
        line, its line breaks folded into spaces (PEP 621 allows no other);
        the package's docstring keeps it as given.
    force:
        Replace the target directory if it already exists.  The whole
        directory is replaced, as ``authoring.package_export.export_package``
        replaces its target: nothing under it is kept, neither an earlier
        scenario's files (which would still be registered through
        ``register_conf_dir`` but no longer import) nor files added by hand.

    Returns
    -------
    ScaffoldResult
        The created package ``root`` and the resolved ``names``.

    Raises
    ------
    PackageExistsError
        *root* exists and *force* is not set.
    FileExistsError
        *root* exists and is not a directory (with *force*).
    OSError
        The package could not be written.  The package is rendered into a
        temporary sibling of *root* and moved into place only once complete,
        so nothing half-written is left behind, and with *force* the earlier
        package is left as it was.
    """
    names = resolve_names(name, scenario_name, description)
    root = Path(output_dir).expanduser() / names["package_name"]

    if _occupied(root):
        if not force:
            raise PackageExistsError(root)
        if not root.is_dir():
            raise FileExistsError(f"{root} exists and is not a directory.")

    env = code_environment(TEMPLATES_DIR)
    root.parent.mkdir(parents=True, exist_ok=True)
    # Rendered into a *sibling* of the target and moved into place once
    # complete, as authoring.package_export does: a failure part-way leaves
    # nothing at *root* (which would make the next run refuse with "already
    # exists"), and --force replaces the old tree rather than writing over it,
    # so an earlier scenario's files do not survive next to the new one's.
    # The package is a directory *inside* the temporary one, made with a plain
    # mkdir: mkdtemp's own directory is mode 0700, and moving that into place
    # would make the package unreadable to anyone but its owner.
    holder = Path(
        tempfile.mkdtemp(prefix=f".{names['package_name']}.new-", dir=root.parent)
    )
    try:
        staging = holder / names["package_name"]
        staging.mkdir()
        for template_name, out_template in _FILE_MANIFEST.items():
            out_path = staging / out_template.format(**names)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rendered = env.get_template(template_name).render(**names)
            out_path.write_text(rendered, encoding="utf-8")
        _move_into_place(staging, root, force=force)
    finally:
        shutil.rmtree(holder, ignore_errors=True)

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
        help=(
            "Replace the target directory if it already exists. The whole "
            "directory is replaced: files of an earlier scenario and files "
            "added by hand are not kept."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the ``scenario-new`` console script."""
    args = _build_parser().parse_args(argv)
    try:
        root, names = create_scenario_package(
            args.name,
            scenario_name=args.scenario_name,
            output_dir=Path(args.output_dir).expanduser(),
            description=args.description,
            force=args.force,
        )
    except PackageExistsError as exc:
        print(  # noqa: T201
            f"Error: target directory already exists: {exc.root} "
            "(use --force to replace it).",
            file=sys.stderr,
        )
        return 1
    except (ValueError, OSError) as exc:
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
