"""The naming rules of ``scenario-new``: package, scenario and class names.

How :mod:`scaffold.generator` turns the raw name it is given into the import,
distribution, scenario and class names of the package it writes, what it
refuses as an identifier, and which names it refuses for clashing once the
package is installed (:func:`check_package_name`) or with the package around
the scenario (:func:`check_scenario_name`, over the names the generator reads
off its templates and the framework's conf tree, #2364).  Kept apart from the
generator, which imports argparse, ast, importlib.metadata and jinja2, so the
static check compiles these rules (#1654); the generator re-exports them.  The rule that a package
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
#: floor under what ``scaffold.generator._dependency_import_names()`` reads
#: off the installed metadata (which is absent when the framework runs from a
#: bare source tree).  A scaffolded package of one of these names would shadow
#: it once installed.
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
        "lxml",
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
        "starlette",
        "tomli",
        "tqdm",
        "typesafe_carla",
        "uvicorn",
        "yaml",  # pyyaml
    }
)


def check_package_name(
    package_name: str,
    *,
    stdlib_names: frozenset[str],
    dependency_names: frozenset[str],
) -> None:
    """Refuse a package name that would clash once the package is installed.

    A package named after the framework depends on itself and replaces the
    installed framework on ``uv pip install -e``; one named after another
    top-level module the framework ships (``hydra_plugins``, #2031), a stdlib
    module (one of *stdlib_names*, ``sys.stdlib_module_names``) or a
    dependency of the framework (one of *dependency_names*, the import names
    the generator reads off the installed metadata) shadows it (#1451).

    Raises:
        ValueError: The name is reserved, saying why.
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
    if package_name in stdlib_names:
        msg = (
            f"package name {package_name!r} is reserved: it would shadow the "
            f"standard library module {package_name!r}. Pass a different name."
        )
        raise ValueError(msg)
    if package_name in dependency_names:
        msg = (
            f"package name {package_name!r} is reserved: it would shadow "
            f"{package_name!r}, which {framework_import} depends on. Pass a "
            f"different name."
        )
        raise ValueError(msg)


def reserved_scenario_names(
    manifest: dict[str, str],
    scenario_template: str,
    package_dir: str,
    init_globals: set[str],
) -> frozenset[str]:
    """Return the names the scenario module may not take.

    Every entry of *manifest* (``template name -> output path template``)
    but *scenario_template* that writes inside the import package (under
    *package_dir*) claims its top-level name there: ``configs.py`` claims
    ``configs``, ``__init__.py`` claims ``__init__`` and ``conf/...`` claims
    ``conf``; one named after the scenario itself (``{scenario_name}``)
    cannot collide.  The globals of the generated ``__init__.py``
    (*init_globals*) are claimed too.  Only lower-case names are kept: a
    scenario name is lower case once normalised.
    """
    reserved: set[str] = set()
    for template_name, out_template in manifest.items():
        if template_name == scenario_template or not out_template.startswith(
            package_dir
        ):
            continue
        top = out_template[len(package_dir) :].split("/", 1)[0]
        if "{" in top:
            # Named after the scenario itself, so it cannot collide.
            continue
        # str.removesuffix(".py"), which Codon's str lacks.
        if top.endswith(".py"):
            top = top[: len(top) - len(".py")]
        reserved.add(top)
    reserved |= init_globals
    # A scenario name is lower case once normalised (``CONF_DIR`` and the
    # config class cannot clash), so only lower-case names are listed.
    return frozenset(name for name in reserved if name == name.lower())


def scenario_class_names(scenario: str) -> tuple[str, str]:
    """Return the scenario and config class names of *scenario*.

    ``PascalCase`` of the scenario name, ``Scenario`` appended unless it
    already ends in it (no awkward doubling) and ``Config`` appended: the
    scenario ``reach_goal`` gets ``ReachGoalScenario`` and
    ``ReachGoalConfig``.
    """
    pascal = _to_pascal(scenario)
    scenario_class = pascal if pascal.endswith("Scenario") else f"{pascal}Scenario"
    return scenario_class, f"{pascal}Config"


def check_scenario_name(
    scenario: str,
    *,
    reserved: frozenset[str],
    imported: frozenset[str],
    groups: tuple[str, ...],
    module_globals: frozenset[str],
) -> None:
    """Refuse a scenario name that clashes with the package around it.

    *scenario* is a valid scenario name (:func:`_validate_scenario_name`).
    It is refused when it is one of *reserved* (a module, directory or global
    the generated package already has, :func:`reserved_scenario_names`), one
    of *imported* (a module the generated package imports, which a scenario
    module beside it would replace in the static check, #2031), a built-in
    scenario (:data:`BUILTIN_SCENARIO_NAMES`, #1802), one of *groups* (the
    framework's own scenario config groups, #1976), or when one of its
    classes (:func:`scenario_class_names`) is one of *module_globals*, a name
    the scenario module already binds.

    Raises:
        ValueError: The name is reserved, saying why.
    """
    if scenario in reserved:
        msg = (
            f"scenario name {scenario!r} is reserved: the generated package "
            f"already has a {scenario!r} module, directory or global (reserved "
            f"names: "
            f"{', '.join(sorted(reserved))}). Pass a different --scenario."
        )
        raise ValueError(msg)
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
    scenario_class, config_class = scenario_class_names(scenario)
    for generated in (scenario_class, config_class):
        if generated in module_globals:
            msg = (
                f"scenario name {scenario!r} is reserved: its class "
                f"{generated!r} would replace the {generated!r} the scenario "
                f"module already binds (an import from the framework). Pass a "
                f"different --scenario."
            )
            raise ValueError(msg)
