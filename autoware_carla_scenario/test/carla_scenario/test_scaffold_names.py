"""The naming rules of ``scenario-new``, moved into the checked
:mod:`autoware_carla_scenario.scaffold.names` (#1654).

``_to_snake`` split camelCase with a lookbehind and a lookahead, which Codon's
re (RE2) has no syntax for, so it now splits with a character loop: these
tests hold the loop to what the regular expression did.

The refusals of a package and a scenario name, the class names a scenario
gets and the manifest walk of the reserved names moved there too (#2364):
the tests below hold them to the code of ``scaffold.generator`` they
replace, message for message.
"""

from __future__ import annotations

import inspect
import random
import re
import sys

import pytest

from autoware_carla_scenario.scaffold import generator, names


def _old_to_snake(raw: str) -> str:
    """``_to_snake`` as it was before #1654, with its lookarounds."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw)
    snake = re.sub(r"[\s\-.]+", "_", spaced).strip("_").lower()
    return re.sub(r"__+", "_", snake)


def _old_default_scenario(package_name: str) -> str:
    """The default scenario name ``resolve_names`` derived before #1654."""
    stripped = re.sub(r"_(package|pkg|scenarios?)$", "", package_name)
    return stripped or package_name


#: The inputs the issue's prototype was compared on.
_ISSUE_INPUTS = [
    "MyScenario",
    "HTTPServer",
    "cut-in.v2",
    "fooBar9Baz",
    "a__B",
    "ÉtéScenario",
    "x1Y",
    "already_snake",
]

#: Characters the random names mix: ASCII upper and lower case and digits, the
#: separators, whitespace, and letters and digits outside ASCII (whose case
#: the ASCII-only ranges of the old pattern ignored).
_ALPHABET = (
    "aBzZ09_-. \t\n"
    "ÉéßẞÀàΣσДдİıǅ"  # non-ASCII upper, lower and title case letters
    "٣１Ａａ"  # non-ASCII digits, full-width letters
    "  "  # non-ASCII whitespace
)


@pytest.mark.parametrize("raw", _ISSUE_INPUTS)
def test_to_snake_matches_the_old_regex_on_the_issue_inputs(raw: str) -> None:
    assert names._to_snake(raw) == _old_to_snake(raw)


def test_to_snake_matches_the_old_regex_on_random_and_unicode_names() -> None:
    rng = random.Random(1654)
    samples = ["", "A", "a", "aA", "9A", "_A", "ÉA", "éA", "ａA", "١A", "aÉ"]
    for _ in range(20000):
        length = rng.randint(0, 12)
        samples.append("".join(rng.choice(_ALPHABET) for _ in range(length)))
    for raw in samples:
        assert names._to_snake(raw) == _old_to_snake(raw), raw
        split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw)
        assert names._split_camel_case(raw) == split, raw


def test_the_split_is_ascii_only() -> None:
    # A non-ASCII lower-case letter or digit before an upper-case letter, or a
    # non-ASCII upper-case letter after one, is no boundary.
    assert names._split_camel_case("éA") == "éA"
    assert names._split_camel_case("١A") == "١A"
    assert names._split_camel_case("aÉ") == "aÉ"
    assert names._split_camel_case("aＡ") == "aＡ"
    # ... so ``é`` before ``S`` is none either, as it was not for the regex.
    assert names._to_snake("ÉtéScenario") == "étéscenario"
    assert names._to_snake("EteScenario") == "ete_scenario"


@pytest.mark.parametrize(
    "package_name",
    [
        "my_scenario_package",
        "reach_goal_pkg",
        "cut_in_scenario",
        "cut_in_scenarios",
        "pkg",
        "_pkg",
        "a_package_pkg",
        "scenarios",
        "my_pkg_tools",
        "x_scenarioss",
    ],
)
def test_the_default_scenario_name_is_the_old_rule(package_name: str) -> None:
    assert names._default_scenario_name(package_name) == _old_default_scenario(
        package_name
    )


def test_the_generator_re_exports_the_moved_names() -> None:
    for name in (
        "_to_snake",
        "_to_pascal",
        "_to_hyphen",
        "_normalise_distribution",
        "_validate_identifier",
        "_validate_scenario_name",
        "_default_scenario_name",
        "BUILTIN_SCENARIO_NAMES",
        "_is_framework_name",
        "_FRAMEWORK_TOP_LEVEL_MODULES",
        "_DEPENDENCY_IMPORT_NAMES",
    ):
        assert getattr(generator, name) is getattr(names, name), name


def test_kind_stays_keyword_only() -> None:
    kind = inspect.signature(names._validate_identifier).parameters["kind"]
    assert kind.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        names._validate_identifier("class", kind="scenario name")
    with pytest.raises(ValueError, match="is empty after normalisation"):
        names._validate_identifier("", kind="package name")
    assert names._validate_identifier("match", kind="scenario name") == "match"


#: Names on either side of :data:`names.SCENARIO_NAME_PATTERN` (#1858).
_SCENARIO_NAME_CASES = [
    "a",
    "reach_goal",
    "cut_in_v2",
    "a1_",
    "",
    "_a",
    "1a",
    "Abc",
    "a-b",
    "a b",
    "café",
    "ａ",
    "a\n",
    "\na",
    "ä",
]


@pytest.mark.parametrize("name", _SCENARIO_NAME_CASES)
def test_is_scenario_name_is_the_pattern(name: str) -> None:
    # fullmatch: ``$`` alone would accept a trailing newline.
    assert names._is_scenario_name(name) is bool(
        re.fullmatch(names.SCENARIO_NAME_PATTERN, name)
    )


def test_the_editor_ids_share_the_pattern() -> None:
    from autoware_carla_scenario.authoring import models  # noqa: PLC0415

    assert models._ID_PATTERN.pattern == names.SCENARIO_NAME_PATTERN


def test_validate_scenario_name_names_the_hydra_grammar() -> None:
    assert names._validate_scenario_name("reach_goal") == "reach_goal"
    with pytest.raises(ValueError, match="Hydra's override grammar") as info:
        names._validate_scenario_name("café")
    assert "'café'" in str(info.value)


# ---------------------------------------------------------------------------
# The refusals of a package and a scenario name (#2364)
# ---------------------------------------------------------------------------


def _old_check_package_name(package_name: str) -> None:
    """``generator._check_package_name`` as it was before #2364."""
    framework_import = names._FRAMEWORK_DISTRIBUTION.replace("-", "_")
    if names._is_framework_name(package_name):
        msg = (
            f"package name {package_name!r} is reserved: it is the framework's "
            f"own ({framework_import!r} / {names._FRAMEWORK_DISTRIBUTION!r}), so the "
            f"package would depend on itself and replace the installed "
            f"framework. Pass a different name."
        )
        raise ValueError(msg)
    if package_name in ("hydra_plugins",):
        msg = (
            f"package name {package_name!r} is reserved: {framework_import} "
            f"ships a {package_name!r} namespace package, which a package of "
            f"that name would hide once installed (framework top-level "
            f"modules: {', '.join(('hydra_plugins',))}). Pass a "
            f"different name."
        )
        raise ValueError(msg)
    if package_name in sys.stdlib_module_names:
        msg = (
            f"package name {package_name!r} is reserved: it would shadow the "
            f"standard library module {package_name!r}. Pass a different name."
        )
        raise ValueError(msg)
    if package_name in generator._dependency_import_names():
        msg = (
            f"package name {package_name!r} is reserved: it would shadow "
            f"{package_name!r}, which {framework_import} depends on. Pass a "
            f"different name."
        )
        raise ValueError(msg)


def _old_reserved_scenario_names() -> frozenset[str]:
    """``generator._reserved_scenario_names`` as it was before #2364."""
    reserved: set[str] = set()
    for template_name, out_template in generator._FILE_MANIFEST.items():
        if template_name == generator._SCENARIO_TEMPLATE or not out_template.startswith(
            generator._PACKAGE_DIR
        ):
            continue
        top = out_template[len(generator._PACKAGE_DIR) :].split("/", 1)[0]
        if "{" in top:
            continue
        reserved.add(top.removesuffix(".py"))
    reserved |= generator._package_init_globals()
    return frozenset(name for name in reserved if name == name.lower())


def _old_check_scenario_name(scenario: str) -> tuple[str, str]:
    """The refusals of ``resolve_names`` as they were before #2364, then the
    class names it derived."""
    reserved = _old_reserved_scenario_names()
    if scenario in reserved:
        msg = (
            f"scenario name {scenario!r} is reserved: the generated package "
            f"already has a {scenario!r} module, directory or global (reserved "
            f"names: "
            f"{', '.join(sorted(reserved))}). Pass a different --scenario."
        )
        raise ValueError(msg)
    imported = generator._template_imported_modules()
    if scenario in imported:
        msg = (
            f"scenario name {scenario!r} is reserved: the generated package "
            f"imports the module {scenario!r}, which a scenario module "
            f"{scenario}.py beside it would replace in the static check "
            f"(imported modules: {', '.join(sorted(imported))}). Pass a "
            f"different --scenario."
        )
        raise ValueError(msg)
    if scenario in names.BUILTIN_SCENARIO_NAMES:
        msg = (
            f"scenario name {scenario!r} is reserved: it names a built-in "
            f"scenario, which the installed package would replace while the "
            f"built-in configs (scenario={scenario}/...) still select it "
            f"(built-in scenarios: {', '.join(names.BUILTIN_SCENARIO_NAMES)}). "
            f"Pass a different --scenario."
        )
        raise ValueError(msg)
    groups = generator._builtin_scenario_groups()
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
    pascal = names._to_pascal(scenario)
    scenario_class = pascal if pascal.endswith("Scenario") else f"{pascal}Scenario"
    config_class = f"{pascal}Config"
    taken = generator._scenario_module_globals()
    for generated in (scenario_class, config_class):
        if generated in taken:
            msg = (
                f"scenario name {scenario!r} is reserved: its class "
                f"{generated!r} would replace the {generated!r} the scenario "
                f"module already binds (an import from the framework). Pass a "
                f"different --scenario."
            )
            raise ValueError(msg)
    return scenario_class, config_class


def _outcome(check: object, *args: object) -> object:
    """What *check* returns for *args*, or the message it raises."""
    try:
        return check(*args)  # type: ignore[operator]
    except ValueError as exc:
        return ("ValueError", str(exc))


def _random_name(rng: random.Random) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz_0123456789"
    first = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return first + "".join(rng.choice(alphabet) for _ in range(rng.randrange(10)))


def _package_name_cases() -> list[str]:
    rng = random.Random(2364)
    return sorted(
        {
            "autoware_carla_scenario",
            "autoware-carla-scenario",
            "Autoware.Carla_Scenario",
            "hydra_plugins",
            "my_scenario_package",
            *sorted(sys.stdlib_module_names),
            *generator._dependency_import_names(),
            *(_random_name(rng) for _ in range(500)),
        }
    )


def _scenario_name_cases() -> list[str]:
    rng = random.Random(2031)
    return sorted(
        {
            "reach_goal",
            "ego",
            "base",
            "scenario",
            "my_scenario",
            "base_scenario",
            *_old_reserved_scenario_names(),
            *generator._template_imported_modules(),
            *names.BUILTIN_SCENARIO_NAMES,
            *generator._builtin_scenario_groups(),
            *(
                names._to_snake(name.removesuffix("Scenario").removesuffix("Config"))
                for name in generator._scenario_module_globals()
            ),
            *(_random_name(rng) for _ in range(500)),
        }
    )


def test_a_package_name_is_refused_as_before() -> None:
    refused = 0
    for name in _package_name_cases():
        expected = _outcome(_old_check_package_name, name)
        assert _outcome(generator._check_package_name, name) == expected, name
        got = _outcome(
            lambda n: names.check_package_name(
                n,
                stdlib_names=sys.stdlib_module_names,
                dependency_names=generator._dependency_import_names(),
            ),
            name,
        )
        assert got == expected, name
        refused += expected is not None
    assert refused > len(sys.stdlib_module_names)


def test_the_reserved_scenario_names_are_walked_as_before() -> None:
    assert generator._reserved_scenario_names() == _old_reserved_scenario_names()
    # The walk on a manifest of its own: a .py and a directory claim their
    # name, the scenario template and files outside the package none.
    manifest = {
        "a.jinja": "src/{package_name}/helpers.py",
        "b.jinja": "src/{package_name}/data/x.yaml",
        "c.jinja": "src/{package_name}/{scenario_name}_extra.py",
        "s.jinja": "src/{package_name}/scenario.py",
        "d.jinja": "README.md",
        "e.jinja": "src/{package_name}/Upper.py",
        "f.jinja": "src/{package_name}/x.py.py",
    }
    assert names.reserved_scenario_names(
        manifest, "s.jinja", "src/{package_name}/", {"register", "CONF_DIR"}
    ) == {"helpers", "data", "register", "x.py"}


def test_a_scenario_name_is_refused_as_before() -> None:
    refused = served = 0

    def new(scenario: str) -> tuple[str, str]:
        names.check_scenario_name(
            scenario,
            reserved=generator._reserved_scenario_names(),
            imported=generator._template_imported_modules(),
            groups=generator._builtin_scenario_groups(),
            module_globals=generator._scenario_module_globals(),
        )
        return names.scenario_class_names(scenario)

    for scenario in _scenario_name_cases():
        expected = _outcome(_old_check_scenario_name, scenario)
        assert _outcome(new, scenario) == expected, scenario
        was_refused = isinstance(expected, tuple) and expected[0] == "ValueError"
        refused += was_refused
        served += not was_refused
        if names._to_snake(scenario) != scenario:
            continue  # resolve_names() would normalise it to another name.
        # resolve_names() refuses it, or derives its classes, as before.
        resolved = _outcome(
            lambda s: generator.resolve_names("pkg_under_test", scenario_name=s),
            scenario,
        )
        if was_refused:
            assert resolved == expected, scenario
        else:
            assert isinstance(resolved, dict)
            classes = (resolved["scenario_class"], resolved["config_class"])
            assert classes == expected, scenario
    # Every refusal is met, and names are served too.
    assert refused > 20 and served > 100


def test_a_scenario_class_is_not_doubled() -> None:
    assert names.scenario_class_names("reach_goal") == (
        "ReachGoalScenario",
        "ReachGoalConfig",
    )
    assert names.scenario_class_names("my_scenario") == (
        "MyScenario",
        "MyScenarioConfig",
    )
