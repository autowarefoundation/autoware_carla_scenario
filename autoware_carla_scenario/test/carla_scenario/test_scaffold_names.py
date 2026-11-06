"""The naming rules of ``scenario-new``, moved into the checked
:mod:`autoware_carla_scenario.scaffold.names` (#1654).

``_to_snake`` split camelCase with a lookbehind and a lookahead, which Codon's
re (RE2) has no syntax for, so it now splits with a character loop: these
tests hold the loop to what the regular expression did.
"""

from __future__ import annotations

import inspect
import random
import re

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
        "_default_scenario_name",
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
