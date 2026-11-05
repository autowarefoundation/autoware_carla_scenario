"""The names an exported package takes from its document id, moved into the
checked :mod:`autoware_carla_scenario.authoring.package_names` (#1695).

The move must not rename the package of any id that exported before; the ids
whose package would be the framework's own are now refused by the validator,
and ids that share a package directory are warned about.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from autoware_carla_scenario.authoring import package_export, package_names
from autoware_carla_scenario.authoring.framework_pin import Pin
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.scaffold import generator, names
from autoware_carla_scenario.typecheck.library import CHECKED

_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def _old_sanitize(raw: str) -> str:
    """``_sanitize_identifier`` as package_export had it before #1695."""
    snake = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_").lower()
    snake = re.sub(r"__+", "_", snake)
    if not snake or snake[0].isdigit():
        snake = f"scenario_{snake}" if snake else "scenario"
    return snake


def _old_names(scenario_id: str) -> dict[str, str]:
    """``package_names`` as package_export had it before #1695."""
    module_id = _old_sanitize(scenario_id)
    package_name = (
        module_id if module_id.endswith("_scenario") else f"{module_id}_scenario"
    )
    return {
        "scenario_id": scenario_id,
        "package_name": package_name,
        "distribution_name": package_name.replace("_", "-"),
        "document_path_env": f"{package_name.upper()}_DOCUMENT_PATH",
    }


def _random_ids(count: int) -> list[str]:
    rng = random.Random(1695)
    alphabet = "abcz09_"
    ids: list[str] = []
    while len(ids) < count:
        candidate = rng.choice("abcz") + "".join(
            rng.choice(alphabet) for _ in range(rng.randint(0, 12))
        )
        ids.append(candidate)
    return ids


def _suffix_ids(count: int) -> list[str]:
    """Ids built from ``_scenario`` suffixes, underscores and the framework name."""
    rng = random.Random(16952)
    heads = ["a", "cut_in", "scenario", "autoware_carla", "autoware"]
    parts = ["_", "__", "scenario", "_scenario", "__scenario", "carla", "a"]
    ids: list[str] = []
    while len(ids) < count:
        candidate = rng.choice(heads) + "".join(
            rng.choice(parts) for _ in range(rng.randint(0, 4))
        )
        ids.append(candidate)
    return ids


_FIXED_IDS = [
    "cut_in",
    "cut_in_scenario",
    "stop__line",
    "stop_line_",
    "scenario",
    "scenario_scenario",
    "class",
    "a",
    "a9",
    "x_scenario_scenario",
]


class TestNames:
    def test_the_module_is_checked(self) -> None:
        assert "autoware_carla_scenario.authoring.package_names" in CHECKED

    @pytest.mark.parametrize("scenario_id", _FIXED_IDS + _random_ids(300))
    def test_valid_ids_give_the_names_they_gave_before(self, scenario_id: str) -> None:
        assert _ID.match(scenario_id)
        assert package_names.names_for_id(scenario_id) == _old_names(scenario_id)
        assert package_names._sanitize_identifier(scenario_id) == _old_sanitize(
            scenario_id
        )

    def test_package_names_wraps_names_for_id(self) -> None:
        document = new_document("cut_in")
        assert package_export.package_names(document) == package_names.names_for_id(
            "cut_in"
        )

    def test_the_exporter_re_exports_the_moved_names(self) -> None:
        for name in ("_sanitize_identifier", "_pin_note", "_pin_summary"):
            assert getattr(package_export, name) is getattr(package_names, name)


class TestFrameworkName:
    @pytest.mark.parametrize(
        "scenario_id", ["autoware_carla", "autoware_carla_scenario", "autoware__carla"]
    )
    def test_the_framework_package_is_refused(self, scenario_id: str) -> None:
        problem = package_names.framework_name_problem(scenario_id)
        assert problem is not None
        assert "framework's own" in problem
        with pytest.raises(ValueError, match="framework's own"):
            package_names.names_for_id(scenario_id)

    @pytest.mark.parametrize("scenario_id", ["autoware", "carla", "autoware_carla_x"])
    def test_other_ids_are_not(self, scenario_id: str) -> None:
        assert package_names.framework_name_problem(scenario_id) is None

    @pytest.mark.parametrize(
        "scenario_id", ["autoware_carla", "autoware_carla_scenario"]
    )
    def test_the_validator_reports_an_id_error(self, scenario_id: str) -> None:
        report = validate_document(new_document(scenario_id))
        assert not report.ok
        errors = [i for i in report.errors if i.path == "id"]
        assert len(errors) == 1
        assert "framework's own" in errors[0].message
        # Not also warned about as a shared directory.
        assert not [i for i in report.warnings if i.path == "id"]

    def test_the_export_refuses_it(self, tmp_path: Path) -> None:
        with pytest.raises(package_export.PackageExportError, match="framework's own"):
            package_export.export_package(new_document("autoware_carla"), tmp_path)

    def test_the_scaffolder_shares_the_rule(self) -> None:
        assert generator._is_framework_name is names._is_framework_name
        assert generator._FRAMEWORK_DISTRIBUTION == names._FRAMEWORK_DISTRIBUTION
        for name in ("autoware_carla_scenario", "autoware-carla-scenario"):
            assert names._is_framework_name(name)
        assert not names._is_framework_name("cut_in_scenario")
        with pytest.raises(ValueError, match="framework's own"):
            generator.resolve_names("autoware_carla_scenario")


class TestSharedDirectory:
    @pytest.mark.parametrize(
        ("scenario_id", "short_id"),
        [
            ("cut_in_scenario", "cut_in"),
            ("cut__in", "cut_in"),
            ("cut_in_", "cut_in"),
            ("scenario_scenario", "scenario"),
            ("cut_in", None),
            ("scenario", None),
            ("a_scenario_scenario", None),
            ("a_scenario__scenario_", "a_scenario_scenario"),
            ("a_scenario", "a"),
            ("autoware_carla_scenario_scenario", None),
            ("autoware_carla_scenario__scenario", "autoware_carla_scenario_scenario"),
            ("autoware_carla", None),
            ("autoware_carla_scenario", None),
        ],
    )
    def test_shared_package_id(self, scenario_id: str, short_id: str | None) -> None:
        assert package_names.shared_package_id(scenario_id) == short_id
        if short_id is not None:
            assert (
                package_names.names_for_id(scenario_id)["package_name"]
                == package_names.names_for_id(short_id)["package_name"]
            )

    def test_a_returned_id_shares_the_package(self) -> None:
        returned = 0
        for scenario_id in _suffix_ids(20000) + _random_ids(2000):
            short_id = package_names.shared_package_id(scenario_id)
            if short_id is None:
                continue
            returned += 1
            assert short_id != scenario_id, scenario_id
            assert len(short_id) <= len(scenario_id), scenario_id
            assert _ID.match(short_id), scenario_id
            assert package_names.framework_name_problem(short_id) is None, scenario_id
            assert (
                package_names.names_for_id(short_id)["package_name"]
                == package_names.names_for_id(scenario_id)["package_name"]
            ), scenario_id
            assert package_names.shared_package_id(short_id) is None, scenario_id
        assert returned > 1000

    def test_the_validator_warns(self) -> None:
        report = validate_document(new_document("cut_in_scenario"))
        assert report.ok
        warnings = [i for i in report.warnings if i.path == "id"]
        assert len(warnings) == 1
        assert "'cut_in'" in warnings[0].message

    def test_the_short_id_is_not_warned_about(self) -> None:
        report = validate_document(new_document("cut_in"))
        assert not [i for i in report.issues if i.path == "id"]


class TestPinLines:
    def test_pin_note(self) -> None:
        assert "exact release" in package_names._pin_note(Pin(version="1.2.3"))
        assert "exact commit" in package_names._pin_note(
            Pin(kind="git", repository="https://x/y", commit="a" * 40)
        )
        assert "does NOT resolve" in package_names._pin_note(
            Pin(kind="path", path="/x")
        )

    def test_pin_summary(self) -> None:
        assert package_names._pin_summary(Pin(version="1.2.3")) == (
            "version `1.2.3` (exact)"
        )
        assert package_names._pin_summary(
            Pin(kind="git", repository="https://x/y", commit="a" * 40)
        ) == (f"commit `{'a' * 40}` of `https://x/y`")
        assert "not portable" in package_names._pin_summary(Pin(kind="path", path="/x"))
