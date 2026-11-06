"""The names an exported package takes from its document id, moved into the
checked :mod:`autoware_carla_scenario.authoring.package_names` (#1695).

The move must not rename the package of any id that exported before; the ids
whose package would be the framework's own are now refused by the validator,
and so are the ids of the built-in scenarios (#1808).  Since #1358 the package
always adds ``_scenario``, so an id already ending in it is the one that
exports under a new name.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from autoware_carla_scenario.authoring import package_export, package_names
from autoware_carla_scenario.authoring.framework_pin import Pin
from autoware_carla_scenario.authoring.starter import blank_document, new_document
from autoware_carla_scenario.authoring.validator import (
    EXPORT_REFUSAL_NOTE,
    validate_document,
)
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


def _names_since_1358(scenario_id: str) -> dict[str, str]:
    """``_old_names``, but ``_scenario`` added even when already there (#1358)."""
    package_name = f"{_old_sanitize(scenario_id)}_scenario"
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


_FIXED_IDS = [
    "my_cut_in",
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
        """Only an id already ending in ``_scenario`` is renamed (#1358)."""
        assert _ID.match(scenario_id)
        names = package_names.names_for_id(scenario_id)
        assert names == _names_since_1358(scenario_id)
        if not _old_sanitize(scenario_id).endswith("_scenario"):
            assert names == _old_names(scenario_id)
        assert package_names._sanitize_identifier(scenario_id) == _old_sanitize(
            scenario_id
        )

    def test_package_names_wraps_names_for_id(self) -> None:
        document = new_document("my_cut_in")
        assert package_export.package_names(document) == package_names.names_for_id(
            "my_cut_in"
        )

    def test_the_exporter_re_exports_the_moved_names(self) -> None:
        for name in ("_sanitize_identifier", "_pin_note", "_pin_summary"):
            assert getattr(package_export, name) is getattr(package_names, name)


class TestFrameworkName:
    @pytest.mark.parametrize(
        "scenario_id", ["autoware_carla", "autoware__carla", "autoware_carla_"]
    )
    def test_the_framework_package_is_refused(self, scenario_id: str) -> None:
        problem = package_names.framework_name_problem(scenario_id)
        assert problem is not None
        assert "framework's own" in problem
        with pytest.raises(ValueError, match="framework's own"):
            package_names.names_for_id(scenario_id)

    @pytest.mark.parametrize(
        "scenario_id",
        # ``autoware_carla_scenario`` exports ``autoware_carla_scenario_scenario``
        # since ``_scenario`` is always added (#1358).
        ["autoware", "carla", "autoware_carla_x", "autoware_carla_scenario"],
    )
    def test_other_ids_are_not(self, scenario_id: str) -> None:
        assert package_names.framework_name_problem(scenario_id) is None

    @pytest.mark.parametrize("scenario_id", ["autoware_carla", "autoware__carla"])
    def test_the_validator_reports_an_id_error(self, scenario_id: str) -> None:
        report = validate_document(new_document(scenario_id))
        assert not report.ok
        errors = [i for i in report.errors if i.path == "id"]
        assert len(errors) == 1
        assert "framework's own" in errors[0].message
        # The refusal is the only issue the id gets.
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


class TestBuiltinScenarioName:
    """An id that names a built-in scenario is refused (#1808).

    The package registers its builder under the id, so once installed it
    would replace the built-in one while the built-in configs still select
    it, as ``scenario-new`` refuses the same name (#1802).
    """

    @pytest.mark.parametrize("scenario_id", names.BUILTIN_SCENARIO_NAMES)
    def test_every_built_in_name_is_refused(self, scenario_id: str) -> None:
        problem = package_names.framework_name_problem(scenario_id)
        assert problem is not None
        assert "names a built-in scenario" in problem
        assert f"scenario={scenario_id}/..." in problem
        with pytest.raises(ValueError, match="names a built-in scenario"):
            package_names.names_for_id(scenario_id)

    @pytest.mark.parametrize(
        "scenario_id", ["my_cut_in", "cut_in_scenario", "cut__in", "lane_change2"]
    )
    def test_names_close_to_one_are_not(self, scenario_id: str) -> None:
        assert package_names.framework_name_problem(scenario_id) is None

    def test_the_message_is_worded_as_scenario_new_words_it(self) -> None:
        problem = package_names.framework_name_problem("cut_in")
        with pytest.raises(ValueError) as raised:
            generator.resolve_names("my_pkg", scenario_name="cut_in")
        # Both name the reason and the built-in scenarios the same way; only
        # what to pass instead differs (an id, a ``--scenario``).
        shared = str(raised.value).split(": ", 1)[1].rsplit(" Pass a", 1)[0]
        assert problem is not None
        assert shared in problem

    def test_the_validator_reports_an_id_error(self) -> None:
        report = validate_document(new_document("cut_in"))
        assert not report.ok
        errors = [i for i in report.errors if i.path == "id"]
        assert len(errors) == 1
        assert "names a built-in scenario" in errors[0].message
        assert not [i for i in report.warnings if i.path == "id"]

    def test_the_export_refuses_it_and_writes_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(
            package_export.PackageExportError, match="names a built-in scenario"
        ):
            package_export.export_package(new_document("cut_in"), tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_the_starter_does_not_take_one(self) -> None:
        assert new_document().id not in names.BUILTIN_SCENARIO_NAMES


class TestSharedPackage:
    """No two ids the export accepts share a package (#1358).

    ``_scenario`` is always added, so an id ending in it has a package of its
    own; ``__`` and a trailing ``_`` still fold into another id's package, so
    the validator warns about them and the export refuses them.
    """

    @pytest.mark.parametrize(
        ("scenario_id", "package_name"),
        [
            ("cut_in_scenario", "cut_in_scenario_scenario"),
            ("cut__in", "cut_in_scenario"),
            ("cut_in_", "cut_in_scenario"),
            ("cut__in_scenario_", "cut_in_scenario_scenario"),
        ],
    )
    def test_the_package_name(self, scenario_id: str, package_name: str) -> None:
        assert package_names.names_for_id(scenario_id)["package_name"] == package_name

    @pytest.mark.parametrize(
        "scenario_id", ["cut_in_scenario", "my_cut_in", "new_scenario"]
    )
    def test_an_id_of_its_own_is_not_warned_about(self, scenario_id: str) -> None:
        assert package_names.scenario_id_problem(scenario_id) is None
        report = validate_document(new_document(scenario_id))
        assert report.ok
        assert not [i for i in report.issues if i.path == "id"]

    @pytest.mark.parametrize("scenario_id", ["cut__in", "cut_in_", "cut__in_scenario_"])
    def test_a_shared_one_is_a_warning(self, scenario_id: str) -> None:
        problem = package_names.scenario_id_problem(scenario_id)
        assert problem is not None
        report = validate_document(new_document(scenario_id))
        assert report.ok
        warnings = [i for i in report.warnings if i.path == "id"]
        assert [w.message for w in warnings] == [f"{problem} {EXPORT_REFUSAL_NOTE}"]

    def test_the_export_refuses_a_shared_one_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(package_export.PackageExportError, match="contains `__`"):
            package_export.export_package(new_document("cut__in"), tmp_path)
        assert list(tmp_path.iterdir()) == []


class TestStarterDocuments:
    """The editor's starting drafts validate with no errors and no warnings."""

    def test_blank_document(self) -> None:
        report = validate_document(blank_document())
        assert report.ok
        assert report.errors == ()
        assert report.warnings == ()

    def test_new_document(self) -> None:
        report = validate_document(new_document())
        assert report.ok
        assert report.errors == ()
        assert report.warnings == ()


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
