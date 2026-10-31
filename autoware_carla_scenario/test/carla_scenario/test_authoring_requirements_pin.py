"""The requirements pinning of a wheelhouse (``authoring.requirements_pin``,
#798), against what ``authoring.wheelhouse`` computed before it was split out,
and that the wheelhouse still reaches the same code under its old names."""

from __future__ import annotations

import itertools
import re

import pytest

from autoware_carla_scenario.authoring import requirements_pin, wheelhouse
from autoware_carla_scenario.authoring.requirements_pin import (
    DIRECT_SCHEMES,
    canonical_name,
    pin_direct_references,
    referenced_distribution,
)

# -- What authoring.wheelhouse computed before #798, verbatim ----------------


def _master_canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


_MASTER_DIRECT_SCHEMES = ("file:", "git+", "http:", "https:", "./", "../", "/")


def _master_referenced_distribution(reference: str) -> str:
    head = reference.split(";", 1)[0].strip()
    head, _, fragment = head.partition("#")
    subdirectory = re.search(r"subdirectory=([^&]+)", fragment)
    if subdirectory:
        return subdirectory.group(1).rstrip("/").split("/")[-1]
    if head.startswith("git+"):
        head = re.sub(r"@[^/@]+$", "", head)
    return head.rstrip("/").split("/")[-1].removesuffix(".git")


def _master_pin_direct_references(requirements: str, wheels: tuple[str, ...]) -> str:
    versions = {}
    for wheel in wheels:
        name, version = wheel.split("-")[:2]
        versions[_master_canonical(name)] = version

    lines = []
    for line in requirements.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            lines.append(line)
            continue

        if " @ " in stripped:
            name, _, rest = stripped.partition(" @ ")
        elif stripped.startswith(_MASTER_DIRECT_SCHEMES):
            name, rest = (
                _master_canonical(_master_referenced_distribution(stripped)),
                stripped,
            )
        else:
            lines.append(line)
            continue

        pinned = versions.get(_master_canonical(name.split("[")[0]))
        if pinned is None:
            lines.append(line)
            continue
        marker = f" ;{rest.split(';', 1)[1]}" if ";" in rest else ""
        lines.append(f"{name}=={pinned}{marker}")
    return "\n".join(lines)


# -- Inputs ------------------------------------------------------------------

NAMES = [
    "",
    "a",
    "A",
    "Foo_Bar",
    "foo.bar",
    "foo--bar",
    "foo-_.-bar",
    "_leading",
    "trailing.",
    "Autoware_Carla_Scenario",
    "zope.interface",
    "ruamel.yaml.clib",
    "UPPER-lower_Mixed.case",
    "ÄÖ_ü",
    "x[extra]",
]

REFERENCES = [
    "",
    " ",
    "/",
    "file:///somewhere/local/local_helper",
    "file:///somewhere/local/local_helper/",
    "file:///somewhere/local/local_helper ; python_version >= '3.10'",
    "file:///somewhere/local/local_helper;python_version<'3.12'",
    "./local_helper",
    "../sibling/pkg_two",
    "/abs/path/to/Pkg.Name",
    "/abs/path/to/repo.git",
    "/abs/path/to/repo.git/",
    "git+https://example.com/org/repo.git",
    "git+https://example.com/org/repo.git@v1.2.3",
    "git+https://example.com/org/repo@abc123",
    "git+https://example.com/org/repo.git@main#egg=repo",
    "git+https://example.com/org/repo@abc#subdirectory=autoware_carla_scenario",
    "git+https://example.com/org/repo@abc#subdirectory=pkgs/inner/",
    "git+https://example.com/org/repo@abc#egg=x&subdirectory=a/b&foo=1",
    "git+https://example.com/org/repo@abc#subdirectory=",
    "git+https://example.com/org/repo@abc#subdirectory=a&b",
    "git+ssh://git@example.com/org/repo.git@deadbeef",
    "git+https://user@example.com/org/repo",
    "git+https://example.com/org/repo/@ref",
    "git+file:///tmp/repo@ref ; sys_platform == 'linux'",
    "https://example.com/dist/pkg-1.0.tar.gz",
    "https://example.com/dist/pkg-1.0-py3-none-any.whl#sha256=abc",
    "http://example.com/simple/thing/",
    "https://example.com/a#b#subdirectory=c",
    "repo.git",
    ".git",
    "plain",
    "  file:///padded/name  ",
]

WHEELS = [
    "annotated_types-0.8.0-py3-none-any.whl",
    "autoware_carla_scenario-0.1.0-py3-none-any.whl",
    "local_helper-2.62.0-py3-none-any.whl",
    "Pkg.Name-3.0-py3-none-any.whl",
    "pkg_two-1.1-cp310-cp310-linux_x86_64.whl",
    "repo-9.9.9-py3-none-any.whl",
    "inner-0.0.1-py3-none-any.whl",
    "b-4-py3-none-any.whl",
    "colorama-0.4.6-py2.py3-none-any.whl",
    "name-1.0.whl",
]

REQUIREMENT_LINES = [
    "",
    "   ",
    "# comment",
    "    # via cut-in-scenario",
    "#   file:///commented/out",
    "annotated-types==0.8.0",
    "annotated-types==0.8.0 \\",
    "    --hash=sha256:abc",
    "colorama==0.4.6 ; sys_platform == 'win32'",
    "autoware-carla-scenario[carla] @ git+https://example.com/r@abc"
    "#subdirectory=autoware_carla_scenario",
    "autoware_carla_scenario @ git+https://example.com/r@abc ; python_version >= '3.10'",
    "Local.Helper @ file:///somewhere/local/local_helper",
    "unbuilt @ git+https://example.com/nope",
    "repo[a,b] @ git+https://example.com/org/repo.git@v1",
    "pkg-name @ https://example.com/pkg.whl;python_version<'3.12';extra",
    "file:///somewhere/local/local_helper",
    "file:///somewhere/local/local_helper ; python_version >= '3.10'",
    "./pkg_two",
    "../up/Pkg.Name",
    "/abs/inner",
    "git+https://example.com/org/repo.git@main",
    "git+https://example.com/x@abc#subdirectory=pkgs/inner",
    "https://example.com/dist/b",
    "http://example.com/unbuilt",
    "-e ./editable",
    "--find-links .",
    "-r other.txt",
    "\tfile:///tabbed/local_helper",
]


# -- Tests -------------------------------------------------------------------


class TestCanonicalName:
    @pytest.mark.parametrize("name", NAMES)
    def test_matches_master(self, name: str) -> None:
        assert canonical_name(name) == _master_canonical(name)


class TestDirectSchemes:
    def test_matches_master(self) -> None:
        assert DIRECT_SCHEMES == _MASTER_DIRECT_SCHEMES

    @pytest.mark.parametrize("line", REFERENCES + REQUIREMENT_LINES)
    def test_is_direct_matches_tuple_startswith(self, line: str) -> None:
        stripped = line.strip()
        assert requirements_pin._is_direct(stripped) == stripped.startswith(
            _MASTER_DIRECT_SCHEMES
        )


class TestReferencedDistribution:
    @pytest.mark.parametrize("reference", REFERENCES)
    def test_matches_master(self, reference: str) -> None:
        assert referenced_distribution(reference) == _master_referenced_distribution(
            reference
        )

    def test_subdirectory_wins_over_the_path(self) -> None:
        assert (
            referenced_distribution(
                "git+https://example.com/org/repo@abc#subdirectory=pkgs/inner/"
            )
            == "inner"
        )

    def test_git_ref_and_suffix_are_dropped(self) -> None:
        assert (
            referenced_distribution("git+https://example.com/org/repo.git@v1.2.3")
            == "repo"
        )


class TestPinDirectReferences:
    @pytest.mark.parametrize("line", REQUIREMENT_LINES)
    def test_each_line_matches_master(self, line: str) -> None:
        wheels = tuple(WHEELS)
        assert pin_direct_references(line, wheels) == _master_pin_direct_references(
            line, wheels
        )

    @pytest.mark.parametrize("count", [0, 1, 3, len(WHEELS)])
    def test_whole_files_match_master(self, count: int) -> None:
        for wheels in itertools.combinations(WHEELS, count):
            for text in (
                "\n".join(REQUIREMENT_LINES),
                "\n".join(REQUIREMENT_LINES) + "\n",
                "\r\n".join(REQUIREMENT_LINES),
                "\n".join(reversed(REQUIREMENT_LINES)),
                "",
                "\n\n",
            ):
                assert pin_direct_references(
                    text, wheels
                ) == _master_pin_direct_references(text, wheels)

    @pytest.mark.parametrize("reference", REFERENCES)
    def test_bare_references_match_master(self, reference: str) -> None:
        wheels = tuple(WHEELS)
        assert pin_direct_references(
            reference, wheels
        ) == _master_pin_direct_references(reference, wheels)

    def test_later_wheels_win_as_in_master(self) -> None:
        wheels = ("repo-1.0-py3-none-any.whl", "Repo-2.0-py3-none-any.whl")
        text = "repo @ git+https://example.com/repo"
        assert pin_direct_references(text, wheels) == "repo==2.0"
        assert pin_direct_references(text, wheels) == _master_pin_direct_references(
            text, wheels
        )

    @pytest.mark.parametrize("wheel", ["nodash.whl", ""])
    def test_a_wheel_without_a_version_raises_as_in_master(self, wheel: str) -> None:
        with pytest.raises(ValueError) as master:
            _master_pin_direct_references("x", (wheel,))
        with pytest.raises(ValueError) as moved:
            pin_direct_references("x", (wheel,))
        assert str(moved.value) == str(master.value)


class TestWheelhouseUsesTheModule:
    def test_old_names_are_the_moved_code(self) -> None:
        assert wheelhouse._canonical is canonical_name
        assert wheelhouse._DIRECT_SCHEMES is DIRECT_SCHEMES
        assert wheelhouse._referenced_distribution is referenced_distribution
        assert wheelhouse._pin_direct_references is pin_direct_references
