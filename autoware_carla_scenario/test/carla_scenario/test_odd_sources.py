"""OpenODD files taken from git repositories at a revision."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.coverage.report import CoverageReport, OddExposure
from autoware_carla_scenario.odd import (
    GitSource,
    OpenOddError,
    load_odd_binding,
    load_openodd,
)
from autoware_carla_scenario.odd.sources import CACHE_ENV, GitSourceError, fetch

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def wind_mps(world: Any) -> float:
    return 0.0


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "--quiet", "--initial-branch=main")
    return _commit(path, files)


def _commit(repo: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "update")
    return repo


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def _taxonomy(limit: str = "") -> str:
    return "TAXONOMY:\n    wind_speed: float velocity\n"


def _modules(limit: str) -> str:
    return (
        "IMPORT: [taxonomy.yml]\n"
        "ODD:\n"
        "    calm:\n"
        "        INCLUDE_AND:\n"
        f'            wind_speed: "< {limit} m/s"\n'
    )


@pytest.fixture(autouse=True)
def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv(CACHE_ENV, str(cache))
    return cache


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, Path]:
    """A taxonomy repository (tagged v1) and a module repository."""
    taxonomy = _repo(tmp_path / "taxonomy", {"taxonomy.yml": _taxonomy()})
    _git(taxonomy, "tag", "-a", "v1", "-m", "v1")
    modules = _repo(tmp_path / "modules", {"odd/calm.yml": _modules("10")})
    return taxonomy, modules


def _binding(tmp_path: Path, taxonomy: Path, modules: Path, rev: str) -> Path:
    binding = tmp_path / "calm.yaml"
    binding.write_text(
        "openodd:\n"
        f"  - git: {taxonomy}\n"
        "    rev: v1\n"
        "    path: taxonomy.yml\n"
        f"  - git: {modules}\n"
        f"    rev: {rev}\n"
        "    path: odd/calm.yml\n"
        "probes:\n"
        f"  wind_speed: {{probe: '{__name__}:wind_mps', unit: m/s}}\n"
    )
    return binding


def _inside(odd: Any, wind: float) -> bool:
    return bool(odd.evaluate({"wind_speed": wind}).inside)


class TestGitSources:
    def test_a_module_repository_imports_a_taxonomy_from_another(
        self, tmp_path: Path, repos: tuple[Path, Path]
    ) -> None:
        taxonomy, modules = repos
        odd = load_odd_binding(_binding(tmp_path, taxonomy, modules, "main"))
        assert _inside(odd, 5.0) and not _inside(odd, 15.0)
        assert odd.sources == [
            {
                "git": str(taxonomy),
                "rev": "v1",
                "commit": _head(taxonomy),
                "path": "taxonomy.yml",
            },
            {
                "git": str(modules),
                "rev": "main",
                "commit": _head(modules),
                "path": "odd/calm.yml",
            },
        ]
        assert odd.describe()["sources"] == odd.sources

    def test_a_commit_pins_the_files_and_a_branch_follows_them(
        self, tmp_path: Path, repos: tuple[Path, Path]
    ) -> None:
        taxonomy, modules = repos
        first = _head(modules)
        _commit(modules, {"odd/calm.yml": _modules("20")})
        pinned = load_odd_binding(_binding(tmp_path, taxonomy, modules, first))
        latest = load_odd_binding(_binding(tmp_path, taxonomy, modules, "main"))
        assert not _inside(pinned, 15.0)
        assert _inside(latest, 15.0)
        assert latest.sources[1]["commit"] == _head(modules)

    def test_an_abbreviated_commit_is_resolved(
        self, tmp_path: Path, repos: tuple[Path, Path]
    ) -> None:
        taxonomy, modules = repos
        first = _head(modules)
        _commit(modules, {"odd/calm.yml": _modules("20")})
        odd = load_odd_binding(_binding(tmp_path, taxonomy, modules, first[:10]))
        assert odd.sources[1]["commit"] == first
        assert not _inside(odd, 15.0)

    def test_a_commit_checked_out_once_needs_no_network(
        self, tmp_path: Path, repos: tuple[Path, Path]
    ) -> None:
        _, modules = repos
        commit = _head(modules)
        first = fetch(str(modules), commit)
        shutil.rmtree(modules)
        again = fetch(str(modules), commit)
        assert again == first
        assert (again.root / "odd" / "calm.yml").is_file()

    def test_an_uppercase_commit_is_read_from_the_cache(
        self, repos: tuple[Path, Path]
    ) -> None:
        _, modules = repos
        commit = _head(modules)
        fetch(str(modules), commit)
        shutil.rmtree(modules)
        assert fetch(str(modules), commit.upper()).commit == commit

    def test_a_tag_moved_upstream_is_followed(
        self, tmp_path: Path, repos: tuple[Path, Path]
    ) -> None:
        _, modules = repos
        _git(modules, "tag", "-a", "v1", "-m", "v1")
        first = _head(modules)
        fetch(str(modules), first[:10])  # the full fetch stores the tags
        _commit(modules, {"odd/calm.yml": _modules("20")})
        _git(modules, "tag", "-f", "-a", "v1", "-m", "v1 again")
        assert fetch(str(modules), "v1").commit == _head(modules)
        assert fetch(str(modules), _head(modules)[:10]).commit == _head(modules)

    @pytest.mark.parametrize("target", ["../../../../outside.yml", "ABSOLUTE"])
    def test_a_git_file_cannot_import_from_outside_the_sources(
        self, tmp_path: Path, target: str
    ) -> None:
        outside = tmp_path / "outside.yml"
        outside.write_text(_taxonomy())
        name = str(outside) if target == "ABSOLUTE" else target
        repo = _repo(tmp_path / "evil", {"odd.yml": f"IMPORT: ['{name}']\nODD: {{}}\n"})
        with pytest.raises(OpenOddError, match="imports only from the sources"):
            load_openodd(GitSource(str(repo), "main", "odd.yml"))

    def test_a_symlink_out_of_the_repository_is_refused(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path / "links", {"odd.yml": "ODD: {}\n"})
        (repo / "escape.yml").symlink_to("/etc/hostname")
        _commit(repo, {})
        with pytest.raises(OpenOddError):
            load_openodd(GitSource(str(repo), "main", "escape.yml"))

    def test_load_openodd_takes_git_sources(self, repos: tuple[Path, Path]) -> None:
        taxonomy, modules = repos
        odd = load_openodd(
            GitSource(str(modules), "main", "odd/calm.yml"),
            GitSource(str(taxonomy), "v1", "taxonomy.yml"),
        )
        assert odd.name == "calm"
        assert [s["path"] for s in odd.sources] == ["odd/calm.yml", "taxonomy.yml"]

    def test_an_import_found_in_two_sources_is_refused(
        self, tmp_path: Path, repos: tuple[Path, Path]
    ) -> None:
        _, modules = repos
        first = _repo(
            tmp_path / "first", {"a.yml": "{}\n", "taxonomy.yml": _taxonomy()}
        )
        second = _repo(
            tmp_path / "second", {"b.yml": "{}\n", "taxonomy.yml": _taxonomy()}
        )
        with pytest.raises(OpenOddError, match="in several sources"):
            load_openodd(
                GitSource(str(first), "main", "a.yml"),
                GitSource(str(second), "main", "b.yml"),
                GitSource(str(modules), "main", "odd/calm.yml"),
            )

    def test_two_files_of_one_name_are_refused(
        self, tmp_path: Path, repos: tuple[Path, Path]
    ) -> None:
        taxonomy, modules = repos
        other = _repo(tmp_path / "other", {"taxonomy.yml": _taxonomy()})
        with pytest.raises(OpenOddError, match="two files named taxonomy.yml"):
            load_openodd(
                GitSource(str(taxonomy), "v1", "taxonomy.yml"),
                GitSource(str(other), "main", "taxonomy.yml"),
                GitSource(str(modules), "main", "odd/calm.yml"),
            )

    def test_a_concept_defined_in_two_files_is_refused(
        self, tmp_path: Path, repos: tuple[Path, Path]
    ) -> None:
        taxonomy, modules = repos
        other = _repo(tmp_path / "other", {"more.yml": _taxonomy()})
        with pytest.raises(
            OpenOddError,
            match=r"TAXONOMY.wind_speed is defined twice \(in taxonomy.yml and more.yml\)",
        ):
            load_openodd(
                GitSource(str(taxonomy), "v1", "taxonomy.yml"),
                GitSource(str(other), "main", "more.yml"),
            )

    def test_a_file_adds_concepts_to_a_taxonomy_another_started(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "base.yml").write_text(
            "TAXONOMY:\n    weather:\n        wind_speed: float velocity\n"
        )
        (tmp_path / "extra.yml").write_text(
            "IMPORT: [base.yml]\n"
            "TAXONOMY:\n    weather:\n        gust_speed: float velocity\n"
            "ODD:\n    calm:\n        INCLUDE_AND:\n"
            '            gust_speed: "< 20 m/s"\n'
            '            wind_speed: "< 10 m/s"\n'
        )
        odd = load_openodd(tmp_path / "extra.yml")
        assert {a.name for a in odd.attributes} >= {
            "weather.wind_speed",
            "weather.gust_speed",
        }

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ("{git: REPO, rev: nope, path: odd/calm.yml}", "no revision 'nope'"),
            ("{git: REPO, rev: main, path: odd/none.yml}", "no file odd/none.yml"),
            ("{git: REPO, rev: main, path: ../../x.yml}", "leaves the repository"),
            ("{git: REPO, rev: main}", r"needs \['path'\]"),
            ("{git: REPO, rev: main, path: a.yml, ref: x}", r"unknown keys \['ref'\]"),
            ("{git: REPO, rev: --upload-pack=x, path: a.yml}", "bad rev"),
        ],
    )
    def test_bad_git_sources_are_refused(
        self, tmp_path: Path, repos: tuple[Path, Path], entry: str, message: str
    ) -> None:
        _, modules = repos
        binding = tmp_path / "bad.yaml"
        binding.write_text(f"openodd:\n  - {entry.replace('REPO', str(modules))}\n")
        with pytest.raises(OpenOddError, match=message):
            load_odd_binding(binding)

    def test_git_source_validates_its_fields(self) -> None:
        with pytest.raises(GitSourceError, match="bad url"):
            GitSource("-oProxyCommand=x", "main", "a.yml")
        with pytest.raises(GitSourceError, match="rev must be"):
            GitSource("repo", "", "a.yml")


class TestReportedSources:
    def _raw(self, commit: str) -> dict[str, Any]:
        return {
            "name": "calm",
            "ticks": {"inside": 1},
            "seconds": {"inside": 0.1},
            "sources": [
                {"git": "u", "rev": "main", "commit": commit, "path": "odd.yml"}
            ],
        }

    def test_the_report_names_the_commits_read(self) -> None:
        exposure = OddExposure("calm")
        exposure.add("s1", self._raw("a" * 40))
        exposure.add("s2", self._raw("a" * 40))
        lines = "\n".join(CoverageReport._odd_markdown(exposure))
        assert "- Source: u odd.yml at main (aaaaaaaaaaaa)" in lines
        assert "different commits" not in lines
        assert exposure.to_dict()["sources"] == self._raw("a" * 40)["sources"]

    def test_runs_on_different_commits_are_flagged(self) -> None:
        exposure = OddExposure("calm")
        exposure.add("s1", self._raw("a" * 40))
        exposure.add("s2", self._raw("b" * 40))
        lines = "\n".join(CoverageReport._odd_markdown(exposure))
        assert "the runs read different commits of a source" in lines
