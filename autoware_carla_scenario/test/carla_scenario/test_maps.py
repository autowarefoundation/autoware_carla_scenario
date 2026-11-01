"""HD maps that live in a git repository.

Everything here runs against a repository created in ``tmp_path``, so the tests
say nothing about HuggingFace being reachable and never wait on it.  That is
also what lets them assert the thing that matters most about the cache: after
the first checkout, the map resolves with the *origin deleted*, which no
mock-based test could establish.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from autoware_carla_scenario.authoring.persistence import DraftStore

from autoware_carla_scenario.maps import (
    GitMapCache,
    MapCacheError,
    MapSource,
    MapSourceError,
    cached_map,
    list_maps,
    map_root,
    pin_source,
    resolve_directory,
    resolve_map,
    resolve_map_paths,
)
from autoware_carla_scenario.maps.cache import (
    AUTOWARE_DATA_DIR_ENV,
    CACHE_ROOT_ENV,
    is_lfs_pointer,
)
from autoware_carla_scenario.maps.opendrive import (
    OpenDriveUnavailable,
    capture_opendrive,
    ensure_xodr,
    installed_xodr,
    map_asset_env_var,
    map_asset_path,
)
from autoware_carla_scenario.maps.resolver import MapResolutionError

# A Lanelet2 map small enough to inline and real enough to be recognised as one.
_OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6">
  <node id="1" lat="0.0" lon="0.0" version="1"/>
</osm>
"""


def _git(cwd: Path, *args: str) -> None:
    """Run a git command in *cwd*, failing the test on a non-zero exit."""
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )


@pytest.fixture
def origin_repo(tmp_path: Path) -> Path:
    """A git repository laid out the way a published map repository is."""
    repo = tmp_path / "origin"
    (repo / "autoware_maps" / "TownA").mkdir(parents=True)
    (repo / "autoware_maps" / "TownB").mkdir(parents=True)
    (repo / "docs").mkdir()

    (repo / "autoware_maps" / "TownA" / "lanelet2_map.osm").write_text(_OSM)
    (repo / "autoware_maps" / "TownA" / "map_projector_info.yaml").write_text(
        "projector_type: Local\n"
    )
    (repo / "autoware_maps" / "TownB" / "lanelet2_map.osm").write_text(_OSM)
    (repo / "autoware_maps" / "TownB" / "TownB.xodr").write_text("<OpenDRIVE/>")
    (repo / "docs" / "README.md").write_text("not a map\n")

    _git(repo, "init", "--quiet", "--initial-branch", "main", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "Publish TownA and TownB")
    return repo


@pytest.fixture
def cache(tmp_path: Path) -> GitMapCache:
    """A map cache rooted somewhere this test owns."""
    return GitMapCache(tmp_path / "map_root")


def _uri(repo: Path, path: str = "autoware_maps/TownA", ref: str = "main") -> str:
    """Return a source URI naming *path* in the local *repo*."""
    return f"git+file://{repo}@{ref}#{path}"


def _git_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the git subcommand of every git invocation the cache makes."""
    seen: list[str] = []
    original = GitMapCache._git

    def _spy(self: Any, cwd: Any, *args: str, **kwargs: Any) -> str:
        seen.append(args[0])
        return str(original(self, cwd, *args, **kwargs))

    monkeypatch.setattr(GitMapCache, "_git", _spy)
    return seen


def _head(repo: Path) -> str:
    """Return the repository's current commit."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Source URIs
# ---------------------------------------------------------------------------


class TestMapSource:
    """Parsing what a person has: a URL out of a repository browser."""

    @pytest.mark.parametrize(
        ("uri", "repo_url", "ref", "path"),
        [
            (
                "https://huggingface.co/datasets/OWNER/NAME/tree/splatsim/maps/T",
                "https://huggingface.co/datasets/OWNER/NAME",
                "splatsim",
                "maps/T",
            ),
            (
                "https://github.com/OWNER/NAME/tree/main/maps/T",
                "https://github.com/OWNER/NAME",
                "main",
                "maps/T",
            ),
            (
                "https://huggingface.co/datasets/O/N/resolve/dev/m/lanelet2_map.osm",
                "https://huggingface.co/datasets/O/N",
                "dev",
                "m/lanelet2_map.osm",
            ),
            (
                "git+https://example.com/o/n@v1.2#maps/T",
                "https://example.com/o/n",
                "v1.2",
                "maps/T",
            ),
            ("https://example.com/o/n", "https://example.com/o/n", "", ""),
            # A ref may contain '/': everything after the last '@' of the
            # repository path is the ref (#671).
            (
                "git+https://github.com/o/r@feature/x#maps",
                "https://github.com/o/r",
                "feature/x",
                "maps",
            ),
            (
                "https://user@example.com/o/n@release/1.0",
                "https://user@example.com/o/n",
                "release/1.0",
                "",
            ),
            (
                "git@github.com:o/r.git@feature/x#maps/T",
                "git@github.com:o/r.git",
                "feature/x",
                "maps/T",
            ),
            (
                "https://github.com/o/n/tree/dev/maps/T@feature/x",
                "https://github.com/o/n",
                "feature/x",
                "maps/T",
            ),
            # An scp-like URL with no '/' at all is still a repository (#671).
            ("git@example.com:maps.git@v1", "git@example.com:maps.git", "v1", ""),
            ("git@example.com:maps.git", "git@example.com:maps.git", "", ""),
            ("git@github.com:o/r", "git@github.com:o/r", "", ""),
            ("git@github.com:o/r@main", "git@github.com:o/r", "main", ""),
            (
                "ssh://git@github.com/o/r.git@v2",
                "ssh://git@github.com/o/r.git",
                "v2",
                "",
            ),
            ("file:///srv/maps@main#T", "file:///srv/maps", "main", "T"),
            (
                "ssh://git@example.com:2222/o/r.git@feature/x",
                "ssh://git@example.com:2222/o/r.git",
                "feature/x",
                "",
            ),
            (
                "git+ssh://git@example.com/o/r@release/1.0#maps",
                "ssh://git@example.com/o/r",
                "release/1.0",
                "maps",
            ),
            # A remote path's own '@' is written '%40' and is then no split.
            (
                "https://example.com/o/a%40b@v1",
                "https://example.com/o/a%40b",
                "v1",
                "",
            ),
            ("/srv/maps@main", "/srv/maps", "main", ""),
            ("/srv/maps", "/srv/maps", "", ""),
            # GitLab's views sit behind a '-' segment (#1046).
            (
                "https://gitlab.com/o/maps/-/tree/main/nishishinjuku",
                "https://gitlab.com/o/maps",
                "main",
                "nishishinjuku",
            ),
            (
                "https://gitlab.com/o/maps/-/blob/v1.0/m/lanelet2_map.osm",
                "https://gitlab.com/o/maps",
                "v1.0",
                "m/lanelet2_map.osm",
            ),
            (
                "https://gitlab.com/o/maps/-/raw/dev/m/lanelet2_map.osm",
                "https://gitlab.com/o/maps",
                "dev",
                "m/lanelet2_map.osm",
            ),
            (
                "https://gitlab.com/o/maps/-/tree/main",
                "https://gitlab.com/o/maps",
                "main",
                "",
            ),
            # The '-' ends the repository, so a subgroup path is unambiguous.
            (
                "https://gitlab.com/group/sub/maps/-/tree/main/maps/T",
                "https://gitlab.com/group/sub/maps",
                "main",
                "maps/T",
            ),
            (
                "https://gitlab.com/group/sub/maps/-/tree/main/maps/T@v2",
                "https://gitlab.com/group/sub/maps",
                "v2",
                "maps/T",
            ),
            # GitHub's form is unchanged.
            (
                "https://github.com/o/maps/blob/main/m/lanelet2_map.osm",
                "https://github.com/o/maps",
                "main",
                "m/lanelet2_map.osm",
            ),
            # A self-hosted GitLab, subgroups included.
            (
                "https://git.example.com/g/s/maps/-/raw/main/m/lanelet2_map.osm",
                "https://git.example.com/g/s/maps",
                "main",
                "m/lanelet2_map.osm",
            ),
            # On GitHub and HuggingFace a '-' is a directory in the path.
            (
                "https://github.com/o/maps/tree/main/-/tree/x",
                "https://github.com/o/maps",
                "main",
                "-/tree/x",
            ),
            (
                "https://huggingface.co/datasets/o/maps/tree/main/-/raw/x",
                "https://huggingface.co/datasets/o/maps",
                "main",
                "-/raw/x",
            ),
            # A GitHub-style view ahead of the '-' wins on any host.
            (
                "https://git.example.com/o/maps/tree/main/-/x",
                "https://git.example.com/o/maps",
                "main",
                "-/x",
            ),
        ],
    )
    def test_parses_the_urls_people_actually_have(
        self, uri: str, repo_url: str, ref: str, path: str
    ) -> None:
        source = MapSource.parse(uri)
        assert (source.repo_url, source.ref, source.path) == (repo_url, ref, path)

    @pytest.mark.parametrize(
        ("uri", "repo_url", "ref", "path"),
        [
            # A HuggingFace PR ref is percent-encoded in the page URL (#1052).
            (
                "https://huggingface.co/datasets/o/maps/tree/refs%2Fpr%2F3/m/T",
                "https://huggingface.co/datasets/o/maps",
                "refs/pr/3",
                "m/T",
            ),
            # A space in the path.
            (
                "https://github.com/o/maps/tree/main/maps/My%20Town",
                "https://github.com/o/maps",
                "main",
                "maps/My Town",
            ),
            # An encoded '/' in a ref stays in the ref: the split comes first.
            (
                "https://github.com/o/maps/tree/feature%2Fx/maps/T",
                "https://github.com/o/maps",
                "feature/x",
                "maps/T",
            ),
            # GitLab page URLs are decoded the same way, split first.
            (
                "https://gitlab.com/g/sub/maps/-/tree/feature%2Fx/maps/My%20Town",
                "https://gitlab.com/g/sub/maps",
                "feature/x",
                "maps/My Town",
            ),
            # An already-decoded space is left as it is.
            (
                "https://github.com/o/maps/tree/main/maps/My Town",
                "https://github.com/o/maps",
                "main",
                "maps/My Town",
            ),
            # Plain GitHub and GitLab page URLs are unchanged.
            (
                "https://github.com/o/maps/tree/main/maps/T",
                "https://github.com/o/maps",
                "main",
                "maps/T",
            ),
            (
                "https://gitlab.com/o/maps/-/tree/main/maps/T",
                "https://gitlab.com/o/maps",
                "main",
                "maps/T",
            ),
        ],
    )
    def test_decodes_the_ref_and_path_of_a_page_url(
        self, uri: str, repo_url: str, ref: str, path: str
    ) -> None:
        source = MapSource.parse(uri)
        assert (source.repo_url, source.ref, source.path) == (repo_url, ref, path)
        assert MapSource.parse(source.uri) == source

    def test_an_explicit_ref_and_a_plain_url_are_not_decoded(self) -> None:
        source = MapSource.parse("https://example.com/o/a%20b@v%201#m%20T")
        assert (source.repo_url, source.ref, source.path) == (
            "https://example.com/o/a%20b",
            "v%201",
            "m%20T",
        )

    def test_canonical_form_round_trips(self) -> None:
        uri = "https://huggingface.co/datasets/O/N/tree/dev/maps/T"
        assert MapSource.parse(MapSource.parse(uri).uri) == MapSource.parse(uri)

    def test_an_explicit_ref_wins_over_the_one_in_a_page_url(self) -> None:
        source = MapSource.parse("https://github.com/o/n/tree/dev/maps/T@v9")
        assert source.ref == "v9"

    def test_userinfo_is_not_mistaken_for_a_ref(self) -> None:
        assert MapSource.parse("https://user@example.com/o/n").ref == ""

    @pytest.mark.parametrize("scheme", ["", "file://"])
    def test_an_at_in_an_existing_local_directory_is_not_a_ref(
        self, tmp_path: Path, scheme: str
    ) -> None:
        # Nix store paths and user@domain home directories are real: a local
        # path that exists is the repository, '@' and all (#671).
        repo = tmp_path / "maps@2" / "repo"
        repo.mkdir(parents=True)
        source = MapSource.parse(f"{scheme}{repo}#T")
        assert (source.repo_url, source.ref, source.path) == (
            f"{scheme}{repo}",
            "",
            "T",
        )

    @pytest.mark.parametrize("scheme", ["", "file://"])
    def test_a_local_ref_follows_the_last_at_that_leaves_a_directory(
        self, tmp_path: Path, scheme: str
    ) -> None:
        repo = tmp_path / "maps@2" / "repo"
        repo.mkdir(parents=True)
        source = MapSource.parse(f"{scheme}{repo}@feature/x#T")
        assert (source.repo_url, source.ref) == (f"{scheme}{repo}", "feature/x")
        assert MapSource.parse(source.uri) == source

    def test_a_missing_local_path_falls_back_to_the_last_at(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "maps@2" / "repo"
        source = MapSource.parse(f"file://{missing}")
        assert (source.repo_url, source.ref) == (f"file://{tmp_path}/maps", "2/repo")

    def test_a_percent_encoded_at_is_never_a_split(self, tmp_path: Path) -> None:
        source = MapSource.parse(f"file://{tmp_path}/maps%402/repo@main")
        assert (source.repo_url, source.ref) == (
            f"file://{tmp_path}/maps%402/repo",
            "main",
        )

    def test_a_ref_with_a_slash_round_trips(self) -> None:
        source = MapSource(repo_url="https://github.com/o/r", ref="feature/x", path="m")
        assert MapSource.parse(source.uri) == source

    def test_only_a_full_commit_counts_as_pinned(self) -> None:
        base = "https://example.com/o/n"
        assert not MapSource.parse(f"{base}@main").pinned
        assert not MapSource.parse(f"{base}@1a2b3c4").pinned
        assert MapSource.parse(f"{base}@{'a1' * 20}").pinned

    def test_a_repositoryless_uri_is_refused(self) -> None:
        with pytest.raises(MapSourceError):
            MapSource.parse("   ")
        with pytest.raises(MapSourceError):
            MapSource.parse("not a url")


# ---------------------------------------------------------------------------
# Where the cache lives
# ---------------------------------------------------------------------------


class TestMapRoot:
    """The default has to match what Autoware's own setup downloads into."""

    def test_defaults_under_the_autoware_data_directory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(CACHE_ROOT_ENV, raising=False)
        monkeypatch.setenv(AUTOWARE_DATA_DIR_ENV, "/somewhere/autoware_data")
        assert map_root() == Path("/somewhere/autoware_data/maps")

    def test_falls_back_to_the_conventional_home_directory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(CACHE_ROOT_ENV, raising=False)
        monkeypatch.delenv(AUTOWARE_DATA_DIR_ENV, raising=False)
        assert map_root() == (Path.home() / "autoware_data" / "maps").resolve()

    def test_an_explicit_override_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path))
        monkeypatch.setenv(AUTOWARE_DATA_DIR_ENV, "/ignored")
        assert map_root() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Cloning and caching
# ---------------------------------------------------------------------------


class TestResolveMap:
    """Turning a source into files."""

    def test_clones_the_named_map(self, origin_repo: Path, cache: GitMapCache) -> None:
        resolved = resolve_map(_uri(origin_repo), cache=cache)

        assert resolved.name == "TownA"
        assert resolved.lanelet2_path.read_text() == _OSM
        assert resolved.projector_info_path is not None
        assert resolved.commit == _head(origin_repo)
        assert not resolved.provisioned

    def test_reads_the_cache_without_the_origin(
        self, origin_repo: Path, cache: GitMapCache, tmp_path: Path
    ) -> None:
        # The cache is the point of the feature, so prove it is one: resolve
        # once, take the repository away, and resolve again.
        first = resolve_map(_uri(origin_repo), cache=cache)
        (origin_repo / ".git").rename(tmp_path / "moved-away")

        again = resolve_map(_uri(origin_repo), cache=cache)
        assert again.lanelet2_path == first.lanelet2_path
        assert again.lanelet2_path.read_text() == _OSM

    def test_a_repository_under_a_directory_with_an_at_resolves(
        self, origin_repo: Path, cache: GitMapCache, tmp_path: Path
    ) -> None:
        moved = tmp_path / "maps@2" / "origin"
        moved.parent.mkdir()
        origin_repo.rename(moved)
        resolved = resolve_map(_uri(moved), cache=cache)
        assert resolved.lanelet2_path.read_text() == _OSM
        assert resolved.commit == _head(moved)

    def test_a_map_the_repository_does_not_have_is_reported(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        with pytest.raises(MapResolutionError):
            resolve_map(_uri(origin_repo, "docs"), cache=cache)

    def test_a_source_pointing_at_the_osm_resolves_its_directory(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        resolved = resolve_map(
            _uri(origin_repo, "autoware_maps/TownA/lanelet2_map.osm"), cache=cache
        )
        # The whole directory comes with it, or the projection beside the map
        # would be missing.
        assert resolved.projector_info_path is not None
        assert resolved.name == "TownA"

    def test_an_unreachable_repository_is_reported(self, cache: GitMapCache) -> None:
        with pytest.raises(MapCacheError):
            resolve_map("git+file:///nonexistent/repo@main#maps/T", cache=cache)

    def test_a_shipped_opendrive_is_found_and_not_derived(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        resolved = resolve_map(_uri(origin_repo, "autoware_maps/TownB"), cache=cache)
        assert resolved.xodr_path is not None
        assert not resolved.xodr_is_derived


class TestOneEntryManyMaps:
    """One repository at one commit holds several maps (#769).

    They share a cache entry, so the checkout of one must not take another's
    files away: a ``ResolvedMap`` handed out earlier -- to the editor's parsed
    map cache, a running case, a sweep over both -- keeps pointing at files.
    """

    @pytest.fixture
    def rooted_repo(self, origin_repo: Path) -> Path:
        """The origin repository with a map at its root as well."""
        (origin_repo / "lanelet2_map.osm").write_text(_OSM)
        _git(origin_repo, "add", "-A")
        _git(origin_repo, "commit", "--quiet", "-m", "A map at the root")
        return origin_repo

    @staticmethod
    def _root_uri(repo: Path) -> str:
        return f"git+file://{repo}@main"

    def test_a_second_map_leaves_the_first_on_disk(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        town_a = resolve_map(_uri(origin_repo, "autoware_maps/TownA"), cache=cache)
        town_b = resolve_map(_uri(origin_repo, "autoware_maps/TownB"), cache=cache)

        assert town_a.directory.parent == town_b.directory.parent
        assert town_a.lanelet2_path.read_text() == _OSM
        assert town_a.projector_info_path is not None
        assert town_a.projector_info_path.is_file()
        assert town_b.lanelet2_path.read_text() == _OSM
        peeked = cache.peek(MapSource.parse(_uri(origin_repo, ref=town_a.commit)))
        assert peeked is not None
        assert peeked.materialised == ("autoware_maps/TownA", "autoware_maps/TownB")

    def test_the_root_after_a_directory_checks_the_whole_tree_out(
        self, rooted_repo: Path, cache: GitMapCache
    ) -> None:
        town_a = resolve_map(_uri(rooted_repo, "autoware_maps/TownA"), cache=cache)
        root = resolve_map(self._root_uri(rooted_repo), cache=cache)

        assert root.lanelet2_path.read_text() == _OSM
        assert root.directory / "autoware_maps" / "TownA" == town_a.directory
        assert town_a.lanelet2_path.read_text() == _OSM
        assert (root.directory / "docs" / "README.md").is_file()

    def test_a_directory_after_the_root_keeps_the_whole_tree(
        self, rooted_repo: Path, cache: GitMapCache
    ) -> None:
        root = resolve_map(self._root_uri(rooted_repo), cache=cache)
        town_b = resolve_map(_uri(rooted_repo, "autoware_maps/TownB"), cache=cache)

        assert root.lanelet2_path.read_text() == _OSM
        assert town_b.lanelet2_path.read_text() == _OSM
        assert (root.directory / "autoware_maps" / "TownA").is_dir()
        assert (root.directory / "docs" / "README.md").is_file()

    def test_going_back_to_the_first_map_is_a_cache_hit(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = resolve_map(_uri(origin_repo, "autoware_maps/TownA"), cache=cache)
        resolve_map(_uri(origin_repo, "autoware_maps/TownB"), cache=cache)

        calls = _git_calls(monkeypatch)
        again = resolve_map(_uri(origin_repo, "autoware_maps/TownA"), cache=cache)

        assert again.lanelet2_path == first.lanelet2_path
        assert again.lanelet2_path.read_text() == _OSM
        # The branch is asked where it is, and nothing is checked out again.
        assert calls == ["ls-remote"]

    def test_an_entry_recording_one_directory_gains_a_second(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        """An entry an older version left on disk: ``materialised`` a string."""
        town_a = resolve_map(_uri(origin_repo, "autoware_maps/TownA"), cache=cache)
        source = MapSource.parse(_uri(origin_repo, ref=town_a.commit))
        meta = cache.entry_dir(source) / "meta.json"
        document = json.loads(meta.read_text(encoding="utf-8"))
        document["materialised"] = "autoware_maps/TownA"
        meta.write_text(json.dumps(document, indent=2), encoding="utf-8")

        resolve_map(_uri(origin_repo, "autoware_maps/TownB"), cache=cache)

        assert town_a.lanelet2_path.read_text() == _OSM
        peeked = cache.peek(source)
        assert peeked is not None
        assert peeked.materialised == ("autoware_maps/TownA", "autoware_maps/TownB")

    def test_a_lost_record_does_not_trim_the_checkout(
        self, rooted_repo: Path, cache: GitMapCache
    ) -> None:
        """Another process's record lost in a race: git, not meta.json, decides."""
        root = resolve_map(self._root_uri(rooted_repo), cache=cache)
        source = MapSource.parse(self._root_uri(rooted_repo)).at(root.commit)
        meta = cache.entry_dir(source) / "meta.json"
        document = json.loads(meta.read_text(encoding="utf-8"))
        document["materialised"] = None
        meta.write_text(json.dumps(document, indent=2), encoding="utf-8")

        town_a = resolve_map(_uri(rooted_repo, "autoware_maps/TownA"), cache=cache)

        assert town_a.lanelet2_path.read_text() == _OSM
        assert root.lanelet2_path.read_text() == _OSM
        assert (root.directory / "docs" / "README.md").is_file()
        assert (root.directory / "autoware_maps" / "TownB").is_dir()

    def test_concurrent_resolves_of_two_maps_keep_both(
        self, rooted_repo: Path, cache: GitMapCache
    ) -> None:
        uris = [
            _uri(rooted_repo, "autoware_maps/TownA"),
            _uri(rooted_repo, "autoware_maps/TownB"),
        ] * 3
        with ThreadPoolExecutor(max_workers=len(uris)) as pool:
            resolved = list(pool.map(lambda uri: resolve_map(uri, cache=cache), uris))

        for each in resolved:
            assert each.lanelet2_path.read_text() == _OSM
        source = MapSource.parse(_uri(rooted_repo, ref=resolved[0].commit))
        peeked = cache.peek(source)
        assert peeked is not None
        assert peeked.materialised == ("autoware_maps/TownA", "autoware_maps/TownB")


class TestCachedMap:
    """What the editor asks on every render: is it here already?"""

    def test_says_nothing_before_anything_is_cached(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        assert cached_map(_uri(origin_repo), cache=cache) is None

    def test_describes_the_map_once_it_is_cached(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        resolve_map(_uri(origin_repo), cache=cache)
        assert cached_map(_uri(origin_repo), cache=cache) is not None

    def test_a_branch_is_found_under_the_commit_it_resolved_to(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The render path never asks the network, so it follows the note.

        A branch is filed under the commit it resolved to. Looking under the
        branch's own name would report a map that is on the machine as missing
        -- and this is what the editor's "downloaded" badge reads.
        """
        resolve_map(_uri(origin_repo), cache=cache)

        seen = _git_calls(monkeypatch)
        found = cached_map(_uri(origin_repo), cache=cache)

        assert found is not None
        assert found.commit == _head(origin_repo)
        assert seen == [], "the render path went to the network"

    def test_an_unpulled_lfs_pointer_does_not_count_as_cached(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        resolved = resolve_map(_uri(origin_repo), cache=cache)
        resolved.lanelet2_path.write_text(
            "version https://git-lfs.github.com/spec/v1\n" "oid sha256:0000\nsize 1\n"
        )
        assert is_lfs_pointer(resolved.lanelet2_path)
        assert cached_map(_uri(origin_repo), cache=cache) is None


class TestAMapThatIsNotAMap:
    """A checkout's contents are the repository's, and git carries symlinks.

    Pointing a scenario at a repository is not the same as trusting it, and the
    editor binds ``0.0.0.0`` with no authentication -- so a ``.osm`` that is a
    link to any readable file on the host has to be refused rather than read.
    """

    @pytest.fixture
    def hostile_repo(self, tmp_path: Path) -> Path:
        """A repository whose map is a symlink to a file outside it."""
        secret = tmp_path / "secret.txt"
        secret.write_text("CANARY")

        repo = tmp_path / "hostile"
        (repo / "autoware_maps" / "EvilTown").mkdir(parents=True)
        (repo / "autoware_maps" / "EvilTown" / "lanelet2_map.osm").symlink_to(secret)
        _git(repo, "init", "--quiet", "--initial-branch", "main", ".")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", "A map that is a link")
        return repo

    def test_the_resolver_refuses_it(
        self, hostile_repo: Path, cache: GitMapCache
    ) -> None:
        """Refused where every consumer arrives, not at each of them."""
        with pytest.raises(MapResolutionError):
            resolve_map(_uri(hostile_repo, "autoware_maps/EvilTown"), cache=cache)

    def test_the_editor_does_not_serve_it(
        self, editor: "TestClient", hostile_repo: Path
    ) -> None:
        """The route this protects: ``/draft/<id>/map.osm``.

        The map is never adopted, so the draft keeps the one it had -- what must
        not happen is the canary coming back down the wire.
        """
        draft_id = _new_draft(editor)
        editor.post(
            f"/draft/{draft_id}/map/use",
            data={"uri": _uri(hostile_repo, "autoware_maps/EvilTown")},
        )

        served = editor.get(f"/draft/{draft_id}/map.osm")
        assert "CANARY" not in served.text


#: How each file the resolver hands out is read back off a ResolvedMap.
_MAP_FILES = {
    "lanelet2_map.osm": "lanelet2_path",
    "TownL.xodr": "xodr_path",
    "map_projector_info.yaml": "projector_info_path",
}


class TestLinksInAMapDirectory:
    """Every file the resolver hands out is held to the ``.osm``'s rule (#760).

    A link that stays inside the map directory is followed; one that leaves it,
    dangles, or names a directory is not a map file.  The ``.xodr`` matters
    most: ``map.overwrite_xodr`` copies it into the CARLA installation.
    """

    @pytest.fixture
    def outside(self, tmp_path: Path) -> Path:
        """A directory outside every repository, holding a canary and a map."""
        directory = tmp_path / "outside"
        directory.mkdir()
        (directory / "secret.txt").write_text("CANARY")
        (directory / "lanelet2_map.osm").write_text(_OSM)
        return directory

    def _repo(self, tmp_path: Path, build: Any) -> Path:
        """A repository with ``maps/TownL``, laid out by *build*."""
        repo = tmp_path / "linked"
        directory = repo / "maps" / "TownL"
        directory.mkdir(parents=True)
        build(directory)
        _git(repo, "init", "--quiet", "--initial-branch", "main", ".")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", "A map with links")
        return repo

    def _resolve_with(
        self, tmp_path: Path, cache: GitMapCache, name: str, target: Any
    ) -> Any:
        """Resolve a map whose *name* is a link to ``target(directory)``.

        Every other map file is a real one, so only *name* is in question.
        """

        def build(directory: Path) -> None:
            for other in _MAP_FILES:
                if other != name:
                    (directory / other).write_text(_OSM)
            (directory / name).symlink_to(target(directory))

        repo = self._repo(tmp_path, build)
        return resolve_map(_uri(repo, "maps/TownL"), cache=cache)

    @pytest.mark.parametrize("name", sorted(_MAP_FILES))
    def test_a_link_inside_the_directory_is_followed(
        self, tmp_path: Path, cache: GitMapCache, name: str
    ) -> None:
        def inside(directory: Path) -> Path:
            (directory / "real").mkdir()
            (directory / "real" / "file").write_text(_OSM)
            return Path("real") / "file"

        resolved = self._resolve_with(tmp_path, cache, name, inside)
        found = getattr(resolved, _MAP_FILES[name])
        assert found == resolved.directory / name
        assert found.read_text() == _OSM

    @pytest.mark.parametrize("name", sorted(_MAP_FILES))
    @pytest.mark.parametrize("kind", ["outside", "dangling", "directory"])
    def test_a_link_that_is_not_a_map_file_is_refused(
        self,
        tmp_path: Path,
        cache: GitMapCache,
        outside: Path,
        name: str,
        kind: str,
    ) -> None:
        targets = {
            "outside": lambda directory: outside / "secret.txt",
            "dangling": lambda directory: directory / "missing",
            "directory": lambda directory: outside,
        }
        if name.endswith(".osm"):
            # The only .osm there: refusing it leaves no map at all.
            with pytest.raises(MapResolutionError):
                self._resolve_with(tmp_path, cache, name, targets[kind])
            return
        resolved = self._resolve_with(tmp_path, cache, name, targets[kind])
        assert getattr(resolved, _MAP_FILES[name]) is None

    @pytest.mark.parametrize("name", ["TownL.xodr", "map_projector_info.yaml"])
    def test_a_link_that_loops_is_ignored(
        self, tmp_path: Path, cache: GitMapCache, name: str
    ) -> None:
        """Resolving a loop raises RuntimeError before Python 3.13."""
        resolved = self._resolve_with(
            tmp_path, cache, name, lambda directory: directory / name
        )
        assert getattr(resolved, _MAP_FILES[name]) is None

    def test_a_derived_xodr_linked_out_of_the_cache_is_refused(
        self, origin_repo: Path, cache: GitMapCache, outside: Path
    ) -> None:
        derived = resolve_map(_uri(origin_repo), cache=cache).derived
        derived.mkdir(parents=True, exist_ok=True)
        (derived / "TownA.xodr").symlink_to(outside / "secret.txt")

        resolved = resolve_map(_uri(origin_repo), cache=cache)
        assert resolved.xodr_path is None

    def test_a_derived_xodr_is_still_found(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        derived = resolve_map(_uri(origin_repo), cache=cache).derived
        derived.mkdir(parents=True, exist_ok=True)
        (derived / "TownA.xodr").write_text("<OpenDRIVE/>")

        resolved = resolve_map(_uri(origin_repo), cache=cache)
        assert resolved.xodr_path == derived / "TownA.xodr"
        assert resolved.xodr_is_derived

    def test_a_stray_xodr_in_the_derived_directory_is_ignored(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        """Only the file :meth:`ResolvedMap.derived_xodr` names is the map's (#1050)."""
        derived = resolve_map(_uri(origin_repo), cache=cache).derived
        derived.mkdir(parents=True, exist_ok=True)
        (derived / "Town01.xodr").write_text("<OpenDRIVE town='Town01'/>")

        resolved = resolve_map(_uri(origin_repo), cache=cache)
        assert resolved.xodr_path is None
        assert not resolved.xodr_is_derived
        assert resolved.cached_derived_xodr("Town01") == derived / "Town01.xodr"

    def test_a_map_directory_linked_out_of_the_checkout_is_refused(
        self, tmp_path: Path, cache: GitMapCache, outside: Path
    ) -> None:
        """Otherwise every file check passes against the link's target."""
        repo = self._repo(tmp_path, lambda d: (d / "lanelet2_map.osm").write_text(_OSM))
        (repo / "maps" / "Evil").symlink_to(outside, target_is_directory=True)
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", "A map directory that is a link")

        with pytest.raises(MapResolutionError, match="outside the repository"):
            resolve_map(_uri(repo, "maps/Evil"), cache=cache)
        assert cached_map(_uri(repo, "maps/Evil"), cache=cache) is None

    @pytest.mark.parametrize(
        "path", ["../outside", "../../../../outside", "maps/../../outside"]
    )
    def test_a_path_that_climbs_out_is_refused(
        self, origin_repo: Path, cache: GitMapCache, outside: Path, path: str
    ) -> None:
        """Joined onto the checkout, the map root and the derived directory."""
        # ``../outside`` from the map root is where ``outside`` really is, and
        # it holds an .osm: without the check it is taken as a provisioned map.
        with pytest.raises(MapResolutionError, match=r"\.\."):
            resolve_map(_uri(origin_repo, path), cache=cache)
        assert cached_map(_uri(origin_repo, path), cache=cache) is None

    def test_a_local_directory_is_held_to_the_same_rule(
        self, tmp_path: Path, outside: Path
    ) -> None:
        """As its ``.osm`` already was: discovery is shared with a checkout."""
        directory = tmp_path / "local"
        directory.mkdir()
        (directory / "lanelet2_map.osm").write_text(_OSM)
        (directory / "TownL.xodr").symlink_to(outside / "secret.txt")
        (directory / "map_projector_info.yaml").symlink_to(outside / "secret.txt")

        resolved = resolve_directory(directory, cache=GitMapCache(tmp_path / "root"))
        assert resolved.xodr_path is None
        assert resolved.projector_info_path is None


class TestProvisionedMaps:
    """A map Autoware's own setup already downloaded is a map we have."""

    def _provision(self, cache: GitMapCache, path: str) -> Path:
        directory = cache.root / path
        directory.mkdir(parents=True)
        (directory / "lanelet2_map.osm").write_text(_OSM)
        return directory

    def test_is_used_without_cloning_anything(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        directory = self._provision(cache, "autoware_maps/TownA")
        (origin_repo / ".git").rename(origin_repo.parent / "gone")

        resolved = resolve_map(_uri(origin_repo), cache=cache)
        assert resolved.provisioned
        assert resolved.lanelet2_path == directory / "lanelet2_map.osm"

    def test_a_pinned_source_ignores_it(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        # An unpacked directory records no revision, so honouring it for a
        # pinned source would answer a question about one commit with the bytes
        # of an unknown other.
        self._provision(cache, "autoware_maps/TownA")

        resolved = resolve_map(_uri(origin_repo, ref=_head(origin_repo)), cache=cache)
        assert not resolved.provisioned
        assert resolved.commit == _head(origin_repo)


class TestPinning:
    """Fixing a scenario's map so two runs are two runs on the same map."""

    def test_replaces_a_branch_with_the_commit_behind_it(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        pinned = pin_source(_uri(origin_repo), cache=cache)
        assert MapSource.parse(pinned).ref == _head(origin_repo)
        assert MapSource.parse(pinned).path == "autoware_maps/TownA"

    def test_is_idempotent(self, origin_repo: Path, cache: GitMapCache) -> None:
        pinned = pin_source(_uri(origin_repo), cache=cache)
        assert pin_source(pinned, cache=cache) == pinned

    def test_a_pin_keeps_resolving_after_the_branch_moves(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        pinned = pin_source(_uri(origin_repo), cache=cache)
        first = _head(origin_repo)

        (origin_repo / "autoware_maps" / "TownA" / "lanelet2_map.osm").write_text(
            _OSM.replace('lat="0.0"', 'lat="1.0"')
        )
        _git(origin_repo, "commit", "--quiet", "-am", "Move the map")

        assert resolve_map(pinned, cache=cache, refresh=True).commit == first
        assert resolve_map(_uri(origin_repo), cache=cache, refresh=True).commit != first


class TestAtTip:
    """Asking the remote which commit a ref names, exactly (#1044)."""

    @pytest.fixture
    def remote(self, origin_repo: Path, tmp_path: Path) -> Path:
        """A bare remote whose refs share names and tails.

        ``archive/main`` and the annotated ``v1.0`` are at the first commit,
        ``main`` has moved on; ``dup`` is both a tag (first commit) and a
        branch (second commit), and ``light`` a lightweight tag.
        """
        _git(origin_repo, "branch", "archive/main")
        _git(origin_repo, "tag", "-a", "v1.0", "-m", "Release")
        _git(origin_repo, "tag", "light")
        _git(origin_repo, "tag", "-a", "dup", "-m", "Tag named like a branch")
        _git(origin_repo, "commit", "--quiet", "--allow-empty", "-m", "Second")
        _git(origin_repo, "branch", "dup")
        bare = tmp_path / "remote.git"
        _git(tmp_path, "clone", "--quiet", "--bare", str(origin_repo), str(bare))
        return bare

    @staticmethod
    def _rev(repo: Path, rev: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", rev],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _tip(self, cache: GitMapCache, remote: Path, ref: str) -> str:
        return cache.at_tip(MapSource(repo_url=f"file://{remote}", ref=ref)).ref

    def test_a_branch_is_not_another_ending_in_its_name(
        self, remote: Path, cache: GitMapCache
    ) -> None:
        assert self._rev(remote, "archive/main") != self._rev(remote, "main")
        assert self._tip(cache, remote, "main") == self._rev(remote, "refs/heads/main")

    def test_an_annotated_tag_is_peeled_to_its_commit(
        self, remote: Path, cache: GitMapCache
    ) -> None:
        tip = self._tip(cache, remote, "v1.0")
        assert tip == self._rev(remote, "v1.0^{commit}")
        assert tip != self._rev(remote, "refs/tags/v1.0")

    def test_a_lightweight_tag(self, remote: Path, cache: GitMapCache) -> None:
        assert self._tip(cache, remote, "light") == self._rev(remote, "light")

    def test_a_tag_wins_over_a_branch_of_the_same_name_as_in_git(
        self, remote: Path, cache: GitMapCache, tmp_path: Path
    ) -> None:
        tip = self._tip(cache, remote, "dup")
        assert tip == self._rev(remote, "refs/tags/dup^{commit}")
        assert tip != self._rev(remote, "refs/heads/dup")
        # The same commit the checkout's `git fetch origin <ref>` gets.
        scratch = tmp_path / "scratch"
        _git(tmp_path, "init", "--quiet", str(scratch))
        _git(scratch, "fetch", "--quiet", str(remote), "dup")
        assert tip == self._rev(scratch, "FETCH_HEAD^{commit}")

    def test_a_full_ref_names_itself(self, remote: Path, cache: GitMapCache) -> None:
        assert self._tip(cache, remote, "refs/heads/dup") == self._rev(
            remote, "refs/heads/dup"
        )

    def test_head(self, remote: Path, cache: GitMapCache) -> None:
        assert self._tip(cache, remote, "HEAD") == self._rev(remote, "HEAD")

    def test_a_ref_only_matching_a_tail_is_not_found(
        self, remote: Path, cache: GitMapCache
    ) -> None:
        source = MapSource(repo_url=f"file://{remote}", ref="archive")
        assert cache.at_tip(source) == source
        tail = MapSource(repo_url=f"file://{remote}", ref="ain")
        assert cache.at_tip(tail) == tail
        # `only` is the tail of `refs/heads/nested/only`, which ls-remote's
        # pattern matches but git's own resolution (and the fetch) does not.
        _git(remote, "branch", "nested/only", "main")
        nested = MapSource(repo_url=f"file://{remote}", ref="only")
        assert cache.at_tip(nested) == nested


# ---------------------------------------------------------------------------
# Browsing
# ---------------------------------------------------------------------------


class TestListMaps:
    """A map is a directory with a Lanelet2 file in it."""

    def test_lists_every_map_and_nothing_else(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        entries = list_maps(f"git+file://{origin_repo}@main", cache=cache)
        assert [entry.name for entry in entries] == ["TownA", "TownB"]
        assert [entry.has_xodr for entry in entries] == [False, True]
        assert entries[0].has_projector_info

    def test_an_entry_names_the_map_it_found(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        entry = list_maps(f"git+file://{origin_repo}@main", cache=cache)[0]
        assert resolve_map(entry.uri, cache=cache).name == "TownA"

    def test_a_path_narrows_the_search(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        entries = list_maps(
            f"git+file://{origin_repo}@main#autoware_maps/TownB", cache=cache
        )
        assert [entry.name for entry in entries] == ["TownB"]


class TestListMapsAtTip:
    """Listing a branch lists what it names now, from resolving's entry (#1051)."""

    @pytest.fixture
    def remote(self, origin_repo: Path, tmp_path: Path) -> Path:
        """A bare clone of the map repository, pushed to like a real remote."""
        bare = tmp_path / "remote.git"
        _git(tmp_path, "clone", "--quiet", "--bare", str(origin_repo), str(bare))
        return bare

    @staticmethod
    def _push_town_c(remote: Path, tmp_path: Path) -> None:
        work = tmp_path / "work"
        _git(tmp_path, "clone", "--quiet", str(remote), str(work))
        (work / "autoware_maps" / "TownC").mkdir()
        (work / "autoware_maps" / "TownC" / "lanelet2_map.osm").write_text(_OSM)
        _git(work, "add", "-A")
        _git(work, "commit", "--quiet", "-m", "Publish TownC")
        _git(work, "push", "--quiet", "origin", "main")

    @staticmethod
    def _entries(cache: GitMapCache) -> list[str]:
        return sorted(
            path.name
            for path in (cache.root / ".repos").iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    def test_a_new_commit_on_the_branch_is_listed(
        self, remote: Path, cache: GitMapCache, tmp_path: Path
    ) -> None:
        uri = f"git+file://{remote}@main"
        assert [e.name for e in list_maps(uri, cache=cache)] == ["TownA", "TownB"]

        self._push_town_c(remote, tmp_path)

        names = [e.name for e in list_maps(uri, cache=cache)]
        assert names == ["TownA", "TownB", "TownC"]

    def test_listing_and_resolving_share_one_entry(
        self, remote: Path, cache: GitMapCache
    ) -> None:
        uri = f"git+file://{remote}@main"
        entry = list_maps(uri, cache=cache)[0]
        listed = self._entries(cache)
        resolved = resolve_map(entry.uri, cache=cache)

        assert len(listed) == 1
        assert self._entries(cache) == listed
        assert resolved.lanelet2_path.is_relative_to(cache.root / ".repos" / listed[0])

    def test_an_unreachable_remote_still_lists_what_was_cached(
        self, remote: Path, cache: GitMapCache, tmp_path: Path
    ) -> None:
        uri = f"git+file://{remote}@main"
        first = [e.name for e in list_maps(uri, cache=cache)]
        remote.rename(tmp_path / "moved-away")

        assert [e.name for e in list_maps(uri, cache=cache)] == first

    def test_an_unreachable_remote_lists_an_entry_filed_under_the_branch(
        self, remote: Path, cache: GitMapCache, tmp_path: Path
    ) -> None:
        # A cache filled before listing asked for the tip: the entry is named
        # after the branch and no tip was ever recorded.
        uri = f"git+file://{remote}@main"
        cache.list_files(MapSource.parse(uri))
        remote.rename(tmp_path / "moved-away")

        assert [e.name for e in list_maps(uri, cache=cache)] == ["TownA", "TownB"]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestResolveMapPaths:
    """What the sweeper, the runner and the editor all ask of a ``map`` group."""

    def test_a_config_naming_nothing_resolves_to_nothing(self) -> None:
        paths = resolve_map_paths({"name": "TownA"})
        assert paths.lanelet2_path is None and paths.xodr_path is None

    def test_explicit_paths_are_used_as_given(self) -> None:
        paths = resolve_map_paths(
            {"lanelet2_path": "a.osm", "xodr_path": "a.xodr", "name": "TownA"}
        )
        assert paths.lanelet2_path == Path("a.osm")
        assert paths.install_xodr == Path("a.xodr")

    def test_a_source_supplies_the_paths_and_the_name(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
        paths = resolve_map_paths({"source": _uri(origin_repo)})
        assert paths.name == "TownA"
        assert paths.lanelet2_path is not None

    def test_an_explicit_path_overrides_the_source(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
        paths = resolve_map_paths(
            {"source": _uri(origin_repo), "lanelet2_path": "/tmp/other.osm"}
        )
        assert paths.lanelet2_path == Path("/tmp/other.osm")

    def test_an_uncached_source_stays_unfetched_when_asked_to(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
        paths = resolve_map_paths({"source": _uri(origin_repo)}, allow_fetch=False)
        assert paths.lanelet2_path is None

    def test_a_shipped_opendrive_is_installed_a_derived_one_is_not(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))

        shipped = resolve_map_paths(
            {"source": _uri(origin_repo, "autoware_maps/TownB")}
        )
        assert shipped.install_xodr is not None

        # TownA has none, so a run has to write one from the loaded world.
        town_a = resolve_map_paths({"source": _uri(origin_repo)})
        assert town_a.install_xodr is None
        assert town_a.opendrive_path is not None
        assert town_a.opendrive_path.name == "TownA.xodr"

    def test_an_opendrive_captured_under_the_carla_map_name_is_found(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run on ``map.name`` writes ``<map.name>.xodr``; the next read finds it."""
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
        config = {"source": _uri(origin_repo), "name": "Town01"}
        destination = resolve_map_paths(config).opendrive_path
        assert destination is not None and destination.name == "Town01.xodr"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("<OpenDRIVE/>")

        again = resolve_map_paths(config)
        assert again.xodr_path == destination
        assert again.xodr_is_derived and again.install_xodr is None

    def test_an_opendrive_captured_under_another_name_is_not_used(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The directory's ``TownA.xodr`` is not the town ``map.name`` loads (#1050)."""
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
        default = resolve_map_paths({"source": _uri(origin_repo)}).opendrive_path
        assert default is not None and default.name == "TownA.xodr"
        default.parent.mkdir(parents=True, exist_ok=True)
        default.write_text("<OpenDRIVE town='TownA'/>")

        renamed = resolve_map_paths({"source": _uri(origin_repo), "name": "Town01"})
        assert renamed.xodr_path is None and not renamed.xodr_is_derived
        assert renamed.opendrive_path == default.parent / "Town01.xodr"

    def test_the_opendrive_is_not_installed_into_carla_unless_asked(self) -> None:
        config = {"lanelet2_path": "a.osm", "xodr_path": "a.xodr", "name": "TownA"}

        assert resolve_map_paths(config).overwrite_xodr is False
        assert resolve_map_paths({**config, "overwrite_xodr": True}).overwrite_xodr
        off = resolve_map_paths({**config, "overwrite_xodr": False})
        assert off.overwrite_xodr is False


def _map_repo(path: Path) -> Path:
    """A repository holding one map at ``map/``, as many single-map repos do."""
    (path / "map").mkdir(parents=True)
    (path / "map" / "lanelet2_map.osm").write_text(_OSM)
    _git(path, "init", "--quiet", "--initial-branch", "main", ".")
    _git(path, "add", "-A")
    _git(path, "commit", "--quiet", "-m", "Publish the map")
    return path


class TestDerivedDirectory:
    """Where an OpenDRIVE taken from CARLA is kept, per repository (#1050)."""

    def test_two_repositories_at_the_same_path_do_not_share_one(
        self, tmp_path: Path, cache: GitMapCache
    ) -> None:
        first = _map_repo(tmp_path / "town01_map")
        second = _map_repo(tmp_path / "town10_map")

        a = resolve_map(_uri(first, "map"), cache=cache)
        captured = a.derived_xodr()
        captured.parent.mkdir(parents=True, exist_ok=True)
        captured.write_text("<OpenDRIVE town='Town01'/>")

        b = resolve_map(_uri(second, "map"), cache=cache)
        assert a.derived != b.derived
        assert b.xodr_path is None and not b.xodr_is_derived
        # The first repository still finds its own.
        assert resolve_map(_uri(first, "map"), cache=cache).xodr_path == captured

    def test_the_ref_is_not_part_of_the_key(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        """The OpenDRIVE comes out of CARLA, so a new commit need not refetch it."""
        on_branch = resolve_map(_uri(origin_repo), cache=cache)
        pinned = resolve_map(_uri(origin_repo, ref=on_branch.commit), cache=cache)
        assert on_branch.derived == pinned.derived

    def test_two_maps_of_one_repository_do_not_share_one(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        town_a = cache.derived_dir(MapSource.parse(_uri(origin_repo)))
        town_b = cache.derived_dir(
            MapSource.parse(_uri(origin_repo, "autoware_maps/TownB"))
        )
        assert town_a != town_b
        # Both under the one repository's directory.
        assert town_a.parent == town_b.parent


# ---------------------------------------------------------------------------
# A local map directory
# ---------------------------------------------------------------------------


@pytest.fixture
def local_map(tmp_path: Path) -> Path:
    """A map directory on this machine, in no git repository."""
    directory = tmp_path / "somewhere" / "TownL"
    directory.mkdir(parents=True)
    (directory / "lanelet2_map.osm").write_text(_OSM)
    (directory / "map_projector_info.yaml").write_text("projector_type: Local\n")
    (directory / "pointcloud_map.pcd").write_text("not read\n")
    return directory


class TestResolveDirectory:
    """``map.directory``: a map read in place, with no repository behind it."""

    def test_finds_the_map_files_the_way_a_checkout_is_searched(
        self, local_map: Path, cache: GitMapCache
    ) -> None:
        resolved = resolve_directory(local_map, cache=cache)

        assert resolved.local and resolved.source is None
        assert resolved.name == "TownL"
        assert resolved.lanelet2_path == local_map / "lanelet2_map.osm"
        assert resolved.projector_info_path == local_map / "map_projector_info.yaml"
        assert resolved.xodr_path is None
        assert resolved.commit == "" and not resolved.provisioned

    def test_runs_no_git_command(
        self, local_map: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _git_calls(monkeypatch)
        resolve_directory(local_map, cache=cache)
        assert calls == []

    def test_a_shipped_opendrive_is_found_and_is_not_derived(
        self, local_map: Path, cache: GitMapCache
    ) -> None:
        (local_map / "TownL.xodr").write_text("<OpenDRIVE/>")
        resolved = resolve_directory(local_map, cache=cache)
        assert resolved.xodr_path == local_map / "TownL.xodr"
        assert not resolved.xodr_is_derived

    def test_accepts_the_osm_itself(self, local_map: Path, cache: GitMapCache) -> None:
        (local_map / "other.osm").write_text(_OSM)
        resolved = resolve_directory(local_map / "other.osm", cache=cache)
        assert resolved.directory == local_map
        assert resolved.lanelet2_path == local_map / "other.osm"

    def test_a_named_osm_that_is_missing_is_an_error_not_a_fallback(
        self, local_map: Path, cache: GitMapCache
    ) -> None:
        with pytest.raises(MapResolutionError, match="wrong.osm does not exist"):
            resolve_directory(local_map / "wrong.osm", cache=cache)

    def test_expands_the_home_directory(
        self, local_map: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(local_map.parent))
        resolved = resolve_directory("~/TownL", cache=cache)
        assert resolved.lanelet2_path == local_map / "lanelet2_map.osm"

    def test_a_missing_directory_is_reported(
        self, tmp_path: Path, cache: GitMapCache
    ) -> None:
        with pytest.raises(MapResolutionError, match="does not exist"):
            resolve_directory(tmp_path / "nowhere", cache=cache)

    def test_a_directory_with_no_map_is_reported(
        self, tmp_path: Path, cache: GitMapCache
    ) -> None:
        with pytest.raises(MapResolutionError, match=r"no Lanelet2 \(\.osm\)"):
            resolve_directory(tmp_path, cache=cache)

    def test_an_unpulled_lfs_pointer_is_reported(
        self, local_map: Path, cache: GitMapCache
    ) -> None:
        (local_map / "lanelet2_map.osm").write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:" + "0" * 64 + "\nsize 123\n"
        )
        with pytest.raises(MapResolutionError, match="git-lfs pointer"):
            resolve_directory(local_map, cache=cache)

    def test_a_derived_opendrive_goes_to_the_map_root_not_the_directory(
        self,
        local_map: Path,
        cache: GitMapCache,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        asset = tmp_path / "TownL.xodr"
        asset.write_text("<OpenDRIVE/>")
        monkeypatch.setenv(map_asset_env_var("TownL"), str(asset))

        written = ensure_xodr(
            resolve_directory(local_map, cache=cache), allow_server=False
        )

        assert cache.root in written.parents
        assert not any(local_map.glob("*.xodr"))
        # Read back on the next resolve, as a derived file.
        again = resolve_directory(local_map, cache=cache)
        assert again.xodr_path == written and again.xodr_is_derived

    def test_two_directories_with_one_name_do_not_share_an_opendrive(
        self, local_map: Path, cache: GitMapCache, tmp_path: Path
    ) -> None:
        twin = tmp_path / "elsewhere" / "TownL"
        twin.mkdir(parents=True)
        (twin / "lanelet2_map.osm").write_text(_OSM)
        assert (
            resolve_directory(local_map, cache=cache).derived
            != resolve_directory(twin, cache=cache).derived
        )

    def test_a_config_names_it_as_map_directory(
        self, local_map: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
        paths = resolve_map_paths({"directory": str(local_map)})

        assert paths.name == "TownL"
        assert paths.lanelet2_path == local_map / "lanelet2_map.osm"
        assert paths.resolved is not None and paths.resolved.local
        assert not paths.from_source
        # No OpenDRIVE shipped, so a run writes one from the loaded world.
        assert paths.install_xodr is None
        assert paths.opendrive_path is not None
        assert cache.root in paths.opendrive_path.parents

    def test_it_wins_over_a_source_and_fetches_nothing(
        self,
        local_map: Path,
        origin_repo: Path,
        cache: GitMapCache,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
        calls = _git_calls(monkeypatch)
        paths = resolve_map_paths(
            {"source": _uri(origin_repo), "directory": str(local_map)}
        )
        assert paths.lanelet2_path == local_map / "lanelet2_map.osm"
        assert calls == []

    def test_explicit_paths_still_override_it(
        self, local_map: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
        paths = resolve_map_paths(
            {
                "directory": str(local_map),
                "lanelet2_path": "/tmp/other.osm",
                "xodr_path": "/tmp/other.xodr",
                "name": "Town10HD_Opt",
            }
        )
        assert paths.name == "Town10HD_Opt"
        assert paths.lanelet2_path == Path("/tmp/other.osm")
        assert paths.install_xodr == Path("/tmp/other.xodr")

    def test_an_unusable_directory_is_an_error_for_a_run(
        self, tmp_path: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
        with pytest.raises(MapResolutionError):
            resolve_map_paths({"directory": str(tmp_path / "nowhere")})

    def test_the_editor_stores_it_and_exports_it(
        self,
        editor: "TestClient",
        drafts: "DraftStore",
        local_map: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from autoware_carla_scenario.authoring.hydra_config import (
            _map_overrides,
        )
        from autoware_carla_scenario.editor import map_preview

        monkeypatch.setenv(map_preview.MAP_ROOTS_ENV, str(local_map.parent))
        draft_id = _new_draft(editor)
        editor.post(
            f"/draft/{draft_id}/scenario", data={"map_directory": str(local_map)}
        )

        document = _stored(drafts, draft_id)
        assert document.map.directory == str(local_map)
        # The starter's Nishishinjuku files are superseded, as they are by a
        # source, and the CARLA map name is the directory's.
        assert document.map.lanelet2_path is None
        assert document.map.xodr_path is None
        assert document.map.name == "TownL"
        assert map_preview.lanelet2_source(document) == (local_map / "lanelet2_map.osm")
        overrides = _map_overrides(document)
        assert overrides["directory"] == str(local_map)
        # Its own map, so the selected group's exclusion list is not inherited.
        assert overrides["no_3d_model_lanelet_ids"] == []

    def test_a_file_typed_alongside_it_in_the_editor_is_kept(
        self, editor: "TestClient", drafts: "DraftStore", local_map: Path
    ) -> None:
        draft_id = _new_draft(editor)
        editor.post(
            f"/draft/{draft_id}/scenario",
            data={
                "map_directory": str(local_map),
                "map_xodr_path": "/maps/TownL.xodr",
                "map_name": "Town10HD_Opt",
            },
        )
        document = _stored(drafts, draft_id)
        assert document.map.xodr_path == "/maps/TownL.xodr"
        assert document.map.name == "Town10HD_Opt"
        assert document.map.lanelet2_path is None

    def test_choosing_a_library_map_clears_the_directory(
        self,
        editor: "TestClient",
        drafts: "DraftStore",
        local_map: Path,
        origin_repo: Path,
    ) -> None:
        draft_id = _new_draft(editor)
        editor.post(
            f"/draft/{draft_id}/scenario", data={"map_directory": str(local_map)}
        )
        editor.post(f"/draft/{draft_id}/map/use", data={"uri": _uri(origin_repo)})

        document = _stored(drafts, draft_id)
        assert document.map.directory is None
        assert resolve_map_paths(document.map).name == "TownA"
        assert document.map.name == "TownA"

    def test_opendrive_is_fetched_for_the_directory_in_use_not_the_source(
        self,
        editor: "TestClient",
        drafts: "DraftStore",
        local_map: Path,
        origin_repo: Path,
        cache: GitMapCache,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        asset = tmp_path / "TownL.xodr"
        asset.write_text("<OpenDRIVE/>")
        monkeypatch.setenv(map_asset_env_var("TownL"), str(asset))

        draft_id = _new_draft(editor)
        editor.post(f"/draft/{draft_id}/map/use", data={"uri": _uri(origin_repo)})
        editor.post(
            f"/draft/{draft_id}/scenario", data={"map_directory": str(local_map)}
        )
        editor.post(
            f"/draft/{draft_id}/map/opendrive",
            data={"carla_host": "127.0.0.1", "carla_port": "1"},
        )

        local = resolve_directory(local_map, cache=cache)
        assert local.xodr_path is not None and local.xodr_is_derived
        # Nothing filed under the source map, which would pick it up later.
        source_map = resolve_map(_uri(origin_repo), cache=cache)
        assert source_map.xodr_path is None
        # And the local map keeps its own name through a re-check of the source.
        editor.post(f"/draft/{draft_id}/map/fetch")
        assert _stored(drafts, draft_id).map.name == "TownL"


# ---------------------------------------------------------------------------
# OpenDRIVE
# ---------------------------------------------------------------------------


class TestOpenDrive:
    """The one part of a map that comes from CARLA rather than the repository."""

    def test_derives_the_environment_variable_autoware_uses(self) -> None:
        assert map_asset_env_var("NishishinjukuMap") == "NISHISHINJUKU_MAP_PATH"
        assert map_asset_env_var("Town01") == "TOWN01_PATH"
        assert map_asset_env_var("Town10HD_Opt") == "TOWN10_HD_OPT_PATH"

    def test_an_unset_variable_names_no_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(map_asset_env_var("TownA"), raising=False)
        assert installed_xodr("TownA") is None

    def test_a_blank_variable_names_no_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(map_asset_env_var("TownA"), "   ")
        assert map_asset_path("TownA") is None
        assert installed_xodr("TownA") is None

    def test_the_variable_is_stripped_and_its_tilde_expanded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue #1031: every reader of the variable resolves it the same way."""
        monkeypatch.setenv("HOME", str(tmp_path))
        asset = tmp_path / "TownA.xodr"
        asset.write_text("<OpenDRIVE/>")
        monkeypatch.setenv(map_asset_env_var("TownA"), " ~/TownA.xodr \n")
        assert map_asset_path("TownA") == asset
        assert installed_xodr("TownA") == asset

    def test_takes_the_file_from_a_carla_installation_when_there_is_one(
        self,
        origin_repo: Path,
        cache: GitMapCache,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        asset = tmp_path / "TownA.xodr"
        asset.write_text("<OpenDRIVE/>")
        monkeypatch.setenv(map_asset_env_var("TownA"), str(asset))

        resolved = resolve_map(_uri(origin_repo), cache=cache)
        written = ensure_xodr(resolved, allow_server=False)

        assert written.read_text() == "<OpenDRIVE/>"
        # In the cache, not in the checkout: generating one must not dirty the
        # working tree it was generated beside.
        assert written.parent == resolved.derived

    def test_a_cached_file_is_reused(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        resolved = resolve_map(_uri(origin_repo, "autoware_maps/TownB"), cache=cache)
        assert ensure_xodr(resolved, allow_server=False) == resolved.xodr_path

    def test_another_carla_maps_cached_file_is_not_reused(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One captured under the directory's name is not ``map_name``'s (#1050)."""
        resolved = resolve_map(_uri(origin_repo), cache=cache)
        resolved.derived.mkdir(parents=True, exist_ok=True)
        resolved.derived_xodr().write_text("<OpenDRIVE town='TownA'/>")
        resolved = resolve_map(_uri(origin_repo), cache=cache)
        assert resolved.xodr_is_derived

        monkeypatch.delenv(map_asset_env_var("Town01"), raising=False)
        with pytest.raises(OpenDriveUnavailable, match=map_asset_env_var("Town01")):
            ensure_xodr(resolved, map_name="Town01", allow_server=False)

        resolved.derived_xodr("Town01").write_text("<OpenDRIVE town='Town01'/>")
        assert ensure_xodr(
            resolved, map_name="Town01", allow_server=False
        ) == resolved.derived_xodr("Town01")

    def test_says_what_is_missing_when_there_is_nowhere_to_get_one(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(map_asset_env_var("TownA"), raising=False)
        resolved = resolve_map(_uri(origin_repo), cache=cache)
        with pytest.raises(OpenDriveUnavailable, match=map_asset_env_var("TownA")):
            ensure_xodr(resolved, allow_server=False)


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------


@pytest.fixture
def drafts(tmp_path: Path) -> "DraftStore":
    """Where the editor under test keeps its drafts."""
    from autoware_carla_scenario.authoring.persistence import DraftStore

    return DraftStore(tmp_path / "drafts")


@pytest.fixture
def editor(
    drafts: "DraftStore", cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
) -> "TestClient":
    """A Scenario Editor whose map cache is the one this test owns."""
    from fastapi.testclient import TestClient

    from autoware_carla_scenario.editor import map_preview
    from autoware_carla_scenario.editor.app import create_app

    monkeypatch.setenv(CACHE_ROOT_ENV, str(cache.root))
    map_preview.clear_cache()
    return TestClient(create_app(draft_dir=drafts.root))


def _new_draft(client: "TestClient") -> str:
    """Create a blank draft and return its id."""
    response = client.post(
        "/new", data={"title": "Map test", "kind": "blank"}, follow_redirects=False
    )
    return str(response.headers["location"]).rsplit("/", 1)[-1]


def _stored(store: "DraftStore", draft_id: str) -> Any:
    """Return the stored document, failing the test when it is gone."""
    draft = store.get(draft_id)
    assert draft is not None, f"draft {draft_id} is gone"
    return draft.document


class TestEditorMapSource:
    """Naming a map repository from the Scenario inspector."""

    def test_a_pasted_page_url_is_stored_canonically(
        self, editor: "TestClient", drafts: "DraftStore", origin_repo: Path
    ) -> None:
        draft_id = _new_draft(editor)
        editor.post(
            f"/draft/{draft_id}/scenario",
            data={"map_source": f"file://{origin_repo}@main#autoware_maps/TownA"},
        )
        assert _stored(drafts, draft_id).map.source == _uri(origin_repo)

    def test_an_unusable_source_is_refused_and_not_stored(
        self, editor: "TestClient", drafts: "DraftStore"
    ) -> None:
        draft_id = _new_draft(editor)
        response = editor.post(
            f"/draft/{draft_id}/scenario", data={"map_source": "not a url"}
        )
        assert "not a repository URL" in response.text
        assert _stored(drafts, draft_id).map.source is None

    def test_choosing_a_map_downloads_it_and_names_it(
        self, editor: "TestClient", drafts: "DraftStore", origin_repo: Path
    ) -> None:
        draft_id = _new_draft(editor)
        response = editor.post(
            f"/draft/{draft_id}/map/use", data={"uri": _uri(origin_repo)}
        )

        assert "Now editing on TownA" in response.text
        stored = _stored(drafts, draft_id).map
        assert stored.name == "TownA"
        # A source supersedes the starter's own file paths, or the document
        # would keep resolving to the map it was created with.
        assert stored.lanelet2_path is None and stored.xodr_path is None

    def test_pinning_fixes_the_revision_and_keeps_the_map(
        self, editor: "TestClient", drafts: "DraftStore", origin_repo: Path
    ) -> None:
        from autoware_carla_scenario.editor import map_preview

        draft_id = _new_draft(editor)
        editor.post(f"/draft/{draft_id}/map/use", data={"uri": _uri(origin_repo)})
        editor.post(f"/draft/{draft_id}/map/pin")

        document = _stored(drafts, draft_id)
        assert MapSource.parse(document.map.source).ref == _head(origin_repo)

        status = map_preview.map_status(document)
        assert status.pinned and status.cached

    def test_the_map_is_servable_to_the_viewer_from_the_cache(
        self, editor: "TestClient", origin_repo: Path
    ) -> None:
        draft_id = _new_draft(editor)
        editor.post(f"/draft/{draft_id}/map/use", data={"uri": _uri(origin_repo)})

        response = editor.get(f"/draft/{draft_id}/map.osm")
        assert response.status_code == 200
        assert response.text == _OSM

    def test_the_places_panel_draws_the_map_it_came_from(
        self, editor: "TestClient", origin_repo: Path
    ) -> None:
        """The panel asks the *resolved* map, not the field a source empties.

        Choosing a map from the library clears ``map.lanelet2_path`` -- the
        source supersedes it -- so a panel that asked that field whether there
        was a map drew nothing for every scenario naming a repository, while
        ``/map.osm`` beside it served the same map perfectly well.
        """
        draft_id = _new_draft(editor)
        editor.post(f"/draft/{draft_id}/map/use", data={"uri": _uri(origin_repo)})

        body = editor.get(f"/draft/{draft_id}/map-view").text
        assert 'data-viewer-key="scenario-map"' in body
        assert "No map" not in body

    def test_a_map_not_downloaded_yet_says_so_rather_than_asking_for_a_path(
        self, editor: "TestClient", origin_repo: Path
    ) -> None:
        """Telling that person to type a path is telling them to undo the source."""
        draft_id = _new_draft(editor)
        editor.post(
            f"/draft/{draft_id}/scenario",
            data={"map_source": _uri(origin_repo), "map_lanelet2_path": ""},
        )

        body = editor.get(f"/draft/{draft_id}/map-view").text
        assert "No map" in body
        assert "has not been downloaded yet" in body

    def test_the_exported_config_carries_the_source(
        self, editor: "TestClient", drafts: "DraftStore", origin_repo: Path
    ) -> None:
        from autoware_carla_scenario.authoring.hydra_config import (
            build_scenario_config,
        )

        draft_id = _new_draft(editor)
        editor.post(f"/draft/{draft_id}/map/use", data={"uri": _uri(origin_repo)})

        config = build_scenario_config(_stored(drafts, draft_id))
        assert config["map"]["source"] == _uri(origin_repo)
        assert config["map"]["name"] == "TownA"
        # Written out even though it is empty: the `map` group a run selects
        # describes another map, whose excluded lanelet ids mean nothing here.
        assert config["map"]["no_3d_model_lanelet_ids"] == []


class TestEditorMapLibrary:
    """Browsing what a repository publishes, from the editor."""

    def test_lists_the_maps_in_a_repository(
        self, editor: "TestClient", origin_repo: Path
    ) -> None:
        response = editor.get("/maps", params={"repo": f"file://{origin_repo}@main"})
        assert "TownA" in response.text and "TownB" in response.text

    def test_an_ambiguous_directory_is_offered_as_the_file_it_advertises(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        """The library must not offer a map the resolver then refuses.

        A directory with several `.osm` and none named the way Autoware names
        one is ambiguous. The entry already chose which file to show, so the
        source names it -- otherwise picking the advertised map fails.
        """
        town = origin_repo / "autoware_maps" / "TownC"
        town.mkdir()
        (town / "alpha.osm").write_text(_OSM)
        (town / "beta.osm").write_text(_OSM)
        _git(origin_repo, "add", "-A")
        _git(origin_repo, "commit", "--quiet", "-m", "A directory with two maps")

        entry = next(
            e
            for e in list_maps(f"file://{origin_repo}@main", cache=cache)
            if e.path == "autoware_maps/TownC"
        )
        assert entry.lanelet2_name == "alpha.osm"
        # The advertised entry resolves rather than raising "several ... and
        # none is named lanelet2_map.osm".
        assert resolve_map(entry.uri, cache=cache).lanelet2_path.name == "alpha.osm"

    def test_offers_the_repositories_it_knows_about(self, editor: "TestClient") -> None:
        assert "carla-ue5-maps" in editor.get("/maps").text

    def test_reports_a_repository_it_cannot_read(self, editor: "TestClient") -> None:
        response = editor.get("/maps", params={"repo": "file:///nowhere@main"})
        assert response.status_code == 200
        assert "could not be read" in response.text

    def test_starts_a_scenario_on_a_chosen_map(
        self, editor: "TestClient", drafts: "DraftStore", origin_repo: Path
    ) -> None:
        response = editor.post(
            "/maps/new", data={"uri": _uri(origin_repo)}, follow_redirects=False
        )
        draft_id = str(response.headers["location"]).rsplit("/", 1)[-1]
        assert _stored(drafts, draft_id).map.name == "TownA"


class TestEditorWithoutOpenDrive:
    """A scenario is written against Lanelet2, so nothing here waits on CARLA."""

    def test_the_map_counts_as_ready_with_no_opendrive_anywhere(
        self,
        editor: "TestClient",
        drafts: "DraftStore",
        origin_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from autoware_carla_scenario.editor import map_preview

        monkeypatch.delenv(map_asset_env_var("TownA"), raising=False)
        draft_id = _new_draft(editor)
        editor.post(f"/draft/{draft_id}/map/use", data={"uri": _uri(origin_repo)})

        status = map_preview.map_status(_stored(drafts, draft_id))
        assert status.xodr_path is None
        assert status.needs_opendrive
        # Cached is the whole bar: the preview and a sweep read Lanelet2.
        assert status.cached

    def test_fetching_opendrive_without_carla_says_why(
        self, editor: "TestClient", origin_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(map_asset_env_var("TownA"), raising=False)
        draft_id = _new_draft(editor)
        editor.post(f"/draft/{draft_id}/map/use", data={"uri": _uri(origin_repo)})

        response = editor.post(
            f"/draft/{draft_id}/map/opendrive",
            data={"carla_host": "127.0.0.1", "carla_port": "1"},
        )
        assert response.status_code == 200
        assert "not applied" in response.text


class TestCacheEconomy:
    """What the cache is for: not doing the work again."""

    def test_a_warm_pinned_resolve_runs_no_git_at_all(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A commit cannot have moved, so nothing is worth asking anyone."""
        pinned = _uri(origin_repo, ref=_head(origin_repo))
        resolve_map(pinned, cache=cache)

        calls = _git_calls(monkeypatch)
        again = resolve_map(pinned, cache=cache)

        assert again.lanelet2_path.read_text() == _OSM
        assert calls == []

    def test_a_warm_branch_resolve_costs_one_ls_remote_and_nothing_else(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A branch can have moved, so it is asked -- and only asked.

        The question is one ``ls-remote``, and the answer being the commit
        already checked out is what stops everything after it: no fetch, no
        clone, no re-checkout of a working tree that has not moved.
        """
        resolve_map(_uri(origin_repo), cache=cache)

        calls = _git_calls(monkeypatch)
        again = resolve_map(_uri(origin_repo), cache=cache)

        assert again.lanelet2_path.read_text() == _OSM
        assert calls == ["ls-remote"]

    def test_a_branch_that_moved_is_resolved_to_the_new_commit(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        """The point of asking: the ref means what it says.

        And the new commit lands at a *new path*, which is what anything
        downstream keyed on the file -- the editor's parsed-map cache above all
        -- needs in order to notice at all.
        """
        first = resolve_map(_uri(origin_repo), cache=cache)

        (origin_repo / "autoware_maps" / "TownA" / "lanelet2_map.osm").write_text(
            _OSM.replace("<osm", "<!--moved--><osm", 1)
        )
        _git(origin_repo, "add", "-A")
        _git(origin_repo, "commit", "--quiet", "-m", "Move the branch on")

        again = resolve_map(_uri(origin_repo), cache=cache)
        assert again.commit == _head(origin_repo) != first.commit
        assert again.lanelet2_path != first.lanelet2_path
        assert "<!--moved-->" in again.lanelet2_path.read_text()

    def test_an_unreachable_remote_falls_back_to_the_commit_it_last_named(
        self, origin_repo: Path, cache: GitMapCache, tmp_path: Path
    ) -> None:
        """Asking is best-effort: a machine with no route still runs.

        The fallback is the *note*, not the branch: the checkout is filed under
        the commit, so handing the branch back would look under a name nothing
        was ever filed under and try to clone from the remote that just failed.
        """
        first = resolve_map(_uri(origin_repo), cache=cache)
        (origin_repo / ".git").rename(tmp_path / "moved-away")

        again = resolve_map(_uri(origin_repo), cache=cache)
        assert again.lanelet2_path == first.lanelet2_path
        assert again.commit == first.commit


def test_a_captured_opendrive_replaces_the_file_whole(tmp_path: Path) -> None:
    """Tasks of a SLURM batch capture one map at once: no half-written file."""
    from types import SimpleNamespace

    world = SimpleNamespace(
        get_map=lambda: SimpleNamespace(to_opendrive=lambda: "<OpenDRIVE/>")
    )
    destination = tmp_path / "maps" / "Town10HD_Opt.xodr"
    destination.parent.mkdir()
    destination.write_text("<old/>", encoding="utf-8")

    assert capture_opendrive(world, destination) == destination  # type: ignore[arg-type]
    assert destination.read_text(encoding="utf-8") == "<OpenDRIVE/>"
    assert [p.name for p in destination.parent.iterdir()] == [destination.name]


def test_a_captured_opendrive_takes_a_scratch_name_of_its_own(tmp_path: Path) -> None:
    """The scratch file is unique whichever host and pid: one another process
    left (here, in the way as a directory) is not the one written."""
    import os
    import stat
    from types import SimpleNamespace

    destination = tmp_path / "Town10HD_Opt.xodr"
    (tmp_path / f".{destination.name}.{os.getpid()}.tmp").mkdir()
    world = SimpleNamespace(
        get_map=lambda: SimpleNamespace(to_opendrive=lambda: "<OpenDRIVE/>")
    )
    assert capture_opendrive(world, destination) == destination  # type: ignore[arg-type]
    assert destination.read_text(encoding="utf-8") == "<OpenDRIVE/>"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [destination.name, f".{destination.name}.{os.getpid()}.tmp"]
    )

    def broken() -> str:
        raise RuntimeError("lost the server")

    failing = SimpleNamespace(get_map=lambda: SimpleNamespace(to_opendrive=broken))
    with pytest.raises(RuntimeError, match="lost the server"):
        capture_opendrive(failing, destination)  # type: ignore[arg-type]
    # Nothing half written is left, and the old file stands.
    assert destination.read_text(encoding="utf-8") == "<OpenDRIVE/>"
    assert len(list(tmp_path.iterdir())) == 2
