"""HD maps that live in a git repository.

Everything here runs against a repository created in ``tmp_path``, so the tests
say nothing about HuggingFace being reachable and never wait on it.  That is
also what lets them assert the thing that matters most about the cache: after
the first checkout, the map resolves with the *origin deleted*, which no
mock-based test could establish.
"""

from __future__ import annotations

import os
import subprocess
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
    ensure_xodr,
    installed_xodr,
    map_asset_env_var,
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
        ],
    )
    def test_parses_the_urls_people_actually_have(
        self, uri: str, repo_url: str, ref: str, path: str
    ) -> None:
        source = MapSource.parse(uri)
        assert (source.repo_url, source.ref, source.path) == (repo_url, ref, path)

    def test_canonical_form_round_trips(self) -> None:
        uri = "https://huggingface.co/datasets/O/N/tree/dev/maps/T"
        assert MapSource.parse(MapSource.parse(uri).uri) == MapSource.parse(uri)

    def test_an_explicit_ref_wins_over_the_one_in_a_page_url(self) -> None:
        source = MapSource.parse("https://github.com/o/n/tree/dev/maps/T@v9")
        assert source.ref == "v9"

    def test_userinfo_is_not_mistaken_for_a_ref(self) -> None:
        assert MapSource.parse("https://user@example.com/o/n").ref == ""

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

    def test_an_unpulled_lfs_pointer_does_not_count_as_cached(
        self, origin_repo: Path, cache: GitMapCache
    ) -> None:
        resolved = resolve_map(_uri(origin_repo), cache=cache)
        resolved.lanelet2_path.write_text(
            "version https://git-lfs.github.com/spec/v1\n" "oid sha256:0000\nsize 1\n"
        )
        assert is_lfs_pointer(resolved.lanelet2_path)
        assert cached_map(_uri(origin_repo), cache=cache) is None


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

    def _git_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Record the git subcommand of every git invocation the cache makes."""
        from autoware_carla_scenario.maps import cache as cache_module

        seen: list[str] = []
        original = cache_module.GitMapCache._git

        def _spy(self: Any, cwd: Any, *args: str, **kwargs: Any) -> str:
            seen.append(args[0])
            return str(original(self, cwd, *args, **kwargs))

        monkeypatch.setattr(cache_module.GitMapCache, "_git", _spy)
        return seen

    def test_a_warm_resolve_runs_no_git_at_all(
        self, origin_repo: Path, cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolve_map(_uri(origin_repo), cache=cache)

        calls = self._git_calls(monkeypatch)
        again = resolve_map(_uri(origin_repo), cache=cache)

        assert again.lanelet2_path.read_text() == _OSM
        assert calls == []
