"""CARLA maps from content packs (``map.pack``, maps/pack.py).

carla-pack-registry is replaced by a fake here: what is tested is how the
framework asks it for a pack's map files and has it mount the pack, not the
registry itself (which has its own suite, against mock GitHub and S3).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario import ScenarioQueue, ScenarioRunner
from autoware_carla_scenario.examples import run
from autoware_carla_scenario.maps import pack as pack_module
from autoware_carla_scenario.maps import resolve_map_paths
from autoware_carla_scenario.maps.pack import (
    PACKS_DIR_ENV,
    MapPack,
    MapPackError,
    packs_dir,
)

SHA = "a" * 64
REF = "acme/carla-packs/TownPack"


class FakeCarlaPackError(Exception):
    """carla-pack's error: args = (message, code)."""


class FakeClient:
    """A carla_pack_registry.Client serving one pack whose data is in *root*."""

    def __init__(self, root: Path, formats: dict[str, dict[str, Any]], maps: list[str]):
        self.root = root
        self._formats = formats
        self.maps = maps
        self.calls: list[tuple[Any, ...]] = []

    def _release(self) -> dict[str, Any]:
        return {
            "ref": f"{REF}@1.2.0",
            "sha256": SHA,
            "name": "TownPack",
            "maps": self.maps,
        }

    def resolve(
        self, reference: str, sha256: Optional[str] = None, **_: Any
    ) -> dict[str, Any]:
        self.calls.append(("resolve", reference, sha256))
        if not reference.startswith(REF):
            raise FakeCarlaPackError(f"{reference}: not found", "not_found")
        return self._release()

    def find_map(
        self, map_name: str, repos: list[str], sha256: Optional[str] = None, **_: Any
    ) -> dict[str, Any]:
        self.calls.append(("find_map", map_name, tuple(repos), sha256))
        return self._release()

    def formats(self, release: dict[str, Any]) -> dict[str, dict[str, Any]]:
        self.calls.append(("formats", release["ref"]))
        return self._formats

    def map_data(self, release: dict[str, Any], map_name: str, fmt: str) -> Path:
        self.calls.append(("map_data", map_name, fmt))
        return self.root / str(self._formats[map_name][fmt]["path"])


def _registry(client: FakeClient, **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(
        Client=lambda: client, CarlaPackError=FakeCarlaPackError, **extra
    )


def _town(
    tmp_path: Path, *, autoware: bool = True, lanelet2: bool = True
) -> FakeClient:
    """A pack carrying Town with the given formats, unpacked in *tmp_path*."""
    formats: dict[str, Any] = {
        "opendrive": {"type": "file", "path": "Content/Town.xodr", "origin": "carla"}
    }
    (tmp_path / "Content").mkdir(parents=True)
    (tmp_path / "Content" / "Town.xodr").write_text("<OpenDRIVE/>")
    if autoware:
        folder = tmp_path / "autoware" / "Town"
        folder.mkdir(parents=True)
        (folder / "lanelet2_map.osm").write_text("<osm/>")
        formats["autoware"] = {
            "type": "folder",
            "path": "autoware/Town",
            "origin": "asset",
        }
    if lanelet2:
        (tmp_path / "lanelet2").mkdir()
        (tmp_path / "lanelet2" / "Town.osm").write_text("<osm/>")
        formats["lanelet2"] = {
            "type": "file",
            "path": "lanelet2/Town.osm",
            "origin": "asset",
        }
    return FakeClient(tmp_path, {"Town": formats}, ["/TownPack/Maps/Town"])


@pytest.fixture
def town(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = _town(tmp_path / "pack")
    monkeypatch.setattr(pack_module, "_registry", lambda: _registry(client))
    return client


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reference", ["acme/packs", "acme/packs/TownPack", "acme/packs/TownPack@1.2.0-rc.1"]
)
def test_a_pack_or_a_repository_is_a_reference(reference: str) -> None:
    assert MapPack(reference).is_repository == (reference.count("/") == 1)


@pytest.mark.parametrize(
    "reference", ["TownPack", "acme/packs/TownPack/extra", "https://github.com/a/b", ""]
)
def test_a_malformed_reference_is_refused(reference: str) -> None:
    with pytest.raises(ValueError, match="map.pack"):
        MapPack(reference)


def test_a_malformed_sha256_is_refused() -> None:
    with pytest.raises(ValueError, match="map.pack_sha256"):
        MapPack(REF, sha256="ABC")


def test_the_packs_dir_is_configured_then_the_environment_then_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PACKS_DIR_ENV, raising=False)
    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_HOME", str(tmp_path / "home"))
    assert packs_dir() == tmp_path / "home" / "packs"
    monkeypatch.setenv(PACKS_DIR_ENV, str(tmp_path / "env"))
    assert packs_dir() == tmp_path / "env"
    assert packs_dir(str(tmp_path / "cfg")) == tmp_path / "cfg"


def test_without_carla_pack_the_error_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "carla_pack_registry", None)
    with pytest.raises(MapPackError, match=r"autoware-carla-scenario\[pack\]"):
        MapPack(REF).files("Town")


# ---------------------------------------------------------------------------
# The map files
# ---------------------------------------------------------------------------


def test_the_autoware_map_and_the_opendrive_are_read(town: FakeClient) -> None:
    files = MapPack(REF).files("Town")
    assert files.map_name == "Town"
    assert files.directory == town.root / "autoware" / "Town"
    assert files.xodr_path == town.root / "Content" / "Town.xodr"
    assert (files.reference, files.sha256) == (f"{REF}@1.2.0", SHA)
    assert ("map_data", "Town", "lanelet2") not in town.calls


def test_a_bare_lanelet2_map_is_read_without_an_autoware_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _town(tmp_path, autoware=False)
    monkeypatch.setattr(pack_module, "_registry", lambda: _registry(client))
    assert MapPack(REF).files("Town").directory == tmp_path / "lanelet2" / "Town.osm"


def test_a_pack_without_a_lanelet2_map_still_gives_its_opendrive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _town(tmp_path, autoware=False, lanelet2=False)
    monkeypatch.setattr(pack_module, "_registry", lambda: _registry(client))
    files = MapPack(REF).files("Town")
    assert files.directory is None
    assert files.xodr_path == tmp_path / "Content" / "Town.xodr"


def test_the_map_is_found_by_its_short_name(town: FakeClient) -> None:
    assert MapPack(REF).files("/Game/Maps/Town").map_name == "Town"


def test_the_only_map_of_a_pack_needs_no_name(town: FakeClient) -> None:
    assert MapPack(REF).files("").map_name == "Town"


def test_a_map_the_pack_does_not_carry_is_refused(town: FakeClient) -> None:
    with pytest.raises(MapPackError, match="does not carry the map Odaiba"):
        MapPack(REF).files("Odaiba")


def test_a_repository_is_searched_for_the_map(town: FakeClient) -> None:
    MapPack("acme/carla-packs", sha256=SHA).files("Town")
    assert ("find_map", "Town", ("acme/carla-packs",), SHA) in town.calls


def test_a_repository_needs_the_map_name(town: FakeClient) -> None:
    with pytest.raises(MapPackError, match="set map.name"):
        MapPack("acme/carla-packs").files("")


def test_a_pack_not_found_says_how_to_get_access(town: FakeClient) -> None:
    with pytest.raises(MapPackError, match="gh auth login"):
        MapPack("acme/carla-packs/Other").files("Town")


# ---------------------------------------------------------------------------
# resolve_map_paths
# ---------------------------------------------------------------------------


def test_a_pack_gives_the_run_its_lanelet2_map_and_opendrive(town: FakeClient) -> None:
    paths = resolve_map_paths({"pack": REF})
    assert paths.name == "Town"
    assert paths.lanelet2_path == town.root / "autoware" / "Town" / "lanelet2_map.osm"
    assert paths.xodr_path == town.root / "Content" / "Town.xodr"
    assert not paths.xodr_is_derived and not paths.overwrite_xodr
    # Pinned to the artifact the files came from.
    assert (paths.pack, paths.pack_sha256) == (f"{REF}@1.2.0", SHA)


def test_a_pack_wins_over_a_source(town: FakeClient) -> None:
    paths = resolve_map_paths(
        {
            "name": "Town",
            "pack": REF,
            "source": "git+https://example.com/maps@main#Town",
        }
    )
    assert paths.lanelet2_path == town.root / "autoware" / "Town" / "lanelet2_map.osm"


def test_a_directory_and_explicit_files_win_over_the_pack(
    town: FakeClient, tmp_path: Path
) -> None:
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "lanelet2_map.osm").write_text("<osm/>")
    xodr = tmp_path / "patched.xodr"
    xodr.write_text("<OpenDRIVE/>")
    paths = resolve_map_paths({"name": "Town", "pack": REF, "directory": str(mine)})
    assert paths.lanelet2_path == mine / "lanelet2_map.osm"
    assert paths.xodr_path == town.root / "Content" / "Town.xodr"
    paths = resolve_map_paths({"name": "Town", "pack": REF, "xodr_path": str(xodr)})
    assert paths.xodr_path == xodr
    assert paths.pack == f"{REF}@1.2.0"


def test_without_fetching_only_the_reference_is_kept(town: FakeClient) -> None:
    paths = resolve_map_paths(
        {"name": "Town", "pack": REF, "pack_sha256": SHA}, allow_fetch=False
    )
    assert (paths.pack, paths.pack_sha256) == (REF, SHA)
    assert paths.lanelet2_path is None
    assert town.calls == []


# ---------------------------------------------------------------------------
# Mounting it
# ---------------------------------------------------------------------------


class _FakeCarlaClient:
    def __init__(self, host: str, port: int) -> None:
        self.address = (host, port)
        self.timeout: Optional[float] = None

    def set_timeout(self, seconds: float) -> None:
        self.timeout = seconds


def _deploying(
    monkeypatch: pytest.MonkeyPatch, **registry: Any
) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr("typesafe_carla.carla.Client", _FakeCarlaClient)
    namespace = SimpleNamespace(CarlaPackError=FakeCarlaPackError, **registry)
    monkeypatch.setattr(pack_module, "_registry", lambda: namespace)
    return calls


def test_a_pack_is_installed_and_mounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    def deploy_pack(ref: str, target: str, *, carla_client: Any, sha256: Any) -> Any:
        seen.append((ref, target, carla_client.address, carla_client.timeout, sha256))
        return {"name": "TownPack", "mounted": True, "maps": ["Town"], "path": target}

    _deploying(monkeypatch, deploy_pack=deploy_pack)
    target = tmp_path / "packs"
    MapPack(REF, sha256=SHA, packs_dir=target).deploy(
        "carla", 2000, "Town", timeout=9.0
    )
    assert seen == [(REF, str(target), ("carla", 2000), 9.0, SHA)]
    assert target.is_dir()


def test_a_pack_the_server_did_not_mount_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _deploying(
        monkeypatch,
        deploy_pack=lambda *a, **k: {"name": "TownPack", "mounted": False, "path": "x"},
    )
    with pytest.raises(MapPackError, match="did not mount"):
        MapPack(REF, packs_dir=tmp_path).deploy("localhost", 2000, "Town")


def test_a_repository_mounts_nothing_when_the_server_has_the_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    def ensure_map(client: Any, map_name: str, target: str, **kw: Any) -> None:
        seen.append((map_name, target, kw))
        return None

    _deploying(monkeypatch, ensure_map=ensure_map)
    MapPack("acme/carla-packs", packs_dir=tmp_path).deploy("localhost", 2000, "Town")
    assert seen == [
        ("Town", str(tmp_path), {"repos": ["acme/carla-packs"], "sha256": None})
    ]


def test_a_failed_deploy_is_a_map_pack_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deploy_pack(*_: Any, **__: Any) -> Any:
        raise FakeCarlaPackError("checksum mismatch", "checksum_mismatch")

    _deploying(monkeypatch, deploy_pack=deploy_pack)
    with pytest.raises(MapPackError, match="checksum mismatch"):
        MapPack(REF, packs_dir=tmp_path).deploy("localhost", 2000, "Town")


def test_the_queue_mounts_the_pack_before_it_loads_the_map() -> None:
    order = MagicMock()
    map_pack = MagicMock(spec=MapPack)
    order.attach_mock(map_pack.deploy, "deploy")
    with (
        patch("autoware_carla_scenario.scenario_queue.CarlaServerManager") as server,
        patch("autoware_carla_scenario.scenario_queue.ScenarioRunner") as runner_cls,
    ):
        server.return_value = MagicMock(timeout=120.0)
        runner = MagicMock(spec=ScenarioRunner)
        order.attach_mock(runner.load_map_by_name, "load_map_by_name")
        runner_cls.return_value = runner
        queue = ScenarioQueue(map_name="Town", map_pack=map_pack)
        queue.start()
    assert [c[0] for c in order.mock_calls] == ["deploy", "load_map_by_name"]
    assert map_pack.deploy.call_args.args == ("localhost", 2000, "Town")


# ---------------------------------------------------------------------------
# The runner's config
# ---------------------------------------------------------------------------


def test_the_run_builds_the_pack_the_files_came_from(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {"map": {"pack": REF}, "server": {"packs_dir": str(tmp_path)}}
    )
    paths = SimpleNamespace(pack=f"{REF}@1.2.0", pack_sha256=SHA)
    assert run.build_map_pack(cfg, paths) == MapPack(
        f"{REF}@1.2.0", sha256=SHA, packs_dir=tmp_path
    )
    assert run.build_map_pack(cfg, SimpleNamespace(pack=None)) is None


def test_the_trajectory_log_names_the_pack_as_the_map_source() -> None:
    assert (
        run._map_source(OmegaConf.create({"map": {"pack": REF}})) == f"carla-pack:{REF}"
    )
    assert (
        run._map_source(
            OmegaConf.create({"map": {"pack": REF, "directory": "/maps/T"}})
        )
        == "/maps/T"
    )


def test_a_batch_must_share_its_pack() -> None:
    a = OmegaConf.create({"map": {"name": "Town", "pack": REF}})
    b = OmegaConf.create({"map": {"name": "Town", "pack": f"{REF}@2.0.0"}})
    with pytest.raises(run.RunConfigError, match="map.pack"):
        run.check_batch_map([a, b])


def test_the_real_carla_pack_takes_the_arguments_the_fakes_do() -> None:
    """The fakes above stand in for carla-pack-registry: its API must take
    what maps/pack.py passes them."""
    import inspect

    registry = pytest.importorskip("carla_pack_registry")
    for method, names in {
        "resolve": ("reference", "sha256"),
        "find_map": ("map_name", "repos", "sha256"),
        "formats": ("release",),
        "map_data": ("release", "map_name", "format"),
    }.items():
        signature = getattr(registry.Client, method).__text_signature__
        assert all(name in signature for name in names), (method, signature)
    assert {"carla_client", "sha256"} <= set(
        inspect.signature(registry.deploy_pack).parameters
    )
    assert {"repos", "sha256"} <= set(inspect.signature(registry.ensure_map).parameters)
    assert issubclass(registry.CarlaPackError, Exception)
