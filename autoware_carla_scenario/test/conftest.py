"""Top-level test configuration for autoware_carla_scenario."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest


# git exports the variables that locate a repository (`git rev-parse
# --local-env-vars`) to its hooks, and the pre-commit hooks run this suite: left
# in place they point every `git` the tests run in a scratch directory at the
# repository being committed to -- writing into its index and config.
for _var in (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_INTERNAL_SUPER_PREFIX",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
):
    os.environ.pop(_var, None)


#: The repository root, wherever pytest was started from.
REPO_ROOT = Path(__file__).resolve().parents[2]
#: The shared fixture maps (the Nishishinjuku ``.osm`` and ``.xodr``).
DATA_DIR = REPO_ROOT / "data"


#: Runs only the K-th of N shards of the selected tests, given as ``K/N``
#: (CI's Codon legs). A test's shard is a stable hash of its node id, the same
#: in every process, xdist worker and run, so the N shards of one selection
#: together run each of its tests exactly once. An environment variable rather
#: than an option: the option of a conftest below the root is unknown to a run
#: that names no path, and every xdist worker inherits the variable.
TEST_SHARD_ENV = "AUTOWARE_CARLA_SCENARIO_TEST_SHARD"


def _shard_of(nodeid: str, shards: int) -> int:
    """The 1-based shard of *nodeid*."""
    digest = hashlib.sha256(nodeid.encode()).digest()
    return int.from_bytes(digest[:8], "big") % shards + 1


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    value = os.environ.get(TEST_SHARD_ENV, "").strip()
    if not value:
        return
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", value)
    if match is None or int(match[1]) > int(match[2]):
        raise pytest.UsageError(
            f"{TEST_SHARD_ENV}: expected K/N with 1 <= K <= N, not {value!r}"
        )
    shard, shards = int(match[1]), int(match[2])
    kept = [item for item in items if _shard_of(item.nodeid, shards) == shard]
    if len(kept) < len(items):
        config.hook.pytest_deselected(
            items=[item for item in items if _shard_of(item.nodeid, shards) != shard]
        )
        items[:] = kept


@pytest.fixture
def at_repo_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the test with the repository root as the working directory.

    The example configs and the editor's starter documents name their maps
    relative to it (``data/nishishinjuku.osm``), as the CLIs are run from
    there; a test that goes through them must not depend on where pytest was
    started (#849).
    """
    monkeypatch.chdir(REPO_ROOT)
    return REPO_ROOT


@pytest.fixture(scope="session", autouse=True)
def _map_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep what the suite derives from maps out of the user's own map cache.

    One directory for every xdist worker of a run (the parent of the workers'
    base temporary directories), so the road network one worker builds is the
    one the others load; entries are written atomically.
    """
    from autoware_carla_scenario.maps.cache import CACHE_ROOT_ENV  # noqa: PLC0415

    root = tmp_path_factory.getbasetemp().parent / "map_cache"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ROOT_ENV, str(root))
        yield


@pytest.fixture(scope="session", autouse=True)
def _typecheck_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Share the Codon verdicts between the xdist workers of a run.

    A directory already given (CI restores one from its previous runs) is kept.
    """
    from autoware_carla_scenario.typecheck.check import CACHE_DIR_ENV  # noqa: PLC0415

    if os.environ.get(CACHE_DIR_ENV):
        yield
        return
    root = tmp_path_factory.getbasetemp().parent / "typecheck_cache"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_DIR_ENV, str(root))
        yield
