"""Top-level test configuration for autoware_carla_scenario."""

from __future__ import annotations

import os
from collections.abc import Iterator

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
