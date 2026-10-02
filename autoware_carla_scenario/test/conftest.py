"""Top-level test configuration for autoware_carla_scenario."""

from __future__ import annotations

import os


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
