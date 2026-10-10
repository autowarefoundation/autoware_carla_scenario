"""autoware_stack - start a fresh Autoware for every scenario, and remove it after.

See :mod:`.launcher` for why the framework owns the stack's lifecycle, and
:mod:`.docker` for running a local Autoware workspace in its dev container.

Usage::

    from autoware_carla_scenario.autoware_stack import (
        DockerAutowareConfig,
        DockerAutowareLauncher,
    )
"""

from __future__ import annotations

from .devcontainer import DevContainer, DevContainerError, read_devcontainer
from .docker import STACK_LABEL, DockerAutowareConfig, DockerAutowareLauncher
from .launcher import (
    AutowareEpisode,
    AutowareLauncher,
    AutowareStackError,
    CommandAutowareLauncher,
    ProcessGroup,
)

__all__ = [
    "STACK_LABEL",
    "AutowareEpisode",
    "AutowareLauncher",
    "AutowareStackError",
    "CommandAutowareLauncher",
    "DevContainer",
    "DevContainerError",
    "DockerAutowareConfig",
    "DockerAutowareLauncher",
    "ProcessGroup",
    "read_devcontainer",
]
