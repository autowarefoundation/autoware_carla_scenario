"""What the host has free for backends: GPU memory and host memory.

Read when backends are about to start, so a run that needs more than the host
has free is refused before anything starts
(:class:`~autoware_carla_scenario.prepare.InsufficientResourcesError`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from . import InsufficientResourcesError, Resources

__all__ = [
    "Gpu",
    "HostResources",
    "assign_gpus",
    "probe_host",
]


@dataclass(frozen=True)
class Gpu:
    """One GPU this process may use.

    Attributes:
        index: Its index, as ``nvidia-smi`` and ``docker --gpus`` count.
        uuid: Its UUID.
        free_mb: Memory free on it, MiB.
    """

    index: int
    uuid: str
    free_mb: int


@dataclass(frozen=True)
class HostResources:
    """What the host has free.

    Attributes:
        gpus: The GPUs this process may use (``CUDA_VISIBLE_DEVICES``).
        memory_mb: Host memory available, MiB (``MemAvailable``).
    """

    gpus: tuple[Gpu, ...]
    memory_mb: int


def probe_host() -> HostResources:
    """What the host has free now."""
    return HostResources(gpus=_visible(_gpus()), memory_mb=_available_memory_mb())


def assign_gpus(
    needs: Sequence[tuple[str, Resources]], host: HostResources
) -> dict[str, Optional[Gpu]]:
    """Place each backend on a GPU with room for it, or refuse.

    Args:
        needs: Each backend's name and what it takes.
        host: What the host has free (:func:`probe_host`).

    Returns:
        Each backend's GPU, ``None`` for one that needs none.

    Raises:
        InsufficientResourcesError: If the host memory or a GPU's memory
            falls short, naming what each backend needs and what is free.
    """
    shortfalls: list[str] = []
    total = Resources()
    for _, need in needs:
        total = total + need
    if total.memory_mb > host.memory_mb:
        shortfalls.append(
            f"host memory: {total.memory_mb} MiB needed, {host.memory_mb} MiB available"
        )
    free = {gpu.index: gpu.free_mb for gpu in host.gpus}
    placed: dict[str, Optional[Gpu]] = {}
    # The largest first, each onto the GPU with the most room left.
    for name, need in sorted(needs, key=lambda item: -item[1].gpu_memory_mb):
        if need.gpu_memory_mb <= 0:
            placed[name] = None
            continue
        best = max(host.gpus, key=lambda gpu: free[gpu.index], default=None)
        if best is None:
            shortfalls.append(f"{name}: needs a GPU, and none is visible")
            continue
        if free[best.index] < need.gpu_memory_mb:
            shortfalls.append(
                f"{name}: {need.gpu_memory_mb} MiB of GPU memory needed, "
                f"{free[best.index]} MiB free on the roomiest GPU ({best.index})"
            )
            continue
        free[best.index] -= need.gpu_memory_mb
        placed[name] = best
    if shortfalls:
        wanted = ", ".join(
            f"{name} ({need.gpu_memory_mb} MiB GPU, {need.memory_mb} MiB host)"
            for name, need in needs
        )
        raise InsufficientResourcesError(
            "the host cannot run the backends this run needs -- "
            + "; ".join(shortfalls)
            + f" (needed: {wanted}). Free the memory, or lower what a backend "
            "is said to take (backends.<name>.gpu_memory_mb / memory_mb)."
        )
    return placed


def _gpus() -> tuple[Gpu, ...]:
    """Every GPU ``nvidia-smi`` lists, with its free memory; none without it."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return ()
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    gpus: list[Gpu] = []
    for line in out.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            gpus.append(
                Gpu(index=int(fields[0]), uuid=fields[1], free_mb=int(fields[2]))
            )
        except ValueError:
            continue
    return tuple(gpus)


def _visible(gpus: tuple[Gpu, ...]) -> tuple[Gpu, ...]:
    """The *gpus* ``CUDA_VISIBLE_DEVICES`` lets this process use (SLURM sets
    it for a job's GPUs), by index or UUID; all of them when it is unset."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return gpus
    wanted = [item.strip() for item in visible.split(",") if item.strip()]
    chosen: list[Gpu] = []
    for item in wanted:
        for gpu in gpus:
            if item == str(gpu.index) or gpu.uuid.startswith(item):
                if gpu not in chosen:
                    chosen.append(gpu)
    return tuple(chosen)


def _available_memory_mb() -> int:
    """``MemAvailable`` of ``/proc/meminfo``, MiB (0 if it cannot be read).

    The host's, not this process's cgroup's: a backend's container runs
    outside the job that starts it.
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0
