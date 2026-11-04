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


#: How many placements :func:`_fit` tries before it gives up.
_FIT_STEPS = 10000


def _fit(
    sizes: list[int], gpus: Sequence[Gpu], free: dict[int, int]
) -> Optional[list[Gpu]]:
    """A GPU for each size (largest first) that leaves none short, or ``None``.

    The first placement tried puts each on the GPU with the most room left
    (the first such GPU on a tie), so a run the greedy placement fits is
    placed the same way. It gives up after :data:`_FIT_STEPS` placements.
    """
    listed = list(gpus)
    room = [free[gpu.index] for gpu in listed]
    chosen = [0 for _ in sizes]
    steps = [_FIT_STEPS]
    if _search(0, sizes, room, chosen, steps):
        return [listed[index] for index in chosen]
    return None


def _search(
    at: int, sizes: list[int], room: list[int], chosen: list[int], steps: list[int]
) -> bool:
    """Place ``sizes[at:]`` in ``room``, recording each GPU in ``chosen``."""
    if at == len(sizes):
        return True
    steps[0] -= 1
    if steps[0] < 0:
        return False
    rest = 0
    for size in sizes[at:]:
        rest += size
    total = 0
    for left in room:
        total += left
    if rest > total:
        return False
    size = sizes[at]
    # The roomiest first; a GPU with as much room as one tried is the same.
    tried: list[int] = []
    for index in sorted(range(len(room)), key=lambda i: -room[i]):
        if room[index] < size:
            break
        if room[index] in tried:
            continue
        tried.append(room[index])
        room[index] -= size
        chosen[at] = index
        if _search(at + 1, sizes, room, chosen, steps):
            return True
        room[index] += size
    return False


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
    memory_mb = 0
    for _, need in needs:
        memory_mb += need.memory_mb
    if memory_mb > host.memory_mb:
        shortfalls.append(
            f"host memory: {memory_mb} MiB needed, {host.memory_mb} MiB available"
        )
    free = {gpu.index: gpu.free_mb for gpu in host.gpus}
    placed: dict[str, Optional[Gpu]] = {}
    # The backends set to a GPU first, so one that is free to go anywhere
    # cannot take their room.
    by_size = sorted(needs, key=lambda item: -item[1].gpu_memory_mb)
    loose: list[tuple[str, Resources]] = []
    for name, need in by_size:
        if need.gpu_memory_mb <= 0:
            placed[name] = None
            continue
        pinned = need.gpu
        if pinned is None:
            loose.append((name, need))
            continue
        best = next((gpu for gpu in host.gpus if gpu.index == pinned), None)
        if best is None:
            shortfalls.append(f"{name}: is set to GPU {pinned}, which is not visible")
            continue
        if free[best.index] < need.gpu_memory_mb:
            shortfalls.append(
                f"{name}: {need.gpu_memory_mb} MiB of GPU memory needed, "
                f"{free[best.index]} MiB free on the GPU it is set to ({best.index})"
            )
            continue
        free[best.index] -= need.gpu_memory_mb
        placed[name] = best
    # Then the rest: the largest first, each onto the GPU with the most room
    # left, or, when that leaves one without room, another placement that
    # fits them all (#1121).
    if loose and host.gpus:
        found = _fit([need.gpu_memory_mb for _, need in loose], host.gpus, free)
        if found is not None:
            for index in range(len(loose)):
                placed[loose[index][0]] = found[index]
            loose = []
    # None fits them all: the greedy placement names the ones left out.
    for name, need in loose:
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
    gpus: list[Gpu] = []
    found = shutil.which("nvidia-smi")
    if found is None:
        return tuple(gpus)
    smi: str = found
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return tuple(gpus)
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
