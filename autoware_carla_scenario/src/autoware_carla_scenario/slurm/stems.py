"""The names a SLURM batch gives its cases' files and the keys of its configs.

:func:`sequential_stems` names each case's files as a local batch would, and
:func:`config_key` names a key of a config as the worker reports a
difference.  Split out of :mod:`.manifest` so the library check compiles them
(#846); reading and writing the batch's JSON (values of any type) stays there.
"""

from __future__ import annotations

from typing import Union

__all__ = ["config_key", "sequential_stems"]


def config_key(prefix: str, key: Union[int, str]) -> str:
    """The name of *key* under *prefix* in a config, as the worker reports a
    difference (``a.b``, ``a.list[2]``)."""
    if isinstance(key, int):
        return f"{prefix}[{key}]"
    return f"{prefix}.{key}" if prefix else str(key)


def sequential_stems(scenario_classes: list[str]) -> list[str]:
    """The file stem each case of a batch is named by, in a local batch's way.

    A local batch names a run's files by its scenario class, and the next run
    of the same class ``{class}-1``, ``{class}-2`` and so on, in the order they
    ran (``scenario_runner._run_file_stem``).  A case keeps that name whichever
    task runs it, and when.
    """
    seen: dict[str, int] = {}
    stems: list[str] = []
    for name in scenario_classes:
        count = seen.get(name, 0)
        stems.append(name if count == 0 else f"{name}-{count}")
        seen[name] = count + 1
    return stems
