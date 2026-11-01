"""Expand a scenario pattern to the config names it matches.

The one resolver behind the ``scenario`` CLI's glob
(``examples.run._resolve_scenario_glob``) and the viewer's launcher
(``ui.app._resolve_scenario_names``), so a pattern runs the same scenarios
from either.  The pattern is matched with :meth:`pathlib.Path.glob` as
``scenario/{pattern}.yaml`` under each config directory: ``*`` stays within
one path component, and a ``**`` component crosses them (zero or more
directories).  A trailing ``**`` means everything below it: it is read as
``**/*``, so ``lane_change/**`` and ``**`` match every config below
``lane_change/`` and ``scenario/``.  A ``**`` inside a component
(``lane_change/**left``) is refused with a :class:`ValueError` naming the
pattern, the same on every Python (3.10-3.12 would raise their own error,
3.13+ would read it as ``*``).  A name without glob characters
(:func:`is_scenario_glob`) matches its one file.  The names are
Hydra config paths relative to ``scenario/`` without the ``.yaml`` suffix
(``"lane_change/left"``), each once, in plain string order.  The viewer's
scenario list (``ui.scanner.list_scenario_configs``) is every name under
the config directories (:func:`scenario_names`), ordered by path component.

Free of heavy imports (only :mod:`pathlib`), so the viewer can resolve a
pattern without importing CARLA through ``examples.run``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "is_scenario_glob",
    "resolve_scenario_glob",
    "scenario_dirs",
    "scenario_names",
]


def is_scenario_glob(value: str) -> bool:
    """Whether *value* is a pattern (has ``*``, ``?`` or ``[``) rather than
    one scenario's name: the one test the CLI and the viewer both apply."""
    return any(ch in value for ch in ("*", "?", "["))


def scenario_dirs(conf_dirs: list[Path]) -> list[Path]:
    """The ``scenario/`` directory of each config directory that has one,
    in the order given."""
    return [
        conf_dir / "scenario"
        for conf_dir in conf_dirs
        if (conf_dir / "scenario").is_dir()
    ]


def _glob_pattern(pattern: str) -> str:
    """*pattern* as :meth:`pathlib.Path.glob` reads it alike on every Python.

    A trailing ``**`` component becomes ``**/*`` (everything below); a
    ``**`` that is not a whole component raises :class:`ValueError`.
    """
    parts = pattern.split("/")
    for part in parts:
        if part != "**" and "**" in part:
            raise ValueError(
                f"invalid scenario pattern {pattern!r}: '**' can only be an "
                "entire path component (e.g. 'lane_change/**' or "
                "'lane_change/**/left')"
            )
    if parts[-1] == "**":
        parts.append("*")
    return "/".join(parts)


def resolve_scenario_glob(pattern: str, conf_dirs: list[Path]) -> list[str]:
    """The config names ``scenario/{pattern}.yaml`` matches under *conf_dirs*.

    Each name once, sorted as strings; empty when nothing matches.  A
    trailing ``**`` component matches everything below it (``**/*``).

    Raises:
        ValueError: *pattern* has a ``**`` that is not a whole path
            component (``lane_change/**left``).
    """
    glob = _glob_pattern(pattern)
    names: set[str] = set()
    for scenario_dir in scenario_dirs(conf_dirs):
        for match in scenario_dir.glob(f"{glob}.yaml"):
            # Hydra config names always use forward slashes, so normalise
            # away any OS-specific separator.
            names.add(match.relative_to(scenario_dir).with_suffix("").as_posix())
    return sorted(names)


def scenario_names(conf_dirs: list[Path]) -> list[str]:
    """Every config name under ``scenario/`` of *conf_dirs*.

    The relative path of each YAML file without the ``.yaml`` suffix (e.g.
    ``"intersection_passing/left_turn"``), each name once, ordered by path
    component.
    """
    names: set[str] = set()
    for conf_dir in conf_dirs:
        scenario_dir = conf_dir / "scenario"
        if not scenario_dir.is_dir():
            continue
        for p in scenario_dir.rglob("*.yaml"):
            # Hydra config names always use forward slashes.
            names.add(p.relative_to(scenario_dir).with_suffix("").as_posix())
    # Ordered by path component, as the directory walk always listed them.
    return sorted(names, key=lambda name: name.split("/"))
