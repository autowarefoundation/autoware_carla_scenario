"""Expand a scenario pattern to the config names it matches.

The one resolver behind the ``scenario`` CLI's glob
(``examples.run._resolve_scenario_glob``) and the viewer's launcher
(``ui.app._resolve_scenario_names``), so a pattern runs the same scenarios
from either.  The pattern is matched with :meth:`pathlib.Path.glob` as
``scenario/{pattern}.yaml`` under each config directory: ``*`` stays within
one path component (``**`` crosses them), and a name without glob
characters matches its one file.  The names are Hydra config paths relative
to ``scenario/`` without the ``.yaml`` suffix (``"lane_change/left"``),
each once, in plain string order.

Free of heavy imports (only :mod:`pathlib`), so the viewer can resolve a
pattern without importing CARLA through ``examples.run``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_scenario_glob", "scenario_dirs"]


def scenario_dirs(conf_dirs: list[Path]) -> list[Path]:
    """The ``scenario/`` directory of each config directory that has one,
    in the order given."""
    return [
        conf_dir / "scenario"
        for conf_dir in conf_dirs
        if (conf_dir / "scenario").is_dir()
    ]


def resolve_scenario_glob(pattern: str, conf_dirs: list[Path]) -> list[str]:
    """The config names ``scenario/{pattern}.yaml`` matches under *conf_dirs*.

    Each name once, sorted as strings; empty when nothing matches.
    """
    names: set[str] = set()
    for scenario_dir in scenario_dirs(conf_dirs):
        for match in scenario_dir.glob(f"{pattern}.yaml"):
            # Hydra config names always use forward slashes, so normalise
            # away any OS-specific separator.
            names.add(match.relative_to(scenario_dir).with_suffix("").as_posix())
    return sorted(names)
