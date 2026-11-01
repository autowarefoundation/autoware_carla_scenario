"""What a command line that does not compose prints, for every console script.

``scenario-check`` and ``scenario-expand`` compose a scenario config from the
command line exactly as a run does.  When it does not compose -- no config
matches ``scenario=``, or an override names a key the config does not have --
that is a mistake on the command line, not a bug: they print one message to
stderr (with the closest config or key) instead of Hydra's traceback, and exit
with 2.  ``HYDRA_FULL_ERROR=1`` raises the error instead, traceback and all.
"""

from __future__ import annotations

import difflib
import os
import re
import sys
from typing import Any

__all__ = ["compose_error", "compose_or_report", "resolve_scenarios", "suggest"]

_OVERRIDE_KEY = re.compile(r"Could not (?:override|delete) '([^']+)'")


def _flatten(node: Any, prefix: str = "") -> list[str]:
    """The dotted keys of the nested mapping *node*."""
    if not isinstance(node, dict):
        return []
    keys: list[str] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        keys.append(path)
        keys += _flatten(value, path)
    return keys


def suggest(word: str, known: list[str]) -> str:
    """``did you mean '<closest>'? ``, the entry of *known* closest to *word*, or ``''``."""
    close = difflib.get_close_matches(word, known, n=1, cutoff=0.6)
    return f"did you mean {close[0]!r}? " if close else ""


def compose_error(exc: Exception, name: str, overrides: list[str]) -> str:
    """One message for the composition error *exc* of config *name*."""
    from . import run  # noqa: PLC0415 - imports the CARLA client

    text = str(exc).strip()
    found = _OVERRIDE_KEY.search(text)
    if found is None:
        return f"Error: {text.splitlines()[0] if text else type(exc).__name__}"
    key = found.group(1)
    try:
        known = _flatten(run._to_dict(run._compose_config(name, [])), "")
    except Exception:  # noqa: BLE001 - the suggestion is a courtesy
        known = []
    value = next(
        (o.split("=", 1)[1] for o in overrides if o.split("=", 1)[0] == key), "..."
    )
    return (
        f"Error: could not override {key!r}: the config has no such key\n"
        f"  ({suggest(key, known)}To add a new key, use +{key}={value})"
    )


def resolve_scenarios(pattern: str) -> list[str] | None:
    """The scenario configs *pattern* (a name or a glob) matches.

    None, once stderr names the closest configs and lists them all, when it
    matches none; None, once stderr says why, when it is not a valid pattern
    (a ``**`` that is not a whole path component, a leading ``/`` or a
    ``..`` component).  Nothing goes to stdout,
    which may be a tool's JSON.
    """
    from ..registry import get_conf_dirs  # noqa: PLC0415
    from ..scenario_glob import resolve_scenario_glob, scenario_dirs  # noqa: PLC0415

    conf_dirs = get_conf_dirs()
    try:
        names = resolve_scenario_glob(pattern, conf_dirs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        return None
    if names:
        return names
    known = resolve_scenario_glob("**/*", conf_dirs)
    print(  # noqa: T201
        f"Error: no scenario configs match pattern '{pattern}' "
        f"under {[str(d) for d in scenario_dirs(conf_dirs)]}",
        file=sys.stderr,
    )
    close = difflib.get_close_matches(pattern, known, n=3, cutoff=0.5)
    if close:
        print(f"  did you mean: {', '.join(close)}?", file=sys.stderr)  # noqa: T201
    print(f"  scenario configs: {', '.join(known)}", file=sys.stderr)  # noqa: T201
    return None


def compose_or_report(name: str, overrides: list[str]) -> Any:
    """Config *name* composed with *overrides* as a run composes it.

    None, once stderr says why, when the command line does not compose: a key,
    a value, what the command line says rather than a bug.  Hydra's own
    switch, ``HYDRA_FULL_ERROR=1``, raises it instead (as any other exception
    is raised).
    """
    from hydra.errors import HydraException  # noqa: PLC0415
    from omegaconf.errors import OmegaConfBaseException  # noqa: PLC0415

    from . import run  # noqa: PLC0415 - imports the CARLA client

    try:
        return run._compose_config(name, overrides)
    except (HydraException, OmegaConfBaseException) as exc:
        if os.environ.get("HYDRA_FULL_ERROR") == "1":
            raise
        print(compose_error(exc, name, overrides), file=sys.stderr)  # noqa: T201
        return None
