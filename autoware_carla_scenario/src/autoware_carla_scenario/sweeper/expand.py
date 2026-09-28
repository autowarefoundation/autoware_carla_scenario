"""Expand a logical scenario into its concrete scenarios.

A scenario config with a ``sweep:`` section is *logical*: its constraints pick
lanelets for one parameter (``sweep.constraints``) and its bindings derive
others from each pick (``sweep.bindings``). Expanding it enumerates every
lanelet of the map that satisfies the constraints -- sorted, so the result is
the same every time for the same map -- and turns each into the Hydra
overrides that run that one concrete scenario::

    [["ego.spawn_lanelet_id=242", "ego.spawn_s=18.6"], ...]

A config without a sweep is already concrete and expands to itself (one empty
override list). The Hydra sweeper (``hydra/sweeper=lanelet_constraint``) runs
these in one process; ``scenario-expand`` hands them to a caller that runs them
elsewhere -- e.g. a test suite fanning them out over a cluster.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from .bindings import Binding, parse_binding
from .constraints import (
    Constraint,
    create_routing_graph,
    find_matching_lanelets,
    parse_constraint,
)

logger = logging.getLogger(__name__)


def expand_sweep(
    sweep: Mapping[Any, Any], lanelet_map: Any, arguments: Sequence[str] = ()
) -> list[list[str]]:
    """The override list of every concrete scenario ``sweep`` describes on ``lanelet_map``.

    ``arguments`` are appended to every list. A lanelet whose binding cannot be
    resolved is skipped (and logged).

    Raises:
        ValueError: If ``sweep`` has no constraints.
    """
    constraints_cfg = sweep.get("constraints") or {}
    # Constraints are keyed by the target parameter (e.g. ego.spawn_lanelet_id);
    # each value is a list of constraint dicts.
    constraints: list[Constraint] = []
    target_key: str | None = None
    for target_key, constraint_list in constraints_cfg.items():
        constraints.extend(parse_constraint(c) for c in constraint_list)
    if not constraints or target_key is None:
        raise ValueError("sweep.constraints is empty; nothing to expand.")

    routing_graph = create_routing_graph(lanelet_map)
    matched_ids = find_matching_lanelets(constraints, lanelet_map, routing_graph)
    if not matched_ids:
        logger.warning("No lanelets match the given constraints.")
        return []

    bindings: list[Binding] = [
        parse_binding(key, b_cfg)
        for key, b_cfg in (sweep.get("bindings") or {}).items()
    ]
    cases: list[list[str]] = []
    for lid in matched_ids:
        overrides = [f"{target_key}={lid}"]
        for binding in bindings:
            try:
                result = binding.resolve(lid, lanelet_map, routing_graph)
            except Exception:
                logger.warning(
                    "Binding %s failed for lanelet %d; skipping this lanelet.",
                    binding.target_key,
                    lid,
                    exc_info=True,
                )
                break
            overrides.append(f"{binding.target_key}={result.value}")
            if result.lanelet_id_override is not None:
                overrides[0] = f"{target_key}={result.lanelet_id_override}"
        else:
            cases.append([*overrides, *arguments])
    if not cases:
        logger.warning("All lanelets were skipped due to binding failures.")
    return cases


def expand_config(cfg: DictConfig, arguments: Sequence[str] = ()) -> list[list[str]]:
    """The concrete scenarios of a composed scenario config (see module docstring)."""
    sweep_cfg = OmegaConf.select(cfg, "sweep")
    sweep = (
        OmegaConf.to_container(sweep_cfg, resolve=True) if sweep_cfg is not None else {}
    )
    if not isinstance(sweep, dict) or not sweep.get("constraints"):
        return [list(arguments)]  # already concrete

    from ..maps import resolve_map_paths  # noqa: PLC0415 -- clones a map on demand
    from .map_loader import load_map  # noqa: PLC0415

    lanelet_map = load_map(resolve_map_paths(OmegaConf.select(cfg, "map")))
    return expand_sweep(sweep, lanelet_map, arguments)


__all__ = ["expand_config", "expand_sweep"]
