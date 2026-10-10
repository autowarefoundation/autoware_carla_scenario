"""Expand a logical scenario into its concrete scenarios.

A scenario config with a ``sweep:`` section is *logical*: its constraints pick
lanelets for one parameter (``sweep.constraints``) and its bindings derive
others from each pick (``sweep.bindings``). Expanding it enumerates every
lanelet of the map that satisfies the constraints -- sorted, so the result is
the same every time for the same map -- and turns each into the Hydra
overrides that run that one concrete scenario::

    [["ego.spawn_lanelet_id=242", "ego.spawn_s=18.6"], ...]

``sweep.odd_sample`` draws the settings a run can be given -- the weather, the
sun, scenario parameters -- from the ODD instead, aimed at what earlier runs
left uncovered when asked to (:mod:`autoware_carla_scenario.odd.sampler`); with
constraints too, each drawn case takes the lanelet cases in turn.

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


def _override_value(value: Any) -> str:
    """Render a binding's value the way a Hydra override reads it back.

    A list becomes ``[a,b]`` with no spaces, which Hydra parses as a list and a
    shell never has to be asked about. An int stays an int: ``goal_lanelet_id``
    does not take ``1234.0``.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_override_value(v) for v in value) + "]"
    return str(value)


def expand_sweep(
    sweep: Mapping[Any, Any],
    lanelet_map: Any,
    arguments: Sequence[str] = (),
    *,
    odd: Any = None,
    controls: Mapping[str, Any] | None = None,
) -> list[list[str]]:
    """The override list of every concrete scenario ``sweep`` describes on ``lanelet_map``.

    ``arguments`` are appended to every list, so they win. A lanelet whose
    binding cannot be resolved is skipped (and logged).

    With ``sweep.odd_sample``, ``count`` cases are drawn from the ODD
    (``odd_sample.odd``, else *odd*, the run's) and each is given the next
    lanelet case in turn -- or none, when the sweep has no constraints.
    Without a ``count``, one case is drawn per lanelet case.  *controls* are
    the scenario's (its config's ``controls``): what its parameters set.

    Raises:
        ValueError: If ``sweep`` has neither constraints nor ``odd_sample``.
    """
    odd_sample = sweep.get("odd_sample") or {}
    if sweep.get("constraints"):
        lanelet_cases = _expand_lanelets(sweep, lanelet_map)
    elif odd_sample:
        lanelet_cases = [[]]
    else:
        raise ValueError("sweep.constraints is empty; nothing to expand.")
    if odd_sample and lanelet_cases:
        lanelet_cases = _with_odd_samples(lanelet_cases, odd_sample, odd, controls)
    return [[*case, *arguments] for case in lanelet_cases]


def _with_odd_samples(
    lanelet_cases: list[list[str]],
    odd_sample: Mapping[Any, Any],
    odd: Any,
    controls: Mapping[str, Any] | None = None,
) -> list[list[str]]:
    """``odd_sample.count`` cases (one per lanelet case by default), each a
    lanelet case and settings drawn from the ODD."""
    from ..odd.sampler import sampler_from_config  # noqa: PLC0415

    sampler, count = sampler_from_config(
        {str(k): v for k, v in odd_sample.items()}, odd=odd, controls=controls
    )
    samples = sampler.sample(len(lanelet_cases) if count is None else count)
    logger.info(
        "Drew %d case(s) from ODD %s over %s (%s)%s",
        len(samples),
        sampler.odd.name,
        ", ".join(sampler.attributes),
        sampler.strategy,
        f"; aiming at situations {sampler.situation_holes}"
        if sampler.situation_holes
        else "",
    )
    return [
        [*lanelet_cases[i % len(lanelet_cases)], *sample.overrides]
        for i, sample in enumerate(samples)
    ]


def _expand_lanelets(sweep: Mapping[Any, Any], lanelet_map: Any) -> list[list[str]]:
    """One override list per lanelet the constraints match, bindings applied."""
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
            overrides.append(f"{binding.target_key}={_override_value(result.value)}")
            if result.lanelet_id_override is not None:
                overrides[0] = f"{target_key}={result.lanelet_id_override}"
        else:
            cases.append(overrides)
    if not cases:
        logger.warning("All lanelets were skipped due to binding failures.")
    return cases


def expand_config(cfg: DictConfig, arguments: Sequence[str] = ()) -> list[list[str]]:
    """The concrete scenarios of a composed scenario config (see module docstring)."""
    sweep_cfg = OmegaConf.select(cfg, "sweep")
    sweep = (
        OmegaConf.to_container(sweep_cfg, resolve=True) if sweep_cfg is not None else {}
    )
    if not isinstance(sweep, dict) or not (
        sweep.get("constraints") or sweep.get("odd_sample")
    ):
        return [list(arguments)]  # already concrete

    odd = OmegaConf.select(cfg, "odd")
    controls = scenario_controls(cfg)
    if not sweep.get("constraints"):
        # Drawing from the ODD alone needs no map.
        return expand_sweep(sweep, None, arguments, odd=odd, controls=controls)

    from ..maps import resolve_map_paths  # noqa: PLC0415 -- clones a map on demand
    from .map_loader import load_map  # noqa: PLC0415

    lanelet_map = load_map(resolve_map_paths(OmegaConf.select(cfg, "map")))
    return expand_sweep(sweep, lanelet_map, arguments, odd=odd, controls=controls)


def scenario_controls(cfg: DictConfig) -> dict[str, Any]:
    """The config's ``controls``, resolved: what the scenario's parameters set."""
    node = OmegaConf.select(cfg, "controls")
    if node is None:
        return {}
    controls = OmegaConf.to_container(node, resolve=True)
    if not isinstance(controls, dict):
        raise ValueError("controls: expected a mapping of attribute -> knob")
    return {str(k): v for k, v in controls.items()}


__all__ = ["expand_config", "expand_sweep", "scenario_controls"]
