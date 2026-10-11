"""Expand a logical scenario into its concrete scenarios.

A scenario config with a ``sweep:`` section is *logical*: its constraints pick
lanelets for one parameter (``sweep.constraints``) and its bindings derive
others from each pick (``sweep.bindings``). Expanding it enumerates every
lanelet of the map that satisfies the constraints -- sorted, so the result is
the same every time for the same map -- and turns each into the Hydra
overrides that run that one concrete scenario::

    [["ego.spawn_lanelet_id=242", "ego.spawn_s=18.6"], ...]

A *logical* scenario names its ego's route as a pattern of road instead
(``sweep.route``, :mod:`autoware_carla_scenario.route`): it expands to one
scenario per route of the map that matches, each with the ego's spawn and goal
and the match itself (``scenario.route.*``) as overrides
(:func:`expand_route`).

``sweep.odd_sample`` draws the settings a run can be given -- the weather, the
sun, scenario parameters -- from the ODD, aimed at what earlier runs left
uncovered when asked to (:mod:`autoware_carla_scenario.odd.sampler`); with
constraints or a route search too, each drawn case takes their cases in turn.

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
    lanelet case or route match in turn -- or none, when the sweep has
    neither. Without a ``count``, one case is drawn per lanelet case or match.  *controls* are
    the scenario's (its config's ``controls``): what its parameters set.

    Raises:
        ValueError: If ``sweep`` has none of constraints, a route search and
            ``odd_sample``, or both constraints and a route search.
    """
    odd_sample = sweep.get("odd_sample") or {}
    if sweep.get("route"):
        if sweep.get("constraints"):
            raise ValueError(
                "sweep has both a route search and constraints; a scenario is "
                "expanded by one of them"
            )
        base_cases = expand_route(sweep["route"], lanelet_map)
    elif sweep.get("constraints"):
        base_cases = _expand_lanelets(sweep, lanelet_map)
    elif odd_sample:
        base_cases = [[]]
    else:
        raise ValueError("sweep.constraints is empty; nothing to expand.")
    if odd_sample and base_cases:
        base_cases = _with_odd_samples(base_cases, odd_sample, odd, controls)
    return [[*case, *arguments] for case in base_cases]


def _with_odd_samples(
    base_cases: list[list[str]],
    odd_sample: Mapping[Any, Any],
    odd: Any,
    controls: Mapping[str, Any] | None = None,
) -> list[list[str]]:
    """``odd_sample.count`` cases (one per base case by default), each a base
    case -- a lanelet or a route match -- and settings drawn from the ODD."""
    from ..odd.sampler import sampler_from_config  # noqa: PLC0415

    sampler, count = sampler_from_config(
        {str(k): v for k, v in odd_sample.items()}, odd=odd, controls=controls
    )
    samples = sampler.sample(len(base_cases) if count is None else count)
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
        [*base_cases[i % len(base_cases)], *sample.overrides]
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


def expand_route(
    route: Mapping[str, Any], lanelet_map: Any, arguments: Sequence[str] = ()
) -> list[list[str]]:
    """The override list of every route of ``lanelet_map`` a route search matches.

    Per match, in the order :func:`~autoware_carla_scenario.route.search.find_route_matches`
    returns them: the ego's spawn (``ego.spawn_lanelet_id`` / ``ego.spawn_s``,
    ``ego_spawn_s`` along the route), its goal (``ego.goal_lanelet_id`` /
    ``ego.goal_s``, the route's end less ``ego_goal_margin``) when
    ``ego_goal`` is set, and the match as the ``scenario.route.*`` keys.  A
    match the ego cannot be placed on is skipped (and logged).

    Raises:
        ValueError: If ``route`` is not a well-formed route search.
    """
    from ..route.frame import RouteFrame, ego_placement  # noqa: PLC0415
    from ..route.model import parse_route_search  # noqa: PLC0415
    from ..route.search import find_route_matches  # noqa: PLC0415

    spec = parse_route_search(route)
    routing_graph = create_routing_graph(lanelet_map)
    matches = find_route_matches(spec, lanelet_map, routing_graph)
    if not matches:
        logger.warning("No route of the map matches the route search.")
        return []
    cases: list[list[str]] = []
    for match in matches:
        frame = RouteFrame(match, lanelet_map, routing_graph)
        try:
            (spawn_id, spawn_s), goal = ego_placement(
                frame, spec.ego_spawn_s, spec.ego_goal, spec.ego_goal_margin
            )
        except ValueError:
            logger.warning(
                "Route match %d (%s) cannot place the ego; skipping it.",
                match.index,
                list(match.lanelet_ids),
                exc_info=True,
            )
            continue
        overrides = [
            f"ego.spawn_lanelet_id={spawn_id}",
            f"ego.spawn_s={round(spawn_s, 4)}",
        ]
        if goal is not None:
            overrides += [
                f"ego.goal_lanelet_id={goal[0]}",
                f"ego.goal_s={round(goal[1], 4)}",
            ]
        overrides += [
            f"scenario.route.{key}={_override_value(value)}"
            for key, value in match.to_config().items()
        ]
        cases.append([*overrides, *arguments])
    return cases


def expand_config(cfg: DictConfig, arguments: Sequence[str] = ()) -> list[list[str]]:
    """The concrete scenarios of a composed scenario config (see module docstring)."""
    sweep_cfg = OmegaConf.select(cfg, "sweep")
    sweep = (
        OmegaConf.to_container(sweep_cfg, resolve=True) if sweep_cfg is not None else {}
    )
    if not isinstance(sweep, dict) or not (
        sweep.get("constraints") or sweep.get("route") or sweep.get("odd_sample")
    ):
        return [list(arguments)]  # already concrete

    odd = OmegaConf.select(cfg, "odd")
    controls = scenario_controls(cfg)
    if not (sweep.get("constraints") or sweep.get("route")):
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
        raise ValueError("controls: expected a mapping of measure -> control")
    return {str(k): v for k, v in controls.items()}


__all__ = ["expand_config", "expand_route", "expand_sweep", "scenario_controls"]
