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
A batch expanding several scenarios hands each the cases the earlier ones
drew (``drawn``, an :class:`~autoware_carla_scenario.odd.sampler.OddDraws`),
so scenarios on one ODD go on from each other rather than draw alike. Each
drawn case also carries itself, encoded (``++odd_case=...``), for its run to
record with its outcome; with ``odd_sample.flaky``, part of the cases are
chosen where the earlier runs left the outcome unsure.

Where the sweep has cases of its own (constraints or a route search), the
ODD's attributes the Lanelet2 map decides -- in a junction or not, how many
lanes, the speed limit: the road the case is on -- are drawn too, among the
buckets the cases' lanelets reach, and each drawn case takes a case on a
lanelet in the buckets drawn (:func:`_choose_cases`).  So the sampler's
strategy -- uniform, or what coverage left uncovered first -- picks the road
as it picks the settings.

``sweep.case_index`` keeps one case of the expansion, the N-th (0-based):
``++sweep.case_index=0`` runs every scenario of a batch on its first case, as
``scenario-expand`` lists them.

A config without a sweep is already concrete and expands to itself (one empty
override list). The Hydra sweeper (``hydra/sweeper=lanelet_constraint``) runs
these in one process; ``scenario-expand`` hands them to a caller that runs them
elsewhere -- e.g. a test suite fanning them out over a cluster.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from .bindings import Binding, parse_binding
from .constraints import (
    Constraint,
    create_routing_graph,
    find_matching_lanelets,
    parse_constraint,
)

if TYPE_CHECKING:
    from ..odd.sampler import OddDraws

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
    drawn: OddDraws | None = None,
) -> list[list[str]]:
    """The override list of every concrete scenario ``sweep`` describes on ``lanelet_map``.

    ``arguments`` are appended to every list, so they win. A lanelet whose
    binding cannot be resolved is skipped (and logged).

    With ``sweep.odd_sample``, ``count`` cases are drawn from the ODD
    (``odd_sample.odd``, else *odd*, the run's) and each is given the next
    lanelet case or route match in turn -- or none, when the sweep has
    neither. Without a ``count``, one case is drawn per lanelet case or match.  *controls* are
    the scenario's (its config's ``controls``): what its parameters set.
    *drawn*, in a batch, holds the cases earlier scenarios drew from the ODD:
    the sampler goes on from them, and the cases drawn here are added to it.

    With ``sweep.case_index``, only the case at that (0-based) index is kept.

    Raises:
        ValueError: If ``sweep`` has none of constraints, a route search and
            ``odd_sample``, or both constraints and a route search, or a
            ``case_index`` past the cases it expands to.
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
        base_cases = _with_odd_samples(
            base_cases,
            odd_sample,
            odd,
            controls,
            drawn,
            lanelet_map=lanelet_map if base_cases != [[]] else None,
        )
    case_index = sweep.get("case_index")
    if case_index is not None:
        index = int(case_index)
        if not 0 <= index < len(base_cases):
            raise ValueError(
                f"sweep.case_index {index} is out of range: the sweep expands "
                f"to {len(base_cases)} case(s)"
            )
        base_cases = [base_cases[index]]
    return [[*case, *arguments] for case in base_cases]


#: The setting a knob of a map-decided attribute writes: none a run reads.
#: It is taken off each drawn case once the case's road is chosen.
_MAP_KNOB_PREFIX = "_odd_map."


def _case_lanelet(case: Sequence[str]) -> int | None:
    """The lanelet a base case is on: its first ``*lanelet_id`` override (the
    constraints' pick, or the route's spawn)."""
    for item in case:
        key, _, value = item.lstrip("+").partition("=")
        if key.endswith("lanelet_id"):
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _case_readings(
    base_cases: list[list[str]], definition: Any, lanelet_map: Any
) -> list[dict[str, tuple[str, Any]]]:
    """Each base case's ``(bucket, value)`` of every attribute the map decides
    on its lanelet (:func:`~autoware_carla_scenario.odd.route.lanelet_readings`)."""
    from ..odd.route import lanelet_readings  # noqa: PLC0415

    graph = create_routing_graph(lanelet_map)
    found: list[dict[str, tuple[str, Any]]] = []
    for case in base_cases:
        lanelet_id = _case_lanelet(case)
        if lanelet_id is None:
            found.append({})
            continue
        try:
            lanelet = lanelet_map.laneletLayer[lanelet_id]
        except Exception:  # noqa: BLE001 - a lanelet the map lacks decides nothing
            found.append({})
            continue
        found.append(lanelet_readings(definition, lanelet, lanelet_map, graph))
    return found


def _map_knobs(
    definition: Any, buckets: list[dict[str, str]], given: Mapping[str, Any]
) -> dict[str, Any]:
    """Knobs for the attributes the map decides, keyed by attribute: each
    reaches the buckets the base cases' lanelets are in, and writes
    :data:`_MAP_KNOB_PREFIX` ``+ name``.  Not for an attribute the sweep
    already has a knob for (*given*)."""
    knobs: dict[str, Any] = {}
    for attribute in definition.attributes:
        name = attribute.name
        # A derived attribute is worked out from those it derives from.
        if name in given or attribute.derived:
            continue
        reached = sorted({b[name] for b in buckets if name in b})
        item = attribute.item
        if not reached or item is None:
            continue
        key = _MAP_KNOB_PREFIX + name
        if not item.numeric:
            knobs[name] = {"key": key, "values": {label: label for label in reached}}
            continue
        # A numeric bucket is drawn inside its own interval: the knob reaches
        # those of the reached buckets, and only them.
        edges = item.edges
        ranges: dict[str, list[float]] = {}
        for index, label in enumerate(item.labels):
            if label in reached and index + 1 < len(edges):
                ranges[label] = [edges[index], edges[index + 1]]
        if not ranges:
            continue
        low = min(r[0] for r in ranges.values())
        high = max(r[1] for r in ranges.values())
        knobs[name] = {"key": key, "values": ranges, "range": [low, high]}
    return knobs


#: How many cases :func:`_on_some_case` draws for each it keeps, at most.
_DRAWS_PER_CASE = 20


def _fits(sample: Any, buckets: dict[str, str], attributes: Sequence[str]) -> bool:
    return all(
        buckets.get(a) == sample.buckets[a] for a in attributes if a in sample.buckets
    )


def _on_some_case(
    sampler: Any, wanted: int, buckets: list[dict[str, str]], attributes: Sequence[str]
) -> list[Any]:
    """*wanted* cases from *sampler* whose buckets for the map's *attributes*
    some base case is in, all of them.

    Each attribute's knob reaches only buckets some case is in, but not every
    combination of them is on one lanelet -- no junction where the limit is
    30, say -- and the knobs draw each attribute on its own.  So more are
    drawn (:data:`_DRAWS_PER_CASE` for each kept, in the sampler's order) and
    those on no case are passed over; if too few are left, the rest are the
    first passed over, put on the case nearest them (:func:`_choose_cases`).
    """
    drawn = sampler.sample(wanted * _DRAWS_PER_CASE)
    kept = [s for s in drawn if any(_fits(s, b, attributes) for b in buckets)]
    if len(kept) >= wanted:
        return kept[:wanted]
    passed = [s for s in drawn if s not in kept]
    logger.warning(
        "Only %d of %d drawn case(s) fit a lanelet of the sweep in every "
        "bucket the map decides; %d go on the nearest one",
        len(kept),
        len(drawn),
        wanted - len(kept),
    )
    return kept + passed[: wanted - len(kept)]


def _choose_cases(
    samples: Sequence[Any],
    buckets: list[dict[str, str]],
    attributes: Sequence[str],
) -> tuple[list[int], int]:
    """The base case for each drawn sample, by index: one on a lanelet in
    every bucket the sample drew for *attributes*, else in as many of them as
    any; among those, the one run least so far (the first of equals), so
    the cases are spread over as many lanelets as the buckets allow.
    Returns the indices and how many matched fully."""
    chosen: list[int] = []
    uses = [0] * len(buckets)
    full = 0
    for sample in samples:
        wanted = {a: sample.buckets[a] for a in attributes if a in sample.buckets}
        scores = [
            sum(1 for a, label in wanted.items() if b.get(a) == label) for b in buckets
        ]
        best = max(scores) if scores else 0
        if best == len(wanted):
            full += 1
        index = min(
            (i for i, score in enumerate(scores) if score == best),
            key=lambda i: (uses[i], i),
        )
        uses[index] += 1
        chosen.append(index)
    return chosen, full


def _as_run(
    sample: Any, readings: dict[str, tuple[str, Any]], attributes: Sequence[str]
) -> Any:
    """*sample* as the case it is put on runs it: the map's attributes are
    its lanelet's buckets and values, not the ones drawn (a nearest case may
    be in other buckets, and a numeric value is drawn anywhere in its
    bucket), and the knobs' own settings are gone."""
    import dataclasses  # noqa: PLC0415

    buckets = dict(sample.buckets)
    values = dict(sample.values)
    for name in attributes:
        if name in readings:
            buckets[name], values[name] = readings[name]
        else:
            buckets.pop(name, None)
            values.pop(name, None)
    overrides = [o for o in sample.overrides if not o.startswith(_MAP_KNOB_PREFIX)]
    return dataclasses.replace(
        sample, overrides=overrides, buckets=buckets, values=values
    )


def _with_odd_samples(
    base_cases: list[list[str]],
    odd_sample: Mapping[Any, Any],
    odd: Any,
    controls: Mapping[str, Any] | None = None,
    drawn: OddDraws | None = None,
    *,
    lanelet_map: Any = None,
) -> list[list[str]]:
    """``odd_sample.count`` cases (as many as base cases by default), each a
    base case -- a lanelet or a route match -- and settings drawn from the ODD.

    With *lanelet_map*, the attributes the map decides are drawn as well and
    each case is put on a base case in the buckets drawn (the module
    docstring), so a base case may run more than once, or not at all; without
    it, the base cases are taken in turn."""
    from openodd import ParameterSpace  # noqa: PLC0415

    from ..odd.registry import resolve_odd  # noqa: PLC0415
    from ..odd.sampler import ODD_CASE_KEY, encode_case, search_from_config  # noqa: PLC0415

    raw = {str(k): v for k, v in odd_sample.items()}
    readings: list[dict[str, tuple[str, Any]]] = []
    buckets: list[dict[str, str]] = []
    map_attributes: list[str] = []
    if lanelet_map is not None:
        definition = resolve_odd(raw.get("odd") or odd)
        readings = _case_readings(base_cases, definition, lanelet_map)
        buckets = [{a: b for a, (b, _v) in r.items()} for r in readings]
        given = dict(raw.get("knobs") or {})
        map_knobs = _map_knobs(definition, buckets, given)
        if map_knobs:
            raw["knobs"] = {**map_knobs, **given}
            map_attributes = list(map_knobs)
    sampler, search, count = search_from_config(
        raw,
        odd=odd,
        controls=controls,
        drawn=drawn,
    )
    wanted = len(base_cases) if count is None else count
    if search is not None:
        samples = search.sample(wanted)
    elif map_attributes:
        samples = _on_some_case(sampler, wanted, buckets, map_attributes)
    else:
        samples = sampler.sample(wanted)
    space = search.space if search is not None else ParameterSpace.of(sampler)
    if map_attributes:
        indices, full = _choose_cases(samples, buckets, map_attributes)
        samples = [
            _as_run(sample, readings[i], map_attributes)
            for sample, i in zip(samples, indices)
        ]
        cases = [base_cases[i] for i in indices]
    else:
        cases = [base_cases[i % len(base_cases)] for i in range(len(samples))]
    if drawn is not None:
        drawn.add(sampler.odd, samples)
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
    if search is not None:
        explored = sum(sample.flaky is not None for sample in samples)
        logger.info(
            "%d of them chosen for instability, learnt from %d earlier run(s)",
            explored,
            len(search.model),
        )
    if map_attributes:
        logger.info(
            "Put them on the road drawn (%s): %d of %d on a lanelet in every "
            "bucket drawn; %d of %d base case(s) run",
            ", ".join(map_attributes),
            full,
            len(samples),
            len(set(indices)),
            len(base_cases),
        )
    # Each run gets its case, to record with its outcome in its coverage file:
    # as it runs it, on the lanelet it is put on.
    return [
        [*case, *sample.overrides, f"++{ODD_CASE_KEY}={encode_case(sample, space)}"]
        for case, sample in zip(cases, samples)
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


def expand_config(
    cfg: DictConfig,
    arguments: Sequence[str] = (),
    *,
    drawn: OddDraws | None = None,
) -> list[list[str]]:
    """The concrete scenarios of a composed scenario config (see module docstring).

    *drawn* is a batch's record of the cases drawn from the ODD so far
    (see :func:`expand_sweep`).
    """
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
        return expand_sweep(
            sweep, None, arguments, odd=odd, controls=controls, drawn=drawn
        )

    from ..maps import resolve_map_paths  # noqa: PLC0415 -- clones a map on demand
    from .map_loader import load_map  # noqa: PLC0415

    lanelet_map = load_map(resolve_map_paths(OmegaConf.select(cfg, "map")))
    return expand_sweep(
        sweep, lanelet_map, arguments, odd=odd, controls=controls, drawn=drawn
    )


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
