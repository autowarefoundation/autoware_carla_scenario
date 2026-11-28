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
constraints or a route search too, each drawn case takes one of their cases
-- in turn, unless the map decides some of the ODD's attributes (below).
The run's own overrides may not set what a drawn case sets
(:func:`expand_sweep`).
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
lanelet in the buckets drawn (:func:`.case_choice.choose_cases`).  So the sampler's
strategy -- uniform, or what coverage left uncovered first -- picks the road
as it picks the settings.

``sweep.case_index`` keeps one case of the expansion, the N-th (0-based):
``++sweep.case_index=0`` runs every scenario of a batch on its first case, as
``scenario-expand`` lists them.

A config without a sweep is already concrete and expands to itself (one empty
override list). Its ``controls`` are still checked, as a sweep's are
(:func:`scenario_controls`). The Hydra sweeper (``hydra/sweeper=lanelet_constraint``) runs
these in one process; ``scenario-expand`` hands them to a caller that runs them
elsewhere -- e.g. a test suite fanning them out over a cluster.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf

from ..seed import DEFAULT_SEED
from ..value_text import shown
from .bindings import Binding, parse_binding

# The lanelet key a walk-back moves is the checked bindings's (#1697); the
# name it had here stays importable from here.
from .bindings import _walk_back_lanelet_key as _walk_back_lanelet_key
from .case_choice import case_lanelet as _case_lanelet
from .case_choice import choose_cases as _choose_cases
from .case_choice import fits as _fits
from .case_choice import override_key as _override_key
from .case_choice import write_override as _write_override
from .constraints import (
    Constraint,
    create_routing_graph,
    find_matching_lanelets,
    parse_constraint,
)
from .distributions import draw_case, fnv1a, is_distribution, parse_distribution
from .relations import (
    RelationSolver,
    RelationsUnsatisfiable,
    constant_value,
    parse_relations,
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


def _case_index(value: Any) -> int | None:
    """``sweep.case_index``, checked before anything is drawn or searched:
    a whole number >= 0, or ``None`` when unset.  Not ``int()``, which reads
    ``1.9`` and ``true`` as 1 and does not name the key on ``"abc"`` (#1819).

    Raises:
        ValueError: Naming ``sweep.case_index``
            (:func:`~autoware_carla_scenario.odd.sampler.whole_number_setting`).
    """
    if value is None:
        return None
    from ..odd.sampler import whole_number_setting  # noqa: PLC0415

    return whole_number_setting("sweep.case_index", value, minimum=0)


def expand_sweep(
    sweep: Mapping[Any, Any],
    lanelet_map: Any,
    arguments: Sequence[str] = (),
    *,
    odd: Any = None,
    controls: Mapping[str, Any] | None = None,
    drawn: OddDraws | None = None,
    seed: int | None = None,
    rules: Mapping[str, Any] | None = None,
    environment_rules: Any = None,
    composed: Sequence[str] = (),
    scenario: str = "",
    config_value: Callable[[str], Any] | None = None,
) -> list[list[str]]:
    """The override list of every concrete scenario ``sweep`` describes on ``lanelet_map``.

    ``arguments`` are appended to every list, so they win. A lanelet whose
    binding cannot be resolved is skipped (and logged).

    With ``sweep.odd_sample``, ``count`` cases are drawn from the ODD
    (``odd_sample.odd``, else *odd*, the run's) and each is given a lanelet
    case or route match -- in turn unless the map decides some of the ODD's
    attributes (docs/odd.md, "The road a case is on"): then each is put on a
    case in the buckets it drew, so a case may run more than once or not at
    all -- or none, when the sweep has neither. Without a ``count``, one case
    is drawn per lanelet case or match.  A drawn case's settings may not be
    overridden: an argument (or one of *composed*, the run's overrides
    already composed into its config, which its caller adds to the cases
    itself) that sets a key a knob or an effect of the ODD draws is refused,
    as the run would follow it while the case records the value drawn
    (#1995); one that sets the grip or the visibility (or the quantile of
    either) turns that draw off, as a case that names it draws none.  *controls* are
    the scenario's (its config's ``controls``): what its parameters set; *rules*
    are the nodes it adds to the rules (its ``scenario_rules``), which a
    control with a ``quantity`` writes; *environment_rules* are the run's
    weather rules (its ``environment.rules``), which the CARLA weather a
    case is given follows, as its run's does.
    *drawn*, in a batch, holds the cases earlier scenarios drew from the ODD:
    the sampler goes on from them, and the cases drawn here are added to it.
    *seed* is the run's (the config's ``seed``): the
    draws' when ``odd_sample.seed`` is not given or null, and
    :data:`~autoware_carla_scenario.seed.DEFAULT_SEED` when both are.

    The distributions in ``sweep.constraints`` draw a value for every key of
    theirs, conditioned on ``sweep.relations`` (solved with Z3,
    :mod:`.relations`), ``sweep.samples`` times for each case, from ``sweep.seed`` (else
    *seed*, else :data:`~autoware_carla_scenario.seed.DEFAULT_SEED`), the
    *scenario*'s name and the case (:mod:`.distributions`).  Before that,
    a case the relations cannot hold for is dropped.  *config_value* reads a
    key of the scenario's config (:func:`config_lookup`): what a case holds
    for a key a relation names and the case does not set.

    With ``sweep.case_index``, only the case at that (0-based) index is kept.

    Raises:
        ValueError: If ``sweep`` has none of constraints, a route search and
            ``odd_sample``, or both constraints and a route search, or a
            route search and bindings, or constraints for more than one
            target key, or a ``case_index``
            that is not a non-negative whole number or is past the cases it
            expands to, or an argument sets a key a drawn case sets.
    """
    lanelet_constraints, distributions = check_constraint_keys(sweep)
    index = _case_index(sweep.get("case_index"))
    # A run without a seed draws as one with DEFAULT_SEED does.
    run_seed = DEFAULT_SEED if seed is None else seed
    odd_sample = sweep.get("odd_sample") or {}
    if odd_sample and odd_sample.get("seed") is None:
        odd_sample = {**odd_sample, "seed": run_seed}
    if sweep.get("route"):
        if lanelet_constraints:
            raise ValueError(
                "sweep has both a route search and constraints; a scenario is "
                "expanded by one of them"
            )
        base_cases = expand_route(
            sweep["route"], lanelet_map, exclude=sweep.get("exclude")
        )
        # A route case is on its spawn.
        lanelets = [_case_lanelet(case) for case in base_cases]
    elif lanelet_constraints:
        base_cases, picks = _lanelet_cases(sweep, lanelet_map)
        # A lanelet case is on its pick, wherever a walk-back moved its
        # spawn (docs/odd.md, "The road a case is on").
        lanelets = list(picks)
    elif odd_sample or distributions:
        base_cases = [[]]
        lanelets = [None]
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
            lanelets=lanelets,
            rules=rules,
            environment_rules=environment_rules,
            arguments=[*arguments, *composed],
        )
    relations = parse_relations(sweep.get("relations"))
    solver = RelationSolver(relations, distributions) if relations else None
    if solver is not None and base_cases:
        base_cases = _with_relations_holding(
            base_cases, solver, [*arguments, *composed], config_value
        )
    if distributions and base_cases:
        draw_seed = _draw_seed(sweep.get("seed"))
        base_cases = _with_distribution_draws(
            base_cases,
            distributions,
            _samples(sweep.get("samples")),
            run_seed if draw_seed is None else draw_seed,
            scenario,
            [*arguments, *composed],
            solver,
            config_value,
        )
    if index is not None:
        if not 0 <= index < len(base_cases):
            raise ValueError(
                f"sweep.case_index {index} is out of range: the sweep expands "
                f"to {len(base_cases)} case(s)"
            )
        base_cases = [base_cases[index]]
    return [[*case, *arguments] for case in base_cases]


def _split_constraints(
    sweep: Mapping[Any, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``sweep.constraints`` split in two: the keys of lanelet constraints
    (with their lists) and the keys drawn from a distribution (with it
    parsed, :mod:`.distributions`)."""
    lanelet: dict[str, Any] = {}
    distributions: dict[str, Any] = {}
    for key, constraint_list in _section(sweep, "constraints").items():
        if is_distribution(constraint_list):
            distributions[str(key)] = parse_distribution(str(key), constraint_list)
        else:
            lanelet[str(key)] = constraint_list
    return lanelet, distributions


def _samples(value: Any) -> int:
    """``sweep.samples``: how many times each case runs; 1 when unset.

    Raises:
        ValueError: Naming ``sweep.samples``, if it is not a whole number of
            at least 1.
    """
    from ..odd.sampler import whole_number_setting  # noqa: PLC0415

    samples = whole_number_setting("sweep.samples", value, minimum=1)
    return 1 if samples is None else samples


def _draw_seed(value: Any) -> int | None:
    """``sweep.seed``, the distributions' seed; ``None`` when unset.

    Raises:
        ValueError: Naming ``sweep.seed``, if it is no seed
            (:func:`~autoware_carla_scenario.odd.sampler.seed_setting`).
    """
    from ..odd.sampler import seed_setting  # noqa: PLC0415

    return seed_setting("sweep.seed", value)


def _clashes(keys: set[str], overrides: Sequence[str]) -> str:
    """The keys of *keys* that *overrides* set, sorted and joined; empty when
    none do."""
    return ", ".join(sorted(keys & {_override_key(o) for o in overrides}))


def config_lookup(cfg: Any) -> Callable[[str], Any]:
    """A reader of *cfg*'s value at a dotted key: ``None`` for a key it does
    not have, a missing value or an interpolation that does not resolve."""

    def value(key: str) -> Any:
        try:
            return OmegaConf.select(cfg, key, default=None, throw_on_missing=False)
        except Exception:  # noqa: BLE001 -- an interpolation that does not resolve
            return None

    return value


def _relation_values(
    solver: RelationSolver,
    overrides: Sequence[str],
    config_value: Callable[[str], Any] | None,
) -> dict[str, Any]:
    """What a case holds for each key the relations name and no distribution
    draws, as relations' constants (:func:`.relations.constant_value`): what
    *overrides* (the case's, then the run's) set, read as Hydra reads them,
    else *config_value*'s; ``None`` when neither holds a value."""
    from hydra.core.override_parser.overrides_parser import (  # noqa: PLC0415
        OverridesParser,
    )

    wanted = set(solver.constant_keys)
    values: dict[str, Any] = {
        key: constant_value(config_value(key)) if config_value else None
        for key in wanted
    }
    parser = OverridesParser.create()
    for override in overrides:
        if _override_key(override) not in wanted:
            continue
        parsed = parser.parse_override(override)
        if not parsed.is_sweep_override():
            values[_override_key(override)] = (
                None if parsed.is_delete() else constant_value(parsed.value())
            )
    return values


def _with_relations_holding(
    base_cases: list[list[str]],
    solver: RelationSolver,
    arguments: Sequence[str],
    config_value: Callable[[str], Any] | None,
) -> list[list[str]]:
    """*base_cases* but those the relations cannot hold for (Z3,
    :class:`.relations.RelationSolver`), with the values each holds
    (:func:`_relation_values`).  How many were dropped is logged.

    Raises:
        ValueError: If the relations hold for none, or name a key that is
            neither drawn nor held by a case, or a value of the wrong type.
    """
    kept: list[list[str]] = []
    dropped: list[str] = []
    for case in base_cases:
        try:
            solver.check(_relation_values(solver, [*case, *arguments], config_value))
        except RelationsUnsatisfiable as exc:
            dropped.append(str(exc))
            continue
        kept.append(case)
    if dropped and not kept:
        raise ValueError(
            f"sweep.relations hold for none of the sweep's {len(base_cases)} "
            f"case(s); for the first: {dropped[0]}"
        )
    if dropped:
        logger.info(
            "sweep.relations dropped %d of %d case(s); the first: %s",
            len(dropped),
            len(base_cases),
            dropped[0],
        )
    return kept


#: The overrides of a route case that say where it stands among the others
#: (:func:`expand_route`), left out of its draws' seed so that a case keeps
#: its values when the map gains or loses another match (#2607).
_UNSEEDED_KEYS = frozenset({"scenario.route.index", "scenario.route.case_count"})


def _with_distribution_draws(
    base_cases: list[list[str]],
    distributions: Mapping[str, Any],
    samples: int,
    seed: int,
    scenario: str,
    arguments: Sequence[str],
    solver: RelationSolver | None,
    config_value: Callable[[str], Any] | None,
) -> list[list[str]]:
    """Each of *base_cases* *samples* times, each time with a value drawn for
    every key of *distributions*, in the order of the base cases.  A draw's
    seed is a hash of *seed*, *scenario*, the base case's overrides (but its
    encoded ODD case, which a change to the ODD's definition alone would move,
    and a route case's ``scenario.route.index`` and
    ``scenario.route.case_count``, which another match appearing or being
    excluded would: :data:`_UNSEEDED_KEYS`, #2607) and the sample's number
    (:mod:`.distributions`).

    The keys *solver*'s relations name are drawn together, conditioned on
    them (:meth:`.relations.RelationSolver.draw`, with the values the case
    holds: :func:`_relation_values`), from a seed of their own; the others
    each on their own as before.

    Raises:
        ValueError: If a base case (a binding, a route match or a case drawn
            from the ODD) or one of *arguments* already sets a key a
            distribution draws: the run would follow that value while the
            sweep said it was drawn.
    """
    from ..odd.control_fields import ODD_CASE_KEY  # noqa: PLC0415

    keys = set(distributions)
    for case in base_cases:
        clash = _clashes(keys, case)
        if clash:
            raise ValueError(
                f"sweep.constraints draws {clash} from a distribution, but the "
                "sweep's bindings, route or ODD sample set it too"
            )
    clash = _clashes(keys, arguments)
    if clash:
        raise ValueError(
            f"sweep.constraints draws {clash} from a distribution; an override "
            "may not set it too (drop the override, or the distribution)"
        )
    # The solver, when its relations relate a drawn key: those it draws.
    joint = solver if solver is not None and solver.related else None
    related = set(joint.related) if joint is not None else set()
    alone = {k: d for k, d in distributions.items() if k not in related}
    unseeded = _UNSEEDED_KEYS | {ODD_CASE_KEY}
    cases: list[list[str]] = []
    for case in base_cases:
        named = [o for o in case if _override_key(o) not in unseeded]
        held = (
            _relation_values(joint, [*case, *arguments], config_value)
            if joint is not None
            else {}
        )
        # The case's part of each draw's identity, hashed once.
        h = fnv1a("\x1f".join([str(seed), scenario, *named]))
        for sample in range(samples):
            h_sample = fnv1a(f"\x1fsample={sample}", h)
            values = draw_case(alone, h_sample)
            if joint is not None:
                rng = random.Random(
                    fnv1a("\x1frelations", h_sample) & 0x7FFFFFFFFFFFFFFF
                )
                values.update(joint.draw(held, rng))
            cases.append(
                [
                    *case,
                    # Rounded to 1e-9, which no scenario parameter resolves
                    # finer, so the override reads back as the value drawn.
                    *(
                        f"{k}={_override_value(round(v, 9))}"
                        for k, v in sorted(values.items())
                    ),
                ]
            )
    return cases


#: The setting a knob of a map-decided attribute writes: none a run reads.
#: It is taken off each drawn case once the case's road is chosen.
_MAP_KNOB_PREFIX = "_odd_map."


def _case_readings(
    lanelets: Sequence[int | None], definition: Any, lanelet_map: Any
) -> list[dict[str, tuple[str, Any]]]:
    """Each base case's ``(bucket, value)`` of every attribute the map decides
    on its lanelet (:func:`~autoware_carla_scenario.odd.route.lanelet_readings`),
    given by id (*lanelets*: a lanelet case's pick, a route case's spawn)."""
    from ..odd.route import lanelet_readings  # noqa: PLC0415

    graph = create_routing_graph(lanelet_map)
    found: list[dict[str, tuple[str, Any]]] = []
    for lanelet_id in lanelets:
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
    first passed over, put on the case nearest them
    (:func:`.case_choice.choose_cases`).
    """
    drawn = sampler.sample(wanted * _DRAWS_PER_CASE)
    kept = [s for s in drawn if any(_fits(s.buckets, b, attributes) for b in buckets)]
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
    lanelets: Sequence[int | None] | None = None,
    rules: Mapping[str, Any] | None = None,
    environment_rules: Any = None,
    arguments: Sequence[str] = (),
) -> list[list[str]]:
    """``odd_sample.count`` cases (as many as base cases by default), each a
    base case -- a lanelet or a route match -- and settings drawn from the ODD.

    With *lanelet_map*, the attributes the map decides are drawn as well and
    each case is put on a base case in the buckets drawn (the module
    docstring), so a base case may run more than once, or not at all; without
    it, the base cases are taken in turn.  The map reads each base case's
    lanelet, one of *lanelets* (the constraints' pick, not the lanelet a
    walk-back moved its spawn to), or, without them, its first
    ``*lanelet_id`` override (:func:`.case_choice.case_lanelet`).

    *arguments* are the run's own overrides: one that sets a key a drawn
    case sets is refused (:func:`_refuse_overridden`); one that sets the grip
    or the visibility, or its quantile, turns that draw off."""
    from openodd import ParameterSpace  # noqa: PLC0415

    from ..odd.control_fields import sets_any  # noqa: PLC0415
    from ..odd.registry import resolve_odd  # noqa: PLC0415
    from ..odd.sampler import (  # noqa: PLC0415
        ODD_CASE_KEY,
        QUANTILE_DRAWS,
        encode_case,
        sample_draws,
        sample_knobs,
        sample_odd,
        search_from_config,
    )

    raw = {str(k): v for k, v in odd_sample.items()}
    for name, key, _quantity, fixed in QUANTILE_DRAWS:
        # Read strictly first (check_sample_settings()): "ture" is refused
        # whatever the arguments say (#1756).  The run names the grip or the
        # visibility, or where it is: the case draws none, as a case that
        # names one keeps it.
        if sample_draws(raw, name) and sets_any(arguments, (key, *fixed)):
            raw[name] = False
    readings: list[dict[str, tuple[str, Any]]] = []
    buckets: list[dict[str, str]] = []
    map_attributes: list[str] = []
    if lanelet_map is not None:
        definition = resolve_odd(sample_odd(raw, odd))
        if lanelets is None:
            lanelets = [_case_lanelet(case) for case in base_cases]
        readings = _case_readings(lanelets, definition, lanelet_map)
        buckets = [{a: b for a, (b, _v) in r.items()} for r in readings]
        given = dict(sample_knobs(raw))
        map_knobs = _map_knobs(definition, buckets, given)
        if map_knobs:
            raw["knobs"] = {**map_knobs, **given}
            map_attributes = list(map_knobs)
    sampler, search, count = search_from_config(
        raw,
        odd=odd,
        controls=controls,
        drawn=drawn,
        rules=rules,
        environment_rules=environment_rules,
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
        indices, full = _choose_cases(
            [s.buckets for s in samples], buckets, map_attributes
        )
        # Every attribute the map decides is the lanelet's, a derived one too
        # (it has no knob, but it was derived from the buckets drawn).
        decided = sorted({*map_attributes, *(n for r in readings for n in r)})
        samples = [
            _as_run(sample, readings[i], decided) for sample, i in zip(samples, indices)
        ]
        cases = [base_cases[i] for i in indices]
    else:
        cases = [base_cases[i % len(base_cases)] for i in range(len(samples))]
    # Before the draws are recorded: a refused sweep draws nothing.
    _refuse_overridden(samples, arguments, raw, odd, controls, rules, environment_rules)
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


def _refuse_overridden(
    samples: Sequence[Any],
    arguments: Sequence[str],
    raw: Mapping[str, Any],
    odd: Any,
    controls: Mapping[str, Any] | None,
    rules: Mapping[str, Any] | None,
    environment_rules: Any,
) -> None:
    """Refuse *arguments* that set a key one of the drawn *samples* sets.

    The run's overrides come after a case's, so Hydra runs the argument's
    value -- while the case records the value drawn (``++odd_case``), its
    weather effects follow that value, coverage counts its bucket and the
    flaky search learns from it (#1995).  Pinning the value instead would
    need it read back into a bucket, which a value outside the ODD's has
    not; the sweep's own knobs pin it exactly, so the message says how.

    Raises:
        ValueError: Naming each key overridden, what draws it, and how to
            hold it instead.
    """
    keys = {_override_key(a) for a in arguments}
    if not keys:
        return
    drawn = {_override_key(o) for sample in samples for o in sample.overrides}
    clashes = sorted(keys & drawn)
    if not clashes:
        return
    from ..odd.registry import resolve_odd  # noqa: PLC0415
    from ..odd.sampler import (  # noqa: PLC0415
        case_settings,
        controls_from_mapping,
        sample_knobs,
        sample_odd,
        scenario_backend,
    )

    settings = case_settings(
        resolve_odd(sample_odd(raw, odd)),
        controls_from_mapping(sample_knobs(raw), "sweep.odd_sample.knobs"),
        controls_from_mapping(controls),
        scenario_backend(rules, environment_rules),
    )
    reasons: list[str] = []
    for key in clashes:
        kind, name = settings.get(key, ("", ""))
        if kind == "attribute":
            reasons.append(
                f"{key} is drawn by the ODD's {name}; pin it in the sweep "
                "instead of overriding it, with a knob for that attribute "
                f"in sweep.odd_sample.knobs ({{key: {key}, values: "
                "{<label>: <value>}} to draw one label, a narrower range for "
                "a number), or narrow the ODD, so that the case records, its "
                "weather follows and coverage counts what runs"
            )
        elif kind == "control":
            reasons.append(
                f"{key} is drawn with each case by the scenario's control "
                f"{name}, from its node of the rules; change that node "
                "(scenario_rules) instead of overriding it"
            )
        elif kind == "weather":
            reasons.append(
                f"{key} follows from the conditions each case draws, by the "
                f"weather rules' {name}; change the rules (environment.rules) "
                "or the conditions drawn instead of overriding it"
            )
        else:
            reasons.append(f"{key} is set by each case drawn from the ODD")
    raise ValueError(
        "sweep.odd_sample: the run's overrides set what a drawn case sets, "
        "and the run would follow them while the case records the value "
        "drawn: " + "; ".join(reasons)
    )


def check_sweep_shape(sweep_cfg: Any) -> None:
    """Refuse a config's ``sweep`` value (``OmegaConf.select(cfg, "sweep")``)
    that is neither null nor a mapping, resolving nothing in it.

    What a run that does not expand the sweep checks of it (``examples/run.py``,
    the lanelet sweeper before its step 0): an interpolation inside a sweep
    that only expanding it reads is not resolved, as before (#2279).

    Raises:
        ValueError: If *sweep_cfg* is neither null nor a mapping.
    """
    if sweep_cfg is None or isinstance(sweep_cfg, (DictConfig, Mapping)):
        return
    raise ValueError(
        "sweep must be a mapping (constraints, route, odd_sample, ...), "
        f"got {_shape(sweep_cfg)}"
    )


def sweep_section(sweep_cfg: Any) -> dict[str, Any] | None:
    """A config's ``sweep`` value (``OmegaConf.select(cfg, "sweep")``) as a
    plain dict, resolved; ``None`` when the config has no sweep.

    Only a missing ``sweep`` or null is no sweep: ``[]``, ``false``, ``0`` or
    ``""`` is the wrong shape, an easy slip in YAML that must not run the
    scenario once, as if it were concrete (#2279).  ``scenario-expand``, a
    ``--multirun`` batch (:func:`expand_config`), a batch run without one
    (``examples/run.py``) and the lanelet sweeper all read it this way.

    Raises:
        ValueError: If *sweep_cfg* is neither null nor a mapping
            (:func:`check_sweep_shape`).
    """
    check_sweep_shape(sweep_cfg)
    if sweep_cfg is None:
        return None
    if isinstance(sweep_cfg, DictConfig):
        container = OmegaConf.to_container(sweep_cfg, resolve=True)
    else:
        container = dict(sweep_cfg)
    assert isinstance(container, dict)  # a DictConfig's container is a dict
    return {str(key): value for key, value in container.items()}


def check_constraint_keys(
    sweep: Mapping[Any, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refuse a ``sweep.constraints`` with more than one target key, and a
    ``sweep.constraints``, ``sweep.bindings``, ``sweep.exclude``,
    ``sweep.route`` or ``sweep.odd_sample`` of the wrong shape.

    A sweep enumerates one parameter; the others a case needs are derived from
    its lanelet by ``sweep.bindings``.  A second key used to be ANDed into the
    first one's search and dropped, so it is refused rather than half honoured.
    ``sweep.constraints`` maps that key to a list of constraints,
    ``sweep.bindings`` maps each key it sets to a binding (a mapping), and
    ``sweep.exclude`` maps each key to a list of values (or null), and
    ``sweep.route`` and ``sweep.odd_sample`` are mappings (or null).
    A route search derives nothing from a pick, so ``sweep.bindings`` beside
    ``sweep.route`` is refused rather than silently dropped (#2454); so are,
    in a sweep that expands, bindings without a lanelet constraint,
    ``sweep.exclude`` without one or a route search, and a ``sweep.samples``
    above 1 without a distribution (#2607).
    Its other keys may each hold a distribution (:mod:`.distributions`),
    checked here with ``sweep.samples``, ``sweep.seed`` and
    ``sweep.relations`` (:mod:`.relations`).
    Callers run this before loading the map, so a bad config fails at once
    rather than after a clone.

    Returns:
        ``sweep.constraints`` split: the lanelet key with its constraints (at
        most one), and the keys drawn from a distribution, each parsed.

    Raises:
        ValueError: If ``sweep.constraints`` has more than one key, or one of
            those sections is not shaped as above (#1951), or a route search
            has bindings, or bindings, ``exclude`` or ``samples`` would be
            ignored (#2607).
    """
    # Only a missing section or null is empty: [], 0, false or "" is not a
    # mapping (#2134).
    for key, what in (("route", "a route search"), ("odd_sample", "its settings")):
        value = sweep.get(key)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(
                f"sweep.{key} must be {what} (a mapping), got {_shape(value)}"
            )
    constraints_cfg = _section(sweep, "constraints")
    if not isinstance(constraints_cfg, Mapping):
        raise ValueError(
            "sweep.constraints must map the one parameter it sweeps (such as "
            "ego.spawn_lanelet_id) to a list of constraints, got "
            f"{_shape(constraints_cfg)}"
        )
    for key, constraint_list in constraints_cfg.items():
        if not _is_list(constraint_list):
            raise ValueError(
                f"sweep.constraints.{key} must be a list of constraints "
                f"(- type: ...), got {_shape(constraint_list)}"
            )
    # A key drawn from a distribution is a scalar parameter of its own; the
    # lanelet constraints still sweep one key.
    lanelet, distributions = _split_constraints(sweep)
    if len(lanelet) > 1:
        raise ValueError(
            "sweep.constraints sweeps one parameter; got keys "
            f"{list(lanelet)}. Keep the constraints of one key and derive the "
            "others from its lanelet with sweep.bindings."
        )
    _samples(sweep.get("samples"))
    _draw_seed(sweep.get("seed"))
    _check_relations(sweep, distributions)
    bindings_cfg = _section(sweep, "bindings")
    if not isinstance(bindings_cfg, Mapping):
        raise ValueError(
            "sweep.bindings must map each parameter it sets to a binding "
            f"({{type: ...}}), got {_shape(bindings_cfg)}"
        )
    for key, binding_cfg in bindings_cfg.items():
        if not isinstance(binding_cfg, Mapping):
            raise ValueError(
                f"sweep.bindings.{key} must be a binding ({{type: ...}}), got "
                f"{_shape(binding_cfg)}"
            )
    if sweep.get("route") and bindings_cfg:
        raise ValueError(
            "sweep has both a route search and bindings; a route search "
            "places the ego itself (ego.spawn_*, ego.goal_*, scenario.route.*) "
            "and derives nothing from a pick, so drop sweep.bindings "
            f"({', '.join(str(key) for key in bindings_cfg)})"
        )
    exclude_cfg = _section(sweep, "exclude")
    if not isinstance(exclude_cfg, Mapping):
        raise ValueError(
            "sweep.exclude must map each parameter to the list of values a case "
            f"may not hold there, got {_shape(exclude_cfg)}"
        )
    for key, values in exclude_cfg.items():
        if values is not None and not _is_list(values):
            raise ValueError(
                f"sweep.exclude.{key} must be a list of values, got {_shape(values)}"
            )
    if constraints_cfg or sweep.get("route") or sweep.get("odd_sample"):
        _check_used(sweep, lanelet, distributions, bindings_cfg, exclude_cfg)
    return lanelet, distributions


def _check_used(
    sweep: Mapping[Any, Any],
    lanelet: Mapping[str, Any],
    distributions: Mapping[str, Any],
    bindings: Mapping[Any, Any],
    exclude: Mapping[Any, Any],
) -> None:
    """Refuse a ``sweep.bindings``, ``sweep.exclude`` or ``sweep.samples``
    (above 1) that nothing in the sweep would read, rather than ignore it
    (#2607): bindings derive from a lanelet constraint's pick, ``exclude``
    drops lanelet cases and route matches, and ``samples`` repeats the
    distributions' draws."""
    if bindings and not lanelet:
        raise ValueError(
            "sweep.bindings derive values from the lanelet a constraint picks, "
            "and the sweep has no lanelet constraint; drop sweep.bindings "
            f"({', '.join(str(key) for key in bindings)})"
        )
    if exclude and not (lanelet or sweep.get("route")):
        raise ValueError(
            "sweep.exclude drops lanelet cases and route matches, and the sweep "
            "has neither a lanelet constraint nor a route search; drop "
            f"sweep.exclude ({', '.join(str(key) for key in exclude)})"
        )
    if _samples(sweep.get("samples")) > 1 and not distributions:
        raise ValueError(
            "sweep.samples repeats each case with values drawn from the "
            "distributions in sweep.constraints, and the sweep has none; drop "
            "sweep.samples"
        )


def _check_relations(
    sweep: Mapping[Any, Any], distributions: Mapping[str, Any]
) -> None:
    """Refuse a ``sweep.relations`` that does not parse, compares a drawn
    parameter with a string or a bool, has nothing to apply to (no
    constraints, route or ODD sample), or -- naming only drawn parameters --
    cannot hold within their distributions' bounds (Z3, :mod:`.relations`):
    before the map is loaded.  Relations naming a value of the case are
    solved for each case instead."""
    relations = parse_relations(sweep.get("relations"))
    if not relations:
        return
    if not (sweep.get("constraints") or sweep.get("route") or sweep.get("odd_sample")):
        raise ValueError(
            "sweep.relations apply to the cases a sweep expands to, and the "
            "sweep has no constraints, route or odd_sample to expand"
        )
    solver = RelationSolver(relations, distributions)
    if not solver.constant_keys:
        solver.check()


def _section(sweep: Mapping[Any, Any], key: str) -> Any:
    """``sweep[key]``, or ``{}`` when it is missing or null: a falsy value of
    another type (``[]``, ``0``, ``false``, ``""``) is kept, for the caller to
    refuse as the wrong shape."""
    value = sweep.get(key)
    return {} if value is None else value


def _is_list(value: Any) -> bool:
    """Whether *value* is a list as YAML writes one (a list or a tuple)."""
    return isinstance(value, (list, tuple, ListConfig))


def _shape(value: Any) -> str:
    """*value* as an error about its shape shows it: a mapping or a list
    named as one, then the value itself."""
    if isinstance(value, Mapping):
        return f"a mapping, {shown(value)}"
    if _is_list(value):
        return f"a list, {shown(value)}"
    return shown(value)


def _lanelet_cases(
    sweep: Mapping[Any, Any], lanelet_map: Any
) -> tuple[list[list[str]], list[int]]:
    """One override list per lanelet the constraints match, bindings applied,
    and the pick each one is for.

    Picks whose cases end the same -- a walk-back moving the target key to
    the same predecessor -- give that case once, the first pick's (#1950),
    which is the pick it is returned with.  The pick is the lanelet the ODD
    reads the case's road on (docs/odd.md, "The road a case is on"): a
    walk-back moves the target key off it, so the case's overrides do not
    name it (#2454).

    *sweep* has passed :func:`check_constraint_keys`.

    Raises:
        ValueError: If the constraints, bindings or exclusions do not parse,
            or the constraints are empty.
    """
    constraints_cfg = _split_constraints(sweep)[0]
    # Constraints are keyed by the target parameter (e.g. ego.spawn_lanelet_id);
    # each value is a list of constraint dicts.  That there is one such key,
    # and each section's shape, expand_sweep (the only caller) has checked
    # already (check_constraint_keys).
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
        return [], []

    bindings: list[Binding] = [
        parse_binding(key, b_cfg)
        for key, b_cfg in (sweep.get("bindings") or {}).items()
    ]
    excluded = _excluded_overrides(sweep)
    cases: list[list[str]] = []
    case_picks: list[int] = []
    # Each distinct case, by its overrides, and the picks that end as it.
    picks_of: dict[tuple[str, ...], list[int]] = {}
    # Why picks were skipped, for the warning when every one was.
    binding_failures = 0
    excluded_picks = 0
    for lid in matched_ids:
        # Key -> (value, who wrote it).  The pick is only the target key's
        # default: a walk-back of the target's own spawn moves it.  Every other
        # write -- a binding's value, or a walk-back moving its spawn's
        # lanelet -- must agree with any other write to the same key.
        values: dict[str, tuple[str, str | None]] = {target_key: (str(lid), None)}
        # Each key's value as written, for sweep.exclude to read a list's
        # elements (written in the same order as *values*, so the last wins).
        raw: dict[str, Any] = {target_key: lid}
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
                binding_failures += 1
                break
            _write_override(
                values,
                binding.target_key,
                _override_value(result.value),
                f"binding {binding.target_key}",
                lid,
            )
            raw[binding.target_key] = result.value
            if result.lanelet_id_override is not None:
                moved = _walk_back_lanelet_key(
                    binding.target_key, result.lanelet_id_override
                )
                _write_override(
                    values,
                    moved,
                    str(result.lanelet_id_override),
                    f"the walk-back of binding {binding.target_key}",
                    lid,
                )
                raw[moved] = result.lanelet_id_override
        else:
            hit = _excluded_key(raw, excluded)
            if hit is not None:
                logger.info(
                    "Lanelet %d's case ends with %s=%s, which sweep.exclude "
                    "names; skipping this lanelet.",
                    lid,
                    hit,
                    values[hit][0],
                )
                excluded_picks += 1
                continue
            case = [f"{key}={value}" for key, (value, _) in values.items()]
            # A walk-back can move the target key off the pick, so picks that
            # walk back to the same lanelet can end as the same case: it runs
            # once, as the first pick's (#1950).
            picks = picks_of.setdefault(tuple(case), [])
            if not picks:
                cases.append(case)
                case_picks.append(lid)
            picks.append(lid)
    if not cases:
        logger.warning(
            "All %d matched lanelets were skipped: a binding failed for %d, "
            "and sweep.exclude names the case of %d.",
            len(matched_ids),
            binding_failures,
            excluded_picks,
        )
    for case_key, picks in picks_of.items():
        if len(picks) > 1:
            logger.info(
                "Lanelets %s give the same case %s once their bindings are "
                "applied; it runs once, as lanelet %d's.",
                ", ".join(str(pick) for pick in picks),
                list(case_key),
                picks[0],
            )
    return cases, case_picks


def _excluded_overrides(sweep: Mapping[Any, Any]) -> dict[str, frozenset[str]]:
    """``sweep.exclude``, each key's values rendered as the case's overrides
    hold them (:func:`_override_value`), so a case is checked on what it runs.

    Its shape -- a mapping of lists -- is checked by :func:`check_constraint_keys`,
    which :func:`expand_sweep` runs first.
    """
    return {
        str(key): frozenset(_override_value(v) for v in (values or ()))
        for key, values in (sweep.get("exclude") or {}).items()
    }


def _excluded_key(
    values: Mapping[str, Any], excluded: Mapping[str, frozenset[str]]
) -> str | None:
    """The first key of a case's *values* (each as written, before it is
    rendered) that holds a value ``sweep.exclude`` names (*excluded*, from
    :func:`_excluded_overrides`), or ``None``.

    A list value is excluded when the whole list is named, or any one of its
    elements: ``scenario.route.lanelet_ids: ${map.no_3d_model_lanelet_ids}``
    drops a route through a lanelet CARLA does not draw (#2454).
    """
    for key, value in values.items():
        names = excluded.get(key)
        if not names:
            continue
        if _override_value(value) in names:
            return key
        if isinstance(value, (list, tuple)) and any(
            _override_value(element) in names for element in value
        ):
            return key
    return None


def expand_route(
    route: Mapping[str, Any],
    lanelet_map: Any,
    arguments: Sequence[str] = (),
    *,
    exclude: Mapping[Any, Any] | None = None,
) -> list[list[str]]:
    """The override list of every route of ``lanelet_map`` a route search matches.

    Per match, in the order :func:`~autoware_carla_scenario.route.search.find_route_matches`
    returns them: the ego's spawn (``ego.spawn_lanelet_id`` / ``ego.spawn_s``,
    ``ego_spawn_s`` along the route), its goal (``ego.goal_lanelet_id`` /
    ``ego.goal_s``, the route's end less ``ego_goal_margin``) when
    ``ego_goal`` is set, the match as the ``scenario.route.*`` keys, and how
    many cases the search expanded to (``scenario.route.case_count``).  A
    match the ego cannot be placed on is skipped (and logged), and not
    counted; so is one that holds a value *exclude* (``sweep.exclude``)
    names on one of those keys -- a spawn or goal lanelet, or a lanelet of
    ``scenario.route.lanelet_ids`` (any element of a list) -- #2454.

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
    excluded = _excluded_overrides({"exclude": exclude})
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
        values: dict[str, Any] = {
            "ego.spawn_lanelet_id": spawn_id,
            "ego.spawn_s": round(spawn_s, 4),
        }
        if goal is not None:
            values["ego.goal_lanelet_id"] = goal[0]
            values["ego.goal_s"] = round(goal[1], 4)
        for key, value in match.to_config().items():
            values[f"scenario.route.{key}"] = value
        hit = _excluded_key(values, excluded)
        if hit is not None:
            logger.info(
                "Route match %d (%s) holds %s=%s, which sweep.exclude names; "
                "skipping it.",
                match.index,
                list(match.lanelet_ids),
                hit,
                _override_value(values[hit]),
            )
            continue
        cases.append([f"{key}={_override_value(v)}" for key, v in values.items()])
    # Every case says how many there are on this map: what a logical
    # scenario's gate counts its case coverage against (docs/gate.md).
    count = f"scenario.route.case_count={len(cases)}"
    return [[*overrides, count, *arguments] for overrides in cases]


#: Why a sweep that expands to no case is refused -- by the lanelet sweeper,
#: a ``--multirun`` batch, the viewer and ``scenario-expand`` alike (#1407).
#: Exit 0 means every case passed: a sweep that tests nothing (a misspelt
#: constraint, the wrong map) must not look like one. A config without a
#: sweep is not refused: it expands to itself, one case.
EMPTY_SWEEP_MESSAGE = (
    "the sweep expands to no case (its constraints, route or ODD sample match "
    "nothing on this map)"
)


def expand_config(
    cfg: DictConfig,
    arguments: Sequence[str] = (),
    *,
    drawn: OddDraws | None = None,
    composed: Sequence[str] = (),
) -> list[list[str]]:
    """The concrete scenarios of a composed scenario config (see module docstring).

    *drawn* is a batch's record of the cases drawn from the ODD so far
    (see :func:`expand_sweep`).  *composed* are the overrides *cfg* was
    composed with when they are not *arguments* (the caller adds them to
    each case itself): a drawn case may not set what they set either
    (:func:`expand_sweep`).

    Raises:
        ControlsError: When the config's ``controls`` are invalid, whether or
            not it has a sweep -- as ``scenario-check`` refuses them
            (:func:`scenario_controls`).
        ValueError: When the sweep itself is invalid -- not a mapping
            (:func:`sweep_section`), a ``case_index`` out
            of range, a route search beside constraints, more than one
            constraint key, an empty constraints section, or a constraint or
            binding the parsers refuse (:func:`expand_sweep`) -- or when the
            run's ``seed``, or ``sweep.odd_sample``'s ``seed`` or ``count``,
            is not a whole number, or an override sets what a case drawn
            from the ODD sets (a negative seed, a count below 1), checked
            before any map is read -- or when the Lanelet2 map a constraint or
            route sweep needs is not named, is missing, or cannot be read,
            whether ``map.lanelet2_path``, ``map.directory`` or ``map.source``
            names it (a map that cannot be resolved or fetched;
            :func:`load_sweep_map`).
    """
    # Checked before anything else, a concrete config's too: the check needs
    # only the config, no map, ODD or CARLA.
    controls = scenario_controls(cfg)
    rules = scenario_rules(cfg)
    weather = environment_rules(cfg)
    # And the run's seed, which every case runs with (a run refuses it too).
    seed = _config_seed(cfg)
    # A ``sweep`` that is not a mapping is refused, not run as concrete.
    sweep = sweep_section(OmegaConf.select(cfg, "sweep")) or {}
    # Before the sweep is judged empty: a ``constraints: []`` is the wrong
    # shape, not an empty sweep (#2134).
    lanelet_constraints = check_constraint_keys(sweep)[0]
    if not (sweep.get("constraints") or sweep.get("route") or sweep.get("odd_sample")):
        return [list(arguments)]  # already concrete

    odd = OmegaConf.select(cfg, "odd")
    # Before any map is read, as the constraint keys are.
    odd_sample = sweep.get("odd_sample")
    if isinstance(odd_sample, Mapping):
        from ..odd.sampler import check_sample_settings  # noqa: PLC0415

        check_sample_settings(odd_sample)
    scenario = scenario_name(cfg)
    if not (lanelet_constraints or sweep.get("route")):
        # Drawing from the ODD or from distributions alone needs no map.
        return expand_sweep(
            sweep,
            None,
            arguments,
            odd=odd,
            controls=controls,
            drawn=drawn,
            seed=seed,
            rules=rules,
            environment_rules=weather,
            composed=composed,
            scenario=scenario,
            config_value=config_lookup(cfg),
        )

    lanelet_map = load_sweep_map(OmegaConf.select(cfg, "map"))
    return expand_sweep(
        sweep,
        lanelet_map,
        arguments,
        odd=odd,
        controls=controls,
        drawn=drawn,
        seed=seed,
        rules=rules,
        environment_rules=weather,
        composed=composed,
        scenario=scenario,
        config_value=config_lookup(cfg),
    )


def load_sweep_map(map_cfg: Any, *, prefix: str = "") -> Any:
    """Resolve and load the Lanelet2 map a sweep's ``map`` config names.

    ``scenario-expand``, a ``--multirun`` batch (:func:`expand_config`) and
    the Hydra sweeper (``LaneletConstraintSweeper``) all load a sweep's map
    this way, so each refuses a map that is missing or unreadable alike: as
    an invalid sweep, an authoring error of this scenario that its caller
    reports by name, not a crash (#1444, #1447).

    Args:
        map_cfg: The config's ``map`` node (``None`` when it has none).
        prefix: Put before the message, to say who refuses the map.

    Raises:
        ValueError: Naming the map, its cause the original error, when it is
            not named, is missing or cannot be read, whether
            ``map.lanelet2_path``, ``map.directory`` or ``map.source`` names
            it -- a map that cannot be resolved or fetched (MapSourceError,
            MapResolutionError, MapCacheError), an ``OSError`` resolving it
            (the map cache's root not a directory, say), a Lanelet2 file that
            is missing (``FileNotFoundError``) or that Lanelet2 cannot read
            (any ``OSError``, or its parser's ``RuntimeError``), or a
            content pack (``map.pack``) that cannot be fetched or does not
            carry the map (MapPackError, PackAccessError: no GitHub token,
            say, #2198); or naming
            ``map.<key>`` and the value, when a setting is refused
            (:func:`~autoware_carla_scenario.maps.config.map_setting`, a
            boolean or a list for a path, #2165).
    """
    from ..maps import (  # noqa: PLC0415 -- clones a map on demand
        MapCacheError,
        MapResolutionError,
        MapSourceError,
        resolve_map_paths,
    )
    from ..maps.pack import MapPackError  # noqa: PLC0415 -- PackAccessError's base
    from .map_loader import load_map  # noqa: PLC0415

    try:
        paths = resolve_map_paths(map_cfg)
    except MapSourceError as exc:  # only map.source is parsed as a URI (#2279)
        raise ValueError(f"{prefix}map.source: {exc}") from exc
    except (MapResolutionError, MapCacheError) as exc:
        raise ValueError(f"{prefix}{exc}") from exc  # each names the map
    except MapPackError as exc:  # a RuntimeError, PackAccessError too (#2198)
        raise ValueError(
            f"{prefix}Could not load the map{_map_label(map_cfg)}: {exc}"
        ) from exc
    except ValueError as exc:  # a map.* setting refused, naming it (#2165)
        raise ValueError(f"{prefix}{exc}") from exc
    except OSError as exc:  # the map cache's own directory, say
        raise ValueError(
            f"{prefix}Could not resolve the map{_map_label(map_cfg)}: {exc}"
        ) from exc
    try:
        return load_map(paths)
    except FileNotFoundError as exc:  # names the file, or says none is named
        raise ValueError(f"{prefix}{exc}") from exc
    except (OSError, RuntimeError) as exc:  # RuntimeError: Lanelet2's parser
        raise ValueError(
            f"{prefix}Could not read the Lanelet2 map {paths.lanelet2_path}: {exc}"
        ) from exc


def _map_label(map_cfg: Any) -> str:
    """`` map.<key>=<value>`` for the setting *map_cfg*'s map is resolved
    from -- ``map.directory`` wins over ``map.pack``, and that over
    ``map.source``, as in
    :func:`~autoware_carla_scenario.maps.resolve_map_paths` -- or its
    ``map.name``; ``""`` when none is set."""
    if not isinstance(map_cfg, DictConfig):
        return ""
    for key in ("directory", "pack", "source", "name"):
        value = OmegaConf.select(map_cfg, key)
        if value is not None and str(value).strip():
            return f" map.{key}={value}"
    return ""


def scenario_name(cfg: DictConfig) -> str:
    """The scenario's name (``scenario.name``), which the draws of its
    distributions are named by (:func:`expand_sweep`); empty when it has none."""
    name = OmegaConf.select(cfg, "scenario.name", default=None)
    return "" if name is None else str(name)


def _config_seed(cfg: DictConfig) -> int | None:
    """The run's seed, the config's top-level ``seed``; ``None`` when it has none
    (:mod:`autoware_carla_scenario.seed`).

    Raises:
        ValueError: When it is not a non-negative whole number
            (:func:`~autoware_carla_scenario.odd.sampler.seed_setting`).
    """
    from ..odd.sampler import seed_setting  # noqa: PLC0415 -- imports openodd

    return seed_setting("seed", OmegaConf.select(cfg, "seed"))


def scenario_controls(cfg: DictConfig) -> dict[str, Any]:
    """The config's ``controls``, resolved: what the scenario's parameters set.

    Raises:
        ControlsError: When they are invalid; its ``diagnostics`` give each
            problem with its config path (:func:`~..odd.controls.validate_controls`).
    """
    from ..odd.controls import ControlsError, validate_controls  # noqa: PLC0415

    node = OmegaConf.select(cfg, "controls")
    if node is None:
        return {}
    controls = OmegaConf.to_container(node, resolve=True)
    errors = [d for d in validate_controls(controls, cfg) if d.severity == "error"]
    if errors:
        raise ControlsError(errors)
    return {str(k): v for k, v in controls.items()}  # type: ignore[union-attr]


def scenario_rules(cfg: DictConfig) -> dict[str, Any] | None:
    """The config's ``scenario_rules``, resolved: the nodes the scenario adds
    to the rules (a rule document, openodd's docs/rules.md, "A scenario's
    nodes"), or ``None`` when it adds none.

    Raises:
        ValueError: When it is not a mapping.
    """
    node = OmegaConf.select(cfg, "scenario_rules")
    if node is None:
        return None
    rules = OmegaConf.to_container(node, resolve=True)
    if not isinstance(rules, dict):
        raise ValueError(f"scenario_rules must be a rule document, got {rules!r}")
    return {str(k): v for k, v in rules.items()}


def environment_rules(cfg: DictConfig) -> Any:
    """The config's ``environment.rules``, resolved: the weather rules its
    runs follow (a file, or ``{default, vehicles}``), or ``None`` for the
    default ones.  A sweep draws its cases' CARLA weather by them
    (:func:`~autoware_carla_scenario.odd.sampler.scenario_backend`), as
    each run does."""
    node = OmegaConf.select(cfg, "environment.rules")
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True)
    return node


__all__ = [
    "EMPTY_SWEEP_MESSAGE",
    "check_constraint_keys",
    "environment_rules",
    "expand_config",
    "expand_route",
    "expand_sweep",
    "scenario_controls",
    "scenario_name",
    "check_sweep_shape",
    "scenario_rules",
    "sweep_section",
]
