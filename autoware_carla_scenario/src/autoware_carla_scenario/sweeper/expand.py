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
from typing import TYPE_CHECKING, Any, Mapping, Sequence

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

    With ``sweep.case_index``, only the case at that (0-based) index is kept.

    Raises:
        ValueError: If ``sweep`` has none of constraints, a route search and
            ``odd_sample``, or both constraints and a route search, or
            constraints for more than one target key, or a ``case_index``
            that is not a non-negative whole number or is past the cases it
            expands to, or an argument sets a key a drawn case sets.
    """
    check_constraint_keys(sweep)
    index = _case_index(sweep.get("case_index"))
    odd_sample = sweep.get("odd_sample") or {}
    if odd_sample and odd_sample.get("seed") is None:
        # A run without a seed draws as one with DEFAULT_SEED does.
        odd_sample = {
            **odd_sample,
            "seed": DEFAULT_SEED if seed is None else seed,
        }
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
            rules=rules,
            environment_rules=environment_rules,
            arguments=[*arguments, *composed],
        )
    if index is not None:
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
    rules: Mapping[str, Any] | None = None,
    environment_rules: Any = None,
    arguments: Sequence[str] = (),
) -> list[list[str]]:
    """``odd_sample.count`` cases (as many as base cases by default), each a
    base case -- a lanelet or a route match -- and settings drawn from the ODD.

    With *lanelet_map*, the attributes the map decides are drawn as well and
    each case is put on a base case in the buckets drawn (the module
    docstring), so a base case may run more than once, or not at all; without
    it, the base cases are taken in turn.

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


def _override_key(override: str) -> str:
    """The config key a Hydra override names: ``a.b`` of ``a.b=1``,
    ``+a.b=1``, ``++a.b=1`` and ``~a.b``."""
    return override.lstrip("~+").split("=", 1)[0]


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
        scenario_backend,
    )

    settings = case_settings(
        resolve_odd(raw.get("odd") or odd),
        controls_from_mapping(raw.get("knobs")),
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


def check_constraint_keys(sweep: Mapping[Any, Any]) -> None:
    """Refuse a ``sweep.constraints`` with more than one target key, and a
    ``sweep.constraints``, ``sweep.bindings`` or ``sweep.exclude`` of the
    wrong shape.

    A sweep enumerates one parameter; the others a case needs are derived from
    its lanelet by ``sweep.bindings``.  A second key used to be ANDed into the
    first one's search and dropped, so it is refused rather than half honoured.
    ``sweep.constraints`` maps that key to a list of constraints,
    ``sweep.bindings`` maps each key it sets to a binding (a mapping), and
    ``sweep.exclude`` maps each key to a list of values (or null).
    Callers run this before loading the map, so a bad config fails at once
    rather than after a clone.

    Raises:
        ValueError: If ``sweep.constraints`` has more than one key, or one of
            those sections is not shaped as above (#1951).
    """
    constraints_cfg = sweep.get("constraints") or {}
    if not isinstance(constraints_cfg, Mapping):
        raise ValueError(
            "sweep.constraints must map the one parameter it sweeps (such as "
            "ego.spawn_lanelet_id) to a list of constraints, got "
            f"{_shape(constraints_cfg)}"
        )
    keys = [str(key) for key in constraints_cfg]
    if len(keys) > 1:
        raise ValueError(
            "sweep.constraints sweeps one parameter; got keys "
            f"{keys}. Keep the constraints of one key and derive the others "
            "from its lanelet with sweep.bindings."
        )
    for key, constraint_list in constraints_cfg.items():
        if not _is_list(constraint_list):
            raise ValueError(
                f"sweep.constraints.{key} must be a list of constraints "
                f"(- type: ...), got {_shape(constraint_list)}"
            )
    bindings_cfg = sweep.get("bindings") or {}
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
    exclude_cfg = sweep.get("exclude") or {}
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


def _write_override(
    values: dict[str, tuple[str, str | None]],
    key: str,
    value: str,
    writer: str,
    lanelet_id: int,
) -> None:
    """Set *key* to *value* in *values*, refusing a write that disagrees.

    A key without a writer holds the pick, which any write replaces.

    Raises:
        ValueError: If *key* was already written by another writer with a
            different value.
    """
    previous = values.get(key)
    if previous is not None and previous[1] is not None and previous[0] != value:
        raise ValueError(
            f"sweep.bindings disagree about {key} on lanelet {lanelet_id}: "
            f"{previous[1]} sets it to {previous[0]}, {writer} to {value}. "
            "Give a stop_line_approach binding the offset of its spawn's "
            "stop_line_offset, or drop one of the two."
        )
    values[key] = (value, writer)


def _expand_lanelets(sweep: Mapping[Any, Any], lanelet_map: Any) -> list[list[str]]:
    """One override list per lanelet the constraints match, bindings applied.

    Picks whose cases end the same -- a walk-back moving the target key to
    the same predecessor -- give that case once, the first pick's (#1950).

    Raises:
        ValueError: If the constraints, bindings or exclusions are of the
            wrong shape or do not parse, or the constraints are empty.
    """
    constraints_cfg = sweep.get("constraints") or {}
    # Constraints are keyed by the target parameter (e.g. ego.spawn_lanelet_id);
    # each value is a list of constraint dicts.  A sweep enumerates one
    # parameter: the others a case needs are derived from its lanelet by the
    # bindings.  A second key used to be ANDed into the first one's search and
    # dropped, so it is refused rather than half honoured.
    check_constraint_keys(sweep)
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
    excluded = _excluded_overrides(sweep)
    cases: list[list[str]] = []
    # Each distinct case, by its overrides, and the picks that end as it.
    picks_of: dict[tuple[str, ...], list[int]] = {}
    for lid in matched_ids:
        # Key -> (value, who wrote it).  The pick is only the target key's
        # default: a walk-back of the target's own spawn moves it.  Every other
        # write -- a binding's value, or a walk-back moving its spawn's
        # lanelet -- must agree with any other write to the same key.
        values: dict[str, tuple[str, str | None]] = {target_key: (str(lid), None)}
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
            _write_override(
                values,
                binding.target_key,
                _override_value(result.value),
                f"binding {binding.target_key}",
                lid,
            )
            if result.lanelet_id_override is not None:
                _write_override(
                    values,
                    _walk_back_lanelet_key(
                        binding.target_key, result.lanelet_id_override
                    ),
                    str(result.lanelet_id_override),
                    f"the walk-back of binding {binding.target_key}",
                    lid,
                )
        else:
            hit = next(
                (
                    key
                    for key, (value, _) in values.items()
                    if value in excluded.get(key, ())
                ),
                None,
            )
            if hit is not None:
                logger.info(
                    "Lanelet %d's case ends with %s=%s, which sweep.exclude "
                    "names; skipping this lanelet.",
                    lid,
                    hit,
                    values[hit][0],
                )
                continue
            case = [f"{key}={value}" for key, (value, _) in values.items()]
            # A walk-back can move the target key off the pick, so picks that
            # walk back to the same lanelet can end as the same case: it runs
            # once, as the first pick's (#1950).
            picks = picks_of.setdefault(tuple(case), [])
            if not picks:
                cases.append(case)
            picks.append(lid)
    if not cases:
        logger.warning("All lanelets were skipped due to binding failures.")
    for case_key, picks in picks_of.items():
        if len(picks) > 1:
            logger.info(
                "Lanelets %s give the same case %s once their bindings are "
                "applied; it runs once, as lanelet %d's.",
                ", ".join(str(pick) for pick in picks),
                list(case_key),
                picks[0],
            )
    return cases


def _excluded_overrides(sweep: Mapping[Any, Any]) -> dict[str, frozenset[str]]:
    """``sweep.exclude``, each key's values rendered as the case's overrides
    hold them (:func:`_override_value`), so a case is checked on what it runs.

    Its shape -- a mapping of lists -- is checked by :func:`check_constraint_keys`,
    which :func:`_expand_lanelets` runs first.
    """
    return {
        str(key): frozenset(_override_value(v) for v in (values or ()))
        for key, values in (sweep.get("exclude") or {}).items()
    }


def expand_route(
    route: Mapping[str, Any], lanelet_map: Any, arguments: Sequence[str] = ()
) -> list[list[str]]:
    """The override list of every route of ``lanelet_map`` a route search matches.

    Per match, in the order :func:`~autoware_carla_scenario.route.search.find_route_matches`
    returns them: the ego's spawn (``ego.spawn_lanelet_id`` / ``ego.spawn_s``,
    ``ego_spawn_s`` along the route), its goal (``ego.goal_lanelet_id`` /
    ``ego.goal_s``, the route's end less ``ego_goal_margin``) when
    ``ego_goal`` is set, the match as the ``scenario.route.*`` keys, and how
    many cases the search expanded to (``scenario.route.case_count``).  A
    match the ego cannot be placed on is skipped (and logged), and not
    counted.

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
        cases.append(overrides)
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
        ValueError: When the sweep itself is invalid -- a ``case_index`` out
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
    sweep_cfg = OmegaConf.select(cfg, "sweep")
    sweep = (
        OmegaConf.to_container(sweep_cfg, resolve=True) if sweep_cfg is not None else {}
    )
    if not isinstance(sweep, dict) or not (
        sweep.get("constraints") or sweep.get("route") or sweep.get("odd_sample")
    ):
        return [list(arguments)]  # already concrete

    check_constraint_keys(sweep)
    odd = OmegaConf.select(cfg, "odd")
    # Before any map is read, as the constraint keys are.
    odd_sample = sweep.get("odd_sample")
    if isinstance(odd_sample, Mapping):
        from ..odd.sampler import check_sample_settings  # noqa: PLC0415

        check_sample_settings(odd_sample)
    if not (sweep.get("constraints") or sweep.get("route")):
        # Drawing from the ODD alone needs no map.
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
            (any ``OSError``, or its parser's ``RuntimeError``).
    """
    from ..maps import (  # noqa: PLC0415 -- clones a map on demand
        MapCacheError,
        MapResolutionError,
        MapSourceError,
        resolve_map_paths,
    )
    from .map_loader import load_map  # noqa: PLC0415

    try:
        paths = resolve_map_paths(map_cfg)
    except (MapSourceError, MapResolutionError, MapCacheError) as exc:
        raise ValueError(f"{prefix}{exc}") from exc  # each names the map
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
    from -- ``map.directory`` wins over ``map.source``, as in
    :func:`~autoware_carla_scenario.maps.resolve_map_paths` -- or its
    ``map.name``; ``""`` when none is set."""
    if not isinstance(map_cfg, DictConfig):
        return ""
    for key in ("directory", "source", "name"):
        value = OmegaConf.select(map_cfg, key)
        if value is not None and str(value).strip():
            return f" map.{key}={value}"
    return ""


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
    "scenario_rules",
]
