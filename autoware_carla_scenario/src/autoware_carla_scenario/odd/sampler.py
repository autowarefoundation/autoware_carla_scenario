"""Sample concrete scenarios from an ODD (openodd's sampler, with this framework's knobs).

What the sampler can draw are the attributes a run *sets*, and three things
say which, none of them knowing the others:

* the ODD maps an attribute onto a scenario measure
  (:func:`~.scenario_measure.scenario_measure`, ``measure:`` in a binding
  file) -- its taxonomy onto the framework's fixed measure keys;
* the scenario says which of its parameters sets a measure, and what it can
  stage, under ``controls`` in its config -- keyed by measure, not by the
  ODD's taxonomy::

      controls:
        vehicle_ahead_gap_m: {key: scenario.npc_gap_m, target: measure, range: [2, 30]}

  A control is a raw setting by default (``target: setting``): the drawn
  value is written as ``value * scale + offset``, a mapping the scenario's
  author vouches for.  ``target: measure`` says the key takes the measure
  itself, as it should read at the start, and the scenario solves for the
  placement that makes it so (:mod:`autoware_carla_scenario.placement`):
  the value is written unchanged, so ``scale``, ``offset``, ``integer`` and
  ``values`` are refused (:func:`controls_from_mapping`);

* the conditions -- the rainfall rate, the visibility through the fog, the
  time of day -- are set for every scenario through the ``environment``
  config, which the built-in probes :func:`~.probes.rainfall_rate_mm_h`,
  :func:`~.probes.fog_visibility_range_m` and :func:`~.probes.time_of_day`
  have knobs on (:data:`DEFAULT_KNOBS`); CARLA's weather fields follow from
  them by the weather rules, drawn with each case as effects at a quantile of
  their own (:func:`~.backend.weather_effects`).

A sweep's ``knobs``, keyed by the ODD's attributes, replace them attribute by
attribute. Attributes nothing sets are left open. See openodd's
:mod:`openodd.sampler` for how cases are drawn.

The road's grip is no attribute of an ODD but follows one: the water the
conditions drawn leave on the road bounds it
(:func:`~autoware_carla_scenario.actions.environment.tire_friction_range`).
Each case is given where in that range its grip is,
``environment.tire_friction_quantile``, a parameter of the case
(``quantile(tire_friction_quantile)``, :data:`QUANTILE_DRAWS`): a root node
of the rules, uniform on [0, 1], written by an effect as the weather fields
are, so the same sweep draws the same grip and the flaky search moves it
(#1995); ``tire_friction: false`` in the sweep leaves it at the middle.
(:func:`with_tire_friction` draws one for cases no sampler drew.)

How far drivers see follows the conditions the same way (the weather rules'
``visibility``): night, rain and fog shorten it.  Each case is given where in that range its visibility
is, ``environment.visibility_quantile`` (``quantile(visibility_quantile)``),
drawn the same way; ``visibility: false`` in the sweep leaves it at the middle.  A
traffic backend that models drivers' sight (``sumo``) holds its traffic to
it.  An ODD that names the visibility itself (the ``visibility`` probe) sets
it through its knob, ``environment.visibility_m``, whatever the weather.

With ``weights`` in the sweep (module -> share, or a YAML file of them),
openodd's weighted strategy puts that share of the cases in each module
named: ``{heavy_rain: 0.3}`` draws 30% of them in heavy rain
(:func:`weights_from_config`).

With ``flaky`` in the sweep, openodd's :class:`~openodd.FlakySearch` draws
part of each batch where a small change flips the outcome, learning from the
runs in ``coverage_from`` that recorded their case and outcome. Every drawn
case travels to its run encoded (:func:`encode_case`), and the run records it,
with its outcome, in its coverage file.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from openodd.flaky import FlakyMetrics, FlakySearch, ParameterSpace
from openodd.sampler import OddDraws, OddKnob, OddSample, knobs_from_mapping
from openodd.sampler import OddSampler as _OddSampler

from ..config_flags import check_flag
from ..route.model import _whole_number
from ..seed import DEFAULT_SEED, check_seed
from openodd import Rules, RulesError

from .backend import (
    CARLA_BACKEND,
    DEFAULT_KNOBS,
    CarlaBackend,
    _effect_control,
    weather_effects,
)
from .control_fields import (
    CONTROL_TARGETS,
    FLAKY_FIELDS,
    MAPPING_FIELDS,
    ODD_CASE_KEY,
    SAMPLE_FIELDS,
    TIRE_FRICTION_KEY,
    VISIBILITY_KEY,
    sets_any,
)
from .model import OddDefinition

__all__ = [
    "DEFAULT_KNOBS",
    "ODD_CASE_KEY",
    "OddDraws",
    "OddCase",
    "OddKnob",
    "OddSample",
    "OddSampler",
    "decode_case",
    "encode_case",
    "CONTROL_TARGETS",
    "controls_from_mapping",
    "knobs_from_mapping",
    "check_sample_settings",
    "case_settings",
    "QUANTILE_DRAWS",
    "sample_count",
    "sample_draws",
    "sample_seed",
    "sampler_from_config",
    "scenario_backend",
    "seed_setting",
    "search_from_config",
    "TIRE_FRICTION_KEY",
    "weights_from_config",
    "VISIBILITY_KEY",
    "whole_number_setting",
    "with_tire_friction",
    "with_visibility",
]

#: A case drawn from the ODD and the space it was drawn in: what a run records.
OddCase = tuple[OddSample, ParameterSpace]

#: The fields a ``target: measure`` control refuses
#: (:data:`.control_fields.MAPPING_FIELDS`).
_MAPPING_FIELDS = MAPPING_FIELDS


class OddSampler(_OddSampler):
    """openodd's sampler, with the CARLA backend's knobs by default, and the
    CARLA weather fields that follow from the conditions the ODD names as
    effects drawn with each case (:func:`~.backend.weather_effects`).

    The ``coverage`` it aims past is read as the scenario measures' current
    definitions read it: the items of a measure past its first version only
    as that version wrote them
    (:func:`~autoware_carla_scenario.coverage.collector.current_measure_coverage`,
    docs/odd.md, "Measure versions")."""

    def __init__(
        self,
        odd: OddDefinition,
        knobs: Optional[Mapping[str, OddKnob]] = None,
        **kw: Any,
    ) -> None:
        from ..coverage.collector import current_measure_coverage  # noqa: PLC0415

        kw.setdefault("backend", CARLA_BACKEND)
        if kw.get("controls"):
            kw["controls"] = _unreplaced_controls(
                odd, knobs or {}, kw["controls"], kw["backend"]
            )
        if "effects" not in kw:
            kw["effects"] = weather_effects(
                odd,
                kw["backend"].active_rules,
                knobs=knobs,
                controls=kw.get("controls"),
                backend=kw["backend"],
            )
        if kw.get("coverage") is not None:
            kw["coverage"] = current_measure_coverage(kw["coverage"], odd)
        super().__init__(odd, knobs, **kw)


def _unreplaced_controls(
    odd: OddDefinition,
    knobs: Mapping[str, OddKnob],
    controls: Mapping[str, OddKnob],
    backend: Any,
) -> dict[str, OddKnob]:
    """*controls* (the scenario's) less those a sweep's knob (*knobs*, by
    attribute) replaces: a control writing a node of the rules that an
    attribute with a knob of *knobs* measures.  Such a control is no knob of
    the attribute (:meth:`CarlaBackend.knob`), so openodd would draw it as an
    effect too, writing its setting over the knob's value; an explicit knob
    replaces it, as it replaces any control an attribute reads (#2072).
    The sampler and :func:`case_settings` both draw with what is left."""
    if not knobs or not isinstance(backend, CarlaBackend):
        return dict(controls)
    replaced = {
        id(control)
        for attribute in odd.attributes
        if attribute.name in knobs
        for control in [_effect_control(attribute, controls)]
        if control is not None
    }
    return {
        name: control
        for name, control in controls.items()
        if id(control) not in replaced
    }


def scenario_backend(
    rules: Optional[Mapping[str, Any]], environment_rules: Any = None
) -> CarlaBackend:
    """The CARLA backend a scenario's cases are drawn with: the shared
    network (:mod:`~autoware_carla_scenario.weather_rules`) -- the default
    rules, with *environment_rules*, the config's ``environment.rules`` (a
    file, or ``{default, vehicles}``), followed over them as a run follows
    them (:func:`~autoware_carla_scenario.weather_rules.use_weather_rules`:
    its ``default``; the ``vehicles`` entries give a vehicle's grip alone,
    which no case draws) -- and, with *rules* -- the scenario's
    ``scenario_rules``, a rule document -- the nodes the scenario adds to it
    (openodd's ``Backend.add_rules``): what it sets up, such as the gap to a
    vehicle about to cut in, which its ``controls`` write by ``quantity``.
    A backend of its own, never the process's
    :data:`~.backend.CARLA_BACKEND`, so the CARLA weather a case is given
    follows the rules its run follows, not those an earlier run left, and
    one scenario's nodes never reach another's draws.

    Raises:
        openodd.RulesError: When *rules* is not a rule document, or defines a
            node the shared network defines, or *environment_rules* is not of
            the form a run reads or its file cannot be read
            (:func:`~autoware_carla_scenario.weather_rules.check_weather_rules`).
    """
    from ..weather_rules import check_weather_rules  # noqa: PLC0415

    default, _vehicles = check_weather_rules(environment_rules)
    backend = CarlaBackend()
    if default is not None:
        backend.use_rules(str(default))
    if rules:
        backend.add_rules(Rules.from_text(yaml.safe_dump(dict(rules), sort_keys=False)))
    return backend


def controls_from_mapping(raw: Optional[Mapping[str, Any]]) -> dict[str, OddKnob]:
    """Knobs from config, by name: :func:`knobs_from_mapping`, plus ``target``.

    ``target: setting`` (the default) is a raw setting, as without it.
    ``target: measure`` declares that the key takes the measure's value at the
    start itself -- the scenario solves for the placement that makes the
    measure read it (:mod:`autoware_carla_scenario.placement`) -- so it is
    written unchanged: such a control may not have ``scale``, ``offset``,
    ``integer`` or ``values``.  ``range`` and ``limits`` it keeps.

    Raises:
        ValueError: On an unknown ``target``, a ``target: measure`` control
            that maps the value, or what :func:`knobs_from_mapping` refuses.
    """
    plain: dict[str, Any] = {}
    for name, spec in (raw or {}).items():
        if not isinstance(spec, Mapping) or "target" not in spec:
            plain[name] = spec
            continue
        target = spec["target"]
        if target not in CONTROL_TARGETS:
            raise ValueError(
                f"control {name}: target must be one of {list(CONTROL_TARGETS)}, "
                f"got {target!r}"
            )
        rest = {k: v for k, v in spec.items() if k != "target"}
        if target == "measure":
            mapped = sorted(f for f in _MAPPING_FIELDS if f in rest)
            if mapped:
                raise ValueError(
                    f"control {name}: target: measure writes the measure's value "
                    f"unchanged, so it takes no {', '.join(mapped)}"
                )
        plain[name] = rest
    return knobs_from_mapping(plain)


def weights_from_config(
    raw: Union[str, Path, Mapping[str, Any], None], base_dir: Optional[Path] = None
) -> Optional[dict[str, float]]:
    """The shares of ``sweep.odd_sample.weights``: module -> the fraction of
    the cases in it, given inline or as the path of a YAML file holding that
    mapping (relative to *base_dir*). ``None`` when not given.

    The modules are the ODD's own, named in its OpenODD files: the shares do
    not name its taxonomy (docs/odd.md, "Sampling by shares of the ODD").

    Raises:
        ValueError: When the file cannot be read, is not UTF-8 or not YAML,
            holds an integer too long to read, or when it or the mapping is
            not module -> number.
    """
    if raw is None:
        return None
    if isinstance(raw, (str, Path)):
        path = Path(raw)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        # A file that is not there, not UTF-8 (UnicodeDecodeError), not YAML,
        # or with an integer of more digits than Python turns from text
        # (PyYAML's int constructor raises a bare ValueError,
        # sys.get_int_max_str_digits()) is refused with the key, on one line
        # (a YAML error's text spans several, with a caret diagram), as the
        # docstring promises (#1703).
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError) as e:
            reason = " ".join(str(e).split()) or type(e).__name__
            raise ValueError(
                f"sweep.odd_sample.weights: cannot read {path}: {reason}"
            ) from e
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"sweep.odd_sample.weights: {path} must hold module -> share"
            )
    if not isinstance(raw, Mapping):
        raise ValueError(
            "sweep.odd_sample.weights: expected module -> share, or a YAML file of them"
        )
    weights: dict[str, float] = {}
    for module, share in raw.items():
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise ValueError(
                f"sweep.odd_sample.weights: the share of {module} must be a number, "
                f"got {share!r}"
            )
        weights[str(module)] = float(share)
    return weights


def sampler_from_config(
    raw: Mapping[str, Any],
    *,
    odd: Union[str, OddDefinition, None] = None,
    controls: Optional[Mapping[str, Any]] = None,
    base_dir: Optional[Path] = None,
    drawn: Optional[OddDraws] = None,
    rules: Optional[Mapping[str, Any]] = None,
    environment_rules: Any = None,
) -> tuple[OddSampler, Optional[int]]:
    """An :class:`OddSampler` and the case count, from a ``sweep.odd_sample``
    mapping without ``flaky`` (see :func:`search_from_config`).

    Raises:
        ValueError: On an unknown field, or a ``flaky`` other than missing,
            null or false: ``true`` and any mapping, ``{}`` included, are a
            flaky search, which needs :func:`search_from_config`.
    """
    # ``{}`` is falsy but a flaky search with the defaults, as ``true`` is.
    flaky = raw.get("flaky")
    if flaky is not None and flaky is not False:
        raise ValueError("sweep.odd_sample.flaky needs search_from_config()")
    sampler, _, count = search_from_config(
        raw,
        odd=odd,
        controls=controls,
        base_dir=base_dir,
        drawn=drawn,
        rules=rules,
        environment_rules=environment_rules,
    )
    return sampler, count


def seed_setting(key: str, value: Any) -> Optional[int]:
    """The seed a config sets under *key* (the run's ``seed``,
    ``sweep.odd_sample.seed``), checked; ``None`` when it is null.

    A seed is a non-negative whole number, as a route search's counts are
    (docs/logical_scenarios.md): ``2`` or ``2.0``, not ``2.5``, a boolean or
    a string that is no integer -- ``int()`` alone would run ``1.9`` and
    ``true`` as seed 1 -- and from 0 to
    :data:`~autoware_carla_scenario.seed.MAX_SEED` (``2**31 - 1``), the range
    CARLA's pedestrian seed and SUMO's ``--seed`` both take
    (:func:`~autoware_carla_scenario.seed.check_seed`).

    Raises:
        ValueError: naming *key*, when *value* is not such a number.
    """
    number = whole_number_setting(key, value)
    if number is None:
        return None
    if number < 0:
        raise ValueError(f"{key} must not be negative, got {value!r}")
    return check_seed(key, number)


def whole_number_setting(
    key: str, value: Any, *, minimum: Optional[int] = None
) -> Optional[int]:
    """The whole number a config sets under *key*, checked; ``None`` when it
    is null.

    ``2`` or ``2.0`` (or the string ``"2"``), not ``2.5``, a boolean, a blank
    or any other string -- ``int()`` alone would read ``1.9`` and ``true`` as
    1 -- and, with *minimum*, not below it.  What every whole-number setting a
    run reads goes through: its seeds (:func:`seed_setting`), and the
    ``background_traffic`` and ``server`` counts.

    Raises:
        ValueError: naming *key*: ``<key> must be a whole number, got ...``,
            or ``<key> must be >= <minimum>, got ...``.
    """
    if value is None:
        return None
    number = _whole_number(key, value)
    if number is None:  # A blank string, which no whole number is.
        raise ValueError(f"{key} must be a whole number, got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value!r}")
    return number


def sample_seed(raw: Mapping[str, Any]) -> int:
    """The seed of a ``sweep.odd_sample`` mapping: its ``seed``, or
    :data:`~autoware_carla_scenario.seed.DEFAULT_SEED` when it is not given or
    null (:func:`~autoware_carla_scenario.sweeper.expand.expand_sweep` fills a
    null one in with the run's first).

    Raises:
        ValueError: When it is not a non-negative whole number
            (:func:`seed_setting`).
    """
    seed = seed_setting("sweep.odd_sample.seed", raw.get("seed"))
    return DEFAULT_SEED if seed is None else seed


def sample_count(raw: Mapping[str, Any]) -> Optional[int]:
    """How many cases a ``sweep.odd_sample`` mapping draws: its ``count``, or
    ``None`` (one per base case) when it is not given or null.

    A whole number, at least 1: ``int()`` alone would draw two cases for
    ``2.7``, and a count of 0 draws no case, a sweep refused anyway
    (:data:`~autoware_carla_scenario.sweeper.expand.EMPTY_SWEEP_MESSAGE`) but
    for a reason it would not name.

    Raises:
        ValueError: naming ``sweep.odd_sample.count``, when it is not a whole
            number of at least 1.
    """
    value = raw.get("count")
    if value is None:
        return None
    try:
        count = _whole_number("sweep.odd_sample.count", value)
    except ValueError:
        count = None  # Refused below, with the bound.
    if count is None or count < 1:
        raise ValueError(
            f"sweep.odd_sample.count must be a whole number of at least 1, got {value!r}"
        )
    return count


#: The switches of ``sweep.odd_sample`` that say whether each case draws its
#: grip (:func:`with_tire_friction`) and its visibility (:func:`with_visibility`).
SAMPLE_DRAWS = ("tire_friction", "visibility")


def sample_draws(raw: Mapping[str, Any], name: str) -> bool:
    """Whether a ``sweep.odd_sample`` mapping draws *name* for each case, one
    of :data:`SAMPLE_DRAWS`: true when not given or null, else its value read
    strictly (:func:`~autoware_carla_scenario.config_flags.check_flag`), so
    ``"false"``, ``"off"`` and ``0`` turn the draw off.

    Raises:
        ValueError: naming ``sweep.odd_sample.<name>``, when it is not true
            or false (``"ture"``, ``2``).
    """
    value = raw.get(name)
    if value is None:
        return True
    return check_flag(f"sweep.odd_sample.{name}", value)


def check_sample_settings(raw: Mapping[str, Any]) -> None:
    """Refuse a ``sweep.odd_sample`` mapping's ``seed``, ``count``,
    ``tire_friction`` and ``visibility`` when drawing from it would
    (:func:`sample_seed`, :func:`sample_count`, :func:`sample_draws`): read
    before anything is drawn, built or resolved.

    Raises:
        ValueError: naming the key of the value refused.
    """
    sample_seed(raw)
    sample_count(raw)
    for name in SAMPLE_DRAWS:
        sample_draws(raw, name)


def search_from_config(
    raw: Mapping[str, Any],
    *,
    odd: Union[str, OddDefinition, None] = None,
    controls: Optional[Mapping[str, Any]] = None,
    base_dir: Optional[Path] = None,
    drawn: Optional[OddDraws] = None,
    rules: Optional[Mapping[str, Any]] = None,
    environment_rules: Any = None,
) -> tuple[OddSampler, Optional[FlakySearch], Optional[int]]:
    """An :class:`OddSampler`, the flaky search over it (``None`` without
    ``flaky``), and the case count (``None`` when not given), from a
    ``sweep.odd_sample`` mapping.

    Fields: ``count``, ``seed`` (:data:`~autoware_carla_scenario.seed.DEFAULT_SEED`
    when not given or null), ``strategy`` (``weighted`` with ``weights``,
    else ``uniform``, when not given), ``odd`` (else *odd*, the
    run's), ``coverage_from`` (paths to earlier runs' coverage files or
    directories: the coverage ``strategy: coverage`` aims past, and the runs
    the flaky search learns from), ``knobs``, ``weights``
    (:func:`weights_from_config`), and ``flaky``
    (:data:`FLAKY_FIELDS`; ``{}`` or ``true`` for the defaults).

    *controls* are the scenario's own (its config's ``controls``), keyed by
    the scenario measures its parameters set. *drawn*, in a batch, holds the
    cases the earlier scenarios drew; the sampler goes on from those drawn
    from its ODD. The caller records its own cases in it once drawn.
    *rules* (the config's ``scenario_rules``) and *environment_rules* (its
    ``environment.rules``) are the rules the cases are drawn by
    (:func:`scenario_backend`): the CARLA weather a case is given follows
    the rules its run follows.

    Raises:
        ValueError: On an unknown field, a ``seed`` or ``count`` that is
            not a whole number, or a ``tire_friction`` or ``visibility`` that
            is not true or false (:func:`check_sample_settings`).
    """
    from ..coverage.report import load_and_merge  # noqa: PLC0415
    from .registry import resolve_odd  # noqa: PLC0415

    unknown = set(raw) - SAMPLE_FIELDS
    if unknown:
        raise ValueError(f"sweep.odd_sample: unknown fields {sorted(unknown)}")
    # Before the ODD is resolved or any coverage read.
    check_sample_settings(raw)
    seed = sample_seed(raw)
    count = sample_count(raw)
    flaky = raw.get("flaky")
    if flaky is True:
        flaky = {}
    if flaky is not None and flaky is not False and not isinstance(flaky, Mapping):
        raise ValueError("sweep.odd_sample.flaky: expected a mapping or true")
    unknown = set(flaky or {}) - FLAKY_FIELDS
    if unknown:
        raise ValueError(f"sweep.odd_sample.flaky: unknown fields {sorted(unknown)}")
    definition = resolve_odd(raw.get("odd") or odd)
    coverage = None
    paths: Iterable[Any] = raw.get("coverage_from") or ()
    if isinstance(paths, (str, Path)):
        paths = [paths]
    resolved = [
        Path(p) if Path(p).is_absolute() or base_dir is None else base_dir / Path(p)
        for p in paths
    ]
    if resolved:
        coverage = load_and_merge(resolved)
    strategy = raw.get("strategy")
    knobs = controls_from_mapping(raw.get("knobs"))
    scenario_controls = controls_from_mapping(controls)
    backend = scenario_backend(rules, environment_rules)
    # The grip's and the visibility's quantiles, drawn with each case as
    # parameters of it (effects of root nodes, uniform on [0, 1]), so the
    # flaky search sees and moves them as it moves the weather effects'
    # (#1995) -- unless a knob already sets them, or what they place.  Not
    # for a sampler that sets nothing else: openodd refuses one that draws
    # nothing of the ODD, and the quantiles alone would hide that.
    written = case_settings(definition, knobs, scenario_controls, backend)
    nodes = [
        quantity
        for name, key, quantity, fixed in QUANTILE_DRAWS
        if written and sample_draws(raw, name) and not set(written) & {key, *fixed}
    ]
    if nodes:
        backend.add_rules(
            Rules.from_text(
                yaml.safe_dump(_rules_with_quantiles(rules, nodes), sort_keys=False)
            )
        )
        scenario_controls = {
            **scenario_controls,
            **{
                f"sweep.odd_sample.{name}": OddKnob(key, quantity=quantity)
                for name, key, quantity, _fixed in QUANTILE_DRAWS
                if quantity in nodes
            },
        }
    sampler = OddSampler(
        definition,
        knobs,
        controls=scenario_controls,
        seed=seed,
        strategy=None if strategy is None else str(strategy),
        coverage=coverage,
        weights=weights_from_config(raw.get("weights"), base_dir),
        drawn=drawn.of(definition) if drawn is not None else (),
        backend=backend,
    )
    search = None
    if flaky is not None and flaky is not False:
        options = {str(k): v for k, v in flaky.items()}
        scenario = options.pop("scenario", None)
        search = FlakySearch(sampler, seed=seed, **options)
        if coverage is not None:
            search.observe_report(coverage, None if scenario is None else str(scenario))
    return sampler, search, count


#: The draws of each case's place in a range its weather leaves: the
#: ``sweep.odd_sample`` switch (:data:`SAMPLE_DRAWS`), the setting it is drawn
#: into, the node of the rules that draws it (a root, uniform on [0, 1], so
#: the value written is the case's quantile, ``quantile(<node>)`` among its
#: values), and the settings that fix what it places, which a case that sets
#: draws no quantile for.
QUANTILE_DRAWS = (
    (
        "tire_friction",
        TIRE_FRICTION_KEY,
        "tire_friction_quantile",
        ("environment.tire_friction",),
    ),
    (
        "visibility",
        VISIBILITY_KEY,
        "visibility_quantile",
        ("environment.visibility_m",),
    ),
)


def case_settings(
    definition: OddDefinition,
    knobs: Mapping[str, OddKnob],
    controls: Mapping[str, OddKnob],
    backend: CarlaBackend,
) -> dict[str, tuple[str, str]]:
    """The settings a case drawn from *definition* is given by a knob or an
    effect, each with what draws it, ``(kind, name)``: ``("attribute",
    <attribute>)`` for an attribute's knob (one of *knobs*, a sweep's, else
    the backend's for it, which *controls*, the scenario's, may give),
    ``("control", <name>)`` for a control that writes a node of the rules
    and no knob of *knobs* replaces (:func:`_unreplaced_controls`),
    and ``("weather", <quantity>)`` for a CARLA weather field that follows
    from the conditions the ODD names (:func:`~.backend.weather_effects`)."""
    settings: dict[str, tuple[str, str]] = {}
    controls = _unreplaced_controls(definition, knobs, controls, backend)
    for attribute in definition.attributes:
        knob = knobs.get(attribute.name) or backend.knob(attribute, controls)
        if knob is not None:
            settings.setdefault(knob.key, ("attribute", attribute.name))
    for name, control in controls.items():
        if control.quantity is not None:
            settings.setdefault(control.key, ("control", name))
    for effect in weather_effects(
        definition,
        backend.active_rules,
        knobs=knobs,
        controls=controls,
        backend=backend,
    ):
        settings.setdefault(effect.key, ("weather", str(effect.quantity)))
    return settings


def _rules_with_quantiles(
    rules: Optional[Mapping[str, Any]], nodes: Sequence[str]
) -> dict[str, Any]:
    """The scenario's rule document (*rules*, its ``scenario_rules``; a bare
    one without), with *nodes* added: roots, uniform on [0, 1].

    Raises:
        openodd.RulesError: When *rules* defines one of *nodes* already.
    """
    doc: dict[str, Any] = (
        {str(k): v for k, v in rules.items()}
        if rules
        else {"version": 3, "taxonomy": {"bundled": "default"}}
    )
    quantities = dict(doc.get("quantities") or {})
    for node in nodes:
        if node in quantities:
            raise RulesError(
                f"scenario_rules: {node} is the node sweep.odd_sample draws "
                "each case's quantile of it from; name the scenario's own "
                "node otherwise"
            )
        quantities[node] = {
            "basis": {
                "kind": "expert",
                "by": "autoware_carla_scenario",
                "text": "Where in the range its weather leaves a case is: "
                "any quantile alike.",
            },
            "parts": {"case": {"uniform": {"low": 0.0, "high": 1.0}}},
        }
    doc["quantities"] = quantities
    return doc


def with_tire_friction(samples: Iterable[OddSample], seed: int = 0) -> list[OddSample]:
    """*samples*, each with where its grip is in the range its weather leaves.

    Each case gets ``environment.tire_friction_quantile=<q>``, *q* uniform in
    [0, 1] and drawn from *seed* and the case itself (its index and settings),
    so the same sweep draws the same grip.  The range is the run's: the
    ``EnvironmentAction`` reads it off the weather it leaves
    (:func:`~autoware_carla_scenario.actions.environment.tire_friction_range`),
    a dry road's around its own grip, a flooded one's about half.  A case that
    names its quantile, or a grip (``environment.tire_friction``), already
    keeps it.
    """
    return _with_quantile(
        samples, seed, TIRE_FRICTION_KEY, "tire_friction", "environment.tire_friction"
    )


def with_visibility(samples: Iterable[OddSample], seed: int = 0) -> list[OddSample]:
    """*samples*, each with where its visibility is in the range its weather leaves.

    Each case gets ``environment.visibility_quantile=<q>``, *q* uniform in
    [0, 1] and drawn from *seed* and the case itself, as
    :func:`with_tire_friction` draws the grip.  The range is the run's: the
    runner works it out from the conditions the case names
    (:func:`~autoware_carla_scenario.weather_rules.resolve_environment`), a
    clear day's 250-400 m, a night's 50-150 m, thick fog's 12-18 m.  A case that
    names its quantile, or a visibility (``environment.visibility_m``),
    already keeps it.
    """
    return _with_quantile(
        samples, seed, VISIBILITY_KEY, "visibility", "environment.visibility_m"
    )


def _with_quantile(
    samples: Iterable[OddSample],
    seed: int,
    key: str,
    name: str,
    *fixed: str,
) -> list[OddSample]:
    """*samples*, each with ``key=<q>`` drawn from *seed*, *name* and the case,
    unless it sets *key* or one of the *fixed* keys already."""
    out: list[OddSample] = []
    prefix = f"{key}="
    given = (key, *fixed)
    for sample in samples:
        if sets_any(sample.overrides, given):
            out.append(sample)
            continue
        rng = random.Random(
            f"{seed}/{name}/{sample.index}/{','.join(sample.overrides)}"
        )
        quantile = round(rng.random(), 3)
        out.append(
            dataclasses.replace(
                sample, overrides=[*sample.overrides, f"{prefix}{quantile}"]
            )
        )
    return out


def encode_case(case: OddSample, space: ParameterSpace) -> str:
    """*case* and the space it was drawn in, as one override value: JSON in
    URL-safe base64, unpadded, which a Hydra override takes as it is."""
    raw = {
        "index": case.index,
        "overrides": list(case.overrides),
        "buckets": dict(case.buckets),
        "values": dict(case.values),
        "situation": case.situation,
        "flaky": None if case.flaky is None else dataclasses.asdict(case.flaky),
        "space": space.to_list(),
    }
    text = json.dumps(raw, separators=(",", ":"), default=str)
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def decode_case(text: str) -> OddCase:
    """The case and space :func:`encode_case` wrote."""
    padded = text + "=" * (-len(text) % 4)
    raw = json.loads(base64.urlsafe_b64decode(padded.encode()))
    flaky = raw.get("flaky")
    case = OddSample(
        index=int(raw["index"]),
        overrides=list(raw["overrides"]),
        buckets=dict(raw["buckets"]),
        values=dict(raw["values"]),
        situation=raw.get("situation"),
        flaky=None if flaky is None else FlakyMetrics(**flaky),
    )
    return case, ParameterSpace.from_list(raw["space"])
