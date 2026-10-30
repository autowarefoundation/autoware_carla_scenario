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

* the weather and the sun are set for every scenario through the
  ``environment`` config, which the built-in probes :func:`~.probes.rain`,
  :func:`~.probes.fog` and :func:`~.probes.illumination` have knobs on
  (:data:`DEFAULT_KNOBS`).

A sweep's ``knobs``, keyed by the ODD's attributes, replace them attribute by
attribute. Attributes nothing sets are left open. See openodd's
:mod:`openodd.sampler` for how cases are drawn.

The road's grip is no attribute of an ODD but follows one: the water the
weather drawn leaves on the road bounds it
(:func:`~autoware_carla_scenario.actions.environment.tire_friction_range`).
Each case is given where in that range its grip is,
``environment.tire_friction_quantile`` (:func:`with_tire_friction`), drawn
from the sweep's seed and the case, so the same sweep draws the same grip;
``tire_friction: false`` in the sweep leaves it at the middle.

How far drivers see follows the weather the same way
(:func:`~autoware_carla_scenario.visibility.visibility_range`): night, rain
and fog shorten it.  Each case is given where in that range its visibility
is, ``environment.visibility_quantile`` (:func:`with_visibility`), drawn the
same way; ``visibility: false`` in the sweep leaves it at the middle.  A
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
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from openodd.flaky import FlakyMetrics, FlakySearch, ParameterSpace
from openodd.sampler import OddDraws, OddKnob, OddSample, knobs_from_mapping
from openodd.sampler import OddSampler as _OddSampler

from .backend import CARLA_BACKEND, DEFAULT_KNOBS
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
    "sampler_from_config",
    "search_from_config",
    "TIRE_FRICTION_KEY",
    "weights_from_config",
    "VISIBILITY_KEY",
    "with_tire_friction",
    "with_visibility",
]

#: The config key (an override a sweep adds) that carries a drawn case to its
#: run, encoded (:func:`encode_case`).
ODD_CASE_KEY = "odd_case"

#: A case drawn from the ODD and the space it was drawn in: what a run records.
OddCase = tuple[OddSample, ParameterSpace]

#: What a control's ``target`` may say: ``setting`` (the default), a raw
#: setting the drawn value is mapped onto by ``scale`` and ``offset``; or
#: ``measure``, a parameter that takes the measure's value at the start itself,
#: which the scenario solves its placement for.
CONTROL_TARGETS = ("setting", "measure")
#: The fields a ``target: measure`` control refuses: what it writes is the
#: measure's own value, unchanged.
_MAPPING_FIELDS = ("scale", "offset", "integer", "values")

#: The fields of ``sweep.odd_sample``: ``tire_friction`` and ``visibility``
#: (default true) say whether each case draws where its grip
#: (:func:`with_tire_friction`) and its visibility (:func:`with_visibility`)
#: are in the ranges its weather leaves; the rest are
#: :func:`search_from_config`'s.
SAMPLE_FIELDS = {
    "count",
    "seed",
    "strategy",
    "odd",
    "coverage_from",
    "knobs",
    "weights",
    "flaky",
    "tire_friction",
    "visibility",
}

#: The setting a case's grip is drawn into: where in the range of grip its
#: weather leaves it is (``EnvironmentAction``'s ``tire_friction_quantile``).
TIRE_FRICTION_KEY = "environment.tire_friction_quantile"
#: The setting a case's visibility is drawn into: where in the range of
#: visibility its weather leaves it is (``EnvironmentAction``'s
#: ``visibility_quantile``).
VISIBILITY_KEY = "environment.visibility_quantile"
#: The fields of ``sweep.odd_sample.flaky``: :class:`~openodd.FlakySearch`'s
#: options, and ``scenario``, the scenario whose earlier runs it learns from
#: (every scenario's, chosen in the same space, when not given).
FLAKY_FIELDS = {
    "explore_fraction",
    "pool",
    "coverage_depth",
    "threshold",
    "epsilon",
    "perturbations",
    "draws",
    "directions",
    "scenario",
}


class OddSampler(_OddSampler):
    """openodd's sampler, with the CARLA backend's knobs by default.

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
        if kw.get("coverage") is not None:
            kw["coverage"] = current_measure_coverage(kw["coverage"], odd)
        super().__init__(odd, knobs, **kw)


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
        ValueError: When the file or the mapping is not module -> number.
    """
    if raw is None:
        return None
    if isinstance(raw, (str, Path)):
        path = Path(raw)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as e:
            raise ValueError(
                f"sweep.odd_sample.weights: cannot read {path}: {e}"
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
) -> tuple[OddSampler, Optional[int]]:
    """An :class:`OddSampler` and the case count, from a ``sweep.odd_sample``
    mapping without ``flaky`` (see :func:`search_from_config`).

    Raises:
        ValueError: On an unknown field, or ``flaky``.
    """
    if raw.get("flaky"):
        raise ValueError("sweep.odd_sample.flaky needs search_from_config()")
    sampler, _, count = search_from_config(
        raw, odd=odd, controls=controls, base_dir=base_dir, drawn=drawn
    )
    return sampler, count


def search_from_config(
    raw: Mapping[str, Any],
    *,
    odd: Union[str, OddDefinition, None] = None,
    controls: Optional[Mapping[str, Any]] = None,
    base_dir: Optional[Path] = None,
    drawn: Optional[OddDraws] = None,
) -> tuple[OddSampler, Optional[FlakySearch], Optional[int]]:
    """An :class:`OddSampler`, the flaky search over it (``None`` without
    ``flaky``), and the case count (``None`` when not given), from a
    ``sweep.odd_sample`` mapping.

    Fields: ``count``, ``seed``, ``strategy`` (``weighted`` with ``weights``,
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

    Raises:
        ValueError: On an unknown field.
    """
    from ..coverage.report import load_and_merge  # noqa: PLC0415
    from .registry import resolve_odd  # noqa: PLC0415

    unknown = set(raw) - SAMPLE_FIELDS
    if unknown:
        raise ValueError(f"sweep.odd_sample: unknown fields {sorted(unknown)}")
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
    seed = int(raw.get("seed", 0))
    strategy = raw.get("strategy")
    sampler = OddSampler(
        definition,
        controls_from_mapping(raw.get("knobs")),
        controls=controls_from_mapping(controls),
        seed=seed,
        strategy=None if strategy is None else str(strategy),
        coverage=coverage,
        weights=weights_from_config(raw.get("weights"), base_dir),
        drawn=drawn.of(definition) if drawn is not None else (),
    )
    search = None
    if flaky is not None and flaky is not False:
        options = {str(k): v for k, v in flaky.items()}
        scenario = options.pop("scenario", None)
        search = FlakySearch(sampler, seed=seed, **options)
        if coverage is not None:
            search.observe_report(coverage, None if scenario is None else str(scenario))
    count = raw.get("count")
    return sampler, search, None if count is None else int(count)


def with_tire_friction(
    samples: Iterable[OddSample], seed: Union[int, str] = 0
) -> list[OddSample]:
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


def with_visibility(
    samples: Iterable[OddSample], seed: Union[int, str] = 0
) -> list[OddSample]:
    """*samples*, each with where its visibility is in the range its weather leaves.

    Each case gets ``environment.visibility_quantile=<q>``, *q* uniform in
    [0, 1] and drawn from *seed* and the case itself, as
    :func:`with_tire_friction` draws the grip.  The range is the run's: the
    ``EnvironmentAction`` reads it off the weather it leaves
    (:func:`~autoware_carla_scenario.visibility.visibility_range`), a clear
    day's 250-400 m, a night's 50-150 m, heavy fog's 15-40 m.  A case that
    names its quantile, or a visibility (``environment.visibility_m``),
    already keeps it.
    """
    return _with_quantile(
        samples, seed, VISIBILITY_KEY, "visibility", "environment.visibility_m"
    )


def _with_quantile(
    samples: Iterable[OddSample],
    seed: Union[int, str],
    key: str,
    name: str,
    *fixed: str,
) -> list[OddSample]:
    """*samples*, each with ``key=<q>`` drawn from *seed*, *name* and the case,
    unless it sets *key* or one of the *fixed* keys already."""
    out: list[OddSample] = []
    prefix = f"{key}="
    given = tuple(f"{k}=" for k in (key, *fixed))
    for sample in samples:
        if any(o.lstrip("+").startswith(given) for o in sample.overrides):
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
