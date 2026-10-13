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
        vehicle_ahead_gap_m: {key: scenario.npc_ahead_m, range: [6, 30]}

* the weather and the sun are set for every scenario through the
  ``environment`` config, which the built-in probes :func:`~.probes.rain`,
  :func:`~.probes.fog` and :func:`~.probes.illumination` have knobs on
  (:data:`DEFAULT_KNOBS`).

A sweep's ``knobs``, keyed by the ODD's attributes, replace them attribute by
attribute. Attributes nothing sets are left open. See openodd's
:mod:`openodd.sampler` for how cases are drawn.

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
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional, Union

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
    "knobs_from_mapping",
    "sampler_from_config",
    "search_from_config",
]

#: The config key (an override a sweep adds) that carries a drawn case to its
#: run, encoded (:func:`encode_case`).
ODD_CASE_KEY = "odd_case"

#: A case drawn from the ODD and the space it was drawn in: what a run records.
OddCase = tuple[OddSample, ParameterSpace]

#: The fields of ``sweep.odd_sample``.
SAMPLE_FIELDS = {"count", "seed", "strategy", "odd", "coverage_from", "knobs", "flaky"}
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
    """openodd's sampler, with the CARLA backend's knobs by default."""

    def __init__(
        self,
        odd: OddDefinition,
        knobs: Optional[Mapping[str, OddKnob]] = None,
        **kw: Any,
    ) -> None:
        kw.setdefault("backend", CARLA_BACKEND)
        super().__init__(odd, knobs, **kw)


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

    Fields: ``count``, ``seed``, ``strategy``, ``odd`` (else *odd*, the
    run's), ``coverage_from`` (paths to earlier runs' coverage files or
    directories: the coverage ``strategy: coverage`` aims past, and the runs
    the flaky search learns from), ``knobs``, and ``flaky``
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
    strategy = str(raw.get("strategy", "uniform"))
    sampler = OddSampler(
        definition,
        knobs_from_mapping(raw.get("knobs")),
        controls=knobs_from_mapping(controls),
        seed=seed,
        strategy=strategy,
        coverage=coverage,
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
