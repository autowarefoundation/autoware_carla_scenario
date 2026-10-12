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
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional, Union

from openodd.sampler import OddDraws, OddKnob, OddSample, knobs_from_mapping
from openodd.sampler import OddSampler as _OddSampler

from .backend import CARLA_BACKEND, DEFAULT_KNOBS
from .model import OddDefinition

__all__ = [
    "DEFAULT_KNOBS",
    "OddDraws",
    "OddKnob",
    "OddSample",
    "OddSampler",
    "knobs_from_mapping",
    "sampler_from_config",
]


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
    """An :class:`OddSampler` and the case count, from a ``sweep.odd_sample`` mapping.

    Fields: ``count`` (``None`` when not given), ``seed``, ``strategy``,
    ``odd`` (else *odd*, the run's), ``coverage_from`` (paths to earlier runs'
    coverage files or directories, for ``strategy: coverage``), ``knobs``.

    *controls* are the scenario's own (its config's ``controls``), keyed by
    the scenario measures its parameters set. *drawn*, in a batch, holds the
    cases the earlier scenarios drew; the sampler goes on from those drawn
    from its ODD. The caller records its own cases in it once drawn.

    Raises:
        ValueError: On an unknown field.
    """
    from ..coverage.report import load_and_merge  # noqa: PLC0415
    from .registry import resolve_odd  # noqa: PLC0415

    unknown = set(raw) - {"count", "seed", "strategy", "odd", "coverage_from", "knobs"}
    if unknown:
        raise ValueError(f"sweep.odd_sample: unknown fields {sorted(unknown)}")
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
    sampler = OddSampler(
        definition,
        knobs_from_mapping(raw.get("knobs")),
        controls=knobs_from_mapping(controls),
        seed=int(raw.get("seed", 0)),
        strategy=str(raw.get("strategy", "uniform")),
        coverage=coverage,
        drawn=drawn.of(definition) if drawn is not None else (),
    )
    count = raw.get("count")
    return sampler, None if count is None else int(count)
