"""Sample concrete scenarios from an ODD -- with openodd_py's sampler.

The sampler is openodd_py's (``openodd.OddSampler``, on its Rust core): it
draws the settings of the next runs from the ODD, only combinations the ODD
admits, and with ``strategy="coverage"`` aims them at the buckets and
situations earlier runs left uncovered.  What the framework adds is which
setting brings each attribute about -- its backend's knobs
(:class:`~.backend.CarlaBackend`):

* the weather and the sun, set for every scenario through the
  ``environment`` config, on the built-in probes :func:`~.probes.rain`,
  :func:`~.probes.fog` and :func:`~.probes.illumination`
  (:data:`DEFAULT_KNOBS`);
* an attribute the ODD maps onto a scenario measure
  (:func:`~.scenario_measure.scenario_measure`, ``measure:`` in a binding
  file), set by the scenario's ``controls`` for that measure::

      controls:
        vehicle_ahead_gap_m: {key: scenario.npc_ahead_m, range: [6, 30]}

A sweep's ``knobs``, keyed by the ODD's attributes, replace them attribute by
attribute.  Attributes nothing sets -- what the map or the drive decides --
are left open: a combination is admitted when the ODD can hold for *some*
value of them.  A batch of scenarios drawing from one ODD goes on where the
last left off (:class:`OddDraws`).  The random stream is ChaCha8 seeded with
the SHA-256 of the seed: the same seed, ODD, knobs and coverage give the same
cases on any platform.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import openodd
from openodd import OddDefinition, OddDraws, OddKnob, OddSample, knobs_from_mapping

from .backend import CARLA_BACKEND, DEFAULT_KNOBS

__all__ = [
    "DEFAULT_KNOBS",
    "OddDraws",
    "OddKnob",
    "OddSample",
    "OddSampler",
    "knobs_from_mapping",
    "sampler_from_config",
]


class OddSampler(openodd.OddSampler):
    """openodd_py's sampler, with the framework's knobs (:class:`~.backend.CarlaBackend`).

    Args:
        odd: The ODD to draw from.
        knobs: ODD attribute name -> how a run sets it, for this ODD alone
            (a sweep's ``knobs``).  Replaces what the backend gives.
        controls: Scenario measure key -> how the scenario sets it (its
            config's ``controls``).  An attribute the ODD maps onto that
            measure (:func:`~.scenario_measure.scenario_measure`) is drawn
            with it; controls of measures the ODD does not map are unused.
            Attributes read by a built-in probe in :data:`DEFAULT_KNOBS` (the
            weather, the sun) need neither.
        seed: The same seed, ODD, knobs and coverage give the same cases.
        strategy: ``"uniform"`` draws every admitted bucket alike;
            ``"coverage"`` draws the least covered first.
        coverage: The coverage of earlier runs, as merged by
            :func:`openodd.load_and_merge`.
            Only read by the ``coverage`` strategy.
        max_tries: Combinations drawn per case before giving up on it.
        drawn: The cases drawn from this ODD by earlier samplers of the
            batch (:meth:`OddDraws.of`).
        backend: Whose knobs; :data:`~.backend.CARLA_BACKEND` by default.

    Raises:
        ValueError: If no attribute of the ODD can be drawn, a knob names an
            attribute the ODD does not have or one without buckets, or the
            strategy is unknown.
    """

    def __init__(
        self,
        odd: OddDefinition,
        knobs: Optional[Mapping[str, OddKnob]] = None,
        *,
        controls: Optional[Mapping[str, OddKnob]] = None,
        seed: Union[int, str] = 0,
        strategy: str = "uniform",
        coverage: Any = None,
        max_tries: int = 200,
        drawn: Sequence[OddSample] = (),
        backend: Optional[openodd.Backend] = None,
    ) -> None:
        super().__init__(
            odd,
            knobs,
            controls=controls,
            seed=seed,
            strategy=strategy,
            coverage=coverage,
            max_tries=max_tries,
            drawn=drawn,
            backend=backend or CARLA_BACKEND,
        )


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
    the scenario measures its parameters set; the ODD's mapping of its
    attributes onto measures joins the two.  ``knobs`` in the sweep are keyed
    by the ODD's attributes and replace both for that sweep.

    *drawn*, in a batch, holds the cases the earlier scenarios drew; the
    sampler goes on from those drawn from its ODD (``OddSampler(drawn=...)``).
    The caller records its own cases in it once drawn.

    Raises:
        ValueError: On an unknown field.
    """
    from openodd import load_and_merge  # noqa: PLC0415
    from .registry import resolve_odd  # noqa: PLC0415

    unknown = set(raw) - {"count", "seed", "strategy", "odd", "coverage_from", "knobs"}
    if unknown:
        raise ValueError(f"sweep.odd_sample: unknown fields {sorted(unknown)}")
    definition = resolve_odd(raw.get("odd") or odd)
    strategy = str(raw.get("strategy", "uniform"))
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
        strategy=strategy,
        coverage=coverage,
        drawn=drawn.of(definition) if drawn is not None else (),
    )
    count = raw.get("count")
    return sampler, None if count is None else int(count)
