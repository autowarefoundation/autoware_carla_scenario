"""Sample concrete scenarios from an ODD.

An ODD says which situations a system is meant to handle; coverage says which
of them the runs so far have driven.  The sampler closes the loop: it draws
the settings of the next runs from the ODD -- only combinations the ODD
admits -- and, given the coverage of earlier runs, aims them at what is still
uncovered.

What the sampler can draw are the attributes a run *sets*: the weather, the
sun, a speed a scenario parameter decides.  Each is tied to the Hydra override
that sets it by a :class:`OddKnob`::

    knobs:
      environment.rain: {key: environment.precipitation,
                         values: {none: 0, light: [1, 30], moderate: [30, 70], heavy: [70, 100]}}
      dynamic.lead_speed: {key: scenario.lead_speed_kmh}

An attribute read by a built-in probe that a run can set (:func:`~.probes.rain`,
:func:`~.probes.fog`, :func:`~.probes.illumination`) has a knob without being
given one (:data:`DEFAULT_KNOBS`).  Attributes with no knob -- what the map or
the drive decides -- are left open: the sampler admits a combination when the
ODD can hold for *some* value of them.

For each case, every knob's attribute gets a bucket -- one of its cover
item's buckets, the ones outside the ODD left out -- and the combination is
kept only if the ODD's own verdict on those buckets is not "outside".  Then a
value is drawn inside each bucket (a numeric bucket's interval, or the range
the knob gives a categorical one) and checked against the ODD again, as a
value this time.

``strategy="coverage"`` draws each attribute from its least covered buckets,
counting the cases already drawn in this batch as if they had covered theirs,
so a batch works through the holes rather than piling into the first.  Only
when the ODD admits no combination of those does it fall back to any bucket,
the less covered the likelier.  Situations (modules covered as situations,
``docs/odd.md``) that are still holes are aimed at first, one case each: a
case meant for one draws only combinations under which the situation can hold.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ..coverage.items import CoverItem
from .model import _OPEN, OddAttribute, OddDefinition, _Bucket
from .probes import fog, illumination, rain

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_KNOBS",
    "OddKnob",
    "OddSample",
    "OddSampler",
    "knobs_from_mapping",
]

#: A value a knob gives a categorical bucket: one value, or ``[low, high]``
#: to draw from.
KnobValue = Union[float, int, str, bool, Sequence[float]]


def _is_range(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _range(value: Any) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"a range is [low, high], not {list(value)!r}")
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class OddKnob:
    """How a run sets an ODD attribute: the Hydra override that does it.

    Attributes:
        key: The config key the value is written to, e.g.
            ``environment.precipitation``.
        values: For a categorical attribute, the value (or ``[low, high]``
            range to draw from) to write for each bucket label.  A numeric
            attribute may give one too, by bucket label, to replace the
            bucket's interval.  A categorical bucket without one cannot be
            drawn.
        scale: A numeric value drawn in the attribute's unit is written as
            ``value * scale + offset`` -- e.g. 1/3.6 for an attribute in km/h
            and a key in m/s.  Values from *values* are written as given.
        offset: See *scale*.
        integer: Write the value rounded to an integer.
    """

    key: str
    values: Mapping[str, KnobValue] = field(default_factory=dict)
    scale: float = 1.0
    offset: float = 0.0
    integer: bool = False

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("OddKnob: key must not be empty")
        for label, value in self.values.items():
            if _is_range(value):
                low, high = _range(value)
                if not low <= high:
                    raise ValueError(
                        f"OddKnob({self.key}): range for {label!r} must be [low, high]"
                    )


#: Knobs for the built-in probes a run can set, through the ``environment``
#: config (:class:`~autoware_carla_scenario.EnvironmentAction`).  The ranges
#: are the probes' own levels (:func:`~.probes.intensity_level`,
#: :func:`~.probes.illumination_level`), read backwards.
DEFAULT_KNOBS: dict[Callable[..., Any], OddKnob] = {
    rain: OddKnob(
        "environment.precipitation",
        values={"none": 0, "light": [1, 30], "moderate": [30, 70], "heavy": [70, 100]},
    ),
    fog: OddKnob(
        "environment.fog_density",
        values={"none": 0, "light": [1, 30], "moderate": [30, 70], "heavy": [70, 100]},
    ),
    illumination: OddKnob(
        "environment.sun_altitude_angle",
        values={
            "day": [15, 90],
            "low_sun": [0, 15],
            "twilight": [-6, 0],
            "night": [-90, -6],
        },
    ),
}


@dataclass(frozen=True)
class OddSample:
    """One concrete case the sampler drew.

    Attributes:
        index: 0-based position in the batch.
        overrides: The Hydra overrides that set it, ``key=value`` each.
        buckets: Attribute name -> the bucket label drawn for it.
        values: Attribute name -> the value drawn, in the attribute's terms
            (a label for a categorical attribute, a number in its unit for a
            numeric one).
        situation: The situation the case was aimed at, if any.
    """

    index: int
    overrides: list[str]
    buckets: dict[str, str]
    values: dict[str, Any]
    situation: Optional[str] = None


def _render(value: Any) -> str:
    """A value the way a Hydra override reads it back."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return repr(round(value, 3))
    return str(value)


def knobs_from_mapping(raw: Mapping[str, Any]) -> dict[str, OddKnob]:
    """Knobs by attribute name, from config (``{attribute: {key: ..., ...}}``).

    A bare string is a key: ``{dynamic.lead_speed: scenario.lead_speed_kmh}``.
    """
    knobs: dict[str, OddKnob] = {}
    for name, spec in raw.items():
        if isinstance(spec, str):
            knobs[str(name)] = OddKnob(spec)
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"knob {name}: expected a key or a mapping, got {spec!r}")
        unknown = set(spec) - {"key", "values", "scale", "offset", "integer"}
        if unknown:
            raise ValueError(f"knob {name}: unknown fields {sorted(unknown)}")
        if "key" not in spec:
            raise ValueError(f"knob {name}: key is required")
        knobs[str(name)] = OddKnob(
            key=str(spec["key"]),
            values={str(k): v for k, v in (spec.get("values") or {}).items()},
            scale=float(spec.get("scale", 1.0)),
            offset=float(spec.get("offset", 0.0)),
            integer=bool(spec.get("integer", False)),
        )
    return knobs


@dataclass
class _Axis:
    """One sampled attribute: its knob, and the buckets it may be drawn in."""

    attribute: OddAttribute
    item: CoverItem
    knob: OddKnob
    labels: list[str]


class OddSampler:
    """Draws concrete cases inside an ODD.

    Args:
        odd: The ODD to draw from.
        knobs: Attribute name -> how a run sets it.  Attributes read by a
            built-in probe in :data:`DEFAULT_KNOBS` need none; a knob given
            here replaces the default.
        seed: The random seed: the same seed, ODD, knobs and coverage give
            the same cases.
        strategy: ``"uniform"`` draws every admitted bucket alike;
            ``"coverage"`` draws the least covered first (see the module
            docstring).
        coverage: The coverage of earlier runs, as merged by
            :func:`~autoware_carla_scenario.coverage.report.load_and_merge`.
            Only read by the ``coverage`` strategy.
        max_tries: Combinations drawn per case before giving up on it.

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
        seed: int = 0,
        strategy: str = "uniform",
        coverage: Any = None,
        max_tries: int = 200,
    ) -> None:
        if strategy not in ("uniform", "coverage"):
            raise ValueError(
                f"OddSampler: strategy must be 'uniform' or 'coverage', not {strategy!r}"
            )
        self.odd = odd
        self.strategy = strategy
        self._rng = random.Random(seed)
        self._max_tries = max_tries
        knobs = dict(knobs or {})
        by_name = {a.name: a for a in odd.attributes}
        unknown = sorted(set(knobs) - set(by_name))
        if unknown:
            raise ValueError(f"OddSampler: ODD {odd.name} has no attributes {unknown}")

        self._axes: list[_Axis] = []
        for attribute in odd.attributes:
            knob = knobs.get(attribute.name)
            if knob is None and callable(attribute.probe):
                knob = DEFAULT_KNOBS.get(attribute.probe)
            if knob is None:
                continue
            if attribute.item is None:
                raise ValueError(
                    f"OddSampler: {attribute.name} has no buckets to draw from"
                )
            outside = set(odd.outside_buckets(attribute))
            labels = [
                label
                for label in attribute.item.labels
                if label not in outside and self._drawable(attribute.item, knob, label)
            ]
            if not labels:
                raise ValueError(
                    f"OddSampler: no bucket of {attribute.name} is both inside "
                    f"ODD {odd.name} and given a value by its knob ({knob.key})"
                )
            self._axes.append(_Axis(attribute, attribute.item, knob, labels))
        if not self._axes:
            raise ValueError(
                f"OddSampler: nothing of ODD {odd.name} can be drawn; give knobs "
                "for the attributes a run sets"
            )

        self._amount: dict[tuple[str, str], float] = {}
        self._situation_holes: list[str] = []
        if strategy == "coverage" and coverage is not None:
            self._read_coverage(coverage)
        #: Cases drawn per (attribute, bucket) so far in this batch.
        self._planned: dict[tuple[str, str], int] = {}

    # -- setup -----------------------------------------------------------

    @staticmethod
    def _drawable(item: CoverItem, knob: OddKnob, label: str) -> bool:
        return item.numeric or label in knob.values

    def _read_coverage(self, coverage: Any) -> None:
        """Remember how covered each bucket is, and which situations are holes."""
        situations = {m.name for m in self.odd.situations()}
        for entry in coverage.entries:
            name = entry.name
            if name.startswith("odd.situation."):
                module = name.removeprefix("odd.situation.")
                if module in situations and entry.holes:
                    self._situation_holes.append(module)
                continue
            if not name.startswith("odd."):
                continue
            attribute = name.removeprefix("odd.")
            target = entry.target if entry.target > 0 else 1.0
            for bucket in entry.buckets:
                self._amount[(attribute, bucket)] = entry.amount(bucket) / target

    @property
    def attributes(self) -> list[str]:
        """The attributes drawn, in the ODD's order."""
        return [axis.attribute.name for axis in self._axes]

    @property
    def situation_holes(self) -> list[str]:
        """The situations still uncovered, which the first cases are aimed at."""
        return list(self._situation_holes)

    # -- drawing ---------------------------------------------------------

    def _load(self, attribute: str, label: str) -> float:
        """How many times over the bucket is covered, or planned in this batch."""
        covered = self._amount.get((attribute, label), 0.0)
        return min(covered, 1.0) + self._planned.get((attribute, label), 0)

    def _pick(self, axis: _Axis, *, least: bool) -> str:
        """A bucket of *axis*: any (uniform), or the least loaded (coverage)."""
        if self.strategy == "uniform":
            return self._rng.choice(axis.labels)
        loads = {label: self._load(axis.attribute.name, label) for label in axis.labels}
        if least:
            lowest = min(loads.values())
            return self._rng.choice([b for b in axis.labels if loads[b] == lowest])
        # The least loaded did not combine into a case the ODD admits: any
        # bucket, the less loaded the likelier.
        weights = [1.0 / (1.0 + loads[label]) for label in axis.labels]
        return self._rng.choices(axis.labels, weights=weights)[0]

    def _open_values(self) -> dict[str, Any]:
        return {a.name: _OPEN for a in self.odd.attributes}

    def _admits(self, values: Mapping[str, Any], situation: Optional[str]) -> bool:
        verdict = self.odd.evaluate(values)
        if not verdict.inside:
            return False
        return situation is None or verdict.modules.get(situation) is not False

    def _draw_value(self, axis: _Axis, label: str) -> tuple[Any, Any]:
        """(value in the attribute's terms, value written to the knob's key)."""
        knob = axis.knob
        given = knob.values.get(label)
        if given is not None:
            written: Any = (
                self._rng.uniform(*_range(given)) if _is_range(given) else given
            )
            if knob.integer and isinstance(written, float):
                written = int(round(written))
            if axis.item.numeric:
                return float(written), written
            return label, written
        # A numeric bucket's interval: [low, high), the last one closed.
        index = axis.item.labels.index(label)
        low, high = axis.item.edges[index], axis.item.edges[index + 1]
        value = self._rng.uniform(low, high)
        if index < len(axis.item.labels) - 1 and value >= high:
            value = math.nextafter(high, low)
        written = value * knob.scale + knob.offset
        if knob.integer:
            written = int(round(written))
            # Rounding must not carry the value into the next bucket.
            value = (written - knob.offset) / knob.scale if knob.scale else value
        return value, written

    def draw(self, index: int = 0, situation: Optional[str] = None) -> OddSample:
        """Draw one case; *situation*, if given, is a module the case aims at.

        Raises:
            ValueError: If no combination the ODD admits was found in
                ``max_tries`` attempts.
        """
        for attempt in range(self._max_tries):
            least = attempt < self._max_tries // 2
            buckets = {
                axis.attribute.name: self._pick(axis, least=least)
                for axis in self._axes
            }
            open_values = self._open_values()
            as_buckets = {
                **open_values,
                **{
                    axis.attribute.name: _Bucket(
                        axis.item, axis.item.labels.index(buckets[axis.attribute.name])
                    )
                    for axis in self._axes
                },
            }
            if not self._admits(as_buckets, situation):
                continue
            values: dict[str, Any] = {}
            overrides: list[str] = []
            for axis in self._axes:
                value, written = self._draw_value(axis, buckets[axis.attribute.name])
                values[axis.attribute.name] = value
                overrides.append(f"{axis.knob.key}={_render(written)}")
            if not self._admits({**open_values, **values}, situation):
                continue
            for name, label in buckets.items():
                self._planned[(name, label)] = self._planned.get((name, label), 0) + 1
            return OddSample(index, overrides, buckets, values, situation)
        aim = f" aimed at situation {situation}" if situation else ""
        raise ValueError(
            f"OddSampler: found no case{aim} inside ODD {self.odd.name} in "
            f"{self._max_tries} tries"
        )

    def sample(self, count: int) -> list[OddSample]:
        """Draw *count* cases.

        With the ``coverage`` strategy, the first cases are aimed at the
        situations still uncovered, one each; a situation no knob can bring
        about is given up on (and logged), and the case is drawn without it.
        """
        if count < 0:
            raise ValueError("OddSampler: count must not be negative")
        aims = list(self._situation_holes)
        cases: list[OddSample] = []
        for index in range(count):
            while aims:
                situation = aims.pop(0)
                try:
                    cases.append(self.draw(index, situation))
                    break
                except ValueError:
                    logger.warning(
                        "OddSampler: no knob brings about situation %s; not aiming at it",
                        situation,
                    )
            else:
                cases.append(self.draw(index))
        return cases


def sampler_from_config(
    raw: Mapping[str, Any],
    *,
    odd: Union[str, OddDefinition, None] = None,
    base_dir: Optional[Path] = None,
) -> tuple[OddSampler, Optional[int]]:
    """An :class:`OddSampler` and the case count, from a ``sweep.odd_sample`` mapping.

    Fields: ``count`` (``None`` when not given), ``seed``, ``strategy``,
    ``odd`` (else *odd*, the run's), ``coverage_from`` (paths to earlier runs'
    coverage files or directories, for ``strategy: coverage``), ``knobs``.

    Raises:
        ValueError: On an unknown field.
    """
    from ..coverage.report import load_and_merge  # noqa: PLC0415
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
        knobs_from_mapping(raw.get("knobs") or {}),
        seed=int(raw.get("seed", 0)),
        strategy=strategy,
        coverage=coverage,
    )
    count = raw.get("count")
    return sampler, None if count is None else int(count)
