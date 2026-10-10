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
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ..coverage.items import CoverItem, value_label
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
        values: Per bucket label.  For a categorical attribute, the value to
            write, or ``[low, high)`` to draw it from; a categorical bucket
            without one cannot be drawn.  For a numeric attribute, a range in
            the attribute's unit to draw from instead of the bucket's
            interval -- which an unbounded bucket (``-inf``/``inf`` edges)
            needs to be drawn at all.
        scale: A numeric attribute's value, drawn in its unit, is written as
            ``value * scale + offset`` -- e.g. 1/3.6 for an attribute in km/h
            and a key in m/s.  Must not be zero.
        offset: See *scale*.
        integer: Write an integer: a numeric value rounded (a case whose
            rounded value falls in another bucket is drawn again), a
            categorical range drawn as an integer.
    """

    key: str
    values: Mapping[str, KnobValue] = field(default_factory=dict)
    scale: float = 1.0
    offset: float = 0.0
    integer: bool = False

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("OddKnob: key must not be empty")
        if self.scale == 0:
            raise ValueError(f"OddKnob({self.key}): scale must not be zero")
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
            (one of its values for a categorical attribute, a number in its
            unit for a numeric one).
        situation: The situation the case was aimed at, if any.
    """

    index: int
    overrides: list[str]
    buckets: dict[str, str]
    values: dict[str, Any]
    situation: Optional[str] = None


def _render(value: Any) -> str:
    """A value the way a Hydra override reads it back -- exactly.

    A float is written in full: rounded, a value drawn just below a level's
    upper edge would be read back as the next level.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return repr(value)
    return str(value)


#: The suffix a coverage report gives an item defined differently in
#: different runs (``odd.environment.rain#2``).
_VARIANT = re.compile(r"#\d+$")


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
            values={value_label(k): v for k, v in (spec.get("values") or {}).items()},
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

        drawn = {axis.attribute.name for axis in self._axes}
        #: Situation -> whether the knobs alone decide it: every attribute it
        #: tests is drawn, and it refers to no other module.
        self._decided: dict[str, bool] = {}
        for module in odd.situations():
            conditions = module._conditions()
            tested = {a.name for c in conditions for a in c._attributes()}
            refers = any(c._references() for c in conditions)
            self._decided[module.name] = not refers and tested <= drawn

        self._amount: dict[tuple[str, str], float] = {}
        self._situation_holes: list[str] = []
        if strategy == "coverage":
            self._read_coverage(coverage)
        #: Cases drawn per (attribute, bucket) so far in this batch.
        self._planned: dict[tuple[str, str], int] = {}

    # -- setup -----------------------------------------------------------

    @staticmethod
    def _drawable(item: CoverItem, knob: OddKnob, label: str) -> bool:
        if not item.numeric:
            return label in knob.values
        if label in knob.values:
            return True
        index = item.labels.index(label)
        # An unbounded bucket has no interval to draw from.
        return math.isfinite(item.edges[index]) and math.isfinite(item.edges[index + 1])

    def _read_coverage(self, coverage: Any) -> None:
        """Remember how covered each bucket is, and which situations are holes.

        An entry counts when its name (less a ``#n`` variant suffix) and its
        buckets are the ones this ODD defines.  Every situation whose entry is
        missing or has a hole is a hole.
        """
        items = {f"odd.{a.name}": a for a in self.odd.attributes if a.item is not None}
        situations = {f"odd.situation.{m.name}": m for m in self.odd.situations()}
        covered_situations: set[str] = set()
        entries = [] if coverage is None else coverage.entries
        for entry in entries:
            name = _VARIANT.sub("", entry.name)
            module = situations.get(name)
            if module is not None:
                assert module.item is not None  # noqa: S101 - a situation has one
                if list(entry.buckets) == module.item.labels and not entry.holes:
                    covered_situations.add(module.name)
                continue
            attribute = items.get(name)
            if attribute is None or attribute.item is None:
                continue
            if list(entry.buckets) != attribute.item.labels:
                continue
            target = entry.target if entry.target > 0 else 1.0
            for bucket in entry.buckets:
                key = (attribute.name, bucket)
                self._amount[key] = (
                    self._amount.get(key, 0.0) + entry.amount(bucket) / target
                )
        self._situation_holes = [
            m.name for m in self.odd.situations() if m.name not in covered_situations
        ]

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

    def _verdict(self, values: dict[str, Any]) -> Any:
        """The ODD's verdict, with the attributes derived from others worked out."""
        for attribute in self.odd._derived:
            if attribute.name in values and values[attribute.name] is not _OPEN:
                continue
            derived = attribute.probe.from_values(values)  # type: ignore[attr-defined]
            values[attribute.name] = _OPEN if derived is None else derived
        return self.odd.evaluate(values)

    def _admits(
        self, values: dict[str, Any], situation: Optional[str], *, strict: bool
    ) -> bool:
        """Whether the ODD admits *values*, and *situation* can hold under them.

        *strict*: a situation the knobs alone decide must hold, not just not fail.
        """
        verdict = self._verdict(values)
        if not verdict.inside:
            return False
        if situation is None:
            return True
        holds = verdict.modules.get(situation)
        if strict and self._decided.get(situation, False):
            return holds is True
        return holds is not False

    def _uniform(
        self, low: float, high: float, *, closed: bool, integer: bool
    ) -> Optional[float]:
        """A value in ``[low, high)`` (``[low, high]`` if *closed*); ``None`` if none."""
        if low == high:
            return low
        if integer:
            first = math.ceil(low)
            last = math.floor(high) if closed else math.ceil(high) - 1
            return None if first > last else self._rng.randint(first, last)
        value = self._rng.uniform(low, high)
        if not closed and value >= high:
            value = math.nextafter(high, low)
        return value

    def _draw_value(self, axis: _Axis, label: str) -> Optional[tuple[Any, Any]]:
        """(value in the attribute's terms, value written), or ``None`` to draw again."""
        knob, item = axis.knob, axis.item
        index = item.labels.index(label)
        given = knob.values.get(label)
        if not item.numeric:
            assert isinstance(item.values, list)  # noqa: S101 - categorical
            value = item.values[index]
            if not _is_range(given):
                return value, given
            chosen = self._uniform(*_range(given), closed=False, integer=knob.integer)
            return None if chosen is None else (value, chosen)

        if given is not None:
            if _is_range(given):
                low, high = _range(given)
            else:
                low = high = float(given)  # type: ignore[arg-type]
            closed = True
        else:
            low, high = item.edges[index], item.edges[index + 1]
            closed = index == len(item.labels) - 1
        drawn = self._uniform(low, high, closed=closed, integer=False)
        if drawn is None or not math.isfinite(drawn):
            return None
        written: Any = float(drawn) * knob.scale + knob.offset
        if knob.integer:
            written = int(round(written))
        # What a run will read back, which must still be in the bucket drawn.
        value = (float(_render(written)) - knob.offset) / knob.scale
        if not math.isfinite(value) or item.bucket_of(value) != label:
            return None
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
            as_buckets: dict[str, Any] = {a.name: _OPEN for a in self.odd.attributes}
            for axis in self._axes:
                label = buckets[axis.attribute.name]
                as_buckets[axis.attribute.name] = _Bucket(
                    axis.item, axis.item.labels.index(label)
                )
            if not self._admits(as_buckets, situation, strict=False):
                continue
            values: dict[str, Any] = {}
            overrides: list[str] = []
            for axis in self._axes:
                drawn = self._draw_value(axis, buckets[axis.attribute.name])
                if drawn is None:
                    break
                values[axis.attribute.name] = drawn[0]
                overrides.append(f"{axis.knob.key}={_render(drawn[1])}")
            else:
                as_values: dict[str, Any] = {a.name: _OPEN for a in self.odd.attributes}
                as_values.update(values)
                if not self._admits(as_values, situation, strict=True):
                    continue
                for name, label in buckets.items():
                    key = (name, label)
                    self._planned[key] = self._planned.get(key, 0) + 1
                return OddSample(index, overrides, buckets, values, situation)
        aim = f" aimed at situation {situation}" if situation else ""
        raise ValueError(
            f"OddSampler: found no case{aim} inside ODD {self.odd.name} in "
            f"{self._max_tries} tries"
        )

    def sample(self, count: int) -> list[OddSample]:
        """Draw *count* cases.

        With the ``coverage`` strategy, the first cases are aimed at the
        situations still uncovered, one each.  A situation the knobs alone
        decide must hold in its case; one that also tests what no knob sets
        (a speed, a road) gets a case under which it can hold.  One the ODD
        and the knobs leave no case for is given up on (and logged), and the
        case is drawn without it.
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
                        "OddSampler: no case brings about situation %s; not aiming at it",
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
