"""Distributions a sweep draws a scalar parameter from.

``sweep.constraints`` maps the parameter it enumerates (a lanelet, such as
``ego.spawn_lanelet_id``) to the constraints its lanelets must meet.  Any other
key there names a scalar parameter of the scenario and holds the one
distribution its value is drawn from, for each case of the sweep::

    sweep:
      constraints:
        ego.spawn_lanelet_id:
          - type: has_traffic_light_stop_line
        scenario.stop_line_distance_m:
          - type: uniform
            low: 0.5
            high: 3.0
        ego.initial_speed_kmh:
          - type: normal
            mean: 40.0
            std: 5.0
            low: 30.0      # optional: truncate the normal to [low, high]
            high: 50.0
      samples: 6           # draws per case (1 when not given)
      seed: 7              # optional; the run's seed when not given

Each case of the sweep (a lanelet the constraints match, a route match, a case
drawn from the ODD, or the one case of a sweep with none of them) runs
``samples`` times, each time with a value drawn for every distribution.  Each
value is drawn from a seed of its own, a hash of ``sweep.seed`` (else the
run's), the scenario's name, the case's overrides, the sample's number and the
key (:func:`fnv1a`).  So the same config and seed draw the same values; a case
keeps its values when the sweep gains or loses other cases (a map update, a
constraint changed); adding a distribution does not move the others; two
scenarios drawing the same key do not draw the same values; and ``sweep.seed``
draws other values without moving the run's other randomness (the traffic, the
pedestrians).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from ..number_settings import finite_number_setting

__all__ = [
    "NormalDistribution",
    "UniformDistribution",
    "draw_case",
    "fnv1a",
    "is_distribution",
    "parse_distribution",
]

#: The standard normal distribution.
_STANDARD = NormalDist()


@dataclass(frozen=True)
class UniformDistribution:
    """Every value in ``[low, high]`` equally likely."""

    low: float
    high: float

    def draw(self, rng: random.Random) -> float:
        """A value of the distribution, from *rng*."""
        return rng.uniform(self.low, self.high)

    def truncated(self, low: float, high: float) -> UniformDistribution:
        """The distribution inside ``[low, high]`` (``low < high``, both within
        its own bounds)."""
        return UniformDistribution(low=low, high=high)


@dataclass(frozen=True)
class NormalDistribution:
    """A normal distribution of ``mean`` and standard deviation ``std``.

    With ``low`` and/or ``high`` it is truncated to ``[low, high]``: drawn from
    the normal's own shape inside the bounds, never clipped to them.  An
    infinite bound is no bound.
    """

    mean: float
    std: float
    low: float = -math.inf
    high: float = math.inf

    def mass(self) -> float:
        """The normal's probability inside ``[low, high]``."""
        return _STANDARD.cdf((self.high - self.mean) / self.std) - _STANDARD.cdf(
            (self.low - self.mean) / self.std
        )

    def truncated(self, low: float, high: float) -> NormalDistribution:
        """The distribution inside ``[low, high]`` (``low < high``, both within
        its own bounds): this normal truncated there.  Where its :meth:`mass`
        is 0 it draws at a bound."""
        return NormalDistribution(mean=self.mean, std=self.std, low=low, high=high)

    def draw(self, rng: random.Random) -> float:
        """A value of the distribution, from *rng*: drawn uniformly between the
        bounds' quantiles and taken through the normal's quantile function, so
        a truncated normal is one draw, not a rejection loop."""
        lo = _STANDARD.cdf((self.low - self.mean) / self.std)
        hi = _STANDARD.cdf((self.high - self.mean) / self.std)
        # Kept inside (0, 1), which the quantile function needs: random() can
        # return 0.0, and a far bound can put lo or hi at 0 or 1.
        p = min(max(rng.uniform(lo, hi), 1e-300), 1.0 - 1e-16)
        value = self.mean + self.std * _STANDARD.inv_cdf(p)
        # Rounding at a bound may step a hair outside it.
        return min(max(value, self.low), self.high)


#: FNV-1a's 64-bit offset basis: the hash of no text.
FNV_OFFSET = 0xCBF29CE484222325


def fnv1a(text: str, h: int = FNV_OFFSET) -> int:
    """64-bit FNV-1a of *text*'s code points, continuing from the hash *h* of
    the text before it: so a prefix shared by many draws is hashed once.
    Stable across runs and Python versions (``hash()`` of a string is not)."""
    for char in text:
        h ^= ord(char)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def draw_case(distributions: Mapping[str, Any], h: int) -> dict[str, float]:
    """A value of every distribution of *distributions* (key -> distribution)
    for one draw, *h* being the hash of its identity (:func:`fnv1a`): each
    from a generator seeded with that hash continued by its key."""
    return {
        key: distributions[key].draw(
            random.Random(fnv1a(f"\x1f{key}", h) & 0x7FFFFFFFFFFFFFFF)
        )
        for key in sorted(distributions)
    }


#: The fields each distribution takes (lists: Codon types a tuple by its length).
_FIELDS: dict[str, list[str]] = {
    "uniform": ["type", "low", "high"],
    "normal": ["type", "mean", "std", "low", "high"],
}


def is_distribution(constraint_list: Sequence[Any]) -> bool:
    """Whether a ``sweep.constraints`` entry (*constraint_list*, its list of
    items) names a distribution: an item whose ``type`` is one of them."""
    return any(
        isinstance(item, Mapping) and item.get("type") in _FIELDS
        for item in constraint_list
    )


def _number(key: str, spec: Mapping[str, Any], name: str, default: float) -> float:
    """The distribution field *name* of *key* (*spec*'s), a finite number;
    *default* when it is missing (``math.nan``: required).

    Raises:
        ValueError: If it is required and missing, or not a finite number.
    """
    value = spec.get(name)
    if value is None:
        if math.isnan(default):
            raise ValueError(f"sweep.constraints.{key}.{name} is required")
        return default
    return finite_number_setting(f"sweep.constraints.{key}.{name}", value)


def parse_distribution(key: str, constraint_list: Sequence[Any]) -> Any:
    """The distribution ``sweep.constraints.<key>`` (*constraint_list*) holds.

    Raises:
        ValueError: Naming the key, if the list holds anything but one
            distribution, a field is unknown, missing or not a finite number,
            ``low`` is not below ``high``, ``std`` is not positive, or a
            truncated normal's bounds leave it no probability to draw from.
    """
    spec = constraint_list[0] if len(constraint_list) == 1 else None
    kind = spec.get("type") if isinstance(spec, Mapping) else None
    if spec is None or kind not in _FIELDS:
        raise ValueError(
            f"sweep.constraints.{key} draws its value from a distribution, so it "
            "holds that one distribution (- type: uniform or - type: normal) and "
            f"nothing else; got {len(constraint_list)} item(s)"
        )
    for name in spec:
        if str(name) not in _FIELDS[kind]:
            raise ValueError(
                f"sweep.constraints.{key}: a {kind} distribution takes "
                f"{', '.join(_FIELDS[kind][1:])}, not {name!r}"
            )
    if kind == "uniform":
        low = _number(key, spec, "low", math.nan)
        high = _number(key, spec, "high", math.nan)
    else:
        low = _number(key, spec, "low", -math.inf)
        high = _number(key, spec, "high", math.inf)
    if not low < high:
        raise ValueError(
            f"sweep.constraints.{key}: low ({low}) must be below high ({high})"
            + (
                "; for one value, set it as an override instead"
                if kind == "uniform"
                else ""
            )
        )
    if kind == "uniform":
        return UniformDistribution(low=low, high=high)
    std = _number(key, spec, "std", math.nan)
    if not std > 0.0:
        raise ValueError(f"sweep.constraints.{key}: std must be above 0, got {std}")
    normal = NormalDistribution(
        mean=_number(key, spec, "mean", math.nan), std=std, low=low, high=high
    )
    if not normal.mass() > 1e-12:
        raise ValueError(
            f"sweep.constraints.{key}: [{low}, {high}] is too far into the tail of "
            f"a normal of mean {normal.mean} and std {std} to draw from"
        )
    return normal
