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

Each case of the sweep (a lanelet the constraints match, a route match, or the
one case of a sweep with neither) runs ``samples`` times, each time with a value
drawn for every distribution.  The draws follow the run's seed: the same config
and seed draw the same values, whatever else changes, and each parameter has a
stream of its own (seeded from the seed and its key), so adding a parameter does
not move the values the others draw.

:func:`parse_distribution` reads one from its YAML, :func:`draw` draws a value
and :func:`draw_values` the values of every case.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "DISTRIBUTION_TYPES",
    "NormalDistribution",
    "UniformDistribution",
    "draw",
    "draw_values",
    "is_distribution",
    "key_seed",
    "normal_ppf",
    "parse_distribution",
]

#: The ``type`` of each distribution a sweep takes.
DISTRIBUTION_TYPES: tuple[str, ...] = ("uniform", "normal")


@dataclass(frozen=True)
class UniformDistribution:
    """Every value in ``[low, high]`` equally likely."""

    low: float
    high: float


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


def normal_cdf(x: float) -> float:
    """The standard normal distribution's CDF at *x*."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# Acklam's rational approximation of the standard normal quantile.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def normal_ppf(p: float) -> float:
    """The standard normal quantile of *p*, ``0 < p < 1``.

    Acklam's approximation (relative error under 1.2e-9), refined by one
    Halley step on :func:`normal_cdf` to about machine precision.

    Raises:
        ValueError: If *p* is not strictly between 0 and 1.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"a normal quantile needs 0 < p < 1, got {p}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (
            ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        ) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
            * q
        ) / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(
            ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        ) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    # One Halley step.
    e = normal_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def draw_uniform(distribution: UniformDistribution, rng: random.Random) -> float:
    """A value of *distribution*."""
    return distribution.low + (distribution.high - distribution.low) * rng.random()


def draw_normal(distribution: NormalDistribution, rng: random.Random) -> float:
    """A value of *distribution*: its bounds' quantiles, and one drawn
    uniformly between them, through the normal's quantile function.  So a
    truncated normal is one draw, not a rejection loop, and the value is
    always inside the bounds."""
    lo = 0.0
    hi = 1.0
    if not math.isinf(distribution.low):
        lo = normal_cdf((distribution.low - distribution.mean) / distribution.std)
    if not math.isinf(distribution.high):
        hi = normal_cdf((distribution.high - distribution.mean) / distribution.std)
    p = lo + (hi - lo) * rng.random()
    # rng.random() can return 0.0, and a bound at the far tail can make lo or
    # hi exactly 0 or 1: keep p inside (0, 1).
    p = min(max(p, 1e-300), 1.0 - 1e-16)
    value = distribution.mean + distribution.std * normal_ppf(p)
    # Rounding at a bound may step a hair outside it.
    return min(max(value, distribution.low), distribution.high)


def draw(distribution: Any, rng: random.Random) -> float:
    """A value of *distribution* (a :class:`UniformDistribution` or a
    :class:`NormalDistribution`), from *rng*."""
    if isinstance(distribution, UniformDistribution):
        return draw_uniform(distribution, rng)
    return draw_normal(distribution, rng)


def key_seed(seed: int, key: str) -> int:
    """The seed of *key*'s stream of draws, from the run's *seed*: FNV-1a of
    the key's code points, mixed with the seed.  Stable across runs and
    Python versions (``hash()`` of a string is not)."""
    h = 0xCBF29CE484222325
    for char in key:
        h ^= ord(char)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return (h ^ (seed & 0xFFFFFFFFFFFFFFFF)) & 0x7FFFFFFFFFFFFFFF


def draw_values(
    distributions: Mapping[str, Any], count: int, seed: int
) -> list[dict[str, float]]:
    """*count* draws of every distribution of *distributions* (key ->
    distribution), each a mapping of key -> value, from *seed*.  Each key
    draws its values in turn from a stream of its own (:func:`key_seed`)."""
    draws: list[dict[str, float]] = [{} for _ in range(count)]
    for key in sorted(distributions):
        rng = random.Random(key_seed(seed, key))
        for values in draws:
            values[key] = draw(distributions[key], rng)
    return draws


def is_distribution(constraint_list: Sequence[Any]) -> bool:
    """Whether a ``sweep.constraints`` entry (*constraint_list*, its list of
    items) names a distribution: an item whose ``type`` is one of
    :data:`DISTRIBUTION_TYPES`."""
    return any(
        isinstance(item, Mapping) and item.get("type") in DISTRIBUTION_TYPES
        for item in constraint_list
    )


def _number(key: str, name: str, value: Any) -> float:
    """*value*, the distribution field *name* of *key*, as a finite float.

    Raises:
        ValueError: If it is missing, a bool, not a number, or not finite.
    """
    where = f"sweep.constraints.{key}: {name}"
    if value is None:
        raise ValueError(f"{where} is required")
    # A bool is an int to Python, and text is no number even when it reads as
    # one; numpy's numbers convert.
    if isinstance(value, (bool, str)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite, got {value!r}")
    return number


#: The fields each distribution takes (lists: Codon types a tuple by its length).
_FIELDS: dict[str, list[str]] = {
    "uniform": ["type", "low", "high"],
    "normal": ["type", "mean", "std", "low", "high"],
}


def parse_distribution(key: str, constraint_list: Sequence[Any]) -> Any:
    """The distribution ``sweep.constraints.<key>`` (*constraint_list*) holds.

    Raises:
        ValueError: Naming the key, if the list holds anything but one
            distribution, the distribution's ``type`` is unknown, a field is
            unknown, missing or not a finite number, ``low`` is not below
            ``high``, ``std`` is not positive, or a truncated normal's bounds
            leave it no probability to draw from.
    """
    if len(constraint_list) != 1 or not isinstance(constraint_list[0], Mapping):
        raise ValueError(
            f"sweep.constraints.{key} draws its value from a distribution, so it "
            "holds that one distribution (- type: uniform or - type: normal) and "
            f"nothing else; got {len(constraint_list)} item(s)"
        )
    spec = constraint_list[0]
    kind = spec.get("type")
    if kind not in _FIELDS:
        raise ValueError(
            f"sweep.constraints.{key}: unknown distribution type {kind!r}; "
            f"expected one of {list(DISTRIBUTION_TYPES)}"
        )
    for name in spec:
        if str(name) not in _FIELDS[kind]:
            raise ValueError(
                f"sweep.constraints.{key}: a {kind} distribution takes "
                f"{', '.join(_FIELDS[kind][1:])}, not {name!r}"
            )
    if kind == "uniform":
        low = _number(key, "low", spec.get("low"))
        high = _number(key, "high", spec.get("high"))
        if not low < high:
            raise ValueError(
                f"sweep.constraints.{key}: low ({low}) must be below high ({high}); "
                "for one value, set it as an override instead"
            )
        return UniformDistribution(low=low, high=high)
    mean = _number(key, "mean", spec.get("mean"))
    std = _number(key, "std", spec.get("std"))
    if not std > 0.0:
        raise ValueError(f"sweep.constraints.{key}: std must be above 0, got {std}")
    low = -math.inf if spec.get("low") is None else _number(key, "low", spec.get("low"))
    high = (
        math.inf if spec.get("high") is None else _number(key, "high", spec.get("high"))
    )
    if not low < high:
        raise ValueError(
            f"sweep.constraints.{key}: low ({low}) must be below high ({high})"
        )
    mass = normal_cdf((high - mean) / std) - normal_cdf((low - mean) / std)
    if not mass > 1e-12:
        raise ValueError(
            f"sweep.constraints.{key}: [{low}, {high}] is too far into the tail of "
            f"a normal of mean {mean} and std {std} to draw from"
        )
    return NormalDistribution(mean=mean, std=std, low=low, high=high)
