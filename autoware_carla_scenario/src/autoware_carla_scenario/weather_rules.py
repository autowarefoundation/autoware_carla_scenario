"""The rules the weather's effects follow: the road's grip and how far drivers see.

A run sets the weather; what it does to the road and to the drivers' sight
follows from it, as a range a run's quantile picks a point of
(:func:`~autoware_carla_scenario.actions.environment.tire_friction`,
:func:`~autoware_carla_scenario.visibility.visibility`).  Which range is
written as fuzzy rules, in a YAML file (``weather_rules.yaml`` next to this
module), and inferred with `simpful <https://github.com/aresio/simpful>`_:

* a **quantity** (``tire_friction``, ``visibility``) is one or more rule
  bases, whose ranges ``combine`` (``min`` or ``max``) joins end by end;
* a **rule base** reads some of the weather's fields (:data:`WEATHER_INPUTS`),
  each sorted into fuzzy terms over its universe, and its rules give, per
  combination of terms, the ``low`` and ``high`` ends of the range: Sugeno's
  zero order, so its range is its rules' ends averaged by how strongly each
  fires.

A rule is simpful's antecedent and the two ends::

    - if: (water IS wet) AND (NOT (fog_density IS dense))
      then: {low: 0.65, high: 0.85}

The file is checked when read (:class:`WeatherRulesError`): every term and
rule well formed, every input's universe covered by its terms, and some rule
firing wherever the inputs can be.  A run reads another file with
``environment.rules`` (:func:`use_weather_rules`); its quantities replace
the default file's, quantity by quantity.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from simpful import (
    FuzzySet,
    FuzzySystem,
    Gaussian_MF,
    LinguisticVariable,
    Trapezoidal_MF,
    Triangular_MF,
)

__all__ = [
    "DEFAULT_RULES",
    "QUANTITIES",
    "WEATHER_INPUTS",
    "WeatherRules",
    "WeatherRulesError",
    "active_weather_rules",
    "load_weather_rules",
    "use_weather_rules",
    "weather_range",
]

#: The rules a run follows unless it names others.
DEFAULT_RULES: Path = Path(__file__).resolve().parent / "weather_rules.yaml"

#: The quantities the framework reads a range of.
QUANTITIES: tuple[str, ...] = ("tire_friction", "visibility")

#: The weather's fields a rule base may read, as CARLA's ``WeatherParameters``
#: name them, and ``water``: the larger of ``wetness`` and
#: ``precipitation_deposits``, which the road's grip follows.
WEATHER_INPUTS: tuple[str, ...] = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "fog_density",
    "fog_distance",
    "sun_altitude_angle",
    "sun_azimuth_angle",
    "water",
)

#: The file format this module reads.
_VERSION = 1
#: How the ranges of a quantity's rule bases are joined.
_COMBINE = {"min": min, "max": max}
#: The points per input the check samples, at most, and in all.
_CHECK_POINTS = 201
_CHECK_TOTAL = 20000
#: A name simpful reads as one: a variable, a term.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
#: A clause of an antecedent: ``(variable IS term)``.
_CLAUSE = re.compile(r"\(\s*(\w+)\s+IS\s+(\w+)\s*\)")


class WeatherRulesError(ValueError):
    """A rule file that cannot be read, or a weather its rules say nothing of."""


def _float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeatherRulesError(f"{where}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise WeatherRulesError(f"{where}: expected a finite number, got {value!r}")
    return number


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WeatherRulesError(f"{where}: expected a mapping, got {value!r}")
    return value


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _NAME.match(value):
        raise WeatherRulesError(
            f"{where}: {value!r} is not a name (letters, digits, _)"
        )
    return value


def _term(spec: Any, where: str) -> Any:
    """simpful's membership function for ``{triangle|trapezoid|gaussian: [...]}``."""
    spec = _mapping(spec, where)
    if len(spec) != 1:
        raise WeatherRulesError(
            f"{where}: expected one of triangle, trapezoid, gaussian, got {dict(spec)!r}"
        )
    ((shape, raw),) = spec.items()
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise WeatherRulesError(f"{where}.{shape}: expected a list, got {raw!r}")
    points = [_float(v, f"{where}.{shape}") for v in raw]
    sizes = {"triangle": 3, "trapezoid": 4, "gaussian": 2}
    if shape not in sizes:
        raise WeatherRulesError(
            f"{where}: unknown term shape {shape!r} (triangle, trapezoid, gaussian)"
        )
    if len(points) != sizes[shape]:
        raise WeatherRulesError(
            f"{where}.{shape}: expected {sizes[shape]} numbers, got {len(points)}"
        )
    if shape == "gaussian":
        if not points[1] > 0.0:
            raise WeatherRulesError(f"{where}.gaussian: sigma must be positive")
        return Gaussian_MF(mu=points[0], sigma=points[1])
    if points != sorted(points):
        raise WeatherRulesError(f"{where}.{shape}: the points must not decrease")
    if shape == "triangle":
        return Triangular_MF(*points)
    return Trapezoidal_MF(*points)


@dataclass(frozen=True)
class _Input:
    """An input of a rule base: its universe, terms and default."""

    name: str
    low: float
    high: float
    terms: dict[str, Any]
    default: Optional[float]

    def clamp(self, value: float) -> float:
        return min(max(value, self.low), self.high)

    def samples(self, count: int) -> list[float]:
        step = (self.high - self.low) / (count - 1)
        return [self.low + i * step for i in range(count)]


def _input(name: str, raw: Any, where: str) -> _Input:
    if name not in WEATHER_INPUTS:
        raise WeatherRulesError(
            f"{where}: unknown input {name!r}; a rule base reads {list(WEATHER_INPUTS)}"
        )
    raw = _mapping(raw, where)
    unknown = sorted(set(raw) - {"universe", "terms", "default"})
    if unknown:
        raise WeatherRulesError(f"{where}: unknown fields {unknown}")
    universe = raw.get("universe")
    if not isinstance(universe, Sequence) or len(universe) != 2:
        raise WeatherRulesError(f"{where}.universe: expected [low, high]")
    low = _float(universe[0], f"{where}.universe")
    high = _float(universe[1], f"{where}.universe")
    if not low < high:
        raise WeatherRulesError(f"{where}.universe: low must be below high")
    terms = {
        _name(term, f"{where}.terms"): _term(spec, f"{where}.terms.{term}")
        for term, spec in _mapping(raw.get("terms"), f"{where}.terms").items()
    }
    if not terms:
        raise WeatherRulesError(f"{where}.terms: no terms")
    default = raw.get("default")
    made = _Input(
        name,
        low,
        high,
        terms,
        None if default is None else _float(default, f"{where}.default"),
    )
    for x in made.samples(_CHECK_POINTS):
        if max(mf(x) for mf in terms.values()) <= 0.0:
            raise WeatherRulesError(
                f"{where}.terms: no term covers {name} = {x:g}, within its universe"
            )
    return made


class _RuleBase:
    """One rule base: a simpful Sugeno system with outputs ``low`` and ``high``."""

    def __init__(self, name: str, raw: Any, where: str) -> None:
        raw = _mapping(raw, where)
        unknown = sorted(set(raw) - {"inputs", "rules"})
        if unknown:
            raise WeatherRulesError(f"{where}: unknown fields {unknown}")
        self.name = name
        self.inputs = [
            _input(_name(n, f"{where}.inputs"), spec, f"{where}.inputs.{n}")
            for n, spec in _mapping(raw.get("inputs"), f"{where}.inputs").items()
        ]
        if not self.inputs:
            raise WeatherRulesError(f"{where}.inputs: no inputs")
        rules = raw.get("rules")
        if not isinstance(rules, Sequence) or isinstance(rules, str) or not rules:
            raise WeatherRulesError(f"{where}.rules: expected a list of rules")
        self._lock = threading.Lock()
        # simpful announces the model type its first output makes it: aloud.
        with contextlib.redirect_stdout(io.StringIO()):
            self._system = FuzzySystem(show_banner=False, verbose=False)
            for i in self.inputs:
                self._system.add_linguistic_variable(
                    i.name,
                    LinguisticVariable(
                        [FuzzySet(function=mf, term=t) for t, mf in i.terms.items()],
                        universe_of_discourse=[i.low, i.high],
                    ),
                )
            texts = [
                text
                for index, rule in enumerate(rules)
                for text in self._rule(index, rule, where)
            ]
            try:
                self._system.add_rules(texts)
            except Exception as e:  # simpful raises Exception
                raise WeatherRulesError(f"{where}.rules: {e}") from e
        self._check(where)

    def _rule(self, index: int, raw: Any, where: str) -> tuple[str, str]:
        """simpful's two rules for rule *index*, one per end; the ends become
        crisp outputs."""
        where = f"{where}.rules[{index}]"
        raw = _mapping(raw, where)
        unknown = sorted(set(raw) - {"if", "then"})
        if unknown:
            raise WeatherRulesError(f"{where}: unknown fields {unknown}")
        antecedent = raw.get("if")
        if not isinstance(antecedent, str) or not antecedent.strip():
            raise WeatherRulesError(f"{where}.if: expected an antecedent")
        clauses = _CLAUSE.findall(antecedent)
        if not clauses:
            raise WeatherRulesError(
                f"{where}.if: expected clauses such as (water IS wet), got {antecedent!r}"
            )
        terms = {i.name: i.terms for i in self.inputs}
        for variable, term in clauses:
            if variable not in terms:
                raise WeatherRulesError(
                    f"{where}.if: {variable!r} is no input of this rule base "
                    f"({sorted(terms)})"
                )
            if term not in terms[variable]:
                raise WeatherRulesError(
                    f"{where}.if: {variable} has no term {term!r} ({sorted(terms[variable])})"
                )
        then = _mapping(raw.get("then"), f"{where}.then")
        if set(then) != {"low", "high"}:
            raise WeatherRulesError(
                f"{where}.then: expected low and high, got {dict(then)!r}"
            )
        low = _float(then["low"], f"{where}.then.low")
        high = _float(then["high"], f"{where}.then.high")
        if not low <= high:
            raise WeatherRulesError(f"{where}.then: low must not be above high")
        self._system.set_crisp_output_value(f"r{index}_low", low)
        self._system.set_crisp_output_value(f"r{index}_high", high)
        return (
            f"IF ({antecedent}) THEN (low IS r{index}_low)",
            f"IF ({antecedent}) THEN (high IS r{index}_high)",
        )

    def _check(self, where: str) -> None:
        """Some rule fires at every point of a grid over the inputs' universes."""
        per_input = max(
            2, min(_CHECK_POINTS, int(_CHECK_TOTAL ** (1 / len(self.inputs))))
        )
        grids = [i.samples(per_input) for i in self.inputs]
        for point in itertools.product(*grids):
            values = {i.name: v for i, v in zip(self.inputs, point)}
            try:
                self.range(values)
            except WeatherRulesError as e:
                raise WeatherRulesError(f"{where}: {e}") from None
            except Exception as e:  # simpful raises Exception
                raise WeatherRulesError(f"{where}.rules: {e}") from e

    def range(self, values: Mapping[str, float]) -> tuple[float, float]:
        """``(low, high)`` at *values*, each input clamped to its universe."""
        with self._lock:
            for i in self.inputs:
                self._system.set_variable(i.name, i.clamp(values[i.name]))
            if sum(self._system.get_firing_strengths()) <= 0.0:
                raise WeatherRulesError(
                    f"no rule fires at {dict((i.name, values[i.name]) for i in self.inputs)}"
                )
            out = self._system.Sugeno_inference(["low", "high"])
        return float(out["low"]), float(out["high"])


class _Quantity:
    def __init__(self, name: str, raw: Any, source: Path) -> None:
        where = f"{source}: quantities.{name}"
        raw = _mapping(raw, where)
        unknown = sorted(set(raw) - {"combine", "rule_bases"})
        if unknown:
            raise WeatherRulesError(f"{where}: unknown fields {unknown}")
        combine = raw.get("combine", "min")
        if combine not in _COMBINE:
            raise WeatherRulesError(
                f"{where}.combine: expected min or max, got {combine!r}"
            )
        self.name = name
        self.source = source
        self.combine = str(combine)
        self.rule_bases = [
            _RuleBase(_name(n, f"{where}.rule_bases"), spec, f"{where}.rule_bases.{n}")
            for n, spec in _mapping(
                raw.get("rule_bases"), f"{where}.rule_bases"
            ).items()
        ]
        if not self.rule_bases:
            raise WeatherRulesError(f"{where}.rule_bases: no rule bases")

    def range(self, values: Mapping[str, float]) -> tuple[float, float]:
        ranges = []
        for base in self.rule_bases:
            given = {}
            for i in base.inputs:
                if i.name in values:
                    given[i.name] = float(values[i.name])
                elif i.default is not None:
                    given[i.name] = i.default
                else:
                    raise WeatherRulesError(
                        f"{self.source}: {self.name}.{base.name} reads {i.name}, "
                        "which was not given and has no default"
                    )
            try:
                ranges.append(base.range(given))
            except WeatherRulesError as e:
                raise WeatherRulesError(
                    f"{self.source}: {self.name}.{base.name}: {e}"
                ) from None
        join = _COMBINE[self.combine]
        return join(r[0] for r in ranges), join(r[1] for r in ranges)


def _read(path: Path) -> dict[str, _Quantity]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise WeatherRulesError(f"cannot read weather rules {path}: {e}") from e
    except yaml.YAMLError as e:
        raise WeatherRulesError(f"{path}: not YAML: {e}") from e
    raw = _mapping(raw, str(path))
    unknown = sorted(set(raw) - {"version", "quantities"})
    if unknown:
        raise WeatherRulesError(f"{path}: unknown fields {unknown}")
    if raw.get("version") != _VERSION:
        raise WeatherRulesError(
            f"{path}: version must be {_VERSION}, got {raw.get('version')!r}"
        )
    quantities = _mapping(raw.get("quantities"), f"{path}: quantities")
    unknown = sorted(set(quantities) - set(QUANTITIES))
    if unknown:
        raise WeatherRulesError(
            f"{path}: unknown quantities {unknown}; the framework reads {list(QUANTITIES)}"
        )
    return {str(n): _Quantity(str(n), spec, path) for n, spec in quantities.items()}


class WeatherRules:
    """The rules of every quantity, read from :data:`DEFAULT_RULES` and, over
    it, *path* (:func:`load_weather_rules`)."""

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        self.path: Optional[Path] = None if path is None else Path(path).resolve()
        self._quantities = _read(DEFAULT_RULES)
        missing = sorted(set(QUANTITIES) - set(self._quantities))
        if missing:  # pragma: no cover -- the packaged file has them all
            raise WeatherRulesError(f"{DEFAULT_RULES}: no rules for {missing}")
        if self.path is not None:
            self._quantities.update(_read(self.path))

    def source(self, quantity: str) -> Path:
        """The file *quantity*'s rules were read from."""
        return self._quantity(quantity).source

    def range(self, quantity: str, inputs: Mapping[str, float]) -> tuple[float, float]:
        """The ``(low, high)`` range of *quantity* in the weather *inputs*.

        *inputs* names the weather's fields (:data:`WEATHER_INPUTS`);
        ``water`` is derived from ``wetness`` and ``precipitation_deposits``
        when not given.

        Raises:
            WeatherRulesError: On an unknown quantity, an input its rules read
                that *inputs* lacks (and has no default), or a weather no rule
                fires at.
        """
        values = {str(k): float(v) for k, v in inputs.items()}
        if "water" not in values and (
            "wetness" in values or "precipitation_deposits" in values
        ):
            values["water"] = max(
                values.get("wetness", 0.0), values.get("precipitation_deposits", 0.0)
            )
        return self._quantity(quantity).range(values)

    def _quantity(self, quantity: str) -> _Quantity:
        if quantity not in self._quantities:
            raise WeatherRulesError(
                f"unknown quantity {quantity!r}; the rules give {sorted(self._quantities)}"
            )
        return self._quantities[quantity]


#: Path (``None`` for the default) and the file's modification time -> its rules.
_LOADED: dict[tuple[Optional[Path], int], WeatherRules] = {}
#: The rules :func:`weather_range` follows; the default until a run names others.
_ACTIVE: Optional[WeatherRules] = None


def load_weather_rules(path: Union[str, Path, None] = None) -> WeatherRules:
    """The rules of *path* over the default ones (the default alone for
    ``None``), read once per version of the file.

    Raises:
        WeatherRulesError: When a file cannot be read or its rules are not
            well formed.
    """
    resolved = None if path is None else Path(path).resolve()
    try:
        stamp = 0 if resolved is None else resolved.stat().st_mtime_ns
    except OSError as e:
        raise WeatherRulesError(f"cannot read weather rules {resolved}: {e}") from e
    key = (resolved, stamp)
    if key not in _LOADED:
        _LOADED[key] = WeatherRules(resolved)
    return _LOADED[key]


def use_weather_rules(path: Union[str, Path, None] = None) -> WeatherRules:
    """Follow *path*'s rules (the default ones for ``None``) from now on; the
    runner calls this for every run with its config's ``environment.rules``."""
    global _ACTIVE
    _ACTIVE = load_weather_rules(path)
    return _ACTIVE


def active_weather_rules() -> WeatherRules:
    """The rules :func:`weather_range` follows."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_weather_rules(None)
    return _ACTIVE


def weather_range(quantity: str, inputs: dict[str, float]) -> tuple[float, float]:
    """The ``(low, high)`` range of *quantity* in the weather *inputs*, by the
    rules followed (:meth:`WeatherRules.range`)."""
    return active_weather_rules().range(quantity, inputs)
