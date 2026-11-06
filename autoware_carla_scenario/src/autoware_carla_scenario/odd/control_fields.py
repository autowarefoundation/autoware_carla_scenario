"""The fields of ``controls`` and ``sweep.odd_sample``, and the keys a sweep sets.

:mod:`.controls` checks a config's ``controls`` and :mod:`.sampler` reads them
and ``sweep.odd_sample``; both name the same fields, targets and keys, which
are here once, with :func:`sets_any`, whether a case's overrides set a key.
Split out of them so the library check compiles them (#1733): :mod:`.sampler`
imports openodd (a Rust extension), which :mod:`.controls` may import only
inside a function, and neither compiles.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "CONTROL_FIELDS",
    "CONTROL_TARGETS",
    "FLAKY_FIELDS",
    "MAPPING_FIELDS",
    "ODD_CASE_KEY",
    "SAMPLE_FIELDS",
    "TIRE_FRICTION_KEY",
    "VISIBILITY_KEY",
    "sets_any",
]

#: The config key (an override a sweep adds) that carries a drawn case to its
#: run, encoded (:func:`~.sampler.encode_case`).
ODD_CASE_KEY = "odd_case"

#: The fields a control may have.
CONTROL_FIELDS = (
    "key",
    "target",
    "values",
    "scale",
    "offset",
    "integer",
    "range",
    "limits",
)
#: What a control's ``target`` may say: ``setting`` (the default), a raw
#: setting the drawn value is mapped onto by ``scale`` and ``offset``; or
#: ``measure``, a parameter that takes the measure's value at the start itself,
#: which the scenario solves its placement for.
CONTROL_TARGETS = ("setting", "measure")
#: The fields a ``target: measure`` control refuses: what it writes is the
#: measure's own value, unchanged.
MAPPING_FIELDS = ("scale", "offset", "integer", "values")

#: The fields of ``sweep.odd_sample``: ``tire_friction`` and ``visibility``
#: (default true) say whether each case draws where its grip
#: (:func:`~.sampler.with_tire_friction`) and its visibility
#: (:func:`~.sampler.with_visibility`) are in the ranges its weather leaves;
#: the rest are :func:`~.sampler.search_from_config`'s.
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

#: The setting a case's grip is drawn into: the quantile of the distribution
#: of grip its weather leaves (``EnvironmentAction``'s ``tire_friction_quantile``).
TIRE_FRICTION_KEY = "environment.tire_friction_quantile"
#: The setting a case's visibility is drawn into: the quantile of the
#: distribution of visibility its weather leaves (``EnvironmentAction``'s
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


def sets_any(overrides: Sequence[str], keys: Sequence[str]) -> bool:
    """Whether one of the Hydra *overrides* (``key=value``, ``+key=value`` or
    ``++key=value``) sets one of *keys*."""
    for override in overrides:
        name = override.lstrip("+")
        for key in keys:
            if name.startswith(key + "="):
                return True
    return False
