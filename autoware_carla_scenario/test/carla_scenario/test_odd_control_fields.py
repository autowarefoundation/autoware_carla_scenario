"""odd.control_fields' names against the copies they replace (#1733).

The values are odd.sampler's and odd.controls' as they were before the move,
copied verbatim; odd.sampler and odd.controls must hold the moved objects
themselves, and :func:`sets_any` must answer as ``_with_quantile``'s test did.
"""

from __future__ import annotations

import itertools

import pytest

from autoware_carla_scenario.odd import control_fields, controls, sampler

_ORIGINAL = {
    "ODD_CASE_KEY": "odd_case",
    "CONTROL_FIELDS": (
        "key",
        "target",
        "values",
        "scale",
        "offset",
        "integer",
        "range",
        "limits",
    ),
    "CONTROL_TARGETS": ("setting", "measure"),
    "MAPPING_FIELDS": ("scale", "offset", "integer", "values"),
    "SAMPLE_FIELDS": {
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
    },
    "TIRE_FRICTION_KEY": "environment.tire_friction_quantile",
    "VISIBILITY_KEY": "environment.visibility_quantile",
    "FLAKY_FIELDS": {
        "explore_fraction",
        "pool",
        "coverage_depth",
        "threshold",
        "epsilon",
        "perturbations",
        "draws",
        "directions",
        "scenario",
    },
}


@pytest.mark.parametrize("name", sorted(_ORIGINAL))
def test_the_moved_values_and_types_are_unchanged(name: str) -> None:
    value = getattr(control_fields, name)
    assert type(value) is type(_ORIGINAL[name])
    assert value == _ORIGINAL[name]


@pytest.mark.parametrize(
    "name",
    [
        "ODD_CASE_KEY",
        "CONTROL_TARGETS",
        "SAMPLE_FIELDS",
        "TIRE_FRICTION_KEY",
        "VISIBILITY_KEY",
        "FLAKY_FIELDS",
    ],
)
def test_sampler_holds_the_moved_objects(name: str) -> None:
    assert getattr(sampler, name) is getattr(control_fields, name)


def test_sampler_still_exports_its_public_names() -> None:
    for name in (
        "ODD_CASE_KEY",
        "CONTROL_TARGETS",
        "TIRE_FRICTION_KEY",
        "VISIBILITY_KEY",
    ):
        assert name in sampler.__all__
    assert sampler._MAPPING_FIELDS is control_fields.MAPPING_FIELDS


def test_controls_holds_the_moved_objects() -> None:
    assert controls._FIELDS is control_fields.CONTROL_FIELDS
    assert controls._CONTROL_TARGETS is control_fields.CONTROL_TARGETS
    assert controls._MAPPING_FIELDS is control_fields.MAPPING_FIELDS


def _original_sets_any(overrides: list[str], key: str, *fixed: str) -> bool:
    """``_with_quantile``'s test before the move, copied verbatim."""
    given = tuple(f"{k}=" for k in (key, *fixed))
    return any(o.lstrip("+").startswith(given) for o in overrides)


_OVERRIDES = [
    "environment.visibility_quantile=0.5",
    "+environment.visibility_quantile=0.5",
    "++environment.visibility_m=30",
    "environment.visibility_quantile_extra=1",
    "environment.visibility=1",
    "a.b=1",
    "+",
    "",
    "=x",
    "environment.tire_friction=0.7",
    "+++environment.tire_friction_quantile=0.1",
]
_KEYS = [
    ("environment.visibility_quantile", "environment.visibility_m"),
    ("environment.tire_friction_quantile", "environment.tire_friction"),
    ("a.b",),
    ("",),
]


@pytest.mark.parametrize(
    "overrides,keys",
    [
        (list(combo), keys)
        for n in range(3)
        for combo in itertools.combinations(_OVERRIDES, n)
        for keys in _KEYS
    ],
)
def test_sets_any_answers_as_the_original(
    overrides: list[str], keys: tuple[str, ...]
) -> None:
    assert control_fields.sets_any(overrides, keys) is _original_sets_any(
        overrides, *keys
    )
