# SPDX-License-Identifier: Apache-2.0
"""The only module in this package that imports ``stlcgpp``.

Two reasons for the seam.  First, packaging: STLCG++ ships a ``setup.py`` with
no ``pyproject.toml``, has no releases or tags, and has no
``stlcgpp/__init__.py`` -- it installs as a PEP 420 namespace package from a
pinned commit.  If that ever stops working, vendoring ``formula.py`` and
``utils.py`` is a change to this file alone.

Second, and more usefully day to day: the library has a handful of sharp edges
that every call site would otherwise have to remember.  They are collected here
once.

Sharp edges, all verified against the pinned commit
---------------------------------------------------

``temperature`` is an *inverse* temperature.
    Larger means closer to the true min/max, not smoother.  It reads backwards
    if you come from a softmax-with-temperature background.

Use ``softmax``, not ``logsumexp``, for gradients.
    Both are smooth, but ``logsumexp``'s min carries a ``-log(N)/temperature``
    bias.  At ``N = 40`` and ``temperature = 1`` that is ``-3.7`` -- which would
    swamp a robustness measured in metres.  ``softmax`` returns a convex
    combination of the signal's own values, so its error is bounded by the
    signal's spread.  :func:`smooth` is the only approved way to ask for a
    differentiable evaluation.

``Always`` / ``Eventually`` / ``Until`` crash on tuple inputs.
    They read ``signal.device`` *before* checking whether the input is a tuple,
    so the multi-``Expression`` idiom from upstream's demo notebook raises
    ``AttributeError`` the moment a temporal operator wraps it.  Consequence:
    a multi-signal specification must be one ``[T, D]`` tensor read by
    :class:`Predicate` slices.  That is what :mod:`stl_driver.signals` builds.

``Expression`` is unhashable.
    It defines ``__eq__`` (returning an ``Equal`` *formula*) without
    ``__hash__``, and every formula is an ``nn.Module``, whose traversals
    (``.to()``, ``.parameters()``, ``named_modules()``) memoise through a set.
    So a tree containing an ``Expression`` raises ``TypeError`` on traversal.
    ``Predicate`` does define ``__hash__``.  We use ``Predicate`` exclusively --
    ``Expression`` is deliberately not re-exported.

Intervals are timestep indices, not seconds.
    ``interval=[a, b]`` means the window ``[t+a, t+b]`` in samples.  With
    ``dt = 0.1 s``, "from one second onward" is ``[10, ...]``.

A bounded interval must fit inside the horizon.
    With ``padding=None`` (the default) a bounded ``Always`` pads beyond the
    horizon with ``-large_number``, so ``interval[1] >= T`` poisons the trace
    tail with ``-1e9``.  We read ``trace[0]``, so the requirement is
    ``interval[1] <= T - 1``; :func:`assert_interval_fits` enforces it at
    construction time rather than letting it surface as a mysterious ``-1e9``.

Prefer the masking operators over the recurrent ones.
    ``AlwaysRecurrent`` / ``EventuallyRecurrent`` / ``UntilRecurrent`` define
    ``robustness`` as ``trace[-1]`` and expect the signal reversed in time, and
    they mutate interval state in place while caching a hidden dimension from
    the first signal they see -- which makes an instance unsafe to reuse across
    signals of different lengths.  The masking operators re-exported here have
    none of those properties.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from stlcgpp.formula import (
    Always as Always,  # re-exported; see assert_interval_fits
)
from stlcgpp.formula import (
    And,
    Eventually,
    Implies,
    Negation,
    Or,
    Predicate,
    STLFormula,
)

__all__ = [
    "And",
    "Eventually",
    "Implies",
    "Negation",
    "Or",
    "Predicate",
    "STLFormula",
    "Always",
    "EXACT",
    "assert_interval_fits",
    "conjoin",
    "smooth",
]

#: Evaluation kwargs for reporting and for accept/reject decisions: exact
#: min/max semantics, so a robustness value means what the STL definition says.
EXACT: dict[str, Any] = {"approx_method": "true"}


def smooth(temperature: float) -> dict[str, Any]:
    """Evaluation kwargs for a differentiable robustness.

    ``temperature`` is an inverse temperature: larger is sharper, converging on
    the exact min/max.  See the module docstring for why this is ``softmax``
    rather than ``logsumexp``.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return {"approx_method": "softmax", "temperature": float(temperature)}


def conjoin(formulas: Iterable[STLFormula]) -> STLFormula:
    """Fold formulas into a single conjunction.

    ``And`` flattens n-ary chains through ``separate_and`` before taking the
    min, so the fold's associativity does not change the result -- including
    under smoothing, where it is an n-way soft-min rather than nested binary
    ones.
    """
    items = list(formulas)
    if not items:
        raise ValueError("cannot conjoin an empty set of formulas")
    result = items[0]
    for formula in items[1:]:
        result = And(result, formula)
    return result


def assert_interval_fits(interval: Sequence[int] | None, horizon: int, *, what: str) -> None:
    """Raise unless a bounded interval lies inside a ``horizon``-step signal.

    Out-of-range upper bounds do not raise upstream; they quietly pad the
    robustness trace with ``-1e9``.  Since a rule's robustness is ``trace[0]``,
    the damage is invisible right up until an interval creeps past the horizon
    and the whole specification reads as catastrophically violated.
    """
    if interval is None:
        return
    lower, upper = int(interval[0]), interval[1]
    if lower < 0:
        raise ValueError(f"{what}: interval lower bound {lower} is negative")
    if upper == float("inf"):
        return
    if int(upper) > horizon - 1:
        raise ValueError(
            f"{what}: interval upper bound {int(upper)} does not fit in a "
            f"{horizon}-step horizon (max {horizon - 1}); the robustness trace "
            "would be padded with -1e9"
        )
    if lower > int(upper):
        raise ValueError(f"{what}: interval {list(interval)} is empty")
