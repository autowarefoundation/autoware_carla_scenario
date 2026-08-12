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

import torch
from stlcgpp.formula import (
    Always as _UpstreamAlways,
)
from stlcgpp.formula import (
    And,
    Implies,
    Negation,
    Or,
    Predicate,
    STLFormula,
)
from stlcgpp.formula import (
    Eventually as _UpstreamEventually,
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

#: Below this a suffix's softmax weights have underflowed and its ratio is
#: 0/0; see :func:`_suffix_reduce`. Comfortably above float32's smallest
#: normal, so the guard fires before the arithmetic does.
_UNDERFLOW = 1e-30

#: Evaluation kwargs for reporting and for accept/reject decisions: exact
#: min/max semantics, so a robustness value means what the STL definition says.
EXACT: dict[str, Any] = {"approx_method": "true"}


def _suffix_reduce(
    values: torch.Tensor,
    *,
    minimum: bool,
    approx_method: str = "true",
    temperature: float = 1.0,
    **_: Any,
) -> torch.Tensor | None:
    """``result[t]`` is ``values[t:]`` reduced, in one pass instead of ``T``.

    Returns ``None`` for an evaluation mode this does not implement, so the
    caller can fall back to upstream rather than quietly answer something else.
    """
    if approx_method not in ("true", "softmax"):
        return None

    # A soft min is a soft max of the negated signal, so only one direction
    # needs writing.
    sign = -1.0 if minimum else 1.0
    x = sign * values
    exact = torch.flip(torch.cummax(torch.flip(x, (0,)), 0)[0], (0,))
    if approx_method == "true":
        return sign * exact

    # Weights are `exp(temperature * x)`, shifted by the global maximum for the
    # same reason `softmax` shifts internally.
    #
    # One shift for the whole signal, where upstream gets one per window. Where
    # a suffix sits far enough below the global maximum, every weight in it
    # underflows and the ratio is 0/0 -- which is not a rounding problem but a
    # NaN that propagates into the gradient and from there into the plan. It
    # happens readily: float32 underflows about 87 below the shift, so a
    # temperature of 8 needs only eleven metres of spread.
    #
    # Where that happens the soft maximum has already converged on the hard
    # one, so the exact scan is not a fallback but the limit being approached.
    weights = torch.exp(temperature * (x - x.max()))

    def suffix_sum(v: torch.Tensor) -> torch.Tensor:
        return torch.flip(torch.cumsum(torch.flip(v, (0,)), 0), (0,))

    total = suffix_sum(weights)
    usable = total > _UNDERFLOW
    soft = suffix_sum(weights * x) / total.clamp_min(_UNDERFLOW)
    return sign * torch.where(usable, soft, exact)


def _suffix_trace(
    values: torch.Tensor,
    interval: Sequence[Any],
    large_number: float,
    *,
    minimum: bool,
    **kwargs: Any,
) -> torch.Tensor | None:
    """A temporal operator's robustness trace, when its window is a suffix.

    Upstream evaluates ``Always`` and ``Eventually`` by materialising a
    ``(T + b + 1, T)`` matrix of every window at once and reducing it: correct,
    and quadratic. Every interval this specification uses ends at the horizon,
    which makes each window a suffix of the last -- so the whole trace is one
    cumulative scan, and the quadratic disappears.

    Anything else returns ``None`` and is left to upstream. That is the point
    of the arrangement: a rule may use whatever interval it likes, and only the
    shapes proven equivalent in :mod:`tests.test_stlcgpp_contract` take the
    fast path.
    """
    n_steps = values.shape[0]
    lower, upper = int(interval[0]), interval[1]
    unbounded = upper == torch.inf
    if not unbounded and int(upper) != n_steps - 1:
        return None

    scan = _suffix_reduce(values, minimum=minimum, **kwargs)
    if scan is None:
        return None

    starts = torch.arange(n_steps, device=values.device) + lower
    reachable = starts <= n_steps - 1
    if not unbounded and minimum:
        # A bounded `Always` pads past the horizon with `-large_number`, and a
        # minimum over a window containing it is that. Only the windows that
        # end inside the horizon survive.
        ends = torch.arange(n_steps, device=values.device) + int(upper)
        reachable = reachable & (ends <= n_steps - 1)

    picked = scan[starts.clamp(max=n_steps - 1)]
    return torch.where(reachable, picked, torch.full_like(picked, -large_number))


class Always(_UpstreamAlways):
    """``Always``, evaluated in linear time when its window is a suffix.

    Exactly upstream's semantics -- see :func:`_suffix_trace` for which shapes
    take the fast path and :mod:`tests.test_stlcgpp_contract` for the
    equivalence that licenses it.
    """

    def robustness_trace(
        self, signal: Any, padding: Any = None, large_number: float = 1e9, **kwargs: Any
    ) -> Any:
        if padding is None:
            values = self.subformula(
                signal, padding=padding, large_number=large_number, **kwargs
            )
            fast = _suffix_trace(
                values, self._interval, large_number, minimum=True, **kwargs
            )
            if fast is not None:
                return fast
        return super().robustness_trace(
            signal, padding=padding, large_number=large_number, **kwargs
        )


class Eventually(_UpstreamEventually):
    """``Eventually``, evaluated in linear time when its window is a suffix."""

    def robustness_trace(
        self, signal: Any, padding: Any = None, large_number: float = 1e9, **kwargs: Any
    ) -> Any:
        if padding is None:
            values = self.subformula(
                signal, padding=padding, large_number=large_number, **kwargs
            )
            fast = _suffix_trace(
                values, self._interval, large_number, minimum=False, **kwargs
            )
            if fast is not None:
                return fast
        return super().robustness_trace(
            signal, padding=padding, large_number=large_number, **kwargs
        )


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
