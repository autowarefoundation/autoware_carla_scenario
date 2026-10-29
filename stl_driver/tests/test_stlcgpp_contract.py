# SPDX-License-Identifier: Apache-2.0
"""Pin the STLCG++ behaviour this package is built on.

STLCG++ is installed from an unreleased, untagged commit of a repository that
ships no ``pyproject.toml`` and no ``stlcgpp/__init__.py``.  Every assumption
:mod:`stl_driver.rules` makes about it is asserted here, so that a bump to the
pin fails in one obvious place rather than as a policy that mysteriously drives
into things.

The sharp edges these tests cover are documented in
:mod:`stl_driver.stl_backend`.
"""

from __future__ import annotations

import math

import pytest
import torch

from stl_driver.signals import N_COLUMNS
from stl_driver.stl_backend import (
    EXACT,
    Always,
    And,
    Eventually,
    Implies,
    Predicate,
    assert_interval_fits,
    conjoin,
    smooth,
)

T = 12


def column(index: int, name: str = "x") -> Predicate:
    return Predicate(name, lambda signal, i=index: signal[..., i])


def signal_from(*columns: list[float]) -> torch.Tensor:
    """Build a ``[T, D]`` signal from per-column value lists."""
    return torch.tensor(columns, dtype=torch.float32).T.contiguous()


def test_the_namespace_package_imports():
    """The install itself -- no __init__.py upstream, so this is not a given."""
    import stlcgpp.formula
    import stlcgpp.utils

    assert hasattr(stlcgpp.formula, "Always")
    assert hasattr(stlcgpp.utils, "maxish")


def test_always_is_the_minimum_over_the_signal():
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
    phi = Always(column(0) > 0.0)
    assert phi.robustness(signal_from(values), **EXACT).item() == pytest.approx(min(values))


def test_eventually_is_the_maximum_over_the_signal():
    values = [-3.0, -1.0, -4.0, -1.0, -5.0, -9.0, -2.0, -6.0, -5.0, -3.0, -5.0, -8.0]
    phi = Eventually(column(0) > 0.0)
    assert phi.robustness(signal_from(values), **EXACT).item() == pytest.approx(max(values))


def test_robustness_is_the_first_element_of_the_trace():
    """Masking operators report ``trace[0]``; the recurrent ones report [-1].

    We use only the masking operators, and :mod:`stl_driver.optimizer` scores
    plans by ``robustness()``.  If this ever flipped, every rule would be
    evaluated over the wrong suffix.
    """
    values = [float(i) for i in range(T)]
    phi = Always(column(0) > 0.0)
    trace = phi(signal_from(values), **EXACT)
    assert phi.robustness(signal_from(values), **EXACT).item() == pytest.approx(trace[0].item())


def test_intervals_are_timestep_indices_not_seconds():
    """The single most dangerous unit confusion in the whole specification."""
    values = [5.0] * T
    values[3] = -1.0  # outside [6, 11]
    inside = list(values)
    inside[3] = 5.0
    inside[8] = -1.0  # inside [6, 11]

    phi = Always(column(0) > 0.0, interval=[6, 11])
    assert phi.robustness(signal_from(values), **EXACT).item() == pytest.approx(5.0)
    assert phi.robustness(signal_from(inside), **EXACT).item() == pytest.approx(-1.0)


def test_implication_is_max_of_negated_antecedent_and_consequent():
    """The identity that sets the traffic-light gate's magnitude.

    ``Implies(a, b)`` is ``max(-rho_a, rho_b)``.  One number -- the antecedent's
    robustness magnitude ``G`` -- therefore controls two things at once:

    * when the antecedent is *inactive* (``rho_a = -G``), the rule contributes
      exactly ``+G``, so ``G`` must stay small enough not to dominate the
      objective; and
    * when it is *active* (``rho_a = +G``), the rule cannot read below ``-G``,
      so ``G`` must stay large enough not to clip a real violation.

    :mod:`stl_driver.config` picks ``gate_magnitude_m`` to satisfy both, and
    the objective's per-rule ``margin`` handles the first concern directly.
    """
    gate = column(0)
    consequent = column(1)
    phi = Always(Implies(gate > 0.0, consequent > 0.0))

    # Inactive antecedent: the value is the gate magnitude, whatever the
    # consequent does. This is bounded vacuous truth.
    inactive = signal_from([-20.0] * T, [-50.0] * T)
    assert phi.robustness(inactive, **EXACT).item() == pytest.approx(20.0)

    # Active antecedent, violation well inside the gate: reported faithfully.
    values = [4.0] * T
    values[5] = -2.0
    active = signal_from([20.0] * T, values)
    assert phi.robustness(active, **EXACT).item() == pytest.approx(-2.0)


def test_an_active_implication_clips_violations_deeper_than_the_gate():
    """The failure mode a too-small gate causes, and why the gate is in metres.

    Once the consequent falls below ``-G`` the max saturates on the antecedent
    branch, and since the gate column is plan-independent the gradient flows
    nowhere useful: the optimizer is told a rule is violated but not which way
    to move.  Sizing ``G`` in metres puts every realistic violation on the
    live side of that boundary.  A +-1 gate would put nearly all of them past
    it -- running even 1.5 m beyond a stop line would already be clipped.
    """
    gate = column(0)
    consequent = column(1)
    phi = Always(Implies(gate > 0.0, consequent > 0.0))

    def consequent_pull(signal: torch.Tensor) -> float:
        gradient = torch.func.grad(lambda s: phi.robustness(s, **smooth(2.0)))(signal)
        return gradient[:, 1].abs().sum().item()

    # A 5 m overshoot: inside a 20 m gate, past a +-1 one.
    overshoot = [-5.0] * T
    assert phi.robustness(signal_from([20.0] * T, overshoot), **EXACT).item() == pytest.approx(-5.0)
    assert phi.robustness(signal_from([1.0] * T, overshoot), **EXACT).item() == pytest.approx(-1.0)

    live = consequent_pull(signal_from([20.0] * T, overshoot))
    clipped = consequent_pull(signal_from([1.0] * T, overshoot))
    assert live > 0.5
    assert live / clipped > 100.0  # two orders of magnitude of attenuation

    # Beyond the gate the value saturates at -G for any gate size; this is the
    # regime the `hard` fallback exists to cover, not one to optimize out of.
    assert phi.robustness(signal_from([20.0] * T, [-50.0] * T), **EXACT).item() == pytest.approx(
        -20.0
    )


def test_an_interval_past_the_horizon_poisons_the_trace():
    """Documents exactly why ``assert_interval_fits`` exists.

    This is upstream behaviour we guard against, not behaviour we want.
    """
    phi = Always(column(0) > 0.0, interval=[0, T + 5])
    poisoned = phi.robustness(signal_from([5.0] * T), **EXACT).item()
    assert poisoned < -1e8

    with pytest.raises(ValueError, match="does not fit"):
        assert_interval_fits([0, T + 5], T, what="test rule")
    assert_interval_fits([0, T - 1], T, what="test rule")
    assert_interval_fits(None, T, what="test rule")
    assert_interval_fits([2, math.inf], T, what="test rule")


def test_conjunction_is_the_minimum_over_rules():
    """The property ``RuleSet.spec_robustness`` relies on for its reporting."""
    signal = signal_from([2.0] * T, [-1.0] * T, [7.0] * T)
    parts = [Always(column(i) > 0.0) for i in range(3)]
    spec = conjoin(parts)
    individually = [p.robustness(signal, **EXACT).item() for p in parts]
    assert spec.robustness(signal, **EXACT).item() == pytest.approx(min(individually))


def test_softmax_robustness_stays_within_the_signal_range():
    """Why gradients use softmax rather than logsumexp.

    A convex combination of the signal's own values cannot leave their range.
    ``logsumexp`` can, by ``-log(N)/temperature``, and that bias would be
    metres-large at the temperatures we use.
    """
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
    signal = signal_from(values)
    phi = Always(column(0) > 0.0)
    approx = phi.robustness(signal, **smooth(2.0)).item()
    assert min(values) <= approx <= max(values)

    # Sharper temperature converges towards the true minimum.
    sharper = phi.robustness(signal, **smooth(50.0)).item()
    assert abs(sharper - min(values)) < abs(approx - min(values))


def test_logsumexp_bias_is_real():
    """Records the measurement behind the softmax choice, so it is not folklore."""
    from stlcgpp.formula import Always as RawAlways

    signal = signal_from([0.0] * T)
    biased = RawAlways(column(0) > 0.0).robustness(
        signal, approx_method="logsumexp", temperature=1.0
    )
    assert biased.item() == pytest.approx(-math.log(T), abs=1e-4)


def test_gradients_flow_through_robustness():
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
    signal = signal_from(values)
    phi = Always(column(0) > 0.0)

    gradient = torch.func.grad(lambda s: phi.robustness(s, **smooth(2.0)))(signal)
    assert gradient.shape == signal.shape
    # softmax spreads gradient across every timestep; `true` would be one-hot.
    assert int((gradient != 0).sum()) == T


def test_vmap_batches_the_robustness_evaluation():
    """The fast path in ``optimizer``: upstream hardcodes ``reshape([T, 1])``.

    A manual leading batch dimension breaks on that reshape, but ``vmap`` is
    invisible to it.  ``optimizer`` falls back to a Python loop if this ever
    stops holding, and this test is what says which path is live.
    """
    phi = Always(column(0) > 0.0)
    batch = torch.stack(
        [
            signal_from([3.0] * T),
            signal_from([-2.0] * T),
            signal_from([float(i) for i in range(T)]),
        ]
    )
    batched = torch.vmap(lambda s: phi.robustness(s, **smooth(2.0)))(batch)
    looped = torch.stack([phi.robustness(s, **smooth(2.0)) for s in batch])
    assert torch.allclose(batched, looped, atol=1e-5)


def test_and_of_comparisons_on_one_tensor_is_the_pointwise_minimum():
    """The idiom every multi-bound rule (``comfort``, ``lane_keeping``) uses."""
    upper = column(0)
    signal = signal_from([0.5, -0.5, 0.9, -0.9, 0.1, 0.0, 0.3, -0.2, 0.7, -0.7, 0.4, -0.4])
    phi = Always((upper < 1.0) & (upper > -1.0))
    # Tightest bound over time: 1 - max|x| = 1 - 0.9.
    assert phi.robustness(signal, **EXACT).item() == pytest.approx(0.1, abs=1e-6)


def test_temporal_operators_reject_tuple_inputs():
    """Why signals are one [T, D] tensor read by Predicates.

    Upstream's demo notebook shows multi-signal formulas built from several
    ``Expression`` objects and fed a tuple.  That idiom does not survive being
    wrapped in a temporal operator, which is the whole point of a driving spec.

    Which exception comes out depends on which layer notices first, and that
    moved when ``Always`` gained its linear-time path: upstream reads
    ``signal.device`` before checking for a tuple and raises ``AttributeError``,
    while ours reaches the subformula and the column index goes out of bounds.
    Either way the idiom is unusable, which is the point being pinned.
    """
    phi = Always(And(column(0) > 0.0, column(1) > 0.0))
    with pytest.raises((AttributeError, TypeError, IndexError)):
        phi.robustness((signal_from([1.0] * T), signal_from([1.0] * T)), **EXACT)


@pytest.mark.parametrize("approx_method", ["true", "softmax"])
@pytest.mark.parametrize("lower", [0, 1, 5, 12])
@pytest.mark.parametrize("bounded", [False, True])
def test_linear_time_temporal_operators_match_upstream(approx_method, lower, bounded):
    """Our `Always`/`Eventually` must be upstream's, only faster.

    Upstream evaluates them by materialising every window as a row of a
    `(T + b + 1, T)` matrix. We replace that with a cumulative scan whenever the
    window is a suffix, which every interval in `rules.py` is. This is the test
    that licenses the substitution: it compares whole traces, not just the
    `trace[0]` a rule reads, so a divergence anywhere shows up here rather than
    surviving until some future rule nests a temporal operator.
    """
    from stlcgpp.formula import Always as UpstreamAlways
    from stlcgpp.formula import Eventually as UpstreamEventually
    from stlcgpp.formula import GreaterThan

    from stl_driver.stl_backend import Always, Eventually

    n_steps = 20
    generator = torch.Generator().manual_seed(lower)
    signal = torch.randn(n_steps, 1, generator=generator) * 3.0
    predicate = GreaterThan(Predicate("x", lambda s: s[..., 0]), 0.0)
    interval = [lower, n_steps - 1] if bounded else [lower, torch.inf]
    evaluation = (
        {"approx_method": "true"}
        if approx_method == "true"
        else {"approx_method": "softmax", "temperature": 2.0}
    )

    for ours, theirs in ((Always, UpstreamAlways), (Eventually, UpstreamEventually)):
        fast = ours(predicate, interval=interval).robustness_trace(signal, **evaluation)
        slow = theirs(predicate, interval=interval).robustness_trace(signal, **evaluation)
        assert torch.allclose(fast, slow, atol=1e-4), f"{ours.__name__} interval={interval}"


def test_the_fast_path_is_actually_being_taken():
    """A guard against the fallback silently swallowing every call.

    The optimisation is invisible in results by construction -- the test above
    asserts exactly that -- so nothing else would notice if the fast path
    stopped being reached.
    """
    from stl_driver.stl_backend import _suffix_trace

    values = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    assert _suffix_trace(values, [0, torch.inf], 1e9, minimum=True) is not None
    assert _suffix_trace(values, [2, 4], 1e9, minimum=True) is not None
    # A window that ends before the horizon is upstream's problem, not ours.
    assert _suffix_trace(values, [0, 2], 1e9, minimum=True) is None
    assert (
        _suffix_trace(values, [0, torch.inf], 1e9, minimum=True, approx_method="logsumexp") is None
    )


def test_a_wide_signal_does_not_produce_nan_under_smoothing():
    """The linear-time scan shifts once for the whole signal, not per window.

    Where a suffix sits far below the global maximum every weight in it
    underflows, and the ratio is 0/0. That is not a rounding error: the NaN
    reaches the gradient, the gradient reaches the plan parameters, and the
    station lookup then indexes a route table with `INT64_MIN`. It showed up
    exactly that way -- three optimizer tests failing on an out-of-bounds
    index, nowhere near the arithmetic that caused it.
    """
    from stlcgpp.formula import GreaterThan

    from stl_driver.stl_backend import Always, smooth

    n_steps = 20
    wide = torch.full((n_steps, 1), -60.0)
    wide[0, 0] = 60.0
    predicate = GreaterThan(Predicate("x", lambda s: s[..., 0]), 0.0)

    trace = Always(predicate).robustness_trace(wide, **smooth(8.0))
    assert torch.isfinite(trace).all()

    parameter = torch.zeros(1, requires_grad=True)
    value = Always(predicate).robustness(wide + parameter, **smooth(8.0))
    value.backward()
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("approx_method", ["true", "softmax"])
@pytest.mark.parametrize("lower", [0, 1, 5, 12])
@pytest.mark.parametrize("bounded", [False, True])
def test_reducing_straight_to_the_value_matches_the_trace(approx_method, lower, bounded):
    """`robustness` skips building a trace it would throw away.

    A rule's robustness is `trace[0]`, and no rule here nests one temporal
    operator inside another, so the rest of the trace is constructed and
    discarded -- which over ten rules, forward and backward, was most of the
    operations in a refinement step. This asserts the shortcut agrees with the
    trace it replaces, including where the interval starts partway in.
    """
    from stlcgpp.formula import Always as UpstreamAlways
    from stlcgpp.formula import Eventually as UpstreamEventually
    from stlcgpp.formula import GreaterThan

    n_steps = 20
    generator = torch.Generator().manual_seed(lower + (0 if bounded else 100))
    signal = torch.randn(n_steps, 1, generator=generator) * 3.0
    predicate = GreaterThan(Predicate("x", lambda s: s[..., 0]), 0.0)
    interval = [lower, n_steps - 1] if bounded else [lower, torch.inf]
    evaluation = (
        {"approx_method": "true"}
        if approx_method == "true"
        else {"approx_method": "softmax", "temperature": 2.0}
    )

    for ours, theirs in ((Always, UpstreamAlways), (Eventually, UpstreamEventually)):
        fast = ours(predicate, interval=interval).robustness(signal, **evaluation)
        slow = theirs(predicate, interval=interval).robustness(signal, **evaluation)
        assert torch.allclose(fast, slow, atol=1e-4), f"{ours.__name__} interval={interval}"


def test_the_shortcut_cuts_the_operation_count():
    """The point of it, stated as a number so it cannot quietly regress."""
    import collections

    from stlcgpp.formula import Always as UpstreamAlways
    from stlcgpp.formula import GreaterThan
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        def __init__(self):
            self.n = 0
            self.by = collections.Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.n += 1
            self.by[str(func)] += 1
            return func(*args, **(kwargs or {}))

    signal = torch.randn(40, 1)
    predicate = GreaterThan(Predicate("x", lambda s: s[..., 0]), 0.0)

    with Count() as ours:
        Always(predicate).robustness(signal, **EXACT)
    with Count() as theirs:
        UpstreamAlways(predicate).robustness(signal, **EXACT)

    assert ours.n * 4 < theirs.n, f"{ours.n} operations against upstream's {theirs.n}"
    assert not any("flip" in name for name in ours.by), "no scan should be built at all"


@pytest.mark.parametrize("approx_method", ["true", "softmax"])
def test_scalar_evaluation_matches_the_formula_it_replaces(approx_method):
    """Collapsing a tree to scalars must not change what it says.

    Every connective here is pointwise in time, so `trace[0]` depends only on
    the children's `trace[0]` -- but only if the n-ary flattening `And` and
    `Or` do before reducing is preserved. Nested pairs of two and a single
    reduction over all operands are the same under an exact min and *not* the
    same under a soft one, so this checks both.
    """
    from stl_driver.stl_backend import Negation, scalar_robustness

    evaluation = (
        {"approx_method": "true"}
        if approx_method == "true"
        else {"approx_method": "softmax", "temperature": 3.0}
    )
    generator = torch.Generator().manual_seed(11)
    a, b, c = (column(i) for i in range(3))
    shapes = {
        "always over a conjunction": Always(And(a > 0.0, And(b > 0.0, c > 0.0))),
        "implication of two temporals": Implies(
            Always(a > 0.0), Eventually(b > 0.0, interval=[0, T - 1])
        ),
        "negation inside": Always(Negation(a > 0.0)),
        "an ampersand chain": Always((a > 0.0) & (b > 0.0) & (c > 0.0)),
        "temporal inside a conjunction": And(
            Always(a > 0.0, interval=[3, T - 1]), Eventually(b > 0.0)
        ),
        "a bare predicate": a > 0.0,
    }
    for label, formula in shapes.items():
        for _ in range(4):
            signal = torch.randn(T, 3, generator=generator) * 2.0
            fast = scalar_robustness(formula, signal, **evaluation)
            slow = formula.robustness(signal, **evaluation)
            assert torch.allclose(fast, slow, atol=1e-4), f"{label} under {approx_method}"


def test_scalar_evaluation_cuts_the_operation_count_of_the_worst_rule():
    """`makes_progress` was three times any other rule, and this is why."""
    import collections

    from torch.utils._python_dispatch import TorchDispatchMode

    from stl_driver.config import STLDriverConfig
    from stl_driver.rules import build_rules
    from stl_driver.stl_backend import scalar_robustness

    class Count(TorchDispatchMode):
        def __init__(self):
            self.n = 0
            self.by = collections.Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.n += 1
            self.by[str(func)] += 1
            return func(*args, **(kwargs or {}))

    config = STLDriverConfig()
    rule = next(r for r in build_rules(config) if r.name == "makes_progress")
    signal = torch.randn(config.horizon.n_steps, N_COLUMNS)

    with Count() as fast:
        scalar_robustness(rule.formula, signal, **EXACT)
    with Count() as slow:
        rule.formula.robustness(signal, **EXACT)

    assert fast.n < slow.n * 0.7, f"{fast.n} operations against {slow.n}"
