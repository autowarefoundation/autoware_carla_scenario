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
    """
    phi = Always(And(column(0) > 0.0, column(1) > 0.0))
    with pytest.raises((AttributeError, TypeError)):
        phi.robustness((signal_from([1.0] * T), signal_from([1.0] * T)), **EXACT)
