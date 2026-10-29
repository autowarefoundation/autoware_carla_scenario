# SPDX-License-Identifier: Apache-2.0
"""The signal the specification reads: one ``[T, D]`` tensor, one column each.

Why a single tensor rather than one signal per quantity: STLCG++'s temporal
operators reject tuple inputs (see :mod:`stl_driver.stl_backend`), so a
specification that mentions both speed and clearance has to read them out of
one array.  :class:`Predicate` slices are how a formula gets at a column.

Why every column is a *slack* -- a difference that should stay positive --
rather than a raw quantity: the thresholds a driving rule cares about change
every step.  The speed limit changes with the road, the stop line moves as we
approach it, the route ends sooner as we drive along it.  Folding those into
the column leaves every formula comparing against the constant ``0``, so the
formula tree is built once when the policy starts rather than rebuilt at 10 Hz.

Read ``GAP > 0`` as "there is clearance to spare", not as "the gap is
positive".
"""

from __future__ import annotations

from enum import IntEnum

import torch

from stl_driver.stl_backend import Predicate

__all__ = ["Col", "N_COLUMNS", "column", "saturate", "signal_from_columns"]


class Col(IntEnum):
    """Column index of each quantity in the signal tensor."""

    #: Speed along the plan, m/s. Never meaningfully negative -- the rollout
    #: floors it smoothly -- but `no_reverse` asserts that rather than assuming.
    SPEED = 0
    #: Longitudinal acceleration, m/s^2.
    ACCEL = 1
    #: Longitudinal jerk, m/s^3.
    JERK = 2
    #: Signed offset from the route centreline, m. Positive is to the left,
    #: matching the rig frame's y axis.
    LATERAL_OFFSET = 3
    #: Lateral acceleration implied by the planned path, m/s^2.
    LATERAL_ACCEL = 4
    #: Clearance to the nearest predicted actor, less the collision margin, m.
    #: Saturates at the configured cap when nothing is near.
    CLEARANCE = 5
    #: Lead-vehicle gap less the required time headway, m. Cap when no lead.
    HEADWAY_SLACK = 6
    #: Distance from the plan to the stop line, m. Cap when no light applies.
    STOP_SLACK = 7
    #: Gate: +gate_magnitude when a red or amber light governs us, -gate
    #: otherwise. See `Limits.gate_magnitude_m` for why the magnitude matters.
    LIGHT_GATE = 8
    #: Distance from the plan to the end of the published route, m.
    ROUTE_SLACK = 9
    #: Distance covered so far, less what `makes_progress` asks for, m.
    PROGRESS_SLACK = 10
    #: Posted limit less the planned speed, m/s.
    SPEED_HEADROOM = 11


N_COLUMNS = len(Col)


def saturate(values: torch.Tensor, cap: float) -> torch.Tensor:
    """Squash a distance-like quantity into ``(-cap, cap)``, smoothly.

    Used wherever a column means "no constraint applies here".  A literal
    infinity, or the ``1e9`` STLCG++ itself pads with, would win every
    ``max``/``min`` in the tree and swamp the weighted objective.  ``tanh``
    keeps the value bounded while leaving a live -- if small -- gradient near
    the cap, so a plan drifting back towards a constraint still feels it.

    Near zero, where every rule's threshold sits, this is within a fraction of
    a percent of the identity; the compression only becomes visible at
    distances no rule is sensitive to.  In float32 it does reach ``cap``
    exactly for large inputs, so ``cap`` is a saturation value rather than a
    strict bound -- do not rely on it as a sentinel distinguishable from real
    data.
    """
    return cap * torch.tanh(values / cap)


def column(name: str, index: Col) -> Predicate:
    """A predicate that reads one column out of the signal.

    Indexes with ``...`` rather than ``:`` so the same predicate works on a
    ``[T, D]`` signal and on anything ``vmap`` hands it.
    """
    position = int(index)

    def read(signal: torch.Tensor, position: int = position) -> torch.Tensor:
        return signal[..., position]

    return Predicate(name, read)


def signal_from_columns(**columns: torch.Tensor) -> torch.Tensor:
    """Assemble a signal from named columns, for tests and fallback plans.

    Missing columns are filled with values that satisfy every rule
    comfortably, so a test can state only the column it is exercising.  The
    filler is deliberately generous rather than marginal: a test that fails
    should fail because of the column it set.
    """
    lengths = {value.shape[0] for value in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"columns disagree on length: {sorted(lengths)}")
    (n_steps,) = lengths

    defaults = {
        Col.SPEED: 8.0,
        Col.ACCEL: 0.0,
        Col.JERK: 0.0,
        Col.LATERAL_OFFSET: 0.0,
        Col.LATERAL_ACCEL: 0.0,
        Col.CLEARANCE: 50.0,
        Col.HEADWAY_SLACK: 50.0,
        Col.STOP_SLACK: 50.0,
        Col.LIGHT_GATE: -20.0,
        Col.ROUTE_SLACK: 50.0,
        Col.PROGRESS_SLACK: 10.0,
        Col.SPEED_HEADROOM: 5.0,
    }

    signal = torch.empty((n_steps, N_COLUMNS), dtype=torch.float32)
    for col, value in defaults.items():
        signal[:, int(col)] = value
    for name, values in columns.items():
        signal[:, int(Col[name.upper()])] = values.to(torch.float32)
    return signal
