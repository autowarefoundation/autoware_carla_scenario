"""The run's seed: the one number every random draw of a run follows.

A run is given one seed (``seed`` in the run's config, or
:attr:`~autoware_carla_scenario.BaseScenario.random_seed`), and everything that
draws at random takes it unless it is given a seed of its own:

* the cases a sweep draws from the ODD (``sweep.odd_sample.seed``), and the
  tire friction and visibility it draws for each;
* the traffic backend -- CARLA's TrafficManager random device and each
  vehicle's parameter draws, SUMO's ``--seed`` and its own draws -- through
  :attr:`~autoware_carla_scenario.traffic.context.TrafficContext.random_seed`;
* CARLA's pedestrian navigation (``World.set_pedestrians_seed``);
* the scenario's actions and conditions that draw: the adversarial actions,
  :class:`~autoware_carla_scenario.actions.adversarial_traffic.AdversarialTrafficAction`,
  :class:`~autoware_carla_scenario.actions.background_traffic.TrafficSourceAction`
  (and the config's ``background_traffic``);
* the driver policy's ``start_session`` seed and its LiDARs' weather noise.

A component given ``seed=None`` (the default) takes the run's seed when it is
built (:func:`resolve_seed`); one given a number keeps it.  The runner sets the
run's seed before it builds the scenario and again before ``setup()``
(:func:`set_run_seed`), so an action built in either place follows it.

This module imports nothing but the standard library.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "DEFAULT_SEED",
    "resolve_seed",
    "run_seed",
    "set_run_seed",
]

#: The seed of a run that names none.
DEFAULT_SEED: int = 0

#: The run's seed, in a list so that setting it needs no ``global``.
_RUN_SEED: list[int] = [DEFAULT_SEED]


def run_seed() -> int:
    """The seed of the run being built or run (:data:`DEFAULT_SEED` before any)."""
    return _RUN_SEED[0]


def set_run_seed(seed: int) -> int:
    """Make *seed* the run's seed, and return the one it replaces.

    Raises:
        ValueError: If *seed* is negative (CARLA's and SUMO's seeds are not).
    """
    if seed < 0:
        raise ValueError(f"seed must not be negative, got {seed}")
    previous = _RUN_SEED[0]
    _RUN_SEED[0] = int(seed)
    return previous


def resolve_seed(seed: Optional[int]) -> int:
    """*seed*, or the run's seed when it is ``None``."""
    if seed is None:
        return run_seed()
    return int(seed)
