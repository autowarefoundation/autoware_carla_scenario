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

Every seed is a whole number from 0 to :data:`MAX_SEED` (``2**31 - 1``):
CARLA's pedestrian seed is a 32-bit unsigned integer and SUMO's ``--seed`` a
32-bit signed one, so a larger seed would be refused by the simulator on every
case, after CARLA had started.  :func:`check_seed` refuses it up front.

A component given ``seed=None`` (the default) takes the run's seed when it is
built (:func:`resolve_seed`); one given a number keeps it.  The runner sets the
run's seed before it builds the scenario and again before ``setup()``
(:func:`set_run_seed`), so an action built in either place follows it.

This module imports nothing but the standard library.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "DEFAULT_SEED",
    "MAX_SEED",
    "check_seed",
    "check_whole_number",
    "resolve_seed",
    "run_seed",
    "set_run_seed",
]

#: The seed of a run that names none.
DEFAULT_SEED: int = 0

#: The largest seed a run takes: ``2**31 - 1``, the largest SUMO's ``--seed``
#: (a 32-bit signed integer) reads, and within CARLA's 32-bit unsigned
#: ``World.set_pedestrians_seed``.
MAX_SEED: int = 2147483647

#: The run's seed, in a list so that setting it needs no ``global``.
_RUN_SEED: list[int] = [DEFAULT_SEED]


def run_seed() -> int:
    """The seed of the run being built or run (:data:`DEFAULT_SEED` before any)."""
    return _RUN_SEED[0]


def set_run_seed(seed: int) -> int:
    """Make *seed* the run's seed, and return the one it replaces.

    Raises:
        ValueError: If *seed* is negative or above :data:`MAX_SEED`
            (:func:`check_seed`).
    """
    number = check_seed("seed", seed)
    previous = _RUN_SEED[0]
    _RUN_SEED[0] = number
    return previous


def check_whole_number(key: str, value: int) -> int:
    """*value*, a config's integer field as it holds it, checked and returned
    as an ``int``: a whole number (``2``, ``2.0`` or ``2e6`` read as an int,
    or a numpy integer).

    A config dataclass is not coerced, so a field typed ``int`` may hold
    whatever its mapping did, and YAML reads ``1.2e6`` as a float.  ``int()``
    alone would take ``1.9`` and ``True`` as 1, so a boolean, a string, a
    fraction, NaN or infinity is refused.  A config normalises its field to
    what this returns, so a protobuf integer field or a CARLA attribute
    (``str(value)``) is given ``2``, never ``2.0``.

    Raises:
        ValueError: naming *key*.
    """
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError(f"{key} must be a whole number, got {value!r}")
    if isinstance(value, int):
        return int(value)
    # A float, or a number that is no int (numpy's integers): float() takes
    # anything with __float__ or __index__, and refuses the rest (None, a
    # list) with a TypeError.
    try:
        number = float(value)
    except TypeError:
        raise ValueError(f"{key} must be a whole number, got {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite, got {value!r}")
    if math.floor(number) != number:
        raise ValueError(f"{key} must be a whole number, got {value!r}")
    return int(value)


def check_seed(key: str, seed: int) -> int:
    """*seed*, a component's own seed as its config holds it, checked: a
    whole number (:func:`check_whole_number`: ``2``, or ``2.0`` read as 2),
    from 0 to :data:`MAX_SEED` (``2**31 - 1``), as the run's ``seed`` is.

    Raises:
        ValueError: naming *key*.
    """
    number = check_whole_number(key, seed)
    if number < 0:
        raise ValueError(f"{key} must not be negative, got {seed!r}")
    if number > MAX_SEED:
        raise ValueError(
            f"{key} must be at most {MAX_SEED} (2**31 - 1: seeds range over "
            f"0..{MAX_SEED}, what CARLA and SUMO take), got {seed!r}"
        )
    return number


def resolve_seed(seed: Optional[int]) -> int:
    """*seed*, or the run's seed when it is ``None``."""
    if seed is None:
        return run_seed()
    return int(seed)
