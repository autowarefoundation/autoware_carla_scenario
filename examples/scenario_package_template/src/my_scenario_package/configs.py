"""Config dataclass(es) owned by this scenario package.

The parameters of a scenario live **with the scenario**, in the package that
defines it -- not inside ``autoware_carla_scenario``.  Shared parameters (ego,
map, server, ...) are imported from the framework's public API instead of being
redefined here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ReachGoalConfig:
    """Parameters for the ``reach_goal`` scenario.

    Every field is overridable from YAML (and the CLI) via Hydra, e.g.::

        uv run scenario scenario=reach_goal/default scenario.timeout_seconds=20
    """

    #: Must match the name passed to ``register_scenario`` and the
    #: ``scenario.name`` key in the YAML config.
    name: str = "reach_goal"

    #: Lanelet IDs whose OpenDRIVE roads the ego is expected to reach, in order.
    goal_lanelet_ids: list[int] = field(default_factory=lambda: [460, 265])

    #: Fail-safe timeout in seconds.
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        """Refuse a timeout that is not positive and finite, naming its key.

        A run builds the config before anything starts
        (``check_batch_configs``), so a bad ``scenario.timeout_seconds=``
        override is refused there, not by the scenario's ``TimeoutCondition``
        once CARLA is up.
        """
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError(
                "scenario.timeout_seconds must be finite and positive, "
                f"not {self.timeout_seconds}"
            )
