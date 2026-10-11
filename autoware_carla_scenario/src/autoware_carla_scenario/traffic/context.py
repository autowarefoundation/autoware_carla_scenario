"""The run a traffic backend joins: :class:`TrafficContext`.

Split out of :mod:`~autoware_carla_scenario.traffic.base`, which re-exports it,
because it is the one part of the traffic seam that names a file
(:class:`pathlib.Path`): the backend interface itself is compiled by the
library check (``typecheck/library.py``), and Codon has no ``pathlib``.  Like
``base``, this module imports neither CARLA nor any traffic simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["TrafficContext"]


@dataclass(frozen=True)
class TrafficContext:
    """Everything a backend needs to know about the run it is joining.

    Built once per scenario by :class:`~autoware_carla_scenario.ScenarioRunner`
    and handed to
    :meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.prepare` before
    anything is spawned, so that a backend which has to start a process, derive
    a road network or refuse the run entirely does so while failing is still
    cheap.

    Attributes:
        client: The CARLA client.
        world: The CARLA world the scenario runs in.
        map_name: Name of the loaded map, as CARLA reports it.
        xodr_path: The OpenDRIVE the world is running, when the run installed
            one.  ``None`` when the map's roads came from CARLA's own assets and
            nothing has read them back yet.
        fixed_delta_seconds: The simulation step the runner applies.  A backend
            that steps a second simulator matches its step length to this.
        random_seed: The scenario's seed.  A backend with any randomness of its
            own derives it from this, so a repeated run repeats.
        output_dir: Where this run's artefacts go; a backend writes its own logs
            under it rather than into the working directory.
    """

    client: Optional["carla.Client"] = None
    world: Optional["carla.World"] = None
    map_name: str = ""
    xodr_path: Optional[Path] = None
    fixed_delta_seconds: float = 0.05
    random_seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path("scenario_outputs"))
