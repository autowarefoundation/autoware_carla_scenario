"""Resolve a scenario's sweep into concrete per-job override lists.

The viewer expands a scenario exactly as ``scenario-batch`` and the lanelet
sweeper do: through :func:`~autoware_carla_scenario.sweeper.expand.expand_config`,
so ``sweep.constraints`` (with ``sweep.bindings``), ``sweep.route``,
``sweep.odd_sample`` and ``sweep.case_index`` all behave as they do from the
CLI.  No jobs are launched and CARLA is not imported; only the Lanelet2 map
is loaded, in the viewer process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from autoware_carla_scenario.registry import load_scenario_plugins

if TYPE_CHECKING:
    from autoware_carla_scenario.odd.sampler import OddDraws

logger = logging.getLogger(__name__)

# Resolved locally to avoid importing examples.run (which pulls in carla).
_CONF_DIR = Path(__file__).resolve().parent.parent / "examples" / "conf"


class SweepResolutionError(ValueError):
    """A scenario's sweep could not be expanded; the message says why."""


def resolve_sweep(
    scenario_name: str,
    extra_overrides: list[str] | None = None,
    *,
    drawn: OddDraws | None = None,
) -> list[list[str]]:
    """The override list of every concrete scenario *scenario_name* expands to.

    Each inner list is ``scenario=<name>`` followed by that case's sweep
    overrides, e.g. ``["scenario=X", "ego.spawn_lanelet_id=5",
    "ego.spawn_s=18.6"]``.  The config is composed with *extra_overrides*
    (e.g. ``["map=nishishinjuku"]``), so the cases are found on the
    configuration the jobs will run with; the caller passes the same
    overrides to every job (they come after the case's, as in
    ``scenario-batch``).  A config without a sweep expands to itself: one
    job, ``["scenario=<name>"]``.

    Args:
        scenario_name: A concrete (non-glob) scenario config name.
        extra_overrides: The Run panel's overrides.
        drawn: In a run of several scenarios, the cases the earlier ones drew
            from the ODD (``sweep.odd_sample``), as ``scenario-batch`` shares
            them.

    Raises:
        SweepResolutionError: When the config cannot be composed or its sweep
            cannot be expanded (a bad constraint, a route search that does
            not parse, a missing map, invalid ``controls``, ...).
    """
    from autoware_carla_scenario.odd.controls import (  # noqa: PLC0415
        ControlsError,
        format_diagnostics,
    )
    from autoware_carla_scenario.sweeper.expand import (  # noqa: PLC0415
        expand_config,
    )

    # A scenario package's configs are on Hydra's search path only once the
    # package has registered its conf dir (hydra_plugins'
    # autoware_scenario_searchpath); loading is a no-op after the first call.
    load_scenario_plugins()
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(_CONF_DIR), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[f"scenario={scenario_name}", *(extra_overrides or [])],
            )
        cases = expand_config(cfg, drawn=drawn)
    except ControlsError as exc:
        raise SweepResolutionError(
            "; ".join(format_diagnostics(exc.diagnostics, scenario_name))
        ) from exc
    except Exception as exc:  # noqa: BLE001 -- Hydra, OmegaConf, Lanelet2, ...
        raise SweepResolutionError(
            f"{scenario_name}: {type(exc).__name__}: {exc}"
        ) from exc

    jobs = [[f"scenario={scenario_name}", *case] for case in cases]
    logger.info("Resolved %d job(s) for %s", len(jobs), scenario_name)
    return jobs
