"""When the runner checks a scenario: the ``typecheck`` config key.

``typecheck: auto`` (the default) checks every scenario registered with
:func:`~autoware_carla_scenario.register_scenario` when a Codon compiler is
installed, and only warns when none is; ``required`` refuses to run without
one; ``off`` does not check.  ``$AUTOWARE_CARLA_SCENARIO_TYPECHECK`` overrides
the config.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Mapping, cast

from .check import ScenarioTypeError, TypeCheckResult, typecheck_scenario

__all__ = [
    "TYPECHECK_ENV",
    "TYPECHECK_MODES",
    "TypecheckMode",
    "check_registered_scenario",
    "typecheck_mode",
]

logger = logging.getLogger(__name__)

TypecheckMode = Literal["auto", "required", "off"]
TYPECHECK_MODES: tuple[TypecheckMode, ...] = ("auto", "required", "off")
TYPECHECK_ENV = "AUTOWARE_CARLA_SCENARIO_TYPECHECK"


def typecheck_mode(cfg: Mapping[str, Any] | None = None) -> TypecheckMode:
    """The mode the environment or *cfg*'s ``typecheck`` key selects."""
    value: Any = os.environ.get(TYPECHECK_ENV)
    if value is None and cfg is not None:
        value = cfg.get("typecheck")
    if value is None:
        return "auto"
    if value is False:
        return "off"
    if value is True:
        return "required"
    mode = str(value).strip().lower()
    if mode not in TYPECHECK_MODES:
        raise ValueError(
            f"typecheck must be one of {', '.join(TYPECHECK_MODES)} (got {value!r})"
        )
    return cast(TypecheckMode, mode)


def check_registered_scenario(
    name: str,
    scenario_dict: Mapping[str, Any],
    mode: TypecheckMode = "auto",
) -> TypeCheckResult | None:
    """Check the scenario registered as *name*, built with *scenario_dict*.

    Returns the result, or ``None`` when nothing was checked: *mode* is
    ``off``, or *name* was registered with a custom builder
    (:func:`~autoware_carla_scenario.register_scenario_builder`), which the
    checker cannot drive.

    Raises:
        ScenarioTypeError: The scenario does not compile, or *mode* is
            ``required`` and there is no Codon compiler.
    """
    if mode == "off":
        return None
    from ..registry import get_scenario_classes  # noqa: PLC0415 - imports CARLA

    classes = get_scenario_classes(name)
    if classes is None:
        logger.info("Scenario %r has a custom builder: not statically checked", name)
        return None
    result = typecheck_scenario(classes[0], classes[1], dict(scenario_dict))
    if result.skipped is not None:
        if mode == "required":
            result.ok = False
            raise ScenarioTypeError(result)
        logger.warning("%s", result.format())
    elif not result.ok:
        raise ScenarioTypeError(result)
    else:
        logger.info("%s", result.format())
    return result
