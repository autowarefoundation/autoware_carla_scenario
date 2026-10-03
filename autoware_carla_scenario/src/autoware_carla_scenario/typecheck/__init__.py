"""Static check of a scenario with Codon, before it runs (docs/typecheck.md).

A scenario is Python, and the runner executes it as Python; the check compiles
the same source with `Codon <https://github.com/exaloop/codon>`_, a statically
typed compiler for Python syntax, against a typed model of this framework
(``codon/``), and refuses a scenario that does not compile::

    from autoware_carla_scenario.typecheck import typecheck_scenario

    result = typecheck_scenario(MyScenario, MyScenarioConfig, {"timeout_seconds": 10.0})
    assert result.ok, result.format()

The runner does this for every scenario registered with
:func:`~autoware_carla_scenario.register_scenario` (the ``typecheck`` config
key), and ``scenario-check`` does it without running anything.  The model
follows the API of `typesafe_carla <https://github.com/hakuturu583/typesafe_carla>`_,
the statically typed CARLA client the framework is moving to.
"""

from .check import (
    Diagnostic,
    ScenarioTypeError,
    TypeCheckResult,
    available_toolchain,
    model_dir,
    typecheck_scenario,
)
from .toolchain import ENV_CODON, Toolchain, ToolchainError, find_codon
from .mode import TYPECHECK_MODES, TypecheckMode, check_registered_scenario

__all__ = [
    "Diagnostic",
    "ENV_CODON",
    "ScenarioTypeError",
    "TYPECHECK_MODES",
    "TypeCheckResult",
    "Toolchain",
    "ToolchainError",
    "TypecheckMode",
    "available_toolchain",
    "check_registered_scenario",
    "find_codon",
    "model_dir",
    "typecheck_scenario",
]
