"""An ODD: the attributes it is described in, and the modules that bound it.

The model is openodd's (ASAM OpenODD 1.0, judged in its Rust core); this
module is where this framework has always exported it from. Probes read the
CARLA world (:mod:`.probes`, :mod:`.backend`).
"""

from __future__ import annotations

from openodd.model import (
    INACTIVE,
    UNDECIDED,
    OddAttribute,
    OddCondition,
    OddDefinition,
    OddModule,
    OddVerdict,
    all_of,
    any_of,
    module_holds,
    read_probe,
)

__all__ = [
    "INACTIVE",
    "OddAttribute",
    "OddCondition",
    "OddDefinition",
    "OddModule",
    "OddVerdict",
    "UNDECIDED",
    "all_of",
    "any_of",
    "module_holds",
    "read_probe",
]
