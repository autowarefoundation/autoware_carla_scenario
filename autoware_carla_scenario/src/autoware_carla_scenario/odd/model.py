"""An ODD: the attributes it is described in, and the modules that bound it.

The model is openodd's (ASAM OpenODD 1.0, judged in its Rust core); this
module is where this framework has always exported it from. Probes read the
CARLA world (:mod:`.probes`, :mod:`.backend`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

#: The class of :data:`UNDECIDED`, which openodd does not export: what an
#: annotation names for "a value only the run can tell" (a probe's
#: ``on_lanelet``).  openodd types the sentinel as ``Any``, and so does mypy;
#: the Codon check reads the class the odd model declares.
if TYPE_CHECKING:
    Undecided = Any
else:
    Undecided = type(UNDECIDED)
