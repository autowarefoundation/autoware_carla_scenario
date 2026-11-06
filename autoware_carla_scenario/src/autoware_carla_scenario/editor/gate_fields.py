"""The gate's thresholds as the editor's inspector edits them.

Kept apart from :mod:`.service` (which imports yaml, zipfile and pydantic)
so the library check compiles it (#1727); the service parses the form
against these fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "GATE_CONSISTENCY_FIELDS",
    "GATE_FIELDS",
    "GATE_ITEM_FIELD",
    "GateField",
]


@dataclass(frozen=True)
class GateField:
    """One threshold of the gate as the inspector edits it: a checkbox that
    says whether it is judged, a bar and a number.

    Attributes:
        name: The :class:`~autoware_carla_scenario.authoring.models.GateSpec`
            field.
        label: What the inspector calls it.
        unit: ``%`` for a share (edited as 0-100, stored as 0-1), ``m``,
            ``m/s``, or ``""`` for a count.
        default: The value a threshold gets when its checkbox is ticked with
            no value of its own, in the unit edited.
        bar_max: Where the bar ends, in the unit edited (the number may go
            past it).
        step: The bar's and the number's step.
        optional: Whether it has a checkbox at all: a count the gate always
            reads has none.
    """

    name: str
    label: str
    unit: str
    default: float
    bar_max: float
    step: float
    optional: bool = True

    @property
    def percent(self) -> bool:
        return self.unit == "%"

    def shown(self, stored: Optional[float]) -> str:
        """*stored* as the number edited (a share as a percentage), its
        default when unset."""
        if stored is None:
            return f"{self.default:g}"
        return f"{round(stored * 100.0, 6) if self.percent else stored:g}"


#: The gate's thresholds the inspector edits, in the order it lists them.
GATE_FIELDS: tuple[GateField, ...] = (
    GateField("min_runs", "Minimum runs", "", 1, 100, 1, optional=False),
    GateField("min_success_rate", "Minimum success rate", "%", 90, 100, 1),
    GateField("min_case_coverage", "Minimum case coverage", "%", 80, 100, 1),
    GateField("min_scenario_coverage", "Minimum scenario coverage", "%", 80, 100, 1),
    GateField("min_odd_coverage", "Minimum ODD coverage", "%", 50, 100, 1),
    GateField("max_lateral_spread_m", "Maximum lateral spread", "m", 0.5, 2.0, 0.05),
    GateField("max_speed_spread_mps", "Maximum speed spread", "m/s", 1.0, 5.0, 0.1),
)

#: How the trajectory spreads are measured: counts, edited only while a spread
#: is judged.
GATE_CONSISTENCY_FIELDS: tuple[GateField, ...] = (
    GateField(
        "consistency_stations", "Stations per segment", "", 20, 200, 1, optional=False
    ),
    GateField("consistency_min_runs", "Runs per station", "", 2, 20, 1, optional=False),
)

#: A cover item's grade, as each row of them edits it: added at 100 %.
GATE_ITEM_FIELD = GateField("item", "Coverage", "%", 100, 100, 1)
