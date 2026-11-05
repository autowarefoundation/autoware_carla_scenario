"""A logical scenario's gate: when it passes over all its runs.

Each run of a logical scenario with a gate writes ``{stem}_gate.json``
(:mod:`.record`, sampled by :mod:`.probe`); ``scenario-gate``
(:mod:`.evaluate`) judges the gate from those records after the runs.  See
``docs/gate.md``.

Only the record is imported here: the evaluator reaches pydantic (the gate's
model, in :mod:`autoware_carla_scenario.authoring.models`) and openodd, and
the probe the scenario's route.
"""

from .record import (
    EGO_COLUMNS,
    GATE_RECORD_SCHEMA,
    GATE_RECORD_SUFFIX,
    GateCase,
    GateRecord,
    read_gate_record,
    write_gate_record,
)

__all__ = [
    "EGO_COLUMNS",
    "GATE_RECORD_SCHEMA",
    "GATE_RECORD_SUFFIX",
    "GateCase",
    "GateRecord",
    "read_gate_record",
    "write_gate_record",
]
