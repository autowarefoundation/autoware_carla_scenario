"""odd.control_diagnostics' ControlDiagnostic and format_diagnostics against
the code they were moved from.

``_OriginalControlDiagnostic`` and ``_original_format_diagnostics`` are
``odd.controls``' as they were before the move (#846), copied verbatim; the
moved ones, and odd.controls' re-exports of them, must give the same text.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from autoware_carla_scenario.odd import control_diagnostics, controls


@dataclass(frozen=True)
class _OriginalControlDiagnostic:
    """One problem with the ``controls`` of a config.

    Attributes:
        path: Where in the config, e.g. ``controls.vehicle_ahead_gap_m.range``.
        message: What is wrong, with a suggestion when there is one.
        severity: ``error`` (the sweep cannot run) or ``warning``.
    """

    path: str
    message: str
    severity: str = "error"

    def format(self, prefix: str = "") -> str:
        """One line: ``<prefix>: <path>: <message>`` (``warning: `` before a warning)."""
        head = f"{prefix}: " if prefix else ""
        tag = "warning: " if self.severity == "warning" else ""
        return f"{head}{tag}{self.path}: {self.message}"


def _original_format_diagnostics(
    diagnostics: Sequence[_OriginalControlDiagnostic], prefix: str = ""
) -> list[str]:
    """The one-line text of each of *diagnostics*."""
    return [d.format(prefix) for d in diagnostics]


_PATHS = ["controls", "controls.gap.range", ""]
_MESSAGES = ["required", "expected a number, got 'x'", ""]
_SEVERITIES = ["error", "warning", "info", ""]
_PREFIXES = ["", "MyScenario", "a: b"]


def test_controls_re_exports_the_moved_names() -> None:
    assert controls.ControlDiagnostic is control_diagnostics.ControlDiagnostic
    assert controls.format_diagnostics is control_diagnostics.format_diagnostics


@pytest.mark.parametrize(
    "path,message,severity,prefix",
    list(itertools.product(_PATHS, _MESSAGES, _SEVERITIES, _PREFIXES)),
)
def test_format_matches_the_original(
    path: str, message: str, severity: str, prefix: str
) -> None:
    new = control_diagnostics.ControlDiagnostic(path, message, severity)
    old = _OriginalControlDiagnostic(path, message, severity)
    assert new.format(prefix) == old.format(prefix)
    assert new.format() == old.format()


def test_default_severity_and_value_semantics_are_unchanged() -> None:
    new = control_diagnostics.ControlDiagnostic("p", "m")
    old = _OriginalControlDiagnostic("p", "m")
    assert new.severity == old.severity == "error"
    assert new == control_diagnostics.ControlDiagnostic("p", "m", "error")
    assert hash(new) == hash(control_diagnostics.ControlDiagnostic("p", "m"))
    with pytest.raises(AttributeError):
        new.path = "q"  # type: ignore[misc]


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_format_diagnostics_matches_the_original(prefix: str) -> None:
    triples = list(itertools.product(_PATHS, _MESSAGES, _SEVERITIES))
    new = [control_diagnostics.ControlDiagnostic(*t) for t in triples]
    old = [_OriginalControlDiagnostic(*t) for t in triples]
    assert control_diagnostics.format_diagnostics(
        new, prefix
    ) == _original_format_diagnostics(old, prefix)
    assert control_diagnostics.format_diagnostics(
        tuple(new), prefix
    ) == _original_format_diagnostics(tuple(old), prefix)
    assert control_diagnostics.format_diagnostics([], prefix) == []


def test_controls_error_text_is_unchanged() -> None:
    diags = [
        control_diagnostics.ControlDiagnostic("controls.a", "bad"),
        control_diagnostics.ControlDiagnostic("controls.b", "worse", "warning"),
    ]
    error = controls.ControlsError(diags)
    assert error.diagnostics == diags
    assert str(error) == "controls.a: bad\nwarning: controls.b: worse"
