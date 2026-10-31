"""A problem with the ``controls`` of a scenario config, and its one-line text.

:class:`ControlDiagnostic` is what :func:`~.controls.validate_controls`
reports: the path in the config, what is wrong and how bad it is.
:meth:`ControlDiagnostic.format` and :func:`format_diagnostics` write it as
the line ``scenario-check`` and ``scenario-expand`` print.  Split out of
:mod:`.controls` so the library check compiles them (#846); checking a config
(omegaconf, values of any type) stays there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ControlDiagnostic", "format_diagnostics"]


@dataclass(frozen=True)
class ControlDiagnostic:
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


def format_diagnostics(
    diagnostics: Sequence[ControlDiagnostic], prefix: str = ""
) -> list[str]:
    """The one-line text of each of *diagnostics*."""
    return [d.format(prefix) for d in diagnostics]
