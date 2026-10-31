"""Validate a scenario config's ``controls`` and report problems with their config path.

:func:`~.sampler.controls_from_mapping` raises on the first problem it meets and
names the knob, not the place in the config.  :func:`validate_controls` checks
every control and returns :class:`ControlDiagnostic` s instead, each with the
path in the config (``controls.vehicle_ahead_gap_m.range``), so that
``scenario-check`` and ``scenario-expand`` can print one line per problem.

An unknown *measure* name is only a warning: a scenario may register measures
of its own (:meth:`~autoware_carla_scenario.BaseScenario.register_measure`) at
run time, which a check of the config cannot see.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ..measures import BUILT_IN_MEASURES
from .control_diagnostics import ControlDiagnostic, format_diagnostics

__all__ = [
    "ControlDiagnostic",
    "ControlsError",
    "config_keys",
    "format_diagnostics",
    "validate_controls",
]

_FIELDS = ("key", "target", "values", "scale", "offset", "integer", "range", "limits")
_CONTROL_TARGETS = ("setting", "measure")
_MAPPING_FIELDS = ("scale", "offset", "integer", "values")
_MISSING = object()


class ControlsError(ValueError):
    """The ``controls`` of a config are invalid (:attr:`diagnostics`, errors only)."""

    def __init__(self, diagnostics: Sequence[ControlDiagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        super().__init__("\n".join(d.format() for d in self.diagnostics))


def config_keys(node: Any, prefix: str = "") -> list[str]:
    """The dotted keys of the nested mapping *node*."""
    if not isinstance(node, Mapping):
        return []
    keys: list[str] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        keys.append(path)
        keys += config_keys(value, path)
    return keys


def _hint(word: str, known: Sequence[str]) -> str:
    close = difflib.get_close_matches(word, list(known), n=1, cutoff=0.75)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _key_exists(cfg: Any, key: str) -> bool:
    """Whether *key* is a key of *cfg*: a ``null`` or ``???`` (mandatory) value is one too."""
    from omegaconf import DictConfig, OmegaConf  # noqa: PLC0415

    try:
        if OmegaConf.select(cfg, key, default=_MISSING) is not _MISSING:
            return True
        # select() answers the default for a mandatory (`???`) value as well.
        parent_key, _, leaf = key.rpartition(".")
        parent = OmegaConf.select(cfg, parent_key) if parent_key else cfg
        return isinstance(parent, DictConfig) and OmegaConf.is_missing(parent, leaf)
    except Exception:  # noqa: BLE001 - a malformed key is not found
        return False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pair(
    value: Any, path: str, what: str, *, open_ends: bool
) -> Optional[ControlDiagnostic]:
    ends_ok = (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(
            _is_number(v) or (open_ends and v is None) for v in value
        )  # null: an open end
    )
    if not ends_ok:
        return ControlDiagnostic(
            path,
            f"expected [low, high] (numbers{', null for an open end' if open_ends else ''}), got {value!r}",
        )
    low = -math.inf if value[0] is None else value[0]
    high = math.inf if value[1] is None else value[1]
    if low > high:
        return ControlDiagnostic(
            path, f"expected [low, high] with low <= high, got {list(value)!r}"
        )
    return None


def _check_control(
    measure: str, spec: Any, keys: Optional[Sequence[str]], cfg: Any
) -> list[ControlDiagnostic]:
    base = f"controls.{measure}"
    if isinstance(spec, str):
        spec = {"key": spec}
    if not isinstance(spec, Mapping):
        return [ControlDiagnostic(base, f"expected a key or a mapping, got {spec!r}")]
    out: list[ControlDiagnostic] = []
    for name in spec:
        if name not in _FIELDS:
            out.append(
                ControlDiagnostic(
                    f"{base}.{name}", f"unknown field{_hint(str(name), _FIELDS)}"
                )
            )
    if "key" not in spec:
        out.append(ControlDiagnostic(f"{base}.key", "required"))
    else:
        key = spec["key"]
        if not isinstance(key, str) or not key:
            out.append(
                ControlDiagnostic(f"{base}.key", f"expected a config key, got {key!r}")
            )
        elif cfg is not None:
            if not _key_exists(cfg, key):
                out.append(
                    ControlDiagnostic(
                        f"{base}.key",
                        f"{key!r} is not a key of the config{_hint(key, keys or ())}",
                    )
                )
    target = spec.get("target", "setting")
    if target not in _CONTROL_TARGETS:
        out.append(
            ControlDiagnostic(
                f"{base}.target",
                f"expected one of {list(_CONTROL_TARGETS)}, got {target!r}",
            )
        )
    elif target == "measure":
        mapped = sorted(f for f in _MAPPING_FIELDS if f in spec)
        for name in mapped:
            out.append(
                ControlDiagnostic(
                    f"{base}.{name}",
                    "not allowed with target: measure (the value is written unchanged)",
                )
            )
    if "range" in spec and spec["range"] is not None:
        bad = _pair(spec["range"], f"{base}.range", "range", open_ends=False)
        if bad:
            out.append(bad)
    if "limits" in spec and spec["limits"] is not None:
        bad = _pair(spec["limits"], f"{base}.limits", "limits", open_ends=True)
        if bad:
            out.append(bad)
    for name in ("scale", "offset"):
        if name in spec and not _is_number(spec[name]):
            out.append(
                ControlDiagnostic(
                    f"{base}.{name}", f"expected a number, got {spec[name]!r}"
                )
            )
    if _is_number(spec.get("scale")) and spec["scale"] == 0:
        out.append(ControlDiagnostic(f"{base}.scale", "must not be zero"))
    if "integer" in spec and not isinstance(spec["integer"], bool):
        out.append(
            ControlDiagnostic(
                f"{base}.integer", f"expected true or false, got {spec['integer']!r}"
            )
        )
    if "values" in spec and spec["values"] is not None:
        if not isinstance(spec["values"], Mapping):
            out.append(
                ControlDiagnostic(
                    f"{base}.values",
                    f"expected a mapping of bucket label -> value, got {spec['values']!r}",
                )
            )
    return out


def validate_controls(controls: Any, cfg: Any = None) -> list[ControlDiagnostic]:
    """Every problem with *controls* (a config's ``controls``, resolved).

    *cfg*, the composed config, lets ``key`` be checked against the config's
    own keys.  A measure that is not a built-in one is a warning, not an
    error: the scenario may register it itself.
    """
    if controls is None:
        return []
    if not isinstance(controls, Mapping):
        return [
            ControlDiagnostic(
                "controls",
                f"expected a mapping of measure -> control, got {controls!r}",
            )
        ]
    keys: Optional[list[str]] = None
    if cfg is not None:
        from omegaconf import OmegaConf  # noqa: PLC0415

        keys = config_keys(OmegaConf.to_container(cfg, resolve=False))
    out: list[ControlDiagnostic] = []
    for measure, spec in controls.items():
        measure = str(measure)
        if measure not in BUILT_IN_MEASURES:
            out.append(
                ControlDiagnostic(
                    f"controls.{measure}",
                    "not a built-in measure"
                    f"{_hint(measure, list(BUILT_IN_MEASURES))}; "
                    "unused unless the scenario registers a measure of this name",
                    "warning",
                )
            )
        out.extend(_check_control(measure, spec, keys, cfg))
    if not any(d.severity == "error" for d in out):
        # Whatever the sampler refuses that the checks above do not know of.
        from .sampler import controls_from_mapping  # noqa: PLC0415

        for measure, spec in controls.items():
            try:
                controls_from_mapping({measure: spec})
            except (ValueError, TypeError) as exc:
                out.append(ControlDiagnostic(f"controls.{measure}", str(exc)))
    return out
