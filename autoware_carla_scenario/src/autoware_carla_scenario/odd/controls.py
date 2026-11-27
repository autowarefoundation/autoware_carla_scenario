"""Validate a scenario config's ``controls`` and report problems with their config path.

:func:`validate_controls` checks every control and returns
:class:`ControlDiagnostic` s, each with the path in the config
(``controls.vehicle_ahead_gap_m.range``), so that ``scenario-check`` and
``scenario-expand`` can print one line per problem.
:func:`~.sampler.controls_from_mapping` refuses the errors among them
(:func:`control_errors`) with a :class:`ControlsError` that reports every
problem, each with its full config path (``<key>.<name>.<field>``), before
openodd reads the knobs.

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
from .control_fields import CONTROL_FIELDS, CONTROL_TARGETS, MAPPING_FIELDS

__all__ = [
    "ControlDiagnostic",
    "ControlsError",
    "config_keys",
    "control_errors",
    "format_diagnostics",
    "validate_controls",
]

_FIELDS = CONTROL_FIELDS
_CONTROL_TARGETS = CONTROL_TARGETS
_MAPPING_FIELDS = MAPPING_FIELDS
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
    # Not ``low > high``: a NaN end compares false either way, and openodd
    # refuses it when the knob is built (#2134).
    if not low <= high:
        return ControlDiagnostic(
            path, f"expected [low, high] with low <= high, got {list(value)!r}"
        )
    return None


def _check_control(
    measure: str,
    spec: Any,
    keys: Optional[Sequence[str]],
    cfg: Any,
    base: Optional[str] = None,
) -> list[ControlDiagnostic]:
    if base is None:
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
    # A field given null is one not given: its default (#2226), as a null
    # target already was (#2171).
    spec = {name: value for name, value in spec.items() if value is not None}
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
    target = spec.get("target")
    if target is None:  # Not given, or null: the default (#2171).
        target = "setting"
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
    if "range" in spec:
        bad = _pair(spec["range"], f"{base}.range", "range", open_ends=False)
        if bad:
            out.append(bad)
    if "limits" in spec:
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
        elif name in spec and not math.isfinite(spec[name]):
            # A NaN or infinite scale or offset maps every draw outside the
            # ODD, which the sampler would only report as no case found.
            out.append(
                ControlDiagnostic(
                    f"{base}.{name}", f"expected a finite number, got {spec[name]!r}"
                )
            )
    if _is_number(spec.get("scale")) and spec["scale"] == 0:
        out.append(ControlDiagnostic(f"{base}.scale", "must not be zero"))
    if "quantity" in spec and (
        not isinstance(spec["quantity"], str) or not spec["quantity"]
    ):
        out.append(
            ControlDiagnostic(
                f"{base}.quantity",
                f"expected the name of a quantity of the rules, got {spec['quantity']!r}",
            )
        )
    if "integer" in spec and not isinstance(spec["integer"], bool):
        out.append(
            ControlDiagnostic(
                f"{base}.integer", f"expected true or false, got {spec['integer']!r}"
            )
        )
    if "values" in spec:
        if not isinstance(spec["values"], Mapping):
            out.append(
                ControlDiagnostic(
                    f"{base}.values",
                    f"expected a mapping of bucket label -> value, got {spec['values']!r}",
                )
            )
    return out


def control_errors(raw: Mapping[Any, Any], key: str) -> list[ControlDiagnostic]:
    """The errors of each control of *raw*, the config's *key* (``controls``,
    ``sweep.odd_sample.knobs``), each with its full path
    (``sweep.odd_sample.knobs.<attribute>.<field>``): what
    :func:`validate_controls` checks of one control, but not its ``key``
    against a config, nor its name against the built-in measures.

    :func:`~.sampler.controls_from_mapping` refuses what this finds before
    openodd reads the knobs, which would take ``integer: "false"`` as true
    and raise a bare ``TypeError`` on ``scale: null`` (#2226).
    """
    out: list[ControlDiagnostic] = []
    for name, spec in raw.items():
        found = _check_control(str(name), spec, None, None, f"{key}.{name}")
        out.extend(d for d in found if d.severity == "error")
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
