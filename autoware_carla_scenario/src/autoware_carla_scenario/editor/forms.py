"""Turn HTML form submissions into typed IR parameters.

The inspector renders controls from :class:`FieldSpec` metadata, so parsing
them back is metadata-driven too: this module never looks at a primitive's
``type``, only at the kinds of fields its spec declares.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..trajectory.authoring import (
    parse_relative_vertices,
    parse_route_vertices,
    parse_vertices,
)
from ..authoring.registry import (
    INT_KINDS,
    INT_LIST_KINDS,
    REFERENCE_PATTERN,
    FieldSpec,
    default_params,
    read_bool,
)

__all__ = [
    "bool_choice",
    "field_present",
    "is_nullable_bool",
    "parse_int_list",
    "parse_params",
    "parse_records",
    "parse_value",
]

#: A ``record_list`` field posts each record's fields as ``<field>.<i>.<sub>``,
#: with ``<field>.__rows`` saying how many records the form holds -- so an
#: empty list, and a record whose checkboxes are all clear, still parse -- and
#: ``<field>.__op`` naming the row button pressed: ``add``, or ``delete``,
#: ``up`` or ``down`` with ``:<i>`` (``partials/fields.html`` writes them).
_ROWS = "__rows"
_OP = "__op"

#: More records than any list is written with, so a forged row count is
#: refused rather than looped over.
_MAX_RECORDS = 1000

#: Separators accepted in a list-of-integers field: commas, spaces, newlines.
_LIST_SEPARATORS = re.compile(r"[,\s]+")


#: What a nullable bool's three-way select posts for each choice: keep the
#: run's setting (``None``), on, off.
_KEEP = ""
_ON = "on"
_OFF = "off"


def is_nullable_bool(spec: FieldSpec) -> bool:
    """Whether *spec* is a bool whose unset value means "keep the run's".

    Such a field has three states, which a checkbox cannot show or post, so the
    inspector renders it as a select of keep / on / off instead.
    """
    return spec.kind == "bool" and spec.default is None and not spec.required


def bool_choice(value: Any) -> str:
    """Return the option of a nullable bool's select that *value* stands for.

    ``None`` and blank text are "keep the run's"; anything else is read as the
    compiler reads a bool (:func:`~autoware_carla_scenario.authoring.registry.read_bool`).
    A stored spelling that is neither true nor false shows as *Off*, as it
    always has; the validator reports it, so the author sees it is wrong.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return _KEEP
    return _ON if read_bool(value) else _OFF


def _parse_bool(raw: Any) -> bool:
    """Return the bool a posted *raw* spells, as the compiler reads one.

    Raises:
        ValueError: If *raw* spells neither true nor false.
    """
    flag = read_bool(raw)
    if flag is None:
        raise ValueError(f"must be true or false, got {raw!r}")
    return flag


def parse_int_list(raw: Any) -> Any:
    """Return *raw* as a list of ints, or verbatim when it is an interpolation."""
    if isinstance(raw, (list, tuple)):
        return [int(str(v).strip()) for v in raw if str(v).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    if REFERENCE_PATTERN.match(text):
        return text
    return [int(part) for part in _LIST_SEPARATORS.split(text) if part]


def parse_value(spec: FieldSpec, raw: Any, *, present: bool) -> Any:
    """Return the value for one field, given what the form submitted.

    Args:
        spec: The field being parsed.
        raw: The submitted value (``None`` when the key was absent).
        present: Whether the key was present at all -- the only way to read an
            unchecked checkbox, which submits nothing.
    """
    if is_nullable_bool(spec):
        # A three-way select: blank keeps the run's setting.
        if not present or raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        return _parse_bool(raw)
    if spec.kind == "bool":
        # A checkbox: absent (or blank) is off, and a checked one posts "on".
        if not present or (isinstance(raw, str) and not raw.strip()):
            return False
        return _parse_bool("on" if raw is None else raw)

    text = raw if raw is not None else ""
    if isinstance(text, str):
        text = text.strip()

    if text == "":
        return spec.default if spec.required else None

    if spec.kind in INT_KINDS:
        return int(text)
    if spec.kind == "number":
        return float(text)
    if spec.kind in (*INT_LIST_KINDS, "int_list_or_ref"):
        return parse_int_list(text)
    if spec.kind == "trajectory":
        return [list(row) for row in parse_vertices(text)]
    if spec.kind == "relative_lane_trajectory":
        return [list(row) for row in parse_relative_vertices(text)]
    if spec.kind == "route_trajectory":
        return parse_route_vertices(text)
    return text


def parse_params(
    fields: "Sequence[FieldSpec]", form: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the ``params`` mapping described by *fields* for this submission.

    Fields the form did not mention keep their default, so a partial form (an
    inspector section that only shows some controls) never silently blanks the
    rest.

    Raises:
        ValueError: If a numeric field was given something that is not a number.
            Callers surface this to the user rather than storing a broken value.
    """
    params: dict[str, Any] = {}
    for spec in fields:
        if not field_present(spec, form):
            params[spec.name] = spec.default
            continue
        if spec.kind == "record_list":
            params[spec.name] = parse_records(spec, form)
            continue
        present = spec.name in form
        try:
            params[spec.name] = parse_value(spec, form.get(spec.name), present=present)
        except ValueError as exc:
            raise ValueError(f"{spec.label}: {exc}") from exc
    return params


def field_present(spec: FieldSpec, form: Mapping[str, Any]) -> bool:
    """Whether *form* posts *spec*'s field at all.

    A checkbox always counts -- an unchecked one posts nothing, and absence is
    its value -- and a ``record_list`` counts when it posts its row count.  A
    nullable bool is a select, not a checkbox, so it counts only when posted:
    a form that leaves it out keeps the stored value rather than storing
    ``False``.
    """
    if is_nullable_bool(spec):
        return spec.name in form
    if spec.kind == "bool":
        return True
    if spec.kind == "record_list":
        return f"{spec.name}.{_ROWS}" in form
    return spec.name in form


def parse_records(spec: FieldSpec, form: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the records a ``record_list`` field's rows post, with the row
    button pressed (if any) applied.

    Raises:
        ValueError: If a record's field cannot be parsed, or the row count or
            the button is malformed.
    """
    count = int(str(form[f"{spec.name}.{_ROWS}"]))
    if not 0 <= count <= _MAX_RECORDS:
        raise ValueError(f"{count} rows")
    # Each row's fields, by row, in one pass over the form.
    posted: list[dict[str, Any]] = [{} for _ in range(count)]
    prefix = f"{spec.name}."
    for key, value in form.items():
        if not key.startswith(prefix):
            continue
        slot, _, sub = key[len(prefix) :].partition(".")
        if sub and slot.isdigit() and int(slot) < count:
            posted[int(slot)][sub] = value
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(posted):
        try:
            rows.append(parse_params(spec.item_fields, row))
        except ValueError as exc:
            raise ValueError(f"row {index + 1}: {exc}") from exc
    op = str(form.get(f"{spec.name}.{_OP}") or "")
    if op:
        _apply_row_op(spec, rows, op)
    return rows


def _apply_row_op(spec: FieldSpec, rows: list[dict[str, Any]], op: str) -> None:
    """Add a record of defaults, or delete or move the one *op* names."""
    if op == "add":
        rows.append(default_params(spec.item_fields))
        return
    verb, _, where = op.partition(":")
    index = int(where)
    if not 0 <= index < len(rows):
        raise ValueError(f"no row {index + 1}")
    if verb == "delete":
        del rows[index]
    elif verb in ("up", "down"):
        other = index - 1 if verb == "up" else index + 1
        if 0 <= other < len(rows):
            rows[index], rows[other] = rows[other], rows[index]
    else:
        raise ValueError(f"unknown row action {op!r}")
