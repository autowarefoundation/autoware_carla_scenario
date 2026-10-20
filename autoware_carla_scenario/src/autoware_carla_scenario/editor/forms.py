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
    TRUTHY_VALUES,
    FieldSpec,
    default_params,
)

__all__ = [
    "RECORD_OP",
    "RECORD_ROWS",
    "parse_int_list",
    "parse_params",
    "parse_records",
    "parse_value",
]

#: A ``record_list`` field posts each record's fields as ``<field>.<i>.<sub>``,
#: with ``<field>.__rows`` saying how many records the form holds -- so an
#: empty list, and a record whose checkboxes are all clear, still parse -- and
#: ``<field>.__op`` naming the row button pressed: ``add``, or ``delete``,
#: ``up`` or ``down`` with ``:<i>``.
RECORD_ROWS = "__rows"
RECORD_OP = "__op"

#: Separators accepted in a list-of-integers field: commas, spaces, newlines.
_LIST_SEPARATORS = re.compile(r"[,\s]+")


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
    if spec.kind == "bool":
        return (
            present and str(raw if raw is not None else "on").lower() in TRUTHY_VALUES
        )

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
        if spec.kind == "record_list":
            if f"{spec.name}.{RECORD_ROWS}" in form:
                params[spec.name] = parse_records(spec, form)
            else:
                params[spec.name] = spec.default
            continue
        present = spec.name in form
        if not present and spec.kind != "bool":
            params[spec.name] = spec.default
            continue
        try:
            params[spec.name] = parse_value(spec, form.get(spec.name), present=present)
        except ValueError as exc:
            raise ValueError(f"{spec.label}: {exc}") from exc
    return params


def parse_records(spec: FieldSpec, form: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the records a ``record_list`` field's rows post, with the row
    button pressed (if any) applied.

    Raises:
        ValueError: If a record's field cannot be parsed, or the row count or
            the button is malformed.
    """
    count = int(str(form[f"{spec.name}.{RECORD_ROWS}"]))
    rows: list[dict[str, Any]] = []
    for index in range(count):
        prefix = f"{spec.name}.{index}."
        posted = {
            key[len(prefix) :]: value
            for key, value in form.items()
            if key.startswith(prefix)
        }
        try:
            rows.append(parse_params(spec.item_fields, posted))
        except ValueError as exc:
            raise ValueError(f"row {index + 1}: {exc}") from exc
    op = str(form.get(f"{spec.name}.{RECORD_OP}") or "")
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
