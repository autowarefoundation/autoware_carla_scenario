"""Turn HTML form submissions into typed IR parameters.

The inspector renders controls from :class:`FieldSpec` metadata, so parsing
them back is metadata-driven too: this module never looks at a primitive's
``type``, only at the kinds of fields its spec declares.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..authoring.registry import REFERENCE_PATTERN, TRUTHY_VALUES, FieldSpec

__all__ = ["parse_int_list", "parse_params", "parse_value"]

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

    if spec.kind in ("int", "lanelet"):
        return int(text)
    if spec.kind == "number":
        return float(text)
    if spec.kind in ("int_list", "int_list_or_ref", "lanelet_list"):
        return parse_int_list(text)
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
        present = spec.name in form
        if not present and spec.kind != "bool":
            params[spec.name] = spec.default
            continue
        try:
            params[spec.name] = parse_value(spec, form.get(spec.name), present=present)
        except ValueError as exc:
            raise ValueError(f"{spec.label}: {exc}") from exc
    return params
