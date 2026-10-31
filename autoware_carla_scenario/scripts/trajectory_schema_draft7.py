"""Write the draft-07 copy of the trajectory log's schema from the 2020-12 one.

``schemas/trajectory.schema.json`` (JSON Schema 2020-12, ``prefixItems``
tuples) is the trajectory log's schema.  ``trajectory_recorder.schema.json``
beside the recorder is the same schema in draft-07, for the code generators
that cannot read 2020-12 tuples (quicktype); it is generated, never edited:

    uv run python autoware_carla_scenario/scripts/trajectory_schema_draft7.py

``test_trajectory_recorder.py`` fails when the two differ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "autoware_carla_scenario"
SOURCE = PACKAGE / "schemas" / "trajectory.schema.json"
TARGET = PACKAGE / "trajectory_recorder.schema.json"
#: The draft-07 copy keeps the id readers of the first format knew it by.
DRAFT7_ID = "urn:autoware-carla-scenario:schema:trajectory"


def _repoint(pointer: str) -> str:
    """*pointer*, a URI reference into ``$defs``, pointing into ``definitions``."""
    return pointer.replace("#/$defs/", "#/definitions/", 1)


def to_draft7(node: Any) -> Any:
    """*node*, a 2020-12 schema or part of one, as draft-07 says the same.

    Only the keywords whose value is a reference are repointed -- ``$ref`` and
    the OpenAPI ``discriminator.mapping`` values -- so a ``const``, ``enum``,
    ``default``, ``examples`` or ``pattern`` that happens to hold ``#/$defs/``
    keeps it.
    """
    if isinstance(node, list):
        return [to_draft7(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "$schema":
            out[key] = "http://json-schema.org/draft-07/schema#"
        elif key == "$id":
            out[key] = DRAFT7_ID
        elif key == "$defs":
            out["definitions"] = to_draft7(value)
        elif key == "$ref" and isinstance(value, str):
            out[key] = _repoint(value)
        elif (
            key == "discriminator"
            and isinstance(value, dict)
            and isinstance(value.get("mapping"), dict)
        ):
            out[key] = {
                **value,
                "mapping": {
                    tag: _repoint(ref) if isinstance(ref, str) else ref
                    for tag, ref in value["mapping"].items()
                },
            }
        elif key == "prefixItems":
            # A tuple: 2020-12's prefixItems + items (false, or the schema of
            # the items after the tuple) is draft-07's items array +
            # additionalItems.
            out["items"] = to_draft7(value)
            if "items" in node:
                out["additionalItems"] = to_draft7(node["items"])
        elif key == "items" and "prefixItems" in node:
            continue
        else:
            out[key] = to_draft7(value)
    return out


def render(schema: dict[str, Any]) -> str:
    return json.dumps(to_draft7(schema), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    TARGET.write_text(
        render(json.loads(SOURCE.read_text(encoding="utf-8"))), encoding="utf-8"
    )
    print(f"wrote {TARGET}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
