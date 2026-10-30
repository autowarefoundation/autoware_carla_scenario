"""The text a dev container's configuration is written in.

JSONC's comments and trailing commas (:func:`_strip_jsonc`), Compose's
``${VAR}`` interpolation (:func:`_interpolate`) and the ``target`` of a
devcontainer ``workspaceMount`` (:func:`_mount_target`).

Split out of ``autoware_stack.devcontainer``, which re-exports every name here
and keeps what needs yaml and the ``Any``-typed documents, so that the library
check compiles it (``docs/typecheck.md``, "Checking the framework itself").
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Optional

__all__ = ["_VARIABLE", "_interpolate", "_mount_target", "_strip_jsonc"]


def _strip_jsonc(text: str) -> str:
    """Drop ``//`` and ``/* */`` comments and trailing commas from JSONC."""
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
        elif c == '"':
            in_string = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            out.append(c)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


_VARIABLE = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:(?P<op>:?-)(?P<default>[^}]*))?\}"
)


def _interpolate(value: str, env: Mapping[str, str]) -> str:
    """Resolve Compose's ``${VAR}``, ``${VAR:-default}`` and ``${VAR-default}``."""

    def _replace(match: re.Match[str]) -> str:
        # One group per call: Codon's Match.group takes a single group.
        name = match.group("name")
        op = match.group("op")
        default = match.group("default")
        if op == ":-":
            return env.get(name) or default
        if op == "-":
            return env[name] if name in env else default
        return env.get(name, "")

    return _VARIABLE.sub(_replace, value)


def _mount_target(spec: str) -> Optional[str]:
    """``target`` of a devcontainer ``workspaceMount`` (``source=…,target=…``)."""
    if not spec:
        return None
    for part in spec.split(","):
        key, _, value = part.partition("=")
        if key.strip() in ("target", "dst", "destination") and "${" not in value:
            return value.strip()
    return None
