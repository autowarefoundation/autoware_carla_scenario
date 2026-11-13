"""What the viewer reads off a request and links a session with.

Whether a request body's ``Content-Type`` names JSON (:func:`_is_json_type`),
the reference of a session in the viewer's links
(``<session_type>/<date>/<time>``, :func:`_session_ref`), and the ODD
attribute a query or a default names, if the view has it
(:func:`_attribute`).

Split out of ``ui.app``, which re-exports every name moved here and keeps
what serves the requests (fastapi, starlette) and reads a session off disk
(``_parse_ref()``, through ``ui.app._base_path()``, which tests patch), so
that the library check compiles it (``docs/typecheck.md``, "Checking the
framework itself").
"""

from __future__ import annotations


def _is_json_type(content_type: str) -> bool:
    """Whether a ``Content-Type`` names JSON (``application/json``, with or
    without parameters such as ``; charset=utf-8``)."""
    return content_type.split(";", 1)[0].strip().lower() == "application/json"


def _session_ref(session_type: str, date: str, time: str) -> str:
    return f"{session_type}/{date}/{time}"


def _attribute(value: str | None, attributes: list[str]) -> str | None:
    return value if value and value in attributes else None
