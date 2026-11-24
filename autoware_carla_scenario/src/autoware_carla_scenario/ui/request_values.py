"""What the viewer reads off a request and links a session with.

Whether a request body's ``Content-Type`` names JSON (:func:`_is_json_type`),
the reference of a session in the viewer's links
(``<session_type>/<date>/<time>``, :func:`_session_ref`), and the ODD
attribute a query or a default names, if the view has it
(:func:`_attribute`), and whether an error is answered with a page rather
than JSON (:func:`wants_page`); and whether a run's extra overrides are
overrides, not flags of ``scenario`` (:func:`check_extra_overrides`), which
the local and the SLURM launch paths share.

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


def wants_page(path: str, accept: str) -> bool:
    """Whether an error is answered with a page: a request for *path* whose
    ``Accept`` header (*accept*, empty when it sends none) accepts HTML (a
    browser's navigation) and that is not one of the API's, whose clients read
    its JSON error (``{"detail": ...}``)."""
    if path == "/api" or path.startswith("/api/"):
        return False
    return "text/html" in accept.lower()


def check_extra_overrides(overrides: list[str]) -> None:
    """Refuse an extra override that starts with ``-``.

    Each is passed to ``scenario`` as it is, so ``--cfg=job`` would make Hydra
    print the config and exit 0 (a job "passed" though nothing ran) and
    ``--multirun`` would send the results elsewhere (#2440).  Hydra's own
    overrides never start with ``-`` (``key=value``, ``+key=value``,
    ``~key``).

    Raises:
        ValueError: An override starts with ``-``.
    """
    for override in overrides:
        if override.startswith("-"):
            raise ValueError(
                f"extra override '{override}' starts with '-': each line is "
                "an override (key=value), not a flag"
            )
