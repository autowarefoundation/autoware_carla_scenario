"""The worker: evaluates frames sent to it as JSON lines on its standard input.

``autoware_carla_scenario.commonroad`` starts it as a process of its own --
CommonRoad's packages pin protobuf 3 and build for CPython 3.10 and 3.11 only,
where the framework needs protobuf 4 or later and runs on 3.10 through 3.14 --
and speaks to it one line at a time, each request answered by one line:

``{"op": "init", "map": "<path of the CommonRoad XML>", "settings": {...}}``
    Read the map; the settings are :class:`~.evaluator.EvaluatorSettings`'
    fields.  Answers ``{"ok": true, "measures": [...]}``, the keys it knows.

``{"op": "evaluate", "measures": [...], "ego": {...}, "others": [{...}, ...]}``
    Evaluate the measures in a frame; each road user is an
    :class:`~.evaluator.Actor`'s fields.  Answers ``{"ok": true, "values":
    {key: value}, "errors": {key: message}}``; a value is a number,
    ``Infinity`` (JSON as Python writes it: no conflict, TTC never comes),
    ``-Infinity`` (ALongReq when no braking avoids the collision within the
    horizon; BTN is ``Infinity`` then) or ``null`` (nothing to measure).

``{"op": "close"}``
    Answers ``{"ok": true}`` and exits.

A request that cannot be served is answered ``{"ok": false, "error": "..."}``:
a line that is not UTF-8 or not JSON, an unknown ``op``, settings or a road
user refused (a NaN or an infinity among its numbers, ``horizon_steps: 0``).
Every request line gets its one answer, so the answers stay in step with the
requests.  A blank line is no request and gets none.
Logging goes to the standard error, never the output, which carries the
answers alone: :func:`main` writes them on a duplicate of descriptor 1 and
points descriptor 1 itself at the standard error, so that what CriMe,
CommonRoad-Reach or their C++ print, ``print()`` or not, cannot break into an
answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable
from typing import IO, Any

from .evaluator import MEASURES, Actor, Evaluator, EvaluatorSettings

logger = logging.getLogger("commonroad_metrics")


class Worker:
    """Serves requests; one evaluator, made by ``init``."""

    def __init__(self) -> None:
        self.evaluator: Evaluator | None = None

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "init":
            settings = EvaluatorSettings.from_mapping(request.get("settings") or {})
            self.evaluator = Evaluator(str(request["map"]), settings)
            return {"ok": True, "measures": list(MEASURES)}
        if op == "evaluate":
            if self.evaluator is None:
                return {"ok": False, "error": "evaluate before init"}
            ego = Actor.from_mapping(request["ego"])
            others = [Actor.from_mapping(o) for o in request.get("others", [])]
            values, errors = self.evaluator.evaluate(ego, others, request.get("measures", []))
            return {"ok": True, "values": values, "errors": errors}
        if op == "close":
            return {"ok": True}
        return {"ok": False, "error": f"unknown op {op!r}"}


def serve(stdin: Iterable[str | bytes], stdout: IO[str]) -> None:
    """Answer the requests on *stdin* on *stdout* until ``close`` or the end.

    *stdin* is text, or bytes decoded here as UTF-8 one line at a time, inside
    the request's ``try``: a line that is not UTF-8 is answered ``ok: false``
    as any other bad request, rather than ending the worker.
    """
    worker = Worker()
    for raw in stdin:
        if not raw.strip():
            continue
        try:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            request = json.loads(line)
            response = worker.handle(request)
        except Exception as exc:
            logger.exception("request failed")
            request = {}
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        stdout.write(json.dumps(response) + "\n")
        stdout.flush()
        if request.get("op") == "close":
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="commonroad-metrics-worker",
        description="Evaluate CriMe and CommonRoad-Reach measures on frames read "
        "as JSON lines from the standard input.",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr)
    # CriMe and Reach print progress, and C++ code may write to descriptor 1
    # directly; the output carries the answers alone (#1954).  The answers go
    # to a duplicate of the output, and descriptor 1 -- sys.stdout's and the
    # C++ code's -- to the standard error.
    sys.stdout.flush()
    answers = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        # Bytes, so that serve decodes each line itself (strictly, inside its
        # try): a text stdin decodes as it iterates, outside it.
        serve(sys.stdin.buffer, answers)
    finally:
        sys.stdout = stdout
        answers.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
