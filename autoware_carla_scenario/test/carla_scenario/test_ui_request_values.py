"""What the viewer reads off a request and links a session with, pinned.

Pins, against verbatim copies of them as they were in ``ui.app`` (the
``_old_*`` oracles below), whether a ``Content-Type`` names JSON
(``_is_json_type()``), the reference of a session in a link
(``_session_ref()``) and the ODD attribute a query or a default names, if the
view has it (``_attribute()``): what each returns (type and value) or raises
(type and args), for random values of every spelling and of other types.
"""

from __future__ import annotations

import random
from typing import Any, Callable, List, Optional, Tuple

from autoware_carla_scenario.ui import app


def _old_is_json_type(content_type: str) -> bool:
    """Whether a ``Content-Type`` names JSON (``application/json``, with or
    without parameters such as ``; charset=utf-8``)."""
    return content_type.split(";", 1)[0].strip().lower() == "application/json"


def _old_session_ref(session_type: str, date: str, time: str) -> str:
    return f"{session_type}/{date}/{time}"


def _old_attribute(value: str | None, attributes: list[str]) -> str | None:
    return value if value and value in attributes else None


def _outcome(function: Callable[..., Any], *args: Any) -> Tuple[Any, ...]:
    try:
        value = function(*args)
    except Exception as exc:  # noqa: BLE001 - the oracle's exceptions are compared
        return ("raised", type(exc), exc.args)
    return ("returned", type(value), value)


_PIECES = [
    "application/json",
    "APPLICATION/JSON",
    "Application/Json",
    "application/jsonx",
    "application/json5",
    "text/plain",
    "text/json",
    "application/ld+json",
    "",
    " ",
    "\t",
    "\n",
    " ",
    " ",
    ";",
    ";;",
    "; charset=utf-8",
    ";charset=UTF-8",
    "application/json;",
    "İ",
    "ß",
    "ǅ",
    "K",
]


def _content_types(rng: random.Random) -> List[str]:
    found = list(_PIECES)
    for _ in range(2000):
        found.append("".join(rng.choice(_PIECES) for _ in range(rng.randrange(0, 5))))
    return found


def test_is_json_type_is_as_before() -> None:
    rng = random.Random(2135)
    for content_type in _content_types(rng):
        assert _outcome(app._is_json_type, content_type) == _outcome(
            _old_is_json_type, content_type
        ), repr(content_type)


def test_is_json_type_refuses_what_is_no_str_as_before() -> None:
    for value in (None, 1, b"application/json", ["application/json"], object()):
        assert _outcome(app._is_json_type, value) == _outcome(
            _old_is_json_type, value
        ), repr(value)


_PARTS = ["batch", "multirun", "single", "", "/", "2026-11-13", "12-00-00", "ü", "%2F"]


def test_session_ref_is_as_before() -> None:
    rng = random.Random(2135)
    triples: List[Tuple[Any, Any, Any]] = [
        (a, b, c) for a in _PARTS[:4] for b in _PARTS[3:6] for c in _PARTS[5:]
    ]
    for _ in range(500):
        triples.append(
            (
                "".join(rng.choice(_PARTS) for _ in range(rng.randrange(0, 3))),
                rng.choice(_PARTS),
                rng.choice(_PARTS),
            )
        )
    # Other types are formatted as f-strings format them.
    triples += [(1, 2.5, None), (True, b"x", ["y"])]
    for triple in triples:
        assert _outcome(app._session_ref, *triple) == _outcome(
            _old_session_ref, *triple
        ), triple


def test_attribute_is_as_before() -> None:
    rng = random.Random(2135)
    names = ["speed", "rain", "", "Speed", "speed ", "a.b", "ü"]
    cases: List[Tuple[Optional[str], Any]] = []
    for _ in range(1000):
        attributes = [rng.choice(names) for _ in range(rng.randrange(0, 4))]
        value = rng.choice([None, *names])
        cases.append((value, attributes))
    # A view's attributes of another container type, as `in` reads them.
    cases += [
        ("speed", ("speed",)),
        ("speed", {"speed": 1}),
        ("sp", "speed"),
        ("", ""),
        (None, None),
        ("speed", None),
        ("speed", 5),
    ]
    for value, attributes in cases:
        assert _outcome(app._attribute, value, attributes) == _outcome(
            _old_attribute, value, attributes
        ), (value, attributes)
