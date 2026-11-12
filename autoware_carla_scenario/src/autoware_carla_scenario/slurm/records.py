"""A SLURM batch's JSON record as the text of each of its members.

A record of a batch (:mod:`.manifest`: a manifest, a submission, a task's
status, a case's outcome) is a JSON object whose values are of mixed types,
which Codon cannot hold beyond the function that parses it.
:class:`JsonMembers` holds such an object as its members, in their order, each
as the text ``json.dumps()`` lays it out with inside the whole object
(``indent=2, ensure_ascii=False``): a member reads back as its value with
``json.loads()``, and the members give back, byte for byte, the text
:func:`.manifest.write_json` writes for the object.

:func:`parse_object` reads an object's text into its members as
``json.loads()`` reads it: a key named twice keeps its first place and its last
value.  :meth:`JsonMembers.from_value` takes the object as Python holds it.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["JsonMembers", "parse_object"]


def _member_text(single: str) -> str:
    """The member of *single*, ``json.dumps({key: value}, indent=2,
    ensure_ascii=False)``, as that member is laid out inside any object at
    the same depth: the text between the braces (``"{\\n"`` and ``"\\n}"``)."""
    return single[2:-2]


def _value_prefix(key: str) -> str:
    """What comes before the value in the member text of *key*."""
    return "  " + json.dumps(key, ensure_ascii=False) + ": "


class JsonMembers:
    """The members of a JSON object, in their order, each as its text.

    Each member's text is what ``json.dumps(obj, indent=2,
    ensure_ascii=False)`` writes for it between the object's braces, without
    the comma that separates it from the next: ``  "key": value``, with the
    value's lines indented as at the object's depth.  A JSON string never
    holds a newline of its own, so the lines of a member are its value's.
    """

    _members: dict[str, str]

    def __init__(self) -> None:
        self._members = dict[str, str]()

    @classmethod
    def from_value(cls, value: Any) -> JsonMembers:
        """The members of *value*, a ``dict`` ``json.dumps()`` writes as an
        object, with its keys as JSON makes them strings (``1``, ``2.5``,
        ``False`` and ``None`` are ``"1"``, ``"2.5"``, ``"false"`` and
        ``"null"``).

        Raises what ``json.dumps()`` raises for a key or a value it cannot
        write, member by member, in their order; a :class:`TypeError` for a
        value that is not a ``dict``, and a :class:`ValueError` for two keys
        JSON writes as the same string (``1`` and ``"1"``), which an object
        of members by their name cannot hold.
        """
        if not isinstance(value, dict):
            raise TypeError(
                f"a JSON object's members are a dict's, not a {type(value).__name__}'s"
            )
        members = cls()
        for key, item in value.items():
            single = json.dumps({key: item}, indent=2, ensure_ascii=False)
            name = next(iter(json.loads(json.dumps({key: None}))))
            if name in members._members:
                raise ValueError(f"the object names the member {name!r} twice")
            members._members[name] = _member_text(single)
        return members

    def __len__(self) -> int:
        return len(self._members)

    def keys(self) -> list[str]:
        """The object's keys, in their order."""
        return list(self._members.keys())

    def has(self, key: str) -> bool:
        """Whether the object has the member *key*."""
        return key in self._members

    def member_text(self, key: str) -> str:
        """The text of the member *key*, as the object lays it out:
        ``  "key": value``.  :class:`KeyError` when there is none."""
        return self._members[key]

    def value_text(self, key: str) -> str:
        """The JSON text of the value of the member *key*:
        ``json.dumps(value, indent=2, ensure_ascii=False)``.  :class:`KeyError`
        when there is none."""
        text = self._members[key][len(_value_prefix(key)) :]
        return text.replace("\n  ", "\n")

    def value(self, key: str) -> Any:
        """The value of the member *key*, as ``json.loads()`` reads the
        object.  :class:`KeyError` when there is none."""
        return json.loads(self.value_text(key))

    def set_text(self, key: str, value_json: str) -> None:
        """Set the member *key* to the value of the JSON text *value_json*:
        in its place when the object has the member, else after the others,
        as a ``dict`` does."""
        self._members[key] = _member_text(
            json.dumps({key: json.loads(value_json)}, indent=2, ensure_ascii=False)
        )

    def object_text(self) -> str:
        """The object's text, as :func:`.manifest.write_json` writes it:
        ``json.dumps(obj, indent=2, ensure_ascii=False)`` and a newline."""
        if len(self._members) == 0:
            return "{}\n"
        return "{\n" + ",\n".join([m for m in self._members.values()]) + "\n}\n"


def parse_object(text: str) -> JsonMembers | None:
    """The members of the JSON object *text*, as ``json.loads()`` reads it,
    or ``None`` when *text* is JSON that is not an object.

    Raises what ``json.loads()`` raises for text that is not JSON (a
    :class:`ValueError`).
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        return None
    members = JsonMembers()
    for key, value in data.items():
        name = str(key)
        members._members[name] = _member_text(
            json.dumps({name: value}, indent=2, ensure_ascii=False)
        )
    return members
