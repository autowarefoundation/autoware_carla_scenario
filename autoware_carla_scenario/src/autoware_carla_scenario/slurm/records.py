"""A SLURM batch's JSON record as the text of each of its members.

A record of a batch (:mod:`.manifest`: a manifest, a submission, a task's
status, a case's outcome) is a JSON object whose values are of mixed types,
which Codon cannot hold beyond the function that parses it.
:class:`JsonMembers` holds such an object as its members, in their order, each
as its value's JSON text (``json.dumps(value, ensure_ascii=False)``), which
reads back as the value with ``json.loads()``; the members give back, byte for
byte, the text :func:`.manifest.write_json` writes for the object
(``indent=2, ensure_ascii=False``).

A value is held as the text the C encoder writes, not as it is laid out
inside the object, because reading must not fail where ``json.loads()`` reads
the object: ``write_json`` lays out ``indent=2`` with ``json.dump()``, which
always takes ``json``'s encoder written in Python, which recurses a Python
call per level of nesting and so fails on a value nested a thousand deep,
while ``json.loads()`` reads ten times that on Python 3.12 (Python 3.10
counts both against the same limit; the C encoder then has a level to spare,
as a member is a level shallower than its object).  Only the text written for
the object (:meth:`JsonMembers.object_text`, :meth:`JsonMembers.member_text`)
is laid out, with ``json.dump()`` as ``write_json`` lays it out, so it fails
where ``write_json`` fails: ``json.dumps()`` takes the C encoder for
``indent=2`` too from Python 3.13 on, which on 3.14 goes as deep as the stack
allows.

:func:`parse_object` reads an object's text into its members as
``json.loads()`` reads it: a key named twice keeps its first place and its last
value.  :meth:`JsonMembers.from_value` takes the object as Python holds it.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["JsonMembers", "parse_object"]


class _Chunks:
    """What ``json.dump()`` writes to: the chunks it writes, joined by
    :meth:`text`."""

    _parts: list[str]

    def __init__(self) -> None:
        self._parts = list[str]()

    def write(self, chunk: str) -> int:
        self._parts.append(chunk)
        return len(chunk)

    def text(self) -> str:
        return "".join(self._parts)


def _member_text(single: str) -> str:
    """The member of *single*, ``json.dumps({key: value}, indent=2,
    ensure_ascii=False)``, as that member is laid out inside any object at
    the same depth: the text between the braces (``"{\\n"`` and ``"\\n}"``)."""
    return single[2:-2]


class JsonMembers:
    """The members of a JSON object, in their order, each as its value's text.

    Each value is held as ``json.dumps(value, ensure_ascii=False)`` writes it
    (:meth:`value_text`), on one line.  A member is laid out as
    ``json.dumps(obj, indent=2, ensure_ascii=False)`` writes it between the
    object's braces only when the object's text is asked for
    (:meth:`member_text`, :meth:`object_text`).
    """

    _values: dict[str, str]
    #: The laid-out text of the members :meth:`from_value` took, written from
    #: the value as Python holds it: two keys of a nested ``dict`` that JSON
    #: writes alike (``1`` and ``"1"``) are both in the text ``write_json``
    #: writes, and only one in the value read back.
    _layouts: dict[str, str]

    def __init__(self) -> None:
        self._values = dict[str, str]()
        self._layouts = dict[str, str]()

    @classmethod
    def from_value(cls, value: Any) -> JsonMembers:
        """The members of *value*, a ``dict`` ``json.dumps()`` writes as an
        object, with its keys as JSON makes them strings (``1``, ``2.5``,
        ``False`` and ``None`` are ``"1"``, ``"2.5"``, ``"false"`` and
        ``"null"``).

        Raises what ``json.dumps()`` raises for a key or a value it cannot
        write, member by member, in their order (laid out as ``write_json``
        lays it out, so a value nested too deep for that raises too); a
        :class:`TypeError` for a value that is not a ``dict``, and a
        :class:`ValueError` for two keys JSON writes as the same string
        (``1`` and ``"1"``), which an object of members by their name cannot
        hold.
        """
        if not isinstance(value, dict):
            raise TypeError(
                f"a JSON object's members are a dict's, not a {type(value).__name__}'s"
            )
        members = cls()
        for key, item in value.items():
            chunks = _Chunks()
            json.dump({key: item}, chunks, indent=2, ensure_ascii=False)
            single = chunks.text()
            name = next(iter(json.loads(json.dumps({key: None}))))
            if name in members._values:
                raise ValueError(f"the object names the member {name!r} twice")
            members._values[name] = json.dumps(item, ensure_ascii=False)
            members._layouts[name] = _member_text(single)
        return members

    def __len__(self) -> int:
        return len(self._values)

    def keys(self) -> list[str]:
        """The object's keys, in their order."""
        return list(self._values.keys())

    def has(self, key: str) -> bool:
        """Whether the object has the member *key*."""
        return key in self._values

    def member_text(self, key: str) -> str:
        """The text of the member *key*, as the object lays it out:
        ``  "key": value``, with the value's lines indented as at the
        object's depth.  :class:`KeyError` when there is none.

        Laid out with ``json``'s encoder written in Python, it raises
        :class:`RecursionError` for a value nested too deep for it, as
        ``write_json`` does."""
        if key in self._layouts:
            return self._layouts[key]
        value = json.loads(self._values[key])
        chunks = _Chunks()
        json.dump({key: value}, chunks, indent=2, ensure_ascii=False)
        return _member_text(chunks.text())

    def value_text(self, key: str) -> str:
        """The JSON text of the value of the member *key*, on one line:
        ``json.dumps(value, ensure_ascii=False)``.  :class:`KeyError` when
        there is none."""
        return self._values[key]

    def value(self, key: str) -> Any:
        """The value of the member *key*, as ``json.loads()`` reads the
        object.  :class:`KeyError` when there is none."""
        return json.loads(self._values[key])

    def set_text(self, key: str, value_json: str) -> None:
        """Set the member *key* to the value of the JSON text *value_json*:
        in its place when the object has the member, else after the others,
        as a ``dict`` does."""
        self._values[key] = json.dumps(json.loads(value_json), ensure_ascii=False)
        if key in self._layouts:
            del self._layouts[key]

    def put(self, key: str, value_json: str) -> None:
        """Set the member *key* to the value *value_json*, the text
        ``json.dumps(value, ensure_ascii=False)`` wrote for it, which is held
        as it is: not read again, so the caller writes it as deep in the
        stack as it read the value (:func:`.manifest.manifest_of_text`).  In
        its place when the object has the member, else after the others."""
        self._values[key] = value_json
        if key in self._layouts:
            del self._layouts[key]

    def object_text(self) -> str:
        """The object's text, as :func:`.manifest.write_json` writes it:
        ``json.dumps(obj, indent=2, ensure_ascii=False)`` and a newline (see
        :meth:`member_text`)."""
        if len(self._values) == 0:
            return "{}\n"
        return "{\n" + ",\n".join([self.member_text(k) for k in self.keys()]) + "\n}\n"


def parse_object(text: str) -> JsonMembers | None:
    """The members of the JSON object *text*, as ``json.loads()`` reads it,
    or ``None`` when *text* is JSON that is not an object.

    Raises what ``json.loads()`` raises for text that is not JSON (a
    :class:`ValueError`, a :class:`RecursionError` for text nested too deep
    for it), and otherwise only the :class:`RecursionError` of the C encoder
    each value is written back with, where it cannot go as deep as
    ``json.loads()`` read: up to Python 3.14.6 it goes at least as deep, while
    3.14.10 bounds both by the stack and reads some hundred thousand levels
    it cannot write back.
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        return None
    members = JsonMembers()
    for key, value in data.items():
        members._values[str(key)] = json.dumps(value, ensure_ascii=False)
    return members
