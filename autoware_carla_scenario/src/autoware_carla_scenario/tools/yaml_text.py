"""How ``detect-no-3d-model`` writes ``no_3d_model_lanelet_ids`` as YAML text.

:func:`format_value` renders the key and its list, :func:`key_line_tail` is
what followed the key's ``:`` on its line when the value starts on a later
one, :func:`comment_start` where a comment starts in that text, and
:func:`insert_after_line` adds an entry after the line a section's last value
ends on.  :mod:`.detect_no_3d_model_lanelets`, which finds where the YAML
nodes are (PyYAML), re-exports them under the names they had there (#2364):
free of YAML, CARLA and argparse, they are compiled by the library check
(docs/typecheck.md).
"""

from __future__ import annotations

__all__ = [
    "NO_3D_MODEL_KEY",
    "comment_start",
    "format_value",
    "insert_after_line",
    "key_line_tail",
]

#: The key the detected lanelet IDs are written to.
NO_3D_MODEL_KEY = "no_3d_model_lanelet_ids"


def comment_start(tail: str) -> int:
    """Where the comment in *tail* starts, with the whitespace before it;
    ``len(tail)`` when it holds none.

    A comment starts at a ``#`` at the start of *tail* or after whitespace
    (an anchor may hold one).  What ``re.search(r"\\s*(?:^|(?<=\\s))#", tail)``
    found, whose lookbehind Codon's re (RE2) lacks: the leftmost match starts
    the run of whitespace before the first such ``#`` (``\\s`` is
    :meth:`str.isspace` for a ``str`` pattern).
    """
    n = len(tail)
    hash_at = -1
    i = 0
    while i < n:
        if tail[i] == "#" and (i == 0 or tail[i - 1].isspace()):
            hash_at = i
            break
        i += 1
    if hash_at < 0:
        return n
    start = hash_at
    while start > 0 and tail[start - 1].isspace():
        start -= 1
    return start


def key_line_tail(text: str, colon_from: int, content_start: int) -> str:
    """Return the text after the key's ``:`` up to the end of its line -- a
    comment, an anchor or a tag -- when the value's content starts on a later
    line, so that replacing the value keeps it; ``""`` otherwise (the line's
    rest is then the value itself, or kept after the replaced span).

    *colon_from* is where the key ends (the ``:`` is the first after it) and
    *content_start* where the value's content starts, both indices in *text*.
    """
    colon = text.find(":", colon_from)
    newline_at = text.find("\n", colon)
    return_at = text.find("\r", colon)
    line_end = len(text)
    if newline_at != -1 and return_at != -1:
        line_end = min(newline_at, return_at)
    elif newline_at != -1:
        line_end = newline_at
    elif return_at != -1:
        line_end = return_at
    if content_start <= line_end:
        return ""
    return text[colon + 1 : line_end]


def format_value(
    lanelet_ids: list[int],
    indent: str,
    newline: str,
    *,
    flow: bool,
    tail: str = "",
) -> str:
    """Render ``no_3d_model_lanelet_ids: <ids>`` starting at the key.

    *tail* is what followed the key's ``:`` on its line (see
    :func:`key_line_tail`): kept after the ``:`` of a block list, and split
    around a flow list -- its anchor or tag before it, its comment after.
    """
    if not lanelet_ids or flow:
        items = ", ".join([str(int(lid)) for lid in lanelet_ids])
        # A comment starts at a "#" after whitespace (an anchor may hold one);
        # the whitespace before it goes with it.
        split = comment_start(tail)
        properties = tail[:split].strip()
        after = tail[split:]
        if after.startswith("#"):
            after = " " + after
        head = (
            f"{NO_3D_MODEL_KEY}: {properties} "
            if properties
            else f"{NO_3D_MODEL_KEY}: "
        )
        return f"{head}[{items}]{after}"
    lines = [f"{NO_3D_MODEL_KEY}:{tail}"]
    for lid in lanelet_ids:
        lines.append(f"{indent}  - {int(lid)}")
    return newline.join(lines)


def insert_after_line(text: str, last_end: int, entry: str, newline: str) -> str:
    """Return *text* with *entry* after the line holding index *last_end*
    (where a section's last value ends).

    On the last line of a file without a final line break, *newline* ends
    that line first.
    """
    line_end = text.find("\n", last_end)
    if line_end == -1:
        ends_a_line = text.endswith("\n") or text.endswith("\r")
        prefix = text if ends_a_line else text + newline
        return prefix + entry
    return text[: line_end + 1] + entry + text[line_end + 1 :]
