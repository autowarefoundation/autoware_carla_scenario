"""Escape free-form text for the TOML and Python files the generators write.

Scenario titles and descriptions are the user's prose, and they end up in a
generated ``pyproject.toml`` and in the docstring of a generated
``__init__.py``.  These functions are what keeps a quote, a backslash or a
control character in that prose from breaking either file.  :mod:`templating`
registers them as the Jinja filters ``toml_string`` and ``py_docstring``.
"""

from __future__ import annotations

import re

__all__ = ["py_docstring", "toml_string"]

#: Control characters a TOML basic string may not hold unescaped (tab, LF and
#: CR are escaped by name in :func:`toml_string`).
_TOML_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Control characters other than tab and newline, which Python source either
#: rejects (NUL) or silently rewrites (CR).
_PY_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def toml_string(text: str) -> str:
    """Return *text* as a TOML basic string, quotes and all.

    Scenario titles and descriptions are free-form text that reaches
    ``pyproject.toml``.  A description containing a quote or a newline -- an
    ordinary thing to write -- produced a file `uv lock` could not parse, so the
    export failed on the user's prose rather than on anything about the
    scenario.
    """
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    # TOML forbids every other control character in a basic string, raw.
    escaped = _TOML_CONTROL.sub(lambda m: f"\\u{ord(m.group()):04X}", escaped)
    return f'"{escaped}"'


def _escape_closing_quotes(text: str) -> str:
    """Return *text* with a backslash before every quote that could close a
    ``\"\"\"`` docstring: one followed by another quote, or the last character
    (the template's closing quotes follow it).

    What ``re.sub(r'"(?="|\\Z)', ...)`` did, which Codon's re (RE2) has
    neither the lookahead nor ``\\Z`` for.
    """
    parts: list[str] = []
    last = len(text) - 1
    for index, char in enumerate(text):
        if char == '"' and (index == last or text[index + 1] == '"'):
            parts.append('\\"')
        else:
            parts.append(char)
    return "".join(parts)


def py_docstring(text: str) -> str:
    r"""Return *text* escaped for the body of a triple-quoted docstring.

    The escaped text goes between the template's own triple quotes and, once
    Python has read it, ``__doc__`` holds *text* again.  A description that
    contained three double quotes closed the docstring early, and one with a
    backslash -- a Windows path such as ``C:\Nishishinjuku`` -- was read as
    an escape (``\N{...}``), so the generated ``__init__.py`` did not compile.
    Only quotes that could close the docstring are escaped, so ordinary prose
    stays readable in the generated source.
    """
    escaped = _escape_closing_quotes(text.replace("\\", "\\\\"))
    return _PY_CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", escaped)
