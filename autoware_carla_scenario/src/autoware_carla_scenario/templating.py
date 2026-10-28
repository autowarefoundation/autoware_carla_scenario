"""Jinja environment shared by the code and config generators.

Both the scaffolder and the Scenario Package exporter render Python, TOML and
YAML rather than HTML, and both need the same strictness: no autoescaping, and
a loud failure on a variable the template asked for but the caller forgot.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

__all__ = ["code_environment", "py_docstring", "toml_string"]


def toml_string(value: object) -> str:
    """Return *value* as a TOML basic string, quotes and all.

    Scenario titles and descriptions are free-form text that reaches
    ``pyproject.toml``.  A description containing a quote or a newline -- an
    ordinary thing to write -- produced a file `uv lock` could not parse, so the
    export failed on the user's prose rather than on anything about the
    scenario.
    """
    text = str(value)
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


#: Control characters a TOML basic string may not hold unescaped (tab, LF and
#: CR are handled by name above).
_TOML_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: A quote that, left alone, could close a ``"""`` docstring: one followed by
#: another quote, or the last character (the template's closing quotes follow).
_DOCSTRING_QUOTE = re.compile(r'"(?="|\Z)')

#: Control characters other than tab and newline, which Python source either
#: rejects (NUL) or silently rewrites (CR).
_PY_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def py_docstring(value: object) -> str:
    r"""Return *value* escaped for the body of a triple-quoted docstring.

    The escaped text goes between the template's own triple quotes and, once
    Python has read it, ``__doc__`` holds *value* again.  A description that
    contained three double quotes closed the docstring early, and one with a
    backslash -- a Windows path such as ``C:\Nishishinjuku`` -- was read as
    an escape (``\N{...}``), so the generated ``__init__.py`` did not compile.
    Only quotes that could close the docstring are escaped, so ordinary prose
    stays readable in the generated source.
    """
    text = str(value).replace("\\", "\\\\")
    text = _DOCSTRING_QUOTE.sub(r'\\"', text)
    return _PY_CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text)


def code_environment(templates_dir: str | Path) -> Environment:
    """Return the Jinja environment used to render generated source files."""
    environment = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,  # we render code and config, never HTML
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,  # fail loudly on a missing variable
    )
    environment.filters["toml_string"] = toml_string
    environment.filters["py_docstring"] = py_docstring
    return environment
