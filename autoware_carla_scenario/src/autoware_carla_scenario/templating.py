"""Jinja environment shared by the code and config generators.

Both the scaffolder and the Scenario Package exporter render Python, TOML and
YAML rather than HTML, and both need the same strictness: no autoescaping, and
a loud failure on a variable the template asked for but the caller forgot.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .text_escape import py_docstring, toml_string

__all__ = ["code_environment", "py_docstring", "toml_string"]


def _toml_string_filter(value: object) -> str:
    """The ``toml_string`` filter: :func:`toml_string` of ``str(value)``."""
    return toml_string(str(value))


def _py_docstring_filter(value: object) -> str:
    """The ``py_docstring`` filter: :func:`py_docstring` of ``str(value)``."""
    return py_docstring(str(value))


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
    environment.filters["toml_string"] = _toml_string_filter
    environment.filters["py_docstring"] = _py_docstring_filter
    return environment
