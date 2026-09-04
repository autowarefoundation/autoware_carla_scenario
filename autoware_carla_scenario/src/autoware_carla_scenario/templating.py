"""Jinja environment shared by the code and config generators.

Both the scaffolder and the Scenario Package exporter render Python, TOML and
YAML rather than HTML, and both need the same strictness: no autoescaping, and
a loud failure on a variable the template asked for but the caller forgot.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

__all__ = ["code_environment", "toml_string"]


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
    return f'"{escaped}"'


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
    return environment
