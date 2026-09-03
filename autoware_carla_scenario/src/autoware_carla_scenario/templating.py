"""Jinja environment shared by the code and config generators.

Both the scaffolder and the Scenario Package exporter render Python, TOML and
YAML rather than HTML, and both need the same strictness: no autoescaping, and
a loud failure on a variable the template asked for but the caller forgot.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

__all__ = ["code_environment"]


def code_environment(templates_dir: str | Path) -> Environment:
    """Return the Jinja environment used to render generated source files."""
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,  # we render code and config, never HTML
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,  # fail loudly on a missing variable
    )
