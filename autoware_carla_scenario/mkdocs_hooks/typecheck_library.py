"""MkDocs hook: fills in what docs/typecheck.md says of the library check (#789).

The page holds ``{{ library.<name> }}`` placeholders where it counts and lists
the ``CHECKED``, ``EXCLUDED`` and ``UNCALLED`` entries; this hook replaces them
with what ``autoware_carla_scenario.typecheck.library_docs`` generates from
``typecheck/library.py``, so a pull request that moves a module into
``CHECKED`` does not edit the page.  Other pages are left as they are.  A
placeholder that names no value fails the build.
"""

from __future__ import annotations

from typing import Any

from autoware_carla_scenario.typecheck.library_docs import (
    library_docs,
    render_library_docs,
)
from mkdocs.exceptions import PluginError

#: The page the placeholders are on, relative to docs_dir.
PAGE = "typecheck.md"


def on_page_markdown(markdown: str, page: Any, **kwargs: Any) -> str:
    if page.file.src_uri != PAGE:
        return markdown
    try:
        return render_library_docs(markdown, library_docs())
    except ValueError as error:
        raise PluginError(f"{PAGE}: {error}") from error
