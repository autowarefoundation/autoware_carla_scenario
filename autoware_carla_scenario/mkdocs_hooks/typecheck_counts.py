"""MkDocs hook: fills in the library check's counts on docs/typecheck.md (#789).

The page holds ``{{ counts.<name> }}`` placeholders where it gives how many
modules are ``CHECKED``, ``EXCLUDED`` and ``UNCALLED``; this hook replaces
them with what ``autoware_carla_scenario.typecheck.doc_counts`` computes from
``typecheck/library.py``, so a pull request that moves a module into
``CHECKED`` does not edit the page.  Other pages are left as they are.  A
placeholder that names no count fails the build.
"""

from __future__ import annotations

from typing import Any

from mkdocs.exceptions import PluginError

from autoware_carla_scenario.typecheck.doc_counts import doc_counts, render_counts

#: The page the placeholders are on, relative to docs_dir.
PAGE = "typecheck.md"


def on_page_markdown(markdown: str, page: Any, **kwargs: Any) -> str:
    if page.file.src_uri != PAGE:
        return markdown
    try:
        return render_counts(markdown, doc_counts())
    except ValueError as error:
        raise PluginError(f"{PAGE}: {error}") from error
