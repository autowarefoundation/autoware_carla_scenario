"""Every ``map.source`` override a doc or a config comment shows composes.

A git map source names its directory in the URL's fragment (``...@REF#PATH``),
and Hydra's override grammar refuses a ``#`` in an unquoted value: the shell's
own quotes are gone by the time Hydra reads the argument, so a copied
``map.source='git+...#maps/T'`` failed to parse before the run started.  The
docs and the config comments show it quoted for Hydra inside the shell's
quotes; each such argument is taken as the shell hands it over and composed as
a run composes it.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from autoware_carla_scenario.examples import run

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_CONF = _PACKAGE_ROOT / "src" / "autoware_carla_scenario" / "examples" / "conf"
_SOURCES = (
    _PACKAGE_ROOT / "docs" / "maps.md",
    _PACKAGE_ROOT / "docs" / "usage.md",
    _CONF / "config.yaml",
    _CONF / "map" / "town10hd_opt.yaml",
)
#: A shell word that sets ``map.source`` to a value with a URL fragment.
_OVERRIDE = re.compile(r"""(["']?map\.source=\S*#\S*)""")


def _arguments() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in _SOURCES:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for match in _OVERRIDE.finditer(line.lstrip("# ")):
                # What the shell passes on: its quotes removed.
                (word,) = shlex.split(match.group(1))
                found.append((f"{path.name}:{number}", word))
    return found


def test_the_docs_show_a_map_source_override() -> None:
    # maps.md, usage.md and config.yaml; town10hd_opt.yaml's placeholder
    # (``<other commit>``) holds a space and is not a value to compose.
    assert len(_arguments()) >= 3


@pytest.mark.parametrize(("where", "argument"), _arguments())
def test_a_documented_map_source_override_composes(where: str, argument: str) -> None:
    cfg = run._compose_config("lane_change/left", ["map=town10hd_opt", argument])
    assert "#" in str(cfg.map.source), where
