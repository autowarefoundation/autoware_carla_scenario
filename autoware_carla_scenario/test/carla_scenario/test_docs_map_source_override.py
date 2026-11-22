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
_DOCS = _PACKAGE_ROOT / "docs"
_CONF = _PACKAGE_ROOT / "src" / "autoware_carla_scenario" / "examples" / "conf"
#: The files known to show such an override; each must still show one.
_EXPECTED = (
    _DOCS / "maps.md",
    _DOCS / "usage.md",
    _CONF / "config.yaml",
    _CONF / "map" / "town10hd_opt.yaml",
)
#: Every file scanned: the docs and the example configs.
_SOURCES = sorted({*_DOCS.rglob("*.md"), *_CONF.rglob("*.yaml"), *_EXPECTED})
#: A shell word that sets ``map.source`` to a value with a URL fragment.
_OVERRIDE = re.compile(r"""(["']?map\.source=\S*#\S*)""")


def _arguments() -> list[tuple[Path, int, str]]:
    found: list[tuple[Path, int, str]] = []
    for path in _SOURCES:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for match in _OVERRIDE.finditer(line):
                # What the shell passes on: its quotes removed.
                (word,) = shlex.split(match.group(1))
                found.append((path, number, word))
    return found


_ARGUMENTS = _arguments()


def test_each_known_file_shows_a_map_source_override() -> None:
    shown = {path for path, _, _ in _ARGUMENTS}
    missing = [
        str(path.relative_to(_PACKAGE_ROOT)) for path in _EXPECTED if path not in shown
    ]
    assert not missing, f"no map.source override with a '#' found in {missing}"


@pytest.mark.parametrize(
    "argument",
    [word for _, _, word in _ARGUMENTS],
    ids=[f"{path.name}:{number}" for path, number, _ in _ARGUMENTS],
)
def test_a_documented_map_source_override_composes(argument: str) -> None:
    value = argument.partition("=")[2]
    # The value is quoted for Hydra; Hydra drops those quotes and keeps the rest.
    assert len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"", value
    cfg = run._compose_config("lane_change/left", ["map=town10hd_opt", argument])
    assert cfg.map.source == value[1:-1]
