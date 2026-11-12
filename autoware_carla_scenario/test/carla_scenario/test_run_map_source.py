"""Where the trajectory log says a run's map came from (no CARLA server)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.examples import run


def test_a_directory_wins_over_a_source_as_the_resolver_lets_it() -> None:
    # resolve_map_paths reads map.directory and ignores map.source when both
    # are set; the log's map record named the source, a map the run never
    # read (#2085).
    cfg = OmegaConf.create(
        {
            "map": {
                "source": "git+https://example.com/o/maps@v1#T",
                "directory": "/maps/T",
            }
        }
    )
    assert run._map_source(cfg) == "/maps/T"


@pytest.mark.parametrize(
    ("map_cfg", "expected"),
    [
        (
            {"source": "git+https://example.com/o/maps@v1#T"},
            "git+https://example.com/o/maps@v1#T",
        ),
        ({"directory": "/maps/T"}, "/maps/T"),
        (
            {"source": "git+https://example.com/o/maps@v1#T", "directory": None},
            "git+https://example.com/o/maps@v1#T",
        ),
        ({"source": "", "directory": "/maps/T"}, "/maps/T"),
        ({}, None),
    ],
)
def test_either_alone_is_named(
    map_cfg: dict[str, object], expected: str | None
) -> None:
    assert run._map_source(OmegaConf.create({"map": map_cfg})) == expected


def test_a_config_without_a_map_names_none() -> None:
    assert run._map_source(OmegaConf.create({})) is None
