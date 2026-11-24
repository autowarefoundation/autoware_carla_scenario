"""Where the trajectory log says a run's map came from (no CARLA server)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.examples import run
from autoware_carla_scenario.maps import MapPaths

#: What resolve_map_paths gives for a config naming no pack.
NO_PACK = MapPaths()


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
    assert run._map_source(cfg, NO_PACK) == "/maps/T"


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
    assert run._map_source(OmegaConf.create({"map": map_cfg}), NO_PACK) == expected


def test_a_config_without_a_map_names_none() -> None:
    assert run._map_source(OmegaConf.create({}), NO_PACK) is None


_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _from_source(ref: str, commit: str) -> MapPaths:
    """What resolve_map_paths gives for a map read from a source at *ref*,
    checked out at *commit*."""
    from types import SimpleNamespace  # noqa: PLC0415

    resolved = SimpleNamespace(source=SimpleNamespace(ref=ref), commit=commit)
    return MapPaths(resolved=resolved)  # type: ignore[arg-type]


def test_a_branch_is_named_with_the_commit_it_resolved_to() -> None:
    # Each task of a batch resolves a branch again: the log says which commit
    # the run read, beside the configured ref (#2446).
    source = "git+https://example.com/o/maps@main#T"
    cfg = OmegaConf.create({"map": {"source": source}})
    assert run._map_source(cfg, _from_source("main", _COMMIT)) == (
        f"{source} (commit {_COMMIT})"
    )


def test_a_pinned_source_is_named_as_it_is() -> None:
    source = f"git+https://example.com/o/maps@{_COMMIT}#T"
    cfg = OmegaConf.create({"map": {"source": source}})
    assert run._map_source(cfg, _from_source(_COMMIT, _COMMIT)) == source


def test_a_source_found_unpacked_names_no_commit() -> None:
    # A provisioned directory records no revision: there is none to name.
    source = "git+https://example.com/o/maps@main#T"
    cfg = OmegaConf.create({"map": {"source": source}})
    assert run._map_source(cfg, _from_source("main", "")) == source
