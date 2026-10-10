"""#34: per-tick ODD samples in the coverage file, exported as an OpenODD COD."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from autoware_carla_scenario.coverage import CoverageCollector
from autoware_carla_scenario.coverage.cod import export_cod
from autoware_carla_scenario.coverage.report import main
from autoware_carla_scenario.odd import (
    OddAttribute,
    OddDefinition,
    load_openodd,
    probes,
)


def _odd() -> OddDefinition:
    return OddDefinition(
        "o",
        [
            OddAttribute(
                "dynamic.speed", lambda w: w.speed, unit="km/h", buckets=[0, 50, 100]
            ),
            OddAttribute("scenery.lanes", lambda w: w.lanes, values=[1, 2, 3]),
            OddAttribute(
                "environment.rain", lambda w: w.rain, values=["none", "heavy"]
            ),
            OddAttribute(
                "scenery.junction", lambda w: w.junction, values=[False, True]
            ),
        ],
    )


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A run of three ticks; the second has no ego position, the third no rain."""
    positions = iter([(35.6, 139.7), None, (35.61, 139.71)])
    monkeypatch.setattr(probes, "ego_geolocation", lambda w: next(positions))
    collector = CoverageCollector([], odd=_odd())
    world = SimpleNamespace(speed=36.5, lanes=2, rain="none", junction=False)
    collector.start(world, 0.0)
    collector.tick(world, 0.05)
    collector.tick(world, 0.1)
    world.rain, world.speed = None, float("nan")
    collector.tick(world, 0.15)
    return collector.to_dict("S")


def test_the_coverage_file_keeps_every_tick(run: dict[str, Any]) -> None:
    samples = run["odd"]["samples"]
    assert [a["name"] for a in samples["attributes"]] == [
        "dynamic.speed",
        "scenery.lanes",
        "environment.rain",
        "scenery.junction",
    ]
    assert samples["attributes"][0]["values"] is None
    assert samples["attributes"][2]["values"] == ["none", "heavy"]
    assert samples["rows"][0] == [0.05, 35.6, 139.7, 36.5, 2, "none", False]
    assert samples["rows"][1][1:3] == [None, None]
    assert samples["rows"][2][3] is None  # NaN is no value
    assert run["odd"]["started_at"].endswith("+00:00")
    json.dumps(run)  # stays JSON


def test_the_cod_follows_openodd(run: dict[str, Any], tmp_path: Path) -> None:
    result = export_cod(run, tmp_path, "S")
    assert result is not None and result.rows == 2
    with result.cod.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [
        "TEMPORAL_EXTENT",
        "SPATIAL_EXTENT",
        "dynamic.speed;km/h",
        "scenery.lanes",
        "environment.rain",
        "scenery.junction",
    ]
    assert rows[1][1:] == ["35.6000000 139.7000000", "36.5", "2", "none", "false"]
    assert rows[2][1:] == ["35.6100000 139.7100000", "", "2", "", "false"]
    started = run["odd"]["started_at"][:10]
    assert rows[1][0].startswith(started) and rows[1][0].count(":") == 2

    with result.manifest.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert manifest[0]["TYPE"] == "taxonomy"
    assert manifest[0]["FILE_NAME"] == "S_taxonomy.yml"
    speed = manifest[1]
    assert (speed["COLUMN"], speed["TAXONOMY_ID"], speed["UNIT"]) == (
        "dynamic.speed;km/h",
        "speed",
        "km/h",
    )

    taxonomy = yaml.safe_load(result.taxonomy.read_text())
    assert taxonomy["TAXONOMY"] == {
        "dynamic": {"speed": "float velocity"},
        "scenery": {"lanes": "integer count", "junction": "boolean"},
        "environment": {"rain": ["none", "heavy"]},
    }
    # The taxonomy is OpenODD the reader takes.
    odd = load_openodd(result.taxonomy)
    assert {a.name for a in odd.attributes} == {
        "dynamic.speed",
        "scenery.lanes",
        "scenery.junction",
        "environment.rain",
    }


def test_a_file_without_samples_is_not_exported(tmp_path: Path) -> None:
    assert export_cod({"odd": None}, tmp_path, "S") is None
    assert export_cod({"odd": {"name": "o"}}, tmp_path, "S") is None


def test_the_cli_exports_every_run(
    run: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for sub in ("a", "b"):
        (tmp_path / "runs" / sub).mkdir(parents=True)
        (tmp_path / "runs" / sub / "S_coverage.json").write_text(json.dumps(run))
    out = tmp_path / "cod"
    assert main([str(tmp_path / "runs"), "--export-cod", str(out)]) == 0
    assert sorted(p.name for p in out.glob("*_cod.csv")) == ["S-2_cod.csv", "S_cod.csv"]
    assert "Exported 2 COD table(s)" in capsys.readouterr().err
