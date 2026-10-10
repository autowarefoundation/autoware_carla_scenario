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


def _doc(attributes: list[dict[str, Any]], rows: list[list[Any]]) -> dict[str, Any]:
    return {
        "odd": {
            "started_at": "2026-01-01T00:00:00.000Z",
            "samples": {"attributes": attributes, "rows": rows},
        }
    }


def test_a_measure_of_an_element_keeps_both(tmp_path: Path) -> None:
    # OpenODD writes a measure of an element as <element>.<measure> (10.2.2.5).
    for order in ([0, 1], [1, 0]):
        metas: list[dict[str, Any]] = [
            {"name": "env.rain", "unit": "mm/h", "values": None},
            {"name": "env.rain.duration", "unit": "s", "values": None},
        ]
        metas = [metas[i] for i in order]
        result = export_cod(_doc(metas, [[0.1, 1.0, 2.0, 3.0, 4.0]]), tmp_path, "S")
        assert result is not None
        taxonomy = yaml.safe_load(result.taxonomy.read_text())["TAXONOMY"]
        assert taxonomy["env"]["rain"] == "float precipitation_rate"
        assert taxonomy["env"]["rain.duration"] == "float time"
        names = {a.name for a in load_openodd(result.taxonomy).attributes}
        assert names == {"env.rain", "env.rain.duration"}


def test_every_literal_is_in_the_taxonomy(tmp_path: Path) -> None:
    metas: list[dict[str, Any]] = [
        {"name": "road", "unit": "", "values": ["RQ28", "RQ43-5", "two lane"]},
        {"name": "free", "unit": "", "values": None},
    ]
    rows = [[0.1, 1.0, 2.0, "RQ43-5", "b"], [0.2, 1.0, 2.0, "RQ28", "a"]]
    result = export_cod(_doc(metas, rows), tmp_path, "S")
    assert result is not None
    taxonomy = yaml.safe_load(result.taxonomy.read_text())["TAXONOMY"]
    assert taxonomy["road"] == ["RQ28", "RQ43-5", "two lane"]
    assert taxonomy["free"] == ["a", "b"]
    assert result.cod.read_text().splitlines()[1].startswith("2026-01-01 00:00:00.100")


def test_numpy_values_keep_their_type() -> None:
    np = pytest.importorskip("numpy")
    from autoware_carla_scenario.coverage.collector import _sample_value

    assert _sample_value(np.int64(3)) == 3 and isinstance(
        _sample_value(np.int64(3)), int
    )
    assert _sample_value(np.bool_(True)) is True
    assert _sample_value(np.float32(0.5)) == 0.5
    assert _sample_value(np.float64("nan")) is None


def test_the_cli_fails_when_nothing_was_exported(tmp_path: Path) -> None:
    doc = {"schema": "autoware_carla_scenario.coverage/1", "scenario": "S", "items": []}
    (tmp_path / "S_coverage.json").write_text(json.dumps(doc))
    assert main([str(tmp_path), "--export-cod", str(tmp_path / "cod")]) == 1
