"""The video is gone (15.0.0): what an old config's video.* gets, and what
the viewer shows instead."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import autoware_carla_scenario
from autoware_carla_scenario.examples.run import warn_if_video_set
from autoware_carla_scenario.ui import scanner


def _shipped(overrides: list[str]):
    from hydra import compose, initialize_config_dir

    from autoware_carla_scenario.examples import conf as conf_package

    conf_dir = str(Path(conf_package.__file__).parent.resolve())
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_the_shipped_config_sets_no_video(capsys) -> None:
    assert not warn_if_video_set(_shipped([]))
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "override",
    ["video.enabled=false", "video.enabled=true", "video.width=960"],
)
def test_an_old_video_setting_is_accepted_and_told_it_does_nothing(
    override: str, capsys
) -> None:
    # Hydra takes the old keys (no struct error), and the run says what to use.
    cfg = _shipped([override])
    assert warn_if_video_set(cfg)
    err = capsys.readouterr().err
    assert override.lower() in err.lower()
    assert "pip install 'autoware-carla-scenario[rerun,viewer]'" in err
    assert "rerun.enabled=true" in err
    assert "python -m autoware_carla_scenario.rerun" in err
    assert "Rerun viewer" in err


def test_a_batch_warns_once_whichever_case_sets_it(capsys) -> None:
    plain, setting = _shipped([]), _shipped(["video.height=540"])
    assert warn_if_video_set(plain, setting, setting)
    err = capsys.readouterr().err
    assert err.count("Warning: video.* is set") == 1
    assert "video.height=540" in err
    assert not warn_if_video_set(plain, plain)


def test_a_config_without_a_video_section_is_fine() -> None:
    assert not warn_if_video_set(OmegaConf.create({"server": {"port": 2000}}))


def test_the_video_api_is_gone() -> None:
    assert not hasattr(autoware_carla_scenario, "CameraRecorder")
    with pytest.raises(ImportError):
        __import__("autoware_carla_scenario.camera_recorder")
    with pytest.raises(ImportError):
        __import__("autoware_carla_scenario.video_render")


def test_the_viewer_points_at_the_trajectory_log(tmp_path: Path) -> None:
    job = tmp_path / "outputs" / "2026-10-18" / "10-00-00"
    job.mkdir(parents=True)
    (job / "CutInScenario_result.json").write_text(
        json.dumps({"passed": True, "message": "ok", "elapsed_seconds": 1.0})
    )
    (job / "CutInScenario.trajectory.jsonl").write_text("{}\n")
    scanner.clear_cache()
    result = scanner.load_scenario(tmp_path, "single", "2026-10-18", "10-00-00", 0)
    assert result is not None
    assert result.trajectory_path == str(
        (job / "CutInScenario.trajectory.jsonl").resolve()
    )
    assert not hasattr(result, "video_filename")
