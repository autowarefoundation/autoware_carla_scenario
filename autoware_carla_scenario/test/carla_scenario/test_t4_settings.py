"""t4_settings, split out of t4 so the library check compiles it (#731): t4
still exports every name it did, and the F-theta lens fit, rewritten without
numpy.polynomial (which Codon's NumPy lacks), returns what it did."""

from __future__ import annotations

from typing import Any

import itertools
import random
import re

import numpy as np
import pytest

from autoware_carla_scenario import t4, t4_settings
from autoware_carla_scenario.driver.base import (
    DriverCameraConfig,
    DriverCameraLens,
    DriverClientConfig,
)
from autoware_carla_scenario.t4_settings import T4Settings

from ._ftheta_lens import COSMOS3_FRONT_WIDE

#: What t4 exported before the split (its ``__all__``).
_T4_EXPORTS = {
    "DATASET_SUFFIX",
    "INSTALL_HINT",
    "LIDAR_CHANNEL",
    "T4Recorder",
    "T4Recording",
    "T4Settings",
    "T4UnavailableError",
    "active_recording",
    "camera_channel",
    "category_of",
    "dataset_path",
    "opencv_of_ftheta",
    "offer_driver_sensors",
    "require_writer",
    "withdraw_driver_sensors",
}


def test_t4_still_exports_every_name() -> None:
    assert set(t4.__all__) >= _T4_EXPORTS
    for name in _T4_EXPORTS:
        assert hasattr(t4, name), name


@pytest.mark.parametrize("name", t4_settings.__all__)
def test_t4_reexports_the_settings_module(name: str) -> None:
    assert name in t4.__all__
    assert getattr(t4, name) is getattr(t4_settings, name)


def _numpy_polynomial_fit(
    lens: DriverCameraLens, width: int, height: int, *, samples: int = 2000
) -> tuple[list[list[float]], list[float], float, float]:
    """t4.opencv_of_ftheta as it was before #731, on numpy.polynomial."""
    poly = np.polynomial.polynomial
    c, d, e = (float(v) for v in lens.linear_cde)
    if abs(e) > 1e-12:
        raise ValueError("an F-theta lens with linear_cde e != 0 has no OpenCV model")
    cx, cy = (float(v) for v in lens.principal_point)
    to_angle = np.asarray(lens.pixeldist_to_angle_poly, np.float64)
    to_radius = np.asarray(lens.angle_to_pixeldist_poly, np.float64)
    focal = float(poly.polyder(to_radius)[0])
    corners_x = np.array([-0.5, width - 0.5, -0.5, width - 0.5]) - cx
    corners_y = np.array([-0.5, -0.5, height - 0.5, height - 0.5]) - cy
    corner_radius = float(np.hypot((corners_x - d * corners_y) / c, corners_y).max())
    max_angle = min(float(lens.max_angle), np.radians(85.0))
    radius = min(corner_radius, float(poly.polyval(max_angle, to_radius)))
    reach = min(float(poly.polyval(radius, to_angle)), max_angle)
    theta = np.linspace(1e-4, reach, samples)
    r = np.tan(theta)
    r2 = r * r
    ratio = poly.polyval(theta, to_radius) / focal / r
    weight = (r * focal)[:, None]
    a = np.stack([r2, r2**2, r2**3, -ratio * r2, -ratio * r2**2, -ratio * r2**3], 1)
    k1, k2, k3, k4, k5, k6 = np.linalg.lstsq(
        a * weight, (ratio - 1.0) * weight[:, 0], rcond=None
    )[0]
    numerator = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    denominator = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    if not (denominator > 0.0).all():
        raise ValueError("no OpenCV rational model fits this F-theta lens")
    error = np.abs(focal * r * (numerator / denominator - ratio))
    intrinsic = [[c * focal, d * focal, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]
    distortion = [float(k) for k in (k1, k2, 0.0, 0.0, k3, k4, k5, k6)]
    return intrinsic, distortion, float(error.mean()), float(error.max())


#: Lenses, as changes to Cosmos3's front wide camera, and the image sizes they
#: are fitted at.
_LENSES: list[tuple[str, dict[str, Any], int, int, int]] = [
    ("cosmos3", {}, 832, 480, 2000),
    ("cosmos3-full-hd", {}, 1920, 1080, 2000),
    ("cosmos3-few-samples", {}, 832, 480, 50),
    ("narrow-max-angle", {"max_angle": 0.6}, 832, 480, 2000),
    (
        "column-skew",
        {"linear_cde": [0.998, 0.0125, 0.0]},
        832,
        480,
        2000,
    ),
    (
        "equidistant",
        {
            "pixeldist_to_angle_poly": [0.0, 1.0 / 500.0],
            "angle_to_pixeldist_poly": [0.0, 500.0],
            "max_angle": 1.4,
            "linear_cde": [1.0, 0.0, 0.0],
            "principal_point": [639.5, 359.5],
        },
        1280,
        720,
        2000,
    ),
    (
        "cubic",
        {
            "pixeldist_to_angle_poly": [0.0, 0.0031, 1.5e-7, -2.0e-10],
            "angle_to_pixeldist_poly": [0.0, 322.5, -4.1, -9.75],
            "max_angle": 1.05,
            "principal_point": [320.25, 241.75],
        },
        640,
        480,
        777,
    ),
    (
        "constant-term",
        {
            "pixeldist_to_angle_poly": [1e-6, 0.0024, -3.9e-8, 6.0e-10],
            "angle_to_pixeldist_poly": [0.02, 410.0, 2.5, -16.0, 10.0, -3.4],
        },
        832,
        480,
        2000,
    ),
]


@pytest.mark.parametrize(
    ("change", "width", "height", "samples"),
    [case[1:] for case in _LENSES],
    ids=[case[0] for case in _LENSES],
)
def test_the_lens_fit_is_the_numpy_polynomial_one(
    change: dict[str, Any], width: int, height: int, samples: int
) -> None:
    """The Horner helpers evaluate as numpy.polynomial.polynomial.polyval does
    (from the highest coefficient, starting at ``c[-1] + x * 0``) and
    ``polyder(c)[0]`` is ``1 * c[1]``, so every operation is the same and the
    fit is bit for bit the old one."""
    lens = DriverCameraLens.from_mapping({**COSMOS3_FRONT_WIDE, **change})
    try:
        expected = _numpy_polynomial_fit(lens, width, height, samples=samples)
    except ValueError as exc:
        with pytest.raises(ValueError, match=str(exc)):
            t4_settings.opencv_of_ftheta(lens, width, height, samples=samples)
        return
    # repr: bit for bit, the sign of a zero included.
    assert repr(t4_settings.opencv_of_ftheta(lens, width, height, samples=samples)) == (
        repr(expected)
    )


@pytest.mark.parametrize(
    "coef",
    [
        [3.0],
        [0.0, 411.286296],
        [0.5, -2.0, 0.25],
        COSMOS3_FRONT_WIDE["angle_to_pixeldist_poly"],
    ],
)
def test_the_horner_helpers_are_numpy_polyval(coef: list[float]) -> None:
    poly = np.polynomial.polynomial
    c = np.asarray(coef, np.float64)
    x = np.linspace(-1.5, 1.5, 101)
    for value in (*x[::10], 0.0, -0.0, 1.226783919):
        assert t4_settings._polyval_at(float(value), c) == float(poly.polyval(value, c))
    np.testing.assert_array_equal(t4_settings._polyval_on(x, c), poly.polyval(x, c))


#: Coefficients and arguments numpy answers with signed zeros, infinities and
#: NaNs.
_SPECIAL = [0.0, -0.0, 1.0, -2.5, 1e308, -1e-320, np.nan, np.inf, -np.inf]


def _bits(value: Any) -> bytes:
    """*value*'s bits, the sign of a zero included; a NaN's sign and payload,
    which the CPU picks, are not compared."""
    array = np.array(value, np.float64)
    array[np.isnan(array)] = np.nan
    return array.tobytes()


@pytest.mark.parametrize("length", [1, 2, 3])
def test_the_horner_helpers_are_numpy_polyval_bit_for_bit(length: int) -> None:
    poly = np.polynomial.polynomial
    x = np.asarray(_SPECIAL, np.float64)
    with np.errstate(all="ignore"):
        for coef in itertools.product(_SPECIAL, repeat=length):
            c = np.asarray(coef, np.float64)
            for value in (*_SPECIAL, np.float64(1.25)):
                assert _bits(t4_settings._polyval_at(value, c)) == _bits(
                    poly.polyval(value, c)
                ), (coef, value)
            on = t4_settings._polyval_on(x, c)
            assert on.dtype == np.float64
            assert _bits(on) == _bits(poly.polyval(x, c)), coef


@pytest.mark.parametrize("length", [1, 2, 3, 6])
def test_the_derivative_at_zero_is_numpy_polyder(length: int) -> None:
    poly = np.polynomial.polynomial
    with np.errstate(all="ignore"):
        for first in _SPECIAL:
            for second in _SPECIAL:
                c = np.asarray([first, second, 0.5, -1.0, 2.0, 3.0][:length])
                assert _bits(t4_settings._polyder_at_zero(c)) == _bits(
                    poly.polyder(c)[0]
                ), c


# ---------------------------------------------------------------------------
# A lens the dataset cannot hold (#1686): refused as the settings are built,
# with no T4 writer installed, not as every run's dataset starts.
# ---------------------------------------------------------------------------

#: An F-theta lens skewed in its rows (``linear_cde``'s e != 0): no OpenCV model.
_SKEWED_LENS = {**COSMOS3_FRONT_WIDE, "linear_cde": [1.0, 0.0, 0.01]}


def _lens_camera(logical_id: str) -> DriverCameraConfig:
    return DriverCameraConfig(
        logical_id=logical_id,
        image_width=832,
        image_height=480,
        lens=DriverCameraLens.from_mapping(_SKEWED_LENS),
    )


def test_settings_refuse_a_lens_the_dataset_cannot_hold() -> None:
    driver = DriverClientConfig(cameras=(_lens_camera("cam_wide"),))
    with pytest.raises(
        ValueError, match=r"camera 'cam_wide'.*linear_cde e != 0 has no OpenCV model"
    ):
        T4Settings.from_driver(driver, {})


def test_settings_do_not_check_a_lens_of_no_camera() -> None:
    # A lens whose logical id names no camera is not written: as before, it
    # is not refused.
    camera = DriverCameraConfig(logical_id="cam_front", image_width=64, image_height=48)
    lens = DriverCameraLens.from_mapping(_SKEWED_LENS)
    settings = T4Settings(
        cameras=(("cam_front", camera.to_sensor_config()),),
        lenses=(("cam_elsewhere", lens),),
    )
    assert settings.lenses == (("cam_elsewhere", lens),)


def test_a_run_refuses_a_lens_the_dataset_cannot_hold_before_it_starts() -> None:
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    from .test_video_removed import _shipped

    camera = {
        "logical_id": "cam_front",
        "image_width": 832,
        "image_height": 480,
        "lens": _SKEWED_LENS,
    }
    cfg = OmegaConf.merge(
        _shipped(["t4.enabled=true"]), {"driver": {"cameras": [camera]}}
    )
    with pytest.raises(
        run.RunConfigError, match=r"^t4: the lens of camera 'cam_front'"
    ):
        run.check_run_config(cfg)  # type: ignore[arg-type]


@pytest.mark.parametrize("block", ["t4", "t4scene"])
@pytest.mark.parametrize("off", ["false", "'false'", "no", "0", "OFF"])
def test_a_switch_spelled_off_as_text_writes_nothing(block: str, off: str) -> None:
    """``${oc.env:VAR,false}`` gives text: ``"false"`` is off, where
    ``bool("false")`` asked for the writer (and refused a run without the
    ``t4`` extra, T4UnavailableError) (#1753)."""
    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    from .test_video_removed import _shipped

    cfg = _shipped([f"{block}.enabled={off}"])
    assert run.build_t4_recorder(cfg) is None
    assert run.build_t4scene_exporter(cfg) is None
    run.check_run_config(cfg)


@pytest.mark.parametrize("block", ["t4", "t4scene"])
@pytest.mark.parametrize(
    ("bad", "shown"),
    [("ture", "'ture'"), ("2", "2"), ("'2'", "'2'"), ("0.0", "0.0")],
)
def test_a_switch_that_spells_neither_is_refused_in_one_keyed_line(
    block: str, bad: str, shown: str
) -> None:
    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    from .test_video_removed import _shipped

    cfg = _shipped([f"{block}.enabled={bad}"])
    message = rf"^{block}\.enabled must be true or false, got {re.escape(shown)}$"
    with pytest.raises(run.RunConfigError, match=message):
        run.check_run_config(cfg)  # type: ignore[arg-type]


_LIDAR_TOP = {"logical_id": "lidar_top"}


@pytest.mark.parametrize("off", ["false", "'false'", "no"])
def test_a_driver_ego_without_renderer_data_cannot_feed_the_dataset_its_lidar(
    off: str,
) -> None:
    """A driver entity attaches its LiDARs only for ``renderer_data``, and a
    T4 dataset of its run takes its sweeps from them: without it the dataset
    has no point cloud (no sample at all with no camera), refused as
    ``driver.t4_scene`` is (#2208)."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    from .test_video_removed import _shipped

    cfg = OmegaConf.merge(
        _shipped(
            [
                "ego.entity=carla_driver",
                "t4.enabled=true",
                f"driver.send_renderer_data={off}",
            ]
        ),
        {"driver": {"lidars": [_LIDAR_TOP]}},
    )
    message = (
        r"^t4\.enabled takes the LiDAR sweeps of an ego\.entity=carla_driver "
        r"from its renderer_data: driver\.lidars needs driver\.send_renderer_data$"
    )
    with pytest.raises(run.RunConfigError, match=message):
        run.check_run_config(cfg)  # type: ignore[arg-type]
    # A batch names the case refused, whichever it is.
    fine = _shipped(["ego.entity=carla_driver"])
    with pytest.raises(run.RunConfigError, match=r"^Bad: t4\.enabled takes"):
        run.check_batch_configs([fine, cfg], ["Fine", "Bad"])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "overrides",
    [
        # Renderer data on: the entity attaches the LiDARs and feeds them.
        ["ego.entity=carla_driver", "t4.enabled=true"],
        # No dataset.
        ["ego.entity=carla_driver", "driver.send_renderer_data=false"],
        # Another ego: the recording attaches the LiDARs itself.
        ["t4.enabled=true", "driver.send_renderer_data=false"],
    ],
)
def test_the_driver_lidars_feed_the_dataset_otherwise(overrides: list[str]) -> None:
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    from .test_video_removed import _shipped

    cfg = OmegaConf.merge(_shipped(overrides), {"driver": {"lidars": [_LIDAR_TOP]}})
    run.check_run_config(cfg)  # type: ignore[arg-type]


def test_a_driver_ego_with_cameras_only_needs_no_renderer_data() -> None:
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    from .test_video_removed import _shipped

    camera = {"logical_id": "cam_front", "image_width": 832, "image_height": 480}
    cfg = OmegaConf.merge(
        _shipped(
            [
                "ego.entity=carla_driver",
                "t4.enabled=true",
                "driver.send_renderer_data=false",
            ]
        ),
        {"driver": {"cameras": [camera]}},
    )
    run.check_run_config(cfg)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "module",
    [
        "autoware_carla_scenario.t4_settings",
        "autoware_carla_scenario.t4",
        "autoware_carla_scenario.entity.carla_driver_entity",
    ],
)
def test_importing_the_settings_loads_no_t4scene_converter(module: str) -> None:
    """The settings and t4 share the T4Scenes' distance check and point
    helpers (#2237) without importing the t4_scene package, whose converter
    (protobuf messages, Lanelet2) a driver entity or a config check never
    needs: t4 imports it only where a recording starts."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    code = (
        f"import sys, {module}; "
        "print('autoware_carla_scenario.t4_scene' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def _reference_fed_selection(
    frame: int,
    sweeps: list[int],
    images: list[tuple[int, str]],
    snapshots: set[int],
    start_frame: int,
    last_frame: int,
    camera_written: dict[str, int],
) -> tuple[list[int], list[int], int]:
    """T4Recording._record_fed's choice before #2355, as the kept sweeps' and
    camera frames' indices and the key frame (-1: nothing written).  #2362
    changed why a sweep is dropped (:func:`_reference_sweep_fate`), not which.
    """

    def is_new(taken_on: int) -> bool:
        return not (taken_on < start_frame or taken_on <= last_frame)

    def has_snapshot(taken_on: int) -> bool:
        return is_new(taken_on) and taken_on in snapshots

    def is_unwritten(taken_on: int, logical_id: str) -> bool:
        written = camera_written.get(logical_id)
        return not (written is not None and taken_on <= written)

    kept_sweeps = [i for i, t in enumerate(sweeps) if has_snapshot(t)]
    kept_images = [
        i for i, (t, cam) in enumerate(images) if is_new(t) and is_unwritten(t, cam)
    ]
    if not kept_sweeps and not kept_images:
        return [], [], -1
    if kept_sweeps:
        key_frame = min(sweeps[i] for i in kept_sweeps)
    else:
        key_frame = min(
            (images[i][0] for i in kept_images if images[i][0] in snapshots),
            default=frame,
        )
    if key_frame <= last_frame:
        return kept_sweeps, kept_images, -1
    on_key = [i for i in kept_sweeps if sweeps[i] == key_frame]
    return on_key, kept_images, key_frame


def _reference_sweep_fate(
    taken_on: int, snapshots: set[int], start_frame: int, last_frame: int
) -> int:
    """Why a sweep is dropped, as t4 logs it since #2362: not new (from before
    the recording, or not after the key frame written last -- the sweep a
    driver entity feeds again on every tick) before a missing snapshot."""
    if taken_on < start_frame or taken_on <= last_frame:
        return t4_settings.FED_NOT_NEW
    if taken_on not in snapshots:
        return t4_settings.FED_NO_SNAPSHOT
    return t4_settings.FED_KEPT


@pytest.mark.parametrize("seed", range(40))
def test_the_fed_selection_is_the_one_t4_made(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(200):
        frame = rng.randint(5, 30)
        start_frame = rng.randint(0, 10)
        last_frame = rng.choice([-1, rng.randint(0, frame - 1)])
        snapshots = {f for f in range(frame + 1) if rng.random() < 0.6} | {frame}
        sweeps = [rng.randint(0, frame) for _ in range(rng.randint(0, 3))]
        cams = ["front", "rear"]
        images = [
            (rng.randint(0, frame), rng.choice(cams)) for _ in range(rng.randint(0, 3))
        ]
        written = {c: rng.randint(0, frame) for c in cams if rng.random() < 0.5}
        sweep_fates, image_fates, key_frame = t4_settings.fed_key_frame(
            frame, sweeps, images, snapshots, start_frame, last_frame, written
        )
        old_sweeps, old_images, old_key = _reference_fed_selection(
            frame, sweeps, images, snapshots, start_frame, last_frame, written
        )
        assert key_frame == old_key
        # t4 logs that nothing can be written exactly when the old code kept
        # nothing.
        kept = t4_settings.FED_KEPT in sweep_fates + image_fates
        assert kept == bool(old_sweeps or old_images)
        if old_key >= 0:
            # And the count of later sweeps it logs is the old one.
            kept_sweeps = sum(
                not (t < start_frame or t <= last_frame or t not in snapshots)
                for t in sweeps
            )
            assert sweep_fates.count(t4_settings.FED_LATER) == kept_sweeps - len(
                old_sweeps
            )
            assert old_sweeps == [
                i for i, f in enumerate(sweep_fates) if f == t4_settings.FED_KEPT
            ]
            assert old_images == [
                i for i, f in enumerate(image_fates) if f == t4_settings.FED_KEPT
            ]
        # Why each dropped one was dropped: a sweep not new apart from one
        # whose frame has no snapshot (#2362), a camera frame as before.
        for t, fate in zip(sweeps, sweep_fates):
            expected = _reference_sweep_fate(t, snapshots, start_frame, last_frame)
            if expected == t4_settings.FED_KEPT:
                assert fate in (t4_settings.FED_KEPT, t4_settings.FED_LATER)
            else:
                assert fate == expected
        for (t, cam), fate in zip(images, image_fates):
            if t < start_frame or t <= last_frame:
                assert fate == t4_settings.FED_NOT_NEW
            elif cam in written and t <= written[cam]:
                assert fate == t4_settings.FED_REPEATED
            else:
                assert fate == t4_settings.FED_KEPT


def test_a_sweep_fed_again_after_its_key_frame_is_not_new() -> None:
    """A driver entity feeds the LiDAR's latest sweep on every tick: once its
    key frame is written, the sweep fed again is not new, not missing its
    snapshot (#2362), and nothing is written."""
    sweep_fates, image_fates, key_frame = t4_settings.fed_key_frame(
        103, [101], [], {99, 100, 101, 102, 103}, 100, 101, {}
    )
    assert sweep_fates == [t4_settings.FED_NOT_NEW]
    assert image_fates == []
    assert key_frame == -1


def test_a_sweep_from_before_the_recording_is_not_new() -> None:
    sweep_fates, _, key_frame = t4_settings.fed_key_frame(
        101, [99], [], {99, 100, 101}, 100, -1, {}
    )
    assert sweep_fates == [t4_settings.FED_NOT_NEW]
    assert key_frame == -1


def test_a_new_sweep_whose_frame_has_no_snapshot_has_none() -> None:
    """A sweep after the key frame written last whose frame the recording
    holds no snapshot of is the one logged as missing its snapshot; a camera
    frame fed with it is written on the step."""
    sweep_fates, image_fates, key_frame = t4_settings.fed_key_frame(
        107, [102], [(107, "front")], {104, 105, 106, 107}, 100, 101, {}
    )
    assert sweep_fates == [t4_settings.FED_NO_SNAPSHOT]
    assert image_fates == [t4_settings.FED_KEPT]
    assert key_frame == 107
