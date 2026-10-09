"""Unit tests for :class:`CarlaCameraSensor` frame retrieval.

The draining behaviour is the subtle part: the sensor delivers frames faster than
the policy consumes them, so ``get_image`` must return the *newest* queued frame
rather than the oldest, or the caller is fed steadily staler images while the
bounded FIFO stays full and blocks the delivery callback.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest

from autoware_carla_scenario.sensor import carla_camera
from autoware_carla_scenario.sensor.carla_camera import (
    CarlaCameraSensor,
    CarlaCameraSensorConfig,
)


def _frame(value: int) -> SimpleNamespace:
    """A 1x1 BGRA frame whose single pixel encodes *value*, as a fake carla.Image."""
    return SimpleNamespace(
        width=1,
        height=1,
        raw_data=bytes([value, value, value, 255]),  # BGRA
    )


def _sensor() -> CarlaCameraSensor:
    return CarlaCameraSensor(CarlaCameraSensorConfig(image_width=1, image_height=1))


def test_get_image_returns_the_only_queued_frame() -> None:
    sensor = _sensor()
    sensor._frame_queue.put(_frame(7))
    image = sensor.get_image()
    assert image is not None
    assert image.shape == (1, 1, 3)
    assert int(image[0, 0, 0]) == 7


def test_get_image_drains_to_the_newest_frame() -> None:
    """Two frames arrive per policy step; the older ones must be dropped, not the
    newest one delivered late."""
    sensor = _sensor()
    sensor._frame_queue = queue.Queue()  # unbounded, to stage several frames
    for value in (1, 2, 3, 4):
        sensor._frame_queue.put(_frame(value))

    image = sensor.get_image()
    assert image is not None
    assert int(image[0, 0, 0]) == 4  # the newest
    assert sensor._frame_queue.empty()  # everything drained


def test_get_image_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # Shrink the blocking wait so an empty queue reports "no frame" quickly.
    monkeypatch.setattr(carla_camera, "_FRAME_TIMEOUT", 0.01)
    sensor = _sensor()
    assert sensor.get_image() is None


def test_get_image_produces_a_bgr_array() -> None:
    sensor = _sensor()
    sensor._frame_queue.put(_frame(5))
    image = sensor.get_image()
    assert image is not None
    assert image.dtype == np.uint8
    assert image.shape[2] == 3  # alpha stripped


# ---------------------------------------------------------------------------
# Blueprint attributes
# ---------------------------------------------------------------------------


def _attach(config: CarlaCameraSensorConfig) -> dict[str, str]:
    """Attach *config* to a fake world; the attributes set on its blueprint."""
    blueprint = SimpleNamespace(attributes={})
    blueprint.set_attribute = blueprint.attributes.__setitem__
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: blueprint),
        spawn_actor=lambda bp, transform, attach_to: SimpleNamespace(
            listen=lambda cb: None
        ),
    )
    CarlaCameraSensor(config).attach(
        world, SimpleNamespace(type_id="vehicle.lincoln.mkz")
    )
    return blueprint.attributes


def test_attach_leaves_exposure_and_optics_to_the_server() -> None:
    """Unset attributes keep the blueprint's defaults: CARLA 0.9.x values (iso 100,
    f/1.4, EV 7..9) pinned on a 0.10 camera render it black."""
    attributes = _attach(
        CarlaCameraSensorConfig(image_width=1920, image_height=1280, fov=50.0, fps=10.0)
    )
    assert attributes == {
        "image_size_x": "1920",
        "image_size_y": "1280",
        "fov": "50.0",
        "sensor_tick": "0.1",
    }


def test_attach_sets_the_attributes_a_config_overrides() -> None:
    attributes = _attach(
        CarlaCameraSensorConfig(iso=100.0, exposure_mode="manual", lens_k=0.5)
    )
    assert attributes["iso"] == "100.0"
    assert attributes["exposure_mode"] == "manual"
    assert attributes["lens_k"] == "0.5"
    assert "fstop" not in attributes


def test_every_blueprint_attribute_is_a_config_field() -> None:
    fields = set(CarlaCameraSensorConfig.__dataclass_fields__)
    assert set(carla_camera.BLUEPRINT_ATTRIBUTES) <= fields
