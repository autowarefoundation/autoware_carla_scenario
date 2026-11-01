"""Unit tests for :class:`CarlaCameraSensor` frame retrieval.

The draining behaviour is the subtle part: the sensor delivers frames faster than
the policy consumes them, so ``get_image`` must return the *newest* queued frame
rather than the oldest, or the caller is fed steadily staler images while the
bounded FIFO stays full and blocks the delivery callback.
"""

from __future__ import annotations

import ast
import inspect
import queue
import threading
import time
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


def _attach(
    config: CarlaCameraSensorConfig, missing: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Attach *config* to a fake world whose camera lacks *missing*; the
    attributes set on its blueprint."""
    blueprint = SimpleNamespace(attributes={})
    blueprint.has_attribute = lambda name: name not in missing

    def set_attribute(name: str, value: str) -> None:
        if name in missing:  # what CARLA does: IndexError
            raise IndexError(f"blueprint has no attribute {name!r}")
        blueprint.attributes[name] = value

    blueprint.set_attribute = set_attribute
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
        # Just under the 0.1 s interval, so two ticks' float deltas reach it.
        "sensor_tick": "0.099999",
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


def test_an_attribute_the_server_lacks_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CARLA's nightly camera has no bloom_intensity: a config naming it still attaches."""
    attributes = _attach(
        CarlaCameraSensorConfig(bloom_intensity=0.675, iso=100.0),
        missing=frozenset({"bloom_intensity"}),
    )
    assert "bloom_intensity" not in attributes
    assert attributes["iso"] == "100.0"
    assert "bloom_intensity" in caplog.text


def test_the_attributes_spelled_out_for_codon_are_the_blueprint_attributes() -> None:
    """attach() names each attribute in its `if TYPE_CHECKING:` branch, which
    Codon compiles: the same ones, in the same order, as the loop Python runs."""
    tree = ast.parse(inspect.getsource(CarlaCameraSensor.attach).strip())
    spelled = [
        (call.args[1].value, ast.unparse(call.args[2]))
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "_override"
        and isinstance(call.args[1], ast.Constant)
    ]
    # Each with the config's value of the attribute it names.
    assert spelled == [
        (name, f"cfg.{name}") for name in carla_camera.BLUEPRINT_ATTRIBUTES
    ]


def test_the_listener_queues_the_frame_it_is_handed() -> None:
    sensor = _sensor()
    frame = _frame(3)
    sensor._on_sensor_data(frame)  # type: ignore[arg-type]
    assert sensor._frame_queue.get_nowait() is frame
    # typesafe_carla's Codon library hands over the SensorData instead.
    data = SimpleNamespace(as_image=lambda: frame)
    sensor._on_sensor_data(data)  # type: ignore[arg-type]
    assert sensor._frame_queue.get_nowait() is frame


def test_the_listener_hands_the_frame_to_on_frame_instead_of_queueing_it() -> None:
    seen: list[object] = []
    sensor = CarlaCameraSensor(
        CarlaCameraSensorConfig(image_width=1, image_height=1), seen.append
    )
    frame = _frame(3)
    sensor._on_sensor_data(frame)  # type: ignore[arg-type]
    sensor._on_sensor_data(SimpleNamespace(as_image=lambda: frame))  # type: ignore[arg-type]
    assert seen == [frame, frame]
    assert sensor._frame_queue.empty()


def test_the_listener_drops_the_oldest_frame_instead_of_blocking() -> None:
    """Nothing drains the queue once a scenario ends; a blocking put would hang
    CARLA's delivery thread, and the replay's recorder would get no frames."""
    sensor = _sensor()

    def deliver() -> None:
        for value in (1, 2, 3, 4):
            sensor._on_sensor_data(_frame(value))  # type: ignore[arg-type]

    delivery = threading.Thread(target=deliver, daemon=True)
    delivery.start()
    delivery.join(timeout=5.0)
    assert not delivery.is_alive(), "the listener blocked on a full queue"
    image = sensor.get_image()
    assert image is not None
    assert int(image[0, 0, 0]) == 4
    assert sensor._frame_queue.empty()


# ---------------------------------------------------------------------------
# Frame timing (#951)
# ---------------------------------------------------------------------------


def _rendered(frame: int, value: int = 0) -> SimpleNamespace:
    """A 1x1 frame as a fake carla.Image rendered on simulator frame *frame*."""
    image = _frame(value)
    image.frame = frame
    return image


@pytest.mark.parametrize(
    ("fps", "interval_ticks"), [(5.0, 4), (7.0, 3), (10.0, 2), (20.0, 1), (30.0, 1)]
)
def test_sensor_tick_has_the_margin_when_slower_than_a_tick(
    fps: float, interval_ticks: int
) -> None:
    sensor_tick = float(
        _attach(CarlaCameraSensorConfig(image_width=1, image_height=1, fps=fps))[
            "sensor_tick"
        ]
    )
    if interval_ticks > 1:
        assert sensor_tick == pytest.approx(1.0 / fps - 1e-6, abs=1e-12)
        assert sensor_tick < 1.0 / fps
    else:  # every tick anyway: left as it is
        assert sensor_tick == pytest.approx(1.0 / fps, abs=1e-12)
    sensor = CarlaCameraSensor(
        CarlaCameraSensorConfig(image_width=1, image_height=1, fps=fps)
    )
    assert sensor.frame_interval_ticks == interval_ticks


@pytest.mark.parametrize(("fps", "interval_ticks"), [(5.0, 4), (10.0, 2)])
@pytest.mark.parametrize("phase", [0, 1])
def test_latest_image_never_waits_when_no_frame_is_due(
    monkeypatch: pytest.MonkeyPatch, fps: float, interval_ticks: int, phase: int
) -> None:
    """A 10 Hz policy (a step every 2nd tick) against a camera at *fps*: no
    step waits, since every frame due has come; each answer is the newest
    frame, with the tick it was rendered on."""
    waits: list[float] = []
    sensor = CarlaCameraSensor(
        CarlaCameraSensorConfig(image_width=1, image_height=1, fps=fps)
    )
    real_get = sensor._frame_queue.get

    def get(block: bool = True, timeout: float | None = None) -> object:
        if block:
            waits.append(timeout if timeout is not None else float("inf"))
        return real_get(block, timeout)

    monkeypatch.setattr(sensor._frame_queue, "get", get)
    newest = None
    for tick in range(1, 25):
        if (tick - phase) % interval_ticks == 0:
            sensor._on_sensor_data(_rendered(tick, tick))  # type: ignore[arg-type]
            newest = tick
        if tick % 2 == 0:
            started = time.monotonic()
            latest = sensor.latest_image(tick)
            assert time.monotonic() - started < 0.5
            if newest is None:
                assert latest is None
            else:
                assert latest is not None
                assert latest[0] == newest
                assert int(latest[1][0, 0, 0]) == newest
    assert waits == []


def test_latest_image_waits_for_a_frame_that_is_due(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame due by this tick is still on its way from CARLA's thread: wait."""
    sensor = CarlaCameraSensor(
        CarlaCameraSensorConfig(image_width=1, image_height=1, fps=10.0)
    )
    sensor._on_sensor_data(_rendered(10, 1))  # type: ignore[arg-type]
    assert sensor.latest_image(10) is not None
    timer = threading.Timer(
        0.05,
        sensor._on_sensor_data,
        (_rendered(12, 2),),
    )
    timer.start()
    try:
        latest = sensor.latest_image(12)
    finally:
        timer.join()
    assert latest is not None
    assert latest[0] == 12


def test_a_due_frame_that_never_comes_costs_one_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(carla_camera, "_FRAME_TIMEOUT", 0.01)
    sensor = CarlaCameraSensor(
        CarlaCameraSensorConfig(image_width=1, image_height=1, fps=10.0)
    )
    sensor._on_sensor_data(_rendered(10))  # type: ignore[arg-type]
    assert sensor.latest_image(10) is not None
    # Nothing on 12: one wait, then the stale frame is not handed out ...
    assert sensor.latest_image(12) is None
    # ... and the next tick does not wait for it again.
    monkeypatch.setattr(carla_camera, "_FRAME_TIMEOUT", 60.0)
    started = time.monotonic()
    assert sensor.latest_image(13) is None
    assert time.monotonic() - started < 0.5


def test_latest_image_keeps_the_newest_frame_however_they_arrive() -> None:
    sensor = _sensor()
    sensor._frame_queue = queue.Queue()
    for frame in (5, 7, 6):
        sensor._frame_queue.put(_rendered(frame, frame))
    latest = sensor.latest_image(7)
    assert latest is not None
    assert latest[0] == 7
    assert int(latest[1][0, 0, 0]) == 7
    assert latest[1].shape == (1, 1, 3)
