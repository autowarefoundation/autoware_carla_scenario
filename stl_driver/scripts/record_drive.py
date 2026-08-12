#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Record a CARLA rollout from a chase camera, with the specification on screen.

Drives a real CARLA server with :class:`~stl_driver.policy.EgoPolicy` and
films it from a camera rigged behind the ego, overlaying each rule's live
robustness on every frame.  The point is to see the specification and the
behaviour at the same time: when the vehicle slows, ``stop_on_red`` or
``collision_free`` is visibly the number going to zero.

    docker compose up -d carla
    uv run --extra carla python scripts/record_drive.py --steps 600 --traffic 30

Green means a rule holds with room to spare, amber that it is close to its
bound, red that it is violated.  The bar under each name is the robustness
scaled against that rule's margin.
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from carla_driver_interface.driver import serving
from carla_driver_interface.runtime import RuntimeConfig, ScenarioSpec
from carla_driver_interface.runtime.carla_runtime import CarlaRuntime
from carla_driver_interface.runtime.carla_world import CarlaWorldAdapter
from PIL import Image, ImageDraw, ImageFont

from stl_driver.config import STLDriverConfig
from stl_driver.policy import EgoPolicy

logger = logging.getLogger("record_drive")

# Chase camera: behind and above the ego, looking slightly down.
CAMERA_OFFSET = (-6.5, 0.0, 3.0)
CAMERA_PITCH_DEG = -12.0

PANEL_WIDTH = 344
PANEL_ALPHA = 195

GREEN = (120, 220, 140)
AMBER = (240, 200, 90)
RED = (240, 110, 110)
GREY = (150, 150, 160)


class RecordingWorld(CarlaWorldAdapter):
    """A CARLA world that also films the ego.

    The camera is registered in ``self._sensors`` so the adapter's own
    teardown destroys it, and its frames are drained once per tick -- in
    synchronous mode the server produces exactly one image per sensor per
    tick, so the queue stays shallow and the film stays in step with the
    simulation.
    """

    def __init__(self, *args: Any, width: int = 1280, height: int = 720, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._film_size = (width, height)
        self._film: queue.Queue = queue.Queue()
        self.frames: list[np.ndarray] = []

    def setup(self):
        result = super().setup()
        import carla

        blueprint = self._world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(self._film_size[0]))
        blueprint.set_attribute("image_size_y", str(self._film_size[1]))
        blueprint.set_attribute("fov", "90")
        transform = carla.Transform(
            carla.Location(x=CAMERA_OFFSET[0], y=CAMERA_OFFSET[1], z=CAMERA_OFFSET[2]),
            carla.Rotation(pitch=CAMERA_PITCH_DEG),
        )
        camera = self._world.spawn_actor(blueprint, transform, attach_to=self._ego)
        camera.listen(self._film.put)
        # Handing it to the adapter's own sensor list means teardown -- which
        # now stops sensors before destroying anything -- covers it too.
        self._sensors.append(camera)
        logger.info("chase camera attached: %dx%d", *self._film_size)
        logger.info(
            "spawn point %s -- pass --spawn-point to pin it, or two recordings "
            "are of different roads and not comparable",
            self._spawn_point_index(),
        )
        return result

    def _spawn_point_index(self) -> int | str:
        """Which of the map's spawn points the ego actually started from.

        Left unset, the runtime draws one from its seeded generator, which is
        reproducible only as long as nothing upstream of the draw changes how
        much of that generator it consumes -- so a run recorded today and one
        recorded last week can silently be of different roads. Recovering the
        index by position means a recording always says where it began, and
        `--spawn-point` can pin it.
        """
        if self.scenario.spawn_point_index is not None:
            return self.scenario.spawn_point_index
        here = self._ego.get_transform().location
        points = self._map.get_spawn_points()
        if not points:  # pragma: no cover - guarded upstream at spawn time
            return "unknown"
        distances = [here.distance(point.location) for point in points]
        return int(min(range(len(points)), key=distances.__getitem__))

    def tick(self):
        snapshot = super().tick()
        try:
            image = self._film.get(timeout=2.0)
        except queue.Empty:  # pragma: no cover - a dropped frame is not fatal
            return snapshot
        buffer = np.frombuffer(image.raw_data, dtype=np.uint8)
        # CARLA delivers BGRA; drop alpha and flip to RGB.
        self.frames.append(
            buffer.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1].copy()
        )
        return snapshot


class SilentWorld(RecordingWorld):
    """The same drive with no camera on it.

    Tuning wants the violation table, not the film, and the film is most of
    the cost: a seven-hundred-step drive spends longer drawing overlays and
    encoding them than it does driving. Attaching no camera also takes the
    rendering load off the simulator.
    """

    def setup(self):
        # Skip `RecordingWorld.setup`, which is the part that attaches one.
        result = super(RecordingWorld, self).setup()
        logger.info(
            "spawn point %s -- pass --spawn-point to pin it, or two runs are "
            "of different roads and not comparable",
            self._spawn_point_index(),
        )
        return result

    def tick(self):
        return super(RecordingWorld, self).tick()


def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def colour_for(value: float, margin: float) -> tuple[int, int, int]:
    if value < 0.0:
        return RED
    if value < 0.25 * margin:
        return AMBER
    return GREEN


def draw_overlay(
    frame: np.ndarray,
    scalars: dict[str, float],
    rules,
    step: int,
    total: int,
) -> Image.Image:
    """Paint the live specification onto one frame."""
    image = Image.fromarray(frame).convert("RGB")
    panel = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)

    title_font = load_font(19)
    body_font = load_font(15)
    small_font = load_font(13)

    height = 74 + 22 * len(rules) + 74
    draw.rectangle([(0, 0), (PANEL_WIDTH, height)], fill=(16, 18, 24, PANEL_ALPHA))

    speed = scalars.get("current_speed_mps", 0.0)
    robustness = scalars.get("stl_robustness")
    fallback = scalars.get("stl_fallback", 0.0) > 0.5

    draw.text((14, 12), "STL specification", font=title_font, fill=(235, 238, 245))
    draw.text(
        (14, 36),
        f"{speed * 3.6:5.1f} km/h    step {step}/{total}",
        font=body_font,
        fill=(190, 196, 210),
    )

    if robustness is None:
        draw.text((14, 58), "waiting for a route", font=body_font, fill=GREY)
    else:
        state = "EMERGENCY STOP" if fallback else "satisfied" if robustness >= 0 else "violated"
        draw.text(
            (14, 58),
            f"robustness {robustness:+6.2f}   {state}",
            font=body_font,
            fill=RED if (fallback or robustness < 0) else GREEN,
        )

    y = 84
    for rule in rules:
        value = scalars.get(f"stl_rule_{rule.name}")
        if value is None:
            colour, text, filled = GREY, "  --  ", 0.0
        else:
            colour = colour_for(value, rule.margin)
            text = f"{value:+6.2f}"
            # Scaled against the rule's own margin, clipped to the bar.
            filled = float(np.clip(value / rule.margin, -1.0, 1.0))

        draw.text((14, y), rule.name, font=body_font, fill=colour)
        draw.text((196, y), text, font=body_font, fill=colour)

        bar_x, bar_w = 258, 58
        draw.rectangle([(bar_x, y + 5), (bar_x + bar_w, y + 12)], fill=(45, 48, 58, 255))
        mid = bar_x + bar_w // 3
        if filled >= 0:
            draw.rectangle(
                [(mid, y + 5), (mid + int(filled * (bar_w - bar_w // 3)), y + 12)], fill=colour
            )
        else:
            draw.rectangle([(mid + int(filled * (bar_w // 3)), y + 5), (mid, y + 12)], fill=colour)
        y += 22

    # Two short lines rather than one long one: the longest single line would
    # run past the panel and sit unreadably on the road behind it.
    y += 8
    clearance = scalars.get("min_clearance_m")
    headway = scalars.get("min_headway_slack_m")
    solve = scalars.get("stl_solve_seconds")
    first = []
    if clearance is not None:
        first.append(f"clearance {min(clearance, 99.0):4.1f} m")
    if headway is not None:
        first.append(f"headway {min(headway, 99.0):4.1f} m")
    draw.text((14, y), "   ".join(first), font=small_font, fill=(180, 186, 200))

    y += 18
    second = []
    if solve is not None:
        second.append(f"solve {solve * 1000:.0f} ms")
    actors = scalars.get("n_actors")
    if actors is not None:
        second.append(f"{int(actors)} actors tracked")
    draw.text((14, y), "   ".join(second), font=small_font, fill=(180, 186, 200))
    if scalars.get("light_active", 0.0) > 0.5:
        draw.text((PANEL_WIDTH - 88, y), "RED LIGHT", font=small_font, fill=RED)

    return Image.alpha_composite(image.convert("RGBA"), panel).convert("RGB")


def _report_rule_violations(per_step: list[dict[str, float]], rules) -> None:
    """How often each rule was violated, and how badly.

    The whole point of writing the behaviour as a specification is that this
    table exists: it says what the vehicle failed to guarantee, in the same
    language the requirement was written in.
    """
    scored = [s for s in per_step if "stl_robustness" in s]
    if not scored:
        return

    satisfied = sum(1 for s in scored if s["stl_robustness"] >= 0.0)
    logger.info(
        "specification satisfied on %d of %d planned steps (%.1f%%)",
        satisfied,
        len(scored),
        100.0 * satisfied / len(scored),
    )
    logger.info("%-18s %8s %8s %10s", "rule", "steps", "share", "worst")
    for rule in rules:
        values = [s.get(f"stl_rule_{rule.name}", 0.0) for s in scored]
        bad = [v for v in values if v < 0.0]
        logger.info(
            "%-18s %8d %7.1f%% %10s",
            rule.name,
            len(bad),
            100.0 * len(bad) / len(scored),
            f"{min(bad):.2f} m" if bad else "-",
        )
    emergencies = sum(1 for s in scored if s.get("stl_fallback", 0.0) > 0.5)
    logger.info(
        "emergency plan driven on %d steps (%.1f%%)",
        emergencies,
        100.0 * emergencies / len(scored),
    )


def encode(frames_dir: Path, output: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the recording")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600, help="policy steps to record")
    parser.add_argument("--traffic", type=int, default=30)
    parser.add_argument("--map", dest="map_name", default="Town10HD_Opt")
    parser.add_argument("--spawn-point", type=int, default=None)
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path, default=Path("drive.mp4"))
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "let the search always take all its refinement steps instead of "
            "stopping at the wall-clock budget, so the same run gives the same "
            "answer. Slower per step, and not how the policy drives for real."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "report the rule violations and skip the film. Drawing the overlay "
            "and encoding it costs several minutes and dominates a run, which "
            "makes the recorder unusable as something to tune against."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # The search normally stops refining when its wall-clock budget runs out,
    # which makes the whole closed loop depend on how busy the machine is: the
    # same drive came back 96.0%, 91.0% and 93.0% satisfied over 340, 238 and
    # 236 metres. Once the ego acts differently at all, the scenario diverges
    # from there, so nothing smaller than that spread can be measured. Letting
    # every step run to completion removes the only nondeterminism the driver
    # has, at the cost of being slower than real time.
    config = STLDriverConfig()
    if args.deterministic:
        config = replace(
            config, optimizer=replace(config.optimizer, time_budget_s=math.inf)
        )
    policy = EgoPolicy(config=config)
    per_step: list[dict[str, float]] = []
    original = policy.drive

    def recording(ctx):
        result = original(ctx)
        per_step.append(dict(result.debug_scalars))
        return result

    policy.drive = recording  # type: ignore[method-assign]

    scenario = ScenarioSpec(
        map_name=args.map_name,
        num_background_vehicles=args.traffic,
        **({"spawn_point_index": args.spawn_point} if args.spawn_point is not None else {}),
    )

    with serving(policy, port=0, host="127.0.0.1") as port:
        config = RuntimeConfig(
            driver_address=f"127.0.0.1:{port}",
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            max_steps=args.steps,
        )
        world_class = SilentWorld if args.no_video else RecordingWorld
        world = world_class(config, scenario, width=args.width, height=args.height)
        outcome = CarlaRuntime(world, config, scenario).run_rollout()

    _report_rule_violations(per_step, policy.rules.rules)

    logger.info("rollout success=%s over %d steps", outcome.success, outcome.steps)
    for key in sorted(outcome.metrics):
        logger.info("  %-26s %.4f", key, outcome.metrics[key])

    if args.no_video:
        return 0

    frames = world.frames
    if not frames:
        logger.error("no frames captured")
        return 1

    # The runtime ticks the simulator several times per policy step, so a
    # frame's scalars are those of the policy step it falls inside.
    ticks_per_step = max(1, round(len(frames) / max(1, len(per_step))))
    fps = int(round(1.0 / config.fixed_delta_s))
    logger.info(
        "captured %d frames over %d policy steps (%d ticks/step) -> %d fps",
        len(frames),
        len(per_step),
        ticks_per_step,
        fps,
    )

    frames_dir = args.frames_dir or args.output.with_suffix("").parent / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in frames_dir.glob("*.png"):
        path.unlink()

    rules = policy.rules.rules
    for index, frame in enumerate(frames):
        step = min(index // ticks_per_step, len(per_step) - 1)
        composed = draw_overlay(frame, per_step[step], rules, step + 1, len(per_step))
        composed.save(frames_dir / f"{index:06d}.png")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode(frames_dir, args.output, fps)
    size_mb = args.output.stat().st_size / 1e6
    logger.info("wrote %s (%.1f MB, %.1f s)", args.output, size_mb, len(frames) / fps)
    return 0 if outcome.success else 1


if __name__ == "__main__":
    sys.exit(main())
