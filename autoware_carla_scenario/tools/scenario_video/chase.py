"""Chase camera on a running scenario's ego, and CARLA's traffic light states.

    python chase.py <out_dir> [--host localhost] [--port 2000]

Waits for the ego (role ``Ego``) to appear in a synchronous world, attaches an
RGB camera behind it, and saves a frame every 0.1 s of simulation time as
``carla_<elapsed>.png``.  Each saved frame also appends every traffic light's
state to ``lights.csv`` (``elapsed,opendrive_id,state``), which
``check_run.py`` compares with SUMO's.  Exits once the ego is gone.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import typesafe_carla.carla as carla

from autoware_carla_scenario.utils.traffic_light import traffic_light_state_name

PERIOD_S = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--wait-s", type=float, default=900.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # render.py stops a chase that outlives its run with SIGTERM; exiting
    # through SystemExit runs the finally below, which removes the camera.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    deadline = time.time() + args.wait_s
    camera = ego = world = None
    while time.time() < deadline and camera is None:
        try:
            world = client.get_world()
            egos = [
                a
                for a in world.get_actors().filter("vehicle.*")
                if a.attributes.get("role_name") == "Ego"
            ]
            if egos and world.get_settings().synchronous_mode:
                ego = egos[0]
                blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
                for key, value in (
                    ("image_size_x", "960"),
                    ("image_size_y", "540"),
                    ("fov", "90"),
                ):
                    blueprint.set_attribute(key, value)
                # Retried: the previous run's teardown may still be reloading the world.
                camera = world.spawn_actor(
                    blueprint,
                    carla.Transform(
                        carla.Location(x=-8.0, z=4.5), carla.Rotation(pitch=-16.0)
                    ),
                    attach_to=ego,
                )
        except RuntimeError:
            camera = None
        # Tight: a lane change is over in under a second of simulation.
        time.sleep(0.02)
    if camera is None or world is None or ego is None:
        raise SystemExit("no ego appeared")

    lights = list(world.get_actors().filter("traffic.traffic_light"))
    log = (args.out / "lights.csv").open("w")
    state = {"next": None, "last": time.time()}

    def save(image: object) -> None:
        stamp = image.timestamp
        state["last"] = time.time()
        if state["next"] is None:
            state["next"] = stamp
        if stamp + 1e-6 < state["next"]:
            return
        image.save_to_disk(str(args.out / f"carla_{stamp:010.3f}.png"))
        state["next"] += PERIOD_S
        try:
            for light in lights:
                log.write(
                    f"{stamp:.3f},{light.get_opendrive_id()},{traffic_light_state_name(light.get_state())}\n"
                )
            log.flush()
        except RuntimeError:
            pass

    try:
        camera.listen(save)
        while time.time() - state["last"] < 20 and time.time() < deadline:
            time.sleep(0.5)
            try:
                if not any(
                    a.id == ego.id
                    for a in client.get_world().get_actors().filter("vehicle.*")
                ):
                    break
            except RuntimeError:
                break
    finally:
        # Also on SIGTERM from render.py: a camera left behind stays attached
        # in the next run's world.
        try:
            camera.stop()
            camera.destroy()
        except RuntimeError:
            pass
        log.close()
    print("frames", len(list(args.out.glob("carla_*.png"))))


if __name__ == "__main__":
    main()
