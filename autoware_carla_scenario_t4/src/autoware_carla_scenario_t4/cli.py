"""``t4-scenario``: write a T4 scene down as a scenario.

::

    # The scene as a scenario document, opened in the Scenario Editor:
    t4-scenario /data/t4/.../scene_0001 \\
        --lanelet2 map/lanelet2_map.osm --xodr map/map.xodr \\
        --map-group my_map --map-name MyMap --to-editor

    # ...or written to a file, and the transcription kept beside it:
    t4-scenario /data/t4/.../scene_0001 --lanelet2 map/lanelet2_map.osm \\
        --output scene_0001.yaml --transcription scene_0001.json

The scene may also be a transcription saved earlier (a ``.json`` file), so the
dataset is read once.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .reader import T4Category, T4SceneTranscription, read_t4_scene

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="t4-scenario",
        description="Write a T4 driving scene down as a scenario document.",
    )
    parser.add_argument(
        "scene", type=Path, help="T4 scene directory, or a saved transcription (.json)"
    )
    parser.add_argument(
        "--lanelet2",
        type=Path,
        required=True,
        help="the scene's area map (Lanelet2 .osm): entities are placed on it",
    )
    parser.add_argument(
        "--xodr", type=Path, help="the map's OpenDRIVE, if the .osm states no origin"
    )
    parser.add_argument(
        "--map-group", default="nishishinjuku", help="the map's Hydra group"
    )
    parser.add_argument("--map-name", default="NishishinjukuMap", help="the map's name")
    parser.add_argument("--output", "-o", type=Path, help="write the document here")
    parser.add_argument(
        "--to-editor",
        action="store_true",
        help="save the document as a Scenario Editor draft",
    )
    parser.add_argument(
        "--transcription", type=Path, help="also save the transcription here"
    )
    parser.add_argument(
        "--categories",
        default="vehicle,bicycle,pedestrian",
        help="road users to replay, comma separated (default: all)",
    )
    parser.add_argument(
        "--following-mode", choices=("position", "follow"), default="position"
    )
    parser.add_argument(
        "--ego-driver", choices=("autopilot", "autoware"), default="autopilot"
    )
    parser.add_argument(
        "--replay-ego",
        action="store_true",
        help="put the ego on the recorded drive too",
    )
    parser.add_argument("--vehicle-type", help="blueprint of every replayed vehicle")
    parser.add_argument(
        "--pedestrian-type", help="blueprint of every pedestrian and cyclist"
    )
    parser.add_argument(
        "--vertex-interval",
        type=float,
        default=0.2,
        help="seconds between kept vertices (0 keeps every frame)",
    )
    parser.add_argument("--max-gap-frames", type=int, default=5)
    parser.add_argument("--min-track-frames", type=int, default=5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command; returns the exit status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    if args.output is None and not args.to_editor:
        print("t4-scenario: give --output, --to-editor or both", file=sys.stderr)
        return 2

    from autoware_carla_scenario.authoring.models import MapRef  # noqa: PLC0415
    from autoware_carla_scenario.authoring.persistence import (  # noqa: PLC0415
        DraftStore,
        save_document,
    )

    from .document import LaneletLocator, load_lanelet_map, transcription_to_document  # noqa: PLC0415

    if args.scene.suffix == ".json":
        transcription = T4SceneTranscription.load(args.scene)
    else:
        transcription = read_t4_scene(
            args.scene,
            max_gap_frames=args.max_gap_frames,
            min_track_frames=args.min_track_frames,
        )
    if args.transcription is not None:
        transcription.save(args.transcription)

    categories = [
        T4Category(name.strip()) for name in args.categories.split(",") if name.strip()
    ]
    map_ref = MapRef(
        group=args.map_group,
        name=args.map_name,
        lanelet2_path=str(args.lanelet2.resolve()),
        xodr_path=None if args.xodr is None else str(args.xodr.resolve()),
    )
    try:
        document = transcription_to_document(
            transcription,
            LaneletLocator(load_lanelet_map(args.lanelet2, args.xodr)),
            map_ref,
            categories=categories,
            following_mode=args.following_mode,
            replay_ego=args.replay_ego,
            ego_driver=args.ego_driver,
            vehicle_type=args.vehicle_type,
            pedestrian_type=args.pedestrian_type,
            vertex_interval_s=args.vertex_interval,
        )
    except ValueError as exc:
        print(f"t4-scenario: {exc}", file=sys.stderr)
        return 1

    replayed = len(document.entities) - 1
    print(
        f"{transcription.scene_name}: {replayed} of {len(transcription.objects)} road "
        f"users, {transcription.duration:.1f} s"
    )
    if args.output is not None:
        print(f"wrote {save_document(document, args.output)}")
    if args.to_editor:
        draft = DraftStore().create(document, title=document.title)
        print(
            f"saved draft {draft.id}: open it from the Scenario Editor (scenario-editor)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
