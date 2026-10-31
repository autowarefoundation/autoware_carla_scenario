"""``scenario-t4scene``: write trajectory logs as T4Scenes (docs/t4_scene.md)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..constants import FIXED_DELTA_SECONDS
from .convert import LOG_SUFFIX, write_t4_scene
from .grid import scene_stem


def _logs(paths: list[Path]) -> list[Path]:
    """The trajectory logs *paths* name: a log itself, or every log under a directory."""
    logs: list[Path] = []
    for path in paths:
        if path.is_dir():
            logs.extend(sorted(path.rglob(f"*{LOG_SUFFIX}")))
        else:
            logs.append(path)
    return logs


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="scenario-t4scene", description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help=f"trajectory logs ({{stem}}{LOG_SUFFIX}), or directories to find them under",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="where to write the scenes (default: beside each log)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help=f"the scene's time step in seconds, a multiple of the {FIXED_DELTA_SECONDS} s tick "
        "(default 0.1)",
    )
    parser.add_argument(
        "--current-index",
        type=int,
        default=0,
        help="the scene's anchor row; negative counts from the end (default 0)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=100.0,
        help="annotate the road users within this many metres of the ego (default 100)",
    )
    parser.add_argument(
        "--all-objects",
        action="store_true",
        help="annotate every road user, however far",
    )
    parser.add_argument(
        "--rear-axle-offset",
        type=float,
        help="the ego's rear axle from its origin, metres along x (default: half its box back)",
    )
    parser.add_argument(
        "--no-map", action="store_true", help="write the scene without a map"
    )
    parser.add_argument(
        "--lanelet2",
        type=Path,
        help="the Lanelet2 map, when it has moved since the run",
    )
    parser.add_argument(
        "--xodr", type=Path, help="the OpenDRIVE file, when it has moved since the run"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logs = _logs(args.paths)
    if not logs:
        parser.error("no trajectory log found")
    if args.output_dir is not None:
        stems: dict[str, Path] = {}
        for log in logs:
            stem = scene_stem(log.name)
            if stem in stems:
                parser.error(
                    f"{stems[stem]} and {log} would both be written to "
                    f"{args.output_dir / stem}.*; convert them without -o"
                )
            stems[stem] = log
    failed = False
    for log in logs:
        try:
            scene, resource = write_t4_scene(
                log,
                args.output_dir,
                step_s=args.step,
                current_index=args.current_index,
                max_distance_m=None if args.all_objects else args.max_distance,
                rear_axle_offset_m=args.rear_axle_offset,
                with_map=not args.no_map,
                lanelet2_path=args.lanelet2,
                xodr_path=args.xodr,
            )
        except Exception as error:  # noqa: BLE001 - one bad log does not stop the rest
            print(f"{log}: {error}", file=sys.stderr)  # noqa: T201
            failed = True
            continue
        print(scene if resource is None else f"{scene} (map: {resource})")  # noqa: T201
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
