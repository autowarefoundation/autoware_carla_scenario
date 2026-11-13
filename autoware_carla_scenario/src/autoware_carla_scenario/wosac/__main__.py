"""``scenario-wosac``: write trajectory logs as WOSAC rollouts, and pack
rollouts as a WOSAC submission (docs/wosac.md)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import orjson
import yaml

from ..authoring.persistence import load_yaml
from ..t4_scene.grid import stem_collision, trajectory_logs
from .convert import CURRENT_INDEX, FUTURE_STEPS, write_rollout
from .pack import (
    build_submission,
    merge_rollouts,
    read_rollouts,
    rollout_files,
    write_submission,
)


def _mapping(path: Path, what: str) -> dict[str, Any]:
    """The mapping in the YAML or JSON file at *path*."""
    try:
        # Only true/false are booleans: a role named `on` or `yes` stays one.
        data = load_yaml(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"{what} {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{what} {path} does not hold a mapping")
    return data


def _export(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    logs = trajectory_logs(args.paths)
    if not logs:
        parser.error("no trajectory log found")
    if args.output_dir is not None:
        collision = stem_collision(logs, args.output_dir)
        if collision is not None:
            parser.error(collision)
    object_ids = None
    if args.object_ids is not None:
        try:
            object_ids = _mapping(args.object_ids, "--object-ids")
        except ValueError as error:
            parser.error(str(error))
    failed = False
    for log in logs:
        try:
            rollouts, agents = write_rollout(
                log,
                args.output_dir,
                scenario_id=args.scenario_id,
                current_index=args.current_index,
                steps=None if args.all_steps else args.steps,
                pad=args.pad,
                include_ego=not args.no_ego,
                map_frame=args.map_frame,
                with_boxes=args.with_boxes,
                object_ids=object_ids,
            )
        except Exception as error:  # noqa: BLE001 - one bad log does not stop the rest
            print(f"{log}: {error}", file=sys.stderr)  # noqa: T201
            failed = True
            continue
        print(f"{rollouts} (agents: {agents})")  # noqa: T201
    return 1 if failed else 0


def _pack(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    files = rollout_files(args.paths)
    if not files:
        parser.error("no rollout (*.wosac.pb) found")
    try:
        metadata = (
            None if args.metadata is None else _mapping(args.metadata, "--metadata")
        )
        merged = merge_rollouts(
            ((read_rollouts(path), str(path)) for path in files),
            num_rollouts=args.rollouts,
            common_agents=args.common_agents,
        )
        submission = build_submission(merged, metadata)
        del merged  # The submission holds the rollouts now.
        paths = write_submission(
            submission, args.output_dir, shards=args.shards, tar=args.tar
        )
    except (ValueError, OSError) as error:
        print(f"scenario-wosac pack: {error}", file=sys.stderr)  # noqa: T201
        return 1
    summary = {
        scenario.scenario_id: len(scenario.joint_scenes)
        for scenario in submission.scenario_rollouts
    }
    print(orjson.dumps({"rollouts": summary}).decode())  # noqa: T201
    for path in paths:
        print(path)  # noqa: T201
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="scenario-wosac", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser(
        "export",
        help="write each trajectory log as one rollout, {stem}.wosac.pb",
        description="Write each trajectory log as a ScenarioRollouts of one "
        "JointScene ({stem}.wosac.pb), and its agents ({stem}.wosac.json).",
    )
    export.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="trajectory logs ({stem}.trajectory.jsonl), or directories to find them under",
    )
    export.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="where to write (default: beside each log)",
    )
    export.add_argument(
        "--scenario-id",
        help="the rollouts' scenario_id (default: the log's stem without its "
        "run counter, CutInScenario-2 -> CutInScenario)",
    )
    export.add_argument(
        "--current-index",
        type=int,
        default=CURRENT_INDEX,
        help=f"the current row, at 10 Hz from the log's first tick (default {CURRENT_INDEX})",
    )
    steps = export.add_mutually_exclusive_group()
    steps.add_argument(
        "--steps",
        type=int,
        default=FUTURE_STEPS,
        help=f"the rows after the current one (default {FUTURE_STEPS}, 8 s)",
    )
    steps.add_argument(
        "--all-steps", action="store_true", help="every row up to the log's end"
    )
    export.add_argument(
        "--pad",
        action="store_true",
        help="hold the last poses past the log's end rather than refuse a short log",
    )
    export.add_argument("--no-ego", action="store_true", help="leave the ego out")
    export.add_argument(
        "--map-frame",
        action="store_true",
        help="positions in Autoware's map frame, MGRS offset and all "
        "(default: its axes at CARLA's origin)",
    )
    export.add_argument(
        "--with-boxes",
        action="store_true",
        help="also write width, length, height, valid and object_type",
    )
    export.add_argument(
        "--object-ids",
        type=Path,
        help="a YAML or JSON mapping of role names to object ids, over the "
        "ids derived from them",
    )

    pack = commands.add_parser(
        "pack",
        help="merge rollouts by scenario into a WOSAC submission",
        description="Merge rollouts (*.wosac.pb) by scenario_id and write them "
        "as SimAgentsChallengeSubmission shards.",
    )
    pack.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="rollouts ({stem}.wosac.pb), or directories to find them under",
    )
    pack.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="where to write the shards"
    )
    pack.add_argument(
        "--rollouts",
        type=int,
        help="how many rollouts every scenario must have (WOSAC's 32; default: any)",
    )
    pack.add_argument(
        "--common-agents",
        action="store_true",
        help="keep only the agents every rollout of a scenario has, rather than refuse",
    )
    pack.add_argument(
        "--metadata",
        type=Path,
        help="a YAML or JSON mapping of the submission's fields (account_name, "
        "unique_method_name, authors, ...)",
    )
    pack.add_argument(
        "--shards",
        type=int,
        default=1,
        help="how many files to split it into (default 1)",
    )
    pack.add_argument(
        "--tar", action="store_true", help="also write submission.tar.gz of the shards"
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "export":
        return _export(args, export)
    return _pack(args, pack)


if __name__ == "__main__":
    sys.exit(main())
