# SPDX-License-Identifier: Apache-2.0
"""Command line entry points.

``serve`` runs the policy as an egodriver gRPC server for a runtime in another
process to connect to.  (autoware_carla_scenario can also load it into its own
process: ``driver.policy=stl_driver.policy:warmed_up_policy``.)  ``demo`` closes the
loop against carla_driver_interface's fake straight road so the whole thing can
be exercised without CARLA, and ``explain`` prints the specification, because a
specification nobody can read is not much better than no specification.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from stl_driver import __version__

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stl-driver",
        description="An ego policy whose driving behaviour is written in Signal Temporal Logic.",
    )
    parser.add_argument("--version", action="version", version=f"stl-driver {__version__}")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the policy as an egodriver gRPC server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=50051)
    serve.add_argument("--max-workers", type=int, default=8)
    serve.add_argument(
        "--cruise-speed",
        type=float,
        default=None,
        help="speed to aim for when nothing else binds, m/s",
    )
    serve.set_defaults(handler=_serve)

    demo = subparsers.add_parser(
        "demo", help="close the loop against a fake straight road, without CARLA"
    )
    demo.add_argument("--steps", type=int, default=120)
    demo.add_argument("--cruise-speed", type=float, default=None)
    demo.add_argument("--metrics-json", default=None, help="write rollout metrics here")
    demo.set_defaults(handler=_demo)

    explain = subparsers.add_parser("explain", help="print the STL specification")
    explain.set_defaults(handler=_explain)

    return parser


def _serve(args: argparse.Namespace) -> int:
    from carla_driver_interface.server import run_server

    from stl_driver.policy import EgoPolicy

    policy = EgoPolicy(cruise_speed_mps=args.cruise_speed)
    logging.info("compiling the search before serving")
    policy.warm_up()
    logging.info(
        "serving %s on %s:%d with %d rules",
        policy.name,
        args.host,
        args.port,
        len(policy.rules),
    )
    run_server(policy, port=args.port, host=args.host, max_workers=args.max_workers)
    return 0


def _demo(args: argparse.Namespace) -> int:
    """Run a closed-loop rollout against carla_driver_interface's fake road.

    Both halves in one process: the driver server and the loop that drives
    it. That is not how a real deployment looks, but it makes the whole loop
    reproducible in one command with no simulator.
    """
    import json

    import numpy as np
    from carla_driver_interface import polyline
    from carla_driver_interface.server import serving
    from carla_driver_interface.testing import FakeLoop

    from stl_driver.policy import EgoPolicy

    policy = EgoPolicy(cruise_speed_mps=args.cruise_speed)
    policy.warm_up()

    with serving(policy, port=0, host="127.0.0.1") as port:
        with FakeLoop(f"127.0.0.1:{port}") as loop:
            result = loop.run(args.steps)

    positions = result.ego.positions
    speeds = result.speeds_mps
    metrics = {
        "distance_travelled_m": float(polyline.arc_lengths(positions)[-1]),
        "speed_mps": float(np.mean(speeds)) if speeds else 0.0,
        "max_lateral_offset_m": float(np.abs(result.lateral_offsets_m).max()),
    }
    success = not result.terminated_by_policy
    logging.info("rollout finished after %d steps: success=%s", result.steps, success)
    for key in sorted(metrics):
        logging.info("  %s = %.3f", key, metrics[key])

    if args.metrics_json:
        with open(args.metrics_json, "w") as handle:
            json.dump({"steps": result.steps, "metrics": metrics}, handle, indent=2)

    return 0 if success else 1


def _explain(args: argparse.Namespace) -> int:
    """Print the specification: what this policy is promising to do."""
    from stl_driver.config import STLDriverConfig
    from stl_driver.rules import build_rules

    config = STLDriverConfig()
    rules = build_rules(config)

    print(
        f"stl-driver {__version__} -- {len(rules)} rules over a "
        f"{config.horizon.duration_s:.1f} s horizon at {config.horizon.step_s:.2f} s\n"
    )
    for rule in rules:
        kind = "hard" if rule.hard else "soft"
        print(f"  {rule.name}  [{kind}, weight {rule.weight:g}, margin {rule.margin:g}]")
        print(f"      {rule.description}")
        print(f"      {rule.formula}")
        print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
