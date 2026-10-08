"""``autoware-carla-egodriver``: serve a reference policy, or drive one without CARLA."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .policies import POLICY_REGISTRY
from .server import run_server
from .testing import FakeCamera, FakeLoop

__all__ = ["main"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autoware-carla-egodriver", description=__doc__
    )
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve a reference policy over gRPC")
    serve.add_argument(
        "--policy", choices=sorted(POLICY_REGISTRY), default="route_follower"
    )
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=50051)
    serve.add_argument("--cruise-speed", type=float, default=None, help="m/s")

    demo = commands.add_parser(
        "demo", help="drive a policy server through the CARLA-free fake loop"
    )
    demo.add_argument("--driver", default="localhost:50051", help="policy host:port")
    demo.add_argument("--steps", type=int, default=100)
    demo.add_argument("--step", type=float, default=0.1, help="seconds per drive call")
    demo.add_argument("--camera-width", type=int, default=FakeCamera.width)
    demo.add_argument("--camera-height", type=int, default=FakeCamera.height)
    demo.add_argument("--camera-fov", type=float, default=FakeCamera.fov_deg)
    demo.add_argument("--camera-id", default=FakeCamera.logical_id)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "serve":
        policy_cls = POLICY_REGISTRY[args.policy]
        kwargs = (
            {} if args.cruise_speed is None else {"cruise_speed_mps": args.cruise_speed}
        )
        run_server(policy_cls(**kwargs), port=args.port, host=args.host)
        return 0

    camera = FakeCamera(
        logical_id=args.camera_id,
        width=args.camera_width,
        height=args.camera_height,
        fov_deg=args.camera_fov,
    )
    with FakeLoop(args.driver, [camera], step_s=args.step) as loop:
        result = loop.run(args.steps)
    final = result.ego.poses[-1].position
    print(f"steps              : {result.steps}")
    print(f"terminated by policy: {result.terminated_by_policy}")
    print(f"distance travelled : {final[0]:.2f} m")
    print(f"final lateral offset: {final[1]:+.2f} m")
    print(f"max lateral offset : {max(abs(y) for y in result.lateral_offsets_m):.2f} m")
    print(f"final speed        : {result.speeds_mps[-1]:.2f} m/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
