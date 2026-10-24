"""``scenario-hils``: run the HILS bridge a configuration file describes.

::

    uvx --from 'autoware-carla-scenario[hils]' scenario-hils --example > hils.yaml
    uvx --from 'autoware-carla-scenario[hils]' scenario-hils hils.yaml
    uvx --from 'autoware-carla-scenario[hils]' --with 'carla-canvas[client]>=0.2' \\
        scenario-hils hils.yaml          # with `canvas:` in hils.yaml

It runs until interrupted (Ctrl-C or SIGTERM), across the runner's episodes.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import sys
from importlib import resources
from types import FrameType
from typing import List, Optional

from ..canvas import CanvasUnavailableError
from .config import HilsConfig, HilsConfigError, load_config
from .signals import SIGNALS

__all__ = ["example_config", "main"]

logger = logging.getLogger(__name__)


def example_config() -> str:
    """The commented example configuration ``--example`` prints."""
    return (
        resources.files("autoware_carla_scenario.hils")
        .joinpath("example.yaml")
        .read_text(encoding="utf-8")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scenario-hils",
        description=(
            "Follow a scenario run's ego in CARLA and stream it to devices under "
            "test (CAN, GNSS, LiDAR, cameras) through hilspy."
        ),
    )
    parser.add_argument("config", nargs="?", help="the bridge's YAML configuration")
    parser.add_argument("--host", help="the CARLA host (overrides carla.host)")
    parser.add_argument(
        "--port", type=int, help="the CARLA port (overrides carla.port)"
    )
    parser.add_argument(
        "--role-name", help="the ego's role_name (overrides ego.role_name)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the configuration, print what it runs, and exit",
    )
    parser.add_argument(
        "--example", action="store_true", help="print an example configuration"
    )
    parser.add_argument(
        "--signals", action="store_true", help="list the signals fields can use"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def _describe(config: HilsConfig) -> str:
    lines = [
        f"CARLA {config.host}:{config.port}, ego {config.role_name!r}, "
        f"{config.clock} clock" + (f", canvas {config.canvas}" if config.canvas else "")
    ]
    for device in config.devices:
        lines.append(f"  {device.name}: {device.url} ({device.codec})")
        if device.sensor is not None:
            lines.append(f"    {device.sensor.blueprint} on the ego")
        for message in device.messages:
            period = "cycle time" if message.period is None else f"{message.period} s"
            lines.append(
                f"    {message.name} every {period}: {', '.join(message.fields)}"
            )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """The ``scenario-hils`` command; its exit status."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.example:
        sys.stdout.write(example_config())
        return 0
    if args.signals:
        width = max(map(len, SIGNALS))
        for name, meaning in SIGNALS.items():
            print(f"{name:<{width}}  {meaning}")
        return 0
    if not args.config:
        _parser().print_usage(sys.stderr)
        print("scenario-hils: a configuration file is needed", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
    except HilsConfigError as exc:
        print(f"scenario-hils: {exc}", file=sys.stderr)
        return 2
    overrides = {
        "host": args.host,
        "port": args.port,
        "role_name": args.role_name,
    }
    config = dataclasses.replace(
        config, **{k: v for k, v in overrides.items() if v is not None}
    )
    if args.check:
        print(_describe(config))
        return 0

    from .bridge import HilsBridge, HilsUnavailableError  # noqa: PLC0415

    try:
        bridge = HilsBridge(config)
    except (HilsUnavailableError, CanvasUnavailableError) as exc:
        print(f"scenario-hils: {exc}", file=sys.stderr)
        return 1

    def _stop(signum: int, frame: Optional[FrameType]) -> None:
        logger.info("stopping (signal %d)", signum)
        bridge.stop()

    signal.signal(signal.SIGTERM, _stop)
    try:
        bridge.run()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
