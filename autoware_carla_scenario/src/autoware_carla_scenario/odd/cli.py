"""``scenario-odd``: check an ODD, and show what it covers and rules out.

    uv run scenario-odd check my_package.odds:urban_odd   # compile with Codon
    uv run scenario-odd check path/to/odd.yaml            # read the OpenODD YAML
    uv run scenario-odd show path/to/odd.yaml             # buckets, in and out
    uv run scenario-odd list                              # ODDs known by name

An ODD is named the way the ``odd`` config key names it: ``default``, a
registered name, an OpenODD ``.yaml`` file, or ``package.module:function``.
``check`` builds it, and compiles a Python one with Codon as the runner would.
The exit status is 0 when every ODD passed, 1 when one failed, and 2 when a
Python ODD could not be compiled for want of Codon.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional, Sequence

from .registry import odd_builder, odd_names, resolve_odd

__all__ = ["main"]


def _check(spec: str) -> int:
    from ..typecheck import typecheck_odd  # noqa: PLC0415

    try:
        odd = resolve_odd(spec)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"[FAILED] {spec}: {exc}")  # noqa: T201
        return 1
    builder = odd_builder(spec)
    if builder is None:
        print(f"[ok] {spec}: {odd.name} reads ({len(odd.attributes)} attributes)")  # noqa: T201
        return 0
    result = typecheck_odd(builder)
    if result.skipped is not None:
        print(f"[SKIPPED] {spec}: {result.format()}")  # noqa: T201
        return 2 if "Codon" in result.skipped else 0
    print(f"[{'ok' if result.ok else 'FAILED'}] {spec}: {result.format()}")  # noqa: T201
    return 0 if result.ok else 1


def _show(spec: str) -> int:
    try:
        odd = resolve_odd(spec)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"scenario-odd: {spec}: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    out = odd.describe()
    out["attributes"] = [
        {**a.item.describe(), "outside_odd": odd.outside_buckets(a)}
        for a in odd.attributes
        if a.item is not None
    ]
    print(json.dumps(out, indent=2))  # noqa: T201
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ``scenario-odd`` with *argv* (``sys.argv[1:]`` by default)."""
    parser = argparse.ArgumentParser(
        prog="scenario-odd",
        description="Check an ODD, and show what it covers and rules out.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check", help="build the ODDs; compile Python ones with Codon"
    )
    check.add_argument(
        "odds", nargs="+", help="ODD names, .yaml files or module:function"
    )
    show = sub.add_parser("show", help="print an ODD's attributes, buckets and modules")
    show.add_argument("odd", help="an ODD name, .yaml file or module:function")
    sub.add_parser("list", help="list the ODDs known by name")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.command == "list":
        for name in odd_names():
            print(name)  # noqa: T201
        return 0
    if args.command == "show":
        return _show(args.odd)
    statuses = [_check(spec) for spec in args.odds]
    if 1 in statuses:
        return 1  # a failure outranks a check that could not be made
    return max(statuses)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
