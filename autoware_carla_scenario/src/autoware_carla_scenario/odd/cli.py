"""``scenario-odd``: check an ODD, and show what it covers and rules out.

    uv run scenario-odd check my_package.odds:urban_odd   # compile with Codon
    uv run scenario-odd check path/to/odd.yaml            # read the OpenODD YAML
    uv run scenario-odd show path/to/odd.yaml             # buckets, in and out
    uv run scenario-odd list                              # ODDs known by name
    uv run scenario-odd route path/to/odd.yaml 'intersection_passing/*'

An ODD is named the way the ``odd`` config key names it: ``default``, a
registered name, an OpenODD ``.yaml`` file, or ``package.module:function``.
``check`` builds it, and compiles a Python one with Codon as the runner would.
The exit status is 0 when every ODD passed, 1 when one failed, and 2 when a
Python ODD could not be compiled for want of Codon.

``route`` plans each scenario config's ego route on its Lanelet2 map (no
CARLA server) and checks it against the ODD (:mod:`.route`).  Scenario
configs are named as ``scenario=`` names them (globs too); arguments with an
``=`` are Hydra overrides for every one of them.  The exit status is 0 when no
route leaves the ODD, 1 when one does, and 2 when a route could not be
planned (no goal, no path, no map) or the ODD could not be read.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Optional, Sequence

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


def _scenario_configs(items: Sequence[str]) -> tuple[list[str], list[str]]:
    """Scenario config names (globs expanded) and the Hydra overrides in *items*."""
    from ..examples import run  # noqa: PLC0415 - registers the built-in scenarios

    patterns: list[str] = []
    overrides: list[str] = []
    for item in items:
        if item.startswith("scenario="):
            patterns.append(item[len("scenario=") :])
        elif "=" in item:
            overrides.append(item)
        else:
            patterns.append(item)
    names: list[str] = []
    for pattern in patterns:
        if run._is_glob_pattern(pattern):
            try:
                names += run._resolve_scenario_glob(pattern)
            except SystemExit:  # it has said what it could not match
                raise ValueError(f"no scenario configs match {pattern!r}") from None
        else:
            names.append(pattern)
    return names, overrides


def _format_route(coverage: Any) -> list[str]:
    ids = [str(i) for i in coverage.lanelet_ids]
    if len(ids) > 12:
        ids = [*ids[:6], "...", *ids[-5:]]
    lines = [
        f"{coverage.name}: lanelets {' -> '.join(ids)} ({coverage.length_m:.1f} m)",
        f"  inside {coverage.inside_m:.1f} m, undecided {coverage.undecided_m:.1f} m, "
        f"outside {coverage.outside_m:.1f} m",
    ]
    for lanelet in coverage.outside():
        lines.append(
            f"  OUTSIDE lanelet {lanelet.lanelet_id} ({lanelet.length_m:.1f} m): "
            f"modules {', '.join(lanelet.failing_modules) or '-'}"
        )
    undecided = coverage.undecided()
    if undecided:
        lines.append(
            "  inside only by assumption: lanelets "
            + ", ".join(str(ll.lanelet_id) for ll in undecided)
        )
    return lines


def _format_buckets(buckets: dict[str, float]) -> str:
    return ", ".join(f"{label} {metres:.1f} m" for label, metres in buckets.items())


def _route(spec: str, items: Sequence[str], as_json: bool) -> int:
    from .route import combine_route_coverage, plan_route_coverage  # noqa: PLC0415

    try:
        odd = resolve_odd(spec)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"scenario-odd: {spec}: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    from ..examples import run  # noqa: PLC0415 - composes configs as a run does
    from ..maps import resolve_map_paths  # noqa: PLC0415
    from ..registry import load_scenario_plugins  # noqa: PLC0415
    from ..sweeper.constraints import create_routing_graph  # noqa: PLC0415
    from ..sweeper.map_loader import load_map  # noqa: PLC0415

    load_scenario_plugins()
    try:
        names, overrides = _scenario_configs(items)
    except ValueError as exc:
        print(f"scenario-odd: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    if not names:
        print("scenario-odd: route: name a scenario config", file=sys.stderr)  # noqa: T201
        return 2

    maps: dict[Any, tuple[Any, Any]] = {}
    routes = []
    failed: list[tuple[str, str]] = []
    for name in names:
        try:
            cfg = run._compose_config(name, overrides)
            route = run.build_planned_route(cfg, name)
            paths = resolve_map_paths(cfg.map)
            key = (paths.lanelet2_path, paths.xodr_path, paths.projector_type)
            if key not in maps:
                lanelet_map = load_map(paths)
                maps[key] = (lanelet_map, create_routing_graph(lanelet_map))
            lanelet_map, graph = maps[key]
            routes.append(
                plan_route_coverage(
                    odd, route, lanelet_map=lanelet_map, routing_graph=graph
                )
            )
        except Exception as exc:  # noqa: BLE001 - reported per scenario
            failed.append((name, str(exc) or type(exc).__name__))
    summary = combine_route_coverage(odd, routes)

    if as_json:
        out = summary.describe()
        out["routes"] = [r.describe() for r in routes]
        out["not_planned"] = dict(failed)
        print(json.dumps(out, indent=2))  # noqa: T201
    else:
        print(f"ODD {odd.name}")  # noqa: T201
        for coverage in routes:
            print("\n".join(_format_route(coverage)))  # noqa: T201
        for name, reason in failed:
            print(f"{name}: NOT PLANNED: {reason}")  # noqa: T201
        if routes:
            print("expected coverage, all routes:")  # noqa: T201
            for attribute in routes[0].map_attributes:
                buckets = summary.expected_m.get(attribute)
                if buckets is not None:
                    print(f"  {attribute}: {_format_buckets(buckets) or '-'}")  # noqa: T201
            for attribute, labels in summary.unreached.items():
                print(f"  unreached {attribute}: {', '.join(labels)}")  # noqa: T201
            for attribute in summary.undetermined:
                print(f"  undetermined {attribute}: undecided on some lanelets")  # noqa: T201
    if summary.leaving:
        return 1
    return 2 if failed else 0


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
    route = sub.add_parser(
        "route",
        help="plan scenarios' ego routes on the Lanelet2 map and check them "
        "against an ODD",
    )
    route.add_argument("odd", help="an ODD name, .yaml file or module:function")
    route.add_argument(
        "scenarios",
        nargs="+",
        help="scenario config names or globs (as scenario= takes them), and "
        "Hydra overrides (key=value) for all of them",
    )
    route.add_argument("--json", action="store_true", help="print JSON")
    # Scenario configs and overrides may come either side of --json.
    args, extra = parser.parse_known_args(argv)
    unknown = [a for a in extra if args.command != "route" or a.startswith("-")]
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.command == "list":
        for name in odd_names():
            print(name)  # noqa: T201
        return 0
    if args.command == "show":
        return _show(args.odd)
    if args.command == "route":
        return _route(args.odd, [*args.scenarios, *extra], args.json)
    statuses = [_check(spec) for spec in args.odds]
    if 1 in statuses:
        return 1  # a failure outranks a check that could not be made
    return max(statuses)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
