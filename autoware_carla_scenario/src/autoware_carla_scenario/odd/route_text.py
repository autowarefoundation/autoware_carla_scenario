"""What ``scenario-odd route`` reads off its command line, and the metres it prints.

Its arguments split into the scenario configs it plans (a ``scenario=``
argument, or one with no ``=``) and the Hydra overrides it composes every one
of them with (any other argument with an ``=``), in the order given
(:func:`_split_items`); and the metres a route is expected to drive in each
bucket of an attribute, written ``<bucket> <metres> m`` (:func:`_format_buckets`);
the lines it prints for a route (:func:`format_route_lines`); and the package
an ODD spec names (:func:`own_package`).

Split out of ``odd.cli``, which re-exports :func:`_format_buckets`, expands
the scenario configs' globs (the registry, omegaconf) and reads openodd's
route coverage into plain values, so that the library check compiles it
(``docs/typecheck.md``, "Checking the framework itself").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

_SCENARIO = "scenario="


def _split_items(items: Sequence[str]) -> tuple[list[str], list[str]]:
    """The scenario config patterns and the Hydra overrides in *items*.

    A ``scenario=`` item is the pattern after it; any other item with an
    ``=`` is an override; any other is a pattern.
    """
    patterns: list[str] = []
    overrides: list[str] = []
    for item in items:
        if item.startswith(_SCENARIO):
            patterns.append(item[len(_SCENARIO) :])
        elif "=" in item:
            overrides.append(item)
        else:
            patterns.append(item)
    return patterns, overrides


def _format_buckets(buckets: dict[str, float]) -> str:
    return ", ".join(f"{label} {metres:.1f} m" for label, metres in buckets.items())


def own_package(spec: str) -> Optional[str]:
    """The top-level package of the module a ``package.module:function`` spec names.

    None for a spec without a ``:``, or one in openodd or this package.
    """
    if ":" not in spec:
        return None
    top = spec.split(":", 1)[0].split(".", 1)[0]
    if top and top != "openodd" and top != "autoware_carla_scenario":
        return top
    return None


def format_route_lines(
    name: str,
    lanelet_ids: list[str],
    length_m: float,
    inside_m: float,
    undecided_m: float,
    outside_m: float,
    outside: list[tuple[str, float, list[str]]],
    undecided: list[str],
) -> list[str]:
    """The lines ``scenario-odd route`` prints for a route's coverage.

    *outside* holds each lanelet outside the ODD as its id, its length and
    the modules that fail on it; *undecided* the ids of the lanelets inside
    only by assumption. A route of more than 12 lanelets shows its first six
    and last five.
    """
    ids = lanelet_ids
    if len(ids) > 12:
        shown: list[str] = []
        for i in range(6):
            shown.append(ids[i])
        shown.append("...")
        for i in range(len(ids) - 5, len(ids)):
            shown.append(ids[i])
        ids = shown
    lines = [
        f"{name}: lanelets {' -> '.join(ids)} ({length_m:.1f} m)",
        f"  inside {inside_m:.1f} m, undecided {undecided_m:.1f} m, "
        f"outside {outside_m:.1f} m",
    ]
    for lanelet_id, lanelet_length_m, failing_modules in outside:
        modules = ", ".join(failing_modules)
        if not modules:
            modules = "-"
        lines.append(
            f"  OUTSIDE lanelet {lanelet_id} ({lanelet_length_m:.1f} m): "
            f"modules {modules}"
        )
    if undecided:
        lines.append("  inside only by assumption: lanelets " + ", ".join(undecided))
    return lines
