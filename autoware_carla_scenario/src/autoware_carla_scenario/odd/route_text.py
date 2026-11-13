"""What ``scenario-odd route`` reads off its command line, and the metres it prints.

Its arguments split into the scenario configs it plans (a ``scenario=``
argument, or one with no ``=``) and the Hydra overrides it composes every one
of them with (any other argument with an ``=``), in the order given
(:func:`_split_items`); and the metres a route is expected to drive in each
bucket of an attribute, written ``<bucket> <metres> m`` (:func:`_format_buckets`).

Split out of ``odd.cli``, which re-exports :func:`_format_buckets` and expands
the scenario configs' globs (the registry, omegaconf), so that the library
check compiles it (``docs/typecheck.md``, "Checking the framework itself").
"""

from __future__ import annotations

from collections.abc import Sequence

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
