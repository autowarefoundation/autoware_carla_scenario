"""The names a versioned measure's cover items are written under, and what a
merged item's name splits into.

A cover item of an ODD attribute mapped onto a measure past its first version
is named for that version (``odd.dynamic.vehicle_ahead_gap@v2``), so
openodd's merge, which keys an item by its name, never adds hits of one
definition to another's (docs/odd.md, "Measure versions").  These are plain
string functions, split out of :mod:`~autoware_carla_scenario.coverage.collector`
(which derives openodd's collector, a Rust extension) so the library check
compiles them; the collector re-exports them.
"""

from __future__ import annotations

__all__ = ["VERSION_MARK", "versioned_item_name"]

#: What a cover item's name is suffixed with for its measure's version.
VERSION_MARK = "@v"


def versioned_item_name(name: str, version: int) -> str:
    """The name a cover item of a measure of *version* is written under:
    *name* itself for the first version, ``name@v<version>`` after it.

    Not ``name#<version>``: ``#<n>`` is the suffix openodd's merge gives the
    n-th differing definition of an item, and a v1 item whose buckets changed
    would take ``#2`` too; openodd's sampler, besides, pools an item's ``#<n>``
    definitions.  :func:`~autoware_carla_scenario.coverage.collector.current_measure_coverage`
    hands a sampler only the current version's."""
    if version <= 1:
        return name
    return f"{name}{VERSION_MARK}{version}"


def _split_versioned(name: str) -> tuple[str, int, str]:
    """``(base, version, rest)`` of a merged item's name: ``name@v2#3`` is
    ``("name", 2, "#3")``, ``name#3`` is ``("name", 1, "#3")``."""
    head = name
    rest = ""
    hash_at = name.find("#")
    if hash_at >= 0:
        head, rest = name[:hash_at], name[hash_at:]
    mark_at = head.rfind(VERSION_MARK)
    if mark_at >= 0 and head[mark_at + len(VERSION_MARK) :].isdigit():
        return head[:mark_at], int(head[mark_at + len(VERSION_MARK) :]), rest
    return head, 1, rest
