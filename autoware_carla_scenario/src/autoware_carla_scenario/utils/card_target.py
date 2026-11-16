"""The module an editor card's ``"..module:Name"`` target names.

A card's target is a class named relative to the package: ``..`` is the
package, and each further ``..`` a dot (``"..pedestrian..ardy:ArdyBackend"``
is ``autoware_carla_scenario.pedestrian.ardy``).  The preparer
(:mod:`..prepare.declare`) and the code generator (:mod:`..authoring.codegen`)
both resolve targets by this rule; it is here so that neither imports the
other.
"""

from __future__ import annotations

__all__ = ["target_module"]

#: The package a target's module path is relative to.
PACKAGE = "autoware_carla_scenario"


def target_module(relative: str) -> str:
    """The absolute module path of a target's *relative* part (before its
    ``:``)."""
    return PACKAGE + relative[1:].replace("..", ".")
