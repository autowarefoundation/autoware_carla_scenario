"""Extensions: vocabulary a separate package adds to scenarios.

A logical scenario is written in the vocabulary of the sweeper's constraints
and bindings and of the editor's specs.  A package that knows more about maps
or about a data source -- one that classifies junctions its own way, say --
adds to that vocabulary without this package depending on it::

    [project.entry-points."autoware_carla_scenario.extensions"]
    my_extension = "my_package:register"

``register`` is a zero-argument callable that calls
:func:`~autoware_carla_scenario.sweeper.constraints.register_constraint`,
:func:`~autoware_carla_scenario.sweeper.bindings.register_binding` and the
``register_*_spec`` functions of :mod:`autoware_carla_scenario.authoring.registry`.

Extensions are loaded on first use -- the first time a spec or a sweep type is
looked up -- so importing this package never imports a plugin.
"""

from __future__ import annotations

__all__ = ["EXTENSION_ENTRY_POINT_GROUP", "load_extensions"]

#: Entry-point group a package advertises an extension through.
EXTENSION_ENTRY_POINT_GROUP = "autoware_carla_scenario.extensions"

#: Whether the entry-point walk has already run in this process.
_loaded = False


def load_extensions() -> None:
    """Import every package advertising an extension.  Idempotent."""
    global _loaded
    if _loaded:
        return
    # Latched before the walk: a plugin's registration looks specs up too.
    _loaded = True

    from .registry import load_entry_point_plugins  # noqa: PLC0415

    load_entry_point_plugins(EXTENSION_ENTRY_POINT_GROUP, "extension")
