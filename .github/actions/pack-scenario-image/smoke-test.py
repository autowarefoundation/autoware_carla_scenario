"""Verify a freshly built scenario image from the inside.

Run with the image's own interpreter::

    docker run --rm --entrypoint python \
        -v "$PWD/.github/actions/pack-scenario-image/smoke-test.py:/tmp/smoke-test.py:ro" \
        my-scenario:latest /tmp/smoke-test.py

It asserts the properties the image exists to guarantee: the CARLA client,
typesafe_carla's CPython package, was compiled into the image and loads
without compiling anything, the official `carla` package is not there, the
scenario package's entry point registers its scenario, and Hydra can reach
that package's config directory.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from importlib.metadata import version


def main(argv: list[str]) -> int:
    """Check the compiled CARLA client and the registered scenarios."""
    if len(argv) != 1:
        print(f"usage: {argv[0]}", file=sys.stderr)
        return 2

    if os.environ.get("TYPESAFE_CARLA_PYCARLA_BUILD") != "0":
        print(
            "TYPESAFE_CARLA_PYCARLA_BUILD is not 0: a container would compile "
            "the CARLA client on its first import",
            file=sys.stderr,
        )
        return 1
    try:
        import typesafe_carla.carla as carla
    except ImportError as exc:
        print(f"The compiled CARLA client does not load: {exc}", file=sys.stderr)
        return 1
    carla.Transform(carla.Location(1.0, 2.0, 3.0))
    if importlib.util.find_spec("carla") is not None:
        print("The official `carla` package is installed in the image", file=sys.stderr)
        return 1
    from typesafe_carla import paths

    installed = (
        f"typesafe-carla {version('typesafe-carla')} "
        f"(CARLA {paths.native_info()['carla_git_ref']})"
    )

    # Imported lazily so a client problem is reported before the heavier import.
    from autoware_carla_scenario.registry import (
        get_conf_dirs,
        get_scenario_registry,
        load_scenario_plugins,
    )

    load_scenario_plugins()
    scenarios = sorted(get_scenario_registry())
    if not scenarios:
        print(
            "No scenario registered: the package's "
            "'autoware_carla_scenario.scenarios' entry point did not load.",
            file=sys.stderr,
        )
        return 1

    # The plugin lives in the `hydra_plugins` namespace package, which is easy
    # to drop from a wheel by accident; without it Hydra never sees the
    # scenario package's conf dir and every `scenario=<name>/...` fails.
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    search_path_plugins = {
        plugin.__name__ for plugin in Plugins.instance().discover(SearchPathPlugin)
    }
    if "AutowareScenarioSearchPathPlugin" not in search_path_plugins:
        print(
            "Hydra did not discover AutowareScenarioSearchPathPlugin: the "
            "hydra_plugins namespace package is missing from the image.",
            file=sys.stderr,
        )
        return 1

    conf_dirs = get_conf_dirs()
    if not any(dir_.is_dir() and any(dir_.rglob("*.yaml")) for dir_ in conf_dirs):
        print(
            f"None of the registered config directories holds a YAML: {conf_dirs}",
            file=sys.stderr,
        )
        return 1

    print(f"CARLA client: {installed}")
    print(f"Registered scenarios: {', '.join(scenarios)}")
    print(f"Config directories: {', '.join(str(d) for d in conf_dirs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
