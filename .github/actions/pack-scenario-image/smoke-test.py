"""Verify a freshly built scenario image from the inside.

Run with the image's own interpreter::

    docker run --rm --entrypoint python \
        -v "$PWD/.github/actions/pack-scenario-image/smoke-test.py:/tmp/smoke-test.py:ro" \
        my-scenario:latest /tmp/smoke-test.py

It asserts the properties the image exists to guarantee: the CARLA client,
typesafe_carla's prebuilt CPython package, is in the image and loads
without compiling anything, the official `carla` package is not there, the
scenario package's entry point registers its scenario, and Hydra can reach
that package's config directory, which exists and holds its scenario configs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path


def check_conf_dirs(conf_dirs: Sequence[Path], framework_conf_dir: Path) -> str | None:
    """Return why the registered config directories fall short, or None.

    The framework's own directory is always registered and always holds
    YAML, so it proves nothing about the scenario package and is ignored.
    ``register_conf_dir`` does not check that a path exists, and a generated
    package registers ``Path(__file__).parent / "conf"`` whether or not its
    wheel shipped it, so a directory merely being registered proves nothing
    either: at least one plugin directory has to exist and hold a YAML under
    its ``scenario/`` group, the one ``scenario=<name>/...`` selects from.
    """
    framework = framework_conf_dir.resolve()
    plugin_dirs = [dir_ for dir_ in conf_dirs if dir_.resolve() != framework]
    if not plugin_dirs:
        return (
            "No scenario package registered a config directory: only the "
            f"framework's own ({framework}) is on the search path."
        )
    for dir_ in plugin_dirs:
        scenario_group = dir_ / "scenario"
        if scenario_group.is_dir() and any(scenario_group.rglob("*.yaml")):
            return None
    shown = ", ".join(str(dir_) for dir_ in plugin_dirs)
    return (
        "No config directory registered by a scenario package holds a YAML "
        f"under scenario/ -- is conf/ missing from the wheel? Registered: {shown}"
    )


def main(argv: list[str]) -> int:
    """Check the prebuilt CARLA client and the registered scenarios."""
    if len(argv) != 1:
        print(f"usage: {argv[0]}", file=sys.stderr)
        return 2

    if os.environ.get("TYPESAFE_CARLA_PYCARLA_BUILD") != "0":
        print(
            "TYPESAFE_CARLA_PYCARLA_BUILD is not 0: a container whose prebuilt "
            "CARLA client did not apply would compile it on its first import",
            file=sys.stderr,
        )
        return 1
    try:
        import typesafe_carla.carla as carla
    except ImportError as exc:
        print(f"The prebuilt CARLA client does not load: {exc}", file=sys.stderr)
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
        _FRAMEWORK_CONF_DIR,
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
    problem = check_conf_dirs(conf_dirs, _FRAMEWORK_CONF_DIR)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 1

    print(f"CARLA client: {installed}")
    print(f"Registered scenarios: {', '.join(scenarios)}")
    print(f"Config directories: {', '.join(str(d) for d in conf_dirs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
