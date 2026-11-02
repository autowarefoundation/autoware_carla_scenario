"""``scenario-expand``: print the concrete scenarios of a scenario as JSON.

Takes the same arguments as ``scenario`` -- ``scenario=<name>`` plus any Hydra
overrides, e.g. ``map=<group>`` -- composes the config exactly as a run would
(external scenario packages included), and prints::

    {"scenario": "<name>", "cases": [["ego.spawn_lanelet_id=242", ...], ...]}

Each case is the overrides that, added to the same arguments, run one concrete
scenario. A logical scenario (one with ``sweep.constraints``) expands to one
case per matching lanelet, in a fixed order; a concrete one to a single empty
case. Nothing is run and no CARLA server is needed -- only the Lanelet2 map,
which is fetched on demand like a run would. Logs go to stderr, so stdout is
the JSON alone.

    scenario-expand scenario=traffic_light_compliance/traffic_light_compliance

As for ``scenario``, ``scenario=`` may be a glob (``'intersection_passing/*'``):
each scenario it matches is expanded with the same overrides and printed as
one such JSON object per line (JSON Lines), in the order a batch runs them
(sorted by name), each headed by its own ``"scenario"``. A single scenario is
the one-line case of that. Every matched scenario is composed and expanded
before anything is printed, so stdout is either every listing or nothing.

The exit status is 0 when every scenario expanded, 1 when the ``controls:`` of
one are invalid, and 2 when the command line does not compose: no
``scenario=``, no config matches it, or an override names a key a config does
not have (the message suggests the closest config or key; ``+key=value`` adds
a new one). Every scenario's problems are reported, and the higher status wins.
``HYDRA_FULL_ERROR=1`` shows the traceback of such an error.
"""

from __future__ import annotations

import json
import logging
import sys

from ..registry import load_scenario_plugins
from ..sweeper.expand import expand_config


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)  # noqa: T201
        return 0
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    scenario = next(
        (a.split("=", 1)[1] for a in args if a.startswith("scenario=")), None
    )
    if not scenario:
        print(
            "usage: scenario-expand scenario=<name> [hydra overrides...]",
            file=sys.stderr,
        )
        return 2
    overrides = [a for a in args if not a.startswith("scenario=")]

    load_scenario_plugins()
    # Composed the way a glob batch run composes each scenario, so the
    # expansion sees the same config (package conf dirs included).
    from ..odd.controls import ControlsError, format_diagnostics  # noqa: PLC0415
    from ..scenario_glob import is_scenario_glob  # noqa: PLC0415
    from .compose_errors import compose_or_report, resolve_scenarios  # noqa: PLC0415

    names = resolve_scenarios(scenario)
    if names is None:
        return 2
    globbed = is_scenario_glob(scenario)
    if not globbed:
        # A plain name is composed (and printed) as given, as a run does.
        names = [scenario]
    listings: list[dict[str, object]] = []
    status = 0
    for name in names:
        cfg = compose_or_report(name, overrides, named=globbed)
        if cfg is None:
            status = 2
            continue
        try:
            cases = expand_config(cfg)
        except ControlsError as exc:
            # An authoring error: one line per problem, with its config path.
            for line in format_diagnostics(exc.diagnostics, name):
                print(line, file=sys.stderr)
            status = max(status, 1)
            continue
        listings.append({"scenario": name, "cases": cases})
    if status:
        return status
    for listing in listings:
        json.dump(listing, sys.stdout)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
