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
Scenarios that draw from one ODD (``sweep.odd_sample``) go on from each
other's draws, as in a batch, so a scenario's listing in a glob can differ
from its listing alone: each is the cases the batch run of the same glob runs.

As for ``scenario`` (and Hydra), the last ``scenario=`` wins when it is
given more than once.

The exit status is 0 when every scenario expanded, 1 when one is composed on
another map than its document names (an exported scenario on a map of its own
run without ``map=<id>_map``, refused before any map is read), the
``controls:`` of one are invalid, its sweep is invalid (``Error: <name>: <why>``: a
``sweep.case_index`` past its cases, a constraint or binding that does not
parse, constraints for more than one key, a Lanelet2 map that is missing,
unreadable or cannot be resolved, ...) or its sweep expands to no case
(its constraints, route or ODD sample match nothing on the map; a scenario
without a sweep is one case), and 2 when the command line does not compose: no
``scenario=``, no config matches it, or an override names a key a config does
not have (the message suggests the closest config or key; ``+key=value`` adds
a new one). Every scenario's problems are reported, and the higher status wins.
``HYDRA_FULL_ERROR=1`` shows the traceback of a command line that does not
compose or of an invalid sweep.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from ..registry import load_scenario_plugins
from ..sweeper.expand import EMPTY_SWEEP_MESSAGE, expand_config
from .argv import _extract_scenario_override


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
    # Parsed as `scenario` parses it: the last `scenario=` wins, as in Hydra,
    # and none of them is passed on as an override.
    scenario, overrides = _extract_scenario_override(["scenario-expand", *args])
    if not scenario:
        print(
            "usage: scenario-expand scenario=<name> [hydra overrides...]",
            file=sys.stderr,
        )
        return 2

    load_scenario_plugins()
    # Composed the way a glob batch run composes each scenario, so the
    # expansion sees the same config (package conf dirs included).
    from ..authoring.hydra_config import scenario_map_refusal  # noqa: PLC0415
    from ..odd.controls import ControlsError, format_diagnostics  # noqa: PLC0415
    from ..odd.sampler import OddDraws  # noqa: PLC0415
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
    # One record of the ODD draws for the whole glob, as a batch keeps: a
    # later scenario on the same ODD goes on from the earlier ones' draws, so
    # each listing is the cases the batch would run for it. A scenario that
    # fails draws nothing, and a failure prints no listing at all.
    drawn = OddDraws()
    for name in names:
        cfg = compose_or_report(name, overrides, named=globbed)
        if cfg is None:
            status = 2
            continue
        # Before the sweep resolves, fetches or searches any map: a scenario
        # composed on another map than its document names would list that
        # map's cases (#1428).
        refusal = scenario_map_refusal(cfg)
        if refusal is not None:
            print(f"Error: {name}: {refusal}", file=sys.stderr)
            status = max(status, 1)
            continue
        try:
            cases = expand_config(cfg, drawn=drawn)
        except ControlsError as exc:
            # An authoring error: one line per problem, with its config path.
            for line in format_diagnostics(exc.diagnostics, name):
                print(line, file=sys.stderr)
            status = max(status, 1)
            continue
        except ValueError as exc:
            # An invalid sweep (a case_index past its cases, a constraint or
            # binding that does not parse, a map that is missing or cannot be
            # resolved, ...): expand_config's documented refusal.
            # HYDRA_FULL_ERROR=1 raises it, traceback and all.
            if os.environ.get("HYDRA_FULL_ERROR") == "1":
                raise
            print(f"Error: {name}: {exc}", file=sys.stderr)
            status = max(status, 1)
            continue
        if not cases:
            # Refused as the lanelet sweeper and a batch refuse it (#1407):
            # a fan-out of no case would test nothing and look like a pass.
            print(f"Error: {name}: {EMPTY_SWEEP_MESSAGE}", file=sys.stderr)
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
