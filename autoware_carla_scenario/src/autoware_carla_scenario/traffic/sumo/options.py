"""The SUMO backend's command-line options, mode bits, vehicle types and type files.

Where a run's SUMO outputs go, the speed and lane-change mode bits the backend
sets, the types a scenario's vehicle joins SUMO as, the additional files that
declare them, SUMO's command line merged with a configuration's extra
arguments, and the command line of ``randomTrips.py`` with the options of
each demand it generates.  :mod:`.backend` re-exports the backend's part of
it, and :mod:`.network` runs ``randomTrips.py`` with the rest; it is split out
because it is plain-typed, so the library check (``typecheck/library.py``)
compiles it, while the backend holds SUMO's connection and CARLA's actors
typed ``Any`` and the network imports roadgen and sumo.  Like
:mod:`.signal_states`, it imports neither simulator.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..context import TrafficContext

__all__ = [
    "EMERGENCY_TYPE",
    "LANE_CHANGE_DURATION_S",
    "LANE_CHANGE_SETTLED_M",
    "MOTORCYCLE_TYPE",
    "OUTPUT_SUFFIX",
    "random_trips_argv",
    "ride_options",
    "run_output_dir",
    "trip_options",
    "walk_options",
]

#: What a run's SUMO outputs (its log, the generated demand, the FCD and the
#: clock) are named by: the directory ``{output_dir}/{file_stem}.sumo``, listed
#: in ``run_outputs.RUN_DIR_SUFFIXES``.
OUTPUT_SUFFIX = ".sumo"


def run_output_dir(context: TrafficContext) -> Path:
    """Where the run *context* names writes its SUMO outputs.

    ``{output_dir}/{file_stem}.sumo``: a batch shares the output directory, so
    each run keeps its own, beside its result (#1544).  A context naming no run
    (a backend prepared on its own) writes to ``{output_dir}/sumo``.
    """
    if context.file_stem:
        return Path(context.output_dir) / f"{context.file_stem}{OUTPUT_SUFFIX}"
    return Path(context.output_dir) / "sumo"


#: SUMO's lane-change duration for a manoeuvre a scenario asks for, seconds.
LANE_CHANGE_DURATION_S: float = 3.0
#: How close to its lane's centre a vehicle must be for a lane change to count
#: as settled, metres.
LANE_CHANGE_SETTLED_M: float = 0.3
#: Speed mode bits of a vehicle SUMO does not control (all checks off).
_SPEED_MODE_EXTERNAL = 0
#: Lane-change mode bits of a vehicle SUMO does not control (none).
_LANE_CHANGE_MODE_EXTERNAL = 0
#: Lane-change mode bits of a vehicle that changes lane only where its route
#: needs it: SUMO's default (1621) without the cooperative, speed-gain and
#: keep-right changes.
_LANE_CHANGE_MODE_STRATEGIC = 1537
#: SUMO's own lane-change mode bits, which ``lane_changes: true`` puts back.
_LANE_CHANGE_MODE_DEFAULT = 1621
#: SUMO's own speed mode bits (every check on).
_SPEED_MODE_DEFAULT = 31
#: The speed mode bit by which a vehicle disregards the right of way of
#: vehicles already in the junction, and the one by which it regards that of
#: vehicles coming to it (off: vehicle_40, waiting short of its left turn's
#: own stop, took those standing across the turn for ones coming).
_SPEED_MODE_THROUGH_JUNCTION = 32
_SPEED_MODE_RIGHT_OF_WAY = 8
#: The speed mode bit by which a vehicle brakes for a red light: off for an
#: emergency vehicle SUMO drives while on a call, which still gives way at
#: the junction (:data:`_SPEED_MODE_RIGHT_OF_WAY`).
_SPEED_MODE_RED_LIGHT = 16

#: SUMO's default bicycle type.
_BICYCLE_TYPE = "DEFAULT_BIKETYPE"
#: The type a scenario's motorcycle joins SUMO as.  SUMO has no default
#: motorcycle type, so every run declares this one: of the class
#: ``motorcycle``, with SUMO's defaults for it (:func:`_write_scenario_types`).
MOTORCYCLE_TYPE = "acs_motorcycle"
#: A scenario vehicle's blueprint ``base_type`` -> the SUMO vehicle class and
#: type it joins SUMO as; any other is a passenger car (:data:`_PASSENGER`).
_SUMO_CLASSES = {
    "bicycle": ("bicycle", _BICYCLE_TYPE),
    "motorcycle": ("motorcycle", MOTORCYCLE_TYPE),
}
_PASSENGER = ("passenger", "DEFAULT_VEHTYPE")
#: The type a scenario's emergency vehicle joins SUMO as: of the class
#: ``emergency``, with a bluelight device (:func:`_write_scenario_types`),
#: whatever its blueprint's ``base_type`` (a police car, an ambulance's van,
#: a fire engine's truck).
EMERGENCY_TYPE = "acs_emergency"
_EMERGENCY = ("emergency", EMERGENCY_TYPE)


def _net_offset(net_file: Path) -> tuple[float, float]:
    """The ``netOffset`` of *net_file*, read from its ``<location>``."""
    with net_file.open() as f:
        for line in f:
            found = re.search(r'netOffset="([-\d.eE+]+),([-\d.eE+]+)"', line)
            if found:
                return float(found[1]), float(found[2])
            if "<edge" in line:
                break
    return 0.0, 0.0


def _with_siren(mode: int, on: bool) -> int:
    """Speed mode *mode* of an emergency vehicle on a call (*on*: it does not
    brake for red lights) or off duty (it does again: every emergency vehicle
    joins with SUMO's own speed mode, which does)."""
    if on:
        return mode & ~_SPEED_MODE_RED_LIGHT
    return mode | _SPEED_MODE_RED_LIGHT


def _write_scenario_types(path: Path) -> Path:
    """Declare in *path* (an additional file) the types SUMO has no default
    for that a scenario's vehicle may join as: :data:`MOTORCYCLE_TYPE`, of the
    class ``motorcycle``, and :data:`EMERGENCY_TYPE`, of the class
    ``emergency`` with a bluelight device; every other attribute SUMO's
    default for the class."""
    path.write_text(
        "<additional>\n"
        f'    <vType id="{MOTORCYCLE_TYPE}" vClass="motorcycle" guiShape="motorcycle"/>\n'
        f'    <vType id="{EMERGENCY_TYPE}" vClass="emergency" guiShape="emergency">\n'
        '        <param key="has.bluelight.device" value="true"/>\n'
        "    </vType>\n"
        "</additional>\n"
    )
    return path


#: SUMO's spellings of the options naming its additional and route files.
_ADDITIONAL_OPTIONS = ("-a", "--additional-files", "--additional")
_ROUTE_OPTIONS = ("-r", "--route-files", "--routes")


def _merge_file_options(
    sumo_args: list[str], additional: list[str], routes: list[str]
) -> list[str]:
    """SUMO's command-line arguments for the backend's own *additional* and
    *routes* files and a configuration's extra *sumo_args*.

    SUMO refuses an option given twice, so an additional or route file option
    in *sumo_args* (``-a X``, ``--additional-files X``, ``--additional X`` or
    ``--additional-files=X``; likewise ``-r``, ``--route-files`` and
    ``--routes``) is taken out of it and its comma-separated files appended to
    the backend's own: theirs first, so the files given in *sumo_args* can use
    the types they declare.  Every other argument keeps its order, after the
    files.  An option without a value is passed on as it is, for SUMO to
    reject.
    """
    files: dict[str, list[str]] = {"-a": list(additional), "-r": list(routes)}
    rest: list[str] = []
    index = 0
    while index < len(sumo_args):
        arg = sumo_args[index]
        index += 1
        name, equals, value = arg.partition("=")
        if name in _ADDITIONAL_OPTIONS:
            key = "-a"
        elif name in _ROUTE_OPTIONS:
            key = "-r"
        else:
            rest.append(arg)
            continue
        if not equals:
            if index >= len(sumo_args):
                rest.append(arg)
                continue
            value = sumo_args[index]
            index += 1
        files[key] += [f for f in value.split(",") if f]
    merged: list[str] = []
    for key in ("-a", "-r"):
        if files[key]:
            merged += [key, ",".join(files[key])]
    return merged + rest


def _write_type_distribution(
    path: Path, type_id: str, vclass: str, members: list[dict[str, float]]
) -> Path:
    """Write *type_id* as a distribution of one type of *vclass* per member,
    each with that member's attributes, to *path* (an additional file)."""
    lines = []
    for index, attributes in enumerate(members):
        written = " ".join(f'{key}="{value:.4g}"' for key, value in attributes.items())
        lines.append(
            f'        <vType id="{vclass}{index}" vClass="{vclass}" {written}/>'
        )
    path.write_text(
        "<additional>\n"
        f'    <vTypeDistribution id="{type_id}">\n'
        + "\n".join(lines)
        + "\n    </vTypeDistribution>\n"
        "</additional>\n"
    )
    return path


def random_trips_argv(
    python: str,
    tool: Path,
    net_file: Path,
    out: Path,
    seed: int,
    period: float,
    end: float,
    prefix: str,
    options: list[str],
) -> list[str]:
    """The command running SUMO's ``randomTrips.py`` (*tool*, with the
    interpreter *python*) on *net_file*, validated routes into *out* (a
    ``.rou.xml``, its trips beside it), with *options* after the ones every
    demand shares (:func:`trip_options`, :func:`walk_options`,
    :func:`ride_options`)."""
    cmd = [
        python,
        str(tool),
        "-n",
        str(net_file),
        "-o",
        str(out.with_suffix(".trips.xml")),
        "-r",
        str(out),
        "-b",
        "0",
        "-e",
        str(end),
        "-p",
        str(period),
        "--seed",
        str(seed),
        "--validate",
        "--prefix",
        prefix,
    ]
    cmd.extend(options)
    return cmd


def trip_options(
    weights_prefix: Path, safe_weights: bool, fringe_factor: float
) -> list[str]:
    """``randomTrips.py``'s options for ambient demand: passenger cars,
    weighted by roadgen's files at *weights_prefix* when *safe_weights* (the
    caller found them), by *fringe_factor* otherwise."""
    options = ["--vehicle-class", "passenger"]
    if safe_weights:
        options.extend(["--weights-prefix", str(weights_prefix)])
    else:
        options.extend(["--fringe-factor", f"{fringe_factor:g}"])
    return options


def walk_options(max_distance: float) -> list[str]:
    """``randomTrips.py``'s options for pedestrian demand, of walks no longer
    than *max_distance* metres."""
    return ["--pedestrians", "--max-distance", f"{max_distance:g}"]


def ride_options(
    out: Path,
    min_distance: float,
    max_distance: float | None,
    bicycle_type: Path | None,
) -> list[str]:
    """``randomTrips.py``'s options for bicycle demand written to *out*: rides
    of SUMO's default bicycle type, ``DEFAULT_BIKETYPE``, between edges a
    bicycle may use, at least *min_distance* metres long and at most
    *max_distance* (``None``: no bound); *bicycle_type* is an additional file
    redefining that type, which the routing has to know."""
    options = [
        # From and to edges a bicycle may use; the type is SUMO's own, which
        # --vehicle-class would shadow with one of randomTrips'.
        "--vclass",
        "bicycle",
        "--trip-attributes",
        'type="DEFAULT_BIKETYPE"',
        "--min-distance",
        f"{min_distance:g}",
    ]
    if max_distance is not None:
        options.extend(["--max-distance", f"{max_distance:g}"])
    if bicycle_type is not None:
        # The routing writes the types its rides drew beside them rather than
        # into them: SUMO loads the type file itself, and refuses a type
        # defined twice.
        options.extend(
            [
                "--additional-files",
                str(bicycle_type),
                "--vtype-output",
                str(out.with_suffix(".vtypes.xml")),
            ]
        )
    return options
