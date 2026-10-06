"""The SUMO network of the road network a CARLA world runs.

The OpenDRIVE comes from the world itself (``world.get_map().to_opendrive()``),
or from the file the run installed, and is converted by roadgen
(https://github.com/hakuturu583/hdmap_generator) into netconvert's plain-XML
input, which netconvert then builds into a ``.net.xml``.  roadgen writes the
network without offset normalisation, so its x/y are OpenDRIVE's.

Ambient demand comes from SUMO's own ``randomTrips.py``, weighted by the
``network.safe.*`` files roadgen writes beside the network (sources and sinks
away from dead ends).

Beside the network go roadgen's traces -- the IR dump, what each element of the
OpenDRIVE became (``map.xodr.trace.json``) and what each became in SUMO -- which
join a CARLA traffic light, by its OpenDRIVE signal id, to the SUMO signal links
it switches (:class:`SignalTable`).

Both are cached under a directory named after a hash of the OpenDRIVE text and
of the roadgen and SUMO versions, so a map is converted once per machine.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import TrafficBackendError, TrafficBackendUnavailable

logger = logging.getLogger(__name__)

__all__ = [
    "SignalTable",
    "SumoNetwork",
    "build_network",
    "generate_trips",
    "sanitize_opendrive",
    "sumo_home",
]

_INSTALL = "install the `sumo` extra (autoware-carla-scenario[sumo])"


@dataclass(frozen=True)
class SumoNetwork:
    """A built network and the files that go with it."""

    net_file: Path
    #: ``randomTrips --weights-prefix``: ``<prefix>.src.xml`` and friends.
    weights_prefix: Path

    @property
    def trace_files(self) -> tuple[Path, Path, Path]:
        """The IR dump, the OpenDRIVE's read trace and the SUMO trace."""
        directory = self.net_file.parent
        return (
            directory / "network.ir.json",
            directory / "map.xodr.trace.json",
            directory / "network.sumo.trace.json",
        )


class SignalTable:
    """Which SUMO signal links each OpenDRIVE signal switches, from roadgen's traces.

    The OpenDRIVE is CARLA's own, so a signal's id there is
    ``TrafficLight.get_opendrive_id()``: the table is the CARLA light to the
    ``(tls id, link index)`` of every movement it governs in SUMO.
    """

    def __init__(self, trace: Any) -> None:
        self._trace = trace

    @classmethod
    def load(cls, network: SumoNetwork) -> SignalTable | None:
        """The table of *network*, or ``None`` when it was built without traces."""
        if not all(path.is_file() for path in network.trace_files):
            return None
        import roadgen  # noqa: PLC0415

        # The files are this cache's own and never change, but they were renamed
        # from roadgen's prefix to `network.*` after the traces recorded them.
        files = (str(path) for path in network.trace_files)
        return cls(roadgen.Trace.load(*files, check_files=False))

    def links(self, signal_id: str) -> list[tuple[str, int]]:
        """The ``(tls id, link index)`` pairs the signal *signal_id* switches."""
        try:
            answers = self._trace.translate(
                "opendrive", f"signal:{signal_id}", to="sumo"
            )
        except ValueError:  # an id roadgen cannot read; one it lacks gives []
            return []
        links = set()
        for answer in answers:
            ref = answer["ref"]
            if answer.get("role") != "link" or not ref.startswith("tls:"):
                continue
            tls, _, index = ref[len("tls:") :].rpartition("/")
            links.add((tls, int(index)))
        return sorted(links)


def sumo_home() -> Path:
    """The SUMO installation: the first of ``$SUMO_HOME`` and the ``eclipse-sumo``
    package that holds SUMO's binaries and tools.

    ``$SUMO_HOME`` alone is not trusted: importing libsumo points it at the
    ``sumo-data`` package, which holds neither.
    """
    candidates = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]))
    try:
        import sumo  # noqa: PLC0415 - the eclipse-sumo package
    except ImportError:
        pass
    else:
        candidates += [Path(sumo.SUMO_HOME), Path(sumo.__file__).parent]
    for home in candidates:
        if (home / "tools" / "randomTrips.py").is_file() and (home / "bin").is_dir():
            return home
    raise TrafficBackendUnavailable(f"SUMO is not installed: {_INSTALL}")


def _binary(name: str) -> str:
    path = sumo_home() / "bin" / name
    if path.is_file():
        return str(path)
    found = shutil.which(name)
    if found is None:
        raise TrafficBackendUnavailable(f"SUMO's {name} was not found: {_INSTALL}")
    return found


def sanitize_opendrive(text: str) -> str:
    """Make CARLA's OpenDRIVE acceptable to roadgen's OpenDRIVE 1.7 parser.

    CARLA's maps (made with RoadRunner) bend the schema in four places the
    strict parser refuses the whole document for, none of which carries road
    geometry: ``<userData>`` without a ``code``, ``<roadMark>`` without a
    ``color``, objects of type ``-1`` and ``<cornerLocal>`` without a
    ``height``.
    """
    text = re.sub(r"<userData\b[^>]*/>|<userData\b.*?</userData>", "", text, flags=re.S)
    text = re.sub(r'(<object [^>]*?)type="-1"', r'\1type="none"', text)
    text = re.sub(r"<roadMark (?![^>]*color=)", '<roadMark color="standard" ', text)
    return re.sub(r"<cornerLocal (?![^>]*height=)", '<cornerLocal height="0" ', text)


def _versions() -> str:
    out = []
    for dist in ("roadgen", "eclipse-sumo"):
        try:
            out.append(f"{dist}={importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            out.append(f"{dist}=?")
    return ",".join(out)


def build_network(
    opendrive: str, cache_dir: Path, curve_lateral_acceleration: float | None = None
) -> SumoNetwork:
    """The SUMO network of *opendrive* (OpenDRIVE text), built or cached.

    Args:
        curve_lateral_acceleration: Hold traffic in bends to this lateral
            acceleration, m/s² (roadgen's option of the same name); ``None``
            converts the map as it is.

    Raises:
        TrafficBackendUnavailable: roadgen or SUMO is not installed.
        TrafficBackendError: The conversion failed.
    """
    try:
        import roadgen  # noqa: PLC0415
    except ImportError as exc:
        raise TrafficBackendUnavailable(
            f"roadgen is not installed: {_INSTALL}"
        ) from exc
    options = f"curve_lateral_acceleration={curve_lateral_acceleration}"
    key = hashlib.sha256(
        "\0".join((_versions(), options, opendrive)).encode()
    ).hexdigest()[:16]
    target = cache_dir / key
    network = SumoNetwork(target / "network.net.xml", target / "network.safe")
    if network.net_file.is_file():
        return network

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_dir))
    try:
        xodr = tmp / "map.xodr"
        xodr.write_text(sanitize_opendrive(opendrive))
        try:
            road_map = roadgen.read_opendrive(str(xodr))
            prefix = _export(road_map, tmp, curve_lateral_acceleration)
            road_map.export_ir(str(tmp / "network.ir.json"))
            road_map.write_read_trace(str(tmp / f"{xodr.name}.trace.json"))
        except Exception as exc:  # roadgen raises ValueError, among others
            raise TrafficBackendError(
                f"roadgen could not convert the map: {exc}"
            ) from exc
        for warning in road_map.sumo_warnings():
            logger.debug("roadgen: %s", warning)
        proc = subprocess.run(  # noqa: S603 - a fixed tool invocation
            [_binary("netconvert"), "-c", f"{prefix}.netccfg"],
            cwd=tmp,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not (tmp / f"{prefix}.net.xml").is_file():
            raise TrafficBackendError(
                f"netconvert failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}"
            )
        if prefix != "network":
            # roadgen names what it writes after the map, which may itself be
            # `map`; the OpenDRIVE and its read trace are this function's own.
            own = {xodr.name, f"{xodr.name}.trace.json"}
            for path in tmp.glob(f"{prefix}.*"):
                if path.name not in own:
                    path.rename(tmp / path.name.replace(prefix, "network", 1))
        try:
            os.rename(tmp, target)  # atomic; another process may have won
        except OSError:
            if not network.net_file.is_file():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    logger.info("Built the SUMO network %s", network.net_file)
    return network


def _export(
    road_map: Any, directory: Path, curve_lateral_acceleration: float | None
) -> str:
    if curve_lateral_acceleration is None:
        return road_map.export_sumo(str(directory))
    try:
        return road_map.export_sumo(
            str(directory), curve_lateral_acceleration=curve_lateral_acceleration
        )
    except TypeError:
        logger.warning(
            "roadgen %s cannot hold traffic to curve speeds; converting without "
            "(SUMO traffic will take bends at the speed limit)",
            _versions(),
        )
        return road_map.export_sumo(str(directory))


def generate_trips(
    network: SumoNetwork,
    out: Path,
    *,
    seed: int,
    period: float,
    end: float,
    safe_weights: bool = True,
    fringe_factor: float = 5.0,
) -> Path:
    """Ambient demand for *network*, written to *out* (a ``.rou.xml``).

    Weighted by roadgen's ``network.safe.*`` files when *safe_weights* and the
    network has them, by *fringe_factor* otherwise.
    """
    tool = sumo_home() / "tools" / "randomTrips.py"
    if safe_weights and Path(f"{network.weights_prefix}.src.xml").is_file():
        weights = ["--weights-prefix", str(network.weights_prefix)]
    else:
        weights = ["--fringe-factor", f"{fringe_factor:g}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    trips = out.with_suffix(".trips.xml")
    cmd = [
        sys.executable,
        str(tool),
        "-n",
        str(network.net_file),
        "-o",
        str(trips),
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
        "--vehicle-class",
        "passenger",
        "--prefix",
        "sumo",
        *weights,
    ]
    env = dict(os.environ, SUMO_HOME=str(sumo_home()))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)  # noqa: S603
    if proc.returncode != 0 or not out.is_file():
        raise TrafficBackendError(
            f"randomTrips.py failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}"
        )
    return out
