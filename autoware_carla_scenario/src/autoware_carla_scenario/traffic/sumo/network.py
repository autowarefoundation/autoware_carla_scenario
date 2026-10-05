"""The SUMO network of the road network a CARLA world runs.

The OpenDRIVE comes from the world itself (``world.get_map().to_opendrive()``),
or from the file the run installed, and is converted by roadgen
(https://github.com/hakuturu583/hdmap_generator) into netconvert's plain-XML
input, which netconvert then builds into a ``.net.xml``.  roadgen writes the
network without offset normalisation, so its x/y are OpenDRIVE's.

Ambient demand comes from SUMO's own ``randomTrips.py``, weighted by the
``network.safe.*`` files roadgen writes beside the network (sources and sinks
away from dead ends).

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

from ..base import TrafficBackendError, TrafficBackendUnavailable

logger = logging.getLogger(__name__)

__all__ = [
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


def build_network(opendrive: str, cache_dir: Path) -> SumoNetwork:
    """The SUMO network of *opendrive* (OpenDRIVE text), built or cached.

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
    key = hashlib.sha256((_versions() + "\0" + opendrive).encode()).hexdigest()[:16]
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
            prefix = road_map.export_sumo(str(tmp))
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
            for path in tmp.glob(f"{prefix}.*"):
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
