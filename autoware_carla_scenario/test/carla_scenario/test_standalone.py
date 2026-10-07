"""Standalone scenario binaries (docs/standalone.md).

The runtime the binaries are compiled against re-implements the parts of the
framework a scenario runs; these tests hold it to the Python package:

* the source rewrite the build adds (``a or b``) keeps lines and semantics;
* the runtime declares every name the static check's model does, so a
  scenario that checks resolves its imports when it is built;
* Python's text formatting, which the runtime replaces Codon's with, prints
  what CPython prints;
* the coordinate conversions, on the geometry the build bakes, land where the
  Python package (Lanelet2 and pyxodr) puts them on the fixture map;
* a scenario builds into a bundle that runs where it is copied, and a scenario
  using API the runtime lacks is refused at the line that uses it.

Everything compiled is skipped where no Codon compiler is installed; running a
scenario needs a CARLA server and is not tested here.
"""

from __future__ import annotations

import ast
import importlib
import math
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from autoware_carla_scenario.standalone.transform import (
    RUNTIME_PRELUDE,
    rewrite_or,
    transform_for_runtime,
)
from autoware_carla_scenario.standalone.build import runtime_dir
from autoware_carla_scenario.standalone.driver import Pose, RunSettings
from autoware_carla_scenario.typecheck import available_toolchain
from autoware_carla_scenario.typecheck.check import model_dir

_REPO = Path(__file__).resolve().parents[3]
XODR_PATH = _REPO / "data" / "nishishinjuku_carla.xodr"
OSM_PATH = _REPO / "data" / "nishishinjuku.osm"

needs_codon = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)


# ---------------------------------------------------------------------------
# The source rewrite
# ---------------------------------------------------------------------------


def test_rewrite_or_keeps_lines_and_nests() -> None:
    source = textwrap.dedent(
        """\
        x = a or b
        y = (a or
             b or c)
        if p and (q or r):
            pass
        s = f"{a or b}"
        """
    )
    out = rewrite_or(source)
    assert out.count("\n") == source.count("\n")
    lines = out.splitlines()
    assert lines[0] == "x = (_acs_or(a, lambda: b))"
    assert "_acs_or(a, lambda: _acs_or(b, lambda: c))" in lines[1]
    assert "(_acs_or(q, lambda: r))" in lines[3]
    assert lines[5] == 's = f"{a or b}"'  # f-string internals are left alone
    ast.parse(out)


def test_rewrite_or_rewrites_inner_or_first() -> None:
    out = rewrite_or("v = f(a or b) or c\n")
    assert out == "v = (_acs_or(f((_acs_or(a, lambda: b))), lambda: c))\n"


def test_transform_for_runtime_keeps_the_prelude_one_line() -> None:
    out = transform_for_runtime("cfg = config or Default()\n")
    assert out.startswith(RUNTIME_PRELUDE)
    assert RUNTIME_PRELUDE.count("\n") == 1
    assert out.splitlines()[1] == "cfg = (_acs_or(config, lambda: Default()))"


# ---------------------------------------------------------------------------
# The runtime declares what the model declares
# ---------------------------------------------------------------------------

_RUNTIME = runtime_dir()
_TOP_LEVEL = re.compile(r"^(?:class|def) (\w+)|^(\w+)\s*(?::[^=]*)?=", re.MULTILINE)


def _top_level_names(path: Path) -> set[str]:
    """The public names *path* defines (the model's private helpers are its own)."""
    text = path.read_text()
    extended = set(re.findall(r"^@extend\nclass (\w+)", text, re.MULTILINE))
    names = {a or b for a, b in _TOP_LEVEL.findall(text)}
    return {n for n in names if not n.startswith("_") and n not in extended}


@pytest.mark.parametrize(
    "module",
    sorted(p.relative_to(model_dir()).as_posix() for p in model_dir().rglob("*.codon")),
)
def test_runtime_declares_every_model_name(module: str) -> None:
    runtime = _RUNTIME / module
    assert runtime.exists(), f"the runtime has no {module}"
    missing = _top_level_names(model_dir() / module) - _top_level_names(runtime)
    assert not missing, f"{module}: the runtime lacks {sorted(missing)}"


# ---------------------------------------------------------------------------
# Programs compiled against the runtime
# ---------------------------------------------------------------------------


def _run_codon(tmp_path: Path, program: str, *args: str) -> str:
    """Compile and run *program* against the runtime; its stdout."""
    from typesafe_carla import paths as tsc_paths

    from autoware_carla_scenario.standalone.build import (
        compile_program,
        prepare_runtime_workspace,
    )
    from autoware_carla_scenario.typecheck.toolchain import codon_environment

    tc = available_toolchain()
    assert tc is not None
    ws = tmp_path / "ws"
    prepare_runtime_workspace(ws)
    (ws / "prog.py").write_text(program)
    exe = tmp_path / "prog"
    proc = compile_program(ws, "prog.py", exe, toolchain=tc, rpath=None)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    env = codon_environment(tc, ws)
    env["TYPESAFE_CARLA_LIB"] = str(tsc_paths.native_library())
    run = subprocess.run(
        [str(exe), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return run.stdout


_FLOATS = [
    0.0,
    -0.0,
    3.0,
    0.1 + 0.2,
    1e-7,
    1e16,
    1e15,
    123456789.125,
    2.5,
    -1.5,
    1 / 3,
    1e22,
    5e-324,
    0.0001,
    0.00001,
    42.62405777575473,
    81654.99999999662,
]
_SPECS = [
    ".2f",
    ".1f",
    ".3f",
    "8.2f",
    "<8.1f",
    "+.2f",
    ".0f",
    "e",
    ".3e",
    "g",
    ".3g",
    "",
    "06.2f",
    "%",
    ".1%",
]
_INTS = [0, 5, -12, 123456]
_ISPECS = ["d", "5d", "03d", "<4d", "+d", ".2f"]
_PRINTF = [
    ("%s=%d", ("a", 5)),
    ("%.2f m/s", (3.14159,)),
    ("%5.1f|%-6s|%r", (2.0, "x", "q")),
    ("%d%%", (50,)),
    ("%s", (1.5,)),
    ("lanelet %d -> road '%s' lane=%d s=%.1f", (183, "12", -1, 5.0)),
]


@needs_codon
def test_formatting_matches_cpython(tmp_path: Path) -> None:
    body = [
        "import autoware_carla_scenario",
        f"FLOATS = {_FLOATS!r}",
        f"SPECS = {_SPECS!r}",
        f"INTS = {_INTS!r}",
        f"ISPECS = {_ISPECS!r}",
        "for x in FLOATS:",
        '    print("R", str(x), repr(x), f"{x}")',
        "    for sp in SPECS:",
        '        print("F", sp, x.__format__(sp))',
        "for n in INTS:",
        "    for sp in ISPECS:",
        '        print("I", sp, n.__format__(sp))',
    ]
    body += [f'print("P", {fmt!r} % {args!r})' for fmt, args in _PRINTF]
    expected = []
    for x in _FLOATS:
        expected.append(f"R {x} {x!r} {x}")
        expected += [f"F {sp} {format(x, sp)}" for sp in _SPECS]
    for n in _INTS:
        expected += [f"I {sp} {format(n, sp)}" for sp in _ISPECS]
    expected += [f"P {fmt % args}" for fmt, args in _PRINTF]
    assert _run_codon(tmp_path, "\n".join(body) + "\n").splitlines() == expected


@needs_codon
def test_or_helper_matches_python(tmp_path: Path) -> None:
    program = textwrap.dedent(
        """\
        from autoware_carla_scenario._lists import _acs_or

        class C:
            x: int
            def __init__(self, x: int = 1):
                self.x = x

        calls = [0]
        def side() -> int:
            calls[0] += 1
            return 7

        def pick(config: Optional[C] = None) -> C:
            return _acs_or(config, lambda: C())

        print(pick(C(5)).x, pick().x)
        print(_acs_or(3, lambda: side()), _acs_or(0, lambda: side()), calls[0])
        empty: Optional[str] = ""
        print(repr(_acs_or(empty, lambda: "dflt")), repr(_acs_or("x", lambda: "y")))
        print(_acs_or(False, lambda: True), _acs_or(List[int](), lambda: [1, 2]))
        """
    )
    assert _run_codon(tmp_path, program).splitlines() == [
        "5 1",
        "3 7 1",
        "'dflt' 'x'",
        "True [1, 2]",
    ]


# --- Coordinates against the Python package --------------------------------------


@pytest.fixture(scope="module")
def coordinate_cases(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, list[str]]:
    """Baked map, the cases, and what the Python package answers for each."""
    from autoware_carla_scenario.coordinate import transform as T
    from autoware_carla_scenario.coordinate.map_manager import MapManager
    from autoware_carla_scenario.coordinate.poses import (
        CarlaWorldPose,
        Lanelet2Pose,
        OpenDrivePose,
    )
    from autoware_carla_scenario.standalone.bake import bake_map

    root = tmp_path_factory.mktemp("standalone_coordinates")
    bundle = root / "map.acsmap"
    bake_map(XODR_PATH, OSM_PATH, bundle)
    MapManager.reset()
    mm = MapManager.get_instance()
    mm.initialize(XODR_PATH, OSM_PATH)
    try:
        rng = random.Random(7)
        cases: list[str] = []
        expected: list[str] = []

        def r(x: float) -> str:
            return repr(float(x))

        lanelet_ids = [lanelet.id for lanelet in mm.lanelet_map.laneletLayer]
        for lid in lanelet_ids[::9]:
            length = T.lanelet_length(lid)
            cases.append(f"len {lid}")
            expected.append(f"len {lid} {r(length)}")
            for _ in range(2):
                s, t, h = (
                    rng.uniform(-2, length + 2),
                    rng.uniform(-2, 2),
                    rng.uniform(-0.5, 0.5),
                )
                p = T.to_carla_world(Lanelet2Pose(lid, s, t, h))
                cases.append(f"l2c {lid} {r(s)} {r(t)} {r(h)}")
                expected.append(f"l2c {r(p.x)} {r(p.y)} {r(p.z)} {r(p.yaw)}")
                q = CarlaWorldPose(
                    p.x + rng.uniform(-1, 1),
                    p.y + rng.uniform(-1, 1),
                    p.z,
                    yaw=rng.uniform(-180, 180),
                )
                on = T.on_lanelet(q, lid)
                pr = T.project_onto_lanelet(q, lid)
                ll = T.to_lanelet2(q)
                cases.append(f"c2l {lid} {r(q.x)} {r(q.y)} {r(q.z)} {r(q.yaw)}")
                expected.append(
                    f"c2l {int(on)} {r(pr.s)} {r(pr.t)} {r(pr.heading)} "
                    f"{ll.lanelet_id} {r(ll.s)} {r(ll.t)} {r(ll.heading)}"
                )
        for rid in list(mm.road_network.road_ids_to_object)[::11]:
            ref = mm.road_network.road_ids_to_object[rid].reference_line
            if len(ref) < 2:
                continue
            arc = T._compute_arc_lengths_2d(ref)
            s, t = rng.uniform(0, float(arc[-1])), rng.uniform(-4, 4)
            p = T.to_carla_world(OpenDrivePose(rid, -1, s, t, 0.1))
            cases.append(f"o2c {rid} {r(s)} {r(t)}")
            expected.append(f"o2c {r(p.x)} {r(p.y)} {r(p.z)} {r(p.yaw)}")
    finally:
        MapManager.reset()
    case_file = root / "cases.txt"
    case_file.write_text("\n".join(cases) + "\n")
    return bundle, case_file, expected


_COORDINATE_PROGRAM = """\
import sys
from autoware_carla_scenario.coordinate import (load_map_data, Lanelet2Pose, OpenDrivePose,
    CarlaWorldPose, to_carla_world, on_lanelet, project_onto_lanelet, to_lanelet2,
    lanelet_length)

load_map_data(sys.argv[1])
for line in open(sys.argv[2]):
    f = line.strip().split(" ")
    if f[0] == "len":
        print("len", f[1], repr(lanelet_length(int(f[1]))))
    elif f[0] == "l2c":
        p = to_carla_world(Lanelet2Pose(int(f[1]), float(f[2]), float(f[3]), float(f[4])))
        print("l2c", repr(p.x), repr(p.y), repr(p.z), repr(p.yaw))
    elif f[0] == "c2l":
        q = CarlaWorldPose(float(f[2]), float(f[3]), float(f[4]), yaw=float(f[5]))
        lid = int(f[1])
        pr = project_onto_lanelet(q, lid)
        ll = to_lanelet2(q)
        print("c2l", 1 if on_lanelet(q, lid) else 0, repr(pr.s), repr(pr.t), repr(pr.heading),
              ll.lanelet_id, repr(ll.s), repr(ll.t), repr(ll.heading))
    elif f[0] == "o2c":
        p = to_carla_world(OpenDrivePose(f[1], -1, float(f[2]), float(f[3]), 0.1))
        print("o2c", repr(p.x), repr(p.y), repr(p.z), repr(p.yaw))
"""


def _close(expected: str, got: str) -> bool:
    a, b = expected.split(), got.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if not math.isclose(float(x), float(y), rel_tol=0.0, abs_tol=1e-9):
                return False
        except ValueError:
            return False
    return True


@needs_codon
def test_coordinates_match_the_python_package(
    tmp_path: Path, coordinate_cases: tuple[Path, Path, list[str]]
) -> None:
    bundle, case_file, expected = coordinate_cases
    got = _run_codon(
        tmp_path, _COORDINATE_PROGRAM, str(bundle), str(case_file)
    ).splitlines()
    assert len(got) == len(expected)
    mismatched = [(e, g) for e, g in zip(expected, got) if not _close(e, g)]
    # A point that projects exactly onto a centerline vertex can take its
    # heading from either adjacent segment: Lanelet2 sums the arc length to the
    # vertex 1 ulp differently. One such case is in this sample.
    assert len(mismatched) <= 1, mismatched[:5]
    for e, g in mismatched:
        # c2l, on_lanelet and s agree (s to an ulp); the heading is what differs
        assert _close(" ".join(e.split()[:3]), " ".join(g.split()[:3]))


# ---------------------------------------------------------------------------
# Building a bundle
# ---------------------------------------------------------------------------


def test_set_rpath_rewrites_in_place(tmp_path: Path) -> None:
    from autoware_carla_scenario.standalone.build import BuildError, set_rpath

    if not shutil.which("cc"):
        pytest.skip("no C compiler to make an ELF file with")
    src = tmp_path / "m.c"
    src.write_text("int main(void) { return 0; }\n")
    exe = tmp_path / "m"
    subprocess.run(
        [
            "cc",
            str(src),
            "-o",
            str(exe),
            "-Wl,--disable-new-dtags,-rpath,/a/long/build/path:/b",
        ],
        check=True,
    )
    assert set_rpath(exe, "$ORIGIN/../lib") == "/a/long/build/path:/b"
    readelf = subprocess.run(
        ["readelf", "-d", str(exe)], capture_output=True, text=True
    )
    assert "[$ORIGIN/../lib]" in readelf.stdout
    assert subprocess.run([str(exe)], check=False).returncode == 0
    with pytest.raises(BuildError):
        set_rpath(exe, "/" + "x" * 200)


_UNSUPPORTED = """
from __future__ import annotations

from dataclasses import dataclass

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    BaseScenario,
    CollisionCondition,
    EgoConfig,
    GroundProjectionConfig,
    Lanelet2Pose,
    TimeoutCondition,
)


@dataclass
class UnsupportedConfig:
    name: str = "unsupported"
    timeout_seconds: float = 10.0


class UnsupportedScenario(BaseScenario):
    _config: UnsupportedConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: UnsupportedConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection)
        self._config = config or UnsupportedConfig()

    def setup(self) -> None:
        self._setup_ego_spawn()
        self.register_fail_condition(CollisionCondition(label="ego_collision"))
        self.register_fail_condition(TimeoutCondition(self._config.timeout_seconds, label="t"))

    def is_done(self) -> bool:
        return False
"""


def _settings() -> RunSettings:
    return RunSettings(
        scenario_name="UnsupportedScenario",
        vehicle_type="vehicle.mini.cooper",
        initial_speed_kmh=0.0,
        spawn_retry_max_count=0,
        spawn_retry_t_step=0.1,
        spawn_retry_z_step=0.5,
        spawn_pose=Pose(183, 5.0),
        goal_pose=None,
        waypoint_poses=(),
        ray_distance_upper=5.0,
        ray_distance_lower=5.0,
        host="localhost",
        port=2000,
        tm_port=8100,
        timeout_seconds=10.0,
        max_tick_rate_hz=None,
        map_name="",
        overwrite_xodr=False,
        xodr_env_var="",
    )


@needs_codon
def test_unsupported_api_is_refused_at_its_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.standalone.build import BuildError, build_bundle

    pkg = tmp_path / "src" / "unsupported_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "scenario.py").write_text(_UNSUPPORTED)
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    module = importlib.import_module("unsupported_pkg.scenario")
    try:
        with pytest.raises(BuildError) as excinfo:
            build_bundle(
                module.UnsupportedScenario,
                module.UnsupportedConfig,
                {},
                _settings(),
                xodr=XODR_PATH,
                lanelet2=OSM_PATH,
                out_dir=tmp_path / "out",
                executable_name="unsupported",
                release=False,
            )
    finally:
        sys.modules.pop("unsupported_pkg.scenario", None)
        sys.modules.pop("unsupported_pkg", None)
    line = (
        _UNSUPPORTED.splitlines().index(
            '        self.register_fail_condition(CollisionCondition(label="ego_collision"))'
        )
        + 1
    )
    [diagnostic] = excinfo.value.diagnostics
    assert (
        "CollisionCondition is not supported by the standalone runtime"
        in diagnostic.message
    )
    assert diagnostic.path == str(pkg / "scenario.py")
    assert diagnostic.line == line


@needs_codon
@pytest.mark.slow
def test_lane_change_builds_into_a_relocatable_bundle(tmp_path: Path) -> None:
    from autoware_carla_scenario.standalone.build import (
        RUNTIME_LIBRARIES,
        build_standalone,
    )

    cwd = os.getcwd()
    os.chdir(_REPO)  # the example configs name the fixture map relative to the repo
    try:
        result = build_standalone("lane_change/left", [], tmp_path / "built")
    finally:
        os.chdir(cwd)
    assert (tmp_path / "built" / "share" / "map.acsmap").stat().st_size > 0
    assert (tmp_path / "built" / "share" / "map.xodr").exists()
    names = {p.name for p in (tmp_path / "built" / "lib").iterdir()}
    assert {"libtypesafe_carla_ffi.so", *RUNTIME_LIBRARIES} <= names

    # Copied elsewhere and run with an empty environment: nothing outside the
    # bundle but the system's C/C++ libraries.
    moved = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(tmp_path / "built", moved)
    exe = moved / "bin" / result.executable.name
    readelf = subprocess.run(
        ["readelf", "-d", str(exe)], capture_output=True, text=True
    )
    assert "Library rpath: [$ORIGIN/../lib]" in readelf.stdout
    ldd = subprocess.run(["ldd", str(exe)], capture_output=True, text=True, env={})
    for lib in ("libcodonrt.so", "libomp.so", "libgfortran.so.5"):
        assert f"{lib} => {moved}/bin/../lib/{lib}" in ldd.stdout
    clean_env = {"PATH": "/usr/bin:/bin"}
    help_run = subprocess.run(
        [str(exe), "--help"], capture_output=True, text=True, env=clean_env, check=False
    )
    assert help_run.returncode == 0, help_run.stderr
    assert (
        "built with: 10.0" in help_run.stdout
    )  # lane_change/left's scenario.timeout_seconds
    # No CARLA server here: the binary loads its map, fails to connect, and says so.
    no_server = subprocess.run(
        [
            str(exe),
            "--port",
            "1",
            "--client-timeout",
            "1",
            "--output-dir",
            str(tmp_path / "o"),
        ],
        capture_output=True,
        text=True,
        env=clean_env,
        check=False,
        timeout=120,
    )
    assert no_server.returncode == 3, no_server.stderr
    assert "CARLA" in no_server.stderr
