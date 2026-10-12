"""The library check: the framework's own modules, compiled with Codon.

The manifest (``typecheck/library.py``) and the source the check generates
are tested without Codon; the compiles are skipped where no Codon compiler is
installed (typesafe-carla's toolchain, a dependency on Linux x86_64 and
aarch64).
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoware_carla_scenario.typecheck import (
    TypeCheckResult,
    available_toolchain,
    check,
    typecheck_library,
)
from autoware_carla_scenario.typecheck import model_dir
from autoware_carla_scenario.typecheck.check import (
    _model_module,
    _uncalled,
    replaced_models,
)
from autoware_carla_scenario.typecheck.cli import main as scenario_check
from autoware_carla_scenario.typecheck.library import (
    CHECKED,
    EXCLUDED,
    MODELLED_IMPORTS,
    NOT_YET_CHECKED,
    REPLACED_MODELS,
    UNCALLED,
    package_modules,
)
from autoware_carla_scenario.typecheck.library_driver import (
    render_library_checks,
    render_library_driver,
    workspace_module,
)
from autoware_carla_scenario.typecheck.transform import redirect_imports

_PACKAGE = "autoware_carla_scenario"

needs_codon = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_every_module_is_either_checked_or_excluded() -> None:
    modules = set(package_modules())
    assert len(CHECKED) == len(set(CHECKED)), "a module is listed twice in CHECKED"
    both = set(CHECKED) & set(EXCLUDED)
    assert not both, f"in both CHECKED and EXCLUDED: {sorted(both)}"
    missing = modules - set(CHECKED) - set(EXCLUDED)
    assert not missing, (
        f"not in typecheck/library.py: {sorted(missing)}; add each to EXCLUDED "
        f"({NOT_YET_CHECKED!r}) or, if it compiles, to CHECKED"
    )
    unknown = (set(CHECKED) | set(EXCLUDED)) - modules
    assert (
        not unknown
    ), f"listed in typecheck/library.py but not modules: {sorted(unknown)}"


def test_every_excluded_module_has_a_reason() -> None:
    assert all(reason.strip() for reason in EXCLUDED.values())


def test_every_uncalled_function_is_a_public_one_of_a_checked_module() -> None:
    modules = package_modules()
    for name, reason in UNCALLED.items():
        assert reason.strip(), f"{name}: no reason"
        # The deepest one: traffic.base.X is in traffic.base, not
        # in traffic, and trajectory.model.X in trajectory.model.
        owners = [m for m in CHECKED if name.startswith(f"{m}.")]
        assert owners, f"{name}: not in a checked module"
        module = max(owners, key=len)
        qualname = name.removeprefix(f"{module}.")
        tree = ast.parse(modules[module].read_text(encoding="utf-8"))
        owner, _, function = qualname.rpartition(".")
        body = tree.body
        if owner:
            classes = [
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
            ]
            assert classes, f"{name}: no class {owner}"
            body = classes[0].body
        if owner and function == "__init__":
            continue  # a generated constructor (a dataclass) has no definition
        assert [
            n
            for n in body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == function
        ], f"{name}: no such function"


def test_every_modelled_import_is_between_checked_modules() -> None:
    for name, imported in MODELLED_IMPORTS.items():
        assert name in CHECKED, name
        assert imported and all(m in CHECKED for m in imported), name


def test_the_checker_itself_is_not_a_module_to_check() -> None:
    assert not [m for m in package_modules() if m.startswith(f"{_PACKAGE}.typecheck")]


# ---------------------------------------------------------------------------
# Imports of the framework, pointed into the workspace
# ---------------------------------------------------------------------------


_CHECKED = {f"{_PACKAGE}.coordinate.frames", f"{_PACKAGE}.kinematics.vector"}
_MODULES = _CHECKED | {
    _PACKAGE,
    f"{_PACKAGE}.coordinate",
    f"{_PACKAGE}.coordinate.poses",
    f"{_PACKAGE}.kinematics",
    f"{_PACKAGE}.utils",
    f"{_PACKAGE}.utils.config",
}


def _target(name: str) -> str | None:
    if name in _CHECKED:
        return workspace_module(name)
    if name.startswith(f"{_PACKAGE}.coordinate"):
        return f"{_PACKAGE}.coordinate"  # the model of the package
    return None


def _redirect(source: str, module: str = f"{_PACKAGE}.kinematics.velocity"):
    return redirect_imports(
        textwrap.dedent(source), module, False, _target, _MODULES.__contains__
    )


def test_an_import_of_a_checked_module_names_its_workspace_module() -> None:
    out, problems = _redirect(
        """
        from autoware_carla_scenario.coordinate.frames import (
            CoordinateFrame,
            frame_of as of,
        )
        from .vector import Vector3
        import math
        """
    )
    assert problems == []
    lines = out.splitlines()
    assert lines[1] == (
        "from _acs_lib.autoware_carla_scenario__coordinate__frames "
        "import CoordinateFrame, frame_of as of"
    )
    assert (
        lines[5]
        == "from _acs_lib.autoware_carla_scenario__kinematics__vector import Vector3"
    )
    assert lines[6] == "import math"
    assert len(lines) == 7  # every statement kept its line


def test_an_import_of_an_unchecked_module_names_its_model() -> None:
    out, problems = _redirect(
        """
        if TYPE_CHECKING:
            from ..coordinate.poses import Lanelet2Pose
        from ..coordinate import frames, poses as p
        """
    )
    assert problems == []
    assert "    from autoware_carla_scenario.coordinate import Lanelet2Pose" in out
    assert (
        "from _acs_lib import autoware_carla_scenario__coordinate__frames as frames; "
        "from autoware_carla_scenario import coordinate as p"
    ) in out


def test_an_import_with_nothing_to_stand_in_is_reported() -> None:
    out, problems = _redirect(
        """
        from autoware_carla_scenario.utils.config import load
        import autoware_carla_scenario.coordinate.frames
        from .vector import *
        """
    )
    assert [line for line, _ in problems] == [2, 3, 4]
    assert "neither checked" in problems[0][1]
    assert "from autoware_carla_scenario.utils.config import load" in out


def test_a_module_of_a_checked_package_can_have_a_model_of_its_own() -> None:
    # utils is checked, and utils/traffic_light.codon models one module of it
    # that is not; the other modules of utils have nothing to stand in.
    assert _model_module(f"{_PACKAGE}.utils.traffic_light") == (
        f"{_PACKAGE}.utils.traffic_light"
    )
    assert _model_module(f"{_PACKAGE}.utils.config") is None
    assert _model_module(f"{_PACKAGE}.coordinate.transform") == f"{_PACKAGE}.coordinate"


@pytest.mark.parametrize(
    ("module", "sibling"),
    [
        ("autoware_bridge.grpc_server", "autoware_bridge.ros_bridge.client"),
        ("autoware_stack.launcher", "autoware_stack.docker"),
        ("driver.egodriver_client", "driver.renderer"),
    ],
)
def test_a_re_exported_module_has_a_model_of_its_own(module: str, sibling: str) -> None:
    # The checked autoware_bridge and driver packages re-export a class of a
    # module that is not checked (grpc), and the checked AutowareEgoEntity
    # takes an AutowareLauncher (subprocess): each such module has a model of
    # its own, under a package directory of the model, and its package and
    # the package's other modules have none.
    assert _model_module(f"{_PACKAGE}.{module}") == f"{_PACKAGE}.{module}"
    assert _model_module(f"{_PACKAGE}.{module.split('.')[0]}") is None
    assert _model_module(f"{_PACKAGE}.{sibling}") is None


def test_a_modelled_import_names_the_model_though_the_module_is_checked() -> None:
    # measures and scenario_base import each other: measures has scenario_base
    # through its model, scenario_base has measures from its source.
    measures, scenario_base = f"{_PACKAGE}.measures", f"{_PACKAGE}.scenario_base"
    assert scenario_base in MODELLED_IMPORTS[measures]
    every = package_modules()
    sources = check._Sources()
    for name in (measures, scenario_base):
        text = every[name].read_text(encoding="utf-8")
        sources.modules[name] = check._Module(every[name], False, text, ast.parse(text))
    files = {
        f.module: f.text
        for f in check._library_files([measures, scenario_base], sources, every)
    }
    assert f"from {_PACKAGE}.scenario_base import BaseScenario" in files[measures]
    assert f"from {workspace_module(measures)} import " in files[scenario_base]


# ---------------------------------------------------------------------------
# Model modules compiled from the checked source instead
# ---------------------------------------------------------------------------


def test_every_replaced_model_names_a_model_file_and_package_modules() -> None:
    modules = package_modules()
    for stem, sources in REPLACED_MODELS.items():
        assert (model_dir() / _PACKAGE / f"{stem}.codon").is_file(), stem
        assert sources and all(m in modules for m in sources), stem


def _trees(*names: str) -> dict[str, SimpleNamespace]:
    every = package_modules()
    return {n: SimpleNamespace(tree=ast.parse(every[n].read_text())) for n in names}


def test_a_replaced_model_re_exports_the_checked_definitions() -> None:
    frames, poses = REPLACED_MODELS["_poses"]
    problems: list = []
    out = replaced_models(_trees(frames, poses), problems)
    assert problems == []
    assert out["_poses"].splitlines() == [
        f"from {workspace_module(frames)} import CoordinateFrame, FrameMismatchError",
        f"from {workspace_module(poses)} import CarlaWorldPose, Lanelet2Pose, "
        "OpenDrivePose",
    ]


def test_the_vehicle_entity_model_re_exports_the_checked_spawn_and_vehicle() -> None:
    # So the model's EgoConfig derives from the checked VehicleEntityConfig and
    # the checked ego hands the checked spawn_vehicle_actor its own SpawnLocation.
    spawn, vehicle = REPLACED_MODELS["_vehicle_entity"]
    problems: list = []
    out = replaced_models(_trees(spawn, vehicle), problems)
    assert problems == []
    assert out["_vehicle_entity"].splitlines() == [
        f"from {workspace_module(spawn)} import SpawnLocation, SpawnPointIndex, "
        "SpawnTransform",
        f"from {workspace_module(vehicle)} import VehicleEntity, VehicleEntityConfig",
    ]


def test_the_ground_projection_model_re_exports_the_checked_config() -> None:
    # So a scenario's GroundProjectionConfig, imported from the top-level
    # package (the model), is the one the checked VehicleEntityConfig holds.
    (snap,) = REPLACED_MODELS["_ground_projection"]
    problems: list = []
    out = replaced_models(_trees(snap), problems)
    assert problems == []
    assert out["_ground_projection"].splitlines() == [
        f"from {workspace_module(snap)} import GroundProjectionConfig",
    ]


def test_the_bridge_pose_model_re_exports_the_checked_pose() -> None:
    # So the pose the model's to_map_frame returns is the one the checked
    # AutowareBridge.configure() takes, which AutowareEgoEntity hands it.
    (base,) = REPLACED_MODELS["_bridge_pose"]
    problems: list = []
    out = replaced_models(_trees(base), problems)
    assert problems == []
    assert out["_bridge_pose"].splitlines() == [
        f"from {workspace_module(base)} import BridgePose, Quaternion, Vector3",
    ]


def test_the_map_source_model_re_exports_the_checked_source() -> None:
    # So the model GitMapCache (maps/cache.codon) takes the MapSource the
    # checked maps.resolver and maps.catalogue hand it.
    (source,) = REPLACED_MODELS["_map_source"]
    problems: list = []
    out = replaced_models(_trees(source), problems)
    assert problems == []
    assert out["_map_source"].splitlines() == [
        f"from {workspace_module(source)} import MapSource",
    ]


def test_a_model_is_replaced_only_once_all_its_modules_are_checked() -> None:
    frames, _poses = REPLACED_MODELS["_poses"]
    problems: list = []
    assert "_poses" not in replaced_models(_trees(frames), problems)
    assert problems == []


def test_a_name_the_checked_source_lacks_is_a_problem() -> None:
    frames, poses = REPLACED_MODELS["_poses"]
    trees = _trees(frames, poses)
    trees[poses].tree.body = [
        node
        for node in trees[poses].tree.body
        if getattr(node, "name", None) != "OpenDrivePose"
    ]
    problems: list = []
    out = replaced_models(trees, problems)
    assert [p.message.split(",")[0] for p in problems] == ["OpenDrivePose"]
    assert "OpenDrivePose" not in out["_poses"]


# ---------------------------------------------------------------------------
# The calls appended to a checked module
# ---------------------------------------------------------------------------


_MODULE = """
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


def angle(a: float, *, wrap: bool = True) -> float:
    return a


def where(pose: Lanelet2Pose | OpenDrivePose | None) -> float:
    return 0.0


def total(xs: Sequence[float]) -> float:
    return 0.0


def _private(x: Any) -> None:
    pass


class Vector:
    def __init__(self, x: float) -> None:
        self.x = x

    def __eq__(self, other: object) -> bool:
        return True

    @property
    def norm(self) -> float:
        return self.x

    @staticmethod
    def zero() -> Vector:
        return Vector(0.0)

    @classmethod
    def unit(cls, scale: float) -> Vector:
        return cls(scale)

    def _helper(self, anything: Any) -> None:
        pass


@dataclass
class Config:
    name: str
    ids: list[int] = field(default_factory=list)
    hidden: int = field(default=0, init=False)


class Level(Enum):
    LOW = 1

    def up(self) -> Level:
        return self


def generic(x: Any, y) -> None:
    pass
"""


def test_every_public_function_and_method_is_called_with_typed_values() -> None:
    checks = render_library_checks(ast.parse(_MODULE))
    calls = checks.source.splitlines()
    assert "    angle(_acs_value(float), wrap=_acs_value(bool))" in calls
    assert "    where(_acs_value(Lanelet2Pose))" in calls
    assert "    where(_acs_value(OpenDrivePose))" in calls
    assert "    total(_acs_value(Sequence[float]))" in calls
    assert "    Vector(_acs_value(float))" in calls
    assert "    _acs_value(Vector).__eq__(_acs_value(Vector))" in calls
    assert "    _acs_value(Vector).norm" in calls
    assert "    Vector.zero()" in calls
    assert "    Vector.unit(_acs_value(float))" in calls
    assert "    Config(_acs_value(str), _acs_value(list[int]))" in calls
    assert "    _acs_value(Level).up()" in calls
    assert not [c for c in calls if "_private" in c or "_helper" in c or "Level(" in c]
    label = checks.labels[calls.index("    where(_acs_value(OpenDrivePose))") + 1]
    assert label == "where(pose: OpenDrivePose)"


def test_a_parameter_the_check_cannot_call_with_is_a_problem_of_the_module() -> None:
    checks = render_library_checks(ast.parse(_MODULE))
    lines = _MODULE.splitlines()
    assert len(checks.problems) == 2
    for line, message in checks.problems:
        assert lines[line - 1].startswith("def generic(")
    assert any("`x`" in m and "Any" in m for _, m in checks.problems)
    assert any("`y` has no annotation" in m for _, m in checks.problems)


_CALLABLES = """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union


def apply(f: Callable[[float, str], Optional[float]]) -> None:
    pass


@dataclass
class Reader:
    read: Callable[["carla.World"], float]


def anything(f: Callable[[Any], Any]) -> None:
    pass


def variadic(f: Callable[..., float]) -> None:
    pass


def register(cb: Union[float, Callable[[float], None]]) -> None:
    pass
"""


def test_a_callable_is_called_with_a_value_of_its_codon_type() -> None:
    checks = render_library_checks(ast.parse(_CALLABLES))
    calls = checks.source.splitlines()
    assert "    apply(_acs_value(Callable[[float, str], Optional[float]]))" in calls
    assert "    Reader(_acs_value(Callable[[carla.World], float]))" in calls
    # A union with a callable in it: called once with each member.
    assert "    register(_acs_value(float))" in calls
    assert "    register(_acs_value(Callable[[float], None]))" in calls
    lines = _CALLABLES.splitlines()
    assert sorted(lines[line - 1].split("(")[0] for line, _ in checks.problems) == [
        "def anything",
        "def variadic",
    ]


_DATACLASS_FIELDS = """
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class Run:
    name: str
    scenario: Straight | Turn = field(default_factory=Straight)
    constraints: dict[str, Any] = field(default_factory=dict)
    npc: Union[Straight, Turn, None] = None
    timeout: float = 5.0


@dataclass
class Required:
    options: dict[str, Any]
"""


def test_a_dataclass_is_constructed_with_each_member_of_a_union_field() -> None:
    checks = render_library_checks(ast.parse(_DATACLASS_FIELDS))
    calls = [c.strip() for c in checks.source.splitlines() if "Run(" in c]
    # A dict[str, Any] field with a default is left out of the call, and the
    # fields after it are passed by keyword.
    assert calls == [
        f"Run(_acs_value(str), _acs_value({scenario}), "
        f"npc=_acs_value({npc}), timeout=_acs_value(float))"
        for scenario in ("Straight", "Turn")
        for npc in ("Straight", "Turn")
    ]
    label = checks.labels[checks.source.splitlines().index("    " + calls[-1]) + 1]
    assert label == "Run.__init__(name: str, scenario: Turn, npc: Turn, timeout: float)"
    # Without a default, the field cannot be left out: a problem, as before.
    lines = _DATACLASS_FIELDS.splitlines()
    assert [lines[line - 1].strip() for line, _ in checks.problems] == [
        "options: dict[str, Any]"
    ]


_DERIVED_DATACLASS = """
from dataclasses import dataclass
from typing import Optional

from .base import CameraConfig


@dataclass(frozen=True)
class Lens:
    k: float = 0.0


@dataclass(frozen=True)
class Camera(CameraConfig):
    sensor_type: str = "sensor.camera.rgb"
    iso: Optional[float] = None
    width: int = 640  # redefined: keeps the base's place


@dataclass
class Fisheye(Lens):
    fov: float = 180.0


class Plain:
    label: str = ""


@dataclass
class Labelled(Plain):
    size: int = 0
"""

_DATACLASS_BASE = """
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1920
    fov: float = 90.0
"""


def test_a_dataclass_is_constructed_with_the_fields_of_its_bases_first() -> None:
    base = ast.parse(_DATACLASS_BASE)
    classes = {n.name: n for n in base.body if isinstance(n, ast.ClassDef)}
    checks = render_library_checks(ast.parse(_DERIVED_DATACLASS), (), classes)
    calls = [c.strip() for c in checks.source.splitlines() if "_acs_value(" in c]
    # A base of another checked module, a base of the same module, and a base
    # that is no dataclass, whose annotations are no fields.
    assert calls == [
        "Lens(_acs_value(float))",
        "Camera(_acs_value(int), _acs_value(float), _acs_value(str), "
        "_acs_value(Optional[float]))",
        "Fisheye(_acs_value(float), _acs_value(float))",
        "Labelled(_acs_value(int))",
    ]
    # Without the classes of the other modules, the base's fields are unknown.
    alone = render_library_checks(ast.parse(_DERIVED_DATACLASS))
    assert (
        "    Camera(_acs_value(str), _acs_value(Optional[float]), _acs_value(int))"
        in alone.source.splitlines()
    )


_MIXED_DATACLASSES = """
from dataclasses import dataclass, field


@dataclass
class Size:
    width: int = 0


@dataclass
class Name:
    label: str = ""


class Plain(Size):
    pass


@dataclass
class Through(Plain):
    depth: float = 0.0


@dataclass
class Hidden(Size):
    width: int = field(default=1, init=False)
    depth: float = 0.0


@dataclass
class Both(Size, Name):
    depth: float = 0.0
"""


def test_a_dataclass_takes_its_base_fields_as_python_does() -> None:
    import dataclasses

    namespace: dict[str, object] = {}
    exec(_MIXED_DATACLASSES, namespace)  # noqa: S102 - the source above
    checks = render_library_checks(ast.parse(_MIXED_DATACLASSES))
    calls = [c.strip() for c in checks.source.splitlines() if "_acs_value(" in c]
    # Through a plain class derived from a dataclass, and without a field
    # redefined with init=False, as Python's constructor takes them.
    assert "Through(_acs_value(int), _acs_value(float))" in calls
    assert "Hidden(_acs_value(float))" in calls
    for name, count in [("Through", 2), ("Hidden", 1)]:
        cls = namespace[name]
        assert dataclasses.is_dataclass(cls)
        assert len([f for f in dataclasses.fields(cls) if f.init]) == count
    # Python takes Name's field first, Codon 0.19 Size's: no call, a problem.
    assert not any(c.startswith("Both(") for c in calls)
    line = _MIXED_DATACLASSES.splitlines().index("class Both(Size, Name):") + 1
    assert [n for n, message in checks.problems if "Both" in message] == [line]


def test_an_uncalled_function_is_left_out() -> None:
    checks = render_library_checks(
        ast.parse(_MODULE), ["angle", "Vector.norm", "Config.__init__", "generic"]
    )
    calls = checks.source.splitlines()
    assert not [c for c in calls if "angle(" in c or ".norm" in c or "Config(" in c]
    assert "    Vector.zero()" in calls
    # generic() is not called, so its parameters are no problem.
    assert checks.problems == []


def test_an_uncalled_name_belongs_to_the_longest_checked_module() -> None:
    # trajectory.model's wrapper is of trajectory.model, not of the
    # trajectory package that is checked too.
    package = "autoware_carla_scenario.trajectory"
    assert "_refusing_vertex_times" in _uncalled(f"{package}.model")
    assert not [n for n in _uncalled(package) if "_refusing_vertex_times" in n]


def test_the_driver_runs_the_calls_of_each_module() -> None:
    driver = render_library_driver([f"{_PACKAGE}.kinematics.angle"])
    assert (
        "from _acs_lib.autoware_carla_scenario__kinematics__angle import "
        "_acs_library_check as _acs_check_0"
    ) in driver.source
    assert "_acs_check_0()" in driver.source


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------


@needs_codon
def test_the_checked_modules_compile() -> None:
    result = typecheck_library()
    assert result.skipped is None
    assert result.ok, result.format()


@needs_codon
def test_scenario_check_checks_the_library(capsys: pytest.CaptureFixture[str]) -> None:
    assert scenario_check(["--library", f"{_PACKAGE}.kinematics.angle"]) == 0
    assert "[ok]" in capsys.readouterr().out


def _dependencies(*names: str) -> list[str]:
    every = package_modules()
    sources = check._Sources()
    for name in names:
        sources.modules[name] = check._read_module(every[name])
    return check._library_dependencies(list(names), sources, every)


def test_a_named_module_brings_the_checked_modules_it_imports() -> None:
    # actions.turn imports find_pedestrian_entity, which only the checked
    # entity.registry defines (the model's entity.codon does not).
    deps = _dependencies(f"{_PACKAGE}.actions.turn")
    assert f"{_PACKAGE}.entity.registry" in deps
    # ... and, transitively, what entity.registry imports.
    assert f"{_PACKAGE}.entity.pedestrian_entity" in deps
    assert set(deps) <= set(CHECKED)


def test_a_modelled_import_is_not_followed() -> None:
    deps = _dependencies(f"{_PACKAGE}.entity.ego")
    assert f"{_PACKAGE}.scenario_base" not in deps  # MODELLED_IMPORTS
    # entity._spawn brings the rest of the vehicle entity's replaced model,
    # so the model re-exports the checked definitions, as in the full run.
    assert {f"{_PACKAGE}.entity._spawn", f"{_PACKAGE}.entity.vehicle_entity"} <= set(
        deps
    )


def test_the_full_run_compiles_only_the_checked_modules() -> None:
    assert _dependencies(*CHECKED) == []


@needs_codon
@pytest.mark.parametrize(
    "module",
    [
        "scenario_base",
        "entity.ego",
        "actions.turn",
        "actions.lane_change",
        # AutowareEgoEntity overrides EgoVehicle.spawn(), and imports
        # scenario_base through the model, as entity.ego does.
        "entity.autoware_entity",
        # The run starts at coordinate.frames, which the model's _poses.codon
        # re-exports: the driver must read the model first.
        "coordinate.poses",
    ],
)
def test_a_module_importing_a_checked_only_name_compiles_alone(module: str) -> None:
    result = typecheck_library([f"{_PACKAGE}.{module}"])
    assert result.ok, result.format()
    assert "more they import" in result.scenario


_FAKE = f"{_PACKAGE}.zz_library_check_case"

_GOOD = """
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

from autoware_carla_scenario.coordinate.poses import Lanelet2Pose, OpenDrivePose
from .entity_role import EntityRole
from .kinematics.vector import Vector3


class Turn(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Level(Enum):
    LOW = auto()
    HIGH = auto()


class LevelError(ValueError):
    level: Level

    def __init__(self, level: Level) -> None:
        super().__init__(f"bad level {level.name}")
        self.level = level


class Thing:
    size: float

    def __init__(self, size: float) -> None:
        self.size = size

    @classmethod
    def unit(cls) -> Thing:
        return cls(1.0)

    @property
    def doubled(self) -> float:
        return self.size * 2

    def scaled(self, v: Vector3) -> Vector3:
        return v * self.size


def lane_s(pose: Lanelet2Pose | OpenDrivePose) -> float:
    return pose.s


@dataclass(frozen=True)
class Reader:
    read: Callable[[float], Optional[float]]


def read_twice(reader: Reader, f: Callable[[float], float]) -> Optional[float]:
    return reader.read(f(1.0))


def describe(turn: Turn, level: Level, role: EntityRole) -> str:
    if level == Level.HIGH:
        raise LevelError(level)
    return turn.value + str(role)
"""


def _check_case(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TypeCheckResult:
    path = tmp_path / "zz_library_check_case.py"
    path.write_text(textwrap.dedent(source))
    modules = package_modules()
    modules[_FAKE] = path
    monkeypatch.setattr(check, "package_modules", lambda: modules)
    return typecheck_library(
        [_FAKE, f"{_PACKAGE}.entity_role", f"{_PACKAGE}.kinematics.vector"]
    )


@needs_codon
def test_enums_exceptions_classmethods_and_unions_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_GOOD, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("wrong", "right"),
    [
        # A body of the wrong type, in each kind of callable.
        ("return turn.value + str(role)", "return level.value"),
        ("return cls(1.0)", "return cls('one')"),
        ("return self.size * 2", "return self.size.real()"),
        # Only one member of the union has the attribute.
        ("return pose.s", "return pose.road_id"),
    ],
    ids=["function", "classmethod", "property", "union-member"],
)
def test_a_wrong_module_is_refused_at_its_line(
    wrong: str, right: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _GOOD.replace(wrong, right)
    assert source != _GOOD
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    stripped = [text.strip() for text in textwrap.dedent(source).splitlines()]
    line = stripped.index(right) + 1
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].path == str(tmp_path / "zz_library_check_case.py")
    assert located[0].line == line, result.format()


@needs_codon
def test_an_abstract_base_class_and_its_subclass_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = """
from __future__ import annotations

from abc import ABC, abstractmethod


class Bridge(ABC):
    @abstractmethod
    def is_ready(self) -> bool:
        \"\"\"Whether it is ready.\"\"\"

    def close(self) -> None:
        pass


class Fake(Bridge):
    polls: int = 0

    def is_ready(self) -> bool:
        self.polls += 1
        return self.polls > 1
"""
    result = _check_case(source, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
def test_pyxodrs_road_network_is_the_one_map_manager_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # utils/traffic_light.py annotates with pyxodr's RoadNetwork, since utils
    # may not import coordinate; the pyxodr shim names the model's class, so
    # the network MapManager returns goes into it.
    source = """
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from autoware_carla_scenario.coordinate.map_manager import MapManager

if TYPE_CHECKING:
    from pyxodr.road_objects.network import RoadNetwork


def first_controller_name(road_network: RoadNetwork) -> Optional[str]:
    for element in road_network.root.iter("controller"):
        return element.get("name")
    return None


def ambient_controller_name() -> Optional[str]:
    return first_controller_name(MapManager.get_instance().road_network)
"""
    result = _check_case(source, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
def test_a_module_importing_what_codon_cannot_compile_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(
        "import yaml\n\n\ndef f() -> int:\n    return 1\n", tmp_path, monkeypatch
    )
    assert not result.ok
    assert result.diagnostics[0].line == 1
    assert "yaml has no Codon model" in result.diagnostics[0].message


@needs_codon
def test_an_unknown_module_is_refused() -> None:
    result = typecheck_library([f"{_PACKAGE}.no_such_module"])
    assert not result.ok
    assert "not a module" in result.diagnostics[0].message


_ABSTRACT = """
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    def summary(self) -> dict[str, Any]:
        import json  # noqa: PLC0415

        return json.loads(json.dumps({"area": self.area(), "kind": "shape"}))


class Square(Shape):
    side: float

    def __init__(self, side: float) -> None:
        self.side = side

    def area(self) -> float:
        return self.side * self.side


def total(shape: Shape) -> float:
    return shape.area()
"""


@needs_codon
def test_abc_and_an_uncalled_method_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without the entry, summary() is called: json is reported.
    result = _check_case(_ABSTRACT, tmp_path, monkeypatch)
    assert not result.ok
    assert "json has no Codon model" in result.diagnostics[0].message
    # With it, neither its import nor its mixed dict reaches Codon.
    monkeypatch.setitem(check.UNCALLED, f"{_FAKE}.Shape.summary", "uses json")
    result = _check_case(_ABSTRACT, tmp_path, monkeypatch)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# What the traffic backends need (#79)
# ---------------------------------------------------------------------------


_OVERRIDDEN = """
from __future__ import annotations

from typing import Any


class Backend:
    def start(self, world: int, *, skip: list[int]) -> None:
        pass

    def turn(self, world: int, **kwargs: Any) -> None:
        pass

    def tick(self, world: int) -> None:
        pass
"""


def test_a_method_with_keyword_only_parameters_is_called_through_its_class() -> None:
    # Codon 0.19 calls a method a subclass may override through a dispatch
    # thunk, which takes neither a keyword argument nor **kwargs.
    calls = render_library_checks(ast.parse(_OVERRIDDEN)).source.splitlines()
    assert (
        "    Backend.start(_acs_value(Backend), _acs_value(int), "
        "skip=_acs_value(list[int]))"
    ) in calls
    assert "    Backend.turn(_acs_value(Backend), _acs_value(int))" in calls
    assert "    _acs_value(Backend).tick(_acs_value(int))" in calls


def test_an_import_inside_an_uncalled_function_is_left_as_written() -> None:
    source = """
        def build(options: dict) -> None:
            from autoware_carla_scenario.utils.config import load
        """
    out, problems = redirect_imports(
        textwrap.dedent(source),
        f"{_PACKAGE}.kinematics.velocity",
        False,
        _target,
        _MODULES.__contains__,
        uncalled=["build"],
    )
    assert problems == []
    assert "from autoware_carla_scenario.utils.config import load" in out


_BACKENDS = """
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


@runtime_checkable
class Steering(Protocol):
    def steer(self, world: carla.World) -> None:
        ...


class Backend:
    def start(self, world: carla.World, *, skip_actor_ids: list[int] = []) -> None:
        pass

    @classmethod
    def from_options(cls, options: dict[str, int]) -> Backend:
        from autoware_carla_scenario.utils.config import checked_options

        checked_options(cls, options)
        return cls()


class Manager(Backend):
    def start(self, world: carla.World, *, skip_actor_ids: list[int] = []) -> None:
        try:
            world.tick()
        except RuntimeError:
            logger.warning("tick failed", exc_info=True)


def vehicle(actor: carla.Actor) -> carla.Vehicle:
    if TYPE_CHECKING:
        return actor.as_vehicle()
    else:
        return actor
"""


@needs_codon
def test_overridden_methods_protocols_and_type_checking_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each override is called by keyword, a framework module with no model is
    # imported only in an uncalled method, a protocol is a declaration, and
    # only the TYPE_CHECKING branch of vehicle() is compiled.
    monkeypatch.setitem(
        check.UNCALLED, f"{_FAKE}.Backend.from_options", "uses utils.config"
    )
    result = _check_case(_BACKENDS, tmp_path, monkeypatch)
    assert result.ok, result.format()


# ---------------------------------------------------------------------------
# The queue shim (the sensors' frame queues)
# ---------------------------------------------------------------------------


_QUEUES = """
from __future__ import annotations

import queue
from typing import Optional


class Frames:
    _frames: queue.Queue[int]
    _unbounded: queue.Queue[tuple[int, str]]

    def __init__(self) -> None:
        self._frames = queue.Queue(maxsize=2)
        self._unbounded = queue.Queue()

    def receive(self, frame: int) -> None:
        self._frames.put(frame)
        self._frames.put(frame, True, 1.0)
        self._frames.put(frame, block=False, timeout=None)
        self._frames.put(frame, True, 1)
        self._frames.put(frame, timeout=2)
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            pass
        self._unbounded.put((frame, "frame"))

    def newest(self, timeout: float) -> Optional[int]:
        try:
            frame = self._frames.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                break
        self._frames.get(True, timeout)
        self._frames.get(block=False)
        self._frames.get(timeout=1)
        return frame

    def size(self) -> int:
        if self._frames.empty() or self._frames.full():
            return 0
        return self._frames.qsize() + queue.Queue[int](0).qsize()
"""


@needs_codon
def test_the_queue_shim_takes_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_QUEUES, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError or an AttributeError in CPython too.
        ("queue.Queue(maxsize=2)", "queue.Queue(size=2)"),
        ("self._frames.get_nowait()", "self._frames.get_nowait(True)"),
        ("self._frames.get(True, timeout)", "self._frames.get(True, timeout, 1)"),
        ("self._frames.qsize() + ", "self._frames.peek() + "),
        ("self._frames.get(timeout=1)", 'self._frames.get(timeout="1")'),
    ],
    ids=[
        "unknown-keyword",
        "get_nowait-argument",
        "get-arguments",
        "no-method",
        "timeout-type",
    ],
)
def test_the_queue_shim_refuses_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _QUEUES.replace(right, wrong, 1)
    assert source != _QUEUES
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if wrong in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()


# ---------------------------------------------------------------------------
# The pathlib and urllib.parse shims (#97)
# ---------------------------------------------------------------------------


_PATHLIB = """
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Run:
    xodr_path: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("outputs"))


def cache_dir(configured: str) -> Path:
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    return Path.home() / ".cache" / "sumo"


def log_file(run: Run) -> str:
    return str(run.output_dir / "sumo" / "fcd.xml")


def is_default(run: Run, configured: str) -> bool:
    # As in Python: a str on the left of `/`, a Path of no parts, and a Path
    # compared with a str (never equal).
    here = "outputs" / Path()
    return run.output_dir == here and run.output_dir != configured
"""


@needs_codon
def test_the_pathlib_shim_builds_joins_and_prints_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "pathlib.codon").is_file()
    result = _check_case(_PATHLIB, tmp_path, monkeypatch)
    assert result.ok, result.format()
    # A Path is not a str: the shim is typed, not a pass-through.
    wrong = _PATHLIB.replace(
        'return str(run.output_dir / "sumo" / "fcd.xml")',
        'return run.output_dir / "sumo" / "fcd.xml"',
    )
    assert wrong != _PATHLIB
    assert not _check_case(wrong, tmp_path, monkeypatch).ok
    # Python builds and joins a Path with a str or a Path only.
    for part in ("Path(configured, 1)", "(Path(configured) / 1)"):
        wrong = _PATHLIB.replace("Path(configured)", part, 1)
        assert wrong != _PATHLIB
        result = _check_case(wrong, tmp_path, monkeypatch)
        assert not result.ok
        assert "a str or a Path only" in result.format(), result.format()


_URLLIB = """
from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit, urlunsplit


def repository(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url
    segments = [s for s in parts.path.split("/") if s]
    return urlunsplit((parts.scheme, parts.netloc, "/" + "/".join(segments), "", ""))


def host(url: str) -> Optional[str]:
    # Python's SplitResult: a 5-tuple of str with the netloc's parts.
    parts = urlsplit(url, scheme="https")
    scheme, netloc, path, query, fragment = parts
    if parts.port is None and parts[0] == scheme:
        return parts.hostname
    return urlunsplit(parts) + parts.geturl()
"""


@needs_codon
def test_the_urllib_parse_shim_splits_and_joins_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "urllib" / "parse.codon").is_file()
    result = _check_case(_URLLIB, tmp_path, monkeypatch)
    assert result.ok, result.format()
    # SplitResult's parts are typed: a part Python's lacks is refused.
    wrong = _URLLIB.replace("parts.netloc:", "parts.host:")
    assert wrong != _URLLIB
    assert not _check_case(wrong, tmp_path, monkeypatch).ok


# ---------------------------------------------------------------------------
# The file-system pathlib calls, hashlib, and str.encode (#116)
# ---------------------------------------------------------------------------


_FILE_SYSTEM = """
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


def maps_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.osm") if p.is_file())


def first_xodr(directory: Path) -> Optional[Path]:
    for candidate in sorted(directory.glob(pattern="*.xodr")):
        if candidate.exists() and not candidate.is_dir():
            found: Optional[Path] = candidate
            return found
    return None


def inside(root: Path, candidate: Path) -> bool:
    real = candidate.absolute().resolve()
    return real.parents[0] <= root or real == root or root in real.parents


def describe(path: Path) -> str:
    names = [parent.name for parent in path.parents]
    suffix = path.suffix
    return f"{path.name}{suffix} in {path.parent} ({len(path.parents)} {names})"


def digest(text: str) -> str:
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    h = hashlib.sha256(usedforsecurity=False)
    h.update(text.encode(errors="strict"))
    h.update(b"-")
    other = hashlib.sha256(string=b"x").copy()
    size = other.digest_size + other.block_size
    return key + h.hexdigest() + other.name + str(size + len(h.digest()))
"""


@needs_codon
def test_the_file_system_shims_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "hashlib.codon").is_file()
    result = _check_case(_FILE_SYSTEM, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError or an AttributeError in CPython 3.10, the oldest
        # Python the package supports (glob() takes `case_sensitive` from 3.12
        # on, sha256() `data=` from 3.13 on), except chmod(): it writes, and
        # the shim declares nothing that writes.
        ('directory.glob("*.osm")', 'directory.glob("*.osm", case_sensitive=True)'),
        ("candidate.exists()", "candidate.exists(True)"),
        ("suffix = path.suffix", "suffix = path.suffix()"),
        ("real.parents[0] <= root", 'real.parents[0] <= "/"'),
        ("candidate.absolute()", "candidate.chmod(0o644)"),
        (".hexdigest()[:10]", ".hexdigest(10)[:10]"),
        ("hashlib.sha256(usedforsecurity=False)", 'hashlib.sha256(b"", False)'),
        ('h.update(b"-")', 'h.update(data=b"-")'),
        ('hashlib.sha256(string=b"x")', 'hashlib.sha256(data=b"x")'),
        ('text.encode("utf-8")', 'text.encode("utf-8", "strict", 1)'),
        ('text.encode(errors="strict")', "text.encode(encoding=1)"),
        ('text.encode("utf-8")', 'text.decode("utf-8")'),
    ],
    ids=[
        "glob-keyword",
        "exists-argument",
        "suffix-is-a-property",
        "path-ordered-with-str",
        "no-method",
        "hexdigest-argument",
        "usedforsecurity-keyword-only",
        "update-positional-only",
        "sha256-keyword",
        "encode-arguments",
        "encode-encoding-type",
        "str-has-no-decode",
    ],
)
def test_the_file_system_shims_refuse_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _FILE_SYSTEM.replace(right, wrong, 1)
    assert source != _FILE_SYSTEM
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if wrong in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()
