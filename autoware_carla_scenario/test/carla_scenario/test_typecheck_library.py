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


def test_every_uncalled_function_is_one_of_a_checked_module() -> None:
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
    # module that is not checked (grpc), and a scenario's AutowareEgoEntity
    # takes an AutowareLauncher (whose module the library check compiles from
    # its source): each such module has a model of its own, under a package
    # directory of the model, and its package and the package's other modules
    # have none.
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


def test_the_ros2_model_re_exports_the_checked_config_and_transport() -> None:
    # So the model CycloneRos2Transport (ros2/cyclone.codon) derives from the
    # checked Ros2EgoTransport and takes the checked Ros2EgoConfig, which the
    # checked entity.ros2_entity._cyclone_transport() hands it.
    config, transport = REPLACED_MODELS["_ros2"]
    problems: list = []
    out = replaced_models(_trees(config, transport), problems)
    assert problems == []
    assert out["_ros2"].splitlines() == [
        f"from {workspace_module(config)} import Ros2EgoConfig",
        f"from {workspace_module(transport)} import Ros2EgoTransport",
    ]


def test_the_cyclone_model_stands_in_for_the_unchecked_module() -> None:
    # ros2.cyclone imports cyclonedds: the checked ros2_entity imports it
    # through its model, a module of the checked ros2 package's own.
    assert f"{_PACKAGE}.ros2.cyclone" in EXCLUDED
    assert f"{_PACKAGE}.ros2" in CHECKED
    assert _model_module(f"{_PACKAGE}.ros2.cyclone") == f"{_PACKAGE}.ros2.cyclone"
    assert _model_module(f"{_PACKAGE}.ros2") is None
    assert _model_module(f"{_PACKAGE}.ros2.messages") is None


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


def _private(x: float) -> None:
    pass


def __getattr__(name: str) -> object:
    return name


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

    def _helper(self, scale: float) -> None:
        pass


@dataclass
class _Walk:
    steps: int
    seen: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.seen = []

    def step(self, n: int) -> None:
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
    assert not [c for c in calls if "Level(" in c]
    label = checks.labels[calls.index("    where(_acs_value(OpenDrivePose))") + 1]
    assert label == "where(pose: OpenDrivePose)"


_DEFAULTS_RENDERED = """
def nearest(x: float, max_distance: float = 150.0) -> float:
    return x


def run(cwd: str, *args: str, timeout: int, env: str = "") -> None:
    pass


def plain(x: float) -> None:
    pass


class Bridge:
    def configure(self, goal: float, waypoints: Optional[list[float]] = None) -> None:
        pass

    def start(self, world: float, *, skip: Collection[int] = ()) -> None:
        pass

    @staticmethod
    def make(size: float = 1.0) -> None:
        pass
"""


def test_a_function_with_defaults_is_called_without_them_too() -> None:
    # Every call passes every argument, so no default would be compiled: one
    # call more leaves out each argument that has one (#183).
    checks = render_library_checks(ast.parse(_DEFAULTS_RENDERED))
    calls = checks.source.splitlines()
    assert "    nearest(_acs_value(float), _acs_value(float))" in calls
    assert "    nearest(_acs_value(float))" in calls
    label = checks.labels[calls.index("    nearest(_acs_value(float))") + 1]
    assert label == "nearest(x: float), with the default of `max_distance`"
    # A `*args: T` is still given one T, so the positional parameters before
    # it are passed; the keyword-only one without a default is too.
    assert "    run(_acs_value(str), _acs_value(str), timeout=_acs_value(int))" in calls
    # A function without a default is called once.
    assert [c for c in calls if c.startswith("    plain(")] == [
        "    plain(_acs_value(float))"
    ]
    # A method through its class: its dispatch thunk fills in no default.
    assert "    Bridge.configure(_acs_value(Bridge), _acs_value(float))" in calls
    assert "    Bridge.start(_acs_value(Bridge), _acs_value(float))" in calls
    assert "    Bridge.make()" in calls


_LEFT_TO_DEFAULTS_RENDERED = """
from typing import Any, Callable, Optional


class Holder:
    def __init__(
        self, size: float, kind: type[Base] | None = None, *, tag: Any = None
    ) -> None:
        pass


def build(x: float, factory: Optional[Callable[..., float]] = None) -> None:
    pass


def needs(x: float, kind: type[Base]) -> None:
    pass


def spread(first: Any = None, *rest: int) -> None:
    pass
"""


def test_a_defaulted_parameter_codon_cannot_express_is_left_to_its_default() -> None:
    # Each parameter Codon cannot express has a default: the function is
    # called leaving them out, and only so, instead of being a problem (#199).
    checks = render_library_checks(ast.parse(_LEFT_TO_DEFAULTS_RENDERED))
    calls = checks.source.splitlines()
    assert [c for c in calls if c.startswith("    Holder(")] == [
        "    Holder(_acs_value(float))"
    ]
    label = checks.labels[calls.index("    Holder(_acs_value(float))") + 1]
    assert label == "Holder.__init__(size: float), with the default of `kind`, `tag`"
    assert [c for c in calls if c.startswith("    build(")] == [
        "    build(_acs_value(float))"
    ]
    # A required one is still a problem, as is a defaulted one the call
    # would pass (before a `*args: T`, which is given one T).
    assert not [c for c in calls if c.startswith(("    needs(", "    spread("))]
    lines = _LEFT_TO_DEFAULTS_RENDERED.splitlines()
    assert sorted(
        (lines[line - 1].split("(")[0], m.split(":")[0]) for line, m in checks.problems
    ) == [("def needs", "needs"), ("def spread", "spread")]
    assert all("Codon cannot express" in m for _, m in checks.problems)


_SETTERS_RENDERED = """
from typing import Any, Optional


class Gauge:
    @property
    def level(self) -> float:
        return 0.0

    @level.setter
    def level(self, value: float | str) -> None:
        pass

    @level.deleter
    def level(self) -> None:
        pass

    @property
    def note(self) -> Optional[str]:
        return None

    @note.setter
    def note(self, value: Optional[str]) -> None:
        pass

    @property
    def anything(self) -> int:
        return 0

    @anything.setter
    def anything(self, value: Any) -> None:
        pass
"""


def test_a_property_setter_is_called_by_assigning_the_property() -> None:
    # Codon compiles a setter only where the property is assigned: the check
    # assigns it, once per member of the value's annotation (#200).
    checks = render_library_checks(ast.parse(_SETTERS_RENDERED))
    calls = checks.source.splitlines()
    assert [c for c in calls if " = " in c] == [
        "    _acs_value(Gauge).level = _acs_value(float)",
        "    _acs_value(Gauge).level = _acs_value(str)",
        "    _acs_value(Gauge).note = _acs_value(Optional[str])",
    ]
    for call in [c for c in calls if " = " in c]:
        name = call.split(".")[1].split(" ")[0]
        assert checks.labels[calls.index(call) + 1] == f"Gauge.{name} (setter)"
    # The getter is read as before; the deleter is not called.
    assert calls.count("    _acs_value(Gauge).level") == 1
    lines = _SETTERS_RENDERED.splitlines()
    [(line, message)] = checks.problems
    assert lines[line - 1].strip() == "def anything(self, value: Any) -> None:"
    assert message.startswith("Gauge.anything (setter): Codon cannot express `Any`")


_OPTIONAL_RETURNS_RENDERED = """
from typing import Optional

import typesafe_carla.carla as carla


def find(world: carla.World) -> Optional[carla.Actor]:
    return None


def count() -> int:
    return 0


class Gauge:
    def where(self) -> carla.Transform | None:
        return None

    @property
    def note(self) -> Optional[str]:
        return None
"""


def test_the_type_an_optional_function_returns_is_realized_before_the_call() -> None:
    # Its `return None` becomes `return Optional[X]()`, which Codon 0.19 fails
    # to generate code for when nothing else realizes X (#202).
    checks = render_library_checks(ast.parse(_OPTIONAL_RETURNS_RENDERED))
    calls = checks.source.splitlines()
    start = calls.index("    _acs_value(carla.Actor)")
    assert calls[start : start + 2] == [
        "    _acs_value(carla.Actor)",
        "    find(_acs_value(carla.World))",
    ]
    assert checks.labels[start + 1] == "find, realizing its return type"
    assert "    count()" in calls
    assert not any("_acs_value(int)" in c for c in calls)
    for value, call in [
        ("carla.Transform", "_acs_value(Gauge).where()"),
        ("str", "_acs_value(Gauge).note"),
    ]:
        at = calls.index(f"    {call}")
        assert calls[at - 1] == f"    _acs_value({value})"


_FIELD_DEFAULTS_RENDERED = """
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class Docker:
    workspace: str
    launch: Sequence[str]
    colcon_args: Sequence[str] = ("--symlink-install",)
    env: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    timeout: float = 10.0
    required: int = field(kw_only=True)


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Defaults:
    k: float = 0.0
"""


def test_a_dataclass_is_constructed_without_its_field_defaults_too() -> None:
    # Every constructor call passes every field, so no field default would
    # be compiled: one call more passes only the fields without one (#188).
    checks = render_library_checks(ast.parse(_FIELD_DEFAULTS_RENDERED))
    calls = checks.source.splitlines()
    assert (
        "    Docker(_acs_value(str), _acs_value(Sequence[str]), "
        "_acs_value(Sequence[str]), _acs_value(dict[str, str]), "
        "timeout=_acs_value(float), required=_acs_value(int))" in calls
    )
    # A field after a defaulted one is passed by keyword; a `field()` with no
    # default (here keyword-only) is required.  The `Any` field is left out
    # of both calls.
    required = (
        "    Docker(_acs_value(str), _acs_value(Sequence[str]), "
        "required=_acs_value(int))"
    )
    assert required in calls
    label = checks.labels[calls.index(required) + 1]
    assert label == (
        "Docker.__init__(workspace: str, launch: Sequence[str], required: int), "
        "with the default of `colcon_args`, `env`, `timeout`"
    )
    # A dataclass without a default is constructed once; one with only
    # defaulted fields is constructed with none too.
    assert [c for c in calls if c.startswith("    Point(")] == [
        "    Point(_acs_value(float), _acs_value(float))"
    ]
    assert "    Defaults()" in calls


_CLASS_DEFAULTS_RENDERED = """
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Optional, TypeVar

T = TypeVar("T")
DEFAULT_PORT = 8000
LOW = 0.0


class Action:
    REISSUES: ClassVar[bool] = False
    KIND: ClassVar = "action"
    port: int = DEFAULT_PORT
    label: Optional[str] = None
    ready: bool = True
    LIMIT: float = 2.0
    TWICE: float = LIMIT * 2
    LOW, HIGH = 0.0, 1.0
    MID: float = (LOW + HIGH) / 2
    SQUARES: List[int] = [i * i for i in range(3)]
    held: T = None
    __slots__ = ("_value",)

    def __init__(self) -> None:
        pass


class Follow(Action):
    REISSUES = True
    port = 9000
    KIND = "follow"
    undeclared = 1


class Mixin:
    backend: Optional[str] = None


class _Memo:
    generation: int = -1


_MEMO = _Memo()


@dataclass
class Config:
    timeout: float = 1.0


class Color(Enum):
    RED: str = "red"
"""


def test_a_class_level_default_is_assigned_to_a_typed_local() -> None:
    # Codon makes `x: T = v` of a class with a constructor a field and drops
    # `v`: the check assigns it to a local of type T, so it is compiled (#190).
    checks = render_library_checks(ast.parse(_CLASS_DEFAULTS_RENDERED))
    lines = checks.source.splitlines()
    defaults = {
        checks.labels[n]: line.split(": ", 1)[1]
        for n, line in enumerate(lines, 1)
        if "_acs_cls_default_" in line
    }
    assert defaults == {
        "class-level default of Action.port": "int = DEFAULT_PORT",
        "class-level default of Action.label": "Optional[str] = None",
        # Codon converts any value assigned to a `bool`; a call does not.
        "class-level default of Action.ready": "bool = _expect_bool(True)",
        "class-level default of Action.LIMIT": "float = 2.0",
        # A comprehension's variable is not a member of the class.
        "class-level default of Action.SQUARES": "List[int] = [i * i for i in range(3)]",
        # An override without an annotation takes its base's, a ClassVar's T.
        "class-level default of Follow.REISSUES": "bool = _expect_bool(True)",
        "class-level default of Follow.port": "int = 9000",
        # A class with no constructor is checked too: what derives from it
        # may have one.
        "class-level default of Mixin.backend": "Optional[str] = None",
    }
    # Each local is named after its line, so no two share a name.
    for n, line in enumerate(lines, 1):
        if "_acs_cls_default_" in line:
            assert line.startswith(f"    _acs_cls_default_{n}: ")
    # Left out: a ClassVar (Codon keeps its value) and an override of a bare
    # one, a value naming another member (`LIMIT * 2`, or `LOW`, which an
    # unpacking binds and the module binds too), a TypeVar, a dunder,
    # a name no base declares, the defaults of a class the module constructs
    # with no argument and no constructor of its own (Codon's generated one
    # takes them), and a dataclass's and an enum's.
    for absent in (
        "Action.REISSUES",
        "Action.KIND",
        "Follow.KIND",
        "Action.TWICE",
        "Action.MID",
        "Action.held",
        "__slots__",
        "Follow.undeclared",
        "_Memo.generation",
        "Config.timeout",
        "Color.RED",
    ):
        assert not [label for label in defaults if label.endswith(absent)]
    assert "from autoware_carla_scenario._tc import _acs_value, _expect_bool" in lines


def test_a_generic_function_and_one_returning_any_are_problems() -> None:
    # `_acs_value(list[T])` does not compile: a TypeVar parameter is a
    # problem.  A function returning `Any` builds what Codon cannot express
    # (`get_details() -> dict[str, Any]`): left out with `skip_any_returns`,
    # which the scenario check passes for want of UNCALLED (#192).
    source = textwrap.dedent(
        """
        from typing import Any, TypeVar

        T = TypeVar("T")


        def first(xs: list[T]) -> T:
            return xs[0]


        class Probe:
            def get_details(self) -> dict[str, Any]:
                return {}

            @property
            def raw(self) -> "Any":
                return 0

            def speed(self) -> float:
                return 0.0
        """
    )
    tree = ast.parse(source)
    checks = render_library_checks(tree)
    assert [line for line, _ in checks.problems] == [7]
    assert "parameter `xs` is annotated with a TypeVar" in checks.problems[0][1]
    assert "first(" not in checks.source
    assert "_acs_value(Probe).get_details()" in checks.source

    checks = render_library_checks(tree, skip_any_returns=True)
    assert [line for line, _ in checks.problems] == [7, 12, 16]
    assert "Probe.get_details: returns `dict[str, Any]`" in checks.problems[1][1]
    assert "Probe.raw: returns `Any`" in checks.problems[2][1]
    assert "get_details" not in checks.source
    assert ".raw" not in checks.source
    assert "_acs_value(Probe).speed()" in checks.source


def test_each_appended_line_names_the_line_it_compiles() -> None:
    # The scenario check reports an error on an appended line at the line of
    # what it compiles (#192): a function's `def`, a class-level default.
    source = _CLASS_DEFAULTS_RENDERED
    checks = render_library_checks(ast.parse(source))
    module = source.splitlines()
    assert set(checks.lines) == set(checks.labels)
    for n, label in checks.labels.items():
        line = module[checks.lines[n] - 1]
        if label.startswith("class-level default of "):
            member = label.rsplit(".", 1)[1]
            assert line.lstrip().startswith(member), (label, line)
        else:
            owner, _, name = label.split("(", 1)[0].rpartition(".")
            # A dataclass's generated constructor is at its `class` line.
            assert f"def {name}(" in line or (
                name == "__init__" and f"class {owner}" in line
            ), (label, line)


def test_protected_functions_and_methods_are_called_too() -> None:
    # A `_` function or method is a hook a subclass or another module uses
    # (BaseScenario._setup_ego_spawn): Codon compiles its body only when
    # something calls it, so the check calls it as it calls a public one.
    calls = render_library_checks(ast.parse(_MODULE)).source.splitlines()
    assert "    _private(_acs_value(float))" in calls
    assert "    _acs_value(Vector)._helper(_acs_value(float))" in calls
    # So are the methods of a `_` class: what uses such a helper may not
    # compile them (an UNCALLED constructor, a call behind `hasattr`), and
    # Codon's constructor does not call a dataclass's __post_init__ (#182).
    assert "    _Walk(_acs_value(int))" in calls
    assert "    _acs_value(_Walk).__post_init__()" in calls
    assert "    _acs_value(_Walk).step(_acs_value(int))" in calls
    # A dunder of the module (PEP 562's lazy `__getattr__`) is Python's hook,
    # not the module's code: it is not called.
    assert not [c for c in calls if "__getattr__" in c]
    # One the check cannot call with is a problem of the module, as a public
    # one is, unless UNCALLED names it.
    tree = ast.parse("def _anything(x: Any) -> None:\n    pass\n")
    assert render_library_checks(tree).problems
    assert not render_library_checks(tree, ["_anything"]).problems


def test_an_optional_union_parameter_is_called_with_each_member() -> None:
    source = """
def lookup(ref: Optional[Union[EntityRole, str]]) -> None:
    pass


def override(value: Optional[float | str]) -> None:
    pass
"""
    checks = render_library_checks(ast.parse(source))
    calls = checks.source.splitlines()
    assert checks.problems == []
    assert "    lookup(_acs_value(EntityRole))" in calls
    assert "    lookup(_acs_value(str))" in calls
    assert "    override(_acs_value(float))" in calls
    assert "    override(_acs_value(str))" in calls


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


def maybe(f: Optional[Callable[[float], int]] = None) -> None:
    pass


def place(f: Callable[[Union[Straight, Turn]], float]) -> None:
    pass


def refer(f: Optional[Callable[[Optional[Union[EntityRole, str]]], Pose]]) -> None:
    pass


def unknown(f: Callable[[Union[Straight, Any]], float]) -> None:
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
    # An Optional one: called with the callable.
    assert "    maybe(_acs_value(Callable[[float], int]))" in calls
    # A union among the arguments, a Callable type Codon 0.19 crashes on: a
    # callable of as many arguments, of any type, returning the result's type.
    assert "    place(_acs_value(_acs_Returning[float, 1]))" in calls
    assert "    refer(_acs_value(_acs_Returning[Pose, 1]))" in calls
    lines = _CALLABLES.splitlines()
    assert sorted(lines[line - 1].split("(")[0] for line, _ in checks.problems) == [
        "def anything",
        "def unknown",
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
    # fields after it are passed by keyword.  The last call passes only the
    # field without a default, so the defaults are compiled (#188).
    assert calls == [
        *(
            f"Run(_acs_value(str), _acs_value({scenario}), "
            f"npc=_acs_value({npc}), timeout=_acs_value(float))"
            for scenario in ("Straight", "Turn")
            for npc in ("Straight", "Turn")
        ),
        "Run(_acs_value(str))",
    ]
    label = checks.labels[checks.source.splitlines().index("    " + calls[-2]) + 1]
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
        # BridgedEgoEntity derives from EgoVehicle, which imports
        # scenario_base through the model, and AutowareEgoEntity from it.
        "entity.bridged_ego",
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


_PROTECTED = """
from __future__ import annotations


class Gauge:
    size: float

    def __init__(self, size: float) -> None:
        self.size = size

    def _scaled(self, k: float) -> float:
        return self.size * k


def _doubled(x: float) -> float:
    return x * 2.0


class _Meter:
    reading: float

    def __init__(self, reading: float) -> None:
        self.reading = reading

    def offset(self, k: float) -> float:
        return self.reading - k
"""


@needs_codon
def test_protected_functions_and_methods_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_PROTECTED, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong", "realized"),
    [
        ("return self.size * k", "return self.size + str(k)", "_scaled"),
        ("return x * 2.0", "return x + 'two'", "_doubled"),
        ("return self.reading - k", "return self.reading - str(k)", "offset"),
    ],
    ids=["method", "function", "private-class-method"],
)
def test_an_error_in_a_protected_function_or_method_is_reported(
    right: str,
    wrong: str,
    realized: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing public calls _scaled(), _doubled() or _Meter.offset(): only the
    # check's own call compiles them, so an error in any is found only
    # because of it.
    source = _PROTECTED.replace(right, wrong)
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    stripped = [text.strip() for text in textwrap.dedent(source).splitlines()]
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].path == str(tmp_path / "zz_library_check_case.py")
    assert located[0].line == stripped.index(wrong) + 1, result.format()
    assert realized in result.format()


_DEFAULTS = """
from __future__ import annotations

from typing import Callable, Collection, Dict, List, Optional, Sequence

from autoware_carla_scenario.kinematics.vector import Vector3


def nearest(x: float, max_distance: float = 150.0) -> float:
    return x + max_distance


class Route:
    _waypoints: List[Vector3]

    def __init__(self, goal: Vector3, waypoints: Sequence[Vector3] = ()) -> None:
        self._waypoints = list(waypoints)


class Launch:
    _args: Sequence[str]

    def __init__(self, args: Sequence[str] = ("--symlink-install", "-v")) -> None:
        self._args = args


def hold(ids: List[int], spare: Collection[int] = ()) -> int:
    return len([i for i in ids if i not in spare])


def first(poses: Optional[Sequence[Vector3]] = None) -> int:
    kept: List[Vector3] = []
    if poses:
        kept = list(poses)
    return len(kept)


def tally(ids: List[int] = [], names: Dict[str, Vector3] = {}) -> float:
    return sum(ids) + sum(v.x for v in names.values())


class Box:
    def add(self, more: List[Vector3] = [], *, by: Dict[str, float] = {}) -> int:
        return len(more) + len(by)


def height(ground: Optional[Callable[[float], Optional[float]]] = None) -> float:
    z: Optional[float] = None
    if ground is not None:
        z = ground(1.0)
    if z is None:
        return 0.0
    return z
"""


@needs_codon
def test_defaults_compile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The calls without the defaults compile a `()` of a Sequence[T] (an empty
    # List[T]), a `None` of an Optional the transform drops, and an empty
    # `[]` / `{}`, which Codon would make a list of None (#183).  A tuple
    # display of a Sequence[T] is a List[T] too, so it can be kept in a
    # Sequence[T] attribute (#188).
    result = _check_case(_DEFAULTS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
def test_a_default_of_the_wrong_type_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every other call passes `max_distance`: only the one leaving it out
    # compiles the default.
    source = _DEFAULTS.replace(
        "max_distance: float = 150.0", 'max_distance: float = "far"'
    )
    assert source != _DEFAULTS
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    assert "with the default of `max_distance`" in result.format(), result.format()


_INIT_AND_SETTER = """
from __future__ import annotations

from typing import TYPE_CHECKING

from autoware_carla_scenario.kinematics.vector import Vector3


class Holder:
    kind: type[Vector3]
    _size: float
    _scale: float

    def __init__(self, size: float, kind: type[Vector3] | None = None) -> None:
        self._size = size
        if not TYPE_CHECKING:
            self.kind = kind or Vector3
        self._scale = 2.0


class Gauge:
    _level: float

    def __init__(self) -> None:
        self._level = 0.0

    @property
    def level(self) -> float:
        return self._level

    @level.setter
    def level(self, value: float | int) -> None:
        self._level = float(value)
"""


@needs_codon
def test_a_constructor_left_to_its_default_and_a_setter_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_INIT_AND_SETTER, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong", "label"),
    [
        (
            "self._scale = 2.0",
            'self._scale = "two"',
            "Holder.__init__(size: float), with the default of `kind`",
        ),
        (
            "self._level = float(value)",
            "self._level = str(value)",
            "Gauge.level (setter)",
        ),
    ],
    ids=["constructor-left-to-default", "setter"],
)
def test_an_error_in_a_constructor_left_to_its_default_or_a_setter_is_reported(
    right: str,
    wrong: str,
    label: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing calls Holder() or assigns Gauge.level: only the check's own
    # calls compile them (#199, #200).
    source = _INIT_AND_SETTER.replace(right, wrong)
    assert source != _INIT_AND_SETTER
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    stripped = [text.strip() for text in textwrap.dedent(source).splitlines()]
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].path == str(tmp_path / "zz_library_check_case.py")
    assert located[0].line == stripped.index(wrong) + 1, result.format()
    assert f"<library check: {label}>" in result.format(), result.format()


_FIELD_DEFAULTS = """
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from autoware_carla_scenario.kinematics.vector import Vector3


@dataclass
class Launch:
    workspace: str
    colcon_args: Sequence[str] = ("--symlink-install", "--merge-install")
    one: Sequence[str] = ("a",)
    sizes: tuple[float, ...] = field(default=(1.0, 2.0))
    origin: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    env: Dict[str, str] = field(default_factory=dict)
    max_distance: float = 150.0

    def arguments(self) -> List[str]:
        return [self.workspace] + list(self.colcon_args) + list(self.one)
"""


@needs_codon
def test_field_defaults_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The constructor call passing only the required fields compiles each
    # field default against its field's type: a tuple display of a
    # `Sequence[str]` or `tuple[float, ...]` field is a list of it (#188).
    result = _check_case(_FIELD_DEFAULTS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
def test_a_field_default_of_the_wrong_type_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every other constructor call passes `max_distance`: only the one
    # passing the required fields alone compiles the default.
    source = _FIELD_DEFAULTS.replace(
        "max_distance: float = 150.0", 'max_distance: float = "far"'
    )
    assert source != _FIELD_DEFAULTS
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    assert "Launch.__init__(workspace: str), with the default of" in (
        result.format()
    ), result.format()


_CLASS_DEFAULTS = """
from __future__ import annotations

from typing import ClassVar, List, Optional

from autoware_carla_scenario.kinematics.vector import Vector3

DEFAULT_PORT = 8000


class Action:
    REISSUES: ClassVar[bool] = False
    port: int = DEFAULT_PORT
    scale: float = 1
    origin: Optional[Vector3] = None
    names: List[str] = []
    ready: bool = True

    def __init__(self) -> None:
        pass

    def describe(self) -> str:
        return str(self.port) + str(self.scale) + str(self.ready)


class Follow(Action):
    REISSUES = True
    port = 9000

    def __init__(self) -> None:
        super().__init__()


def tuples(xs: List[int]) -> tuple[int, ...]:
    return tuple(xs)
"""


@needs_codon
def test_class_level_defaults_and_tuple_returns_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each class-level default is compiled against its annotation (#190), an
    # override without one against its base's, and a `tuple[X, ...]` return
    # is checked as a List[X] (#191).
    result = _check_case(_CLASS_DEFAULTS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@pytest.mark.parametrize(
    ("right", "wrong", "reported"),
    [
        (
            "port: int = DEFAULT_PORT",
            'port: int = "x"',
            "class-level default of Action.port",
        ),
        (
            "origin: Optional[Vector3] = None",
            "origin: Optional[Vector3] = 1.0",
            "class-level default of Action.origin",
        ),
        # Codon converts any value assigned to a `bool`: the check calls.
        ("ready: bool = True", 'ready: bool = "x"', "Action.ready"),
        ("    REISSUES = True", '    REISSUES = "x"', "Follow.REISSUES"),
        ("    port = 9000", '    port = "x"', "Follow.port"),
        # Nothing uses what `tuples` returns: only its annotation checks it.
        (
            "    return tuple(xs)",
            "    return 1",
            "'List[int]' does not match expected type 'int'",
        ),
    ],
)
@needs_codon
def test_a_wrong_class_level_default_or_tuple_return_is_reported(
    right: str,
    wrong: str,
    reported: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Before #190 and #191, each of these compiled: Codon dropped the default
    # of a class with a constructor, and the check the tuple return.
    assert _CLASS_DEFAULTS.count(right) == 1
    result = _check_case(_CLASS_DEFAULTS.replace(right, wrong), tmp_path, monkeypatch)
    assert not result.ok
    assert reported in result.format(), result.format()


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


_SINGLETON_AND_UNION_CALLABLE = """
from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union

from autoware_carla_scenario.entity_role import EntityRole
from autoware_carla_scenario.kinematics.vector import Vector3

#: A callable alias, which Codon 0.19 refuses as an expression.
Lookup = Callable[[Optional[Union[EntityRole, str]]], Vector3]


class Single:
    _instance: ClassVar[Optional["Single"]] = None
    size: Optional[float]

    def __new__(cls) -> "Single":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.size = None
        return cls._instance

    def __init__(self) -> None:
        pass


def length(
    lookup: Callable[[Optional[Union[EntityRole, str]]], Vector3],
) -> float:
    return lookup(EntityRole.ego()).dot(lookup(None)) + (Single().size or 0.0)
"""


@needs_codon
def test_a_singleton_and_a_callable_of_a_union_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # super().__new__(cls) allocates the class, and a callable with a union
    # argument is called with one of any arguments returning its result type.
    right = _SINGLETON_AND_UNION_CALLABLE
    result = _check_case(right, tmp_path, monkeypatch)
    assert result.ok, result.format()
    # What the callable returns is typed: a Vector3 has no norm_of().
    wrong = right.replace(".dot(lookup(None))", ".norm_of(lookup(None))")
    assert wrong != right
    assert not _check_case(wrong, tmp_path, monkeypatch).ok
    # Nor does a call with another number of arguments than the Callable's.
    arity = right.replace(".dot(lookup(None))", ".dot(lookup(None, None))")
    assert arity != right
    assert not _check_case(arity, tmp_path, monkeypatch).ok


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


_ELSE_IMPORT = """
from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
else:
    from tqdm import tqdm


def wait(steps: int) -> None:
    if TYPE_CHECKING:
        for _ in range(steps):
            time.sleep(0.1)
    else:
        import json  # noqa: PLC0415

        for _ in tqdm(range(steps), desc=json.dumps("wait")):
            time.sleep(0.1)
"""


@needs_codon
def test_an_import_in_the_else_branch_of_type_checking_is_not_compiled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codon reads TYPE_CHECKING as a compile-time true and compiles only the
    # first branch, so what the `else` branch imports needs no model (#126).
    result = _check_case(_ELSE_IMPORT, tmp_path, monkeypatch)
    assert result.ok, result.format()


def test_an_import_in_the_first_branch_of_type_checking_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swapped = _ELSE_IMPORT.replace(
        "    pass\nelse:\n    from tqdm import tqdm",
        "    from tqdm import tqdm\nelse:\n    pass",
    )
    assert swapped != _ELSE_IMPORT
    result = _check_case(swapped, tmp_path, monkeypatch)
    assert not result.ok
    assert "tqdm has no Codon model" in result.diagnostics[0].message


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


def log_text(run: Run) -> str:
    return (run.output_dir / "fcd.xml").read_text(encoding="utf-8")


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
    # Python builds and joins a Path with a str or an os.PathLike only.
    for part in ("Path(configured, 1)", "(Path(configured) / 1)"):
        wrong = _PATHLIB.replace("Path(configured)", part, 1)
        assert wrong != _PATHLIB
        result = _check_case(wrong, tmp_path, monkeypatch)
        assert not result.ok
        assert "a str or an os.PathLike only" in result.format(), result.format()


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


_STEM = """
from __future__ import annotations

from pathlib import Path


def numbered(path: Path, counter: int) -> Path:
    return path.with_name(f"{path.stem}_{counter}{path.suffix}")
"""


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        ("", ""),
        # A property in CPython, and with_name() takes the name: each is a
        # TypeError in CPython 3.10.
        ("{path.stem}", "{path.stem()}"),
        ("path.with_name(f", "path.with_name(1, f"),
    ],
    ids=["right", "stem-is-a-property", "with-name-takes-one-name"],
)
def test_the_pathlib_shim_has_the_stem_and_with_name(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # scenario_runner numbers a run's video after the file it would replace (#126).
    source = _STEM.replace(right, wrong, 1) if right else _STEM
    result = _check_case(source, tmp_path, monkeypatch)
    assert result.ok is (not right), result.format()


# ---------------------------------------------------------------------------
# The os and shutil shims, dict.get(key), and the missing exceptions (#117)
# ---------------------------------------------------------------------------


_OS_SHUTIL = """
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


class MissingTool(FileNotFoundError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def configured(name: str) -> Optional[Path]:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def environment(name: str) -> str:
    unset: Optional[str] = os.environ.get(name)
    fallback = os.getenv(name, "fallback")
    maybe: Optional[str] = os.getenv(key=name)
    os.environ[name] = fallback
    env = os.environ.copy()
    names = [key for key in os.environ if key.startswith("ACS_")]
    return f"{unset} {maybe} {len(env)} {os.pathsep.join(names)}"


def lookup(table: dict[str, int], key: str) -> Optional[int]:
    return table.get(key)


def processes(pgid: int) -> int:
    os.killpg(os.getpgid(os.getpid()), 15)
    os.kill(pgid, 9)
    return os.system("true")


def install(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("x", encoding="utf-8")
    shutil.copy2(source, destination)
    copied: str = shutil.copy2(str(source), "target.xodr", follow_symlinks=False)
    moved = shutil.move(src=source, dst=destination)
    shutil.rmtree(destination.parent, ignore_errors=True)
    found: Optional[str] = shutil.which("git")
    joined = os.path.join(destination, "a")
    return f"{copied} {Path(moved)} {found} {joined}"


def guarded(path: Path) -> str:
    try:
        return path.name
    except ModuleNotFoundError as exc:
        return str(exc)
    except ImportError:
        raise MissingTool("no tool")
    except PermissionError:
        raise TimeoutError("slow")
    except BrokenPipeError:
        raise ProcessLookupError(path)
    except ConnectionError as exc:
        raise OSError(str(exc))
"""


@needs_codon
def test_the_os_and_shutil_shims_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "os" / "__init__.codon").is_file()
    assert (model_dir() / "shutil.codon").is_file()
    result = _check_case(_OS_SHUTIL, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError or an AttributeError in CPython 3.10, the oldest
        # Python the package supports (rmtree() takes `onexc` from 3.12 on).
        ('os.environ.get(name, "")', 'os.environ.get(name, "", "")'),
        ("os.environ.copy()", 'os.environ.fromkeys(["a"])'),
        ("os.environ[name] = fallback", "os.environ[name] = len(fallback)"),
        ("os.getenv(key=name)", "os.getenv(name=name)"),
        ("table.get(key)", "table.get(key=key)"),
        ("os.kill(pgid, 9)", "os.kill(pid=pgid, signal=9)"),
        ("os.getpgid(os.getpid())", "os.getpgid(os.getpid(), 0)"),
        ('os.system("true")', 'os.system(cmd="true")'),
        ("mkdir(parents=True, exist_ok=True)", "mkdir(exist_ok=True, missing_ok=True)"),
        ('write_text("x", encoding="utf-8")', "write_text(1)"),
        ("follow_symlinks=False)", "False)"),
        ("ignore_errors=True)", "ignore_errors=True, onexc=None)"),
        ('shutil.which("git")', 'shutil.which("git", 1, None, True)'),
        ("dst=destination)", "dest=destination)"),
        ('os.path.join(destination, "a")', "os.path.join(destination, 1)"),
        ('raise TimeoutError("slow")', 'raise TimeoutError(message="slow")'),
        ('raise MissingTool("no tool")', 'raise ImportError(message="no tool")'),
    ],
    ids=[
        "environ-get-arguments",
        "environ-is-no-dict",
        "environ-value-str",
        "getenv-keyword",
        "dict-get-positional-only",
        "kill-positional-only",
        "getpgid-arguments",
        "system-keyword",
        "mkdir-keyword",
        "write-text-str",
        "copy2-follow-symlinks-keyword-only",
        "rmtree-onexc",
        "which-arguments",
        "move-keyword",
        "join-path",
        "oserror-no-keyword",
        "importerror-keyword",
    ],
)
def test_the_os_and_shutil_shims_refuse_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _OS_SHUTIL.replace(right, wrong, 1)
    assert source != _OS_SHUTIL
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


@needs_codon
def test_what_shutil_returns_for_a_path_is_only_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # copy2() and move() return their Path destination, or a str when it names
    # a directory: the check takes str() and Path() of it, and nothing else.
    wrong = _OS_SHUTIL.replace("{Path(moved)}", "{moved.name}", 1)
    assert wrong != _OS_SHUTIL
    result = _check_case(wrong, tmp_path, monkeypatch)
    assert not result.ok
    assert "name" in result.format()


# ---------------------------------------------------------------------------
# The subprocess and importlib.metadata shims, str.removesuffix(), a typed
# *args and except (A, B) (#118)
# ---------------------------------------------------------------------------


_SUBPROCESS = """
from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional


def installed(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        return str(exc)


def git(repo: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            cwd=repo,
            env={"LANG": "C"},
            check=False,
        )
        done = subprocess.run(["git"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or done.stderr is None:
        return None
    return f"{result.stdout} {done.args[0]}".strip()


def url(text: str, repo: Path, base: Path) -> str:
    inside = repo.relative_to(base).as_posix()
    return text.removesuffix(".git").removeprefix("git+") + inside + str(Path.cwd())


def spawn(repo: Path) -> int:
    process: subprocess.Popen[bytes] = subprocess.Popen(
        ["sleep", "1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        cwd=str(repo),
        start_new_session=True,
    )
    if process.poll() is None:
        process.terminate()
    return process.wait(timeout=5.0)


def checked() -> subprocess.CompletedProcess[str]:
    done = subprocess.CompletedProcess[str](["true"], 0, "out", None)
    try:
        done.check_returncode()
    except subprocess.CalledProcessError as exc:
        raise subprocess.TimeoutExpired(exc.cmd, 1, output=exc.stdout)
    return done
"""


@needs_codon
def test_the_subprocess_and_metadata_shims_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (model_dir() / "subprocess.codon").is_file()
    assert (model_dir() / "importlib" / "metadata.codon").is_file()
    result = _check_case(_SUBPROCESS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each is a TypeError or an AttributeError in CPython 3.10, the oldest
        # Python the package supports (process_group is 3.11's, walk_up
        # 3.12's), but relative-to-parts, which 3.10 takes and 3.14 refuses,
        # and split-handler-body, a body of the wrong type in the
        # handler `except (A, B):` becomes two of. The ones after that handler
        # are located past the lines its copy adds.
        ("metadata.version(name)", "metadata.version(distribution=name)"),
        ("metadata.version(name)", 'metadata.version(name, "x")'),
        ('["git"], capture_output=True', '["git"], capture_outputs=True'),
        (
            '["git"], capture_output=True',
            '["git"], process_group=0, capture_output=True',
        ),
        ("        return None\n    if result", "        return 1\n    if result"),
        ("done.stderr is None", "done.output is None"),
        ('text.removesuffix(".git")', 'text.removesuffix(suffix=".git")'),
        ("repo.relative_to(base)", "repo.relative_to(base, walk_up=True)"),
        ("repo.relative_to(base)", 'repo.relative_to(base, "x")'),
        ("repo.relative_to(base)", "repo.relative_to(other=base)"),
        ("str(Path.cwd())", "str(Path.cwd(repo))"),
        ("process.wait(timeout=5.0)", "process.wait(5.0, 1)"),
        ("subprocess.TimeoutExpired(exc.cmd, 1,", "subprocess.TimeoutExpired(exc.cmd,"),
    ],
    ids=[
        "version-keyword",
        "version-arguments",
        "run-unknown-keyword",
        "run-process-group",
        "split-handler-body",
        "completed-has-no-output",
        "removesuffix-positional-only",
        "relative-to-walk-up",
        "relative-to-parts",
        "relative-to-keyword",
        "cwd-arguments",
        "wait-arguments",
        "timeout-expired-arguments",
    ],
)
def test_the_subprocess_and_metadata_shims_refuse_what_cpython_refuses(
    right: str, wrong: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _SUBPROCESS.replace(right, wrong, 1)
    assert source != _SUBPROCESS
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    first = wrong.splitlines()[0]
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if first in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()


def test_a_typed_star_args_is_called_with_one_value() -> None:
    checks = render_library_checks(
        ast.parse(
            "def git(repo: Path, *args: str, timeout: int) -> None: ...\n"
            "def log(*values: object) -> None: ...\n"
            "def anything(*values) -> None: ...\n"
            "class Context:\n"
            "    def __exit__(self, *args: object) -> None: ...\n"
        )
    )
    calls = checks.source.splitlines()
    assert (
        "    git(_acs_value(Path), _acs_value(str), timeout=_acs_value(int))" in calls
    )
    assert "    log()" in calls
    assert "    anything()" in calls
    assert "    _acs_value(Context).__exit__()" in calls
    assert not checks.problems


# ---------------------------------------------------------------------------
# The signal and atexit shims, a Popen's pipes, and the dotenv, ffmpeg and
# carla_install models (#119)
# ---------------------------------------------------------------------------


_PROCESS_SIGNALS = """
from __future__ import annotations

import atexit
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Optional

import ffmpeg
import numpy as np
from dotenv import load_dotenv

from .carla_install import installed_executable


class Recorder:
    _process: Optional[subprocess.Popen[bytes]]

    def __init__(self) -> None:
        self._process = None

    def start(self, size: str) -> None:
        load_dotenv(override=False)
        load_dotenv(Path(".env"), None, True)
        self._process = (
            ffmpeg.input("pipe:", format="rawvideo", s=size, r=20.0)
            .output("out.mp4", vcodec="libx264", crf=23)
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True)
        )
        atexit.register(self.stop)
        atexit.register(self.finish, 3, reason="exit")
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _on_signal(self, signum: int, frame: Any) -> None:
        self.stop()

    def finish(self, code: int, reason: str = "") -> None:
        self.stop()

    def write(self) -> int:
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        if self._process is None or self._process.stdin is None:
            return 0
        return self._process.stdin.write(bgr.data) + self._process.stdin.write(b"raw")

    def stop(self) -> None:
        if self._process is not None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            self._process.wait()
            if self._process.stderr is not None:
                logged: str = self._process.stderr.read()
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            atexit.unregister(self.stop)
            self._process = None


def launcher() -> Optional[Path]:
    return installed_executable()
"""


def _check_signals_case(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TypeCheckResult:
    # A signal handler's frame is CPython's frame object, which Codon has no
    # type for (`Any`): the handler is compiled through signal.signal(), which
    # calls it as CPython does, not called with values of its own.
    monkeypatch.setitem(
        check.UNCALLED,
        f"{_FAKE}.Recorder._on_signal",
        "a signal handler, whose frame Codon has no type for",
    )
    return _check_case(source, tmp_path, monkeypatch)


@needs_codon
def test_the_signal_atexit_and_pipe_shims_and_their_models_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for shim in ("signal.codon", "atexit.codon", "dotenv", "ffmpeg"):
        assert (model_dir() / shim).exists(), shim
    result = _check_signals_case(_PROCESS_SIGNALS, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong", "at"),
    [
        # Each fails in CPython 3.10 (the oldest Python the package supports)
        # with the installed python-dotenv and ffmpeg-python: a TypeError, an
        # AttributeError or ffmpeg-python's ValueError when it is called, but
        # handler-arity and register-arguments, which CPython takes and fails
        # on later, with a TypeError when the signal arrives and one it
        # reports at exit. Codon reports a method a stream of the chain lacks,
        # and an error inside one, at the chain's first line (`at`).
        ("signal.SIG_IGN)", "5)", "signal.SIGINT, 5)"),
        ("signal.SIGKILL)", "signal.SIGKILLED)", None),
        ("signal.SIGTERM, self._on_signal)", "signal.SIGTERM, self.stop)", None),
        ("atexit.register(self.stop)", "atexit.register(func=self.stop)", None),
        ('3, reason="exit")', '3, why="exit")', None),
        ("atexit.unregister(self.stop)", "atexit.unregister(func=self.stop)", None),
        ("load_dotenv(override=False)", "load_dotenv(overwrite=False)", None),
        (
            '            .output("out.mp4", vcodec="libx264", crf=23)\n',
            "",
            "ffmpeg.input(",
        ),
        ('.output("out.mp4",', '.output(Path("out.mp4"),', "ffmpeg.input("),
        ("run_async(pipe_stdin=True,", "run_async(pipe_input=True,", None),
        ("stdin.write(bgr.data)", "stdin.write(bgr.size)", None),
        ("installed_executable()", "installed_executable(True)", None),
    ],
    ids=[
        "signal-int-handler",
        "signal-unknown-number",
        "handler-arity",
        "register-keyword",
        "register-arguments",
        "unregister-keyword",
        "dotenv-keyword",
        "input-has-no-overwrite-output",
        "output-file-name-str",
        "run-async-keyword",
        "write-bytes-like",
        "installed-executable-arguments",
    ],
)
def test_the_signal_atexit_and_pipe_shims_and_their_models_refuse_what_cpython_refuses(
    right: str,
    wrong: str,
    at: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _PROCESS_SIGNALS.replace(right, wrong, 1)
    assert source != _PROCESS_SIGNALS
    result = _check_signals_case(source, tmp_path, monkeypatch)
    assert not result.ok
    needle = at if at is not None else wrong
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if needle in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()


# ---------------------------------------------------------------------------
# The shlex and sys shims, signal.Signals, a file object (Path.open(), a
# Popen stream) and the symlink queries (#127)
# ---------------------------------------------------------------------------


_STACK = """
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import IO, Optional


def command(root: str, args: list[str]) -> str:
    argv = ["python3", "-m", "x y", *args]
    return shlex.quote(root) + " " + shlex.join(argv)


def interpreter() -> list[str]:
    if sys.platform != "linux":
        return list(sys.argv)
    return [sys.executable, "-c", "pass"]


def stop(pid: int, sig: signal.Signals) -> str:
    try:
        os.killpg(pid, sig)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    signals = [sig, signal.SIGTERM, signal.Signals(9)]
    name = signals[0].name
    return f"{name} {int(signal.SIGINT)} {signal.SIGINT == 2}"


def link(directory: Path, target: Path) -> Path:
    path = directory / "link"
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target, target_is_directory=True)
    return path


def opened(directory: Path) -> IO[bytes]:
    return (directory / "x.log").open("ab")


def logged(directory: Path) -> Optional[int]:
    log = opened(directory)
    log.write(b"start\\n")
    process: subprocess.Popen[bytes] = subprocess.Popen(
        ["true"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
    )
    code = process.wait()
    process.send_signal(signal.SIGTERM)
    log.close()
    return code
"""


@needs_codon
def test_a_checked_module_has_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Python sets a module's __file__, which Codon does not: the library check
    # gives each checked module its source's path.
    source = """
    from __future__ import annotations

    from pathlib import Path

    from .entity_role import __file__ as role_file


    def directory() -> Path:
        return Path(role_file).resolve().parent
    """
    result = _check_case(source, tmp_path, monkeypatch)
    assert result.ok, result.format()


def _top_level_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


@needs_codon
def test_the_sys_shim_keeps_all_of_codons_sys() -> None:
    # The shim replaces the sys module Codon ships, which Codon's own modules
    # (time, random, itertools) and typesafe_carla's library import: it
    # declares every name Codon's does.
    tc = available_toolchain()
    assert tc is not None
    codon_sys = tc.codon_dir / "lib" / "codon" / "stdlib" / "sys.codon"
    if not codon_sys.is_file():
        pytest.skip("this Codon ships no stdlib/sys.codon")
    shim = _top_level_names(model_dir() / "sys.codon")
    assert _top_level_names(codon_sys) <= shim
    assert {"platform", "executable"} <= shim


@needs_codon
def test_the_shlex_sys_signal_and_file_shims_take_what_cpython_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for shim in ("shlex.codon", "sys.codon", "signal.codon", "typing.codon"):
        assert (model_dir() / shim).is_file(), shim
    result = _check_case(_STACK, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong", "at"),
    [
        # Each fails in CPython 3.10 (the oldest Python the package supports)
        # and 3.13 when it is called: a TypeError, an AttributeError, or the
        # ValueError of a Signals looked up by anything but its number. Codon
        # reports a bad argument of a call written over several lines at the
        # call's first line (`at`).
        ("shlex.quote(root)", "shlex.quote(len(root))", None),
        ("shlex.join(argv)", "shlex.join([len(root)])", None),
        ('sys.platform != "linux"', 'sys.platfrom != "linux"', None),
        ("signals[0].name", "signals[0].label", None),
        ("os.killpg(pid, sig)", "os.killpg(pid, sig.name)", None),
        ("signal.Signals(9)", 'signal.Signals("SIGKILL")', None),
        ("target_is_directory=True)", "target_is_dir=True)", None),
        ("path.is_symlink()", "path.is_symlink(True)", None),
        ('.open("ab")', '.open("ab", buffer=0)', None),
        ('log.write(b"start\\n")', 'log.write(len(b"start\\n"))', None),
        ("stdout=log,", 'stdout="x.log",', "subprocess.Popen("),
    ],
    ids=[
        "quote-str",
        "join-str",
        "sys-unknown-name",
        "signal-unknown-attribute",
        "killpg-signal-number",
        "signals-by-number",
        "symlink-to-keyword",
        "is-symlink-arguments",
        "open-keyword",
        "write-bytes",
        "popen-stream-file",
    ],
)
def test_the_shlex_sys_signal_and_file_shims_refuse_what_cpython_refuses(
    right: str,
    wrong: str,
    at: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _STACK.replace(right, wrong, 1)
    assert source != _STACK
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    needle = at if at is not None else wrong
    line = next(
        i
        for i, text in enumerate(textwrap.dedent(source).splitlines(), 1)
        if needle in text
    )
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].line == line, result.format()


# ---------------------------------------------------------------------------
# What autoware_stack.docker reads: the ids, a link, a file's text and bytes,
# a recursive glob, the path queries and a Popen's output by line (#128)
# ---------------------------------------------------------------------------


_DOCKER = """
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def label() -> str:
    uid = os.getuid()
    gid = os.getgid()
    return f"uid-{uid}:{gid}"


def followed(path: Path, mount: str) -> Path:
    target = Path(os.readlink(path))
    if not target.is_absolute():
        target = path.parent / target
    elif target.is_relative_to(mount):
        target = path / target.relative_to(mount)
    return target


def digest(source: Path) -> str:
    sha = hashlib.sha256()
    for path in sorted(source.rglob("package.xml")):
        sha.update(path.read_bytes())
    return sha.hexdigest()


def supported(path: Path) -> bool:
    return 'name="x"' in path.read_text()


def streamed(argv: list[str]) -> int:
    process: subprocess.Popen[str] = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
    return process.wait()
"""


@needs_codon
def test_the_os_pathlib_and_popen_shims_take_what_docker_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_DOCKER, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        # Each fails in CPython 3.10 (the oldest Python the package supports)
        # and 3.14 when it is called: a TypeError or an AttributeError; but
        # is-relative-to-parts, which 3.10 takes and 3.14 refuses, and
        # read-text-newline, which 3.13 takes and 3.10 refuses.
        ("os.getuid()", "os.getuid(0)"),
        ("os.getgid()", "os.getgid(0)"),
        ("os.readlink(path)", "os.readlink(path, None)"),
        ("target.is_absolute()", "target.is_absolute(True)"),
        ("target.is_relative_to(mount)", "target.is_relative_to(len(mount))"),
        ("target.is_relative_to(mount)", 'target.is_relative_to(mount, "x")'),
        ("target.is_relative_to(mount)", "target.is_relative_to(other=mount)"),
        ('source.rglob("package.xml")', "source.rglob()"),
        ("path.read_bytes()", 'path.read_bytes("rb")'),
        ("path.read_text()", 'path.read_text(encodig="utf-8")'),
        ("path.read_text()", 'path.read_text(newline="")'),
        ("sys.stdout.write(line)", "sys.stdout.write(line.decode())"),
    ],
    ids=[
        "getuid-arguments",
        "getgid-arguments",
        "readlink-dir-fd-keyword",
        "is-absolute-arguments",
        "is-relative-to-path",
        "is-relative-to-parts",
        "is-relative-to-keyword",
        "rglob-pattern",
        "read-bytes-arguments",
        "read-text-keyword",
        "read-text-newline",
        "popen-line-is-str",
    ],
)
def test_the_os_pathlib_and_popen_shims_refuse_what_cpython_refuses(
    right: str,
    wrong: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _DOCKER.replace(right, wrong, 1)
    assert source != _DOCKER
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
