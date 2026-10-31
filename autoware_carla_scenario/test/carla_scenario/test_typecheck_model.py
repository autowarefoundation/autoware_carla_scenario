"""The Codon model declares the public API as the Python package defines it.

The static check compiles scenarios against ``typecheck/codon/`` instead of
the Python package, so a parameter renamed in Python and not in the model
would reject a correct scenario (or accept a wrong one).  These tests read
the model as Python syntax -- it is written to parse as Python -- and compare
every class and function ``autoware_carla_scenario/__init__.codon`` exports
with the object of the same name: its parameters (names, order, which are
keyword-only, which have defaults), the methods and properties it declares,
its enum members.  Every public name of a modelled module is declared or
listed, with the reason, in ``typecheck/not_modelled.py`` (#237), and every
model file is compared or exempt with the reason (#240).  They need no Codon.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import functools
import importlib
import importlib.util
import inspect
import itertools
import re
import textwrap
import types
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dotenv
import lxml.etree
import pytest
from pyxodr.road_objects.network import RoadNetwork as PyxodrRoadNetwork
from pyxodr.road_objects.road import Road as PyxodrRoad

import autoware_carla_scenario as acs
from autoware_carla_scenario import (
    action_state,
    carla_install,
    conditions,
    constants,
    coordinate,
    debug_markers,
    entity_role,
    environment,
    failure,
    kinematics,
    measures,
    metrics,
    odd,
    ros2,
    trajectory_recorder,
    weather_rules,
)
from autoware_carla_scenario import rerun as rerun_module
from autoware_carla_scenario import t4 as t4_module
from autoware_carla_scenario import t4_scene as t4_scene_module
from autoware_carla_scenario import waypoint_arrival
from autoware_carla_scenario.autoware_bridge import base as bridge_base
from autoware_carla_scenario.autoware_bridge import grpc_server as bridge_grpc_server
from autoware_carla_scenario.autoware_stack import devcontainer as stack_devcontainer
from autoware_carla_scenario.autoware_stack import launcher as stack_launcher
from autoware_carla_scenario.conditions.comparison import ComparisonRule
from autoware_carla_scenario.coordinate import map_manager as coordinate_map_manager
from autoware_carla_scenario.coordinate import projection as coordinate_projection
from autoware_carla_scenario.coordinate import transform as coordinate_transform
from autoware_carla_scenario.coverage import collector as coverage_collector
from autoware_carla_scenario.coverage import items as coverage_items
from autoware_carla_scenario.coverage import report as coverage_report
from autoware_carla_scenario.driver import egodriver_client
from autoware_carla_scenario.entity import carla_driver_entity, hils_entity
from autoware_carla_scenario.maps import cache as maps_cache
from autoware_carla_scenario.maps import source as maps_source
from autoware_carla_scenario.maps import road_network_cache as maps_road_network_cache
from autoware_carla_scenario.maps import z_offset_cache as maps_z_offset_cache
from autoware_carla_scenario.odd import model as odd_model
from autoware_carla_scenario.odd import probes as odd_probes
from autoware_carla_scenario.odd import registry as odd_registry
from autoware_carla_scenario.odd import route as odd_route
from autoware_carla_scenario.route import active as route_active
from autoware_carla_scenario.route import frame as route_frame
from autoware_carla_scenario.route import model as route_model
from autoware_carla_scenario.traffic import base as traffic_base
from autoware_carla_scenario.traffic import context as traffic_context
from autoware_carla_scenario.traffic import registry as traffic_registry
from autoware_carla_scenario.trajectory import model as trajectory_model
from autoware_carla_scenario.typecheck import available_toolchain, model_dir
from autoware_carla_scenario.typecheck.check import (
    _model_module,
    _model_modules,
    _model_names,
    _unmodelled_module,
)
from autoware_carla_scenario.typecheck.library import package_modules
from autoware_carla_scenario.typecheck.not_modelled import (
    NOT_MODELLED,
    NOT_MODELLED_MEMBERS,
    defines_members,
    not_modelled_member,
)
from autoware_carla_scenario.ui import models as ui_models
from autoware_carla_scenario.utils import traffic_light as utils_traffic_light
from carla_driver_interface import geometry as cdi_geometry

_MODEL = model_dir() / "autoware_carla_scenario"

#: Python members the model leaves out on purpose, by class.  Each is a
#: free-form mapping (dict[str, Any]), which has no Codon type.
_OMITTED_PARAMETERS: dict[str, set[str]] = {
    "SweepConfig": {"constraints", "bindings", "odd_sample"},
    # A condition's details (scenario_runner fills them, outside the check).
    "ConditionStatus": {"details"},
}

#: Model constants that are no literal, each compared by a test of its own.
_CONSTANTS_COMPARED_ON_THEIR_OWN = {
    "BUILT_IN_MEASURES",  # test_the_measures_model_holds_python_s_values
    "BUILT_IN_METRICS",  # test_the_metrics_model_holds_python_s_values
    "EGO_ROLE_NAME",  # test_the_constants_model_holds_python_s_values
    "URBAN_CONSTRUCTION",  # test_the_urban_construction_model_holds_python_s_values
    "UNDECIDED",  # test_the_undecided_model_is_openodd_s_sentinel
}

#: (kind, name, has default): kind is "pos", "kw", "*args" or "**kwargs".
Param = tuple[str, str, bool]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _exports() -> dict[str, str]:
    """Name -> model module (stem) for every name ``__init__.codon`` imports."""
    out: dict[str, str] = {}
    for node in _parse(_MODEL / "__init__.codon").body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = node.module.replace(".", "/")
    return out


#: The model module that defines a class (`id` of its node, which the cache of
#: _definitions keeps alive), to look its bases up where it names them.
_ORIGIN: dict[int, tuple[str, Path]] = {}


@functools.cache
def _definitions(module: str, root: Path = _MODEL) -> dict[str, list[ast.stmt]]:
    """Top-level definitions of a model module (a function may be overloaded).

    Cached: every module re-reads the modules it imports, and the tests read
    the same few modules for each of their names. Treat the result as
    read-only."""
    out: dict[str, list[ast.stmt]] = {}
    for node in _parse(root / f"{module}.codon").body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            out.setdefault(node.name, []).append(node)
            if isinstance(node, ast.ClassDef):
                _ORIGIN[id(node)] = (module, root)
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for target in targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(node)
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            # A module beside it (`.coordinate`) or in a package beside it
            # (`.autoware_bridge.grpc_server`).
            source = (Path(module).parent / node.module.replace(".", "/")).as_posix()
            for alias in node.names:
                out.setdefault(alias.asname or alias.name, []).extend(
                    _definitions(source, root).get(alias.name, [])
                )
    return out


def _is_required_marker(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_Required"
    )


def _model_params(fn: ast.FunctionDef, *, drop_first: bool) -> list[Param]:
    args = fn.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    out: list[Param] = [
        ("pos", a.arg, d is not None and not _is_required_marker(d))
        for a, d in zip(positional, defaults)
    ]
    if drop_first:
        out = out[1:]
    if args.vararg is not None and args.vararg.arg != "_kw":
        out.append(("*args", args.vararg.arg, False))
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        out.append(("kw", a.arg, d is not None and not _is_required_marker(d)))
    if args.kwarg is not None:
        out.append(("**kwargs", args.kwarg.arg, False))
    return out


def _model_fields(cls: ast.ClassDef) -> list[Param]:
    """The generated __init__ of a model class without one: its fields."""
    out: list[Param] = []
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if "ClassVar" in ast.unparse(stmt.annotation):
                continue
            out.append(("pos", stmt.target.id, stmt.value is not None))
    return out


_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: "pos",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "pos",
    inspect.Parameter.KEYWORD_ONLY: "kw",
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
}


def _python_params(obj: Any, *, drop_first: bool) -> list[Param]:
    params = list(inspect.signature(obj).parameters.values())
    if drop_first:
        params = params[1:]
    return [
        (_KINDS[p.kind], p.name, p.default is not inspect.Parameter.empty)
        for p in params
    ]


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    out: dict[str, ast.FunctionDef] = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name == "__init__":
                # The constructor is the last __init__: one refusing what
                # Codon would build otherwise comes ahead of it (_inits).
                out[stmt.name] = stmt
            else:
                out.setdefault(stmt.name, stmt)  # a property's getter, not its setter
    return out


def _inits(cls: ast.ClassDef) -> list[ast.FunctionDef]:
    return [
        stmt
        for stmt in cls.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__"
    ]


def _is_compile_error(fn: ast.FunctionDef) -> bool:
    """*fn*'s body is a compile_error() call and nothing else."""
    body = fn.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]  # its docstring
    return (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Call)
        and ast.unparse(body[0].value.func) == "compile_error"
    )


def _refusing_init(cls: ast.ClassDef, arity: int) -> ast.FunctionDef | None:
    """The __init__ of *arity* positional parameters (self aside) that is a
    compile_error and comes ahead of the constructor, or None. Codon tries a
    class's overloads from the last declared, so the constructor wins every
    call it takes, and this one only refuses what Codon would build else."""
    inits = _inits(cls)
    for init in inits[:-1]:
        args = init.args
        if (
            _is_compile_error(init)
            and len(args.posonlyargs) + len(args.args) == arity + 1
            and not args.defaults
            and args.vararg is None
            and not args.kwonlyargs
            and args.kwarg is None
        ):
            return init
    return None


def _decorators(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {ast.unparse(d) for d in fn.decorator_list}


def _class_vars(cls: ast.ClassDef) -> set[str]:
    return {
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and "ClassVar" in ast.unparse(stmt.annotation)
    }


_EXPORTS = _exports()


def test_the_model_exports_only_names_the_package_exports() -> None:
    unknown = sorted(set(_EXPORTS) - set(acs.__all__) - set(acs._LAZY_IMPORTS))
    assert not unknown, f"the model exports names the package does not: {unknown}"


@pytest.mark.parametrize("name", sorted(_EXPORTS))
def test_an_exported_name_is_declared_as_in_python(name: str) -> None:
    _assert_declared_as_in_python(name, _EXPORTS[name], getattr(acs, name))


def test_every_exported_enum_is_named_by_the_model_as_iterable() -> None:
    # The checker rewrites `for m in E` and `E["A"]` for the names the model
    # declares `_acs_members()` for (#437): exactly the exported enums, none
    # of them left out, none that is no enum.
    from autoware_carla_scenario.typecheck.transform import model_enum_names

    declared = {
        **{name: getattr(acs, name) for name in _EXPORTS},
        **{name: python for name, (_, python) in _BOUNDARY.items()},
        "ActionState": action_state.ActionState,
    }
    exported = {
        name
        for name, python in declared.items()
        if inspect.isclass(python) and issubclass(python, enum.Enum)
    }
    assert exported
    assert model_enum_names() == exported


#: Names the model declares for the modules the library check compiles
#: (typecheck/library.py), or that a scenario imports from a submodule,
#: though the package does not export them: name -> (model module, the
#: Python object).
_BOUNDARY: dict[str, tuple[str, Any]] = {
    "CarlaDriverEntity": ("entity", carla_driver_entity.CarlaDriverEntity),
    # What the checked actions.communication_loss uses of entity.hils_entity.
    "HilsEgoEntity": ("entity", hils_entity.HilsEgoEntity),
    "find_hils_entity": ("entity", hils_entity.find_hils_entity),
    "CoverItem": ("coverage", coverage_items.CoverItem),
    "CrossItem": ("coverage", coverage_items.CrossItem),
    "CoverageCollector": ("coverage", coverage_collector.CoverageCollector),
    # What the checked coverage package re-exports.
    "COVERAGE_SCHEMA": ("coverage", coverage_collector.COVERAGE_SCHEMA),
    "CoverGroup": ("coverage", coverage_items.CoverGroup),
    "CoverageReport": ("coverage", coverage_report.CoverageReport),
    "merge_coverage": ("coverage", coverage_report.merge_coverage),
    "MapManager": ("coordinate", coordinate_map_manager.MapManager),
    # What map_manager and sweeper.map_loader import of coordinate.projection,
    # which the library check now compiles from its source too.
    **{
        name: ("coordinate", getattr(coordinate_projection, name))
        for name in ("_parse_geo_reference", "map_origin", "resolve_projector")
    },
    "TrafficContext": ("traffic", traffic_context.TrafficContext),
    # The type of a vehicle entity's simulation_fidelity, which the traffic
    # package exports (and the package does not).
    "SimulationFidelity": ("traffic", traffic_base.SimulationFidelity),
    **{
        name: ("traffic", getattr(traffic_registry, name))
        for name in (
            "TRAFFIC_BACKEND_ENTRY_POINT_GROUP",
            "available_backends",
            "build_backend",
            "get_backend_factory",
            "load_traffic_backend_plugins",
            "register_backend",
            "unregister_backend",
        )
    },
    "ResolvedTrajectory": ("trajectory", trajectory_model.ResolvedTrajectory),
    # A Union alias: its members compared by name, in order.
    "TrajectoryPosition": ("trajectory", trajectory_model.TrajectoryPosition),
    # What a scenario imports of coordinate beside the package's exports.
    **{
        name: ("coordinate", getattr(coordinate, name))
        for name in (
            "find_traffic_lights_for_lanelet2_id",
            "get_signal_ids_for_controller",
            "get_stop_line_linestrings",
            "get_stop_line_linestrings_with_following",
            "lanelet_length",
            "on_lanelet",
            "project_onto_lanelet",
            "project_onto_road",
            "road_traffic_rule",
            "route_road_ids",
        )
    },
    # What a scenario imports of conditions beside the package's exports.
    **{
        name: ("conditions", getattr(conditions, name))
        for name in (
            "ActionStateCondition",
            "ConditionStatus",
            "LaneChangeSettledCondition",
            "beside_lane_id",
            "compare",
        )
    },
    # Defined by entity_role, as in Python; conditions re-exports it.
    "find_actor_in_list": ("entity_role", entity_role.find_actor_in_list),
    "find_actor_in_world": ("entity_role", entity_role.find_actor_in_world),
    # What a scenario imports of kinematics beside the package's exports.
    "normalize_angle_deg": ("kinematics", kinematics.normalize_angle_deg),
    # A Union alias, for a scenario's own annotations: its members by name.
    "AnyPose": ("coordinate", coordinate.AnyPose),
    # A frozen dataclass a scenario reads off follow_with_spectator().
    # Every constant of constants.codon (EGO_ROLE_NAME is exported), each
    # compared by value.
    **{
        name: ("constants", getattr(constants, name))
        for name in (
            "BUMPER_PROBE_ROLE_NAME",
            "DEFAULT_TM_PORT",
            "EGO_ROLES",
            "FIXED_DELTA_SECONDS",
            "LANE_CHANGE_CENTER_TOLERANCE_M",
            "LANE_CHANGE_HEADING_TOLERANCE_DEG",
            "US_PER_S",
        )
    },
    # Everything route.codon declares: the running route, its frame, the
    # anchors, and the values (compared by value) the route poses check.
    "RouteMatch": ("route", route_model.RouteMatch),
    "RouteSegmentMatch": ("route", route_model.RouteSegmentMatch),
    "parse_anchor": ("route", route_model.parse_anchor),
    "scenario_route": ("route", route_active.scenario_route),
    "scenario_route_frame": ("route", route_active.scenario_route_frame),
    **{
        name: ("route", getattr(route_model, name))
        for name in ("APPROACHES", "PATH_APPROACHES", "CROSSWALK_LEGS", "SIDES")
    },
    "RouteFrame": ("route", route_frame.RouteFrame),
    "TrajectorySample": ("trajectory", trajectory_model.TrajectorySample),
    # What BaseAction.take_waypoint_arrivals() returns.
    "WaypointArrival": ("actions", waypoint_arrival.WaypointArrival),
    "to_map_frame": ("coordinate", coordinate_transform.to_map_frame),
    "reset_probes": ("odd", odd_probes.reset_probes),
    "resolve_odd": ("odd", odd_registry.resolve_odd),
    # What scenario_runner's _warn_if_route_leaves_odd uses of odd.route.
    **{
        name: ("odd", getattr(odd_route, name))
        for name in ("RouteCoverage", "RouteLanelet", "plan_route_coverage")
    },
}


@pytest.mark.parametrize("name", sorted(_BOUNDARY))
def test_a_boundary_name_is_declared_as_in_python(name: str) -> None:
    module, python = _BOUNDARY[name]
    _assert_declared_as_in_python(name, module, python)


def _assert_declared_as_in_python(
    name: str, module: str, python: Any, root: Path = _MODEL
) -> None:
    definitions = _definitions(module, root).get(name)
    assert definitions, f"{name} is not defined in {module}.codon"
    node = definitions[0]

    if isinstance(node, ast.FunctionDef):
        assert inspect.isfunction(python), f"{name} is a function in the model only"
        expected = _python_params(python, drop_first=False)
        for overload in definitions:
            assert isinstance(overload, ast.FunctionDef)
            assert _model_params(overload, drop_first=False) == expected, name
        return

    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        _assert_constant_as_in_python(name, node, python)
        return

    if not isinstance(node, ast.ClassDef):
        return

    if typing.get_origin(python) is typing.Union:
        # A Union alias (SpawnLocation): the base class of its members here.
        for member in typing.get_args(python):
            classes = _definitions(_EXPORTS[member.__name__]).get(member.__name__, [])
            assert classes and isinstance(classes[0], ast.ClassDef)
            assert name in {
                ast.unparse(b) for b in classes[0].bases
            }, f"{member.__name__} does not derive from {name} in the model"
        return

    assert inspect.isclass(python), f"{name} is a class in the model only"
    methods = _methods(node)
    if issubclass(python, enum.Enum):
        assert _class_vars(node) == {
            m.name for m in python
        }, f"{name}: the model's members differ from the enum's"
        _assert_enum_as_in_python(name, node, python)
    else:
        if "__init__" in methods:
            model_init = _model_params(methods["__init__"], drop_first=True)
        else:
            model_init = _model_fields(node)
        omitted = _OMITTED_PARAMETERS.get(name, set())
        python_init = [
            p for p in _python_params(python, drop_first=False) if p[1] not in omitted
        ]
        assert model_init == python_init, f"{name}(): the model's parameters differ"

    if dataclasses.is_dataclass(python) and "__init__" not in methods:
        python_fields = {f.name for f in dataclasses.fields(python)}
        declared = {p[1] for p in _model_fields(node)}
        assert declared <= python_fields, f"{name}: fields the dataclass does not have"

    for attr in _class_vars(node):
        assert hasattr(python, attr), f"{name}.{attr} is in the model only"

    fields = (
        {f.name for f in dataclasses.fields(python)}
        if dataclasses.is_dataclass(python)
        else set()
    )
    for method_name, method in methods.items():
        if method_name.startswith("__") or method_name.startswith("_acs_"):
            continue
        if method_name in fields and "property" in _decorators(method):
            # A dataclass field read as a property (RouteMatch.lanelet_ids,
            # #298): a field would change the constructor Codon generates.
            continue
        try:
            python_attr = inspect.getattr_static(python, method_name)
        except AttributeError:
            # An attribute Python sets on the instance (#307), read as a
            # property: a field would change the constructor Codon generates.
            assert method_name in _python_members(
                python
            ), f"{name}.{method_name} is in the model only"
            assert "property" in _decorators(
                method
            ), f"{name}.{method_name}: an instance attribute, read as a property"
            continue
        if "property" in _decorators(method):
            # An enum's name and value are Enum's: enum.property (3.11+) or
            # a DynamicClassAttribute (3.10), each a DynamicClassAttribute.
            assert isinstance(
                python_attr, (property, types.DynamicClassAttribute)
            ), f"{name}.{method_name} is a property"
            continue
        if "staticmethod" in _decorators(method):
            # A classmethod in Python (EntityRole.ego): bound, it takes the same.
            assert isinstance(
                python_attr, (staticmethod, classmethod)
            ), f"{name}.{method_name} is not a static or class method in Python"
            assert _model_params(method, drop_first=False) == _python_params(
                getattr(python, method_name), drop_first=False
            ), f"{name}.{method_name}(): the model's parameters differ"
            continue
        assert _model_params(method, drop_first=True) == _python_params(
            getattr(python, method_name), drop_first=True
        ), f"{name}.{method_name}(): the model's parameters differ"


def _enum_values(node: ast.ClassDef) -> dict[str, Any]:
    """Member -> the value a model enum's member is built with
    (`LESS_THAN: ClassVar[ComparisonRule] = ComparisonRule(2)`)."""
    out: dict[str, Any] = {}
    for stmt in node.body:
        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and "ClassVar" in ast.unparse(stmt.annotation)
        ):
            value = stmt.value
            assert (
                isinstance(value, ast.Call)
                and ast.unparse(value.func) == node.name
                and len(value.args) == 1
                and not value.keywords
            ), f"{node.name}.{stmt.target.id}: not built from its value alone"
            out[stmt.target.id] = ast.literal_eval(value.args[0])
    return out


def _enum_names(node: ast.ClassDef) -> list[tuple[str, str]]:
    """(member, the str returned for it) of a model enum's `name` property:
    a chain of `if self == Enum.MEMBER: return "MEMBER"`, which raises for a
    value that is no member's (#307)."""
    getter = _methods(node).get("name")
    assert getter is not None, f"{node.name} has no name property"
    assert "property" in _decorators(getter), f"{node.name}.name is a property"
    out: list[tuple[str, str]] = []
    *chain, last = getter.body
    for stmt in chain:
        assert isinstance(stmt, ast.If) and not stmt.orelse, ast.unparse(stmt)
        test, body = stmt.test, stmt.body
        assert (
            isinstance(test, ast.Compare)
            and ast.unparse(test.left) == "self"
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Attribute)
            and ast.unparse(test.comparators[0].value) == node.name
        ), f"{node.name}.name: {ast.unparse(test)}"
        assert (
            len(body) == 1
            and isinstance(body[0], ast.Return)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ), f"{node.name}.name: {ast.unparse(stmt)}"
        out.append((test.comparators[0].attr, body[0].value.value))
    assert isinstance(last, ast.Raise), f"{node.name}.name ends with a raise"
    return out


def _enum_iteration(node: ast.ClassDef) -> tuple[list[str], list[str]]:
    """(the members `_acs_members()` returns, the names `_acs_by_name()` looks
    up) of a model enum (#437): what `for m in E`, `list(E)` and `E["A"]` read
    once the checker rewrites them. Both are static methods of a fixed shape:
    `return [E.A, E.B]` and a chain of `if name == "A": return E.A`, ending in
    `raise KeyError(name)`."""
    methods = _methods(node)
    listing, lookup = methods.get("_acs_members"), methods.get("_acs_by_name")
    assert listing is not None, f"{node.name} has no _acs_members()"
    assert lookup is not None, f"{node.name} has no _acs_by_name()"
    for method in (listing, lookup):
        assert "staticmethod" in _decorators(method), f"{node.name}.{method.name}"
    assert listing.returns is not None and lookup.returns is not None
    assert ast.unparse(listing.returns) == f"List[{node.name}]"
    assert ast.unparse(lookup.returns) == node.name
    assert [a.arg for a in lookup.args.args] == ["name"]
    (ret,) = listing.body
    assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.List)
    members = []
    for element in ret.value.elts:
        assert isinstance(element, ast.Attribute)
        assert ast.unparse(element.value) == node.name, ast.unparse(element)
        members.append(element.attr)
    names: list[str] = []
    *chain, last = lookup.body
    for stmt in chain:
        assert isinstance(stmt, ast.If) and not stmt.orelse, ast.unparse(stmt)
        test = stmt.test
        assert (
            isinstance(test, ast.Compare)
            and ast.unparse(test.left) == "name"
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, str)
        ), ast.unparse(test)
        member = test.comparators[0].value
        (ret,) = stmt.body
        assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.Attribute)
        assert ast.unparse(ret.value) == f"{node.name}.{member}"
        names.append(member)
    assert isinstance(last, ast.Raise) and ast.unparse(last) == "raise KeyError(name)"
    return members, names


def _assert_enum_as_in_python(name: str, node: ast.ClassDef, python: Any) -> None:
    """A model enum (a `@tuple` of its value, #307) builds each member from
    Python's value, of Python's type, and its `name` property returns each
    member's Python name: Codon has no Enum to give a framework enum one."""
    values = _enum_values(node)
    for member in python:
        assert type(values[member.name]) is type(
            member.value
        ), f"{name}.{member.name}: the type of its value"
        assert (
            values[member.name] == member.value
        ), f"{name}.{member.name}: the model's value differs"
    assert _enum_names(node) == [
        (member.name, member.name) for member in python
    ], f"{name}.name: not each member's name, in Python's order"
    members, by_name = _enum_iteration(node)
    assert members == [
        member.name for member in python
    ], f"{name}._acs_members(): not Python's iteration order (aliases skipped)"
    assert by_name == list(
        python.__members__
    ), f"{name}._acs_by_name(): not each member, aliases included, in Python's order"


def _union_member_name(member: Any) -> str:
    # A member Python names as a string (a forward reference) by that string.
    if isinstance(member, typing.ForwardRef):
        return member.__forward_arg__
    return str(member.__name__)


def _assert_constant_as_in_python(
    name: str, node: ast.Assign | ast.AnnAssign, python: Any
) -> None:
    """A model constant holds Python's value: a literal (str, int, float, a
    tuple of them) by its value and type, a Union alias by its members' names,
    in order, a Literal alias as the type of its values. The few that are none
    of these are compared on their own (_CONSTANTS_COMPARED_ON_THEIR_OWN)."""
    value = node.value
    assert value is not None, f"{name} has no value in the model"
    if name in _CONSTANTS_COMPARED_ON_THEIR_OWN:
        return
    if typing.get_origin(python) is typing.Literal:
        # A Literal alias (ui.models.Status): the type of its values.
        types = {type(v).__name__ for v in typing.get_args(python)}
        assert [ast.unparse(value)] == sorted(types), f"{name}: a Literal alias"
        return
    if typing.get_origin(python) is typing.Union:
        assert isinstance(value, ast.Subscript), f"{name} is a Union in Python"
        assert ast.unparse(value.value) == "Union", name
        members = value.slice.elts if isinstance(value.slice, ast.Tuple) else []
        assert [ast.unparse(m) for m in members] == [
            _union_member_name(m) for m in typing.get_args(python)
        ], f"{name}: the model's Union members differ"
        return
    literal = ast.literal_eval(value)  # a ValueError: no literal
    assert _literal_types(literal) == _literal_types(python), f"{name}: its type"
    assert literal == python, f"{name}: the model's value differs"


def _literal_types(value: Any) -> Any:
    """*value*'s type, a tuple's being its items' (bool is no int, int no
    float)."""
    if isinstance(value, tuple):
        return tuple(_literal_types(v) for v in value)
    return type(value)


#: Model modules of one framework module each, which the library check
#: compiles a checked module against (typecheck/library.py) or a scenario
#: imports: model module -> the Python module it models and the names it
#: declares, every one compared with the Python object below.
_MODULE_MODELS: dict[str, tuple[Any, tuple[str, ...]]] = {
    "_bridge_pose": (bridge_base, ("BridgePose", "Quaternion", "Vector3")),
    # What ActionStateCondition takes and BaseAction.state returns; the model
    # of actions imports it from here, as Python's actions.base does.
    "action_state": (action_state, ("ActionState",)),
    "carla_install": (carla_install, ("installed_executable",)),
    # What an ego entity a scenario defines returns from debug_markers().
    "debug_markers": (
        debug_markers,
        ("DRIVEN_PATH_COLOR", "DebugMarkers", "DebugPolyline", "TRAJECTORY_COLOR"),
    ),
    "trajectory_recorder": (
        trajectory_recorder,
        ("EGO_ROLES", "RECORDED_KINDS", "SCHEMA_VERSION", "TrajectoryRecorder"),
    ),
    # What the checked scenario_runner and scenario_queue hold of it, and the
    # log suffix the checked t4_scene.grid reads.
    "rerun": (
        rerun_module,
        ("RerunConverter", "TRAJECTORY_SUFFIX", "recording_path"),
    ),
    # What the checked scenario_runner and scenario_queue hold of it.
    "t4": (t4_module, ("T4Recorder", "T4Recording")),
    # What the checked scenario_runner and scenario_queue hold of it.
    "t4_scene": (t4_scene_module, ("T4SceneExporter",)),
    # What the checked visibility and actions.environment read the weather
    # rules through.
    "weather_rules": (
        weather_rules,
        ("vehicle_range", "vehicle_rules_in_use", "weather_range"),
    ),
    # What scenario_runner's _record_run reads of the unchecked environment.
    "environment": (
        environment,
        ("ENVIRONMENT_SUFFIX", "LOCK_SUFFIX", "RunEnvironment", "capture"),
    ),
    "autoware_bridge/grpc_server": (bridge_grpc_server, ("GrpcAutowareBridgeServer",)),
    "autoware_stack/launcher": (
        stack_launcher,
        ("AutowareEpisode", "AutowareLauncher"),
    ),
    # What the checked autoware_stack.docker and the package take of the
    # unchecked devcontainer (yaml, json).
    "autoware_stack/devcontainer": (
        stack_devcontainer,
        (
            "DEFAULT_WORKSPACE_MOUNT",
            "DevContainer",
            "DevContainerError",
            "read_devcontainer",
        ),
    ),
    "driver/egodriver_client": (egodriver_client, ("EgoDriverGrpcClient",)),
    # MapSource is shared with the checked maps.source (REPLACED_MODELS).
    "_map_source": (maps_source, ("MapSource",)),
    # What a scenario sees of the map cache; the library check compiles the
    # checked maps.cache from its source.
    "maps/cache": (
        maps_cache,
        (
            "AUTOWARE_DATA_DIR_ENV",
            "CACHE_ROOT_ENV",
            "CachedRepo",
            "DERIVED_SUBDIR",
            "GitMapCache",
            "MapCacheError",
            "is_lfs_pointer",
            "map_root",
            "TIP_SUFFIX",
        ),
    ),
    # What the checked coordinate.map_manager (MapManager.initialize) uses of
    # the unchecked z_offset_cache (json, contextlib).
    "maps/z_offset_cache": (maps_z_offset_cache, ("cached_z_offset", "z_offset_key")),
    # What the checked coordinate.map_manager (MapManager.initialize and
    # road_network) uses of the unchecked road_network_cache (pickle, lxml).
    "maps/road_network_cache": (
        maps_road_network_cache,
        ("load_roads", "open_road_network"),
    ),
    # A pydantic model: its fields by keyword only, as Python's signature.
    "ui/models": (ui_models, ("RunProgress", "Status")),
    # The base and config of ros2/cyclone.codon's CycloneRos2Transport, shared
    # with the checked ros2.transport and ros2.config (REPLACED_MODELS).
    "_ros2": (ros2, ("Ros2EgoConfig", "Ros2EgoTransport")),
    # What a scenario imports of the checked metrics: every name of its
    # __all__, the values compared below.
    "metrics": (metrics, tuple(metrics.__all__)),
    # What a scenario imports of the checked failure: every name of its
    # __all__ (FailureCategory, to register a fail condition with).
    "failure": (failure, tuple(failure.__all__)),
    # What a scenario or a Python ODD imports of the checked measures: every
    # name of its __all__, the values compared below.
    "measures": (measures, tuple(measures.__all__)),
    # The model of one module of the checked utils package, which the scenario
    # check resolves (typecheck/check.py).
    "utils/traffic_light": (
        utils_traffic_light,
        (
            "get_signal_ids_for_controller",
            "lanelet2_traffic_light_id_to_opendrive_controller_id",
            "traffic_light_state_name",
        ),
    ),
}

#: Classes the model declares only so a checked package can re-export them:
#: their constructors are a compile error, so a scenario cannot build one.
_UNBUILDABLE: dict[str, str] = {
    "AutowareLauncher": "autoware_stack/launcher",
    "CarlaDriverEntity": "entity",
    "CoverageReport": "coverage",
    "EgoDriverGrpcClient": "driver/egodriver_client",
    "GrpcAutowareBridgeServer": "autoware_bridge/grpc_server",
    "HilsEgoEntity": "entity",
    # Stand-ins of checked classes the library check replaces (_ros2.codon):
    # a scenario has no ROS 2 transport to build.
    "Ros2EgoConfig": "_ros2",
    "Ros2EgoTransport": "_ros2",
}


def test_a_module_model_declares_only_the_compared_names() -> None:
    # A name the module lacks would pass a scenario's check and fail its run.
    # A model-only helper (`_acs_` prefix) is no name of the module, and must
    # not stand in for one.
    for module, (python, names) in _MODULE_MODELS.items():
        definitions = _definitions(module)
        declared = [n for n in definitions if not n.startswith("_acs_")]
        assert sorted(declared) == sorted(names), module
        helpers = [n for n in definitions if n.startswith("_acs_")]
        assert not [n for n in helpers if hasattr(python, n)], module


_MODULE_MODEL_NAMES = [
    (module, name)
    for module, (_python, names) in _MODULE_MODELS.items()
    for name in names
    if name != "AutowareLauncher"  # a Protocol: compared on its own below
]


@pytest.mark.parametrize(("module", "name"), _MODULE_MODEL_NAMES)
def test_a_module_model_is_declared_as_in_python(module: str, name: str) -> None:
    python_module, _names = _MODULE_MODELS[module]
    # A cached function (environment.capture(), `@lru_cache`) is compared as
    # the function it wraps: the cache takes the same parameters.
    python = inspect.unwrap(getattr(python_module, name))
    _assert_declared_as_in_python(name, module, python)


def test_the_devcontainer_model_holds_python_s_values() -> None:
    # A constant is compared by its value, and the frozen dataclass is a
    # @tuple (an assignment to a field fails the check as it fails the run).
    definitions = _definitions("autoware_stack/devcontainer")
    constant = definitions["DEFAULT_WORKSPACE_MOUNT"][0]
    assert isinstance(constant, ast.Assign)
    assert (
        ast.literal_eval(constant.value) == stack_devcontainer.DEFAULT_WORKSPACE_MOUNT
    )
    node = definitions["DevContainer"][0]
    assert isinstance(node, ast.ClassDef)
    assert [ast.unparse(d) for d in node.decorator_list] == ["tuple"]
    params = stack_devcontainer.DevContainer.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen
    assert issubclass(stack_devcontainer.DevContainerError, ValueError)


def test_the_coverage_model_holds_python_s_values() -> None:
    # The constant is compared by its value, and the enum stand-in (a @tuple
    # of ClassVars) by the value of each member.
    definitions = _definitions("coverage")
    constant = definitions["COVERAGE_SCHEMA"][0]
    assert isinstance(constant, ast.AnnAssign) and constant.value is not None
    assert ast.literal_eval(constant.value) == coverage_collector.COVERAGE_SCHEMA
    node = definitions["CoverGroup"][0]
    assert isinstance(node, ast.ClassDef)
    members = {
        stmt.target.id: ast.literal_eval(stmt.value.args[0])
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.value, ast.Call)
    }
    assert members == {m.name: m.value for m in coverage_items.CoverGroup}


def _literal(node: ast.expr, constants: dict[str, Any]) -> Any:
    """*node*'s value, a name being one of *constants*."""
    if isinstance(node, ast.Name):
        return constants[node.id]
    return ast.literal_eval(node)


def _model_constants(module: str, python: Any) -> dict[str, Any]:
    """The constants of *python*'s ``__all__`` (str, int, float), each compared
    with its model's value: name -> value."""
    definitions = _definitions(module)
    constants: dict[str, Any] = {}
    for name in python.__all__:
        value = getattr(python, name)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            node = definitions[name][0]
            assert isinstance(node, ast.AnnAssign) and node.value is not None
            assert ast.unparse(node.annotation) == type(value).__name__, name
            constants[name] = ast.literal_eval(node.value)
            assert constants[name] == value, name
    return constants


def _assert_frozen_fields(module: str, python: type) -> dict[str, ast.AnnAssign]:
    """Model class *python* is a @tuple of the frozen dataclass's fields, in
    order, of the same types and defaults (but a `read`, a callable): the
    model's fields, by name."""
    assert python.__dataclass_params__.frozen  # type: ignore[attr-defined]
    node = _definitions(module)[python.__name__][0]
    assert isinstance(node, ast.ClassDef)
    assert [ast.unparse(d) for d in node.decorator_list] == ["tuple"]
    model_fields = {
        stmt.target.id: stmt
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    assert list(model_fields) == [f.name for f in dataclasses.fields(python)]
    for field in dataclasses.fields(python):
        stmt = model_fields[field.name]
        if field.name == "read":
            assert (
                ast.unparse(stmt.annotation)
                == "Callable[[carla.World], Optional[float]]"
            )
            continue
        assert ast.unparse(stmt.annotation) == field.type, field.name
        if field.default is not dataclasses.MISSING:
            assert stmt.value is not None, field.name
            assert ast.literal_eval(stmt.value) == field.default, field.name
    new = _methods(node).get("__new__")
    # A `read` field needs the __new__ that takes any callable: Codon 0.19
    # crashes on a `Callable` field given a function of another type.
    assert new is not None or "read" not in model_fields, python.__name__
    if new is not None:
        # A __new__ that takes any callable as `read` takes Python's
        # parameters: the fields, in order, with their defaults.
        assert _model_params(new, drop_first=False) == _python_params(
            python, drop_first=False
        )
        for arg, default in zip(reversed(new.args.args), reversed(new.args.defaults)):
            field_default = model_fields[arg.arg].value
            assert field_default is not None, arg.arg
            assert ast.literal_eval(default) == ast.literal_eval(field_default)
    return model_fields


def _assert_built_ins(
    module: str, table: str, python: dict[str, Any], constants: dict[str, Any]
) -> None:
    """The model's *table* holds *python*'s entries, by key and in order, each
    built with every field's value but its `read`: Python's is a private
    function of the module, the model's a stub (_acs_unread), and nothing in
    the model runs."""
    node = _definitions(module)[table][0]
    entry = type(next(iter(python.values())))
    assert isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Dict)
    assert ast.unparse(node.annotation) == f"Dict[str, {entry.__name__}]"
    fields = [f.name for f in dataclasses.fields(entry)]
    modelled: dict[str, dict[str, Any]] = {}
    for key_node, call in zip(node.value.keys, node.value.values):
        assert key_node is not None and isinstance(call, ast.Call)
        assert ast.unparse(call.func) == entry.__name__
        arguments = dict(zip(fields, call.args))
        arguments.update({k.arg: k.value for k in call.keywords if k.arg})
        assert ast.unparse(arguments.pop("read")) == "_acs_unread"
        values = {f: _literal(a, constants) for f, a in arguments.items()}
        modelled[_literal(key_node, constants)] = values
    assert list(modelled) == list(python)
    for key, built_in in python.items():
        assert built_in.key == key
        defaults = {
            f.name: f.default
            for f in dataclasses.fields(built_in)
            if f.default is not dataclasses.MISSING
        }
        expected = {f: getattr(built_in, f) for f in fields if f != "read"}
        assert {**defaults, **modelled[key]} == expected, key


def test_the_metrics_model_holds_python_s_values() -> None:
    # The keys are compared by value, each built-in metric by every field but
    # `read`, and Metric's fields by type and default.
    constants = _model_constants("metrics", metrics)
    _assert_built_ins(
        "metrics", "BUILT_IN_METRICS", metrics.BUILT_IN_METRICS, constants
    )
    _assert_frozen_fields("metrics", metrics.Metric)


def test_the_measures_model_holds_python_s_values() -> None:
    # Scenarios and Python ODDs import the keys and the other constants; each
    # built-in measure is compared by every field but `read`, and Measure's
    # and Bumpers' fields by type and default.
    constants = _model_constants("measures", measures)
    assert set(constants) >= {
        "VEHICLE_AHEAD_GAP_M",
        "VEHICLE_AHEAD_RELATIVE_SPEED_KPH",
        "CROSSING_PEDESTRIAN_GAP_M",
        "CROSSING_PEDESTRIAN_SPEED_MS",
        "VEHICLE_AHEAD_VERSION",
    }
    _assert_built_ins(
        "measures", "BUILT_IN_MEASURES", measures.BUILT_IN_MEASURES, constants
    )
    _assert_frozen_fields("measures", measures.Measure)
    _assert_frozen_fields("measures", measures.Bumpers)


#: Names a model declares as Python cannot, compared by a test of their own:
#: name -> (model module, why).
_DIFFERING: dict[str, tuple[str, str]] = {
    "TrafficBackendFactory": (
        "traffic",
        "a Callable alias in Python, of a mapping of any values (Hydra's "
        "options) to the checked TrafficBackend: Codon has no type for the "
        "mapping, and the model cannot import the checked traffic.base, so "
        "the model declares an empty class, only so the checked traffic "
        "package's re-export resolves; nothing annotates with it "
        "(register_backend's factory is untyped)",
    ),
    "OddSample": (
        "odd",
        "openodd's sample (a Rust extension), which the checked "
        "scenario_runner only hands on: the model declares an opaque class "
        "whose constructor is a compile error, as openodd's sampler draws it",
    ),
    "OddCase": (
        "odd",
        "a `tuple[OddSample, ParameterSpace]` alias in Python: a @tuple class "
        "of the two fields here, Codon having no tuple alias a class can name",
    ),
}


def _own_names(module: str) -> set[str]:
    """The public names *module*'s model defines itself (not imports). Any
    other top-level statement (an `if`, a `try`, an unpacking) fails, so that
    no name it would define goes uncompared."""
    names: set[str] = set()
    for node in _parse(_MODEL / f"{module}.codon").body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            assert all(isinstance(t, ast.Name) for t in targets), ast.unparse(node)
            names.update(t.id for t in targets if isinstance(t, ast.Name))
        else:
            assert isinstance(
                node, (ast.Import, ast.ImportFrom, ast.Expr, ast.Pass)
            ), f"{module}.codon: {ast.unparse(node)}"
    return {n for n in names if not n.startswith("_")}


@pytest.mark.parametrize(
    "module", ["conditions", "constants", "route", "traffic", "trajectory"]
)
def test_a_model_declares_only_the_compared_names(module: str) -> None:
    # Every public name the model defines is compared: an export, a boundary
    # name or one of _DIFFERING. A name Python lacks would pass a scenario's
    # check and fail its run.
    compared = {
        *(n for n, m in _EXPORTS.items() if m == module),
        *(n for n, (m, _python) in _BOUNDARY.items() if m == module),
        *(n for n, (m, _why) in _DIFFERING.items() if m == module),
    }
    assert sorted(_own_names(module)) == sorted(compared), module


def test_the_constants_model_holds_python_s_values() -> None:
    # Every constant of constants.py is in the model, with its value: the
    # literals through _BOUNDARY, EGO_ROLE_NAME (built from EntityRole) here.
    python = {
        name
        for name, value in vars(constants).items()
        if name.isupper() and not inspect.ismodule(value)
    }
    assert _own_names("constants") == python
    node = _definitions("constants")["EGO_ROLE_NAME"][0]
    assert isinstance(node, ast.AnnAssign) and node.value is not None
    assert ast.unparse(node.annotation) == "EntityRole"
    assert ast.unparse(node.value) == "EntityRole.ego()"
    assert constants.EGO_ROLE_NAME == acs.EntityRole.ego()


def test_the_urban_construction_model_holds_python_s_values() -> None:
    # The preset is a ConstructionSettings call: the model's keywords are the
    # fields Python's differs from the defaults in, with the same values.
    node = _definitions("actions")["URBAN_CONSTRUCTION"][0]
    assert isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call)
    assert ast.unparse(node.annotation) == "ConstructionSettings"
    assert ast.unparse(node.value.func) == "ConstructionSettings"
    assert not node.value.args
    model = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
    default = acs.ConstructionSettings()
    python = {
        field.name: getattr(acs.URBAN_CONSTRUCTION, field.name)
        for field in dataclasses.fields(acs.ConstructionSettings)
        if getattr(acs.URBAN_CONSTRUCTION, field.name) != getattr(default, field.name)
    }
    assert model == python


#: Every class the tests above compare with Python: name -> (model module,
#: the Python class).
_COMPARED_CLASSES: dict[str, tuple[str, Any]] = {
    name: (module, python)
    for name, module, python in [
        *((n, m, getattr(acs, n)) for n, m in _EXPORTS.items()),
        *((n, m, p) for n, (m, p) in _BOUNDARY.items()),
        *((n, m, getattr(_MODULE_MODELS[m][0], n)) for m, n in _MODULE_MODEL_NAMES),
    ]
    if inspect.isclass(python) and not issubclass(python, enum.Enum)
}


#: What the model's exceptions derive from (`Static[Exception]`, ...): Codon
#: 0.19's own exception classes, as its standard library declares them
#: (stdlib/internal/types/error.codon), reduced to their fields and
#: constructors. test_the_codon_exceptions_are_declared_as_in_codon compares
#: them with the standard library where Codon is installed.
_CODON_EXCEPTIONS_SOURCE = """
class BaseException:
    typename: str
    message: str
    func: str
    file: str
    line: int
    col: int
    python_type: cobj
    cause: Optional[BaseException]

    def __init__(self, typename: str, message: str = ""): ...

class Exception(Static[BaseException]):
    def __init__(self, typename: str, msg: str = ""): ...

class ValueError(Static[Exception]):
    def __init__(self, message: str = ""): ...

class RuntimeError(Static[Exception]):
    def __init__(self, typename: str, message: str = ""): ...
    def __init__(self, message: str = ""): ...
"""
_CODON_EXCEPTIONS: dict[str, ast.ClassDef] = {
    node.name: node
    for node in ast.parse(_CODON_EXCEPTIONS_SOURCE).body
    if isinstance(node, ast.ClassDef)
}


def _model_class(name: str) -> ast.ClassDef:
    module, _python = _COMPARED_CLASSES[name]
    node = _definitions(module)[name][0]
    assert isinstance(node, ast.ClassDef), name
    return node


def _base_class(cls: ast.ClassDef, base: ast.expr) -> tuple[ast.ClassDef, bool]:
    """The class *base* names, where *cls* is defined, and whether *cls*
    derives from it by `Static[...]`, as from one of Codon's exceptions."""
    text = ast.unparse(base)
    static = text.startswith("Static[") and text.endswith("]")
    name = text[len("Static[") : -1] if static else text
    if id(cls) in _ORIGIN:
        module, root = _ORIGIN[id(cls)]
        definitions = _definitions(module, root).get(name, [])
        for node in _parse(root / f"{module}.codon").body:
            # An import from a parent package (`from .._ros2 import ...`),
            # which _definitions leaves out.
            if isinstance(node, ast.ImportFrom) and node.level > 1 and node.module:
                source = Path(module).parents[node.level - 1]
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        definitions = _definitions(
                            (source / node.module.replace(".", "/")).as_posix(), root
                        ).get(alias.name, [])
        if definitions and isinstance(definitions[0], ast.ClassDef):
            return definitions[0], static
    assert name in _CODON_EXCEPTIONS, f"{cls.name}: base {text} is not declared"
    return _CODON_EXCEPTIONS[name], static


def _codon_fields(cls: ast.ClassDef) -> list[tuple[str, bool]]:
    """The fields Codon gives *cls*, its bases' first: (name, has default)."""
    out: list[tuple[str, bool]] = []
    for base in cls.bases:
        out += _codon_fields(_base_class(cls, base)[0])
    return out + [(p[1], p[2]) for p in _model_fields(cls)]


def _inherited_inits(cls: ast.ClassDef) -> list[ast.FunctionDef]:
    """The __init__s of the bases *cls* derives from by `Static[...]`: Codon
    tries them after the class's own (a `FrameMismatchError("x")` took
    Exception's), but not those of a base it derives from plainly."""
    out: list[ast.FunctionDef] = []
    for base in cls.bases:
        node, static = _base_class(cls, base)
        if static:
            out += [i for i in _inits(node) if not _is_compile_error(i)]
            out += _inherited_inits(node)
    return out


def _model_signature(fn: ast.FunctionDef) -> inspect.Signature:
    """*fn*'s parameters but self, as a call binds them in Codon: a
    `_Required()` default is a default too (the body refuses it)."""
    args = fn.args
    positional = [*args.posonlyargs, *args.args][1:]
    defaults = [False] * (len(positional) - len(args.defaults)) + [True] * len(
        args.defaults
    )
    empty = inspect.Parameter.empty
    params = [
        inspect.Parameter(
            a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None if d else empty
        )
        for a, d in zip(positional, defaults)
    ]
    if args.vararg is not None:
        params.append(
            inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
        )
    params += [
        inspect.Parameter(
            a.arg, inspect.Parameter.KEYWORD_ONLY, default=empty if d is None else None
        )
        for a, d in zip(args.kwonlyargs, args.kw_defaults)
    ]
    return inspect.Signature(params)


def _fields_signature(fields: list[tuple[str, bool]]) -> inspect.Signature:
    """The constructor Codon generates: the fields, by position or name."""
    empty = inspect.Parameter.empty
    return inspect.Signature(
        [
            inspect.Parameter(
                n, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None if d else empty
            )
            for n, d in fields
        ]
    )


def _takes(signature: inspect.Signature, arity: int) -> bool:
    """*signature* binds *arity* positional arguments."""
    try:
        signature.bind(*([None] * arity))
    except TypeError:
        return False
    return True


def _positional_count(signature: inspect.Signature) -> int:
    return sum(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for p in signature.parameters.values()
    )


def _refuses_extra_positionals(fn: ast.FunctionDef) -> bool:
    """*fn* refuses what its `*_kw` takes (`_keyword_only(_kw, ...)`), or
    every call (its body is a compile_error)."""
    return _is_compile_error(fn) or any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_keyword_only"
        and node.args
        and ast.unparse(node.args[0]) == "_kw"
        for node in ast.walk(fn)
    )


def _arities_only_codon_takes(name: str) -> list[int]:
    """The numbers of positional arguments Python's constructor of *name*
    refuses, its model's constructor does not take, and Codon would still
    build it by: with the constructor it generates of the fields (#225,
    #236, #244), or an __init__ of a `Static[...]` base (an exception's).

    Codon tries a class's own __init__s from the last declared and those two
    after them, so each arity found needs an __init__ of as many arguments,
    a compile_error, ahead of the constructor (_refusing_init)."""
    _module, python = _COMPARED_CLASSES[name]
    node = _model_class(name)
    inits = _inits(node)
    if not inits:
        return []  # the generated constructor is its own, compared with Python's
    own = _model_signature(inits[-1])
    fallbacks = [_fields_signature(_codon_fields(node))]
    fallbacks += [_model_signature(i) for i in _inherited_inits(node)]
    expected = inspect.signature(python)
    out: list[int] = []
    for arity in range(max(_positional_count(f) for f in fallbacks) + 1):
        if _takes(expected, arity):
            continue
        if _takes(own, arity):
            # Only through `*_kw`, the model's parameters being Python's.
            assert _refuses_extra_positionals(inits[-1]), (
                f"{name}(): the constructor takes {arity} positional "
                "arguments, which Python refuses, and does not refuse them"
            )
            continue
        if any(_takes(f, arity) for f in fallbacks):
            out.append(arity)
    return out


#: The compared classes Codon would build by a number of positional arguments
#: Python refuses: name -> those numbers.
_ARITIES_ONLY_CODON_TAKES: dict[str, list[int]] = {
    name: arities
    for name in sorted(_COMPARED_CLASSES)
    if (arities := _arities_only_codon_takes(name))
}

#: The conditions and actions: each derives from a base with one field, its
#: label, and Python's constructor takes no label by position.
_LABELLED = sorted(
    name
    for name, (module, _python) in _COMPARED_CLASSES.items()
    if module in ("conditions", "actions")
    and _model_class(name).bases
    and [n for n, _ in _codon_fields(_model_class(name))] == ["label"]
)


def test_the_classes_codon_builds_by_other_arities_are_found() -> None:
    # What #225, #236 and #244 found: the rule would pass for nothing if the
    # detector found nothing. A class without fields (RouteMatch, #224, and
    # those of #225) by no argument; a condition or an action by its label;
    # DebugMarkers, CoverItem, OddCondition and RunProgress by their fields
    # (RouteCoverage's are Python's since #298); the exceptions by Exception's constructor and their
    # fields (BaseException's eight first).
    found = _ARITIES_ONLY_CODON_TAKES
    for name in (
        "CoverageCollector",
        "CoverageReport",
        "EgoDriverGrpcClient",
        "PedestrianEntity",
        "RouteFrame",
        "RouteMatch",
        "RouteSegmentMatch",
        "Ros2EgoTransport",
        "VehicleEntity",
    ):
        assert found.get(name) == [0], name
    # A condition or an action whose label Python takes by no position;
    # every one of them declares the guard of one all the same (below).
    assert len(_LABELLED) > 30
    assert {"SpeedCondition", "TurnAction"} <= set(found)
    for name in _LABELLED:
        assert found.get(name, [1]) == [1], name
    assert {
        name: found.get(name)
        for name in (
            "CoverItem",
            "DebugMarkers",
            "DevContainerError",
            "FrameMismatchError",
            "MapCacheError",
            "OddCondition",
            "RunProgress",
        )
    } == {
        "CoverItem": [1],
        "DebugMarkers": [1],
        "DevContainerError": [0, 2, 8],
        "FrameMismatchError": [1, 2, 10],
        "MapCacheError": [0, 2, 8],
        "OddCondition": [1],
        "RunProgress": [4],
    }
    # Their `*_kw` refuses what the fields would take (#244's review).
    for name in ("BaseScenario", "OddAttribute", "OddModule"):
        assert name not in found, name
    # Its fields are Python's, in Python's order (#298): the constructor
    # Codon generates takes no call Python's refuses.
    assert "RouteCoverage" not in found


@pytest.mark.parametrize("name", sorted(_ARITIES_ONLY_CODON_TAKES))
def test_codon_cannot_build_a_class_by_what_python_refuses(name: str) -> None:
    # Codon builds a class by its fields too, by position or by name (#225,
    # #236, #244), and an exception by Exception's constructor, after the
    # __init__s the class declares. Where Python refuses as many positional
    # arguments, the model declares an __init__ of that many, a
    # compile_error, ahead of the constructor; the one of the fields' arity
    # takes their names, so a call by the fields' names is refused too.
    node = _model_class(name)
    fields = [n for n, _ in _codon_fields(node)]
    for arity in _ARITIES_ONLY_CODON_TAKES[name]:
        guard = _refusing_init(node, arity)
        assert guard is not None, (
            f"{name}: Codon builds it with {arity} positional argument"
            f"{'s' * (arity != 1)}, which "
            f"Python refuses; declare `def __init__(self{', ...' * bool(arity)})`"
            " of as many, a compile_error, ahead of the constructor"
        )
        if arity == len(fields):
            names = [a.arg for a in [*guard.args.posonlyargs, *guard.args.args][1:]]
            assert names == fields, f"{name}: the guard of {arity} takes its fields"


#: A call: how many positional arguments, and the names passed by keyword.
_Shape = tuple[int, tuple[str, ...]]


def _call_shapes(signature: inspect.Signature) -> list[_Shape]:
    """Every call *signature* (positional-or-keyword parameters) binds."""
    params = list(signature.parameters.values())
    empty = inspect.Parameter.empty
    out: list[_Shape] = []
    for count in range(len(params) + 1):
        rest = params[count:]
        required = tuple(p.name for p in rest if p.default is empty)
        optional = [p.name for p in rest if p.default is not empty]
        for k in range(len(optional) + 1):
            for chosen in itertools.combinations(optional, k):
                out.append((count, required + chosen))
    return out


def _binds(signature: inspect.Signature, shape: _Shape) -> bool:
    count, names = shape
    try:
        signature.bind(*([None] * count), **dict.fromkeys(names))
    except TypeError:
        return False
    return True


def _shapes_only_codon_takes(name: str) -> list[_Shape]:
    """The calls, by position or by keyword, that Python's constructor of
    *name* refuses, its model's constructor does not take, and the
    constructor Codon generates of the fields or an __init__ of a
    `Static[...]` base would: _arities_only_codon_takes, keywords included
    (`DevContainerError(typename="x")` took Exception's, #244's review)."""
    _module, python = _COMPARED_CLASSES[name]
    node = _model_class(name)
    inits = _inits(node)
    if not inits:
        return []
    own = _model_signature(inits[-1])
    fallbacks = [_fields_signature(_codon_fields(node))]
    fallbacks += [_model_signature(i) for i in _inherited_inits(node)]
    expected = inspect.signature(python)
    out: list[_Shape] = []
    for fallback in fallbacks:
        for shape in _call_shapes(fallback):
            if shape in out or _binds(expected, shape):
                continue
            if _binds(own, shape):
                # Only through `*_kw`: _arities_only_codon_takes checks it
                # refuses them.
                continue
            out.append(shape)
    return out


def test_the_calls_codon_builds_by_keyword_are_found() -> None:
    # The keyword calls #244's review found passing the check: the rule
    # below would pass for nothing if the detector found none of them.
    found = {
        name: _shapes_only_codon_takes(name)
        for name in ("DevContainerError", "FrameMismatchError", "MapCacheError")
    }
    for name in found:
        assert (1, ("message",)) in found[name], name
        assert (0, ("typename", "message")) in found[name], name
    assert (0, ("typename",)) in found["DevContainerError"]
    assert (0, ("typename",)) in found["MapCacheError"]
    assert (0, ("name",)) in _shapes_only_codon_takes("CoverItem")
    assert (0, ("current", "total", "scenario_name", "status")) not in (
        _shapes_only_codon_takes("RunProgress")
    )


@pytest.mark.parametrize(
    "name", sorted(n for n in _COMPARED_CLASSES if _shapes_only_codon_takes(n))
)
def test_codon_cannot_build_a_class_by_keywords_python_refuses(name: str) -> None:
    # Beside the positional rule above: the constructor Codon generates and
    # an exception base's take their parameters' names too, so a call by
    # keyword Python refuses (`DevContainerError(typename="x")`,
    # `FrameMismatchError("x", message="y")`) needs an __init__, a
    # compile_error, ahead of the constructor, that takes the same names.
    node = _model_class(name)
    guards = [_model_signature(i) for i in _inits(node)[:-1] if _is_compile_error(i)]
    missing = [
        f"{count} positional{''.join(f', {n}=' for n in names)}"
        for count, names in _shapes_only_codon_takes(name)
        if not any(_binds(g, (count, names)) for g in guards)
    ]
    assert not missing, (
        f"{name}: Codon builds it by calls Python refuses ({'; '.join(missing)});"
        " declare an __init__ of those parameter names, a compile_error, ahead of"
        " the constructor"
    )


@pytest.mark.parametrize("name", _LABELLED)
def test_a_condition_or_action_cannot_be_built_by_its_label_alone(name: str) -> None:
    # Beside the rule above: where Python takes one positional argument
    # (`TimeoutCondition(10)`), a call the constructor refuses by its type
    # still reaches the one Codon generates of the label it inherits
    # (`TimeoutCondition("10")` passed, #236). Each condition and action
    # declares the guard of one argument, ahead of its constructor.
    guard = _refusing_init(_model_class(name), 1)
    assert guard is not None, name
    assert [a.arg for a in guard.args.args[1:]] == ["label"], name


def test_the_codon_exceptions_are_declared_as_in_codon() -> None:
    # _CODON_EXCEPTIONS is Codon's standard library, as far as the model's
    # exceptions derive from it.
    toolchain = available_toolchain()
    if toolchain is None:
        pytest.skip("no Codon compiler")
    path = toolchain.codon_dir / "lib/codon/stdlib/internal/types/error.codon"
    stdlib = {
        node.name: node for node in _parse(path).body if isinstance(node, ast.ClassDef)
    }
    for name, node in _CODON_EXCEPTIONS.items():
        codon = stdlib[name]
        assert [ast.unparse(b) for b in codon.bases] == [
            ast.unparse(b) for b in node.bases
        ], name
        assert _model_fields(codon) == _model_fields(node), name
        assert [_model_params(i, drop_first=True) for i in _inits(codon)] == [
            _model_params(i, drop_first=True) for i in _inits(node)
        ], name


def test_the_cyclone_transport_model_cannot_be_built_with_no_arguments() -> None:
    # ros2.cyclone is compared only where cyclonedds is installed (below);
    # its model needs the same __init__ as any class without fields.
    node = _definitions("ros2/cyclone")["CycloneRos2Transport"][0]
    assert isinstance(node, ast.ClassDef)
    assert [ast.unparse(b) for b in node.bases] == ["Ros2EgoTransport"]
    assert _codon_fields(node) == []
    assert _refusing_init(node, 0) is not None


def test_the_traffic_backend_factory_model_is_a_stand_in() -> None:
    # Python's is a Callable alias, the model's an empty class (_DIFFERING):
    # neither declares anything a scenario could call or build differently.
    assert typing.get_origin(traffic_registry.TrafficBackendFactory) is (
        typing.get_origin(typing.Callable)
    )
    node = _definitions("traffic")["TrafficBackendFactory"][0]
    assert isinstance(node, ast.ClassDef)
    assert not node.bases and [ast.unparse(s) for s in node.body] == ["pass"]


def test_the_undecided_model_is_openodd_s_sentinel() -> None:
    # openodd's UNDECIDED is the one instance of a class Python cannot build
    # another of; odd.model names the class Undecided (openodd does not
    # export it).  The model allocates its one instance without the
    # constructor, which is a compile error, and writes it as Python does.
    sentinel = _definitions("odd")["UNDECIDED"][0]
    assert isinstance(sentinel, ast.AnnAssign) and sentinel.value is not None
    assert ast.unparse(sentinel.annotation) == "Undecided"
    assert ast.unparse(sentinel.value) == "Undecided.__new__()"
    cls = _definitions("odd")["Undecided"][0]
    assert isinstance(cls, ast.ClassDef)
    methods = _methods(cls)
    assert _model_params(methods["__init__"], drop_first=True) == []
    assert "compile_error" in ast.unparse(methods["__init__"])
    for method in ("__repr__", "__str__"):
        returned = [n for n in ast.walk(methods[method]) if isinstance(n, ast.Return)]
        assert [ast.literal_eval(r.value) for r in returned if r.value] == [
            getattr(odd.UNDECIDED, method)()
        ], method
    assert odd_model.Undecided is type(odd.UNDECIDED)
    with pytest.raises(TypeError):
        odd_model.Undecided()


def test_the_odd_case_models_are_stand_ins() -> None:
    # OddSample and OddCase (_DIFFERING): an opaque, unbuildable class for
    # openodd's sample, and the pair Python's alias names, in order.
    sample = _definitions("odd")["OddSample"][0]
    assert isinstance(sample, ast.ClassDef)
    init = _methods(sample)["__init__"]
    assert _model_params(init, drop_first=True) == []
    calls = [s.value for s in init.body if isinstance(s, ast.Expr)]
    assert [ast.unparse(c.func) for c in calls if isinstance(c, ast.Call)] == [
        "compile_error"
    ]
    case = _definitions("odd")["OddCase"][0]
    assert isinstance(case, ast.ClassDef)
    assert [ast.unparse(d) for d in case.decorator_list] == ["tuple"]
    fields = [
        ast.unparse(stmt.annotation)
        for stmt in case.body
        if isinstance(stmt, ast.AnnAssign)
    ]
    assert typing.get_origin(odd.OddCase) is tuple
    assert fields == [t.__name__ for t in typing.get_args(odd.OddCase)]


def test_the_launcher_model_is_the_protocol_s_methods() -> None:
    # Python cannot instantiate a Protocol, so the model's constructor is a
    # compile error (below); what AutowareEgoEntity and ScenarioQueue call is
    # declared as Python's.
    python = stack_launcher.AutowareLauncher
    assert getattr(python, "_is_protocol", False)
    node = _definitions("autoware_stack/launcher")["AutowareLauncher"][0]
    assert isinstance(node, ast.ClassDef)
    methods = {n: m for n, m in _methods(node).items() if n != "__init__"}
    assert sorted(methods) == ["close", "poll", "prepare", "start", "stop"]
    for method_name, method in methods.items():
        assert _model_params(method, drop_first=True) == _python_params(
            getattr(python, method_name), drop_first=True
        ), f"AutowareLauncher.{method_name}(): the model's parameters differ"


#: What a scenario reads of an unbuildable class a modelled function returns
#: (merge_coverage() a CoverageReport, #298): the members Codon has types for.
_READ_ON_WHAT_A_FUNCTION_RETURNS: dict[str, tuple[str, ...]] = {
    "CoverageReport": ("group_grade", "groups", "runs", "scenarios", "to_markdown"),
    # What CommunicationLossAction calls on the HILS ego find_hils_entity()
    # returns.
    "HilsEgoEntity": ("disconnect", "reconnect"),
}


@pytest.mark.parametrize("name", sorted(_UNBUILDABLE))
def test_a_class_only_re_exported_cannot_be_built_in_a_scenario(name: str) -> None:
    node = _definitions(_UNBUILDABLE[name])[name][0]
    assert isinstance(node, ast.ClassDef)
    assert not node.bases, f"{name}: a base would bring methods to compare"
    methods = _methods(node)
    assert set(methods) <= {
        "__init__",
        "close",
        "poll",
        "prepare",
        "start",
        "stop",
        *_READ_ON_WHAT_A_FUNCTION_RETURNS.get(name, ()),
    }, name
    body = methods["__init__"].body
    assert len(body) == 1 and isinstance(body[0], ast.Expr), name
    call = body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == "compile_error", name


def test_the_cyclone_model_is_the_transport_ros2_entity_builds() -> None:
    # ros2.cyclone imports cyclonedds (the `ros2` extra), so it is compared
    # only where that is installed. The model declares the constructor
    # entity.ros2_entity._cyclone_transport() calls, and derives from the
    # Ros2EgoTransport that the library check replaces with the checked one.
    # Skip on cyclonedds alone: an ImportError of ros2.cyclone's own (a moved
    # name) must fail the test, not skip it.
    pytest.importorskip("cyclonedds")
    from autoware_carla_scenario.ros2 import cyclone  # noqa: PLC0415

    assert sorted(_definitions("ros2/cyclone")) == ["CycloneRos2Transport"]
    node = _definitions("ros2/cyclone")["CycloneRos2Transport"][0]
    assert isinstance(node, ast.ClassDef)
    assert [ast.unparse(b) for b in node.bases] == ["Ros2EgoTransport"]
    assert cyclone.CycloneRos2Transport.__mro__[1] is ros2.Ros2EgoTransport
    _assert_declared_as_in_python(
        "CycloneRos2Transport", "ros2/cyclone", cyclone.CycloneRos2Transport
    )


#: The model of the third-party carla_driver_interface (its geometry): what the
#: driver modules use of it, compared with the installed package.
_CDI_MODEL = model_dir() / "carla_driver_interface"
_CDI_NAMES = ("Pose", "Trajectory", "waypoints_to_proto")

#: (class, field) -> (the package's annotation, the model's): an array is
#: Codon's of its dtype and number of dimensions (the docstrings give them).
_CDI_FIELDS = {
    ("Pose", "position"): ("NDArray[np.float64]", "np.ndarray[np.float64, 1]"),
    ("Pose", "quat_xyzw"): ("NDArray[np.float64]", "np.ndarray[np.float64, 1]"),
    ("Trajectory", "timestamps_us"): ("List[int]", "List[int]"),
    ("Trajectory", "poses"): ("List[Pose]", "List[Pose]"),
}


def test_the_carla_driver_interface_model_declares_only_the_compared_names() -> None:
    # A name the package lacks would pass a scenario's check and fail its run:
    # whatever the model declares is compared below, dunder methods included.
    definitions = _definitions("geometry", _CDI_MODEL)
    assert sorted(definitions) == sorted(_CDI_NAMES)
    for name in _CDI_NAMES:
        node = definitions[name][0]
        if isinstance(node, ast.ClassDef):
            python = getattr(cdi_geometry, name)
            for method in _methods(node):
                assert hasattr(python, method), f"{name}.{method} is in the model only"


@pytest.mark.parametrize("name", _CDI_NAMES)
def test_the_carla_driver_interface_model_is_declared_as_in_the_package(
    name: str,
) -> None:
    _assert_declared_as_in_python(
        name, "geometry", getattr(cdi_geometry, name), root=_CDI_MODEL
    )


@pytest.mark.parametrize("name", ["Pose", "Trajectory"])
def test_the_carla_driver_interface_fields_have_the_packages_types(name: str) -> None:
    node = _definitions("geometry", _CDI_MODEL)[name][0]
    assert isinstance(node, ast.ClassDef)
    model = {
        stmt.target.id: ast.unparse(stmt.annotation)
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    python = getattr(cdi_geometry, name)
    init_fields = [f.name for f in dataclasses.fields(python) if f.init]
    assert list(model) == init_fields, f"{name}: the model's fields differ"
    for field_name, annotation in model.items():
        expected_python, expected_model = _CDI_FIELDS[(name, field_name)]
        assert (
            python.__annotations__[field_name] == expected_python
        ), f"{name}.{field_name} changed type in carla_driver_interface"
        assert annotation == expected_model, f"{name}.{field_name}"


#: The model of the third-party python-dotenv: what the checked server uses
#: of it, compared with the installed package.
_DOTENV_MODEL = model_dir() / "dotenv"


def _public(definitions: dict[str, list[ast.stmt]]) -> list[str]:
    return sorted(name for name in definitions if not name.startswith("_"))


def test_the_dotenv_model_is_load_dotenv_as_in_the_package() -> None:
    assert _public(_definitions("__init__", _DOTENV_MODEL)) == ["load_dotenv"]
    _assert_declared_as_in_python(
        "load_dotenv", "__init__", dotenv.load_dotenv, root=_DOTENV_MODEL
    )


# ---------------------------------------------------------------------------
# What Python exports, the model declares or leaves out on purpose (#237)
# ---------------------------------------------------------------------------


def _model_stem(module: str) -> str:
    """The model file of *module* relative to _MODEL, without its suffix."""
    parts = module.split(".")[1:]
    return "/".join(parts) if parts else "__init__"


#: Every model module the scenario check resolves an import of a Python
#: module against (check._model_modules()): the helpers (`_tc`, `_poses`, ...)
#: have no Python module, so no scenario imports them.
_MODELLED_MODULES = sorted(
    module
    for module in _model_modules()
    if importlib.util.find_spec(module) is not None
)


def _python_public_names(module: str) -> list[str]:
    """What *module* exports in Python: its ``__all__``, else the public names
    it defines. Read from its source where it cannot be imported (a missing
    optional dependency: ros2.cyclone's cyclonedds)."""
    try:
        python = importlib.import_module(module)
    except ImportError:
        python = None
    if python is not None and hasattr(python, "__all__"):
        return list(python.__all__)
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None, module
    names: list[str] = []
    for node in _parse(Path(spec.origin)).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return list(ast.literal_eval(node.value))  # type: ignore[arg-type]
                if isinstance(target, ast.Name):
                    names.append(target.id)
    return [name for name in names if not name.startswith("_")]


def test_the_modelled_modules_are_the_public_ones() -> None:
    # The packages the scenario check reads, and the modules of the library
    # check's boundary, are all here: a model file without a Python module
    # is a helper (`_` prefix), and none of those is listed.
    assert "autoware_carla_scenario" in _MODELLED_MODULES
    assert "autoware_carla_scenario.action_state" in _MODELLED_MODULES
    assert "autoware_carla_scenario.conditions" in _MODELLED_MODULES
    assert "autoware_carla_scenario.maps.z_offset_cache" in _MODELLED_MODULES
    assert not [m for m in _MODELLED_MODULES if m.rsplit(".", 1)[-1].startswith("_")]
    assert set(NOT_MODELLED) <= set(_MODELLED_MODULES)


@pytest.mark.parametrize("module", _MODELLED_MODULES)
def test_a_public_name_is_modelled_or_listed_as_not_modelled(module: str) -> None:
    # A name Python exports that the model lacks fails a correct scenario's
    # check (`cannot import name`), unless the check says why (NOT_MODELLED).
    # And a name listed there must still be one: public in Python and absent
    # from the model.
    public = set(_python_public_names(module))
    undeclared = public - _model_names(module)
    listed = set(NOT_MODELLED.get(module, {}))
    assert not sorted(
        undeclared - listed
    ), f"{module}: public in Python, neither modelled nor in NOT_MODELLED"
    assert not sorted(
        listed - undeclared
    ), f"{module}: in NOT_MODELLED, but modelled or not public in Python"
    assert all(NOT_MODELLED.get(module, {}).values()), "a reason for each name"


#: Modules of a modelled package the model has no module for
#: (conditions.comparison, actions.base): the check refuses an import of one,
#: and says where else the model declares the names.
_UNMODELLED_SUBMODULES = sorted(
    module
    for module in package_modules()
    if module not in _model_modules()
    and (_model_module(module) or "autoware_carla_scenario")
    != "autoware_carla_scenario"
)

_SUGGESTION = re.compile(r"import (?P<names>[\w, ]+) from (?P<module>[\w.]+)")


@pytest.mark.parametrize("module", _UNMODELLED_SUBMODULES)
def test_the_advice_for_an_unmodelled_module_runs_in_python(module: str) -> None:
    # "import X from M instead" is given only where Python's M has the same
    # X: the advice "import from autoware_carla_scenario instead" named a
    # package without ActionState (#237).
    try:
        python = importlib.import_module(module)
    except ImportError as exc:
        pytest.skip(f"{module}: {exc}")
    names = [n for n in _python_public_names(module) if hasattr(python, n)]
    message = _unmodelled_module(module, names)
    assert message.startswith(f"{module} is not part of the Codon model"), message
    for match in _SUGGESTION.finditer(message):
        target = importlib.import_module(match["module"])
        for name in match["names"].split(", "):
            assert getattr(target, name) is getattr(python, name), (name, message)
            assert name in _model_names(match["module"]), (name, message)
    suggested = {
        name
        for match in _SUGGESTION.finditer(message)
        for name in match["names"].split(", ")
    }
    # Every name is suggested somewhere, or said to be missing from the model.
    for name in names:
        assert name in suggested or name in message, (name, message)


#: Public names the model declares that are compared by a test of their own
#: rather than by _assert_declared_as_in_python: name -> the test.
_COMPARED_APART = {
    "AutowareLauncher": "test_the_launcher_model_is_the_protocol_s_methods",
    "TrafficBackendFactory": "test_the_traffic_backend_factory_model_is_a_stand_in",
}


def _already_compared(module: str, name: str) -> bool:
    stem = _model_stem(module)
    return (
        (stem == "__init__" and name in _EXPORTS)
        or (name in _BOUNDARY and _BOUNDARY[name][0] == stem)
        or name in _MODULE_MODELS.get(stem, (None, ()))[1]
        or name in _COMPARED_APART
        or (name in _DIFFERING and _DIFFERING[name][0] == stem)
    )


_DECLARED_PUBLIC_NAMES = [
    (module, name)
    for module in _MODELLED_MODULES
    for name in sorted(set(_python_public_names(module)) & _model_names(module))
    if not _already_compared(module, name)
]


@pytest.mark.parametrize(("module", "name"), _DECLARED_PUBLIC_NAMES)
def test_a_re_exported_public_name_is_declared_as_in_python(
    module: str, name: str
) -> None:
    # A public name a module re-exports (kinematics' frame_of, conditions'
    # find_actor_in_list) is compared where it is imported too, so the
    # comparisons above extend to every name a scenario can import.
    try:
        python = importlib.import_module(module)
    except ImportError as exc:
        pytest.skip(f"{module}: {exc}")
    _assert_declared_as_in_python(
        name, _model_stem(module), inspect.unwrap(getattr(python, name))
    )


# ---------------------------------------------------------------------------
# Every public member of a compared class is modelled or listed (#265)
# ---------------------------------------------------------------------------


#: The optional dependencies a compared module may import: where one is not
#: installed (cyclonedds has no wheel for every Python and arch CI runs), the
#: module's classes are left out of the member check, as
#: test_the_cyclone_model_is_the_transport_ros2_entity_builds leaves them out.
_OPTIONAL_DEPENDENCIES = frozenset({"cyclonedds"})


def _optional_import(module: str) -> Any:
    """*module*, or None when it imports an optional dependency that is not
    installed. Any other ImportError (a moved name) still fails collection."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in _OPTIONAL_DEPENDENCIES:
            _UNIMPORTABLE.update(n for m, n in _DECLARED_PUBLIC_NAMES if m == module)
            return None
        raise


#: The compared classes left out because their module's optional dependency
#: is not installed: their NOT_MODELLED_MEMBERS entries excuse nothing here.
_UNIMPORTABLE: set[str] = set()


def _compared_classes() -> dict[str, tuple[str, type]]:
    """Every class the tests above compare with its model class, by name:
    (model module, the Python class), enums included (#307): an enum's
    members are compared above, its `name`, `value` and its own methods and
    properties here. Union aliases are not classes the model declares
    members of."""
    candidates: list[tuple[str, str, Any]] = [
        *((name, module, getattr(acs, name)) for name, module in _EXPORTS.items()),
        *((name, module, python) for name, (module, python) in _BOUNDARY.items()),
        *(
            (name, module, getattr(python_module, name))
            for module, (python_module, names) in _MODULE_MODELS.items()
            for name in names
        ),
        *(
            (name, _model_stem(module), getattr(python_module, name))
            for module, name in _DECLARED_PUBLIC_NAMES
            if (python_module := _optional_import(module)) is not None
        ),
    ]
    out: dict[str, tuple[str, type]] = {}
    for name, module, python in candidates:
        if not inspect.isclass(python):
            continue
        node = _definitions(module).get(name, [None])[0]
        if isinstance(node, ast.ClassDef):
            out.setdefault(name, (module, python))
    return out


_MEMBER_CLASSES = _compared_classes()


def _model_members(cls: ast.ClassDef) -> set[str]:
    """What the model class declares, its model bases' included: methods,
    properties, fields and class attributes."""
    out: set[str] = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            out.add(stmt.name)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            out |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
    for base in cls.bases:
        node, _static = _base_class(cls, base)
        if node.name not in _CODON_EXCEPTIONS:
            out |= _model_members(node)
    return out


def _python_members(python: type) -> set[str]:
    """The public methods, properties, class attributes, dataclass fields and
    instance attributes that the package's Python or openodd's (whose classes
    the package re-exports, #298) defines on *python* or a class it derives
    from. A dataclass's field is a member too, read on the instance, beside
    the constructor's parameter it is compared as; so is an attribute a
    method sets on the instance (`self.timeout_seconds = ...` in __init__,
    #307), which Python reads like any other. An enum's members are its
    class attributes; it has Enum's `name` and `value` too (#307)."""
    out: set[str] = set()
    if issubclass(python, enum.Enum):
        out |= {"name", "value"}
    for klass in python.__mro__:
        if not defines_members(klass.__module__):
            continue
        out |= {n for n in vars(klass) if not n.startswith("_")}
        if dataclasses.is_dataclass(klass):
            out |= {
                f.name for f in dataclasses.fields(klass) if not f.name.startswith("_")
            }
        out |= _instance_attributes(klass)
    return out


def _instance_attributes(klass: type) -> set[str]:
    """The public attributes the methods of *klass* (its own, not those of
    the classes it derives from) set on the instance: each `self.<name>`
    stored to (`=`, `+=`, annotated, a tuple's target, a `for` or `with`
    target) and each `setattr(self, "<name>", ...)`, in the body of a method
    whose first parameter is the instance, read from the source's AST. A
    static method's first parameter is no instance, and a class method's sets
    a class attribute, which vars() reads. A class or a function nested in
    the method that names its own `self` is another object's, and is not
    read. A class with no Python source (openodd's Rust extension) has none
    to read; an attribute set under a name computed at run time is not
    found."""
    try:
        source = textwrap.dedent(inspect.getsource(klass))
    except (OSError, TypeError):
        return set()
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.ClassDef) and node.name == klass.__name__, klass
    out: set[str] = set()
    for method in node.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _decorators(method) & {"staticmethod", "classmethod"}:
            continue
        positional = [*method.args.posonlyargs, *method.args.args]
        if not positional:
            continue
        instance = positional[0].arg
        for each in _instance_scope(method.body, instance):
            if (
                isinstance(each, ast.Attribute)
                and isinstance(each.ctx, ast.Store)
                and isinstance(each.value, ast.Name)
                and each.value.id == instance
            ):
                name = each.attr
            elif (
                isinstance(each, ast.Call)
                and ast.unparse(each.func) == "setattr"
                and len(each.args) >= 2
                and isinstance(each.args[0], ast.Name)
                and each.args[0].id == instance
                and isinstance(each.args[1], ast.Constant)
                and isinstance(each.args[1].value, str)
            ):
                name = each.args[1].value
            else:
                continue
            if not name.startswith("_"):
                out.add(name)
    return out


def _instance_scope(body: list[ast.stmt], instance: str) -> Iterator[ast.AST]:
    """Every node of a method's *body* in which *instance* is still the
    method's: not inside a nested class, nor inside a nested function or
    lambda that takes a parameter of that name."""
    stack: list[ast.AST] = list(body)
    while stack:
        each = stack.pop()
        if isinstance(each, ast.ClassDef):
            continue
        if isinstance(each, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = each.args
            params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            params += [a for a in (args.vararg, args.kwarg) if a is not None]
            if any(p.arg == instance for p in params):
                continue
        yield each
        stack.extend(ast.iter_child_nodes(each))


def _unmodelled_members(name: str) -> set[str]:
    module, python = _MEMBER_CLASSES[name]
    node = _definitions(module)[name][0]
    assert isinstance(node, ast.ClassDef)
    return _python_members(python) - _model_members(node)


class _SetsAttributes:
    """What _instance_attributes() reads, and what it leaves out (#307)."""

    def __init__(self, me: int) -> None:
        self.plain = me
        self.first, (self.second, *self.rest) = 1, (2, 3)
        self.counter: int = 0
        self.counter += 1
        self.items: list[int] = []
        self.items[0:0] = [1]  # an item stored, not an attribute
        self._private = 0
        setattr(self, "by_setattr", 0)
        for self.looped in range(1):
            pass

        def helper() -> None:
            self.in_closure = 0  # still the instance

        def other(self: Any) -> None:
            self.of_another = 0  # its own `self`

        class Inner:
            def __init__(self) -> None:
                self.of_inner = 0

        helper()
        other(Inner())

    @staticmethod
    def build(entry: Any) -> None:
        entry.of_an_argument = 0


def test_instance_attributes_are_read_from_the_method_s_own_scope() -> None:
    assert _instance_attributes(_SetsAttributes) == {
        "plain",
        "first",
        "second",
        "rest",
        "counter",
        "items",
        "by_setattr",
        "looped",
        "in_closure",
    }


def test_the_member_comparison_reads_every_kind_of_compared_class() -> None:
    # The classes of each table above, a class the model declares by name
    # only among them.
    assert {"BaseScenario", "EgoVehicle", "VehicleEntity", "RoutingAction"} <= set(
        _MEMBER_CLASSES
    )
    assert {"MapManager", "RouteFrame", "CarlaDriverEntity", "Measure"} <= set(
        _MEMBER_CLASSES
    )
    # An enum: Enum's name and value, and its own methods and properties
    # (#307), beside its members.
    assert {"TickTiming", "ComparisonRule", "LaneChangeDirection"} <= set(
        _MEMBER_CLASSES
    )
    assert {"name", "value", "text", "LESS_THAN"} <= _python_members(ComparisonRule)
    assert "to_carla_bool" in _python_members(acs.LaneChangeDirection)
    # An attribute a method sets on the instance (#307), the package's
    # (__init__'s, and a counter execute() adds to) or openodd's.
    assert "timeout_seconds" in _python_members(acs.TimeoutCondition)
    assert "ego_type" in _python_members(acs.BaseScenario)
    assert "removed" in _python_members(acs.TrafficSinkAction)
    assert {"target", "item", "probe"} <= _python_members(acs.OddAttribute)
    assert "attributes" in _python_members(acs.OddDefinition)
    # A static method's first parameter is no instance: openodd's collector
    # sets `entry.failed` in one.
    assert "failed" not in _python_members(coverage_collector.CoverageCollector)
    # Inherited members count: set_desired_speed is BackendDriven's in Python.
    assert "set_desired_speed" in _python_members(acs.VehicleEntity)
    assert "measures" in _python_members(acs.BaseScenario)
    # A dataclass field is a member too, read on the instance (#298), beside
    # the constructor's parameter it is compared as.
    assert "lanelet_id" in _python_members(acs.Lanelet2Pose)
    assert "position" in _python_members(acs.TrajectoryVertex)
    # openodd's classes, which the package re-exports, are compared (#298),
    # a Rust extension's among them (OddCondition is openodd._core's).
    assert {"OddModule", "OddCondition", "CoverItem", "CoverageReport"} <= set(
        _MEMBER_CLASSES
    )
    assert "as_situation" in _python_members(acs.OddModule)
    assert "describe" in _python_members(_MEMBER_CLASSES["OddCondition"][1])


@pytest.mark.parametrize("name", sorted(_MEMBER_CLASSES))
def test_a_public_member_is_modelled_or_listed_as_not_modelled(name: str) -> None:
    # The converse of _assert_declared_as_in_python, which checks that what
    # the model declares exists in Python: a member Python has and the model
    # lacks fails a correct scenario with Codon's bare "has no attribute"
    # (BaseScenario.measures, VehicleEntity.set_desired_speed), unless
    # NOT_MODELLED_MEMBERS says why.
    _module, python = _MEMBER_CLASSES[name]
    missing = sorted(
        m
        for m in _unmodelled_members(name)
        if not_modelled_member(python, m, name) is None
    )
    # Python's own name says where it comes from: an openodd upgrade adds
    # members to openodd.model.OddModule, the package's OddModule (#298).
    assert not missing, (
        f"{name} ({python.__module__}.{python.__qualname__}): public in "
        f"Python, neither modelled nor in NOT_MODELLED_MEMBERS: {missing}"
    )


def test_a_listed_member_is_still_public_and_unmodelled() -> None:
    # Each entry of NOT_MODELLED_MEMBERS excuses a member some compared class
    # has in Python and lacks in the model: one that is modelled now, or no
    # longer public, is stale.
    used: set[tuple[str, str]] = set()
    for name, (_module, python) in _MEMBER_CLASSES.items():
        for member in _unmodelled_members(name):
            excused = not_modelled_member(python, member, name)
            if excused is not None:
                used.add((excused[0], member))
        node = _definitions(_MEMBER_CLASSES[name][0])[name][0]
        assert isinstance(node, ast.ClassDef)
        modelled_and_listed = sorted(
            m
            for m in _python_members(python) & _model_members(node)
            if not_modelled_member(python, m, name) is not None
        )
        assert (
            not modelled_and_listed
        ), f"{name}: modelled, and in NOT_MODELLED_MEMBERS: {modelled_and_listed}"
    listed = {
        (owner, member)
        for owner, members in NOT_MODELLED_MEMBERS.items()
        for member in members
    }
    stale = sorted(
        (owner, member) for owner, member in listed - used if owner not in _UNIMPORTABLE
    )
    assert not stale, f"in NOT_MODELLED_MEMBERS, but excusing nothing: {stale}"
    assert all(
        reason
        for members in NOT_MODELLED_MEMBERS.values()
        for reason in members.values()
    ), "a reason for each member"


# ---------------------------------------------------------------------------
# Every model file is compared with Python, or exempt with the reason (#240)
# ---------------------------------------------------------------------------

#: Model files (relative to _MODEL, without the suffix) that no table above
#: compares, each with the reason.
_UNCOMPARED_MODEL_FILES: dict[str, str] = {
    "_tc": "helpers of the model (Codon's missing typing), no Python module",
    "_lists": "helpers of the model (lists of conditions and actions), no Python module",
    "_reader": "helpers of the model (a measure's reader), no Python module; "
    "and Measure, which measures.codon re-exports: compared through that re-export",
    "_exceptions": "helpers of the model (the exceptions Codon 0.19 lacks), no Python module",
    "_poses": "shared by coordinate.codon, which re-exports it: compared through those re-exports",
    "_vehicle_entity": "shared by entity.codon, which re-exports it: compared through those re-exports",
    "_ground_projection": "shared by coordinate.codon, which re-exports it: compared through those re-exports",
    "ros2/cyclone": "compared by test_the_cyclone_model_is_the_transport_ros2_entity_builds, where cyclonedds is installed",
    **{
        f"{package}/__init__": "a checked package's __init__: models nothing, so a scenario finds none of its names"
        for package in (
            "autoware_bridge",
            "autoware_stack",
            "driver",
            "maps",
            "ros2",
            "ui",
            "utils",
        )
    },
}


def _compared_model_files() -> set[str]:
    return {
        "__init__",  # _EXPORTS is what it imports
        *_EXPORTS.values(),
        *(module for module, _python in _BOUNDARY.values()),
        *_MODULE_MODELS,
        *(module for module, _why in _DIFFERING.values()),
    }


def test_every_model_file_is_compared_or_exempt() -> None:
    # A model file no test compares drifts from Python unnoticed: what the
    # library check compiles against it would be checked against a stale
    # signature (maps/z_offset_cache.codon, #240).
    files = {
        path.relative_to(_MODEL).with_suffix("").as_posix()
        for path in _MODEL.rglob("*.codon")
    }
    compared = _compared_model_files()
    assert not sorted(
        files - compared - set(_UNCOMPARED_MODEL_FILES)
    ), "model files no test compares with Python: add them to a table or exempt them"
    assert not sorted(set(_UNCOMPARED_MODEL_FILES) - files), "exempt, but no such file"
    assert not sorted(
        set(_UNCOMPARED_MODEL_FILES) & compared
    ), "exempt, but compared: drop the exemption"


def test_every_model_file_parses_as_python() -> None:
    # The tests above read the model with ast; keep it valid Python syntax.
    for path in sorted(model_dir().rglob("*.codon")):
        _parse(path)


# ---------------------------------------------------------------------------
# pyxodr's RoadNetwork, as coordinate.codon declares how it is loaded
# ---------------------------------------------------------------------------

_XODR_PATH = Path(__file__).resolve().parents[3] / "data" / "nishishinjuku_carla.xodr"


def _road_network_model() -> ast.ClassDef:
    node = _definitions("coordinate")["RoadNetwork"][0]
    assert isinstance(node, ast.ClassDef)
    return node


@pytest.mark.parametrize("method", ["__init__", "get_roads"])
def test_the_road_network_model_takes_pyxodr_s_parameters(method: str) -> None:
    # pyxodr's own signature, but for the trailing parameters with a default
    # the framework leaves at pyxodr's (the lane types to ignore).
    model = _model_params(_methods(_road_network_model())[method], drop_first=True)
    python = _python_params(getattr(PyxodrRoadNetwork, method), drop_first=True)
    assert model == python[: len(model)], method
    assert all(has_default for _kind, _name, has_default in python[len(model) :])


def test_the_road_network_model_loads_the_fixture_map() -> None:
    # What MapManager.initialize() and road_network do with it.
    network = PyxodrRoadNetwork(str(_XODR_PATH))
    roads = network.get_roads()
    assert isinstance(roads, list) and roads
    assert all(isinstance(road, PyxodrRoad) for road in roads)
    assert isinstance(network.road_ids_to_object, dict)
    assert network.road_ids_to_object
    for road_id, road in network.road_ids_to_object.items():
        assert isinstance(road_id, str)
        assert isinstance(road, PyxodrRoad)


def test_the_lxml_model_is_the_element_pyxodr_holds() -> None:
    # codon/lxml/etree.codon names the XML element coordinate.codon declares
    # as lxml.etree._Element (coordinate.transform._cubic()): pyxodr's roads,
    # lane sections and lanes hold lxml elements, and every method the model
    # declares of one (and of its attributes) is lxml's.
    etree_model = _parse(model_dir() / "lxml" / "etree.codon")
    assert [ast.unparse(node) for node in etree_model.body] == [
        "from autoware_carla_scenario.coordinate import _XmlElement as _Element"
    ]
    element = _definitions("coordinate")["_XmlElement"][0]
    attributes = _definitions("coordinate")["_XmlAttributes"][0]
    assert isinstance(element, ast.ClassDef) and isinstance(attributes, ast.ClassDef)

    road = PyxodrRoadNetwork(str(_XODR_PATH)).get_roads()[0]
    section = road.lane_sections[0]
    xml = [road.road_xml, section.lane_section_xml, section.lanes[0].lane_xml]
    assert all(isinstance(each, lxml.etree._Element) for each in xml)
    fields = [
        stmt.target.id
        for stmt in element.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]
    assert fields == ["attrib"]
    for name in [*_methods(element), *fields]:
        assert hasattr(road.road_xml, name), name
    for name in _methods(attributes):
        assert hasattr(road.road_xml.attrib, name), name


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("scenario_base", "BaseScenario"),
        ("conditions", "BaseCondition"),
        ("actions", "BaseAction"),
        ("entity", "EgoVehicle"),
    ],
)
def test_a_base_class_checks_every_override_against_its_return_type(
    module: str, name: str
) -> None:
    # The scenario check binds the result of each method a class of the
    # scenario's modules overrides through `_acs_hook_returns` (#238): one
    # branch per method of the model class, with the method's own return
    # annotation (None for none), and a message naming both.
    cls = _definitions(module)[name][0]
    assert isinstance(cls, ast.ClassDef)
    expected: dict[str, tuple[str, str]] = {}
    for stmt in cls.body:
        if not isinstance(stmt, ast.FunctionDef) or stmt.name.startswith(
            ("__", "_acs_")
        ):
            continue
        decorators = _decorators(stmt)
        if any(d.endswith(".setter") for d in decorators):
            continue
        returns = "None" if stmt.returns is None else ast.unparse(stmt.returns)
        shown = f"{name}.{stmt.name}" + ("" if "property" in decorators else "()")
        expected.setdefault(
            stmt.name,
            (
                "NoneType" if returns == "None" else returns,
                f"{shown} returns {returns}",
            ),
        )
    hooks = [
        s
        for s in cls.body
        if isinstance(s, ast.FunctionDef) and s.name == "_acs_hook_returns"
    ]
    assert len(hooks) == 1, f"{name} declares no _acs_hook_returns"
    declared: dict[str, tuple[str, str]] = {}
    for node in ast.walk(hooks[0]):
        if isinstance(node, ast.If):
            test = node.test
            assert isinstance(test, ast.Compare) and isinstance(
                test.comparators[0], ast.Constant
            )
            call = node.body[0]
            assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
            _, type_, message = call.value.args
            hook, text = test.comparators[0].value, message
            assert isinstance(hook, str)
            assert isinstance(text, ast.Constant) and isinstance(text.value, str)
            declared[hook] = (ast.unparse(type_), text.value)
    assert declared == expected


@pytest.mark.parametrize(
    ("path", "python"),
    [
        ("_poses.codon", acs.OpenDrivePose),
        ("autoware_stack/devcontainer.codon", stack_devcontainer.DevContainer),
    ],
)
def test_the_replace_methods_of_a_model_match_the_dataclass_fields(
    path: str, python: Any
) -> None:
    # dataclasses.replace(obj, f=x) is lowered to obj._acs_with__f(x)
    # (#487). The model of a framework dataclass spells those methods by
    # hand: a field added to the Python class and not to them would make the
    # check refuse a replace() the run accepts.
    cls = next(
        n
        for n in _parse(_MODEL / path).body
        if isinstance(n, ast.ClassDef) and n.name == python.__name__
    )
    methods = {
        s.name: s
        for s in cls.body
        if isinstance(s, ast.FunctionDef) and s.name.startswith("_acs_with__")
    }
    fields = [f for f in dataclasses.fields(python) if f.init]
    assert sorted(methods) == sorted(f"_acs_with__{f.name}" for f in fields)
    for f in fields:
        method = methods[f"_acs_with__{f.name}"]
        assert [a.arg for a in method.args.args] == ["self", f.name]
        # It rebuilds the object from every field, in order.
        returned = method.body[-1]
        assert isinstance(returned, ast.Return)
        assert isinstance(returned.value, ast.Call)
        built = [ast.unparse(a) for a in returned.value.args]
        assert built == [
            f.name if g.name == f.name else f"self.{g.name}" for g in fields
        ]


# ---------------------------------------------------------------------------
# The logging model against CPython's logging (#496)
# ---------------------------------------------------------------------------

#: The public methods of logging.Logger the model leaves out: a scenario does
#: not build log records or walk the logger tree.
_LOGGER_NOT_MODELLED = {
    "callHandlers",
    "findCaller",
    "getChildren",
    "handle",
    "makeRecord",
    "manager",
    "root",
}

#: logging.__all__ names the model leaves out: records, filters, adapters and
#: the global configuration of the logging module.
_LOGGING_NOT_MODELLED = {
    "BufferingFormatter",
    "Filter",
    "LogRecord",
    "LoggerAdapter",
    "addLevelName",
    "captureWarnings",
    "disable",
    "getHandlerByName",
    "getHandlerNames",
    "getLevelNamesMapping",
    "getLogRecordFactory",
    "getLoggerClass",
    "lastResort",
    "makeLogRecord",
    "raiseExceptions",
    "setLogRecordFactory",
    "setLoggerClass",
    "shutdown",
}


def _codon_logging() -> ast.Module:
    return _parse(model_dir() / "logging.codon")


def _declared(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in getattr(node, "body", []):
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)):
            names.add(item.name)
        elif isinstance(item, ast.Assign):
            names |= {t.id for t in item.targets if isinstance(t, ast.Name)}
    return names


def test_the_logging_model_declares_the_loggers_methods() -> None:
    import logging

    (logger,) = [
        n
        for n in _codon_logging().body
        if isinstance(n, ast.ClassDef) and n.name == "Logger"
    ]
    modelled = {n for n in _declared(logger) if not n.startswith("_")}
    methods = {
        n
        for n, v in vars(logging.Logger).items()
        if not n.startswith("_") and callable(v)
    }
    assert methods - modelled <= _LOGGER_NOT_MODELLED, "public, not modelled"
    assert not modelled & _LOGGER_NOT_MODELLED, "listed, but modelled"
    # The model's methods are CPython's (the instance attributes are Python's).
    assert modelled - {"name", "level", "propagate", "disabled"} <= set(
        dir(logging.Logger)
    )


def test_the_logging_model_declares_the_modules_names() -> None:
    import logging

    modelled = {n for n in _declared(_codon_logging()) if not n.startswith("_")}
    exported = set(logging.__all__)
    assert exported - modelled <= _LOGGING_NOT_MODELLED, "exported, not modelled"
    assert not modelled & _LOGGING_NOT_MODELLED, "listed, but modelled"
    assert (modelled - {"BASIC_FORMAT"}) - exported == set(), "not in logging"


def test_the_logging_model_holds_the_levels_and_keywords_of_python() -> None:
    import logging

    tree = _codon_logging()
    levels = {
        t.id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
        for t in n.targets
        if isinstance(t, ast.Name) and t.id.isupper() and t.id != "BASIC_FORMAT"
    }
    assert levels == {
        name: getattr(logging, name)
        for name in ("NOTSET DEBUG INFO WARNING WARN ERROR CRITICAL FATAL".split())
    }
    assert (
        ast.literal_eval(
            [
                n.value
                for n in tree.body
                if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "BASIC_FORMAT" for t in n.targets)
            ][0]
        )
        == logging.BASIC_FORMAT
    )
    # Every level method takes the keywords CPython's _log() does.
    keywords = {"exc_info", "stack_info", "stacklevel", "extra"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    functions += [
        m
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "Logger"
        for m in c.body
        if isinstance(m, ast.FunctionDef)
    ]
    level_functions = [
        f
        for f in functions
        if f.name
        in {"debug", "info", "warning", "warn", "error", "exception", "critical"}
        | {"fatal", "log"}
    ]
    assert len(level_functions) == 18
    for function in level_functions:
        assert {a.arg for a in function.args.kwonlyargs} == keywords, function.name


def test_no_model_file_defines_a_top_level_function_twice() -> None:
    # Codon 0.19 tries only the last of two typed overloads, so a second
    # `def f(x: A)` after `def f(x: B)` refuses a B that Python accepts (#536,
    # as to_carla_world had before). One untyped definition with an
    # isinstance() check does what both are meant to; an @overload set is
    # Codon's own dispatch and is left alone.
    duplicated: list[str] = []
    for path in sorted(model_dir().rglob("*.codon")):
        seen: set[str] = set()
        for node in _parse(path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Codon's own @overload dispatches on the arguments.
                if any(ast.unparse(d) == "overload" for d in node.decorator_list):
                    continue
                if node.name in seen:
                    duplicated.append(f"{path.relative_to(model_dir())}: {node.name}")
                seen.add(node.name)
    assert not duplicated, duplicated
