"""Which framework modules the library check compiles (docs/typecheck.md).

The scenario check compiles a scenario against the hand-written model of the
framework (``codon/``).  The library check (:func:`.check.typecheck_library`,
``scenario-check --library``) compiles the framework's own source instead, a
module at a time, as each is made to compile.  This file says which:

* :data:`CHECKED`: modules compiled from their real source.  Every public
  function and every method of every public class is called with a value of
  each parameter's annotated type, so their bodies are checked.
* :data:`EXCLUDED`: every other module, with the reason it is not compiled.
  :data:`NOT_YET_CHECKED` marks one that could be: moving it into
  :data:`CHECKED` is a matter of making its source compile.  "imports X"
  names what keeps a module out: a package or a standard module Codon cannot
  compile.

Every module of the package except ``typecheck/`` is in exactly one of the
two (``test_typecheck_library.py``), so a new module needs an entry here.

:data:`UNCALLED` names the few public functions and methods of a checked
module the check leaves out, each with its reason: one whose job Codon cannot
express (a dict of mixed value types, ``json``), in a module that otherwise
compiles.

To move a module into :data:`CHECKED`, delete its :data:`EXCLUDED` entry, add
it to :data:`CHECKED`, run ``uv run scenario-check --library`` and fix what
that reports, in the module itself: annotations and Codon-friendly rewrites
only, so the module behaves exactly as before.  A framework module a checked
module imports that is not checked itself stands in through the model
(``codon/``); a type the checked module defines is not the same type as its
twin in the model, so move modules bottom-up, before the modules that use
them.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "CHECKED",
    "EXCLUDED",
    "NOT_YET_CHECKED",
    "REPLACED_MODELS",
    "UNCALLED",
    "package_modules",
]

#: The reason of a module nobody has made compile yet.
NOT_YET_CHECKED = "not yet checked (#45)"
_NOT_YET = NOT_YET_CHECKED

#: Modules compiled from their real source, by dotted name (a package by its
#: own name, for its ``__init__.py``).
CHECKED: tuple[str, ...] = (
    "autoware_carla_scenario.action_state",
    "autoware_carla_scenario.actions",
    "autoware_carla_scenario.actions._departures",
    "autoware_carla_scenario.actions.base",
    "autoware_carla_scenario.actions.environment",
    "autoware_carla_scenario.actions.lane_change",
    "autoware_carla_scenario.actions.set_speed",
    "autoware_carla_scenario.actions.traffic_signal",
    "autoware_carla_scenario.actions.turn",
    "autoware_carla_scenario.actions.walk_straight",
    "autoware_carla_scenario.autoware_bridge._proto",
    "autoware_carla_scenario.autoware_bridge.base",
    "autoware_carla_scenario.autoware_bridge.ros_bridge",
    "autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api",
    "autoware_carla_scenario.conditions.action_state",
    "autoware_carla_scenario.conditions.always_true",
    "autoware_carla_scenario.conditions.and_condition",
    "autoware_carla_scenario.conditions.base",
    "autoware_carla_scenario.conditions.collision",
    "autoware_carla_scenario.conditions.comparison",
    "autoware_carla_scenario.conditions.composition.acceleration",
    "autoware_carla_scenario.conditions.composition.base",
    "autoware_carla_scenario.conditions.composition.distance_measure",
    "autoware_carla_scenario.conditions.composition.entity_distance",
    "autoware_carla_scenario.conditions.composition.entity_lane_position",
    "autoware_carla_scenario.conditions.composition.entity_position_distance",
    "autoware_carla_scenario.conditions.composition.relative_speed",
    "autoware_carla_scenario.conditions.composition.speed",
    "autoware_carla_scenario.conditions.composition.standstill",
    "autoware_carla_scenario.conditions.composition.time_headway",
    "autoware_carla_scenario.conditions.composition.time_to_collision",
    "autoware_carla_scenario.conditions.composition.waypoint",
    "autoware_carla_scenario.conditions.elapsed_time",
    "autoware_carla_scenario.conditions.entity_existence",
    "autoware_carla_scenario.conditions.lane_change_settled",
    "autoware_carla_scenario.conditions.not_condition",
    "autoware_carla_scenario.conditions.or_condition",
    "autoware_carla_scenario.conditions.persistent",
    "autoware_carla_scenario.conditions.route_progress",
    "autoware_carla_scenario.conditions.sticky",
    "autoware_carla_scenario.conditions.timeout",
    "autoware_carla_scenario.conditions.traffic_signal",
    "autoware_carla_scenario.conditions.traffic_signal_controller",
    "autoware_carla_scenario.conditions.trajectory_time",
    "autoware_carla_scenario.constants",
    "autoware_carla_scenario.coordinate",
    "autoware_carla_scenario.coordinate.frames",
    "autoware_carla_scenario.coordinate.lane_distance",
    "autoware_carla_scenario.coordinate.poses",
    "autoware_carla_scenario.coordinate.snap",
    "autoware_carla_scenario.coordinate.traffic_light",
    "autoware_carla_scenario.entity._spawn",
    "autoware_carla_scenario.entity.ego",
    "autoware_carla_scenario.entity.pedestrian_entity",
    "autoware_carla_scenario.entity.registry",
    "autoware_carla_scenario.entity.vehicle_entity",
    "autoware_carla_scenario.entity_role",
    "autoware_carla_scenario.examples.conf",
    "autoware_carla_scenario.kinematics",
    "autoware_carla_scenario.kinematics.acceleration",
    "autoware_carla_scenario.kinematics.angle",
    "autoware_carla_scenario.kinematics.frames",
    "autoware_carla_scenario.kinematics.vector",
    "autoware_carla_scenario.tools",
    "autoware_carla_scenario.kinematics.velocity",
    "autoware_carla_scenario.measures",
    "autoware_carla_scenario.odd.scenario_measure",
    "autoware_carla_scenario.odd.units",
    "autoware_carla_scenario.traffic.sumo.geometry",
    "autoware_carla_scenario.world_reset",
    "autoware_carla_scenario.traffic.sumo.physics_control",
    "autoware_carla_scenario.trajectory",
    "autoware_carla_scenario.utils",
    "autoware_carla_scenario.utils.opendrive",
    "autoware_carla_scenario.utils.vehicles",
)

#: The reason of a method that builds a dict of mixed value types.
_DETAILS = "builds a dict[str, Any] of mixed value types, which Codon cannot express"

#: Public functions and methods of checked modules the check does not call,
#: by dotted name (``module.Class.method``), with the reason.  Codon compiles
#: a function only when something calls it, so the body of one listed here is
#: not checked; nor are the imports made inside it, which is where a module
#: imports what Codon has nothing for (``json``) and only that function uses.
#: Everything else in the module still is.
UNCALLED: dict[str, str] = {
    "autoware_carla_scenario.conditions.action_state.ActionStateCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.and_condition.AndCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.base.BaseCondition.__init_subclass__": "wraps check() when a subclass is created (cls.__dict__, functools.wraps), which Codon has no hook for",
    "autoware_carla_scenario.conditions.base.BaseCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.base.BaseCondition.to_summary_dict": _DETAILS,
    "autoware_carla_scenario.conditions.base.ConditionStatus.__init__": "takes `details: dict[str, Any]`, a dict of mixed value types, which Codon cannot express",
    "autoware_carla_scenario.conditions.base.ConditionStatus.to_dict": _DETAILS,
    "autoware_carla_scenario.conditions.base.ScenarioResult.to_dict": _DETAILS,
    "autoware_carla_scenario.conditions.base.ScenarioResult.to_json": "serialises with json, which Codon does not have",
    "autoware_carla_scenario.conditions.comparison.ScalarComparisonRule.to_dict": _DETAILS,
    "autoware_carla_scenario.conditions.collision.CollisionCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.acceleration.AccelerationCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.base.CompositionCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.entity_distance.EntityDistanceCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.entity_lane_position.EntityLaneOfCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.entity_lane_position.EntityLanePositionCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.entity_position_distance.EntityPositionDistanceCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.relative_speed.RelativeSpeedCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.speed.SpeedCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.standstill.StandstillCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.time_headway.TimeHeadwayCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.time_to_collision.TimeToCollisionCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.waypoint.WaypointCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.elapsed_time.ElapsedTimeCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.lane_change_settled.LaneChangeSettledCondition.get_details": _DETAILS,
    "autoware_carla_scenario.entity.registry.find_entity_by_role_name": "returns the entity of any kind registered under a role (`Optional[Any]`), which Codon cannot express; find_vehicle_entity and find_pedestrian_entity are its typed lookups",
    "autoware_carla_scenario.entity.registry.register_entity": "takes an entity of any kind (`Any`: a vehicle, a pedestrian or a duck-typed object); a checked caller compiles it with the entity it passes",
    "autoware_carla_scenario.conditions.not_condition.NotCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.or_condition.OrCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.persistent.PersistentCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.route_progress.RouteProgressCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.sticky.StickyCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.traffic_signal.TrafficSignalCondition.get_details": _DETAILS,
}

#: Every other module, with the reason it is not compiled.
EXCLUDED: dict[str, str] = {
    "autoware_carla_scenario": "loads its public names lazily with importlib (a PEP 562 `__getattr__` returning `object`), and Codon reads its `TYPE_CHECKING` re-exports as imports of every subpackage, numpy, grpc and pytest ones included",
    "autoware_carla_scenario.actions.background_traffic": "imports yaml",
    "autoware_carla_scenario.actions.follow_trajectory": "imports carla_driver_interface",
    "autoware_carla_scenario.actions.routing": "holds its waypoints as a variable-length tuple, which its public `waypoints` property returns and which it passes on as `**extra` only when there are some, and its public `entity_name` returns the `Union[EntityRole, str]` it was given, an attribute Codon 0.19 cannot hold; it routes whatever entity has the role (find_entity_by_role_name, `Optional[Any]`), since entity.registry has no typed lookup for an entity with route_to",
    "autoware_carla_scenario.actions.traffic_signal_controller": "imports signals.registry, which is neither checked nor modelled",
    "autoware_carla_scenario.authoring": "re-exports authoring.models (pydantic), authoring.persistence (yaml), authoring.package_export and authoring.wheelhouse (subprocess, shutil)",
    "autoware_carla_scenario.authoring._builders_generated": "its builders take authoring.compiler's CompiledCondition, CompiledAction and BuildContext, which reach pydantic models through authoring.models",
    "autoware_carla_scenario.authoring.builders": "dispatches to builders by name through vars() and globals() as Callable[..., Any]; its parameters reach pydantic models through authoring.compiler",
    "autoware_carla_scenario.authoring.codegen": "imports pathlib, json, argparse, importlib, inspect, difflib",
    "autoware_carla_scenario.authoring.compiler": "reaches pydantic models through authoring.models",
    "autoware_carla_scenario.authoring.framework_pin": "imports pathlib, subprocess, shutil, importlib",
    "autoware_carla_scenario.authoring.hydra_config": "reaches pydantic models through authoring.models; imports authoring.persistence (yaml)",
    "autoware_carla_scenario.authoring.models": "imports pydantic, uuid",
    "autoware_carla_scenario.authoring.package_export": "imports pathlib, subprocess, shutil, tempfile, importlib, platform",
    "autoware_carla_scenario.authoring.persistence": "imports pathlib, yaml",
    "autoware_carla_scenario.authoring.registry": "FieldSpec.default is Any and holds str, int, float, bool, list or None across the spec table, which Codon has no type for; rule_symbol takes object",
    "autoware_carla_scenario.authoring.starter": "reaches pydantic models through authoring.models",
    "autoware_carla_scenario.authoring.uv_tool": "imports pathlib, subprocess, shutil",
    "autoware_carla_scenario.authoring.validator": "reaches pydantic models through authoring.models",
    "autoware_carla_scenario.authoring.wheelhouse": "imports pathlib, jinja2, packaging, tomllib, tomli, subprocess, shutil, tempfile, platform",
    "autoware_carla_scenario.autoware_bridge": "re-exports autoware_bridge.grpc_server (grpc, concurrent)",
    "autoware_carla_scenario.autoware_bridge._proto.autoware_bridge_pb2": "imports google",
    "autoware_carla_scenario.autoware_bridge._proto.autoware_bridge_pb2_grpc": "imports grpc",
    "autoware_carla_scenario.autoware_bridge.fake": "overrides AutowareBridge.configure, whose Sequence[BridgePose] parameter is generic in Codon, and Codon 0.19 cannot call an overridden method with a generic parameter",
    "autoware_carla_scenario.autoware_bridge.grpc_server": "imports grpc, concurrent",
    "autoware_carla_scenario.autoware_bridge.ros_bridge.__main__": "runs the scenario_bridge node (rclpy)",
    "autoware_carla_scenario.autoware_bridge.ros_bridge.client": "imports grpc",
    "autoware_carla_scenario.autoware_bridge.ros_bridge.node": "imports rclpy, autoware_adapi_v1_msgs, geometry_msgs, grpc",
    "autoware_carla_scenario.autoware_stack": "re-exports autoware_stack.devcontainer (yaml, json), autoware_stack.docker and autoware_stack.launcher (subprocess)",
    "autoware_carla_scenario.autoware_stack.bridge_node": "imports pathlib, hashlib, shlex",
    "autoware_carla_scenario.autoware_stack.devcontainer": "imports pathlib, yaml, json",
    "autoware_carla_scenario.autoware_stack.docker": "imports pathlib, subprocess, shlex",
    "autoware_carla_scenario.autoware_stack.launcher": "imports pathlib, subprocess, shutil, signal",
    "autoware_carla_scenario.camera_recorder": "imports numpy, pathlib, ffmpeg, subprocess, queue",
    "autoware_carla_scenario.carla_install": "imports pathlib, tqdm, shutil, json, argparse, urllib, http, email, tarfile, platform",
    "autoware_carla_scenario.conditions": "re-exports conditions.composition, whose temporary_stop imports numpy",
    "autoware_carla_scenario.conditions.composition": "re-exports composition.temporary_stop (numpy)",
    "autoware_carla_scenario.conditions.composition.temporary_stop": "imports numpy, to sum the length of pyxodr's reference line (a numpy array the model does not have)",
    "autoware_carla_scenario.coordinate.map_manager": "imports numpy, lanelet2, pyxodr, pathlib",
    "autoware_carla_scenario.coordinate.projection": "imports lanelet2, autoware_lanelet2_extension_python, pathlib, yaml",
    "autoware_carla_scenario.coordinate.stop_line": "imports lanelet2",
    "autoware_carla_scenario.coordinate.transform": "imports numpy, lanelet2, autoware_lanelet2_extension_python",
    "autoware_carla_scenario.coverage": "re-exports coverage.collector and coverage.report, which import pathlib, json",
    "autoware_carla_scenario.coverage.cod": "imports pathlib, yaml, csv",
    "autoware_carla_scenario.coverage.collector": "imports pathlib, json",
    "autoware_carla_scenario.coverage.items": (
        "samples values of any type (value_label, bucket_of and a cover "
        "expression take or return Any: an enum, a bool, a number or a label), "
        "and keeps an event that is a SamplingEvent or a condition, a Union "
        "field Codon 0.19 cannot hold"
    ),
    "autoware_carla_scenario.coverage.report": "imports pathlib, json, argparse",
    "autoware_carla_scenario.declarative": "imports pathlib",
    "autoware_carla_scenario.driver": "re-exports driver.base and driver.control (numpy), driver.egodriver_client (grpc)",
    "autoware_carla_scenario.driver.base": "imports numpy",
    "autoware_carla_scenario.driver.control": "imports numpy",
    "autoware_carla_scenario.driver.egodriver_client": "imports numpy, grpc, google, carla_driver_interface, atexit",
    "autoware_carla_scenario.driver.geometry": "imports carla_driver_interface",
    "autoware_carla_scenario.driver.hdmap_export": "imports pathlib, roadgen, shutil, tempfile, json, hashlib",
    "autoware_carla_scenario.driver.observation": "imports numpy, cv2",
    "autoware_carla_scenario.driver.renderer": "imports numpy, carla_driver_interface",
    "autoware_carla_scenario.editor": "imports pathlib, uvicorn",
    "autoware_carla_scenario.editor.app": "imports pathlib, fastapi",
    "autoware_carla_scenario.editor.forms": "imports authoring.registry, which is not checked and has no model; parses form values typed Any",
    "autoware_carla_scenario.editor.map_preview": "imports pathlib, json",
    "autoware_carla_scenario.editor.routes": "imports fastapi, starlette, anyio",
    "autoware_carla_scenario.editor.scenario_map": "reaches pydantic models through authoring.models; imports editor.map_preview (pathlib, json)",
    "autoware_carla_scenario.editor.service": "imports pathlib, yaml, shutil, tempfile, zipfile, contextlib",
    "autoware_carla_scenario.entity": "re-exports entity.carla_driver_entity (carla_driver_interface, uuid) and entity.autoware_entity, which are not checked and which the model does not declare",
    "autoware_carla_scenario.entity._policy_warmup": "imports utils.powertrain (numpy)",
    "autoware_carla_scenario.entity.autoware_entity": "imports autoware_stack.launcher (subprocess) and driver.observation (numpy), which are neither checked nor modelled",
    "autoware_carla_scenario.entity.carla_driver_entity": "imports carla_driver_interface, uuid",
    "autoware_carla_scenario.examples": "re-exports examples.run (pathlib, omegaconf, hydra, json) and the registry (pathlib, omegaconf, importlib)",
    "autoware_carla_scenario.examples.configs": "ScenarioRunConfig.scenario, the scenario group of the Hydra root config, is a union of the six scenario configs, which Codon cannot express (and Codon 0.19 crashes on a union-typed attribute); Hydra reads the union, so it stays",
    "autoware_carla_scenario.examples.cut_in": "imports examples.configs, which cannot be checked (its own reason) and has no model; the scenario check compiles it as a scenario, against the model",
    "autoware_carla_scenario.examples.expand": "imports json, sys, the registry (pathlib, omegaconf, importlib) and sweeper.expand (omegaconf)",
    "autoware_carla_scenario.examples.intersection_passing": "imports examples.configs, which cannot be checked (its own reason) and has no model; the scenario check compiles it as a scenario, against the model",
    "autoware_carla_scenario.examples.lane_change": "imports examples.configs, which cannot be checked (its own reason) and has no model; the scenario check compiles it as a scenario, against the model",
    "autoware_carla_scenario.examples.pedestrian_dart_out": "imports examples.configs, which cannot be checked (its own reason) and has no model; the scenario check compiles it as a scenario, against the model",
    "autoware_carla_scenario.examples.run": "imports pathlib, omegaconf, hydra, json",
    "autoware_carla_scenario.examples.temporary_stop": "imports examples.configs, which cannot be checked (its own reason) and has no model; the scenario check compiles it as a scenario, against the model",
    "autoware_carla_scenario.examples.traffic_light_compliance": "imports examples.configs, which cannot be checked (its own reason) and has no model; the scenario check compiles it as a scenario, against the model",
    "autoware_carla_scenario.maps": "re-exports maps.cache (subprocess, json), maps.config (pathlib) and maps.source (urllib)",
    "autoware_carla_scenario.maps.cache": "imports pathlib, subprocess, shutil, tempfile, json, hashlib",
    "autoware_carla_scenario.maps.catalogue": "imports maps.cache (subprocess, json), maps.resolver (pathlib, hashlib) and maps.source (urllib), which are not checked and have no model",
    "autoware_carla_scenario.maps.config": "imports pathlib",
    "autoware_carla_scenario.maps.opendrive": "imports pathlib, shutil",
    "autoware_carla_scenario.maps.resolver": "imports pathlib, hashlib",
    "autoware_carla_scenario.maps.source": "imports urllib",
    "autoware_carla_scenario.odd": (
        "re-exports odd.probes and odd.route (lanelet2), odd.openodd (pathlib, "
        "yaml), odd.sources, odd.sampler and odd.registry (pathlib, importlib)"
    ),
    "autoware_carla_scenario.odd.cli": "imports json, argparse",
    "autoware_carla_scenario.odd.model": (
        "evaluates attribute values of any type (Mapping[str, Any] of numbers, "
        "labels, None, UNDECIDED and buckets; three-valued verdicts mixed with "
        'the "inactive" label), and builds on coverage.items, which is not checked'
    ),
    "autoware_carla_scenario.odd.openodd": "imports pathlib, yaml",
    "autoware_carla_scenario.odd.probes": "imports lanelet2",
    "autoware_carla_scenario.odd.registry": "imports pathlib, importlib",
    "autoware_carla_scenario.odd.route": "imports lanelet2",
    "autoware_carla_scenario.odd.sampler": "imports pathlib",
    "autoware_carla_scenario.odd.sources": "imports pathlib, subprocess, shutil, tempfile, tarfile, hashlib, fcntl, io, contextlib",
    "autoware_carla_scenario.pytest_fixtures": "imports pytest",
    "autoware_carla_scenario.registry": "imports pathlib, omegaconf, importlib",
    "autoware_carla_scenario.route": "imports route.model, which is neither checked nor modelled",
    "autoware_carla_scenario.route.active": "imports route.model and route.frame, which are neither checked nor modelled, and takes lanelet2's map and routing graph (`Any`)",
    "autoware_carla_scenario.route.frame": "imports route.geometry and trajectory.relative_lane, which use lanelet2, and holds lanelet2's map, routing graph and lanelets (`Any`)",
    "autoware_carla_scenario.route.geometry": "imports lanelet2, autoware_lanelet2_extension_python",
    "autoware_carla_scenario.route.model": "holds variable-length tuples (`tuple[int, ...]`), which Codon cannot express, and parses YAML mappings of `Any` (`Range.parse`, `RouteMatch.from_config`) into `dict[str, Any]`",
    "autoware_carla_scenario.route.positions": "imports route.frame and trajectory.relative_lane, which use lanelet2",
    "autoware_carla_scenario.route.search": "imports route.geometry (lanelet2) and sweeper.constraints, which are neither checked nor modelled, and walks lanelet2's lanelets (`Any`)",
    "autoware_carla_scenario.scaffold": "re-exports scaffold.generator (pathlib, argparse)",
    "autoware_carla_scenario.scaffold.generator": "imports pathlib, argparse, keyword",
    "autoware_carla_scenario.scenario_base": "takes the ego's class (`type[EgoVehicle]`), callbacks that are an action or a function (`Union[BaseAction, Callable]`), and measure and cover expressions of any value (`Callable[[carla.World], Any]`, `Iterable[Any]`, `Callable[[Any], bool]`), none of which Codon can express; imports entity.registry (entities as `Any`) and coverage.items, which are not checked, and traffic.base (pathlib), which the model does not declare",
    "autoware_carla_scenario.scenario_config": "imports omegaconf",
    "autoware_carla_scenario.scenario_queue": "imports pathlib, tqdm, pytest",
    "autoware_carla_scenario.scenario_runner": "imports pathlib, shutil",
    "autoware_carla_scenario.sensor": "re-exports sensor.base, sensor.carla_camera and sensor.carla_lidar (numpy)",
    "autoware_carla_scenario.sensor.base": "imports numpy",
    "autoware_carla_scenario.sensor.carla_camera": "imports numpy, queue",
    "autoware_carla_scenario.sensor.carla_lidar": "imports numpy, queue",
    "autoware_carla_scenario.server": "imports pathlib, dotenv, subprocess, signal, atexit",
    "autoware_carla_scenario.signals": "imports signals.controller, which is neither checked nor modelled",
    "autoware_carla_scenario.signals.controller": "holds variable-length tuples (`tuple[Phase, ...]`), which Codon cannot express, and build_controllers reads the authoring package's pydantic specs, typed `list[Any]`",
    "autoware_carla_scenario.signals.registry": "imports signals.controller, which is neither checked nor modelled, and returns a variable-length tuple",
    "autoware_carla_scenario.sweeper": "re-exports sweeper.bindings and sweeper.constraints (lanelet2), sweeper.expand (omegaconf)",
    "autoware_carla_scenario.sweeper.bindings": "imports lanelet2",
    "autoware_carla_scenario.sweeper.constraints": "imports lanelet2",
    "autoware_carla_scenario.sweeper.expand": "imports omegaconf",
    "autoware_carla_scenario.sweeper.lanelet_constraint_sweeper": "imports pathlib, omegaconf, hydra, tqdm, json, signal",
    "autoware_carla_scenario.sweeper.map_loader": "imports lanelet2, pathlib",
    "autoware_carla_scenario.templating": "imports pathlib, jinja2",
    "autoware_carla_scenario.tools.detect_no_3d_model_lanelets": "imports lanelet2, pathlib, omegaconf, tqdm, shutil, argparse",
    "autoware_carla_scenario.traffic": "imports traffic.base, which is neither checked nor modelled, and its backend factories take `Mapping[str, Any]` options",
    "autoware_carla_scenario.traffic.base": "imports pathlib",
    "autoware_carla_scenario.traffic.config": "imports utils.config, which is neither checked nor modelled, and holds a backend's options as `dict[str, Any]`",
    "autoware_carla_scenario.traffic.driven": "imports traffic.base, which is neither checked nor modelled, and holds the map a lane change was read from as `Any`",
    "autoware_carla_scenario.traffic.registry": "imports traffic.base, which is neither checked nor modelled, and registry (pathlib, omegaconf, importlib); a backend factory is a `Callable`",
    "autoware_carla_scenario.traffic.sumo": "imports traffic.sumo.config (pathlib), which is neither checked nor modelled",
    "autoware_carla_scenario.traffic.sumo.backend": "imports pathlib, traci, sumolib, libsumo",
    "autoware_carla_scenario.traffic.sumo.config": "imports pathlib",
    "autoware_carla_scenario.traffic.sumo.network": "imports pathlib, roadgen, sumo, subprocess, shutil, tempfile, importlib, hashlib",
    "autoware_carla_scenario.traffic.traffic_manager": "subclasses traffic.base.TrafficBackend, which is neither checked nor modelled, and drives duck-typed entities and CARLA handles (`Any`), read with getattr defaults",
    "autoware_carla_scenario.trajectory.authoring": "parses document values of `Any` (YAML rows, editor fields) into `dict[str, Any]`, which Codon cannot express",
    "autoware_carla_scenario.trajectory.model": "imports inspect (TrajectoryVertex.__init__.__signature__); TrajectoryVertex holds a position of any of ten pose types and takes `*args: Any, **kwargs: Any`, and RelativeLanePose.entity_ref is a union-typed field (EntityRole or str), which crashes Codon 0.19",
    "autoware_carla_scenario.trajectory.relative_lane": "imports lanelet2, autoware_lanelet2_extension_python",
    "autoware_carla_scenario.trajectory.resolve": "imports route.active and route.positions, which are neither checked nor modelled, and takes and returns callables (`PositionResolver`, `GroundHeight`, `ReferencePose`)",
    "autoware_carla_scenario.trajectory_recorder": "imports pathlib, json",
    "autoware_carla_scenario.ui": "imports pathlib, uvicorn",
    "autoware_carla_scenario.ui.app": "imports pathlib, fastapi, json, asyncio, fnmatch",
    "autoware_carla_scenario.ui.models": "imports pydantic",
    "autoware_carla_scenario.ui.runner": "imports pathlib, subprocess",
    "autoware_carla_scenario.ui.scanner": "imports pathlib, yaml, json",
    "autoware_carla_scenario.ui.sweep_resolver": "imports pathlib, omegaconf, hydra",
    "autoware_carla_scenario.utils.config": "reads the fields of a dataclass it is handed as a class (`type`, dataclasses.fields), which Codon cannot express",
    "autoware_carla_scenario.utils.powertrain": "imports numpy",
    "autoware_carla_scenario.utils.stop_line": "imports lanelet2",
    "autoware_carla_scenario.utils.traffic_light": "reads pyxodr's road network as an XML tree, and calls carla.TrafficLightState, which typesafe_carla's Codon library has as a value, not a type",
}

#: Model modules (``codon/autoware_carla_scenario/<name>.codon``) the library
#: check compiles from the checked source they model instead: once all of the
#: modules named are checked, the model module re-exports their definitions,
#: so every module of the check -- checked or model -- has the same types.
#: Without it, a checked module handing its own pose to a model function
#: (``to_opendrive``) would hand the wrong type.
REPLACED_MODELS: dict[str, tuple[str, ...]] = {
    "_poses": (
        "autoware_carla_scenario.coordinate.frames",
        "autoware_carla_scenario.coordinate.poses",
    ),
    "_vehicle_entity": (
        "autoware_carla_scenario.entity._spawn",
        "autoware_carla_scenario.entity.vehicle_entity",
    ),
}


def package_modules() -> dict[str, Path]:
    """Every module of the package but ``typecheck/``: dotted name -> source file."""
    root = Path(__file__).resolve().parent.parent
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[0] == "typecheck":
            continue
        if parts[-1] == "__init__":
            parts.pop()
        out[".".join([root.name, *parts])] = path
    return out
