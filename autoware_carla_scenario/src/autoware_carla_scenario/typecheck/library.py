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

__all__ = ["CHECKED", "EXCLUDED", "NOT_YET_CHECKED", "package_modules"]

#: The reason of a module nobody has made compile yet.
NOT_YET_CHECKED = "not yet checked (#45)"
_NOT_YET = NOT_YET_CHECKED

#: Modules compiled from their real source, by dotted name (a package by its
#: own name, for its ``__init__.py``).
CHECKED: tuple[str, ...] = (
    "autoware_carla_scenario.constants",
    "autoware_carla_scenario.coordinate.frames",
    "autoware_carla_scenario.coordinate.poses",
    "autoware_carla_scenario.entity_role",
    "autoware_carla_scenario.kinematics",
    "autoware_carla_scenario.kinematics.acceleration",
    "autoware_carla_scenario.kinematics.angle",
    "autoware_carla_scenario.kinematics.frames",
    "autoware_carla_scenario.kinematics.vector",
    "autoware_carla_scenario.kinematics.velocity",
    "autoware_carla_scenario.utils",
    "autoware_carla_scenario.utils.opendrive",
    "autoware_carla_scenario.utils.vehicles",
)

#: Every other module, with the reason it is not compiled.
EXCLUDED: dict[str, str] = {
    "autoware_carla_scenario": _NOT_YET,
    "autoware_carla_scenario.action_state": _NOT_YET,
    "autoware_carla_scenario.actions": _NOT_YET,
    "autoware_carla_scenario.actions._departures": _NOT_YET,
    "autoware_carla_scenario.actions.background_traffic": "imports yaml",
    "autoware_carla_scenario.actions.base": _NOT_YET,
    "autoware_carla_scenario.actions.environment": _NOT_YET,
    "autoware_carla_scenario.actions.follow_trajectory": "imports carla_driver_interface",
    "autoware_carla_scenario.actions.lane_change": _NOT_YET,
    "autoware_carla_scenario.actions.routing": _NOT_YET,
    "autoware_carla_scenario.actions.set_speed": _NOT_YET,
    "autoware_carla_scenario.actions.traffic_signal": _NOT_YET,
    "autoware_carla_scenario.actions.traffic_signal_controller": _NOT_YET,
    "autoware_carla_scenario.actions.turn": _NOT_YET,
    "autoware_carla_scenario.actions.walk_straight": _NOT_YET,
    "autoware_carla_scenario.authoring": _NOT_YET,
    "autoware_carla_scenario.authoring._builders_generated": _NOT_YET,
    "autoware_carla_scenario.authoring.builders": _NOT_YET,
    "autoware_carla_scenario.authoring.codegen": "imports pathlib, json, argparse, importlib, inspect, difflib",
    "autoware_carla_scenario.authoring.compiler": _NOT_YET,
    "autoware_carla_scenario.authoring.framework_pin": "imports pathlib, subprocess, shutil, importlib",
    "autoware_carla_scenario.authoring.hydra_config": _NOT_YET,
    "autoware_carla_scenario.authoring.models": "imports pydantic, uuid",
    "autoware_carla_scenario.authoring.package_export": "imports pathlib, subprocess, shutil, tempfile, importlib, platform",
    "autoware_carla_scenario.authoring.persistence": "imports pathlib, yaml",
    "autoware_carla_scenario.authoring.registry": _NOT_YET,
    "autoware_carla_scenario.authoring.starter": _NOT_YET,
    "autoware_carla_scenario.authoring.uv_tool": "imports pathlib, subprocess, shutil",
    "autoware_carla_scenario.authoring.validator": _NOT_YET,
    "autoware_carla_scenario.authoring.wheelhouse": "imports pathlib, jinja2, packaging, tomllib, tomli, subprocess, shutil, tempfile, platform",
    "autoware_carla_scenario.autoware_bridge": _NOT_YET,
    "autoware_carla_scenario.autoware_bridge._proto": _NOT_YET,
    "autoware_carla_scenario.autoware_bridge._proto.autoware_bridge_pb2": "imports google",
    "autoware_carla_scenario.autoware_bridge._proto.autoware_bridge_pb2_grpc": "imports grpc",
    "autoware_carla_scenario.autoware_bridge.base": _NOT_YET,
    "autoware_carla_scenario.autoware_bridge.fake": _NOT_YET,
    "autoware_carla_scenario.autoware_bridge.grpc_server": "imports grpc, concurrent",
    "autoware_carla_scenario.autoware_stack": _NOT_YET,
    "autoware_carla_scenario.autoware_stack.devcontainer": "imports pathlib, yaml, json",
    "autoware_carla_scenario.autoware_stack.docker": "imports pathlib, subprocess, shlex",
    "autoware_carla_scenario.autoware_stack.launcher": "imports pathlib, subprocess, shutil, signal",
    "autoware_carla_scenario.camera_recorder": "imports numpy, pathlib, ffmpeg, subprocess, queue",
    "autoware_carla_scenario.carla_install": "imports pathlib, tqdm, shutil, json, argparse, urllib, http, email, tarfile, platform",
    "autoware_carla_scenario.conditions": _NOT_YET,
    "autoware_carla_scenario.conditions.action_state": _NOT_YET,
    "autoware_carla_scenario.conditions.always_true": _NOT_YET,
    "autoware_carla_scenario.conditions.and_condition": _NOT_YET,
    "autoware_carla_scenario.conditions.base": _NOT_YET,
    "autoware_carla_scenario.conditions.collision": _NOT_YET,
    "autoware_carla_scenario.conditions.comparison": _NOT_YET,
    "autoware_carla_scenario.conditions.composition": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.acceleration": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.base": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.distance_measure": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.entity_distance": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.entity_lane_position": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.entity_position_distance": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.relative_speed": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.speed": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.standstill": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.temporary_stop": "imports numpy",
    "autoware_carla_scenario.conditions.composition.time_headway": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.time_to_collision": _NOT_YET,
    "autoware_carla_scenario.conditions.composition.waypoint": _NOT_YET,
    "autoware_carla_scenario.conditions.elapsed_time": _NOT_YET,
    "autoware_carla_scenario.conditions.entity_existence": _NOT_YET,
    "autoware_carla_scenario.conditions.lane_change_settled": _NOT_YET,
    "autoware_carla_scenario.conditions.not_condition": _NOT_YET,
    "autoware_carla_scenario.conditions.or_condition": _NOT_YET,
    "autoware_carla_scenario.conditions.persistent": _NOT_YET,
    "autoware_carla_scenario.conditions.route_progress": _NOT_YET,
    "autoware_carla_scenario.conditions.sticky": _NOT_YET,
    "autoware_carla_scenario.conditions.timeout": _NOT_YET,
    "autoware_carla_scenario.conditions.traffic_signal": _NOT_YET,
    "autoware_carla_scenario.conditions.traffic_signal_controller": _NOT_YET,
    "autoware_carla_scenario.conditions.trajectory_time": _NOT_YET,
    "autoware_carla_scenario.coordinate": _NOT_YET,
    "autoware_carla_scenario.coordinate.lane_distance": _NOT_YET,
    "autoware_carla_scenario.coordinate.map_manager": "imports numpy, lanelet2, pyxodr, pathlib",
    "autoware_carla_scenario.coordinate.projection": "imports lanelet2, autoware_lanelet2_extension_python, pathlib, yaml",
    "autoware_carla_scenario.coordinate.snap": _NOT_YET,
    "autoware_carla_scenario.coordinate.stop_line": "imports lanelet2",
    "autoware_carla_scenario.coordinate.traffic_light": "imports utils.traffic_light, which is neither checked nor modelled",
    "autoware_carla_scenario.coordinate.transform": "imports numpy, lanelet2, autoware_lanelet2_extension_python",
    "autoware_carla_scenario.coverage": _NOT_YET,
    "autoware_carla_scenario.coverage.cod": "imports pathlib, yaml, csv",
    "autoware_carla_scenario.coverage.collector": "imports pathlib, json",
    "autoware_carla_scenario.coverage.items": _NOT_YET,
    "autoware_carla_scenario.coverage.report": "imports pathlib, json, argparse",
    "autoware_carla_scenario.declarative": "imports pathlib",
    "autoware_carla_scenario.driver": _NOT_YET,
    "autoware_carla_scenario.driver.base": "imports numpy",
    "autoware_carla_scenario.driver.control": "imports numpy",
    "autoware_carla_scenario.driver.egodriver_client": "imports numpy, grpc, google, carla_driver_interface, atexit",
    "autoware_carla_scenario.driver.geometry": "imports carla_driver_interface",
    "autoware_carla_scenario.driver.hdmap_export": "imports pathlib, roadgen, shutil, tempfile, json, hashlib",
    "autoware_carla_scenario.driver.observation": "imports numpy, cv2",
    "autoware_carla_scenario.driver.renderer": "imports numpy, carla_driver_interface",
    "autoware_carla_scenario.editor": "imports pathlib, uvicorn",
    "autoware_carla_scenario.editor.app": "imports pathlib, fastapi",
    "autoware_carla_scenario.editor.forms": _NOT_YET,
    "autoware_carla_scenario.editor.map_preview": "imports pathlib, json",
    "autoware_carla_scenario.editor.routes": "imports fastapi, starlette, anyio",
    "autoware_carla_scenario.editor.scenario_map": _NOT_YET,
    "autoware_carla_scenario.editor.service": "imports pathlib, yaml, shutil, tempfile, zipfile, contextlib",
    "autoware_carla_scenario.entity": _NOT_YET,
    "autoware_carla_scenario.entity._policy_warmup": _NOT_YET,
    "autoware_carla_scenario.entity._spawn": _NOT_YET,
    "autoware_carla_scenario.entity.autoware_entity": _NOT_YET,
    "autoware_carla_scenario.entity.carla_driver_entity": "imports carla_driver_interface, uuid",
    "autoware_carla_scenario.entity.ego": _NOT_YET,
    "autoware_carla_scenario.entity.pedestrian_entity": _NOT_YET,
    "autoware_carla_scenario.entity.registry": _NOT_YET,
    "autoware_carla_scenario.entity.vehicle_entity": _NOT_YET,
    "autoware_carla_scenario.examples": _NOT_YET,
    "autoware_carla_scenario.examples.conf": _NOT_YET,
    "autoware_carla_scenario.examples.configs": _NOT_YET,
    "autoware_carla_scenario.examples.cut_in": _NOT_YET,
    "autoware_carla_scenario.examples.expand": _NOT_YET,
    "autoware_carla_scenario.examples.intersection_passing": _NOT_YET,
    "autoware_carla_scenario.examples.lane_change": _NOT_YET,
    "autoware_carla_scenario.examples.pedestrian_dart_out": _NOT_YET,
    "autoware_carla_scenario.examples.run": "imports pathlib, omegaconf, hydra, json",
    "autoware_carla_scenario.examples.temporary_stop": _NOT_YET,
    "autoware_carla_scenario.examples.traffic_light_compliance": _NOT_YET,
    "autoware_carla_scenario.maps": _NOT_YET,
    "autoware_carla_scenario.maps.cache": "imports pathlib, subprocess, shutil, tempfile, json, hashlib",
    "autoware_carla_scenario.maps.catalogue": _NOT_YET,
    "autoware_carla_scenario.maps.config": "imports pathlib",
    "autoware_carla_scenario.maps.opendrive": "imports pathlib, shutil",
    "autoware_carla_scenario.maps.resolver": "imports pathlib, hashlib",
    "autoware_carla_scenario.maps.source": "imports urllib",
    "autoware_carla_scenario.measures": _NOT_YET,
    "autoware_carla_scenario.odd": _NOT_YET,
    "autoware_carla_scenario.odd.cli": "imports json, argparse",
    "autoware_carla_scenario.odd.model": _NOT_YET,
    "autoware_carla_scenario.odd.openodd": "imports pathlib, yaml",
    "autoware_carla_scenario.odd.probes": "imports lanelet2",
    "autoware_carla_scenario.odd.registry": "imports pathlib, importlib",
    "autoware_carla_scenario.odd.route": "imports lanelet2",
    "autoware_carla_scenario.odd.sampler": "imports pathlib",
    "autoware_carla_scenario.odd.scenario_measure": _NOT_YET,
    "autoware_carla_scenario.odd.sources": "imports pathlib, subprocess, shutil, tempfile, tarfile, hashlib, fcntl, io, contextlib",
    "autoware_carla_scenario.odd.units": _NOT_YET,
    "autoware_carla_scenario.pytest_fixtures": "imports pytest",
    "autoware_carla_scenario.registry": "imports pathlib, omegaconf, importlib",
    "autoware_carla_scenario.route": _NOT_YET,
    "autoware_carla_scenario.route.active": _NOT_YET,
    "autoware_carla_scenario.route.frame": _NOT_YET,
    "autoware_carla_scenario.route.geometry": "imports lanelet2, autoware_lanelet2_extension_python",
    "autoware_carla_scenario.route.model": _NOT_YET,
    "autoware_carla_scenario.route.positions": _NOT_YET,
    "autoware_carla_scenario.route.search": _NOT_YET,
    "autoware_carla_scenario.scaffold": _NOT_YET,
    "autoware_carla_scenario.scaffold.generator": "imports pathlib, argparse, keyword",
    "autoware_carla_scenario.scenario_base": _NOT_YET,
    "autoware_carla_scenario.scenario_config": "imports omegaconf",
    "autoware_carla_scenario.scenario_queue": "imports pathlib, tqdm, pytest",
    "autoware_carla_scenario.scenario_runner": "imports pathlib, shutil",
    "autoware_carla_scenario.sensor": _NOT_YET,
    "autoware_carla_scenario.sensor.base": "imports numpy",
    "autoware_carla_scenario.sensor.carla_camera": "imports numpy, queue",
    "autoware_carla_scenario.sensor.carla_lidar": "imports numpy, queue",
    "autoware_carla_scenario.server": "imports pathlib, dotenv, subprocess, signal, atexit",
    "autoware_carla_scenario.signals": _NOT_YET,
    "autoware_carla_scenario.signals.controller": _NOT_YET,
    "autoware_carla_scenario.signals.registry": _NOT_YET,
    "autoware_carla_scenario.sweeper": _NOT_YET,
    "autoware_carla_scenario.sweeper.bindings": "imports lanelet2",
    "autoware_carla_scenario.sweeper.constraints": "imports lanelet2",
    "autoware_carla_scenario.sweeper.expand": "imports omegaconf",
    "autoware_carla_scenario.sweeper.lanelet_constraint_sweeper": "imports pathlib, omegaconf, hydra, tqdm, json, signal",
    "autoware_carla_scenario.sweeper.map_loader": "imports lanelet2, pathlib",
    "autoware_carla_scenario.templating": "imports pathlib, jinja2",
    "autoware_carla_scenario.tools": _NOT_YET,
    "autoware_carla_scenario.tools.detect_no_3d_model_lanelets": "imports lanelet2, pathlib, omegaconf, tqdm, shutil, argparse",
    "autoware_carla_scenario.traffic": _NOT_YET,
    "autoware_carla_scenario.traffic.base": "imports pathlib",
    "autoware_carla_scenario.traffic.config": _NOT_YET,
    "autoware_carla_scenario.traffic.driven": _NOT_YET,
    "autoware_carla_scenario.traffic.registry": _NOT_YET,
    "autoware_carla_scenario.traffic.sumo": _NOT_YET,
    "autoware_carla_scenario.traffic.sumo.backend": "imports pathlib, traci, sumolib, libsumo",
    "autoware_carla_scenario.traffic.sumo.config": "imports pathlib",
    "autoware_carla_scenario.traffic.sumo.geometry": _NOT_YET,
    "autoware_carla_scenario.traffic.sumo.network": "imports pathlib, roadgen, sumo, subprocess, shutil, tempfile, importlib, hashlib",
    "autoware_carla_scenario.traffic.sumo.physics_control": _NOT_YET,
    "autoware_carla_scenario.traffic.traffic_manager": _NOT_YET,
    "autoware_carla_scenario.trajectory": _NOT_YET,
    "autoware_carla_scenario.trajectory.authoring": _NOT_YET,
    "autoware_carla_scenario.trajectory.model": _NOT_YET,
    "autoware_carla_scenario.trajectory.relative_lane": "imports lanelet2, autoware_lanelet2_extension_python",
    "autoware_carla_scenario.trajectory.resolve": _NOT_YET,
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
