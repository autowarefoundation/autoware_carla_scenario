"""Which framework modules the library check compiles (docs/typecheck.md).

The scenario check compiles a scenario against the hand-written model of the
framework (``codon/``).  The library check (:func:`.check.typecheck_library`,
``scenario-check --library``) compiles the framework's own source instead, a
module at a time, as each is made to compile.  This file says which:

* :data:`CHECKED`: modules compiled from their real source.  Every function
  and every method of every class, protected (``_``) ones included, is
  called with a value of each parameter's annotated type, so their bodies
  are checked.
* :data:`EXCLUDED`: every other module, with the reason it is not compiled.
  :data:`NOT_YET_CHECKED` marks one that could be: moving it into
  :data:`CHECKED` is a matter of making its source compile.  "imports X"
  names what keeps a module out: a package or a standard module Codon cannot
  compile.

Every module of the package except ``typecheck/`` is in exactly one of the
two (``test_typecheck_library.py``), so a new module needs an entry here.

:data:`UNCALLED` names the few functions and methods of a checked module the
check leaves out, each with its reason: one whose job Codon cannot express (a
dict of mixed value types, ``json``, a parameter of ``Any``), in a module that
otherwise compiles.

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
    "MODELLED_IMPORTS",
    "NOT_YET_CHECKED",
    "REPLACED_MODELS",
    "UNCALLED",
    "package_modules",
]

#: The reason of a module nobody has made compile yet.
NOT_YET_CHECKED = "not yet checked (#45)"

#: Modules compiled from their real source, by dotted name (a package by its
#: own name, for its ``__init__.py``).
CHECKED: tuple[str, ...] = (
    "autoware_carla_scenario.action_state",
    "autoware_carla_scenario.actions",
    "autoware_carla_scenario.actions._departures",
    "autoware_carla_scenario.actions.background_traffic",
    "autoware_carla_scenario.actions.base",
    "autoware_carla_scenario.actions.environment",
    "autoware_carla_scenario.actions.follow_trajectory",
    "autoware_carla_scenario.actions.lane_change",
    "autoware_carla_scenario.actions.routing",
    "autoware_carla_scenario.actions.set_speed",
    "autoware_carla_scenario.actions.traffic_signal",
    "autoware_carla_scenario.actions.traffic_signal_controller",
    "autoware_carla_scenario.actions.turn",
    "autoware_carla_scenario.actions.turn_indicator",
    "autoware_carla_scenario.actions.walk_straight",
    "autoware_carla_scenario.authoring.framework_pin",
    "autoware_carla_scenario.authoring.uv_tool",
    "autoware_carla_scenario.autoware_bridge",
    "autoware_carla_scenario.autoware_bridge._proto",
    "autoware_carla_scenario.autoware_bridge.base",
    "autoware_carla_scenario.autoware_bridge.fake",
    "autoware_carla_scenario.autoware_bridge.ros_bridge",
    "autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api",
    "autoware_carla_scenario.autoware_stack",
    "autoware_carla_scenario.autoware_stack.bridge_node",
    "autoware_carla_scenario.autoware_stack.docker",
    "autoware_carla_scenario.autoware_stack.launcher",
    "autoware_carla_scenario.camera_recorder",
    "autoware_carla_scenario.conditions",
    "autoware_carla_scenario.conditions.action_state",
    "autoware_carla_scenario.conditions.always_true",
    "autoware_carla_scenario.conditions.and_condition",
    "autoware_carla_scenario.conditions.base",
    "autoware_carla_scenario.conditions.collision",
    "autoware_carla_scenario.conditions.comparison",
    "autoware_carla_scenario.conditions.composition",
    "autoware_carla_scenario.conditions.composition.acceleration",
    "autoware_carla_scenario.conditions.composition.base",
    "autoware_carla_scenario.conditions.composition.distance_measure",
    "autoware_carla_scenario.conditions.composition.entity_distance",
    "autoware_carla_scenario.conditions.composition.entity_lane_position",
    "autoware_carla_scenario.conditions.composition.entity_position_distance",
    "autoware_carla_scenario.conditions.composition.relative_speed",
    "autoware_carla_scenario.conditions.composition.speed",
    "autoware_carla_scenario.conditions.composition.standstill",
    "autoware_carla_scenario.conditions.composition.temporary_stop",
    "autoware_carla_scenario.conditions.composition.time_headway",
    "autoware_carla_scenario.conditions.composition.time_to_collision",
    "autoware_carla_scenario.conditions.composition.turn_indicator",
    "autoware_carla_scenario.conditions.composition.waypoint",
    "autoware_carla_scenario.conditions.elapsed_time",
    "autoware_carla_scenario.conditions.metric",
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
    "autoware_carla_scenario.coordinate.map_manager",
    "autoware_carla_scenario.coordinate.poses",
    "autoware_carla_scenario.coordinate.projection",
    "autoware_carla_scenario.coordinate.snap",
    "autoware_carla_scenario.coordinate.stop_line",
    "autoware_carla_scenario.coordinate.traffic_light",
    "autoware_carla_scenario.coordinate.transform",
    "autoware_carla_scenario.coverage",
    "autoware_carla_scenario.debug_markers",
    "autoware_carla_scenario.driver",
    "autoware_carla_scenario.driver.base",
    "autoware_carla_scenario.driver.control",
    "autoware_carla_scenario.driver.geometry",
    "autoware_carla_scenario.driver.observation",
    "autoware_carla_scenario.entity",
    "autoware_carla_scenario.entity._policy_warmup",
    "autoware_carla_scenario.entity.autoware_entity",
    "autoware_carla_scenario.entity.bridged_ego",
    "autoware_carla_scenario.entity._spawn",
    "autoware_carla_scenario.entity.ego",
    "autoware_carla_scenario.entity.externally_driven",
    "autoware_carla_scenario.entity.pedestrian_entity",
    "autoware_carla_scenario.entity.registry",
    "autoware_carla_scenario.entity.ros2_entity",
    "autoware_carla_scenario.entity.vehicle_entity",
    "autoware_carla_scenario.entity_role",
    "autoware_carla_scenario.examples.conf",
    "autoware_carla_scenario.examples.configs",
    "autoware_carla_scenario.examples.cut_in",
    "autoware_carla_scenario.examples.intersection_passing",
    "autoware_carla_scenario.examples.lane_change",
    "autoware_carla_scenario.examples.pedestrian_dart_out",
    "autoware_carla_scenario.examples.temporary_stop",
    "autoware_carla_scenario.examples.traffic_light_compliance",
    "autoware_carla_scenario.kinematics",
    "autoware_carla_scenario.kinematics.acceleration",
    "autoware_carla_scenario.kinematics.angle",
    "autoware_carla_scenario.kinematics.frames",
    "autoware_carla_scenario.kinematics.vector",
    "autoware_carla_scenario.ros2",
    "autoware_carla_scenario.ros2.commands",
    "autoware_carla_scenario.ros2.config",
    "autoware_carla_scenario.ros2.transport",
    "autoware_carla_scenario.route",
    "autoware_carla_scenario.route.active",
    "autoware_carla_scenario.route.frame",
    "autoware_carla_scenario.route.geometry",
    "autoware_carla_scenario.route.model",
    "autoware_carla_scenario.route.positions",
    "autoware_carla_scenario.route.search",
    "autoware_carla_scenario.sweeper.map_loader",
    "autoware_carla_scenario.tools",
    "autoware_carla_scenario.traffic",
    "autoware_carla_scenario.traffic.base",
    "autoware_carla_scenario.traffic.config",
    "autoware_carla_scenario.traffic.context",
    "autoware_carla_scenario.traffic.driven",
    "autoware_carla_scenario.traffic.traffic_manager",
    "autoware_carla_scenario.kinematics.velocity",
    "autoware_carla_scenario.maps",
    "autoware_carla_scenario.maps.catalogue",
    "autoware_carla_scenario.maps.config",
    "autoware_carla_scenario.maps.opendrive",
    "autoware_carla_scenario.maps.resolver",
    "autoware_carla_scenario.maps.source",
    "autoware_carla_scenario.measures",
    "autoware_carla_scenario.metrics",
    "autoware_carla_scenario.placement",
    "autoware_carla_scenario.scenario_base",
    "autoware_carla_scenario.scenario_config",
    "autoware_carla_scenario.scenario_queue",
    "autoware_carla_scenario.scenario_runner",
    "autoware_carla_scenario.server",
    "autoware_carla_scenario.sensor",
    "autoware_carla_scenario.sensor.base",
    "autoware_carla_scenario.sensor.carla_camera",
    "autoware_carla_scenario.sensor.carla_lidar",
    "autoware_carla_scenario.odd.route_plan",
    "autoware_carla_scenario.odd.scenario_measure",
    "autoware_carla_scenario.odd.unit_names",
    "autoware_carla_scenario.signals",
    "autoware_carla_scenario.signals.controller",
    "autoware_carla_scenario.signals.registry",
    "autoware_carla_scenario.traffic.sumo",
    "autoware_carla_scenario.traffic.sumo.config",
    "autoware_carla_scenario.traffic.sumo.geometry",
    "autoware_carla_scenario.trajectory.resolve",
    "autoware_carla_scenario.world_reset",
    "autoware_carla_scenario.traffic.sumo.physics_control",
    "autoware_carla_scenario.trajectory",
    "autoware_carla_scenario.trajectory.model",
    "autoware_carla_scenario.sweeper.bindings",
    "autoware_carla_scenario.sweeper.constraints",
    "autoware_carla_scenario.trajectory.relative_lane",
    "autoware_carla_scenario.ui.runner",
    "autoware_carla_scenario.utils",
    "autoware_carla_scenario.utils.opendrive",
    "autoware_carla_scenario.utils.powertrain",
    "autoware_carla_scenario.utils.stop_line",
    "autoware_carla_scenario.utils.traffic_light",
    "autoware_carla_scenario.utils.turn_indicator",
    "autoware_carla_scenario.utils.vehicles",
)

#: The reason of a method that builds a dict of mixed value types.
_DETAILS = "builds a dict[str, Any] of mixed value types, which Codon cannot express"

#: The reason of a function that takes a backend's options node.
_OPTIONS = (
    "takes a backend's options as `Mapping[str, Any]`, a Hydra node of mixed "
    "value types, which Codon cannot express"
)
#: The reason of a driver config built from a Hydra node.
_DRIVER_FROM_MAPPING = (
    "takes a Hydra node (`Mapping[str, Any]`) of mixed value types, which "
    "Codon cannot express, checks its keys with utils.config, which reads a "
    "dataclass's fields from its class (`type`), and builds the config with "
    "`**` of a dict; utils.config is imported inside it, so it is not compiled"
)
#: The reason of a config built from such a node.
_FROM_MAPPING = (
    f"{_OPTIONS}, and checks its keys with utils.config, which reads a "
    "dataclass's fields from its class (`type`); utils.config is imported "
    "inside it, so it is not compiled"
)

#: The reason of what takes the bridge node's ROS parameters.
_ROS_PARAMETERS = (
    "takes the scenario_bridge node's ROS parameters as `Mapping[str, Any]` "
    "(bool, int, float or str values), which Codon cannot express: Codon 0.19 "
    "fails on a Union in a parameter annotation (`union already sealed`), so "
    "the checker drops it and the check has nothing to call with"
)

#: The reason of a background-traffic constructor that takes lanelet constraints.
_CONSTRAINT_SPECS = (
    "takes lanelet constraints as `Sequence[ConstraintSpec]`, each a Hydra "
    "node (`Mapping[str, Any]`) or a parsed constraint "
    "(`ConstraintSpec = Union[Mapping[str, Any], Any]`), which Codon cannot "
    "express; LaneletRegion.lanelet_ids parses them in the `else` branch of "
    "`if TYPE_CHECKING:`"
)

#: The reason of a function that reads a YAML document's values.
_YAML = (
    "reads a YAML mapping of values of any type (`Any`: numbers, strings, "
    "booleans, lists, mappings or None), which Codon cannot express"
)

#: Functions and methods of checked modules the check does not call,
#: by dotted name (``module.Class.method``), with the reason.  Codon compiles
#: a function only when something calls it, so the body of one listed here is
#: not checked; nor are the imports made inside it, which is where a module
#: imports what Codon has nothing for (``json``) and only that function uses.
#: Everything else in the module still is.
UNCALLED: dict[str, str] = {
    "autoware_carla_scenario.actions.background_traffic.constraints_from_text": "reads YAML (yaml, imported inside it) into a list of `dict[str, Any]` constraint mappings of mixed value types, which Codon cannot express",
    "autoware_carla_scenario.actions.background_traffic.LaneletRegion.__init__": _CONSTRAINT_SPECS,
    "autoware_carla_scenario.actions.background_traffic.TrafficSinkAction.__init__": _CONSTRAINT_SPECS,
    "autoware_carla_scenario.actions.background_traffic.TrafficSourceAction.__init__": _CONSTRAINT_SPECS,
    "autoware_carla_scenario.authoring.framework_pin.Pin.uv_source": 'builds the `[tool.uv.sources]` entry, a dict[str, Any] of str and bool values (`{"path": ..., "editable": True}`), which Codon cannot express',
    "autoware_carla_scenario.autoware_stack.bridge_node.bridge_ros_args": _ROS_PARAMETERS,
    "autoware_carla_scenario.autoware_stack.bridge_node.bridge_command": f"{_ROS_PARAMETERS}; CommandAutowareLauncher.start() and DockerAutowareLauncher.start() call it in the `else` branch of `if TYPE_CHECKING:`",
    "autoware_carla_scenario.conditions.and_condition.AndCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.base.BaseCondition.__init_subclass__": "wraps check() when a subclass is created (cls.__dict__, functools.wraps), which Codon has no hook for",
    "autoware_carla_scenario.conditions.base.BaseCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.base.BaseCondition.to_summary_dict": _DETAILS,
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
    "autoware_carla_scenario.conditions.composition.turn_indicator.TurnIndicatorCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.composition.waypoint.WaypointCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.elapsed_time.ElapsedTimeCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.metric.MetricCondition.get_details": _DETAILS,
    "autoware_carla_scenario.driver.base.DriverCameraConfig.from_mapping": _DRIVER_FROM_MAPPING,
    "autoware_carla_scenario.driver.base.DriverClientConfig.from_mapping": _DRIVER_FROM_MAPPING,
    "autoware_carla_scenario.driver.base.DriverLidarConfig.from_mapping": _DRIVER_FROM_MAPPING,
    "autoware_carla_scenario.driver.control.ControlConfig.from_mapping": "takes a Hydra node (`Mapping[str, Any]`) of mixed value types, which Codon cannot express, reads the dataclass's fields from its class (`dataclasses.fields(cls)`) and builds the config with `**` of a dict",
    "autoware_carla_scenario.camera_recorder.draw_caption": "draws with cv2 (OpenCV), which Codon cannot compile; cv2 is imported inside it, and CameraRecorder.write_frame calls it in the `else` branch of `if TYPE_CHECKING:`",
    "autoware_carla_scenario.driver.observation.encode_frame_jpeg": "encodes with cv2 (OpenCV), which Codon cannot compile; cv2 is imported inside it, so it is not compiled",
    "autoware_carla_scenario.entity.registry.find_entity_by_role_name": "returns the entity of any kind registered under a role (`Optional[Any]`), which Codon cannot express; find_vehicle_entity and find_pedestrian_entity are its typed lookups",
    "autoware_carla_scenario.entity.ros2_entity._publishing_camera": "builds the camera of ros2.camera (imported inside it), whose listener the check cannot type through CarlaCameraSensor's listen callback; Ros2EgoEntity.on_scenario_start calls it in the `else` branch of `if TYPE_CHECKING:`",
    "autoware_carla_scenario.maps.config.resolve_map_paths": "takes the `map` config node as `Any` (a MapConfig, an OmegaConf node or the editor's MapRef, read with hasattr/getattr and indexing), which Codon cannot express",
    "autoware_carla_scenario.entity.registry.register_entity": "takes an entity of any kind (`Any`: a vehicle, a pedestrian or a duck-typed object); Codon resolves its isinstance() dispatch by the static type rather than the class, so a subclass of BackendDriven takes the wrong branch; register_vehicle_entity and register_pedestrian_entity are its typed counterparts, which scenario_base calls",
    "autoware_carla_scenario.conditions.not_condition.NotCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.or_condition.OrCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.persistent.PersistentCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.route_progress.RouteProgressCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.sticky.StickyCondition.get_details": _DETAILS,
    "autoware_carla_scenario.conditions.traffic_signal.TrafficSignalCondition.get_details": _DETAILS,
    "autoware_carla_scenario.trajectory.model._refusing_vertex_times": "the run-time wrapper of TrajectoryVertex.__init__ that refuses the calls written for a vertex with a time (`*args: Any, **kwargs: Any`, functools.wraps, an inspect.Signature); the check drops the decorator and compiles __init__ as written",
    "autoware_carla_scenario.traffic.build_none": _OPTIONS,
    "autoware_carla_scenario.traffic.build_sumo": _OPTIONS,
    "autoware_carla_scenario.traffic.build_traffic_manager": _OPTIONS,
    "autoware_carla_scenario.traffic.base.TrafficBackend.describe": _DETAILS,
    "autoware_carla_scenario.traffic.config.TrafficConfig.from_mapping": _FROM_MAPPING,
    "autoware_carla_scenario.traffic.config.TrafficManagerBackendConfig.from_mapping": _FROM_MAPPING,
    "autoware_carla_scenario.traffic.sumo.config.AmbientTrafficConfig.from_mapping": _FROM_MAPPING,
    "autoware_carla_scenario.traffic.sumo.config.SumoBackendConfig.from_mapping": _FROM_MAPPING,
    "autoware_carla_scenario.traffic.traffic_manager.TrafficManagerBackend.describe": _DETAILS,
    "autoware_carla_scenario.coordinate.projection._projector_info": f"{_YAML} (map_projector_info.yaml, with yaml, imported inside it, which Codon does not have); only read_map_origin and read_projector_type call it",
    "autoware_carla_scenario.coordinate.projection.read_map_origin": f"{_YAML} (map_projector_info.yaml); map_origin calls it in the `else` branch of `if TYPE_CHECKING:`",
    "autoware_carla_scenario.coordinate.projection.read_projector_type": f"{_YAML} (map_projector_info.yaml); resolve_projector calls it in the `else` branch of `if TYPE_CHECKING:`",
    "autoware_carla_scenario.ros2.config.CommandSourceConfig.from_mapping": _DRIVER_FROM_MAPPING,
    "autoware_carla_scenario.ros2.config.Ros2CameraConfig.from_mapping": _DRIVER_FROM_MAPPING,
    "autoware_carla_scenario.ros2.config.Ros2EgoConfig.from_mapping": _DRIVER_FROM_MAPPING,
    "autoware_carla_scenario.route.model.Range.parse": _YAML,
    "autoware_carla_scenario.route.model.RouteMatch.from_config": _YAML,
    "autoware_carla_scenario.route.model.RouteMatch.to_config": _DETAILS,
    "autoware_carla_scenario.route.model.parse_route_search": _YAML,
    "autoware_carla_scenario.scenario_base.BaseScenario.create_ego": "instantiates the ego's class held in `ego_type`, which Codon cannot hold as a value",
    "autoware_carla_scenario.scenario_base.BaseScenario.register_cover": "builds a CoverItem of values of any type (`expression: Callable[[carla.World], Any]`, `values: Iterable[Any]`, `ignore: Callable[[Any], bool]`, `event: SamplingEvent | BaseCondition`), which Codon cannot express; coverage.items is not checked",
    "autoware_carla_scenario.scenario_queue.ScenarioQueue.as_fixture": "builds a pytest fixture (pytest, imported inside it), a generator function returned as a `Callable[[], Generator[ScenarioQueue, None, None]]`",
    "autoware_carla_scenario.scenario_runner._collect_condition_statuses": "builds ConditionStatus values, which hold a condition's details (`dict[str, Any]`), which Codon cannot express; _condition_statuses calls it in the `else` branch of `if TYPE_CHECKING:`",
    "autoware_carla_scenario.scenario_runner._write_result": "writes the result with json, and the environment with environment.write() (platform, json), imported inside it, which Codon does not have; run_scenario calls it in the `else` branch of `if TYPE_CHECKING:`",
    "autoware_carla_scenario.signals.controller.build_controllers": "reads the authoring package's pydantic specs (MapRef.traffic_signal_controllers and signal_groups), typed `list[Any]` so the runtime does not import authoring, with getattr defaults for fields older documents lack",
    "autoware_carla_scenario.sweeper.bindings.parse_binding": _YAML,
    "autoware_carla_scenario.sweeper.constraints.parse_constraint": _YAML,
    "autoware_carla_scenario.utils.stop_line._road_marking_stop_lines": "returns a road marking's `refers` parameters as line strings, and a `refers` parameter may be any of lanelet2's primitives (a polygon, a lanelet, ...), which the lanelet2 model types as what they all have (id, attributes); regulatory_element_stop_lines calls it in the `else` branch of `if TYPE_CHECKING:`",
    "autoware_carla_scenario.maps.config._setting": "reads one `map` setting off a config of any shape (`map_config: Any`: a MapConfig, an OmegaConf node or the editor's MapRef, read with hasattr/getattr and indexing), which Codon cannot express; only resolve_map_paths, itself uncalled, calls it",
    "autoware_carla_scenario.maps.config._flag": "reads a boolean `map` setting with _setting, off a config of any shape (`map_config: Any`), which Codon cannot express; only resolve_map_paths, itself uncalled, calls it",
    "autoware_carla_scenario.route.model._tristate": f"{_YAML} (a segment's opposite_lane, stop_line, ... flags); only _parse_segment, itself uncalled, calls it",
    "autoware_carla_scenario.route.model._parse_segment": f"{_YAML} (one segment of a route search); only parse_route_search, itself uncalled, calls it",
    "autoware_carla_scenario.scenario_base.BaseScenario._register_tick": "takes the list a plain callback goes into (`List[Callable[[carla.World], None]]`), and the check has no value of a list of callables to call with; register_init(), register_pre_tick() and register_post_tick() call it with BaseScenario's own lists, so its body is compiled through them",
    "autoware_carla_scenario.signals.controller._group_members": "reads the authoring package's pydantic signal groups (`list[Any]`, so the runtime does not import authoring); only build_controllers, itself uncalled, calls it",
    "autoware_carla_scenario.signals.controller._phase_states": "reads one of the authoring package's pydantic phases (`Any`, so the runtime does not import authoring); only build_controllers, itself uncalled, calls it",
    "autoware_carla_scenario.sweeper.constraints._bind_set_level_constraints": "walks a constraint tree of any node (`constraint: Any`: a Constraint, or any object with `constraints` or `constraint`), which Codon cannot express; find_matching_lanelets calls it with each Constraint, so its body is compiled through it",
    "autoware_carla_scenario.sweeper.constraints._bind_children": "reads the children of a constraint of any kind with hasattr (`constraint: Any`), which Codon cannot express; the constraints' _bind_set_level() call it with themselves, so its body is compiled through them",
    "autoware_carla_scenario.sweeper.constraints._cache_ids": "sets `_cached_ids` on a frozen constraint of any kind with `object.__setattr__` (`constraint: Any`), which Codon cannot express; the set-level constraints' _bind_set_level() call it with themselves, so its body is compiled through them",
    "autoware_carla_scenario.trajectory.model._whole": "checks that a value of any type (`value: Any`: whatever a YAML document or a caller wrote for a lane or junction index) is a whole number, which Codon cannot express; the route poses' __post_init__ call it with their own fields, so its body is compiled through them",
    "autoware_carla_scenario.trajectory.model._vertex_advance": "takes a TrajectoryVertex call's `*args` and `**kwargs` as written (`tuple[Any, ...]`, `dict[str, Any]`), which Codon cannot express; only _refusing_vertex_times, itself uncalled, calls it",
    "autoware_carla_scenario.scenario_runner._build_condition_status": "builds a ConditionStatus, which holds the condition's details (`dict[str, Any]`), which Codon cannot express; only _collect_condition_statuses, itself uncalled, calls it",
}

#: Every other module, with the reason it is not compiled.
EXCLUDED: dict[str, str] = {
    "autoware_carla_scenario": "loads its public names lazily with importlib (a PEP 562 `__getattr__` returning `object`), and Codon reads its `TYPE_CHECKING` re-exports as imports of every subpackage, two of which are neither checked nor modelled: pytest_fixtures (pytest) and registry (omegaconf)",
    "autoware_carla_scenario.authoring": "re-exports authoring.models (pydantic), authoring.persistence (yaml), authoring.package_export and authoring.wheelhouse (tempfile)",
    "autoware_carla_scenario.authoring._builders_generated": "its builders take authoring.compiler's CompiledCondition, CompiledAction and BuildContext, which reach pydantic models through authoring.models",
    "autoware_carla_scenario.authoring.builders": "dispatches to builders by name through vars() and globals() as Callable[..., Any]; its parameters reach pydantic models through authoring.compiler, and it imports authoring._builders_generated and authoring.models (pydantic) inside functions, which are not checked",
    "autoware_carla_scenario.authoring.codegen": "imports json, argparse, importlib, inspect, difflib, and templating (jinja2) and authoring.registry (`Any` field defaults), which are not checked",
    "autoware_carla_scenario.authoring.compiler": "reaches pydantic models through authoring.models",
    "autoware_carla_scenario.authoring.hydra_config": "reaches pydantic models through authoring.models; imports authoring.persistence (yaml)",
    "autoware_carla_scenario.authoring.models": "imports pydantic, uuid",
    "autoware_carla_scenario.authoring.package_export": "imports tempfile, platform, more of functools than Codon's declares (`lru_cache()`) and of importlib.metadata than its shim declares (`metadata()`), and, at module level, authoring.models (pydantic), authoring.persistence (yaml), authoring.hydra_config and authoring.validator (pydantic, through authoring.models), templating (jinja2) and authoring.wheelhouse (tempfile), which are not checked and have no model",
    "autoware_carla_scenario.authoring.persistence": "imports yaml, and authoring.models (pydantic), which is not checked and has no model",
    "autoware_carla_scenario.authoring.registry": "FieldSpec.default is Any and holds str, int, float, bool, list or None across the spec table, which Codon has no type for; rule_symbol takes object, and searchable_lanelet_fields a union of spec classes",
    "autoware_carla_scenario.authoring.starter": "reaches pydantic models through authoring.models",
    "autoware_carla_scenario.authoring.validator": "reaches pydantic models through authoring.models",
    "autoware_carla_scenario.authoring.wheelhouse": "imports jinja2, packaging, tomllib, tomli, tempfile, platform, and more of pathlib than its shim declares (`Path.iterdir()`, `Path.stat()`)",
    "autoware_carla_scenario.autoware_bridge._proto.autoware_bridge_pb2": "imports google.protobuf",
    "autoware_carla_scenario.autoware_bridge._proto.autoware_bridge_pb2_grpc": "imports grpc",
    "autoware_carla_scenario.autoware_bridge.grpc_server": "imports grpc, concurrent.futures",
    "autoware_carla_scenario.autoware_bridge.ros_bridge.__main__": "runs the scenario_bridge node: imports autoware_bridge.ros_bridge.node (rclpy), which is neither checked nor modelled",
    "autoware_carla_scenario.autoware_bridge.ros_bridge.client": "imports grpc",
    "autoware_carla_scenario.autoware_bridge.ros_bridge.node": "imports rclpy, autoware_adapi_v1_msgs, geometry_msgs, grpc",
    "autoware_carla_scenario.autoware_stack.devcontainer": "imports yaml, json; reads a JSONC and a Compose document of values of any type (`Any`), which Codon cannot express. The checked autoware_stack.docker and the package take what they use of it (`DevContainer`, `DevContainerError`, `read_devcontainer()`, `DEFAULT_WORKSPACE_MOUNT`) from its model (autoware_stack/devcontainer.codon)",
    "autoware_carla_scenario.carla_install": "imports tqdm, json, argparse, urllib.request, http, email, tarfile, platform, and more of pathlib and shutil than their shims declare (`PurePosixPath`, `Path.iterdir()`, `Path.rename()`, `shutil.disk_usage()`)",
    "autoware_carla_scenario.coverage.cod": "re-exports openodd's COD export (a Rust extension)",
    "autoware_carla_scenario.coverage.collector": "derives openodd's collector (a Rust extension)",
    "autoware_carla_scenario.coverage.items": "re-exports openodd's cover items (a Rust extension)",
    "autoware_carla_scenario.coverage.report": "re-exports openodd's coverage merge and report (a Rust extension)",
    "autoware_carla_scenario.declarative": "imports authoring.builders, authoring.compiler, authoring.models (pydantic) and authoring.persistence (yaml), which are not checked and have no model",
    "autoware_carla_scenario.driver.egodriver_client": "imports grpc, google.protobuf, numpy.typing (`NDArray`), which Codon's numpy lacks, and carla_driver_interface's contract, policies, server and protocol, which its model (geometry only) does not declare",
    "autoware_carla_scenario.driver.hdmap_export": "imports roadgen, tempfile, json, and more of pathlib and shutil than their shims declare (`Path.rename()`, `shutil.copyfile()`)",
    "autoware_carla_scenario.driver.renderer": "imports carla_driver_interface.protocol (the protobuf messages), which its model (geometry only) does not declare, and numpy.typing (`NDArray`), which Codon's numpy lacks",
    "autoware_carla_scenario.editor": "imports uvicorn and editor.app (fastapi) inside main()",
    "autoware_carla_scenario.editor.app": "imports fastapi",
    "autoware_carla_scenario.editor.forms": "imports authoring.registry, which is not checked and has no model; parses form values typed Any",
    "autoware_carla_scenario.editor.map_preview": "imports json, and authoring.models (pydantic), which is not checked and has no model",
    "autoware_carla_scenario.editor.routes": "imports fastapi, starlette, anyio",
    "autoware_carla_scenario.editor.scenario_map": "reaches pydantic models through authoring.models; imports editor.map_preview (json)",
    "autoware_carla_scenario.editor.service": "imports yaml, tempfile, zipfile, contextlib; reaches pydantic models through authoring.models; takes form fields as `Mapping[str, Any]`, which Codon cannot express",
    "autoware_carla_scenario.entity.carla_driver_entity": "imports carla_driver_interface.driver (`TurnIndicator`) and carla_driver_interface.protocol, which its model (geometry only) does not declare, uuid, and driver.renderer (the protocol's messages) and driver.hdmap_export (roadgen, json), which are neither checked nor modelled; builds an EgoDriverGrpcClient (grpc), which its model refuses to build, and encodes camera frames with encode_frame_jpeg (cv2)",
    "autoware_carla_scenario.examples": "re-exports examples.run (omegaconf, hydra, json) and the registry (omegaconf, importlib)",
    "autoware_carla_scenario.examples.expand": "imports json, the registry (omegaconf, importlib) and sweeper.expand (omegaconf), which are not checked, and examples.run (hydra) inside it",
    "autoware_carla_scenario.environment": "imports platform, json, more of functools and sys than Codon's declare (`lru_cache()`, `sys.prefix`), and more of importlib.metadata than its shim declares (`distributions()`, `packages_distributions()`, `Distribution`) and of urllib.parse (`urlparse()`, `unquote()`); records packages as `dict[str, Any]`, which Codon cannot express",
    "autoware_carla_scenario.examples.run": "imports omegaconf, hydra, json, and more of pathlib than its shim declares (`Path.with_suffix()`), and typecheck and typecheck.mode (the checker, outside the library check), registry (omegaconf, importlib) and, inside a function, sweeper.expand (omegaconf), which are neither checked nor modelled; imports `ScenarioQueue` (and, inside build_ego_entity(), `AutowareBridgeConfig`, `AutowareEgoEntity`, `GrpcAutowareBridgeServer` and `CarlaDriverEntity`) from the package, and `PlannedRoute`, `RouteError`, `ODD_CASE_KEY` and `decode_case()` from odd.route and odd.sampler, which the package and odd models do not declare",
    "autoware_carla_scenario.maps.z_offset_cache": "imports json (`loads()`, `dumps()`) and contextlib (`suppress()`), more of pathlib than its shim declares (`Path.replace()`), and calls the two-argument `iter()`; maps/z_offset_cache.codon models what MapManager.initialize uses of it",
    "autoware_carla_scenario.maps.cache": "imports json (`loads()`, `dumps()`) and tempfile (`mkdtemp()`), and more of pathlib than its shim declares (`Path.replace()`); keeps a cache entry's metadata as `dict[str, Any]` of mixed value types, which Codon cannot express; maps/cache.codon models what the maps package uses of it",
    "autoware_carla_scenario.odd": "`from . import backend, probes` does not resolve against the package model, which declares no submodules; behind that, it re-exports odd.model, odd.sampler, odd.openodd, odd.sources and odd.backend (openodd, a Rust extension), and odd.probes, odd.route and odd.registry, which are excluded",
    "autoware_carla_scenario.odd.backend": "derives openodd's Backend (a Rust extension), a dataclass of probe callables and knobs (`Mapping[Any, OddKnob]`)",
    "autoware_carla_scenario.odd.cli": "imports json, argparse, and `odd_builder()` and `odd_names()` from odd.registry, which the odd model does not declare; inside its commands, it imports `combine_route_coverage()` from odd.route, which the odd model does not declare either (it declares only `plan_route_coverage()`), and typecheck (the checker, outside the library check), examples.run (omegaconf, hydra) and registry (omegaconf, importlib), which are neither checked nor modelled",
    "autoware_carla_scenario.odd.model": "re-exports openodd's model (a Rust extension)",
    "autoware_carla_scenario.odd.openodd": "re-exports openodd's reader (a Rust extension)",
    "autoware_carla_scenario.odd.probes": "imports `UNDECIDED` from odd.model, which the odd model does not declare; behind that, it attaches its `on_lanelet` readers to probe functions with `setattr`, marks an unread cache entry with an `object()` sentinel, caches probe values of any type (`dict[str, Any]`) and walks lanelet2's lanelets, map and routing graph typed `Any`, which Codon cannot express",
    "autoware_carla_scenario.odd.registry": "imports odd.model (openodd's model, a Rust extension) at module level, and importlib (import_callable resolves a `module:attr` spec to a `Callable[..., Any]`); imports registry (omegaconf, importlib), which is not checked, inside it",
    "autoware_carla_scenario.odd.route": "the route-coverage half of checking a planned route against an ODD (it imports the planning, `PlannedRoute`, `RouteError` and `plan_route()`, from odd.route_plan, which is checked, and re-exports it): names what the odd and coverage models do not declare (`coverage.items.value_label()`, `odd.model.UNDECIDED`, and the probe, cover item and derivation of `OddAttribute` and the attributes, `complete()`, `evaluate()`, `situations()` and `outside_buckets()` of `OddDefinition`), and holds values Codon has no type for: each lanelet's attribute values (`dict[str, Any]`: whatever a probe reads, a bool, int, float or str, `None`, or the `UNDECIDED` sentinel, compared by identity), a probe's optional `on_lanelet` reader (`getattr` with a default, any callable) and the `describe()` reports (`dict[str, Any]` of mixed values)",
    "autoware_carla_scenario.odd.sampler": "wraps openodd's sampler (a Rust extension) and reads `Mapping[str, Any]` config nodes (`sampler_from_config()`), which Codon cannot express",
    "autoware_carla_scenario.odd.sources": "re-exports openodd's git sources (a Rust extension)",
    "autoware_carla_scenario.odd.units": "re-exports openodd's units (a Rust extension)",
    "autoware_carla_scenario.pytest_fixtures": "imports pytest, and takes the scenario's class (`Type[BaseScenario]`), which Codon cannot hold as a value",
    "autoware_carla_scenario.registry": "imports omegaconf (`DictConfig`, under `TYPE_CHECKING`) and more of importlib.metadata than its shim declares (`entry_points()`); holds scenario classes (`type[BaseScenario]`) and their config classes as values, which Codon cannot hold",
    "autoware_carla_scenario.ros2.camera": "subclasses CarlaCameraSensor with a listener that hands each frame back to the ROS 2 ego from a CARLA sensor thread, which the check cannot type through the listen callback",
    "autoware_carla_scenario.ros2.cyclone": "imports cyclonedds (its domain participant, readers, writers, topics and QoS policies) and ros2.messages, which imports it too; ros2/cyclone.codon models the CycloneRos2Transport that entity.ros2_entity builds",
    "autoware_carla_scenario.ros2.messages": "imports cyclonedds (the IdlStruct base of every message, and the IDL field types `array[float64, 36]` and `sequence[TrajectoryPoint]`, which Codon cannot express)",
    "autoware_carla_scenario.scaffold": "re-exports scaffold.generator (argparse, jinja2)",
    "autoware_carla_scenario.scaffold.generator": "imports argparse, keyword, templating (jinja2) and, inside a function, authoring.package_export (tempfile), which are not checked",
    "autoware_carla_scenario.sweeper": "re-exports, besides the checked sweeper.bindings, sweeper.constraints and sweeper.map_loader, sweeper.expand (omegaconf) and sweeper.lanelet_constraint_sweeper (hydra)",
    "autoware_carla_scenario.sweeper.expand": "imports omegaconf, and openodd inside a function; imports `OddDraws` (under `TYPE_CHECKING`) and, inside a function, `ODD_CASE_KEY`, `encode_case()` and `search_from_config()` from odd.sampler, which the odd model does not declare; expand_sweep() and expand_route() take the sweep, the route search, the map and the ODD as `Any` and `Mapping[..., Any]`, which Codon cannot express",
    "autoware_carla_scenario.sweeper.lanelet_constraint_sweeper": "imports omegaconf, hydra, tqdm, json, and more of signal and os than their shims declare (`signal.alarm()`, `signal.getsignal()`, `os.fork()`, `os.setpgrp()`, `os.waitpid()`, `os._exit()` and the wait-status macros)",
    "autoware_carla_scenario.templating": "imports jinja2; toml_string() takes `object`",
    "autoware_carla_scenario.tools.detect_no_3d_model_lanelets": "imports omegaconf, tqdm, argparse",
    "autoware_carla_scenario.traffic.registry": "holds backend factories, each a `Callable[[Mapping[str, Any]], TrafficBackend]`, and takes their options as `Mapping[str, Any]`, which Codon cannot express; registry (omegaconf, importlib), which is neither checked nor modelled, is imported inside load_traffic_backend_plugins; traffic.codon models it for the traffic package",
    "autoware_carla_scenario.traffic.sumo.backend": "imports traci, sumolib, libsumo, and more of pathlib than its shim declares (`Path.with_suffix()`)",
    "autoware_carla_scenario.traffic.sumo.network": "imports roadgen, sumo, tempfile, and more of pathlib and os than their shims declare (`Path.rename()`, `Path.with_suffix()`, `os.rename()`)",
    "autoware_carla_scenario.trajectory.authoring": "parses document values of `Any` (YAML rows, editor fields) into `dict[str, Any]`, which Codon cannot express",
    "autoware_carla_scenario.trajectory_recorder": "writes JSON-lines records of mixed value types (`dict[str, Any]`) serialised with json.dumps(), which Codon has neither the type nor the module for; trajectory_recorder.codon models what scenario_runner uses of it",
    "autoware_carla_scenario.ui": "imports uvicorn and ui.app (fastapi) inside main()",
    "autoware_carla_scenario.ui.app": "imports fastapi, json, asyncio, fnmatch",
    "autoware_carla_scenario.ui.models": "imports pydantic",
    "autoware_carla_scenario.ui.scanner": "imports yaml, json, and more of pathlib than its shim declares (`Path.iterdir()`, `Path.stat()`, `Path.with_suffix()`)",
    "autoware_carla_scenario.ui.sweep_resolver": "imports omegaconf, hydra",
    "autoware_carla_scenario.utils.config": "reads the fields of a dataclass it is handed as a class (`type`, dataclasses.fields), which Codon cannot express",
}

#: Model modules (``codon/autoware_carla_scenario/<name>.codon``) the library
#: check compiles from the checked source they model instead: once all of the
#: modules named are checked, the model module re-exports their definitions,
#: so every module of the check -- checked or model -- has the same types.
#: Without it, a checked module handing its own pose to a model function
#: (``to_opendrive``) would hand the wrong type.
REPLACED_MODELS: dict[str, tuple[str, ...]] = {
    # So the checked BridgedEgoEntity hands the pose the model's to_map_frame
    # returns to the checked AutowareBridge.configure().
    "_bridge_pose": ("autoware_carla_scenario.autoware_bridge.base",),
    # So a scenario hands the GroundProjectionConfig it imports from the
    # top-level package (the model) to the checked VehicleEntityConfig (#99).
    "_ground_projection": ("autoware_carla_scenario.coordinate.snap",),
    # So the model GitMapCache (maps/cache.codon) takes and returns the
    # checked MapSource that the checked maps.resolver hands it.
    "_map_source": ("autoware_carla_scenario.maps.source",),
    "_poses": (
        "autoware_carla_scenario.coordinate.frames",
        "autoware_carla_scenario.coordinate.poses",
    ),
    # So the model CycloneRos2Transport (ros2/cyclone.codon), which the checked
    # entity.ros2_entity builds, derives from the checked Ros2EgoTransport and
    # takes the checked Ros2EgoConfig.
    "_ros2": (
        "autoware_carla_scenario.ros2.config",
        "autoware_carla_scenario.ros2.transport",
    ),
    "_vehicle_entity": (
        "autoware_carla_scenario.entity._spawn",
        "autoware_carla_scenario.entity.vehicle_entity",
    ),
}

#: Checked modules that import another checked module through its model
#: rather than its source, by importing module -> the modules so imported.
#: For an import cycle Codon cannot resolve: each module names the other's
#: types where Codon resolves them when it reads the module (a class-level
#: attribute, a global, a signature), so neither can be read first.  The
#: module keeps the model's types: they never meet the checked module's.
MODELLED_IMPORTS: dict[str, tuple[str, ...]] = {
    # BaseScenario holds Measures (`_measures: dict[str, Measure]`), and
    # measures holds the running BaseScenario (`_MEASURED`).
    "autoware_carla_scenario.measures": ("autoware_carla_scenario.scenario_base",),
    # BaseScenario holds an EgoVehicle (`ego_entity`), and EgoVehicle.spawn()
    # takes an EgoConfig.
    "autoware_carla_scenario.entity.ego": ("autoware_carla_scenario.scenario_base",),
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
