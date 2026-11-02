"""Public names the Codon model leaves out on purpose (docs/typecheck.md).

The scenario check compiles a scenario against the model (``codon/``), so a
name a modelled module exports in Python (its ``__all__``, or the public
names it defines when it has none) and the model does not declare cannot be
imported by a checked scenario.  Each such name is listed here with the
reason, by module: :func:`.check.typecheck_scenario` refuses an import of one
with that reason before compiling, and ``test_typecheck_model.py`` fails when
a public name is neither declared by the model nor listed here, or when a
name listed here is declared by the model or no longer public in Python.

The same holds for the public methods, properties, class attributes,
dataclass fields and instance attributes of the classes the model declares,
openodd's that the package re-exports and enums (`name`, `value` and their
own methods) included (#265, #298, #307): :data:`NOT_MODELLED_MEMBERS`
lists, with the reason, each one the model's class leaves out, and the
check adds that reason to Codon's "object has no attribute" error.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "NOT_MODELLED",
    "NOT_MODELLED_MEMBERS",
    "defines_members",
    "not_modelled_member",
]

_PACKAGE = "autoware_carla_scenario"
_MEMBER_MODULES = (_PACKAGE, "openodd")


def _each(names: str, reason: str) -> dict[str, str]:
    return {name: reason for name in names.split()}


_EGO_STACK = (
    "an ego the runner builds from the run's ego config (examples.run's "
    "build_ego_entity()), not a scenario: a scenario is handed its ego as an "
    "EgoVehicle; the library check compiles it from its source"
)
_AUTOWARE_BRIDGE = (
    "the bridge to an Autoware stack, which the runner builds for an Autoware "
    "ego from the run's config, not a scenario; the library check compiles it "
    "from its source"
)
_DRIVER = (
    "the egodriver client and its config, which the runner builds from the "
    "run's `driver` config (DriverConfig, which is modelled), not a scenario; "
    "the library check compiles it from its source"
)
_CAMERA = (
    "the camera sensors of an external driver's rig, which the ego entity "
    "attaches from the run's `driver` config, not a scenario; the library "
    "check compiles them from its source"
)
_RUNNER = (
    "the runner, which builds and runs the scenario: a scenario does not run "
    "itself; the library check compiles it from its source"
)
_REGISTRY = (
    "the scenario registry (registry.py: omegaconf, and scenario classes held "
    "as values, which Codon cannot hold), which a scenario package's "
    "__init__.py calls; the check compiles the scenario module and builds the "
    "scenario as register_scenario() would"
)
_TRAFFIC_BACKEND = (
    "a traffic backend or its config, which the runner builds from the run's "
    "`traffic` config (Hydra options, `Mapping[str, Any]`, which Codon cannot "
    "express), not a scenario: a scenario drives traffic through its actions; "
    "the library check compiles it from its source"
)
_TRAFFIC_PROTOCOL = (
    "what a traffic backend's entities implement (traffic.base, "
    "traffic.driven), which the checked entities and actions use and the "
    "library check compiles from its source; a scenario asks its entities "
    "through actions and conditions"
)
_OPENODD = (
    "openodd's (a Rust extension, with no model), which an ODD written in "
    "Python does not need: odd.codon models the ODD a builder function "
    "returns"
)
_ODD_RUN = (
    "the ODD's sampling and route coverage, which the runner, the sweeper and "
    "odd.cli use with openodd (a Rust extension) and values Codon has no type "
    "for (`Mapping[str, Any]`, `dict[str, Any]`), not an ODD or a scenario"
)
_ODD_REGISTRY = (
    "the ODD registry's lookup (odd.registry, over openodd's: importlib, entry points), which "
    "the runner uses; an ODD written in Python registers itself with "
    "register_odd(), which is modelled"
)
_ROUTE_SEARCH = (
    "the route search, parsed from a document's YAML (parse_route_search() "
    "takes `Mapping[str, Any]`, which Codon has no type for) into frozen "
    "dataclasses; a scenario reads the route its sweep "
    "matched with scenario_route(), which is modelled"
)
_ROUTE_SETTER = (
    "the runner's: it sets the running route before setup() and clears it "
    "after; a scenario reads it with scenario_route(), which is modelled"
)
_ENTITY_REGISTRY = (
    "the registry of spawned entities, which BaseScenario and the runner keep: "
    "an entity holds any kind of entity (`Any`), which Codon cannot express; "
    "a scenario spawns and finds its entities through BaseScenario"
)
_STACK_LAUNCHER = (
    "the launcher's implementation (subprocess, process groups), which the "
    "runner drives and the library check compiles from its source; "
    "autoware_stack/launcher.codon models what the ego entity and the queue "
    "call of a launcher they are given"
)
_GRPC = (
    "grpc's settings of the client or the server (grpc, which has no model); "
    "the model declares the class only by name, so the checked package's "
    "re-export resolves"
)
_CARLA_INSTALL = (
    "the CARLA installer and its command line (tqdm, urllib.request, tarfile, "
    "argparse); carla_install.codon models only installed_executable(), "
    "which the checked server manager uses"
)
_CYCLONE = (
    "converts ROS 2 messages (ros2.messages: cyclonedds's IdlStruct, which "
    "has no model); ros2/cyclone.codon models the transport "
    "entity.ros2_entity builds"
)
_UI = (
    "a pydantic model of the results UI (ui.app, fastapi), which the checked "
    "ui.runner does not use; ui/models.codon models what it does"
)

#: Model module (dotted, as Python imports it) -> name -> why the model does
#: not declare it.
NOT_MODELLED: dict[str, dict[str, str]] = {
    _PACKAGE: {
        **_each(
            "AutowareBridge AutowareBridgeConfig FakeAutowareBridge", _AUTOWARE_BRIDGE
        ),
        **_each("AutowareEgoEntity ExternallyDrivenEgo", _EGO_STACK),
        **_each(
            "BaseEgoDriverClient ControlConfig DriverCameraConfig "
            "DriverClientConfig DriverLidarConfig",
            _DRIVER,
        ),
        **_each(
            "CameraSensorBase CameraSensorConfig CarlaCameraSensor "
            "CarlaCameraSensorConfig",
            _CAMERA,
        ),
        "CarlaScenarioFixture": (
            "the pytest fixture (pytest_fixtures: pytest, and the scenario's "
            "class held as a value, which Codon cannot hold), used by a test, "
            "not a scenario"
        ),
        "CarlaServerManager": (
            "starts and stops the CARLA server for the runner and the pytest "
            "fixture, not a scenario; the library check compiles it from its "
            "source"
        ),
        **_each("ScenarioQueue ScenarioRunner", _RUNNER),
        **_each(
            "BuildScenarioFn ScenarioBuilder get_conf_dirs get_scenario_builder "
            "get_scenario_classes get_scenario_registry load_scenario_plugins "
            "register_conf_dir register_scenario register_scenario_builder "
            "unregister_conf_dir unregister_scenario",
            _REGISTRY,
        ),
        **_each(
            "NullTrafficBackend TrafficBackend TrafficBackendError "
            "TrafficBackendUnavailable TrafficConfig TrafficManagerBackend "
            "TrafficManagerBackendConfig TrafficParameterError",
            _TRAFFIC_BACKEND,
        ),
        "load_openodd": _OPENODD,
    },
    f"{_PACKAGE}.autoware_bridge.grpc_server": {"server_options": _GRPC},
    f"{_PACKAGE}.autoware_stack.launcher": _each(
        "AutowareStackError CommandAutowareLauncher DEFAULT_BRIDGE_PARAMETERS "
        "ProcessGroup episode_log_path fill_placeholders",
        _STACK_LAUNCHER,
    ),
    f"{_PACKAGE}.carla_install": _each(
        "HOME_ENV LAUNCHER NIGHTLY_MACHINES NIGHTLY_URL install install_dir main "
        "package_url",
        _CARLA_INSTALL,
    ),
    f"{_PACKAGE}.debug_markers": _each(
        "plan_markers plan_polyline world_point",
        "draws a driver's plan (carla_driver_interface's Trajectory and Pose), "
        "which only the driver-backed entities hold (checked from their "
        "source); the model of carla_driver_interface declares them for the "
        "driver modules, not for a scenario",
    ),
    f"{_PACKAGE}.driver.egodriver_client": _each(
        "EGODRIVER_SERVICE_FULL_NAME MAX_MESSAGE_BYTES channel_options", _GRPC
    ),
    f"{_PACKAGE}.conditions": {
        "entity_filter_from_fields": (
            "a scenario document's way to an EntityFilter (its generated "
            "builder); a scenario builds the EntityFilter itself"
        ),
    },
    f"{_PACKAGE}.constants": {
        "tick_multiple": (
            "checks a tick-quantised setting of the run's config (the driver "
            "policy's step, a LiDAR's sweep, the T4 and T4Scene steps) when the "
            "runner builds it, not a scenario; the library check compiles it "
            "from its source"
        )
    },
    f"{_PACKAGE}.entity": {
        **_each("AutowareEgoEntity BridgedEgoEntity ExternallyDrivenEgo", _EGO_STACK),
        **_each(
            "clear_entities find_entity_by_role_name register_entity unregister_entity",
            _ENTITY_REGISTRY,
        ),
    },
    f"{_PACKAGE}.weather_rules": _each(
        "DEFAULT_RULES EGO_KEY QUANTITIES RulesError WEATHER_INPUTS default_rules "
        "rule_values use_weather_rules vehicle_range weather_rule_files",
        "reads rule files with openodd (a Rust extension) and takes "
        "`environment.rules` as `Mapping[str, Any]`, the runner's; a scenario "
        "sets the weather, not the rules (weather_rules.codon models what "
        "visibility and actions.environment read the rules through)",
    ),
    f"{_PACKAGE}.environment": {
        "write": (
            "writes the run's environment with json (`dict[str, Any]`), the "
            "runner's; environment.codon models what it records with a run"
        )
    },
    f"{_PACKAGE}.odd": {
        **_each(
            "DEFAULT_KNOBS GitSource GitSourceError OddDraws OddKnob OddVerdict "
            "OpenOddError knobs_from_mapping load_odd_binding "
            "load_odd_file load_openodd read_probe",
            _OPENODD,
        ),
        **_each(
            "OddSampler PlannedRoute RouteCoverageSummary RouteError "
            "ScenarioMeasure combine_route_coverage controls_from_mapping "
            "plan_route",
            _ODD_RUN,
        ),
        **_each(
            "DEFAULT_ODD ENTRY_POINT_GROUP import_callable odd_names", _ODD_REGISTRY
        ),
        **_each(
            "backend probes",
            "a submodule (openodd's backend; the probes, which the library "
            "check compiles from their source), not a name an ODD or a "
            "scenario imports; the model declares no submodules of odd",
        ),
    },
    f"{_PACKAGE}.ros2.cyclone": _each(
        "control_from_message odometry_message ros_time trajectory_from_message",
        _CYCLONE,
    ),
    f"{_PACKAGE}.route": {
        **_each(
            "JunctionSegmentSpec LaneSegmentSpec Range RouteSearchSpec "
            "parse_route_search",
            _ROUTE_SEARCH,
        ),
        **_each("clear_scenario_route set_scenario_route", _ROUTE_SETTER),
    },
    f"{_PACKAGE}.scenario_base": {
        "logger": "the module's logger; a scenario logs with its own",
    },
    f"{_PACKAGE}.scenario_config": _each(
        "TrafficConfig TrafficManagerBackendConfig PedestrianParameters Spread "
        "VehicleParameters BicycleParameters",
        _TRAFFIC_BACKEND,
    ),
    f"{_PACKAGE}.traffic": {
        **_each(
            "NullTrafficBackend TrafficBackend TrafficBackendError "
            "TrafficBackendUnavailable TrafficConfig TrafficManagerBackend "
            "TrafficManagerBackendConfig TrafficParameterError "
            "VEHICLE_PARAMETER_BACKENDS "
            "PEDESTRIAN_PARAMETER_BACKENDS BICYCLE_PARAMETER_BACKENDS build_none "
            "build_sumo build_traffic_manager "
            "register_builtin_backends",
            _TRAFFIC_BACKEND,
        ),
        **_each(
            "BackendDriven DrivenEntity LaneChanging SettingSpeed TurningAtJunctions "
            "WalkedEntity",
            _TRAFFIC_PROTOCOL,
        ),
    },
    f"{_PACKAGE}.rerun": _each(
        "INSTALL_HINT RECORDING_SUFFIX RerunConversion RerunUnavailableError "
        "convert_outputs main report require_converter",
        "makes Rerun recordings of a run's trajectory logs with the "
        "autoware-carla-scenario-rerun package, which the runner does after a "
        "run (rerun.enabled), not a scenario; the model declares only the "
        "RerunConverter the checked runner hands each log to and "
        "recording_path, which names the recording it makes",
    ),
    f"{_PACKAGE}.t4": _each(
        "DATASET_SUFFIX INSTALL_HINT LIDAR_CHANNEL T4Settings T4UnavailableError "
        "active_recording camera_channel category_of dataset_path opencv_of_ftheta "
        "offer_driver_sensors offered_rear_axle require_writer withdraw_driver_sensors",
        "writes a run as a T4 dataset with the autoware-carla-scenario-t4 "
        "package, which the runner does (t4.enabled), not a scenario; the "
        "model declares only the T4Recorder and T4Recording the checked "
        "runner drives",
    ),
    f"{_PACKAGE}.t4_scene": _each(
        "LOG_SUFFIX MAP_SUFFIX SCENE_SUFFIX T4SceneExport map_resource "
        "trajectory_log_to_t4_scene write_t4_scene",
        "writes a run's trajectory log as a T4Scene, which the runner does "
        "after a run (t4scene.enabled) and scenario-t4scene does from the "
        "shell, not a scenario; the model declares only the T4SceneExporter "
        "the checked runner hands each log to",
    ),
    f"{_PACKAGE}.trajectory_recorder": {
        "SCHEMA_PATH": (
            "the path of the JSON Schema beside the Python module "
            "(`Path(__file__)`), a value the model, compiled elsewhere, has "
            "no counterpart of; it is for the log's readers, not a scenario"
        ),
        "DRAFT7_SCHEMA_PATH": (
            "the path of the schema's draft-07 copy beside the Python module "
            "(`Path(__file__)`), a value the model, compiled elsewhere, has "
            "no counterpart of; it is for the log's readers, not a scenario"
        ),
    },
    f"{_PACKAGE}.ui.models": _each(
        "ConditionNode ScenarioResultView SessionItem SessionSummary", _UI
    ),
}


# ---------------------------------------------------------------------------
# Members of the classes the model declares (#265)
# ---------------------------------------------------------------------------

_AS_DICT = (
    "returns a `dict[str, Any]` (what the runner writes to a result or a log "
    "as JSON), which Codon has no type for; a scenario reads the typed "
    "fields and properties instead"
)
_FROM_MAPPING = (
    "builds one from a `Mapping[str, Any]` (a document's YAML or a run's "
    "config), which Codon has no type for; the runner does, not a scenario"
)
_CONDITION_DETAILS = (
    "a condition's details for the runner's report, a `dict[str, Any]`, which "
    "Codon has no type for; a scenario reads a condition through check() and "
    "the properties of each condition"
)
_TRAFFIC_DRAWS = (
    "draws a road user's value with the backend's random stream: the backend "
    "draws for each vehicle and pedestrian, not a scenario, which hands the "
    "spread to a TrafficParametersAction"
)
_TRAFFIC_PARAMETER_PLUMBING = (
    "what the traffic backend and the TrafficParametersAction do with the "
    "parameters (refusing unsupported fields, merging an override); a "
    "scenario builds them and hands them to the action"
)
_SET_TRAFFIC_BACKEND = (
    "takes or returns the run's TrafficBackend, which is not modelled (see "
    "autoware_carla_scenario.TrafficBackend): the runner injects the backend "
    "the run selected; a scenario drives traffic through its actions and its "
    "entities' manoeuvres (set_desired_speed(), change_lane(), ...), which are "
    "modelled"
)
_BY_NAME = (
    "the model declares the class by name only, so the checked package's "
    "re-export resolves, and a scenario cannot build one (its constructor is "
    "a compile error): the runner builds it from the run's config; the "
    "library check compiles it from its source"
)
_ROS2_TRANSPORT = (
    "the ROS 2 transport the checked ros2_entity drives (cyclonedds), which "
    "the library check compiles from its source; a scenario has no ROS 2 "
    "transport to build"
)
_MAP_CACHE = (
    "the map cache's own (the checked maps.resolver's), which the library "
    "check compiles from its source; a scenario's maps are the run's, loaded "
    "before setup()"
)
_ANY_FIELD = (
    "holds a `dict[str, Any]` (a value per ODD attribute, or per key of the "
    "module's metadata, of any type), which Codon has no type for; a scenario "
    "reads the typed fields instead"
)
_SWEEP_FIELD = (
    "a Hydra mapping of the sweep (`dict[str, Any]`), which Codon has no type "
    "for; the sweeper reads it, not a scenario, which is handed its case's "
    "values through its own config"
)
_ROS2_CONFIG = (
    "a field of the ROS 2 transport's config, which the runner builds from the "
    "run's config: the model's constructor is a compile error, so a scenario "
    "holds none to read; the library check compiles ros2.config in this "
    "model's place"
)
_CALLABLE_ANY = (
    "a callable of `Any` (`Callable[[Any], Any]`, `Callable[[Any], bool]`), "
    "which Codon cannot hold as a value of one type; register_cover() checks "
    "the one a scenario passes"
)
_COVER_VALUES = (
    "the bucket values (`Iterable[Any]`, a `list[Any]` once built), which "
    "Codon has no type for; a scenario reads the buckets' labels (labels) "
    "or edges (edges)"
)
_ODD_RUN_VALUES = (
    "openodd's evaluation and sampling of an ODD over a run's values "
    "(`Mapping[str, Any]`, `dict[str, Any]`, OddVerdict), which the runner, "
    "the sweeper and odd.cli use; Codon has no type for the values, and an "
    "ODD written in Python does not call them"
)
_ODD_ARCHIVE = (
    "openodd's archive of an ODD loaded from files (`bytes`, its manifest "
    "and files as `dict[str, Any]`), which odd.cli and the report write; an "
    "ODD written in Python has none, and Codon has no type for them"
)
_REPORT_ROWS = (
    "openodd's rows of a merged report (MergedEntry, CaseRun, CellFlakiness, "
    "CellRow, OddExposure, WeightedOutcome), which have no model: the report tools read them, "
    "not a scenario or an ODD"
)
_RUN_ENVIRONMENT = (
    "the runner's record of the run's environment (json, `dict[str, Any]`); "
    "environment.codon models what the runner's _record_run reads"
)
_EGO_TYPE = (
    "the EgoVehicle subclass the scenario builds its ego from: a class held "
    "as a value, which Codon cannot hold (BaseScenario.__init__ sets it "
    "outside the check, under `if not TYPE_CHECKING:`); a scenario reads "
    "ego_requires_goal and create_ego(), which are modelled"
)
_PROBE = (
    "the attribute's probe, a callable of `Any` (`Callable[[Any], Any]`, or "
    "a measure's reader), which Codon cannot hold as a value of one type; "
    "OddAttribute() checks the one an ODD passes"
)
_MAP_FEATURES = (
    "route.geometry's MapFeatures, the lanelet2 queries the checked route "
    "modules share (the library check compiles them from their source); a "
    "scenario reads the route's lanelets (lanelets, junction_way()) instead"
)

_RUN_CLEANUP = (
    "what the runner's cleanup calls once the run is over, to stop and "
    "destroy a sensor a condition started listening through its own handle "
    "(#371); a scenario never calls it, and a condition that overrides it "
    "needs no declaration in the model to do so"
)


def _members(names: str, reason: str) -> dict[str, str]:
    return {name: reason for name in names.split()}


#: Python class name (the package's name of an openodd class: `OddCondition`
#: for openodd's `Condition`) -> public member -> why the model's class does
#: not declare it.  A member of a class is listed under that class or any
#: class of the package or openodd it derives from in Python (`BaseCondition`
#: lists `get_details` for every condition), so a key need not be a modelled
#: class itself (`BackendDriven`).
NOT_MODELLED_MEMBERS: dict[str, dict[str, str]] = {
    "AdversityActiveCondition": {
        "active": (
            "the adversities its last check found, as adversity_activity "
            "ActiveAdversity records, for a run's report; a scenario reads the "
            "condition's verdict, not them"
        ),
    },
    "PedestrianEntity": {
        "walk_path": (
            "the walk route_to() laid out, a trajectory.model ResolvedTrajectory "
            "the routing action paces a walker along; a scenario routes the "
            "pedestrian, it does not read its walk"
        ),
        "walk_along": (
            "lays a walk out whole as a trajectory.model ResolvedTrajectory, "
            "what the routing action does with a recorded walk; a scenario "
            "routes the pedestrian with a RoutingAction instead"
        ),
        "set_traffic_backend": _SET_TRAFFIC_BACKEND,
        "set_turn_back": (
            "how the walks it is given wait at a kerb and turn back, which the "
            "routing action sets from its turn_back_* thresholds; a scenario "
            "sets them on the RoutingAction"
        ),
    },
    "SpawnAction": {
        "apply": (
            "spawns at once with a blueprint library and a ground lookup shared "
            "by many props, what ConstructionZoneAction does; a scenario "
            "registers the action instead"
        ),
    },
    "AdversarialTrafficAction": {
        "choices": (
            "the adversities NADE sampled, as adversity.nade's NadeChoice, "
            "which holds an AdversarialCommand of the adversity package that a "
            "scenario does not import; read its weight instead (#397 makes it a "
            "metric)"
        ),
        "set_traffic_backend": _SET_TRAFFIC_BACKEND,
    },
    "StalledObjectAction": {
        "pose": (
            "the pose the action was given, one of three pose classes, which "
            "Codon has no union to return; a scenario keeps the pose it passed"
        ),
    },
    "NadeSettings": {
        "avoidance": (
            "adversity.avoidance's AvoidanceSettings, TeraSim's avoidance "
            "numbers, of the adversity package that a scenario does not import; "
            "a scenario leaves them at TeraSim's (the library check compiles "
            "them from their source)"
        ),
    },
    "AdversityFeasibleCondition": _members(
        "command scene",
        "what the adversarial actions read off the condition they fire on: an "
        "AdversarialCommand and the adversity Scene, of the adversity "
        "package, which a scenario does not import; the library check "
        "compiles them from their source",
    ),
    "BackendDriven": {"set_traffic_backend": _SET_TRAFFIC_BACKEND},
    "BaseAction": {"close": _RUN_CLEANUP},
    "BaseCondition": {
        **_members("get_details to_summary_dict", _CONDITION_DETAILS),
        "close": _RUN_CLEANUP,
    },
    "BaseScenario": {
        "close_conditions": _RUN_CLEANUP,
        "registered_actions": (
            "the runner's: what a run's backends are prepared from (prepare), "
            "after setup(), not a scenario's to call"
        ),
        **_members("set_traffic_backend traffic_backend", _SET_TRAFFIC_BACKEND),
        "ego_type": _EGO_TYPE,
    },
    "CarlaDriverEntity": _members(
        "actor attaches_to_existing_actor carried_actor_ids change_lane client "
        "config debug_markers debug_state destroy drive_count end_route finish_at follow_route "
        "forget_lane_change "
        "handshake is_initialized lane_change_finished on_scenario_end "
        "on_scenario_start on_tick release_from_traffic require_microscopic "
        "requires_goal resume_traffic route_to set_client set_desired_speed "
        "set_initial_speeds set_traffic_backend simulation_fidelity "
        "spawn stack_name termination_reason termination_requested "
        "turn_at_junction turn_indicator use_autopilot",
        _BY_NAME,
    ),
    "ConditionStatus": {"details": _CONDITION_DETAILS, "to_dict": _AS_DICT},
    "CoverageCollector": {"to_dict": _AS_DICT},
    "CoverageReport": {
        **_members(
            "cases cell_flakiness entries entry flaky_cells odds weighted_outcomes",
            _REPORT_ROWS,
        ),
        "to_dict": _AS_DICT,
    },
    "CoverItem": {
        **_members("expression ignore", _CALLABLE_ANY),
        "describe": _AS_DICT,
        "values": _COVER_VALUES,
    },
    "CrossItem": {"describe": _AS_DICT},
    "CycloneRos2Transport": _members(
        "close open poll poll_turn_indicators publish_image publish_state",
        _ROS2_TRANSPORT,
    ),
    "EgoDriverGrpcClient": _members(
        "camera_sensor_configs close_session config drive session_uuid "
        "start_session submit_egomotion_observation submit_image_observation "
        "submit_recording_ground_truth submit_route",
        _BY_NAME,
    ),
    "GitMapCache": _members("entry_dir remember_tip tip_file", _MAP_CACHE),
    "GrpcAutowareBridgeServer": _members(
        "client_address client_connected close config configure is_ready port "
        "reroute start take_turn_indicators",
        _BY_NAME,
    ),
    "HilsEgoEntity": _members(
        "actor attaches_to_existing_actor carried_actor_ids change_lane command "
        "config debug_markers destroy end_route finish_at follow_route forget_lane_change "
        "is_initialized lane_change_finished on_scenario_end on_scenario_start "
        "on_tick release_from_traffic require_microscopic requires_goal "
        "resume_traffic route_to set_client set_desired_speed set_initial_speeds "
        "simulation_fidelity spawn stack_name termination_reason "
        "termination_requested turn_at_junction use_autopilot",
        "the model declares only what the checked CommunicationLossAction "
        "calls on the HILS ego (disconnect, reconnect), and a scenario cannot "
        "build one (its constructor is a compile error): the runner builds it "
        "from the run's config (ego.entity=hils); entity.hils_entity imports "
        "socket, so the library check does not compile it",
    ),
    "MapSource": _members(
        "at directory name parse pinned repo_name uri with_path", _MAP_CACHE
    ),
    "OddAttribute": {"probe": _PROBE},
    "OddCondition": {"evaluate": _ODD_RUN_VALUES},
    "OddDefinition": {
        **_members("complete evaluate sample", _ODD_RUN_VALUES),
        **_members("archive files manifest", _ODD_ARCHIVE),
        "describe": _AS_DICT,
    },
    "OddModule": {"describe": _AS_DICT, "metadata": _ANY_FIELD},
    "Ros2EgoConfig": {
        **_members(
            "cameras child_frame_id clock_topic domain_id frame_id "
            "kinematic_state_topic on_stale rear_axle_offset_m sources "
            "speed_topic steering_status_topic turn_indicators_command_topic "
            "turn_indicators_report_topic",
            _ROS2_CONFIG,
        ),
        "from_mapping": _FROM_MAPPING,
    },
    "Ros2EgoTransport": _members(
        "close open poll poll_turn_indicators publish_image publish_state",
        _ROS2_TRANSPORT,
    ),
    "RouteCoverage": {"describe": _AS_DICT},
    "RouteFrame": {"features": _MAP_FEATURES},
    "RouteLanelet": {"describe": _AS_DICT, "values": _ANY_FIELD},
    "RouteMatch": {
        "from_config": _FROM_MAPPING,
        "to_config": _AS_DICT,
    },
    "RerunConverter": {
        **_members(
            "close strict wait",
            "waits for the conversions and reports them, which the runner's "
            "caller (examples.run) does once the runs are over, not the checked "
            "runner, which only submits each log",
        ),
        "records_sensors": (
            "whether rerun.lidar or rerun.cameras is set, which the converter's "
            "start_sensors checks itself; the checked runner calls "
            "start_sensors and stop_sensors whatever it is"
        ),
    },
    "T4SceneExporter": _members(
        "max_distance_m step_s",
        "the settings the exporter was made with, which only its own export() "
        "reads; the checked runner calls export() whatever they are",
    ),
    "T4Recorder": {
        "settings": (
            "the T4Settings the recorder was made with (the sensors, the "
            "rate), which only its own start() reads; the checked runner "
            "calls start() whatever they are"
        ),
    },
    "T4Recording": _members(
        "fed_by_driver feed_camera_jpeg feed_lidar",
        "what the driver policy's ego (entity.carla_driver_entity, itself "
        "excluded) hands the dataset, not the checked runner, which only "
        "records each tick and closes it",
    ),
    "RunEnvironment": _members(
        "by_name implementation lock machine packages pins platform python "
        "scenario_package summary to_dict",
        _RUN_ENVIRONMENT,
    ),
    "RunProgress": {
        "model_config": (
            "pydantic's configuration of the model (a `dict`), no field of it; "
            "ui/models.codon models the fields"
        )
    },
    "ScalarComparisonRule": {"to_dict": _AS_DICT},
    "ScenarioResult": {"to_dict": _AS_DICT},
    "Failure": {
        "to_dict": _AS_DICT,
        "from_dict": _FROM_MAPPING,
    },
    "SweepConfig": _members("bindings constraints odd_sample", _SWEEP_FIELD),
    "TrafficSinkAction": {"set_traffic_backend": _SET_TRAFFIC_BACKEND},
    "ConstructionZoneAction": {"set_traffic_backend": _SET_TRAFFIC_BACKEND},
    "TrafficSourceAction": {"set_traffic_backend": _SET_TRAFFIC_BACKEND},
    "TrafficParametersAction": {"set_traffic_backend": _SET_TRAFFIC_BACKEND},
    "Spread": {
        "describe": _AS_DICT,
        "draw": _TRAFFIC_DRAWS,
    },
    "VehicleParameters": {
        "describe": _AS_DICT,
        "from_mapping": _FROM_MAPPING,
        **_members(
            "check_supported overridden_by restricted_to set_fields",
            _TRAFFIC_PARAMETER_PLUMBING,
        ),
    },
    "PedestrianParameters": {
        "describe": _AS_DICT,
        "from_mapping": _FROM_MAPPING,
        **_members(
            "check_supported overridden_by set_fields", _TRAFFIC_PARAMETER_PLUMBING
        ),
    },
    "BicycleParameters": {
        "describe": _AS_DICT,
        "from_mapping": _FROM_MAPPING,
        **_members(
            "check_supported overridden_by set_fields spreads",
            _TRAFFIC_PARAMETER_PLUMBING,
        ),
    },
}


def defines_members(module: str) -> bool:
    """Whether a class of *module* is one whose public members the model
    must declare or list (#265): the package's, or openodd's, whose classes
    the package re-exports (OddModule, CoverItem, ..., #298)."""
    return any(module == m or module.startswith(m + ".") for m in _MEMBER_MODULES)


def not_modelled_member(
    cls: type, member: str, name: Optional[str] = None
) -> Optional[tuple[str, str]]:
    """``(owner, reason)`` when *member* of Python class *cls* is listed in
    :data:`NOT_MODELLED_MEMBERS`, under *cls* (or *name*, the name the
    package gives it, when that differs from its own: openodd's
    ``Condition`` is ``OddCondition``) or a class of the package or openodd
    it derives from (*owner*); ``None`` otherwise."""
    for klass in getattr(cls, "__mro__", ()):
        if not defines_members(getattr(klass, "__module__", "")):
            continue
        keys = [klass.__name__]
        if klass is cls and name is not None and name != klass.__name__:
            keys.insert(0, name)
        for key in keys:
            reason = NOT_MODELLED_MEMBERS.get(key, {}).get(member)
            if reason is not None:
                return key, reason
    return None
