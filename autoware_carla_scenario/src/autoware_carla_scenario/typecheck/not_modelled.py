"""Public names the Codon model leaves out on purpose (docs/typecheck.md).

The scenario check compiles a scenario against the model (``codon/``), so a
name a modelled module exports in Python (its ``__all__``, or the public
names it defines when it has none) and the model does not declare cannot be
imported by a checked scenario.  Each such name is listed here with the
reason, by module: :func:`.check.typecheck_scenario` refuses an import of one
with that reason before compiling, and ``test_typecheck_model.py`` fails when
a public name is neither declared by the model nor listed here, or when a
name listed here is declared by the model or no longer public in Python.

The same holds for the public methods, properties and class attributes of
the classes the model declares (#265): :data:`NOT_MODELLED_MEMBERS` lists,
with the reason, each one the model's class leaves out, and the check adds
that reason to Codon's "object has no attribute" error.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["NOT_MODELLED", "NOT_MODELLED_MEMBERS", "not_modelled_member"]

_PACKAGE = "autoware_carla_scenario"


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
    "the ODD registry's lookup (odd.registry: importlib, entry points), which "
    "the runner uses; an ODD written in Python registers itself with "
    "register_odd(), which is modelled"
)
_ROUTE_SEARCH = (
    "the route search, parsed from a document's YAML (parse_route_search() "
    "takes `Mapping[str, Any]`) into frozen dataclasses of variadic tuples, "
    "which Codon has no type for; a scenario reads the route its sweep "
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
            "TrafficManagerBackendConfig",
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
        "HOME_ENV LAUNCHER NIGHTLY_MACHINES NIGHTLY_URL install install_dir main",
        _CARLA_INSTALL,
    ),
    f"{_PACKAGE}.debug_markers": {
        "plan_markers": (
            "takes carla_driver_interface's Trajectory, a driver's plan, which "
            "only the driver-backed entities hold (checked from their source); "
            "the model of carla_driver_interface declares it for the driver "
            "modules, not for a scenario"
        ),
    },
    f"{_PACKAGE}.driver.egodriver_client": _each(
        "EGODRIVER_SERVICE_FULL_NAME MAX_MESSAGE_BYTES channel_options", _GRPC
    ),
    f"{_PACKAGE}.entity": {
        **_each("AutowareEgoEntity BridgedEgoEntity ExternallyDrivenEgo", _EGO_STACK),
        **_each(
            "clear_entities find_entity_by_role_name register_entity unregister_entity",
            _ENTITY_REGISTRY,
        ),
    },
    f"{_PACKAGE}.environment": {
        "write": (
            "writes the run's environment with json (`dict[str, Any]`), the "
            "runner's; environment.codon models what it records with a run"
        )
    },
    f"{_PACKAGE}.odd": {
        **_each(
            "DEFAULT_KNOBS GitSource GitSourceError OddDraws OddKnob OddVerdict "
            "OpenOddError UNDECIDED knobs_from_mapping load_odd_binding "
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
            "a submodule (openodd's backend; probe values of any type), not a "
            "name an ODD or a scenario imports; the model declares no "
            "submodules of odd",
        ),
    },
    f"{_PACKAGE}.ros2.cyclone": _each(
        "control_from_message odometry_message ros_time trajectory_from_message",
        _CYCLONE,
    ),
    f"{_PACKAGE}.route": {
        **_each(
            "JunctionSegmentSpec LaneSegmentSpec Range RouteSearchSpec "
            "RouteSegmentMatch parse_route_search",
            _ROUTE_SEARCH,
        ),
        **_each("clear_scenario_route set_scenario_route", _ROUTE_SETTER),
    },
    f"{_PACKAGE}.scenario_base": {
        "logger": "the module's logger; a scenario logs with its own",
    },
    f"{_PACKAGE}.scenario_config": _each(
        "TrafficConfig TrafficManagerBackendConfig", _TRAFFIC_BACKEND
    ),
    f"{_PACKAGE}.traffic": {
        **_each(
            "NullTrafficBackend TrafficBackend TrafficBackendError "
            "TrafficBackendUnavailable TrafficConfig TrafficManagerBackend "
            "TrafficManagerBackendConfig build_none build_sumo "
            "build_traffic_manager register_builtin_backends",
            _TRAFFIC_BACKEND,
        ),
        **_each(
            "BackendDriven DrivenEntity LaneChanging SettingSpeed TurningAtJunctions",
            _TRAFFIC_PROTOCOL,
        ),
    },
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
_SET_TRAFFIC_BACKEND = (
    "takes or returns the run's TrafficBackend, which is not modelled (see "
    "autoware_carla_scenario.TrafficBackend): the runner injects the backend "
    "the run selected; a scenario drives traffic through its actions and its "
    "entities' manoeuvres (set_desired_speed(), change_lane(), ...), which are "
    "modelled"
)
_ROUTE_SEGMENTS = (
    "returns a RouteSegmentMatch, a frozen dataclass of variadic tuples, "
    "which Codon has no type for (see autoware_carla_scenario.route); a "
    "scenario reads a junction's lanelets with "
    "scenario_route_frame().junction_way()"
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
_RUN_ENVIRONMENT = (
    "the runner's record of the run's environment (json, `dict[str, Any]`); "
    "environment.codon models what the runner's _record_run reads"
)


def _members(names: str, reason: str) -> dict[str, str]:
    return {name: reason for name in names.split()}


#: Python class name -> public member -> why the model's class does not
#: declare it.  A member of a class is listed under that class or any class
#: of the package it derives from in Python (`BaseCondition` lists
#: `get_details` for every condition), so a key need not be a modelled class
#: itself (`BackendDriven`).
NOT_MODELLED_MEMBERS: dict[str, dict[str, str]] = {
    "BackendDriven": {"set_traffic_backend": _SET_TRAFFIC_BACKEND},
    "BaseCondition": _members("get_details to_summary_dict", _CONDITION_DETAILS),
    "BaseScenario": _members(
        "set_traffic_backend traffic_backend", _SET_TRAFFIC_BACKEND
    ),
    "CarlaDriverEntity": _members(
        "actor attaches_to_existing_actor carried_actor_ids change_lane client "
        "config debug_markers debug_state destroy drive_count forget_lane_change "
        "handshake is_initialized lane_change_finished on_scenario_end "
        "on_scenario_start on_tick release_from_traffic requires_goal route_to "
        "set_client set_desired_speed set_initial_speeds set_traffic_backend "
        "spawn stack_name termination_reason termination_requested "
        "turn_at_junction turn_indicator use_autopilot",
        _BY_NAME,
    ),
    "ConditionStatus": {"to_dict": _AS_DICT},
    "CoverageCollector": {"to_dict": _AS_DICT},
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
        "start take_turn_indicators",
        _BY_NAME,
    ),
    "MapSource": _members(
        "at directory name parse pinned repo_name uri with_path", _MAP_CACHE
    ),
    "Ros2EgoConfig": {"from_mapping": _FROM_MAPPING},
    "Ros2EgoTransport": _members(
        "close open poll poll_turn_indicators publish_image publish_state",
        _ROS2_TRANSPORT,
    ),
    "RouteCoverage": {"describe": _AS_DICT},
    "RouteLanelet": {"describe": _AS_DICT},
    "RouteMatch": {
        "from_config": _FROM_MAPPING,
        "junction": _ROUTE_SEGMENTS,
        "junctions": _ROUTE_SEGMENTS,
        "to_config": _AS_DICT,
    },
    "RunEnvironment": _members(
        "lock pins scenario_package summary to_dict", _RUN_ENVIRONMENT
    ),
    "RunProgress": {
        "model_config": (
            "pydantic's configuration of the model (a `dict`), no field of it; "
            "ui/models.codon models the fields"
        )
    },
    "ScalarComparisonRule": {"to_dict": _AS_DICT},
    "ScenarioResult": {"to_dict": _AS_DICT},
    "TrafficSinkAction": {"set_traffic_backend": _SET_TRAFFIC_BACKEND},
    "TrafficSourceAction": {"set_traffic_backend": _SET_TRAFFIC_BACKEND},
}


def not_modelled_member(cls: type, member: str) -> Optional[tuple[str, str]]:
    """``(owner, reason)`` when *member* of Python class *cls* is listed in
    :data:`NOT_MODELLED_MEMBERS`, under *cls* or a class of the package it
    derives from (*owner*); ``None`` otherwise."""
    for klass in getattr(cls, "__mro__", ()):
        if not getattr(klass, "__module__", "").startswith(_PACKAGE):
            continue
        reason = NOT_MODELLED_MEMBERS.get(klass.__name__, {}).get(member)
        if reason is not None:
            return klass.__name__, reason
    return None
