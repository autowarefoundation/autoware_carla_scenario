"""Authoring: the Scenario IR, its metadata, and the road from IR to a package.

The pipeline this package implements::

    ScenarioDocument      the canonical, declarative scenario (models)
        |                 described by ActionSpec / ConditionSpec / ... (registry)
        v
    validate_document     metadata-driven checks (validator)
        v
    compile_document      typed, resolved plan with no CARLA import (compiler)
        v
    builders              the framework's own BaseAction / BaseCondition
        v
    DeclarativeScenario   a BaseScenario like any other

and, sideways from the document, :func:`export_package`, which writes a
reproducible Scenario Package.

Importing this package pulls in neither CARLA nor lanelet2, so the Scenario
Editor -- which never talks to a simulator -- can use all of it.
"""

from __future__ import annotations

from .compiler import (
    BuildContext,
    CompilationError,
    CompiledScenario,
    compile_document,
)
from .hydra_config import build_scenario_config, dump_scenario_config
from .models import (
    ActionNode,
    Assertions,
    BindingRef,
    ConditionNode,
    ConstraintNode,
    Entity,
    MapRef,
    ScenarioDocument,
    SpawnSpec,
    SValue,
    UiLayout,
)
from .package_export import ExportResult, PackageExportError, export_package
from .persistence import Draft, DraftStore, load_document, save_document
from .registry import (
    ActionSpec,
    BindingSpec,
    ConditionSpec,
    ConditionVisual,
    ConstraintSpec,
    FieldSpec,
    action_specs,
    binding_specs,
    condition_specs,
    constraint_specs,
    register_action_spec,
    register_binding_spec,
    register_condition_spec,
    register_constraint_spec,
)
from .starter import blank_document, new_document
from .validator import Issue, ValidationReport, validate_document

__all__ = [
    "ActionNode",
    "ActionSpec",
    "Assertions",
    "BindingRef",
    "BindingSpec",
    "BuildContext",
    "CompilationError",
    "CompiledScenario",
    "ConditionNode",
    "ConditionSpec",
    "ConditionVisual",
    "ConstraintNode",
    "ConstraintSpec",
    "Draft",
    "DraftStore",
    "Entity",
    "ExportResult",
    "FieldSpec",
    "Issue",
    "MapRef",
    "PackageExportError",
    "SValue",
    "ScenarioDocument",
    "SpawnSpec",
    "UiLayout",
    "ValidationReport",
    "action_specs",
    "binding_specs",
    "blank_document",
    "build_scenario_config",
    "compile_document",
    "condition_specs",
    "constraint_specs",
    "dump_scenario_config",
    "export_package",
    "load_document",
    "new_document",
    "register_action_spec",
    "register_binding_spec",
    "register_condition_spec",
    "register_constraint_spec",
    "save_document",
    "validate_document",
]
