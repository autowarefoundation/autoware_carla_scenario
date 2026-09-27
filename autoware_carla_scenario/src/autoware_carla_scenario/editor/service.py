"""Document mutations behind the editor's HTTP routes.

Routes parse a request and render a template; every change to a
:class:`~autoware_carla_scenario.authoring.models.ScenarioDocument` happens
here, so the rules about *what* an edit means live in one place and can be
tested without a web client.

Mutations are metadata-driven: adding a node seeds it from its spec's field
defaults and updating one parses the form through the same spec, so a newly
registered primitive is editable with no change to this module.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Sequence, get_args

from ..authoring.models import (
    ActionNode,
    BindingRef,
    ConditionNode,
    ConstraintNode,
    EgoDriver,
    Entity,
    GoalSpec,
    LaneletSlot,
    ScenarioDocument,
    SpawnSpec,
    as_action_phase,
    condition_refs,
)
from ..authoring.persistence import Draft, DraftStore
from ..authoring.registry import (
    default_params,
    get_action_spec,
    get_binding_spec,
    get_condition_spec,
    get_constraint_spec,
)
from ..authoring.starter import blank_document, new_document
from ..maps import (
    KNOWN_REPOSITORIES,
    MapCacheError,
    MapResolutionError,
    MapRepository,
    MapSource,
    MapSourceError,
    OpenDriveUnavailable,
    ensure_xodr,
    list_maps,
    pin_source,
    resolve_map,
)
from ..authoring.validator import ValidationReport, validate_document
from .forms import parse_params

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..maps import MapEntry

logger = logging.getLogger(__name__)

__all__ = [
    "EditorError",
    "EditorService",
    "as_int",
    "SLOT_FAIL",
    "SLOT_PASS",
    "condition_actions",
]

#: Condition slots that are not attached to an action.
SLOT_PASS = "pass"
SLOT_FAIL = "fail"


class EditorError(Exception):
    """Raised when a request asks for something the document cannot do."""


def _zip_directory(source: Path, archive: Path) -> None:
    """Zip *source* to *archive*, keeping the directory itself as the one root.

    Stored rather than deflated. The payload is a wheelhouse -- a couple of
    hundred megabytes of wheels, which are themselves deflate-compressed zips --
    so re-compressing it costs about eight seconds of the request thread to
    shave one percent off the download. ``shutil.make_archive`` has no way to
    say that, hence the explicit loop.
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                bundle.write(path, source.name / path.relative_to(source))


@contextmanager
def _map_errors(prefix: str = "") -> "Iterator[None]":
    """Turn any map problem into something the editor can show.

    Every map operation can fail four ways -- an unusable URI, an unreachable
    repository, a repository with no map in it, no OpenDRIVE to be had -- and
    all four mean the same thing to the person who pressed the button.  Deciding
    that once is what stops a fifth failure mode from escaping as a 500 because
    one method's ``except`` clause was not updated with the others.
    """
    try:
        yield
    except (
        MapSourceError,
        MapCacheError,
        MapResolutionError,
        OpenDriveUnavailable,
    ) as exc:
        raise EditorError(f"{prefix}: {exc}" if prefix else str(exc)) from exc


class EditorService:
    """Draft storage plus every document mutation the editor performs."""

    def __init__(self, store: DraftStore, export_dir: Path | None = None) -> None:
        """
        Args:
            store: Where drafts are read and written.
            export_dir: Where a finished export's wheelhouse ``.zip`` is staged
                until the browser has fetched it.  The editor hands packages to
                whoever is using it rather than leaving them on the machine it
                happens to run on -- over a LAN those are not the same machine
                -- so this is a holding area, not a destination anyone browses.
        """
        self.store = store
        self.export_dir = (
            Path(export_dir)
            if export_dir is not None
            else Path.cwd() / "scenario_packages"
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def archive_path(self, draft: Draft) -> Path:
        """Return where *draft*'s exported wheelhouse is staged."""
        return self.export_dir / f"{draft.document.id}-wheelhouse.zip"

    def export_archive(self, draft: Draft, **options: Any) -> Any:
        """Export *draft* and zip its wheelhouse for download.

        What comes back is the **wheelhouse**, not the uv project it was built
        from.  The project needs `uv`, `git` and a network to install; the
        wheelhouse needs pip and none of them, which is all the environment a
        scenario actually runs in is guaranteed to have.  The project is a build
        input, so it stays in the temporary directory and is removed with it.

        Only the archive outlives the request, so an export never leaves a
        half-written tree behind and re-exporting needs no ``force`` flag to
        overwrite one.

        Args:
            draft: The draft to export.
            **options: Forwarded to
                :func:`~autoware_carla_scenario.authoring.package_export.export_package`.

        Returns:
            The :class:`ExportResult`.  The archive itself is at
            :meth:`archive_path`, which the download route asks for directly.

        Raises:
            PackageExportError: If the export itself failed, or produced no
                wheelhouse.  Nothing is staged in that case.
        """
        from ..authoring.package_export import (  # noqa: PLC0415
            PackageExportError,
            export_package,
        )

        build_dir = Path(tempfile.mkdtemp(prefix="scenario-export-"))
        try:
            result = export_package(draft.document, build_dir, **options)
            if result.wheelhouse is None:
                # Only reachable by asking for an unlocked export, which this
                # form cannot; the log is carried anyway, since it is the only
                # thing that would explain it.
                raise PackageExportError(
                    "The export produced no wheelhouse, so there is nothing "
                    "that can be installed without uv and a network.",
                    log=result.log,
                )
            _zip_directory(result.wheelhouse.root, self.archive_path(draft))
            return result
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Draft lifecycle
    # ------------------------------------------------------------------

    def list_drafts(self) -> list[Draft]:
        """Return every stored draft, newest first."""
        return self.store.list()

    def create_draft(self, kind: str = "cut_in", title: str = "") -> Draft:
        """Create a draft from a starter document.

        Args:
            kind: ``"cut_in"`` for the worked example, anything else for the
                minimal document.
            title: Optional title override.
        """
        document = (
            new_document() if kind == "cut_in" else blank_document(title=title or "")
        )
        if title:
            document.title = title
        return self.store.create(document, title=document.title)

    def require_draft(self, draft_id: str) -> Draft:
        """Return the draft, or raise.

        Raises:
            EditorError: If no draft with that id is stored.
        """
        try:
            draft = self.store.get(draft_id)
        except ValueError as exc:
            raise EditorError(str(exc)) from exc
        if draft is None:
            raise EditorError(f"No draft named {draft_id!r}.")
        return draft

    def save(self, draft: Draft) -> Draft:
        """Normalise the layout and persist *draft*."""
        draft.document.sync_layout()
        return self.store.save(draft)

    def delete_draft(self, draft_id: str) -> bool:
        """Delete a draft.  Returns whether anything was removed."""
        return self.store.delete(draft_id)

    def validate(self, draft: Draft) -> ValidationReport:
        """Return the validation report for a draft's document."""
        return validate_document(draft.document)

    # ------------------------------------------------------------------
    # Scenario metadata
    # ------------------------------------------------------------------

    def update_scenario(
        self, document: ScenarioDocument, form: Mapping[str, Any]
    ) -> None:
        """Apply the scenario-level inspector form.

        Raises:
            EditorError: If a submitted value is not usable.
        """
        if "title" in form:
            document.title = str(form["title"]).strip() or document.title
        if "description" in form:
            document.description = str(form["description"]).strip()
        if "scenario_id" in form:
            new_id = str(form["scenario_id"]).strip()
            if new_id and new_id != document.id:
                try:
                    document.id = new_id
                except Exception as exc:  # pydantic validation
                    raise EditorError(f"Scenario id: {exc}") from exc
        if "timeout_seconds" in form:
            document.timeout_seconds = _as_float(
                form["timeout_seconds"], "Timeout", document.timeout_seconds
            )
        for attribute in ("group", "name", "source", "xodr_path", "lanelet2_path"):
            key = f"map_{attribute}"
            if key in form:
                setattr(document.map, attribute, str(form[key]).strip() or None)
        if document.map.source:
            # Normalise whatever was pasted -- a browser URL, most likely --
            # into the canonical form, so the field shows what will be stored
            # and two spellings of one map are one string.
            document.map.source = self._canonical_source(document.map.source)
        # ``group`` and ``name`` are plain strings, never None.
        document.map.group = document.map.group or "nishishinjuku"
        document.map.name = document.map.name or "Town10HD_Opt"
        if "map_no_3d_model_lanelet_ids" in form:
            from .forms import parse_int_list  # noqa: PLC0415

            parsed = parse_int_list(form["map_no_3d_model_lanelet_ids"])
            document.map.no_3d_model_lanelet_ids = (
                parsed if isinstance(parsed, list) else []
            )

    # ------------------------------------------------------------------
    # Traffic signals
    # ------------------------------------------------------------------
    #
    # Groups and controllers are addressed by their index in the document's
    # list, not by name.  A name is the obvious key and the wrong one: it is
    # editable in the same form, and a document mid-edit may legitimately hold
    # two groups called the same thing for as long as it takes to rename the
    # second.  Every mutation re-renders the panel, so an index is never stale
    # by the time the next one is posted.

    @staticmethod
    def _signal_group(document: ScenarioDocument, index: int) -> Any:
        groups = document.map.signal_groups
        if not 0 <= index < len(groups):
            raise EditorError("That signal group is no longer there.")
        return groups[index]

    @staticmethod
    def _signal_controller(document: ScenarioDocument, index: int) -> Any:
        controllers = document.map.traffic_signal_controllers
        if not 0 <= index < len(controllers):
            raise EditorError("That signal controller is no longer there.")
        return controllers[index]

    @staticmethod
    def _signal_phase(document: ScenarioDocument, index: int, phase_index: int) -> Any:
        controller = EditorService._signal_controller(document, index)
        if not 0 <= phase_index < len(controller.phases):
            raise EditorError("That phase is no longer there.")
        return controller.phases[phase_index]

    def add_signal_group(self, document: ScenarioDocument) -> None:
        """Declare one more movement on this map."""
        from ..authoring.models import SignalGroupRef  # noqa: PLC0415

        document.map.signal_groups.append(
            SignalGroupRef(name=_unused_name("group", _signal_group_names(document)))
        )

    def update_signal_group(
        self, document: ScenarioDocument, index: int, form: Mapping[str, Any]
    ) -> None:
        """Apply one signal group's form.

        A rename carries: every phase state naming the old name is rewritten,
        as is every ``conflicts_with`` entry pointing at it.  Leaving them to
        the validator would turn a rename into a screenful of errors about
        edits nobody made.
        """
        from .forms import parse_int_list  # noqa: PLC0415

        group = self._signal_group(document, index)
        if "name" in form:
            new_name = str(form["name"]).strip()
            if new_name and new_name != group.name:
                self._rename_signal_group(document, group.name, new_name)
                group.name = new_name
        if "lanelet2_regulatory_element_ids" in form:
            parsed = parse_int_list(form["lanelet2_regulatory_element_ids"])
            group.lanelet2_regulatory_element_ids = (
                parsed if isinstance(parsed, list) else []
            )
        if "conflicts_with" in form:
            names = set(_signal_group_names(document)) - {group.name}
            chosen = form["conflicts_with"]
            values = chosen if isinstance(chosen, list) else [chosen]
            group.conflicts_with = [
                str(value) for value in values if str(value) in names
            ]

    @staticmethod
    def _rename_signal_group(document: ScenarioDocument, old: str, new: str) -> None:
        """Point every reference to *old* at *new*."""
        for other in document.map.signal_groups:
            other.conflicts_with = [
                new if name == old else name for name in other.conflicts_with
            ]
        for controller in document.map.traffic_signal_controllers:
            for phase in controller.phases:
                for state in phase.states:
                    if state.group == old:
                        state.group = new

    def delete_signal_group(self, document: ScenarioDocument, index: int) -> None:
        """Remove a group, and every phase state and conflict naming it.

        The states go with it because a state whose group is gone sets nothing:
        leaving them behind would be a phase that reads as driving a movement
        this map no longer has.
        """
        group = self._signal_group(document, index)
        name = group.name
        document.map.signal_groups.pop(index)
        for other in document.map.signal_groups:
            other.conflicts_with = [n for n in other.conflicts_with if n != name]
        for controller in document.map.traffic_signal_controllers:
            for phase in controller.phases:
                phase.states = [s for s in phase.states if s.group != name]

    def add_signal_controller(self, document: ScenarioDocument) -> None:
        """Declare one more junction cycle, with a phase to start from.

        A controller with no phases is a validation error the moment it exists,
        so it is born with one rather than with a complaint attached.
        """
        from ..authoring.models import (  # noqa: PLC0415
            SignalControllerRef,
            SignalPhaseRef,
        )

        names = [c.name for c in document.map.traffic_signal_controllers]
        controller = SignalControllerRef(
            name=_unused_name("junction", names),
            phases=[SignalPhaseRef(name="phase_1")],
        )
        _fill_phase_states(document, controller.phases[0])
        document.map.traffic_signal_controllers.append(controller)

    def update_signal_controller(
        self, document: ScenarioDocument, index: int, form: Mapping[str, Any]
    ) -> None:
        """Apply one controller's own form -- its name and its offset."""
        controller = self._signal_controller(document, index)
        if "name" in form:
            new_name = str(form["name"]).strip()
            if new_name and new_name != controller.name:
                old = controller.name
                for other in document.map.traffic_signal_controllers:
                    if other.reference == old:
                        other.reference = new_name
                for path, node in _signal_controller_uses(document):
                    if str(node.params.get("controller") or "") == old:
                        node.params["controller"] = new_name
                controller.name = new_name
        if "reference" in form:
            reference = str(form["reference"]).strip()
            controller.reference = reference or None
        if "delay_seconds" in form:
            controller.delay_seconds = _as_float(
                form["delay_seconds"], "Start delay", controller.delay_seconds
            )
        if not controller.reference:
            # A delay measures from a reference; without one it is a number
            # that does nothing, and keeping it would be a validation error
            # about a field the author had just cleared.
            controller.delay_seconds = 0.0

    def delete_signal_controller(self, document: ScenarioDocument, index: int) -> None:
        """Remove a controller, and drop the offsets that measured from it."""
        controller = self._signal_controller(document, index)
        name = controller.name
        document.map.traffic_signal_controllers.pop(index)
        for other in document.map.traffic_signal_controllers:
            if other.reference == name:
                other.reference = None
                other.delay_seconds = 0.0

    def add_signal_phase(self, document: ScenarioDocument, index: int) -> None:
        """Add a step to a cycle, already naming every group it drives."""
        from ..authoring.models import SignalPhaseRef  # noqa: PLC0415

        controller = self._signal_controller(document, index)
        phase = SignalPhaseRef(
            name=_unused_name("phase", [p.name for p in controller.phases])
        )
        _fill_phase_states(document, phase)
        controller.phases.append(phase)

    def update_signal_phase(
        self,
        document: ScenarioDocument,
        index: int,
        phase_index: int,
        form: Mapping[str, Any],
    ) -> None:
        """Apply one phase's form: its name, its duration, and its colours.

        The colours arrive one per declared group, under ``state_<group>``.
        States naming a bare regulatory element are left exactly as they are:
        they are the escape hatch for a junction the groups cannot describe,
        they are written by hand, and a form that does not show them must not
        be able to delete them either.
        """
        from ..authoring.models import SignalStateRef  # noqa: PLC0415

        controller = self._signal_controller(document, index)
        phase = self._signal_phase(document, index, phase_index)

        if "name" in form:
            new_name = str(form["name"]).strip()
            if new_name and new_name != phase.name:
                old = phase.name
                for path, node in _signal_controller_uses(document):
                    if (
                        str(node.params.get("controller") or "") == controller.name
                        and str(node.params.get("signal_phase") or "") == old
                    ):
                        node.params["signal_phase"] = new_name
                phase.name = new_name
        if "duration_seconds" in form:
            phase.duration_seconds = _as_float(
                form["duration_seconds"], "Duration", phase.duration_seconds
            )

        kept = [state for state in phase.states if not state.group]
        by_group = {state.group: state for state in phase.states if state.group}
        rebuilt: list[Any] = []
        for group in document.map.signal_groups:
            key = f"state_{group.name}"
            if key in form:
                colour = str(form[key]).strip().lower()
            else:
                existing = by_group.get(group.name)
                colour = existing.state if existing is not None else "red"
            rebuilt.append(SignalStateRef(group=group.name, state=colour))
        phase.states = rebuilt + kept

    def delete_signal_phase(
        self, document: ScenarioDocument, index: int, phase_index: int
    ) -> None:
        """Remove one step of a cycle, and any card that named it."""
        controller = self._signal_controller(document, index)
        phase = self._signal_phase(document, index, phase_index)
        name = phase.name
        controller.phases.pop(phase_index)
        for path, node in _signal_controller_uses(document):
            if (
                str(node.params.get("controller") or "") == controller.name
                and str(node.params.get("signal_phase") or "") == name
            ):
                node.params["signal_phase"] = ""

    def move_signal_phase(
        self, document: ScenarioDocument, index: int, phase_index: int, delta: int
    ) -> None:
        """Shift a phase along its cycle.

        Order is semantics here, unlike the canvas's own move: a cycle is
        walked in the order it is written, so this is what puts the amber
        between the two greens.
        """
        controller = self._signal_controller(document, index)
        self._signal_phase(document, index, phase_index)
        target = phase_index + delta
        if not 0 <= target < len(controller.phases):
            return
        phases = controller.phases
        phases[phase_index], phases[target] = phases[target], phases[phase_index]

    # ------------------------------------------------------------------
    # Maps
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_source(uri: str) -> str:
        """Return *uri* in canonical form, or raise a message worth showing.

        Raises:
            EditorError: If it names no repository.
        """
        with _map_errors("Map source"):
            return MapSource.parse(uri).uri

    def map_repositories(self) -> tuple[MapRepository, ...]:
        """Return the map repositories the library offers before anyone types one."""
        return KNOWN_REPOSITORIES

    def use_map(self, document: ScenarioDocument, uri: str) -> None:
        """Point *document* at the map *uri* names, and download it.

        Naming a map and having it are one action here: a scenario cannot be
        edited against a map that is not on the machine, so choosing one from
        the library that left it undownloaded would only defer the same wait to
        the first thing that needed it.

        Raises:
            EditorError: If the map could not be fetched.
        """
        document.map.source = self._canonical_source(uri)
        # A source supersedes whatever files the document named before it;
        # leaving them would silently keep resolving to the old map.
        document.map.lanelet2_path = None
        document.map.xodr_path = None
        self.fetch_map(document)

    def fetch_map(self, document: ScenarioDocument, *, refresh: bool = False) -> str:
        """Download *document*'s map, and return a line saying what happened.

        Args:
            document: The scenario being edited.
            refresh: Fetch the repository again rather than trusting the cache.

        Raises:
            EditorError: If the document names no source, or it could not be
                resolved.
        """
        source = self._require_source(document)
        with _map_errors():
            resolved = resolve_map(source, refresh=refresh)

        # The directory name is the CARLA world name -- that is the convention
        # a published Autoware map follows -- so taking it saves the one piece
        # of the map config a source cannot otherwise supply.
        document.map.name = resolved.name
        where = "already on disk" if resolved.provisioned else "cached"
        commit = f" at {resolved.commit[:10]}" if resolved.commit else ""
        return f"{resolved.name} {where}{commit}."

    def pin_map(self, document: ScenarioDocument) -> str:
        """Rewrite the map source to the exact commit it resolves to.

        Raises:
            EditorError: If the document names no source, or the commit behind
                it could not be established.
        """
        source = self._require_source(document)
        with _map_errors():
            # Refreshed, so the pin names the repository's current tip rather
            # than whichever revision this machine happens to have cached.
            pinned = pin_source(source, refresh=True)
        document.map.source = pinned
        # The cache is keyed by ref, so the pinned URI addresses an entry the
        # branch's own checkout is not in.  Resolving it now means pinning
        # leaves the map ready rather than apparently un-downloaded -- and it
        # costs no network, because the cache adopts the entry already at that
        # commit.
        self.fetch_map(document)
        return f"Pinned to {MapSource.parse(pinned).ref[:10]}."

    def fetch_opendrive(
        self, document: ScenarioDocument, *, host: str, port: int
    ) -> str:
        """Read the map's OpenDRIVE from a CARLA server and cache it.

        Raises:
            EditorError: If the map is not downloaded yet, or CARLA could not
                be reached.
        """
        source = self._require_source(
            document,
            "OpenDRIVE is fetched for a map named by a source. This scenario "
            "names its files directly, so set the .xodr path.",
        )
        with _map_errors():
            path = ensure_xodr(
                resolve_map(source),
                map_name=document.map.name or None,
                host=host,
                port=port,
                refresh=True,
            )
        from . import map_preview  # noqa: PLC0415 -- imports this module back

        # The preview keys its parsed maps on the file paths, and the map just
        # gained one it did not have; without this the next preview would be
        # answered from a parse made when there was no OpenDRIVE.
        map_preview.clear_cache()
        return f"OpenDRIVE written to {path.name}."

    def browse_maps(self, uri: str, *, refresh: bool = False) -> list[MapEntry]:
        """Return the maps the repository *uri* names offers.

        Raises:
            EditorError: If the repository could not be read.
        """
        with _map_errors():
            return list(list_maps(uri, refresh=refresh))

    @staticmethod
    def _require_source(document: ScenarioDocument, message: str = "") -> str:
        """Return the document's map source, or say it has none.

        Raises:
            EditorError: If the document names no map source.
        """
        if not document.map.source:
            raise EditorError(message or "This scenario names no map source.")
        return document.map.source

    def create_draft_for_map(self, uri: str, title: str = "") -> Draft:
        """Create a draft whose map is the one *uri* names.

        Raises:
            EditorError: If the map could not be fetched.
        """
        draft = self.create_draft(kind="blank", title=title)
        self.use_map(draft.document, uri)
        if not title:
            draft.document.title = f"New scenario on {draft.document.map.name}"
        draft.title = draft.document.title
        return self.save(draft)

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    #: Id prefix per entity kind.  A walker called ``npc3`` reads as a vehicle
    #: in every log line and every condition that names it, so the kinds are
    #: told apart by their names as well as by their field.
    _ID_PREFIXES: ClassVar[dict[str, str]] = {
        "ego": "ego",
        "vehicle": "npc",
        "pedestrian": "walker",
    }

    def add_entity(self, document: ScenarioDocument, kind: str = "vehicle") -> Entity:
        """Append a new entity and return it."""
        if kind not in self._ID_PREFIXES:
            raise EditorError(f"Unknown entity kind {kind!r}.")
        if kind == "ego" and document.ego is not None:
            raise EditorError("The scenario already has an ego entity.")
        entity_id = _unique_entity_id(document, self._ID_PREFIXES[kind])
        entity = Entity(
            id=entity_id,
            # The blueprint follows from the kind; `Entity` fills it in.
            kind=kind,  # type: ignore[arg-type]
            title="Ego" if kind == "ego" else entity_id.upper(),
            spawn=SpawnSpec(lanelet_id=document.entities[0].spawn.lanelet_id)
            if document.entities
            else SpawnSpec(),
        )
        document.entities.append(entity)
        document.sync_layout()
        return entity

    def delete_entity(self, document: ScenarioDocument, entity_id: str) -> None:
        """Remove an entity, its actions, and every condition that referenced it.

        Leaving a dangling reference behind would turn a delete into a
        validation error the user did not cause, so the references go too.
        """
        entity = document.entity(entity_id)
        if entity is None:
            raise EditorError(f"No entity named {entity_id!r}.")
        document.entities.remove(entity)
        # Through delete_action, so the conditions that waited on the entity's
        # actions go too -- dropping the actions alone would leave references
        # to ids that no longer exist.
        for owned in [a for a in document.actions if a.actor == entity_id]:
            self.delete_action(document, owned.id)
        _purge_references(document, "entity", entity_id)
        document.sync_layout()

    def update_entity(
        self, document: ScenarioDocument, entity_id: str, form: Mapping[str, Any]
    ) -> None:
        """Apply the entity inspector form, including its spawn and its goal."""
        entity = document.entity(entity_id)
        if entity is None:
            raise EditorError(f"No entity named {entity_id!r}.")

        if "title" in form:
            entity.title = str(form["title"]).strip()
        if "vehicle_type" in form:
            entity.vehicle_type = (
                str(form["vehicle_type"]).strip() or entity.vehicle_type
            )
        if "initial_speed_kmh" in form:
            entity.initial_speed_kmh = _as_float(
                form["initial_speed_kmh"], "Initial speed", entity.initial_speed_kmh
            )

        if entity.kind == "ego" and "driven_by" in form:
            driven_by = str(form["driven_by"])
            if driven_by in get_args(EgoDriver):
                entity.driven_by = driven_by  # type: ignore[assignment]

        self._update_goal(entity, form)

        spawn = entity.spawn
        # Fixed or searched is not asked here: it is the same question every
        # lanelet field asks, and `set_lanelet_mode` is the one place that
        # answers it -- see `/draft/<id>/lanelet-mode`.
        if "spawn_lanelet_id" in form:
            spawn.lanelet_id = as_int(
                form["spawn_lanelet_id"], "Lanelet ID", spawn.lanelet_id
            )
        if "spawn_s_mode" in form:
            s_mode = str(form["spawn_s_mode"])
            if s_mode in ("fixed", "derived"):
                spawn.s.mode = s_mode  # type: ignore[assignment]
        if "spawn_s" in form:
            spawn.s.value = _as_float(form["spawn_s"], "Offset", spawn.s.value)

        # Only a derived offset needs a binding, and switching back to Fixed
        # leaves the old one in place: it is inert (nothing emits it) and it
        # means flipping the radio back does not lose what was configured.
        if spawn.s.mode == "derived":
            binding_type = str(form.get("binding_type") or "").strip()
            if not binding_type and spawn.s.binding is not None:
                binding_type = spawn.s.binding.type
            if not binding_type:
                binding_type = "stop_line_offset"
            spec = get_binding_spec(binding_type)
            if spec is None:
                raise EditorError(f"Unknown binding type {binding_type!r}.")
            existing = (
                spawn.s.binding.params
                if spawn.s.binding is not None and spawn.s.binding.type == binding_type
                else default_params(spec.fields)
            )
            params = dict(existing)
            params.update(_parse(spec.fields, form, prefix="binding_"))
            spawn.s.binding = BindingRef(type=binding_type, params=params)

    @staticmethod
    def _update_goal(entity: Entity, form: Mapping[str, Any]) -> None:
        """Apply the goal fields of the entity form.

        The stored goal is whatever the field says, including nothing: an ego
        the TrafficManager drives may have no destination, and blanking the
        lanelet is how it goes back to having none.  An ``autoware`` ego cleared
        that way is reported by
        :func:`~autoware_carla_scenario.authoring.validator.validate_document`
        rather than refused here -- an incomplete draft stays saveable.

        Only the ego carries a goal, so no other entity's form is read for one:
        the inspector does not offer the controls, and a goal stored elsewhere
        is a validation error.

        Raises:
            EditorError: If the goal offset is not a number.
        """
        if entity.kind != "ego" or "goal_lanelet_id" not in form:
            return
        raw = str(form["goal_lanelet_id"]).strip()
        if not raw:
            entity.goal = None
            return
        if entity.goal is None:
            entity.goal = GoalSpec()
        entity.goal.lanelet_id = as_int(raw, "Goal lanelet ID", entity.goal.lanelet_id)
        if "goal_s" in form:
            entity.goal.s = _as_float(form["goal_s"], "Goal offset", entity.goal.s)

    # ------------------------------------------------------------------
    # Lanelet searches
    # ------------------------------------------------------------------

    @staticmethod
    def require_slot(document: ScenarioDocument, slot_key: str) -> LaneletSlot:
        """Return the lanelet slot *slot_key* addresses, ready to be written to.

        Always with ``create=True``: every caller here is about to change the
        document, and the read-only view a missing goal otherwise answers with
        would swallow the change silently.

        Raises:
            EditorError: If nothing in the document is at that address.
        """
        slot = document.lanelet_slot(slot_key, create=True)
        if slot is None:
            raise EditorError(f"No lanelet field named {slot_key!r}.")
        return slot

    def set_lanelet_mode(
        self, document: ScenarioDocument, slot_key: str, mode: str
    ) -> str:
        """Pin a lanelet slot or hand it to the constraint sweeper.

        The constraint tree survives a flip back to Fixed: it is inert there --
        nothing emits it -- and keeping it means changing one's mind twice does
        not cost the search that was already written.

        Returns:
            The id of the object the inspector should show, so the picker
            re-opens on the thing that was just edited.
        """
        if mode not in ("fixed", "constraint_search"):
            raise EditorError(f"Unknown lanelet mode {mode!r}.")
        slot = self.require_slot(document, slot_key)
        slot.attach().mode = mode  # type: ignore[assignment]
        return slot.owner_id

    def add_constraint(
        self,
        document: ScenarioDocument,
        slot_key: str,
        type_id: str,
        parent_id: str | None = None,
    ) -> ConstraintNode:
        """Add a constraint to the search that chooses one lanelet."""
        slot = self.require_slot(document, slot_key)
        spec = get_constraint_spec(type_id)
        if spec is None:
            raise EditorError(f"Unknown constraint type {type_id!r}.")

        node = ConstraintNode(type=type_id, params=default_params(spec.fields))
        if parent_id:
            _, parent = find_constraint(document, parent_id)
            if parent is None:
                raise EditorError(f"No constraint named {parent_id!r}.")
            parent_spec = get_constraint_spec(parent.type)
            if parent_spec is None or not parent_spec.accepts_children:
                raise EditorError(f"{parent.type!r} does not take child constraints.")
            if (
                parent_spec.max_children is not None
                and len(parent.constraints) >= parent_spec.max_children
            ):
                raise EditorError(
                    f"{parent_spec.title} takes at most "
                    f"{parent_spec.max_children} child constraint(s)."
                )
            parent.constraints.append(node)
        else:
            slot.attach().constraints.append(node)
        return node

    def update_constraint(
        self, document: ScenarioDocument, node_id: str, form: Mapping[str, Any]
    ) -> None:
        """Apply the constraint inspector form."""
        _, node = find_constraint(document, node_id)
        if node is None:
            raise EditorError(f"No constraint named {node_id!r}.")
        spec = get_constraint_spec(node.type)
        if spec is None:
            raise EditorError(f"Unknown constraint type {node.type!r}.")
        node.params.update(_parse(spec.fields, form))

    def delete_constraint(self, document: ScenarioDocument, node_id: str) -> None:
        """Remove a constraint subtree from whichever search holds it."""
        for slot in document.lanelet_slots():
            roots = slot.choice.constraints
            for index, root in enumerate(roots):
                if root.id == node_id:
                    del roots[index]
                    return
                if root.remove(node_id):
                    return
        raise EditorError(f"No constraint named {node_id!r}.")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def add_action(
        self,
        document: ScenarioDocument,
        type_id: str,
        actor: str | None,
        phase: str | None = None,
    ) -> ActionNode:
        """Append an action to an actor's lane, in *phase* or the spec's default.

        The phase comes from the control that was used: the add button inside a
        lane's init cell asks for ``init``, because a card added there that
        landed on the tick loop would simply not appear where it was put.
        """
        spec = get_action_spec(type_id)
        if spec is None:
            raise EditorError(f"Unknown action type {type_id!r}.")
        if spec.actor_required and not actor:
            raise EditorError(f"{spec.title} needs an actor.")
        if actor and document.entity(actor) is None:
            raise EditorError(f"No entity named {actor!r}.")

        action = ActionNode(
            type=type_id,
            title=spec.title,
            # Refused rather than stored: an environment action is performed by
            # the world, and the track it is drawn in has to be the one the
            # runtime actually acts on.
            actor=None if spec.scope == "environment" else (actor or None),
            params=default_params(spec.fields),
            phase=as_action_phase(phase) or spec.default_phase,
        )
        document.actions.append(action)
        document.sync_layout()
        return action

    def update_action(
        self, document: ScenarioDocument, action_id: str, form: Mapping[str, Any]
    ) -> None:
        """Apply the action inspector form."""
        action = document.action(action_id)
        if action is None:
            raise EditorError(f"No action named {action_id!r}.")
        spec = get_action_spec(action.type)
        if spec is None:
            raise EditorError(f"Unknown action type {action.type!r}.")

        if "title" in form:
            action.title = str(form["title"]).strip()
        if "actor" in form and spec.scope != "environment":
            actor = str(form["actor"]).strip()
            if actor and document.entity(actor) is None:
                raise EditorError(f"No entity named {actor!r}.")
            action.actor = actor or None
        if "phase" in form:
            requested = as_action_phase(str(form["phase"]))
            if requested is not None:
                # Refused rather than dropped: the trigger is something the
                # author wrote, and deleting it to satisfy a phase change would
                # be a silent edit they did not ask for.
                if requested == "init" and action.trigger is not None:
                    raise EditorError(
                        f"{action.title or action.type} has a trigger, and the "
                        "initialization phase runs once and begins immediately, "
                        "so nothing there can wait. Remove the trigger first."
                    )
                action.phase = requested
        action.once = "once" in form
        action.params.update(_parse(spec.fields, form))

    def delete_action(self, document: ScenarioDocument, action_id: str) -> None:
        """Remove an action, its trigger, and every condition that waited on it.

        Leaving a dangling reference behind would turn a delete into a
        validation error the user did not cause -- and one that blocks
        compilation and export until they hunt down each dependent condition by
        hand.  Entity deletion already drops its references; an action
        reference is no different.
        """
        action = document.action(action_id)
        if action is None:
            raise EditorError(f"No action named {action_id!r}.")
        document.actions.remove(action)
        document.ui.nodes.pop(action_id, None)

        _purge_references(document, "action", action_id)
        document.sync_layout()

    def move_action(
        self, document: ScenarioDocument, action_id: str, delta: int
    ) -> None:
        """Shift an action one step along its lane.

        This writes ``ui.column_hint`` only -- what an action does and when it
        fires are its own type and trigger, so moving a card never changes what
        the scenario does.

        A card moves into any step, empty or occupied.  A step is a *set* of
        actions rather than a slot for one: everything is armed from the first
        tick, so actions that nothing sequences really do run alongside each
        other, and they are drawn stacked to say so.

        The one thing a move may not do is put an action level with, or ahead
        of, something its trigger waits on -- within a step nothing is ordered,
        so that would draw a dependency the runtime cannot honour.  The layout
        is repaired afterwards rather than the move refused, so a card lands as
        close to where it was aimed as its dependencies allow.
        """
        action = document.action(action_id)
        if action is None:
            raise EditorError(f"No action named {action_id!r}.")
        column = document.ui.column_of(action.id)
        target = max(0, column + delta)
        if target == column:
            return
        document.ui.set_column(action.id, target)
        document.enforce_dependency_order()

    def reorder_actors(self, document: ScenarioDocument, order: list[str]) -> None:
        """Set the swimlane order.  Presentation only."""
        known = [e.id for e in document.entities]
        document.ui.actor_order = [e for e in order if e in known]
        document.sync_layout()

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def add_condition(
        self, document: ScenarioDocument, slot: str, type_id: str
    ) -> ConditionNode:
        """Add a condition into *slot*.

        Slots are ``trigger:<action_id>`` (the action's trigger),
        ``node:<node_id>`` (a child of a composition), ``pass`` or ``fail``.
        Attaching a second condition to an action whose trigger is a single leaf
        wraps both in an ``ALL`` -- the reading the swimlane already implies.
        """
        spec = get_condition_spec(type_id)
        if spec is None:
            raise EditorError(f"Unknown condition type {type_id!r}.")
        node = ConditionNode(type=type_id, params=default_params(spec.fields))

        if slot == SLOT_PASS:
            document.assertions.pass_conditions.append(node)
            return node
        if slot == SLOT_FAIL:
            document.assertions.fail_conditions.append(node)
            return node

        target, _, identifier = slot.partition(":")
        if target == "trigger":
            action = document.action(identifier)
            if action is None:
                raise EditorError(f"No action named {identifier!r}.")
            if not action.takes_trigger:
                raise EditorError(
                    f"{action.title or action.type} is in the initialization "
                    "phase, which runs once and begins immediately, so there is "
                    "nothing for a condition to wait for. Move it onto the tick "
                    "loop to give it a trigger."
                )
            action.trigger = _attach_trigger(action.trigger, node)
            return node
        if target == "node":
            parent = document.condition(identifier)
            if parent is None:
                raise EditorError(f"No condition named {identifier!r}.")
            parent_spec = get_condition_spec(parent.type)
            if parent_spec is None or not parent_spec.accepts_children:
                raise EditorError(f"{parent.type!r} does not take child conditions.")
            if (
                parent_spec.max_children is not None
                and len(parent.children) >= parent_spec.max_children
            ):
                raise EditorError(
                    f"{parent_spec.title} already has its "
                    f"{parent_spec.max_children} child condition(s)."
                )
            parent.children.append(node)
            return node

        raise EditorError(f"Unknown condition slot {slot!r}.")

    def update_condition(
        self, document: ScenarioDocument, node_id: str, form: Mapping[str, Any]
    ) -> None:
        """Apply the condition inspector form."""
        node = document.condition(node_id)
        if node is None:
            raise EditorError(f"No condition named {node_id!r}.")
        spec = get_condition_spec(node.type)
        if spec is None:
            raise EditorError(f"Unknown condition type {node.type!r}.")
        node.params.update(_parse(spec.fields, form))

    def delete_condition(self, document: ScenarioDocument, node_id: str) -> None:
        """Remove a condition subtree from wherever it sits."""
        for action in document.actions:
            trigger = action.trigger
            if trigger is None:
                continue
            if trigger.id == node_id:
                action.trigger = None
                return
            if trigger.remove(node_id):
                return
        for bucket in (
            document.assertions.pass_conditions,
            document.assertions.fail_conditions,
        ):
            for index, root in enumerate(bucket):
                if root.id == node_id:
                    del bucket[index]
                    return
                if root.remove(node_id):
                    return
        raise EditorError(f"No condition named {node_id!r}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(fields: Any, form: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Parse the subset of *form* that names fields, honouring an optional prefix.

    Raises:
        EditorError: If a value cannot be parsed into its declared type.
    """
    if prefix:
        stripped = {
            key[len(prefix) :]: value
            for key, value in form.items()
            if key.startswith(prefix)
        }
    else:
        stripped = dict(form)
    present = [f for f in fields if f.name in stripped or f.kind == "bool"]
    try:
        return parse_params(present, stripped)
    except ValueError as exc:
        raise EditorError(str(exc)) from exc


def _signal_group_names(document: ScenarioDocument) -> "list[str]":
    """Every declared group name, in declaration order."""
    return [group.name for group in document.map.signal_groups]


def _unused_name(stem: str, taken: "Sequence[str]") -> str:
    """Return ``stem_1``, ``stem_2``... -- the first one not already in use.

    A new row needs a name it can be referred to by before anyone has typed
    one, and two rows sharing a name is a validation error the author did not
    make.
    """
    existing = set(taken)
    index = 1
    while f"{stem}_{index}" in existing:
        index += 1
    return f"{stem}_{index}"


def _fill_phase_states(document: ScenarioDocument, phase: Any) -> None:
    """Give *phase* one state per declared group, all red.

    A phase states every group its controller drives, which is what makes
    applying it put the junction into a known whole.  Starting from all-red
    means a half-filled cycle is a junction stopped rather than one letting
    two crossing movements through.
    """
    from ..authoring.models import SignalStateRef  # noqa: PLC0415

    phase.states = [
        SignalStateRef(group=name, state="red")
        for name in _signal_group_names(document)
    ]


def _signal_controller_uses(document: ScenarioDocument) -> "list[tuple[str, Any]]":
    """Every action and condition of the junction type, wherever it is written.

    The same walk the validator does, for the same reason: a rename has to
    reach a card whether it sits in a trigger, a nested composition or an
    assertion.
    """
    found: "list[tuple[str, Any]]" = []

    def walk(path: str, node: Any) -> None:
        if node.type == "traffic_signal_controller":
            found.append((path, node))
        for index, child in enumerate(node.children):
            walk(f"{path}.children[{index}]", child)

    for index, action in enumerate(document.actions):
        path = f"actions[{index}]"
        if action.type == "traffic_signal_controller":
            found.append((path, action))
        if action.trigger is not None:
            walk(f"{path}.trigger", action.trigger)
    for index, condition in enumerate(document.assertions.pass_conditions):
        walk(f"assertions.pass[{index}]", condition)
    for index, condition in enumerate(document.assertions.fail_conditions):
        walk(f"assertions.fail[{index}]", condition)
    return found


def _as_float(raw: Any, label: str, fallback: float) -> float:
    """Return *raw* as a float, or raise a user-facing error."""
    text = str(raw).strip()
    if not text:
        return fallback
    try:
        return float(text)
    except ValueError as exc:
        raise EditorError(f"{label} must be a number, got {raw!r}.") from exc


def as_int(raw: Any, label: str, fallback: int) -> int:
    """Return *raw* as an int, or raise a user-facing error."""
    text = str(raw).strip()
    if not text:
        return fallback
    try:
        return int(text)
    except ValueError as exc:
        raise EditorError(f"{label} must be a whole number, got {raw!r}.") from exc


def _unique_entity_id(document: ScenarioDocument, stem: str) -> str:
    """Return an entity id based on *stem* that nothing else uses.

    Non-ego entities are always numbered (``npc1``, ``npc2``, ...) so that ids
    line up with the ``npc<N>`` CARLA role names the compiler assigns.
    """
    taken = {e.id for e in document.entities}
    if stem == "ego":
        if stem not in taken:
            return stem
    index = 1
    while f"{stem}{index}" in taken:
        index += 1
    return f"{stem}{index}"


def condition_actions(node: ConditionNode) -> list[str]:
    """Return the action ids *node* waits on, for the canvas's causal links.

    A thin name over :func:`~autoware_carla_scenario.authoring.models.condition_refs`
    so the template global reads as what the canvas wants.  The link data is the
    document's own reference: card positions are ``ui.column_hint``, which the
    compiler never reads, so inferring causality from them would draw a line
    that moving a card could invent or erase.
    """
    return condition_refs(node, "action")


def _purge_references(document: ScenarioDocument, kind: str, target: str) -> None:
    """Drop every trigger and assertion that names *target* through a *kind* field.

    One purge for both delete paths: an entity and an action are referenced the
    same way, so a third kind of reference cannot be remembered in one deletion
    and forgotten in the other.
    """

    def names_target(root: ConditionNode) -> bool:
        return any(target in condition_refs(node, kind) for node in root.walk())

    for action in document.actions:
        if action.trigger is not None and names_target(action.trigger):
            action.trigger = None
    assertions = document.assertions
    assertions.pass_conditions = [
        c for c in assertions.pass_conditions if not names_target(c)
    ]
    assertions.fail_conditions = [
        c for c in assertions.fail_conditions if not names_target(c)
    ]


def _attach_trigger(
    existing: ConditionNode | None, node: ConditionNode
) -> ConditionNode:
    """Return the action's new trigger after adding *node* to *existing*."""
    if existing is None:
        return node
    spec = get_condition_spec(existing.type)
    if spec is not None and spec.kind == "composite":
        existing.children.append(node)
        return existing
    return ConditionNode(type="all", children=[existing, node])


def find_constraint(
    document: ScenarioDocument, node_id: str
) -> tuple[str | None, ConstraintNode | None]:
    """Find a constraint by id, with the id of the object whose search holds it.

    Every lanelet in a document may be searched for, not just a spawn, so the
    owner is whatever the inspector selects to get back to the search: an
    entity, an action, or a condition.
    """
    for slot in document.lanelet_slots():
        for root in slot.choice.constraints:
            for candidate in root.walk():
                if candidate.id == node_id:
                    return slot.owner_id, candidate
    return None, None
