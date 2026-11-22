"""Scenario Editor: what the page does around an edit (#2345).

Two tabs on one draft must not silently undo each other, one tab's quick edits
must not trip over each other, the top bar must follow a rename, and every
inspector must be usable without seeing it: each control named, each icon-only
button labelled. The typing that survives a swap is ``editor.js``'s and is
checked in a browser, not here.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.authoring.persistence import Draft, DraftStore
from autoware_carla_scenario.authoring.registry import (
    action_specs,
    condition_specs,
    constraint_specs,
)
from autoware_carla_scenario.editor.app import create_app
from autoware_carla_scenario.editor.routes import CLIENT_HEADER, REVISION_FIELD
from autoware_carla_scenario.editor.service import EditorError, EditorService

_REVISION = re.compile(r'name="draft_revision" value="(\d+)"')


@pytest.fixture
def client(tmp_path: Path, at_repo_root: Path) -> TestClient:
    return TestClient(
        create_app(draft_dir=tmp_path / "drafts", export_dir=tmp_path / "packages"),
        base_url="http://127.0.0.1",
    )


def _new_draft(client: TestClient, kind: str = "cut_in") -> str:
    response = client.post(
        "/new", data={"kind": kind, "title": "Cut in"}, follow_redirects=False
    )
    assert response.status_code == 303
    return str(response.headers["location"].rsplit("/", 1)[-1])


@pytest.fixture
def draft_id(client: TestClient) -> str:
    return _new_draft(client)


def _service(client: TestClient) -> EditorService:
    service = client.app.state.service  # type: ignore[attr-defined]
    assert isinstance(service, EditorService)
    return service


def _stored(client: TestClient, draft_id: str) -> Draft:
    draft = _service(client).store.get(draft_id)
    assert draft is not None
    return draft


def _revision(page: str) -> str:
    """The one draft revision every inspector form on *page* carries."""
    found = set(_REVISION.findall(page))
    assert len(found) == 1, f"forms disagree on the revision: {found}"
    return found.pop()


class TestDraftRevision:
    def test_a_save_bumps_the_revision_and_stores_it(
        self, tmp_path: Path, client: TestClient, draft_id: str
    ) -> None:
        before = _stored(client, draft_id).revision
        page = client.get(f"/draft/{draft_id}/inspector/scenario").text
        assert _revision(page) == str(before)
        client.post(
            f"/draft/{draft_id}/scenario",
            data={"title": "Renamed", REVISION_FIELD: str(before)},
        )
        # Read back from disk by a store of its own: the revision is persisted.
        reread = DraftStore(tmp_path / "drafts").get(draft_id)
        assert reread is not None and reread.revision == before + 1

    def test_a_draft_written_before_revisions_reads_as_zero(self) -> None:
        raw = {"id": "draft_x", "document": {}}
        assert Draft.from_dict(raw).revision == 0
        assert Draft.from_dict({**raw, "revision": "bogus"}).revision == 0
        assert Draft.from_dict({**raw, "revision": 7}).revision == 7

    def test_every_inspector_form_carries_the_revision(
        self, client: TestClient, draft_id: str
    ) -> None:
        for object_id in ("scenario", "ego", "npc1"):
            page = client.get(f"/draft/{draft_id}/inspector/{object_id}").text
            forms = re.findall(r"<form\b[^>]*change delay[^>]*>", page)
            assert forms, object_id
            assert page.count('name="draft_revision"') >= len(forms), object_id


class TestStaleForms:
    def test_a_form_from_another_tab_is_refused(
        self, client: TestClient, draft_id: str
    ) -> None:
        # Two tabs open the same entity.
        page = client.get(f"/draft/{draft_id}/inspector/npc1").text
        stale = _revision(page)
        speed = _stored(client, draft_id).document.entity("npc1")
        assert speed is not None
        # Tab A changes the speed...
        first = client.post(
            f"/draft/{draft_id}/entity/npc1",
            data={"initial_speed_kmh": "42", REVISION_FIELD: stale},
            headers={CLIENT_HEADER: "tab-a"},
        )
        assert first.status_code == 200
        # ...and tab B, still showing the old form, posts all of its fields.
        second = client.post(
            f"/draft/{draft_id}/entity/npc1",
            data={
                "title": "Renamed in B",
                "initial_speed_kmh": str(speed.initial_speed_kmh),
                REVISION_FIELD: stale,
            },
            headers={CLIENT_HEADER: "tab-b"},
        )
        assert second.status_code == 409
        # What editor.js reads out of the answer for its notice.
        assert "data-error-heading" in second.text
        assert "Changed elsewhere" in second.text
        assert "Reload the page" in html.unescape(second.text)
        stored = _stored(client, draft_id).document.entity("npc1")
        assert stored is not None
        assert stored.initial_speed_kmh == 42
        assert stored.title != "Renamed in B"

    def test_a_form_without_a_page_token_is_refused_once_stale(
        self, client: TestClient, draft_id: str
    ) -> None:
        stale = _revision(client.get(f"/draft/{draft_id}/inspector/scenario").text)
        client.post(
            f"/draft/{draft_id}/scenario", data={"title": "One", REVISION_FIELD: stale}
        )
        again = client.post(
            f"/draft/{draft_id}/scenario", data={"title": "Two", REVISION_FIELD: stale}
        )
        assert again.status_code == 409
        assert _stored(client, draft_id).document.title == "One"

    @pytest.mark.parametrize("bad", ["", "x", "-1", "999999"])
    def test_a_revision_that_is_not_a_past_one_is_refused(
        self, client: TestClient, draft_id: str, bad: str
    ) -> None:
        response = client.post(
            f"/draft/{draft_id}/scenario",
            data={"title": "Forged", REVISION_FIELD: bad},
            headers={CLIENT_HEADER: "tab-a"},
        )
        assert response.status_code == 409
        assert _stored(client, draft_id).document.title != "Forged"

    def test_a_post_without_a_revision_is_not_checked(
        self, client: TestClient, draft_id: str
    ) -> None:
        # Buttons and scripted posts carry no revision, and stay as they were.
        response = client.post(f"/draft/{draft_id}/scenario", data={"title": "Bare"})
        assert response.status_code == 200
        assert _stored(client, draft_id).document.title == "Bare"


class TestOneTabsQuickEdits:
    def test_sequential_edits_each_carry_the_new_revision(
        self, client: TestClient, draft_id: str
    ) -> None:
        page = client.get(f"/draft/{draft_id}/inspector/scenario").text
        for title in ("First", "Second", "Third"):
            revision = _revision(page)
            response = client.post(
                f"/draft/{draft_id}/scenario",
                data={"title": title, REVISION_FIELD: revision},
            )
            assert response.status_code == 200, title
            page = response.text
            # The swap numbers every form afresh.
            assert _revision(page) == str(int(revision) + 1)
        assert _stored(client, draft_id).document.title == "Third"

    def test_two_quick_edits_from_one_tab_both_land(
        self, client: TestClient, draft_id: str
    ) -> None:
        # The second change is posted before the answer to the first is
        # swapped in, so it still carries the revision the first one started
        # from. Every save since is this tab's own, so it is not stale.
        revision = _revision(client.get(f"/draft/{draft_id}/inspector/scenario").text)
        tab = {CLIENT_HEADER: "tab-a"}
        first = client.post(
            f"/draft/{draft_id}/scenario",
            data={"title": "Quick", "description": "", REVISION_FIELD: revision},
            headers=tab,
        )
        second = client.post(
            f"/draft/{draft_id}/scenario",
            data={"title": "Quick", "description": "abc", REVISION_FIELD: revision},
            headers=tab,
        )
        assert first.status_code == 200
        assert second.status_code == 200
        document = _stored(client, draft_id).document
        assert (document.title, document.description) == ("Quick", "abc")

    def test_a_quick_edit_after_another_tabs_save_is_refused(
        self, client: TestClient, draft_id: str
    ) -> None:
        revision = _revision(client.get(f"/draft/{draft_id}/inspector/scenario").text)
        client.post(
            f"/draft/{draft_id}/scenario",
            data={"title": "A", REVISION_FIELD: revision},
            headers={CLIENT_HEADER: "tab-a"},
        )
        client.post(
            f"/draft/{draft_id}/scenario",
            data={"description": "from B", REVISION_FIELD: str(int(revision) + 1)},
            headers={CLIENT_HEADER: "tab-b"},
        )
        late = client.post(
            f"/draft/{draft_id}/scenario",
            data={"title": "A", "description": "", REVISION_FIELD: revision},
            headers={CLIENT_HEADER: "tab-a"},
        )
        assert late.status_code == 409
        assert _stored(client, draft_id).document.description == "from B"

    def test_save_draft_does_not_strand_the_tab_that_pressed_it(
        self, client: TestClient, draft_id: str
    ) -> None:
        # Save Draft answers with a status line, so the forms keep the revision
        # from before it; the tab's next edit must still go through.
        revision = _revision(client.get(f"/draft/{draft_id}/inspector/scenario").text)
        tab = {CLIENT_HEADER: "tab-a"}
        assert client.post(f"/draft/{draft_id}/save", headers=tab).status_code == 200
        response = client.post(
            f"/draft/{draft_id}/scenario",
            data={"title": "After save", REVISION_FIELD: revision},
            headers=tab,
        )
        assert response.status_code == 200
        assert _stored(client, draft_id).document.title == "After save"


class TestHeaderFollowsTheDocument:
    def test_a_rename_updates_the_crumb_and_the_tab_title(
        self, client: TestClient, draft_id: str
    ) -> None:
        page = client.get(f"/draft/{draft_id}").text
        assert page.count('id="editor-crumb"') == 1
        assert "hx-swap-oob" not in page
        revision = _revision(page)
        body = client.post(
            f"/draft/{draft_id}/scenario",
            data={
                "title": "Renamed <here>",
                "scenario_id": "renamed_here",
                REVISION_FIELD: revision,
            },
        ).text
        # A top-level <title>, which htmx makes the document's.
        assert body.lstrip().startswith("<title>Renamed &lt;here&gt; · Scenario Editor")
        crumb = re.search(
            r'<div class="ed-crumb" id="editor-crumb"[^>]*>(.*?)</div>', body, re.S
        )
        assert crumb is not None
        assert 'hx-swap-oob="true"' in crumb.group(0)
        assert "Renamed &lt;here&gt;" in crumb.group(1)
        assert "renamed_here" in crumb.group(1)
        # And the full page agrees after a reload.
        reloaded = client.get(f"/draft/{draft_id}").text
        assert "<title>Renamed &lt;here&gt; · Scenario Editor</title>" in reloaded


# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------

_GENERATED_ID = re.compile(r"\b[a-z]+_[0-9a-f]{8}\b")
_UNLABELLED_INPUT_TYPES = {"hidden", "submit", "button", "reset", "image"}
_NAMING_ATTRS = ("aria-label", "aria-labelledby")


class _A11yAudit(HTMLParser):
    """Find controls a screen reader cannot name, and repeated ids."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._labels = 0
        self._label_for: set[str] = set()
        self._controls: list[tuple[str, str, str | None]] = []  # (tag, desc, id)
        self._buttons: list[tuple[dict[str, str], list[str]]] = []
        self._ids: list[str] = []
        self._svg = 0
        self._icon_only: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if a.get("id"):
            self._ids.append(a["id"])
        if tag == "svg":
            self._svg += 1
        if tag == "label":
            self._labels += 1
            if a.get("for"):
                self._label_for.add(a["for"])
        elif tag == "button":
            self._buttons.append((a, []))
        elif tag in ("input", "select", "textarea"):
            if tag == "input" and a.get("type", "text") in _UNLABELLED_INPUT_TYPES:
                return
            named = self._labels > 0 or any(a.get(k, "").strip() for k in _NAMING_ATTRS)
            if not named:
                desc = f"<{tag} name={a.get('name')!r} type={a.get('type')!r}>"
                self._controls.append((tag, desc, a.get("id")))

    def handle_endtag(self, tag: str) -> None:
        if tag == "svg":
            self._svg = max(0, self._svg - 1)
        elif tag == "label":
            self._labels = max(0, self._labels - 1)
        elif tag == "button" and self._buttons:
            attrs, text = self._buttons[-1]
            self._finish_button(attrs, "".join(text))

    def handle_data(self, data: str) -> None:
        if self._buttons and not self._svg:
            self._buttons[-1][1].append(data)

    def _finish_button(self, attrs: dict[str, str], text: str) -> None:
        self._buttons.pop()
        if any(attrs.get(k, "").strip() for k in _NAMING_ATTRS):
            return
        if not re.search(r"\w", text):
            self._icon_only.append(f"<button {attrs}> text={text.strip()!r}")

    def audit(self, page: str) -> tuple[list[str], list[str], list[str]]:
        """``(unlabelled controls, unnamed icon-only buttons, repeated ids)``."""
        self.feed(page)
        self.close()
        unlabelled = [
            desc
            for _, desc, id_ in self._controls
            if not id_ or id_ not in self._label_for
        ]
        repeated = sorted(i for i, n in Counter(self._ids).items() if n > 1)
        return unlabelled, self._icon_only, repeated


def _kitchen_sink(client: TestClient) -> str:
    """A draft holding one of every action, condition and constraint type.

    Each is added where the editor would accept it; one the editor refuses
    there (a metric that is only a fail condition, an action no entity of the
    starter can perform) is tried in the next place, or skipped.
    """
    draft_id = _new_draft(client)
    service = _service(client)
    with service.editing(draft_id):
        draft = service.require_draft(draft_id)
        document = draft.document
        actors = [None] + [entity.id for entity in document.entities]
        for action_spec in action_specs():
            for actor in actors:
                try:
                    service.add_action(document, action_spec.type_id, actor, None)
                    break
                except EditorError:
                    continue
        for condition_spec in condition_specs():
            for slot in ("fail", "pass"):
                try:
                    service.add_condition(document, slot, condition_spec.type_id)
                    break
                except EditorError:
                    continue
        slot_key = document.lanelet_slots()[0].key
        service.set_lanelet_mode(document, slot_key, "constraint_search")
        for constraint_spec in constraint_specs():
            try:
                service.add_constraint(
                    document, slot_key, constraint_spec.type_id, None
                )
            except EditorError:
                continue
        service.save(draft)
    return draft_id


def _object_ids(client: TestClient, draft_id: str) -> Iterator[str]:
    document = _stored(client, draft_id).document
    yield "scenario"
    yield "environment"
    for entity in document.entities:
        yield entity.id
    for action in document.actions:
        yield action.id
    for slot in document.lanelet_slots():
        for constraint in slot.choice.constraints:
            yield from (n.id for n in constraint.walk())
    stack = list(document.assertions.pass_conditions) + list(
        document.assertions.fail_conditions
    )
    stack += [a.trigger for a in document.actions if a.trigger is not None]
    while stack:
        node = stack.pop()
        yield node.id
        stack.extend(node.children)


class TestAccessibility:
    def test_every_inspector_names_its_controls(self, client: TestClient) -> None:
        pages: dict[str, str] = {}
        for draft_id in (_new_draft(client, "blank"), _kitchen_sink(client)):
            pages[f"{draft_id} editor"] = client.get(f"/draft/{draft_id}").text
            for object_id in _object_ids(client, draft_id):
                response = client.get(f"/draft/{draft_id}/inspector/{object_id}")
                assert response.status_code == 200, object_id
                # The page as it stands once that inspector is open: the
                # body with it selected, inside the editor page.
                pages[object_id] = client.get(
                    f"/draft/{draft_id}", params={"selected": object_id}
                ).text
        assert len(pages) > 20
        problems: set[str] = set()
        for page in pages.values():
            unlabelled, icon_only, repeated = _A11yAudit().audit(page)
            problems.update(f"unlabelled {d}" for d in unlabelled)
            problems.update(f"unnamed {d}" for d in icon_only)
            problems.update(f"repeated id {d!r}" for d in repeated)
        # One line per kind of problem, not per node: the ids differ.
        kinds = sorted({_GENERATED_ID.sub("<id>", p) for p in problems})
        assert not kinds, "\n".join(kinds)
