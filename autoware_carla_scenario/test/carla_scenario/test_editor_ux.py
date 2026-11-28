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
from typing import Any, Iterator
from urllib.parse import urlencode

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


class TestRowButtons:
    """Buttons that name a row by its index are checked strictly (#2348)."""

    @staticmethod
    def _phases(client: TestClient, draft_id: str) -> list[str]:
        controllers = _stored(client, draft_id).document.map.traffic_signal_controllers
        return [phase.name for phase in controllers[0].phases]

    @pytest.fixture
    def cycle(self, client: TestClient, draft_id: str) -> str:
        """A controller of three phases; returns the revision a page shows."""
        assert client.post(f"/draft/{draft_id}/signal-group").status_code == 200
        assert client.post(f"/draft/{draft_id}/signal-controller").status_code == 200
        for _ in range(2):
            response = client.post(f"/draft/{draft_id}/signal-controller/0/phase")
            assert response.status_code == 200
        assert self._phases(client, draft_id) == ["phase_1", "phase_2", "phase_3"]
        return _revision(client.get(f"/draft/{draft_id}/inspector/environment").text)

    def test_every_row_button_carries_the_revision(
        self, client: TestClient, draft_id: str, cycle: str
    ) -> None:
        page = client.get(f"/draft/{draft_id}/inspector/environment").text
        buttons = re.findall(
            r'<form hx-post="[^"]*/(?:delete|move|phase)"[^>]*>\s*'
            r'(?:<input type="hidden" name="delta" value="-?1">\s*)?'
            r'<input type="hidden" name="draft_revision" value="(\d+)">',
            page,
        )
        # A group, a controller and three phases' deletes, four moves, + Phase.
        assert len(buttons) == 1 + 1 + 3 + 4 + 1
        assert set(buttons) == {cycle}

    def test_a_single_click_works_and_renumbers_the_buttons(
        self, client: TestClient, draft_id: str, cycle: str
    ) -> None:
        response = client.post(
            f"/draft/{draft_id}/signal-controller/0/phase/1/delete",
            data={REVISION_FIELD: cycle},
            headers={CLIENT_HEADER: "tab-a"},
        )
        assert response.status_code == 200
        assert self._phases(client, draft_id) == ["phase_1", "phase_3"]
        assert _revision(response.text) == str(int(cycle) + 1)

    def test_a_stale_tabs_delete_is_refused(
        self, client: TestClient, draft_id: str, cycle: str
    ) -> None:
        first = client.post(
            f"/draft/{draft_id}/signal-controller/0/phase/0/delete",
            data={REVISION_FIELD: cycle},
            headers={CLIENT_HEADER: "tab-a"},
        )
        assert first.status_code == 200
        # Tab B still shows phase_1 at index 0; index 0 is now phase_2.
        second = client.post(
            f"/draft/{draft_id}/signal-controller/0/phase/0/delete",
            data={REVISION_FIELD: cycle},
            headers={CLIENT_HEADER: "tab-b"},
        )
        assert second.status_code == 409
        assert "Changed elsewhere" in second.text
        assert "Reload the page" in html.unescape(second.text)
        assert self._phases(client, draft_id) == ["phase_2", "phase_3"]

    def test_a_stale_tabs_move_is_refused(
        self, client: TestClient, draft_id: str, cycle: str
    ) -> None:
        moved = client.post(
            f"/draft/{draft_id}/signal-controller/0/phase/0/move",
            data={"delta": "1", REVISION_FIELD: cycle},
            headers={CLIENT_HEADER: "tab-a"},
        )
        assert moved.status_code == 200
        assert self._phases(client, draft_id) == ["phase_2", "phase_1", "phase_3"]
        stale = client.post(
            f"/draft/{draft_id}/signal-controller/0/phase/2/move",
            data={"delta": "-1", REVISION_FIELD: cycle},
            headers={CLIENT_HEADER: "tab-b"},
        )
        assert stale.status_code == 409
        assert self._phases(client, draft_id) == ["phase_2", "phase_1", "phase_3"]

    @pytest.mark.parametrize(
        "button",
        [
            "signal-controller/0/phase/0/delete",
            "signal-controller/0/phase/0/move",
            "signal-group/0/delete",
            "signal-controller/0/phase",
        ],
    )
    def test_a_double_click_in_one_tab_lands_once(
        self, client: TestClient, draft_id: str, cycle: str, button: str
    ) -> None:
        # The second click leaves before the answer to the first renumbers the
        # page: the same page made every save since, yet the rows have moved.
        tab = {CLIENT_HEADER: "tab-a"}
        data = {"delta": "1", REVISION_FIELD: cycle}
        url = f"/draft/{draft_id}/{button}"
        assert client.post(url, data=data, headers=tab).status_code == 200
        after_one = _stored(client, draft_id)
        assert client.post(url, data=data, headers=tab).status_code == 409
        again = _stored(client, draft_id)
        assert again.revision == after_one.revision
        assert again.document.to_yaml_dict() == after_one.document.to_yaml_dict()

    def test_a_stale_waypoint_condition_delete_is_refused(
        self, client: TestClient, draft_id: str
    ) -> None:
        shown = str(_stored(client, draft_id).revision)
        tab = {CLIENT_HEADER: "tab-a"}
        client.post(f"/draft/{draft_id}/scenario", data={"title": "T"}, headers=tab)
        response = client.post(
            f"/draft/{draft_id}/action/nothing/advance/0/delete",
            data={REVISION_FIELD: shown},
            headers=tab,
        )
        # Refused before the (missing) action is even looked up.
        assert response.status_code == 409

    def test_a_stale_actor_order_is_refused(
        self, client: TestClient, draft_id: str
    ) -> None:
        before = _stored(client, draft_id)
        order = [entity.id for entity in before.document.ordered_entities()]
        current = str(before.revision)
        tab = {CLIENT_HEADER: "tab-a"}
        reordered = client.post(
            f"/draft/{draft_id}/actors",
            data={"order": order[::-1], REVISION_FIELD: current},
            headers=tab,
        )
        assert reordered.status_code == 200
        stale = client.post(
            f"/draft/{draft_id}/actors",
            data={"order": order, REVISION_FIELD: current},
            headers=tab,
        )
        assert stale.status_code == 409
        document = _stored(client, draft_id).document
        assert [entity.id for entity in document.ordered_entities()] == order[::-1]

    def test_the_waypoint_delete_button_sends_the_revision(self) -> None:
        from autoware_carla_scenario.editor import app as app_module

        template = (
            Path(app_module.__file__).parent
            / "templates"
            / "partials"
            / "vertex_conditions.html"
        ).read_text(encoding="utf-8")
        button = template.split("/advance/{{ index }}/delete")[1].split(">")[0]
        assert """hx-vals='{"draft_revision": "{{ draft.revision }}"}'""" in button


def _post(client: TestClient, url: str, fields: dict[str, str], tab: str) -> Any:
    """Post *fields* urlencoded, as a page does (``data=`` mis-encodes lists)."""
    return client.post(
        url,
        content=urlencode(fields),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            CLIENT_HEADER: tab,
        },
    )


class TestEntityButtons:
    """A button that names an entity by its id carries the revision (#2632).

    Entity ids are reused -- a new actor takes the lowest free ``npcN`` -- so
    a stale tab's click would act on the actor that took the id since.
    """

    @staticmethod
    def _reuse_npc1(client: TestClient, draft_id: str) -> str:
        """Tab B replaces npc1 with a new car under the same id; returns the
        revision tab A, opened before, still shows."""
        page = client.get(f"/draft/{draft_id}/inspector/npc1").text
        shown = _revision(page)
        delete = re.search(
            r'hx-post="[^"]*/entity/npc1/delete"\s+hx-vals=\'([^\']*)\'', page
        )
        assert delete is not None, "the delete button names no revision"
        assert f'"{REVISION_FIELD}": "{shown}"' in delete.group(1)
        base = f"/draft/{draft_id}"
        assert (
            _post(
                client, f"{base}/entity/npc1/delete", {REVISION_FIELD: shown}, "tab-b"
            )
        ).status_code == 200
        assert (
            _post(client, f"{base}/entity", {"kind": "vehicle"}, "tab-b").status_code
            == 200
        )
        now = str(_stored(client, draft_id).revision)
        renamed = _post(
            client,
            f"{base}/entity/npc1",
            {"title": "B's new car", REVISION_FIELD: now},
            "tab-b",
        )
        assert renamed.status_code == 200
        return shown

    @pytest.mark.parametrize(
        ("path", "fields"),
        [
            ("entity/npc1/delete", {}),
            ("lanelet-mode", {"slot": "npc1.spawn", "mode": "constraint_search"}),
            ("action", {"type_id": "set_speed", "actor": "npc1"}),
            ("constraint", {"slot": "npc1.spawn", "type_id": "lanelet_length"}),
        ],
    )
    def test_a_stale_tabs_button_does_not_act_on_a_reused_id(
        self, client: TestClient, draft_id: str, path: str, fields: dict[str, str]
    ) -> None:
        shown = self._reuse_npc1(client, draft_id)
        before = _stored(client, draft_id)
        stale = _post(
            client,
            f"/draft/{draft_id}/{path}",
            {**fields, REVISION_FIELD: shown},
            "tab-a",
        )
        assert stale.status_code == 409
        assert "Changed elsewhere" in stale.text
        after = _stored(client, draft_id)
        assert after.revision == before.revision
        assert after.document.to_yaml_dict() == before.document.to_yaml_dict()
        car = after.document.entity("npc1")
        assert car is not None and car.title == "B's new car"

    @pytest.mark.parametrize(
        ("path", "fields"),
        [
            ("lanelet-mode", {"slot": "npc1.spawn", "mode": "constraint_search"}),
            ("action", {"type_id": "set_speed", "actor": "npc1"}),
            ("entity/npc1/delete", {}),
        ],
    )
    def test_a_current_click_goes_through_and_a_quick_one_after_it(
        self, client: TestClient, draft_id: str, path: str, fields: dict[str, str]
    ) -> None:
        # Checked leniently: this page's own save since does not refuse it.
        shown = str(_stored(client, draft_id).revision)
        base = f"/draft/{draft_id}"
        quick = _post(
            client, f"{base}/scenario", {"title": "T", REVISION_FIELD: shown}, "tab-a"
        )
        assert quick.status_code == 200
        response = _post(
            client, f"{base}/{path}", {**fields, REVISION_FIELD: shown}, "tab-a"
        )
        assert response.status_code == 200
        assert _stored(client, draft_id).revision == int(shown) + 2

    def test_the_lane_and_search_buttons_carry_the_revision(
        self, client: TestClient, draft_id: str
    ) -> None:
        base = f"/draft/{draft_id}"
        shown = str(_stored(client, draft_id).revision)
        assert (
            _post(
                client,
                f"{base}/lanelet-mode",
                {
                    "slot": "npc1.spawn",
                    "mode": "constraint_search",
                    REVISION_FIELD: shown,
                },
                "tab-a",
            )
        ).status_code == 200
        page = client.get(f"{base}/inspector/npc1").text
        now = _revision(page)
        radios = re.findall(
            r"hx-post=\"[^\"]*/lanelet-mode\"\s+hx-vals='([^']*)'", page
        )
        assert radios
        assert all(f'"{REVISION_FIELD}": "{now}"' in vals for vals in radios)
        add_constraint = re.findall(
            r'<form hx-post="[^"]*/constraint" [^>]*>\s*'
            r'<input type="hidden" name="draft_revision" value="(\d+)">',
            page,
        )
        assert add_constraint and set(add_constraint) == {now}
        canvas = client.get(f"{base}/canvas").text
        lanes = re.findall(
            r'<form hx-post="[^"]*/action" [^>]*>\s*'
            r'<input type="hidden" name="draft_revision" value="(\d+)">',
            canvas,
        )
        assert lanes and set(lanes) == {now}


class TestInspectorFormsSkipBrowserValidation:
    """htmx sends no form the browser finds invalid and says nothing, so an
    inspector form whose inputs carry ``min`` or ``step`` would drop every
    change in it: the server's messages stand in for the browser's (#2632)."""

    _FORM = re.compile(r"(<form\b[^>]*>)(.*?)</form>", re.S)
    _CONSTRAINED = re.compile(
        r'<input\b[^>]*\b(?:min|max)="|<input\b[^>]*step="(?!any")'
    )

    def test_every_constrained_form_is_novalidate(
        self, client: TestClient, draft_id: str
    ) -> None:
        base = f"/draft/{draft_id}"
        tab = "tab-a"
        added = _post(
            client, f"{base}/action", {"type_id": "routing", "actor": "npc1"}, tab
        )
        assert added.status_code == 200
        document = _stored(client, draft_id).document
        routing = next(a for a in document.actions if a.type == "routing")
        gate = _post(
            client,
            f"{base}/condition",
            {
                "slot": f"advance:{routing.id}",
                "vertex": "1",
                "type_id": "trajectory_time",
            },
            tab,
        )
        assert gate.status_code == 200
        counted = _post(
            client,
            f"{base}/condition",
            {"slot": "fail", "type_id": "entity_count"},
            tab,
        )
        assert counted.status_code == 200
        assert (
            _post(
                client,
                f"{base}/lanelet-mode",
                {"slot": "npc1.spawn", "mode": "constraint_search"},
                tab,
            ).status_code
            == 200
        )
        assert (
            _post(
                client,
                f"{base}/constraint",
                {"slot": "npc1.spawn", "type_id": "lanelet_length"},
                tab,
            ).status_code
            == 200
        )
        document = _stored(client, draft_id).document
        objects = ["scenario", *(e.id for e in document.entities)]
        objects += [a.id for a in document.actions]
        objects += [c.id for root in document.condition_roots() for c in root.walk()]
        objects += [
            node.id for entity in document.entities for node in entity.spawn.constraints
        ]
        seen: set[str] = set()
        for object_id in objects:
            page = client.get(f"{base}/inspector/{object_id}").text
            for tag, inner in self._FORM.findall(page):
                if "hx-post" not in tag or not self._CONSTRAINED.search(inner):
                    continue
                url = re.search(r'hx-post="([^"]+)"', tag)
                assert url is not None
                assert "novalidate" in tag, (object_id, url.group(1))
                seen.add(re.sub(r"[^/]+$", "", url.group(1).split(base, 1)[1]))
        # The entity's, an action's, a condition's and a waypoint vertex's.
        assert {"/entity/", "/action/", "/condition/"} <= {
            re.sub(r"^(/[^/]+/).*", r"\1", url) for url in seen
        }
        assert any("/advance/" in url for url in seen)


class TestRowForms:
    """A row's form names its row, so its index cannot land on another (#2348).

    These forms are checked leniently: a page's own quick edits pass the
    revision check, and one of them may have removed a row before this one.
    """

    TAB = {CLIENT_HEADER: "tab-a"}

    @staticmethod
    def _groups(client: TestClient, draft_id: str) -> list[str]:
        return [g.name for g in _stored(client, draft_id).document.map.signal_groups]

    @pytest.fixture
    def three_groups(self, client: TestClient, draft_id: str) -> str:
        for _ in range(3):
            assert client.post(f"/draft/{draft_id}/signal-group").status_code == 200
        return _revision(client.get(f"/draft/{draft_id}/inspector/environment").text)

    def test_every_row_form_names_its_row(
        self, client: TestClient, draft_id: str, three_groups: str
    ) -> None:
        client.post(f"/draft/{draft_id}/signal-controller")
        page = client.get(f"/draft/{draft_id}/inspector/environment").text
        for name in self._groups(client, draft_id):
            assert f'name="row" value="{name}"' in page
        controller = _stored(client, draft_id).document.map.traffic_signal_controllers[
            0
        ]
        assert f'name="row" value="{controller.name}"' in page
        assert f'name="row" value="{controller.phases[0].name}"' in page
        assert f'name="parent_row" value="{controller.name}"' in page

    def test_an_edit_behind_a_delete_before_its_row_is_refused(
        self, client: TestClient, draft_id: str, three_groups: str
    ) -> None:
        first, second, third = self._groups(client, draft_id)
        deleted = client.post(
            f"/draft/{draft_id}/signal-group/0/delete",
            data={REVISION_FIELD: three_groups},
            headers=self.TAB,
        )
        assert deleted.status_code == 200
        # The form of the second group, rendered before the delete and posted
        # after it: index 1 is now the third group.
        stale = client.post(
            f"/draft/{draft_id}/signal-group/1",
            data={"name": "renamed", "row": second, REVISION_FIELD: three_groups},
            headers=self.TAB,
        )
        assert stale.status_code == 409
        assert "has moved" in html.unescape(stale.text)
        assert self._groups(client, draft_id) == [second, third]

    def test_a_rename_and_a_quick_edit_after_it_both_land(
        self, client: TestClient, draft_id: str, three_groups: str
    ) -> None:
        first = self._groups(client, draft_id)[0]
        url = f"/draft/{draft_id}/signal-group/0"
        form = {"name": "east", "row": first, REVISION_FIELD: three_groups}
        assert client.post(url, data=form, headers=self.TAB).status_code == 200
        # The same form again, before the answer to the rename renumbered it.
        again = {**form, "lanelet2_regulatory_element_ids": "1330"}
        assert client.post(url, data=again, headers=self.TAB).status_code == 200
        group = _stored(client, draft_id).document.map.signal_groups[0]
        assert group.name == "east"
        assert group.lanelet2_regulatory_element_ids == [1330]

    def test_a_phase_of_a_controller_that_moved_up_is_refused(
        self, client: TestClient, draft_id: str
    ) -> None:
        for _ in range(3):
            client.post(f"/draft/{draft_id}/signal-controller")
        page = client.get(f"/draft/{draft_id}/inspector/environment").text
        shown = _revision(page)
        controllers = _stored(client, draft_id).document.map.traffic_signal_controllers
        middle = controllers[1]
        # Every controller is born with a phase of the same name.
        assert len({c.phases[0].name for c in controllers}) == 1
        deleted = client.post(
            f"/draft/{draft_id}/signal-controller/0/delete",
            data={REVISION_FIELD: shown},
            headers=self.TAB,
        )
        assert deleted.status_code == 200
        stale = client.post(
            f"/draft/{draft_id}/signal-controller/1/phase/0",
            data={
                "duration_seconds": "99",
                "row": middle.phases[0].name,
                "parent_row": middle.name,
                REVISION_FIELD: shown,
            },
            headers=self.TAB,
        )
        assert stale.status_code == 409
        after = _stored(client, draft_id).document.map.traffic_signal_controllers
        assert all(c.phases[0].duration_seconds != 99 for c in after)

    def test_a_form_without_a_row_is_not_checked(
        self, client: TestClient, draft_id: str, three_groups: str
    ) -> None:
        response = client.post(
            f"/draft/{draft_id}/signal-group/1", data={"name": "scripted"}
        )
        assert response.status_code == 200
        assert self._groups(client, draft_id)[1] == "scripted"


class TestSaveDraft:
    def test_saving_an_unchanged_draft_keeps_the_revision(
        self, client: TestClient, draft_id: str
    ) -> None:
        before = _stored(client, draft_id)
        # More presses than the editor remembers writers for (#2348).
        for _ in range(100):
            response = client.post(
                f"/draft/{draft_id}/save", headers={CLIENT_HEADER: "tab-a"}
            )
            assert response.status_code == 200
            assert "Draft saved" in response.text
        after = _stored(client, draft_id)
        assert after.revision == before.revision
        assert after.updated_at == before.updated_at
        # The page's forms are still current, from any tab.
        response = client.post(
            f"/draft/{draft_id}/scenario",
            data={"title": "Still current", REVISION_FIELD: str(before.revision)},
            headers={CLIENT_HEADER: "tab-b"},
        )
        assert response.status_code == 200

    def test_saving_a_new_title_bumps_the_revision(
        self, client: TestClient, draft_id: str
    ) -> None:
        before = _stored(client, draft_id).revision
        client.post(f"/draft/{draft_id}/save", data={"title": "Retitled"})
        after = _stored(client, draft_id)
        assert after.revision == before + 1
        assert after.document.title == "Retitled"


class TestWriterMemory:
    def test_only_the_latest_drafts_writers_are_kept(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoware_carla_scenario.editor import service as service_module

        monkeypatch.setattr(service_module, "_WRITER_DRAFTS", 3)
        service = _service(client)
        drafts = [_new_draft(client) for _ in range(5)]
        for draft_id in drafts:
            client.post(
                f"/draft/{draft_id}/scenario",
                data={"title": f"T {draft_id}"},
                headers={CLIENT_HEADER: "tab-a"},
            )
        assert list(service._writers) == drafts[-3:]
        # Saving an older one again makes it the newest.
        client.post(
            f"/draft/{drafts[2]}/scenario",
            data={"title": "again"},
            headers={CLIENT_HEADER: "tab-a"},
        )
        assert list(service._writers) == [drafts[3], drafts[4], drafts[2]]


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
