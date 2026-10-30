"""autoware_stack.devcontainer_text, split out of autoware_stack.devcontainer so
the library check compiles it (#763): JSONC's comments and trailing commas,
Compose's ``${VAR}`` interpolation and a ``workspaceMount``'s target, with the
values master's devcontainer gave, and devcontainer still exporting every
moved name, as the same objects."""

from __future__ import annotations

import json

import pytest

from autoware_carla_scenario.autoware_stack import devcontainer, devcontainer_text

#: What devcontainer held before the split that now lives in devcontainer_text.
_MOVED = ("_VARIABLE", "_interpolate", "_mount_target", "_strip_jsonc")


@pytest.mark.parametrize("name", _MOVED)
def test_devcontainer_re_exports_each_moved_name(name: str) -> None:
    assert getattr(devcontainer, name) is getattr(devcontainer_text, name)


def test_devcontainer_text_exports_exactly_the_moved_names() -> None:
    assert sorted(devcontainer_text.__all__) == sorted(_MOVED)


# ---------------------------------------------------------------------------
# _strip_jsonc
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('{"a": 1} // trailing', '{"a": 1} '),
        ('{\n  // a comment\n  "a": 1\n}', '{\n  \n  "a": 1\n}'),
        ('{"a": /* inline */ 1}', '{"a":  1}'),
        ('{"a": 1 /* multi\nline */}', '{"a": 1 }'),
        # Unterminated comments run to the end.
        ('{"a": 1} /* open', '{"a": 1} '),
        ('{"a": 1}//', '{"a": 1}'),
        # Comment markers inside strings are kept.
        ('{"url": "http://host/x"}', '{"url": "http://host/x"}'),
        ('{"glob": "a/*b*/c"}', '{"glob": "a/*b*/c"}'),
        # An escaped quote does not end the string.
        ('{"q": "say \\"//hi\\""}', '{"q": "say \\"//hi\\""}'),
        ('{"b": "back\\\\"} // c', '{"b": "back\\\\"} '),
        # Trailing commas before } and ], with or without whitespace.
        ('{"a": [1, 2,], "b": 3,}', '{"a": [1, 2], "b": 3}'),
        ('{"a": [1,\n  ],\n}', '{"a": [1\n  ]\n}'),
        ('{"a": 1, // c\n}', '{"a": 1 \n}'),
        ("", ""),
    ],
)
def test_strip_jsonc(given: str, expected: str) -> None:
    assert devcontainer_text._strip_jsonc(given) == expected


def test_strip_jsonc_leaves_parseable_json() -> None:
    text = """{
      // The image.
      "image": "ghcr.io/x/y:latest", /* pinned */
      "mounts": ["source=a,target=/b",],
      "env": {"URL": "https://example.com//path",},
    }"""
    assert json.loads(devcontainer_text._strip_jsonc(text)) == {
        "image": "ghcr.io/x/y:latest",
        "mounts": ["source=a,target=/b"],
        "env": {"URL": "https://example.com//path"},
    }


# ---------------------------------------------------------------------------
# _interpolate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "env", "expected"),
    [
        ("plain", {}, "plain"),
        ("${A}", {"A": "x"}, "x"),
        ("${A}", {}, ""),
        ("${A}", {"A": ""}, ""),
        ("pre-${A}-post", {"A": "x"}, "pre-x-post"),
        ("${A}/${B_2}", {"A": "1", "B_2": "2"}, "1/2"),
        # ${VAR:-default}: the default when unset or empty.
        ("${A:-d}", {"A": "x"}, "x"),
        ("${A:-d}", {}, "d"),
        ("${A:-d}", {"A": ""}, "d"),
        ("${A:-}", {}, ""),
        ("${A:-a/b:c}", {}, "a/b:c"),
        # ${VAR-default}: the default only when unset.
        ("${A-d}", {"A": "x"}, "x"),
        ("${A-d}", {}, "d"),
        ("${A-d}", {"A": ""}, ""),
        # Not a variable: left as it is.
        ("$A", {"A": "x"}, "$A"),
        ("${1A}", {"1A": "x"}, "${1A}"),
        ("${A", {"A": "x"}, "${A"),
        ("${A:+d}", {"A": "x"}, "${A:+d}"),
    ],
)
def test_interpolate(value: str, env: dict[str, str], expected: str) -> None:
    assert devcontainer_text._interpolate(value, env) == expected


# ---------------------------------------------------------------------------
# _mount_target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("", None),
        ("source=/a,target=/b,type=bind", "/b"),
        ("source=/a, target = /b ,type=bind", "/b"),
        ("src=/a,dst=/b", "/b"),
        ("source=/a,destination=/b", "/b"),
        ("source=/a,type=bind", None),
        # A target naming a variable only VS Code resolves is no answer.
        ("source=/a,target=${containerWorkspaceFolder}", None),
        ("target=${x},dst=/b", "/b"),
        ("target=/first,dst=/second", "/first"),
        ("target", ""),
    ],
)
def test_mount_target(spec: str, expected: str | None) -> None:
    assert devcontainer_text._mount_target(spec) == expected
