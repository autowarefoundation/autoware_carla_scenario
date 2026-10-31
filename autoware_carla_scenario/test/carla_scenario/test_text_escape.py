"""text_escape, split out of templating so the library check compiles it
(#773): the TOML and docstring escapes, with the values master's templating
gave, and templating still exporting both names, as the same objects."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# tomllib landed in 3.11; tomli is the framework's own dependency on 3.10.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib

from autoware_carla_scenario import templating, text_escape
from autoware_carla_scenario.text_escape import py_docstring, toml_string

#: Inputs that exercise every rule of both escapes.
EDGE_CASES = [
    "",
    "plain prose",
    '"',
    '""',
    '"""',
    '""""',
    'a "quoted" word',
    'ends with a quote"',
    '"starts with a quote',
    'The ego says """stop"""',
    "'''single triple quotes'''",
    "\\",
    "\\\\",
    'backslash then quote \\"',
    "ends with a backslash\\",
    r"C:\Nishishinjuku",
    r"\N{BULLET} \x41 \u0041 \U00000041 \0 \n \t",
    "line\nbreak",
    "carriage\rreturn",
    "crlf\r\nline",
    "tab\there",
    "nul\x00bell\x07esc\x1bdel\x7f",
    "vertical\x0btab form\x0cfeed",
    "".join(chr(c) for c in range(0x20)),
    "unicode: 西新宿 é \u2028 \u2029 \U0001f697",
    "\x80 \x85 \xa0",
    "{{ jinja }} {% raw %}",
]

#: What master's templating.toml_string returned for each of EDGE_CASES.
TOML_EXPECTED = [
    '""',
    '"plain prose"',
    '"\\""',
    '"\\"\\""',
    '"\\"\\"\\""',
    '"\\"\\"\\"\\""',
    '"a \\"quoted\\" word"',
    '"ends with a quote\\""',
    '"\\"starts with a quote"',
    '"The ego says \\"\\"\\"stop\\"\\"\\""',
    "\"'''single triple quotes'''\"",
    '"\\\\"',
    '"\\\\\\\\"',
    '"backslash then quote \\\\\\""',
    '"ends with a backslash\\\\"',
    '"C:\\\\Nishishinjuku"',
    '"\\\\N{BULLET} \\\\x41 \\\\u0041 \\\\U00000041 \\\\0 \\\\n \\\\t"',
    '"line\\nbreak"',
    '"carriage\\rreturn"',
    '"crlf\\r\\nline"',
    '"tab\\there"',
    '"nul\\u0000bell\\u0007esc\\u001Bdel\\u007F"',
    '"vertical\\u000Btab form\\u000Cfeed"',
    '"\\u0000\\u0001\\u0002\\u0003\\u0004\\u0005\\u0006\\u0007\\u0008\\t\\n'
    "\\u000B\\u000C\\r\\u000E\\u000F\\u0010\\u0011\\u0012\\u0013\\u0014\\u0015"
    '\\u0016\\u0017\\u0018\\u0019\\u001A\\u001B\\u001C\\u001D\\u001E\\u001F"',
    '"unicode: 西新宿 é \u2028 \u2029 \U0001f697"',
    '"\x80 \x85 \xa0"',
    '"{{ jinja }} {% raw %}"',
]

#: What master's templating.py_docstring returned for each of EDGE_CASES.
DOCSTRING_EXPECTED = [
    "",
    "plain prose",
    '\\"',
    '\\"\\"',
    '\\"\\"\\"',
    '\\"\\"\\"\\"',
    'a "quoted" word',
    'ends with a quote\\"',
    '"starts with a quote',
    # The third quote of an opening run is followed by `s`: it cannot close.
    'The ego says \\"\\""stop\\"\\"\\"',
    "'''single triple quotes'''",
    "\\\\",
    "\\\\\\\\",
    'backslash then quote \\\\\\"',
    "ends with a backslash\\\\",
    "C:\\\\Nishishinjuku",
    "\\\\N{BULLET} \\\\x41 \\\\u0041 \\\\U00000041 \\\\0 \\\\n \\\\t",
    "line\nbreak",
    "carriage\\x0dreturn",
    "crlf\\x0d\nline",
    "tab\there",
    "nul\\x00bell\\x07esc\\x1bdel\\x7f",
    "vertical\\x0btab form\\x0cfeed",
    "\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\t\n\\x0b\\x0c\\x0d\\x0e\\x0f"
    "\\x10\\x11\\x12\\x13\\x14\\x15\\x16\\x17\\x18\\x19\\x1a\\x1b\\x1c\\x1d"
    "\\x1e\\x1f",
    "unicode: 西新宿 é \u2028 \u2029 \U0001f697",
    "\x80 \x85 \xa0",
    "{{ jinja }} {% raw %}",
]


@pytest.mark.parametrize("name", ["py_docstring", "toml_string"])
def test_templating_re_exports_each_moved_name(name: str) -> None:
    assert getattr(templating, name) is getattr(text_escape, name)


def test_text_escape_exports_exactly_the_moved_names() -> None:
    assert sorted(text_escape.__all__) == ["py_docstring", "toml_string"]


@pytest.mark.parametrize(
    ("value", "expected"), list(zip(EDGE_CASES, TOML_EXPECTED, strict=True))
)
def test_toml_string_matches_master(value: str, expected: str) -> None:
    assert toml_string(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"), list(zip(EDGE_CASES, DOCSTRING_EXPECTED, strict=True))
)
def test_py_docstring_matches_master(value: str, expected: str) -> None:
    assert py_docstring(value) == expected


@pytest.mark.parametrize("value", EDGE_CASES)
def test_toml_string_round_trips(value: str) -> None:
    assert tomllib.loads(f"v = {toml_string(value)}")["v"] == value


@pytest.mark.parametrize("value", EDGE_CASES)
def test_py_docstring_round_trips(value: str) -> None:
    tree = ast.parse(f'"""{py_docstring(value)}"""\n')
    assert ast.get_docstring(tree, clean=False) == value


def test_py_docstring_leaves_inner_quotes_alone() -> None:
    # Only a quote another quote or the end follows could close the docstring.
    assert py_docstring('say "hi" now') == 'say "hi" now'
    assert py_docstring('a""b') == 'a\\""b'
    assert py_docstring('"a"b"') == '"a"b\\"'


@pytest.mark.parametrize(
    ("filter_name", "value", "expected"),
    [
        ("toml_string", 3, '"3"'),
        ("toml_string", None, '"None"'),
        ("toml_string", 'a "b"', '"a \\"b\\""'),
        ("py_docstring", 3, "3"),
        ("py_docstring", Path("C:/x"), "C:/x"),
        ("py_docstring", 'end"', 'end\\"'),
    ],
)
def test_jinja_filters_escape_str_of_any_value(
    tmp_path: Path, filter_name: str, value: object, expected: str
) -> None:
    (tmp_path / "t.txt").write_text(f"{{{{ value | {filter_name} }}}}")
    environment = templating.code_environment(tmp_path)
    assert environment.get_template("t.txt").render(value=value) == expected
