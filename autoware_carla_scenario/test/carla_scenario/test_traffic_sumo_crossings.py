"""The pedestrian crossings of a SUMO network (``traffic.sumo.network.crossing_edges``).

Pins, without SUMO, what crossing_edges() reads off a network file: each
``function="crossing"`` edge by its id, the roads it crosses as a set, and
what it makes of an edge without an id or a ``crossingEdges``, of a tag in a
namespace, of an id named twice, and of a file it cannot read.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import ParseError

import pytest

from autoware_carla_scenario.traffic.sumo.network import crossing_edges


def _net(tmp_path: Path, body: str, name: str = "network.net.xml") -> Path:
    path = tmp_path / name
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<net version="1.20">\n'
        + body
        + "\n</net>\n",
        encoding="utf-8",
    )
    return path


def test_each_crossing_and_the_roads_it_crosses(tmp_path: Path) -> None:
    net = _net(
        tmp_path,
        """
  <location netOffset="0.00,0.00"/>
  <edge id=":J0_c0" function="crossing" crossingEdges="-E0 E0">
    <lane id=":J0_c0_0" index="0" allow="pedestrian" speed="2.78" length="9.60"/>
  </edge>
  <edge id=":J0_c1" function="crossing" crossingEdges="  E1\t-E1\n E1  ">
    <lane id=":J0_c1_0" index="0"/>
  </edge>
  <edge id=":J0_w0" function="walkingarea"/>
  <edge id=":J0_0" function="internal" crossingEdges="E9"/>
  <edge id="E0" from="J0" to="J1" priority="-1"/>
  <junction id="J0" type="traffic_light"/>
  <connection from="E0" to="E1"/>
""",
    )
    assert crossing_edges(net) == {
        ":J0_c0": frozenset({"-E0", "E0"}),
        ":J0_c1": frozenset({"E1", "-E1"}),
    }
    assert all(type(v) is frozenset for v in crossing_edges(net).values())


def test_a_network_without_crossings_has_none(tmp_path: Path) -> None:
    assert crossing_edges(_net(tmp_path, '<edge id="E0"/>')) == {}
    assert crossing_edges(_net(tmp_path, "", "empty.net.xml")) == {}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # No crossingEdges, or an empty one: a crossing of no road.
        ('<edge id="c" function="crossing"/>', {"c": frozenset()}),
        ('<edge id="c" function="crossing" crossingEdges=""/>', {"c": frozenset()}),
        ('<edge id="c" function="crossing" crossingEdges="   "/>', {"c": frozenset()}),
        # No id: the key is "None".
        ('<edge function="crossing" crossingEdges="E0"/>', {"None": frozenset({"E0"})}),
        # An empty id is kept as it is.
        (
            '<edge id="" function="crossing" crossingEdges="E0"/>',
            {"": frozenset({"E0"})},
        ),
        # An id named twice: the last edge's roads, in the first's place.
        (
            '<edge id="a" function="crossing" crossingEdges="E0"/>'
            '<edge id="b" function="crossing" crossingEdges="E2"/>'
            '<edge id="a" function="crossing" crossingEdges="E1"/>',
            {"a": frozenset({"E1"}), "b": frozenset({"E2"})},
        ),
        # The function is matched exactly, the tag too (no namespace).
        ('<edge id="c" function="Crossing" crossingEdges="E0"/>', {}),
        ('<edge id="c" function=" crossing" crossingEdges="E0"/>', {}),
        ('<Edge id="c" function="crossing" crossingEdges="E0"/>', {}),
        (
            '<s:edge xmlns:s="urn:sumo" id="c" function="crossing" crossingEdges="E0"/>',
            {},
        ),
        # An edge anywhere in the document counts, a nested one too.
        (
            '<group><edge id="n" function="crossing" crossingEdges="E5"/></group>',
            {"n": frozenset({"E5"})},
        ),
        (
            '<edge id="outer" function="crossing" crossingEdges="E6">'
            '<edge id="inner" function="crossing" crossingEdges="E7"/></edge>',
            {"inner": frozenset({"E7"}), "outer": frozenset({"E6"})},
        ),
        # Non-ASCII ids and entities, read as text.
        (
            '<edge id="横断&amp;歩道" function="crossing" crossingEdges="道路&#x41; &lt;b&gt;"/>',
            {"横断&歩道": frozenset({"道路A", "<b>"})},
        ),
    ],
    ids=[
        "no-crossing-edges",
        "empty-crossing-edges",
        "blank-crossing-edges",
        "no-id",
        "empty-id",
        "id-twice",
        "function-case",
        "function-space",
        "tag-case",
        "namespace",
        "nested",
        "edge-in-edge",
        "non-ascii",
    ],
)
def test_what_crossing_edges_makes_of_an_edge(
    tmp_path: Path, body: str, expected: dict[str, frozenset[str]]
) -> None:
    found = crossing_edges(_net(tmp_path, body))
    assert found == expected
    assert list(found) == list(expected)


def test_the_root_itself_may_be_the_crossing(tmp_path: Path) -> None:
    path = tmp_path / "root.xml"
    path.write_text('<edge id="r" function="crossing" crossingEdges="E0"/>')
    assert crossing_edges(path) == {"r": frozenset({"E0"})}


def test_a_file_that_is_not_xml_raises_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.net.xml"
    path.write_text('<net><edge id="c" function="crossing" crossingEdges="E0"/>')
    with pytest.raises(ParseError):
        crossing_edges(path)
    path.write_text("")
    with pytest.raises(ParseError):
        crossing_edges(path)


def test_a_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        crossing_edges(tmp_path / "missing.net.xml")


def test_a_file_of_another_encoding_is_read_as_it_declares(tmp_path: Path) -> None:
    path = tmp_path / "latin1.net.xml"
    path.write_bytes(
        b'<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        b'<net><edge id="\xe9" function="crossing" crossingEdges="\xe0 b"/></net>'
    )
    assert crossing_edges(path) == {"é": frozenset({"à", "b"})}
