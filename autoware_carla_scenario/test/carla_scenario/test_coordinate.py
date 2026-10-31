"""Integration tests for coordinate transformation using real nishishinjuku map data.

Tests use real .xodr and .osm files from the test data directory.
No CARLA connection is required.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from autoware_carla_scenario.coordinate import (
    CarlaWorldPose,
    Lanelet2Pose,
    MapManager,
    OpenDrivePose,
    to_carla_world,
    to_lanelet2,
    to_map_frame,
    to_opendrive,
)
from autoware_carla_scenario.coordinate.map_manager import _parse_geo_reference
from autoware_carla_scenario.coordinate.projection import (
    LOCAL_XY,
    PROJECTOR_BY_AUTOWARE_TYPE,
    load_lanelet2,
    map_origin,
)
from autoware_carla_scenario.coordinate.projection import (
    read_projector_type as _read_projector_type,
)
from autoware_carla_scenario.coordinate.transform import (
    _carla_to_lanelet2,
    _carla_to_opendrive,
    _lanelet2_to_carla,
    _opendrive_to_carla,
    _project_to_centerline,
)

# Generous bound on the nishishinjuku map's half-extent, in metres. Used to
# assert that a coordinate is XODR-relative rather than MGRS-absolute (MGRS
# easting/northing around Tokyo are ~81 km / ~33 km).
_MAP_EXTENT_M = 5_000.0

DATA_DIR = Path(__file__).parent / "data"
_CONVERTER_TEST_DATA = Path(__file__).resolve().parents[3] / "data"
XODR_PATH = _CONVERTER_TEST_DATA / "nishishinjuku_carla.xodr"
OSM_PATH = _CONVERTER_TEST_DATA / "nishishinjuku.osm"
TWO_WAY = Path(__file__).resolve().parents[1] / "data" / "two_way_street"


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def map_manager():
    """Load nishishinjuku map once for the entire test module."""
    MapManager.reset()
    mm = MapManager.get_instance()
    mm.initialize(XODR_PATH, OSM_PATH)
    yield mm
    # Release lanelet2/pyxodr objects explicitly during teardown so they are
    # destroyed while the C++ runtime is still in a valid state, not during
    # Python interpreter shutdown (which can trigger std::terminate).
    mm._lanelet_map = None
    mm._road_network = None
    mm._geo_origin = None
    mm._mgrs_offset = None
    MapManager.reset()


# ---------------------------------------------------------------------------
# TestParseGeoReference – pure string parsing, no map files needed
# ---------------------------------------------------------------------------


class TestParseGeoReference:
    _PROJ = "+proj=tmerc +lat_0=35.123 +lon_0=139.456 +k=1 +x_0=0 +y_0=0"
    _XODR = (
        "<OpenDRIVE><header>"
        f"<geoReference><![CDATA[{_PROJ}]]></geoReference>"
        "</header></OpenDRIVE>"
    )

    def test_lat_lon_parsed_correctly(self):
        lat, lon, alt = _parse_geo_reference(self._XODR)
        assert lat == pytest.approx(35.123)
        assert lon == pytest.approx(139.456)
        assert alt == pytest.approx(0.0)

    def test_missing_lat_raises(self):
        bad = "<geoReference><![CDATA[+proj=tmerc +lon_0=139.0]]></geoReference>"
        with pytest.raises(ValueError, match="lat_0"):
            _parse_geo_reference(bad)

    def test_missing_geo_reference_raises(self):
        with pytest.raises(ValueError, match="geoReference"):
            _parse_geo_reference("<OpenDRIVE></OpenDRIVE>")

    def test_without_cdata_wrapper(self):
        xodr = "<geoReference>+proj=tmerc +lat_0=10.0 +lon_0=20.0</geoReference>"
        lat, lon, alt = _parse_geo_reference(xodr)
        assert lat == pytest.approx(10.0)
        assert lon == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# TestReadProjectorType – the projection comes from the map, not from a guess
# ---------------------------------------------------------------------------


class TestReadProjectorType:
    """Autoware writes the projection beside the map; this reads that file."""

    @staticmethod
    def _map_with_info(tmp_path, body: str):
        """Write a map and its projector descriptor; return the map path."""
        (tmp_path / "map_projector_info.yaml").write_text(body, encoding="utf-8")
        map_path = tmp_path / "lanelet2_map.osm"
        map_path.write_text("<osm/>", encoding="utf-8")
        return map_path

    def test_local_maps_are_read_from_their_local_tags(self, tmp_path):
        # local_xy, whose lat/lon fallback is UTM at the map origin; MGRS would
        # tear it apart at a zone boundary, which is exactly where the CARLA
        # towns sit.
        map_path = self._map_with_info(tmp_path, "projector_type: Local\n")
        assert _read_projector_type(map_path) == LOCAL_XY

    def test_every_autoware_projection_is_read_as_autoware_reads_it(self):
        # autoware_core's get_lanelet2_projector() (autoware_geography_utils):
        # LocalCartesianUTM is UtmProjector, LocalCartesian LocalCartesianProjector
        # (#859).  Local is Autoware's local_x/local_y reading.
        assert PROJECTOR_BY_AUTOWARE_TYPE == {
            "Local": LOCAL_XY,
            "LocalCartesian": "local_cartesian",
            "LocalCartesianUTM": "utm",
            "MGRS": "mgrs",
            "TransverseMercator": "transverse_mercator",
        }

    def test_local_cartesian_utm_maps_are_read_with_utm(self, tmp_path):
        map_path = self._map_with_info(tmp_path, "projector_type: LocalCartesianUTM\n")
        assert _read_projector_type(map_path) == "utm"

    def test_mgrs_maps_are_read_with_mgrs(self, tmp_path):
        map_path = self._map_with_info(tmp_path, "projector_type: MGRS\n")
        assert _read_projector_type(map_path) == "mgrs"

    def test_no_descriptor_leaves_the_choice_to_the_caller(self, tmp_path):
        map_path = tmp_path / "lanelet2_map.osm"
        map_path.write_text("<osm/>", encoding="utf-8")
        assert _read_projector_type(map_path) is None

    def test_unreadable_projection_leaves_the_choice_to_the_caller(self, tmp_path):
        map_path = self._map_with_info(tmp_path, "projector_type: Galactic\n")
        assert _read_projector_type(map_path) is None

    def test_malformed_descriptor_leaves_the_choice_to_the_caller(self, tmp_path):
        map_path = self._map_with_info(tmp_path, "projector_type: [unclosed\n")
        assert _read_projector_type(map_path) is None

    def test_a_descriptor_that_is_not_a_mapping_is_ignored(self, tmp_path):
        # Valid YAML, wrong shape: .get() on a list would be an AttributeError
        # rather than the documented fallback.
        map_path = self._map_with_info(tmp_path, "- projector_type: Local\n")
        assert _read_projector_type(map_path) is None


class TestLocalCartesianUtmMap:
    """A ``LocalCartesianUTM`` map lands where Autoware puts it (#859)."""

    ORIGIN = (35.68, 139.69, 0.0)

    def test_the_fixture_map_is_read_as_autoware_reads_it(self, tmp_path):
        import shutil

        import lanelet2.io
        import lanelet2.projection

        map_path = tmp_path / "lanelet2_map.osm"
        shutil.copyfile(OSM_PATH, map_path)
        lat, lon, alt = self.ORIGIN
        (tmp_path / "map_projector_info.yaml").write_text(
            "projector_type: LocalCartesianUTM\n"
            "vertical_datum: WGS84\n"
            "map_origin:\n"
            f"  latitude: {lat}\n"
            f"  longitude: {lon}\n"
            f"  altitude: {alt}\n",
            encoding="utf-8",
        )
        stated, source = map_origin(map_path)
        assert (stated, source) == (self.ORIGIN, "map_projector_info.yaml")
        origin = lanelet2.io.Origin(*stated)

        lanelet_map, projector, name = load_lanelet2(map_path, origin)
        assert name == "utm"
        assert isinstance(projector, lanelet2.projection.UtmProjector)

        # Autoware: lanelet2.io.load() through UtmProjector at map_origin.
        autoware = lanelet2.io.load(
            str(map_path), lanelet2.projection.UtmProjector(origin)
        )
        expected = {p.id: (p.x, p.y, p.z) for p in autoware.pointLayer}
        assert len(expected) == len(lanelet_map.pointLayer) > 0
        worst = max(
            math.dist((p.x, p.y, p.z), expected[p.id]) for p in lanelet_map.pointLayer
        )
        assert worst < 1e-6

        # What the map used to be read with lands metres away, so the check
        # above tells the two apart by geometry, not by name alone.
        tangent = lanelet2.io.load(
            str(map_path), lanelet2.projection.LocalCartesianProjector(origin)
        )
        nearest = min(
            math.dist((p.x, p.y), expected[p.id][:2]) for p in tangent.pointLayer
        )
        assert nearest > 1.0


class TestLocalCartesianOriginAltitude:
    """A ``LocalCartesian`` map's z is measured from the origin's altitude (#863).

    Autoware's ``get_lanelet2_projector()`` anchors the projector at
    ``GPSPoint{latitude, longitude, altitude}``; ``LocalCartesianProjector``
    subtracts that altitude from every z, so an origin read without it puts
    the whole map ``altitude`` metres off.
    """

    ALTITUDE = 40.0

    @classmethod
    def _map(cls, tmp_path, altitude=None):
        """The fixture map beside a LocalCartesian descriptor; (osm, origin)."""
        import shutil

        import lanelet2.io

        if altitude is None:
            altitude = cls.ALTITUDE
        map_path = tmp_path / "lanelet2_map.osm"
        shutil.copyfile(OSM_PATH, map_path)
        lat, lon, _ = _parse_geo_reference(XODR_PATH.read_text(encoding="utf-8"))
        (tmp_path / "map_projector_info.yaml").write_text(
            "projector_type: LocalCartesian\n"
            "vertical_datum: WGS84\n"
            "map_origin:\n"
            f"  latitude: {lat}\n"
            f"  longitude: {lon}\n"
            f"  altitude: {altitude}\n",
            encoding="utf-8",
        )
        return map_path, lanelet2.io.Origin(lat, lon, altitude)

    @staticmethod
    def _autoware(map_path, origin):
        """The map as Autoware reads it: LocalCartesianProjector at map_origin."""
        import lanelet2.io
        import lanelet2.projection

        return lanelet2.io.load(
            str(map_path), lanelet2.projection.LocalCartesianProjector(origin)
        )

    @staticmethod
    def _worst(lanelet_map, expected_map) -> float:
        expected = {p.id: (p.x, p.y, p.z) for p in expected_map.pointLayer}
        assert len(expected) == len(lanelet_map.pointLayer) > 0
        return max(
            math.dist((p.x, p.y, p.z), expected[p.id]) for p in lanelet_map.pointLayer
        )

    def test_the_origin_carries_the_altitude(self, tmp_path):
        map_path, _ = self._map(tmp_path)
        stated, source = map_origin(map_path, XODR_PATH)
        assert source == "map_projector_info.yaml"
        assert stated[2] == self.ALTITUDE

    def test_the_sweeper_reads_it_as_autoware_does(self, tmp_path):
        from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

        map_path, origin = self._map(tmp_path)
        swept = load_lanelet2_map(map_path, XODR_PATH)
        assert self._worst(swept, self._autoware(map_path, origin)) < 1e-6

    def test_the_run_reads_it_as_autoware_does(self, tmp_path):
        map_path, origin = self._map(tmp_path)
        autoware = self._autoware(map_path, origin)

        MapManager.reset()
        mm = MapManager.get_instance()
        try:
            mm.initialize(XODR_PATH, map_path)
            assert mm.geo_origin[2] == self.ALTITUDE
            assert self._worst(mm.lanelet_map, autoware) < 1e-6

            # What the run hands Autoware as a pose lands on Autoware's own
            # map: the map-frame z of a lanelet's first centerline point is
            # that point's z in the map Autoware loaded.
            lanelet = next(iter(autoware.laneletLayer))
            sent = to_map_frame(Lanelet2Pose(lanelet_id=lanelet.id, s=0.0))
            assert sent.position.z == pytest.approx(lanelet.centerline[0].z, abs=1e-3)
        finally:
            mm._lanelet_map = None
            mm._road_network = None
            mm._geo_origin = None
            mm._mgrs_offset = None
            MapManager.reset()

    def test_the_carla_side_does_not_move(self, tmp_path):
        """The altitude moves the map frame's z, never CARLA's: the z offset
        takes it up, so a lanelet pose lands at the same CARLA height, and
        back on the same lanelet, whatever the altitude."""
        pose = Lanelet2Pose(lanelet_id=0, s=1.0)
        seen = {}
        for altitude in (0.0, self.ALTITUDE):
            directory = tmp_path / f"alt{altitude:g}"
            directory.mkdir()
            map_path, _ = self._map(directory, altitude)
            MapManager.reset()
            mm = MapManager.get_instance()
            try:
                mm.initialize(XODR_PATH, map_path)
                if pose.lanelet_id == 0:
                    lanelet = next(iter(mm.lanelet_map.laneletLayer))
                    pose = Lanelet2Pose(lanelet_id=lanelet.id, s=1.0)
                carla = to_carla_world(pose)
                seen[altitude] = (
                    mm.z_offset,
                    to_map_frame(pose).position.z,
                    carla,
                    to_lanelet2(carla).lanelet_id,
                )
            finally:
                mm._lanelet_map = None
                mm._road_network = None
                mm._geo_origin = None
                mm._mgrs_offset = None
                MapManager.reset()

        low, high = seen[0.0], seen[self.ALTITUDE]
        assert low[0] - high[0] == pytest.approx(self.ALTITUDE, abs=1e-6)
        assert low[1] - high[1] == pytest.approx(self.ALTITUDE, abs=1e-6)
        for a, b in ((low[2].x, high[2].x), (low[2].y, high[2].y)):
            assert a == pytest.approx(b, abs=1e-6)
        assert low[2].z == pytest.approx(high[2].z, abs=1e-6)
        assert low[3] == high[3] == pose.lanelet_id


# ---------------------------------------------------------------------------
# TestLocalProjection – how a projector_type: Local map is read
# ---------------------------------------------------------------------------


class TestLocalProjection:
    """A ``projector_type: Local`` map is read from its local_x/local_y (#285)."""

    ORIGIN = (35.68, 139.69)
    #: (id, local_x, local_y) of a 200 m lanelet, 3.5 m wide: long enough for
    #: UTM and a tangent plane to part by more than a centimetre at its end.
    NODES = [
        (1, 0.0, 0.0),
        (2, 200.0, 0.0),
        (3, 0.0, 3.5),
        (4, 200.0, 3.5),
    ]

    @classmethod
    def _origin(cls):
        import lanelet2.io

        return lanelet2.io.Origin(*cls.ORIGIN)

    @classmethod
    def _write_map(
        cls,
        tmp_path: Path,
        latlon_projector,
        *,
        untagged: frozenset[int] = frozenset(),
        descriptor: str = "projector_type: Local\n",
    ) -> Path:
        """Write a Local map whose lat/lon *latlon_projector* derived.

        Nodes in *untagged* carry no local_x/local_y.
        """
        import lanelet2.core

        lines = ['<?xml version="1.0"?>', '<osm version="0.6" generator="test">']
        for node_id, x, y in cls.NODES:
            gps = latlon_projector.reverse(lanelet2.core.BasicPoint3d(x, y, 0.0))
            lines.append(
                f'  <node id="{node_id}" version="1" '
                f'lat="{gps.lat:.12f}" lon="{gps.lon:.12f}">'
            )
            if node_id not in untagged:
                lines.append(f'    <tag k="local_x" v="{x:.4f}"/>')
                lines.append(f'    <tag k="local_y" v="{y:.4f}"/>')
            lines.append("  </node>")
        for way_id, (first, last) in ((10, (1, 2)), (11, (3, 4))):
            lines += [
                f'  <way id="{way_id}" version="1">',
                f'    <nd ref="{first}"/>',
                f'    <nd ref="{last}"/>',
                '    <tag k="type" v="line_thin"/>',
                '    <tag k="subtype" v="solid"/>',
                "  </way>",
            ]
        lines += [
            '  <relation id="100" version="1">',
            '    <member type="way" role="left" ref="11"/>',
            '    <member type="way" role="right" ref="10"/>',
            '    <tag k="type" v="lanelet"/>',
            '    <tag k="subtype" v="road"/>',
            "  </relation>",
            "</osm>",
        ]
        map_path = tmp_path / "lanelet2_map.osm"
        map_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (tmp_path / "map_projector_info.yaml").write_text(descriptor, encoding="utf-8")
        return map_path

    @classmethod
    def _tangent_plane_map(
        cls, tmp_path: Path, projector: str = "LocalCartesianProjector", **kwargs
    ) -> Path:
        """A Local map whose lat/lon *projector* derived: not UTM, by default."""
        import lanelet2.projection

        latlon = getattr(lanelet2.projection, projector)(cls._origin())
        return cls._write_map(tmp_path, latlon, **kwargs)

    @classmethod
    def _offsets(cls, lanelet_map) -> dict[int, float]:
        """How far each point of the map lies from its local_x/local_y."""
        nodes = {node_id: (x, y) for node_id, x, y in cls.NODES}
        return {
            point.id: math.hypot(
                point.x - nodes[point.id][0], point.y - nodes[point.id][1]
            )
            for point in lanelet_map.pointLayer
        }

    def test_a_map_is_read_from_its_local_tags_by_default(self, tmp_path, caplog):
        map_path = self._tangent_plane_map(tmp_path)
        with caplog.at_level("INFO"):
            lanelet_map, _projector, name = load_lanelet2(map_path, self._origin())
        assert name == LOCAL_XY
        assert max(self._offsets(lanelet_map).values()) < 1e-6
        assert "from local_x/local_y" in caplog.text
        # Its lat/lon did not come from UTM: the two readings disagree.
        assert "disagree" in caplog.text

    def test_lat_lon_through_utm_misplaces_a_map_projected_otherwise(self, tmp_path):
        # What the issue found: lat/lon that did not come from UTM, read back
        # through it, land measurably off.
        map_path = self._tangent_plane_map(tmp_path)
        lanelet_map, _projector, name = load_lanelet2(
            map_path, self._origin(), projector_type="utm"
        )
        assert name == "utm"
        assert max(self._offsets(lanelet_map).values()) > 0.01

    @pytest.mark.parametrize("projector_type", [None, "utm"])
    def test_a_carla_town_reads_the_same_either_way(
        self, tmp_path, caplog, projector_type
    ):
        # The official CARLA towns derive their lat/lon from local_x/local_y
        # with UTM at the origin: both readings agree, and nothing is warned.
        map_path = self._tangent_plane_map(tmp_path, projector="UtmProjector")
        with caplog.at_level("WARNING"):
            lanelet_map, _projector, _name = load_lanelet2(
                map_path, self._origin(), projector_type=projector_type
            )
        assert max(self._offsets(lanelet_map).values()) < 0.01
        assert "disagree" not in caplog.text

    def test_a_node_without_local_tags_falls_back_to_its_lat_lon(self, tmp_path):
        map_path = self._tangent_plane_map(tmp_path, untagged=frozenset({4}))
        lanelet_map, _projector, _name = load_lanelet2(map_path, self._origin())
        offsets = self._offsets(lanelet_map)
        assert max(offsets[1], offsets[2], offsets[3]) < 1e-6
        # Node 4 is where utm puts it.
        through_utm, _projector, _name = load_lanelet2(
            map_path, self._origin(), projector_type="utm"
        )
        assert offsets[4] == pytest.approx(self._offsets(through_utm)[4], abs=1e-9)
        assert offsets[4] > 0.01

    def test_the_moved_map_is_searched_where_its_points_now_are(self, tmp_path):
        import lanelet2.core

        map_path = self._tangent_plane_map(tmp_path)
        lanelet_map, _projector, _name = load_lanelet2(map_path, self._origin())
        assert len(lanelet_map.laneletLayer) == 1
        assert len(lanelet_map.lineStringLayer) == 2
        assert len(lanelet_map.pointLayer) == 4
        box = lanelet2.core.BoundingBox2d(
            lanelet2.core.BasicPoint2d(199.999, 3.499),
            lanelet2.core.BasicPoint2d(200.001, 3.501),
        )
        assert [point.id for point in lanelet_map.pointLayer.search(box)] == [4]

    def test_a_map_that_is_not_local_ignores_its_local_tags(self, tmp_path):
        map_path = self._tangent_plane_map(
            tmp_path, descriptor="projector_type: TransverseMercator\n"
        )
        lanelet_map, _projector, name = load_lanelet2(map_path, self._origin())
        assert name == "transverse_mercator"
        assert max(self._offsets(lanelet_map).values()) > 1e-6

    def test_local_xy_can_be_asked_for(self, tmp_path):
        map_path = self._tangent_plane_map(
            tmp_path, descriptor="projector_type: MGRS\n"
        )
        lanelet_map, _projector, name = load_lanelet2(
            map_path, self._origin(), projector_type=LOCAL_XY
        )
        assert name == LOCAL_XY
        assert max(self._offsets(lanelet_map).values()) < 1e-6

    def test_both_loaders_read_the_two_way_street_from_its_local_tags(self):
        # roadgen derived the fixture's lat/lon on a tangent plane, 1.3 m from
        # where UTM reads its far end.  The sweeper and the run read it alike,
        # so the lanelet a sweep picks is the one the run poses a vehicle on.
        from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

        osm = TWO_WAY / "lanelet2_map.osm"
        xodr = TWO_WAY / "two_way_street.xodr"
        swept = load_lanelet2_map(osm, xodr).pointLayer[1001]
        assert (swept.x, swept.y) == pytest.approx((100.0, 3.5), abs=1e-6)

        MapManager.reset()
        mm = MapManager.get_instance()
        try:
            mm.initialize(xodr, osm)
            run = mm.lanelet_map.pointLayer[1001]
            assert (run.x, run.y) == pytest.approx((swept.x, swept.y), abs=1e-9)
            assert mm.mgrs_offset == pytest.approx((0.0, 0.0), abs=1e-6)
        finally:
            mm._lanelet_map = None
            mm._road_network = None
            MapManager.reset()


# ---------------------------------------------------------------------------
# TestMapManagerInit – verify map loaded successfully
# ---------------------------------------------------------------------------


class TestMapManagerInit:
    def test_map_loads_successfully(self, map_manager):
        assert map_manager.lanelet_map is not None
        assert map_manager.road_network is not None

    def test_geo_origin_is_nishishinjuku(self, map_manager):
        """Origin should be in the Tokyo / Nishishinjuku area."""
        lat, lon, _alt = map_manager.geo_origin
        assert lat == pytest.approx(35.688, abs=0.01)
        assert lon == pytest.approx(139.692, abs=0.01)

    def test_mgrs_offset_is_nonzero(self, map_manager):
        ox, oy = map_manager.mgrs_offset
        assert ox > 0
        assert oy > 0

    def test_lanelet_map_has_lanelets(self, map_manager):
        lanelets = list(map_manager.lanelet_map.laneletLayer)
        assert len(lanelets) > 0

    def test_road_network_has_roads(self, map_manager):
        roads = map_manager.road_network.road_ids_to_object
        assert len(roads) > 0

    def test_singleton_same_instance(self, map_manager):
        assert MapManager.get_instance() is map_manager

    def test_double_init_raises(self, map_manager):
        with pytest.raises(RuntimeError, match="already initialized"):
            map_manager.initialize(XODR_PATH, OSM_PATH)

    def test_file_not_found_raises(self, map_manager):
        # Use object.__new__ to bypass the singleton and get a raw uninitialized
        # instance without touching the shared singleton's state.
        raw_mm = object.__new__(MapManager)
        raw_mm._lanelet_map = None
        raw_mm._road_network = None
        raw_mm._geo_origin = None
        raw_mm._mgrs_offset = None
        with pytest.raises(FileNotFoundError):
            raw_mm.initialize(Path("nonexistent.xodr"), Path("nonexistent.osm"))


# ---------------------------------------------------------------------------
# TestLanelet2ToCarla
# ---------------------------------------------------------------------------


class TestLanelet2ToCarla:
    def test_result_is_carla_world_pose(self, map_manager):
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        result = _lanelet2_to_carla(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0))
        assert isinstance(result, CarlaWorldPose)

    def test_carla_coords_are_finite(self, map_manager):
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        result = _lanelet2_to_carla(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0))
        assert math.isfinite(result.x)
        assert math.isfinite(result.y)
        assert math.isfinite(result.z)
        assert math.isfinite(result.yaw)

    def test_carla_coords_are_xodr_relative(self, map_manager):
        """The MGRS offset must be subtracted, leaving XODR-relative metres.

        MGRS easting/northing for Tokyo are ~81 km / ~33 km; the map itself is
        about a kilometre across. Anything near the MGRS magnitude means the
        offset was not applied, and a value far outside the map extent means it
        was applied twice or with the wrong sign.
        """
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        result = _lanelet2_to_carla(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0))
        assert abs(result.x) < _MAP_EXTENT_M
        assert abs(result.y) < _MAP_EXTENT_M

    def test_lateral_offset_displaces_by_correct_distance(self, map_manager):
        """t=1 m should shift position by exactly 1 m."""
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        r0 = _lanelet2_to_carla(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0, t=0.0))
        r1 = _lanelet2_to_carla(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0, t=1.0))
        dist = math.sqrt((r0.x - r1.x) ** 2 + (r0.y - r1.y) ** 2)
        assert dist == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# TestOpenDriveToCarla
# ---------------------------------------------------------------------------


class TestOpenDriveToCarla:
    def test_result_is_carla_world_pose(self, map_manager):
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        result = _opendrive_to_carla(OpenDrivePose(road_id=road_id, lane_id=-1, s=0.0))
        assert isinstance(result, CarlaWorldPose)

    def test_carla_coords_are_finite(self, map_manager):
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        result = _opendrive_to_carla(OpenDrivePose(road_id=road_id, lane_id=-1, s=0.0))
        assert math.isfinite(result.x)
        assert math.isfinite(result.y)
        assert math.isfinite(result.z)

    def test_carla_coords_are_xodr_relative(self, map_manager):
        """OpenDRIVE coordinates are already XODR-relative; no offset is added.

        Roads sit on either side of the geoReference origin, so the sign of x
        carries no information — the invariant is the magnitude.
        """
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        result = _opendrive_to_carla(OpenDrivePose(road_id=road_id, lane_id=-1, s=0.0))
        assert abs(result.x) < _MAP_EXTENT_M
        assert abs(result.y) < _MAP_EXTENT_M

    def test_lateral_offset_displaces_by_correct_distance(self, map_manager):
        """t=1 m should shift position by exactly 1 m."""
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        r0 = _opendrive_to_carla(
            OpenDrivePose(road_id=road_id, lane_id=0, s=0.0, t=0.0)
        )
        r1 = _opendrive_to_carla(
            OpenDrivePose(road_id=road_id, lane_id=1, s=0.0, t=1.0)
        )
        dist = math.sqrt((r0.x - r1.x) ** 2 + (r0.y - r1.y) ** 2)
        assert dist == pytest.approx(1.0, abs=0.01)

    def test_lanelet2_and_opendrive_origins_are_close(self, map_manager):
        """Lanelet2 and OpenDRIVE share the same map area; positions should overlap."""
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        road_id = next(iter(map_manager.road_network.road_ids_to_object))

        ll2_carla = _lanelet2_to_carla(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0))
        od_carla = _opendrive_to_carla(
            OpenDrivePose(road_id=road_id, lane_id=-1, s=0.0)
        )

        # Both should be in the Nishishinjuku area (x roughly 81000–83000 in MGRS)
        assert abs(ll2_carla.x - od_carla.x) < 3000
        assert abs(ll2_carla.y - od_carla.y) < 3000


# ---------------------------------------------------------------------------
# TestRoundTripLanelet2 – Lanelet2Pose → CarlaWorldPose → Lanelet2Pose
# ---------------------------------------------------------------------------


class TestRoundTripLanelet2:
    def _ll2_carla_ll2_position_error(
        self, map_manager, lanelet_id: int, s: float, t: float
    ) -> float:
        """Return the Euclidean error (in CARLA space) for a round-trip."""
        original = Lanelet2Pose(lanelet_id=lanelet_id, s=s, t=t)
        carla = _lanelet2_to_carla(original)
        recovered = _carla_to_lanelet2(carla)
        recovered_carla = _lanelet2_to_carla(recovered)
        return math.sqrt(
            (carla.x - recovered_carla.x) ** 2 + (carla.y - recovered_carla.y) ** 2
        )

    def test_round_trip_on_centerline(self, map_manager):
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        error = self._ll2_carla_ll2_position_error(
            map_manager, lanelet_id, s=0.0, t=0.0
        )
        assert error < 1.0  # within 1 m

    def test_round_trip_with_lateral_offset(self, map_manager):
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        error = self._ll2_carla_ll2_position_error(
            map_manager, lanelet_id, s=0.0, t=0.5
        )
        assert error < 1.0


# ---------------------------------------------------------------------------
# TestProjectToCenterline – the foot of the perpendicular, not a vertex
# ---------------------------------------------------------------------------


class TestProjectToCenterline:
    """``_project_to_centerline`` projects onto the polyline's segments."""

    # An L: 100 m East, then 50 m North.
    _L = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (100.0, 50.0, 0.0)]

    def test_a_point_between_far_apart_vertices(self):
        s, t, heading = _project_to_centerline(self._L, 37.5, 2.0)
        assert s == pytest.approx(37.5, abs=1e-12)
        assert t == pytest.approx(2.0, abs=1e-12)
        assert heading == pytest.approx(0.0, abs=1e-12)

    def test_right_of_the_line_is_negative(self):
        s, t, _heading = _project_to_centerline(self._L, 62.0, -1.5)
        assert s == pytest.approx(62.0, abs=1e-12)
        assert t == pytest.approx(-1.5, abs=1e-12)

    def test_near_a_corner_reads_the_segment_it_is_beside(self):
        # 1 m before the corner, inside it: the first segment, not the vertex.
        s, t, heading = _project_to_centerline(self._L, 99.0, 0.5)
        assert s == pytest.approx(99.0, abs=1e-12)
        assert t == pytest.approx(0.5, abs=1e-12)
        assert heading == pytest.approx(0.0, abs=1e-12)
        # 1 m after it: the second segment, its heading and its own t.
        s, t, heading = _project_to_centerline(self._L, 99.5, 1.0)
        assert s == pytest.approx(101.0, abs=1e-12)
        assert t == pytest.approx(0.5, abs=1e-12)
        assert heading == pytest.approx(math.pi / 2, abs=1e-12)

    def test_a_foot_on_a_vertex_belongs_to_the_segment_starting_there(self):
        # Outside the corner both segments' feet are the corner itself.
        s, t, heading = _project_to_centerline(self._L, 102.0, -2.0)
        assert s == pytest.approx(100.0, abs=1e-12)
        assert heading == pytest.approx(math.pi / 2, abs=1e-12)
        assert t == pytest.approx(-2.0, abs=1e-12)
        # A point on the vertex reads the same.
        s, t, heading = _project_to_centerline(self._L, 100.0, 0.0)
        assert (s, t) == (pytest.approx(100.0), pytest.approx(0.0))
        assert heading == pytest.approx(math.pi / 2, abs=1e-12)

    def test_past_the_ends_is_clamped(self):
        s, t, heading = _project_to_centerline(self._L, -3.0, 1.0)
        assert (s, t, heading) == (0.0, pytest.approx(1.0), pytest.approx(0.0))
        s, t, heading = _project_to_centerline(self._L, 101.0, 60.0)
        assert s == pytest.approx(150.0, abs=1e-12)
        assert t == pytest.approx(-1.0, abs=1e-12)
        assert heading == pytest.approx(math.pi / 2, abs=1e-12)

    def test_repeated_points_are_skipped(self):
        line = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
        ]
        s, t, heading = _project_to_centerline(line, 10.5, 4.0)
        assert s == pytest.approx(14.0, abs=1e-12)
        assert t == pytest.approx(-0.5, abs=1e-12)
        assert heading == pytest.approx(math.pi / 2, abs=1e-12)
        # The corner, through a zero-length segment, still starts the next one.
        s, _t, heading = _project_to_centerline(line, 11.0, -1.0)
        assert s == pytest.approx(10.0, abs=1e-12)
        assert heading == pytest.approx(math.pi / 2, abs=1e-12)

    def test_a_single_point(self):
        assert _project_to_centerline([(1.0, 2.0, 0.0)], 4.0, 5.0) == (
            0.0,
            3.0,
            0.0,
        )
        assert _project_to_centerline([(1.0, 2.0, 0.0)] * 3, 4.0, 1.0) == (
            0.0,
            -1.0,
            0.0,
        )


class TestToLanelet2ProjectsOntoThePolyline:
    """``to_lanelet2`` reads ``s`` and ``t`` off the polyline, not a vertex."""

    def test_a_point_mid_segment_on_the_map(self, map_manager):
        # The lanelet with the longest centreline segment, and its midpoint.
        def longest(lanelet) -> tuple[float, int]:
            pts = [(p.x, p.y) for p in lanelet.centerline]
            return max(
                (math.dist(a, b), i) for i, (a, b) in enumerate(zip(pts, pts[1:]))
            )

        lanelet = max(map_manager.lanelet_map.laneletLayer, key=longest)
        length, index = longest(lanelet)
        assert length > 20.0  # far enough apart for a vertex snap to show
        pts = [(p.x, p.y) for p in lanelet.centerline]
        s = sum(math.dist(a, b) for a, b in zip(pts[:index], pts[1 : index + 1]))
        s += length / 2

        pose = to_carla_world(Lanelet2Pose(lanelet_id=lanelet.id, s=s, t=0.4))
        recovered = to_lanelet2(pose)

        assert recovered.lanelet_id == lanelet.id
        assert recovered.s == pytest.approx(s, abs=1e-6)
        assert recovered.t == pytest.approx(0.4, abs=1e-6)
        assert recovered.heading == pytest.approx(0.0, abs=1e-9)

    def test_round_trip_on_every_centreline(self, map_manager):
        """to_carla_world(to_lanelet2(p)) is p for a point on a centreline."""
        for lanelet in map_manager.lanelet_map.laneletLayer:
            pts = [(p.x, p.y) for p in lanelet.centerline]
            total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
            for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
                point = to_carla_world(
                    Lanelet2Pose(lanelet_id=lanelet.id, s=fraction * total)
                )
                back = to_carla_world(to_lanelet2(point))
                assert math.hypot(back.x - point.x, back.y - point.y) < 1e-6, (
                    lanelet.id,
                    fraction,
                )


# ---------------------------------------------------------------------------
# TestRoundTripOpenDrive – OpenDrivePose → CarlaWorldPose → OpenDrivePose
# ---------------------------------------------------------------------------


class TestRoundTripOpenDrive:
    def _od_carla_od_position_error(
        self, map_manager, road_id: str, s: float, t: float
    ) -> float:
        """Return the Euclidean error (in CARLA space) for a round-trip."""
        original = OpenDrivePose(road_id=road_id, lane_id=0, s=s, t=t)
        carla = _opendrive_to_carla(original)
        recovered = _carla_to_opendrive(carla)
        recovered_carla = _opendrive_to_carla(recovered)
        return math.sqrt(
            (carla.x - recovered_carla.x) ** 2 + (carla.y - recovered_carla.y) ** 2
        )

    def test_round_trip_on_reference_line(self, map_manager):
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        error = self._od_carla_od_position_error(map_manager, road_id, s=0.0, t=0.0)
        assert error < 1.0

    def test_round_trip_with_lateral_offset(self, map_manager):
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        error = self._od_carla_od_position_error(map_manager, road_id, s=0.0, t=-1.0)
        assert error < 1.0


# ---------------------------------------------------------------------------
# TestCrossSystem – Lanelet2 ↔ OpenDRIVE via CARLA intermediate
# ---------------------------------------------------------------------------


class TestCrossSystem:
    def test_a_road_is_not_a_lane(self, map_manager):
        """Neighbouring lanelets share one OpenDRIVE road.

        On this map lanelets 183 and 184 are two lanes of the same road, so a
        caller that resolved a lanelet to a road alone would make "NPC1 is on
        lanelet 183" true while NPC1 is still in the neighbouring lane -- the
        cut-in example would pass without the cut-in ever happening.  This is
        the coordinate-level fact behind that; that the conditions carry the
        lane through is pinned in ``test_composition_conditions.py``.
        """
        address = to_opendrive(Lanelet2Pose(lanelet_id=183, s=0.0))
        neighbour = to_opendrive(Lanelet2Pose(lanelet_id=184, s=0.0))

        assert address.road_id == neighbour.road_id
        assert address.lane_id != neighbour.lane_id

    def test_lanelet2_to_opendrive_and_back(self, map_manager):
        """Lanelet2 → CARLA → OpenDRIVE → CARLA: position should be close."""
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        ll2_pose = Lanelet2Pose(lanelet_id=lanelet_id, s=0.0, t=0.0)

        carla_orig = _lanelet2_to_carla(ll2_pose)
        od_pose = _carla_to_opendrive(carla_orig)
        carla_recovered = _opendrive_to_carla(od_pose)

        dist = math.sqrt(
            (carla_orig.x - carla_recovered.x) ** 2
            + (carla_orig.y - carla_recovered.y) ** 2
        )
        # Allow up to 5 m – lanelet centerlines may not coincide with road ref lines
        assert dist < 5.0

    def test_opendrive_to_lanelet2_and_back(self, map_manager):
        """OpenDRIVE → CARLA → Lanelet2 → CARLA: position should be close."""
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        od_pose = OpenDrivePose(road_id=road_id, lane_id=-1, s=0.0, t=0.0)

        carla_orig = _opendrive_to_carla(od_pose)
        ll2_pose = _carla_to_lanelet2(carla_orig)
        carla_recovered = _lanelet2_to_carla(ll2_pose)

        dist = math.sqrt(
            (carla_orig.x - carla_recovered.x) ** 2
            + (carla_orig.y - carla_recovered.y) ** 2
        )
        assert dist < 5.0


# ---------------------------------------------------------------------------
# TestPublicAPI – overloaded dispatch
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_to_carla_world_from_lanelet2(self, map_manager):
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        result = to_carla_world(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0))
        assert isinstance(result, CarlaWorldPose)

    def test_to_carla_world_from_opendrive(self, map_manager):
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        result = to_carla_world(OpenDrivePose(road_id=road_id, lane_id=-1, s=0.0))
        assert isinstance(result, CarlaWorldPose)

    def test_to_opendrive_from_carla(self, map_manager):
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        carla = to_carla_world(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0))
        result = to_opendrive(carla)
        assert isinstance(result, OpenDrivePose)

    def test_to_lanelet2_from_carla(self, map_manager):
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        carla = to_carla_world(OpenDrivePose(road_id=road_id, lane_id=-1, s=0.0))
        result = to_lanelet2(carla)
        assert isinstance(result, Lanelet2Pose)

    def test_to_opendrive_from_lanelet2(self, map_manager):
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        result = to_opendrive(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0))
        assert isinstance(result, OpenDrivePose)

    def test_to_lanelet2_from_opendrive(self, map_manager):
        road_id = next(iter(map_manager.road_network.road_ids_to_object))
        result = to_lanelet2(OpenDrivePose(road_id=road_id, lane_id=-1, s=0.0))
        assert isinstance(result, Lanelet2Pose)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            to_carla_world("invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestToMapFrame – the frame Autoware's own map_loader produces
# ---------------------------------------------------------------------------


class TestToMapFrame:
    """A map-frame pose is the Lanelet2 frame, not the CARLA/XODR one.

    Autoware reads the same Lanelet2 file with the same projector, so the pose
    it is handed has to be expressed the way that file is -- offset and all.
    On nishishinjuku (MGRS) the offset is tens of kilometres, so a pose left in
    CARLA coordinates would land in a different part of Tokyo.
    """

    def test_a_lanelet_centerline_point_maps_back_onto_itself(self, map_manager):
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        centerline = map_manager.lanelet_map.laneletLayer[lanelet_id].centerline
        start = centerline[0]

        pose = to_map_frame(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0))

        assert pose.position.x == pytest.approx(start.x, abs=1e-6)
        assert pose.position.y == pytest.approx(start.y, abs=1e-6)
        assert pose.position.z == pytest.approx(start.z, abs=1e-6)

    def test_the_offset_is_added_back(self, map_manager):
        lanelet_id = next(iter(map_manager.lanelet_map.laneletLayer)).id
        carla_pose = to_carla_world(Lanelet2Pose(lanelet_id=lanelet_id, s=0.0))

        pose = to_map_frame(carla_pose)

        offset_x, offset_y = map_manager.mgrs_offset
        assert pose.position.x == pytest.approx(carla_pose.x + offset_x)
        assert pose.position.y == pytest.approx(-carla_pose.y + offset_y)
        assert pose.position.z == pytest.approx(carla_pose.z + map_manager.z_offset)
        # An MGRS map's own coordinates are far outside the town's extent.
        assert abs(pose.position.x) > _MAP_EXTENT_M

    def test_the_heading_flips_with_the_handedness(self, map_manager):
        pose = to_map_frame(CarlaWorldPose(x=0.0, y=0.0, z=0.0, yaw=90.0))

        # CARLA yaw +90 deg (clockwise from East) is -90 deg in the map frame.
        yaw = 2.0 * math.atan2(pose.rotation.z, pose.rotation.w)
        assert yaw == pytest.approx(-math.pi / 2)

    def test_pitch_and_roll_survive_the_conversion(self, map_manager):
        # A pose snapped onto a ramp carries pitch; dropping it would have
        # Autoware initialize localization in an attitude the vehicle is not in.
        flat = to_map_frame(CarlaWorldPose(x=0.0, y=0.0, z=0.0))
        tilted = to_map_frame(CarlaWorldPose(x=0.0, y=0.0, z=0.0, pitch=10.0, roll=5.0))
        assert tilted.rotation != flat.rotation


# ---------------------------------------------------------------------------
# TestZOffsetFromSpawnPoints
# ---------------------------------------------------------------------------


class TestZOffsetFromSpawnPoints:
    """The offset compares the Lanelet2 centerline with the road, not the spawn."""

    @staticmethod
    def _world(map_manager, road_below_lanelet2: float, spawn_above_road: float):
        from types import SimpleNamespace

        ox, oy = map_manager.mgrs_offset
        points = [
            ll.centerline[0]
            for _, ll in zip(range(20), map_manager.lanelet_map.laneletLayer)
        ]
        spawns = [
            SimpleNamespace(
                location=SimpleNamespace(
                    x=pt.x - ox,
                    y=-(pt.y - oy),
                    z=pt.z - road_below_lanelet2 + spawn_above_road,
                )
            )
            for pt in points
        ]

        class _Map:
            def get_spawn_points(self):
                return spawns

            def get_waypoint(self, location, project_to_road=True):
                road = SimpleNamespace(z=location.z - spawn_above_road)
                return SimpleNamespace(transform=SimpleNamespace(location=road))

        return SimpleNamespace(get_map=_Map)

    def test_a_spawn_point_held_above_the_road_does_not_shift_the_offset(
        self, map_manager
    ):
        # roadgen's cooked maps hold spawn points 4 m up: read as the road, they
        # put every pose handed to Autoware 4 m under it, and NDT, starting
        # there, settled where the ground it sees matches nothing in the map.
        world = self._world(map_manager, road_below_lanelet2=0.5, spawn_above_road=4.0)
        assert map_manager._z_offset_from_spawn_points(world) == pytest.approx(
            0.5, abs=0.05
        )

    def test_without_a_road_the_spawn_point_is_used(self, map_manager):
        world = self._world(map_manager, road_below_lanelet2=0.0, spawn_above_road=0.0)
        world.get_map().__class__.get_waypoint = lambda self, *a, **k: None
        assert map_manager._z_offset_from_spawn_points(world) == pytest.approx(
            0.0, abs=0.05
        )


def test_the_reference_lines_kept_are_the_loaded_networks() -> None:
    """Kept per network, not per map name: a queue loads the next run's map
    into a new network, whose roads must not be measured against the last."""
    from types import SimpleNamespace

    import numpy as np

    from autoware_carla_scenario.coordinate import transform

    def network(length: float) -> SimpleNamespace:
        line = np.array([[0.0, 0.0], [length, 0.0]])
        short = np.array([[0.0, 0.0]])  # fewer than two vertices: left out
        return SimpleNamespace(
            road_ids_to_object={
                "1": SimpleNamespace(reference_line=line),
                "2": SimpleNamespace(reference_line=short),
            }
        )

    first, second = network(10.0), network(20.0)
    assert transform._road_arc_lengths(first, "1").tolist() == [0.0, 10.0]
    assert transform._ref_lines(first) is transform._ref_lines(first)
    assert transform._road_arc_lengths(second, "1").tolist() == [0.0, 20.0]
    assert transform._road_arc_lengths(second, "2").tolist() == [0.0]
    _network, roads, stacked, starts, _arcs = transform._ref_lines(second)
    assert [road[0] for road in roads] == ["1"] and starts == [0]
    assert stacked.shape == (2, 2)
