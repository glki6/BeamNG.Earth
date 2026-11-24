"""Hong Kong's bus-only lanes (hkbuslanes.py): the Transport Department's BUS_ONLY_LANE lines matched to the map's roads
and turned into the kerb lane's thick solid white line (markings.lane_plan, roads.extras); OpenStreetMap's own
bus:lanes first; a contra-flow lane left alone; nothing while the layer cannot be had (and the build not kept); the
layer fetched page by page, kept on disk and read from there; the road tiles' key in Hong Kong only.

The fixture's lines are SYNTHETIC: the service (portal.csdi.gov.hk) could not be reached from where this was written,
so the layer's attribute names are not pinned and the lines are made on Nathan Road's alignment for the test. They are
read the way the layer's own features are read (ArcGIS JSON, outSR 4326, paths or rings)."""
import gzip
import json
import math
import os
import sys
import tempfile
import time
import unittest
import urllib.parse
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hkbuslanes  # noqa: E402
import mapgeom  # noqa: E402
import markings  # noqa: E402
import revisit  # noqa: E402
import roads  # noqa: E402

KX = 111320.0 * math.cos(math.radians(22.31))
KY = 110574.0
ROAD = [[114.17150, 22.30200], [114.17150, 22.30500]]      # a 333 m road due north (Nathan Road's line, roughly)


def shifted(line, north_m=0.0, east_m=0.0):
    return [[lo + east_m / KX, la + north_m / KY] for lo, la in line]


def feature(paths=None, rings=None, **attrs):
    g = {"paths": paths} if paths is not None else {"rings": rings}
    return {"attributes": {"OBJECTID": 1, **attrs}, "geometry": g}


WEST = shifted(ROAD, east_m=-4.5)           # the lane by the west kerb: the left of a northbound road
TD = {"features": [feature([WEST], REMARKS="7:30am-9:30am Mon-Sat")]}


def road_feats(lines):
    """A roads Features of lon / lat lines, one part and ring each."""
    lonlat, coord_off, ring_off, part_off = [], [0], [0], [0]
    for ln in lines:
        lonlat.append(np.asarray(ln, np.float64))
        coord_off.append(coord_off[-1] + len(ln))
        ring_off.append(ring_off[-1] + 1)
        part_off.append(part_off[-1] + 1)
    n = len(lines)
    return mapgeom.Features(layer="roads", ids=np.array([f"r{i}" for i in range(n)]),
                            kind=np.full(n, mapgeom.LINE, np.uint8), part_off=np.array(part_off, np.int64),
                            ring_off=np.array(ring_off, np.int64), coord_off=np.array(coord_off, np.int64),
                            lonlat=np.concatenate(lonlat))


def sides(td_lines, road=ROAD):
    lanes = hkbuslanes.Lanes(hkbuslanes.parse([feature([ln]) for ln in td_lines]))
    return lanes.sides(hkbuslanes._local(road))


class ParseTest(unittest.TestCase):
    def test_paths_and_rings_are_lines_and_bad_ones_are_left_out(self):
        rows = hkbuslanes.parse([
            feature([WEST], HOURS="24 hours"),
            feature(rings=[WEST + WEST[::-1]]),
            feature([[[114.1, 22.3]]]),                         # (one point)
            feature([[[114.1, float("nan")], [114.2, 22.3]]]),  # (a bad point dropped: one left)
            {"attributes": {}, "geometry": None}, "not a feature"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], {"OBJECTID": 1, "HOURS": "24 hours"}, "every attribute kept (schema not pinned)")
        self.assertEqual(len(rows[1][1]), 4)


class SidesTest(unittest.TestCase):
    def test_a_lane_beside_the_centreline_is_that_sides(self):
        self.assertEqual(sides([WEST]), {"left"})
        self.assertEqual(sides([shifted(ROAD, east_m=4.5)]), {"right"})
        self.assertEqual(sides([WEST[::-1]]), {"left"}, "the side, whichever way the line was drawn")
        self.assertEqual(sides([WEST, shifted(ROAD, east_m=4.5)]), {"left", "right"}, "a lane each way")

    def test_a_lane_on_the_centreline_runs_along_or_against_the_road(self):
        self.assertEqual(sides([shifted(ROAD, east_m=0.5)]), {"along"})
        self.assertEqual(sides([shifted(ROAD, east_m=0.5)[::-1]]), {"against"})

    def test_a_cross_street_a_parallel_road_and_a_short_stretch_take_none(self):
        mid = (ROAD[0][1] + ROAD[1][1]) / 2
        self.assertEqual(sides([[[114.1700, mid], [114.1730, mid]]]), set(), "across: the direction disagrees")
        self.assertEqual(sides([shifted(ROAD, east_m=20.0)]), set(), "20 m away: another road's")
        short = [ROAD[0], [ROAD[0][0], ROAD[0][1] + 0.3 * (ROAD[1][1] - ROAD[0][1])]]
        self.assertEqual(sides([shifted(short, east_m=-4.5)]), set(), "under half the road")
        self.assertEqual(sides([]), set())


class LaneTagsTest(unittest.TestCase):
    def setUp(self):
        self._p = [mock.patch.object(hkbuslanes, "LOCAL", hkbuslanes.parse(TD["features"])),
                   mock.patch.dict(hkbuslanes._state, {"lanes": None})]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()

    def test_a_road_with_a_bus_lane_gets_the_key_beside_its_own_tags(self):
        feats = road_feats([ROAD, shifted(ROAD, east_m=200.0), [[-79.38, 43.65], [-79.379, 43.65]]])
        self.assertEqual(hkbuslanes.lane_tags(feats, 0, None), {hkbuslanes.KEY: "left"})
        self.assertEqual(hkbuslanes.lane_tags(feats, 0, {"lanes": "4"}), {"lanes": "4", hkbuslanes.KEY: "left"})
        self.assertIsNone(hkbuslanes.lane_tags(feats, 1, None), "no bus lane near")
        self.assertEqual(hkbuslanes.lane_tags(feats, 2, {"lanes": "2"}), {"lanes": "2"}, "Toronto: untouched")

    def test_roads_lane_tags_and_parts_carry_it(self):
        feats = road_feats([ROAD])
        self.assertEqual(roads.lane_tags(feats, 0), {hkbuslanes.KEY: "left"})
        self.assertEqual([t for *_, t in roads.lane_tag_parts(feats, 0)], [{hkbuslanes.KEY: "left"}])

    def test_off_leaves_the_tags_alone(self):
        with mock.patch.object(hkbuslanes, "ON", False):
            self.assertIsNone(hkbuslanes.lane_tags(road_feats([ROAD]), 0, None))
            self.assertEqual(hkbuslanes.identity(), "off")

    def test_without_the_layer_nothing_and_the_build_is_not_kept(self):
        with mock.patch.object(hkbuslanes, "LOCAL", None), mock.patch.object(hkbuslanes, "lanes", return_value=None):
            before = revisit.faults()
            feats = road_feats([ROAD])
            self.assertIsNone(hkbuslanes.lane_tags(feats, 0, None))
            self.assertGreater(revisit.faults(), before)
            self.assertNotIn(0, feats.__dict__.get("_hkbus_sides", {}), "asked again once the layer is there")


class PlanTest(unittest.TestCase):
    """markings.lane_plan with the key: the kerb lane's line is a bus lane's ("bus"); Hong Kong keeps to the left."""

    @staticmethod
    def bus_offsets(plan):
        return sorted(round(off, 2) for off, role, how in plan if role == "lane" and how == "bus")

    def test_one_way_the_kerb_lane_on_the_left(self):
        plan = markings.lane_plan({hkbuslanes.KEY: "along"}, False, 9.0, 3, left_hand=True)
        self.assertEqual(self.bus_offsets(plan), [1.5], "between the left kerb's lane and the next")
        self.assertEqual(self.bus_offsets(markings.lane_plan({hkbuslanes.KEY: "right"}, False, 9.0, 3, True)), [-1.5])
        self.assertEqual(self.bus_offsets(markings.lane_plan({hkbuslanes.KEY: "left"}, False, 9.0, 3, True)), [1.5])

    def test_a_contra_flow_lane_is_not_drawn(self):
        self.assertEqual(self.bus_offsets(markings.lane_plan({hkbuslanes.KEY: "against"}, False, 9.0, 3, True)), [])

    def test_two_way_each_directions_kerb(self):
        plan = markings.lane_plan({hkbuslanes.KEY: "along;against"}, True, 12.0, 4, left_hand=True)
        self.assertEqual(self.bus_offsets(plan), [-3.0, 3.0])
        plan = markings.lane_plan({hkbuslanes.KEY: "along"}, True, 12.0, 4, left_hand=True)
        self.assertEqual(self.bus_offsets(plan), [3.0], "with the road's direction: the left kerb in Hong Kong")
        plan = markings.lane_plan({hkbuslanes.KEY: "along"}, True, 12.0, 4, left_hand=False)
        self.assertEqual(self.bus_offsets(plan), [-3.0], "(right-hand traffic: the right kerb)")

    def test_a_single_lane_direction_has_no_line_to_draw(self):
        self.assertEqual(self.bus_offsets(markings.lane_plan({hkbuslanes.KEY: "left"}, True, 7.0, 2, True)), [])

    def test_openstreetmaps_own_lanes_first(self):
        osm = {"bus:lanes": "no|no|designated", hkbuslanes.KEY: "along"}
        self.assertEqual(self.bus_offsets(markings.lane_plan(osm, False, 9.0, 3, left_hand=True)), [-1.5])


class PaintTest(unittest.TestCase):
    """roads.extras in Hong Kong: Figure No. 504's thick solid white line, the other lines as before."""
    LINE = np.array([[0.0, 0, 0], [118, 0, 0]])

    @staticmethod
    def lines(ex, kind):
        return sorted((round(float(w), 2), round(float(ln[0][1]), 2), round(float(np.hypot(*(ln[-1] - ln[0])[:2])), 1))
                      for lk, w, ln in ex if lk.get("kind") == kind)

    def extras(self, sp, osm, country="HK"):
        return roads.extras(sp, self.LINE, 114.17, left_hand=True, country=country, osm=osm)

    def test_the_bus_lane_line_is_thick_and_solid(self):
        sp = roads.Spec(10.0, "paved", True, "primary", True)._replace(lanes=3)
        lane = self.lines(self.extras(sp, {hkbuslanes.KEY: "along"}), "lane")
        self.assertIn((markings.HK_BUS_W, round(10.0 / 2 - 10.0 / 3, 2), 118.0), lane)
        self.assertEqual(sum(1 for w, _, _ in lane if w == markings.HK_BUS_W), 1)
        self.assertEqual({w for w, _, n in lane if w != markings.HK_BUS_W}, {0.10}, "the other lane line: RM 1101")
        plain = self.lines(self.extras(sp, None), "lane")
        self.assertEqual({w for w, _, _ in plain}, {0.10}, "without a bus lane: no thick line")

    def test_a_wide_two_way_road_keeps_its_solid_centre_line(self):
        sp = roads.Spec(14.0, "paved", True, "primary", False)._replace(lanes=4)
        before = self.lines(self.extras(sp, None), "centre")
        after = self.lines(self.extras(sp, {hkbuslanes.KEY: "along;against"}), "centre")
        self.assertEqual(after, before)
        self.assertEqual(sum(1 for w, _, _ in self.lines(self.extras(sp, {hkbuslanes.KEY: "along;against"}), "lane")
                             if w == markings.HK_BUS_W), 2)

    def test_elsewhere_the_style_decides(self):
        sp = roads.Spec(7.0, "paved", True, "secondary", True)._replace(lanes=2)
        se = roads.extras(sp, self.LINE, 12.0, country="SE", osm={"bus:lanes": "designated|no"})
        self.assertEqual({round(float(w), 2) for lk, w, _ in se if lk.get("kind") == "lane"}, {markings.M6_W})


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._p = [mock.patch.dict(hkbuslanes._state, {"cache": self.tmp, "fetcher": None, "lanes": None,
                                                       "down_at": 0.0}),
                   mock.patch.object(hkbuslanes, "LOCAL", None)]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()

    def fetcher(self, pages):
        asked = []

        def get(url, kind=None):
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
            asked.append(q)
            page = int(q["resultOffset"]) // int(q["resultRecordCount"])
            return SimpleNamespace(body=json.dumps(pages[page]).encode())
        return SimpleNamespace(get=get), asked

    def test_fetched_paged_kept_and_read_again(self):
        f2 = feature([shifted(ROAD, east_m=4.5)])
        fetcher, asked = self.fetcher([{"features": TD["features"], "exceededTransferLimit": True},
                                       {"features": [f2]}])
        hkbuslanes._state["fetcher"] = fetcher
        got = hkbuslanes.lanes()
        self.assertEqual(len(got.rows), 2)
        self.assertEqual(len(asked), 2)
        self.assertEqual((asked[0]["outFields"], asked[0]["outSR"]), ("*", "4326"))
        path = os.path.join(self.tmp, "hkbuslanes", hkbuslanes.VERSION, "lanes.json.gz")
        with open(path, "rb") as fh:
            self.assertEqual(len(json.loads(gzip.decompress(fh.read()))), 2)
        ident = got.ident
        hkbuslanes._state["lanes"] = None
        hkbuslanes._state["fetcher"] = SimpleNamespace(get=mock.Mock(side_effect=AssertionError("asked again")))
        self.assertEqual(hkbuslanes.lanes().ident, ident, "fresh on disk: not asked again")
        self.assertEqual(hkbuslanes.identity(), ident)

    def test_down_keeps_the_old_copy_and_backs_off(self):
        self.assertIsNone(hkbuslanes.lanes(), "no fetcher, nothing on disk")
        self.assertEqual(hkbuslanes.identity(), "none")
        hkbuslanes._state["fetcher"] = SimpleNamespace(get=mock.Mock(side_effect=OSError("offline")))
        self.assertIsNone(hkbuslanes.lanes())
        self.assertGreater(hkbuslanes._state["down_at"], 0)
        path = os.path.join(self.tmp, "hkbuslanes", hkbuslanes.VERSION, "lanes.json.gz")
        hkbuslanes._write(path, hkbuslanes.parse(TD["features"]))
        old = time.time() - 2 * hkbuslanes.MAX_AGE_S
        os.utime(path, (old, old))
        self.assertEqual(len(hkbuslanes.lanes().rows), 1, "stale but kept while the service is down")


class KeyTest(unittest.TestCase):
    def test_the_road_tiles_key_in_hong_kong_only(self):
        import blockworker
        self.assertEqual(blockworker._hk_bus_key("SE"), ())
        with mock.patch.object(hkbuslanes, "identity", return_value="abc"):
            self.assertEqual(blockworker._hk_bus_key("HK"), (("hkbuslanes", "abc"),))


if __name__ == "__main__":
    unittest.main()
