"""Hong Kong's zebra crossings from the Transport Department's road network (hkzebras.py): the layer's points, where
they stand on the roads, their stripes, give-way lines, zig-zags and beacons, and the junctions' crossings unstriped
(studs at signals).

The sandbox that wrote these tests could not reach the CSDI service, so the service's answer here is a saved sample:
points and attributes copied from the live layer (TRAFFIC_FEATURES, layer 1), trimmed to the fields parse reads."""
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geodesy  # noqa: E402
import guardrails  # noqa: E402
import hkzebras  # noqa: E402
import revisit  # noqa: E402
import roads  # noqa: E402

# The layer's answer (outSR 4326): FEATURE_TYPE 1 zebra, 2 yellow box (several RD_IDs), 3 toll point, 4 cul-de-sac.
SAMPLE = [
    {"attributes": {"OBJECTID": 2, "FEATURE_TYPE": 1}, "geometry": {"x": 114.1600307749322, "y": 22.324228910991497}},
    {"attributes": {"OBJECTID": 4, "FEATURE_TYPE": 1}, "geometry": {"x": 114.16605303783867, "y": 22.311029042125369}},
    {"attributes": {"OBJECTID": 6, "FEATURE_TYPE": 1}, "geometry": {"x": 114.16721227624853, "y": 22.342939196046572}},
    {"attributes": {"OBJECTID": 30, "FEATURE_TYPE": 2}, "geometry": {"x": 114.16459374616829, "y": 22.312943651190739}},
    {"attributes": {"OBJECTID": 35, "FEATURE_TYPE": 2}, "geometry": {"x": 114.16513676508394, "y": 22.323444215881953}},
    {"attributes": {"OBJECTID": 2680, "FEATURE_TYPE": 3}, "geometry": {"x": 114.14227090897825, "y": 22.289983832589236}},
    {"attributes": {"OBJECTID": 866, "FEATURE_TYPE": 4}, "geometry": {"x": 114.13842366790142, "y": 22.381165325122517}},
    # (not the layer's: one outside Hong Kong, one without a point, one without a type)
    {"attributes": {"FEATURE_TYPE": 1}, "geometry": {"x": 113.5, "y": 22.2}},
    {"attributes": {"FEATURE_TYPE": 1}, "geometry": None},
    {"attributes": {}, "geometry": {"x": 114.16, "y": 22.32}},
]
ZEBRAS = [[114.1600308, 22.3242289], [114.166053, 22.311029], [114.1672123, 22.3429392]]


def reset():
    hkzebras._state.update(cache=None, fetcher=None, points=None, ident=None, down_at=0.0)
    hkzebras.LOCAL = None


def road(fid, pts=((0.0, 0.0), (200.0, 0.0)), width=8.0, kind="paved", cls="tertiary"):
    line = np.c_[np.array(pts, np.float64), np.zeros(len(pts))]
    return fid, roads.Spec(width, kind, True, cls, False, kmh=50.0), line, 0.0


def found(*xy):
    return {"hk_zebras": [tuple(map(float, p)) for p in xy], "priority": None}


def crosswalk(x, y0=-4.0, y1=4.0, **kw):
    return dict({"a": np.array([x, y0, 0.0]), "b": np.array([x, y1, 0.0]), "w": 3.0}, **kw)


class Flat:
    """A frame where a degree is 10 m (east: lon, north: lat)."""
    def geodetic_to_enu(self, lat, lon, h=0.0):
        return np.asarray(lon, np.float64) * 10.0, np.asarray(lat, np.float64) * 10.0, np.zeros(np.shape(lat))

    def enu_to_geodetic(self, e, n, u=0.0):
        return np.asarray(n, np.float64) / 10.0, np.asarray(e, np.float64) / 10.0, np.zeros(np.shape(e))


class ParseTest(unittest.TestCase):
    def test_zebras_only(self):
        self.assertEqual(hkzebras.parse(SAMPLE), ZEBRAS)

    def test_the_same_from_any_order(self):
        self.assertEqual(hkzebras.parse(list(reversed(SAMPLE)) + SAMPLE[:2]), ZEBRAS)

    def test_nothing(self):
        self.assertEqual(hkzebras.parse(None), [])
        self.assertEqual(hkzebras.parse([]), [])


class WantedTest(unittest.TestCase):
    def tearDown(self):
        hkzebras.ON = True

    def test_hong_kong_only(self):
        self.assertTrue(hkzebras.wanted("HK"))
        self.assertTrue(hkzebras.wanted("hk"))
        for c in ("SE", "DE", "GB", "", None):
            self.assertFalse(hkzebras.wanted(c), c)

    def test_switched_off(self):
        hkzebras.ON = False
        self.assertFalse(hkzebras.wanted("HK"))
        self.assertEqual(hkzebras.identity(), "off")


class SourceTest(unittest.TestCase):
    def setUp(self):
        reset()
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        reset()
        self.dir.cleanup()

    def fetcher(self, pages):
        asked = []

        def get(url, kind=None):
            asked.append(url)
            return SimpleNamespace(body=json.dumps(pages[len(asked) - 1]).encode())
        return SimpleNamespace(get=get), asked

    def test_fetched_once_and_kept(self):
        f, asked = self.fetcher([{"features": SAMPLE[:3], "exceededTransferLimit": True}, {"features": SAMPLE[3:]}])
        hkzebras.configure(self.dir.name, f)
        self.assertEqual(hkzebras.points(), ZEBRAS)
        self.assertEqual(len(asked), 2)
        self.assertIn("FEATURE_TYPE%3D1", asked[0])
        self.assertIn("outSR=4326", asked[0])
        path = os.path.join(self.dir.name, "hkzebras", hkzebras.VERSION, "zebras.json.gz")
        with gzip.open(path, "rt") as fh:
            self.assertEqual(json.load(fh), ZEBRAS)
        ident = hkzebras.identity()
        self.assertRegex(ident, r"^[0-9a-f]{12}$")
        # a new process reads the kept file, asking nothing
        reset()
        f2, asked2 = self.fetcher([])
        hkzebras.configure(self.dir.name, f2)
        self.assertEqual(hkzebras.points(), ZEBRAS)
        self.assertEqual(asked2, [])
        self.assertEqual(hkzebras.identity(), ident)

    def test_offline_is_none_and_a_fault(self):
        def down(url, kind=None):
            raise OSError("no route")
        hkzebras.configure(self.dir.name, SimpleNamespace(get=down))
        self.assertEqual(hkzebras.identity(), "none")
        before = revisit.faults()
        self.assertIsNone(hkzebras.read(None, Flat(), (0, 0, 100, 100)))
        self.assertEqual(revisit.faults(), before + 1)
        self.assertFalse(os.path.exists(os.path.join(self.dir.name, "hkzebras")))

    def test_the_services_error_is_offline(self):
        f, _ = self.fetcher([{"error": {"code": 400}}])
        hkzebras.configure(self.dir.name, f)
        self.assertIsNone(hkzebras.points())

    def test_local(self):
        hkzebras.LOCAL = [[11.0, 1.0], [99.0, 9.0]]
        self.assertEqual(hkzebras.identity(), "local")
        got = hkzebras.read(None, Flat(), (100.0, 0.0, 120.0, 20.0))
        self.assertEqual(got, {"hk_zebras": [(110.0, 10.0)], "priority": None})


class NodesTest(unittest.TestCase):
    def test_within_reach_of_the_tile(self):
        pts = [[-5.0, 5.0], [0.0, 0.0], [5.0, 5.0], [15.0, 5.0], [17.0, 5.0]]   # (Flat: a degree is 10 m)
        got = hkzebras.nodes(Flat(), pts, (40.0, 40.0, 80.0, 80.0))["hk_zebras"]     # (REACH_M 80 round it)
        self.assertEqual(got, [(0.0, 0.0), (50.0, 50.0), (150.0, 50.0)])
        self.assertTrue(hkzebras.is_hk(hkzebras.nodes(Flat(), [], (0, 0, 1, 1))))
        self.assertIsNone(hkzebras.nodes(Flat(), [], (0, 0, 1, 1))["priority"])

    def test_not_hong_kongs(self):
        self.assertFalse(hkzebras.is_hk(None))
        self.assertFalse(hkzebras.is_hk({"crossings": [], "priority": np.zeros((0, 3))}))


class PlaceTest(unittest.TestCase):
    def test_across_the_road_kerb_to_kerb(self):
        (z,) = hkzebras.place([road(7)], found((100.0, 2.0)))
        self.assertEqual(z["fid"], 7)
        np.testing.assert_allclose(z["c"], [100.0, 0.0, 0.0])
        self.assertAlmostEqual(z["s"], 100.0)
        cw = hkzebras.walk(z)
        np.testing.assert_allclose(sorted([cw["a"][1], cw["b"][1]]), [-4.0, 4.0])
        self.assertEqual(cw["w"], hkzebras.ZEBRA_W)

    def test_off_the_road_or_not_a_road(self):
        self.assertEqual(hkzebras.place([road(1)], found((100.0, 9.0))), [])
        self.assertEqual(hkzebras.place([road(1, kind="gravel")], found((100.0, 0.0))), [])
        self.assertEqual(hkzebras.place([road(1, cls="motorway")], found((100.0, 0.0))), [])
        self.assertEqual(hkzebras.place([], found((100.0, 0.0))), [])

    def test_one_crossing_of_two_points(self):
        self.assertEqual(len(hkzebras.place([road(1)], found((100.0, 0.0), (102.0, 1.0)))), 1)
        self.assertEqual(len(hkzebras.place([road(1)], found((100.0, 0.0), (120.0, 1.0)))), 2)

    def test_the_nearest_road(self):
        cross = road(2, pts=((100.0, -100.0), (100.0, 100.0)))
        (z,) = hkzebras.place([road(1), cross], found((101.0, 40.0)))
        self.assertEqual(z["fid"], 2)


class MarksTest(unittest.TestCase):
    def test_give_way_lines_and_zig_zags(self):
        (z,) = hkzebras.place([road(1)], found((100.0, 0.0)))
        got = hkzebras.marks(z)
        # 8 m across: 9 dashes of 0.6 m a side; 8 zig-zags along each kerb on each side
        self.assertEqual(len(got), 2 * 9 + 4 * hkzebras.ZIG_N)
        xs = sorted({round(float(q[:, 0].mean()), 2) for q in got[:9]})
        self.assertEqual(len(xs), 1)
        self.assertAlmostEqual(abs(xs[0] - 100.0), hkzebras.ZEBRA_W / 2 + hkzebras.GIVE_WAY_M, places=6)
        for q in got:
            self.assertEqual(q.shape, (4, 3))
            self.assertLessEqual(float(np.abs(q[:, 1]).max()), 4.0 + 1e-6)

    def test_zig_zags_stop_short_of_the_roads_end(self):
        # (the layer's zebras stand 8 to 19 m from a street's end)
        (z,) = hkzebras.place([road(1)], found((188.0, 0.0)))
        n = len(hkzebras.marks(z)) - 2 * 9
        self.assertEqual(n, 2 * hkzebras.ZIG_N + 2 * 3)
        (z,) = hkzebras.place([road(1)], found((192.0, 0.0)))
        self.assertEqual(len(hkzebras.marks(z)) - 2 * 9, 2 * hkzebras.ZIG_N)
        for q in hkzebras.marks(z):
            self.assertLessEqual(float(q[:, 0].max()), 200.0 - hkzebras.ZIG_END_M + 0.2)


class PaintTest(unittest.TestCase):
    def test_a_zebra_takes_the_junctions_crossing(self):
        walks = [crosswalk(101.0), crosswalk(150.0), crosswalk(50.0, stop=True)]
        out, ramps, zw, extra = hkzebras.paint([road(1)], found((100.0, 0.0)), walks, {}, None)
        self.assertEqual(len(zw), 1)
        self.assertTrue(any(c is zw[0] for c in out))
        plain = [c for c in out if c.get("paint") is False]
        self.assertEqual([float(c["a"][0]) for c in plain], [150.0])
        self.assertTrue(any(c is walks[2] for c in out))                # (a stop line stays)
        self.assertNotIn("paint", walks[1])                             # (the caller's left as it was)
        self.assertEqual(sorted(ramps), [(1, -1), (1, 1)])
        self.assertEqual(len(extra), 2 * 9 + 4 * hkzebras.ZIG_N)       # (no studs: no signals)

    def test_studs_at_signals_only(self):
        walks = [crosswalk(50.0), crosswalk(150.0)]
        _, _, _, extra = hkzebras.paint([road(1)], found(), walks, {}, np.array([[52.0, 6.0]]))
        self.assertEqual(len(extra), len(hkzebras.studs(walks[0])))
        self.assertGreater(len(extra), 0)
        for q in extra:
            self.assertAlmostEqual(float(q[:, 0].mean()), 50.0 + 1.5 * np.sign(q[0, 0] - 50.0), places=6)

    def test_studs_two_rows(self):
        got = hkzebras.studs(crosswalk(50.0))
        self.assertEqual(len(got) % 2, 0)
        self.assertEqual(sorted({round(float(q[:, 0].mean()), 3) for q in got}), [48.5, 51.5])
        self.assertEqual(hkzebras.studs(crosswalk(50.0, -0.2, 0.2)), [])

    def test_the_junctions_ramps_kept(self):
        jr = {(5, 1): [(np.zeros(2), np.array([1.0, 0.0]), -1.0, 1.0)]}
        _, ramps, _, _ = hkzebras.paint([road(1)], found((100.0, 0.0)), [], jr, None)
        self.assertEqual(len(ramps[(5, 1)]), 1)
        self.assertEqual(len(jr), 1)


class BeaconTest(unittest.TestCase):
    def test_one_at_each_kerb(self):
        got = hkzebras.beacons([road(1)], found((100.0, 0.0)))
        self.assertEqual(sorted(round(float(p[1]), 3) for p, _ in got), [-4.6, 4.6])
        for p, facing in got:
            self.assertAlmostEqual(float(p[0]), 100.0)
            np.testing.assert_allclose(np.abs(facing), [1.0, 0.0])

    def test_rows_of_the_tile(self):
        frame = geodesy.EnuFrame(22.3242, 114.16)

        def elev(la, lo):
            return np.full(np.shape(la), 12.0)
        rows = hkzebras.tile_rows(frame, [road(1)], found((100.0, 0.0)), (0.0, 0.0, 200.0, 200.0), elev)
        self.assertEqual(len(rows), 1)                                  # (the other stands in the tile below)
        r = rows[0]
        self.assertEqual(guardrails.GROUPS[r[0]], hkzebras.BEACON_GROUP)
        self.assertAlmostEqual(r[1], 100.0, places=2)
        self.assertAlmostEqual(r[2], 4.6, places=2)
        self.assertAlmostEqual(r[3], 12.0, delta=0.05)
        self.assertEqual(r[6], hkzebras.BEACON_H)
        self.assertEqual(len(r), 17)
        self.assertEqual(hkzebras.tile_rows(frame, [road(1)], None, (0.0, 0.0, 200.0, 200.0), elev), [])

    def test_the_model_is_shipped(self):
        mod = Path(__file__).resolve().parents[2] / "mod/beamng_earth"
        shapes = mod / "levels/beamng_earth/art/shapes/beamngEarth"
        self.assertTrue((shapes / "bne_hk_beacon.dae").is_file())
        mats = json.loads((shapes / "hk_beacon.materials.json").read_text())
        dae = (shapes / "bne_hk_beacon.dae").read_text()
        for name in mats:
            self.assertIn(f'name="{name}"', dae)
        lua = (mod / "lua/ge/extensions/beamngEarth/trees.lua").read_text()
        self.assertIn("M.CATALOG.hk_beacon = { { '/levels/beamng_earth/art/shapes/beamngEarth/bne_hk_beacon', "
                      f"{hkzebras.BEACON_H} }} }}", lua)


class TileTest(unittest.TestCase):
    """Through build_tile: a street and a side street in Hong Kong, a zebra on the street, and a junction crossing at
    x 150 (handed to build_tile in junctions.find's answer: these streets have no sidewalks for it to find one)."""
    def tile(self, nodes, signals=None, country="HK"):
        import test_roads as tr
        import junctions
        from unittest import mock
        real = junctions.find

        def find(*a, **kw):
            corners, cuts, _cw, ramps, lines, internal = real(*a, **kw)
            return corners, cuts, [crosswalk(150.0, -2.0, 6.0)], ramps, lines, internal
        feats = tr.features([[(-300, 2), (600, 2)], [(100, 2), (100, 300)]], ["tertiary", "residential"])
        urban = lambda la, lo: np.ones(np.shape(la), bool)      # noqa: E731
        with mock.patch.object(junctions, "find", find):
            return roads.build_tile(tr.FRAME, 0, 0, [feats], lambda a, b: np.zeros(np.shape(a)), urban, walks=[],
                                    country=country, mph=True, left_hand=True, signals=signals, nodes=nodes)

    @staticmethod
    def stripes(ch):
        return sorted(round(float(np.mean(np.array(c["n"])[:, 0])), 1) for c in ch if c["m"] == "crossing_white")

    @staticmethod
    def studs(ch):
        return [c for c in ch if c["m"] == "line_white" and c.get("poly")
                and np.ptp(np.array(c["poly"])[:, 0]) < 0.2 and np.ptp(np.array(c["poly"])[:, 1]) < 0.2]

    def test_the_zebra_striped_and_the_junction_not(self):
        self.assertEqual(self.stripes(self.tile(None, signals=np.zeros((0, 2)))), [150.0])     # (no layer: as before)
        ch = self.tile(found((60.0, 2.0)), signals=np.zeros((0, 2)))
        self.assertEqual(self.stripes(ch), [60.0])
        (z,) = [c for c in ch if c["m"] == "crossing_white"]
        self.assertEqual(z["w"], hkzebras.ZEBRA_W)
        marks = [c for c in ch if c["m"] == "line_white" and c.get("poly")
                 and 45.0 < np.array(c["poly"])[:, 0].mean() < 75.0]
        self.assertGreaterEqual(len(marks), 2 * 9)                      # (the give-way lines, the zig-zags)
        self.assertEqual(self.studs(ch), [])

    def test_studs_at_the_signals(self):
        lit = self.tile(found(), signals=np.array([[154.0, 8.0]]))
        self.assertEqual(self.stripes(lit), [])
        self.assertTrue(self.studs(lit))
        self.assertEqual(self.studs(self.tile(found(), signals=np.array([[250.0, 8.0]]))), [])

    def test_elsewhere_as_before(self):
        self.assertEqual(self.stripes(self.tile(None, signals=np.zeros((0, 2)), country="GB")), [150.0])


class KeyTest(unittest.TestCase):
    def setUp(self):
        reset()

    def tearDown(self):
        reset()

    def test_road_tile_key(self):
        import blockworker
        self.assertEqual(blockworker._hk_zebra_key("SE"), ())
        self.assertEqual(blockworker._hk_zebra_key(None), ())
        hkzebras.LOCAL = [[114.16, 22.32]]
        self.assertEqual(blockworker._hk_zebra_key("HK"), (("hkzebras", "local"),))


if __name__ == "__main__":
    unittest.main()
