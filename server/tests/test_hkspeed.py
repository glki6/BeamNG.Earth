"""Hong Kong's speed limits (hkspeed.py): the Transport Department's posted limits matched to the map's roads, before
the map's own limit; the law's 50 km/h on the ordinary classes the layer does not list once it is loaded; the class
defaults as before while it cannot be had (and the build not kept); the layer fetched page by page, kept on disk and
read from there; the sign tiles' key in Hong Kong only. The fixture's lines are four the service answered on
2026-11-24 (SPEED_LIMIT layer, OBJECTID 1-4, outSR 4326, geometryPrecision 6)."""
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
import hkspeed  # noqa: E402
import mapgeom  # noqa: E402
import navgraph  # noqa: E402
import revisit  # noqa: E402
import roads  # noqa: E402

TD = {"features": [
    {"attributes": {"OBJECTID": 1, "SPEED_LIMIT": "80 km/h", "BOUND": 0},
     "geometry": {"paths": [[[114.147838, 22.288336], [114.147854, 22.288334], [114.148067, 22.288313],
                             [114.148075, 22.288312], [114.148143, 22.288304], [114.148295, 22.288282],
                             [114.14838, 22.288288], [114.148496, 22.288296]]]}},
    {"attributes": {"OBJECTID": 2, "SPEED_LIMIT": "80 km/h", "BOUND": 0},
     "geometry": {"paths": [[[114.122663, 22.376105], [114.122986, 22.376258], [114.123008, 22.376291]]]}},
    {"attributes": {"OBJECTID": 3, "SPEED_LIMIT": "30 km/h", "BOUND": 0},
     "geometry": {"paths": [[[113.976937, 22.243304], [113.977109, 22.24331]]]}},
    {"attributes": {"OBJECTID": 4, "SPEED_LIMIT": "30 km/h", "BOUND": 0},
     "geometry": {"paths": [[[114.15351, 22.553143], [114.153485, 22.5531], [114.153487, 22.553061]]]}},
]}
KX = 111320.0 * math.cos(math.radians(22.29))
KY = 110574.0


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


def shifted(line, north_m=0.0, east_m=0.0):
    return [[lo + east_m / KX, la + north_m / KY] for lo, la in line]


LINE1 = TD["features"][0]["geometry"]["paths"][0]
ALONG = shifted(LINE1, north_m=2.5)                 # the map's road 2.5 m off the TD line (80 km/h)
ACROSS = [[114.14817, 22.28800], [114.14817, 22.28860]]     # a street across it at its middle
BESIDE = shifted(LINE1, north_m=20.0)               # a parallel road 20 m away
FAR = [[114.20, 22.40], [114.201, 22.40]]           # Sha Tin: no posted line near
TORONTO = [[-79.38, 43.65], [-79.379, 43.65]]


class Loaded(unittest.TestCase):
    def setUp(self):
        self._p = [mock.patch.object(hkspeed, "LOCAL", hkspeed.parse(TD["features"])),
                   mock.patch.dict(hkspeed._state, {"limits": None})]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()


class ValuesTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(hkspeed.value_kmh("80 km/h"), 80.0)
        self.assertEqual(hkspeed.value_kmh("30km/h"), 30.0)
        self.assertEqual(hkspeed.value_kmh("110 KM/H"), 110.0)
        self.assertIsNone(hkspeed.value_kmh(None))
        self.assertIsNone(hkspeed.value_kmh("varies"))
        self.assertIsNone(hkspeed.value_kmh("800 km/h"), "not a limit")

    def test_parse(self):
        rows = hkspeed.parse(TD["features"] + [{"attributes": {"SPEED_LIMIT": None}, "geometry": {"paths": []}},
                                               {"attributes": {"SPEED_LIMIT": "50 km/h"},
                                                "geometry": {"paths": [[[114.1, 22.3]]]}}])
        self.assertEqual([r[0] for r in rows], [80.0, 80.0, 30.0, 30.0])
        self.assertEqual(len(rows[0][2]), 8)


class MatchTest(Loaded):
    def test_matched_along_its_line_only(self):
        f = road_feats([ALONG, ACROSS, BESIDE, FAR, TORONTO])
        self.assertEqual(hkspeed.kmh(f, 0), 80.0)
        self.assertEqual(hkspeed.kmh(f, 0, at=0.05), 80.0)
        self.assertIsNone(hkspeed.kmh(f, 1), "a cross street does not take the line it crosses")
        self.assertIsNone(hkspeed.kmh(f, 2), "a parallel road 20 m away is not on it")
        self.assertIsNone(hkspeed.kmh(f, 3), "no posted line near: none")
        self.assertIsNone(hkspeed.kmh(f, 4), "outside Hong Kong: never asked")

    def test_reversed_road_matches_too(self):
        f = road_feats([list(reversed(ALONG))])
        self.assertEqual(hkspeed.kmh(f, 0), 80.0)

    def test_a_short_line_covers_its_own_stretch(self):
        line3 = TD["features"][2]["geometry"]["paths"][0]
        road = [[line3[0][0] - 60 / KX, line3[0][1]], [line3[1][0] + 60 / KX, line3[1][1]]]   # 18 m of 138 m posted
        f = road_feats([road])
        self.assertEqual(hkspeed.kmh(f, 0, at=0.5), 30.0)
        self.assertIsNone(hkspeed.kmh(f, 0, at=0.05), "the road's end, 60 m from the posted stretch")

    def test_off(self):
        f = road_feats([ALONG])
        with mock.patch.object(hkspeed, "ON", False):
            self.assertIsNone(hkspeed.kmh(f, 0))
            self.assertEqual(hkspeed.identity(), "off")


class RoadMphTest(Loaded):
    def kmh(self, f, i, spec, country="HK", **kw):
        return navgraph.road_mph(f, i, spec, mph=False, country=country, **kw) * 1.609344

    def test_posted_limit_then_the_law(self):
        f = road_feats([ALONG, FAR])
        f.rules = {}
        primary = roads.Spec(9.0, "paved", False, "primary", False)
        motorway = roads.Spec(11.0, "paved", False, "motorway", True)
        self.assertAlmostEqual(self.kmh(f, 0, primary), 80.0)
        self.assertAlmostEqual(self.kmh(f, 0, motorway), 80.0, msg="a motorway on a posted line: its limit")
        self.assertTrue(navgraph.mapped_limit(f, 0))
        self.assertAlmostEqual(self.kmh(f, 1, primary), 50.0, msg="not listed: the law's 50 km/h (was 80 out of town)")
        self.assertAlmostEqual(self.kmh(f, 1, primary._replace(cls="residential", urban=True)), 50.0)
        self.assertFalse(navgraph.mapped_limit(f, 1))
        self.assertAlmostEqual(self.kmh(f, 1, motorway), 120.0, msg="a motorway it does not list: a miss, as before")
        self.assertLess(self.kmh(f, 1, primary._replace(cls="service")), 30.0, "a service road: as before")
        self.assertAlmostEqual(self.kmh(f, 1, primary, country="CN"), 80.0, msg="not Hong Kong's country: as before")

    def test_posted_limit_before_the_maps(self):
        f = road_feats([ALONG, FAR])

        class T:
            start, end, value = [0.0], [1.0], [70.0]

            def rows(self, i):
                return [0]
        f.rules = {"max_speed_kmh": T()}
        primary = roads.Spec(9.0, "paved", False, "primary", False)
        self.assertAlmostEqual(self.kmh(f, 0, primary), 80.0, msg="the Department's gazetted limit wins")
        self.assertAlmostEqual(self.kmh(f, 1, primary), 70.0, msg="not posted: the map's")

    def test_tests_feature_stubs_unchanged(self):
        class F:            # (the other tests' stand-ins: no geometry)
            rules = {}
        mw = roads.Spec(11.0, "paved", False, "motorway", True)
        self.assertAlmostEqual(navgraph.road_mph(F, 0, mw), 65.0)
        self.assertFalse(navgraph.mapped_limit(F, 0))


class UnavailableTest(unittest.TestCase):
    def test_class_defaults_and_not_kept(self):
        f = road_feats([ALONG])
        f.rules = {}
        primary = roads.Spec(9.0, "paved", False, "primary", False)
        with mock.patch.object(hkspeed, "LOCAL", None), \
                mock.patch.dict(hkspeed._state, {"limits": None, "cache": None, "fetcher": None, "down_at": 0.0}):
            before = revisit.faults()
            kmh = navgraph.road_mph(f, 0, primary, mph=False, country="HK") * 1.609344
            self.assertAlmostEqual(kmh, 80.0, msg="the class default out of town, not the law's 50: nothing loaded")
            self.assertGreater(revisit.faults(), before, "made short of the layer: not kept")
            self.assertEqual(hkspeed.identity(), "none")


class FetchTest(unittest.TestCase):
    def test_fetched_in_pages_kept_and_read_back(self):
        asked = []

        class Fetcher:
            def get(self, url, kind="demand"):
                asked.append(url)
                q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                off = int(q["resultOffset"][0])
                feats = TD["features"][off // hkspeed.hklidar.PAGE * 2:][:2]
                body = {"features": feats, "exceededTransferLimit": off == 0}
                return SimpleNamespace(body=json.dumps(body).encode())
        with tempfile.TemporaryDirectory() as d, mock.patch.object(hkspeed, "LOCAL", None), \
                mock.patch.dict(hkspeed._state, {"limits": None, "cache": d, "fetcher": Fetcher(), "down_at": 0.0}):
            lim = hkspeed.limits()
            self.assertEqual(len(asked), 2, "two pages")
            self.assertTrue(all(u.startswith(hkspeed.URL + "?") and "outSR=4326" in u for u in asked))
            self.assertEqual(sorted(set(lim.kmh.tolist())), [30.0, 80.0])
            path = os.path.join(d, "hkspeed", hkspeed.VERSION, "limits.json.gz")
            with open(path, "rb") as fh:
                self.assertEqual(len(json.loads(gzip.decompress(fh.read()))), 4)
            ident = hkspeed.identity()
            self.assertEqual(len(ident), 12)
            # another process: read from disk, not asked again
            hkspeed._state["limits"] = None
            self.assertEqual(hkspeed.identity(), ident)
            self.assertEqual(len(asked), 2)
            # a month on: asked again; the service down keeps the old copy
            old = time.time() - hkspeed.MAX_AGE_S - 60
            os.utime(path, (old, old))
            hkspeed._state["limits"] = None

            class Down:
                def get(self, url, kind="demand"):
                    raise OSError("unreachable")
            hkspeed._state["fetcher"] = Down()
            self.assertIsNotNone(hkspeed.limits())
            self.assertEqual(hkspeed.limits().ident, ident)


class SignKeyTest(unittest.TestCase):
    def test_key_in_hong_kong_only(self):
        import blockworker
        self.assertEqual(blockworker._hk_speed_key("DE"), ())
        self.assertEqual(blockworker._hk_speed_key(None), ())
        with mock.patch.object(hkspeed, "identity", lambda: "abc"):
            self.assertEqual(blockworker._hk_speed_key("HK"), (("hkspeed", "abc"),))


if __name__ == "__main__":
    unittest.main()
