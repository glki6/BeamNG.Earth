"""Hong Kong's bus stop signs (hkbusstops.py), and sebusstops' poles in left-hand traffic."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import geodesy
import hkbusstops
import mapgeom
import roads
import sebusstops
import signstream


def features(frame, stops):
    """An infrastructure layer of points: stops = [(fid, x, y, class, name, tags)]."""
    xy = np.array([[s[1], s[2]] for s in stops], np.float64).reshape(-1, 2)
    lat, lon, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], np.zeros(len(xy)))
    off = np.arange(len(stops) + 1, dtype=np.int64)
    return mapgeom.Features(
        layer="infrastructure", ids=np.array([s[0] for s in stops]), kind=np.full(len(stops), mapgeom.POINT, np.uint8),
        part_off=off, ring_off=off, coord_off=off, lonlat=np.column_stack((lon, lat)),
        attrs={"subtype": np.array(["transit"] * len(stops)), "class": np.array([s[3] for s in stops]),
               "name": np.array([s[4] for s in stops], object),
               "tags": np.array([json.dumps(s[5], ensure_ascii=False) for s in stops])})


class Map:
    def __init__(self, feats):
        self.feats = feats

    def cell(self, key, layers=(), kind=None):
        return type("Cell", (), {"layers": {"infrastructure": self.feats}})()


class Roads:
    """A sign context with one road along x at y, 7 m wide."""
    def __init__(self, y):
        self.y = y

    def pavements(self, frame, points, left_hand=False):
        return [("rd", None, np.array([[-5000.0, self.y, 0.0], [5000.0, self.y, 0.0]]), 7.0)]


class NameTest(unittest.TestCase):
    def setUp(self):
        self.frame = geodesy.EnuFrame(22.28, 114.18, 0.0)

    def test_hong_kongs_two_names_in_one(self):
        self.assertEqual(hkbusstops.split_name(" 銅鑼灣 (怡和街)  Causeway Bay (Yee Wo Street) "),
                         ("銅鑼灣 (怡和街)", "Causeway Bay (Yee Wo Street)"))
        self.assertEqual(hkbusstops.split_name("中環 - Central"), ("中環", "Central"))
        self.assertEqual(hkbusstops.split_name("港鐵旺角站A2出口"), ("港鐵旺角站A2出口", ""))
        self.assertEqual(hkbusstops.split_name(None), ("", ""))
        self.assertEqual(hkbusstops.split_name("nan"), ("", ""))

    def test_label_reads_the_name_else_the_language_tags_english_in_capitals(self):
        f = features(self.frame, [("a", 0, 0, "bus_stop", "灣仔碼頭 Wan Chai Ferry Pier", {}),
                                  ("b", 0, 0, "bus_stop", "金鐘站", {"name:en": "admiralty station"}),
                                  ("c", 0, 0, "bus_stop", "Stanley Village", {"name:zh": "赤柱村"}),
                                  ("d", 0, 0, "bus_stop", None, {"name:zh-hant": "大埔墟站", "name:en": "tai po market"}),
                                  ("e", 0, 0, "bus_stop", None, {"name:zh": "abc"}),
                                  ("f", 0, 0, "bus_stop", None, {})])
        self.assertEqual(hkbusstops.label(f, 0), ("灣仔碼頭", "WAN CHAI FERRY PIER"))
        self.assertEqual(hkbusstops.label(f, 1), ("金鐘站", "ADMIRALTY STATION"))
        self.assertEqual(hkbusstops.label(f, 2), ("赤柱村", "STANLEY VILLAGE"))
        self.assertEqual(hkbusstops.label(f, 3), ("大埔墟站", "TAI PO MARKET"))
        self.assertEqual(hkbusstops.label(f, 4), ("", ""))         # (a name:zh with no Chinese character in it)
        self.assertEqual(hkbusstops.label(f, 5), ("", ""))

    def test_only_hong_kong(self):
        self.assertTrue(hkbusstops.wanted("hk"))
        self.assertFalse(hkbusstops.wanted("CN"))
        self.assertFalse(hkbusstops.wanted("SE"))
        self.assertFalse(hkbusstops.wanted(None))


class LeftHandPoleTest(unittest.TestCase):
    near = [("rd", type("S", (), {"width": 7.0})(), np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]))]

    def test_at_the_kerb_on_the_traffics_left_ahead_of_the_stop(self):
        x, y, gx, gy = sebusstops.pole(50.0, -5.0, self.near, left_hand=True)    # south of a road east: traffic west
        self.assertAlmostEqual(y, -(3.5 + sebusstops.KERB_M))
        self.assertAlmostEqual(x, 50.0 - sebusstops.AHEAD_M)
        self.assertEqual((gx, gy), (-1.0, 0.0))
        x, y, gx, gy = sebusstops.pole(50.0, 2.0, self.near, left_hand=True)     # north: east
        self.assertAlmostEqual(y, 3.5 + sebusstops.KERB_M)
        self.assertAlmostEqual(x, 50.0 + sebusstops.AHEAD_M)
        self.assertEqual(gx, 1.0)

    def test_right_hand_unchanged(self):
        self.assertEqual(sebusstops.pole(50.0, -5.0, self.near)[2:], (1.0, 0.0))

    def test_a_halt_on_a_one_way_road_stands_on_its_left(self):
        for d, want in ((1, 1.0), (-1, -1.0)):
            near = [("rd", type("S", (), {"width": 7.0, "oneway": True, "dir": d})(), self.near[0][2])]
            for fid in "abcd":
                x, y, gx, gy = sebusstops.pole(50.0, 0.0, near, (), fid, left_hand=True)
                self.assertAlmostEqual(y, want * (3.5 + sebusstops.KERB_M))
                self.assertEqual(gx, float(d))                      # (the pole faces the one way's own traffic)

    def test_rows_of_passes_it_on(self):
        got = sebusstops.rows_of([("a", 50.0, -5.0, "中環", "CENTRAL")], self.near, left_hand=True)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][2:], (-1.0, 0.0, "中環", "CENTRAL"))


@unittest.skipUnless(hkbusstops.cjk_font(), "no Chinese typeface on this computer")
class PictureTest(unittest.TestCase):
    def test_the_flag_is_black_on_white(self):
        im = hkbusstops.flag_picture()
        self.assertEqual(im.shape, (int(round(hkbusstops.FLAG_H_M * hkbusstops.PX_PER_M)),
                                    int(round(hkbusstops.FLAG_W_M * hkbusstops.PX_PER_M)), 3))
        self.assertEqual(tuple(im[1, 1]), hkbusstops.WHITE)
        self.assertEqual(tuple(im[im.shape[0] // 2, 4]), hkbusstops.BLACK)    # the edge line
        self.assertEqual(tuple(im[int(0.27 * im.shape[0]), int(0.45 * im.shape[1])]), hkbusstops.BLACK)  # the bus
        colours = {tuple(c) for c in im.reshape(-1, 3)[::7]}
        self.assertTrue(all(max(c) - min(c) <= 2 for c in colours))           # (greys only: no operator's colours)
        self.assertTrue(np.array_equal(im, hkbusstops.flag_picture()))

    def test_a_name_plate_fits_its_names(self):
        rgb, w, h = hkbusstops.name_picture("中環", "CENTRAL")
        self.assertAlmostEqual(w, hkbusstops.MIN_W_M, delta=0.01)
        self.assertEqual(rgb.shape[:2], (int(round(h * hkbusstops.PX_PER_M)), int(round(w * hkbusstops.PX_PER_M))))
        rgb2, w2, h2 = hkbusstops.name_picture("", "CENTRAL")
        self.assertLess(h2, h)                                      # (the English only: one line)
        long = hkbusstops.name_picture("尖沙咀碼頭巴士總站北面上落客區", "TSIM SHA TSUI STAR FERRY PIER BUS TERMINUS")
        self.assertAlmostEqual(long[1], hkbusstops.MAX_W_M)
        self.assertIsNone(hkbusstops.name_picture("", ""))


class TileTest(unittest.TestCase):
    def setUp(self):
        self.frame = geodesy.EnuFrame(22.28, 114.18, 0.0)
        self.box = roads.tile_bbox(0, 0)
        self.x0, self.yc = self.box[0] + 60.0, (self.box[1] + self.box[3]) / 2.0     # the road along the tile's middle

    def tile(self, stops, country="HK", tx=0, ty=0):
        stops = [(s[0], s[1], s[2] + self.yc, *s[3:]) for s in stops]
        with tempfile.TemporaryDirectory() as out:
            mesh, faces = hkbusstops.sign_tile(Map(features(self.frame, stops)), self.frame, tx, ty,
                                               lambda la, lo: np.full(np.shape(la), 12.0), out, "/v", Roads(self.yc), (),
                                               country, True)
            return mesh, faces, sorted(os.listdir(out))

    def test_a_named_stop_has_its_flag_and_name_on_both_sides(self):
        mesh, faces, files = self.tile([("a", self.x0, -5.0, "bus_stop", "北角碼頭 North Point Ferry Pier", {})])
        p, t, uv, materials, mi = mesh
        self.assertEqual(len(materials), 3)                         # metal, the flag, the name
        self.assertEqual(sorted(faces), sorted(materials[1:]))
        self.assertEqual(files, sorted(m + ".color.dds" for m in materials[1:]))
        for m in (1, 2):                                            # each face twice: one from each way
            self.assertEqual(int((np.asarray(mi) == m).sum()), 4)
        y = p[:, 1] + self.box[1] - self.yc
        self.assertLess(float(y.max()), -3.5)                       # nothing over the carriageway
        self.assertGreater(float(y.min()), -3.5 - sebusstops.KERB_M - hkbusstops.MAX_W_M)
        z = p[:, 2]
        self.assertGreater(float(z.max()) - 12.0, hkbusstops.MOUNT_M + hkbusstops.GAP_M + hkbusstops.FLAG_H_M)
        self.assertLess(float(z.max()) - 12.0, 3.4)

    def test_stops_of_one_name_share_its_picture_and_an_unnamed_one_has_the_flag(self):
        mesh, faces, _ = self.tile([("a", self.x0, -5.0, "bus_stop", "天后 Tin Hau", {}),
                                    ("b", self.x0 + 30.0, 5.0, "bus_stop", "天后 Tin Hau", {}),
                                    ("c", self.x0 + 90.0, 5.0, "bus_stop", None, {})])
        mi = np.asarray(mesh[4])
        self.assertEqual(len(faces), 2)
        self.assertEqual(int((mi == 1).sum()), 12)                  # the flag: three poles, two faces of two triangles
        self.assertEqual(int((mi == 2).sum()), 8)

    def test_none_elsewhere_nor_for_another_tiles_stop(self):
        stop = [("a", self.x0, -5.0, "bus_stop", "天后 Tin Hau", {})]
        self.assertEqual(self.tile(stop, country="SE")[:2], (None, {}))
        self.assertEqual(self.tile(stop, tx=1)[:2], (None, {}))

    def test_off_switch(self):
        stop = [("a", self.x0, -5.0, "bus_stop", "天后 Tin Hau", {})]
        old, hkbusstops.ON = hkbusstops.ON, False
        try:
            self.assertEqual(self.tile(stop)[:2], (None, {}))
        finally:
            hkbusstops.ON = old

    def test_the_sign_tile_asks_for_them_in_hong_kong(self):
        import inspect
        src = inspect.getsource(signstream.build_tile)
        self.assertIn("hkbusstops.sign_tile", src)
        self.assertIn("hkbusstops.wanted(country)", src)


if __name__ == "__main__":
    unittest.main()
