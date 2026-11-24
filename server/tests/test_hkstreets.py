"""Hong Kong's street name plates (hkstreets.py): names, the HyD plate picture, where plates stand, the tile's mesh."""
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import geodesy
import hkstreets
import roads
import signroads
import signstream


def spec(width, cls="secondary", link=False):
    return signroads.Spec(width, cls, False, 0, link, 50.0, False, ())


class Feats:
    """A roads layer with just ids and names (all _names reads)."""
    def __init__(self, names):
        self.ids = np.array(list(names), dtype=str)
        self.attrs = {"name": np.array(list(names.values()), object)}

    def __len__(self):
        return len(self.ids)


class Context:
    """A sign context: the road rows (fid, spec, line, width) and one cell of their names."""
    def __init__(self, rows, names):
        self.rows, self.feats = rows, Feats(names)

    def pavements(self, frame, points, left_hand=False):
        return list(self.rows)

    def cell(self, key):
        return self.feats


def cross(cx, cy, main="彌敦道 Nathan Road", side="柯士甸道 Austin Road", main_w=14.0, side_w=8.0, side_cls="tertiary"):
    """A + junction at (cx, cy): main along x, side along y, each split at the junction (Overture's connectors)."""
    rows = [("m1", spec(main_w), np.array([[cx - 150.0, cy, 0], [cx, cy, 0]]), main_w),
            ("m2", spec(main_w), np.array([[cx, cy, 0], [cx + 150.0, cy, 0]]), main_w),
            ("s1", spec(side_w, side_cls), np.array([[cx, cy - 120.0, 0], [cx, cy, 0]]), side_w),
            ("s2", spec(side_w, side_cls), np.array([[cx, cy, 0], [cx, cy + 120.0, 0]]), side_w)]
    names = {"m1": main, "m2": main, "s1": side, "s2": side}
    return rows, names


def png(data):
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


class NameTest(unittest.TestCase):
    def test_osm_hong_kong_names_split_into_english_and_chinese(self):
        self.assertEqual(hkstreets.split_name("彌敦道 Nathan Road"), ("Nathan Road", "彌敦道"))
        self.assertEqual(hkstreets.split_name("皇后大道中 Queen's Road Central"), ("Queen's Road Central", "皇后大道中"))
        self.assertEqual(hkstreets.split_name("C段第八街 Section C, 8th Street"), ("Section C, 8th Street", "C段第八街"))
        self.assertEqual(hkstreets.split_name("青山公路－荃灣段 Castle Peak Road – Tsuen Wan"),
                         ("Castle Peak Road - Tsuen Wan", "青山公路－荃灣段"))    # (Transport has no en dash)
        self.assertEqual(hkstreets.split_name("  通州街   Tung Chau​Street "), ("Tung Chau Street", "通州街"))

    def test_one_language_and_no_names(self):
        self.assertEqual(hkstreets.split_name("北橋"), ("", "北橋"))
        self.assertEqual(hkstreets.split_name("Smithfield Road"), ("Smithfield Road", ""))
        self.assertEqual(hkstreets.split_name("D3"), ("", ""))         # a reference, not a name
        for empty in ("", None, "None", "   "):
            self.assertEqual(hkstreets.split_name(empty), ("", ""))

    def test_the_cut_font_has_hong_kong_names_not_simplified_ones(self):
        for zh in ("深水埗", "鰂魚涌街", "彌敦道", "德輔道中", "旺角", "銅鑼灣道"):
            self.assertTrue(hkstreets.covered(zh), zh)
        self.assertFalse(hkstreets.covered("铜锣湾"))                  # (Shenzhen's: simplified)


class PictureTest(unittest.TestCase):
    def test_the_plate_is_hyd_sized_white_with_a_black_border(self):
        data, w, h = hkstreets.picture("Nathan Road", "彌敦道")
        self.assertEqual(h, hkstreets.PLATE_H)
        self.assertTrue(hkstreets.MIN_W <= w <= hkstreets.MAX_W)
        self.assertAlmostEqual(round(w / hkstreets.STEP_W, 6) % 1, 0.0)  # 50 mm steps
        im = png(data)
        self.assertEqual(im.size, (round(w * hkstreets.PX_PER_M), round(h * hkstreets.PX_PER_M)))
        px = np.asarray(im, np.int32)
        b = round(hkstreets.BORDER * hkstreets.PX_PER_M)
        self.assertTrue((px[im.height // 2, : b - 1] < 40).all())        # the border: black
        self.assertTrue((px[b + 4, im.width // 2] > 230).all())          # the field above the lettering: white
        ink = (px.sum(2) < 200)[b + 12:-b - 12, b + 12:-b - 12]   # (inside the corners)
        rows = np.nonzero(ink.any(1))[0]
        self.assertGreater(len(rows), 0)
        # two lines: English capitals ~56 px, Chinese ~78 px, a white gap between them
        gaps = np.nonzero(np.diff(rows) > 15)[0]
        self.assertEqual(len(gaps), 1)
        top_h, bottom_h = rows[gaps[0]] - rows[0], rows[-1] - rows[gaps[0] + 1]
        self.assertGreater(bottom_h, top_h)                               # Chinese under English, the larger

    def test_long_names_grow_the_plate_then_condense(self):
        short = hkstreets.plate_width("Nathan Road", "彌敦道")
        longer = hkstreets.plate_width("Castle Peak Road - Tsuen Wan", "青山公路－荃灣段")
        self.assertEqual(short, hkstreets.MIN_W)
        self.assertGreater(longer, short)
        very = "Shing Mun Forest Track - Needle Hill Section Upper Ridge Path"
        self.assertEqual(hkstreets.plate_width(very, "城門林道－針山段"), hkstreets.MAX_W)
        im = png(hkstreets.picture(very, "城門林道－針山段")[0])
        px = np.asarray(im, np.int32)
        side = round(hkstreets.SIDE * hkstreets.PX_PER_M)
        self.assertTrue((px[:, side - 8:side - 2][px.shape[0] // 4: px.shape[0] // 2] > 230).all())   # not cut

    def test_one_language_plates_and_the_same_picture_twice(self):
        a = hkstreets.picture("Smithfield Road", "")[0]
        self.assertEqual(a, hkstreets.picture("Smithfield Road", "")[0])
        self.assertEqual(png(hkstreets.picture("", "北橋")[0]).height, round(hkstreets.PLATE_H * hkstreets.PX_PER_M))

    def test_a_name_transport_lacks_uses_the_fallback_face(self):
        self.assertEqual(hkstreets._en_face("Nathan Road"), hkstreets.EN_FONT)
        self.assertEqual(hkstreets._en_face("Café Street"), hkstreets.EN_FALLBACK)
        self.assertGreater(len(hkstreets.picture("Café Street", "")[0]), 0)


class PlacementTest(unittest.TestCase):
    BOX = (-500.0, -500.0, 500.0, 500.0)

    def lines(self, rows):
        return [(r[3] / 2, np.asarray(r[2])[:, :2]) for r in rows]

    def test_each_named_arm_gets_a_plate_clear_of_the_carriageways(self):
        rows, names = cross(0.0, 0.0)
        names = {k: hkstreets.split_name(v) for k, v in names.items()}
        found = hkstreets.plates(rows, names, self.BOX)
        self.assertEqual(len(found), 4)
        self.assertEqual(sorted(f[4] for f in found), ["Austin Road"] * 2 + ["Nathan Road"] * 2)
        for x, y, ux, uy, en, zh in found:
            q = np.array([x, y])
            w = hkstreets.plate_width(en, zh)
            along = np.array([ux, uy])
            for e in (q, q + along * (w / 2 + 0.1), q - along * (w / 2 + 0.1)):
                for hw, ln in self.lines(rows):
                    self.assertGreaterEqual(hkstreets._seg_dist(e, ln), hw + hkstreets.CLEAR - 1e-9)
            # parallel to the street it names
            if en == "Nathan Road":
                self.assertAlmostEqual(abs(ux), 1.0)
            else:
                self.assertAlmostEqual(abs(uy), 1.0)
        # a street's two plates on diagonally opposite corners (left of each arm: left-hand traffic)
        nathan = sorted((f for f in found if f[4] == "Nathan Road"), key=lambda f: f[0])
        self.assertLess(nathan[0][0], 0.0)
        self.assertGreater(nathan[1][0], 0.0)
        self.assertLess(nathan[0][1] * nathan[1][1], 0.0)

    def test_no_plate_where_a_street_only_continues_or_on_links_and_motorways(self):
        rows, names = cross(0.0, 0.0, side="彌敦道 Nathan Road")
        names = {k: hkstreets.split_name(v) for k, v in names.items()}
        self.assertEqual(hkstreets.plates(rows, names, self.BOX), [])
        rows, names = cross(0.0, 0.0, side_cls="motorway")
        names = {k: hkstreets.split_name(v) for k, v in names.items()}
        self.assertEqual({f[4] for f in hkstreets.plates(rows, names, self.BOX)}, {"Nathan Road"})

    def test_a_plate_belongs_to_the_tile_its_post_stands_in(self):
        rows, names = cross(0.0, 0.0)
        names = {k: hkstreets.split_name(v) for k, v in names.items()}
        everything = hkstreets.plates(rows, names, self.BOX)
        halves = hkstreets.plates(rows, names, (-500.0, -500.0, 0.0, 500.0)) + \
            hkstreets.plates(rows, names, (0.0, -500.0, 500.0, 500.0))
        self.assertEqual(sorted(everything), sorted(halves))

    def test_no_plate_on_a_bridge_deck(self):
        rows, names = cross(0.0, 0.0)
        names = {k: hkstreets.split_name(v) for k, v in names.items()}
        deck = [{"deck": True, "n": np.array([[-200.0, -30.0], [200.0, -30.0], [200.0, 30.0]]), "w": 120.0}]
        self.assertEqual(hkstreets.plates(rows, names, self.BOX, deck), [])


class TileTest(unittest.TestCase):
    def setUp(self):
        self.frame = geodesy.EnuFrame(22.30, 114.17, 0.0)
        self.box = roads.tile_bbox(0, 0)
        self.cx, self.cy = (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    def tile(self, country="HK", **kw):
        rows, names = cross(self.cx, self.cy, **kw)
        with tempfile.TemporaryDirectory() as out:
            mesh, faces = hkstreets.sign_tile(None, self.frame, 0, 0, lambda la, lo: np.full(np.shape(la), 20.0), out,
                                              "/v", Context(rows, names), (), country, True)
            return mesh, faces, sorted(os.listdir(out))

    def test_plates_their_pictures_and_posts(self):
        mesh, faces, files = self.tile()
        p, t, uv, materials, mi = mesh
        self.assertEqual(materials[0], hkstreets.METAL)
        self.assertEqual(len(materials), 3)                         # metal, Nathan Road, Austin Road
        self.assertEqual(sorted(faces), sorted(materials[1:]))
        self.assertEqual(files, sorted(m + ".color.dds" for m in materials[1:]))
        mi = np.asarray(mi)
        for m in (1, 2):
            self.assertEqual(int((mi == m).sum()), 8)               # two plates, two faces, two triangles
        z = p[:, 2]
        self.assertAlmostEqual(float(z.max()), 20.0 + hkstreets.MOUNT + hkstreets.PLATE_H + 0.03, delta=0.01)
        self.assertAlmostEqual(float(z.min()), 20.0 - 0.3, delta=0.01)       # (the frame's curvature: mm)
        self.assertTrue(all(faces[m] == f"/v/{m}.color.dds" for m in faces))

    def test_both_faces_read_left_to_right_from_their_side(self):
        mesh, *_ = self.tile()
        p, t, uv, materials, mi = mesh
        t, mi, uv = np.asarray(t), np.asarray(mi), np.asarray(uv)
        tri = t[mi == 1]
        for a, b, c in tri[::2]:                                    # the first triangle of each face quad
            normal = np.cross(p[b] - p[a], p[c] - p[a])
            right = p[b] - p[a]                                     # u = 0 -> 1 along the first edge
            self.assertAlmostEqual(float(uv[b][0] - uv[a][0]), 1.0)
            # the viewer stands on the normal's side, looking against it: their right is up x normal
            seen_right = np.cross(np.array([0.0, 0.0, 1.0]), normal)
            self.assertGreater(float((right * seen_right).sum()), 0.0)
        normals = [np.cross(p[b] - p[a], p[c] - p[a])[:2] for a, b, c in tri[::2]]
        self.assertLess(float((normals[0] * normals[1]).sum()), 0.0)   # the two faces of a plate: back to back

    def test_none_outside_hong_kong_or_switched_off(self):
        self.assertEqual(self.tile(country="CN")[:2], (None, {}))
        with mock.patch.dict(os.environ, {"BNE_HK_STREETS": "off"}):
            self.assertEqual(self.tile()[:2], (None, {}))

    def test_a_name_the_font_lacks_is_english_only(self):
        mesh, faces, _ = self.tile(side="铜锣湾路 Tongluowan Road")
        self.assertEqual(len(faces), 2)
        self.assertTrue(hkstreets.covered("柯士甸道"))
        self.assertFalse(hkstreets.covered("铜锣湾路"))

    def test_the_sign_tile_asks_for_them_in_hong_kong(self):
        import inspect
        src = inspect.getsource(signstream.build_tile)
        self.assertIn("hkstreets.sign_tile", src)
        self.assertIn("hkstreets.wanted(country)", src)


if __name__ == "__main__":
    unittest.main()
