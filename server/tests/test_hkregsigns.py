"""Hong Kong's regulatory signs from the Transport Department's road network (hkregsigns.py): the layers' records,
which sign each stands for, where its post stands and which way it faces, the faces' pictures and the mesh.

The sandbox that wrote these tests could not reach the CSDI service from its shell, so the service's answer here is a
saved sample: records copied from the live layers PROHIBITION (5) and VEHICLE_RESTRICTION (0) on 2026-11-24 (outSR
4326), trimmed to the fields parse reads."""
import gzip
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geodesy  # noqa: E402
import hkregsigns  # noqa: E402
import revisit  # noqa: E402
import roads  # noqa: E402


def _pro(oid, inc, exc, other, part, remarks, x, y):
    return {"attributes": {"OBJECTID": oid, "INC_VEH_TYPE": inc, "EXC_VEH_TYPE": exc, "OTHER_REST_TYPE_GV": other,
                           "PART_TIME_PROHIBITION": part, "REMARKS": remarks}, "geometry": {"x": x, "y": y}}


def _vr(oid, length, weight, height, remarks, x, y, width=None, axle=None):
    return {"attributes": {"OBJECTID": oid, "MAX_LENGTH": length, "MAX_WEIGHT": weight, "MAX_HEIGHT": height,
                           "MAX_WIDTH": width, "MAX_AXLE_WEIGHT": axle, "REMARKS": remarks},
            "geometry": {"x": x, "y": y}}


PROHIBITION = [
    _pro(1, "GV,PLB", "OTH", "Weight", "N", "PLB Proh/ GV Proh/ Over 12 Tonnes/ E FA", 114.21258703940903, 22.333675431853646),
    _pro(2, "GV", "NA", "NA", "N", "GV/ Proh", 114.15066935294762, 22.288306481545494),
    _pro(3, "PLB", "NA", "NA", "N", "PLB Proh", 114.21687636243034, 22.288931797212843),
    _pro(5, "OTH", "NA", "NA", "N", "LD Proh", 114.12782250965611, 22.371595608388915),
    _pro(6, "ALL", "OTH", "Length", "N", "Proh/ E FA/ Over 10m", 114.18915230287453, 22.279310496305492),
    _pro(7, "ALL", "GMB", "NA", "N", "Proh/E GMB", 114.20347971544247, 22.338386234297406),
    _pro(8, "ALL", "OTH,PC", "NA", "N", "Proh/ E PC/ E MC", 113.95325069951137, 22.314975480178603),
    _pro(9, "OTH", "NA", "NA", "Y", "LD Proh", 114.1767470406794, 22.318085474106677),
    _pro(10, "OTH,PLB", "NA", "NA", "N", "PLB Proh/ LD Proh", 114.15980540386929, 22.319575590832603),
    _pro(11, "ALL", "NA", "NA", "N", "Proh/ E WP", 114.14843453667004, 22.244297402207806),
    # (not the layer's: outside Hong Kong, no point)
    _pro(90, "GV", "NA", "NA", "N", "GV Proh", 113.5, 22.2),
    {"attributes": {"OBJECTID": 91, "INC_VEH_TYPE": "GV"}, "geometry": None},
]
RESTRICTION = [
    _vr(1, "8", None, None, None, 114.12834567920839, 22.356397530748321),
    _vr(2, None, "2", None, None, 114.07876963203077, 22.50973344328284),
    _vr(3, "12", None, None, "E WP", 113.99140871231288, 22.428848882834114),
    _vr(6, None, "5.5", None, None, 114.15303769318646, 22.282536932911395),
    _vr(9, None, None, "5.3", "Safe Ht 5.3 m", 113.96911732575761, 22.407271141214231),
    _vr(10, "8", None, None, "E FB", 114.11152676766869, 22.367690296985032),
    _vr(12, "7", None, None, "Proh/Over 7m", 114.07058911180758, 22.442578644619971),
    _vr(13, None, None, None, None, 114.10, 22.30, width="2", axle="10"),
    _vr(14, None, "x", None, None, 114.10, 22.30),
]
LAT, LON = 22.3, 114.17


def reset():
    hkregsigns._state.update(cache=None, fetcher=None, records=None, ident=None, down_at=0.0)
    hkregsigns.LOCAL = None


def road(fid, pts=((0.0, 0.0), (200.0, 0.0)), width=8.0, oneway=False, dir=0, kmh=50.0, kind="paved"):
    line = np.c_[np.array(pts, np.float64), np.zeros(len(pts))]
    return fid, roads.Spec(width, kind, True, "tertiary", oneway, dir=dir, kmh=kmh), line, width


def rec_at(frame, x, y, **fields):
    la, lo, _ = frame.enu_to_geodetic(np.array([x]), np.array([y]), np.zeros(1))
    return dict({"k": "pro", "lon": float(lo[0]), "lat": float(la[0])}, **fields)


class RecordsTest(unittest.TestCase):
    def test_parse_keeps_hong_kong_points_in_order(self):
        recs = hkregsigns.parse("pro", PROHIBITION)
        self.assertEqual([r["id"] for r in recs], [1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(recs[0]["INC_VEH_TYPE"], "GV,PLB")
        self.assertEqual((recs[0]["lon"], recs[0]["lat"]), (114.212587, 22.3336754))
        self.assertEqual(len(hkregsigns.parse("vr", RESTRICTION)), len(RESTRICTION))

    def test_prohibitions_read(self):
        got = {r["id"]: hkregsigns.faces(r) for r in hkregsigns.parse("pro", PROHIBITION)}
        self.assertEqual(got[1], [("lorry", "", ("Over 12 tonnes", "Except for access")),
                                  ("minibus", "", ("Over 12 tonnes", "Except for access"))])
        self.assertEqual(got[2], [("lorry", "", ())])
        self.assertEqual(got[3], [("minibus", "", ())])
        self.assertEqual(got[5], [("learner", "", ())])
        self.assertEqual(got[6], [("ring", "", ("Over 10 m", "Except for access"))])
        self.assertEqual(got[7], [("ring", "", ("Except green minibuses",))])
        self.assertEqual(got[8], [("ring", "", ("Except private cars", "Except motor cycles"))])
        self.assertEqual(got[10], [("minibus", "", ()), ("learner", "", ())])
        self.assertEqual(got[11], [("ring", "", ("Except with permit",))])

    def test_unknown_classes_and_a_bare_ring_make_no_sign(self):
        self.assertEqual(hkregsigns.faces({"k": "pro", "INC_VEH_TYPE": "OTH", "REMARKS": "Something Proh"}), [])
        self.assertEqual(hkregsigns.faces({"k": "pro", "INC_VEH_TYPE": "ALL", "REMARKS": "Proh"}), [])
        self.assertEqual(hkregsigns.faces({"k": "other"}), [])

    def test_restrictions_read(self):
        got = {r["id"]: hkregsigns.faces(r) for r in hkregsigns.parse("vr", RESTRICTION)}
        self.assertEqual(got[1], [("length", "8m", ())])
        self.assertEqual(got[2], [("weight", "2t", ())])
        self.assertEqual(got[3], [("length", "12m", ("Except with permit",))])
        self.assertEqual(got[6], [("weight", "5.5t", ())])
        self.assertEqual(got[9], [("height", "5.3m", ())])
        self.assertEqual(got[10], [("length", "8m", ("Except franchised buses",))])
        self.assertEqual(got[13], [("width", "2m", ()), ("axle", "10t", ())])
        self.assertEqual(got[14], [])

    def test_exceptions_once_in_order(self):
        self.assertEqual(hkregsigns.exceptions("Proh/ E WP/ E FA", "OTH"), ["Except for access", "Except with permit"])
        self.assertEqual(hkregsigns.exceptions(None, "GMB"), ["Except green minibuses"])
        self.assertEqual(hkregsigns.exceptions("E XYZ", "NA"), [])


class CacheTest(unittest.TestCase):
    def setUp(self):
        reset()

    def tearDown(self):
        reset()

    def test_fetched_once_kept_and_keyed(self):
        calls = []

        class Fetcher:
            def get(self, url, kind=None):
                calls.append(url)
                layer = "/5/query" in url
                body = {"features": PROHIBITION if layer else RESTRICTION}
                return SimpleNamespace(body=json.dumps(body).encode())
        with tempfile.TemporaryDirectory() as d:
            hkregsigns.configure(d, Fetcher())
            got = hkregsigns.records()
            self.assertEqual(len(got), 10 + len(RESTRICTION))
            self.assertEqual(len(calls), 2)
            path = os.path.join(d, "hkregsigns", hkregsigns.VERSION, "signs.json.gz")
            with open(path, "rb") as fh:
                self.assertEqual(json.loads(gzip.decompress(fh.read())), got)
            ident = hkregsigns.identity()
            self.assertTrue(ident.startswith(f"{hkregsigns.FACE_VERSION}:"))
            reset()
            hkregsigns.configure(d, Fetcher())
            self.assertEqual(hkregsigns.records(), got)          # (from the disk)
            self.assertEqual(len(calls), 2)
            self.assertEqual(hkregsigns.identity(), ident)

    def test_offline_no_signs_and_the_tile_not_kept(self):
        class Down:
            def get(self, url, kind=None):
                raise OSError("offline")
        with tempfile.TemporaryDirectory() as d:
            hkregsigns.configure(d, Down())
            faults = []
            old = revisit.fault
            revisit.fault = lambda here=False: faults.append(here)
            try:
                self.assertIsNone(hkregsigns.read(None))
            finally:
                revisit.fault = old
            self.assertEqual(faults, [True])
            self.assertEqual(hkregsigns.identity(), "none")

    def test_off_and_countries(self):
        self.assertTrue(hkregsigns.wanted("hk"))
        self.assertFalse(hkregsigns.wanted("GB"))
        old = hkregsigns.ON
        hkregsigns.ON = False
        try:
            self.assertFalse(hkregsigns.wanted("HK"))
            self.assertEqual(hkregsigns.identity(), "off")
        finally:
            hkregsigns.ON = old


class PlacementTest(unittest.TestCase):
    def test_two_way_road_end_faces_the_entering_traffic_on_its_left(self):
        rows = [road(1)]
        got = hkregsigns.spots(rows, [(10.0, 0.5)])
        self.assertEqual(len(got), 1)
        x, y, fx, fy, i = got[0]
        self.assertAlmostEqual(x, 10.0, places=6)
        self.assertAlmostEqual(y, 4.0 + 0.6, places=6)      # (left of eastbound traffic: north, past the kerb)
        self.assertEqual((fx, fy), (-1.0, -0.0))            # (facing the traffic coming east)
        x, y, fx, fy, _ = hkregsigns.spots(rows, [(190.0, -0.5)])[0]
        self.assertAlmostEqual(y, -4.6, places=6)           # (the other end: westbound traffic's left, south)
        self.assertEqual((fx, fy), (1.0, 0.0))

    def test_mid_block_both_ways_and_right_hand_mirrors(self):
        rows = [road(1)]
        got = hkregsigns.spots(rows, [(100.0, 0.0)])
        self.assertEqual(sorted((round(y, 3), fx) for _, y, fx, _, _ in got), [(-4.6, 1.0), (4.6, -1.0)])
        rh = hkregsigns.spots(rows, [(10.0, 0.0)], left_hand=False)
        self.assertAlmostEqual(rh[0][1], -4.6, places=6)

    def test_one_way_faces_its_own_traffic(self):
        got = hkregsigns.spots([road(1, oneway=True, dir=-1)], [(10.0, 0.0)])
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][2:4], (1.0, 0.0))           # (westbound traffic, facing east)
        self.assertAlmostEqual(got[0][1], -4.6, places=6)

    def test_clear_of_other_roads_and_far_points_dropped(self):
        rows = [road(1), road(2, pts=((0.0, 9.0), (200.0, 9.0)), width=6.0)]
        got = hkregsigns.spots(rows, [(10.0, -1.0)])
        self.assertTrue(got)
        for x, y, *_ in got:
            self.assertGreater(min(abs(y) - 4.0, abs(y - 9.0) - 3.0), hkregsigns.CLEAR_M - 1e-9)
        self.assertEqual(hkregsigns.spots([road(1)], [(50.0, 40.0)]), [])
        self.assertEqual(hkregsigns.spots([road(1, kind="unpaved")], [(10.0, 0.0)]), [])

    def test_tiles_agree_and_one_face_once(self):
        frame = geodesy.EnuFrame(LAT, LON)
        rows = [road(1, pts=((-200.0, 0.0), (200.0, 0.0)))]
        recs = [rec_at(frame, -190.0, 0.0, INC_VEH_TYPE="GV", REMARKS="GV Proh"),
                rec_at(frame, -188.0, 0.0, INC_VEH_TYPE="GV", REMARKS="GV Proh"),       # (the same sign twice)
                rec_at(frame, 0.5, 0.0, INC_VEH_TYPE="PLB", REMARKS="PLB Proh"),
                {**rec_at(frame, 190.0, 0.0), "k": "vr", "MAX_HEIGHT": "4.5"}]
        whole = hkregsigns.signs(frame, recs, (-300.0, -300.0, 300.0, 300.0), rows)
        left = hkregsigns.signs(frame, recs, (-300.0, -300.0, 0.0, 300.0), rows)
        right = hkregsigns.signs(frame, recs, (0.0, -300.0, 300.0, 300.0), rows)
        key = lambda s: (round(s[0], 6), round(s[1], 6), s[5])
        self.assertEqual(sorted(map(key, whole)), sorted(map(key, left + right)))
        lorries = [s for s in whole if s[5] == [("lorry", "", ())]]
        self.assertEqual(len(lorries), 1)
        self.assertEqual(len([s for s in whole if s[5] == [("minibus", "", ())]]), 2)   # (mid-block: both ways)
        self.assertTrue(all(s[4] == hkregsigns.DISC[0] for s in whole))

    def test_fast_road_bigger_disc_and_stack_limit(self):
        frame = geodesy.EnuFrame(LAT, LON)
        rows = [road(1, kmh=80.0)]
        recs = [{**rec_at(frame, 10.0, 0.0), "k": "vr", "MAX_HEIGHT": "4.5", "MAX_WIDTH": "2", "MAX_LENGTH": "7",
                 "MAX_WEIGHT": "24"}]
        got = hkregsigns.signs(frame, recs, (-50.0, -50.0, 50.0, 50.0), rows)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][4], hkregsigns.DISC[1])
        self.assertEqual(len(got[0][5]), hkregsigns.MAX_STACK)


class FaceTest(unittest.TestCase):
    def test_picture_sizes_and_plates(self):
        from PIL import Image
        png, w, h, rows = hkregsigns.picture("lorry", "", ("Over 12 tonnes", "Except for access"), 0.6)
        self.assertGreaterEqual(w, 0.6)
        self.assertAlmostEqual(h, 0.6 + 2 * (hkregsigns.GAP + hkregsigns.PLATE_H), places=9)
        im = Image.open(io.BytesIO(png))
        self.assertEqual(im.size, (round(w * hkregsigns.PX_PER_M), round(h * hkregsigns.PX_PER_M)))
        cx, cy = im.width // 2, round(0.3 * hkregsigns.PX_PER_M)
        self.assertEqual(im.getpixel((cx, cy - round(0.27 * hkregsigns.PX_PER_M))), hkregsigns.RED)    # (the ring)
        self.assertEqual(im.getpixel((cx, cy - round(0.2 * hkregsigns.PX_PER_M))), hkregsigns.WHITE)   # (the disc)
        self.assertEqual([r[0] for r in rows], ["Over 12 tonnes", "Except for access"])
        self.assertAlmostEqual(hkregsigns._disc_of(h, rows), 0.6, places=9)

    def test_plates_bilingual(self):
        self.assertEqual(hkregsigns.chinese("Over 12 tonnes"), "超逾12公噸")
        self.assertEqual(hkregsigns.chinese("Over 5.5 m"), "超逾5.5米")
        self.assertEqual(hkregsigns.chinese("Except buses"), "巴士例外")
        self.assertEqual(hkregsigns.chinese("Something else"), "")
        for text in hkregsigns.EXCEPT.values():                     # (every plate has its Chinese, in the font)
            zh = hkregsigns.chinese(text)
            self.assertTrue(zh, text)
            self.assertGreater(hkregsigns._zh_img(zh, 40).width, 40 * len(zh) * 0.6, text)
        wide = hkregsigns.plate_width("Except public light buses", 0.6)
        self.assertGreater(wide, 0.6)
        self.assertEqual(round(wide / 0.05, 6) % 1, 0)

    def test_every_symbol_draws(self):
        for kind, fig in (("lorry", ""), ("minibus", ""), ("bus", ""), ("motorcycle", ""), ("learner", ""),
                          ("ring", ""), ("height", "5.3m"), ("width", "2m"), ("length", "10m"), ("weight", "5.5t"),
                          ("axle", "10t")):
            png, w, h, rows = hkregsigns.picture(kind, fig, (), 0.6)
            self.assertEqual((w, h, rows), (0.6, 0.6, []))
            self.assertEqual(png, hkregsigns.picture(kind, fig, (), 0.6)[0])         # (the same bytes again)

    def test_mesh_faces_the_traffic(self):
        found = [(10.0, 4.6, -1.0, 0.0, 0.6, [("lorry", "", ("Except for access",)), ("height", "4.5m", ())])]
        with tempfile.TemporaryDirectory() as d:
            mesh, pics = hkregsigns.build(found, [0.0], (0.0, 0.0, 0.0), d, "vfs")
            self.assertEqual(len(pics), 2)
            for mat, path in pics.items():
                self.assertTrue(os.path.exists(os.path.join(d, mat + ".color.dds")))
                self.assertEqual(path, f"vfs/{mat}.color.dds")
        p, t, uv, mats, mi = mesh
        self.assertEqual(mats[0], hkregsigns.METAL)
        self.assertEqual(len(t), len(mi))
        self.assertTrue(((uv >= -1e-9) & (uv <= 1 + 1e-9)).all())
        front = t[mi > 0]
        n = np.cross(p[front[:, 1]] - p[front[:, 0]], p[front[:, 2]] - p[front[:, 0]])
        self.assertTrue((n[:, 0] < 0).all())                # (every picture triangle faces -x: the oncoming traffic)
        for m in set(mi[mi > 0].tolist()):                  # (each picture read left to right, upright)
            fv = np.unique(t[mi == m].ravel())
            self.assertLess(np.corrcoef(uv[fv, 0], p[fv, 1])[0, 1], -0.99)   # (looking east, the right is south)
            self.assertGreater(np.corrcoef(uv[fv, 1], p[fv, 2])[0, 1], 0.99)
        self.assertGreater(p[:, 2].max(), hkregsigns.MOUNT + 1.2)
        self.assertLess(p[front][:, :, 2].min(), hkregsigns.MOUNT + hkregsigns.PLATE_H + 1e-6)
        self.assertGreaterEqual(p[front][:, :, 2].min(), hkregsigns.MOUNT - 1e-9)

    def test_no_signs_no_mesh(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(hkregsigns.build([], [], (0.0, 0.0, 0.0), d, "vfs"), (None, {}))


if __name__ == "__main__":
    unittest.main()
