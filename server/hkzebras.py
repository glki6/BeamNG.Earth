"""Hong Kong's zebra crossings as the Transport Department maps them (roads.build_tile, country HK), and their beacons.

junctions.find draws a striped crosswalk on every arm with two corners, as West Coast USA's streets look. A Hong Kong
street is not striped that way: "'Zebra' crossings are marked with black and white stripes, road studs and zigzag
lines on either side with yellow flashing beacon", signal-controlled crossings are "delineated by road studs", and an
uncontrolled crossing is "normally indicated by dropped kerbs" (Transport Department, At-grade Pedestrian Crossings;
Road Users' Code ch. 2 and 5). As Sweden's crossings (secrossings.py) paint only where its law paints, Hong Kong's:

    a zebra         where the Transport Department has one: the black and white stripes across the road, a broken
                    give-way line before them on each side ("stop behind the 'Give way' line", Road Users' Code ch. 5),
                    zig-zag lines along each kerb on each side, and a beacon (a yellow globe on a black and white post)
                    at each end of the crossing (tile_rows: the Forest rows of group hk_beacon)
    a junction's    crossing under mapped signals: no stripes, two rows of road studs along its edges
    any other       junction crossing: no paint, its dropped kerbs kept

The zebras are the Transport Department's Road Network (2nd generation) layer TRAFFIC_FEATURES ("Zebra crossing,
yellow box, toll plaza and cul-de-sac": points), on the CSDI Portal's ArcGIS service roadz.py reads the centrelines of
(layer 10). Its FEATURE_TYPE codes are not published in the service (no domain, no renderer labels, an empty
description: read 2026-11-24) and the data dictionary is a zip on static.data.gov.hk. They were worked out from the
layer itself on 2026-11-24: of 2,982 points, code 1 is 264, 2 is 1,895, 3 is 19, 4 is 804, which is the layer's own
order. Code 3's remarks name the Western Harbour Crossing's backup toll point (a toll plaza); code 2's points carry
four or five road ids (RD_ID_1..), a junction's arms (a yellow box); code 4's lie on closed loops of the centreline
(Fuk Lee Street's, 67 m round: a cul-de-sac's turning circle); code 1's lie on a street 8 to 19 m from its end (Cotton
Path, Taikoo Shing Road, Taikoo Wan Road): ZEBRA_TYPE. Yellow boxes are not drawn: no other region draws one yet.

The layer's zebras are a few hundred points: fetched whole once (hklidar.arcgis) and kept on disk
(<cache>/hkzebras/<VERSION>/zebras.json.gz), asked again after MAX_AGE_S (the layer changes monthly) with the old copy
kept while the service does not answer. Without them (offline, nothing kept) a tile is built as before and not kept
(revisit.fault). identity(): the kept file's hash, for the road tiles' key.

Measures not in the pages read (the Transport Planning and Design Manual's road marking chapter was not reachable):
the stripes are the stock crosswalk's (junctions.CROSSWALK, 3.0 m along the road); the give-way line, zig-zags and
beacons follow the United Kingdom's, whose road markings Hong Kong's follow (TSRGD 2016: a broken line of 0.6 m marks,
0.3 m gaps, 0.2 m wide, 1.5 m before the stripes; eight zig-zag marks of 2 m, 0.1 m wide; a globe about 0.33 m across
near 2.75 m up on a banded post). The zig-zags are white, as the United Kingdom's (the pages read do not say).

Off: BNE_HK_ZEBRA=off (Hong Kong's crossings as before). Its source and terms are in CREDITS.md (the roadz "hk" row:
the same service and terms).
"""
import gzip
import hashlib
import json
import logging
import math
import os
import threading
import time

import numpy as np

import dmath
import dnp

import hklidar

log = logging.getLogger("earthd.hkzebras")

ON = os.environ.get("BNE_HK_ZEBRA", "on").lower() not in ("off", "0", "false")
VERSION = "hkzebras-v1"
URL = "https://portal.csdi.gov.hk/server/rest/services/common/td_rcd_1638949160594_2844/FeatureServer/1/query"
ZEBRA_TYPE = 1                      # TRAFFIC_FEATURES.FEATURE_TYPE of a zebra crossing (worked out above)
MAX_AGE_S = 30 * 86400              # s: the layer is asked again after this (it changes monthly)
RETRY_S = 3600                      # s: unreachable, asked again after this
REACH_M = 80.0                      # m past a tile's box its zebras are read (their zig-zags and beacons reach in)

NODE_NEAR_M = 6.0                   # m from a carriageway's centreline a zebra is on that road (two surveys' lines)
ONE_ZEBRA_M = 5.0                   # m along a road within which two points are one crossing
SAME_CROSSWALK_M = 6.0              # m between a zebra and a junction's drawn crosswalk: the same crossing
ZEBRA_W = 3.0                       # m along the road: the stripes' band (junctions.CROSSWALK's)
GIVE_WAY_M = 1.5                    # m from the stripes to the give-way line's middle
GIVE_WAY_W, GIVE_WAY_DASH = 0.20, (0.6, 0.3)
ZIG_N, ZIG_L, ZIG_W = 8, 2.0, 0.10  # zig-zag marks a side, each this long along the road, this wide
ZIG_AMP = 0.30                      # m from one side of the zig-zag to the other
ZIG_KERB = 0.45                     # m in from the kerb its middle runs (inside an edge line)
ZIG_FROM = 0.40                     # m past the give-way line it starts
ZIG_END_M = 1.0                     # m short of the road's end it stops (the junction there)
ZIG_MIN = 2                         # marks: fewer fit, none
STUD = 0.12                         # m square: a road stud
STUD_STEP = 0.60                    # m between them
SIGNALS_NEAR_M = 30.0               # m from a mapped signal a junction's crossing is under it
BEACON_OUT_M = 0.6                  # m past the kerb a beacon stands
BEACON_GROUP = "hk_beacon"          # guardrails.GROUPS, trees.lua's CATALOG (tools/build_hk_beacon.py)
BEACON_H = 2.95                     # m: the model's height (its catalog height: scale 1)

_lock = threading.Lock()
_state = {"cache": None, "fetcher": None, "points": None, "ident": None, "down_at": 0.0}
LOCAL = None                        # (tests: [[lon, lat], ...] used instead of the cache and the service)


def wanted(country):
    """Whether a place's road tiles draw Hong Kong's crossings: Hong Kong's, unless BNE_HK_ZEBRA=off."""
    return ON and str(country or "").upper() == "HK"


def configure(cache_dir, fetcher):
    """Where the layer is kept and what fetches it (read() does it from its MapData); until then nothing is fetched."""
    with _lock:
        if _state["fetcher"] is None:
            _state["cache"], _state["fetcher"] = cache_dir, fetcher


def parse(features):
    """[[lon, lat], ...] of an ArcGIS query's point features (outSR 4326) that are zebras (FEATURE_TYPE ZEBRA_TYPE),
    in Hong Kong, rounded to 1e-7 degrees, ordered (the same file from the same layer)."""
    out = set()
    for f in features or ():
        a = f.get("attributes") or {}
        g = f.get("geometry") or {}
        try:
            if int(a.get("FEATURE_TYPE")) != ZEBRA_TYPE:
                continue
            x, y = float(g["x"]), float(g["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y) and hklidar.BOUNDS[0] <= y <= hklidar.BOUNDS[2] \
                and hklidar.BOUNDS[1] <= x <= hklidar.BOUNDS[3]:
            out.add((round(x, 7), round(y, 7)))
    return [list(p) for p in sorted(out)]


def _path():
    cache = _state["cache"]
    return os.path.join(cache, "hkzebras", VERSION, "zebras.json.gz") if cache else None


def _read(path):
    """(points, identity) of the kept file, None when it cannot be read."""
    try:
        with open(path, "rb") as fh:
            body = fh.read()
        return json.loads(gzip.decompress(body).decode("utf-8")), hashlib.sha256(body).hexdigest()[:12]
    except (OSError, ValueError, EOFError) as e:
        log.debug("hk zebras unreadable (%s): %s", path, e)
        return None


def _fetch():
    """The layer's zebras from the service, None when it does not answer (backs off RETRY_S)."""
    fetcher = _state["fetcher"]
    if fetcher is None or time.time() - _state["down_at"] < RETRY_S:
        return None

    def get_json(url):
        return json.loads(fetcher.get(url, kind="demand").body.decode("utf-8"))
    try:
        return parse(hklidar.arcgis(get_json, URL, {
            "where": f"FEATURE_TYPE={ZEBRA_TYPE}", "outFields": "FEATURE_TYPE", "returnGeometry": "true",
            "outSR": 4326, "orderByFields": "OBJECTID"}))
    except Exception as e:      # (offline, or the service changed: the crossings as before, and the build is not kept)
        log.info("hk zebras not fetched: %s: %s", type(e).__name__, e)
        _state["down_at"] = time.time()
        return None


def _write(path, points):
    body = gzip.compress(json.dumps(points, separators=(",", ":")).encode("utf-8"), mtime=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as fh:
        fh.write(body)
    os.replace(path + ".part", path)
    return hashlib.sha256(body).hexdigest()[:12]


def points(fetch=True):
    """The process's zebras [[lon, lat], ...] (loaded once and kept), None while they cannot be had. fetch=False: the
    disk only."""
    if _state["points"] is not None:
        return _state["points"]
    with _lock:
        if _state["points"] is not None:
            return _state["points"]
        if LOCAL is not None:
            _state["points"], _state["ident"] = [list(p) for p in LOCAL], "local"
            return _state["points"]
        path = _path()
        kept = _read(path) if path and os.path.exists(path) else None
        fresh = kept is not None and time.time() - os.path.getmtime(path) < MAX_AGE_S
        if not fresh and fetch:
            got = _fetch()
            if got:
                kept = (got, _write(path, got) if path else hashlib.sha256(json.dumps(got).encode()).hexdigest()[:12])
        if kept is None:
            return None
        _state["points"], _state["ident"] = kept
        log.info("hk zebras: %d (%s)", len(_state["points"]), _state["ident"])
        return _state["points"]


def identity():
    """The zebras' identity for cache keys: "off", "none" (none on disk yet) or the kept file's hash."""
    if not ON:
        return "off"
    return _state["ident"] if points(fetch=False) is not None else "none"


def nodes(frame, pts, box):
    """build_tile's nodes= of Hong Kong's zebras pts [[lon, lat], ...] within REACH_M of ENU box (x0, y0, x1, y1):
    {"hk_zebras": [(x, y)]}, with no "priority" (None: the junctions' signs guessed as before) nor refuges."""
    out = []
    if pts:
        ll = np.asarray(pts, np.float64).reshape(-1, 2)
        e, n, _ = frame.geodetic_to_enu(ll[:, 1], ll[:, 0], 0.0)
        e, n = np.asarray(e, np.float64).ravel(), np.asarray(n, np.float64).ravel()
        x0, y0, x1, y1 = box
        keep = (e >= x0 - REACH_M) & (e <= x1 + REACH_M) & (n >= y0 - REACH_M) & (n <= y1 + REACH_M)
        out = [(float(a), float(b)) for a, b in zip(e[keep], n[keep])]
    return {"hk_zebras": out, "priority": None}


def read(md, frame, box):
    """nodes of the zebras near ENU box (md: mapdata.MapData, its cache and fetcher), None while the layer cannot be had
    (a tile then draws its crossings as before, and is not kept)."""
    if md is not None:
        configure(getattr(md, "cache_dir", None), getattr(md, "fetcher", None))
    got = points()
    if got is None:
        import revisit
        revisit.fault(here=True)
        return None
    return nodes(frame, got, box)


def is_hk(found):
    return isinstance(found, dict) and found.get("hk_zebras") is not None


def _stations(pts):
    return np.r_[0.0, np.cumsum(dnp.hypot(*np.diff(pts[:, :2], axis=0).T))]


def _at(pts, s, st):
    """(point (3,), unit tangent (2,)) of polyline pts at station st (s: its stations), clamped to its ends."""
    st = min(max(float(st), 0.0), float(s[-1]))
    k = int(np.clip(np.searchsorted(s, st, side="right") - 1, 0, len(s) - 2))
    seg = pts[k + 1] - pts[k]
    L = max(float(s[k + 1] - s[k]), 1e-9)
    f = (st - float(s[k])) / L
    u = seg[:2] / L
    return pts[k] + seg * f, u


def place(roads, found):
    """[{"fid", "spec", "c" (3,), "u", "n" (2,), "half", "s", "pts", "stations"}] of the zebras of found (nodes) on the
    paved roads of roads [(fid, spec, centreline (k, 3), ...)]: each on its nearest carriageway within NODE_NEAR_M, one
    per ONE_ZEBRA_M of a road, in the zebras' order. The same from every tile that has the road whole."""
    lines = [(fid, spec, np.asarray(pts, np.float64)) for fid, spec, pts, *_ in roads
             if spec.kind in ("paved", "cobble") and spec.cls != "motorway" and len(pts) >= 2]
    out = []
    if not lines or not found or not found.get("hk_zebras"):
        return out
    lines = [(fid, spec, np.c_[p[:, :2], p[:, 2] if p.shape[1] > 2 else np.zeros(len(p))]) for fid, spec, p in lines]
    a = np.vstack([p[:-1] for _, _, p in lines])
    b = np.vstack([p[1:] for _, _, p in lines])
    own = np.repeat(np.arange(len(lines)), [len(p) - 1 for _, _, p in lines])
    first = np.r_[0, np.cumsum([len(p) - 1 for _, _, p in lines])[:-1]]
    d = b[:, :2] - a[:, :2]
    l2 = np.maximum((d * d).sum(1), 1e-9)
    stations = {}
    for x, y in found["hk_zebras"]:
        t = np.clip(((np.array([x, y]) - a[:, :2]) * d).sum(1) / l2, 0.0, 1.0)
        q = a[:, :2] + d * t[:, None]
        off = dnp.hypot(q[:, 0] - x, q[:, 1] - y)
        i = int(np.argmin(off))
        if off[i] > NODE_NEAR_M:
            continue                    # (on a footway, or a road not in this list)
        j = int(own[i])
        fid, spec, pts = lines[j]
        if j not in stations:
            stations[j] = _stations(pts)
        s = stations[j]
        k = i - int(first[j])
        st = float(s[k] + t[i] * (s[k + 1] - s[k]))
        if any(z["fid"] == fid and abs(z["s"] - st) < ONE_ZEBRA_M for z in out):
            continue
        c, u = _at(pts, s, st)
        out.append({"fid": fid, "spec": spec, "c": c, "u": u, "n": np.array([-u[1], u[0]]),
                     "half": float(spec.width) / 2, "s": st, "pts": pts, "stations": s})
    return out


def _quad(p0, p1, w):
    """The convex quad (4, 3) of a stroke from p0 to p1 (3,), w wide."""
    d = p1[:2] - p0[:2]
    L = float(dnp.hypot(*d))
    if L < 1e-6:
        return None
    h = np.array([-d[1], d[0], 0.0]) / L * w / 2
    return np.stack([p0 + h, p1 + h, p1 - h, p0 - h])


def _broken(p0, p1, w, pattern):
    """The marks (quads) of a broken line from p0 to p1 (3,), w wide, its marks and gaps pattern, centred on it."""
    L = float(dnp.hypot(*(p1 - p0)[:2]))
    mark, gap = pattern
    if L < mark:
        return []
    n = int((L + gap) // (mark + gap))
    lead = (L - (n * mark + (n - 1) * gap)) / 2
    out = []
    for k in range(n):
        f0 = (lead + k * (mark + gap)) / L
        f1 = (lead + k * (mark + gap) + mark) / L
        q = _quad(p0 + (p1 - p0) * f0, p0 + (p1 - p0) * f1, w)
        if q is not None:
            out.append(q)
    return out


def _zigzag(z, side, offset):
    """The zig-zag marks (quads) on zebra z's road on side (1: its stations beyond the crossing, -1: before it), its
    middle offset m left of the centreline: ZIG_N marks of ZIG_L, short of the road's end by ZIG_END_M."""
    pts, s = z["pts"], z["stations"]
    start = ZEBRA_W / 2 + GIVE_WAY_M + GIVE_WAY_W / 2 + ZIG_FROM
    room = (float(s[-1]) - z["s"] if side > 0 else z["s"]) - ZIG_END_M - start
    n = min(ZIG_N, int(room // ZIG_L))
    if n < ZIG_MIN:
        return []
    verts = []
    for k in range(n + 1):
        p, u = _at(pts, s, z["s"] + side * (start + k * ZIG_L))
        nrm = np.array([-u[1], u[0], 0.0])
        verts.append(p + nrm * (offset + (ZIG_AMP / 2) * (1 if k % 2 else -1)))
    return [q for q in (_quad(verts[k], verts[k + 1], ZIG_W) for k in range(n)) if q is not None]


def marks(z):
    """The give-way lines and zig-zags (quads) of zebra z (place): a broken line across the road GIVE_WAY_M before the
    stripes on each side, and zig-zags along each kerb beyond it."""
    out = []
    nrm = np.r_[z["n"], 0.0]
    for side in (1, -1):
        p, _ = _at(z["pts"], z["stations"], z["s"] + side * (ZEBRA_W / 2 + GIVE_WAY_M))
        out += _broken(p + nrm * z["half"], p - nrm * z["half"], GIVE_WAY_W, GIVE_WAY_DASH)
        for edge in (1, -1):
            out += _zigzag(z, side, edge * (z["half"] - ZIG_KERB))
    return out


def walk(z):
    """Zebra z's stripes: a crosswalk {"a", "b", "w"} from kerb to kerb."""
    nrm = np.r_[z["n"], 0.0]
    return {"a": z["c"] + nrm * z["half"], "b": z["c"] - nrm * z["half"], "w": ZEBRA_W}


def studs(cw):
    """Two rows of road studs (quads) along the edges of crossing cw {"a", "b", "w"} (its band's sides)."""
    a, b = np.asarray(cw["a"], np.float64)[:3], np.asarray(cw["b"], np.float64)[:3]
    d = b[:2] - a[:2]
    L = float(dnp.hypot(*d))
    if L < STUD_STEP:
        return []
    u = d / L
    along = np.array([-u[1], u[0], 0.0]) * (float(cw["w"]) / 2)
    out = []
    for sd in (1, -1):
        out += _broken(a + along * sd, b + along * sd, STUD, (STUD, STUD_STEP - STUD))
    return out


def _mid(cw):
    return 0.5 * (np.asarray(cw["a"], np.float64)[:2] + np.asarray(cw["b"], np.float64)[:2])


def paint(roads, found, crosswalks, jramps, signals):
    """(crosswalks, jramps, zebra crosswalks, marks) of a Hong Kong road tile: the junctions' drawn crosswalks
    (junctions.find's, stop lines kept as they are) without stripes, studs along those under mapped signals (signals:
    furniture.mapped_signals' (n, 2), None: no studs), one that a zebra stands on (SAME_CROSSWALK_M) left out for it;
    the zebras of found (nodes) striped, with their kerb ramps, give-way lines and zig-zags."""
    zs = place(roads, found)
    zwalks = [walk(z) for z in zs]
    zmid = np.array([z["c"][:2] for z in zs]).reshape(-1, 2)
    sig = None if signals is None else np.asarray(signals, np.float64).reshape(-1, 2)
    out, extra = [], []
    for cw in crosswalks:
        if cw.get("stop"):
            out.append(cw)
            continue
        m = _mid(cw)
        if len(zmid) and float(dnp.hypot(*(zmid - m).T).min()) < SAME_CROSSWALK_M:
            continue
        out.append(dict(cw, paint=False))
        if sig is not None and len(sig) and float(dnp.hypot(*(sig - m).T).min()) < SIGNALS_NEAR_M:
            extra += studs(cw)
    ramps = {k: list(v) for k, v in (jramps or {}).items()}
    for z in zs:
        for side in (1, -1):
            ramps.setdefault((z["fid"], side), []).append((z["c"][:2].copy(), z["u"], -ZEBRA_W / 2, ZEBRA_W / 2))
        extra += marks(z)
    return out + zwalks, ramps, zwalks, extra


def _row(p, facing, frame, elev):
    """The Forest row of a beacon at p (2,), its model's -Y towards facing (2,) (sesigns._row's form)."""
    fx, fy = float(facing[1]), float(-facing[0])
    x, y = float(p[0]), float(p[1])
    la, lo, _ = frame.enu_to_geodetic(np.array([x]), np.array([y]), np.zeros(1))
    z = float(np.asarray(frame.geodetic_to_enu(la, lo, np.asarray(elev(la, lo), np.float64))[2]).ravel()[0])
    import guardrails
    return [guardrails.GROUPS.index(BEACON_GROUP), round(x, 3), round(y, 3), round(z, 3),
            round(math.degrees(dmath.atan2(fy, fx)), 2), 0.0, BEACON_H, 0.0, fx, fy, 0.0, -fy, fx, 0.0, 0.0, 0.0, 1.0]


def beacons(near_roads, found):
    """[(p (2,), facing (2,))]: a beacon at each end of each zebra of found on near_roads (build_tile's roads_out),
    BEACON_OUT_M past the kerb, facing along the road."""
    out = []
    for z in place(near_roads, found):
        for sd in (1, -1):
            out.append((z["c"][:2] + z["n"] * sd * (z["half"] + BEACON_OUT_M), z["u"]))
    return out


def tile_rows(frame, near_roads, found, box, elev):
    """The beacons' Forest rows of road tile box (those standing in it)."""
    if not is_hk(found):
        return []
    x0, y0, x1, y1 = box[:4]
    return [_row(p, t, frame, elev) for p, t in beacons(near_roads, found)
            if x0 <= p[0] < x1 and y0 <= p[1] < y1]
