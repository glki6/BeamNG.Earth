"""Hong Kong's bus-only lanes: the Transport Department's BUS_ONLY_LANE layer (Road Network, 2nd generation, layer 9)
marked on the map's roads, as OpenStreetMap's bus:lanes / psv:lanes are everywhere (markings.lane_plan: Sweden's M6, a
solid line elsewhere).

Why: Hong Kong's roads got a bus lane line only where OpenStreetMap tags bus:lanes, and few of its 97 bus-only lanes
(Transport Department, June 2026) are tagged lane by lane. The Department keeps every gazetted bus-only lane as a line
of its Road Network's BUS_ONLY_LANE layer (with its hours and days), on the CSDI Portal's ArcGIS service whose
centrelines roadz.py reads (layer 10) and whose speed limits hkspeed.py reads (layer 2). Hong Kong drives on the left
and its with-flow bus lanes are the kerb lane: the lane by the left kerb in the direction of travel (Road Users' Code,
"Bus lanes": "A thick solid white line on the road marks the edge of a bus lane"; Road Traffic (Traffic Control)
Regulations, Cap. 374G reg. 2 and 12: a bus lane is bounded by the continuous white line of Figure No. 504, or the broken
line of Figure No. 518).

The whole layer is one or two requests (hklidar.arcgis: 2,000 a page), fetched once with every field (outFields *, the
schema is not pinned here) and kept on disk (<cache>/hkbuslanes/<VERSION>/lanes.json.gz), asked again after MAX_AGE_S
(the layer changes monthly) with the old copy kept while the service does not answer. Each process loads it once
(identity: the road tiles' key).

lane_tags(feats, i, tags): road feature i's OpenStreetMap lane tags with KEY added where a bus lane runs along it: the
geometry's sides its bus lanes are on ("left" / "right", where the layer's line runs off the road's centreline) or the
direction its line runs relative to the road's ("along" / "against", where it runs on the centreline). markings.lane_plan
turns that into the kerb lane's solid line on the side traffic keeps to; a contra-flow lane (against a one-way road)
is left alone. OpenStreetMap's own bus:lanes win where they are mapped. A road takes a side when at least COVER of its
stations have a bus lane line within MATCH_M whose direction agrees within MATCH_DEG.

Off: BNE_HK_BUS_LANES=off. Its source and terms are in CREDITS.md.
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

log = logging.getLogger("earthd.hkbuslanes")

ON = os.environ.get("BNE_HK_BUS_LANES", "on").lower() not in ("off", "0", "false")
VERSION = "hkbuslanes-v1"
URL = "https://portal.csdi.gov.hk/server/rest/services/common/td_rcd_1638949160594_2844/FeatureServer/9/query"
KEY = "bne:hk_bus_lane"             # (not an OpenStreetMap key: the lane tags' own, markings.lane_plan reads it)
BOUNDS = hklidar.BOUNDS             # lat0, lon0, lat1, lon1: the Hong Kong SAR
LAT0, LON0 = 22.35, 114.13          # the local metres' origin (equirectangular: < 0.1 % over the SAR)
MAX_AGE_S = 30 * 86400              # s: the layer is asked again after this (it changes monthly)
RETRY_S = 3600                      # s: unreachable, asked again after this
MATCH_M = 8.0                       # m from a road's station to a bus lane's line: a kerb lane of a 4-lane road ~5 m off
MATCH_DEG = 30.0                    # degrees at most between the road's and the line's direction (either way)
SIDE_M = 1.5                        # m off the road's centreline from which the line is a side's lane, not the road's
COVER = 0.5                         # share of the road's stations that must see the same bus lane
STEP_M = 5.0                        # m between the road's stations
CELL_M = 50.0                       # grid the lines' segments are looked up in

_KX = 111320.0 * dmath.cos(dmath.radians(LAT0))
_KY = 110574.0
_lock = threading.Lock()
_state = {"cache": None, "fetcher": None, "lanes": None, "down_at": 0.0}
LOCAL = None                        # (tests: a parsed list of rows used instead of the cache and the service)


def configure(cache_dir, fetcher):
    """MapData does it in every process that has one; until then nothing is fetched."""
    with _lock:
        if _state["fetcher"] is None:
            _state["cache"], _state["fetcher"] = cache_dir, fetcher


def parse(features):
    """The rows [{attributes}, [[lon, lat], ...]] of an ArcGIS query's features (outSR 4326): each path its own row, and
    each ring of a polygon layer (a lane drawn as an area: its two long edges lie either side of the lane's middle, the
    side found the same). Features without a line of two points are left out."""
    rows = []
    for f in features or ():
        if not isinstance(f, dict):
            continue
        a = {k: v for k, v in (f.get("attributes") or {}).items() if isinstance(v, (str, int, float)) or v is None}
        g = f.get("geometry") or {}
        for p in (g.get("paths") or []) + (g.get("rings") or []):
            try:
                pts = [[round(float(q[0]), 7), round(float(q[1]), 7)] for q in p]
            except (TypeError, ValueError, IndexError):
                continue
            pts = [q for q in pts if math.isfinite(q[0]) and math.isfinite(q[1])]
            if len(pts) >= 2:
                rows.append([a, pts])
    return rows


def _local(ll):
    p = np.asarray(ll, np.float64).reshape(-1, 2)
    return np.column_stack([(p[:, 0] - LON0) * _KX, (p[:, 1] - LAT0) * _KY])


class Lanes:
    """The layer's lines as segments in local metres, in a grid: near(x, y) the segments around a point."""

    def __init__(self, rows, ident=""):
        self.ident = ident
        self.rows = rows
        a, b = [], []
        for _attrs, pts in rows:
            xy = _local(pts)
            a.append(xy[:-1])
            b.append(xy[1:])
        self.a = np.concatenate(a) if a else np.zeros((0, 2))
        self.b = np.concatenate(b) if b else np.zeros((0, 2))
        d = self.b - self.a
        self.len = dnp.hypot(d[:, 0], d[:, 1])
        keep = self.len > 1e-6
        self.a, self.b, self.len, d = self.a[keep], self.b[keep], self.len[keep], d[keep]
        self.u = d / self.len[:, None] if len(d) else d
        self.grid = {}
        lo = np.floor(np.minimum(self.a, self.b) / CELL_M).astype(np.int64)
        hi = np.floor(np.maximum(self.a, self.b) / CELL_M).astype(np.int64)
        for s in range(len(self.a)):
            for gx in range(lo[s, 0], hi[s, 0] + 1):
                for gy in range(lo[s, 1], hi[s, 1] + 1):
                    self.grid.setdefault((gx, gy), []).append(s)

    def __len__(self):
        return len(self.a)

    def near(self, x, y):
        gx, gy = int(math.floor(x / CELL_M)), int(math.floor(y / CELL_M))
        cand = set()
        for ix in (gx - 1, gx, gx + 1):
            for iy in (gy - 1, gy, gy + 1):
                cand.update(self.grid.get((ix, iy), ()))
        return np.fromiter(sorted(cand), np.int64) if cand else None

    def sides(self, xy):
        """The bus lanes along a road line xy (k, 2) local metres: a set of "left" / "right" (the line's side of the
        road, off its centreline) and "along" / "against" (on its centreline, running with or against the road)."""
        d = np.diff(xy, axis=0)
        seg = dnp.hypot(d[:, 0], d[:, 1])
        total = float(seg.sum())
        if total <= 0 or not len(self):
            return set()
        n = max(2, int(math.ceil(total / STEP_M)) + 1)
        s = np.r_[0.0, np.cumsum(seg)]
        at = np.linspace(0.0, total, n)
        k = np.clip(np.searchsorted(s, at, side="right") - 1, 0, len(seg) - 1)
        f = (at - s[k]) / np.maximum(seg[k], 1e-9)
        px = xy[k, 0] + d[k, 0] * f
        py = xy[k, 1] + d[k, 1] * f
        ux = d[k, 0] / np.maximum(seg[k], 1e-9)
        uy = d[k, 1] / np.maximum(seg[k], 1e-9)
        cos_ok = dmath.cos(dmath.radians(MATCH_DEG))
        counts = {"left": 0, "right": 0, "along": 0, "against": 0}
        for j in range(n):
            cand = self.near(float(px[j]), float(py[j]))
            if cand is None:
                continue
            a, u, L = self.a[cand], self.u[cand], self.len[cand]
            t = np.clip((px[j] - a[:, 0]) * u[:, 0] + (py[j] - a[:, 1]) * u[:, 1], 0.0, L)
            qx, qy = a[:, 0] + u[:, 0] * t, a[:, 1] + u[:, 1] * t
            dist = dnp.hypot(qx - px[j], qy - py[j])
            dot = u[:, 0] * ux[j] + u[:, 1] * uy[j]
            ok = (dist <= MATCH_M) & (np.abs(dot) >= cos_ok)
            seen = set()
            for c in np.flatnonzero(ok):
                off = float(ux[j] * (qy[c] - py[j]) - uy[j] * (qx[c] - px[j]))     # (+ left of the road's direction)
                if off >= SIDE_M:
                    seen.add("left")
                elif off <= -SIDE_M:
                    seen.add("right")
                else:
                    seen.add("along" if dot[c] > 0 else "against")
            for side in seen:
                counts[side] += 1
        return {side for side, c in counts.items() if c >= COVER * n}


def _path():
    cache = _state["cache"]
    return os.path.join(cache, "hkbuslanes", VERSION, "lanes.json.gz") if cache else None


def _read(path):
    """(rows, identity) of the kept file, None when it cannot be read."""
    try:
        with open(path, "rb") as fh:
            body = fh.read()
        return json.loads(gzip.decompress(body).decode("utf-8")), hashlib.sha256(body).hexdigest()[:12]
    except (OSError, ValueError, EOFError) as e:
        log.debug("hk bus lanes unreadable (%s): %s", path, e)
        return None


def _fetch():
    """The layer's rows from the service, None when it does not answer (backs off RETRY_S)."""
    fetcher = _state["fetcher"]
    if fetcher is None or time.time() - _state["down_at"] < RETRY_S:
        return None

    def get_json(url):
        return json.loads(fetcher.get(url, kind="demand").body.decode("utf-8"))
    try:
        return parse(hklidar.arcgis(get_json, URL, {
            "where": "1=1", "outFields": "*", "returnGeometry": "true", "outSR": 4326,
            "orderByFields": "OBJECTID", "geometryPrecision": 6}))
    except Exception as e:      # (offline, or the service changed: OpenStreetMap's bus lanes only, the build not kept)
        log.info("hk bus lanes not fetched: %s: %s", type(e).__name__, e)
        _state["down_at"] = time.time()
        return None


def _write(path, rows):
    body = gzip.compress(json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8"), mtime=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as fh:
        fh.write(body)
    os.replace(path + ".part", path)
    return hashlib.sha256(body).hexdigest()[:12]


def lanes(fetch=True):
    """The process's Lanes (loaded once and kept), None while they cannot be had. fetch=False: the disk only."""
    got = _state["lanes"]
    if got is not None:
        return got
    with _lock:
        if _state["lanes"] is not None:
            return _state["lanes"]
        if LOCAL is not None:
            _state["lanes"] = Lanes(LOCAL, "local")
            return _state["lanes"]
        path = _path()
        kept = _read(path) if path and os.path.exists(path) else None
        fresh = kept is not None and time.time() - os.path.getmtime(path) < MAX_AGE_S
        if not fresh and fetch:
            rows = _fetch()
            if rows:
                ident = _write(path, rows) if path else hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:12]
                kept = (rows, ident)
        if kept is None:
            return None
        _state["lanes"] = Lanes(*kept)
        log.info("hk bus lanes: %d lines, %d segments (%s)", len(_state["lanes"].rows), len(_state["lanes"]),
                 _state["lanes"].ident)
        return _state["lanes"]


def identity():
    """The bus lanes' identity for cache keys: "off", "none" (none on disk yet) or the kept file's hash."""
    if not ON:
        return "off"
    got = lanes(fetch=False)
    return got.ident if got is not None else "none"


def _in_hk(lon, lat):
    return BOUNDS[0] <= lat <= BOUNDS[2] and BOUNDS[1] <= lon <= BOUNDS[3]


def _line(feats, i):
    """Road feature i's first line in local metres, None outside Hong Kong or without one."""
    try:
        p = np.asarray(feats.parts(i)[0][0], np.float64)[:, :2]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    if len(p) < 2 or not np.isfinite(p).all() or not _in_hk(float(p[:, 0].mean()), float(p[:, 1].mean())):
        return None
    return _local(p)


def sides(feats, i):
    """The bus lanes of road feature i (Lanes.sides; kept on feats), an empty set outside Hong Kong or with none."""
    memo = feats.__dict__.setdefault("_hkbus_sides", {}) if hasattr(feats, "__dict__") else {}
    got = memo.get(i)
    if got is not None:
        return got
    got = frozenset()
    xy = _line(feats, i) if ON else None
    if xy is not None:
        lane_set = lanes()
        if lane_set is None:
            import revisit
            revisit.fault(here=True)        # (made on OpenStreetMap's bus lanes alone for want of the layer: not kept)
            return got                      # (not kept on feats either: asked again)
        got = frozenset(lane_set.sides(xy))
    memo[i] = got
    return got


def value(found):
    """KEY's value for a set of sides (Lanes.sides), None for none: the sides in a fixed order, ";" between them."""
    order = ("left", "right", "along", "against")
    return ";".join(s for s in order if s in found) or None


def lane_tags(feats, i, tags):
    """Road feature i's lane tags (roads.lane_tags: {key: value} or None) with KEY where a bus lane runs along it."""
    if not ON or not hasattr(feats, "parts"):
        return tags
    v = value(sides(feats, i))
    if v is None:
        return tags
    return {**(tags or {}), KEY: v}
