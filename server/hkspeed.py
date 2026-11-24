"""Hong Kong's speed limits: the Transport Department's posted limits (Road Network, 2nd generation, layer SPEED_LIMIT)
and the law's 50 km/h on every other road, as Sweden's and the US states' defaults are in navgraph.road_mph.

Why: a Hong Kong road without a limit in the map took its class's default for the traffic and the speed-limit signs
(an expressway 120 km/h, a primary road out of town 80), where the law says 50 unless a sign says otherwise:
"where there is no sign on speed limit erected on the road, the speed limit is 50 km/h" (section 40 of the Road
Traffic Ordinance, Cap. 374; the Government's reply of 2009-09-14). The Transport Department keeps every stretch
whose limit was varied by Gazette notice as a line of its Road Network's SPEED_LIMIT layer (4,385 lines on 2026-11-24,
"80 km/h", "30 km/h" ...; monthly), on the CSDI Portal's ArcGIS service roadz.py reads the centrelines of (layer 10).

The whole layer is a few requests (hklidar.arcgis: 2,000 a page), fetched once and kept on disk
(<cache>/hkspeed/<VERSION>/limits.json.gz), asked again after MAX_AGE_S (the layer changes monthly) with the old copy
kept while the service does not answer. Each process loads it once and keeps it (identity: the sign tiles' key).

kmh(feats, i, at): the posted limit of road feature i at fraction at of its line: the nearest line of the layer within
MATCH_M of that point whose direction agrees within MATCH_DEG (a cross street's end at a junction does not take the
expressway's). None outside Hong Kong, where no line matches, or while the layer cannot be had (revisit.fault: a sign
tile made then is not kept). road_mph takes it before the map's own limit (the Department's is the one gazetted), and
where nothing matches gives the ordinary classes 50 km/h once the layer is loaded (loaded()): a road the layer does not
list is at the law's limit. Motorways the layer does not list keep their class default (a miss, not a 50 km/h road).

Off: BNE_HK_SPEED=off. Its source and terms are in CREDITS.md (the roadz "hk" row: the same service and terms).
"""
import gzip
import hashlib
import json
import logging
import math
import os
import re
import threading
import time

import numpy as np

import dmath
import dnp

import hklidar

log = logging.getLogger("earthd.hkspeed")

ON = os.environ.get("BNE_HK_SPEED", "on").lower() not in ("off", "0", "false")
VERSION = "hkspeed-v1"
URL = ("https://portal.csdi.gov.hk/server/rest/services/common/td_rcd_1638949160594_2844/FeatureServer/2/query")
BOUNDS = hklidar.BOUNDS             # lat0, lon0, lat1, lon1: the Hong Kong SAR
LAT0, LON0 = 22.35, 114.13          # the local metres' origin (equirectangular: < 0.1 % over the SAR)
MAX_AGE_S = 30 * 86400              # s: the layer is asked again after this (it changes monthly)
RETRY_S = 3600                      # s: unreachable, asked again after this
MATCH_M = 8.0                       # m from a road's point to a limit's line (the two centrelines' own few metres)
MATCH_DEG = 30.0                    # degrees at most between the road's and the line's direction (either way)
CELL_M = 50.0                       # grid the lines' segments are looked up in
KMH = (5, 130)                      # a value outside this is not a limit (not taken)
LAW_KMH = 50.0                      # Road Traffic Ordinance s. 40: every road without a sign
LAW_CLASSES = {"trunk", "primary", "secondary", "tertiary", "unclassified", "residential"}

_KX = 111320.0 * dmath.cos(dmath.radians(LAT0))
_KY = 110574.0
_VALUE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:km|kph)", re.I)
_lock = threading.Lock()
_state = {"cache": None, "fetcher": None, "limits": None, "down_at": 0.0}
LOCAL = None                        # (tests: a parsed list of rows used instead of the cache and the service)


def configure(cache_dir, fetcher):
    """MapData does it in every process that has one; until then nothing is fetched."""
    with _lock:
        if _state["fetcher"] is None:
            _state["cache"], _state["fetcher"] = cache_dir, fetcher


def value_kmh(text):
    """The km/h of a SPEED_LIMIT value ("80 km/h" -> 80.0), None for anything else."""
    m = _VALUE.search(str(text or ""))
    if not m:
        return None
    v = float(m.group(1))
    return v if KMH[0] <= v <= KMH[1] else None


def parse(features):
    """The rows [kmh, bound, [[lon, lat], ...]] of an ArcGIS query's features (outSR 4326), each path its own row;
    features without a readable limit or a line of two points are left out."""
    rows, skipped = [], 0
    for f in features or ():
        a = f.get("attributes") or {}
        kmh = value_kmh(a.get("SPEED_LIMIT"))
        paths = (f.get("geometry") or {}).get("paths") or []
        if kmh is None:
            skipped += 1
            continue
        for p in paths:
            pts = [[round(float(x), 7), round(float(y), 7)] for x, y, *_ in p
                   if math.isfinite(float(x)) and math.isfinite(float(y))]
            if len(pts) >= 2:
                rows.append([kmh, int(a.get("BOUND") or 0), pts])
    if skipped:
        log.info("hk speed limits: %d lines without a km/h value left out", skipped)
    return rows


class Limits:
    """The layer's lines as segments in local metres, in a grid: at(x, y, dx, dy) the limit there."""

    def __init__(self, rows, ident=""):
        self.ident = ident
        a, b, k = [], [], []
        for kmh, _bound, pts in rows:
            p = np.asarray(pts, np.float64)
            xy = np.column_stack([(p[:, 0] - LON0) * _KX, (p[:, 1] - LAT0) * _KY])
            a.append(xy[:-1])
            b.append(xy[1:])
            k.append(np.full(len(xy) - 1, float(kmh)))
        self.a = np.concatenate(a) if a else np.zeros((0, 2))
        self.b = np.concatenate(b) if b else np.zeros((0, 2))
        self.kmh = np.concatenate(k) if k else np.zeros(0)
        d = self.b - self.a
        self.len = dnp.hypot(d[:, 0], d[:, 1])
        keep = self.len > 1e-6
        self.a, self.b, self.kmh, self.len, d = self.a[keep], self.b[keep], self.kmh[keep], self.len[keep], d[keep]
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

    def at(self, x, y, dx, dy):
        """The km/h of the nearest segment within MATCH_M of local point (x, y) whose direction agrees with (dx, dy)
        within MATCH_DEG either way, else None."""
        n = math.hypot(dx, dy)
        if n <= 0:
            return None
        dx, dy = dx / n, dy / n
        gx, gy = int(math.floor(x / CELL_M)), int(math.floor(y / CELL_M))
        cand = set()
        for ix in (gx - 1, gx, gx + 1):
            for iy in (gy - 1, gy, gy + 1):
                cand.update(self.grid.get((ix, iy), ()))
        if not cand:
            return None
        s = np.fromiter(cand, np.int64)
        a, u, L = self.a[s], self.u[s], self.len[s]
        t = np.clip((x - a[:, 0]) * u[:, 0] + (y - a[:, 1]) * u[:, 1], 0.0, L)
        dist = dnp.hypot(a[:, 0] + u[:, 0] * t - x, a[:, 1] + u[:, 1] * t - y)
        agree = np.abs(u[:, 0] * dx + u[:, 1] * dy) >= dmath.cos(dmath.radians(MATCH_DEG))
        ok = (dist <= MATCH_M) & agree
        if not ok.any():
            return None
        j = int(np.argmin(np.where(ok, dist, np.inf)))
        return float(self.kmh[s[j]])


def _path():
    cache = _state["cache"]
    return os.path.join(cache, "hkspeed", VERSION, "limits.json.gz") if cache else None


def _read(path):
    """(rows, identity) of the kept file, None when it cannot be read."""
    try:
        with open(path, "rb") as fh:
            body = fh.read()
        return json.loads(gzip.decompress(body).decode("utf-8")), hashlib.sha256(body).hexdigest()[:12]
    except (OSError, ValueError, EOFError) as e:
        log.debug("hk speed limits unreadable (%s): %s", path, e)
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
            "where": "1=1", "outFields": "SPEED_LIMIT,BOUND", "returnGeometry": "true", "outSR": 4326,
            "orderByFields": "OBJECTID", "geometryPrecision": 6}))
    except Exception as e:      # (offline, or the service changed: the class defaults, and the build is not kept)
        log.info("hk speed limits not fetched: %s: %s", type(e).__name__, e)
        _state["down_at"] = time.time()
        return None


def _write(path, rows):
    body = gzip.compress(json.dumps(rows, separators=(",", ":")).encode("utf-8"), mtime=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as fh:
        fh.write(body)
    os.replace(path + ".part", path)
    return hashlib.sha256(body).hexdigest()[:12]


def limits(fetch=True):
    """The process's Limits (loaded once and kept), None while they cannot be had. fetch=False: the disk only."""
    got = _state["limits"]
    if got is not None:
        return got
    with _lock:
        if _state["limits"] is not None:
            return _state["limits"]
        if LOCAL is not None:
            _state["limits"] = Limits(LOCAL, "local")
            return _state["limits"]
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
        _state["limits"] = Limits(*kept)
        log.info("hk speed limits: %d segments (%s)", len(_state["limits"]), _state["limits"].ident)
        return _state["limits"]


def loaded():
    """Are the posted limits loaded in this process (an unlisted road is then at the law's limit)?"""
    return _state["limits"] is not None


def identity():
    """The posted limits' identity for cache keys: "off", "none" (none on disk yet) or the kept file's hash."""
    if not ON:
        return "off"
    got = limits(fetch=False)
    return got.ident if got is not None else "none"


def _in_hk(lon, lat):
    return BOUNDS[0] <= lat <= BOUNDS[2] and BOUNDS[1] <= lon <= BOUNDS[3]


def _line(feats, i):
    """Road feature i's first line in local metres and its cumulative lengths, kept on feats; None without one."""
    memo = feats.__dict__.setdefault("_hkspeed_lines", {}) if hasattr(feats, "__dict__") and not isinstance(
        feats, type) else {}
    got = memo.get(i, False)
    if got is not False:
        return got
    got = None
    try:
        ring = feats.parts(i)[0][0]
        p = np.asarray(ring, np.float64)[:, :2]
        if len(p) >= 2 and np.isfinite(p).all() and _in_hk(float(p[:, 0].mean()), float(p[:, 1].mean())):
            xy = np.column_stack([(p[:, 0] - LON0) * _KX, (p[:, 1] - LAT0) * _KY])
            s = np.concatenate([[0.0], np.cumsum(dnp.hypot(*np.diff(xy, axis=0).T))])
            if s[-1] > 0:
                got = (xy, s)
    except (AttributeError, IndexError, TypeError, ValueError):
        got = None
    memo[i] = got
    return got


def kmh(feats, i, at=0.5):
    """The posted limit (km/h) of road feature i at fraction at of its line, None where none applies (module doc)."""
    if not ON or not hasattr(feats, "parts"):
        return None
    line = _line(feats, i)
    if line is None:
        return None
    lim = limits()
    if lim is None:
        import revisit
        revisit.fault(here=True)        # (made on the class defaults for want of the layer: not kept)
        return None
    xy, s = line
    d = min(max(float(at), 0.0), 1.0) * s[-1]
    k = int(np.clip(np.searchsorted(s, d, side="right") - 1, 0, len(s) - 2))
    seg = xy[k + 1] - xy[k]
    f = (d - s[k]) / max(s[k + 1] - s[k], 1e-9)
    x, y = xy[k] + seg * f
    return lim.at(float(x), float(y), float(seg[0]), float(seg[1]))
