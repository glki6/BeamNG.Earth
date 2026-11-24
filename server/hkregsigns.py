"""Hong Kong's regulatory road signs where the Transport Department maps them (the user, 2026-11-24: what other regions
have, in Hong Kong's style). Germany, Poland, Czechia, Ukraine, Sweden and Bulgaria place their regulatory signs from
the road data (roadsigns.py and the countries' modules); Hong Kong's Transport Department publishes where its signs
stand, so they are placed from that.

The signs are the Road Network (2nd generation) layers on the CSDI Portal's ArcGIS service that roadz.py reads the
centrelines of (`td_rcd_1638949160594_2844`):

    layer 5 PROHIBITION           points: "traffic sign of prohibition" (2,339 on 2026-11-24): the vehicle classes
                                  barred (INC_VEH_TYPE), those excepted (EXC_VEH_TYPE), a weight or length the ban
                                  applies over (OTHER_REST_TYPE_GV), and REMARKS ("GV Proh/ Over 12 Tonnes/ E FA")
    layer 0 VEHICLE_RESTRICTION   points: "traffic sign of vehicle restriction" (838): MAX_HEIGHT, MAX_WIDTH,
                                  MAX_LENGTH, MAX_WEIGHT, MAX_AXLE_WEIGHT (metres, tonnes) and REMARKS ("E WP")

The service publishes no code lists (every field's domain is empty, read 2026-11-24) and the data dictionary is a zip
on static.data.gov.hk that could not be read here. The codes below are read from the records' own REMARKS, which spell
them out beside the codes (INC_VEH_TYPE "GV" beside "GV Proh", "PLB" beside "PLB Proh"; INC "OTH" beside "LD Proh"),
and from the vehicle classes of the Road Users' Code's signs (Transport Department, "Traffic signs giving orders",
signs 32-39 and 47-51, read 2026-11-24):

    GV      goods vehicles           "No goods vehicles" (sign 38): a lorry
    PLB     public light buses       "No public light buses" (sign 37): a minibus
    BUS     buses and coaches        "No buses and coaches" (sign 36): a bus          (code inferred: not in the sample)
    MC      motor cycles             "No motor cycles and motor tricycles" (sign 35)   (code from REMARKS "E MC")
    LD      learner drivers          "No motor vehicles driven by learner drivers" (sign 39): an L   (from REMARKS)
    ALL     every vehicle            "Vehicles prohibited as indicated by supplementary plate" (sign 32): the plain ring
    E xx    an exception plate under the sign: FA "Except for access" (the code's wording), WP "Except with permit",
            PC private cars, MC, GMB green minibuses, FB franchised buses, PLB, GV, BUS (inferred from the codes)

A ban over a weight or a length gets the plate "Over 12 tonnes" / "Over 10 m" (the Road Users' Code's own pictures'
words, signs 55 and 56). A restriction point is the red ring with its figure: the height between two arrowheads, the
width between two, the length over a double arrow, the gross weight alone, the axle weight over an axle (signs 47-51).
A part-time ban's times are not in the data: its time plate is left out. A record whose classes are none of these
(OTH without LD) gets no sign.

Hong Kong's signs follow the United Kingdom's (Road Traffic (Traffic Control) Regulations Schedule 1): white discs in a
red ring, black symbols. The pages read give no drawing's measures: the ring is a tenth of the diameter as the UK's,
the disc 600 mm (750 mm on fast roads), the plates' English in Transport Heavy (the light-background face, TPDM Vol 3
1.3.1.7). Each plate is two lines, English over Traditional Chinese (as Hong Kong's signs set their two languages,
English above), the Chinese in Noto Sans HK Bold (the street name plates' cut, NotoSansHK-Bold-streets.otf).

The plates' Chinese (CHINESE): no page read showed a plate's Chinese as printed (the Road Users' Code's plate pictures
could not be read), so it is built from the Transport Department's own words for each part, and the pattern "X例外"
the regulations give the bilingual plate "Except General Holidays / 公眾假期例外" (LN 86 of 2001) and the Road Users'
Code's heading "巴士例外" (its "except bus" plate):
    seen as such     巴士例外 (except buses)
    parts seen       專營巴士 (sign 65), 電單車 (sign 35), 公共小型巴士 (sign 37), 貨車 (sign 38) + 例外;
                     超逾 N 公噸 (TD notices: "車輛總重超逾4公噸"), 超逾 N 米 (米: TD's metre)
    guessed          出入例外 (except for access; sign 54's caption is prose), 持許可證者例外 (permit: the notices'
                     "許可證"), 私家車例外, 專線小巴例外 (green minibuses)

Placement: a sign stands at the kerb beside its point, facing the traffic it applies to. The layers' BOUND field is
not documented and is not used. A point snaps to the nearest road within SNAP_M; on a one-way road the sign faces its
traffic, on its left (left-hand traffic); on a two-way road within END_M of a line's end the sign faces the traffic
entering from that end (a ban is signed where a road is entered); elsewhere one sign faces each direction. Several
signs of one point share a post, stacked.

Each face (a sign with its plates) is a picture drawn here (one per face, named by its hash, as the street name plates
and Sweden's bus stop plates are) and joins the sign tile's direction-board mesh; the disc is cut out by the mesh.

Fetched whole once per layer (hklidar.arcgis), kept on disk (<cache>/hkregsigns/<VERSION>/signs.json.gz) and asked
again after MAX_AGE_S; without it (offline, nothing kept) a sign tile is built without them and not kept
(revisit.fault). identity(): the kept file's hash, for the sign tiles' key. BNE_HK_REGSIGNS=off turns them off.
"""
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import time
from functools import lru_cache

import numpy as np

import dmath
import dnp
import hklidar

log = logging.getLogger("earthd.hkregsigns")

ON = os.environ.get("BNE_HK_REGSIGNS", "on").strip().lower() not in ("off", "0", "false", "no")
COUNTRIES = {"HK"}
VERSION = "hkregsigns-v1"
FACE_VERSION = 2
BASE = "https://portal.csdi.gov.hk/server/rest/services/common/td_rcd_1638949160594_2844/FeatureServer"
LAYERS = {"pro": (5, ("INC_VEH_TYPE", "EXC_VEH_TYPE", "OTHER_REST_TYPE_GV", "PART_TIME_PROHIBITION", "REMARKS")),
          "vr": (0, ("MAX_HEIGHT", "MAX_WIDTH", "MAX_LENGTH", "MAX_WEIGHT", "MAX_AXLE_WEIGHT", "REMARKS"))}
MAX_AGE_S = 30 * 86400              # s: asked again after this (the layers change monthly)
RETRY_S = 3600                      # s: unreachable, asked again after this
REACH_M = 40.0                      # m past a tile's box its points are read (a post stands up to this far off)

FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fonts", "AG-Transport-Heavy.ttf")
ZH_FONT = os.path.join(os.path.dirname(FONT), "NotoSansHK-Bold-streets.otf")
METAL = "bne_dir_metal"             # (dirsigns.METAL: the boards' bare metal, their backs)
RED, BLACK, WHITE = (196, 18, 32), (12, 12, 12), (255, 255, 255)
PX_PER_M = 600.0
DISC = (0.6, 0.75)                  # m: the disc on a town road, on a road over URBAN_KMH
URBAN_KMH = 70.0
RING = 0.10                         # the red ring's width over the diameter
PLATE_H = 0.22                      # m: a plate's height (its English over its Chinese)
PLATE_CAP = 0.05                    # m: its capitals
ZH_H = 0.065                        # m: its Chinese characters' ink height
ZH_GAP = 0.025                      # m between the English and the Chinese
PLATE_SIDE = 0.04                   # m: lettering to its ends
PLATE_BORDER = 0.008
GAP = 0.03                          # m between a disc and its plate, and between stacked signs
MOUNT = 2.1                         # m: the lowest edge over the ground (a sign over a footway)
POST_W = 0.076                      # m: the post (square here)
THICK = 0.004
FAN = 32                            # the disc's segments
KERB_M = (0.6, 1.0, 1.5)            # m past the carriageway's edge, tried in order
SHIFT_M = (0.0, 4.0, -4.0, 8.0, -8.0)   # m along the road from the point, tried in order
CLEAR_M = 0.3                       # m from any other carriageway's edge
SNAP_M = 15.0                       # m: a point this far from every road is no road's
END_M = 30.0                        # m from a line's end: a two-way road's sign faces the traffic entering there
SAME_M = 8.0                        # m: a face this near the same face (both layers' records of one sign) is one
DECK_M = 1.0
MAX_STACK = 3
MAX_FACES = 40                      # pictures a tile makes at most
MAX_SIGNS = 120

PICTOGRAMS = {"GV": "lorry", "PLB": "minibus", "BUS": "bus", "MC": "motorcycle", "LD": "learner", "ALL": "ring"}
EXCEPT = {"FA": "Except for access", "WP": "Except with permit", "PC": "Except private cars",
          "MC": "Except motor cycles", "GMB": "Except green minibuses", "FB": "Except franchised buses",
          "PLB": "Except public light buses", "GV": "Except goods vehicles", "BUS": "Except buses"}
CHINESE = {"Except for access": "出入例外", "Except with permit": "持許可證者例外", "Except private cars": "私家車例外",
           "Except motor cycles": "電單車例外", "Except green minibuses": "專線小巴例外",
           "Except franchised buses": "專營巴士例外", "Except public light buses": "公共小型巴士例外",
           "Except goods vehicles": "貨車例外", "Except buses": "巴士例外"}
LIMITS = ("MAX_HEIGHT", "MAX_WIDTH", "MAX_LENGTH", "MAX_WEIGHT", "MAX_AXLE_WEIGHT")
LIMIT_KIND = {"MAX_HEIGHT": "height", "MAX_WIDTH": "width", "MAX_LENGTH": "length", "MAX_WEIGHT": "weight",
              "MAX_AXLE_WEIGHT": "axle"}
UNIT = {"height": "m", "width": "m", "length": "m", "weight": "t", "axle": "t"}

_lock = threading.Lock()
_state = {"cache": None, "fetcher": None, "records": None, "ident": None, "down_at": 0.0}
LOCAL = None                        # (tests: a list of records used instead of the cache and the service)


def wanted(country):
    """Whether a place's sign tiles carry Hong Kong's regulatory signs: Hong Kong's, unless BNE_HK_REGSIGNS=off."""
    return ON and str(country or "").upper() in COUNTRIES


# ---------------------------------------------------------------------------------------------------- the records

def _text(v):
    v = "" if v is None else str(v).strip()
    return "" if v.upper() in ("", "NA", "NULL", "NONE") else v


def _codes(v):
    return [c.strip().upper() for c in _text(v).split(",") if c.strip()]


def _number(v):
    """A limit's figure as the sign shows it ("5.5", "12"), "" when it is none."""
    t = _text(v)
    try:
        f = float(t)
    except ValueError:
        return ""
    if not (0 < f < 100) or not math.isfinite(f):
        return ""
    return f"{f:g}"


_EXC = re.compile(r"(?:^|[/\s])E\s+([A-Z]+)\b")
_OVER = re.compile(r"over\s+([0-9]+(?:\.[0-9]+)?)\s*(tonnes?|t|m)\b", re.I)


def exceptions(remarks, exc=""):
    """The exception plates' wordings of a record, in EXCEPT's order, each once: from REMARKS' "E FA" tokens and the
    excepted classes field."""
    found = {m.group(1) for m in _EXC.finditer(_text(remarks).upper().replace("/", " / "))}
    found |= {c for c in _codes(exc) if c in EXCEPT}
    return [EXCEPT[c] for c in EXCEPT if c in found]


def faces(rec):
    """[(pictogram or limit kind, figure, (plate, ...))] of one record ({"k": "pro" | "vr", fields}), in a stable
    order; [] for a record no sign here stands for."""
    plates = tuple(exceptions(rec.get("REMARKS"), rec.get("EXC_VEH_TYPE")))
    out = []
    if rec.get("k") == "pro":
        inc = _codes(rec.get("INC_VEH_TYPE"))
        remarks = _text(rec.get("REMARKS")).upper()
        classes = [c for c in inc if c in PICTOGRAMS]
        if "OTH" in inc and re.search(r"\bLD\b", remarks):
            classes.append("LD")
        over = ()
        m = _OVER.search(_text(rec.get("REMARKS")))
        kind = _text(rec.get("OTHER_REST_TYPE_GV")).lower()
        if m and kind in ("weight", "length"):
            unit = "tonnes" if m.group(2).lower().startswith("t") else "m"
            over = (f"Over {float(m.group(1)):g} {unit}",)
        if "ALL" in classes:            # (every vehicle: the plain ring, whatever else is listed)
            classes = ["ALL"]
        for c in sorted(set(classes), key=list(PICTOGRAMS).index):
            if c == "ALL" and not (plates or over):
                continue                # (a plain ring with nothing under it says nothing: no sign)
            out.append((PICTOGRAMS[c], "", over + plates))
    elif rec.get("k") == "vr":
        for f in LIMITS:
            v = _number(rec.get(f))
            if v:
                out.append((LIMIT_KIND[f], v + UNIT[LIMIT_KIND[f]], plates))
    return out


def parse(layer, features):
    """[{"k", lon, lat, fields}] of an ArcGIS query's point features (outSR 4326) of layer "pro" | "vr", in Hong Kong,
    rounded to 1e-7 degrees, in OBJECTID order."""
    out = []
    names = LAYERS[layer][1]
    for f in features or ():
        a = f.get("attributes") or {}
        g = f.get("geometry") or {}
        try:
            x, y = float(g["x"]), float(g["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y) and hklidar.BOUNDS[0] <= y <= hklidar.BOUNDS[2]
                and hklidar.BOUNDS[1] <= x <= hklidar.BOUNDS[3]):
            continue
        rec = {"k": layer, "lon": round(x, 7), "lat": round(y, 7), "id": a.get("OBJECTID")}
        rec.update({n: a.get(n) for n in names})
        out.append(rec)
    out.sort(key=lambda r: (r["id"] if isinstance(r["id"], int) else 0, r["lon"], r["lat"]))
    return out


def configure(cache_dir, fetcher):
    """Where the layers are kept and what fetches them (read() does it from its MapData); until then nothing is fetched."""
    with _lock:
        if _state["fetcher"] is None:
            _state["cache"], _state["fetcher"] = cache_dir, fetcher


def _path():
    cache = _state["cache"]
    return os.path.join(cache, "hkregsigns", VERSION, "signs.json.gz") if cache else None


def _read(path):
    try:
        with open(path, "rb") as fh:
            body = fh.read()
        return json.loads(gzip.decompress(body).decode("utf-8")), hashlib.sha256(body).hexdigest()[:12]
    except (OSError, ValueError, EOFError) as e:
        log.debug("hk regulatory signs unreadable (%s): %s", path, e)
        return None


def _write(path, records):
    body = gzip.compress(json.dumps(records, separators=(",", ":"), sort_keys=True).encode("utf-8"), mtime=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as fh:
        fh.write(body)
    os.replace(path + ".part", path)
    return hashlib.sha256(body).hexdigest()[:12]


def _fetch():
    """Both layers' records from the service, None when it does not answer (backs off RETRY_S)."""
    fetcher = _state["fetcher"]
    if fetcher is None or time.time() - _state["down_at"] < RETRY_S:
        return None

    def get_json(url):
        return json.loads(fetcher.get(url, kind="demand").body.decode("utf-8"))
    try:
        out = []
        for layer, (n, names) in LAYERS.items():
            out += parse(layer, hklidar.arcgis(get_json, f"{BASE}/{n}/query", {
                "where": "1=1", "outFields": ",".join(("OBJECTID",) + names), "returnGeometry": "true",
                "outSR": 4326, "orderByFields": "OBJECTID"}))
        return out
    except Exception as e:      # (offline, or the service changed: no signs, and the tile is not kept)
        log.info("hk regulatory signs not fetched: %s: %s", type(e).__name__, e)
        _state["down_at"] = time.time()
        return None


def records(fetch=True):
    """The process's records (loaded once and kept), None while they cannot be had. fetch=False: the disk only."""
    if _state["records"] is not None:
        return _state["records"]
    with _lock:
        if _state["records"] is not None:
            return _state["records"]
        if LOCAL is not None:
            _state["records"], _state["ident"] = [dict(r) for r in LOCAL], "local"
            return _state["records"]
        path = _path()
        kept = _read(path) if path and os.path.exists(path) else None
        fresh = kept is not None and time.time() - os.path.getmtime(path) < MAX_AGE_S
        if not fresh and fetch:
            got = _fetch()
            if got:
                kept = (got, _write(path, got) if path else
                        hashlib.sha256(json.dumps(got, sort_keys=True).encode()).hexdigest()[:12])
        if kept is None:
            return None
        _state["records"], _state["ident"] = kept
        log.info("hk regulatory signs: %d records (%s)", len(_state["records"]), _state["ident"])
        return _state["records"]


def identity():
    """For the sign tiles' key: "off", "none" (none on disk yet) or the drawing's version and the kept file's hash."""
    if not ON:
        return "off"
    return f"{FACE_VERSION}:{_state['ident']}" if records(fetch=False) is not None else "none"


# ---------------------------------------------------------------------------------------------------- the pictures

@lru_cache(maxsize=None)
def _font(px):
    from PIL import ImageFont
    return ImageFont.truetype(FONT, max(8, int(px)))


def _text_img(text, cap_px):
    """(L image of text's ink, sized so a capital is cap_px high)."""
    from PIL import Image, ImageDraw
    ref = _font(200)
    l, t, r, b = ref.getbbox("H", anchor="ls")
    font = _font(200 * cap_px / max(b - t, 1))
    l, t, r, b = font.getbbox(text, anchor="ls")
    im = Image.new("L", (max(1, r - l), max(1, b - t)))
    ImageDraw.Draw(im).text((-l, -t), text, font=font, fill=255, anchor="ls")
    return im


def chinese(text):
    """A plate's Chinese ("" where none is known): CHINESE's, or "超逾 N 公噸" / "超逾 N 米" for "Over N tonnes|m"."""
    if text in CHINESE:
        return CHINESE[text]
    m = re.fullmatch(r"Over ([0-9.]+) (tonnes|m)", text)
    return f"超逾{m.group(1)}{'公噸' if m.group(2) == 'tonnes' else '米'}" if m else ""


@lru_cache(maxsize=None)
def _zh_font(px):
    from PIL import ImageFont
    return ImageFont.truetype(ZH_FONT, max(8, int(px)))


def _zh_img(text, h_px):
    """(L image of Chinese text's ink, sized so a character's ink is about h_px high: 永's ink over the em)."""
    from PIL import Image, ImageDraw
    ref = _zh_font(200)
    l, t, r, b = ref.getbbox("永", anchor="ls")
    font = _zh_font(200 * h_px / max(b - t, 1))
    l, t, r, b = font.getbbox(text, anchor="ls")
    im = Image.new("L", (max(1, r - l), max(1, b - t)))
    ImageDraw.Draw(im).text((-l, -t), text, font=font, fill=255, anchor="ls")
    return im


def plate_width(text, disc):
    """A plate's width (m): its longer line and PLATE_SIDE either end, at least the disc's diameter, in 0.05 m
    steps."""
    zh = chinese(text)
    need = max(_text_img(text, PLATE_CAP * PX_PER_M).width,
               _zh_img(zh, ZH_H * PX_PER_M).width if zh else 0) / PX_PER_M + 2 * PLATE_SIDE
    return float(max(disc, math.ceil(round(need / 0.05, 6)) * 0.05))


def _poly(d, pts, cx, cy, s, fill):
    d.polygon([(cx + x * s, cy - y * s) for x, y in pts], fill=fill)


def _arrowhead(d, cx, cy, s, dx, dy, fill):
    """A solid arrowhead at unit-square point (dx, dy) pointing that way from the centre."""
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    tip = (dx, dy)
    back = (dx - ux * 0.24, dy - uy * 0.24)
    _poly(d, [tip, (back[0] + px * 0.17, back[1] + py * 0.17), (back[0] - px * 0.17, back[1] - py * 0.17)],
          cx, cy, s, fill)


def _wheel(d, cx, cy, s, x, y, r):
    d.ellipse((cx + (x - r) * s, cy - (y + r) * s, cx + (x + r) * s, cy - (y - r) * s), fill=BLACK)


def _symbol(d, kind, figure, cx, cy, r):
    """The symbol of a sign inside its ring: cx, cy the disc's centre (px), r its inner radius (px). Shapes in a unit
    square [-1, 1]^2 (y up), drawn at scale s."""
    s = r * 0.78
    if kind == "lorry":                 # a rigid lorry: cab on the left, box body, two wheels
        _poly(d, [(-0.85, -0.25), (-0.85, 0.15), (-0.62, 0.35), (-0.42, 0.35), (-0.42, -0.25)], cx, cy, s, BLACK)
        _poly(d, [(-0.36, -0.25), (-0.36, 0.5), (0.85, 0.5), (0.85, -0.25)], cx, cy, s, BLACK)
        for x in (-0.6, 0.5):
            _wheel(d, cx, cy, s, x, -0.32, 0.16)
    elif kind == "minibus":             # a van-shaped light bus, a row of windows
        _poly(d, [(-0.85, -0.25), (-0.85, 0.2), (-0.65, 0.45), (0.85, 0.45), (0.85, -0.25)], cx, cy, s, BLACK)
        for k in range(4):
            x0 = -0.6 + k * 0.36
            _poly(d, [(x0, 0.1), (x0, 0.33), (x0 + 0.26, 0.33), (x0 + 0.26, 0.1)], cx, cy, s, WHITE)
        for x in (-0.55, 0.55):
            _wheel(d, cx, cy, s, x, -0.3, 0.15)
    elif kind == "bus":                 # a single-deck bus, longer and taller
        _poly(d, [(-0.9, -0.3), (-0.9, 0.5), (0.9, 0.5), (0.9, -0.3)], cx, cy, s, BLACK)
        for k in range(5):
            x0 = -0.8 + k * 0.33
            _poly(d, [(x0, 0.12), (x0, 0.38), (x0 + 0.24, 0.38), (x0 + 0.24, 0.12)], cx, cy, s, WHITE)
        for x in (-0.55, 0.6):
            _wheel(d, cx, cy, s, x, -0.35, 0.15)
    elif kind == "motorcycle":          # two wheels, a frame and a rider's bulk
        for x in (-0.55, 0.55):
            d.ellipse((cx + (x - 0.27) * s, cy - (-0.2 + 0.27) * s, cx + (x + 0.27) * s, cy - (-0.2 - 0.27) * s),
                      outline=BLACK, width=max(2, int(0.09 * s)))
        _poly(d, [(-0.55, -0.2), (-0.2, 0.15), (0.35, 0.15), (0.55, -0.2), (0.45, -0.2), (0.3, 0.05), (-0.15, 0.05),
                  (-0.45, -0.2)], cx, cy, s, BLACK)
        _poly(d, [(-0.15, 0.15), (-0.05, 0.6), (0.15, 0.6), (0.2, 0.15)], cx, cy, s, BLACK)
        d.ellipse((cx - 0.02 * s, cy - 0.85 * s, cx + 0.22 * s, cy - 0.61 * s), fill=BLACK)
    elif kind == "learner":             # an L
        _poly(d, [(-0.3, -0.5), (-0.3, 0.55), (-0.08, 0.55), (-0.08, -0.3), (0.35, -0.3), (0.35, -0.5)],
              cx, cy, s, BLACK)
    elif kind in ("height", "width", "length", "weight", "axle"):
        im = _text_img(figure, 0.42 * r)
        if im.width > 1.5 * r:
            from PIL import Image
            im = im.resize((int(1.5 * r), im.height), Image.LANCZOS)
        ty = cy - im.height / 2 + (0.12 * r if kind in ("length", "axle") else 0)
        d.bitmap((int(cx - im.width / 2), int(ty)), im, fill=BLACK)
        if kind == "height":
            _arrowhead(d, cx, cy, r, 0, 0.88, BLACK)
            _arrowhead(d, cx, cy, r, 0, -0.88, BLACK)
        elif kind == "width":
            _arrowhead(d, cx, cy, r, 0.9, 0, BLACK)
            _arrowhead(d, cx, cy, r, -0.9, 0, BLACK)
        elif kind == "length":          # a double arrow over the figure
            _poly(d, [(-0.5, 0.43), (0.5, 0.43), (0.5, 0.49), (-0.5, 0.49)], cx, cy, r, BLACK)
            _arrowhead(d, cx, cy, r, -0.72, 0.46, BLACK)
            _arrowhead(d, cx, cy, r, 0.72, 0.46, BLACK)
        elif kind == "axle":            # an axle and its two wheels over the figure
            _poly(d, [(-0.4, 0.42), (0.4, 0.42), (0.4, 0.5), (-0.4, 0.5)], cx, cy, r, BLACK)
            for x in (-0.45, 0.45):
                _poly(d, [(x - 0.07, 0.3), (x - 0.07, 0.62), (x + 0.07, 0.62), (x + 0.07, 0.3)], cx, cy, r, BLACK)


def layout(kind, plates, disc):
    """(width m, height m, [(plate text, top m, width m)]) of a face: the disc on top, its plates under it."""
    w = disc
    rows, top = [], disc + GAP
    for p in plates:
        pw = plate_width(p, disc)
        rows.append((p, top, pw))
        w = max(w, pw)
        top += PLATE_H + GAP
    return w, (top - GAP if plates else disc), rows


def picture(kind, figure, plates, disc):
    """(PNG bytes, width m, height m, plate rows) of a face: the disc centred at the top (the mesh cuts it out), each
    plate centred under it; outside them the picture is the ring's red (no mip level bleeds white into the ring)."""
    from PIL import Image, ImageDraw
    w_m, h_m, rows = layout(kind, plates, disc)
    W, H = int(round(w_m * PX_PER_M)), int(round(h_m * PX_PER_M))
    img = Image.new("RGB", (W, H), RED)
    d = ImageDraw.Draw(img)
    D = disc * PX_PER_M
    cx, cy = W / 2, D / 2
    d.ellipse((cx - D / 2, cy - D / 2, cx + D / 2, cy + D / 2), fill=RED)
    ri = D / 2 * (1 - 2 * RING)
    d.ellipse((cx - ri, cy - ri, cx + ri, cy + ri), fill=WHITE)
    _symbol(d, kind, figure, cx, cy, ri)
    b = max(2, int(round(PLATE_BORDER * PX_PER_M)))
    for text, top, pw in rows:
        x0, y0 = (W - pw * PX_PER_M) / 2, top * PX_PER_M
        x1, y1 = x0 + pw * PX_PER_M, y0 + PLATE_H * PX_PER_M
        d.rectangle((x0, y0, x1, y1), fill=BLACK)
        d.rectangle((x0 + b, y0 + b, x1 - b, y1 - b), fill=WHITE)
        room = int(pw * PX_PER_M - 2 * PLATE_SIDE * PX_PER_M)
        zh = chinese(text)
        lines = [_text_img(text, PLATE_CAP * PX_PER_M)] + ([_zh_img(zh, ZH_H * PX_PER_M)] if zh else [])
        lines = [im.resize((room, im.height), Image.LANCZOS) if im.width > room else im for im in lines]
        gap = ZH_GAP * PX_PER_M if zh else 0.0
        y = y0 + (PLATE_H * PX_PER_M - sum(im.height for im in lines) - gap) / 2
        for im in lines:                # (the English over the Chinese)
            d.bitmap((int((W - im.width) / 2), int(round(y))), im, fill=BLACK)
            y += im.height + gap
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue(), w_m, h_m, rows


def face(out_dir, vfs_dir, kind, figure, plates, disc):
    """(material, vfs path, width m, height m, plate rows) of a face's picture: drawn once (named by its hash)."""
    h = hashlib.sha1(json.dumps([FACE_VERSION, kind, figure, list(plates), disc, PX_PER_M])
                     .encode("utf-8")).hexdigest()[:12]
    material = f"bne_hkreg_{h}"
    path = os.path.join(out_dir, material + ".color.dds")
    png, w, hh, rows = picture(kind, figure, plates, disc)
    if not os.path.exists(path):
        import imagery
        imagery.write_picture_dds(path, png)    # (a DDS, never a PNG the engine cooks)
    return material, f"{vfs_dir}/{material}.color.dds", w, hh, rows


# ---------------------------------------------------------------------------------------------------- placement

def _stations(line):
    return np.r_[0.0, np.cumsum(dnp.norm(np.diff(line[:, :2], axis=0), axis=1))]


def _project(p, line):
    """(distance, station, point (2,), unit tangent (2,)) of p's nearest point on polyline line."""
    a, b = line[:-1, :2], line[1:, :2]
    ab = b - a
    l2 = np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / l2, 0.0, 1.0)
    q = a + t[:, None] * ab
    dist = dnp.norm(q - p, axis=1)
    k = int(np.argmin(dist))
    st = _stations(line)
    u = ab[k] / max(math.sqrt(float(l2[k])), 1e-9)
    return float(dist[k]), float(st[k] + t[k] * math.sqrt(float(l2[k]))), q[k], u


def _at(line, s):
    st = _stations(line)
    s = min(max(s, 0.0), float(st[-1]))
    k = int(np.clip(np.searchsorted(st, s, side="right") - 1, 0, len(line) - 2))
    seg = line[k + 1, :2] - line[k, :2]
    n = max(float(dnp.norm(seg)), 1e-9)
    return line[k, :2] + seg / n * (s - float(st[k])), seg / n


def _seg_dist(p, line):
    a, b = line[:-1, :2], line[1:, :2]
    ab = b - a
    l2 = np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / l2, 0.0, 1.0)
    return float(dnp.norm(a + t[:, None] * ab - p, axis=1).min())


def directions(spec, station, length):
    """The travel directions (+1 along the line's geometry, -1 against) a point's signs face: a one-way road's own; a
    two-way road's entering from an end within END_M (both ends: both); else both."""
    if spec.oneway:
        return [int(spec.dir or 1)]
    near0, near1 = station <= END_M, length - station <= END_M
    if near0 and not near1:
        return [1]
    if near1 and not near0:
        return [-1]
    return [1, -1]


def spots(rows, points, decks=(), left_hand=True):
    """[(x, y, facing x, facing y, point index)] of the posts of points [(x, y)]: for each travel direction of its
    road (directions()), the first clear kerb spot (SHIFT_M along, KERB_M out, on the travel's near side), facing the
    oncoming traffic. rows: (fid, spec, line, width) of the paved roads near."""
    from exitsigns import _on_deck
    lines = [(fid, spec, np.asarray(line, float)[:, :2], float(width)) for fid, spec, line, width in rows
             if len(line) >= 2 and getattr(spec, "kind", "paved") in ("paved", "cobble")]
    out = []
    if not lines:
        return out
    for i, (x, y) in enumerate(points):
        p = np.array([x, y], float)
        best = None
        for fid, spec, line, width in lines:
            dist, st, q, u = _project(p, line)
            if dist <= SNAP_M + width / 2 and (best is None or dist < best[0]):
                best = (dist, st, spec, line, width)
        if best is None:
            continue
        _, st, spec, line, width = best
        length = float(_stations(line)[-1])
        for travel in directions(spec, st, length):
            for shift in SHIFT_M:
                q, u = _at(line, st - travel * shift)
                fwd = u * travel
                side = np.array([-fwd[1], fwd[0]]) if left_hand else np.array([fwd[1], -fwd[0]])
                found = None
                for off in KERB_M:
                    c = q + side * (width / 2 + off)
                    if any(_seg_dist(c, ln) < w / 2 + CLEAR_M for f, _, ln, w in lines):
                        continue
                    if decks and _on_deck(c, decks, DECK_M):
                        continue
                    found = c
                    break
                if found is not None:
                    out.append((float(found[0]), float(found[1]), float(-fwd[0]), float(-fwd[1]), i))
                    break
    return out


def signs(frame, recs, box, rows, decks=(), left_hand=True):
    """[(x, y, facing x, facing y, disc m, [(kind, figure, plates)])] of the posts standing in ENU box (x0, y0, x1,
    y1): every record within REACH_M placed as if there were no tile (the same from every tile), the faces of one point
    stacked (at most MAX_STACK), a face dropped where the same face already faces the same way within SAME_M."""
    if not recs:
        return []
    x0, y0, x1, y1 = (float(v) for v in box)
    ll = np.array([[r["lon"], r["lat"]] for r in recs], np.float64)
    e, n, _ = frame.geodetic_to_enu(ll[:, 1], ll[:, 0], 0.0)
    e, n = np.asarray(e, float).ravel(), np.asarray(n, float).ravel()
    near = np.flatnonzero((e >= x0 - REACH_M) & (e < x1 + REACH_M) & (n >= y0 - REACH_M) & (n < y1 + REACH_M))
    pts, fcs = [], []
    for k in near:
        f = faces(recs[int(k)])
        if f:
            pts.append((float(e[k]), float(n[k])))
            fcs.append(f)
    if not pts:
        return []
    order = sorted(range(len(pts)), key=lambda i: (round(pts[i][0], 2), round(pts[i][1], 2)))
    pts, fcs = [pts[i] for i in order], [fcs[i] for i in order]
    out, seen = [], []
    for x, y, fx, fy, i in spots(rows, pts, decks, left_hand):
        kmh = _kmh_near(rows, pts[i])
        disc = DISC[1] if kmh and kmh > URBAN_KMH else DISC[0]
        keep = []
        for fc in fcs[i]:
            if any(fc == s_fc and (s_x - x) * (s_x - x) + (s_y - y) * (s_y - y) < SAME_M * SAME_M
                   and s_fx * fx + s_fy * fy > 0.7
                   for s_x, s_y, s_fx, s_fy, s_fc in seen):
                continue
            seen.append((x, y, fx, fy, fc))
            keep.append(fc)
        if keep and x0 <= x < x1 and y0 <= y < y1:
            out.append((x, y, fx, fy, disc, keep[:MAX_STACK]))
        if len(out) >= MAX_SIGNS:
            break
    return out


def _kmh_near(rows, p):
    best, kmh = None, None
    p = np.asarray(p, float)
    for fid, spec, line, width in rows:
        if len(line) < 2:
            continue
        d = _seg_dist(p, np.asarray(line, float)[:, :2])
        if best is None or d < best:
            best, kmh = d, getattr(spec, "kmh", None)
    return kmh


# ---------------------------------------------------------------------------------------------------- the mesh

class _Mesh:
    def __init__(self, origin):
        self.origin = np.asarray(origin, float)
        self.p, self.t, self.uv, self.mi = [], [], [], []

    def tri(self, pts, uvs, mat):
        k = len(self.p)
        self.p.extend(np.asarray(pts, float) - self.origin)
        self.uv.extend(uvs)
        self.t.append((k, k + 1, k + 2))
        self.mi.append(mat)

    def quad(self, pts, mat, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
        self.tri(pts[:3], uv[:3], mat)
        self.tri([pts[0], pts[2], pts[3]], [uv[0], uv[2], uv[3]], mat)

    def cuboid(self, c, a, b, h0, h1, ha, hb, mat):
        a3, b3 = np.array([a[0], a[1], 0.0]), np.array([b[0], b[1], 0.0])
        c3 = np.array([c[0], c[1], 0.0])
        v = lambda i, j, z: c3 + a3 * ha * i + b3 * hb * j + np.array([0.0, 0.0, z])
        for (i0, j0), (i1, j1) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            self.quad([v(i0, j0, h0), v(i1, j1, h0), v(i1, j1, h1), v(i0, j0, h1)], mat)
        self.quad([v(-1, -1, h1), v(1, -1, h1), v(1, 1, h1), v(-1, 1, h1)], mat)

    def arrays(self):
        return (np.asarray(self.p, np.float64).reshape(-1, 3), np.asarray(self.t, np.int64).reshape(-1, 3),
                np.asarray(self.uv, np.float64).reshape(-1, 2), np.asarray(self.mi, np.int64))


def face_mesh(mesh, centre, f, z_bottom, w, h, disc, rows, material):
    """One face: its disc (a FAN-sided polygon) and plates, front towards f (unit xy, the oncoming traffic), the bare
    metal behind; z_bottom its lowest edge. A front triangle runs anticlockwise seen from f (the kernel's: a quad's
    visible side is its first edge x up), its picture mapped as drawn (v up)."""
    f = np.asarray(f, float)
    r2 = np.array([-f[1], f[0]])            # the viewer's right (looking along -f: -f x up)
    right = np.array([r2[0], r2[1], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    off = np.array([f[0], f[1], 0.0]) * THICK / 2
    base = np.array([centre[0], centre[1], z_bottom])

    def at(u, v, back=False):
        """(3D point, uv) of picture position u across (m from its left), v up from its bottom (m)."""
        p = base + right * (u - w / 2) + up * v + (-off if back else off)
        return p, (u / w, v / h)
    # the disc, centred at (w/2, h - disc/2)
    cu, cv, R = w / 2, h - disc / 2, disc / 2
    ring = [(cu + R * dmath.cos(2 * math.pi * k / FAN), cv + R * dmath.sin(2 * math.pi * k / FAN)) for k in range(FAN)]
    c3, cuv = at(cu, cv)
    cb, _ = at(cu, cv, back=True)
    for k in range(FAN):
        (u0, v0), (u1, v1) = ring[k], ring[(k + 1) % FAN]
        p0, uv0 = at(u0, v0)
        p1, uv1 = at(u1, v1)
        mesh.tri([c3, p0, p1], [cuv, uv0, uv1], material)
        b0, _ = at(u0, v0, back=True)
        b1, _ = at(u1, v1, back=True)
        mesh.tri([cb, b1, b0], [(0, 0), (0, 0), (0, 0)], 0)
    for text, top, pw in rows:
        u0, u1 = (w - pw) / 2, (w + pw) / 2
        v1 = h - top
        v0 = v1 - PLATE_H
        corners = [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)]
        mesh.quad([p for p, _ in corners], material, uv=[uv for _, uv in corners])
        back = [at(u1, v0, True)[0], at(u0, v0, True)[0], at(u0, v1, True)[0], at(u1, v1, True)[0]]
        mesh.quad(back, 0, uv=((0, 0), (0, 0), (0, 0), (0, 0)))


def build(found, z, origin, out_dir, vfs_dir):
    """(mesh, faces) of the posts found (signs()) on ground heights z: dirsigns.build_tile's form (vertices, triangles,
    uv, materials, material per triangle) and {material: vfs path of its picture}."""
    materials, pics, drawn = [METAL], {}, {}
    mesh = _Mesh(origin)
    for (x, y, fx, fy, disc, fcs), zz in zip(found, z):
        made = []
        for kind, figure, plates in fcs:
            key = (kind, figure, tuple(plates), disc)
            if key not in drawn:
                if len(drawn) >= MAX_FACES:
                    continue
                mat, tex, w, h, rows = face(out_dir, vfs_dir, kind, figure, plates, disc)
                drawn[key] = (mat, w, h, rows)
                if mat not in pics:
                    pics[mat] = tex
                    materials.append(mat)
            made.append(drawn[key])
        if not made:
            continue
        zb = float(zz) + MOUNT
        f = np.array([fx, fy], float)
        f /= max(float(dnp.norm(f)), 1e-9)
        c = np.array([x, y], float)
        for mat, w, h, rows in reversed(made):      # (the first face on top)
            face_mesh(mesh, c + f * (THICK / 2 + POST_W / 2 + 0.005), f, zb, w, h, _disc_of(h, rows), rows,
                      materials.index(mat))
            zb += h + GAP
        side = np.array([-f[1], f[0]])
        mesh.cuboid(c, f, side, float(zz) - 0.3, zb - GAP - 0.05, POST_W / 2, POST_W / 2, 0)
    p, t, uv, mi = mesh.arrays()
    if not len(t):
        return None, {}
    return (p, t, uv, materials, mi), pics


def _disc_of(h, rows):
    """A face's disc diameter from its height and plates (layout()'s inverse)."""
    return h - sum(PLATE_H + GAP for _ in rows)


# ---------------------------------------------------------------------------------------------------- the tile

def read(md):
    """The records (md: mapdata.MapData, its cache and fetcher), None while the layers cannot be had (the tile is then
    built without them and not kept)."""
    if md is not None:
        configure(getattr(md, "cache_dir", None), getattr(md, "fetcher", None))
    got = records()
    if got is None:
        import revisit
        revisit.fault(here=True)
    return got


def sign_tile(md, frame, tx, ty, elev, out_dir, vfs_dir, context, decks=(), country=None, left_hand=True):
    """The regulatory signs of road tile (tx, ty): (mesh, faces) in dirsigns.build_tile's form, or (None, {})."""
    if not wanted(country):
        return None, {}
    recs = read(md)
    if not recs:
        return None, {}
    import roads
    box = roads.tile_bbox(tx, ty)
    x0, y0, x1, y1 = box
    la, lo, _ = frame.enu_to_geodetic(np.array([x0 - REACH_M, x1 + REACH_M]),
                                      np.array([y0 - REACH_M, y1 + REACH_M]), np.zeros(2))
    rows = list(context.pavements(frame, np.c_[lo, la], left_hand))
    if not rows:
        return None, {}
    found = signs(frame, recs, box, rows, decks, left_hand)
    if not found:
        return None, {}
    xy = np.array([[s[0], s[1]] for s in found], np.float64)
    la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], np.zeros(len(xy)))
    hz = np.asarray(elev(la, lo), np.float64).ravel()
    z = np.asarray(frame.geodetic_to_enu(la, lo, hz)[2], np.float64).ravel()
    return build(found, z, (x0, y0, 0.0), out_dir, vfs_dir)
