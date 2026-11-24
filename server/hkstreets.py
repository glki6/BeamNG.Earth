"""Hong Kong's street name plates (the user, 2026-11-24: what other regions have, in Hong Kong's style; France's are
frstreets.py): at the junctions of differently named streets, a plate beside each named arm, on two posts at the back
of the kerb, parallel to the street it names, so the traffic crossing that street reads it from either side.

The plate is the Highways Department's standard (HyD standard drawing H 2206D, "Street name plate with building
numbers", read 2026-11-24): 324 mm high, 750 mm to 2000 mm long in 50 mm steps, 25 mm corner radius, white with a black
border and black lettering; the English name in Transport (70 mm capitals, mixed case) over the Chinese name in a bold
Hei face (98 mm), both centred. Here English is AG Transport Heavy (the dark-on-light Transport, TPDM Vol 3 1.3.1.7)
and Chinese Noto Sans HK Bold (OFL 1.1; Hong Kong's glyph forms), cut down by tools/build_hk_street_font.py. The
drawing's building numbers and arrows are left out: the map has no number ranges.

Names come from the map (Overture's names.primary of the road: OpenStreetMap's Hong Kong convention "彌敦道 Nathan
Road"), split into its Chinese and English parts. Plate positions are scenery inferred from the junction, not mapped.

The shared sign letter atlas (signmesh) has no Chinese, so each plate is a picture drawn here with Pillow, one per
name (named by the hash of what it shows, so a tile drawn again reuses it), as Sweden's stop name plates are
(sebusstops.py); the plates join the sign tile's direction-board mesh (dirsigns.lua makes their Materials). A name
with a character the cut font lacks gets an English-only plate.

    sign_tile(md, frame, tx, ty, elev, out_dir, vfs_dir, context, decks, country) -> (mesh, faces) or (None, {})

Tile independence: a plate belongs to the tile its post stands in; its junction and the roads it must clear are read
from the sign context round the tile (signcontext.MARGIN_M), the same from every tile.

BNE_HK_STREETS=off turns them off.
"""
from functools import lru_cache
import hashlib
import io
import json
import os
import re

import numpy as np

import dnp

COUNTRIES = {"HK"}
FACE_VERSION = 1
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fonts")
ZH_FONT = os.path.join(FONT_DIR, "NotoSansHK-Bold-streets.otf")
ZH_CHARS = os.path.join(FONT_DIR, "NotoSansHK-Bold-streets.chars.txt")
EN_FONT = os.path.join(FONT_DIR, "AG-Transport-Heavy.ttf")
EN_FALLBACK = os.path.join(FONT_DIR, "Overpass-Bold.ttf")    # (a character Transport lacks: é, &, ...)
METAL = "bne_dir_metal"                                         # (dirsigns.METAL: the boards' bare metal)

# HyD H 2206D, in metres
PLATE_H = 0.324
MIN_W, MAX_W, STEP_W = 0.75, 2.0, 0.05
RADIUS = 0.025
BORDER = 0.012
CAP_EN = 0.070                  # English capitals
HEIGHT_ZH = 0.098               # Chinese characters
GAP = 0.034                     # English baseline to the top of the Chinese
SIDE = 0.06                     # lettering to the plate's ends
PX_PER_M = 800.0                # picture pixels a metre (the Chinese 78 px high)
WHITE, BLACK = (255, 255, 255), (12, 12, 12)

CLASSES = {"primary", "secondary", "tertiary", "residential", "unclassified", "living_street"}
MOUNT = 2.1                     # the plate's bottom over the footway
POST_W = 0.05                   # square galvanised posts
POST_GAP = 0.02                 # post to the plate's end
THICK = 0.006                   # the plate (two faces 3 mm apart from its middle)
KERB = (0.9, 1.3, 1.8)          # posts' line past the carriageway's edge, tried in order
ALONG = (9.0, 12.0, 16.0, 21.0) # plate centre from the junction along the arm, tried in order
CLEAR = 0.35                    # plate and posts from any other carriageway's edge
NODE_M = 1.5                    # line ends this near are one junction
SAME_M = 30.0                   # one plate a name a side within this
DECK_M = 1.5
MAX_NAMES = 30                  # pictures (Materials) a tile makes at most
MAX_PLATES = 60


def wanted(country):
    return (str(country or "").upper() in COUNTRIES
            and os.environ.get("BNE_HK_STREETS", "on").strip().lower() not in {"0", "off", "false", "no"})


# ---------------------------------------------------------------------------------------------------- names

_HAN = re.compile(r"[㐀-鿿豈-﫿\U00020000-\U0002ffff]")
_SPLIT = re.compile(r"^(?P<zh>.*?[㐀-鿿豈-﫿\U00020000-\U0002ffff][^A-Za-z]*?)\s+(?P<en>[A-Za-z].*)$")
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "’": "'", "‘": "'"})


def _clean(text):
    text = "".join(c if c.isprintable() else " " for c in str(text or ""))
    return " ".join(text.split())


def split_name(name):
    """(english, chinese) of a mapped Hong Kong road name ("彌敦道 Nathan Road" -> ("Nathan Road", "彌敦道")); either may
    be "" (a name in one language only; a bare route number such as "D3" is no street name: ("", ""))."""
    name = _clean(name)
    if not name or name in {"None", "null"}:
        return "", ""
    m = _SPLIT.match(name)
    if m:
        return m.group("en").translate(_DASHES).strip(), m.group("zh").strip()
    if _HAN.search(name):
        return "", name
    if not re.search(r"[a-z]", name) and not re.search(r"\s", name):
        return "", ""           # (a reference, not a name)
    return name.translate(_DASHES), ""


@lru_cache(maxsize=1)
def zh_chars():
    with open(ZH_CHARS, encoding="utf-8") as f:
        return frozenset(f.read().strip())


def covered(text):
    """Whether the cut Chinese font has every character of text (spaces aside)."""
    have = zh_chars()
    return all(c.isspace() or c in have for c in text)


# ---------------------------------------------------------------------------------------------------- the picture

@lru_cache(maxsize=None)
def _font(path, px):
    from PIL import ImageFont
    return ImageFont.truetype(path, px)


@lru_cache(maxsize=None)
def _en_has(path):
    from fontTools.ttLib import TTFont
    return frozenset(chr(c) for c in TTFont(path).getBestCmap())


def _en_face(text):
    """The English typeface's path: Transport, or the fallback when Transport lacks a character of text."""
    try:
        have = _en_has(EN_FONT)
    except ImportError:          # (no fontTools: Transport's own set, read 2026-11-24)
        have = frozenset("%&'(),-./0123456789:?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
    return EN_FONT if all(c == " " or c in have for c in text) else EN_FALLBACK


def _cap_px(path, size):
    l, t, r, b = _font(path, size).getbbox("H", anchor="ls")
    return b - t


def _line(text, path, height_px, measure_char):
    """(image L mode with the ink of text, its baseline's row in it, measure_char's ink top over the baseline px): the
    font sized so measure_char's ink is height_px high (H: the capitals; 國: the Chinese)."""
    from PIL import Image, ImageDraw
    ref = 200
    l, t, r, b = _font(path, ref).getbbox(measure_char, anchor="ls")
    size = max(8, int(round(ref * height_px / max(b - t, 1))))
    font = _font(path, size)
    top = -font.getbbox(measure_char, anchor="ls")[1]
    l, t, r, b = font.getbbox(text, anchor="ls")
    im = Image.new("L", (max(1, r - l), max(1, b - t)))
    ImageDraw.Draw(im).text((-l, -t), text, font=font, fill=255, anchor="ls")
    return im, -t, top


@lru_cache(maxsize=4096)
def plate_width(en, zh):
    """The plate's length (m) for its two lines: their ink and SIDE either end, in STEP_W steps, MIN_W..MAX_W."""
    need = 0.0
    if en:
        need = max(need, _line(en, _en_face(en), CAP_EN * PX_PER_M, "H")[0].width / PX_PER_M)
    if zh:
        need = max(need, _line(zh, ZH_FONT, HEIGHT_ZH * PX_PER_M, "國")[0].width / PX_PER_M)
    w = np.ceil(round((need + 2 * SIDE) / STEP_W, 6)) * STEP_W
    return float(min(MAX_W, max(MIN_W, round(float(w), 3))))


def picture(en, zh):
    """(PNG bytes, width m, height m) of the plate showing en over zh (either may be "")."""
    from PIL import Image, ImageDraw
    w_m = plate_width(en, zh)
    W, H = int(round(w_m * PX_PER_M)), int(round(PLATE_H * PX_PER_M))
    # the picture's edge is the plate's (a rectangle: its corners outside the radius black, as the border)
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)
    s = PX_PER_M
    b = int(round(BORDER * s))
    d.rounded_rectangle((b, b, W - 1 - b, H - 1 - b), radius=int((RADIUS - BORDER) * s), fill=WHITE)
    room = W - 2 * int(round(SIDE * s))
    lines = []
    if en:
        lines.append(_line(en, _en_face(en), CAP_EN * s, "H") + (CAP_EN * s,))
    if zh:
        lines.append(_line(zh, ZH_FONT, HEIGHT_ZH * s, "國") + (HEIGHT_ZH * s,))
    # vertical: the capitals and the Chinese with GAP between, the pair centred on the plate
    total = sum(h for *_, h in lines) + GAP * s * (len(lines) - 1)
    top = (H - total) / 2
    for im, base, ink_top, h in lines:
        if im.width > room:      # a long name: condensed to fit, never cut
            im = im.resize((room, im.height), Image.LANCZOS)
        x = (W - im.width) // 2
        # set on its baseline: the measuring character's ink top (a capital's, a Chinese character's) at top
        y = int(round(top + ink_top - base))
        img.paste(Image.new("RGB", im.size, BLACK), (x, y), im)
        top += h + GAP * s
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue(), w_m, PLATE_H


def face(out_dir, vfs_dir, en, zh):
    """(material, vfs path, width m, height m) of the plate's picture: drawn once (named by its hash)."""
    h = hashlib.sha1(json.dumps([FACE_VERSION, en, zh, PX_PER_M, PLATE_H, BORDER], ensure_ascii=False)
                     .encode("utf-8")).hexdigest()[:12]
    material = f"bne_hkstreet_{h}"
    path = os.path.join(out_dir, material + ".color.dds")
    png, w, hh = picture(en, zh)
    if not os.path.exists(path):
        import imagery
        imagery.write_picture_dds(path, png)    # (a DDS, never a PNG the engine cooks)
    return material, f"{vfs_dir}/{material}.color.dds", w, hh


# ---------------------------------------------------------------------------------------------------- placement

def _arclen(line):
    seg = dnp.norm(np.diff(line[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _at(line, s):
    """(point xy, unit tangent) at arc length s along line (clamped)."""
    cum = _arclen(line)
    s = min(max(s, 0.0), float(cum[-1]))
    i = int(min(np.searchsorted(cum, s, side="right") - 1, len(line) - 2))
    seg = line[i + 1, :2] - line[i, :2]
    n = float(dnp.norm(seg))
    u = seg / max(n, 1e-9)
    return line[i, :2] + u * (s - float(cum[i])), u


def _seg_dist(p, line):
    """Distance from point p (2,) to polyline line (k, 2)."""
    a, b = line[:-1], line[1:]
    ab = b - a
    l2 = np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / l2, 0.0, 1.0)
    return float(dnp.norm(a + t[:, None] * ab - p, axis=1).min())


def junctions(rows, names):
    """[(centre xy, [(row index, end 0|1)])] of the line ends where two or more differently named roads meet, in a
    stable order. rows: (fid, spec, line, width); names: fid -> (english, chinese)."""
    ends = []
    for k, (fid, spec, line, width) in enumerate(rows):
        if len(line) < 2:
            continue
        ends.append((np.asarray(line[0, :2], float), k, 0))
        ends.append((np.asarray(line[-1, :2], float), k, 1))
    cell = {}
    for j, (p, _, _) in enumerate(ends):
        cell.setdefault((int(np.floor(p[0] / NODE_M)), int(np.floor(p[1] / NODE_M))), []).append(j)
    parent = list(range(len(ends)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (cx, cy), members in cell.items():
        near = [m for dx in (-1, 0, 1) for dy in (-1, 0, 1) for m in cell.get((cx + dx, cy + dy), ())]
        for a in members:
            for b in near:
                if b > a and float(dnp.norm(ends[a][0] - ends[b][0])) < NODE_M:
                    parent[find(b)] = find(a)
    groups = {}
    for j in range(len(ends)):
        groups.setdefault(find(j), []).append(j)
    out = []
    for members in groups.values():
        arms = sorted({(ends[j][1], ends[j][2]) for j in members}, key=lambda a: (str(rows[a[0]][0]), a[1]))
        keys = {_key(names.get(rows[k][0], ("", ""))) for k, _ in arms} - {""}
        if len(arms) < 2 or len(keys) < 2:
            continue
        centre = np.mean([ends[j][0] for j in members], axis=0)
        out.append((centre, arms))
    out.sort(key=lambda j: (round(float(j[0][0]), 1), round(float(j[0][1]), 1)))
    return out


def _key(name):
    en, zh = name
    return (en or zh).casefold()


def plates(rows, names, box, decks=(), left_hand=True):
    """[(x, y, along x, along y, english, chinese)] of the plates whose posts stand in box (x0, y0, x1, y1): a plate's
    centre at the back of the kerb of a named arm of each junction, its face along the arm. Each candidate spot is
    tried in a fixed order (ALONG, then KERB, the left side first in left-hand traffic) and kept when the plate and
    its posts clear every carriageway near it, no bridge deck is under it and no plate of the same name stands on
    the same side within SAME_M. Every plate is chosen as if there were no tile, then kept by its post's tile: the
    same from every tile."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return [p for p in _all_plates(rows, names, decks, left_hand) if x0 <= p[0] < x1 and y0 <= p[1] < y1]


def _all_plates(rows, names, decks, left_hand):
    from exitsigns import _on_deck
    lines = [(float(width) / 2, np.asarray(line, float)[:, :2]) for _, _, line, width in rows]
    out = []
    for centre, arms in junctions(rows, names):
        for k, end in arms:
            fid, spec, line, width = rows[k]
            en, zh = names.get(fid, ("", ""))
            if not (en or zh) or spec.cls not in CLASSES or spec.link:
                continue
            line = np.asarray(line, float)
            line = line if end == 0 else line[::-1]
            spot = _spot(line, float(width), plate_width(en, zh), lines, decks, left_hand, _on_deck)
            if spot is None:
                continue
            q, u, normal, side = spot
            if any(o[4:] == (en, zh) and float(dnp.norm(np.asarray(o[:2]) - q)) < SAME_M
                   and ((o[0] - q[0]) * normal[0] + (o[1] - q[1]) * normal[1]) * side > -1.0 for o in out):
                continue                # (its other carriageway's plate, or the next arm's)
            out.append((float(q[0]), float(q[1]), float(u[0]), float(u[1]), en, zh))
            if len(out) >= MAX_PLATES:
                break
    return out


def _spot(line, width, w, lines, decks, left_hand, on_deck):
    """(post line centre, along, normal, side) of the first clear spot beside an arm (line from its junction), or
    None."""
    length = float(_arclen(line)[-1])
    for s in ALONG:
        if s - w / 2 < width / 2 + 2.0 or s + w / 2 > length:
            continue
        p, u = _at(line, s)
        normal = np.array([-u[1], u[0]])
        for off in KERB:
            for side in ((1.0, -1.0) if left_hand else (-1.0, 1.0)):
                q = p + normal * side * (width / 2 + off)
                ends = [q + u * (w / 2 + POST_GAP + POST_W), q - u * (w / 2 + POST_GAP + POST_W)]
                if any(_seg_dist(e, ln) < hw + CLEAR for hw, ln in lines for e in (q, *ends)):
                    continue
                if on_deck(q, decks, DECK_M):
                    continue
                return q, u, normal, side
    return None


# ---------------------------------------------------------------------------------------------------- the mesh

class _Mesh:
    def __init__(self, origin):
        self.origin = np.asarray(origin, float)
        self.p, self.t, self.uv, self.mi = [], [], [], []

    def quad(self, pts, mat, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
        k = len(self.p)
        self.p.extend(np.asarray(pts, float) - self.origin)
        self.uv.extend(uv)
        self.t.extend([(k, k + 1, k + 2), (k, k + 2, k + 3)])
        self.mi.extend([mat, mat])

    def cuboid(self, c, a, b, h0, h1, ha, hb, mat):
        """A box centred at c (xy) with half extents ha along a, hb along b (unit xy vectors), z h0..h1."""
        a3, b3 = np.array([a[0], a[1], 0.0]), np.array([b[0], b[1], 0.0])
        c3 = np.array([c[0], c[1], 0.0])
        v = lambda i, j, z: c3 + a3 * ha * i + b3 * hb * j + np.array([0.0, 0.0, z])
        for (i0, j0), (i1, j1) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            self.quad([v(i0, j0, h0), v(i1, j1, h0), v(i1, j1, h1), v(i0, j0, h1)], mat)
        self.quad([v(-1, -1, h1), v(1, -1, h1), v(1, 1, h1), v(-1, 1, h1)], mat)

    def arrays(self):
        return (np.asarray(self.p, np.float64).reshape(-1, 3), np.asarray(self.t, np.int64).reshape(-1, 3),
                np.asarray(self.uv, np.float64).reshape(-1, 2), np.asarray(self.mi, np.int64))


def plate_mesh(mesh, x, y, z, ux, uy, w, h, material):
    """A two-faced plate at (x, y), its face along (ux, uy), bottom MOUNT over z, on two posts just past its ends."""
    u = np.array([ux, uy], float)
    u /= max(float(dnp.norm(u)), 1e-9)
    n = np.array([-u[1], u[0]])
    c = np.array([x, y], float)
    lo, hi = z + MOUNT, z + MOUNT + h
    u3, n3 = np.array([u[0], u[1], 0.0]), np.array([n[0], n[1], 0.0])
    base = np.array([x, y, 0.0])

    def corner(i, side, zz):
        return base + u3 * (w / 2) * i + n3 * side * THICK / 2 + np.array([0.0, 0.0, zz])
    # A quad's visible side is (its first edge) x up, its picture read left to right along that edge (the kernel's
    # dirsign_mesh): the face on the -n side runs along u, the one on the +n side along -u, so both read right.
    mesh.quad([corner(-1, -1, lo), corner(1, -1, lo), corner(1, -1, hi), corner(-1, -1, hi)], material)
    mesh.quad([corner(1, 1, lo), corner(-1, 1, lo), corner(-1, 1, hi), corner(1, 1, hi)], material)
    # the plate's edge, and its posts
    mesh.cuboid(c, u, n, lo, hi, w / 2, THICK / 2 * 0.9, 0)
    for i in (-1.0, 1.0):
        mesh.cuboid(c + u * i * (w / 2 + POST_GAP + POST_W / 2), u, n, z - 0.3, hi + 0.03, POST_W / 2, POST_W / 2, 0)


# ---------------------------------------------------------------------------------------------------- the tile

def _names(context, frame, box):
    """fid (signroads.feature_id) -> (english, chinese) of the roads in the sign context round box."""
    import mapcells
    import signcontext
    import signroads
    x0, y0, x1, y1 = box
    m = signcontext.MARGIN_M
    la, lo, _ = frame.enu_to_geodetic(np.array([x0 - m, x1 + m]), np.array([y0 - m, y1 + m]), np.zeros(2))
    out = {}
    for key in mapcells.cells_for_bbox(float(la.min()), float(lo.min()), float(la.max()), float(lo.max())):
        feats = context.cell(key)
        if feats is None or not len(feats) or "name" not in feats.attrs:
            continue
        col = feats.attrs["name"]
        for i in range(len(feats)):
            fid = signroads.feature_id(feats, i)
            if fid not in out:
                out[fid] = split_name(col[i])
    return out


def _in_hk(frame, pts):
    """Whether each plate stands in Hong Kong by the bundled borders (all True where they are not at hand)."""
    try:
        import countrybounds
        index = countrybounds.bundled()
        la, lo, _ = frame.enu_to_geodetic(pts[:, 0], pts[:, 1], np.zeros(len(pts)))
        codes = index.codes_at([(float(a), float(b)) for a, b in zip(la, lo)])
    except Exception:
        return [True] * len(pts)
    return [c in (None, "", "HK") for c in codes]


def sign_tile(md, frame, tx, ty, elev, out_dir, vfs_dir, context, decks=(), country=None, left_hand=True):
    """The street name plates of road tile (tx, ty): (mesh, faces) in dirsigns.build_tile's form (vertices,
    triangles, uv, materials, material per triangle), or (None, {})."""
    if not wanted(country):
        return None, {}
    import roads
    box = roads.tile_bbox(tx, ty)
    x0, y0, x1, y1 = box
    la, lo, _ = frame.enu_to_geodetic(np.array([x0, x1]), np.array([y0, y1]), np.zeros(2))
    rows = list(context.pavements(frame, np.c_[lo, la], left_hand))
    if not rows:
        return None, {}
    names = _names(context, frame, box)
    found = plates(rows, names, box, decks, left_hand)
    if not found:
        return None, {}
    xy = np.array([[p[0], p[1]] for p in found], np.float64)
    keep = _in_hk(frame, xy)
    found = [p for p, k in zip(found, keep) if k]
    if not found:
        return None, {}
    xy = np.array([[p[0], p[1]] for p in found], np.float64)
    la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], np.zeros(len(xy)))
    hz = np.asarray(elev(la, lo), np.float64).ravel()
    z = np.asarray(frame.geodetic_to_enu(la, lo, hz)[2], np.float64).ravel()
    return build(found, z, (x0, y0, 0.0), out_dir, vfs_dir)


def build(found, z, origin, out_dir, vfs_dir):
    """(mesh, faces) of the plates found (plates()) standing on ground heights z."""
    materials, faces, drawn = [METAL], {}, {}
    mesh = _Mesh(origin)
    for (x, y, ux, uy, en, zh), zz in zip(found, z):
        if zh and not covered(zh):
            zh = ""             # (a character the cut font lacks: English only)
        if not (en or zh):
            continue
        if (en, zh) not in drawn:
            if len(drawn) >= MAX_NAMES:
                continue
            mat, tex, w, h = face(out_dir, vfs_dir, en, zh)
            drawn[(en, zh)] = (mat, w, h)
            if mat not in faces:
                faces[mat] = tex
                materials.append(mat)
        mat, w, h = drawn[(en, zh)]
        plate_mesh(mesh, x, y, float(zz), ux, uy, w, h, materials.index(mat))
    p, t, uv, mi = mesh.arrays()
    if not len(t):
        return None, {}
    return (p, t, uv, materials, mi), faces
