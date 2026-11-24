"""Hong Kong's bus stop signs (glki6, 2026-11-24: "what other regions already found ways of implementing, but for Hong
Kong, in HK style"): Sweden's (sebusstops.py, Mob1K7105 #1417) for Hong Kong's left-hand roads. At every mapped bus
stop a pole at the kerb, on the traffic's left, with a stop flag across the pavement read from both ways: the
words "巴士站" and "BUS STOP" (as the stop's own road marking writes them) under a bus, and under the flag a white plate
with the stop's name in Chinese and in English, the English in capitals (as the franchised operators' flags write it).

Hong Kong has no statutory bus stop sign: the Commissioner for Transport designates a stop (Public Bus Services
Ordinance, Cap. 230, s.3) and the operator puts up a flag of a type the Commissioner approves, each operator its own
colours and logo. These flags carry no operator's colours or logo: one plain black on white flag for every stop.

The stops are the ones Sweden's are (OpenStreetMap's, read by sebusstops.stops): a stop beside the road or the node on
the carriageway where the bus halts. Hong Kong maps a stop's name as "中文 English" (name), or in name:zh / name:en:
label() splits the one or reads the others. The Chinese is drawn in the computer's own Chinese typeface (Windows'
Microsoft JhengHei, the traditional characters' forms, then YaHei, Noto Sans CJK, WenQuanYi): with none found a plate
has its English only. BNE_HK_BUSSTOPS=off leaves them out.

    sign_tile(md, frame, tx, ty, elev, out_dir, vfs_dir, context, decks, country, left_hand) -> (mesh, faces)
"""
import hashlib
import io
import json
import os
import re
from types import SimpleNamespace

import numpy as np

import dirsigns
import fast
import imagery
import roads
import sebusstops
import shelters

COUNTRIES = {"HK"}
ON = os.environ.get("BNE_HK_BUSSTOPS", "").strip().lower() not in ("off", "0", "no", "false")
KERNEL = sebusstops.KERNEL      # dirsign_mesh
FACE_VERSION = 1
PX_PER_M = 400.0
WHITE, BLACK = (255, 255, 255), (16, 16, 16)
FLAG_W_M, FLAG_H_M = 0.45, 0.55     # the stop flag
NAME_ZH_M = 0.085                   # the Chinese name's character height on the plate
NAME_EN_M = 0.05                    # the English name's cap height
PAD_M = 0.035
BORDER_M = 0.008
MIN_W_M, MAX_W_M = FLAG_W_M, 1.3
MOUNT_M = 2.1                       # the plate's bottom over the ground
GAP_M = 0.03                        # between the plate and the flag over it
MAX_NAMES = 30                      # name plates a tile makes at most (a Material per name); the stops past them keep
MAX_NAME_LEN = 40                   # their flag
CJK = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")
CJK_FONTS = ("msjhbd.ttc", "msjh.ttc", "msyhbd.ttc", "msyh.ttc", "NotoSansCJK-Bold.ttc", "NotoSansCJKtc-Bold.otf",
             "NotoSansTC-Bold.otf", "wqy-zenhei.ttc")
EDGE = " -–—/|·,;:"


def wanted(country):
    return ON and str(country or "").upper() in COUNTRIES


def _clean(text):
    s = "" if text is None else str(text).strip()
    if s.lower() in ("", "none", "nan"):
        return ""
    return " ".join(s.split()).strip(EDGE)[:MAX_NAME_LEN].strip(EDGE)


def split_name(name):
    """(Chinese, English) of a name as Hong Kong maps it, "中文 English": the words with a Chinese character in them, and
    the rest."""
    zh, en = [], []
    for word in _clean(name).split():
        (zh if CJK.search(word) else en).append(word)
    return _clean(" ".join(zh)), _clean(" ".join(en))


def label(feats, i):
    """(Chinese, English) of stop i: from its name ("中文 English"), else its name:zh (name:zh-hant ...) and name:en; ""
    for none. The English in capitals (the tags are kept lower-case)."""
    names = feats.attrs.get("name")
    zh, en = split_name(names[i] if names is not None else "")
    tags = shelters._tags(feats, i)
    if not zh:
        for key in ("name:zh-hant", "name:zh_hant", "name:zh-hk", "name:zh"):
            got = _clean(tags.get(key, ""))
            if CJK.search(got):
                zh = got
                break
    if not en:
        en = _clean(tags.get("name:en", ""))
    return zh, en.upper()


_CJK_FONT = None


def cjk_font():
    """The file name of the first of CJK_FONTS on this computer, or "" (none: the plates have their English only)."""
    global _CJK_FONT
    if _CJK_FONT is None:
        from PIL import ImageFont
        _CJK_FONT = ""
        for name in CJK_FONTS:
            try:
                ImageFont.truetype(name, 12)
            except OSError:
                continue
            _CJK_FONT = name
            break
    return _CJK_FONT


def _latin(px):
    from PIL import ImageFont
    return ImageFont.truetype(dirsigns.FONTS[0], px)


def _chinese(px):
    from PIL import ImageFont
    return ImageFont.truetype(cjk_font(), px) if cjk_font() else None


def _bus(draw, x0, y0, w, h):
    """A bus from the side, black, in the box (x0, y0, w, h) px: body, windows, wheels."""
    def r(a, b, c, d, fill):
        draw.rectangle([x0 + a * w, y0 + b * h, x0 + c * w - 1, y0 + d * h - 1], fill=fill)
    draw.rounded_rectangle([x0, y0, x0 + w - 1, y0 + 0.82 * h - 1], radius=max(1, int(0.08 * h)), fill=BLACK)
    for k in range(5):                              # the windows, the door's last
        a = 0.05 + k * 0.185
        r(a, 0.12, a + 0.15, 0.45 if k < 4 else 0.74, WHITE)
    for cx in (0.2, 0.8):                           # the wheels
        rr = 0.17 * h
        draw.ellipse([x0 + cx * w - rr, y0 + 0.82 * h - rr, x0 + cx * w + rr, y0 + 0.82 * h + rr], fill=BLACK)
        rr *= 0.4
        draw.ellipse([x0 + cx * w - rr, y0 + 0.82 * h - rr, x0 + cx * w + rr, y0 + 0.82 * h + rr], fill=WHITE)


def _text_w(draw, text, font):
    a = draw.textbbox((0, 0), text, font=font)
    return a[2] - a[0]


def _centred(draw, text, font, cx, top):
    a = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (a[0] + a[2]) / 2.0, top - a[1]), text, font=font, fill=BLACK)


def flag_picture():
    """The stop flag as (h, w, 3) uint8: a white field with a black edge line, the bus, "巴士站" and "BUS STOP"."""
    from PIL import Image, ImageDraw
    w, h = int(round(FLAG_W_M * PX_PER_M)), int(round(FLAG_H_M * PX_PER_M))
    im = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(im)
    b = int(round(BORDER_M * PX_PER_M))
    d.rectangle([b, b, w - 1 - b, h - 1 - b], outline=BLACK, width=b)
    _bus(d, 0.18 * w, 0.1 * h, 0.64 * w, 0.32 * h)
    zh = _chinese(int(0.16 * h))
    en = _latin(int(0.11 * h))
    if zh is not None:
        _centred(d, "巴士站", zh, w / 2.0, 0.5 * h)
        _centred(d, "BUS STOP", en, w / 2.0, 0.74 * h)
    else:
        _centred(d, "BUS STOP", en, w / 2.0, 0.62 * h)
    return np.asarray(im, np.uint8)


def name_picture(zh, en):
    """A stop's name plate as (h, w, 3) uint8, its (w, h) in m: white with a black edge line, the Chinese name over the
    English, each made narrower to fit MAX_W_M. None with neither."""
    from PIL import Image, ImageDraw
    zf = _chinese(int(round(NAME_ZH_M * PX_PER_M * 1.15))) if zh else None
    lines = ([(zh, zf)] if zf is not None else []) + ([(en, _latin(int(round(NAME_EN_M * PX_PER_M * 1.4))))] if en else [])
    if not lines:
        return None
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    pad = PAD_M * PX_PER_M
    gap = 0.4 * pad
    heights = [NAME_ZH_M * PX_PER_M * 1.15 if f is zf else NAME_EN_M * PX_PER_M * 1.4 for _, f in lines]
    widths = [_text_w(probe, t, f) for t, f in lines]
    max_px = MAX_W_M * PX_PER_M - 2 * pad
    w = int(round(min(MAX_W_M * PX_PER_M, max(MIN_W_M * PX_PER_M, max(widths) + 2 * pad))))
    h = int(round(2 * pad + sum(heights) + gap * (len(lines) - 1)))
    im = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(im)
    top = pad
    for (text, font), lh, tw in zip(lines, heights, widths):
        if tw > max_px:                             # (a long name: drawn wide, then pressed to the plate)
            line = Image.new("RGB", (int(tw) + 4, int(lh * 1.3) + 4), WHITE)
            _centred(ImageDraw.Draw(line), text, font, line.width / 2.0, 2)
            line = line.resize((int(max_px), line.height), Image.LANCZOS)
            im.paste(line, (int((w - line.width) / 2), int(top)))
        else:
            _centred(d, text, font, w / 2.0, top)
        top += lh + gap
    b = int(round(BORDER_M * PX_PER_M))
    d.rectangle([b, b, w - 1 - b, h - 1 - b], outline=BLACK, width=b)
    return np.asarray(im, np.uint8), w / PX_PER_M, h / PX_PER_M


def _png(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _hash(*parts):
    flat = [FACE_VERSION, WHITE, BLACK, PX_PER_M, cjk_font(), *parts]
    return hashlib.sha1(json.dumps(flat, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def _flag_face(out_dir, vfs_dir):
    """(material, vfs path of its picture): one for every stop."""
    material = f"bne_hkstop_flag_{_hash(FLAG_W_M, FLAG_H_M)}"
    path = os.path.join(out_dir, material + ".color.dds")
    if not os.path.exists(path):
        imagery.write_picture_dds(path, _png(flag_picture()))     # (a DDS, never a PNG the engine cooks)
    return material, f"{vfs_dir}/{material}.color.dds"


def _name_face(out_dir, vfs_dir, zh, en):
    """(material, vfs path, width m, height m) of a stop's name plate (drawn once, named by its hash), or None."""
    got = name_picture(zh, en)
    if got is None:
        return None
    rgb, w_m, h_m = got
    material = f"bne_hkstop_{_hash(zh, en, NAME_ZH_M, NAME_EN_M, MAX_W_M)}"
    path = os.path.join(out_dir, material + ".color.dds")
    if not os.path.exists(path):
        imagery.write_picture_dds(path, _png(rgb))
    return material, f"{vfs_dir}/{material}.color.dds", w_m, h_m


def sign_tile(md, frame, tx, ty, elev, out_dir, vfs_dir, context, decks=(), country=None, left_hand=True):
    """The stop signs of road tile (tx, ty): (mesh, faces) in dirsigns.build_tile's form (vertices, triangles, uv,
    materials, material per triangle), or (None, {})."""
    if not wanted(country):
        return None, {}
    box = roads.tile_bbox(tx, ty)
    found, platforms = sebusstops.stops(md, frame, box, label)
    if not found:
        return None, {}
    import busstationkeep               # (a bus station has its own plates)
    keep = busstationkeep.keepout(md, frame, box)
    if keep:
        inside = keep.contains(np.array([[f[1], f[2]] for f in found], np.float64))
        found = [f for f, drop in zip(found, inside) if not drop]
        if not found:
            return None, {}
    pts = np.array([[f[1], f[2]] for f in found], np.float64)
    la, lo, _ = frame.enu_to_geodetic(pts[:, 0], pts[:, 1], 0.0)
    near = [(r[0], SimpleNamespace(width=float(r[3]), oneway=bool(getattr(r[1], "oneway", False)),
                                   dir=getattr(r[1], "dir", 0)), r[2])
            for r in context.pavements(frame, np.c_[lo, la], left_hand)]
    poles = sebusstops.rows_of(found, near, decks, platforms, left_hand)
    if not poles:
        return None, {}
    fast.require(KERNEL, "bus stop signs")
    xy = np.array([[p[0], p[1]] for p in poles], np.float64)
    la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
    hz = np.asarray(elev(la, lo), np.float64).ravel()
    z = np.asarray(frame.geodetic_to_enu(la, lo, hz)[2], np.float64).ravel()
    flag, flag_tex = _flag_face(out_dir, vfs_dir)
    materials, faces, rows, named = [dirsigns.METAL, flag], {flag: flag_tex}, [], set()
    side = -1.0 if left_hand else 1.0
    for (x, y, gx, gy, zh, en), zz in zip(poles, z):
        top = MOUNT_M
        if (zh or en) and ((zh, en) in named or len(named) < MAX_NAMES):
            got = _name_face(out_dir, vfs_dir, zh, en)
            if got is not None:
                named.add((zh, en))
                mat, tex, w, h = got
                if mat not in faces:
                    faces[mat] = tex
                    materials.append(mat)
                _both(rows, x, y, zz, gx, gy, w, h, materials.index(mat), top, side)
                top += h + GAP_M
        _both(rows, x, y, zz, gx, gy, FLAG_W_M, FLAG_H_M, 1, top, side)
    p, t, uv, mi = fast._ef.dirsign_mesh(np.ascontiguousarray(rows, np.float64), [float(box[0]), float(box[1]), 0.0])
    if not len(t):
        return None, {}
    return (p, t, uv, materials, mi), faces


def _both(rows, x, y, z, gx, gy, w, h, material, mount, side):
    """A board on the stand at (x, y) and the same on its back: the first met by the traffic going (gx, gy), reaching
    away from the road (its left: side -1 in left-hand traffic), the second from the other way, over the same span."""
    rows.append([x, y, z, z, gx, gy, w, h, material, mount, sebusstops.MODE_STAND, side])
    rows.append([x, y, z, z, -gx, -gy, w, h, material, mount, sebusstops.MODE_STAND, -side])
