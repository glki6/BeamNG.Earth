"""build_hk_beacon.py: Hong Kong's zebra crossing beacon (mod/.../art/shapes/beamngEarth/bne_hk_beacon.dae).

The Transport Department's road safety leaflet shows a zebra crossing's beacon as "a yellow globe on a black and
white post": one at each kerb end of the crossing (server/hkzebras.beacons). Z up, origin at the post's foot.

The globe glows steadily (a forest row is static: no flashing). The post's bands and the globe's size are a
common Belisha beacon's, not measured in Hong Kong (hkzebras' docstring).

    python tools/build_hk_beacon.py
"""
import json
import os
import sys
import uuid

try:        # (first: numpy's AVX-512 kernels held back before anything imports numpy, tools/npyfirst.py)
    import npyfirst  # noqa: E402,F401
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_rail_crossing_props import Mesh, OUT, write_dae  # noqa: E402

NAME = "bne_hk_beacon"
POST_R = 0.045          # m: the post's radius
POST_H = 2.60           # m: the post's top (the globe sits on it)
BAND = 0.30             # m: the post's black and white bands
GLOBE_R = 0.165         # m: the globe's radius (about 330 mm across)
GLOBE_Z = POST_H + 0.02 + GLOBE_R    # m: the globe's centre; its top near 2.95 m (hkzebras.BEACON_H)
GLOBE_BANDS = 6         # the globe's latitude rings

COLOURS = {
    "bne_hk_beacon_black": ((0.03, 0.03, 0.03), 0.0, 0.55, None),
    "bne_hk_beacon_white": ((0.86, 0.86, 0.84), 0.0, 0.5, None),
    "bne_hk_beacon_globe": ((0.95, 0.62, 0.08), 0.0, 0.3, (0.9, 0.55, 0.05)),
}


def _globe(m, segs, bands):
    """A UV sphere, flat-shaded."""
    import math
    rings = [(-math.pi / 2 + math.pi * i / bands) for i in range(bands + 1)]
    for i in range(bands):
        a0, a1 = rings[i], rings[i + 1]
        for j in range(segs):
            b0, b1 = 2 * math.pi * j / segs, 2 * math.pi * (j + 1) / segs
            p = lambda a, b: (GLOBE_R * math.cos(a) * math.cos(b), GLOBE_R * math.cos(a) * math.sin(b),
                              GLOBE_Z + GLOBE_R * math.sin(a))
            q = [p(a0, b0), p(a0, b1), p(a1, b1), p(a1, b0)]
            if i == 0:
                q = [q[0], q[2], q[3]]          # (the bottom pole: a triangle)
            elif i == bands - 1:
                q = [q[0], q[1], q[2]]          # (the top pole)
            m.face("bne_hk_beacon_globe", q)


def beacon(segs, bands):
    m = Mesh(segs)
    z, k = 0.0, 0
    while z < POST_H - 1e-9:
        top = min(POST_H, z + BAND)
        m.cylinder("bne_hk_beacon_black" if k % 2 == 0 else "bne_hk_beacon_white", 0.0, 0.0, z, top, POST_R,
                   caps=False)
        z, k = top, k + 1
    m.cylinder("bne_hk_beacon_black", 0.0, 0.0, POST_H, POST_H + 0.02, POST_R * 1.6)     # the globe's collar
    _globe(m, segs, bands)
    return m


def collision():
    m = Mesh(4)
    m.box("bne_hk_beacon_black", (0.0, 0.0, POST_H / 2), (0.05, 0.05, POST_H / 2))
    return m


def materials():
    out = {}
    pid = lambda n: str(uuid.uuid5(uuid.NAMESPACE_URL, "beamng-earth/" + n))
    for k, (col, met, rough, em) in COLOURS.items():
        stage = {"baseColorFactor": [*col, 1.0], "metallicFactor": met, "roughnessFactor": rough}
        if em is not None:
            stage["emissiveFactor"] = list(em)
        out[k] = {"name": k, "mapTo": k, "class": "Material", "version": 1.5, "persistentId": pid(k),
                  "Stages": [stage, {}, {}, {}]}
    with open(os.path.join(OUT, "hk_beacon.materials.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")


def main():
    materials()
    path = os.path.join(OUT, NAME + ".dae")
    lods = [(64, beacon(16, GLOBE_BANDS)), (8, beacon(6, 3))]
    write_dae(path, NAME, lods, collision())
    print(path, os.path.getsize(path), "bytes", sum(len(t) for t in lods[0][1].tri.values()), "triangles")


if __name__ == "__main__":
    main()
