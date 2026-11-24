"""Hong Kong street name plates' Chinese typeface (server/hkstreets.py): Noto Sans HK Bold cut down to the characters
Hong Kong's street names use, so the server ships a small font file and draws every plate's picture with it.

A developer tool, run with a Python that has fontTools:

    python3 tools/build_hk_street_font.py /path/to/NotoSansHK-Bold.otf

The full font (SIL Open Font License 1.1) is
https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/HK/NotoSansHK-Bold.otf (the HK subset of Noto Sans CJK:
Hong Kong's own glyph forms, HKSCS characters such as 埗 in 深水埗 included). Kept: Big5's 5,401 frequent characters,
every character of tools/fonts/hk_streets/street_chars.txt (the Chinese characters of the road names in Hong Kong's
box of Overture's 2026-11-18.0 segments: 113.82-114.45 E, 22.15-22.57 N), CJK punctuation and ASCII. A character the
cut font lacks leaves its plate English only (hkstreets.covered).

Writes server/data/fonts/NotoSansHK-Bold-streets.otf, the characters it has (NotoSansHK-Bold-streets.chars.txt: what
hkstreets.covered reads, so the server needs no fontTools) and its licence beside it (NotoSansHK-OFL.txt).
"""
from pathlib import Path
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
CHARS = ROOT / "tools/fonts/hk_streets/street_chars.txt"
OUT = ROOT / "server/data/fonts/NotoSansHK-Bold-streets.otf"
LICENSE = ROOT / "tools/fonts/hk_streets/OFL.txt"


def big5_frequent():
    """Big5's level-1 characters (lead bytes A4-C6): the 5,401 most used in traditional Chinese text."""
    out = set()
    for lead in range(0xA4, 0xC7):
        for trail in (*range(0x40, 0x7F), *range(0xA1, 0xFF)):
            try:
                ch = bytes([lead, trail]).decode("big5")
            except UnicodeDecodeError:
                continue
            if "一" <= ch <= "鿿":
                out.add(ch)
    return out


def wanted():
    chars = big5_frequent() | set(CHARS.read_text(encoding="utf-8").strip())
    chars |= {chr(c) for c in range(0x20, 0x7F)}
    chars |= set("　、。・「」『』（）－–—～：；，！？．")
    return chars


def build(source):
    cmap = TTFont(source).getBestCmap()
    keep = sorted(ord(c) for c in wanted() if ord(c) in cmap)
    opts = subset.Options()
    opts.layout_features = ["kern", "palt", "vert"]
    opts.name_IDs = ["*"]
    opts.name_languages = ["*"]
    opts.notdef_outline = True
    opts.hinting = False
    opts.desubroutinize = False
    font = subset.load_font(source, opts)
    s = subset.Subsetter(opts)
    s.populate(unicodes=keep)
    s.subset(font)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subset.save_font(font, str(OUT), opts)
    (OUT.parent / "NotoSansHK-OFL.txt").write_bytes(LICENSE.read_bytes())
    han = "".join(chr(c) for c in keep if c > 0x2000)
    OUT.with_suffix(".chars.txt").write_text(han + "\n", encoding="utf-8")
    print(len(keep), "characters,", OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    build(sys.argv[1])
