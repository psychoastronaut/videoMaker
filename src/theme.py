"""Visual themes: palette, fonts and drawing helpers.

``theme: psycho_terminal`` mirrors the psychoastronaut.xyz look (near-black navy, phosphor-green
neon with glow, cyan accents, magenta buttons, monospace/pixel CRT fonts).  ``theme: classic``
keeps the original blue look.  Colours were sampled from the website screenshot.

Fonts (SIL OFL 1.1, files + licences in ``assets/fonts``): Share Tech Mono (text, legible at
small sizes) and VT323 (display / ASCII logo).  Fallback: DejaVu Sans Mono.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

TERMINAL = "psycho_terminal"
CLASSIC = "classic"
THEMES = (CLASSIC, TERMINAL)

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "assets" / "fonts"
FONT_TEXT = FONT_DIR / "ShareTechMono-Regular.ttf"
FONT_DISPLAY = FONT_DIR / "VT323-Regular.ttf"
FONT_FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

RGB = tuple[int, int, int]

# --- sampled from psychoastronaut.xyz ---------------------------------------------------------
BG: RGB = (0, 3, 8)
PANEL: RGB = (2, 10, 14)
GRID: RGB = (5, 20, 17)
SCRIM: tuple[int, int, int, int] = (1, 6, 9, 238)
GREEN: RGB = (0, 255, 65)        # neon text
GREEN_MID: RGB = (0, 170, 43)    # logo / status text
GREEN_DIM: RGB = (0, 139, 36)    # rules
RULE: RGB = (0, 53, 19)
CYAN: RGB = (0, 255, 249)
MAGENTA: RGB = (255, 0, 255)
YELLOW: RGB = (232, 255, 40)     # hype: amber -> neon yellow
RED: RGB = (255, 98, 130)         # fail: red/magenta
MINT: RGB = (214, 255, 228)      # label text

STYLE_ACCENTS: dict[str, RGB] = {
    "normal": GREEN,
    "deadpan": (0, 190, 60),
    "hype": YELLOW,
    "technical": CYAN,
    "fail": RED,
}
STYLE_TAGS = {"fail": "ERR", "technical": "DATA", "hype": "OK", "deadpan": "SYS",
              "normal": "SECTOR"}

CLASSIC_BG: RGB = (10, 10, 26)
CLASSIC_PANEL: RGB = (16, 24, 48)


# --------------------------------------------------------------------------- config

_cfg_cache: dict[str, tuple[float, str]] = {}


def active() -> str:
    """Theme name from the active config (``theme:``; missing = classic)."""
    from src.common import load_config

    override = os.environ.get("VIDEOMAKER_CONFIG")
    try:
        key = override or "default"
        mtime = Path(override).stat().st_mtime if override else 0.0
        if key in _cfg_cache and _cfg_cache[key][0] == mtime:
            return _cfg_cache[key][1]
        name = normalize(load_config().get("theme"))
        _cfg_cache[key] = (mtime, name)
        return name
    except (OSError, ValueError):
        return CLASSIC


def normalize(name: Any) -> str:
    n = str(name or CLASSIC).strip().lower()
    return n if n in THEMES else CLASSIC


def is_terminal(theme: str | None = None) -> bool:
    return (theme if theme is not None else active()) == TERMINAL


def bg(theme: str | None = None) -> RGB:
    return BG if is_terminal(theme) else CLASSIC_BG


def panel(theme: str | None = None) -> RGB:
    return PANEL if is_terminal(theme) else CLASSIC_PANEL


def accent(style: str, theme: str | None = None) -> RGB:
    if is_terminal(theme):
        return STYLE_ACCENTS.get(style, GREEN)
    return {"normal": (74, 144, 217), "deadpan": (150, 162, 185), "hype": (255, 184, 0),
            "technical": (0, 214, 255), "fail": (255, 64, 64)}.get(style, (74, 144, 217))


def tag_for(style: str) -> str:
    return STYLE_TAGS.get(style, "SECTOR")


# --------------------------------------------------------------------------- contrast


def _lin(c: int) -> float:
    v = c / 255
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4


def luminance(rgb: RGB) -> float:
    return 0.2126 * _lin(rgb[0]) + 0.7152 * _lin(rgb[1]) + 0.0722 * _lin(rgb[2])


def contrast(a: RGB, b: RGB) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# --------------------------------------------------------------------------- fonts


@functools.lru_cache(maxsize=64)
def font(size: int, display: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ((FONT_DISPLAY if display else FONT_TEXT), FONT_FALLBACK):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def text_width(text: str, fnt, tracking: int = 0) -> int:
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    return round(probe.textlength(text, font=fnt)) + tracking * max(len(text) - 1, 0)


def fit_mono(text: str, size: int, max_width: int, tracking: int = 0, display: bool = False):
    """Shrink (down to 50 %) then truncate with an ellipsis so text never exceeds max_width.
    Returns (text, font, tracking)."""
    sz = size
    fnt = font(sz, display)
    while sz > max(12, size // 2) and text_width(text, fnt, tracking) > max_width:
        sz -= 2
        fnt = font(sz, display)
    while len(text) > 1 and text_width(text, fnt, tracking) > max_width:
        text = text[:-2].rstrip() + "…"
    return text, fnt, tracking


def draw_tracked(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt,
                 fill: tuple[int, ...], tracking: int = 0) -> None:
    x, y = xy
    if not tracking:
        d.text((x, y), text, font=fnt, fill=fill)
        return
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        x += d.textlength(ch, font=fnt) + tracking


def glow_text(text: str, fnt, color: RGB, tracking: int = 0, glow: int = 6,
              strength: float = 0.9, pad: int | None = None) -> Image.Image:
    """RGBA layer: crisp text over a soft blurred halo of the same colour."""
    pad = pad if pad is not None else glow * 3
    w = max(text_width(text, fnt, tracking), 1) + 2 * pad
    asc, desc = fnt.getmetrics()
    h = asc + desc + 2 * pad
    mask = Image.new("L", (w, h), 0)
    draw_tracked(ImageDraw.Draw(mask), (pad, pad), text, fnt, (255,), tracking)
    halo = mask.filter(ImageFilter.GaussianBlur(glow)).point(lambda v: min(255, int(v * 2.2 * strength)))
    out = Image.new("RGBA", (w, h), color + (0,))
    out.putalpha(halo)
    crisp = Image.new("RGBA", (w, h), color + (0,))
    crisp.putalpha(mask)
    out.alpha_composite(crisp)
    return out


def glow_box(size: tuple[int, int], color: RGB, width: int = 2, glow: int = 8, radius: int = 3,
             fill: tuple[int, int, int, int] | None = SCRIM) -> Image.Image:
    """RGBA box (thin border, dark scrim, outer glow) of ``size`` inner pixels; the returned
    image is bigger by ``glow*3`` on each side (use ``pad = glow * 3``)."""
    pad = glow * 3
    w, h = size[0] + 2 * pad, size[1] + 2 * pad
    ring = Image.new("L", (w, h), 0)
    ImageDraw.Draw(ring).rounded_rectangle((pad, pad, pad + size[0] - 1, pad + size[1] - 1),
                                           radius=radius, outline=255, width=width + 1)
    halo = ring.filter(ImageFilter.GaussianBlur(glow / 2)).point(lambda v: min(255, int(v * 0.9)))
    img = Image.new("RGBA", (w, h), color + (0,))
    img.putalpha(halo)
    d = ImageDraw.Draw(img)
    if fill:
        d.rounded_rectangle((pad, pad, pad + size[0] - 1, pad + size[1] - 1), radius=radius,
                            fill=fill)
    d.rounded_rectangle((pad, pad, pad + size[0] - 1, pad + size[1] - 1), radius=radius,
                        outline=color + (255,), width=width)
    return img


def terminal_badge(text: str, style: str, font_size: int = 46, max_width: int = 1100,
                   cursor: bool = True) -> Image.Image:
    """``[ SPHERICAL ABERRATION ]_`` tag: thin border, dark scrim, neon text with glow."""
    col = accent(style, TERMINAL)
    label = f"[ {text.upper().strip()} ]" + ("_" if cursor else "")
    glow = 7
    pad_x, pad_y, tracking = 26, 16, 2
    label, fnt, tracking = fit_mono(label, font_size, max_width - 2 * pad_x - 2 * glow * 3,
                                    tracking)
    txt = glow_text(label, fnt, col, tracking, glow=4, pad=4)
    inner = (txt.width - 8 + 2 * pad_x, txt.height - 8 + 2 * pad_y)
    box = glow_box(inner, col, width=2, glow=glow)
    p = glow * 3
    box.alpha_composite(txt, (p + pad_x - 4, p + pad_y - 4))
    return box


def terminal_lower_third(text: str, style: str, font_size: int = 40, max_width: int = 1300
                         ) -> Image.Image:
    """``[DATA] > label`` bar: accent prefix, pale-mint label on a dark scrim, accent edge."""
    col = accent(style, TERMINAL)
    prefix = f"[{tag_for(style)}] "
    tracking = 1
    body, fnt, tracking = fit_mono("> " + text.upper().strip(), font_size,
                                   max_width - 2 * 34 - 14 - text_width(prefix,
                                                                         font(font_size), 1),
                                   tracking)
    pfnt = fnt
    pw = text_width(prefix, pfnt, tracking) + tracking
    bw = text_width(body, fnt, tracking)
    asc, desc = fnt.getmetrics()
    pad_x, pad_y, bar = 34, 18, 8
    w, h = bar + pad_x * 2 + pw + bw, asc + desc + pad_y * 2
    img = Image.new("RGBA", (w, h), SCRIM)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, bar - 1, h - 1], fill=col + (255,))
    d.rectangle([0, 0, w - 1, h - 1], outline=col + (150,), width=1)
    pre = glow_text(prefix, pfnt, col, tracking, glow=3, pad=3)
    img.alpha_composite(pre, (bar + pad_x - 3, pad_y - 3))
    draw_tracked(d, (bar + pad_x + pw, pad_y), body, fnt, MINT + (255,), tracking)
    return img


def grid_background(width: int, height: int, step: int = 40, strength: float = 1.0
                    ) -> Image.Image:
    """Near-black navy with a very faint green grid (as on the website)."""
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    c = tuple(round(BG[i] + (GRID[i] - BG[i]) * strength) for i in range(3))
    for x in range(0, width, step):
        d.line([(x, 0), (x, height)], fill=c, width=1)
    for y in range(0, height, step):
        d.line([(0, y), (width, y)], fill=c, width=1)
    return img  # type: ignore[return-value]


# --------------------------------------------------------------------------- ASCII logo

_GLYPHS = {
    "P": ["XXXX.", "X...X", "X...X", "XXXX.", "X....", "X....", "X...."],
    "S": [".XXXX", "X....", "X....", ".XXX.", "....X", "....X", "XXXX."],
    "Y": ["X...X", "X...X", ".X.X.", "..X..", "..X..", "..X..", "..X.."],
    "C": [".XXXX", "X....", "X....", "X....", "X....", "X....", ".XXXX"],
    "H": ["X...X", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"],
    "0": [".XXX.", "X...X", "X..XX", "X.X.X", "XX..X", "X...X", ".XXX."],
}


def logo_cells(word: str = "PSYCH0") -> list[tuple[int, int]]:
    cells, ox = [], 0
    for ch in word:
        for r, row in enumerate(_GLYPHS[ch]):
            cells += [(ox + c, r) for c, v in enumerate(row) if v == "X"]
        ox += 6
    return cells


def logo_image(cell: int = 22, word: str = "PSYCH0") -> Image.Image:
    """Big blocky pixel logo with the website's double-layer green and a soft glow."""
    cols, rows = len(word) * 6 - 1, 7
    pad = cell * 2
    w, h = cols * cell + 2 * pad, rows * cell + 2 * pad
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    off = max(cell // 4, 2)
    cells = logo_cells(word)
    for cx, cy in cells:
        x, y = pad + cx * cell, pad + cy * cell
        d.rectangle([x + off, y + off, x + cell - 1 + off, y + cell - 1 + off], fill=GREEN_DIM + (255,))
    for cx, cy in cells:
        x, y = pad + cx * cell, pad + cy * cell
        d.rectangle([x, y, x + cell - 2, y + cell - 2], fill=GREEN_MID + (255,))
        d.rectangle([x, y, x + cell - 2, y + max(cell // 6, 2)], fill=(30, 235, 80, 255))
    halo = base.filter(ImageFilter.GaussianBlur(cell * 0.6))
    halo.putalpha(halo.getchannel("A").point(lambda v: min(255, int(v * 1.4))))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.alpha_composite(halo)
    out.alpha_composite(base)
    return out


# --------------------------------------------------------------------------- boot cards

# Boot-card timeline in seconds (shared with the sound design): typing starts / chars per second.
INTRO_T = {"logo": (0.10, 0.60), "sub": (0.55, 45.0), "rule": 1.20, "status": (1.25, 100.0),
           "glitch_out": 0.28}
OUTRO_T = {"l1": (0.15, 30.0), "l2": (0.65, 70.0), "rule": 0.95, "l3": (1.00, 45.0),
           "github": 1.70}
INTRO_STATUS = "[OK] CONNECTION ESTABLISHED   |   [OK] SIGNAL LOCKED   |   [SECTOR] CYBERSPACE"
INTRO_SUB = "ASTRONA4UT // SYSTEM ONLINE"
OUTRO_L1 = "> END OF TRANSMISSION"
OUTRO_L2 = "[OK] SIGNAL RELEASED  |  [OK] SEE YOU IN THE NEXT SECTOR"
OUTRO_L3 = "// EOF // psychoastronaut.xyz"

INTRO_LINES = ["[OK] CONNECTION ESTABLISHED", "[OK] SIGNAL LOCKED", "[SECTOR] CYBERSPACE"]


def _typed(text: str, t: float, start: float, cps: float = 38.0) -> str:
    n = int(max(t - start, 0) * cps)
    return text[:n]


def _paste_center(img: Image.Image, layer: Image.Image, cy: int) -> None:
    img.alpha_composite(layer, ((img.width - layer.width) // 2, cy - layer.height // 2))


def _colored_line(img: Image.Image, text: str, cy: int, fnt, tracking: int,
                  default: RGB, glow: int = 3) -> None:
    """Line whose leading ``[TAG]`` words are neon-green and the rest ``default`` (status look)."""
    parts, cur = [], ""
    for tok in text.split(" "):
        parts.append((tok, GREEN if tok.startswith("[") and tok.endswith("]") else default))
    total = text_width(text, fnt, tracking)
    x = (img.width - total) // 2
    space = text_width(" ", fnt, tracking) + tracking
    for tok, col in parts:
        layer = glow_text(tok, fnt, col, tracking, glow=glow, pad=glow * 2)
        img.alpha_composite(layer, (x - glow * 2, cy - layer.height // 2))
        x += text_width(tok, fnt, tracking) + tracking + space
    del cur


def boot_frame(kind: str, t: float, width: int = 1920, height: int = 1080,
               duration: float | None = None, fade_in: float = 0.0, fade_out: float = 0.0
               ) -> Image.Image:
    """One frame of the intro (``kind='intro'``) or outro boot card at time ``t`` seconds."""
    k = width / 1920
    img = _bg_cached(width, height).copy().convert("RGBA")
    d = ImageDraw.Draw(img)
    cx = width // 2
    rule_c = RULE + (255,)
    if kind == "intro":
        logo = _logo_cached(max(round(22 * k), 8))
        # logo reveals row by row from 0.15 s to 0.85 s, with a one-frame RGB jitter early on
        l0, l1 = INTRO_T["logo"]
        frac = min(max((t - l0) / (l1 - l0), 0.0), 1.0)
        rows = max(int(logo.height * frac), 0)
        top = int(height * 0.40) - logo.height // 2
        if rows:
            part = logo.crop((0, 0, logo.width, rows))
            if 0.1 < t < 0.45 and int(t * 30) % 3 == 0:
                j = round(8 * k)
                red = part.copy()
                red.putalpha(part.getchannel("A").point(lambda v: v // 3))
                img.alpha_composite(red, ((width - logo.width) // 2 + j, top))
            img.alpha_composite(part, ((width - logo.width) // 2, top))
        sub = _typed(INTRO_SUB, t, *INTRO_T["sub"])
        if sub:
            sfnt = font(round(34 * k))
            layer = glow_text(sub, sfnt, CYAN, round(9 * k), glow=5)
            _paste_center(img, layer, int(height * 0.40) + logo.height // 2 + round(40 * k))
        if t > INTRO_T["rule"]:
            y = int(height * 0.40) + logo.height // 2 + round(110 * k)
            span = int(min((t - INTRO_T["rule"]) / 0.2, 1.0) * 820 * k)
            d.line([(cx - span // 2, y), (cx + span // 2, y)], fill=rule_c, width=2)
        line = INTRO_STATUS
        shown = _typed(line, t, *INTRO_T["status"])
        if shown:
            lfnt = font(round(26 * k))
            y = int(height * 0.40) + logo.height // 2 + round(160 * k)
            tw = text_width(line, lfnt, 1)
            x0 = cx - tw // 2
            x = x0
            for tok in shown.split(" "):
                col = GREEN if tok.startswith("[") else GREEN_MID
                lay = glow_text(tok, lfnt, col, 1, glow=3, pad=6)
                img.alpha_composite(lay, (int(x) - 6, y - lay.height // 2))
                x += text_width(tok, lfnt, 1) + text_width(" ", lfnt, 1) + 2
        if int(t * 4) % 2 == 0 and t > 0.1:
            cfnt = font(round(30 * k))
            layer = glow_text("_", cfnt, GREEN, 0, glow=4)
            img.alpha_composite(layer, (round(70 * k), round(60 * k)))
    else:
        l1 = _typed(OUTRO_L1, t, *OUTRO_T["l1"])
        lfnt = font(round(54 * k), True)
        if l1:
            layer = glow_text(l1 + ("_" if int(t * 5) % 2 == 0 else ""), lfnt, GREEN, 3, glow=6)
            _paste_center(img, layer, int(height * 0.36))
        l2 = _typed(OUTRO_L2, t, *OUTRO_T["l2"])
        if l2:
            sfnt = font(round(26 * k))
            layer = glow_text(l2, sfnt, GREEN_MID, 1, glow=3)
            _paste_center(img, layer, int(height * 0.36) + round(80 * k))
        if t > OUTRO_T["rule"]:
            y = int(height * 0.62)
            span = int(min((t - OUTRO_T["rule"]) / 0.25, 1.0) * 900 * k)
            d.line([(cx - span // 2, y), (cx + span // 2, y)], fill=GREEN_DIM + (255,), width=2)
        l3 = _typed(OUTRO_L3, t, *OUTRO_T["l3"])
        if l3:
            efnt = font(round(32 * k))
            layer = glow_text(l3, efnt, GREEN_MID, 2, glow=4)
            _paste_center(img, layer, int(height * 0.62) + round(54 * k))
        if t > OUTRO_T["github"]:
            bfnt = font(round(30 * k))
            label = ">_ GITHUB"
            tw = text_width(label, bfnt, 2)
            box = glow_box((tw + round(56 * k), round(56 * k)), MAGENTA, width=2, glow=8,
                           fill=(18, 0, 18, 235))
            by = int(height * 0.62) + round(150 * k)
            _paste_center(img, box, by)
            txt = glow_text(label, bfnt, MAGENTA, 2, glow=4, pad=4)
            _paste_center(img, txt, by)
    out = img.convert("RGB")
    if kind == "intro" and duration is not None and t > duration - INTRO_T["glitch_out"]:
        out = _glitch(out, (t - (duration - INTRO_T["glitch_out"])) / INTRO_T["glitch_out"],
                      int(t * 1000))
    f = 1.0
    if fade_in > 0 and t < fade_in:
        f = min(f, t / fade_in)
    if duration is not None and fade_out > 0 and t > duration - fade_out:
        f = min(f, max((duration - t) / fade_out, 0.0))
    if f < 1.0:
        out = Image.eval(out, lambda v: int(v * f))
    return out


def _glitch(img: Image.Image, amount: float, seed: int) -> Image.Image:
    """Transition into the first beat: RGB split plus horizontal slice jitter, ramping up."""
    import random

    rnd = random.Random(seed)
    w, h = img.size
    amount = min(max(amount, 0.0), 1.0)
    shift = round(w * 0.012 * amount) + 1
    r, g, b = img.split()
    r = r.transform(r.size, Image.AFFINE, (1, 0, -shift, 0, 1, 0))
    b = b.transform(b.size, Image.AFFINE, (1, 0, shift, 0, 1, 0))
    out = Image.merge("RGB", (r, g, b))
    for _ in range(2 + int(8 * amount)):
        y0 = rnd.randrange(0, h - 8)
        bh = rnd.randrange(6, max(h // 14, 8))
        dx = rnd.randrange(-int(w * 0.08 * amount) - 4, int(w * 0.08 * amount) + 5)
        band = out.crop((0, y0, w, min(y0 + bh, h)))
        out.paste(band, (dx, y0))
    return out


@functools.lru_cache(maxsize=4)
def _bg_cached(width: int, height: int) -> Image.Image:
    return grid_background(width, height, step=max(round(40 * width / 1920), 8))


@functools.lru_cache(maxsize=4)
def _logo_cached(cell: int) -> Image.Image:
    return logo_image(cell)


def write_boot_frames(kind: str, out_dir: Path, fps: int, seconds: float, width: int,
                      height: int, fade_in: float = 0.0, fade_out: float = 0.0) -> int:
    """Writes ``f%04d.png`` for the whole card; returns the frame count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = round(seconds * fps)
    for i in range(n):
        boot_frame(kind, i / fps, width, height, seconds, fade_in, fade_out).save(
            out_dir / f"f{i:04d}.png", compress_level=1)
    return n


# --------------------------------------------------------------------------- CRT overlay

CRT_LEVELS = ("off", "subtle", "strong")


def normalize_crt(value: Any) -> str:
    if value is True:
        return "subtle"
    v = str(value if value is not None else "off").strip().lower()
    return v if v in CRT_LEVELS else "off"


def make_scanlines(path: Path, width: int, height: int, level: str) -> Path:
    """RGBA overlay of faint dark scanlines (every 3rd row) for ``crt`` level."""
    alpha = {"subtle": 20, "strong": 46}.get(level, 0)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    step = max(round(3 * height / 1080), 2)
    for y in range(0, height, step):
        d.line([(0, y), (width, y)], fill=(0, 6, 4, alpha), width=max(step // 3, 1))
    img.save(path)
    return path


def crt_chain(level: str, src: str, dst: str, scan_input: int | None, width: int) -> list[str]:
    """ffmpeg chains: green-cyan grade, light chromatic aberration, vignette, scanlines."""
    if level == "off":
        return [f"[{src}]null[{dst}]"]
    strong = level == "strong"
    shift = max(round((2 if strong else 1) * width / 1920), 1)
    grade = ("colorbalance=rs=-0.04:gs=0.02:bs=0.035:rm=-0.05:gm=0.015:bm=0.045:rh=-0.02:bh=0.02"
             if strong else
             "colorbalance=rs=-0.02:gs=0.01:bs=0.02:rm=-0.025:gm=0.01:bm=0.025")
    vig = "vignette=angle=0.62" if strong else "vignette=angle=0.42"
    chain = f"[{src}]{grade},rgbashift=rh=-{shift}:bh={shift}:edge=smear,{vig}"
    if strong:  # soft bloom: blurred copy screened over the picture
        chain += (",split[crt_a][crt_b];[crt_b]gblur=sigma=14,eq=brightness=-0.05[crt_bl];"
                  "[crt_a][crt_bl]blend=all_mode=screen:all_opacity=0.16")
    if scan_input is None:
        return [chain + f"[{dst}]"]
    if strong:
        return [chain + "[crt_m]", f"[crt_m][{scan_input}:v]overlay=format=auto[{dst}]"]
    return [chain + "[crt_m]", f"[crt_m][{scan_input}:v]overlay=format=auto[{dst}]"]


def hud_chain(src: str, dst: str, t0: float, width: int, height: int) -> str:
    """Corner HUD: blinking ``REC`` dot + ``T+mm:ss`` timecode in dim green (top-right,
    inside the safe margin).  ``t0`` is the beat start in seconds."""
    k = width / 1920
    size = max(round(30 * k), 12)
    y = round(54 * k)
    f = str(FONT_TEXT).replace(":", "\\:")
    f2 = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    tc = (f"T+%{{eif\\:floor((t+{t0:.3f})/60)\\:d\\:2}}\\:%{{eif\\:mod(floor(t+{t0:.3f})\\,60)\\:d\\:2}}")
    common = f"fontsize={size}:borderw=2:bordercolor=0x000a05@0.9"
    x0 = width - round(96 * k) - round(300 * k)
    return (f"[{src}]drawtext=fontfile={f2}:text='\u25cf':x={x0}:y={y}:"
            f"fontcolor=0xff3860:enable='lt(mod(t+{t0:.3f}\\,1)\\,0.6)':{common},"
            f"drawtext=fontfile={f}:text='REC':x={x0 + round(40 * k)}:y={y}:"
            f"fontcolor=0x00aa2b:{common},"
            f"drawtext=fontfile={f}:text='{tc}':x={x0 + round(130 * k)}:y={y}:"
            f"fontcolor=0x00aa2b:{common}[{dst}]")
