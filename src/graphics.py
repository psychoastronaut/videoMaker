"""Pillow-based graphics: badges, lower-thirds, cards, shadows, grid backgrounds."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src import theme

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BG_COLOR = (10, 10, 26)
PANEL_COLOR = (16, 24, 48)

# Accent per beat style (RGB).
STYLE_ACCENTS: dict[str, tuple[int, int, int]] = {
    "normal": (74, 144, 217),
    "deadpan": (150, 162, 185),
    "hype": (255, 184, 0),
    "technical": (0, 214, 255),
    "fail": (255, 64, 64),
}


def accent_for(style: str) -> tuple[int, int, int]:
    if theme.is_terminal():
        return theme.accent(style, theme.TERMINAL)
    return STYLE_ACCENTS.get(style, STYLE_ACCENTS["normal"])


def panel_color() -> tuple[int, int, int]:
    return theme.panel()


def bg_color() -> tuple[int, int, int]:
    return theme.bg()


def accent_hex(style: str) -> str:
    r, g, b = accent_for(style)
    return f"0x{r:02x}{g:02x}{b:02x}"


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if theme.is_terminal():
        return theme.font(size)  # type: ignore[return-value]
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)
    except OSError:
        return ImageFont.load_default(size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font)


def _fit_font(text: str, size: int, max_width: int, bold: bool = True):
    """Shrink the font until the text fits in max_width (minimum 50% of requested size)."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    min_size = max(12, size // 2)
    font = load_font(size, bold)
    while size > min_size:
        box = _text_size(probe, text, font)
        if box[2] - box[0] <= max_width:
            break
        size -= 2
        font = load_font(size, bold)
    return font


def _fit_text(text: str, size: int, max_width: int, bold: bool = True):
    """Like _fit_font but also truncates with an ellipsis if even the smallest size overflows,
    so text can never run past its safe area. Returns (text, font)."""
    font = _fit_font(text, size, max_width, bold)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    while len(text) > 1:
        box = _text_size(probe, text, font)
        if box[2] - box[0] <= max_width:
            break
        text = text[:-2].rstrip() + "\u2026"
    return text, font


def render_badge(text: str, style: str, font_size: int = 46, max_width: int = 1100) -> Image.Image:
    """Dark rounded badge with accent border and bar; tight to the text. Terminal theme:
    ``[ TEXT ]_`` tag with glow."""
    if theme.is_terminal():
        return theme.terminal_badge(text, style, font_size, max_width)
    text = text.upper()
    accent = accent_for(style)
    text, font = _fit_text(text, font_size, max_width - 100)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x0, y0, x1, y1 = _text_size(probe, text, font)
    tw, th = x1 - x0, y1 - y0
    pad_x, pad_y, bar = 34, 22, 12
    w, h = tw + pad_x * 2 + bar, th + pad_y * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=14, fill=(8, 10, 24, 238),
                        outline=accent + (255,), width=3)
    d.rounded_rectangle([0, 0, bar + 6, h - 1], radius=14, fill=accent + (255,))
    d.rectangle([bar, 0, bar + 8, h - 1], fill=(8, 10, 24, 238))
    d.text((bar + pad_x - x0, pad_y - y0), text, font=font, fill=(240, 246, 255, 255))
    return img


def display_case(text: str) -> str:
    """Title Case that keeps acronyms/mixed-case words (PID, LiPo) as written."""
    return " ".join(w if w[1:] != w[1:].lower() else w[:1].upper() + w[1:].lower()
                    for w in text.split())


def render_lower_third(text: str, style: str, font_size: int = 40) -> Image.Image:
    """Lower-third bar (Title Case; badges are UPPERCASE): accent line on top, dark body.
    Terminal theme: ``[DATA] > LABEL`` with a style prefix."""
    if theme.is_terminal():
        return theme.terminal_lower_third(text, style, font_size)
    text = display_case(text)
    accent = accent_for(style)
    text, font = _fit_text(text, font_size, 1300, bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x0, y0, x1, y1 = _text_size(probe, text, font)
    tw, th = x1 - x0, y1 - y0
    pad_x, pad_y = 36, 24
    w, h = tw + pad_x * 2, th + pad_y * 2 + 6
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], fill=(8, 10, 24, 226))
    d.rectangle([0, 0, w - 1, 5], fill=accent + (255,))
    d.text((pad_x - x0, pad_y + 6 - y0), text, font=font, fill=(240, 246, 255, 255))
    return img


def with_shadow(img: Image.Image, offset: tuple[int, int] = (14, 18), blur: int = 18,
                opacity: int = 170) -> Image.Image:
    """Return img composited over a blurred drop shadow (canvas grows to fit)."""
    img = img.convert("RGBA")
    margin = blur * 3
    w, h = img.size
    canvas = Image.new("RGBA", (w + margin * 2 + abs(offset[0]), h + margin * 2 + abs(offset[1])),
                       (0, 0, 0, 0))
    alpha = img.getchannel("A").point(lambda v: v * opacity // 255)
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 255))
    shadow.putalpha(alpha)
    canvas.paste(shadow, (margin + max(offset[0], 0), margin + max(offset[1], 0)), shadow)
    canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(img, (margin + max(-offset[0], 0), margin + max(-offset[1], 0)))
    return canvas


def draw_grid(img: Image.Image, step: int, color: tuple[int, int, int]) -> None:
    d = ImageDraw.Draw(img)
    for x in range(0, img.width, step):
        d.line([(x, 0), (x, img.height)], fill=color, width=1)
    for y in range(0, img.height, step):
        d.line([(0, y), (img.width, y)], fill=color, width=1)


def make_grid_background(path: Path, width: int, height: int, opacity: float = 0.15) -> Path:
    """Dark engineering grid. ``opacity`` blends the grid lines towards a visible blue."""
    if theme.is_terminal():
        img = theme.grid_background(width, height, strength=min(0.6 + opacity * 3, 1.5))
        img.save(path)
        return path
    img = Image.new("RGB", (width, height), BG_COLOR)
    base = BG_COLOR
    line = (40, 70, 130)
    color = tuple(int(base[i] + (line[i] - base[i]) * min(opacity * 3, 1.0)) for i in range(3))
    draw_grid(img, 60, color)  # type: ignore[arg-type]
    img.save(path)
    return path


def make_card(path: Path, title: str, subtitle: str = "", width: int = 1920, height: int = 1080,
              accent: tuple[int, int, int] = (74, 144, 217)) -> Path:
    """Dark-mode engineering card used as an explicit fallback visual."""
    if theme.is_terminal():
        return _terminal_card(path, title, subtitle, width, height, accent)
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw_grid(img, 40, (26, 26, 58))
    d = ImageDraw.Draw(img)
    title = title.strip()[:60] or "Technical Visual"
    font = _fit_font(title, 64, width - 300)
    x0, y0, x1, y1 = _text_size(d, title, font)
    tw, th = x1 - x0, y1 - y0
    cx, cy = width // 2, height // 2
    d.rounded_rectangle([cx - tw // 2 - 50, cy - th // 2 - 36, cx + tw // 2 + 50,
                         cy + th // 2 + 36], radius=16, fill=(16, 20, 44), outline=accent, width=4)
    d.text((cx - tw // 2 - x0, cy - th // 2 - y0), title, font=font, fill=(235, 242, 255))
    if subtitle:
        sfont = _fit_font(subtitle, 30, width - 300, bold=False)
        sx0, _, sx1, _ = _text_size(d, subtitle, sfont)
        d.text((cx - (sx1 - sx0) // 2 - sx0, cy + th // 2 + 70), subtitle, font=sfont,
               fill=(150, 160, 190))
    img.save(path)
    return path


def _terminal_card(path: Path, title: str, subtitle: str, width: int, height: int,
                   accent: tuple[int, int, int]) -> Path:
    img = theme.grid_background(width, height, step=max(round(40 * width / 1920), 8)).convert("RGBA")
    col = accent if accent != (74, 144, 217) else theme.GREEN
    text, fnt, tr = theme.fit_mono(f"[ {(title.strip()[:60] or 'TECHNICAL VISUAL').upper()} ]",
                                   round(70 * width / 1920), width - 360, 3, display=True)
    layer = theme.glow_text(text, fnt, col, tr, glow=6)
    box = theme.glow_box((layer.width + 60, layer.height + 20), col, width=3, glow=10)
    cx, cy = width // 2, height // 2
    img.alpha_composite(box, (cx - box.width // 2, cy - box.height // 2))
    img.alpha_composite(layer, (cx - layer.width // 2, cy - layer.height // 2))
    if subtitle:
        s2, sf, st = theme.fit_mono("> " + subtitle.upper(), round(30 * width / 1920),
                                    width - 360, 2)
        sl = theme.glow_text(s2, sf, theme.GREEN_MID, st, glow=3)
        img.alpha_composite(sl, (cx - sl.width // 2, cy + box.height // 2 + 10))
    img.convert("RGB").save(path)
    return path


def _terminal_title_card(path: Path, text: str, style: str, width: int, height: int) -> Path:
    col = accent_for(style)
    k = width / 1920
    img = theme.grid_background(width, height, step=max(round(40 * k), 8)).convert("RGBA")
    d = ImageDraw.Draw(img)
    words = (display_case(text.strip()) or "Key Point").upper().split()[:5]
    lines = [" ".join(words)] if len(words) <= 2 else [" ".join(words[:len(words) // 2 + 1]),
                                                       " ".join(words[len(words) // 2 + 1:])]
    lines = [ln for ln in lines if ln]
    for x, y, sx, sy in [(70, 70, 1, 1), (width - 70, 70, -1, 1),
                         (70, height - 70, 1, -1), (width - 70, height - 70, -1, -1)]:
        d.line([(x, y), (x + 110 * sx, y)], fill=col, width=4)
        d.line([(x, y), (x, y + 110 * sy)], fill=col, width=4)
    d.line([(160, 84), (width - 160, 84)], fill=theme.RULE, width=2)
    tag = theme.glow_text(f"[{theme.tag_for(style)}] SIGNAL LOCKED", theme.font(round(30 * k)),
                          col, 3, glow=4)
    img.alpha_composite(tag, (150, 100))
    longest = max(lines, key=len)
    _t, fnt, tr = theme.fit_mono(longest, round(190 * k), width - 420, 4, display=True)
    y = (height - len(lines) * round(fnt.size * 1.0)) // 2
    for i, ln in enumerate(lines):
        c = col if i == len(lines) - 1 else theme.MINT
        txt = ln + ("_" if i == len(lines) - 1 else "")
        layer = theme.glow_text(txt, fnt, c, tr, glow=8)
        img.alpha_composite(layer, (190 - 24, y - 24))
        y += round(fnt.size * 1.0)
    img.convert("RGB").save(path)
    return path


def make_title_card(path: Path, text: str, style: str = "normal", width: int = 1920,
                    height: int = 1080) -> Path:
    """Designed full-frame title card (used only when no real media exists): big Title Case
    text, accent word and bar, no debug text."""
    if theme.is_terminal():
        return _terminal_title_card(path, text, style, width, height)
    accent = accent_for(style)
    words = (display_case(text.strip()) or "Key Point").split()[:5]
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw_grid(img, 60, (22, 28, 60))
    d = ImageDraw.Draw(img)
    # vertical accent bar + corner brackets
    d.rectangle([110, 300, 126, height - 300], fill=accent)
    for x, y, sx, sy in [(70, 70, 1, 1), (width - 70, 70, -1, 1),
                         (70, height - 70, 1, -1), (width - 70, height - 70, -1, -1)]:
        d.line([(x, y), (x + 110 * sx, y)], fill=accent, width=8)
        d.line([(x, y), (x, y + 110 * sy)], fill=accent, width=8)
    # two lines at most; the last word is the accent word
    lines = [" ".join(words)] if len(words) <= 2 else [" ".join(words[:len(words) // 2 + 1]),
                                                       " ".join(words[len(words) // 2 + 1:])]
    lines = [ln for ln in lines if ln]
    font = _fit_font(max(lines, key=len), 170, width - 420)
    boxes = [d.textbbox((0, 0), ln, font=font) for ln in lines]
    line_h = max(b[3] - b[1] for b in boxes) + 28
    y = (height - line_h * len(lines)) // 2
    for i, (ln, box) in enumerate(zip(lines, boxes, strict=True)):
        x = 190
        if i == len(lines) - 1 and len(ln.split()) > 0:
            head, _, tail = ln.rpartition(" ")
            if head:
                d.text((x - box[0], y - box[1]), head + " ", font=font, fill=(235, 242, 255))
                hw = d.textlength(head + " ", font=font)
                d.text((x + hw - box[0], y - box[1]), tail, font=font, fill=accent)
            else:
                d.text((x - box[0], y - box[1]), ln, font=font, fill=accent)
        else:
            d.text((x - box[0], y - box[1]), ln, font=font, fill=(235, 242, 255))
        y += line_h
    img.save(path)
    return path


def photo_card(image: Image.Image, radius: int = 28, border: int = 6,
               accent: tuple[int, int, int] = (235, 240, 255)) -> Image.Image:
    """Rounded-corner photo with a light border (RGBA); the shadow is added by with_shadow."""
    from PIL import ImageChops

    if theme.is_terminal():
        radius, border = 3, 3
        accent = theme.GREEN if accent == (235, 240, 255) else accent
    photo = image.convert("RGB")
    w, h = photo.size
    radius = (3 if theme.is_terminal() else min(radius * max(w, h) // 900 + 12, min(w, h) // 4))
    border = max(border * max(w, h) // 900, 4)
    out = Image.new("RGBA", (w + 2 * border, h + 2 * border), accent + (255,))
    out.paste(photo, (border, border))
    mask = Image.new("L", out.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, out.width - 1, out.height - 1),
                                           radius=radius + border, fill=255)
    out.putalpha(ImageChops.multiply(out.getchannel("A"), mask))
    return out
