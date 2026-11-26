"""Picture-in-picture insets and animated highlights: Pillow-drawn assets + sidecar hints.

Everything here is deterministic. Stage 6 turns the PNGs into ffmpeg overlays; Stage 3 calls
:func:`apply_sidecar_hints` to copy sidecar hints (``audio:``, ``pip:``, ``highlight:``) onto
beats that did not set the fields themselves (hand-written plan values always win).

Sidecar lines (``<asset file>.txt``)::

    audio: speech | ambient | none       clip audio: duck narration | mix under it | off
    kind: screen | tablet | face         a small screen clip suited for a picture-in-picture
    pip_for: <asset id>                  show THIS asset as an inset on beats of that asset
    pip: <asset id>                      show that asset as an inset on beats of THIS asset
    highlight: x,y,w,h | label | start   box on a diagram (0-1 fractions of the picture, or
                                         pixels at 1920x1080); region names: top-left, top,
                                         top-right, left, center, right, bottom-left, bottom,
                                         bottom-right.  "arrow:" instead of "highlight:" draws
                                         an arrow pointing at the region.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from src import graphics, theme

REGIONS: dict[str, tuple[float, float, float, float]] = {
    "top-left": (0.04, 0.04, 0.4, 0.3), "top": (0.3, 0.04, 0.4, 0.3),
    "top-right": (0.56, 0.04, 0.4, 0.3), "left": (0.04, 0.35, 0.4, 0.3),
    "center": (0.3, 0.35, 0.4, 0.3), "right": (0.56, 0.35, 0.4, 0.3),
    "bottom-left": (0.04, 0.66, 0.4, 0.3), "bottom": (0.3, 0.66, 0.4, 0.3),
    "bottom-right": (0.56, 0.66, 0.4, 0.3),
}
PIP_KINDS = ("screen", "tablet", "face")
PIP_CORNERS = ("tr", "tl", "br", "bl")


# --------------------------------------------------------------------------- sidecar hints


def _sidecar(asset: dict[str, Any] | None) -> str:
    path = (asset or {}).get("filepath")
    try:
        return Path(str(path) + ".txt").read_text() if path else ""
    except OSError:
        return ""


def _field(text: str, key: str) -> str | None:
    m = re.search(rf"^\s*{key}\s*[:=]\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else None


def sidecar_kind(asset: dict[str, Any] | None) -> str | None:
    v = (_field(_sidecar(asset), "kind") or "").lower()
    return v if v in PIP_KINDS else None


def sidecar_audio(asset: dict[str, Any] | None) -> str | None:
    """``audio: speech|ambient|none`` -> clip_audio duck|on|off."""
    v = (_field(_sidecar(asset), "audio") or "").lower()
    return {"speech": "duck", "ambient": "on", "none": "off"}.get(v)


def parse_highlight(spec: str, shape: str = "box") -> dict[str, Any] | None:
    """``x,y,w,h | label | start`` or ``region | label | start`` -> highlight dict."""
    parts = [p.strip() for p in spec.split("|")]
    head = parts[0].lower().replace("_", "-")
    h: dict[str, Any] = {"shape": shape}
    nums = re.findall(r"-?\d+(?:\.\d+)?", parts[0])
    if len(nums) >= 4 and not head.isalpha():
        h.update(dict(zip("xywh", (float(n) for n in nums[:4]), strict=True)))
    elif head in REGIONS:
        h["region"] = head
    else:
        return None
    if len(parts) > 1 and parts[1]:
        h["label"] = parts[1][:40]
    if len(parts) > 2:
        try:
            h["start"] = float(parts[2])
        except ValueError:
            pass
    return h


def sidecar_highlights(asset: dict[str, Any] | None) -> list[dict[str, Any]]:
    out = []
    for m in re.finditer(r"^\s*(highlights?|arrow)\s*[:=]\s*(.+?)\s*$", _sidecar(asset),
                         re.IGNORECASE | re.MULTILINE):
        h = parse_highlight(m.group(2), "arrow" if m.group(1).lower() == "arrow" else "box")
        if h:
            out.append(h)
    return out


def apply_sidecar_hints(beats: list[Any], assets: list[dict[str, Any]]) -> int:
    """Copy sidecar hints onto SceneBeat objects (only into fields still unset). Returns the
    number of beats changed."""
    by_id = {a["id"]: a for a in assets}
    pip_for: dict[str, str] = {}
    for a in assets:
        target = _field(_sidecar(a), "pip_for")
        if target and sidecar_kind(a):
            pip_for[target.strip()] = a["id"]
    changed = 0
    for b in beats:
        main = by_id.get(b.source_id)
        ov = by_id.get(b.overlay_asset or "")
        touched = False
        if main is not None and b.clip_audio is None and main.get("asset_type") == "clip":
            hint = sidecar_audio(main)
            if hint:
                b.clip_audio, touched = hint, True
        if (b.overlay_mode is None and b.visual_template.value in
                ("fullscreen_clip", "punch_in_emphasis") and main is not None):
            pid = pip_for.get(b.source_id) or (_field(_sidecar(main), "pip") or "").strip()
            if pid and pid in by_id and pid != b.source_id:
                b.overlay_mode, b.pip_asset, touched = "pip", pid, True
        if b.highlights is None:
            hl = []
            for a in (main, ov):
                if a is not None and a.get("asset_type") == "schematic":
                    hl += sidecar_highlights(a)
            if hl:
                b.highlights, touched = hl, True
        changed += touched
    return changed


# --------------------------------------------------------------------------- picture-in-picture


def pip_geometry(width: int, height: int, aspect: float, corner: str, scale: float = 0.27
                 ) -> tuple[int, int, int, int, int]:
    """Inset (w, h, x, y, border) in output pixels; inside the 96/54 px (1920x1080) safe margins."""
    aspect = min(max(aspect, 0.5), 2.4)
    w = int(width * scale)
    h = int(w / aspect)
    if h > height * 0.4:
        h = int(height * 0.4)
        w = int(h * aspect)
    w, h = w - w % 2, h - h % 2
    mx, my = round(width * 0.05), round(height * 0.05)
    border = max(round(width * 0.004), 3)
    x = mx if corner.endswith("l") else width - mx - w - 2 * border
    y = my if corner.startswith("t") else height - my - h - 2 * border
    return w, h, x, y, border


def make_pip_frame(prefix: Path, w: int, h: int, border: int, accent: tuple[int, int, int],
                   radius: int | None = None) -> tuple[Path, Path, int]:
    """Writes ``<prefix>_frame.png`` (shadow + rounded border plate, size (w+2b+2pad)) and
    ``<prefix>_mask.png`` (rounded L mask of the video area). Returns (frame, mask, pad)."""
    radius = radius if radius is not None else (3 if theme.is_terminal()
                                                 else max(round(w * 0.06), 8))
    pad = max(round(w * 0.06), 12)
    pw, ph = w + 2 * border, h + 2 * border
    plate = Image.new("RGBA", (pw, ph), accent + (255,))
    m = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=radius + border, fill=255)
    plate.putalpha(ImageChops.multiply(plate.getchannel("A"), m))
    framed = graphics.with_shadow(plate, offset=(0, round(pad * 0.5)), blur=max(pad // 2, 6),
                                  opacity=190)
    # with_shadow pads by blur*3 on every side: report the real pad so callers can align
    real_pad = max(pad // 2, 6) * 3
    frame = prefix.with_name(prefix.name + "_frame.png")
    framed.save(frame)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    mask_path = prefix.with_name(prefix.name + "_mask.png")
    mask.save(mask_path)
    return frame, mask_path, real_pad


# --------------------------------------------------------------------------- highlights


def resolve_box(h: dict[str, Any], ref: tuple[int, int, int, int], width: int
                ) -> tuple[int, int, int, int]:
    """Highlight -> pixel box (x, y, w, h) inside ``ref`` = (x, y, w, h) of the picture area.
    Values <= 1 are fractions of ``ref``; larger values are pixels at 1920 px width."""
    rx, ry, rw, rh = ref
    if h.get("region") in REGIONS:
        x, y, w, hh = REGIONS[h["region"]]
    else:
        x, y, w, hh = (float(h.get(k, 0.0)) for k in "xywh")
    if max(x, y, w, hh) > 1.0:
        k = width / 1920
        px, py, pw, ph = x * k, y * k, w * k, hh * k
    else:
        px, py, pw, ph = x * rw, y * rh, w * rw, hh * rh
    return round(rx + px), round(ry + py), max(round(pw), 8), max(round(ph), 8)


def make_highlight(path: Path, w: int, h: int, label: str, accent: tuple[int, int, int],
                   thickness: int, font_px: int, shape: str = "box"
                   ) -> tuple[Path, int, int]:
    """RGBA plate for one highlight (target box w x h). Returns (png, dx, dy): the plate's
    top-left relative to the target's top-left (negative). Pulsing is done by ffmpeg."""
    terminal = theme.is_terminal()
    if terminal and label:
        label = f"[ {label.upper()} ]"
    m = thickness * 3
    chip_h = 0
    font = graphics.load_font(font_px)
    tw = 0
    if label:
        dummy = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        l, t, r, b = dummy.textbbox((0, 0), label, font=font)
        tw, chip_h = r - l + 2 * font_px // 2, b - t + font_px // 2 + thickness
    arrow_h = round(font_px * 2.2) if shape == "arrow" else 0
    top = chip_h + arrow_h
    cw, ch = max(w + 2 * m, tw + 2 * m), h + 2 * m + top
    cw, ch = cw + cw % 2, ch + ch % 2
    img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bx0 = (cw - w) // 2
    by0 = m + top
    glow = accent + (70,)
    d.rounded_rectangle((bx0 - m // 2, by0 - m // 2, bx0 + w + m // 2, by0 + h + m // 2),
                        radius=thickness * 3, outline=glow, width=thickness * 2)
    d.rounded_rectangle((bx0, by0, bx0 + w, by0 + h),
                        radius=2 if terminal else thickness * 2,
                        outline=accent + (255,), width=thickness)
    if shape == "arrow":
        cx, tip = cw // 2, by0 - thickness * 2
        a = arrow_h - thickness * 2
        d.polygon([(cx, tip), (cx - a // 2, tip - a // 2), (cx - a // 6, tip - a // 2),
                   (cx - a // 6, tip - a), (cx + a // 6, tip - a), (cx + a // 6, tip - a // 2),
                   (cx + a // 2, tip - a // 2)], fill=accent + (255,))
    if label:
        cx0 = (cw - tw) // 2
        cy0 = m // 2
        d.rounded_rectangle((cx0, cy0, cx0 + tw, cy0 + chip_h), radius=chip_h // 3,
                            fill=theme.SCRIM if terminal else (12, 14, 22, 235),
                            outline=accent + (255,), width=2)
        d.text((cx0 + tw // 2, cy0 + chip_h // 2), label, font=font,
               fill=theme.MINT + (255,) if terminal else (255, 255, 255, 255), anchor="mm")
    img.save(path)
    return path, -bx0, -by0
