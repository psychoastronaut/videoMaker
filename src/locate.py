"""Localise text on a diagram so an automatic highlight lands on something real.

A highlight box is only produced when a *real* location is found AND it matches the narration:

1. OCR (optional: ``easyocr`` or ``pytesseract`` + the tesseract binary): words with boxes;
   the narration's content words/numbers pick the matching words; the box is their union.
2. Otherwise the local VLM (Ollama ``qwen2.5vl``) proposes a bbox for the narrated text; the
   box is then *validated*: the cropped region is read back by the VLM and must contain a
   narrated word/number.

Nothing found -> ``None`` (no highlight is drawn; the safe default). Boxes are fractions
``[x, y, w, h]`` of the picture, padded a little, never covering the title band or the
caption band, and never larger than ``MAX_AREA`` of the picture.
"""

from __future__ import annotations

import base64
import io
import re
import shutil
from pathlib import Path
from typing import Any

import httpx

TITLE_BAND = 0.12      # a box whose centre lies in the top band would cover the title
CAPTION_BAND = 0.92    # ... or the bottom caption
MAX_AREA = 0.30
PAD = 0.012
_STOP = frozenset(["the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are", "was", "were", "it", "its", "this", "that", "with", "for", "as", "at", "by", "from", "be", "has", "have", "had", "not", "but", "we", "you", "they", "he", "she", "i"])


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9.\-]*", text.lower())
            if w not in _STOP and (len(w) > 3 or any(c.isdigit() for c in w))}


def _clean(w: str) -> str:
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", w.lower())


def ocr_words(path: str | Path) -> list[tuple[str, list[float]]] | None:
    """[(word, [x, y, w, h] fractions)] from an optional local OCR engine, or None if there
    is none installed."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
        try:
            import easyocr  # type: ignore[import-not-found]
            import numpy as np

            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            out = []
            for box, text, conf in reader.readtext(np.asarray(Image.open(path).convert("RGB"))):
                if conf < 0.3:
                    continue
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                parts = text.split()
                x0, x1 = min(xs), max(xs)
                for k, part in enumerate(parts):  # spread words evenly over the line box
                    a = x0 + (x1 - x0) * k / len(parts)
                    b = x0 + (x1 - x0) * (k + 1) / len(parts)
                    out.append((part, [a / W, min(ys) / H, (b - a) / W, (max(ys) - min(ys)) / H]))
            return out
        except ImportError:
            pass
        if shutil.which("tesseract"):
            import pytesseract  # type: ignore[import-not-found]

            d = pytesseract.image_to_data(Image.open(path), output_type=pytesseract.Output.DICT)
            return [(t, [x / W, y / H, w / W, h / H]) for t, x, y, w, h, c in zip(
                d["text"], d["left"], d["top"], d["width"], d["height"], d["conf"], strict=True)
                if t.strip() and float(c) > 30]
    except Exception:  # noqa: BLE001 - optional engine
        return None
    return None


def box_from_words(words: list[tuple[str, list[float]]], narration: str) -> list[float] | None:
    """Union box (fractions) of the OCR words that appear in the narration, if they form a
    coherent cluster (one text line / block); else None."""
    want = content_words(narration)
    hit = [b for w, b in words if _clean(w) in want]
    if not hit:
        return None
    # keep the densest cluster: words within 1.5 line heights vertically of the first hit
    hit.sort(key=lambda b: (b[1], b[0]))
    lh = max(b[3] for b in hit) or 0.03
    base = hit[0]
    cl = [b for b in hit if abs(b[1] - base[1]) <= 1.5 * lh + 0.02]
    x0, y0 = min(b[0] for b in cl), min(b[1] for b in cl)
    x1, y1 = max(b[0] + b[2] for b in cl), max(b[1] + b[3] for b in cl)
    return [x0, y0, x1 - x0, y1 - y0]


def sane_box(box: list[float] | None) -> list[float] | None:
    """Pad, clamp and reject boxes that would cover the title/caption or most of the picture."""
    if not box or len(box) != 4:
        return None
    x, y, w, h = (float(v) for v in box)
    if w <= 0.01 or h <= 0.008:
        return None
    x, y, w, h = x - PAD, y - PAD, w + 2 * PAD, h + 2 * PAD
    x, y = max(x, 0.0), max(y, 0.0)
    w, h = min(w, 1 - x), min(h, 1 - y)
    cy = y + h / 2
    if cy < TITLE_BAND or cy > CAPTION_BAND or w * h > MAX_AREA:
        return None
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


# --------------------------------------------------------------------------- VLM route

_VLM_BOX = {"type": "object", "required": ["found"], "properties": {
    "found": {"type": "boolean"}, "text": {"type": "string"},
    "x1": {"type": "integer"}, "y1": {"type": "integer"},
    "x2": {"type": "integer"}, "y2": {"type": "integer"}}}
_VLM_READ = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}


def _vlm(base: str, model: str, img_b64: str, prompt: str, schema: dict[str, Any],
         timeout: float = 90.0) -> dict[str, Any] | None:
    import json

    payload = {"model": model, "stream": False, "format": schema, "keep_alive": "2m",
               "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
               "options": {"temperature": 0, "num_predict": 120}}
    try:
        r = httpx.post(f"{base.rstrip('/')}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _b64(im: Any, max_side: int = 1024) -> tuple[str, tuple[int, int]]:
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), im.size


def vlm_box(path: str | Path, narration: str, base: str = "http://localhost:11434",
            model: str = "qwen2.5vl:3b") -> list[float] | None:
    """VLM-proposed bbox of the diagram text the narration is about, validated by reading the
    cropped region back (it must contain a narrated word/number). None if unsure."""
    from PIL import Image

    want = content_words(narration)
    if not want:
        return None
    with Image.open(path) as src:
        img = src.convert("RGB")
    b64, (w, h) = _b64(img)
    ans = _vlm(base, model, b64,
               f"This is a diagram. Narration: \"{narration}\". Find the label or text in the "
               f"image that the narration is talking about. Give its bounding box in pixels of "
               f"this {w}x{h} image (x1,y1,x2,y2), or found=false if no such text is visible.",
               _VLM_BOX)
    if not ans or not ans.get("found"):
        return None
    x1, y1, x2, y2 = (ans.get(k) for k in ("x1", "y1", "x2", "y2"))
    if None in (x1, y1, x2, y2) or x2 <= x1 or y2 <= y1:
        return None
    box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
    if not all(-0.01 <= v <= 1.01 for v in box[:2]) or box[2] > 1 or box[3] > 1:
        return None
    W, H = img.size
    m = 0.01
    crop = img.crop((max(round((box[0] - m) * W), 0), max(round((box[1] - m) * H), 0),
                     min(round((box[0] + box[2] + m) * W), W),
                     min(round((box[1] + box[3] + m) * H), H)))
    cb64, _ = _b64(crop, 768)
    read = _vlm(base, model, cb64, "Read all the text in this image crop, exactly.", _VLM_READ)
    got = {_clean(t) for t in (read or {}).get("text", "").split()}
    if not got & want:
        return None
    return box


def locate(path: str | Path, narration: str, base: str = "http://localhost:11434",
           model: str = "qwen2.5vl:3b", use_vlm: bool = True) -> dict[str, Any] | None:
    """A trustworthy highlight dict for the diagram text the narration mentions, or None."""
    words = ocr_words(path)
    box: list[float] | None = None
    source = "ocr"
    if words:
        box = box_from_words(words, narration)
    elif use_vlm:
        box, source = vlm_box(path, narration, base, model), "vlm"
    box = sane_box(box)
    if box is None:
        return None
    return {"shape": "box", "x": box[0], "y": box[1], "w": box[2], "h": box[3],
            "source": source}
