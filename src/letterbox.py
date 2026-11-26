"""Letterbox detection: find black bars (even ones carrying burned-in caption text).

``cropdetect`` ignores bars that contain bright text; here a bar is a top/bottom band whose
rows are mostly near-black (text only covers a minority of pixels) and that ends in a sharp
edge against the content. Frames are sampled across the clip; the bar must be present in
most of them. ``crop_box`` is ``[x, y, w, h]`` in the source's pixel coordinates.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

DARK = 24            # luma <= DARK counts as "black"
ROW_DARK_FRAC = 0.55  # a bar row: >= this share of dark pixels (caption glyphs fill the rest)
PURE_FRAC = 0.97     # a pure black row (the bar's inner edge)
EDGE_DARK_MAX = 0.80  # ...or the share of black pixels must drop to at most this under the bar
EDGE_STEP = 25       # mean luma jump required between the bar and the first content rows
MIN_BAR_FRAC = 0.04  # ignore bars thinner than 4 % of the height
MAX_BAR_FRAC = 0.35  # one bar never exceeds 35 %
MIN_AGREE = 0.6      # share of frames that must agree on a bar
_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _bar_height(luma: np.ndarray) -> int:
    """Height in rows of the dark bar at the TOP of ``luma`` (0 if none)."""
    h = luma.shape[0]
    max_rows = int(h * MAX_BAR_FRAC)
    frac = (luma[:max_rows] <= DARK).mean(axis=1)
    mostly = frac >= ROW_DARK_FRAC  # caption rows are mostly black with light glyphs
    n = 0
    while n < max_rows and mostly[n]:
        n += 1
    # trim back to the last pure-black row so dark content under the bar is not swallowed
    pure = np.nonzero(frac[:n] >= PURE_FRAC)[0]
    n = int(pure[-1]) + 1 if pure.size else 0
    if n < max(2, int(h * MIN_BAR_FRAC)):
        return 0
    # sharp edge: the rows right below the bar must be clearly brighter than the bar's rows
    # (a dark scene fading to black has no step, a letterbox does)
    k = max(3, h // 100)
    if n + k > h:
        return 0
    step = luma[n:n + k].mean() - luma[n - k:n].mean()
    sharp = step >= EDGE_STEP or (luma[n:n + k] <= DARK).mean() <= EDGE_DARK_MAX
    if not sharp:
        return 0
    return n


def detect_crop(frames: list[np.ndarray]) -> list[int] | None:
    """``frames``: 2-D luma arrays (same size). Returns ``[x, y, w, h]`` or None."""
    if not frames:
        return None
    h, w = frames[0].shape
    tops = [_bar_height(f) for f in frames]
    bots = [_bar_height(f[::-1]) for f in frames]

    def agree(vals: list[int]) -> int:
        hits = sorted(v for v in vals if v > 0)
        if len(hits) < max(1, int(np.ceil(len(vals) * MIN_AGREE))):
            return 0
        return hits[0]  # most conservative (smallest) bar

    top, bot = agree(tops), agree(bots)
    # letterbox bars come in pairs of similar size; a lone dark band is just a dark scene/poster
    if not top or not bot or max(top, bot) > 2.5 * min(top, bot):
        return None
    if h - top - bot < h * 0.3:
        return None
    return [0, top, w, h - top - bot]


def _frames_video(path: Path, n: int, dur: float | None) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for k in range(n):
        t = (dur or 0) * (k + 0.5) / n
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
               "-f", "image2pipe", "-vcodec", "png", "-"]
        r = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        if r.returncode == 0 and r.stdout:
            out.append(np.asarray(Image.open(io.BytesIO(r.stdout)).convert("L")))
    return out


def detect_for_file(path: str | Path, duration: float | None = None, n: int = 5) -> list[int] | None:
    p = Path(path)
    try:
        if p.suffix.lower() in _VIDEO_EXT:
            frames = _frames_video(p, n, duration)
        else:
            with Image.open(p) as im:
                frames = [np.asarray(im.convert("L"))]
    except Exception:  # noqa: BLE001 - detection is best effort
        return None
    return detect_crop(frames)


def crop_filter(box: list[int], src_size: tuple[int, int], ref_size: tuple[int, int]) -> str:
    """ffmpeg crop expression for ``box`` (given in ``ref_size`` pixels) on a ``src_size`` file."""
    sx, sy = src_size[0] / ref_size[0], src_size[1] / ref_size[1]
    x, y, w, h = box
    ev = lambda v: round(v) // 2 * 2
    return f"crop={ev(w * sx)}:{ev(h * sy)}:{ev(x * sx)}:{ev(y * sy)}"
