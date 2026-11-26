"""Virtual sub-assets: when an asset has to be shown again, every reuse gets a different view.

* photo (large)  -> distinct detail crop (``crop_region`` = [x, y, w, h] fractions), chosen among
  a few candidate regions by SigLIP score of the region embedding against the beat's query;
* clip           -> a different, non-overlapping ``media_start/media_end`` segment (shot
  boundaries from the semantic index, best shot for the beat first);
* diagram        -> its next sidecar ``highlights`` region (never invented).
The first use keeps the plain asset; ``diversify_views`` only touches the 2nd, 3rd... use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.schemas import VisualTemplate

CROPPABLE = (VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.PUNCH_IN_EMPHASIS)
MIN_CROP_PX = 1700  # a detail crop must still be this wide (output 1920) to stay sharp
ANCHORS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5), (0.0, 0.5), (1.0, 0.5))


def crop_scale(width: int | None, scale: float = 0.6) -> float | None:
    """Fraction of the picture a detail crop keeps (None when the photo is too small/portrait)."""
    if not width:
        return None
    sc = max(scale, MIN_CROP_PX / width)
    return round(sc, 3) if sc <= 0.85 else None


def candidate_regions(scale: float) -> list[list[float]]:
    return [[round(ax * (1 - scale), 3), round(ay * (1 - scale), 3), scale, scale]
            for ax, ay in ANCHORS]


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def pick_region(scores: list[float], regions: list[list[float]], used: list[list[float]],
                max_iou: float = 0.45) -> list[float] | None:
    """Best-scoring region that does not overlap (IoU) a view already shown."""
    order = sorted(range(len(regions)), key=lambda i: -scores[i])
    for i in order:
        if all(iou(regions[i], u) <= max_iou for u in used):
            return regions[i]
    return None


def free_window(used: list[tuple[float, float]], length: float, duration: float,
                shots: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """Window [s, s+length] inside the clip with the least overlap with ``used`` intervals;
    candidates start at each shot (best score first), then slide in 0.5 s steps. Returns None when
    even the best candidate overlaps more than 35 % of its length."""
    length = min(length, duration)
    if length <= 0:
        return None

    def overlap(s: float) -> float:
        return sum(max(0.0, min(s + length, e) - max(s, b)) for b, e in used) / length

    cands = [min(s, max(duration - length, 0.0)) for s, _, _ in sorted(shots, key=lambda t: -t[2])]
    cands += [i * 0.5 for i in range(int(max(duration - length, 0) / 0.5) + 1)]
    best = None
    for k, s in enumerate(cands):  # earlier = better score; strictly lower overlap wins
        o = overlap(s)
        if best is None or o < best[0] - 1e-9:
            best = (o, s)
        if o == 0 and k < len(shots):
            break
    if best is None or best[0] > 0.35:
        return None
    return round(best[1], 3), round(best[1] + length, 3)


def _shot_scores(index: Any, asset_id: str, qs: list[np.ndarray]
                 ) -> list[tuple[float, float, float]]:
    entry = index.data["assets"].get(asset_id) or {}
    best: dict[tuple[float, float], float] = {}
    for f in entry.get("frames", []):
        if not f.get("shot"):
            continue
        sc = max(float(index.vecs[f["row"]] @ q) for q in qs) if qs else 0.0
        key = (f["shot"][0], f["shot"][1])
        best[key] = max(best.get(key, -9.0), sc)
    return [(s, e, v) for (s, e), v in best.items()]


def _interval(b: Any, duration: float) -> tuple[float, float]:
    s = b.media_start or 0.0
    return s, (b.media_end if b.media_end else min(s + b.duration, duration))


def diversify_views(beats: list[Any], ctx: Any, index: Any, by_beat: dict[int, list[np.ndarray]],
                    backend: Any, cfg: dict[str, Any], work: Path,
                    clip_hints: Any = None) -> dict[str, int]:
    """Give every re-use of an asset a different view. Returns counts per kind."""
    from src.overlays import sidecar_highlights

    div = cfg.get("diversity") or {}
    stats = {"crop": 0, "segment": 0, "highlight": 0}
    seen: dict[str, list[Any]] = {}
    for b in sorted(beats, key=lambda x: x.start_time):
        main = b.source_id if b.primary_source.value == "local" else ""
        if main:
            seen.setdefault(main, []).append(b)
        if b.visual_template == VisualTemplate.SPLIT_SCHEMATIC and b.overlay_asset:
            seen.setdefault(b.overlay_asset, []).append(b)
    for aid, bs in seen.items():
        if len(bs) < 2:
            continue
        a = ctx.assets_by_id.get(aid) or {}
        kind = a.get("asset_type")
        if kind == "clip" and div.get("clip_segments", True):
            dur = float(a.get("duration") or 0)
            used = []
            for k, b in enumerate(bs):
                if b.source_id != aid:
                    continue
                if k and dur:
                    shots = _shot_scores(index, aid, by_beat.get(b.id, []))
                    win = free_window(used, b.duration, dur, shots)
                    if win and clip_hints is not None:
                        win = clip_hints(win[0], win[1] - win[0], dur, a)
                    if win:
                        b.media_start, b.media_end = win
                        stats["segment"] += 1
                if dur:
                    used.append(_interval(b, dur))
        elif kind == "image" and div.get("photo_crops", True):
            targets = [b for b in bs[1:] if b.source_id == aid and b.visual_template in CROPPABLE
                       and not b.crop_region]
            sc = crop_scale(a.get("width"), float(div.get("crop_scale", 0.6)))
            w, h = a.get("width") or 0, a.get("height") or 0
            if not targets or sc is None or not h or not 0.95 <= w / h <= 2.2:
                continue
            regions = candidate_regions(sc)
            vecs = _embed_regions(backend, a, regions, work / "semantic_cache" / "crops", aid)
            used = [[0.0, 0.0, 1.0, 1.0]]
            for b in targets:
                qs = by_beat.get(b.id, [])
                scores = [max((float(v @ q) for q in qs), default=0.0) for v in vecs] \
                    if len(vecs) == len(regions) else [0.0] * len(regions)
                r = pick_region(scores, regions, used)
                if r is not None:
                    b.crop_region = r
                    used.append(r)
                    stats["crop"] += 1
        elif kind == "schematic" and div.get("diagram_highlights", True):
            hl = sidecar_highlights(a) or []
            order = _ink_regions(a)  # busiest areas first: where the diagram has content
            for k, b in enumerate(bs):
                if b.highlights is not None:
                    continue
                if hl:
                    b.highlights = [hl[k % len(hl)]]
                elif k and order:
                    b.highlights = [{"shape": "box", "region": order[(k - 1) % len(order)]}]
                else:
                    continue
                stats["highlight"] += k > 0
    return stats


def _embed_regions(backend: Any, asset: dict[str, Any], regions: list[list[float]],
                   out_dir: Path, aid: str) -> np.ndarray:
    """Embed each candidate region (crop saved as a small JPEG) with the active backend."""
    try:
        from PIL import Image

        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        with Image.open(asset["filepath"]) as im:
            im = im.convert("RGB")
            W, H = im.size
            for i, (x, y, w, h) in enumerate(regions):
                c = im.crop((round(x * W), round(y * H), round((x + w) * W), round((y + h) * H)))
                c.thumbnail((768, 768))
                p = out_dir / f"{aid[:60]}_{i}.jpg"
                c.save(p, quality=88)
                paths.append(p)
        vecs, _ = backend.embed_frames(paths)
        return np.asarray(vecs)
    except Exception:  # noqa: BLE001 - views are optional; fall back to anchor order
        return np.zeros((0, 1))


def _ink_regions(asset: dict[str, Any], top: int = 4) -> list[str]:
    """Named diagram regions ordered by how much non-background content they hold."""
    try:
        from PIL import Image

        from src.overlays import REGIONS

        with Image.open(asset["filepath"]) as im:
            g = np.asarray(im.convert("L").resize((320, 180)), dtype=float)
        bg = float(np.median(g))
        ink = np.abs(g - bg) > 40
        h, w = ink.shape
        dens = {n: float(ink[int(y * h):int((y + hh) * h), int(x * w):int((x + ww) * w)].mean())
                for n, (x, y, ww, hh) in REGIONS.items()}
        return [n for n, d in sorted(dens.items(), key=lambda t: -t[1])[:top] if d > 0.04]
    except Exception:  # noqa: BLE001
        return []
