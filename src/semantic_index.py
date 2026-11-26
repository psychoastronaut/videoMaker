"""Stage 2a: semantic index of the local assets.

Clips are split into shots (PySceneDetect; fixed windows when it is missing), 1-3 keyframes per
shot are embedded by the active ``SemanticBackend``; images/schematics are embedded directly.
Results are cached per file hash in ``work/semantic/`` and written to
``work/semantic_index.json`` (+ ``work/semantic_index.npy`` with the vectors).

Retrieval scores an asset by its best frame (MaxFrame) and merges adjacent high-scoring frames
into a trim interval (``start``/``end`` in seconds; ``None`` for stills).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from src.common import read_json, write_json
from src.semantic import SemanticBackend, get_backend, semantic_config

console = Console()

INDEX_JSON = "semantic_index.json"
INDEX_NPY = "semantic_index.npy"
WINDOW = 3.0  # fallback shot length when scenedetect is unavailable


# --------------------------------------------------------------------------- shots/frames


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def fixed_shots(duration: float, window: float = WINDOW) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    n = max(1, round(duration / window))
    step = duration / n
    return [(round(i * step, 3), round((i + 1) * step, 3)) for i in range(n)]


def detect_shots(path: Path, duration: float, threshold: float = 27.0
                 ) -> tuple[list[tuple[float, float]], str]:
    """Shots as (start, end) seconds. Returns (shots, method)."""
    try:
        from scenedetect import ContentDetector, detect

        scenes = detect(str(path), ContentDetector(threshold=threshold, min_scene_len=15))
        shots = [(round(s.get_seconds(), 3), round(e.get_seconds(), 3)) for s, e in scenes]
        shots = [(s, e) for s, e in shots if e - s > 0.2]
        if shots:
            # very long shots are split too so the trim interval stays meaningful
            out: list[tuple[float, float]] = []
            for s, e in shots:
                if e - s > 8.0:
                    out += [(s + a, s + b) for a, b in fixed_shots(e - s, 4.0)]
                else:
                    out.append((s, e))
            return out, "scenedetect"
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001 - never let shot detection break indexing
        console.print(f"  [yellow]scenedetect failed on {path.name}: {e}[/]")
    return fixed_shots(duration), "fixed"


def keyframe_times(start: float, end: float) -> list[float]:
    length = end - start
    fracs = (0.5,) if length < 3.0 else (0.25, 0.75) if length < 8.0 else (1 / 6, 0.5, 5 / 6)
    return [round(start + f * length, 3) for f in fracs]


def extract_frame(path: Path, t: float, out: Path) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
           "-vf", "scale=-2:480", "-q:v", "3", str(out)]
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return False
    return out.is_file() and out.stat().st_size > 0


# --------------------------------------------------------------------------- build


def _embed_asset(asset: dict[str, Any], backend: SemanticBackend, tmp: Path, cfg: dict[str, Any]
                 ) -> tuple[list[dict[str, Any]], np.ndarray]:
    path = Path(asset["filepath"])
    frames: list[dict[str, Any]] = []
    paths: list[Path] = []
    if asset["asset_type"] == "clip":
        dur = float(asset.get("duration") or 0.0)
        shots, _ = detect_shots(path, dur, float(cfg.get("shot_threshold", 27.0)))
        for si, (s, e) in enumerate(shots):
            for t in keyframe_times(s, e):
                out = tmp / f"{asset['id']}_{len(paths)}.jpg"
                if extract_frame(path, min(t, max(dur - 0.05, 0.0)), out):
                    paths.append(out)
                    frames.append({"t": t, "shot": [s, e], "shot_index": si})
    else:
        paths.append(path)
        frames.append({"t": None, "shot": None, "shot_index": 0})
    if not paths:
        return [], np.zeros((0, 1), np.float32)
    vecs, caps = backend.embed_frames(paths)
    for f, c in zip(frames, caps, strict=True):
        f["caption"] = c
    return frames, vecs


def build_index(work: Path, backend: SemanticBackend, assets: list[dict[str, Any]],
                cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Embed every asset (cached by file hash); write the JSON + NPY index; return the JSON."""
    cfg = cfg or {}
    cache = work / "semantic"
    cache.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    chunks: list[np.ndarray] = []
    row = 0
    tmp = Path(tempfile.mkdtemp(prefix="semidx_"))
    hits = 0
    try:
        for a in assets:
            p = Path(a["filepath"])
            if not p.is_file():
                continue
            key = f"{file_hash(p)}.{backend.tag}"
            meta_f, vec_f = cache / f"{key}.json", cache / f"{key}.npy"
            if meta_f.is_file() and vec_f.is_file():
                frames = read_json(meta_f)["frames"]
                vecs = np.load(vec_f)
                hits += 1
            else:
                console.print(f"  indexing {a['id']} ...")
                frames, vecs = _embed_asset(a, backend, tmp, cfg)
                if not frames:
                    continue
                write_json(meta_f, {"frames": frames})
                np.save(vec_f, vecs)
            for i, f in enumerate(frames):
                f["row"] = row + i
            row += len(frames)
            chunks.append(vecs)
            entries[a["id"]] = {"type": a["asset_type"], "duration": a.get("duration"),
                                "frames": frames}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    all_vecs = np.concatenate(chunks) if chunks else np.zeros((0, 1), np.float32)
    np.save(work / INDEX_NPY, all_vecs)
    data = {"backend": backend.name, "model": backend.model, "tag": backend.tag,
            "dim": int(all_vecs.shape[1]), "frames": int(all_vecs.shape[0]), "assets": entries}
    write_json(work / INDEX_JSON, data)
    console.print(f"  Semantic index: {len(entries)} assets, {all_vecs.shape[0]} frames "
                  f"({hits} from cache) [{backend.name}]")
    return data


def run(episode_id: str, work_dir: Path) -> dict[str, Any]:
    """Stage 2a. Skips (with a warning) when no semantic backend is available."""
    from src.common import load_config

    config = load_config()
    backend = get_backend(config)
    if backend is None:
        console.print("  [yellow]Semantic index skipped: planner will use keyword matching[/]")
        return {}
    assets = read_json(work_dir / "local_assets.json").get("assets", [])
    try:
        return build_index(work_dir, backend, assets, semantic_config(config))
    finally:
        backend.free()


# --------------------------------------------------------------------------- retrieval


@dataclass
class Hit:
    asset_id: str
    score: float
    start: float | None = None
    end: float | None = None
    caption: str = ""
    asset_type: str = ""
    rank: float = 0.0  # score + keyword prior (ordering only; ``score`` stays the raw cosine)

    def as_dict(self) -> dict[str, Any]:
        return {"asset_id": self.asset_id, "score": round(self.score, 4), "start": self.start,
                "end": self.end, "caption": self.caption, "type": self.asset_type}


@dataclass
class SemanticIndex:
    data: dict[str, Any]
    vecs: np.ndarray
    merge_margin: float = 0.03
    _rows: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    # optional per-asset ``media_start`` (sidecar hint): shots ending before it are ignored
    hints: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, work: Path, merge_margin: float = 0.03) -> SemanticIndex | None:
        jp, npy = work / INDEX_JSON, work / INDEX_NPY
        if not (jp.is_file() and npy.is_file()):
            return None
        try:
            return cls(read_json(jp), np.load(npy), merge_margin)
        except (json.JSONDecodeError, OSError, ValueError):
            return None

    @property
    def tag(self) -> str:
        return str(self.data.get("tag", ""))

    def asset_ids(self) -> list[str]:
        return list(self.data.get("assets", {}))

    def score_asset(self, asset_id: str, q: np.ndarray) -> Hit | None:
        """MaxFrame score + merged trim interval for one asset against query vector ``q``."""
        entry = self.data.get("assets", {}).get(asset_id)
        if not entry or not entry["frames"]:
            return None
        frames = entry["frames"]
        hint = self.hints.get(asset_id)
        if hint and entry["type"] == "clip":
            kept = [f for f in frames if f["shot"][1] > hint]
            frames = kept or frames
        sims = np.array([float(self.vecs[f["row"]] @ q) for f in frames])
        b = int(sims.argmax())
        best = float(sims[b])
        if entry["type"] != "clip":
            return Hit(asset_id, best, None, None, frames[b].get("caption", ""), entry["type"])
        lo = hi = b  # merge adjacent frames scoring close to the best one
        while lo > 0 and sims[lo - 1] >= best - self.merge_margin:
            lo -= 1
        while hi < len(frames) - 1 and sims[hi + 1] >= best - self.merge_margin:
            hi += 1
        start = min(f["shot"][0] for f in frames[lo:hi + 1])
        end = max(f["shot"][1] for f in frames[lo:hi + 1])
        return Hit(asset_id, best, round(start, 3), round(end, 3),
                   frames[b].get("caption", ""), "clip")

    def search(self, q: np.ndarray, types: tuple[str, ...] = ("clip", "image", "schematic"),
               k: int = 5, exclude: set[str] | None = None,
               prior: dict[str, float] | None = None) -> list[Hit]:
        """Top-k assets by MaxFrame score (+ optional weak keyword ``prior`` per asset id)."""
        hits = []
        for aid, entry in self.data.get("assets", {}).items():
            if entry["type"] not in types or (exclude and aid in exclude):
                continue
            h = self.score_asset(aid, q)
            if h:
                h.rank = h.score + (prior or {}).get(aid, 0.0)
                hits.append(h)
        hits.sort(key=lambda h: -h.rank)
        return hits[:k]


def fit_window(hit: Hit, beat_dur: float, clip_dur: float | None) -> tuple[float | None,
                                                                          float | None]:
    """Choose [media_start, media_end] of length ~beat_dur inside the clip, centred on the hit."""
    if hit.start is None or not clip_dur or clip_dur <= 0:
        return None, None
    length = min(beat_dur + 0.3, clip_dur)
    if hit.end is not None and hit.end - hit.start >= length:
        start = hit.start
    else:
        start = (hit.start + (hit.end or hit.start)) / 2 - length / 2
    start = max(0.0, min(start, clip_dur - length))
    return round(start, 3), round(start + length, 3)


# --------------------------------------------------------------------------- stills (stock)


def first_frame(path: Path, tmp: Path, name: str) -> Path | None:
    """The image itself, or a frame near the start of a video."""
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return path
    out = tmp / f"{name}.jpg"
    return out if extract_frame(path, 0.5, out) else None


def embed_stills(work: Path, backend: SemanticBackend, items: dict[str, Path]
                 ) -> dict[str, tuple[np.ndarray, str]]:
    """Embed arbitrary stills/videos (stock candidates) with the per-hash cache.
    Returns {key: (vector, caption)}; unreadable files are omitted."""
    cache = work / "semantic"
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[np.ndarray, str]] = {}
    tmp = Path(tempfile.mkdtemp(prefix="semstill_"))
    try:
        todo: list[tuple[str, Path, Path, Path]] = []
        for key, p in items.items():
            if not p.is_file():
                continue
            ck = f"{file_hash(p)}.still.{backend.tag}"
            mf, vf = cache / f"{ck}.json", cache / f"{ck}.npy"
            if mf.is_file() and vf.is_file():
                out[key] = (np.load(vf)[0], read_json(mf).get("caption", ""))
            else:
                todo.append((key, p, mf, vf))
        paths, keep = [], []
        for key, p, mf, vf in todo:
            fr = first_frame(p, tmp, key.replace("/", "_"))
            if fr is not None:
                paths.append(fr)
                keep.append((key, mf, vf))
        if paths:
            vecs, caps = backend.embed_frames(paths)
            for (key, mf, vf), v, c in zip(keep, vecs, caps, strict=True):
                np.save(vf, v[None, :])
                write_json(mf, {"caption": c})
                out[key] = (v, c)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out
