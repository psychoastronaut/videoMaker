"""Local AI upscaling of low-resolution photos (Real-ESRGAN x4plus via spandrel).

Optional extra (``pip install -e .[upscale]``). Photos of 550-700 px get stretched to 1080p by
the renderer and look soft; this writes a sharper, bigger copy to ``work/upscaled/`` and records
it in ``local_assets.json`` (``upscaled_path`` ...). ``src.media.AssetResolver`` then prefers it.

Design: tiled fp16 inference with feathered overlap (a whole frame OOMs on a shared GPU), smaller
tiles then CPU on OOM, VRAM freed afterwards, result cached by content hash (idempotent), the
4x output Lanczos-resized to ``min(1920, 2x original)`` and optionally blended with a plain
Lanczos resize (``upscale.blend``) to tame plastic textures. Original files are never touched.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import time
from pathlib import Path
from typing import Any, Self

import numpy as np
from PIL import Image, ImageOps

from src.common import is_offline

MODEL_NAME = "RealESRGAN_x4plus"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"  # BSD-3-Clause
DEFAULT_MODEL_DIR = "~/.cache/videomaker/models"
SCALE = 4
OVERLAP = 16  # input pixels shared between neighbouring tiles
TILE_LADDER = (256, 192, 128, 64)

DEFAULTS: dict[str, Any] = {
    "enabled": "auto",        # auto = on when the [upscale] extra is installed; true | false
    "model_dir": DEFAULT_MODEL_DIR,
    "min_long_side": 1400,    # images whose long side is already >= this are skipped
    "max_long_side": 1920,    # never upscale beyond this
    "max_factor": 2.0,        # target long side <= max_factor x original
    "tile": 256,              # starting tile (input px); shrinks on OOM, then CPU
    "blend": 0.7,             # 1.0 = pure AI result, lower mixes in a plain Lanczos resize
    "schematics": False,      # also upscale raw/schematics (diagrams) when true
    "flat_graphics": False,   # also upscale flat-colour graphics / logos when true
}


class UpscaleUnavailable(RuntimeError):
    """Deps or weights missing (and cannot be fetched): the caller should warn and skip."""


# --------------------------------------------------------------------------- config / deps


def upscale_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    cfg.update((config or {}).get("upscale") or {})
    return cfg


def deps_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "spandrel", "einops"))


def is_enabled(cfg: dict[str, Any]) -> bool:
    en = cfg.get("enabled", "auto")
    if isinstance(en, str):
        en = en.strip().lower()
        if en == "auto":
            return deps_available()
        return en in {"true", "yes", "on", "1"}
    return bool(en)


def model_path(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("model_dir") or DEFAULT_MODEL_DIR)).expanduser() / f"{MODEL_NAME}.pth"


def ensure_model(cfg: dict[str, Any]) -> Path:
    """Path of the cached weights, downloading them once (sha256-verified) if missing."""
    path = model_path(cfg)
    if path.is_file():
        return path
    if is_offline():
        raise UpscaleUnavailable(f"{path.name} not cached and VIDEOMAKER_OFFLINE is set")
    import httpx

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pth.part")
    h = hashlib.sha256()
    try:
        with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
    except (httpx.HTTPError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise UpscaleUnavailable(f"cannot download {MODEL_URL} ({type(e).__name__}: {e})") from e
    if h.hexdigest() != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise UpscaleUnavailable("downloaded weights failed the sha256 check")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- decisions


def _looks_flat(img: Image.Image) -> bool:
    """Flat-colour graphic (logo, diagram, UI): a handful of colours cover nearly all pixels."""
    small = img.convert("RGB").resize((192, 192), Image.NEAREST)
    arr = np.asarray(small).reshape(-1, 3)
    _, counts = np.unique(arr, axis=0, return_counts=True)
    top = np.sort(counts)[::-1][:12].sum()
    return bool(top / arr.shape[0] > 0.9)


def target_size(w: int, h: int, cfg: dict[str, Any]) -> tuple[int, int]:
    """Output size: long side = min(max_long_side, max_factor x original), never smaller."""
    long_side = max(w, h)
    target = min(int(cfg["max_long_side"]), round(long_side * float(cfg["max_factor"])))
    if target <= long_side:
        return w, h
    k = target / long_side
    return max(1, round(w * k)), max(1, round(h * k))


def skip_reason(path: Path, asset_type: str, cfg: dict[str, Any]) -> str | None:
    """Why ``path`` should not be upscaled, or None."""
    if asset_type == "schematic" and not cfg.get("schematics"):
        return "schematic"
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            if max(w, h) >= int(cfg["min_long_side"]):
                return f"already {max(w, h)}px"
            if target_size(w, h, cfg) == (w, h):
                return "no upscaling needed"
            if not cfg.get("flat_graphics") and _looks_flat(im):
                return "flat graphic"
    except Exception as e:  # noqa: BLE001 - undecodable images are handled elsewhere
        return f"unreadable ({type(e).__name__})"
    return None


def _cache_key(path: Path, cfg: dict[str, Any]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(f"{MODEL_NAME}|{cfg['max_long_side']}|{cfg['max_factor']}|{cfg['blend']}".encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- engine


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


class Upscaler:
    """Holds the loaded model; use as a context manager so VRAM is released afterwards."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model: Any = None
        self.device = "cpu"
        self.tile = int(cfg.get("tile", 256))
        self.peak_vram_mib = 0.0

    def __enter__(self) -> Self:
        if not deps_available():
            raise UpscaleUnavailable("install the extra: pip install -e '.[upscale]'")
        weights = ensure_model(self.cfg)
        import torch
        from spandrel import ModelLoader

        desc = ModelLoader().load_from_file(str(weights))
        self.model = desc.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.reset_peak_memory_stats()
            try:
                self.model.to("cuda").half()
            except Exception as e:
                if not _is_oom(e):
                    raise
                self._to_cpu()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            return

    def _to_cpu(self) -> None:
        import torch

        self.model.float().to("cpu")
        self.device = "cpu"
        gc.collect()
        torch.cuda.empty_cache()

    def _run_tile(self, tile: np.ndarray) -> np.ndarray:
        import torch

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        x = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype) / 255.0
        with torch.inference_mode():
            y = self.model(x)
        out = y.squeeze(0).float().clamp_(0, 1).permute(1, 2, 0).cpu().numpy()
        if self.device == "cuda":
            self.peak_vram_mib = max(self.peak_vram_mib,
                                     torch.cuda.max_memory_allocated() / 2**20)
        return out

    def enhance_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 -> float32 (H*4)x(W*4)x3 in [0,1], tiled with feathered overlap."""
        while True:
            try:
                return self._enhance(rgb, self.tile)
            except Exception as e:
                if not _is_oom(e):
                    raise
                gc.collect()
                import torch

                if self.device == "cuda":
                    torch.cuda.empty_cache()
                smaller = [t for t in TILE_LADDER if t < self.tile]
                if smaller:
                    self.tile = smaller[0]
                elif self.device == "cuda":
                    self._to_cpu()
                    self.tile = 128
                else:
                    raise

    def _enhance(self, rgb: np.ndarray, tile: int) -> np.ndarray:
        h, w, _ = rgb.shape
        s = SCALE
        out = np.zeros((h * s, w * s, 3), np.float32)
        wsum = np.zeros((h * s, w * s, 1), np.float32)
        stride = tile - OVERLAP
        ys = list(range(0, max(h - OVERLAP, 1), stride))
        xs = list(range(0, max(w - OVERLAP, 1), stride))
        ramp = OVERLAP * s
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                y0c, x0c = max(min(y0, y1 - tile), 0), max(min(x0, x1 - tile), 0)
                sr = self._run_tile(np.ascontiguousarray(rgb[y0c:y1, x0c:x1]))
                th, tw = sr.shape[:2]
                wy = np.ones(th, np.float32)
                wx = np.ones(tw, np.float32)
                edge = np.linspace(1 / ramp, 1, ramp, dtype=np.float32)
                if y0c > 0:
                    wy[:ramp] = edge
                if y1 < h:
                    wy[-ramp:] = np.minimum(wy[-ramp:], edge[::-1])
                if x0c > 0:
                    wx[:ramp] = edge
                if x1 < w:
                    wx[-ramp:] = np.minimum(wx[-ramp:], edge[::-1])
                wgt = (wy[:, None] * wx[None, :])[..., None]
                oy, ox = y0c * s, x0c * s
                out[oy:oy + th, ox:ox + tw] += sr * wgt
                wsum[oy:oy + th, ox:ox + tw] += wgt
        return out / np.maximum(wsum, 1e-6)

    def upscale(self, src: Path, dst: Path) -> dict[str, Any]:
        """Upscale ``src`` into the PNG ``dst``; returns size / timing info."""
        t0 = time.perf_counter()
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            tw, th = target_size(w, h, self.cfg)
            alpha = im.getchannel("A") if im.mode in ("RGBA", "LA", "PA") or \
                "transparency" in im.info else None
            if alpha is None and im.mode in ("RGBA", "LA", "PA"):
                alpha = im.convert("RGBA").getchannel("A")
            rgb_im = im.convert("RGBA").convert("RGB") if alpha is not None else im.convert("RGB")
            rgb = np.asarray(rgb_im)
        sr = self.enhance_rgb(rgb)
        sr_img = Image.fromarray((sr * 255.0 + 0.5).astype(np.uint8))
        sr_img = sr_img.resize((tw, th), Image.LANCZOS)
        blend = float(self.cfg.get("blend", 1.0))
        if blend < 1.0:
            plain = rgb_im.resize((tw, th), Image.LANCZOS)
            sr_img = Image.blend(plain, sr_img, blend)
        if alpha is not None:
            sr_img.putalpha(alpha.resize((tw, th), Image.LANCZOS))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        sr_img.save(tmp, format="PNG", compress_level=3)
        tmp.replace(dst)
        return {"width": tw, "height": th, "seconds": round(time.perf_counter() - t0, 2),
                "device": self.device, "tile": self.tile,
                "peak_vram_mib": round(self.peak_vram_mib)}


# --------------------------------------------------------------------------- public API


def cached_path(path: Path, out_dir: Path, cfg: dict[str, Any]) -> Path:
    return out_dir / f"{path.stem}-{_cache_key(path, cfg)}.png"


def upscale_image(path: str | Path, out_path: str | Path | None = None,
                  cfg: dict[str, Any] | None = None, *, work_dir: Path | None = None,
                  asset_type: str = "image", engine: Upscaler | None = None,
                  ) -> dict[str, Any] | None:
    """Upscale one image if worthwhile. Returns ``{"path", "width", "height", ...}`` or None.

    ``out_path`` defaults to a content-hashed file in ``work_dir/upscaled`` (idempotent: an
    existing result is reused without loading the model). Pass ``engine`` to share one loaded
    model across many images. Raises ``UpscaleUnavailable`` when deps/weights are missing.
    """
    cfg = {**DEFAULTS, **(cfg or {})}  # the ``upscale:`` section (partial is fine)
    src = Path(path)
    if skip_reason(src, asset_type, cfg):
        return None
    if out_path is not None:
        dst = Path(out_path)
    else:
        if work_dir is None:
            raise ValueError("work_dir or out_path required")
        dst = cached_path(src, work_dir / "upscaled", cfg)
    if dst.is_file():
        with Image.open(dst) as im:
            return {"path": str(dst), "width": im.width, "height": im.height, "cached": True}
    owns = engine is None
    eng = engine or Upscaler(cfg).__enter__()
    try:
        info = eng.upscale(src, dst)
    finally:
        if owns:
            eng.close()
    return {"path": str(dst), "cached": False, **info}


def apply_to_index(index: dict[str, Any], work_dir: Path, config: dict[str, Any],
                   log: Any = print) -> dict[str, Any]:
    """Upscale the eligible images of an asset-index dict in place; returns a summary.

    Adds ``upscaled_path``, ``upscaled_from``, ``original_width/height`` and rewrites
    ``width/height`` to the new size. Missing deps/weights => warning and no change.
    """
    cfg = upscale_config(config)
    summary: dict[str, Any] = {"upscaled": [], "skipped": 0, "enabled": is_enabled(cfg)}
    assets = [a for a in index.get("assets", []) if a.get("asset_type") in ("image", "schematic")]
    if not summary["enabled"]:
        if str(cfg.get("enabled", "auto")).lower() == "auto":
            log("  [yellow]Upscaling skipped: optional deps missing "
                "(pip install -e '.[upscale]')[/]")
        return summary
    todo = [a for a in assets if not skip_reason(Path(a["filepath"]), a["asset_type"], cfg)]
    summary["skipped"] = len(assets) - len(todo)
    if not todo:
        return summary
    out_dir = work_dir / "upscaled"
    engine: Upscaler | None = None
    try:
        for a in todo:
            src = Path(a["filepath"])
            dst = cached_path(src, out_dir, cfg)
            if not dst.is_file() and engine is None:
                engine = Upscaler(cfg).__enter__()
            res = upscale_image(src, dst, cfg, asset_type=a["asset_type"], engine=engine)
            if not res:
                continue
            a.setdefault("original_width", a.get("width"))
            a.setdefault("original_height", a.get("height"))
            a["upscaled_path"], a["upscaled_from"] = res["path"], str(src)
            a["width"], a["height"] = res["width"], res["height"]
            summary["upscaled"].append({"id": a["id"], **res})
            log(f"  Upscaled {src.name}: {a['original_width']}x{a['original_height']} -> "
                f"{res['width']}x{res['height']}" +
                (" (cached)" if res.get("cached") else f" in {res['seconds']}s"))
    except UpscaleUnavailable as e:
        log(f"  [yellow]Upscaling skipped: {e}[/]")
    finally:
        if engine is not None:
            engine.close()
    return summary


def upscale_stock_image(path: str | Path, work_dir: Path,
                        config: dict[str, Any] | None = None) -> str:
    """Helper for the stock stage: path to use for a downloaded stock image.

    Returns the upscaled copy when upscaling is enabled and worthwhile, else ``path`` unchanged
    (never raises for missing deps/weights).
    """
    cfg = upscale_config(config)
    if not is_enabled(cfg):
        return str(path)
    try:
        res = upscale_image(path, None, cfg, work_dir=work_dir)
    except UpscaleUnavailable:
        return str(path)
    return res["path"] if res else str(path)
