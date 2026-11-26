"""Stage 5: cutouts & graphic overlays.

For every beat this stage creates the overlay images its template needs (transparent subject
cutout via rembg, callout badge, lower-third) and writes their paths back into
``scene_plan.json`` so Stage 6 can pass them to ffmpeg as real inputs.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from rich.console import Console

from src import graphics
from src.common import load_config, probe_duration, read_json, write_json
from src.media import AssetResolver, ResolvedMedia
from src.schemas import ScenePlan

console = Console()

_session_cache: dict[str, Any] = {}


def run(episode_id: str, work_dir: Path) -> dict[str, Any]:
    """Run Stage 5: generate cutouts, badges and lower-thirds; persist paths in the plan."""
    plan_path = work_dir / "scene_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Scene plan not found: {plan_path}")
    if not (work_dir / "local_assets.json").is_file():
        raise FileNotFoundError(f"Asset index not found: {work_dir / 'local_assets.json'}")

    plan = ScenePlan.model_validate(read_json(plan_path))
    cfg = load_config()
    cut_cfg = cfg.get("cutouts", {})
    res = cfg.get("pipeline", {}).get("resolution", {})
    width, height = int(res.get("width", 1920)), int(res.get("height", 1080))
    resolver = AssetResolver(work_dir)

    out_dir = work_dir / "cutouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ready = out_dir / ".ready"
    ready.unlink(missing_ok=True)

    grid = graphics.make_grid_background(work_dir / "grid_background.png", width, height,
                                         float(cut_cfg.get("grid_bg_opacity", 0.15)))

    cutouts = badges = thirds = 0
    decisions: list[dict[str, Any]] = []
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        console.print(f"  [yellow]{msg}[/]")

    for beat in plan.beats:
        beat.cutout_path = beat.badge_path = beat.lower_third_path = None
        style = beat.beat_style.value
        template = beat.visual_template.value

        if template == "cutout_callout":
            subject, why = pick_subject(beat.model_dump(), resolver)
            if subject is None:
                warn(f"beat {beat.id}: no cutout ({why}); rendering without cutout")
            else:
                try:
                    path = _make_cutout(subject, beat.id, out_dir, cut_cfg, height, warn, decisions)
                    beat.cutout_path = str(path)
                    cutouts += 1
                    console.print(f"  beat {beat.id}: cutout {path.name}")
                except Exception as e:  # noqa: BLE001 - corrupt image, ffmpeg failure, ...
                    warn(f"beat {beat.id}: cutout failed for {Path(subject.path).name} "
                         f"({type(e).__name__}: {e}); rendering without cutout")

        if not beat.callout_text:
            continue
        if beat.license_type == "generated-fallback":
            continue  # the title card already shows the text: no duplicate lower-third
        try:
            if template in ("cutout_callout", "split_schematic"):
                img = graphics.render_badge(beat.callout_text, style,
                                            int(cut_cfg.get("callout_font_size", 46)))
                path = out_dir / f"badge_{beat.id:03d}.png"
                img.save(path)
                beat.badge_path = str(path)
                badges += 1
            else:
                img = graphics.render_lower_third(beat.callout_text, style,
                                                  int(cut_cfg.get("lower_third_font_size", 48)))
                path = out_dir / f"lower_third_{beat.id:03d}.png"
                img.save(path)
                beat.lower_third_path = str(path)
                thirds += 1
        except Exception as e:  # noqa: BLE001
            warn(f"beat {beat.id}: could not render callout graphic ({e})")

    write_json(plan_path, plan.model_dump(mode="json"))
    manifest = {"cutouts_dir": str(out_dir), "grid_background": str(grid),
                "cutout_count": cutouts, "badge_count": badges, "lower_third_count": thirds,
                "cutout_decisions": decisions, "warnings": warnings}
    write_json(work_dir / "cutouts_manifest.json", manifest)
    ready.touch()  # last: marks the stage complete
    console.print(f"  Overlays complete: {cutouts} cutouts, {badges} badges, "
                  f"{thirds} lower-thirds")
    return manifest


# --------------------------------------------------------------------------- cutouts


DEFAULT_MODELS = ["birefnet-general-lite", "isnet-general-use", "u2net"]


def _model_chain(cfg: dict[str, Any]) -> list[str]:
    models = cfg.get("models") or ([cfg["rembg_session"]] if cfg.get("rembg_session") else [])
    return [str(m) for m in (models or DEFAULT_MODELS)]


def _get_session(cfg: dict[str, Any], model: str):
    """Create (and cache) a rembg session; threads are limited via onnxruntime options."""
    threads = int(cfg.get("rembg_num_threads", 4))
    key = f"{model}:{threads}"
    if key not in _session_cache:
        import onnxruntime as ort
        from rembg import new_session

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        _session_cache[key] = new_session(model, sess_options=opts,
                                          providers=["CPUExecutionProvider"])
    return _session_cache[key]


def _extract_frame(video: str, out: Path) -> Path:
    """Grab the middle frame of a video as a PNG (subject source for a clip)."""
    dur = probe_duration(video) or 0.0
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{dur / 2:.2f}", "-i", video,
           "-frames:v", "1", str(out)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError(f"could not extract frame from {video}: {result.stderr[-200:]}")
    return out


def pick_subject(beat: dict[str, Any], resolver: AssetResolver
                 ) -> tuple[ResolvedMedia | None, str]:
    """Subject to cut out: the explicit ``overlay_asset`` (image/clip), else the beat's own
    source if it is a still image. The background footage is never used as a subject."""
    if beat.get("overlay_asset"):
        sub = resolver.any_asset(beat["overlay_asset"])
        if sub is None:
            return None, f"overlay_asset '{beat['overlay_asset']}' not found"
        if sub.asset_type == "schematic":
            return None, "a schematic is not a cutout subject (use split_schematic)"
        return sub, ""
    main = resolver.beat_main(beat)
    if main is None:
        return None, f"source '{beat.get('source_id')}' not found"
    if main.is_video or main.asset_type not in ("image", "stock"):
        return None, "no overlay_asset and the background is footage/schematic"
    return main, ""


def _path_key(path: str) -> str:
    return hashlib.md5(str(Path(path).resolve()).encode()).hexdigest()[:8]


def _make_cutout(subject: ResolvedMedia, beat_id: int, out_dir: Path,
                 cfg: dict[str, Any], frame_h: int, warn: Callable[[str], None],
                 decisions: list[dict[str, Any]] | None = None) -> Path:
    """Subject -> display-ready RGBA: a cleaned cutout, or a framed card if the matte is bad."""
    key = f"{Path(subject.path).stem}_{_path_key(subject.path)}"
    src = Path(subject.path)
    if subject.is_video:
        src = _extract_frame(subject.path, out_dir / f"frame_{key}.png")
    image = _load_limited(src, int(cfg.get("max_input_px", 1600)))

    mode, reason, stats = "cutout", "", {}
    model = ""
    result: Image.Image | None = None
    native = native_alpha_stats(src) if not subject.is_video else None
    bg_reason = "" if native else dark_uniform_background(image)
    if native:  # already a real cutout (icon/sticker PNG): use as is, no rembg / background test
        result, model, stats = _load_rgba_limited(src, int(cfg.get("max_input_px", 1600))), \
            "native-alpha", native
        reason = f"input already has alpha ({native['transparent']:.0%} transparent)"
    elif bg_reason and cfg.get("card_on_dark_background", True):
        mode, reason = "card", bg_reason
    else:
        raw, model = _cached_matte(image, src, key, out_dir, cfg)
        result, stats, bad = clean_cutout(raw, cfg)
        if bad:
            mode, reason = "card", bad
    if mode == "card":
        result = make_card_image(image)
        warn(f"{src.name}: using framed card instead of cutout ({reason})")
    assert result is not None

    bbox = result.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if bbox:
        result = result.crop(bbox)
    max_h = int(frame_h * float(cfg.get("max_cutout_height_ratio", 0.45)))
    max_w = int(max_h * 1.8)
    scale = min(max_h / result.height, max_w / result.width)
    result = result.resize((max(int(result.width * scale), 1), max(int(result.height * scale), 1)),
                           Image.Resampling.LANCZOS)
    final = graphics.with_shadow(result)
    out = out_dir / f"cutout_{beat_id:03d}_{key}.png"
    final.save(out)
    if decisions is not None:
        decisions.append({"beat": beat_id, "subject": src.name, "mode": mode, "reason": reason,
                          "model": model if mode == "cutout" else "", "stats": stats,
                          "path": str(out)})
    return out


def native_alpha_stats(path: Path, lo: float = 0.05, hi: float = 0.95) -> dict[str, float] | None:
    """Stats if the image has a useful alpha channel (``lo`` < transparent fraction < ``hi``),
    else None."""
    try:
        img = Image.open(path)
        if "A" not in img.getbands() and "transparency" not in img.info:
            return None
        alpha = np.asarray(img.convert("RGBA").getchannel("A"))
    except OSError:
        return None
    frac = float((alpha < 128).mean())
    return {"transparent": round(frac, 4)} if lo < frac < hi else None


def _load_rgba_limited(path: Path, max_px: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    return img


def _load_limited(path: Path, max_px: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    return img


def _cached_matte(image: Image.Image, src: Path, key: str, out_dir: Path,
                  cfg: dict[str, Any]) -> tuple[Image.Image, str]:
    """Raw rembg RGBA through the model fallback chain (cached per model on disk)."""
    chain = _model_chain(cfg)
    for model in chain:
        cache = out_dir / f"matte_{model}_{key}.png"
        if cache.is_file() and cache.stat().st_mtime >= src.stat().st_mtime:
            return Image.open(cache).convert("RGBA"), model
        try:
            from rembg import remove

            result = remove(image, session=_get_session(cfg, model))
            if not isinstance(result, Image.Image):
                raise TypeError(f"unexpected rembg output {type(result)}")
            result = result.convert("RGBA")
            result.save(cache)
            return result, model
        except Exception as e:  # noqa: BLE001 - model download/onnx failure -> next model
            console.print(f"  [yellow]rembg model {model} failed for {src.name} ({e})[/]")
    console.print(f"  [yellow]all rembg models failed for {src.name}; using fallback keying[/]")
    return _fallback_cutout(image), "fallback-keying"


# ------------------------------------------------------------------ mask post-processing


def dark_uniform_background(image: Image.Image) -> str:
    """Reason string if the photo already sits on a dark/uniform backdrop (e.g. space shots)
    so a cutout would only add artifacts; '' otherwise."""
    arr = np.asarray(image.convert("RGB").resize((160, int(160 * image.height / image.width) or 1)),
                     dtype=np.float32)
    h, w = arr.shape[:2]
    t = max(2, int(0.06 * min(h, w)))
    border = np.concatenate([arr[:t].reshape(-1, 3), arr[-t:].reshape(-1, 3),
                             arr[:, :t].reshape(-1, 3), arr[:, -t:].reshape(-1, 3)])
    lum = border @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if float(np.median(lum)) < 40 and float((lum < 60).mean()) > 0.8:
        return f"dark background (border luma {np.median(lum):.0f})"
    if float(border.std(axis=0).mean()) < 10:
        return "uniform background already clean"
    return ""


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def refine_alpha(alpha: np.ndarray, lo: float = 0.2, hi: float = 0.8,
                 min_component: float = 0.15, hole_frac: float = 0.01,
                 feather: float = 1.5) -> np.ndarray:
    """Clean a float [0,1] matte: contrast (kills low-alpha haze), drop floating fragments,
    fill small holes, open/close, erode 1px and feather."""
    from scipy import ndimage as ndi

    a = _smoothstep(alpha.astype(np.float32), lo, hi)
    binary = a > 0.5
    # drop fragments smaller than min_component of the largest one
    labels, n = ndi.label(binary)
    if n:
        sizes = ndi.sum(binary, labels, index=np.arange(1, n + 1))
        keep = np.flatnonzero(sizes >= max(sizes.max() * min_component, 16)) + 1
        binary = np.isin(labels, keep)
    # fill holes that are small relative to the subject
    holes = ndi.binary_fill_holes(binary) & ~binary
    hl, hn = ndi.label(holes)
    if hn:
        hs = ndi.sum(holes, hl, index=np.arange(1, hn + 1))
        small = np.flatnonzero(hs <= max(binary.sum() * hole_frac, 8)) + 1
        binary = binary | np.isin(hl, small)
    st = ndi.generate_binary_structure(2, 1)
    binary = ndi.binary_closing(binary, st, iterations=2)
    binary = ndi.binary_opening(binary, st, iterations=1)
    binary = ndi.binary_erosion(binary, st, iterations=1)
    # soft edge, keeping the model's interior detail where it was solid
    soft = ndi.gaussian_filter(binary.astype(np.float32), feather)
    out = np.minimum(soft, np.maximum(a, ndi.binary_erosion(binary, st, iterations=2)))
    # hard-zero everything outside the (dilated) cleaned region
    out[~ndi.binary_dilation(binary, st, iterations=3)] = 0
    return np.clip(out, 0, 1)


def defringe(rgb: np.ndarray, alpha: np.ndarray, solid: float = 0.97) -> np.ndarray:
    """Replace colours of non-solid (edge) pixels by the nearest solid pixel's colour, which
    removes the old-background halo."""
    from scipy import ndimage as ndi

    core = alpha >= solid
    if not core.any():
        return rgb
    inner = ndi.binary_erosion(core, iterations=1)
    if inner.any():
        core = inner
    idx = ndi.distance_transform_edt(~core, return_distances=False, return_indices=True)
    nearest = rgb[idx[0], idx[1]]
    edge = (alpha > 0) & ~core
    out = rgb.copy()
    out[edge] = nearest[edge]
    return out


def mask_stats(alpha: np.ndarray) -> dict[str, Any]:
    """Quality stats of a float [0,1] matte."""
    from scipy import ndimage as ndi

    fg = alpha > 0.5
    labels, n = ndi.label(fg)
    sizes = ndi.sum(fg, labels, index=np.arange(1, n + 1)) if n else np.array([])
    big = int((sizes >= max(sizes.max() * 0.15, 16)).sum()) if n else 0
    seen = alpha > 0.05
    band = (alpha > 0.05) & (alpha < 0.95)
    perim = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
    return {"coverage": float(fg.mean()), "components": big,
            "uncertain_frac": float(band.sum() / max(seen.sum(), 1)),
            "border_touch": float(perim.mean()),
            "sides_touched": sum(float(e.mean()) > 0.03 for e in
                                 (fg[0], fg[-1], fg[:, 0], fg[:, -1]))}


def clean_cutout(raw: Image.Image, cfg: dict[str, Any]) -> tuple[Image.Image, dict[str, Any], str]:
    """Refine a raw rembg RGBA. Returns (cleaned RGBA, stats, bad_reason or '')."""
    rgba = np.asarray(raw.convert("RGBA"))
    alpha0 = rgba[..., 3].astype(np.float32) / 255
    before = mask_stats(alpha0)
    alpha = refine_alpha(alpha0, feather=float(cfg.get("feather_px", 1.5)))
    rgb = defringe(rgba[..., :3], alpha)
    out = np.dstack([rgb, (alpha * 255).round().astype(np.uint8)])
    stats = mask_stats(alpha)
    stats["uncertain_frac_raw"] = before["uncertain_frac"]
    stats = {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}
    reason = ""
    if stats["coverage"] < float(cfg.get("min_coverage", MIN_SUBJECT_COVERAGE)):
        reason = f"subject too small ({stats['coverage']:.1%} coverage)"
    elif stats["coverage"] > float(cfg.get("max_coverage", 0.88)):
        reason = f"nothing to remove ({stats['coverage']:.0%} coverage)"
    elif stats["components"] > int(cfg.get("max_components", 4)):
        reason = f"fragmented mask ({stats['components']} components)"
    elif before["uncertain_frac"] > float(cfg.get("max_uncertain", 0.30)):
        reason = f"hazy matte ({before['uncertain_frac']:.0%} uncertain pixels)"
    elif before["sides_touched"] >= 2 and stats["coverage"] < 0.10:
        reason = "partial subject cut off by the frame"
    elif stats["border_touch"] > float(cfg.get("max_border_touch", 0.5)):
        reason = f"subject cropped by frame ({stats['border_touch']:.0%} of border)"
    return Image.fromarray(out, "RGBA"), stats, reason


MIN_SUBJECT_COVERAGE = 0.05


def make_card_image(image: Image.Image) -> Image.Image:
    return graphics.photo_card(image)


def _fallback_cutout(image: Image.Image) -> Image.Image:
    """Keep existing transparency, else key out the dominant border colour."""
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    if (alpha < 250).mean() > 0.02:
        return rgba
    arr = np.asarray(rgba.convert("RGB")).astype(np.int16)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.abs(arr - bg).sum(axis=2)
    mask = Image.fromarray(((dist > 48) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    coverage = np.asarray(mask).mean() / 255
    if coverage < 0.02 or coverage > 0.98:
        return rgba  # cannot isolate a subject: keep the whole image rather than fake one
    rgba.putalpha(mask)
    return rgba
