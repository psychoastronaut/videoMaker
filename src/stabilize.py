"""Optional source stabilisation of shaky video clips (ffmpeg vidstab, OpenCV fallback).

Old/handheld footage (e.g. NASA film) is shaky by itself. ``shake_score`` measures the
high-frequency jitter of the camera path (px at 1920 wide, residual of per-frame motion after a ~2-frame Gaussian low-pass; cut spikes ignored);
clips above ``stabilize.threshold`` are re-encoded smoothed into ``work/stabilized/`` (cached by
content hash + settings) and the index gets ``stabilized_path`` (preferred by the resolver).
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "enabled": "auto",    # auto = stabilise clips whose shake score > threshold | true = all | false
    "threshold": 1.25,     # shake score (px @1920 of high-frequency camera jitter)
    "smoothing": 15,      # vidstab: frames of camera-path smoothing each side
    "shakiness": 5,       # vidstab detection sensitivity 1..10
    "zoom": 0.0,          # extra static zoom % (negative/zero keeps the full frame)
    "optzoom": 1,         # 0 none | 1 static optimal zoom (hides borders) | 2 adaptive
    "crf": 16,
}


def stabilize_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    cfg.update(config.get("stabilize") or {})
    return cfg


def has_vidstab() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, check=False,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return "vidstabdetect" in r.stdout and "vidstabtransform" in r.stdout


# --------------------------------------------------------------------------- measuring


def camera_path(path: str | Path, t0: float, dur: float, width: int = 480) -> tuple[Any, float]:
    """Per-frame global translation (dx, dy) in px @1920 via robust LK tracking (RANSAC affine)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    prev = None
    steps: list[tuple[float, float]] = []
    for _ in range(int(dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (width, max(int(g.shape[0] * width / g.shape[1]), 8)),
                       interpolation=cv2.INTER_AREA)
        if prev is not None:
            dx = dy = 0.0
            cut = float(np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean()) > 30
            pts = None if cut else cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
            if pts is not None and len(pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21),
                                                      maxLevel=3)
                m = st.ravel() == 1
                if m.sum() >= 8:
                    mat, inl = cv2.estimateAffinePartial2D(pts[m], nxt[m], method=cv2.RANSAC,
                                                           ransacReprojThreshold=1.0)
                    if mat is not None and inl is not None and inl.sum() >= 6:
                        dx = float(mat[0, 2] + (mat[0, 0] - 1) * g.shape[1] / 2)
                        dy = float(mat[1, 2] + (mat[0, 0] - 1) * g.shape[0] / 2)
            if abs(dx) > 40 or abs(dy) > 40:  # scene cut / tracking failure
                dx = dy = 0.0
            steps.append((dx, dy))
        prev = g
    cap.release()
    return np.array(steps) * (1920 / width), fps


def shake_score(path: str | Path, t0: float = 0.0, dur: float = 6.0) -> float | None:
    """RMS frame-to-frame jitter (px @1920) of the camera path; None if too short."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    steps, _fps = camera_path(path, t0, dur)
    if len(steps) < 12:
        return None
    # high-frequency part of the per-frame motion (sigma ~2 frames); cut spikes are dropped
    hf = steps - gaussian_filter1d(steps, sigma=2.0, axis=0, mode="nearest")
    mag = np.sqrt((hf ** 2).sum(1))
    mag = mag[mag < 20.0]
    return float(np.sqrt((mag ** 2).mean())) if len(mag) else 0.0


# --------------------------------------------------------------------------- stabilising


def cached_path(src: Path, out_dir: Path, cfg: dict[str, Any]) -> Path:
    st = src.stat()
    h = hashlib.sha1()
    with src.open("rb") as f:
        h.update(f.read(1 << 20))
    key = f"{st.st_size}:{h.hexdigest()}:{cfg['smoothing']}:{cfg['shakiness']}:{cfg['zoom']}:" \
          f"{cfg['optzoom']}:{cfg['crf']}"
    return out_dir / f"{src.stem}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.mp4"


def _vidstab(src: Path, dst: Path, cfg: dict[str, Any]) -> bool:
    with tempfile.TemporaryDirectory(prefix="vidstab_") as tmp:
        trf = Path(tmp) / "t.trf"
        det = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf",
             f"vidstabdetect=shakiness={int(cfg['shakiness'])}:result={trf}", "-f", "null", "-"],
            capture_output=True, text=True, check=False)
        if det.returncode or not trf.is_file():
            return False
        tr = (f"vidstabtransform=input={trf}:smoothing={int(cfg['smoothing'])}:"
              f"optzoom={int(cfg['optzoom'])}:zoom={float(cfg['zoom'])}:interpol=bicubic:"
              "crop=keep,unsharp=5:5:0.4:3:3:0.0")
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", tr, "-an", "-c:v", "libx264",
             "-preset", "fast", "-crf", str(int(cfg["crf"])), "-pix_fmt", "yuv420p", str(dst)],
            capture_output=True, text=True, check=False)
        return r.returncode == 0 and dst.is_file()


def _opencv(src: Path, dst: Path, cfg: dict[str, Any]) -> bool:
    """Fallback without libvidstab: track, low-pass the camera path, warp, crop-zoom."""
    import cv2
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    steps, fps = camera_path(src, 0.0, 1e6)
    if len(steps) < 3:
        return False
    cap = cv2.VideoCapture(str(src))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path = np.vstack([[0, 0], np.cumsum(steps, axis=0)])
    corr = gaussian_filter1d(path, sigma=max(float(cfg["smoothing"]) / 2, 1), axis=0,
                             mode="nearest") - path
    corr *= w / 1920
    zoom = 1.0 + float(np.abs(corr).max()) * 2 / w + float(cfg["zoom"]) / 100
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf",
         str(int(cfg["crf"])), "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    i = 0
    while ff.stdin is not None:
        ok, frame = cap.read()
        if not ok:
            break
        dx, dy = corr[min(i, len(corr) - 1)]
        m = np.float32([[zoom, 0, dx + w * (1 - zoom) / 2], [0, zoom, dy + h * (1 - zoom) / 2]])
        ff.stdin.write(cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE).tobytes())
        i += 1
    if ff.stdin is not None:
        ff.stdin.close()
    return ff.wait() == 0 and dst.is_file()


def stabilize_clip(src: Path, dst: Path, cfg: dict[str, Any]) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    ok = False
    try:
        ok = (_vidstab(src, tmp, cfg) if has_vidstab() else False) or _opencv(src, tmp, cfg)
    except Exception:  # noqa: BLE001 - stabilisation is optional, never fatal
        ok = False
    if ok:
        shutil.move(str(tmp), dst)
    else:
        tmp.unlink(missing_ok=True)
    return ok


def apply_to_index(index: dict[str, Any], work_dir: Path, config: dict[str, Any],
                   log: Any = print, force: bool = False) -> dict[str, Any]:
    """Stabilise shaky video clips of an asset-index dict in place (adds ``stabilized_path``,
    ``shake_score``). Returns {stabilized: [...], scores: {id: score}, skipped: n}."""
    cfg = stabilize_config(config)
    mode = str(cfg["enabled"]).lower()
    summary: dict[str, Any] = {"stabilized": [], "scores": {}, "skipped": 0}
    if mode in ("false", "0", "no", "off"):
        return summary
    out_dir = work_dir / "stabilized"
    for a in index.get("assets", []):
        src = Path(a.get("filepath", ""))
        if a.get("asset_type") != "clip" or not src.is_file():
            continue
        dst = cached_path(src, out_dir, cfg)
        if dst.is_file() and not force:
            a["stabilized_path"] = str(dst)
            summary["stabilized"].append(a["id"])
            continue
        score = shake_score(src)
        a["shake_score"] = None if score is None else round(score, 2)
        summary["scores"][a["id"]] = a["shake_score"]
        if mode == "auto" and (score is None or score <= float(cfg["threshold"])):
            a.pop("stabilized_path", None)
            summary["skipped"] += 1
            continue
        log(f"  Stabilizing {src.name} (shake {score if score is None else round(score, 2)})")
        if stabilize_clip(src, dst, cfg):
            after = shake_score(dst)
            if mode == "auto" and score is not None and after is not None and after >= score * 0.9:
                dst.unlink(missing_ok=True)  # no measurable gain: keep the original
                log(f"  {src.name}: no gain ({score:.2f} -> {after:.2f}), kept original")
                summary["skipped"] += 1
                continue
            a["stabilized_path"] = str(dst)
            a["shake_after"] = None if after is None else round(after, 2)
            summary["stabilized"].append(a["id"])
        else:
            log(f"  [yellow]stabilization failed for {src.name}; using the original[/]")
    return summary
