"""Exact asset resolution shared by stages 5 and 6 (no fuzzy matching)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.common import read_json

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


_decode_cache: dict[tuple[str, int], str | None] = {}


def undecodable_reason(path: str | Path) -> str | None:
    """None if ``path`` is a decodable image / video with a video stream, else the reason.

    Images are verified and fully loaded with Pillow; videos need ffprobe to report a video
    stream. Results are cached per (path, mtime).
    """
    p = Path(path)
    try:
        key = (str(p), int(p.stat().st_mtime_ns))
    except OSError as e:
        return f"unreadable ({e.strerror})"
    if key in _decode_cache:
        return _decode_cache[key]
    reason: str | None = None
    if is_video_path(p):
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=codec_name:format=duration", "-of", "json", str(p)],
                check=False, capture_output=True, text=True, timeout=30)
            info = json.loads(r.stdout or "{}")
            if r.returncode != 0 or not info.get("streams"):
                reason = "no decodable video stream"
            else:
                # A truncated file probes fine (header intact) but fails to decode near the
                # end: decode the last ~second to catch it.
                dur = float(info.get("format", {}).get("duration") or 0)
                tail = subprocess.run(
                    ["ffmpeg", "-v", "error", "-xerror", "-ss", f"{max(dur - 1.5, 0):.2f}",
                     "-t", "1", "-i", str(p), "-f", "null", "-"],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=60)
                if tail.returncode != 0:
                    reason = "truncated or corrupt (decode failed near the end)"
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            reason = f"ffprobe failed ({e})"
    else:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
        except Exception as e:  # noqa: BLE001 - any decode problem means unusable
            reason = f"not a decodable image ({type(e).__name__})"
    _decode_cache[key] = reason
    return reason


@dataclass(frozen=True)
class ResolvedMedia:
    path: str
    is_video: bool
    asset_id: str
    asset_type: str  # clip | image | schematic | stock
    trim_start: float | None = None  # semantic trim: start of the chosen interval (clips only)

    @property
    def exists(self) -> bool:
        return Path(self.path).is_file()


def is_video_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _stock_policy(work_dir: Path) -> Any:
    """Predicate ``asset_dict -> bool`` for the CURRENT stock license policy."""
    from src.common import load_config
    from src.schemas import StockAsset
    from src.stage4_fetch_stock import StockFetcher

    fetcher = StockFetcher(load_config().get("stock_media", {}), work_dir, offline=True)

    def allowed(a: dict[str, Any]) -> bool:
        try:
            return fetcher.license_ok(StockAsset.model_validate(a))
        except ValueError:
            return False

    return allowed


class AssetResolver:
    """Resolve asset ids from ``local_assets.json`` and ``licenses_audit.json`` exactly."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.local: dict[str, dict[str, Any]] = {}
        self.stock: dict[str, dict[str, Any]] = {}
        self.refused: dict[str, str] = {}  # stock asset id -> reason (license policy)
        assets_path = work_dir / "local_assets.json"
        if assets_path.is_file():
            for a in read_json(assets_path).get("assets", []):
                self.local[a["id"]] = a
        audit_path = work_dir / "licenses_audit.json"
        if audit_path.is_file():
            allowed = _stock_policy(work_dir)
            for a in read_json(audit_path).get("assets", []):
                self.stock[a["id"]] = a
                if not allowed(a):
                    self.refused[a["id"]] = (
                        f"{a.get('source_platform')} / {a.get('license_type')} is not allowed "
                        "by the current stock license policy")

    def local_asset(self, asset_id: str | None) -> ResolvedMedia | None:
        media = self._local_asset(asset_id)
        box = self.local.get(asset_id or "", {}).get("crop_box")
        if media is not None and box:
            return self._cropped(media, box, self.local[asset_id])
        return media

    def _cropped(self, media: ResolvedMedia, box: list[int], a: dict[str, Any]) -> ResolvedMedia:
        """Copy of ``media`` cut to the letterbox content window (cached in work/letterbox);
        the renderer then cover-scales it as usual. Falls back to the original on any failure."""
        from dataclasses import replace

        from src.letterbox import crop_filter

        try:
            with Image.open(media.path) as im:  # stills; videos are probed below
                size = im.size
        except Exception:  # noqa: BLE001
            size = None
        try:
            if size is None:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream=width,height", "-of", "csv=p=0:s=x", media.path],
                    capture_output=True, text=True, timeout=30, check=False)
                w, h = r.stdout.strip().split("x")
                size = (int(w), int(h))
            ref = (int(a["width"]), int(a["height"]))
            vf = crop_filter(box, size, ref)
            src = Path(media.path)
            out_dir = self.work_dir / "letterbox"
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{int(src.stat().st_mtime)}-{hashlib.md5(f'{vf}{src.stat().st_size}'.encode()).hexdigest()[:8]}"
            out = out_dir / f"{src.stem}-{tag}{'.mp4' if media.is_video else '.png'}"
            if not out.is_file():
                cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf]
                cmd += (["-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p",
                         "-c:a", "copy"] if media.is_video else [])
                subprocess.run(cmd + [str(out)], check=True, capture_output=True, timeout=600)
            return replace(media, path=str(out)) if self._make(str(out), "", "") else media
        except Exception:  # noqa: BLE001
            return media

    def _local_asset(self, asset_id: str | None) -> ResolvedMedia | None:
        if not asset_id or asset_id not in self.local:
            return None
        a = self.local[asset_id]
        st = a.get("stabilized_path")  # shake-smoothed clip (src/stabilize.py), full length
        if st and Path(st).is_file():
            media = self._make(st, asset_id, a["asset_type"])
            if media is not None:
                return media
        up = a.get("upscaled_path")  # AI-upscaled copy (src/upscale.py), preferred when valid
        if up and Path(up).is_file():
            media = self._make(up, asset_id, a["asset_type"])
            if media is not None:
                return media
        return self._make(a["filepath"], asset_id, a["asset_type"])

    def stock_asset(self, asset_id: str | None) -> ResolvedMedia | None:
        if not asset_id or asset_id not in self.stock or asset_id in self.refused:
            return None
        return self._make(self.stock[asset_id]["local_path"], asset_id, "stock")

    @staticmethod
    def _make(path: str, asset_id: str, asset_type: str) -> ResolvedMedia | None:
        if not Path(path).is_file() or undecodable_reason(path):
            return None
        return ResolvedMedia(path=path, is_video=is_video_path(path), asset_id=asset_id,
                             asset_type=asset_type)

    def any_asset(self, asset_id: str | None) -> ResolvedMedia | None:
        return self.local_asset(asset_id) or self.stock_asset(asset_id)

    def beat_main(self, beat: dict[str, Any]) -> ResolvedMedia | None:
        """Main media of a beat: stock beats use media_path/stock_asset_id, local use source_id."""
        if beat.get("primary_source") == "stock":
            sid = beat.get("stock_asset_id")
            if not sid or sid not in self.stock or sid in self.refused:
                return None  # unaudited or policy-refused stock is never rendered
            mp = beat.get("media_path")
            if mp and Path(mp).is_file() and not undecodable_reason(mp):
                return ResolvedMedia(path=mp, is_video=is_video_path(mp), asset_id=sid,
                                     asset_type="stock")
            return self.stock_asset(sid)
        media = self.local_asset(beat.get("source_id"))
        start, end = beat.get("media_start"), beat.get("media_end")
        if media is not None and media.is_video and start is not None and start >= 0 \
                and (end is None or end > start):
            from dataclasses import replace

            return replace(media, trim_start=float(start))
        return media

    def stock_refusal(self, beat: dict[str, Any]) -> str | None:
        """Why a stock beat's media cannot be used (policy / missing audit), else None."""
        if beat.get("primary_source") != "stock":
            return None
        sid = beat.get("stock_asset_id")
        if sid in self.refused:
            return self.refused[sid]
        if not sid or sid not in self.stock:
            return "stock asset is not in licenses_audit.json"
        return None

    def beat_overlay(self, beat: dict[str, Any]) -> ResolvedMedia | None:
        """Right panel / cutout subject; a clip is trimmed to ``overlay_start`` (time sync)."""
        return self._trimmed(self.any_asset(beat.get("overlay_asset")), beat.get("overlay_start"))

    def beat_pip(self, beat: dict[str, Any]) -> ResolvedMedia | None:
        """Picture-in-picture inset media (``pip_asset``, clip trimmed to ``pip_start``)."""
        if beat.get("overlay_mode") != "pip":
            return None
        return self._trimmed(self.any_asset(beat.get("pip_asset")), beat.get("pip_start"))

    @staticmethod
    def _trimmed(media: ResolvedMedia | None, start: Any) -> ResolvedMedia | None:
        if media is not None and media.is_video and start is not None and float(start) > 0:
            from dataclasses import replace

            return replace(media, trim_start=float(start))
        return media
