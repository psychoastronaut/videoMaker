"""Helpers for stock *video*: candidate selection, capped download, ffprobe verification, trim."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx

# Archive.org licenseurl values that are explicitly CC0 / public domain.
ARCHIVE_PD = re.compile(
    r"https?://(www\.)?creativecommons\.org/"
    r"(publicdomain/(zero|mark)/1\.0|licenses/publicdomain)/?", re.IGNORECASE)


def is_archive_pd(license_url: str | None) -> bool:
    """The *whole* licenseurl must be a CC0 / public-domain-mark URL (no substring tricks)."""
    return bool(license_url and ARCHIVE_PD.fullmatch(license_url.strip()))


NASA_LICENSE = "Public Domain (NASA)"
PEXELS_LICENSE = "Pexels License"
PIXABAY_LICENSE = "Pixabay Content License"
# Free-to-use platform licenses that are NOT CC0/public domain (flagged in the timeline).
PLATFORM_LICENSES = {PEXELS_LICENSE, PIXABAY_LICENSE}


def pick_height_file(files: list[dict[str, Any]], min_h: int, max_h: int) -> dict[str, Any] | None:
    """Best landscape file: tallest height within [min_h, max_h] (smaller than max preferred
    over a downscale-needing 4K file). Each item needs ``width``/``height``/``url``."""
    ok = [f for f in files
          if f.get("url") and (f.get("width") or 0) >= (f.get("height") or 1)
          and min_h <= (f.get("height") or 0) <= max_h]
    return max(ok, key=lambda f: f["height"]) if ok else None


def pexels_files(video: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"url": f.get("link"), "width": f.get("width"), "height": f.get("height"),
             "size": 0}
            for f in video.get("video_files", []) if f.get("file_type") == "video/mp4"]


def pixabay_files(hit: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"url": v.get("url"), "width": v.get("width"), "height": v.get("height"),
             "size": v.get("size") or 0} for v in (hit.get("videos") or {}).values()]


def download_capped(client: httpx.Client, url: str, dest: Path, max_bytes: int) -> bool:
    """Stream ``url`` to ``dest``; abort (and delete) if it exceeds ``max_bytes``."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                return False
            length = int(resp.headers.get("content-length") or 0)
            if length and length > max_bytes:
                return False
            total = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes(1 << 16):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("too large")
                    f.write(chunk)
        tmp.replace(dest)
        return True
    except (httpx.HTTPError, ValueError, OSError):
        tmp.unlink(missing_ok=True)
        return False


def probe_video(path: Path) -> tuple[int, int, float] | None:
    """(width, height, duration) if ffprobe finds a decodable video stream, else None."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height:format=duration", "-of", "json", str(path)],
            check=False, capture_output=True, text=True, timeout=30)
        data = json.loads(r.stdout or "{}")
        st = data["streams"][0]
        dur = float(data.get("format", {}).get("duration") or 0)
        return int(st["width"]), int(st["height"]), dur
    except (subprocess.SubprocessError, KeyError, IndexError, ValueError, OSError):
        return None


def trim_video(src: Path, dst: Path, seconds: float, max_h: int = 1080) -> bool:
    """Keep the first ``seconds`` seconds, drop audio, cap height, normalise to H.264 mp4."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-xerror", "-i", str(src), "-t", f"{seconds:.2f}",
           "-an", "-vf", f"scale=-2:'min(ih,{max_h})',format=yuv420p", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(dst)]
    try:
        r = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=max(60, seconds * 20))
    except subprocess.SubprocessError:
        dst.unlink(missing_ok=True)
        return False
    if r.returncode != 0:
        dst.unlink(missing_ok=True)
        return False
    return dst.is_file() and dst.stat().st_size > 0
