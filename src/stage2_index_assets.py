"""Stage 2: Local Asset Indexing.

Scans raw/clips/, raw/images/, and raw/schematics/ directories, extracts
metadata via ffprobe, and generates semantic descriptions/tags for each asset.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from rich.console import Console

from src.common import episode_dir, load_config, write_json
from src.media import undecodable_reason
from src.schemas import AssetIndex, LocalAsset

console = Console()


def run(episode_id: str, work_dir: Path) -> dict:
    """Run Stage 2: index local assets."""
    raw = episode_dir(episode_id) / "raw"
    config = load_config()
    asset_config = config.get("assets", {})

    assets: list[LocalAsset] = []
    skipped: list[dict[str, str]] = []

    # Index video clips
    clips_dir = raw / "clips"
    if clips_dir.exists():
        clips = _scan_directory(clips_dir, "clip", asset_config.get("supported_video_extensions", []), skipped)
        assets.extend(clips)
        console.print(f"  Found {len(clips)} video clips")

    # Index images
    images_dir = raw / "images"
    if images_dir.exists():
        images = _scan_directory(images_dir, "image", asset_config.get("supported_image_extensions", []), skipped)
        assets.extend(images)
        console.print(f"  Found {len(images)} images")

    # Index schematics
    schematics_dir = raw / "schematics"
    if schematics_dir.exists():
        schematics = _scan_directory(
            schematics_dir, "schematic", asset_config.get("supported_schematic_extensions", []), skipped
        )
        assets.extend(schematics)
        console.print(f"  Found {len(schematics)} schematics")

    index = AssetIndex(assets=assets, skipped=skipped)

    data = index.model_dump(mode="json")
    try:  # optional AI upscaling of small photos (never blocks indexing)
        from src.upscale import apply_to_index

        apply_to_index(data, work_dir, config, log=console.print)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [yellow]Upscaling failed, continuing without it: {e}[/]")

    try:  # optional shake smoothing of handheld/old footage (never blocks indexing)
        from src.stabilize import apply_to_index as stabilize_index

        stabilize_index(data, work_dir, config, log=console.print)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [yellow]Stabilization failed, continuing without it: {e}[/]")

    try:  # black bars (possibly with burned-in caption text): record the content window
        from src.letterbox import detect_for_file

        for a in data["assets"]:
            if a["asset_type"] in ("clip", "image"):
                box = detect_for_file(a["filepath"], a.get("duration"))
                if box:
                    a["crop_box"] = box
                    console.print(f"  Letterbox detected in {a['filename']}: crop {box}")
    except Exception as e:  # noqa: BLE001
        console.print(f"  [yellow]Letterbox detection failed, continuing without it: {e}[/]")

    output_path = work_dir / "local_assets.json"
    write_json(output_path, data)

    console.print(f"  Asset index saved: {output_path} ({len(assets)} assets)")
    return data


def _scan_directory(dir_path: Path, asset_type: str, extensions: list[str],
                    skipped: list[dict[str, str]] | None = None) -> list[LocalAsset]:
    """Scan a directory and extract metadata from each file."""
    assets: list[LocalAsset] = []
    seen_ids: set[str] = set()

    for file_path in sorted(dir_path.iterdir()):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in extensions:
            continue

        reason = undecodable_reason(file_path)
        if reason:  # corrupt/unsupported files must never reach the planner or ffmpeg
            console.print(f"  [yellow]Skipping {file_path.name}: {reason}[/]")
            if skipped is not None:
                skipped.append({"file": str(file_path), "reason": reason})
            continue

        asset_id = f"{asset_type}_{file_path.stem}"
        if asset_id in seen_ids:  # same stem, different extension
            asset_id = f"{asset_type}_{file_path.stem}_{ext.lstrip('.')}"
        seen_ids.add(asset_id)
        metadata = _extract_metadata(file_path, asset_type)

        asset = LocalAsset(
            id=asset_id,
            filepath=str(file_path.resolve()),
            asset_type=asset_type,
            filename=file_path.name,
            description=metadata["description"],
            tags=metadata["tags"],
            duration=metadata.get("duration") if asset_type == "clip" else None,
            width=metadata.get("width"),
            height=metadata.get("height"),
            fps=metadata.get("fps"),
            aspect_ratio=metadata.get("aspect_ratio"),
        )
        assets.append(asset)

    return assets


def _extract_metadata(file_path: Path, asset_type: str) -> dict:
    """Extract metadata from a media file using ffprobe."""
    metadata: dict = {
        "description": "",
        "tags": [],
    }

    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        probe = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        metadata["description"] = f"{asset_type} from {file_path.name}"
        metadata["tags"] = [asset_type, file_path.stem]
        return metadata

    # Extract video stream info
    video_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream:
        metadata["width"] = video_stream.get("width")
        metadata["height"] = video_stream.get("height")
        fps = video_stream.get("r_frame_rate")
        if fps and "/" in fps:
            num, den = fps.split("/")
            metadata["fps"] = float(num) / float(den) if float(den) > 0 else 30.0
        elif fps:
            metadata["fps"] = float(fps)

        if metadata.get("width") and metadata.get("height"):
            metadata["aspect_ratio"] = metadata["width"] / metadata["height"]

    # Extract format info
    fmt = probe.get("format", {})
    duration = fmt.get("duration")
    if duration:
        metadata["duration"] = float(duration)

    # Generate description from filename and sidecar notes
    stem = file_path.stem.lower()
    desc_parts = [f"{asset_type}"]

    # Parse descriptive words from filename
    descriptive_words = stem.replace("_", " ").replace("-", " ")
    desc_parts.append(descriptive_words)

    # Check for sidecar notes
    sidecar = file_path.with_suffix(file_path.suffix + ".txt")
    if sidecar.exists():
        desc_parts.append(sidecar.read_text().strip())

    metadata["description"] = " | ".join(p for p in desc_parts if p)
    metadata["tags"] = [asset_type] + [t for t in stem.replace("_", " ").replace("-", " ").split() if t]

    return metadata
