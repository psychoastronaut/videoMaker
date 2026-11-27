"""Shared helpers: paths, config, JSON I/O, ffprobe, plan signatures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

ENV_CONFIG = "VIDEOMAKER_CONFIG"
ENV_EPISODES = "VIDEOMAKER_EPISODES_DIR"
ENV_OFFLINE = "VIDEOMAKER_OFFLINE"

USER_AGENT = "videomaker-pipeline/0.1 (open-license media fetcher)"


# --------------------------------------------------------------------------- paths / env


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing variables win)."""
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def episodes_root() -> Path:
    override = os.environ.get(ENV_EPISODES)
    return Path(override) if override else PROJECT_ROOT / "episodes"


def episode_dir(episode_id: str) -> Path:
    return episodes_root() / episode_id


def voice_path(ep: Path) -> Path:
    """Voice-over used by the pipeline: work/voiceover_clean.wav when the audio stage ran
    (and ``audio.enabled`` is not false), else raw/voiceover.wav."""
    clean = ep / "work" / "voiceover_clean.wav"
    try:
        en = (load_config().get("audio") or {}).get("enabled", "auto")
    except FileNotFoundError:
        en = "auto"
    if clean.is_file() and en not in (False, "false", "off", "no"):
        return clean
    return ep / "raw" / "voiceover.wav"


def is_offline() -> bool:
    return os.environ.get(ENV_OFFLINE, "").strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> dict[str, Any]:
    """Load the YAML config (``--config`` / VIDEOMAKER_CONFIG deep-merged over config/default.yaml)."""
    override = os.environ.get(ENV_CONFIG)
    path = Path(override).expanduser() if override else DEFAULT_CONFIG_PATH
    if not path.is_file():
        if override:
            raise FileNotFoundError(f"Config file not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if override and DEFAULT_CONFIG_PATH.is_file() and path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        # an episode config only lists overrides: deep-merge them over config/default.yaml
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            data = deep_merge(yaml.safe_load(f) or {}, data)
    return data


def deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge (``over`` wins; lists and scalars are replaced)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# --------------------------------------------------------------------------- JSON


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temp file + rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- excluded assets

EXCLUDED_FILE = "excluded_assets.json"   # work/: [{"id", "file", "reason", "ts"}] (fix mode)
BURNED_FILE = "burned_text.json"         # work/: {asset id: {"text", "kind", "region", ...}}


def sidecar_excluded(filepath: str | Path | None) -> bool:
    """True when the asset's sidecar (``<file>.txt``) has a line ``exclude: true``."""
    if not filepath:
        return False
    try:
        text = Path(str(filepath) + ".txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    m = re.search(r"^\s*exclude\s*[:=]\s*(\S+)", text, re.IGNORECASE | re.MULTILINE)
    return bool(m and m.group(1).strip().lower() in ("true", "yes", "1", "on"))


def excluded_ids(work: Path, assets: list[dict[str, Any]] | None = None) -> set[str]:
    """Asset ids that must never be used: listed in work/excluded_assets.json (by id or file
    name) or flagged ``exclude: true`` in the sidecar. Respected by stages 2/2a/3 and fix mode."""
    ids: set[str] = set()
    names: set[str] = set()
    p = work / EXCLUDED_FILE
    if p.is_file():
        try:
            for e in read_json(p):
                if e.get("id"):
                    ids.add(str(e["id"]))
                if e.get("file"):
                    names.add(str(e["file"]))
        except (json.JSONDecodeError, OSError, AttributeError, TypeError):
            pass
    if assets is None:
        ap = work / "local_assets.json"
        assets = read_json(ap).get("assets", []) if ap.is_file() else []
    for a in assets:
        if a.get("filename") in names or sidecar_excluded(a.get("filepath")):
            ids.add(a["id"])
    return ids


def drop_excluded(assets: list[dict[str, Any]], work: Path) -> list[dict[str, Any]]:
    bad = excluded_ids(work, assets)
    return [a for a in assets if a["id"] not in bad] if bad else assets


def load_burned(work: Path) -> dict[str, Any]:
    p = work / BURNED_FILE
    try:
        return read_json(p) if p.is_file() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def drop_burned(assets: list[dict[str, Any]], work: Path) -> list[dict[str, Any]]:
    """Planner candidates: assets with burned-in text/captions/watermarks are avoided unless a
    crop (``crop_box``) already cuts the text band away."""
    flagged = {k for k, v in load_burned(work).items() if v.get("text")}
    return [a for a in assets if not ((a["id"] in flagged or a.get("burned_text"))
                                      and not a.get("crop_box"))] if flagged or any(
        a.get("burned_text") for a in assets) else assets


# --------------------------------------------------------------------------- ffprobe


def probe_duration(path: str | Path) -> float | None:
    """Container duration in seconds, or None when ffprobe cannot read the file."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(path)],
            check=False, capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


# --------------------------------------------------------------------------- plan signatures

_AUTHORING_FIELDS = (
    "start_time", "end_time", "visual_template", "primary_source", "source_id",
    "overlay_asset", "callout_text", "beat_style", "search_queries", "emphasis", "stock_media",
    "media_start", "media_end", "transition", "transition_duration",
    "overlay_start", "clip_audio", "clip_audio_gain", "overlay_mode", "pip_asset", "pip_corner",
    "pip_start", "pip_delay", "highlights", "crop_region",
)
_RESOLVED_FIELDS = ("media_path", "stock_asset_id", "stock_fallback_of")


def _sig(beats: list[dict], fields: tuple[str, ...]) -> str:
    # optional trim fields only count once set, so v1 plans keep their signature
    optional = ("media_start", "media_end", "transition", "transition_duration",
                "overlay_start", "clip_audio", "clip_audio_gain", "overlay_mode", "pip_asset",
                "pip_corner", "pip_start", "pip_delay", "highlights", "crop_region")
    payload = [{k: b.get(k) for k in fields if not (k in optional and b.get(k) is None)}
               for b in beats]
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def authoring_signature(plan: dict) -> str:
    """Hash of everything a human/LLM decides (excludes generated file paths)."""
    return _sig(plan.get("beats", []), _AUTHORING_FIELDS)


def resolved_signature(plan: dict) -> str:
    """Authoring signature plus the stock assets Stage 4 resolved."""
    return _sig(plan.get("beats", []), _AUTHORING_FIELDS + _RESOLVED_FIELDS)
