"""Pydantic schemas for the video automation pipeline."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class VisualTemplate(str, Enum):
    FULLSCREEN_CLIP = "fullscreen_clip"
    SPLIT_SCHEMATIC = "split_schematic"
    CUTOUT_CALLOUT = "cutout_callout"
    PUNCH_IN_EMPHASIS = "punch_in_emphasis"


class BeatStyle(str, Enum):
    NORMAL = "normal"
    DEADPAN = "deadpan"
    HYPE = "hype"
    TECHNICAL = "technical"
    FAIL = "fail"


class PrimarySource(str, Enum):
    LOCAL = "local"
    STOCK = "stock"


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class Segment(BaseModel):
    id: int = 0
    start: float
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)

    @field_validator("words", mode="before")
    @classmethod
    def ensure_word_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []


class Transcript(BaseModel):
    full_text: str = ""
    language: str = "en"
    duration: float = 0.0
    segments: list[Segment] = Field(default_factory=list)
    cut_boundaries: list[float] = Field(default_factory=list)
    synthetic: bool = False  # demo transcript: never overwritten by an automatic re-run

    @property
    def total_duration(self) -> float:
        if self.segments:
            return self.segments[-1].end - self.segments[0].start
        return self.duration

    @field_validator("segments")
    @classmethod
    def sort_segments(cls, v):
        if v:
            return sorted(v, key=lambda s: s.start)
        return v


class LocalAsset(BaseModel):
    id: str
    filepath: str
    asset_type: str  # "clip", "image", "schematic"
    filename: str
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    aspect_ratio: float | None = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    # letterbox content window [x, y, w, h] in source pixels (src/letterbox.py); None = no bars
    crop_box: list[int] | None = None


class AssetIndex(BaseModel):
    assets: list[LocalAsset] = Field(default_factory=list)
    # files ignored because they are corrupt/undecodable: [{"file", "reason"}]
    skipped: list[dict[str, str]] = Field(default_factory=list)

    def by_type(self, asset_type: str) -> list[LocalAsset]:
        return [a for a in self.assets if a.asset_type == asset_type]

    def by_id(self, asset_id: str) -> LocalAsset | None:
        for a in self.assets:
            if a.id == asset_id:
                return a
        return None


class SceneBeat(BaseModel):
    id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    visual_template: VisualTemplate = VisualTemplate.FULLSCREEN_CLIP
    primary_source: PrimarySource = PrimarySource.LOCAL
    # fullscreen/punch: the media. split_schematic: left media. cutout_callout: background.
    source_id: str = ""
    # split_schematic: the schematic (right panel). cutout_callout: the subject to cut out
    # (defaults to source_id). Ignored by the other templates.
    overlay_asset: str | None = None
    callout_text: str | None = None
    # beat_style is the source of truth; ``tone`` is a read-compat alias kept in sync.
    tone: str = "normal"
    beat_style: BeatStyle = BeatStyle.NORMAL
    director_notes: str = ""
    beat_reason: str = ""
    emphasis: bool = False
    illustrative_defocus: bool = False  # narration describes blur: shown photo is blurred on purpose
    search_queries: list[str] = Field(default_factory=list)
    # what stage 4 should fetch for a stock beat: any | image (entity establishing shot)
    stock_media: str = "any"
    zoompan_config: dict[str, Any] | None = None
    # Semantic matching (all optional; absent in v1 plans)
    visual_query: str | None = None    # concrete English query derived from the narration
    match_score: float | None = None   # backend similarity of query vs chosen media
    media_start: float | None = None   # trim interval inside the local clip (seconds)
    media_end: float | None = None
    # split_schematic with a clip on the right: where that clip starts (seconds), so both
    # panels can be time-synced with ``media_start`` of the left one.
    overlay_start: float | None = None
    # Clip audio (mixed in the final mux, never per beat): off (default) | on | duck (the
    # narration is ducked while the clip speaks). ``clip_audio_gain`` is a linear gain.
    clip_audio: str | None = None
    clip_audio_gain: float | None = None
    # Picture-in-picture inset (small rounded clip/still over the main picture):
    # overlay_mode "pip" + pip_asset; pip_corner tr|tl|br|bl; pip_start trims a clip;
    # pip_delay = seconds into the beat before it slides in.
    overlay_mode: str | None = None
    pip_asset: str | None = None
    pip_corner: str | None = None
    pip_start: float | None = None
    pip_delay: float | None = None
    # Animated highlights on the picture: [{x,y,w,h | region, label, start, duration}]
    highlights: list[dict[str, Any]] | None = None
    # Detail crop of a still ("virtual sub-asset"): [x, y, w, h] as fractions of the picture
    crop_region: list[float] | None = None
    # Transition INTO this beat from the previous one (None = automatic by style/template;
    # "none" = hard cut). See src/transitions.py for the vetted names.
    transition: str | None = None
    transition_duration: float | None = None

    # Resolved by stages 4/5 (written back to scene_plan.json).
    media_path: str | None = None
    # set when a stock beat found no media and was downgraded to this local asset id
    stock_fallback_of: str | None = None
    stock_asset_id: str | None = None
    license_type: str | None = None
    cutout_path: str | None = None
    badge_path: str | None = None
    lower_third_path: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_style(cls, data: Any) -> Any:
        """Accept ``tone`` and unknown style strings; normalise into the BeatStyle enum."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        valid = {s.value for s in BeatStyle}

        def norm(value: Any) -> str | None:
            if isinstance(value, BeatStyle):
                return value.value
            if isinstance(value, str) and value.strip().lower() in valid:
                return value.strip().lower()
            return None

        style = norm(data.get("beat_style")) or norm(data.get("tone")) or "normal"
        # An explicit non-normal tone wins over a defaulted beat_style.
        if style == "normal" and norm(data.get("tone")) not in (None, "normal"):
            style = norm(data.get("tone")) or "normal"
        data["beat_style"] = style
        data["tone"] = style
        for key in ("overlay_asset", "callout_text", "media_path", "stock_asset_id",
                    "license_type", "cutout_path", "badge_path", "lower_third_path",
                    "stock_fallback_of", "pip_asset"):
            if data.get(key) == "":
                data[key] = None
        if data.get("search_queries") is None:
            data["search_queries"] = []
        if data.get("source_id") is None:
            data["source_id"] = ""
        if data.get("director_notes") is None:
            data["director_notes"] = ""
        return data

    @field_validator("clip_audio")
    @classmethod
    def clean_clip_audio(cls, v: str | None) -> str | None:
        v = (v or "").strip().lower()
        return v if v in ("off", "on", "duck") else None

    @field_validator("overlay_mode")
    @classmethod
    def clean_overlay_mode(cls, v: str | None) -> str | None:
        v = (v or "").strip().lower()
        return "pip" if v in ("pip", "inset") else None

    @field_validator("highlights", mode="before")
    @classmethod
    def clean_highlights(cls, v: Any) -> Any:
        if not v:
            return None
        return [h for h in v if isinstance(h, dict)] or None

    @field_validator("crop_region", mode="before")
    @classmethod
    def clean_crop_region(cls, v: Any) -> Any:
        try:
            x, y, w, h = (float(t) for t in v)
        except (TypeError, ValueError):
            return None
        if not (0 <= x < 1 and 0 <= y < 1 and 0.2 <= w <= 1 and 0.2 <= h <= 1
                and x + w <= 1.001 and y + h <= 1.001):
            return None
        return [x, y, w, h]

    @field_validator("transition")
    @classmethod
    def clean_transition(cls, v: str | None) -> str | None:
        if v is None or not str(v).strip():
            return None
        from src.transitions import normalize_name
        return normalize_name(str(v))

    @field_validator("transition_duration")
    @classmethod
    def clean_transition_duration(cls, v: float | None) -> float | None:
        return None if v is None else min(max(float(v), 0.05), 1.5)

    @field_validator("callout_text")
    @classmethod
    def clean_callout(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = re.sub(r"[^\w\s'&+/.\-]", " ", v, flags=re.UNICODE)
        words = cleaned.split()[:5]
        text = " ".join(words)[:40].strip()
        return text or None

    @model_validator(mode="after")
    def sync_duration(self) -> SceneBeat:
        self.tone = self.beat_style.value
        self.duration = round(max(self.end_time - self.start_time, 0.0), 3)
        return self

    def set_times(self, start: float, end: float) -> None:
        self.start_time = round(start, 3)
        self.end_time = round(end, 3)
        self.duration = round(max(end - start, 0.0), 3)


class ScenePlan(BaseModel):
    episode_id: str = ""
    beats: list[SceneBeat] = Field(default_factory=list)

    @field_validator("beats")
    @classmethod
    def validate_plan(cls, v: list[SceneBeat]) -> list[SceneBeat]:
        v = sorted(v, key=lambda b: b.start_time)
        for i, beat in enumerate(v):
            beat.id = i
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_duration(self) -> float:
        if self.beats:
            return round(self.beats[-1].end_time, 3)
        return 0.0


class StockAsset(BaseModel):
    id: str
    source_platform: str
    source_url: str
    local_path: str
    license_type: str
    license_url: str | None = None
    width: int | None = None
    height: int | None = None
    thumb_url: str | None = None
    downloaded_at: str = ""
    filename: str = ""
    author: str | None = None
    beat_id: int | None = None
    query: str = ""
    media_type: str = "image"  # image | video
    duration: float | None = None  # seconds, videos only (after trimming)

    @field_validator("author")
    @classmethod
    def clean_author(cls, v: str | None) -> str | None:
        """Plain single-line author: no HTML, no stray whitespace, no 'unknown author'."""
        if v is None:
            return None
        text = re.sub(r"<[^>]+>", " ", v)
        text = re.sub(r"\s+", " ", text).strip(" ,;-")
        if not text or re.fullmatch(r"(unknown|anonymous|n/?a|none)( author)?", text, re.IGNORECASE):
            return None
        # "X Unknown author X" style duplicates from Commons credit lines
        text = re.sub(r"\s*\bunknown author\b\s*", " ", text, flags=re.IGNORECASE).strip()
        return text or None


class LicenseAudit(BaseModel):
    assets: list[StockAsset] = Field(default_factory=list)

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for a in self.assets:
            counts[a.license_type] = counts.get(a.license_type, 0) + 1
        return counts


class RenderConfig(BaseModel):
    resolution_width: int = 1920
    resolution_height: int = 1080
    fps: int = 30
    video_codec: str = "h264_nvenc"
    audio_codec: str = "aac"
    loudness_target: float = -16.0
