"""Stage 6: FFmpeg composition & render.

Compiles ``scene_plan.json`` into one ffmpeg invocation per beat (each with a correct
``filter_complex`` and real ``-i`` inputs for every overlay), encodes the intermediates once at
final quality, concatenates them with ``-c:v copy`` and muxes the loudness-normalised voiceover.

A beat that fails to render is re-rendered as an explicit fallback card (with a warning in the
log and in ``timeline_summary.md``); it never corrupts the concat.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console

from src import branding_sound, graphics, music, overlays, theme
from src.common import (
    episode_dir,
    load_config,
    probe_duration,
    read_json,
    voice_path,
    write_json,
)
from src.media import AssetResolver, ResolvedMedia
from src.schemas import ScenePlan
from src.stock_video import PLATFORM_LICENSES
from src.transitions import XFADE, resolve_joins

console = Console()

# Layout of split_schematic (1920x1080 reference; scaled when the output size differs).
SPLIT_MARGIN = 48
SPLIT_GAP = 48
SPLIT_TOP = 190
SPLIT_H = 700
SPLIT_LEFT_W = 640
SPLIT_RIGHT_W = 1136
SMALL_PHOTO_W = 1000  # stills narrower than this (at 1920 output) are framed, not upscaled
MIN_TEXT_SCALE = 0.5
PANO_ASPECT = 2.2      # wider stills pan across the strip instead of being cropped
PANO_MAX = 4.5         # beyond this the strip is letterboxed (ffmpeg size limits)
PORTRAIT_ASPECT = 0.95  # taller stills sit on a framed card over a cinematic fill


EXTREME_ASPECT = 3.0
PUNCH_ANCHOR_Y = 0.35  # 0.5 = centre crop; lower keeps more of the top of the frame


def _media_size(media: ResolvedMedia) -> tuple[int, int] | None:
    try:
        if media.is_video:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height", "-of", "csv=p=0:s=x", media.path],
                check=False, capture_output=True, text=True, timeout=30)
            w, h = r.stdout.strip().split("x")[:2]
            return int(w), int(h)
        with Image.open(media.path) as im:
            return im.size
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class BeatCompileError(Exception):
    """A beat cannot be compiled into a filter graph (e.g. missing media)."""


# --------------------------------------------------------------------------- settings


@dataclass
class RenderSettings:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    codec: str = "libx264"
    nvenc_preset: str = "p4"
    x264_preset: str = "medium"
    crf: int = 18
    audio_rate: int = 48000
    audio_bitrate: str = "192k"
    loudness_target: float = -16.0
    loudness_tp: float = -1.5
    loudness_lra: float = 11.0
    drift_zoom: float = 0.05
    hype_drift_zoom: float = 0.09
    technical_drift_zoom: float = 0.025
    punch_zoom: float = 1.18
    punch_ramp: float = 0.25
    micro_punch_zoom: float = 1.06
    micro_punch_seconds: float = 0.15
    video_drift: bool = False     # synthetic Ken Burns on video clips (off: only punch-ins)
    outro: bool = True            # hold + fade-out after the last spoken word
    outro_hold: float = 1.0       # seconds of picture/silence after the voiceover ends
    outro_fade: float = 0.8       # fade to black (video) and silence (audio) at the very end
    intro_fade: float = 0.3       # fade in from black / silence at the start (0 = off)
    entrance: float = 0.25
    transitions: bool = True
    transition_duration: float = 0.35
    transition_max_group: int = 6
    workers: int = 3              # parallel beat encodes (NVENC gave ~1.5x at 6 sessions)
    theme: str = "classic"        # classic | psycho_terminal
    crt: str = "off"              # off | subtle | strong (grade, scanlines, vignette)
    branding: bool = False        # boot-sequence intro/outro cards
    intro_seconds: float = 3.0
    outro_seconds: float = 3.0
    branding_sound: bool = True
    branding_sound_gain: float = 1.0
    hud: bool = False             # REC dot + T+mm:ss corner timecode
    glitch_transitions: bool = False
    # render.split_layout: {kind: {left_w: px at 1920}}; "default" applies to every other media
    split_layout: dict[str, Any] = field(default_factory=dict)
    draft: bool = False           # fast preview: low res, ultrafast x264, static camera
    draft_size: tuple[int, int] = (640, 360)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RenderSettings:
        pipe = config.get("pipeline", {})
        res = pipe.get("resolution", {})
        r = config.get("render", {})
        th = theme.normalize(config.get("theme"))
        term = th == theme.TERMINAL
        return cls(
            theme=th,
            crt=theme.normalize_crt(r.get("crt", "subtle" if term else "off")),
            branding=bool(r.get("branding", term)),
            intro_seconds=max(float(r.get("intro_seconds", 3.0)), 0.5),
            outro_seconds=max(float(r.get("outro_seconds", 3.0)), 0.5),
            branding_sound=bool(r.get("branding_sound", True)),
            branding_sound_gain=max(float(r.get("branding_sound_gain", 1.0)), 0.0),
            hud=bool(r.get("hud", False)),
            glitch_transitions=bool(r.get("glitch_transitions", term)),
            width=int(res.get("width", 1920)),
            height=int(res.get("height", 1080)),
            fps=int(pipe.get("fps", 30)),
            codec=str(pipe.get("video_codec", "h264_nvenc")),
            nvenc_preset=str(pipe.get("preset", "p4")),
            x264_preset=str(pipe.get("x264_preset", "medium")),
            crf=int(pipe.get("crf", 18)),
            audio_rate=int(pipe.get("audio_sample_rate", 48000)),
            audio_bitrate=str(pipe.get("audio_bitrate", "192k")),
            loudness_target=float(r.get("loudness_target", -16.0)),
            loudness_tp=float(r.get("loudness_true_peak", -1.5)),
            loudness_lra=float(r.get("loudness_lra", 11.0)),
            drift_zoom=float(r.get("drift_zoom", 0.05)),
            hype_drift_zoom=float(r.get("hype_drift_zoom", 0.09)),
            technical_drift_zoom=float(r.get("technical_drift_zoom", 0.025)),
            punch_zoom=float(r.get("punch_zoom", 1.18)),
            punch_ramp=float(r.get("punch_ramp_seconds", 0.25)),
            micro_punch_zoom=float(r.get("micro_punch_zoom", 1.06)),
            micro_punch_seconds=float(r.get("micro_punch_seconds", 0.15)),
            entrance=float(r.get("entrance_seconds", 0.25)),
            video_drift=bool(r.get("video_drift", False)),
            outro=bool(r.get("outro", True)),
            outro_hold=max(float(r.get("outro_hold", 1.0)), 0.0),
            outro_fade=max(float(r.get("outro_fade", 0.8)), 0.0),
            intro_fade=max(float(r.get("intro_fade", 0.3)), 0.0),
            transitions=bool(r.get("transitions", True)),
            transition_duration=float(r.get("transition_duration", 0.35)),
            transition_max_group=int(r.get("transition_max_group", 6)),
            workers=max(1, min(int(r.get("workers", 3)), 6)),
            split_layout=dict(r.get("split_layout") or {}),
        )

    def as_draft(self) -> RenderSettings:
        return dataclasses.replace(self, draft=True, codec="libx264", x264_preset="ultrafast",
                                   crf=30, transitions=False, crt="off", branding=False,
                                   hud=False)

    def sx(self, v: int) -> int:
        """Scale a 1920-reference horizontal measure to the output width (even)."""
        return int(round(v * self.width / 1920 / 2) * 2)

    def sy(self, v: int) -> int:
        return int(round(v * self.height / 1080 / 2) * 2)

    def encoder_args(self) -> list[str]:
        if self.codec == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", self.nvenc_preset, "-tune", "hq",
                    "-rc", "vbr", "-cq", str(self.crf), "-b:v", "0", "-bf", "0",
                    "-pix_fmt", "yuv420p"]
        return ["-c:v", "libx264", "-preset", self.x264_preset, "-crf", str(self.crf),
                "-pix_fmt", "yuv420p"]


# --------------------------------------------------------------------------- motion


@dataclass(frozen=True)
class Motion:
    """Camera move. ``kind``: static | drift (slow Ken Burns) | punch (ease-out zoom, then hold)."""

    kind: str = "static"
    start_zoom: float = 1.0
    end_zoom: float = 1.0
    ramp_frames: int = 1
    total_frames: int = 1
    fx: float = 0.5  # horizontal anchor of the crop inside its slack (0 left .. 1 right)
    fy: float = 0.5

    def zoom_expr(self, var: str = "in") -> str:
        if self.kind == "punch":
            r = max(self.ramp_frames, 1)
            return f"1+({self.end_zoom:.4f}-1)*(1-pow(1-min({var}/{r},1),3))"
        n = max(self.total_frames - 1, 1)
        return f"{self.start_zoom:.4f}+({self.end_zoom - self.start_zoom:.4f})*min({var}/{n},1)"


STATIC = Motion()


def motion_for(beat: dict[str, Any], template: str, s: RenderSettings, frames: int) -> Motion:
    """Pick the camera move from template + beat_style (+ emphasis)."""
    if s.draft:
        return STATIC
    style = beat.get("beat_style", "normal")
    direction = 1 if int(beat.get("id", 0)) % 2 == 0 else -1
    ramp = max(round(s.punch_ramp * s.fps), 2)
    micro = max(round(s.micro_punch_seconds * s.fps), 2)

    if template == "punch_in_emphasis":
        zoom = {"hype": s.punch_zoom + 0.04, "technical": s.punch_zoom - 0.05,
                "deadpan": s.punch_zoom - 0.08}.get(style, s.punch_zoom)
        # crop anchored slightly above centre so top text/heads are not cut off
        return Motion("punch", end_zoom=zoom, ramp_frames=ramp, total_frames=frames,
                      fy=PUNCH_ANCHOR_Y)

    if style == "deadpan" and not beat.get("emphasis"):
        return STATIC
    if style == "fail" or beat.get("emphasis"):
        return Motion("punch", end_zoom=s.micro_punch_zoom, ramp_frames=micro, total_frames=frames)

    amount = {"hype": s.hype_drift_zoom, "technical": s.technical_drift_zoom}.get(
        style, s.drift_zoom)
    if template in ("cutout_callout", "split_schematic"):
        amount *= 0.6
    fx = 0.5 + 0.3 * direction
    if direction > 0:
        return Motion("drift", 1.0, 1.0 + amount, total_frames=frames, fx=fx)
    return Motion("drift", 1.0 + amount, 1.0, total_frames=frames, fx=fx)


# --------------------------------------------------------------------------- graph builder


@dataclass
class FilterJob:
    input_args: list[str] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    n_inputs: int = 0

    def add_media(self, media: ResolvedMedia, fps: int) -> int:
        if media.is_video:
            if media.trim_start:  # semantic trim: start at the best-matching interval
                self.input_args += ["-ss", f"{media.trim_start:.3f}"]
            self.input_args += ["-stream_loop", "-1", "-i", media.path]
        else:
            self.input_args += ["-loop", "1", "-framerate", str(fps), "-i", media.path]
        self.n_inputs += 1
        return self.n_inputs - 1

    def add_image(self, path: str, fps: int) -> int:
        self.input_args += ["-loop", "1", "-framerate", str(fps), "-i", path]
        self.n_inputs += 1
        return self.n_inputs - 1

    def add_chain(self, chain: str) -> None:
        self.chains.append(chain)

    @property
    def graph(self) -> str:
        return ";".join(self.chains)


def _fit(w: int, h: int, contain: bool = False) -> str:
    """Cover-crop to w x h; ``contain`` letterboxes instead (extreme aspect ratios)."""
    if contain:
        return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0a0a1a,setsar=1")
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"


def motion_chain(src: int | str, is_video: bool, w: int, h: int, motion: Motion, fps: int,
                 out: str, contain: bool = False, video_drift: bool = False) -> str:
    """``[src:v]`` -> normalised w x h stream ``[out]`` with the camera move applied.

    The media is cover-fitted to w x h once and the move is a ``perspective`` warp whose source
    window (zoom + anchor) is evaluated per frame in floating point and resampled with cubic
    interpolation, so there is no integer rounding of the crop position (zoompan's rounding was
    the "shimmer"). Slow Ken Burns drift is skipped for videos unless ``video_drift`` (real
    footage already moves; a synthetic drift only amplifies its shake); punch-ins still apply.
    Videos are conformed to ``fps``; stills arrive as a looped image stream at ``fps``.
    """
    pre = f"fps={fps}," if is_video else ""
    label = f"[{src}]" if isinstance(src, str) else f"[{src}:v]"
    if motion.kind == "drift" and is_video and not video_drift:
        motion = STATIC
    if motion.kind == "static":
        return f"{label}{pre}{_fit(w, h, contain)},format=yuv420p[{out}]"
    z = f"({motion.zoom_expr('on')})"
    ww, hh = f"({w}/{z})", f"({h}/{z})"
    x0 = f"(({w}-{ww})*{motion.fx:.3f})"
    y0 = f"(({h}-{hh})*{motion.fy:.3f})"
    persp = (f"perspective=x0='{x0}':y0='{y0}':x1='{x0}+{ww}':y1='{y0}':"
             f"x2='{x0}':y2='{y0}+{hh}':x3='{x0}+{ww}':y3='{y0}+{hh}':"
             "interpolation=cubic:eval=frame:sense=source")
    return f"{label}{pre}{_fit(w, h, contain)},{persp},setsar=1,format=yuv420p[{out}]"


def _slide(final: str, travel: int, entrance: float, delay: float = 0.0, axis: int = 1) -> str:
    """overlay coordinate expression: eases from ``final+axis*travel`` to ``final`` (cubic)."""
    t = f"max(t-{delay:.3f},0)" if delay else "t"
    sign = "+" if axis > 0 else "-"
    return f"{final}{sign}pow(1-min({t}/{entrance:.3f},1),3)*{travel}"


def _fade_in_chain(src: int, out: str, start: float, fade_out_at: float | None = None) -> str:
    chain = f"[{src}:v]format=rgba,fade=t=in:st={start:.3f}:d=0.2:alpha=1"
    if fade_out_at is not None and fade_out_at > start + 0.4:
        chain += f",fade=t=out:st={fade_out_at:.3f}:d=0.2:alpha=1"
    return chain + f"[{out}]"


def _exists(path: str | None) -> bool:
    return bool(path) and Path(str(path)).is_file()


def asset_kind(media: ResolvedMedia | None) -> str:
    """``kind: terminal`` (etc.) hint from the asset's sidecar ``<file>.txt``; '' if absent."""
    if media is None:
        return ""
    try:
        text = Path(media.path + ".txt").read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"\bkind\s*[:=]\s*([\w-]+)", text, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def split_widths(beat: dict[str, Any], ctx: CompileContext) -> tuple[int, int]:
    """(left media width, right diagram width) in 1920-px reference units for a split beat.
    ``render.split_layout.<kind>.left_w`` overrides the default 640 px media panel for assets
    whose sidecar says ``kind: <kind>`` (e.g. terminal/code recordings need more room)."""
    layout = ctx.settings.split_layout
    kind = asset_kind(ctx.resolver.beat_main(beat))
    conf = layout.get(kind) or layout.get("default") or {}
    total = SPLIT_LEFT_W + SPLIT_RIGHT_W
    left = int(conf.get("left_w", SPLIT_LEFT_W))
    left = min(max(left, 320), total - 480)
    return left, total - left


@dataclass
class CompileContext:
    resolver: AssetResolver
    settings: RenderSettings
    work_dir: Path
    tmp_dir: Path
    warnings: list[str] = field(default_factory=list)
    codec_switched: bool = False
    head: float = 0.0   # seconds of the incoming transition: overlay entrances wait for it
    tail: float = 0.0   # seconds of the outgoing transition: lower-thirds are gone before it
    edges: dict[int, tuple[bool, bool]] = field(default_factory=dict)  # beat id -> (first, last)
    _sizes: dict[str, tuple[int, int] | None] = field(default_factory=dict)
    _warned: set[str] = field(default_factory=set)
    _cards: set[Any] | None = None
    render_stats: dict[str, int] = field(default_factory=dict)

    def size(self, media: ResolvedMedia) -> tuple[int, int] | None:
        if media.path not in self._sizes:
            self._sizes[media.path] = _media_size(media)
        size = self._sizes[media.path]
        return size if size and min(size) > 0 else None

    def contain(self, media: ResolvedMedia) -> bool:
        """True for extreme aspect ratios (> 3:1 either way): letterbox instead of crop so the
        2x upscale in the motion chain cannot overflow ffmpeg's size limits."""
        if media.path not in self._sizes:
            self._sizes[media.path] = _media_size(media)
        size = self._sizes[media.path]
        if not size or min(size) <= 0:
            return False
        ratio = size[0] / size[1]
        extreme = ratio > EXTREME_ASPECT or ratio < 1 / EXTREME_ASPECT
        if extreme and media.path not in self._warned:
            self._warned.add(media.path)
            self.warnings.append(f"{Path(media.path).name} has an extreme aspect ratio "
                                 f"({size[0]}x{size[1]}); letterboxed instead of cropped")
        return extreme

    def is_card_beat(self, beat_id: Any) -> bool:
        """Stage 5 replaced this beat's cutout by a framed card (quality gate)."""
        if self._cards is None:
            self._cards = set()
            try:
                man = json.loads((self.work_dir / "cutouts_manifest.json").read_text())
                self._cards = {d.get("beat") for d in man.get("cutout_decisions", [])
                               if d.get("mode") == "card"}
            except (OSError, ValueError):
                pass
        return beat_id in self._cards

    @property
    def grid_path(self) -> str:
        grid = self.work_dir / "grid_background.png"
        if not grid.is_file():
            graphics.make_grid_background(grid, self.settings.width, self.settings.height)
        return str(grid)

    @property
    def scanlines_path(self) -> str:
        path = self.work_dir / f"scanlines_{self.settings.crt}_{self.settings.width}x" \
                               f"{self.settings.height}.png"
        if not path.is_file():
            theme.make_scanlines(path, self.settings.width, self.settings.height,
                                 self.settings.crt)
        return str(path)

    def card(self, name: str, title: str, subtitle: str, w: int, h: int) -> ResolvedMedia:
        path = self.tmp_dir / f"{name}.png"
        graphics.make_card(path, title, subtitle, w, h)
        return ResolvedMedia(path=str(path), is_video=False, asset_id=name, asset_type="card")


def compile_beat(beat: dict[str, Any], frames: int, ctx: CompileContext) -> FilterJob:
    """Build the ffmpeg inputs + filter_complex (final label ``[vout]``) for one beat."""
    template = beat.get("visual_template", "fullscreen_clip")
    if template == "split_schematic":
        job = _compile_split(beat, frames, ctx)
    elif template == "cutout_callout":
        job = _compile_cutout(beat, frames, ctx)
    else:
        job = _compile_fullscreen(beat, frames, ctx, template)
    if beat.get("highlights") or beat.get("overlay_mode") == "pip":
        m = re.fullmatch(r"\[(\w+)\]format=yuv420p\[vout\]", job.chains[-1]) \
            if job.chains else None
        if m:
            job.chains.pop()
            last = m.group(1)
            if beat.get("highlights"):
                last = _add_highlights(job, beat, ctx, last, frames)
            if beat.get("overlay_mode") == "pip":
                last = _add_pip(job, beat, ctx, last, frames)
            job.add_chain(f"[{last}]format=yuv420p[vout]")
    return job


def _fit_rect(size: tuple[int, int] | None, box: tuple[int, int, int, int]
              ) -> tuple[int, int, int, int]:
    """Rectangle a picture of ``size`` occupies when fitted (contain) inside ``box``."""
    x, y, w, h = box
    if not size:
        return box
    k = min(w / size[0], h / size[1])
    fw, fh = round(size[0] * k), round(size[1] * k)
    return x + (w - fw) // 2, y + (h - fh) // 2, fw, fh


def highlight_trusted(h: dict[str, Any]) -> tuple[bool, str]:
    """A highlight is drawn only when its position is explicit (sidecar / hand-written numeric
    box) or located by OCR/VLM. A bare named region ('top', 'bottom'...) with no provenance is
    an old guess that lands on titles and unrelated text: skipped."""
    if h.get("source") in ("sidecar", "ocr", "vlm", "manual"):
        return True, ""
    if h.get("region") and not any(k in h for k in "xywh"):
        return False, f"unlocated region '{h['region']}' (no OCR/VLM/sidecar position)"
    try:
        _x, _y, w, hh = (float(h.get(k, 0)) for k in "xywh")
    except (TypeError, ValueError):
        return False, "invalid box"
    if 0 < max(w, hh) <= 1.0 and (w < 0.01 or hh < 0.008):
        return False, "empty box"
    return True, ""


def _add_highlights(job: FilterJob, beat: dict[str, Any], ctx: CompileContext, last: str,
                    frames: int) -> str:
    """Pulsing boxes/arrows with a label over the picture (Pillow plates, ffmpeg alpha pulse)."""
    s = ctx.settings
    ref = (0, 0, s.width, s.height)
    if beat.get("visual_template") == "split_schematic":
        right = ctx.resolver.beat_overlay(beat)
        left_w, right_w = split_widths(beat, ctx)
        panel = (s.sx(SPLIT_MARGIN + left_w + SPLIT_GAP), s.sy(SPLIT_TOP),
                 s.sx(right_w), s.sy(SPLIT_H))
        ref = _fit_rect(ctx.size(right) if right else None, panel)
    else:
        main = ctx.resolver.beat_main(beat)
        if main is not None and main.asset_type == "schematic":
            ref = _fit_rect(ctx.size(main), ref)
    accent = graphics.accent_for(beat.get("beat_style", "normal"))
    dur = frames / s.fps
    for n, h in enumerate((beat.get("highlights") or [])[:4]):
        ok, why = highlight_trusted(h)
        if not ok:
            ctx.warnings.append(f"beat {beat.get('id')}: highlight skipped ({why})")
            continue
        x, y, w, hh = overlays.resolve_box(h, ref, s.width)
        png, dx, dy = overlays.make_highlight(
            ctx.tmp_dir / f"hl_{beat.get('id')}_{n}.png", w, hh, str(h.get("label") or ""),
            accent, max(s.sx(5), 3), max(s.sx(30), 14), str(h.get("shape") or "box"))
        k = job.add_image(str(png), s.fps)
        start = ctx.head + float(h.get("start") if h.get("start") is not None else 0.6)
        chain = (f"[{k}:v]format=yuva420p,geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':"
                 f"a='alpha(X,Y)*(0.62+0.38*sin(2*PI*1.8*T))',"
                 f"fade=t=in:st={start:.3f}:d=0.2:alpha=1")
        end = h.get("duration")
        out_at = start + float(end) if end else dur - ctx.tail - 0.3
        if out_at > start + 0.4:
            chain += f",fade=t=out:st={out_at:.3f}:d=0.2:alpha=1"
        job.add_chain(chain + f"[hl{n}]")
        job.add_chain(f"[{last}][hl{n}]overlay=x={max(x + dx, 0)}:y={max(y + dy, 0)}:"
                      f"format=auto[hlo{n}]")
        last = f"hlo{n}"
    return last


def _add_pip(job: FilterJob, beat: dict[str, Any], ctx: CompileContext, last: str,
             frames: int) -> str:
    """Small rounded inset (clip or still) in a corner: border, shadow, slide+fade entrance."""
    s = ctx.settings
    media = ctx.resolver.beat_pip(beat)
    if beat.get("visual_template") == "split_schematic" or media is None:
        ctx.warnings.append(f"beat {beat.get('id')}: picture-in-picture skipped "
                            "(split_schematic or pip_asset missing)")
        return last
    corner = str(beat.get("pip_corner") or
                 ("tl" if beat.get("visual_template") == "cutout_callout" else "tr"))
    if corner not in overlays.PIP_CORNERS:
        corner = "tr"
    size = ctx.size(media)
    pw, ph, x, y, b = overlays.pip_geometry(s.width, s.height,
                                            size[0] / size[1] if size else 16 / 9, corner)
    accent = graphics.accent_for(beat.get("beat_style", "normal"))
    frame, mask, pad = overlays.make_pip_frame(ctx.tmp_dir / f"pip_{beat.get('id')}", pw, ph, b,
                                               accent)
    vi = job.add_media(media, s.fps)
    fi = job.add_image(str(frame), s.fps)
    mi = job.add_image(str(mask), s.fps)
    delay = ctx.head + float(beat["pip_delay"] if beat.get("pip_delay") is not None else 0.4)
    fade = f"fade=t=in:st={delay:.3f}:d=0.25:alpha=1"
    pre = f"fps={s.fps}," if media.is_video else ""
    job.add_chain(f"[{vi}:v]{pre}scale={pw}:{ph}:force_original_aspect_ratio=increase,"
                  f"crop={pw}:{ph},setsar=1,format=yuv420p[pipsrc]")
    job.add_chain(f"[{mi}:v]format=gray[pipmask]")
    job.add_chain(f"[pipsrc][pipmask]alphamerge,{fade}[pipv]")
    job.add_chain(f"[{fi}:v]format=rgba,{fade}[pipf]")
    axis = -1 if corner.endswith("l") else 1
    travel = pw // 2
    fx = _slide(str(x - pad), travel, s.entrance + 0.1, axis=axis, delay=delay)
    vx = _slide(str(x + b), travel, s.entrance + 0.1, axis=axis, delay=delay)
    job.add_chain(f"[{last}][pipf]overlay=x='{fx}':y={y - pad}:format=auto[pipo1]")
    job.add_chain(f"[pipo1][pipv]overlay=x='{vx}':y={y + b}:format=auto[pipo2]")
    return "pipo2"


def _require_main(beat: dict[str, Any], ctx: CompileContext) -> ResolvedMedia:
    main = ctx.resolver.beat_main(beat)
    if main is None:
        refusal = ctx.resolver.stock_refusal(beat)
        if refusal:
            raise BeatCompileError(f"stock media refused: {refusal}")
        which = beat.get("stock_asset_id") or beat.get("source_id") or "(none)"
        raise BeatCompileError(
            f"media for {beat.get('primary_source', 'local')} source '{which}' is missing or "
            "undecodable")
    return main


def _warn_small_text(beat: dict[str, Any], media: ResolvedMedia, size: tuple[int, int] | None,
                     shown_w: int, ctx: CompileContext) -> None:
    """Diagram text scales with the displayed width: warn when it ends below ~50% of native."""
    if size and shown_w / size[0] < MIN_TEXT_SCALE:
        ctx.warnings.append(f"beat {beat.get('id')}: diagram {Path(media.path).name} is shown at "
                            f"{shown_w / size[0]:.0%} of its size; its text may be unreadable")


def _framed_photo(job: FilterJob, i: int, media: ResolvedMedia, size: tuple[int, int],
                  beat: dict[str, Any], frames: int, ctx: CompileContext) -> None:
    """Low-resolution still: sharp card (at most 1.5x native, mild drift) on a blurred fill of
    itself instead of a full-frame upscale. Output label ``[bg]``."""
    s = ctx.settings
    aspect = size[0] / size[1]
    max_w, max_h = round(s.width * 0.86), round(s.height * 0.8)
    fw, fh = (max_w, round(max_w / aspect)) if max_w / max_h < aspect \
        else (round(max_h * aspect), max_h)
    cap = min(1.5 * size[0] / fw, 1.0)
    fw, fh = round(fw * cap), round(fh * cap)
    fw, fh = fw - fw % 2, fh - fh % 2
    job.add_chain(f"[{i}:v]split[ps_a][ps_b]")
    # cinematic fill: the still cover-scaled, heavily blurred, darkened, with a vignette
    job.add_chain(f"[ps_a]{_fit(s.width, s.height)},gblur=sigma=45:steps=3,"
                  f"eq=brightness=-0.38:saturation=0.8,vignette=angle=0.5,setsar=1[ps_fill]")
    motion = motion_for(beat, "cutout_callout", s, frames)  # the mild (0.6x) drift
    job.add_chain(motion_chain("ps_b", False, fw, fh, motion, s.fps, "ps_fg0"))
    job.add_chain(f"[ps_fg0]drawbox=x=0:y=0:w=iw:h=ih:color={'0x00ff41@0.55' if s.theme == theme.TERMINAL else 'white@0.3'}:t={max(s.sx(3), 2)}[ps_fg]")
    job.add_chain("[ps_fill][ps_fg]overlay=x=(W-w)/2:y=(H-h)/2,setsar=1[bg]")


def _lt_width(path: str) -> int:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size[0]
    except OSError:
        return 0


def _compile_fullscreen(beat: dict[str, Any], frames: int, ctx: CompileContext,
                        template: str) -> FilterJob:
    s = ctx.settings
    job = FilterJob()
    main = _require_main(beat, ctx)
    i = job.add_media(main, s.fps)
    size = ctx.size(main)
    if main.asset_type == "schematic":
        # a diagram is shown whole, at native scale, never cropped or zoomed (text stays legible)
        pc = graphics.panel_color()
        job.add_chain(
            f"[{i}:v]{f'fps={s.fps},' if main.is_video else ''}scale={s.width}:{s.height}:"
            f"force_original_aspect_ratio=decrease,pad={s.width}:{s.height}:(ow-iw)/2:(oh-ih)/2:"
            f"color=0x{pc[0]:02x}{pc[1]:02x}{pc[2]:02x},setsar=1[bg]")
        _warn_small_text(beat, main, size, s.width, ctx)
    elif not main.is_video and size and PANO_ASPECT <= size[0] / size[1] <= PANO_MAX:
        # panoramic strip: a slow horizontal pan across the whole strip instead of a crop
        dur = max(frames / s.fps, 0.1)
        p = f"t/{dur:.3f}" if int(beat.get("id") or 0) % 2 == 0 else f"(1-t/{dur:.3f})"
        job.add_chain(f"[{i}:v]scale=-2:{s.height}:flags=lanczos,"
                      f"crop={s.width}:{s.height}:x='(iw-ow)*{p}':y=0,setsar=1,"
                      "format=yuv420p[bg]")
    elif (not main.is_video and size and template == "fullscreen_clip"
          and (size[0] < SMALL_PHOTO_W * s.width / 1920
               or size[0] / size[1] < PORTRAIT_ASPECT)):
        _framed_photo(job, i, main, size, beat, frames, ctx)
    else:
        src: int | str = i
        cr = beat.get("crop_region")  # reuse view: detail crop of a still (plan field)
        if cr and len(cr) == 4:  # stills and clips (fix mode can crop a clip)
            job.add_chain(f"[{i}:v]crop=w=trunc(iw*{cr[2]:.4f}/2)*2:h=trunc(ih*{cr[3]:.4f}/2)*2:"
                          f"x=trunc(iw*{cr[0]:.4f}/2)*2:y=trunc(ih*{cr[1]:.4f}/2)*2[crsrc]")
            src = "crsrc"
        job.add_chain(motion_chain(src, main.is_video, s.width, s.height,
                                   motion_for(beat, template, s, frames), s.fps, "bg",
                                   ctx.contain(main), s.video_drift))
    last = "bg"
    if beat.get("illustrative_defocus") and main.asset_type != "schematic":
        job.add_chain("[bg]gblur=sigma=14:steps=2[bgdf]")  # narration says "blurry": illustrative
        last = "bgdf"
        ctx.warnings.append(f"beat {beat.get('id')}: illustrative defocus (picture blurred on "
                            "purpose to match the narration)")
    lt = beat.get("lower_third_path")
    if beat.get("callout_text") and _exists(lt) and main.asset_type == "schematic":
        # the diagram carries its own text: a lower-third would sit on top of its labels
        ctx.warnings.append(f"overlap: beat {beat.get('id')}: lower-third '{beat['callout_text']}'"
                            " suppressed (it would cover the diagram's own labels)")
    elif beat.get("callout_text") and _exists(lt):
        if size is not None and _lt_width(str(lt)) > s.width - 2 * s.sx(96) + 8:
            ctx.warnings.append(f"overlap: beat {beat.get('id')}: lower-third wider than the "
                                "safe area (text may be clipped)")
        k = job.add_image(str(lt), s.fps)
        dur = frames / s.fps
        job.add_chain(_fade_in_chain(k, "lt", 0.15 + ctx.head,
                                     fade_out_at=dur - ctx.tail - 0.35))
        # diagrams carry their own caption near the bottom: sit the lower-third below it
        lt_bottom = s.sy(40) if main.asset_type == "schematic" else s.sy(170)
        job.add_chain(f"[{last}][lt]overlay=x={s.sx(96)}:y=main_h-overlay_h-{lt_bottom}:"
                      f"format=auto[with_lt]")
        last = "with_lt"
    job.add_chain(f"[{last}]format=yuv420p[vout]")
    return job


def _compile_cutout(beat: dict[str, Any], frames: int, ctx: CompileContext) -> FilterJob:
    s = ctx.settings
    job = FilterJob()
    bg_media = ctx.resolver.beat_main(beat)
    if bg_media is None:
        bg_media = ResolvedMedia(ctx.grid_path, False, "grid", "card")
        ctx.warnings.append(f"beat {beat.get('id')}: background media missing, using grid")
    cut = beat.get("cutout_path")
    has_cut = _exists(cut)
    clean_bg = has_cut and (ctx.is_card_beat(beat.get("id"))
                            or beat.get("source_id") == beat.get("overlay_asset"))
    if clean_bg:  # the card/cutout would sit on a blurred copy of itself: dark grid instead
        bg_media = ResolvedMedia(ctx.grid_path, False, "grid", "card")
    i = job.add_media(bg_media, s.fps)
    if not has_cut:
        ctx.warnings.append(f"beat {beat.get('id')}: cutout missing (run stage 5), "
                            "rendering without cutout")
    motion = motion_for(beat, "cutout_callout", s, frames)
    job.add_chain(motion_chain(i, bg_media.is_video, s.width, s.height, motion, s.fps, "bg0",
                               ctx.contain(bg_media), s.video_drift))
    if has_cut and not clean_bg:
        job.add_chain("[bg0]gblur=sigma=6:steps=2,eq=brightness=-0.14:saturation=0.75[bg]")
    else:
        job.add_chain("[bg0]null[bg]")
    last = "bg"
    e = s.entrance
    if has_cut:
        k = job.add_image(str(cut), s.fps)
        job.add_chain(_fade_in_chain(k, "cut", ctx.head))
        # short entrance (fade + a 60 px settle, not a long slide) and a slow continuous drift
        # (a few px/s horizontally, gentle float vertically) so the subject never looks frozen
        t0 = f"max(t-{ctx.head:.3f},0)"
        x = _slide(f"main_w-overlay_w-{s.sx(130)}", s.sx(60), e, delay=ctx.head)
        x = f"{x}-{t0}*{s.sx(6)}"
        y = f"(main_h-overlay_h)/2+sin({t0}*0.9)*{s.sy(6)}"
        job.add_chain(f"[{last}][cut]overlay=x='{x}':y='{y}':format=auto[o1]")
        last = "o1"
    badge = beat.get("badge_path")
    if beat.get("callout_text") and _exists(badge):
        k = job.add_image(str(badge), s.fps)
        job.add_chain(_fade_in_chain(k, "badge", 0.1 + ctx.head))
        y = _slide(f"main_h-overlay_h-{s.sy(150)}", s.sy(70), e, delay=0.1 + ctx.head)
        job.add_chain(f"[{last}][badge]overlay=x={s.sx(130)}:y='{y}':format=auto[o2]")
        last = "o2"
    elif beat.get("callout_text"):
        ctx.warnings.append(f"beat {beat.get('id')}: badge image missing (run stage 5)")
    job.add_chain(f"[{last}]format=yuv420p[vout]")
    return job


def _compile_split(beat: dict[str, Any], frames: int, ctx: CompileContext) -> FilterJob:
    s = ctx.settings
    job = FilterJob()
    left = ctx.resolver.beat_main(beat)
    right = ctx.resolver.beat_overlay(beat)
    if left is None and right is None:
        raise BeatCompileError("neither the left media nor the schematic could be resolved")
    left_w, right_w = split_widths(beat, ctx)
    lw, lh, rw = s.sx(left_w), s.sy(SPLIT_H), s.sx(right_w)
    top = s.sy(SPLIT_TOP)
    if left is None:
        ctx.warnings.append(f"beat {beat.get('id')}: left media missing, using a card")
        left = ctx.card(f"left_{beat.get('id')}", "Media missing", str(beat.get("source_id", "")),
                        lw, lh)
    if right is None:
        ctx.warnings.append(f"beat {beat.get('id')}: schematic missing, using a card")
        right = ctx.card(f"right_{beat.get('id')}", "Schematic missing",
                         str(beat.get("overlay_asset") or ""), rw, lh)
    accent = graphics.accent_hex(beat.get("beat_style", "normal"))
    border = max(s.sx(4), 2)
    if not right.is_video:
        _warn_small_text(beat, right, ctx.size(right), rw - 2 * s.sx(16), ctx)

    g = job.add_image(ctx.grid_path, s.fps)
    job.add_chain(f"[{g}:v]scale={s.width}:{s.height},setsar=1,format=yuv420p[base]")

    li = job.add_media(left, s.fps)
    motion = motion_for(beat, "split_schematic", s, frames)
    lsize = ctx.size(left)
    fw, fh = lw, lh
    if lsize and not ctx.contain(left):
        # Fit the whole frame inside the panel (blurred fill behind) instead of cropping the
        # sides off, so content near the edges of the source survives.
        aspect = lsize[0] / lsize[1]
        fw, fh = (lw, round(lw / aspect)) if lw / lh < aspect else (round(lh * aspect), lh)
        fw, fh = fw - fw % 2, fh - fh % 2
    if (fw, fh) != (lw, lh):
        job.add_chain(f"[{li}:v]split[lsrc_a][lsrc_b]")
        job.add_chain(f"[lsrc_a]{f'fps={s.fps},' if left.is_video else ''}"
                      f"{_fit(lw, lh)},gblur=sigma=24:steps=2,eq=brightness=-0.25[lfill]")
        job.add_chain(motion_chain("lsrc_b", left.is_video, fw, fh, motion, s.fps, "lfg",
                                   video_drift=s.video_drift))
        job.add_chain("[lfill][lfg]overlay=x=(W-w)/2:y=(H-h)/2,setsar=1[left0]")
    else:
        job.add_chain(motion_chain(li, left.is_video, lw, lh, motion, s.fps, "left0",
                                   ctx.contain(left), s.video_drift))
    job.add_chain(f"[left0]drawbox=x=0:y=0:w={lw}:h={lh}:color={accent}:t={border}[left]")

    ri = job.add_media(right, s.fps)
    pre = f"fps={s.fps}," if right.is_video else ""
    inner_w, inner_h = rw - 2 * s.sx(16), lh - 2 * s.sy(16)
    job.add_chain(
        f"[{ri}:v]{pre}scale={inner_w}:{inner_h}:force_original_aspect_ratio=decrease,"
        f"pad={rw}:{lh}:(ow-iw)/2:(oh-ih)/2:color=0x{graphics.panel_color()[0]:02x}"
        f"{graphics.panel_color()[1]:02x}{graphics.panel_color()[2]:02x},"
        f"drawbox=x=0:y=0:w={rw}:h={lh}:color={accent}:t={border},setsar=1,format=yuv420p[right]")

    e = s.entrance
    lx = _slide(str(s.sx(SPLIT_MARGIN)), s.sx(320), e, axis=-1, delay=ctx.head)
    rx_final = s.sx(SPLIT_MARGIN + left_w + SPLIT_GAP)
    rx = _slide(str(rx_final), s.sx(320), e, delay=ctx.head)
    job.add_chain(f"[base][left]overlay=x='{lx}':y={top}[b1]")
    job.add_chain(f"[b1][right]overlay=x='{rx}':y={top}[b2]")
    last = "b2"
    badge = beat.get("badge_path")
    if beat.get("callout_text") and _exists(badge):
        k = job.add_image(str(badge), s.fps)
        job.add_chain(_fade_in_chain(k, "badge", 0.1 + ctx.head))
        job.add_chain(f"[{last}][badge]overlay=x='min({rx_final},main_w-overlay_w-"
                      f"{s.sx(96)})':y={top - s.sy(16)}-overlay_h:format=auto[b3]")
        last = "b3"
    job.add_chain(f"[{last}]format=yuv420p[vout]")
    return job


def compile_card(beat: dict[str, Any], frames: int, ctx: CompileContext, reason: str) -> FilterJob:
    """Last-resort visual for a beat that could not be rendered: a designed title card. The
    technical reason is reported in the log/timeline warnings, never drawn on screen."""
    s = ctx.settings
    title = beat.get("callout_text") or str(beat.get("visual_template", "")).replace("_", " ")
    path = ctx.tmp_dir / f"fallback_{beat.get('id')}.png"
    graphics.make_title_card(path, title, beat.get("beat_style", "normal"), s.width, s.height)
    card = ResolvedMedia(path=str(path), is_video=False, asset_id=path.stem, asset_type="card")
    job = FilterJob()
    i = job.add_media(card, s.fps)
    job.add_chain(motion_chain(i, False, s.width, s.height, STATIC, s.fps, "bg"))
    job.add_chain("[bg]format=yuv420p[vout]")
    return job


# --------------------------------------------------------------------------- ffmpeg


def nvenc_usable() -> bool:
    """True if h264_nvenc can actually encode (encoder listed AND a GPU session opens)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=30",
             "-frames:v", "3", "-c:v", "h264_nvenc", "-f", "null", "-"],
            check=False, capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _beat_command(job: FilterJob, frames: int, out: Path, s: RenderSettings) -> list[str]:
    # -xerror: abort on the first decode/filter error (a corrupt input otherwise loops forever)
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-xerror", *job.input_args,
            "-filter_complex", job.graph, "-map", "[vout]", "-frames:v", str(frames),
            "-r", str(s.fps), "-fps_mode", "cfr", *s.encoder_args(), "-an",
            "-video_track_timescale", "15360", str(out)]


TIMEOUT_RC = 124
STDERR_TAIL = 6000


def _run(cmd: list[str], timeout: float = 900) -> subprocess.CompletedProcess[str]:
    """Run a command with stderr streamed to a temp file (only its tail is kept), so a noisy
    or looping ffmpeg cannot balloon memory. On timeout the process is killed and the result
    has ``returncode == TIMEOUT_RC``."""
    with tempfile.TemporaryFile(mode="w+b") as err:
        rc = 0
        try:
            proc = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=err,
                                  timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = TIMEOUT_RC
        size = err.seek(0, 2)
        err.seek(max(size - STDERR_TAIL, 0))
        tail = err.read().decode("utf-8", "replace")
    if rc == TIMEOUT_RC:
        tail = f"TIMEOUT after {timeout:.0f}s\n{tail}"
    return subprocess.CompletedProcess(cmd, rc, "", tail)


def beat_timeout(frames: int, fps: int) -> float:
    """Generous but bounded: 10x realtime, at least 60s."""
    return max(60.0, frames / fps * 10)


def _encode(job: FilterJob, frames: int, out: Path, ctx: CompileContext, label: str
            ) -> subprocess.CompletedProcess[str]:
    """Encode one beat. A genuine NVENC error (not a timeout) is retried on libx264, and on
    success the whole render switches to libx264."""
    s = ctx.settings
    limit = beat_timeout(frames, s.fps)
    cmd = _beat_command(job, frames, out, s)
    result = _run(cmd, timeout=limit)
    if result.returncode == 0:
        return result
    _log_failure(label, cmd, result)
    if s.codec == "h264_nvenc" and result.returncode != TIMEOUT_RC:
        cpu = dataclasses.replace(s, codec="libx264")
        retry = _run(_beat_command(job, frames, out, cpu), timeout=limit)
        if retry.returncode == 0:
            console.print("  [yellow]NVENC failed mid-run; switching to libx264 for this "
                          "and all remaining beats[/]")
            s.codec = "libx264"
            ctx.codec_switched = True
            ctx.warnings.append("NVENC failed mid-render; remaining beats used libx264")
        return retry
    return result


def _log_failure(label: str, cmd: list[str], result: subprocess.CompletedProcess[str]) -> None:
    console.print(f"  [red]{label} failed (exit {result.returncode})[/]")
    console.print("  cmd: " + " ".join(_quote(c) for c in cmd), markup=False, highlight=False)
    console.print((result.stderr or "")[-1500:], markup=False, highlight=False)


def _quote(arg: str) -> str:
    return f"'{arg}'" if re.search(r"[\s;\[\]()]", arg) else arg


# --------------------------------------------------------------------------- run


def prepare_beats(beats: list[dict[str, Any]], resolver: AssetResolver) -> list[str]:
    """Non-destructive render-time validation of a (possibly hand-written) plan.

    Timing is never changed here beyond what frame_spans does (end times define the cuts).
    Fixes only what would otherwise render wrongly -- a schematic on the left of a split is
    swapped to the right -- and returns warnings for everything questionable. Mutates the
    in-memory beats only; scene_plan.json is untouched.
    """
    from src.stage5_cutouts import pick_subject

    warnings: list[str] = []
    known = {"fullscreen_clip", "split_schematic", "cutout_callout", "punch_in_emphasis"}
    for b in beats:
        i = b["id"]
        tmpl = b.get("visual_template")
        if tmpl not in known:
            warnings.append(f"beat {i}: unknown template '{tmpl}'")
        dur = b["end_time"] - b["start_time"]
        if dur < 1.0:
            warnings.append(f"beat {i} is very short ({dur:.2f}s)")
        if dur > 8.0:
            warnings.append(f"beat {i} is long ({dur:.2f}s)")
        main = resolver.beat_main(b)
        if main is None and b.get("primary_source") == "local":
            warnings.append(f"beat {i}: source '{b.get('source_id')}' not found")
        if tmpl == "split_schematic":
            ov = resolver.beat_overlay(b)
            if main is not None and main.asset_type == "schematic" and (
                    ov is None or ov.asset_type != "schematic") and b.get("source_id"):
                warnings.append(f"beat {i}: schematic was on the left of a split; swapped "
                                "(source_id <-> overlay_asset)")
                b["source_id"], b["overlay_asset"] = b.get("overlay_asset") or "", b["source_id"]
                if not b["source_id"]:  # nothing for the left panel: show it as the main visual
                    b["source_id"], b["overlay_asset"] = b["overlay_asset"], None
                    b["visual_template"] = "fullscreen_clip"
            elif ov is None:
                warnings.append(f"beat {i}: split_schematic has no valid overlay_asset "
                                "(schematic image or clip)")
            elif ov.is_video and main is not None and main.is_video:
                if b.get("overlay_start") is None and b.get("media_start") is None:
                    warnings.append(f"beat {i}: both panels are clips without media_start/"
                                    "overlay_start; they start at their first frames")
            elif ov.is_video:
                pass
        if b.get("overlay_mode") == "pip" and resolver.beat_pip(b) is None:
            warnings.append(f"beat {i}: overlay_mode pip without a valid pip_asset")
        if b.get("clip_audio") in ("on", "duck") and (main is None or not main.is_video):
            warnings.append(f"beat {i}: clip_audio {b['clip_audio']} but the main media is not "
                            "a clip")
        if tmpl == "cutout_callout":
            subject, why = pick_subject(b, resolver)
            if subject is None:
                warnings.append(f"beat {i}: cutout_callout without a usable subject ({why})")
    for i in range(1, len(beats)):
        if abs(beats[i]["start_time"] - beats[i - 1]["end_time"]) > 0.05:
            warnings.append(f"gap/overlap between beats {i - 1} and {i}; beat {i} was snapped "
                            "to the previous end")
    return warnings


def frame_spans(beats: list[dict[str, Any]], fps: int, total: float) -> list[int]:
    """Frames per beat from cumulative end times (no drift); last beat ends at ``total``."""
    spans: list[int] = []
    prev = 0
    for i, beat in enumerate(beats):
        end = total if i == len(beats) - 1 else float(beat["end_time"])
        cur = max(round(end * fps), prev)
        spans.append(cur - prev)
        prev = cur
    return spans


def run(episode_id: str, work_dir: Path, draft: bool = False) -> dict[str, Any]:
    """Run Stage 6: render the final cut (``draft``: fast low-res preview -> out/draft.mp4)."""
    config = load_config()
    s = RenderSettings.from_config(config)
    if draft:
        s = s.as_draft()
    ep_dir = episode_dir(episode_id)
    plan_path = work_dir / "scene_plan.json"
    transcript_path = work_dir / "transcript.json"
    audio_path = voice_path(ep_dir)
    for p in (plan_path, transcript_path, audio_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required input not found: {p}")

    plan = ScenePlan.model_validate(read_json(plan_path))
    transcript = read_json(transcript_path)
    beats = [b.model_dump(mode="json") for b in plan.beats]
    if not beats:
        raise ValueError("scene_plan.json has no beats")

    audio_dur = probe_duration(audio_path) or float(transcript.get("duration", 0.0))
    if audio_dur <= 0:
        raise ValueError(f"Could not determine voiceover duration: {audio_path}")
    plan_end = beats[-1]["end_time"]
    warnings: list[str] = []
    if abs(plan_end - audio_dur) > 0.05:
        warnings.append(f"plan ends at {plan_end:.2f}s but voiceover is {audio_dur:.2f}s; "
                        "the last beat was stretched/trimmed to match the audio")
    resolver = AssetResolver(work_dir)
    warnings.extend(prepare_beats(beats, resolver))
    try:  # editing QA: short beats, unreadable diagrams, sideways joins, static overlays, repeats
        from src import polish

        la = work_dir / "local_assets.json"
        assets_by = polish.assets_by_id(read_json(la)) if la.is_file() else {}
        warnings.extend(f"polish: beat {bid} ({kind}): {msg}" for bid, kind, msg in polish.audit(
            beats, assets_by, polish.transcript_words(transcript), config,
            glitch=s.glitch_transitions))
    except Exception as e:  # noqa: BLE001 - QA hints must never stop a render
        warnings.append(f"polish audit skipped ({e})")

    branded = s.branding and not draft
    intro_dur = round(s.intro_seconds * s.fps) / s.fps if branded else 0.0
    outro_dur = round(s.outro_seconds * s.fps) / s.fps if branded else 0.0
    hold = 0.0 if branded else (s.outro_hold if s.outro else 0.0)
    total_frames = round((audio_dur + hold) * s.fps)
    video_dur = total_frames / s.fps
    spans = frame_spans(beats, s.fps, video_dur)

    if s.codec == "h264_nvenc" and not nvenc_usable():
        console.print("  [yellow]NVENC not usable, using libx264[/]")
        s.codec = "libx264"
    console.print(f"  Encoder: {s.codec} | {s.width}x{s.height}@{s.fps} | {len(beats)} beats | "
                  f"{video_dur:.2f}s")

    out_dir = ep_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / ("draft.mp4" if draft else "final_cut.mp4")
    beat_notes: dict[int, str] = {}
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="videomaker_") as tmp:
        tmp_dir = Path(tmp)
        ctx = CompileContext(resolver, s, work_dir, tmp_dir)
        # branded: the boot cards carry the fade in/out, the beats themselves do not fade
        ctx.edges = {int(b["id"]): (i == 0 and not branded, i == len(beats) - 1 and not branded)
                     for i, b in enumerate(beats)}
        joins = resolve_joins(beats, spans, s.fps, enabled=s.transitions,
                              glitch=s.glitch_transitions,
                              default_duration=s.transition_duration,
                              chapters=_chapter_starts(beats, transcript),
                              max_group=s.transition_max_group)
        for b, j in zip(beats, joins, strict=True):
            b["_join"] = j.name if j.frames else "none"
            b["_join_frames"] = j.frames
        cache_dir = work_dir / "render_cache"
        cache_dir.mkdir(exist_ok=True)
        code_rev = _code_revision()
        if s.crt != "off":
            _ = ctx.scanlines_path  # create once before worker threads
        graphics.make_grid_background(Path(ctx.grid_path), s.width, s.height)  # theme-correct
        jobs: list[tuple[int, dict[str, Any], int, int, int]] = []
        for bi, (beat, span) in enumerate(zip(beats, spans, strict=True)):
            head = joins[bi].frames
            tail = joins[bi + 1].frames if bi + 1 < len(beats) else 0
            if span + tail <= 0:
                warnings.append(f"beat {int(beat['id'])} has zero frames and was skipped")
                continue
            jobs.append((bi, beat, span + tail, head, tail))
        stats = {"cached": 0, "rendered": 0}
        done: list[tuple[dict[str, Any], int, Path, int, int]] = []
        used_cache: set[Path] = set()
        for attempt in range(2):  # a mid-render NVENC -> x264 switch re-renders everything once
            codec_before = s.codec
            done.clear()
            used_cache.clear()
            results = _render_all(jobs, ctx, cache_dir, tmp_dir, code_rev, len(beats), spans, s)
            for (bi, beat, frames, head, tail), (seg, note, bw, secs, hit, key) in zip(
                    jobs, results, strict=True):
                idx = int(beat["id"])
                if note:
                    beat_notes[idx] = note
                    warnings.append(f"beat {idx}: {note}")
                warnings.extend(bw)
                stats["cached" if hit else "rendered"] += 1
                done.append((beat, frames, seg, head, tail))
                if seg.parent == cache_dir:
                    used_cache.add(seg)
                console.print(f"  beat {idx + 1:02d}/{len(beats)} {beat['visual_template']:<18} "
                              f"{spans[bi] / s.fps:5.2f}s {beat['_join']:<9} "
                              f"{'FALLBACK' if note else 'cached' if hit else 'ok'} "
                              f"({secs:.1f}s)")
            if s.codec == codec_before or attempt:
                break
            console.print("  [yellow]NVENC failed mid-render: re-rendering every beat with "
                          "libx264 so the segments match[/]")
        _prune_cache(cache_dir, used_cache)
        ctx.render_stats = stats
        warnings.extend(_merge_stage5_warnings(ctx.warnings, work_dir))

        parts = _assemble(done, spans_by_id={int(b["id"]): sp for b, sp
                                             in zip(beats, spans, strict=True)},
                          ctx=ctx, tmp_dir=tmp_dir, warnings=warnings)
        if branded:
            parts = [_boot_card("intro", intro_dur, s, tmp_dir, warnings), *parts,
                     _boot_card("outro", outro_dur, s, tmp_dir, warnings)]
        concat_file = tmp_dir / "concat.txt"
        concat_file.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
        events = collect_clip_audio(
            beats, spans, resolver, s.fps,
            float((config.get("render") or {}).get("clip_audio_gain", 1.0)), warnings)
        sfx = None
        if branded and s.branding_sound and s.branding_sound_gain > 0:
            sfx = branding_sound.write_wav(tmp_dir / "branding_sfx.wav", intro_dur, video_dur,
                                           outro_dur, s.audio_rate)
        music_mix = _music_input(config, work_dir, ep_dir, video_dur + intro_dur + outro_dur,
                                 draft, warnings, intro_dur, outro_dur)
        stems = (tmp_dir / "stem_voice.wav", tmp_dir / "stem_music.wav") if music_mix else None
        _mux(concat_file, audio_path, output_path, video_dur, s, events, warnings,
             lead=intro_dur, tail_extra=outro_dur, sfx=sfx, music_mix=music_mix, stems=stems)
        music_qa = None
        if stems and all(p.is_file() for p in stems):
            try:
                music_qa = music.qa_music(stems[0], stems[1])
            except (OSError, ValueError, EOFError) as e:
                warnings.append(f"music QA failed: {e}")

    if draft:
        duration = probe_duration(output_path) or 0.0
        console.print(f"  Draft complete: {output_path} ({duration:.2f}s, "
                      f"{time.time() - started:.0f}s)")
        return {"output_path": str(output_path), "codec": s.codec, "duration": duration,
                "draft": True, "warnings": warnings}
    if (config.get("render") or {}).get("generate_srt", False):
        _generate_srt(transcript, out_dir / "final_cut.srt")
    _write_output_audit(work_dir, out_dir, beats, resolver, beat_notes)

    duration = probe_duration(output_path) or 0.0
    join_info = {int(b["id"]): (b["_join"], b["_join_frames"] / s.fps) for b in beats}
    _generate_timeline_summary(plan.model_dump(mode="json"), transcript, resolver, warnings,
                               duration, episode_id, out_dir / "timeline_summary.md", join_info,
                               outro=hold + outro_dur, intro=intro_dur)

    static_spans = [
        (intro_dur + float(b["start_time"]), intro_dur + float(b["end_time"])) for b in beats
        if str(b.get("source_id") or "").startswith("schematic")
        or str(b.get("visual_template")) in ("split_schematic",
                                              "VisualTemplate.split_schematic")]
    qa = qa_gate(output_path, out_dir, warnings, video_dur + intro_dur + outro_dur,
                 lead=intro_dur, tail_card=outro_dur, static_spans=static_spans)
    with (out_dir / "timeline_summary.md").open("a", encoding="utf-8") as fh:
        fh.write("\n## QA gate\n\n" + (
            "Passed: no black gaps, no freeze > 4 s, no silence > 1.5 s, loudness on target.\n"
            if qa["passed"] else "\n".join(f"- {i}" for i in qa["issues"]) + "\n"))
        fh.write(f"\n- loudness: {qa['loudness']}\n- callout overlap warnings: "
                 f"{qa['callout_overlap_warnings']}\n- segments: {ctx.render_stats}\n")
    if music_qa is not None:
        _finish_music_qa(music_qa, qa, out_dir, music_mix[1])
    for i in qa["issues"]:
        warnings.append(f"QA: {i}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"  Render complete: {output_path} ({size_mb:.1f} MB, {duration:.2f}s, "
                  f"{time.time() - started:.0f}s)")
    for w in warnings:
        console.print(f"  [yellow]warning:[/] {w}")
    return {"output_path": str(output_path), "codec": s.codec,
            "resolution": f"{s.width}x{s.height}", "fps": s.fps, "duration": duration,
            "file_size_mb": round(size_mb, 1), "warnings": warnings}


_RENDER_KEYS = ("id", "visual_template", "source_id", "overlay_asset", "stock_asset_id",
                "primary_source", "media_path", "media_start", "media_end", "callout_text",
                "overlay_start", "overlay_mode", "pip_asset", "pip_corner", "pip_start",
                "pip_delay", "highlights", "crop_region",
                "beat_style", "emphasis", "illustrative_defocus", "cutout_path", "badge_path", "lower_third_path",
                "start_time", "end_time", "_join", "_join_frames")
_file_hashes: dict[tuple[str, int, int], str] = {}


def _code_revision() -> str:
    """Hash of the render code: any change to the layouts invalidates the segment cache."""
    h = hashlib.sha256()
    for mod in (__file__, graphics.__file__, overlays.__file__, theme.__file__,
                branding_sound.__file__):
        h.update(Path(mod).read_bytes())
    return h.hexdigest()[:12]


def _file_sig(path: str | None, content: bool = False) -> str:
    if not path:
        return ""
    try:
        st = Path(path).stat()
    except OSError:
        return f"{path}:missing"
    if not content:
        return f"{path}:{st.st_size}:{int(st.st_mtime)}"
    k = (path, st.st_size, int(st.st_mtime))
    if k not in _file_hashes:
        _file_hashes[k] = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return f"{path}:{_file_hashes[k]}"


def beat_cache_key(beat: dict[str, Any], frames: int, head: int, tail: int,
                   ctx: CompileContext, code_rev: str) -> str:
    """sha256 over everything that can change the rendered segment of one beat."""
    main = ctx.resolver.beat_main(beat)
    over = ctx.resolver.beat_overlay(beat)
    parts = {
        "beat": {k: beat.get(k) for k in _RENDER_KEYS},
        "frames": frames, "head": head, "tail": tail,
        "edges": ctx.edges.get(int(beat["id"])),
        "settings": dataclasses.asdict(ctx.settings),
        "main": _file_sig(main.path) if main else "", "overlay": _file_sig(over.path)
        if over else "",
        "files": [_file_sig(beat.get(k), content=True)
                  for k in ("cutout_path", "badge_path", "lower_third_path")],
        "card": ctx.is_card_beat(beat.get("id")), "rev": code_rev,
        "split": split_widths(beat, ctx) if beat.get("visual_template") == "split_schematic"
        else None,
        "grid": _file_sig(ctx.grid_path, content=True),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _render_all(jobs: list[tuple[int, dict[str, Any], int, int, int]], ctx: CompileContext,
                cache_dir: Path, tmp_dir: Path, code_rev: str, n_beats: int,
                spans: list[int], s: RenderSettings
                ) -> list[tuple[Path, str, list[str], float, bool, str]]:
    """Render (or reuse from ``work/render_cache``) every beat, up to ``s.workers`` at once.
    Returns per job: (segment path, fallback note, warnings, seconds, cache hit, key)."""

    def one(job: tuple[int, dict[str, Any], int, int, int]
            ) -> tuple[Path, str, list[str], float, bool, str]:
        _bi, beat, frames, head, tail = job
        t0 = time.time()
        bctx = copy.copy(ctx)  # per-beat head/tail/warnings (the heavy parts are shared)
        bctx.warnings = []
        key = beat_cache_key(beat, frames, head, tail, bctx, code_rev)
        cached = cache_dir / f"beat_{key[:20]}.mp4"
        if cached.is_file() and cached.stat().st_size > 0:
            return cached, "", [], time.time() - t0, True, key
        seg = tmp_dir / f"beat_{int(beat['id']):03d}.mp4"
        note = _render_beat(beat, frames, seg, bctx, head, tail)
        if note:  # fallback cards are never cached: the next run retries the real visual
            return seg, note, bctx.warnings, time.time() - t0, False, key
        part = cache_dir / f".{cached.name}.part"
        shutil.copyfile(seg, part)
        part.replace(cached)
        return cached, "", bctx.warnings, time.time() - t0, False, key

    workers = max(1, s.workers if s.codec == "h264_nvenc" else min(s.workers, 2))
    if workers == 1:
        return [one(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def _prune_cache(cache_dir: Path, keep: set[Path], max_age_days: float = 14.0) -> None:
    """Drop cached segments that this run did not use and that are old."""
    cutoff = time.time() - max_age_days * 86400
    for f in cache_dir.glob("beat_*.mp4"):
        if f not in keep and f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)


def qa_gate(video: Path, out_dir: Path, warnings: list[str], expected: float,
            lead: float = 0.0, tail_card: float = 0.0,
            static_spans: list[tuple[float, float]] | None = None) -> dict[str, Any]:
    """Post-render QA in ONE ffmpeg pass: blackdetect + freezedetect (>4 s only: slow pans on
    stills trip shorter windows) + silencedetect + ebur128. Writes out/qa_report.json."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video),
           "-vf", "blackdetect=d=0.2:pix_th=0.08,freezedetect=n=0.001:d=4",
           "-af", "silencedetect=n=-50dB:d=1.5,ebur128=peak=true", "-f", "null", "-"]
    try:
        err = subprocess.run(cmd, check=False, capture_output=True, text=True,
                             timeout=900).stderr or ""
    except (subprocess.SubprocessError, OSError):
        err = ""

    def spans(start_key: str, end_key: str) -> list[dict[str, float]]:
        starts = [float(x) for x in re.findall(rf"{start_key}:\s*([0-9.]+)", err)]
        ends = [float(x) for x in re.findall(rf"{end_key}:\s*([0-9.]+)", err)]
        return [{"start": a, "end": b} for a, b in zip(starts, ends, strict=False)]

    black = spans("black_start", "black_end")
    freeze = spans("lavfi.freezedetect.freeze_start", "lavfi.freezedetect.freeze_end")
    silence = spans("silence_start", "silence_end")
    # diagrams are meant to sit still while the viewer reads: not a QA freeze
    freeze = [f for f in freeze
              if not any(f["start"] >= a - 0.3 and f["end"] <= b + 0.3
                         for a, b in (static_spans or []))]
    summary = err[err.rfind("Summary:"):] if "Summary:" in err else ""

    def num(label: str) -> float | None:
        m = re.search(rf"{label}:\s*(-?[0-9.]+)", summary)
        return float(m.group(1)) if m else None

    loud = {"integrated_lufs": num("I"), "lra": num("LRA"), "true_peak_dbfs": num("Peak")}
    dur = probe_duration(video) or 0.0
    issues = []
    issues += [f"black frames {b['start']:.2f}-{b['end']:.2f}s" for b in black
               if not (b["start"] < 0.1 or b["end"] > dur - 0.1)  # fades at both ends are wanted
               and not (tail_card and b["start"] >= dur - tail_card - 0.1)]  # dark boot card
    issues += [f"frozen picture {f['start']:.1f}-{f['end']:.1f}s (>4 s)" for f in freeze]
    issues += [f"silence {x['start']:.1f}-{x['end']:.1f}s (>1.5 s)" for x in silence
               if x["end"] < dur - 1.6 and not (lead and x["start"] < 0.1
                                                and x["end"] <= lead + 0.1)]
    if loud["integrated_lufs"] is not None and abs(loud["integrated_lufs"] + 16) > 2.5:
        issues.append(f"loudness {loud['integrated_lufs']:.1f} LUFS is far from the -16 target")
    if abs(dur - expected) > 1 / 30 + 0.02:
        issues.append(f"duration {dur:.3f}s differs from the expected {expected:.3f}s")
    overlaps = [w for w in warnings if w.startswith("overlap:")]
    report = {"video": str(video), "duration": dur, "expected_duration": expected,
              "black": black, "freeze_over_4s": freeze, "silence_over_1_5s": silence,
              "loudness": loud, "callout_overlap_warnings": len(overlaps), "issues": issues,
              "passed": not issues}
    write_json(out_dir / "qa_report.json", report)
    return report


def _finish_music_qa(mqa: dict[str, Any], qa: dict[str, Any], out_dir: Path,
                     cfg: music.MusicCfg) -> None:
    """Fold the music/voice check into qa_report.json (voice must stay intelligible)."""
    mqa["final_lufs"] = qa["loudness"].get("integrated_lufs")
    try:
        meta = read_json(out_dir.parent / "work" / "music.json")
        mqa.update({"source": meta.get("source"), "mood": meta.get("mood"),
                    "license": meta.get("license"), "level_lufs": cfg.level_lufs})
    except (OSError, ValueError):
        pass
    try:
        mqa.update(read_json(out_dir.parent / "work" / "music_duck.json"))
    except (OSError, ValueError):
        pass
    qa["music"] = mqa
    if mqa.get("pumping_db", 0) > max(3.0, cfg.duck_db + 0.5):
        qa["issues"].append(f"music pumping: gain swings {mqa['pumping_db']} dB within 1 s")
        qa["passed"] = False
    if not mqa["passed"]:
        qa["issues"].append(
            f"music too loud under speech: {mqa['music_vs_voice_db_pooled']} dB vs voice "
            f"(limit {mqa['max_allowed_db']} dB)")
        qa["passed"] = False
    write_json(out_dir / "qa_report.json", qa)


def _write_output_audit(work_dir: Path, out_dir: Path, beats: list[dict[str, Any]],
                        resolver: AssetResolver, beat_notes: dict[int, str]) -> None:
    """out/licenses_audit.json: only assets actually used in the cut. Policy-refused assets
    and assets of beats that fell back are moved to ``excluded`` with the reason."""
    src = work_dir / "licenses_audit.json"
    raw = read_json(src) if src.is_file() else {"assets": []}
    refs: dict[str, list[dict[str, Any]]] = {}
    for b in beats:  # every beat that references an asset, as main media or as overlay
        for ident in {b.get("stock_asset_id"), b.get("overlay_asset")} - {None, ""}:
            refs.setdefault(str(ident), []).append(b)
    used: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for a in raw.get("assets", []):
        users = refs.get(a["id"], [])
        rendered = [b for b in users if int(b["id"]) not in beat_notes]
        if a["id"] in resolver.refused:
            excluded.append({**a, "excluded_reason": resolver.refused[a["id"]]})
        elif not users:
            excluded.append({**a, "excluded_reason": "not referenced by the current plan"})
        elif not rendered:
            excluded.append({**a, "excluded_reason": "beat rendered as fallback card"})
        else:
            used.append(a)
    extra: dict[str, Any] = {}
    if (work_dir / "music.wav").is_file() and (work_dir / "music_audit.json").is_file():
        try:
            extra["music"] = read_json(work_dir / "music_audit.json")
        except (OSError, ValueError):
            pass
    write_json(out_dir / "licenses_audit.json",
               {**raw, "assets": used, "excluded": excluded, **extra})


def _merge_stage5_warnings(render_warnings: list[str], work_dir: Path) -> list[str]:
    """Add Stage 5's real failure reasons; drop the generic "cutout missing" notes they explain."""
    manifest = work_dir / "cutouts_manifest.json"
    stage5: list[str] = []
    if manifest.is_file():
        try:
            stage5 = [str(w) for w in read_json(manifest).get("warnings", [])]
        except (OSError, ValueError):
            stage5 = []
    explained = {m.group(0) for w in stage5 if (m := re.match(r"beat \d+:", w))}
    kept = [w for w in render_warnings
            if not ("cutout missing" in w and any(w.startswith(e) for e in explained))]
    return kept + [f"stage 5: {w}" for w in stage5]


def _chapter_starts(beats: list[dict[str, Any]], transcript: dict[str, Any]) -> set[int]:
    """Ids of beats that open after a real pause (>= 0.7s between narration segments)."""
    segs = transcript.get("segments") or []
    out: set[int] = set()
    for a, b in itertools.pairwise(segs):
        try:
            end, nxt = float(a["end"]), float(b["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if nxt - end >= 0.7:
            for beat in beats:
                if end - 0.2 <= float(beat["start_time"]) <= nxt + 0.2:
                    out.add(int(beat["id"]))
                    break
    return out


def _xfade_command(inputs: list[Path], spans: list[int], joins: list[tuple[str, int]], out: Path,
                   s: RenderSettings, hard: bool = False) -> list[str]:
    """Join consecutive beat files (each = span + tail frames) with xfade; the output has
    exactly sum(spans) frames. ``hard`` builds a cut-only equivalent (trim + concat)."""
    fps = s.fps
    norm = f"fps={fps},settb=1/{fps},format=yuv420p,setpts=PTS-STARTPTS"
    parts = [f"[{k}:v]{norm}" + (f",trim=end_frame={spans[k]},setpts=PTS-STARTPTS" if hard
                                 else "") + f"[v{k}]" for k in range(len(inputs))]
    if hard:
        parts.append("".join(f"[v{k}]" for k in range(len(inputs)))
                     + f"concat=n={len(inputs)}:v=1:a=0[vout]")
    else:
        cur, acc = "v0", 0
        for k in range(1, len(inputs)):
            acc += spans[k - 1]
            name, n = joins[k]
            label = "vout" if k == len(inputs) - 1 else f"x{k}"
            parts.append(f"[{cur}][v{k}]xfade=transition={XFADE[name]}:duration={n / fps:.4f}:"
                         f"offset={acc / fps:.4f}[{label}]")
            cur = label
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-xerror"]
    for p in inputs:
        cmd += ["-i", str(p)]
    return cmd + ["-filter_complex", ";".join(parts), "-map", "[vout]",
                  "-frames:v", str(sum(spans)), "-r", str(fps), "-fps_mode", "cfr",
                  *s.encoder_args(), "-an", "-video_track_timescale", "15360", str(out)]


def _assemble(done: list[tuple[dict[str, Any], int, Path, int, int]],
              spans_by_id: dict[int, int], ctx: CompileContext, tmp_dir: Path,
              warnings: list[str]) -> list[Path]:
    """Rendered beat files -> files ready for the stream-copy concat. Beats joined by a
    transition are merged (in runs of at most ``transition_max_group``) into one file."""
    s = ctx.settings
    runs: list[list[int]] = []
    for i, (beat, _f, _p, head, _t) in enumerate(done):
        if i and head and beat["_join"] != "none":
            runs[-1].append(i)
        else:
            runs.append([i])
    parts: list[Path] = []
    for gi, run in enumerate(runs):
        if len(run) == 1:
            parts.append(done[run[0]][2])
            continue
        files = [done[i][2] for i in run]
        spans = [spans_by_id[int(done[i][0]["id"])] for i in run]
        joins = [(done[i][0]["_join"], done[i][3]) for i in run]
        out = tmp_dir / f"join_{gi:03d}.mp4"
        total = sum(spans)
        limit = beat_timeout(total, s.fps)
        cmd = _xfade_command(files, spans, joins, out, s)
        res = _run(cmd, timeout=limit)
        if res.returncode != 0 and s.codec == "h264_nvenc" and res.returncode != TIMEOUT_RC:
            _log_failure(f"transition run {gi}", cmd, res)
            s.codec = "libx264"
            warnings.append("NVENC failed while joining transitions; used libx264")
            res = _run(_xfade_command(files, spans, joins, out, s), timeout=limit)
        if res.returncode != 0:
            _log_failure(f"transition run {gi}", cmd, res)
            res = _run(_xfade_command(files, spans, joins, out, s, hard=True), timeout=limit)
            warnings.append(f"transitions failed for beats {done[run[0]][0]['id']}-"
                            f"{done[run[-1]][0]['id']}; joined with hard cuts")
            if res.returncode != 0:
                raise RuntimeError("Could not join beats (xfade nor cut fallback)")
        parts.append(out)
    return parts


def _boot_card(kind: str, seconds: float, s: RenderSettings, tmp_dir: Path,
               warnings: list[str]) -> Path:
    """Render the intro/outro boot-sequence card (Pillow frames -> encoder args of the run, so
    the stream-copy concat stays valid). Intro fades in, outro fades out."""
    frames_dir = tmp_dir / f"boot_{kind}"
    fade_in = s.intro_fade if kind == "intro" else 0.0
    fade_out = s.outro_fade if kind == "outro" and s.outro else 0.0
    n = theme.write_boot_frames(kind, frames_dir, s.fps, seconds, s.width, s.height,
                                fade_in, fade_out)
    out = tmp_dir / f"boot_{kind}.mp4"

    def cmd(cs: RenderSettings) -> list[str]:
        return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(s.fps),
                "-i", str(frames_dir / "f%04d.png"), "-frames:v", str(n),
                "-vf", "setsar=1,format=yuv420p", "-r", str(s.fps), "-fps_mode", "cfr",
                *cs.encoder_args(), "-an", "-video_track_timescale", "15360", str(out)]
    res = _run(cmd(s), timeout=300)
    if res.returncode != 0 and s.codec == "h264_nvenc":
        s.codec = "libx264"
        warnings.append("NVENC failed on the boot card; used libx264")
        res = _run(cmd(s), timeout=300)
    if res.returncode != 0:
        _log_failure(f"boot card {kind}", cmd(s), res)
        raise RuntimeError(f"Could not render the {kind} boot card")
    return out


def _apply_edge_fades(job: FilterJob, beat: dict[str, Any], frames: int,
                      ctx: CompileContext) -> None:
    """Fade the first beat in from black and the last beat out to black (``ctx.edges``)."""
    s = ctx.settings
    first, last = ctx.edges.get(int(beat.get("id", -1)), (False, False))
    fades = []
    if first and s.intro_fade > 0:
        fades.append(f"fade=t=in:st=0:d={s.intro_fade:.3f}")
    if last and s.outro and s.outro_fade > 0:
        d = min(s.outro_fade, frames / s.fps)
        fades.append(f"fade=t=out:st={frames / s.fps - d:.3f}:d={d:.3f}")
    if not fades or not job.chains or not job.chains[-1].endswith("[vout]"):
        return
    job.chains[-1] = job.chains[-1][:-6] + "[vfx]"
    job.chains.append("[vfx]" + ",".join(fades) + "[vout]")


def _apply_theme_fx(job: FilterJob, beat: dict[str, Any], ctx: CompileContext) -> None:
    """CRT grade/scanlines and the optional corner HUD on the finished beat picture."""
    s = ctx.settings
    if (s.crt == "off" and not s.hud) or not job.chains or not job.chains[-1].endswith("[vout]"):
        return
    job.chains[-1] = job.chains[-1][:-6] + "[vpre]"
    last = "vpre"
    if s.crt != "off":
        k = job.add_image(ctx.scanlines_path, s.fps)
        dst = "vcrt" if s.hud else "vout"
        job.chains += theme.crt_chain(s.crt, last, dst, k, s.width)
        last = dst
    if s.hud:
        job.chains.append(theme.hud_chain(last, "vout", float(beat.get("start_time") or 0.0),
                                          s.width, s.height))
    if last == "vpre":
        job.chains.append("[vpre]null[vout]")


def _draft_scale(job: FilterJob, s: RenderSettings) -> None:
    """Draft renders compile the full-size layout and shrink only the final frame."""
    if not s.draft or not job.chains or not job.chains[-1].endswith("[vout]"):
        return
    w, h = s.draft_size
    job.chains[-1] = job.chains[-1][:-6] + "[vfull]"
    job.chains.append(f"[vfull]scale={w}:{h}:flags=bilinear,setsar=1[vout]")


def _render_beat(beat: dict[str, Any], frames: int, out: Path, ctx: CompileContext,
                 head: int = 0, tail: int = 0) -> str:
    """Render one beat (``frames`` includes the ``tail`` overlap frames); returns "" on
    success or the fallback reason."""
    ctx.head = head / ctx.settings.fps
    ctx.tail = tail / ctx.settings.fps
    reason = ""
    try:
        job = compile_beat(beat, frames, ctx)
        _apply_theme_fx(job, beat, ctx)
        _apply_edge_fades(job, beat, frames, ctx)
        _draft_scale(job, ctx.settings)
        result = _encode(job, frames, out, ctx,
                         f"beat {beat.get('id')} ({beat.get('visual_template')})")
        if result.returncode == 0:
            return ""
        why = "timed out" if result.returncode == TIMEOUT_RC else "ffmpeg failed"
        reason = f"{why}; rendered fallback card"
    except BeatCompileError as e:
        reason = f"{e}; rendered fallback card"
        console.print(f"  [yellow]beat {beat.get('id')}: {e}[/]")

    job = compile_card(beat, frames, ctx, reason.split(";")[0])
    _apply_theme_fx(job, beat, ctx)
    _apply_edge_fades(job, beat, frames, ctx)
    _draft_scale(job, ctx.settings)
    result = _encode(job, frames, out, ctx, f"fallback card for beat {beat.get('id')}")
    if result.returncode != 0:
        raise RuntimeError(f"Could not render beat {beat.get('id')} nor its fallback card")
    return reason


def _measure_loudness(audio: Path, s: RenderSettings) -> dict[str, str] | None:
    """First loudnorm pass; returns measured_* values or None if parsing fails."""
    af = f"loudnorm=I={s.loudness_target}:TP={s.loudness_tp}:LRA={s.loudness_lra}:print_format=json"
    r = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio), "-af", af, "-f", "null",
              "-"], timeout=300)
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.DOTALL)
    if r.returncode != 0 or not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {k: str(data[k]) for k in
                ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except (json.JSONDecodeError, KeyError):
        return None


@dataclasses.dataclass
class ClipAudio:
    path: str
    start: float      # position on the final timeline (seconds)
    dur: float
    trim: float       # offset inside the clip
    gain: float
    duck: bool


def has_audio_stream(path: str) -> bool:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=codec_type", "-of", "csv=p=0", path],
                           check=False, capture_output=True, text=True, timeout=30)
        return "audio" in r.stdout
    except (subprocess.SubprocessError, OSError):
        return False


def collect_clip_audio(beats: list[dict[str, Any]], spans: list[int], resolver: AssetResolver,
                       fps: int, default_gain: float, warnings: list[str]) -> list[ClipAudio]:
    """Per beat with ``clip_audio`` on|duck: the main clip's audio, trimmed with media_start and
    placed at the beat's position. Clips without an audio stream are skipped (never fail)."""
    events: list[ClipAudio] = []
    pos = 0
    for beat, span in zip(beats, spans, strict=True):
        start, pos = pos / fps, pos + span
        mode = beat.get("clip_audio")
        if mode not in ("on", "duck") or span <= 0 or beat.get("primary_source") == "stock":
            continue
        main = resolver.beat_main(beat)
        if main is None or not main.is_video:
            continue
        src = (resolver.local.get(beat.get("source_id") or "") or {}).get("filepath")
        path = src if src and Path(src).is_file() else main.path  # stabilised copies drop audio
        if not has_audio_stream(path):
            warnings.append(f"beat {beat.get('id')}: clip_audio {mode} but the clip has no audio")
            continue
        gain = beat.get("clip_audio_gain")
        events.append(ClipAudio(path, start, span / fps, main.trim_start or 0.0,
                                float(gain if gain is not None else default_gain),
                                mode == "duck"))
    return events


def clip_audio_filter(events: list[ClipAudio], rate: int) -> str:
    """filter_complex turning input 0 (voiceover) + inputs 1..n (clips) into ``[mix]``. Clips
    that speak (``duck``) sidechain-compress the narration; all clips are summed on top."""
    parts: list[str] = []
    for i, e in enumerate(events, start=1):
        fade = min(0.04, e.dur / 4)
        parts.append(f"[{i}:a]aresample={rate},aformat=channel_layouts=stereo,"
                     f"atrim=0:{e.dur:.3f},asetpts=PTS-STARTPTS,volume={e.gain:.3f},"
                     f"afade=t=in:d={fade:.3f},afade=t=out:st={e.dur - fade:.3f}:d={fade:.3f},"
                     f"adelay={round(e.start * 1000)}:all=1"
                     + (f",asplit=2[m{i}][s{i}]" if e.duck else f"[m{i}]"))
    n = len(events)
    parts.append("".join(f"[m{i}]" for i in range(1, n + 1))
                 + f"amix=inputs={n}:normalize=0:duration=longest[cbus]")
    voice = f"[0:a]aresample={rate}[v0]"
    duck_ids = [i for i, e in enumerate(events, start=1) if e.duck]
    if duck_ids:
        parts.append("".join(f"[s{i}]" for i in duck_ids)
                     + f"amix=inputs={len(duck_ids)}:normalize=0:duration=longest[sc0]")
        parts.append(voice)
        parts.append("[v0][sc0]sidechaincompress=threshold=0.03:ratio=9:attack=15:"
                     "release=350:makeup=1[vduck]")
        parts.append("[vduck][cbus]amix=inputs=2:normalize=0:duration=longest[mix]")
    else:
        parts.append(voice)
        parts.append("[v0][cbus]amix=inputs=2:normalize=0:duration=longest[mix]")
    return ";".join(parts)


def _mix_clip_audio(audio: Path, events: list[ClipAudio], out: Path, video_dur: float,
                    s: RenderSettings) -> Path | None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(audio)]
    for e in events:
        cmd += ["-stream_loop", "-1", "-ss", f"{e.trim:.3f}", "-t", f"{e.dur:.3f}", "-i", e.path]
    cmd += ["-filter_complex", clip_audio_filter(events, s.audio_rate), "-map", "[mix]",
            "-t", f"{video_dur:.4f}", "-ar", str(s.audio_rate), "-ac", "2", str(out)]
    res = _run(cmd, timeout=600)
    if res.returncode != 0:
        _log_failure("clip audio mix", cmd, res)
        return None
    return out


def _music_input(config: dict[str, Any], work_dir: Path, ep_dir: Path, total: float,
                 draft: bool, warnings: list[str], lead: float = 0.0, outro: float = 3.0) -> tuple[Path, music.MusicCfg] | None:
    """``work/music.wav`` when the music stage is enabled and has produced a track."""
    cfg = music.MusicCfg.from_config(config)
    if draft or cfg.off:
        return None
    track = work_dir / "music.wav"
    if not track.is_file() and cfg.enabled == "true":  # explicitly on: make it now
        try:
            music.build(ep_dir, cfg)
        except (OSError, ValueError, RuntimeError) as e:
            warnings.append(f"music: could not build a track ({e}); rendered without music")
    if not track.is_file():
        return None
    dur = probe_duration(track) or 0.0
    try:
        stats = music.make_ducked(ep_dir, cfg, lead, total, outro)
        track = work_dir / "music_ducked.wav"
        if stats["pumping_db"] > max(3.0, cfg.duck_db + 0.5):
            warnings.append(f"music gain swings {stats['pumping_db']} dB within 1 s (pumping)")
    except (OSError, ValueError) as e:
        warnings.append(f"music ducking failed ({e}); rendered without music")
        return None
    if abs(dur - total) > 1.0:
        warnings.append(f"music.wav is {dur:.1f}s but the cut is {total:.1f}s (stale?); "
                        "run `music <ep>` - it is looped/trimmed meanwhile")
    return track, cfg


def _mux(concat_file: Path, audio: Path, output: Path, video_dur: float,
         s: RenderSettings, events: list[ClipAudio] | None = None,
         warnings: list[str] | None = None, lead: float = 0.0,
         tail_extra: float = 0.0, sfx: Path | None = None,
         music_mix: tuple[Path, music.MusicCfg] | None = None,
         stems: tuple[Path, Path] | None = None) -> None:
    """Concat the beat videos (stream copy) and mux the normalised voiceover (+ clip audio,
    mixed first so loudnorm measures the final mix)."""
    if events:
        mixed = _mix_clip_audio(audio, events, concat_file.with_name("clip_mix.wav"),
                                video_dur, s)
        if mixed is not None:
            audio = mixed
        elif warnings is not None:
            warnings.append("clip audio mix failed; rendered the voiceover only")
    af = f"loudnorm=I={s.loudness_target}:TP={s.loudness_tp}:LRA={s.loudness_lra}"
    measured = _measure_loudness(audio, s)
    if measured:
        af += (f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
               f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
               f":offset={measured['target_offset']}:linear=true")
    else:
        console.print("  [yellow]loudnorm measurement unavailable, using single-pass[/]")
    total_dur = video_dur + lead + tail_extra  # intro card + body (voiceover) + outro card
    af += f",aresample={s.audio_rate}"
    if lead > 0:  # voiceover starts when the intro card ends: same offset as the picture
        af += f",adelay={round(lead * 1000)}:all=1"
    af += f",apad=whole_dur={total_dur:.4f}"
    video_dur = total_dur
    if s.intro_fade > 0:
        af += f",afade=t=in:st=0:d={s.intro_fade:.3f}"
    if s.outro and s.outro_fade > 0:
        d = min(s.outro_fade, video_dur)
        af += f",afade=t=out:st={video_dur - d:.3f}:d={d:.3f}"
    sfx_in: list[str] = []
    audio_args = ["-map", "1:a:0", "-af", af]
    extra_out: list[str] = []
    if sfx is not None or music_mix is not None:
        # voice (loudness-normalised) + boot sound + ducked music around it
        idx = 2
        parts = [f"[1:a]{af}[vo0]"]
        labels = ["vo"]
        if music_mix is not None:
            parts.append("[vo0]" + ("asplit=2[vo][vst]" if stems else "anull[vo]"))
            parts.append(music.mix_filter(music_mix[1], f"{idx}:a", "md", lead, video_dur,
                                          "mst" if stems else None))
            sfx_in += ["-i", str(music_mix[0])]
            labels.append("md")
            idx += 1
            if stems:
                extra_out = ["-map", "[mst]", "-c:a", "pcm_s16le", "-ar", str(s.audio_rate),
                             str(stems[1]), "-map", "[vst]", "-c:a", "pcm_s16le", "-ar",
                             str(s.audio_rate), str(stems[0])]
        else:
            parts.append("[vo0]anull[vo]")
        if sfx is not None:
            sfx_in += ["-i", str(sfx)]
            parts.append(f"[{idx}:a]volume={s.branding_sound_gain:.3f}[sfx]")
            labels.append("sfx")
        parts.append("".join(f"[{x}]" for x in labels)
                     + f"amix=inputs={len(labels)}:normalize=0:duration=first,"
                     "alimiter=limit=0.95:level=disabled[aout]")
        audio_args = ["-filter_complex", ";".join(parts), "-map", "[aout]"]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(concat_file), "-i", str(audio), *sfx_in,
           "-map", "0:v:0", *audio_args, "-c:v", "copy",
           "-c:a", "aac", "-b:a", s.audio_bitrate, "-ar", str(s.audio_rate), "-ac", "2",
           "-t", f"{video_dur:.4f}", "-movflags", "+faststart", str(output), *extra_out]
    result = _run(cmd, timeout=1800)
    if result.returncode != 0:
        _log_failure("final mux", cmd, result)
        raise RuntimeError("FFmpeg final mux failed")


# --------------------------------------------------------------------------- sidecars


def _seconds_to_srt_time(seconds: float) -> str:
    total_ms = round(max(seconds, 0.0) * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    sec, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _srt_cues(transcript: dict[str, Any], max_chars: int = 84,
              max_seconds: float = 4.5) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    for seg in transcript.get("segments", []):
        words = seg.get("words") or []
        if not words:
            cues.append((seg["start"], seg["end"], seg["text"].strip()))
            continue
        cur: list[dict[str, Any]] = []
        for w in words:
            cur.append(w)
            text = " ".join(x["word"].strip() for x in cur)
            if len(text) >= max_chars or w["end"] - cur[0]["start"] >= max_seconds:
                cues.append((cur[0]["start"], w["end"], text))
                cur = []
        if cur:
            cues.append((cur[0]["start"], cur[-1]["end"],
                         " ".join(x["word"].strip() for x in cur)))
    return cues


def _generate_srt(transcript: dict[str, Any], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(f"{i}\n{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}\n"
                    f"{text}\n\n" for i, (start, end, text) in enumerate(_srt_cues(transcript), 1))


def _stock_label(beat: dict[str, Any], resolver: AssetResolver) -> str:
    """Source + license of a stock beat; platform licenses are flagged as non-CC0."""
    if (beat.get("stock_asset_id") or "") in resolver.refused:
        return "stock: REFUSED by license policy -> fallback card"
    lic = beat.get("license_type") or "unresolved"
    kind = resolver.stock.get(beat.get("stock_asset_id") or "", {}).get("media_type", "image")
    if lic == "generated-fallback":
        return "stock: fallback card (no free media found)"
    if lic in PLATFORM_LICENSES:
        return f"stock {kind}: free-license (non-CC0) - {lic}"
    return f"stock {kind} ({lic})"


def _cell(value: Any) -> str:
    text = str(value) if value not in (None, "") else "-"
    return text.replace("|", "/").replace("\n", " ")


def _semantic_section(beats: list[dict[str, Any]], work: Path) -> list[str]:
    """Query / score / trim per beat; beats below ``semantic.min_score`` are flagged."""
    from src.semantic import min_score_for

    try:
        backend = read_json(work / "semantic_index.json").get("backend", "")
    except (OSError, ValueError):
        backend = ""
    threshold = min_score_for(backend)
    lines = ["", f"## Semantic matching ({backend or 'n/a'}, review threshold {threshold:.2f})", "",
             "| # | Visual query | Score | Trim | Review |", "|---|---|---|---|---|"]
    for b in beats:
        sc = b.get("match_score")
        trim = (f"{b['media_start']:.1f}-{b['media_end']:.1f}s"
                if b.get("media_start") is not None and b.get("media_end") is not None else "-")
        flag = "REVIEW (low score)" if sc is not None and sc < threshold else ""
        lines.append(f"| {b['id'] + 1} | {_cell(b.get('visual_query'))} | "
                     f"{'-' if sc is None else f'{sc:.2f}'} | {trim} | {flag} |")
    return lines


def _join_cell(b: dict[str, Any], joins: dict[int, tuple[str, float]]) -> str:
    name, dur = joins.get(int(b["id"]), ("-", 0.0))
    return f"{name} {dur:.2f}s" if name not in ("none", "-") else name


def _generate_timeline_summary(plan: dict[str, Any], transcript: dict[str, Any],
                               resolver: AssetResolver, warnings: list[str], duration: float,
                               episode_id: str, output_path: Path,
                               joins: dict[int, tuple[str, float]] | None = None,
                               outro: float = 0.0, intro: float = 0.0) -> None:
    beats = plan.get("beats", [])
    joins = joins or {}
    outro_note = (f"_Includes a {outro:.2f}s outro (voiceover ends at {duration - outro:.2f}s, "
                  "then hold/outro card + fade to black)._"
                  + (f" _Boot intro card: {intro:.2f}s; the voiceover starts at {intro:.2f}s._"
                     if intro else ""))
    lines = [
        "# Timeline Summary", "",
        f"**Episode:** {plan.get('episode_id') or episode_id}", "",
        (f"**Total Duration (rendered):** {duration:.2f}s  |  **Plan:** "
         f"{plan.get('total_duration', 0):.2f}s  |  **Beats:** {len(beats)}"), "",
        *([outro_note, ""] if outro else []),
        "| # | Start | End | Dur | Template | Transition | Source | Style | Asset | Callout | Reason |",
        "|---|-------|-----|-----|----------|------------|--------|-------|-------|---------|--------|",
    ]
    for b in beats:
        if b.get("primary_source") == "stock":
            source = _stock_label(b, resolver)
            asset = b.get("stock_asset_id") or "-"
        else:
            source = "local"
            asset = b.get("source_id") or "-"
        if b.get("overlay_asset"):
            asset = f"{asset} + {b['overlay_asset']}"
        reason = b.get("beat_reason") or b.get("director_notes")
        lines.append(
            f"| {b['id'] + 1} | {b['start_time']:.2f} | {b['end_time']:.2f} | "
            f"{b['duration']:.2f} | {b['visual_template']} | {_join_cell(b, joins)} | "
            f"{_cell(source)} | "
            f"{b['beat_style']} | {_cell(asset)} | {_cell(b.get('callout_text'))} | "
            f"{_cell(reason)} |")

    sem = [b for b in beats if b.get("visual_query") or b.get("match_score") is not None]
    if sem:
        lines += _semantic_section(sem, output_path.parent.parent / "work")

    stock_rows = [resolver.stock[b["stock_asset_id"]] for b in beats
                  if b.get("stock_asset_id") in resolver.stock
                  and b["stock_asset_id"] not in resolver.refused
                  and resolver.stock[b["stock_asset_id"]]["license_type"] != "generated-fallback"]
    if stock_rows:
        lines += ["", "## Stock media licenses", ""]
        for a in stock_rows:
            flag = " **(free-license, non-CC0)**" if a["license_type"] in PLATFORM_LICENSES \
                else ""
            lines.append(f"- `{a['id']}` ({a.get('media_type', 'image')}, {a['source_platform']}): "
                         f"{a['license_type']}{flag}; author: {a.get('author') or 'n/a'}; "
                         f"{a['source_url']}")

    lines += ["", "## Warnings", ""]
    extra = []
    for b in beats:
        if b["duration"] < 1.2:
            extra.append(f"beat {b['id']} is very short ({b['duration']:.2f}s)")
        if b["duration"] > 7.0:
            extra.append(f"beat {b['id']} is long ({b['duration']:.2f}s)")
        if b.get("primary_source") == "local" and b.get("source_id") \
                and resolver.local_asset(b["source_id"]) is None:
            extra.append(f"beat {b['id']}: local asset '{b['source_id']}' missing")
        if b.get("license_type") == "generated-fallback":
            extra.append(f"beat {b['id']}: no license-free stock media found, using a "
                         "generated card")
    all_warnings = list(dict.fromkeys(warnings + extra))
    lines += [f"- {w}" for w in all_warnings] if all_warnings else ["None."]

    lines += [
        "", "## Editing this timeline", "",
        ("Edit `work/scene_plan.json` (e.g. `visual_template`, `beat_style`, `callout_text`, "
         "`source_id`, `overlay_asset`, `transition` / `transition_duration`, start/end times) and run "
         f"`python -m src.cli render {episode_id}`. Stage 5 re-runs automatically when the plan "
         "changed; Stage 6 always re-renders."), "",
        "## Beat Details", "",
    ]
    for b in beats:
        lines.append(f"### Beat {b['id'] + 1} ({b['start_time']:.2f}s - {b['end_time']:.2f}s)")
        lines.append("")
        lines.append(f"- **Template:** {b['visual_template']}")
        lines.append(f"- **Style:** {b['beat_style']}")
        if b.get("beat_reason"):
            lines.append(f"- **Cut reason:** {b['beat_reason']}")
        if b.get("director_notes"):
            lines.append(f"- **Director notes:** {b['director_notes']}")
        if b.get("callout_text"):
            lines.append(f"- **Callout:** {b['callout_text']}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
