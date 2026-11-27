"""Terminal dashboard ("AGENT PIPELINE" panel) for screen recording and screenshots.

Pure ``rich`` + PIL (no new dependencies). Three ways to use it, via ``cli dashboard``:

* live/follow: poll the episode files and redraw (``--follow`` while ``run``/``render`` works);
* ``--replay``: animate a finished run (run_log + scene_plan + final_cut.mp4 frames);
* ``--snapshot out.svg|png|txt``: render ONE static frame, no terminal needed.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.control import ControlType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.styled import Styled
from rich.table import Table
from rich.terminal_theme import TerminalTheme
from rich.text import Text

RGB = tuple[int, int, int]

# --------------------------------------------------------------------------- themes


@dataclass(frozen=True)
class Theme:
    name: str
    bg: RGB
    fg: RGB
    dim: RGB
    accents: tuple[RGB, ...]
    ok: RGB
    run: RGB
    err: RGB

    def c(self, rgb: RGB) -> str:
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    @property
    def base(self) -> str:
        return f"{self.c(self.fg)} on {self.c(self.bg)}"


THEMES = {
    "slate": Theme(
        "slate", (43, 47, 58), (201, 209, 224), (122, 131, 150),
        ((232, 161, 90), (111, 168, 255), (242, 143, 189)),
        (143, 209, 143), (255, 213, 107), (255, 107, 107)),
    "psycho": Theme(
        "psycho", (0, 3, 8), (200, 255, 215), (0, 130, 40),
        ((0, 255, 65), (0, 255, 249), (255, 0, 255)),
        (0, 255, 65), (0, 255, 249), (255, 0, 255)),
}


def terminal_theme(t: Theme) -> TerminalTheme:
    pal = [t.bg, t.err, t.ok, t.run, t.accents[1], t.accents[2], t.accents[1], t.fg]
    return TerminalTheme(t.bg, t.fg, pal, pal)


# --------------------------------------------------------------------------- data

STAGE_DEFS: list[dict[str, str]] = [
    {"name": "Voice chain", "model": "ffmpeg", "desc": "denoise, EQ, loudnorm"},
    {"name": "Transcribe", "model": "whisper small", "desc": "words + pause cut points"},
    {"name": "Asset index", "model": "SigLIP2", "desc": "ffprobe + embeddings"},
    {"name": "Planner", "model": "qwen2.5:14b", "desc": "Ollama, blocks of 5, JSON"},
    {"name": "Stock + audit", "model": "CC0 / PD", "desc": "free stock, license audit"},
    {"name": "Cutouts", "model": "rembg birefnet", "desc": "alpha + quality gate"},
    {"name": "Music", "model": "CC0 / synth", "desc": "track, mood, sidechain duck"},
    {"name": "Render", "model": "FFmpeg NVENC", "desc": "templates, xfade, loudnorm"},
]
LOG_STAGE = {"audio": 0, "transcribe": 1, "index": 2, "semantic": 2, "prefetch": 4, "plan": 3,
             "fetch": 4, "cutouts": 5, "music": 6, "render": 7}
NB = 5  # planner block size


def _rj(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass
class Data:
    ep: Path
    ep_id: str
    log: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    assets: dict[str, Any] = field(default_factory=dict)
    cutouts: dict[str, Any] = field(default_factory=dict)
    qa: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    music: dict[str, Any] = field(default_factory=dict)
    audit: Any = None
    transcript: dict[str, Any] = field(default_factory=dict)
    sem: dict[str, Any] = field(default_factory=dict)
    planner_model: str = ""
    mtimes: dict[str, float] = field(default_factory=dict)

    @property
    def beats(self) -> list[dict[str, Any]]:
        return self.plan.get("beats") or []

    @property
    def video(self) -> Path:
        return self.ep / "out" / "final_cut.mp4"


def load_data(ep: Path, ep_id: str, cfg: dict[str, Any] | None = None) -> Data:
    w = ep / "work"
    d = Data(ep, ep_id)
    try:
        d.log = [json.loads(x) for x in (w / "run_log.jsonl").read_text("utf-8").splitlines() if x.strip()]
    except (OSError, ValueError):
        d.log = []
    d.plan = _rj(w / "scene_plan.json") or {}
    d.assets = _rj(w / "local_assets.json") or {}
    d.cutouts = _rj(w / "cutouts_manifest.json") or {}
    d.qa = _rj(ep / "out" / "qa_report.json") or _rj(w / "qa_report.json") or {}
    d.audio = _rj(w / "audio_report.json") or {}
    d.music = _rj(w / "music.json") or {}
    d.audit = _rj(w / "licenses_audit.json")
    if d.audit is None:
        d.audit = _rj(ep / "out" / "licenses_audit.json")
    d.transcript = _rj(w / "transcript.json") or {}
    d.sem = _rj(w / "semantic_index.json") or {}
    if cfg:
        llm = cfg.get("llm") or cfg.get("planner") or {}
        d.planner_model = str(llm.get("model") or "").replace("-instruct", "")
    names = {"audio": w / "voiceover_clean.wav", "transcribe": w / "transcript.json",
             "index": w / "local_assets.json", "plan": w / "scene_plan.json",
             "cutouts": w / "cutouts_manifest.json", "render": ep / "out" / "final_cut.mp4",
             "log": w / "run_log.jsonl"}
    for k, p in names.items():
        try:
            d.mtimes[k] = p.stat().st_mtime
        except OSError:
            pass
    return d


@dataclass
class StageView:
    name: str
    model: str
    desc: str
    status: str = "idle"   # idle | running | done | error
    seconds: float | None = None
    counters: list[tuple[str, int | str]] = field(default_factory=list)
    extra: str = ""
    loop: tuple[int, int] | None = None   # (done, total) blocks
    link: str = ""


def _scale(v: int | str, p: float) -> int | str:
    return max(0, round(v * p)) if isinstance(v, int) else v


def stage_views(d: Data) -> list[StageView]:
    """Final (as-complete-as-the-files-show) view of each stage."""
    views = [StageView(**s) for s in STAGE_DEFS]
    secs: dict[int, float] = {}
    errs: dict[int, str] = {}
    for e in d.log:
        i = LOG_STAGE.get(e.get("stage", ""))
        if i is None:
            continue
        if e.get("status") == "ok":
            secs[i] = float(e.get("seconds") or 0)
            errs.pop(i, None)
        else:
            errs[i] = str(e.get("error", "error"))
    sem_s = [float(e.get("seconds") or 0) for e in d.log if e.get("stage") == "semantic" and e.get("status") == "ok"]
    if 2 in secs and sem_s:
        secs[2] += sem_s[-1]
    beats = d.beats
    n = len(beats)
    w = d.ep / "work"
    v0 = views[0]
    if (w / "voiceover_clean.wav").exists() or d.audio:
        v0.status = "done"
        rep = d.audio.get("after") or d.audio
        lufs = rep.get("integrated_lufs") if isinstance(rep, dict) else None
        v0.counters = [("lufs", f"{lufs:.1f}")] if isinstance(lufs, (int, float)) else []
    elif (d.ep / "raw" / "voiceover.wav").exists():
        v0.status = "done"
        v0.extra = "raw voiceover (chain off)"
    v1 = views[1]
    if d.transcript:
        v1.status = "done"
        segs = d.transcript.get("segments") or []
        words = sum(len(s.get("words") or []) for s in segs)
        v1.counters = [("words", words), ("seg", len(segs)),
                       ("dur", f"{float(d.transcript.get('duration') or 0):.0f}s")]
    v2 = views[2]
    assets = d.assets.get("assets") or []
    if assets:
        v2.status = "done"
        kinds = Counter(a.get("asset_type", "?") for a in assets)
        v2.counters = [("assets", len(assets)), ("clips", kinds.get("clip", 0)), ("img", kinds.get("image", 0))]
        if d.sem:
            v2.extra = f"{len(d.sem.get('assets') or [])} embedded, dim {d.sem.get('dim', '?')}"
        v2.link = "delegates > plan"
    v3 = views[3]
    if d.planner_model:
        v3.model = d.planner_model
    if n:
        v3.status = "done"
        scores = [b.get("match_score") for b in beats if isinstance(b.get("match_score"), (int, float))]
        fb = sum(1 for b in beats if "fallback" in str(b.get("beat_reason", "")).lower())
        v3.counters = [("beats", n), ("blocks", math.ceil(n / NB)), ("fallback", fb)]
        if scores:
            v3.extra = f"mean match {sum(scores) / len(scores):.2f}"
        v3.loop = (math.ceil(n / NB), math.ceil(n / NB))
        v3.link = "delegates > stock"
    v4 = views[4]
    audit_items = d.audit.get("assets") if isinstance(d.audit, dict) else d.audit
    if d.audit is not None or n:
        stock = sum(1 for b in beats if (b.get("primary_source") or "local") != "local")
        if d.audit is not None:
            v4.status = "done"
        v4.counters = [("stock", stock), ("audited", len(audit_items or []))]
        v4.link = "audit"
    v5 = views[5]
    if d.cutouts:
        v5.status = "done"
        decisions = d.cutouts.get("cutout_decisions") or []
        gated = sum(1 for x in decisions if x.get("mode") != "cutout")
        v5.counters = [("cutouts", int(d.cutouts.get("cutout_count") or 0)),
                       ("badges", int(d.cutouts.get("badge_count") or 0)), ("gate", gated)]
        v5.loop = (len(decisions), max(len(decisions), 1))
        warn = len(d.cutouts.get("warnings") or [])
        v5.extra = f"{warn} warnings" if warn else ""
    v6 = views[6]
    if d.music:
        v6.status = "done"
        v6.model = str(d.music.get("source") or "-")[:12]
        lu = d.music.get("level_lufs")
        v6.counters = [("mood", d.music.get("mood") or "-")]
        if isinstance(lu, (int, float)):
            v6.counters.append(("lufs", f"{lu:.1f}"))
        v6.extra = str(d.music.get("license") or "")[:40]
    elif (d.ep / "work" / "music_audit.json").exists():
        v6.status = "done"
    v7 = views[7]
    if d.video.exists():
        v7.status = "done"
        dur = d.qa.get("duration")
        lu = (d.qa.get("loudness") or {}).get("integrated_lufs")
        dur = dur or d.plan.get("total_duration")
        v7.counters = [("dur", f"{dur:.0f}s" if dur else "-"), ("lufs", f"{lu:.1f}" if lu else "-")]
        if lu is None:
            v7.counters.pop()
        if d.qa:
            v7.extra = "QA pass" if d.qa.get("passed") else f"QA {len(d.qa.get('issues') or [])} issues"
        cache = d.ep / "work" / "render_cache"
        if cache.is_dir():
            v7.counters.append(("cache", sum(1 for _ in cache.iterdir())))
    for i, v in enumerate(views):
        if i in secs:
            v.seconds = secs[i]
        if i in errs and v.status != "done":
            v.status = "error"
            v.extra = errs[i][:40]
    return views


def mark_running(views: list[StageView], d: Data, now: float | None = None) -> None:
    """--follow heuristic: first not-done stage is running while the work dir is being written."""
    now = now or time.time()
    if max(d.mtimes.values(), default=0) < now - 120:
        return
    for v in views:
        if v.status == "idle":
            v.status = "running"
            return


# --------------------------------------------------------------------------- GPU

_gpu_cache: tuple[float, dict[str, Any] | None] = (0.0, None)


def gpu_info() -> dict[str, Any] | None:
    global _gpu_cache
    if time.time() - _gpu_cache[0] < 1.0:
        return _gpu_cache[1]
    info = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=True,
            ).stdout.splitlines()[0].split(",")
            info = {"name": out[0].strip().replace("NVIDIA GeForce ", ""), "used": float(out[1]),
                    "total": float(out[2]), "util": float(out[3]), "temp": float(out[4])}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            info = None
    _gpu_cache = (time.time(), info)
    return info


# --------------------------------------------------------------------------- preview

PREVIEW_MODES = ("auto", "kitty", "iterm", "sixel", "blocks", "quadrants")
_GRAPHIC = ("kitty", "iterm", "sixel")
_preview_mode = "quadrants"  # resolved mode (never "auto")
_QUAD_CHARS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"  # bit0=TL bit1=TR bit2=BL bit3=BR


def detect_preview_mode(pref: str = "auto", env: dict[str, str] | None = None, tty: bool | None = None) -> str:
    """Resolve ``--preview``: a graphics protocol when the terminal is known to support it, else quadrants."""
    env = os.environ if env is None else env
    pref = (pref or "auto").lower()
    if pref in ("blocks", "quadrants"):
        return pref
    if tty is None:
        tty = sys.stdout.isatty()
    if pref == "sixel":
        return "sixel" if shutil.which("chafa") else "quadrants"
    if pref in ("kitty", "iterm"):
        return pref
    if not tty or env.get("TMUX") or env.get("STY"):
        return "quadrants"
    term, prog = env.get("TERM", ""), env.get("TERM_PROGRAM", "")
    if env.get("KITTY_WINDOW_ID") or term in ("xterm-kitty", "xterm-ghostty") or prog == "ghostty":
        return "kitty"
    if prog in ("iTerm.app", "WezTerm") or env.get("WEZTERM_EXECUTABLE"):
        return "iterm"
    if shutil.which("chafa") and ("sixel" in term or term in ("foot", "mlterm", "yaft-256color")):
        return "sixel"
    return "quadrants"


def set_preview_mode(pref: str = "auto") -> str:
    global _preview_mode
    _preview_mode = detect_preview_mode(pref)
    env = os.environ
    if pref == "auto" and _preview_mode == "quadrants" and (
            env.get("KITTY_WINDOW_ID") or env.get("TERM") == "xterm-kitty"):
        why = "not a tty" if not sys.stdout.isatty() else "inside tmux/screen"
        print(f"warning: kitty terminal detected but {why}; preview falls back to quadrants "
              "(use --preview kitty to force)", file=sys.stderr)
    if _preview_mode == "kitty":
        import atexit

        atexit.register(lambda: (sys.stdout.write(graphics_cleanup()), sys.stdout.flush()))
    return _preview_mode


def _fit_cells(iw: int, ih: int, cols: int, rows: int) -> tuple[int, int]:
    """Largest (cols, rows) box with the image's aspect, cells being ~1:2 (physical box = cols x 2*rows)."""
    s = min(cols / iw, 2 * rows / ih)
    return max(1, min(cols, round(iw * s))), max(1, min(rows, round(ih * s / 2)))


def _load_rgb(path: str):
    from PIL import Image, ImageEnhance, ImageFilter

    im = Image.open(path).convert("RGB")
    im = ImageEnhance.Contrast(im).enhance(1.08)
    return ImageEnhance.Brightness(im).enhance(1.03), ImageFilter


def _quadrant_cells(px, cols: int, rows: int):
    """px: (2*rows, 2*cols, 3) float array -> (chars index, fg, bg) per cell, minimal 2-colour error."""
    import numpy as np

    c = px.reshape(rows, 2, cols, 2, 3).transpose(0, 2, 1, 3, 4).reshape(rows, cols, 4, 3)
    mean = c.mean(axis=2)
    best_err = ((c - mean[:, :, None]) ** 2).sum(axis=(2, 3)) - 1e-6  # uniform cell (full block) wins ties
    best_m = np.full((rows, cols), 15, dtype=np.int64)
    fg_out, bg_out = mean.copy(), mean.copy()
    for m in range(1, 15):
        sel = np.array([(m >> k) & 1 for k in range(4)], dtype=bool)
        fg = c[:, :, sel].mean(axis=2)
        bg = c[:, :, ~sel].mean(axis=2)
        err = ((c[:, :, sel] - fg[:, :, None]) ** 2).sum(axis=(2, 3)) + \
              ((c[:, :, ~sel] - bg[:, :, None]) ** 2).sum(axis=(2, 3))
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_m = np.where(better, m, best_m)
        fg_out = np.where(better[:, :, None], fg, fg_out)
        bg_out = np.where(better[:, :, None], bg, bg_out)
    return best_m, fg_out, bg_out


@lru_cache(maxsize=256)
def _cell_lines(path: str, mtime: float, cols: int, rows: int, bg: RGB, mode: str = "blocks") -> tuple[Text, ...]:
    from PIL import Image

    try:
        im, _ = _load_rgb(path)
    except (OSError, ValueError):
        return tuple(Text(" " * cols) for _ in range(rows))
    cu, ru = _fit_cells(im.width, im.height, cols, rows)
    quad = mode == "quadrants"
    sx = 2 if quad else 1
    im = im.resize((cu * sx, ru * 2), Image.LANCZOS)
    canvas = Image.new("RGB", (cols * sx, rows * 2), bg)
    canvas.paste(im, (((cols - cu) // 2) * sx, ((rows - ru) // 2) * 2))
    lines = []
    if quad:
        import numpy as np

        m, fg, bgc = _quadrant_cells(np.asarray(canvas, dtype=np.float64), cols, rows)
        fg, bgc = fg.round().astype(int), bgc.round().astype(int)
        for y in range(rows):
            t = Text(no_wrap=True, overflow="crop")
            for x in range(cols):
                f, b = fg[y, x], bgc[y, x]
                t.append(_QUAD_CHARS[int(m[y, x])], style=f"rgb({f[0]},{f[1]},{f[2]}) on rgb({b[0]},{b[1]},{b[2]})")
            lines.append(t)
        return tuple(lines)
    px = canvas.load()
    for y in range(rows):
        t = Text(no_wrap=True, overflow="crop")
        for x in range(cols):
            top, bot = px[x, 2 * y], px[x, 2 * y + 1]
            t.append("▀", style=f"rgb({top[0]},{top[1]},{top[2]}) on rgb({bot[0]},{bot[1]},{bot[2]})")
        lines.append(t)
    return tuple(lines)


def _cell_px() -> tuple[int, int]:
    try:
        import fcntl
        import struct
        import termios

        r, c, xp, yp = struct.unpack("HHHH", fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8))
        if xp and yp and c and r:
            return max(xp // c, 4), max(yp // r, 8)
    except (OSError, ImportError, ValueError):
        pass
    return 9, 18


@lru_cache(maxsize=32)
def _graphic_payload(path: str, mtime: float, cu: int, ru: int, mode: str, cw: int, ch: int) -> bytes:
    """Encoded image for a graphics protocol: base64 PNG/JPEG (kitty/iterm) or chafa sixel bytes."""
    import base64

    from PIL import Image

    if mode == "sixel":
        out = subprocess.run(["chafa", "-f", "sixel", "-s", f"{cu}x{ru}", "--animate=off", path],
                             capture_output=True, timeout=5, check=False).stdout
        return out.rstrip(b"\r\n")
    im, _ = _load_rgb(path)
    im = im.resize((cu * cw, ru * ch), Image.LANCZOS)
    buf = io.BytesIO()
    if mode == "kitty":
        im.save(buf, "PNG", compress_level=1)
    else:
        im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue())


_kitty_sent: dict[str, float] = {}
_KITTY_ID = 7341


def _graphic_escape(path: Path, cu: int, ru: int, mode: str) -> str:
    cw, ch = _cell_px()
    data = _graphic_payload(str(path), path.stat().st_mtime, cu, ru, mode, cw, ch)
    if mode == "sixel":
        return data.decode("latin-1")
    if mode == "iterm":
        return (f"\x1b]1337;File=inline=1;width={cu};height={ru};preserveAspectRatio=0;size={len(data) * 3 // 4}:"
                f"{data.decode()}\x07")
    key = f"{path}:{path.stat().st_mtime}:{cu}x{ru}:{shutil.get_terminal_size()}"
    now = time.time()
    place = f"\x1b_Ga=p,i={_KITTY_ID},p=1,c={cu},r={ru},C=1,q=2\x1b\\"
    if now - _kitty_sent.get(key, 0) < 4.0:
        return ""  # kitty keeps the placement while rich rewrites the cells: draw only on change (resync every 4 s)
    _kitty_sent.clear()
    _kitty_sent[key] = now
    s = data.decode()
    parts = [s[i:i + 4096] for i in range(0, len(s), 4096)] or [""]
    out = []
    for i, chunk in enumerate(parts):
        more = 1 if i < len(parts) - 1 else 0
        head = f"a=t,i={_KITTY_ID},f=100,q=2," if i == 0 else ""
        out.append(f"\x1b_G{head}m={more};{chunk}\x1b\\")
    return "".join(out) + place


@contextlib.contextmanager
def cell_preview():
    """Render with cell characters only (cast frames, snapshots): graphics escapes never go there."""
    global _preview_mode
    saved = _preview_mode
    if saved in _GRAPHIC:
        _preview_mode = "quadrants"
    try:
        yield
    finally:
        _preview_mode = saved


def graphics_cleanup() -> str:
    return f"\x1b_Ga=d,d=I,i={_KITTY_ID},q=2\x1b\\" if _preview_mode == "kitty" else ""


class HalfBlock:
    """Image drawn with block characters (mode: blocks=▀ 1x2, quadrants=▖▗▘▝▚.. 2x2 px per cell)."""

    def __init__(self, path: Path | None, cols: int, rows: int, theme: Theme, label: str = "no preview",
                 big: bool = False) -> None:
        self.path, self.cols, self.rows, self.theme, self.label = path, max(cols, 1), max(rows, 1), theme, label
        self.big = big

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self.path is None or not Path(self.path).exists():
            yield Text(self.label.center(self.cols)[: self.cols], style=self.theme.c(self.theme.dim))
            for _ in range(self.rows - 1):
                yield Text("")
            return
        p = Path(self.path)
        mode = _preview_mode
        if self.big and mode in _GRAPHIC and console.is_terminal and not console.is_dumb_terminal:
            try:
                yield from self._graphic(console, p, mode)
                return
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        cmode = "blocks" if mode == "blocks" else "quadrants"
        yield from _cell_lines(str(p), p.stat().st_mtime, self.cols, self.rows, self.theme.bg, cmode)

    def _graphic(self, console: Console, p: Path, mode: str) -> RenderResult:
        from PIL import Image

        with Image.open(p) as im:
            iw, ih = im.size
        cu, ru = _fit_cells(iw, ih, self.cols, self.rows)
        pad_l, pad_t = (self.cols - cu) // 2, (self.rows - ru) // 2
        esc = _graphic_escape(p, cu, ru, mode)
        sty = Style.parse(self.theme.base)
        ctl = [(ControlType.BELL,)]  # marker only: makes rich pass the raw escape through, width 0
        for y in range(self.rows):
            if y == pad_t:
                yield Segment(" " * pad_l, sty)
                yield Segment(esc, None, ctl)
                yield Segment(" " * (self.cols - pad_l), sty)
            else:
                yield Segment(" " * self.cols, sty)
            yield Segment.line()


def ensure_frames(d: Data, video_fps: int = 0) -> dict[str, Any]:
    """Extract preview frames from final_cut.mp4: one per beat, plus an optional ~4 fps sequence."""
    out = d.ep / "work" / "dashboard_frames"
    res: dict[str, Any] = {"beat": {}, "seq": [], "fps": video_fps}
    if not d.video.exists() or not shutil.which("ffmpeg"):
        return res
    out.mkdir(parents=True, exist_ok=True)
    stamp = d.video.stat().st_mtime
    marker = out / ".stamp"
    try:
        fresh = marker.exists() and abs(float(marker.read_text() or 0) - stamp) < 1e-3
    except ValueError:
        fresh = False
    if not fresh:
        for f in out.glob("*.jpg"):
            f.unlink()
        marker.write_text(str(stamp))
    for i, b in enumerate(d.beats):
        p = out / f"beat_{i:03d}.jpg"
        if not p.exists():
            mid = (float(b.get("start_time", 0)) + float(b.get("end_time", 0))) / 2
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{mid:.2f}", "-i", str(d.video),
                            "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(p)],
                           capture_output=True, timeout=60, check=False)
        if p.exists():
            res["beat"][i] = p
    if video_fps:
        seq = sorted(out.glob("f_*.jpg"))
        if not seq:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d.video), "-vf",
                            f"fps={video_fps},scale=320:-2", "-q:v", "5", str(out / "f_%05d.jpg")],
                           capture_output=True, timeout=600, check=False)
            seq = sorted(out.glob("f_*.jpg"))
        res["seq"] = seq
    return res


# --------------------------------------------------------------------------- rendering


@dataclass
class View:
    """Everything the renderer needs for one frame."""
    ep_id: str
    stages: list[StageView]
    beats: list[dict[str, Any]]
    shown_beats: int            # how many beats are 'produced' so far
    cur: int                    # current beat index
    log: list[str]
    preview: Path | None
    thumbs: list[Path | None]
    mode: str = "live"
    elapsed: float = 0.0
    video_t: float | None = None


def _bar(frac: float, width: int, theme: Theme, color: RGB) -> Text:
    frac = min(max(frac, 0.0), 1.0)
    full = round(frac * width)
    t = Text()
    t.append("█" * full, style=theme.c(color))
    t.append("░" * (width - full), style=theme.c(theme.dim))
    return t


def _fmt_s(s: float | None) -> str:
    if s is None:
        return ""
    return f"{s:.1f}s" if s < 100 else f"{int(s) // 60}m{int(s) % 60:02d}s"


def stage_box(i: int, v: StageView, height: int, theme: Theme, width: int = 60) -> Panel:
    accent = theme.accents[i % len(theme.accents)]
    glyph, gcol = {"idle": ("○", theme.dim), "running": ("●", theme.run),
                   "done": ("✓", theme.ok), "error": ("✗", theme.err)}[v.status]
    title = Text()
    title.append(f" {i} ", style=f"bold {theme.c(theme.bg)} on {theme.c(accent)}")
    title.append(f" {v.name} ", style=f"bold {theme.c(accent)}")
    head = Table.grid(expand=True, padding=(0, 1))
    head.add_column(no_wrap=True)
    head.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    head.add_column(justify="right", no_wrap=True)
    state = Text()
    state.append(glyph + " ", style=f"bold {theme.c(gcol)}")
    state.append(f"{v.status:<7}", style=theme.c(gcol))
    cnt = Text(no_wrap=True, overflow="ellipsis")
    avail = width - 4 - 9 - len(v.model) - 4
    items: list[tuple[str, str]] = []
    if v.seconds is not None:
        items.append(("", _fmt_s(v.seconds)))
    items += [(k, str(val)) for k, val in v.counters]
    used = 0
    for k, val in items:
        need = len(k) + (1 if k else 0) + len(val) + 2
        if used + need > avail:
            break
        used += need
        if k:
            cnt.append(f"{k} ", style=theme.c(theme.dim))
        cnt.append(f"{val}  ", style=f"bold {theme.c(theme.fg)}" if k else theme.c(theme.fg))
    head.add_row(state, cnt, Text(v.model, style=f"bold {theme.c(accent)}"))
    third = Text(no_wrap=True, overflow="ellipsis", justify="right")
    if v.link:
        third.append(f"{v.link}  ", style=theme.c(accent))
    if v.loop:
        d, t = v.loop
        cells = min(t, 5)
        filled = round(cells * d / t) if t else 0
        third.append("loop ", style=theme.c(theme.dim))
        third.append("■" * filled + "□" * (cells - filled), style=theme.c(accent))
    row2 = Table.grid(expand=True, padding=(0, 1))
    row2.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    row2.add_column(justify="right", no_wrap=True)
    row2.add_row(Text(v.desc, style=theme.c(theme.dim), no_wrap=True, overflow="ellipsis"), third)
    lines: list[Any] = [head, row2]
    if v.extra:
        lines.append(Text(v.extra, style=theme.c(theme.fg), no_wrap=True, overflow="ellipsis"))
    return Panel(Group(*lines[: max(height - 2, 1)]), title=title, title_align="left", box=box.ROUNDED,
                 border_style=theme.c(accent if v.status != "idle" else theme.dim), padding=(0, 1),
                 height=height, style=theme.base)


def _title(text: str, color: RGB, theme: Theme) -> Text:
    return Text(f" {text} ", style=f"bold {theme.c(color)}")


def _preview_panel(v: View, theme: Theme, w: int, h: int) -> Panel:
    accent = theme.accents[2]
    inner_w, inner_h = w - 4, h - 2
    strip_rows = 3
    img_rows = max(inner_h - strip_rows - 1, 4)
    img_cols = max(min(int(img_rows * 2 * 16 / 9), inner_w - 20), 8)
    b = v.beats[v.cur] if v.beats and 0 <= v.cur < len(v.beats) else {}
    info = Text(overflow="fold")
    if b:
        info.append(f"beat {v.cur + 1}/{len(v.beats)}\n", style=f"bold {theme.c(accent)}")
        info.append(f"{b.get('start_time', 0):.1f}-{b.get('end_time', 0):.1f}s\n", style=theme.c(theme.fg))
        info.append(_tpl(b.get("visual_template", "")) + "\n", style=theme.c(theme.accents[1]))
        co = b.get("callout_text")
        if co:
            info.append(f"'{co}'\n", style=f"bold {theme.c(theme.accents[0])}")
        info.append(str(b.get("visual_query") or b.get("director_notes") or "")[:80], style=theme.c(theme.dim))
    top = Table.grid(padding=(0, 1))
    top.add_column(width=img_cols)
    top.add_column(width=max(inner_w - img_cols - 1, 8))
    top.add_row(HalfBlock(v.preview, img_cols, img_rows, theme, "awaiting render", big=True), info)
    ncols = max(min(inner_w // 9, len(v.beats) or 1), 1)
    strip = Table.grid(padding=(0, 1))
    for _ in range(ncols):
        strip.add_column(width=8)
    start = min(max(v.cur - ncols // 2, 0), max(len(v.beats) - ncols, 0))
    cells, marks = [], []
    for j in range(start, min(start + ncols, len(v.beats))):
        cells.append(HalfBlock(v.thumbs[j] if j < len(v.thumbs) else None, 8, strip_rows, theme, "·"))
        marks.append(Text(f"{j + 1}".center(8), style=f"bold {theme.c(theme.bg)} on {theme.c(accent)}"
                          if j == v.cur else theme.c(theme.dim)))
    if cells:
        strip.add_row(*cells)
        strip.add_row(*marks)
    return Panel(Group(top, strip), title=_title("PREVIEW", accent, theme), title_align="left",
                 subtitle=Text(f" {v.mode} ", style=theme.c(theme.dim)), subtitle_align="right",
                 box=box.ROUNDED, border_style=theme.c(accent), padding=(0, 1), height=h, style=theme.base)


def _tpl(name: Any) -> str:
    n = str(name).replace("_emphasis", "")
    for a, b in (("fullscreen_clip", "fullscreen"), ("fullscreen_image", "fullscreen"), ("split_schematic", "split"),
                 ("cutout_callout", "cutout"), ("punch_in", "punch in")):
        if n == a:
            return b
    return n.replace("_", " ")


_cpu_prev: tuple[float, float] | None = None


def cpu_info() -> tuple[float, float] | None:
    """(cpu %, mem %) from /proc since the previous call; None if unavailable."""
    global _cpu_prev
    try:
        f = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        idle, total = f[3] + f[4], sum(f)
        mem = {ln.split(":")[0]: float(ln.split()[1]) for ln in Path("/proc/meminfo").read_text().splitlines()}
    except (OSError, ValueError, IndexError):
        return None
    prev, _cpu_prev = _cpu_prev, (idle, total)
    pct = 0.0 if prev is None or total == prev[1] else 100 * (1 - (idle - prev[0]) / (total - prev[1]))
    return pct, 100 * (1 - mem["MemAvailable"] / mem["MemTotal"])


def _gpu_panel(theme: Theme, w: int, h: int) -> Panel:
    accent = theme.accents[1]
    g = gpu_info()
    bw = max(w - 17, 4)
    rows: list[Any] = []
    if g is None:
        rows.append(Text("no nvidia-smi", style=theme.c(theme.dim)))
    else:
        rows.append(Text(g["name"], style=f"bold {theme.c(theme.fg)}", no_wrap=True, overflow="ellipsis"))
        vr = Text("vram ", style=theme.c(theme.dim), no_wrap=True)
        vr.append_text(_bar(g["used"] / g["total"], bw, theme, accent))
        vr.append(f" {g['used'] / 1024:.1f}/{g['total'] / 1024:.0f}G", style=theme.c(theme.fg))
        ut = Text("util ", style=theme.c(theme.dim), no_wrap=True)
        ut.append_text(_bar(g["util"] / 100, bw, theme, theme.accents[0]))
        ut.append(f" {g['util']:3.0f}% {g['temp']:.0f}C", style=theme.c(theme.fg))
        rows += [vr, ut]
    c = cpu_info()
    if c is not None:
        ct = Text("cpu  ", style=theme.c(theme.dim), no_wrap=True)
        ct.append_text(_bar(c[0] / 100, bw, theme, theme.accents[2]))
        ct.append(f" {c[0]:3.0f}% m{c[1]:.0f}", style=theme.c(theme.fg))
        rows.append(ct)
    return Panel(Group(*rows[: h - 2]), title=_title("GPU / VRAM", accent, theme), title_align="left",
                 box=box.ROUNDED, border_style=theme.c(accent), padding=(0, 1), height=h, style=theme.base)


def _mix_panel(v: View, theme: Theme, w: int, h: int) -> Panel:
    accent = theme.accents[0]
    beats = v.beats[: max(v.shown_beats, 0)]
    rows: list[Any] = []
    if beats:
        tpl = Counter(_tpl(b.get("visual_template", "?")) for b in beats)
        sty = Counter(str(b.get("beat_style", "normal")) for b in beats)
        total = len(beats)
        bw = max(w - 25, 4)
        n_sty = 1 if h <= 6 else 2
        for k, c in tpl.most_common(max(h - 2 - n_sty, 1)):
            t = Text(f"{k[:10]:<10} ", style=theme.c(theme.fg), no_wrap=True)
            t.append_text(_bar(c / total, bw, theme, accent))
            t.append(f" {c:>2} {100 * c // total:>3}%", style=theme.c(theme.dim))
            rows.append(t)
        for k, c in sty.most_common(n_sty):
            t = Text(f"st {k[:7]:<7} ", style=theme.c(theme.dim), no_wrap=True)
            t.append_text(_bar(c / total, bw, theme, theme.accents[2]))
            t.append(f" {c:>2} {100 * c // total:>3}%", style=theme.c(theme.dim))
            rows.append(t)
    else:
        rows.append(Text("no scene plan yet", style=theme.c(theme.dim)))
    return Panel(Group(*rows[: h - 2]), title=_title("TEMPLATE / STYLE MIX", accent, theme), title_align="left",
                 box=box.ROUNDED, border_style=theme.c(accent), padding=(0, 1), height=h, style=theme.base)


def _short(s: str) -> str:
    for p in ("clip_clip_", "clip_", "schematic_", "image_", "stock_"):
        if s.startswith(p):
            return s[len(p):]
    return s


def _beats_panel(v: View, theme: Theme, w: int, h: int) -> Panel:
    accent = theme.accents[1]
    nrows = max(h - 3, 1)
    shown = v.beats[: max(v.shown_beats, 0)]
    end = min(max(v.cur + 1, nrows), len(shown))
    idx = list(range(max(end - nrows, 0), end))
    tb = Table(box=None, expand=True, padding=(0, 1), show_edge=False, header_style=f"bold {theme.c(theme.dim)}",
               style=theme.base)
    tb.add_column("time", no_wrap=True)
    tb.add_column("template", no_wrap=True, max_width=10)
    tb.add_column("asset", no_wrap=True, overflow="ellipsis", ratio=1)
    tb.add_column("score", no_wrap=True, justify="right")
    tb.add_column("callout", no_wrap=True, overflow="ellipsis", max_width=max(w // 4, 8))
    for i in idx:
        b = shown[i]
        sc = b.get("match_score")
        st = f"bold {theme.c(accent)}" if i == v.cur else theme.c(theme.fg)
        tb.add_row(Text(f"{b.get('start_time', 0):6.1f}s", style=st),
                   Text(_tpl(b.get("visual_template", "")), style=st),
                   Text(_short(str(b.get("source_id", ""))), style=st),
                   Text(f"{sc:.2f}" if isinstance(sc, (int, float)) else "-", style=st),
                   Text(str(b.get("callout_text") or "-"), style=st))
    return Panel(tb, title=_title("LAST BEATS", accent, theme), title_align="left", box=box.ROUNDED,
                 border_style=theme.c(accent), padding=(0, 1), height=h, style=theme.base)


def _log_panel(v: View, theme: Theme, w: int, h: int) -> Panel:
    accent = theme.accents[2]
    lines = []
    for ln in v.log[-(h - 2):]:
        t = Text(no_wrap=True, overflow="ellipsis")
        ts, _, body = ln.partition(" ")
        t.append(ts + " ", style=theme.c(theme.dim))
        words = body.split()
        col = theme.err if "error" in words else theme.ok if "ok" in words else theme.fg
        t.append(body, style=theme.c(col))
        lines.append(t)
    if not lines:
        lines = [Text("no run_log.jsonl entries yet", style=theme.c(theme.dim))]
    return Panel(Group(*lines), title=_title("session log", accent, theme), title_align="left", box=box.ROUNDED,
                 border_style=theme.c(accent), padding=(0, 1), height=h, style=theme.base)


def log_lines(entries: list[dict[str, Any]]) -> list[str]:
    out = []
    for e in entries:
        ts = str(e.get("ts", ""))[11:19]
        if e.get("status") == "ok":
            tail = f"{float(e.get('seconds') or 0):6.1f}s  ok"
        else:
            tail = f"error {str(e.get('error', ''))[:70]}"
        out.append(f"{ts} {e.get('stage', '')!s:<10} {tail}")
    return out


def render(v: View, theme: Theme, width: int, height: int) -> Styled:
    """Compose the full dashboard for a width x height terminal."""
    log_h = 8 if height < 44 else 10
    main_h = height - 2 - log_h
    left_w = width // 2
    right_w = width - left_w
    hs = [main_h // 8 + (1 if k < main_h % 8 else 0) for k in range(8)]
    left = Layout()
    left.split_column(*[Layout(stage_box(i, v.stages[i], hs[i], theme, left_w), size=hs[i]) for i in range(8)])
    mix_h = 6 if main_h <= 30 else 8
    beats_h = 8 if main_h <= 40 else 10
    prev_h = max(main_h - mix_h - beats_h, 16)
    beats_h = main_h - mix_h - prev_h
    gpu_w = right_w * 2 // 5
    mixrow = Layout()
    mixrow.split_row(Layout(_gpu_panel(theme, gpu_w, mix_h), size=gpu_w),
                     Layout(_mix_panel(v, theme, right_w - gpu_w, mix_h)))
    right = Layout()
    right.split_column(Layout(_preview_panel(v, theme, right_w, prev_h), size=prev_h),
                       Layout(mixrow, size=mix_h),
                       Layout(_beats_panel(v, theme, right_w, beats_h), size=beats_h))
    main = Layout()
    main.split_row(Layout(left, size=left_w), Layout(right))
    head = Text(no_wrap=True, overflow="ellipsis")
    head.append(" AGENT PIPELINE ", style=f"bold {theme.c(theme.bg)} on {theme.c(theme.accents[0])}")
    head.append(f"  {v.ep_id}", style=f"bold {theme.c(theme.accents[1])}")
    head.append(f"   {sum(s.status == 'done' for s in v.stages)}/{len(v.stages)} stages done", style=theme.c(theme.dim))
    status = Text(no_wrap=True, overflow="ellipsis")
    run = next((s for s in v.stages if s.status == "running"), None)
    status.append(f" {v.mode.upper()} ", style=f"bold {theme.c(theme.bg)} on {theme.c(theme.accents[1])}")
    status.append(f" {run.name + ' running' if run else 'idle'}", style=theme.c(theme.run if run else theme.dim))
    status.append(f"  beats {v.shown_beats}/{len(v.beats)}", style=theme.c(theme.fg))
    if v.video_t is not None:
        status.append(f"  t={v.video_t:5.1f}s", style=theme.c(theme.fg))
    status.append(f"  elapsed {v.elapsed:5.1f}s", style=theme.c(theme.dim))
    status.append("   ctrl-c quit", style=theme.c(theme.dim))
    root = Layout()
    root.split_column(Layout(head, size=1), Layout(main), Layout(_log_panel(v, theme, width, log_h), size=log_h),
                      Layout(status, size=1))
    return Styled(root, theme.base)


# --------------------------------------------------------------------------- views


def static_view(d: Data, frames: dict[str, Any], cur: int | None = None, running: bool = False) -> View:
    stages = stage_views(d)
    if running:
        mark_running(stages, d)
    n = len(d.beats)
    cur = min(cur if cur is not None else n // 3, max(n - 1, 0))
    bf = frames.get("beat", {})
    return View(d.ep_id, stages, d.beats, n, cur, log_lines(d.log)[-10:], bf.get(cur),
                [bf.get(i) for i in range(n)], mode="follow" if running else "snapshot")


def replay_view(d: Data, frames: dict[str, Any], t: float, speed: float, use_seq: bool) -> tuple[View, bool]:
    """Frame of a replay at real time ``t`` (seconds since start). Returns (view, finished)."""
    final = stage_views(d)
    durs = [min(max(s.seconds or 1.5, 2.0), 9.0) / speed if s.status != "idle" else 0.0 for s in final]
    starts, acc = [], 0.0
    for dur in durs:
        starts.append(acc)
        acc += dur
    n = len(d.beats)
    views: list[StageView] = []
    log: list[str] = []

    def clock(sec: float) -> str:
        return time.strftime("%H:%M:%S", time.gmtime(max(sec, 0) * speed))

    for i, f in enumerate(final):
        sv = StageView(f.name, f.model, f.desc, link=f.link)
        p = (t - starts[i]) / durs[i] if durs[i] else 0.0
        if p <= 0:
            views.append(sv)
            continue
        pp = min(p, 1.0)
        sv.status = "done" if p >= 1 else "running"
        sv.seconds = (f.seconds or 0) * pp if f.seconds else None
        sv.counters = [(k, _scale(val, pp)) for k, val in f.counters]
        if f.loop:
            sv.loop = (round(f.loop[1] * pp), f.loop[1])
        if p >= 1:
            sv.extra, sv.seconds = f.extra, f.seconds
        views.append(sv)
        log.append(f"{clock(starts[i])} {f.name.lower():<10} start  {f.model}")
        if p >= 1:
            log.append(f"{clock(starts[i] + durs[i])} {f.name.lower():<10} {(f.seconds or 0):6.1f}s  ok")
    rp = 0.0 if durs[7] == 0 else min(max((t - starts[7]) / durs[7], 0.0), 1.0)
    pp3 = 0.0 if durs[3] == 0 else min(max((t - starts[3]) / durs[3], 0.0), 1.0)
    shown = n if rp > 0 or pp3 >= 1 else round(n * pp3)
    total_dur = float(d.plan.get("total_duration") or (d.beats[-1].get("end_time", 0) if n else 0))
    vt = rp * total_dur
    cur = 0
    for i, b in enumerate(d.beats):
        if float(b.get("start_time", 0)) <= vt:
            cur = i
    if rp == 0 and n:
        cur = max(min(shown - 1, n - 1), 0)
    preview = None
    seq = frames.get("seq") or []
    bf = frames.get("beat", {})
    if rp > 0:
        if use_seq and seq:
            preview = seq[min(int(vt * (frames.get("fps") or 4)), len(seq) - 1)]
        else:
            preview = bf.get(cur)
    thumbs = [bf.get(i) if rp > 0 and i <= cur else None for i in range(n)]
    v = View(d.ep_id, views, d.beats, shown, cur, log[-10:], preview, thumbs, mode="replay", elapsed=t,
             video_t=vt if rp > 0 else None)
    return v, t >= acc + 2.0


# --------------------------------------------------------------------------- output


def build_console(width: int, height: int) -> Console:
    return Console(width=width, height=height, record=True, file=io.StringIO(), force_terminal=True,
                   color_system="truecolor", legacy_windows=False)


def export_snapshot(v: View, theme: Theme, out: Path, width: int = 120, height: int = 40) -> Path:
    with cell_preview():
        return _export_snapshot(v, theme, out, width, height)


def _export_snapshot(v: View, theme: Theme, out: Path, width: int, height: int) -> Path:
    out = Path(out)
    con = build_console(width, height)
    root = render(v, theme, width, height)
    out.parent.mkdir(parents=True, exist_ok=True)
    suf = out.suffix.lower()
    if suf == ".png":
        _save_png(con, root, theme, out)
        return out
    con.print(root)
    if suf == ".svg":
        con.save_svg(str(out), theme=terminal_theme(theme), title=f"AGENT PIPELINE - {v.ep_id}")
    else:
        out.write_text(con.export_text(styles=False), encoding="utf-8")
    return out


def _save_png(con: Console, root: Any, theme: Theme, out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    size = 18
    try:
        fr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size)
    except OSError:
        fr = fb = ImageFont.load_default()
    cw = round(fr.getlength("M"))
    ch = int(size * 1.25)
    lines = con.render_lines(root, con.options, pad=True, new_lines=False)
    img = Image.new("RGB", (cw * con.width, ch * len(lines)), theme.bg)
    dr = ImageDraw.Draw(img)
    for y, line in enumerate(lines):
        x = 0
        for seg in line:
            st = seg.style
            fg = st.color.get_truecolor() if st and st.color else None
            bgc = st.bgcolor.get_truecolor() if st and st.bgcolor else None
            fgc = (fg.red, fg.green, fg.blue) if fg else theme.fg
            bgt = (bgc.red, bgc.green, bgc.blue) if bgc else theme.bg
            if st and st.reverse:
                fgc, bgt = bgt, fgc
            for chh in seg.text:
                x0, y0 = x * cw, y * ch
                if bgt != theme.bg:
                    dr.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=bgt)
                if chh in _QUAD_CHARS and chh != " ":
                    m = _QUAD_CHARS.index(chh)
                    for k in range(4):
                        if m >> k & 1:
                            qx, qy = x0 + (k & 1) * cw // 2, y0 + (k >> 1) * ch // 2
                            dr.rectangle([qx, qy, qx + (cw + 1) // 2 - 1 if k & 1 else qx + cw // 2 - 1,
                                          qy + (ch + 1) // 2 - 1 if k >> 1 else qy + ch // 2 - 1], fill=fgc)
                elif chh != " ":
                    dr.text((x0, y0 + 1), chh, font=fb if (st and st.bold) else fr, fill=fgc)
                x += 1
    img.save(out)


def run_live(ep: Path, ep_id: str, cfg: dict[str, Any] | None, theme: Theme, *, replay: bool, speed: float,
             follow: bool, video: bool) -> None:
    console = Console(color_system="truecolor")
    d = load_data(ep, ep_id, cfg)
    frames = ensure_frames(d, 4 if (replay and video) else 0)
    t0 = time.time()
    last_load = 0.0
    with Live(console=console, screen=True, auto_refresh=False, redirect_stderr=False) as live:
        try:
            while True:
                now = time.time()
                w, h = console.size
                if replay:
                    v, done = replay_view(d, frames, now - t0, speed, video)
                    if done:
                        t0 = now  # loop the replay
                else:
                    if now - last_load > 0.5:
                        d = load_data(ep, ep_id, cfg)
                        if now - last_load > 5:
                            frames = ensure_frames(d)
                        last_load = now
                    cur = int((now - t0) / 2) % max(len(d.beats), 1)
                    v = static_view(d, frames, cur, running=follow)
                    v.elapsed = now - t0
                live.update(render(v, theme, max(w, 100), max(h, 30)), refresh=True)
                time.sleep(0.12 if replay else 0.4)
        except KeyboardInterrupt:
            pass


# --------------------------------------------------------------------------- live run (--run)

TITLE_STAGE = {"0": 0, "1": 1, "2": 2, "2a": 2, "2b": -1, "3": 3, "4": 4, "5": 5, "5b": 6, "6": 7}
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX = " │┃╭╮╰╯─┏┓┗┛━"
_RE_SKIP = re.compile(r"Skipping: Stage (\w+):")
_RE_DONE = re.compile(r"^Stage (\w+): .+ complete$")
_RE_ERR = re.compile(r"^Error in Stage (\w+):")
_RE_START = re.compile(r"^Stage (\w+): [^:]+: (?!.*complete$).+")
_RE_BEAT = re.compile(r"^beat (\d+)/(\d+) ")
_RE_CHUNK = re.compile(r"^chunk (\d+)/(\d+) (ok|fallback)")
_RE_FOUND = re.compile(r"^Found (\d+) (video clips|images|schematics)")
_RE_ASSETS = re.compile(r"Asset index saved: .*\((\d+) assets\)")
_RE_CUT = re.compile(r"^beat (\d+): cutout ")


class RunState:
    """Thread-safe state fed by parsing the pipeline subprocess output."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.status: dict[int, str] = {}
        self.started: dict[int, float] = {}
        self.ended: dict[int, float] = {}
        self.skipped: set[int] = set()
        self.counts: dict[str, int] = {}
        self.chunk: tuple[int, int] = (0, 0)
        self.beat: tuple[int, int] = (0, 0)
        self.cur_beat = 0
        self.log: list[str] = []
        self.current = -1
        self.exit_code: int | None = None
        self.cut_beats: list[int] = []
        self.polled = 0   # beats re-rendered since stage 6 started, counted from render_cache

    def _stamp(self) -> str:
        return time.strftime("%H:%M:%S")

    def feed(self, raw: str) -> None:
        text = _ANSI.sub("", raw).strip(_BOX).strip()
        if not text:
            return
        with self.lock:
            self.log.append(f"{self._stamp()} {text[:200]}")
            del self.log[:-200]
            now = time.time()
            if m := _RE_SKIP.search(text):
                i = TITLE_STAGE.get(m.group(1), -1)
                if i >= 0 and self.status.get(i) != "done":
                    self.status[i] = "done"
                    self.skipped.add(i)
                return
            if m := _RE_ERR.match(text):
                i = TITLE_STAGE.get(m.group(1), -1)
                if i >= 0:
                    self.status[i] = "error"
                    self.ended[i] = now
                return
            if m := _RE_DONE.match(text):
                i = TITLE_STAGE.get(m.group(1), -1)
                if i >= 0:
                    self.status[i] = "done"
                    self.ended[i] = now
                return
            if m := _RE_START.match(text):
                i = TITLE_STAGE.get(m.group(1), -1)
                if i >= 0 and "up to date" not in text:
                    if self.status.get(i) != "running":
                        self.started.setdefault(i, now)
                    self.status[i] = "running"
                    self.current = i
                return
            if m := _RE_BEAT.match(text):
                self.beat = (int(m.group(1)), int(m.group(2)))
                self.cur_beat = int(m.group(1)) - 1
            elif m := _RE_CHUNK.match(text):
                self.chunk = (int(m.group(1)), int(m.group(2)))
            elif m := _RE_FOUND.match(text):
                self.counts["clips" if m.group(2).startswith("video") else m.group(2)] = int(m.group(1))
            elif m := _RE_ASSETS.search(text):
                self.counts["assets"] = int(m.group(1))
            elif m := _RE_CUT.match(text):
                self.cut_beats.append(int(m.group(1)))
                self.counts["cutouts"] = len(self.cut_beats)

    def beats_done(self) -> tuple[int, int]:
        done, total = self.beat
        return max(done, self.polled), total

    def eta(self) -> float | None:
        done, total = self.beats_done()
        if self.status.get(7) == "running" and done > 0 and 7 in self.started:
            return (time.time() - self.started[7]) / done * (total - done)
        done, total = self.chunk
        if self.status.get(3) == "running" and done > 0 and 3 in self.started:
            return (time.time() - self.started[3]) / done * (total - done)
        return None


def live_stage_views(rs: RunState, d: Data) -> list[StageView]:
    base = stage_views(d)
    out: list[StageView] = []
    now = time.time()
    with rs.lock:
        for i, b in enumerate(base):
            sv = StageView(b.name, b.model, b.desc, link=b.link)
            st = rs.status.get(i, "idle")
            if st == "done":
                sv.status, sv.counters, sv.extra, sv.loop = "done", b.counters, b.extra, b.loop
                sv.seconds = b.seconds if i in rs.skipped else rs.ended.get(i, now) - rs.started.get(i, now)
                if i in rs.skipped:
                    sv.extra = "up to date"
            elif st == "error":
                sv.status, sv.extra = "error", "failed - see log"
                sv.seconds = rs.ended.get(i, now) - rs.started.get(i, now)
            elif st == "running":
                sv.status = "running"
                sv.seconds = now - rs.started.get(i, now)
                c: list[tuple[str, int | str]] = []
                if i == 2:
                    c = [(k, v) for k, v in rs.counts.items() if k in ("clips", "images", "schematics", "assets")]
                elif i == 3 and rs.chunk[1]:
                    c = [("chunk", f"{rs.chunk[0]}/{rs.chunk[1]}")]
                    sv.loop = rs.chunk
                elif i == 5:
                    c = [("cutouts", rs.counts.get("cutouts", 0))]
                elif i == 7 and rs.beats_done()[1]:
                    bd = rs.beats_done()
                    c = [("beat", f"{bd[0]}/{bd[1]}")]
                    sv.loop = bd
                eta = rs.eta()
                if eta is not None and i in (3, 7):
                    c.append(("eta", _fmt_s(eta)))
                sv.counters = c
            out.append(sv)
    return out


_live_frames: dict[str, Path] = {}


def _frame_of(video: Path, out: Path, at: float) -> Path | None:
    if out.exists():
        return out
    if not shutil.which("ffmpeg"):
        return None
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1",
                    "-vf", "scale=480:-2", "-q:v", "4", str(out)], capture_output=True, timeout=30, check=False)
    return out if out.exists() else None


def live_preview(d: Data, rs: RunState, cur: int) -> Path | None:
    """Frame of the beat being planned/rendered: newest render_cache segment, else the beat's asset."""
    out = d.ep / "work" / "dashboard_frames"
    out.mkdir(parents=True, exist_ok=True)
    if rs.status.get(7) == "running":
        cache = d.ep / "work" / "render_cache"
        try:
            segs = [p for p in cache.glob("beat_*.mp4") if p.stat().st_mtime >= rs.t0 - 1]
            newest = max(segs, key=lambda p: p.stat().st_mtime) if segs else None
        except OSError:
            newest = None
        if newest is not None:
            return _frame_of(newest, out / f"live_{newest.stem}.jpg", 0.4)
    if not d.beats:
        return None
    b = d.beats[min(max(cur, 0), len(d.beats) - 1)]
    sid = str(b.get("source_id", ""))
    for a in d.assets.get("assets") or []:
        if a.get("id") == sid:
            path = Path(a.get("upscaled_path") or a.get("filepath") or "")
            if a.get("asset_type") == "image" and path.exists():
                return path
            if path.exists():
                return _frame_of(path, out / f"asset_{sid[:60]}.jpg", float(b.get("media_start") or 0.5))
    return None


def _ansi_frame(root: Any, width: int, height: int) -> str:
    con = Console(width=width, height=height, file=io.StringIO(), force_terminal=True, color_system="truecolor",
                  legacy_windows=False)
    con.print(root, end="")
    return con.file.getvalue()  # type: ignore[attr-defined]


def run_dashboard(ep: Path, ep_id: str, cfg: dict[str, Any] | None, theme: Theme, *, force: bool = False,
                  from_stage: str | None = None, record: Path | None = None,
                  snapshot_at_end: Path | None = None) -> int:
    """Run ``cli run`` as a subprocess and show the dashboard live. Returns the pipeline exit code."""
    from src.common import ENV_CONFIG

    root_dir = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "src.cli"]
    if os.environ.get(ENV_CONFIG):
        cmd += ["--config", os.environ[ENV_CONFIG]]
    cmd += ["run", ep_id] + (["--force"] if force else []) + (["--from-stage", from_stage] if from_stage else [])
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}
    proc = subprocess.Popen(cmd, cwd=root_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=0, start_new_session=True)
    rs = RunState()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.default_int_handler)  # also when inherited as SIG_IGN

    def reader() -> None:
        assert proc.stdout is not None
        fd, buf = proc.stdout.fileno(), b""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            *parts, buf = re.split(rb"[\r\n]", buf)
            for part in parts:
                rs.feed(part.decode("utf-8", "replace"))
        if buf:
            rs.feed(buf.decode("utf-8", "replace"))
        rs.exit_code = proc.wait()

    threading.Thread(target=reader, daemon=True).start()
    console = Console(color_system="truecolor")
    cast = None
    if record:
        w0, h0 = console.size
        cast = open(record, "w", encoding="utf-8")  # noqa: SIM115
        cast.write(json.dumps({"version": 2, "width": w0, "height": h0, "timestamp": int(time.time()),
                               "title": f"AGENT PIPELINE {ep_id}"}) + "\n")
    d = load_data(ep, ep_id, cfg)
    last_load = last_rec = 0.0
    last_ansi = ""
    interrupted = False

    def view(final_frames: dict[str, Any] | None = None) -> View:
        nonlocal d, last_load
        now = time.time()
        if final_frames is not None or now - last_load > 1.0:
            d = load_data(ep, ep_id, cfg)
            last_load = now
        n = len(d.beats)
        if rs.status.get(7) == "running":
            try:
                rs.polled = sum(1 for p in (d.ep / "work" / "render_cache").glob("beat_*.mp4")
                                if p.stat().st_mtime >= rs.t0 - 1)
            except OSError:
                pass
            if not rs.beat[1]:
                rs.beat = (0, n)
        stages = live_stage_views(rs, d)
        plan_ready = rs.status.get(3) == "done" or rs.status.get(7) in ("running", "done")
        shown = n if plan_ready else 0
        if final_frames is not None:
            bf = final_frames.get("beat", {})
            cur = min(n // 2, max(n - 1, 0))
            preview = bf.get(cur)
            thumbs = [bf.get(i) for i in range(n)]
        else:
            cur = max(rs.cur_beat, rs.beats_done()[0] - 1) if rs.status.get(7) == "running" else int((now - rs.t0) / 2) % max(n, 1)
            cur = min(max(cur, 0), max(n - 1, 0))
            preview = live_preview(d, rs, cur)
            thumbs = [None] * n
            if rs.status.get(7) == "running":
                shown = min(rs.beats_done()[0], n)
        with rs.lock:
            log = rs.log[-10:]
        v = View(ep_id, stages, d.beats, shown, cur, log, preview, thumbs, mode="run", elapsed=now - rs.t0)
        return v

    def emit(v: View, w: int, h: int) -> None:
        nonlocal last_rec, last_ansi
        if cast and time.time() - last_rec > 1.0:
            with cell_preview():
                frame = _ansi_frame(render(v, theme, w, h), w, h)
            if frame != last_ansi:
                cast.write(json.dumps([round(time.time() - rs.t0, 3), "o", "\x1b[H" + frame.replace("\n", "\r\n")])
                           + "\n")
                last_ansi = frame
            last_rec = time.time()

    try:
        with Live(console=console, screen=True, auto_refresh=False, redirect_stderr=False) as live:
            try:
                while rs.exit_code is None:
                    w, h = console.size
                    w, h = max(w, 100), max(h, 30)
                    v = view()
                    live.update(render(v, theme, w, h), refresh=True)
                    emit(v, w, h)
                    time.sleep(0.25)
            except KeyboardInterrupt:
                interrupted = True
                rs.feed("interrupted: stopping pipeline (ctrl-c)")
                _stop(proc)
                live.update(render(view(), theme, *(max(x, m) for x, m in zip(console.size, (100, 30), strict=True))),
                            refresh=True)
    finally:
        if proc.poll() is None:
            _stop(proc)
    code = proc.wait() if not interrupted else 130
    time.sleep(0.2)
    if code == 0:
        for i in range(8):
            if rs.status.get(i) == "running":
                rs.status[i] = "done"
    w, h = console.size
    w, h = max(w, 100), max(h, 30)
    frames = ensure_frames(load_data(ep, ep_id, cfg)) if code == 0 else None
    v = view(frames if frames is not None else None)
    v.mode = "done" if code == 0 else "failed" if not interrupted else "stopped"
    final = render(v, theme, w, max(h - 5, 30))
    if cast:
        emit_frame = _ansi_frame(final, w, h)
        cast.write(json.dumps([round(time.time() - rs.t0, 3), "o", "\x1b[H" + emit_frame.replace("\n", "\r\n")])
                   + "\n")
        cast.close()
    console.print(final)
    if snapshot_at_end:
        export_snapshot(v, theme, snapshot_at_end, w, h)
        console.print(f"snapshot: {snapshot_at_end}")
    video = d.video
    console.print(f"exit code {code}" + (f" | video: {video}" if code == 0 and video.exists() else ""))
    return code


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """SIGINT the pipeline's process group, then escalate."""
    for sig, wait in ((signal.SIGINT, 8), (signal.SIGTERM, 4), (signal.SIGKILL, 2)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue
