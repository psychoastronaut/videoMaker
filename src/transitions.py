"""Beat-to-beat transitions (Stage 6 xfade) and their automatic selection.

A beat's ``transition`` describes the join INTO that beat from the previous one. Names are a
short vetted list (all verified against ``ffmpeg -h filter=xfade``); anything else becomes
``None`` (= automatic). Pure functions, no ffmpeg needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# public name -> ffmpeg xfade transition ("none" = hard cut)
XFADE: dict[str, str] = {
    "none": "",
    "crossfade": "fade",
    "dissolve": "dissolve",
    "fadeblack": "fadeblack",
    "fadewhite": "fadewhite",
    "zoomin": "zoomin",
    "smoothleft": "smoothleft",
    "slideleft": "slideleft",
    "wipeleft": "wipeleft",
    # glitch family (psycho_terminal): pixelate / slice / horizontal-blur joins
    "glitch": "pixelize",
    "glitchslice": "hlslice",
    "glitchblur": "hblur",
}
GLITCH = ("glitch", "glitchslice", "glitchblur")
# joins that slide/wipe the picture sideways: valid names, but never used unless asked for
# (``allow_motion``): viewers read them as amateur, and the polish pass rewrites them
MOTION = ("smoothleft", "slideleft", "wipeleft")
NAMES = tuple(XFADE)
_ALIASES = {"cut": "none", "hard_cut": "none", "fade": "crossfade", "xfade": "crossfade",
            "fade_black": "fadeblack", "dip_to_black": "fadeblack", "fade_white": "fadewhite"}

DEFAULT_DURATION = 0.35
MAX_FRACTION = 0.40       # a transition may use < 40% of the shorter neighbouring beat
MIN_FRAMES = 3
MAX_GROUP = 6             # beats joined by one xfade graph (longer runs are split by a cut)
STYLE_FACTOR = {"hype": 0.6, "fail": 0.6, "normal": 0.85, "technical": 1.15, "deadpan": 1.15}


def normalize_name(name: str) -> str | None:
    n = name.strip().lower().replace(" ", "")
    n = _ALIASES.get(n, n)
    return n if n in XFADE else None


def auto_transition(beat: dict[str, Any], prev: dict[str, Any] | None, chapter: bool,
                    glitch: bool = False) -> str:
    """Default for the join into ``beat``: hard cuts for energetic beats, soft joins for
    explanatory ones, a dip/zoom at chapter changes."""
    if prev is None:
        return "none"
    style = str(beat.get("beat_style", "normal"))
    if glitch and style in ("hype", "fail") and (chapter or beat.get("emphasis")
                                                   or style == "fail"):
        return "glitchslice" if style == "fail" else "glitch"
    if glitch and chapter and style == "technical":
        return "glitchblur"
    if chapter:
        return "zoomin" if style in ("hype", "fail") else "fadeblack"
    if style in ("hype", "fail") or beat.get("emphasis") \
            or beat.get("visual_template") == "punch_in_emphasis":
        return "none"
    # soft joins only between two long beats; on fast cuts a hard cut reads as edited-on-beat
    slow = float(beat.get("duration") or 0) >= 3.2 and float(prev.get("duration") or 0) >= 3.2
    if not slow:
        return "none"
    if style == "technical" or beat.get("visual_template") == "split_schematic":
        return "dissolve"
    if style == "deadpan":
        return "crossfade"
    return "none"


@dataclass(frozen=True)
class Join:
    """Resolved join into a beat: ffmpeg-facing name and overlap in frames (0 = cut)."""
    name: str = "none"
    frames: int = 0


def resolve_joins(beats: list[dict[str, Any]], spans: list[int], fps: int, *,
                  enabled: bool = True, default_duration: float = DEFAULT_DURATION,
                  chapters: set[int] | None = None, max_group: int = MAX_GROUP, glitch: bool = False,
                  allow_motion: bool = False) -> list[Join]:
    """One Join per beat (the first is always a cut). Enforces: overlap < 40% of the shorter
    neighbour, >= MIN_FRAMES, head+tail of a middle beat leave a visible body, and at most
    ``max_group`` beats per xfade run. Beat ids in ``chapters`` start a chapter."""
    joins = [Join() for _ in beats]
    if not enabled:
        return joins
    chapters = chapters or set()
    run = 1
    for i in range(1, len(beats)):
        b = beats[i]
        name = b.get("transition") or auto_transition(b, beats[i - 1], int(b["id"]) in chapters, glitch)
        if name in MOTION and not allow_motion:
            name = "crossfade"
        if not XFADE.get(name):
            run = 1
            continue
        dur = b.get("transition_duration")
        if dur is None:
            dur = default_duration * STYLE_FACTOR.get(str(b.get("beat_style", "normal")), 1.0)
        n = round(float(dur) * fps)
        if n < MIN_FRAMES or n >= MAX_FRACTION * min(spans[i - 1], spans[i]) or run >= max_group:
            run = 1
            continue
        if joins[i - 1].frames and joins[i - 1].frames + n > 0.7 * spans[i - 1]:
            run = 1
            continue
        joins[i] = Join(name, n)
        run += 1
    return joins
