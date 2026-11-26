"""Template-mix rebalancing: explicit target shares per visual template + max-run rules.

The planner's per-beat choices are content-driven but drift to "fullscreen everywhere". This
pass runs after the semantic assignment: for every template it asks the caller whether a
given fullscreen beat could *really* use it (a matching clear-subject still for a cutout, a
diagram that shares terms with the line for a split, an emphasis beat for a punch-in) and how
well, then converts the best-fitting beats until the mix is inside the targets and no template
repeats in a monotone stretch. Beats that cannot support a template are never converted, so the
mix is a goal, not a quota that overrides the content.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from src.schemas import VisualTemplate

FS = VisualTemplate.FULLSCREEN_CLIP
PUNCH = VisualTemplate.PUNCH_IN_EMPHASIS
CUT = VisualTemplate.CUTOUT_CALLOUT
SPLIT = VisualTemplate.SPLIT_SCHEMATIC

DEFAULTS: dict[str, Any] = {
    "fullscreen_max": 0.55,           # share of plain fullscreen_clip beats (upper bound)
    "punch": [0.12, 0.18],            # punch_in_emphasis (emphasis / fail / hype)
    "cutout": [0.10, 0.15],           # cutout_callout (one clear concrete subject)
    "split": [0.08, 0.12],            # split_schematic (a diagram supports the line)
    "max_fullscreen_run": 3,          # longest allowed stretch of plain fullscreen beats
}

Option = tuple[float, Callable[[], None]]


def settings(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if k in DEFAULTS})
    return out


def mix(beats: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for b in beats:
        out[b.visual_template.value] = out.get(b.visual_template.value, 0) + 1
    return out


def _spaced(beats: list[Any], i: int, tmpl: VisualTemplate, gap: int = 1) -> bool:
    """No beat within ``gap`` positions already uses the same special template."""
    lo, hi = max(0, i - gap), min(len(beats), i + gap + 1)
    return all(beats[j].visual_template != tmpl for j in range(lo, hi) if j != i)


def _fs_runs(beats: list[Any]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, b in enumerate(beats):
        if b.visual_template == FS:
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(beats) - 1))
    return runs


def rebalance(beats: list[Any], options: dict[VisualTemplate, Callable[[Any], Option | None]],
              cfg: dict[str, Any] | None = None) -> dict[str, int]:
    """Convert fullscreen beats until the mix meets the targets; returns the final mix.

    ``options[tmpl](beat)`` -> ``(fit, apply)`` when the beat can take ``tmpl`` (``apply()``
    performs the change) else None. Only plain fullscreen beats are converted."""
    s = settings(cfg)
    n = len(beats)
    if n < 6:
        return mix(beats)
    caps = {t: math.ceil(float(s[k][1]) * n) for t, k in
            ((PUNCH, "punch"), (CUT, "cutout"), (SPLIT, "split"))}
    floors = {t: math.ceil(float(s[k][0]) * n) for t, k in
              ((PUNCH, "punch"), (CUT, "cutout"), (SPLIT, "split"))}

    def count(t: VisualTemplate) -> int:
        return sum(b.visual_template == t for b in beats)

    def convert(tmpl: VisualTemplate, only: range | None = None) -> bool:
        """Best-fitting spaced fullscreen beat -> tmpl (optionally restricted to indices)."""
        fn = options.get(tmpl)
        if fn is None:
            return False
        scored = []
        for i, b in enumerate(beats):
            if b.visual_template != FS or (only is not None and i not in only):
                continue
            if not _spaced(beats, i, tmpl):
                continue
            opt = fn(b)
            if opt is not None:
                scored.append((opt[0], i, opt[1]))
        if not scored:
            return False
        scored.sort(key=lambda x: -x[0])
        scored[0][2]()
        beats[scored[0][1]].visual_template = tmpl
        return True

    # scarce, content-bound templates first (split, cutout), then the flexible punch-in
    for tmpl in (SPLIT, CUT, PUNCH):
        while count(tmpl) < floors[tmpl] and convert(tmpl):
            pass
    # too many plain fullscreen beats / monotone stretches: spend the remaining headroom
    limit = int(float(s["fullscreen_max"]) * n)
    max_run = int(s["max_fullscreen_run"])
    stuck: set[tuple[int, int]] = set()
    for _ in range(2 * n):
        runs = [r for r in _fs_runs(beats) if r[1] - r[0] + 1 > max_run and r not in stuck]
        over = count(FS) > limit
        if not runs and not over:
            break
        rng = None
        if runs:
            a, z = runs[0]
            # break the stretch from the inside (the last beat may be used when the run ends
            # the video: nothing follows it)
            rng = range(a + 1, z + 1 if z == n - 1 else z)
        done = False
        for tmpl in (CUT, SPLIT, PUNCH):
            if count(tmpl) < caps[tmpl] and convert(tmpl, rng):
                done = True
                break
        if not done:
            soft = math.ceil(0.20 * n)  # last resort while fullscreen is still over target
            if ((runs and count(PUNCH) < math.ceil(caps[PUNCH] * 1.4))
                    or (over and count(PUNCH) < soft)) and convert(PUNCH, rng):
                done = True
        if not done:
            if not runs:
                break
            stuck.add(runs[0])  # this stretch cannot be broken: try the others
    return mix(beats)
