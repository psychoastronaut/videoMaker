"""Score a scene plan against the ground-truth scene map (raw/scene_map.json).

Each map entry has the narration ``line`` and the asset-id substrings that are ``intended`` or
``acceptable`` for it. Beats are mapped to scenes by word overlap; a beat hits when any asset it
shows (source, overlay, pip) contains an intended/acceptable substring."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rich.console import Console

from src.common import read_json

console = Console()
WINDOW = 22.0


def _words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def beat_text(beat: dict[str, Any], words: list[dict[str, Any]]) -> str:
    return " ".join(w["word"] for w in words
                    if beat["start_time"] - 1e-6 <= (w["start"] + w["end"]) / 2 < beat["end_time"] + 1e-6)


def transcript_words(tr: dict[str, Any]) -> list[dict[str, Any]]:
    return [w for s in tr.get("segments", []) for w in s.get("words", [])]


def evaluate(plan: dict[str, Any], scene_map: list[dict[str, Any]], words: list[dict[str, Any]]
             ) -> dict[str, Any]:
    beats = plan["beats"]
    line_w = [_words(s["line"]) for s in scene_map]
    per_beat: list[dict[str, Any]] = []
    for b in beats:
        bw = _words(beat_text(b, words))
        best, ratio = None, 0.0
        for i, lw in enumerate(line_w):
            r = len(bw & lw) / max(len(bw), 1)
            if r > ratio:
                best, ratio = i, r
        if best is None or ratio < 0.5:
            continue
        shown = [a for a in (b.get("source_id"), b.get("overlay_asset"), b.get("pip_asset")) if a]
        sm = scene_map[best]
        inten = any(k in a for a in shown for k in sm["intended"])
        acc = inten or any(k in a for a in shown for k in sm["acceptable"])
        per_beat.append({"id": b["id"], "scene": sm["scene"], "intended": inten, "acceptable": acc,
                         "shown": shown})
    n = max(len(per_beat), 1)
    scenes: dict[int, dict[str, bool]] = {}
    for p in per_beat:
        s = scenes.setdefault(p["scene"], {"intended": False, "acceptable": False})
        s["intended"] |= p["intended"]
        s["acceptable"] |= p["acceptable"]
    uses: dict[str, int] = {}
    close = 0
    last: dict[str, float] = {}
    for b in beats:
        a = b.get("source_id")
        if not a or b.get("primary_source") == "stock":
            continue
        uses[a] = uses.get(a, 0) + 1
        if a in last and b["start_time"] - last[a] < WINDOW:
            close += 1
        last[a] = b["end_time"]
    return {
        "beats": len(beats), "mapped": len(per_beat),
        "beat_intended": sum(p["intended"] for p in per_beat) / n,
        "beat_acceptable": sum(p["acceptable"] for p in per_beat) / n,
        "scene_intended": sum(s["intended"] for s in scenes.values()) / max(len(scenes), 1),
        "scene_acceptable": sum(s["acceptable"] for s in scenes.values()) / max(len(scenes), 1),
        "close_repeats": close, "max_uses": max(uses.values(), default=0),
        "distinct_assets": len(uses),
        "templates": {t: sum(b["visual_template"] == t for b in beats)
                      for t in sorted({b["visual_template"] for b in beats})},
        "styles": {t: sum(b["beat_style"] == t for b in beats)
                   for t in sorted({b["beat_style"] for b in beats})},
        "transitions": {t: sum((b.get("transition") or "none") == t for b in beats)
                        for t in sorted({b.get("transition") or "none" for b in beats})},
    }


def run(work: Path, scene_map: Path | None = None) -> dict[str, Any]:
    ep = work.parent
    sm = read_json(scene_map or ep / "raw" / "scene_map.json")
    res = evaluate(read_json(work / "scene_plan.json"), sm,
                   transcript_words(read_json(work / "transcript.json")))
    for k, v in res.items():
        console.print(f"  {k}: {v:.1%}" if isinstance(v, float) else f"  {k}: {v}")
    return res
