"""Montage decisions from a Jev-style decision model (src/decision.py), text only.

Per beat the model reads the narration (+ neighbouring lines, position in the video) and
answers typed questions with probabilities:

* ``asset``     choice among the top-K candidates, each described in text (sidecar + caption);
* ``style``, ``template``, ``transition``  choices with a one-line "when to use" per option;
* ``cutout_named``, ``callout_worthy``, ``topic_change``  yes/no probabilities.

Probabilities are used two ways: the asset matrix P[beat][asset] is blended into the global
assignment score (``decision.weight``, normalised per beat) so reuse caps/windows still apply;
style/template/transition replace the current (LLM/rule) decision only above a confidence
threshold. Every applied decision is recorded in ``beat_reason``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rich.console import Console

from src.decision import (
    DecisionClient,
    DecisionError,
    Question,
    choice,
    normalise,
    noul,
)

console = Console()

DEFAULTS: dict[str, Any] = {
    "model": "nimble",
    "base_url": "http://localhost:11434",
    "timeout": 120,
    "retries": 2,
    "keep_alive": "5m",
    "cache": True,
    "unload": True,
    "weight": 0.6,            # score added to the best candidate of a beat (others scale by p/p_max)
    "candidates": 16,          # K assets offered per beat
    "gate_p": 0.35,            # a diagram skipped by the term-overlap gate re-enters above this p
    "thresholds": {"style": 0.55, "template": 0.55, "transition": 0.55,
                   "cutout_named": 0.6, "callout_worthy": 0.30, "topic_change": 0.6},
}

STYLE_CRITERIA = {
    "normal": "plain informative narration, nothing special",
    "deadpan": "dry, understated or ironic delivery of something surprising or absurd",
    "hype": "excited, triumphant: a big reveal, launch or achievement",
    "technical": "explains a mechanism, measurement, number or engineering detail",
    "fail": "something goes wrong: a failure, flaw, mistake, bug or disaster",
}
TEMPLATE_CRITERIA = {
    "fullscreen_clip": "one full-screen photo or footage that simply illustrates the line",
    "punch_in": "a quick emphatic zoom on a key fact, hook or punchline",
    "split_schematic": "explains how something works: footage on the left, labelled diagram on the right",
    "cutout_callout": "the line names one object, person or part that works as an isolated "
                      "cut-out subject with a label",
}
TEMPLATE_VALUE = {"fullscreen_clip": "fullscreen_clip", "punch_in": "punch_in_emphasis",
                  "split_schematic": "split_schematic", "cutout_callout": "cutout_callout"}
TRANSITION_CRITERIA = {
    "none": "hard cut: the same thought continues, or an energetic beat",
    "crossfade": "soft blend between two related consecutive ideas",
    "dissolve": "slower blend: time passes or the subject shifts gently",
    "fadeblack": "dip to black: a new chapter, a dramatic pause or a big change of topic",
}
_HARD = {"hype", "fail"}  # energetic beats are cut, not blended (same rule as the LLM planner)


def decision_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(config.get("decision") or {})}
    cfg["thresholds"] = {**DEFAULTS["thresholds"], **((config.get("decision") or {})
                                                      .get("thresholds") or {})}
    return cfg


def planner_mode(config: dict[str, Any]) -> str:
    import os

    mode = str(os.environ.get("VIDEOMAKER_PLANNER")
               or (config.get("llm") or {}).get("planner") or "llm").lower()
    return mode if mode in {"llm", "decision", "hybrid"} else "llm"


def short(text: str, n: int) -> str:
    return " ".join(str(text).split())[:n]


def asset_description(asset: dict[str, Any] | None, caption: str = "", n: int = 230) -> str:
    """Short text for the model: type, sidecar free text (or the indexed description), caption."""
    a = asset or {}
    kind = {"clip": "video clip", "image": "photo", "schematic": "diagram"}.get(
        a.get("asset_type", ""), a.get("asset_type", "asset"))
    desc = ""
    if a.get("filepath"):
        from src.describe import sidecar_parts

        free, gen, _ = sidecar_parts(Path(str(a["filepath"]) + ".txt"))
        desc = free or gen
    if not desc:
        desc = re.sub(r"^\w+ \|[^|]*\|\s*", "", str(a.get("description", "")))
    parts = [short(desc, n)]
    if caption and caption.lower() not in desc.lower():
        parts.append(f"caption: {short(caption, 90)}")
    return f"{kind}: " + "; ".join(p for p in parts if p)


class DecisionPlanner:
    def __init__(self, config: dict[str, Any], work: Path,
                 client: DecisionClient | None = None) -> None:
        self.cfg = decision_config(config)
        self.work = work
        self.client = client or DecisionClient(self.cfg, work / "decision_cache")
        self.name = self.client.model
        self.enabled = True
        self.p_asset: dict[int, dict[str, float]] = {}
        self.beat_answers: dict[int, dict[str, Any]] = {}
        self.errors = 0
        self.times: dict[int, tuple[Any, Any]] = {}
        self._freed = False

    # ------------------------------------------------------------------ helpers

    def _prepare(self) -> None:
        if not self._freed and self.cfg.get("unload", True):
            freed = self.client.free_others()
            if freed:
                console.print(f"  [dim]Unloaded {', '.join(freed)} before the decision model[/]")
        self._freed = True

    def _fail(self, e: Exception) -> None:
        self.errors += 1
        console.print(f"  [yellow]decision model call failed ({type(e).__name__}: "
                      f"{str(e)[:100]}); current decision kept[/]")
        from src.decision import DecisionUnavailable

        if isinstance(e, DecisionUnavailable) or self.errors >= 3:
            self.enabled = False
            console.print("  [yellow]decision model disabled for this run[/]")

    @staticmethod
    def state_for(i: int, beats: list[Any], texts: dict[int, str], total: float) -> dict[str, Any]:
        b = beats[i]
        frac = (b.start_time + b.end_time) / 2 / max(total, 1e-6)
        where = "opening (hook)" if frac < 0.08 else "closing" if frac > 0.92 else \
            "first half" if frac < 0.5 else "second half"
        return {
            "narration": texts.get(b.id, ""),
            "previous_line": texts.get(beats[i - 1].id, "") if i > 0 else "(start of video)",
            "next_line": texts.get(beats[i + 1].id, "") if i + 1 < len(beats) else "(end of video)",
            "position": f"beat {i + 1} of {len(beats)}, {where} of the video",
            "duration_seconds": round(b.end_time - b.start_time, 1),
            "video_kind": "short science/engineering explainer with archival photos, footage and diagrams",
        }

    # ------------------------------------------------------------------ beat-level questions

    def beat_questions(self) -> dict[str, Question]:
        allq = {
            "style": choice("Which delivery style does this narration line have?", STYLE_CRITERIA),
            "template": choice("Which visual layout suits this line best?", TEMPLATE_CRITERIA),
            "transition": choice("How should the picture join INTO this line from the previous "
                                 "one?", TRANSITION_CRITERIA),
            "cutout_named": noul("Does the narration name a specific object, person or part that "
                                 "could be shown on its own as a cut-out subject?"),
            "callout_worthy": noul("Does this line contain a key term, name or number worth a "
                                   "short on-screen label?"),
            "topic_change": noul("Does this line start a new topic compared with the previous "
                                 "line?"),
        }
        want = self.cfg.get("questions")
        return {k: q for k, q in allq.items() if not want or k in want}

    def decide_beats(self, beats: list[Any], texts: dict[int, str], total: float
                     ) -> dict[int, dict[str, Any]]:
        """Ask the beat-level questions for every beat (cached). {beat_id: {name: Answer}}."""
        out: dict[int, dict[str, Any]] = {}
        if not self.enabled:
            return out
        self._prepare()
        qs = self.beat_questions()
        for i, b in enumerate(beats):
            if not self.enabled:
                break
            if not texts.get(b.id):
                continue
            try:
                out[b.id] = self.client.ask(self.state_for(i, beats, texts, total), qs)
            except DecisionError as e:
                self._fail(e)
        self.beat_answers = out
        return out

    def apply_beat_decisions(self, beats: list[Any], answers: dict[int, dict[str, Any]]
                             ) -> dict[str, int]:
        """Overwrite style/template/transition when the model is confident (probability of the
        top option >= threshold); otherwise the existing decision stays. Returns counts."""
        from src.schemas import BeatStyle, VisualTemplate

        th = self.cfg["thresholds"]
        n = {"style": 0, "template": 0, "transition": 0, "callout_off": 0}
        for i, b in enumerate(beats):
            ans = answers.get(b.id)
            if not ans:
                continue
            notes: list[str] = []
            st = ans.get("style")
            if st and st.choice in STYLE_CRITERIA and st.p >= th["style"] \
                    and st.choice != b.beat_style.value:
                b.beat_style = BeatStyle(st.choice)
                n["style"] += 1
                notes.append(f"style={st.choice} p={st.p:.2f}")
            tp = ans.get("template")
            tv = TEMPLATE_VALUE.get(tp.choice or "") if tp else None
            cn = ans["cutout_named"].p if "cutout_named" in ans else 0.5
            if tp and tv and tp.p >= th["template"] and tv != b.visual_template.value:
                cur = b.visual_template.value
                simple = {"fullscreen_clip", "punch_in_emphasis"}
                ok = (cur in simple and tv in simple) or (
                    tv == "cutout_callout" and cur in simple
                    and cn >= th["cutout_named"]) or (
                    tv == "split_schematic" and cur in simple)
                if cur == "cutout_callout" and tv != cur and cn < 0.2:
                    ok = tv in simple
                if ok:
                    b.visual_template = VisualTemplate(tv)
                    n["template"] += 1
                    notes.append(f"template={tp.choice} p={tp.p:.2f}")
            tr = ans.get("transition")
            probs = dict(tr.probabilities) if tr else {}
            tc = ans["topic_change"].p if "topic_change" in ans else 0.0
            for k in ("dissolve", "fadeblack"):  # a topic change favours the heavier joins
                if k in probs:
                    probs[k] += 0.25 * tc * probs[k]
            probs = normalise(probs)
            best = max(probs, key=probs.__getitem__) if probs else None
            if best and probs[best] >= th["transition"] and i > 0:
                if b.beat_style.value in _HARD and best not in {"none"}:
                    best = "none"
                if best != (b.transition or "none"):
                    b.transition = best
                    b.transition_duration = None
                    n["transition"] += 1
                    notes.append(f"transition={best} p={probs.get(best, 0):.2f}")
            cw = ans.get("callout_worthy")
            if cw and cw.p < th["callout_worthy"] and b.callout_text:
                b.callout_text = None
                n["callout_off"] += 1
                notes.append(f"callout off p={cw.p:.2f}")
            if notes:
                reason = f"decision:{self.name} " + ", ".join(notes)
                b.beat_reason = f"{b.beat_reason} | {reason}" if b.beat_reason else reason
        return n

    # ------------------------------------------------------------------ asset choice

    def decide_assets(self, rows: list[dict[str, Any]], assets_by_id: dict[str, dict[str, Any]]
                      ) -> dict[int, dict[str, float]]:
        """rows: {id, state, cands:[(asset_id, caption)]} -> {beat_id: {asset_id: p}} (sums to 1
        per beat). Beats whose call fails are simply absent (no bonus)."""
        out: dict[int, dict[str, float]] = {}
        if not self.enabled:
            return out
        self._prepare()
        for r in rows:
            if not self.enabled:
                break
            if len(r["cands"]) < 2:
                continue
            crit = {aid: asset_description(assets_by_id.get(aid), cap)
                    for aid, cap in r["cands"]}
            q = choice("Which asset best illustrates this narration line? Prefer the one that "
                       "shows what the line is literally about (objects, action, year); a "
                       "diagram only when the line explains how something works.", crit)
            try:
                ans = self.client.ask(r["state"], {"asset": q})["asset"]
            except DecisionError as e:
                self._fail(e)
                continue
            out[r["id"]] = normalise({k: ans.probabilities.get(k, 0.0) for k in crit})
            self.times[r["id"]] = (r.get("start"), r.get("end"))
        self.p_asset = out
        return out

    def asset_bonus(self, beat_id: int, asset_id: str) -> float | None:
        """Score to add in the global assignment: weight * p / p_max (per-beat normalisation);
        None when the model did not rate this asset for this beat."""
        probs = self.p_asset.get(beat_id)
        if not probs or asset_id not in probs:
            return None
        top = max(probs.values()) or 1.0
        return float(self.cfg["weight"]) * probs[asset_id] / top

    def close(self) -> None:
        import json

        console.print(f"  Decision model {self.name}: {self.client.summary()}")
        try:
            (self.work / "decision_stats.json").write_text(json.dumps(
                {"model": self.name, **self.client.stats, "errors": self.errors,
                 "enabled": self.enabled}))
            (self.work / "decision_assets.json").write_text(json.dumps(
                [{"id": i, "start": self.times.get(i, (None, None))[0],
                  "end": self.times.get(i, (None, None))[1], "p": p}
                 for i, p in self.p_asset.items()]))
        except OSError:
            pass
        self.client.close()


def candidate_union(index: Any, vectors: list[Any], prior: dict[str, float], types: tuple[str, ...],
                    k: int) -> list[tuple[str, str]]:
    """Top-K assets for a beat: ~60 % by embedding rank (best query vector wins), the rest by the
    sidecar/description prior (diagrams included). Returns [(asset_id, caption)]."""
    best: dict[str, Any] = {}
    for v in vectors:
        for h in index.search(v, types, k, None, None):
            if h.asset_id not in best or h.score > best[h.asset_id].score:
                best[h.asset_id] = h
    ranked = sorted(best.values(), key=lambda h: -h.score)
    n_vis = max(1, round(k * 0.6))
    chosen = {h.asset_id: h.caption for h in ranked[:n_vis]}
    for aid, _v in sorted(prior.items(), key=lambda kv: -kv[1]):
        if len(chosen) >= k:
            break
        if aid not in chosen and _v > 0:
            chosen[aid] = ""
    for h in ranked[n_vis:]:  # still room: next embedding hits
        if len(chosen) >= k:
            break
        chosen.setdefault(h.asset_id, h.caption)
    return list(chosen.items())
