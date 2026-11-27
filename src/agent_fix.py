"""Agentic fix mode: small, validated, reversible edits of work/scene_plan.json.

``fix <ep> "<instruction>"``: a local LLM (Ollama /api/chat, JSON-schema output) turns a PT/EN
instruction into *actions*; code (not the LLM) locates beats, picks replacement assets from the
semantic index, validates everything, shows a diff and only then writes the plan. Everything is
local: no network, no cloud. The LLM never writes the plan; it only fills the action schema.

Pieces (all injectable for tests): ``chat`` (LLM), ``searcher`` (asset retrieval),
``renderer`` (re-render changed beats), ``confirm``.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from src.common import (
    BURNED_FILE,
    EXCLUDED_FILE,
    episode_dir,
    excluded_ids,
    load_burned,
    load_config,
    read_json,
    write_json,
)

console = Console()

ACTIONS = ("replace_asset", "exclude_asset", "change_template", "set_crop", "set_zoom",
           "set_trim", "set_callout", "remove_callout", "set_transition", "split_beat",
           "merge_beats", "move_overlay", "remove_highlight")
TEMPLATES = ("fullscreen_clip", "punch_in_emphasis", "split_schematic", "cutout_callout")
FOCUS = {"center": (0.5, 0.5), "top": (0.5, 0.0), "bottom": (0.5, 1.0), "left": (0.0, 0.5),
         "right": (1.0, 0.5), "top-left": (0.0, 0.0), "top-right": (1.0, 0.0),
         "bottom-left": (0.0, 1.0), "bottom-right": (1.0, 1.0)}
CORNERS = ("tr", "tl", "br", "bl")
BY = ("video_time", "narration_time", "quote", "beat_id", "asset", "previous")
MIN_PIECE = 0.8   # seconds: shortest beat a split may create
REUSE_WINDOW = 22.0


class FixError(Exception):
    """A problem the user can act on (Ollama down, unknown beat, invalid action...)."""


# --------------------------------------------------------------------------- schema

_REF = {"type": "object", "required": ["by", "value"], "properties": {
    "by": {"type": "string", "enum": list(BY)}, "value": {"type": "string"}}}
ACTION_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["actions"], "properties": {
        "explanation": {"type": "string"},
        "actions": {"type": "array", "items": {"type": "object", "required": ["type"], "properties": {
            "type": {"type": "string", "enum": list(ACTIONS)},
            "beat": _REF,                       # which beat (not needed for a global exclude)
            "asset": {"type": "string"},        # asset id/name, "@last" or "@current"
            "query": {"type": "string"},        # English visual query for a semantic search
            "slot": {"type": "string", "enum": ["main", "overlay", "pip"]},
            "template": {"type": "string", "enum": list(TEMPLATES)},
            "start": {"type": "number"}, "end": {"type": "number"},
            "focus": {"type": "string", "enum": list(FOCUS)},
            "zoom": {"type": "number"},
            "region": {"type": "array", "items": {"type": "number"}},
            "text": {"type": "string"},
            "transition": {"type": "string"}, "duration": {"type": "number"},
            "at": _REF,                          # split point
            "with_next": {"type": "boolean"},
            "corner": {"type": "string", "enum": list(CORNERS)},
        }}}}}

SYSTEM = """You convert an editing instruction for a narrated video into JSON actions.
The instruction may be Portuguese or English. Output ONLY the JSON schema you were given.
Each action targets a beat with "beat": {"by", "value"}:
- by=video_time: a time in the FINAL video ("0:08", "no 0:08", "8s"); value "0:08".
- by=narration_time: a time in the narration/voice-over only.
- by=quote: words the narrator SAYS ("quando eu falo 1990" -> value "1990"; give the spoken
  words only, in the transcript language, as short as possible).
- by=beat_id: "beat 12" -> "12".   by=asset: every beat showing that asset.
- by=previous: the beat targeted by the previous fix.
Actions: replace_asset (asset=<id> OR query=<English description of what to show>; slot main by
default), exclude_asset (asset=<id> or "@last" meaning "this clip" = the asset the previous fix
removed or discussed; it is banned from the whole project and replaced in every beat using it),
change_template, set_crop / set_zoom (focus + zoom>1, or region [x,y,w,h] fractions),
set_trim (start/end seconds inside the clip), set_callout / remove_callout (text),
set_transition (transition name + duration), split_beat (at: where), merge_beats (with_next),
move_overlay (picture-in-picture corner tl/tr/bl/br), remove_highlight.
A query is ONE rich English sentence (8-15 words: subject, action, setting), e.g.
"space shuttle robotic arm releasing the Hubble Space Telescope into orbit above Earth"; never
a keyword fragment and never "query=...". Put it in "query", not in "asset". Use asset ids only if copied exactly from the list; for anything else give a query, never an invented id.
Do not invent actions the user did not ask for."""


# --------------------------------------------------------------------------- context


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.'][a-z0-9]+)*", _fold(text))


def fmt_time(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def parse_time(v: str) -> float | None:
    """'0:08', '1:02.5', '8s', '8.5', '00:01:02' -> seconds, else None."""
    v = v.strip().lower().replace(",", ".")
    m = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", v)
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seg|segundos?)?", v)
    return float(m.group(1)) if m else None


def intro_offset(cfg: dict[str, Any]) -> float:
    """Seconds the final video is ahead of the narration/plan timeline (boot intro card)."""
    r = cfg.get("render") or {}
    if not r.get("branding", False):
        return 0.0
    fps = int((cfg.get("pipeline") or {}).get("fps", 30)) or 30
    return round(max(float(r.get("intro_seconds", 3.0)), 0.5) * fps) / fps


@dataclass
class Ctx:
    ep: Path
    work: Path
    plan: dict[str, Any]
    words: list[dict[str, Any]]
    assets: dict[str, dict[str, Any]]
    cfg: dict[str, Any]
    intro: float = 0.0
    last_targets: list[dict[str, Any]] = field(default_factory=list)

    @property
    def beats(self) -> list[dict[str, Any]]:
        return self.plan["beats"]

    def speech(self, b: dict[str, Any]) -> str:
        return " ".join(w["word"].strip() for w in self.words
                        if b["start_time"] - 1e-3 <= w["start"] < b["end_time"] - 1e-3)

    def video_time(self, t: float) -> float:
        return t + self.intro


def load_ctx(episode_id: str) -> Ctx:
    ep = episode_dir(episode_id)
    work = ep / "work"
    if not (work / "scene_plan.json").is_file():
        raise FixError(f"No scene plan in {work}: run `plan {episode_id}` first.")
    plan = read_json(work / "scene_plan.json")
    tr = read_json(work / "transcript.json") if (work / "transcript.json").is_file() else {}
    words = [w for seg in tr.get("segments", []) for w in (seg.get("words") or [])]
    assets = {}
    if (work / "local_assets.json").is_file():
        assets = {a["id"]: a for a in read_json(work / "local_assets.json").get("assets", [])}
    cfg = load_config()
    ctx = Ctx(ep, work, plan, words, assets, cfg, intro_offset(cfg))
    ctx.last_targets = last_targets(work)
    return ctx


def last_targets(work: Path) -> list[dict[str, Any]]:
    log = work / "fix_log.jsonl"
    if not log.is_file():
        return []
    for line in reversed(log.read_text(encoding="utf-8").splitlines()):
        try:
            return json.loads(line).get("targets", [])
        except json.JSONDecodeError:
            continue
    return []


# --------------------------------------------------------------------------- locating beats


def beat_at(ctx: Ctx, t: float) -> dict[str, Any]:
    for b in ctx.beats:
        if b["start_time"] - 1e-6 <= t < b["end_time"] - 1e-6:
            return b
    if ctx.beats and abs(t - ctx.beats[-1]["end_time"]) < 0.05:
        return ctx.beats[-1]
    raise FixError(f"No beat at narration time {fmt_time(t)} (the narration is "
                   f"{fmt_time(ctx.beats[-1]['end_time'])} long)")


def find_quote(ctx: Ctx, quote: str) -> list[float]:
    """Start times (narration seconds) of every occurrence of the quoted words in the
    transcript (accent/case/punctuation-insensitive; falls back to fuzzy matching)."""
    q = _norm_words(quote)
    if not q:
        return []
    toks: list[tuple[str, float]] = []
    for w in ctx.words:
        for part in _norm_words(w["word"]):
            toks.append((part, float(w["start"])))
    seq = [t for t, _ in toks]
    hits = [toks[i][1] for i in range(len(seq) - len(q) + 1) if seq[i:i + len(q)] == q]
    if hits:
        return hits
    import difflib

    best, at = 0.0, None
    for i in range(len(seq) - len(q) + 1):
        r = difflib.SequenceMatcher(None, " ".join(seq[i:i + len(q)]), " ".join(q)).ratio()
        if r > best:
            best, at = r, toks[i][1]
    return [at] if at is not None and best >= 0.8 else []


def ref_time(ctx: Ctx, ref: dict[str, str]) -> float:
    """A split point: narration seconds from a time or a quote reference."""
    by, val = ref.get("by"), str(ref.get("value", ""))
    if by == "video_time":
        t = parse_time(val)
        if t is None:
            raise FixError(f"Cannot read the time '{val}'")
        return t - ctx.intro
    if by == "narration_time":
        t = parse_time(val)
        if t is None:
            raise FixError(f"Cannot read the time '{val}'")
        return t
    if by == "quote":
        hits = find_quote(ctx, val)
        if not hits:
            raise FixError(f"The words '{val}' are not in the transcript")
        return hits[0]
    raise FixError(f"Cannot use '{by}' as a time")


def locate_beats(ctx: Ctx, ref: dict[str, str] | None, notes: list[str]) -> list[dict[str, Any]]:
    if not ref:
        raise FixError("The action needs a beat (time, quote or id)")
    by, val = ref.get("by"), str(ref.get("value", "")).strip()
    if by in ("video_time", "narration_time"):
        t = ref_time(ctx, ref)
        if t < 0:
            raise FixError(f"{val} is inside the {ctx.intro:.1f}s intro card, not in a beat")
        return [beat_at(ctx, t)]
    if by == "quote":
        hits = find_quote(ctx, val)
        if not hits:
            raise FixError(f"The words '{val}' are not in the transcript")
        beats = [beat_at(ctx, hits[0])]
        if len(hits) > 1:
            notes.append(f"'{val}' is said {len(hits)} times; using the first "
                         f"({fmt_time(ctx.video_time(hits[0]))}). Say a time to pick another.")
        return beats
    if by == "beat_id":
        m = re.search(r"\d+", val)
        if m and 0 <= int(m.group()) < len(ctx.beats):
            return [ctx.beats[int(m.group())]]
        raise FixError(f"No beat '{val}' (0..{len(ctx.beats) - 1})")
    if by == "asset":
        aid = resolve_asset(ctx, val)
        out = [b for b in ctx.beats if aid in (b.get("source_id"), b.get("overlay_asset"),
                                               b.get("pip_asset"))]
        if not out:
            raise FixError(f"No beat uses '{aid}'")
        return out
    if by == "previous":
        ids = {t.get("beat") for t in ctx.last_targets}
        out = [b for b in ctx.beats if b["id"] in ids]
        if not out:
            raise FixError("There is no previous fix to refer to")
        return out
    raise FixError(f"Unknown beat reference {ref}")


def resolve_asset(ctx: Ctx, name: str, beat: dict[str, Any] | None = None) -> str:
    """Exact id, file name, stem, '@last' / '@current', or a unique fuzzy match."""
    n = (name or "").strip()
    if n == "@current":
        if beat and beat.get("source_id"):
            return str(beat["source_id"])
        raise FixError("'@current' needs a beat")
    if n in ("@last", "last", "this", "esse clipe"):
        olds = [t["asset"] for t in ctx.last_targets if t.get("asset")]
        if not olds:
            raise FixError("There is no previous fix: name the clip to remove")
        return str(olds[0])
    known = set(ctx.assets) | {b.get(k) for b in ctx.beats
                               for k in ("source_id", "overlay_asset", "pip_asset") if b.get(k)}
    if n in known:
        return n
    low = _fold(n)
    for aid, a in ctx.assets.items():
        if low in (_fold(a.get("filename", "")), _fold(Path(a.get("filename", "")).stem)):
            return aid
    cands = [aid for aid in known if low and low in _fold(aid)]
    if len(cands) == 1:
        return cands[0]
    toks = set(_norm_words(n)) - {"clip", "image", "schematic"}
    if toks:
        scored = sorted(((len(toks & set(_norm_words(aid + " " + ctx.assets.get(aid, {}).get(
            "description", "")))), aid) for aid in known), reverse=True)
        if scored and scored[0][0] >= max(2, len(toks) // 2) and (
                len(scored) < 2 or scored[0][0] > scored[1][0]):
            return scored[0][1]
    raise FixError(f"Unknown asset '{name}'")


# --------------------------------------------------------------------------- retrieval


class Searcher:
    """Semantic (SigLIP / Ollama) retrieval over the episode index; keyword fallback."""

    def __init__(self, ctx: Ctx) -> None:
        from src.semantic import get_backend
        from src.semantic_index import SemanticIndex

        self.ctx = ctx
        self.index = SemanticIndex.load(ctx.work)
        self.backend = get_backend(ctx.cfg, quiet=True) if self.index else None
        self.mode = "semantic" if self.backend and self.index else "keyword"

    def free(self) -> None:
        if self.backend is not None:
            self.backend.free()

    def search(self, query: str, types: tuple[str, ...], exclude: set[str],
               prior: dict[str, float], k: int = 5) -> list[dict[str, Any]]:
        if self.backend is not None and self.index is not None:
            q = self.backend.embed_queries([query])[0]
            hits = self.index.search(q, types, k, exclude, prior)
            return [{"asset": h.asset_id, "score": h.score, "start": h.start, "end": h.end,
                     "caption": h.caption} for h in hits]
        toks = set(_norm_words(query)) - {"a", "the", "of", "in"}
        out = []
        for aid, a in self.ctx.assets.items():
            if a["asset_type"] not in types or aid in exclude:
                continue
            hay = set(_norm_words(aid + " " + a.get("description", "")))
            out.append({"asset": aid, "score": len(toks & hay) / (len(toks) or 1)
                        + prior.get(aid, 0.0), "start": None, "end": None, "caption": ""})
        out.sort(key=lambda d: -d["score"])
        return out[:k]


def _usage_prior(ctx: Ctx, beat: dict[str, Any], burned: dict[str, Any]) -> dict[str, float]:
    """Soft penalties so a replacement is not the picture already on screen nearby."""
    prior: dict[str, float] = {}
    for b in ctx.beats:
        if b is beat:
            continue
        near = abs(b["start_time"] - beat["start_time"]) < REUSE_WINDOW
        for k in ("source_id", "overlay_asset", "pip_asset"):
            if b.get(k):
                prior[b[k]] = prior.get(b[k], 0.0) - (0.06 if near else 0.015)
    for aid in burned:
        prior[aid] = prior.get(aid, 0.0) - 0.25
    return prior


# --------------------------------------------------------------------------- applying actions


@dataclass
class Change:
    beat: int
    field: str
    before: Any
    after: Any
    note: str = ""


@dataclass
class Proposal:
    instruction: str
    actions: list[dict[str, Any]]
    plan: dict[str, Any]
    changes: list[Change]
    notes: list[str]
    side_effects: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    explanation: str = ""

    @property
    def changed_beats(self) -> list[int]:
        return sorted({c.beat for c in self.changes})


def _asset_ok(ctx: Ctx, aid: str, types: tuple[str, ...]) -> dict[str, Any]:
    a = ctx.assets.get(aid)
    if a is None:
        raise FixError(f"Asset '{aid}' is not in the local index (stock is never picked here)")
    if a["asset_type"] not in types:
        raise FixError(f"'{aid}' is a {a['asset_type']}; this slot needs {'/'.join(types)}")
    return a


def _window(ctx: Ctx, a: dict[str, Any], beat: dict[str, Any], hit: dict[str, Any] | None,
            ) -> tuple[float | None, float | None]:
    """Trim [media_start, media_end] for a clip: around the retrieval hit, honouring sidecar
    hints (media_start/media_end/avoid/segments)."""
    if a["asset_type"] != "clip":
        return None, None
    from src.semantic_index import Hit, fit_window
    from src.semantic_planner import apply_clip_hints, sidecar_clip_hints

    dur = a.get("duration") or 0.0
    length = beat["end_time"] - beat["start_time"] + 0.3
    if hit and hit.get("start") is not None:
        st, en = fit_window(Hit(hit["asset"], hit["score"], hit["start"], hit["end"]),
                            length - 0.3, dur)
    else:
        st, en = 0.0, min(length, dur) if dur else None
    if st is None:
        st, en = 0.0, min(length, dur) if dur else None
    try:
        st, en = apply_clip_hints(st or 0.0, (en or st or 0.0) - (st or 0.0) or length, dur,
                                  sidecar_clip_hints(a))
    except (KeyError, TypeError, ValueError):
        st, en = st or 0.0, en or (st or 0.0) + length
    return round(st, 3), round(en, 3)


def _set(ctx: Ctx, changes: list[Change], oid: int, b: dict[str, Any], key: str, val: Any,
         note: str = "") -> None:
    if b.get(key) != val:
        changes.append(Change(oid, key, b.get(key), val, note))
        b[key] = val


def _swap_asset(ctx: Ctx, searcher: Any, changes: list[Change], oid: int, b: dict[str, Any],
                act: dict[str, Any], notes: list[str], banned: set[str]) -> str:
    """replace_asset on one beat; returns the new asset id."""
    slot = act.get("slot") or "main"
    key = {"main": "source_id", "overlay": "overlay_asset", "pip": "pip_asset"}[slot]
    types = {"main": ("clip", "image"), "overlay": ("schematic", "image"),
             "pip": ("clip", "image")}[slot]
    if slot == "main" and b.get("visual_template") == "fullscreen_clip" and \
            ctx.assets.get(b.get("source_id", ""), {}).get("asset_type") == "schematic":
        types = ("schematic", "clip", "image")
    old = b.get(key)
    bad = set(banned) | ({old} if old else set()) | excluded_ids(ctx.work, list(ctx.assets.values()))
    hit: dict[str, Any] | None = None
    why = ""
    if isinstance(act.get("asset"), str) and re.match(r"\s*query\s*[=:]", act["asset"], re.IGNORECASE):
        act = {**act, "query": act.get("query") or re.sub(r"^\s*query\s*[=:]\s*", "", act["asset"], flags=re.IGNORECASE),
               "asset": None}
    if act.get("query"):
        act = {**act, "query": re.sub(r"^\s*query\s*[=:]\s*", "", act["query"], flags=re.IGNORECASE)}
    if act.get("asset"):
        try:
            aid = resolve_asset(ctx, act["asset"], b)
        except FixError:
            if searcher is None:
                raise
            # the LLM named something that is not in the library: treat it as a description
            act = {**act, "asset": None, "query": act.get("query") or
                   re.sub(r"[_\-]+", " ", str(act["asset"]))}
            notes.append(f"asset '{act['query']}' is not in the library: searched by description")
            return _swap_asset(ctx, searcher, changes, oid, b, act, notes, banned)
        a = _asset_ok(ctx, aid, ("clip", "image", "schematic"))
        if aid in bad - {old}:
            raise FixError(f"'{aid}' is excluded/banned")
        query = act.get("query") or b.get("visual_query") or ctx.speech(b)
        if searcher is not None and a["asset_type"] == "clip":
            hits = searcher.search(query, ("clip",), set(ctx.assets) - {aid}, {}, 1)
            hit = hits[0] if hits else None
        why = "named asset"
    else:
        query = act.get("query") or b.get("visual_query") or ctx.speech(b)
        if not query:
            raise FixError(f"Beat {oid}: replace_asset needs an asset or a query")
        if searcher is None:
            raise FixError("No retrieval backend for a query-based replacement")
        prior = _usage_prior(ctx, b, {k: v for k, v in load_burned(ctx.work).items()
                                      if v.get("text")})
        old_type = ctx.assets.get(old or "", {}).get("asset_type")
        for aid2, a2 in ctx.assets.items():  # keep the kind of media the beat already had
            if old_type and a2["asset_type"] == old_type:
                prior[aid2] = prior.get(aid2, 0.0) + 0.03
        hits = searcher.search(query, types, bad, prior, 5)
        if not hits:
            raise FixError(f"Beat {oid}: nothing in the library matches '{query}'")
        hit = hits[0]
        aid = hit["asset"]
        a = ctx.assets[aid]
        alts = ", ".join(f"{h['asset']} ({h['score']:.2f})" for h in hits[1:3])
        why = f"query '{query}' -> {hit['score']:.2f}" + (f"; next: {alts}" if alts else "")
    if aid == old:
        raise FixError(f"Beat {oid} already shows '{aid}'")
    for k, v in (("primary_source", "local"), ("media_path", None), ("stock_asset_id", None),
                 ("license_type", None), ("stock_fallback_of", None)):
        if b.get(k) != v and slot == "main":
            b[k] = v
    _set(ctx, changes, oid, b, key, aid, why)
    if slot == "main":
        st, en = _window(ctx, a, b, hit)
        _set(ctx, changes, oid, b, "media_start", st)
        _set(ctx, changes, oid, b, "media_end", en)
        if b.get("crop_region") is not None:
            _set(ctx, changes, oid, b, "crop_region", None)
        if b.get("highlights"):
            _set(ctx, changes, oid, b, "highlights", None)
        if hit:
            b["match_score"] = round(float(hit["score"]), 3)
        if act.get("query"):
            b["visual_query"] = act["query"]
        b["beat_reason"] = (str(b.get("beat_reason") or "") + " | fix: replaced").strip(" |")
    return aid


def _zoom_region(focus: str, zoom: float) -> list[float] | None:
    if zoom <= 1.0:
        return None
    z = min(zoom, 5.0)
    w = h = round(1.0 / z, 4)
    fx, fy = FOCUS[focus]
    return [round((1 - w) * fx, 4), round((1 - h) * fy, 4), w, h]


def apply_actions(ctx: Ctx, actions: list[dict[str, Any]], searcher: Any, instruction: str = "",
                  explanation: str = "") -> Proposal:
    """Validate and apply ``actions`` to a COPY of the plan; nothing is written."""
    work_ctx = copy.copy(ctx)
    work_ctx.plan = copy.deepcopy(ctx.plan)
    c = work_ctx
    orig_id = {id(b): b["id"] for b in c.beats}
    # resolve every beat reference up front (ids shift after a split/merge)
    notes: list[str] = []
    resolved: list[list[dict[str, Any]]] = []
    for act in actions:
        if act.get("type") not in ACTIONS:
            raise FixError(f"Unknown action type '{act.get('type')}'")
        if act["type"] == "exclude_asset" and not act.get("beat"):
            resolved.append([])
        else:
            resolved.append(locate_beats(c, act.get("beat"), notes))
    changes: list[Change] = []
    side: list[dict[str, Any]] = []
    banned: set[str] = set()
    targets: list[dict[str, Any]] = []
    structural = False
    for act, beats in zip(actions, resolved, strict=True):
        t = act["type"]
        if t == "exclude_asset":
            aid = resolve_asset(c, act.get("asset") or "@current", beats[0] if beats else None)
            banned.add(aid)
            a = c.assets.get(aid)
            side.append({"op": "exclude", "id": aid, "file": a["filename"] if a else None,
                         "path": a["filepath"] if a else None})
            users = [b for b in c.beats if aid in (b.get("source_id"), b.get("overlay_asset"),
                                                   b.get("pip_asset"))]
            for b in users:
                slot = "main" if b.get("source_id") == aid else \
                    "overlay" if b.get("overlay_asset") == aid else "pip"
                _swap_asset(c, searcher, changes, orig_id[id(b)], b,
                            {"slot": slot, "query": act.get("query")}, notes, banned)
                targets.append({"beat": orig_id[id(b)], "asset": aid})
            if not users:
                notes.append(f"'{aid}' is not used by any beat; it is only excluded")
            continue
        for b in beats:
            oid = orig_id.get(id(b), b["id"])
            dur = b["end_time"] - b["start_time"]
            if t == "replace_asset":
                old = b.get({"main": "source_id", "overlay": "overlay_asset",
                             "pip": "pip_asset"}[act.get("slot") or "main"])
                new = _swap_asset(c, searcher, changes, oid, b, act, notes, banned)
                targets.append({"beat": oid, "asset": old, "new": new})
            elif t == "change_template":
                tpl = act.get("template")
                if tpl not in TEMPLATES:
                    raise FixError(f"Unknown template '{tpl}'")
                if tpl == "split_schematic" and not b.get("overlay_asset"):
                    if searcher is None:
                        raise FixError("split_schematic needs a diagram (no retrieval backend)")
                    hits = searcher.search(b.get("visual_query") or c.speech(b), ("schematic",),
                                           excluded_ids(c.work), {}, 1)
                    if not hits:
                        raise FixError(f"Beat {oid}: no diagram available for a split layout")
                    _set(c, changes, oid, b, "overlay_asset", hits[0]["asset"], "diagram for the split")
                if tpl in ("fullscreen_clip", "punch_in_emphasis") and b.get("overlay_asset") \
                        and b.get("visual_template") in ("split_schematic", "cutout_callout"):
                    _set(c, changes, oid, b, "overlay_asset", None)
                _set(c, changes, oid, b, "visual_template", tpl)
            elif t in ("set_crop", "set_zoom"):
                region = act.get("region")
                if not region and (act.get("focus") or act.get("zoom") is not None):
                    region = _zoom_region(act.get("focus") or "center",
                                          float(act.get("zoom") or 1.0))
                    if region is None:
                        _set(c, changes, oid, b, "crop_region", None, "crop removed")
                        continue
                if not region or len(region) != 4:
                    raise FixError(f"Beat {oid}: give a focus+zoom or a [x,y,w,h] region")
                x, y, w, h = (float(v) for v in region)
                if not (0 <= x < 1 and 0 <= y < 1 and 0.2 <= w <= 1 and 0.2 <= h <= 1
                        and x + w <= 1.001 and y + h <= 1.001):
                    raise FixError(f"Beat {oid}: crop region {region} is outside the picture "
                                   "(min size 20%)")
                _set(c, changes, oid, b, "crop_region", [round(x, 4), round(y, 4), round(w, 4),
                                                         round(h, 4)])
            elif t == "set_trim":
                a = c.assets.get(b.get("source_id", ""))
                if not a or a["asset_type"] != "clip":
                    raise FixError(f"Beat {oid}: trim needs a clip as main media")
                cd = float(a.get("duration") or 0)
                st = float(act["start"]) if act.get("start") is not None else b.get("media_start") or 0.0
                en = float(act["end"]) if act.get("end") is not None else None
                if st < 0 or (cd and st >= cd - 0.2):
                    raise FixError(f"Beat {oid}: start {st}s is outside the {cd:.1f}s clip")
                if en is not None and (en <= st + 0.2 or (cd and en > cd + 0.05)):
                    raise FixError(f"Beat {oid}: end {en}s must be after {st}s and within {cd:.1f}s")
                if cd and (en or st + dur) - st < dur - 0.05 and (en or cd) - st < dur:
                    notes.append(f"beat {oid}: only {(en or cd) - st:.1f}s of footage for a "
                                 f"{dur:.1f}s beat (the last frame will hold)")
                _set(c, changes, oid, b, "media_start", round(st, 3))
                _set(c, changes, oid, b, "media_end", round(en, 3) if en is not None else None)
            elif t == "set_callout":
                text = (act.get("text") or "").strip()
                if not text:
                    raise FixError(f"Beat {oid}: set_callout needs text")
                _set(c, changes, oid, b, "callout_text", " ".join(text.split()[:5])[:40])
            elif t == "remove_callout":
                _set(c, changes, oid, b, "callout_text", None)
            elif t == "set_transition":
                from src.transitions import normalize_name

                name = normalize_name(str(act.get("transition") or ""))
                if name is None:
                    from src.transitions import NAMES
                    raise FixError(f"Unknown transition '{act.get('transition')}' "
                                   f"(use: {', '.join(NAMES)})")
                _set(c, changes, oid, b, "transition", name)
                if act.get("duration") is not None:
                    _set(c, changes, oid, b, "transition_duration",
                         min(max(float(act["duration"]), 0.1), 1.5))
            elif t == "move_overlay":
                if b.get("overlay_mode") != "pip":
                    raise FixError(f"Beat {oid} has no picture-in-picture to move")
                if act.get("corner") not in CORNERS:
                    raise FixError("corner must be tl, tr, bl or br")
                _set(c, changes, oid, b, "pip_corner", act["corner"])
            elif t == "remove_highlight":
                if not b.get("highlights"):
                    raise FixError(f"Beat {oid} has no highlight")
                _set(c, changes, oid, b, "highlights", None)
            elif t == "split_beat":
                at = ref_time(c, act["at"]) if act.get("at") else (b["start_time"] + b["end_time"]) / 2
                at = round(at, 3)
                if not (b["start_time"] + MIN_PIECE <= at <= b["end_time"] - MIN_PIECE):
                    raise FixError(f"Beat {oid}: split at {fmt_time(c.video_time(at))} leaves a "
                                   f"piece shorter than {MIN_PIECE}s")
                nb = copy.deepcopy(b)
                nb.update({"start_time": at, "callout_text": None, "emphasis": False,
                           "transition": "none", "transition_duration": None,
                           "lower_third_path": None, "badge_path": None, "cutout_path": None,
                           "highlights": None})
                if b.get("media_start") is not None and \
                        c.assets.get(b.get("source_id", ""), {}).get("asset_type") == "clip":
                    nb["media_start"] = round(float(b["media_start"]) + (at - b["start_time"]), 3)
                    b["media_end"] = None
                nb["duration"] = round(nb["end_time"] - at, 3)
                changes.append(Change(oid, "split", f"{fmt_time(c.video_time(b['start_time']))}-"
                                      f"{fmt_time(c.video_time(b['end_time']))}",
                                      f"two beats at {fmt_time(c.video_time(at))}"))
                b["end_time"] = at
                b["duration"] = round(at - b["start_time"], 3)
                c.beats.insert(c.beats.index(b) + 1, nb)
                structural = True
            elif t == "merge_beats":
                i = c.beats.index(b)
                j = i + 1 if act.get("with_next", True) else i - 1
                if not 0 <= j < len(c.beats):
                    raise FixError(f"Beat {oid} has no {'next' if j > i else 'previous'} beat")
                keep, drop = (b, c.beats[j]) if j > i else (c.beats[j], b)
                changes.append(Change(oid, "merge", "2 beats",
                                      f"1 beat ({fmt_time(c.video_time(keep['start_time']))}-"
                                      f"{fmt_time(c.video_time(drop['end_time']))})"))
                keep["end_time"] = drop["end_time"]
                keep["duration"] = round(keep["end_time"] - keep["start_time"], 3)
                keep["media_end"] = None
                c.beats.remove(drop)
                structural = True
    if structural:
        for bb in c.beats:
            for k in ("lower_third_path", "badge_path", "cutout_path"):
                bb[k] = None
    if not changes and not side:
        raise FixError("The instruction produced no change")
    return Proposal(instruction, actions, c.plan, changes, notes, side, targets, explanation)


# --------------------------------------------------------------------------- LLM


def ollama_check(cfg: dict[str, Any], model: str) -> None:
    base = str((cfg.get("llm") or {}).get("base_url", "http://localhost:11434/v1"))
    root = re.sub(r"/v1/?$", "", base.rstrip("/"))
    try:
        r = httpx.get(f"{root}/api/tags", timeout=3.0)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError) as e:
        raise FixError(f"Ollama is not reachable at {root} ({type(e).__name__}). Fix mode runs "
                       "100% locally: start it with `ollama serve`.") from e
    if model not in names and f"{model}:latest" not in names:
        raise FixError(f"Ollama model '{model}' is not pulled (`ollama pull {model}`).")


def default_chat(ctx: Ctx) -> Callable[[list[dict[str, str]], dict[str, Any]], str]:
    from src.stage3_plan_scenes import llm_chat

    llm = dict(ctx.cfg.get("llm") or {})
    model = str(llm.get("model", "qwen2.5:14b-instruct"))
    ollama_check(ctx.cfg, model)
    llm["num_ctx"] = max(int(llm.get("num_ctx", 4096)), 8192)
    base = str(llm.get("base_url", "http://localhost:11434/v1"))

    def chat(messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        with httpx.Client(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
            return llm_chat(client, base, llm, model, messages, schema, max_tokens=1500,
                            temperature=0.0, deadline=None, request_timeout=180.0)

    return chat


def beat_table(ctx: Ctx) -> str:
    rows = []
    for b in ctx.beats:
        sp = ctx.speech(b).split()
        sp_s = " ".join(sp[:9]) + (" ..." if len(sp) > 9 else "")
        asset = b.get("source_id") or b.get("stock_asset_id") or ""
        ov = f" + {b['overlay_asset']}" if b.get("overlay_asset") else ""
        rows.append(f"{b['id']:>3} | {fmt_time(ctx.video_time(b['start_time']))} | "
                    f"{b['visual_template'][:10]} | {asset}{ov} | \"{sp_s}\"")
    return "\n".join(rows)


def interpret(ctx: Ctx, instruction: str, chat: Callable[..., str],
              history: list[str] | None = None) -> tuple[list[dict[str, Any]], str]:
    prev = ""
    if ctx.last_targets:
        prev = ("Previous fix touched: " + "; ".join(
            f"beat {t.get('beat')} (asset {t.get('asset')})" for t in ctx.last_targets[:4]) + "\n")
    hist = ("Earlier in this session: " + " / ".join(history[-4:]) + "\n") if history else ""
    user = (f"Final video time = narration time + {ctx.intro:.2f}s (intro card).\n{prev}{hist}"
            f"Beats (id | final-video time | template | asset | narration):\n{beat_table(ctx)}\n\n"
            f"Instruction: {instruction}")
    raw = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
               ACTION_SCHEMA)
    try:
        data = json.loads(raw)
        acts = [a for a in data.get("actions", []) if isinstance(a, dict)]
    except (json.JSONDecodeError, AttributeError) as e:
        raise FixError(f"The local LLM returned invalid JSON ({e})") from e
    if not acts:
        raise FixError("The LLM did not find anything to change. Rephrase with a time, "
                       "a quote or a beat id.")
    return acts, str(data.get("explanation", ""))


# --------------------------------------------------------------------------- diff / write / undo


def show_diff(ctx: Ctx, p: Proposal) -> None:
    t = Table(title="Proposed changes", show_lines=True)
    for col in ("Beat", "Video", "Speech", "Change"):
        t.add_column(col, overflow="fold")
    by_beat: dict[int, list[Change]] = {}
    for ch in p.changes:
        by_beat.setdefault(ch.beat, []).append(ch)
    byid = {b["id"]: b for b in ctx.beats}
    for bid, chs in sorted(by_beat.items()):
        b = byid.get(bid)
        sp = " ".join(ctx.speech(b).split()[:14]) if b else ""
        when = fmt_time(ctx.video_time(b["start_time"])) if b else "?"
        lines = [f"{c.field}: {_short(c.before)} -> {_short(c.after)}" + (f"  [dim]({c.note})[/]"
                                                                          if c.note else "")
                 for c in chs]
        t.add_row(str(bid), when, f'"{sp}"', "\n".join(lines))
    console.print(t)
    for s in p.side_effects:
        console.print(f"  [yellow]exclude project-wide:[/] {s['id']} "
                      f"(work/{EXCLUDED_FILE}, sidecar `exclude: true`)")
    for n in p.notes:
        console.print(f"  [dim]note: {n}[/]")


def _short(v: Any) -> str:
    s = "-" if v is None else str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _undo_dir(work: Path) -> Path:
    d = work / "fix_undo"
    d.mkdir(exist_ok=True)
    return d


def mark_hand_edited(work: Path) -> None:
    sp = work / ".stage_state.json"
    try:
        state = read_json(sp) if sp.is_file() else {}
    except json.JSONDecodeError:
        state = {}
    state["hand_edited"] = True
    write_json(sp, state)


def write_proposal(ctx: Ctx, p: Proposal) -> int:
    """Back up, write the plan + side effects, log. Returns the undo number."""
    from src.schemas import ScenePlan

    plan_path = ctx.work / "scene_plan.json"
    new = ScenePlan.model_validate(p.plan).model_dump(mode="json")  # renumbers ids, validates
    n = len(list(_undo_dir(ctx.work).glob("*.json"))) + 1
    shutil.copy2(plan_path, ctx.work / "scene_plan.prev.json")
    undo: dict[str, Any] = {"plan": ctx.plan, "excluded_added": [], "sidecars": []}
    for s in p.side_effects:
        if s["op"] == "exclude":
            ep = ctx.work / EXCLUDED_FILE
            cur = read_json(ep) if ep.is_file() else []
            if not any(e.get("id") == s["id"] for e in cur):
                cur.append({"id": s["id"], "file": s.get("file"), "reason": p.instruction,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
                write_json(ep, cur)
                undo["excluded_added"].append(s["id"])
            if s.get("path") and Path(s["path"]).is_file():
                sc = Path(str(s["path"]) + ".txt")
                text = sc.read_text(encoding="utf-8") if sc.is_file() else ""
                if not re.search(r"^\s*exclude\s*[:=]", text, re.IGNORECASE | re.MULTILINE):
                    sc.write_text(text.rstrip("\n") + ("\n" if text else "") + "exclude: true\n",
                                  encoding="utf-8")
                    undo["sidecars"].append({"path": str(sc), "existed": sc.is_file() and bool(text),
                                             "text": text})
    write_json(_undo_dir(ctx.work) / f"{n:04d}.json", undo)
    write_json(plan_path, new)
    mark_hand_edited(ctx.work)
    with (ctx.work / "fix_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "instruction": p.instruction,
            "actions": p.actions, "undo": n, "targets": p.targets,
            "changes": [{"beat": c.beat, "field": c.field, "before": _short(c.before),
                         "after": _short(c.after)} for c in p.changes]},
            ensure_ascii=False, default=str) + "\n")
    ctx.plan = new
    ctx.last_targets = p.targets
    return n


def undo_last(ctx: Ctx) -> bool:
    ud = _undo_dir(ctx.work)
    files = sorted(ud.glob("*.json"))
    if not files:
        return False
    data = read_json(files[-1])
    for sc in data.get("sidecars", []):
        pth = Path(sc["path"])
        if sc.get("text"):
            pth.write_text(sc["text"], encoding="utf-8")
        else:
            pth.unlink(missing_ok=True)
    if data.get("excluded_added"):
        ep = ctx.work / EXCLUDED_FILE
        cur = [e for e in (read_json(ep) if ep.is_file() else [])
               if e.get("id") not in data["excluded_added"]]
        write_json(ep, cur)
    write_json(ctx.work / "scene_plan.json", data["plan"])
    files[-1].unlink()
    ctx.plan = data["plan"]
    ctx.last_targets = []
    with (ctx.work / "fix_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "undo_of": files[-1].name,
                             "targets": []}) + "\n")
    return True


# --------------------------------------------------------------------------- frames / preview


def _grab(src: Path, t: float, out: Path) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(t, 0):.2f}", "-i", str(src),
                        "-frames:v", "1", "-vf", "scale=480:-2", str(out)],
                       capture_output=True, timeout=60, check=False)
    return r.returncode == 0 and out.is_file()


def grab_frames(ctx: Ctx, video: Path | None, plan: dict[str, Any], beat_ids: list[int],
                tag: str) -> dict[int, Path]:
    """One frame per beat (middle) from the rendered video, else from the beat's asset."""
    out: dict[int, Path] = {}
    d = ctx.work / "fix_frames"
    d.mkdir(exist_ok=True)
    for b in plan["beats"]:
        if b["id"] not in beat_ids:
            continue
        mid = (b["start_time"] + b["end_time"]) / 2
        f = d / f"{tag}_{b['id']:03d}.png"
        if video and video.is_file() and _grab(video, mid + ctx.intro, f):
            out[b["id"]] = f
            continue
        a = ctx.assets.get(b.get("source_id", ""))
        if a and _grab(Path(a["filepath"]), (b.get("media_start") or 0) + 0.5, f):
            out[b["id"]] = f
    return out


def contact_sheet(ctx: Ctx, before: dict[int, Path], after: dict[int, Path],
                  rows: list[tuple[int, str]], out: Path) -> Path | None:
    from PIL import Image, ImageDraw

    rows = [r for r in rows if r[0] in before or r[0] in after][:8]
    if not rows:
        return None
    w, h, pad, head = 480, 270, 8, 34
    img = Image.new("RGB", (2 * w + 3 * pad, len(rows) * (h + head + pad) + pad), (16, 18, 26))
    d = ImageDraw.Draw(img)
    for i, (bid, label) in enumerate(rows):
        y = pad + i * (h + head + pad)
        d.text((pad, y + 4), f"beat {bid}  |  {label}"[:110], fill=(230, 230, 240))
        for j, src in enumerate((before.get(bid), after.get(bid))):
            x = pad + j * (w + pad)
            d.text((x, y + 18), "BEFORE" if j == 0 else "AFTER", fill=(120, 200, 255))
            if src and src.is_file():
                with Image.open(src) as im:
                    im = im.convert("RGB").resize((w, h))
                    img.paste(im, (x, y + head))
    img.save(out)
    return out


# --------------------------------------------------------------------------- render


def default_render(episode_id: str) -> Path:
    """Re-render through the normal CLI path; unchanged beats come from work/render_cache."""
    import os

    from src import cli

    prev = os.environ.get("VIDEOMAKER_OFFLINE")
    os.environ["VIDEOMAKER_OFFLINE"] = "1"   # fix mode never touches the network
    try:
        cli.render(episode_id, draft=False)
    finally:
        if prev is None:
            os.environ.pop("VIDEOMAKER_OFFLINE", None)
        else:
            os.environ["VIDEOMAKER_OFFLINE"] = prev
    return episode_dir(episode_id) / "out" / "final_cut.mp4"


# --------------------------------------------------------------------------- session / REPL


class Session:
    """State of a fix run: the plan at session start, applied fixes, the pending render."""

    def __init__(self, episode_id: str, chat: Callable[..., str] | None = None,
                 searcher: Any = None, renderer: Callable[[str], Path] | None = None,
                 confirm: Callable[[str], bool] | None = None, render: bool = True) -> None:
        self.ep_id = episode_id
        self.ctx = load_ctx(episode_id)
        self.chat = chat or None
        self._searcher = searcher
        self.renderer = renderer or default_render
        self.confirm = confirm or (lambda q: input(q + " [y/N] ").strip().lower() in ("y", "yes", "s", "sim"))
        self.do_render = render
        self.start_plan = copy.deepcopy(self.ctx.plan)
        self.history: list[str] = []
        self.applied = 0
        self.before_video = self._old_video()

    def _old_video(self) -> Path | None:
        v = self.ctx.ep / "out" / "final_cut.mp4"
        if not v.is_file():
            return None
        keep = self.ctx.work / "fix_before.mp4"
        shutil.copy2(v, keep)
        return keep

    @property
    def searcher(self) -> Any:
        if self._searcher is None:
            self._searcher = Searcher(self.ctx)
        return self._searcher

    def propose(self, instruction: str) -> Proposal:
        if self.chat is None:
            self.chat = default_chat(self.ctx)
        acts, expl = interpret(self.ctx, instruction, self.chat, self.history)
        return apply_actions(self.ctx, acts, self.searcher, instruction, expl)

    def apply(self, p: Proposal) -> int:
        n = write_proposal(self.ctx, p)
        self.history.append(p.instruction)
        self.applied += 1
        return n

    def changed_beats(self) -> list[int]:
        """Beat ids (in the current plan) that differ from the start of the session."""
        old = {(b["start_time"], b["end_time"]): b for b in self.start_plan["beats"]}
        out = []
        for b in self.ctx.beats:
            o = old.get((b["start_time"], b["end_time"]))
            keys = ("source_id", "overlay_asset", "visual_template", "crop_region", "media_start",
                    "media_end", "callout_text", "transition", "pip_corner", "highlights")
            if o is None or any(o.get(k) != b.get(k) for k in keys):
                out.append(b["id"])
        return out

    def finish(self) -> Path | None:
        """Re-render (cache: only changed beats), then the before/after sheet."""
        ids = self.changed_beats()
        if not ids:
            console.print("  Nothing changed: no render needed.")
            return None
        before = grab_frames(self.ctx, self.before_video, self.start_plan, ids, "before") \
            if self.before_video else {}
        video: Path | None = None
        if self.do_render:
            video = self.renderer(self.ep_id)
        after = grab_frames(self.ctx, video, self.ctx.plan, ids, "after")
        if not before:  # no earlier render: show the old asset instead
            before = grab_frames(self.ctx, None, self.start_plan, ids, "before")
        rows = []
        for b in self.ctx.beats:
            if b["id"] in ids:
                when = fmt_time(self.ctx.video_time(b["start_time"]))
                said = " ".join(self.ctx.speech(b).split()[:10])
                rows.append((b["id"], f'{when}  "{said}"'))
        sheet = contact_sheet(self.ctx, before, after, rows, self.ctx.work / "fix_preview.png")
        if sheet:
            console.print(f"  Before/after sheet: {sheet}")
        if video:
            console.print(f"  [green]Video:[/] {video}")
        self.start_plan = copy.deepcopy(self.ctx.plan)
        return video

    def close(self) -> None:
        if self._searcher is not None and hasattr(self._searcher, "free"):
            self._searcher.free()


def run_fix(episode_id: str, instruction: str, *, yes: bool = False, dry_run: bool = False,
            chat: Callable[..., str] | None = None, searcher: Any = None,
            renderer: Callable[[str], Path] | None = None,
            confirm: Callable[[str], bool] | None = None, render: bool = True) -> Proposal | None:
    s = Session(episode_id, chat, searcher, renderer, confirm, render)
    try:
        p = s.propose(instruction)
        if p.explanation:
            console.print(f"  [dim]{p.explanation}[/]")
        show_diff(s.ctx, p)
        if dry_run:
            console.print("  [yellow]--dry-run: nothing written.[/]")
            return p
        if not (yes or s.confirm("Apply?")):
            console.print("  Cancelled.")
            return None
        n = s.apply(p)
        console.print(f"  Plan updated (backup: work/scene_plan.prev.json, undo #{n}, "
                      "log: work/fix_log.jsonl).")
        s.finish()
        return p
    finally:
        s.close()


HELP = ("Type a correction (PT or EN). Commands: undo, render, show, help, quit. "
        "`render` re-renders only the changed beats.")


def interactive(episode_id: str, *, yes: bool = False, chat: Callable[..., str] | None = None,
                searcher: Any = None, renderer: Callable[[str], Path] | None = None,
                read: Callable[[str], str] = input, render: bool = True) -> None:
    s = Session(episode_id, chat, searcher, renderer, None, render)
    if yes:
        s.confirm = lambda q: True
    else:
        s.confirm = lambda q: read(q + " [y/N] ").strip().lower() in ("y", "yes", "s", "sim")
    console.print(HELP)
    try:
        while True:
            try:
                line = read("fix> ").strip()
            except EOFError:
                line = "quit"
            if not line:
                continue
            cmd = line.lower()
            if cmd in ("quit", "exit", "q"):
                if s.changed_beats() and s.confirm("Render the pending changes before leaving?"):
                    s.finish()
                break
            if cmd == "help":
                console.print(HELP)
            elif cmd == "undo":
                console.print("  Undone." if undo_last(s.ctx) else "  Nothing to undo.")
            elif cmd == "render":
                s.finish()
            elif cmd == "show":
                console.print(beat_table(s.ctx))
            else:
                try:
                    p = s.propose(line)
                    show_diff(s.ctx, p)
                    if s.confirm("Apply?"):
                        s.apply(p)
                        console.print("  Applied (type `undo` to revert, `render` when done).")
                except FixError as e:
                    console.print(f"  [red]{e}[/]")
    finally:
        s.close()


# --------------------------------------------------------------------------- review


_VLM_SCHEMA = {"type": "object", "required": ["matches", "burned_text"], "properties": {
    "matches": {"type": "integer"},        # 0-10: does the picture illustrate the narration
    "burned_text": {"type": "string"},     # any caption/watermark/date/logo text, "" if none
    "text_kind": {"type": "string", "enum": ["none", "caption", "watermark", "date", "logo",
                                             "diagram_label"]},
    "text_band": {"type": "string", "enum": ["none", "top", "bottom", "middle"]},
    "reason": {"type": "string"}}}
YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")


def _vlm_cfg(ctx: Ctx) -> tuple[str, str]:
    ol = ((ctx.cfg.get("semantic") or {}).get("ollama") or {})
    base = str(ol.get("base_url", "http://localhost:11434")).removesuffix("/v1")
    return base, str(ol.get("caption_model", "qwen2.5vl:3b"))


def vlm_judge(ctx: Ctx, frame: Path, narration: str) -> dict[str, Any] | None:
    from PIL import Image

    from src.locate import _b64, _vlm

    base, model = _vlm_cfg(ctx)
    with Image.open(frame) as im:
        b64, _ = _b64(im, 896)
    return _vlm(base, model, b64,
                f"Narration: \"{narration}\"\nRate 0-10 (matches) how well this picture illustrates "
                "the narration. Transcribe any text burned into the picture itself (subtitle, "
                "caption, watermark, date, logo) in burned_text (empty if none) and say where it "
                "is (text_band). Be literal; do not invent text.", _VLM_SCHEMA)


def asset_frame(ctx: Ctx, b: dict[str, Any], tmp: Path) -> Path | None:
    a = ctx.assets.get(b.get("source_id", ""))
    if not a:
        return None
    out = tmp / f"rv_{b['id']:03d}.png"
    t = (b.get("media_start") or 0.0) + 0.5 * (b["end_time"] - b["start_time"])
    if a["asset_type"] == "clip":
        return out if _grab(Path(a["filepath"]), t, out) else None
    return Path(a.get("upscaled_path") or a["filepath"])


@dataclass
class Suspect:
    beat: int
    reasons: list[str]
    action: dict[str, Any] | None = None


def highlight_problem(b: dict[str, Any]) -> str | None:
    """Why a beat's highlight is wrong (covers the title/caption, unlocated, huge) or None."""
    from src.locate import CAPTION_BAND, MAX_AREA, TITLE_BAND
    from src.stage6_render import highlight_trusted

    for h in b.get("highlights") or []:
        ok, why = highlight_trusted(h)
        if not ok:
            return f"highlight without a real location ({why}); it can cover the title/text"
        try:
            from src.overlays import REGIONS

            x, y, w, hh = REGIONS[h["region"]] if h.get("region") in REGIONS else (
                float(h.get("x", 0)), float(h.get("y", 0)), float(h.get("w", 0)),
                float(h.get("h", 0)))
        except (KeyError, TypeError, ValueError):
            continue
        if max(x, y, w, hh) <= 1.0 and h.get("source") != "sidecar":
            cy = y + hh / 2
            if cy < TITLE_BAND or cy > CAPTION_BAND or w * hh > MAX_AREA:
                return "highlight covers the title/caption band or too much of the picture"
    return None


def scan_burned_text(ctx: Ctx, asset_ids: list[str], log: Callable[[str], None] = console.print
                     ) -> dict[str, Any]:
    """VLM scan for burned-in text on clips/images; results cached in work/burned_text.json and
    flagged as ``burned_text`` on the assets in local_assets.json (planner avoids them)."""
    import tempfile

    store = load_burned(ctx.work)
    tmp = Path(tempfile.mkdtemp(prefix="fixscan_"))
    try:
        for aid in asset_ids:
            a = ctx.assets.get(aid)
            if not a or a["asset_type"] == "schematic":
                continue
            p = Path(a["filepath"])
            sig = f"{p.stat().st_size}:{int(p.stat().st_mtime)}" if p.is_file() else ""
            if aid in store and store[aid].get("sig") == sig:
                continue
            texts, bands = [], []
            dur = a.get("duration") or 0
            times = [0.5] if a["asset_type"] == "image" else [dur * f for f in (0.15, 0.5, 0.85)]
            for k, t in enumerate(times):
                fr = tmp / f"{aid[:40]}_{k}.png"
                src = p if a["asset_type"] == "image" else None
                if src is None and not _grab(p, t, fr):
                    continue
                r = vlm_judge(ctx, src or fr, "(describe only)")
                if r and r.get("burned_text", "").strip():
                    texts.append(r["burned_text"].strip())
                    bands.append(r.get("text_band", "none"))
            entry: dict[str, Any] = {"sig": sig}
            if texts:
                band = max(set(bands), key=bands.count)
                entry.update({"text": texts[0][:120], "band": band})
                w, h = a.get("width") or 0, a.get("height") or 0
                if w and h and band == "bottom":
                    entry["crop_box"] = [0, 0, w, int(h * 0.86)]
                elif w and h and band == "top":
                    entry["crop_box"] = [0, int(h * 0.14), w, h - int(h * 0.14)]
                log(f"  burned text in {aid}: {entry['text']!r} ({band})")
            store[aid] = entry
        write_json(ctx.work / BURNED_FILE, store)
        ap = ctx.work / "local_assets.json"
        data = read_json(ap)
        for a in data.get("assets", []):
            e = store.get(a["id"])
            if e and e.get("text"):
                a["burned_text"] = {k: v for k, v in e.items() if k != "sig"}
                if e.get("crop_box") and not a.get("crop_box"):
                    a["crop_box"] = e["crop_box"]
        write_json(ap, data)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return store


def review(episode_id: str, *, apply: bool = False, yes: bool = False, scan_all: bool = False,
           vlm: Callable[[Ctx, Path, str], dict[str, Any] | None] | None = None,
           searcher: Any = None, renderer: Callable[[str], Path] | None = None,
           confirm: Callable[[str], bool] | None = None, render: bool = True
           ) -> list[Suspect]:
    """Judge every beat: does the picture fit the speech, is there burned-in text that
    contradicts it, does a highlight cover text? Proposes replacements; ``apply`` writes them."""
    import tempfile

    s = Session(episode_id, None, searcher, renderer, confirm, render)
    ctx = s.ctx
    judge = vlm or vlm_judge
    suspects: list[Suspect] = []
    tmp = Path(tempfile.mkdtemp(prefix="fixrev_"))
    try:
        if scan_all:
            scan_burned_text(ctx, [a for a in ctx.assets])
        scores: dict[int, float] = {}
        for b in ctx.beats:
            reasons: list[str] = []
            narr = ctx.speech(b)
            hp = highlight_problem(b)
            fr = asset_frame(ctx, b, tmp)
            r = judge(ctx, fr, narr) if fr else None
            burned = (r or {}).get("burned_text", "").strip()
            kind = (r or {}).get("text_kind", "none")
            if burned and kind not in ("none", "diagram_label") and \
                    ctx.assets.get(b.get("source_id", ""), {}).get("asset_type") != "schematic":
                years = set(YEAR.findall(burned))
                said = set(YEAR.findall(narr))
                if years - said:
                    reasons.append(f"burned-in text {burned!r} has a date/year ({', '.join(sorted(years))}) "
                                   f"the narration does not say")
                else:
                    reasons.append(f"burned-in {kind} text {burned!r} (subtitle/watermark on the picture)")
            if r is not None and int(r.get("matches", 10)) <= 3:
                reasons.append(f"VLM: picture does not illustrate the speech ({r.get('reason', '')[:80]})")
            if hp:
                reasons.append(hp)
            if reasons:
                suspects.append(Suspect(b["id"], reasons))
            scores[b["id"]] = float((r or {}).get("matches", 10))
        # SigLIP second opinion for the picture/speech mismatches (weak VLM alone is not enough)
        sr = s.searcher
        for sp in suspects:
            b = ctx.beats[sp.beat]
            aid = b.get("source_id")
            text_problem = any("burned-in" in x for x in sp.reasons)
            vlm_bad = any(x.startswith("VLM") for x in sp.reasons)
            only_hl = all("highlight" in x for x in sp.reasons)
            if only_hl:
                sp.action = {"type": "remove_highlight", "beat": {"by": "beat_id",
                                                                  "value": str(b["id"])}}
            elif text_problem or vlm_bad:
                q = b.get("visual_query") or ctx.speech(b)
                bad_ids = {aid} | excluded_ids(ctx.work)
                hits = sr.search(q, ("clip", "image"), bad_ids, {}, 3) if q else []
                if vlm_bad and not text_problem:
                    # the 3B VLM alone is noisy: demand a clearly better SigLIP candidate
                    cur = sr.search(q, ("clip", "image", "schematic"),
                                    set(ctx.assets) - {aid}, {}, 1) if q and aid else []
                    cs = cur[0]["score"] if cur else 0.0
                    if not hits or hits[0]["score"] - cs < 0.03:
                        sp.reasons = [x for x in sp.reasons if not x.startswith("VLM")]
                        if sp.reasons and highlight_problem(b):
                            sp.action = {"type": "remove_highlight",
                                         "beat": {"by": "beat_id", "value": str(b["id"])}}
                        continue
                    sp.reasons.append(f"better candidate: {hits[0]['asset']} "
                                      f"({hits[0]['score']:.2f} vs {cs:.2f})")
                if hits:
                    sp.action = {"type": "replace_asset",
                                 "beat": {"by": "beat_id", "value": str(b["id"])},
                                 "query": q, "asset": hits[0]["asset"]}
        suspects = [x for x in suspects if x.reasons]
        _print_review(ctx, suspects)
        write_json(ctx.work / "fix_review.json", [
            {"beat": x.beat, "reasons": x.reasons, "action": x.action} for x in suspects])
        acts = [x.action for x in suspects if x.action]
        if apply and acts:
            p = apply_actions(ctx, acts, sr, "review --apply")
            show_diff(ctx, p)
            if yes or s.confirm("Apply the proposed fixes?"):
                s.apply(p)
                s.finish()
        elif acts:
            console.print("  Re-run with --apply to apply the proposals (with confirmation).")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        s.close()
    return suspects


def _print_review(ctx: Ctx, suspects: list[Suspect]) -> None:
    if not suspects:
        console.print("  [green]No suspicious beats.[/]")
        return
    t = Table(title=f"{len(suspects)} suspicious beat(s)", show_lines=True)
    for col in ("Beat", "Video", "Speech", "Why", "Proposal"):
        t.add_column(col, overflow="fold")
    for sp in suspects:
        b = ctx.beats[sp.beat]
        prop = "-"
        if sp.action:
            prop = sp.action["type"] + (f" -> {sp.action['asset']}" if sp.action.get("asset") else "")
        t.add_row(str(sp.beat), fmt_time(ctx.video_time(b["start_time"])),
                  " ".join(ctx.speech(b).split()[:10]), "\n".join(sp.reasons), prop)
    console.print(t)
