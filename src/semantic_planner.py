"""Stage 3 semantic layer: narration -> visual queries -> retrieved local assets (with trim).

Runs after the base plan is built and repaired. For every beat it
1. asks the local LLM to rewrite the narration into 1-2 concrete English visual queries (+kind),
2. retrieves the top-k local assets (with a trim interval) from the semantic index,
3. lets the LLM choose among ONLY those k candidates (validated; deterministic fallback = top
   ranked with a diversity penalty),
and records ``visual_query``, ``match_score``, ``media_start/end`` on the beat. Without a
backend/index/LLM the plan is returned untouched (keyword planner result).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from rich.console import Console

from src.common import is_offline
from src.schemas import PrimarySource, ScenePlan, VisualTemplate
from src.semantic import SemanticBackend, get_backend, min_score_for, semantic_config
from src.semantic_index import Hit, SemanticIndex, fit_window

console = Console()
RECENT = 20.0  # default seconds before the same asset may be reused (semantic.reuse_window)
DEBUG: dict[str, Any] = {}  # last run's candidates/queries per beat (read by eval scripts)

QUERY_SYSTEM = (
    "You rewrite narration lines from a science/engineering YouTube video into concrete visual "
    "search queries for a b-roll library. Reply with JSON only.")

PICK_SYSTEM = (
    "You are a video editor choosing b-roll. For each beat pick the candidate that best "
    "illustrates the narration. Choose ONLY from the listed candidate ids. Reply with JSON only.")


class LLM:
    """Minimal chat client reusing the planner's endpoint/model discovery."""

    def __init__(self, config: dict[str, Any], role: str = "") -> None:
        from src import stage3_plan_scenes as s3

        self.ok = False
        self.fails = 0
        self.deadline: float | None = None
        self.config = config
        if os.environ.get("VIDEOMAKER_OFFLINE", "").lower() in {"1", "true", "yes", "on"}:
            return
        llm = config.get("llm", {})
        urls = [os.environ.get("LLM_BASE_URL") or llm.get("base_url", "")]
        urls = [u for u in urls if u] or ["http://localhost:11434/v1"]
        for u in urls:
            models = s3._list_models(u)
            if models:
                wanted = (llm.get(f"{role}_model") if role else None) \
                    or os.environ.get("LLM_MODEL") or llm.get("model", "qwen2.5:14b-instruct")
                model = s3.pick_model(wanted, models)
                if model:
                    self.base, self.model, self.ok = u.rstrip("/"), model, True
                    return

    def json(self, system: str, user: str, max_tokens: int = 1200,
             timeout: float | None = None, schema: dict[str, Any] | None = None) -> Any:
        """One small JSON chat call with its own timeout; None on failure. After three failures
        in a row the LLM is considered unusable for the rest of the run."""
        from src import stage3_plan_scenes as s3

        if not self.ok:
            return None
        llm = self.config.get("llm", {})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        per = float(timeout or llm.get("chunk_timeout", min(float(llm.get("timeout", 150)), 60)))
        if self.deadline is None:
            self.deadline = time.monotonic() + float(llm.get("semantic_budget", 420))
        try:
            with httpx.Client(timeout=httpx.Timeout(per, connect=5.0)) as c:
                content = s3.llm_chat(
                    c, self.base, llm, self.model, messages, schema, max_tokens=max_tokens,
                    temperature=0.2, deadline=min(self.deadline, time.monotonic() + per + 1),
                    request_timeout=per)
            data = s3._extract_json(content)
            self.fails = 0
            return data
        except (httpx.HTTPError, KeyError, ValueError) as e:
            self.fails += 1
            console.print(f"  [yellow]semantic LLM call failed: {type(e).__name__}[/]")
            if self.fails >= 3:
                console.print("  [yellow]LLM unresponsive; deterministic ranking for the rest[/]")
                self.ok = False
            return None


def _as_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return []
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


CHUNK = 5  # beats per LLM call: small prompts and answers keep a 7B model inside its timeout


def _chunks(items: dict[int, str], size: int = CHUNK) -> list[dict[int, str]]:
    keys = list(items)
    return [{k: items[k] for k in keys[i: i + size]} for i in range(0, len(keys), size)]


def _cache_key(*parts: Any) -> str:
    import hashlib
    import json

    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _cached(path: Path | None, key: str) -> Any:
    import json

    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get(key)
    except (OSError, ValueError, AttributeError):
        return None


def _store(path: Path | None, key: str, data: Any) -> None:
    import json

    if path is None:
        return
    try:
        blob = json.loads(path.read_text()) if path.is_file() else {}
        blob = blob if isinstance(blob, dict) else {}
        blob[key] = data
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(list(blob.items())[-12:])))  # keep the last entries
    except (OSError, ValueError):
        pass


def script_context(llm: LLM, texts: dict[int, str], cache: Path | None = None) -> str:
    """One call per episode: topic + the concrete things the video is about (cached). The
    per-line query rewrite is conditioned on it, so "the computer" becomes "Apollo guidance
    computer" instead of a generic laptop."""
    full = " ".join(texts.values())
    key = _cache_key("ctx-v1", getattr(llm, "model", "") if llm.ok else "", full)
    hit = _cached(cache, key)
    if isinstance(hit, str):
        return hit
    if not llm.ok or not full.strip():
        return ""
    schema = {"type": "object", "required": ["topic", "things"], "properties": {
        "topic": {"type": "string"},
        "things": {"type": "array", "items": {"type": "string"}, "maxItems": 12}}}
    data = llm.json("You summarise a video script for a b-roll editor. Reply with JSON only.",
                    'Script:\n"' + full[:5000] + '"\nReturn {"topic":"one sentence","things":'
                    '["the concrete physical things, places, machines, documents and diagrams '
                    'the script talks about, 2-4 words each"]}.', max_tokens=500, schema=schema)
    if not isinstance(data, dict):
        return ""
    things = [str(t).strip() for t in (data.get("things") or []) if str(t).strip()][:12]
    out = f"Video topic: {str(data.get('topic', '')).strip()} Things shown: {'; '.join(things)}."
    _store(cache, key, out)
    return out


def rewrite_queries(llm: LLM, texts: dict[int, str], context: str = "",
                    neighbours: bool = True, cache: Path | None = None
                    ) -> dict[int, dict[str, Any]]:
    """{beat_id: {"queries": [...], "kind": local|stock|schematic|cutout}} (may be partial).

    Chunked: a failed chunk is retried once for its missing lines only; the beats still missing
    simply use their narration as the query. ``context`` (script summary) and the neighbouring
    lines condition each rewrite; the whole result is cached per (model, texts, context)."""
    key = _cache_key("q-v2", getattr(llm, "model", "") if llm.ok else "", context, neighbours, texts)
    hit = _cached(cache, key)
    if isinstance(hit, dict) and hit:
        return {int(k): v for k, v in hit.items()}
    ids = list(texts)
    out: dict[int, dict[str, Any]] = {}
    for part in _chunks(texts):
        near = {}
        if neighbours:
            for i in part:
                n = ids.index(i)
                near[i] = (texts[ids[n - 1]] if n else "", texts[ids[n + 1]] if n + 1 < len(ids) else "")
        got = _rewrite_once(llm, part, context, near)
        missing = {i: t for i, t in part.items() if i not in got}
        if missing and llm.ok:
            got.update(_rewrite_once(llm, missing, context, near))
        out.update(got)
    if out and len(out) == len(texts):
        _store(cache, key, out)
    return out


def _rewrite_once(llm: LLM, texts: dict[int, str], context: str = "",
                  near: dict[int, tuple[str, str]] | None = None) -> dict[int, dict[str, Any]]:
    near = near or {}
    lines = "\n".join(
        f'{i}: "{t}"' + (f' (before: "{near[i][0][:90]}" | after: "{near[i][1][:90]}")'
                         if i in near and any(near[i]) else "") for i, t in texts.items())
    user = (
        (context + "\n" if context else "") +
        "For each numbered narration line write 1-2 short, concrete English visual queries "
        "(objects, setting, action; no abstract words, no names of people) that describe what "
        "the camera should show. Also give kind: 'local' (a machine/workshop/electronics shot), "
        "'schematic' (the line explains an abstract idea such as a control loop, signal, "
        "latency, algorithm: a diagram fits), 'stock' (a real-world place or generic scene) or "
        "'cutout' (a single object to call out). Describe only what THIS line says or clearly "
        "implies (use the video topic to resolve pronouns like 'it'/'the computer'), never "
        "carry over objects from other lines; abstract lines (latency, loop, control, "
        "feedback, algorithm) are 'schematic'.\n"
        'Return {"beats":[{"id":0,"queries":["..."],"kind":"local"}]}.\n' + lines)
    out: dict[int, dict[str, Any]] = {}
    ids = list(texts)
    schema = {"type": "object", "required": ["beats"], "properties": {"beats": {
        "type": "array", "minItems": len(ids), "maxItems": len(ids), "items": {
            "type": "object", "required": ["id", "queries", "kind"], "properties": {
                "id": {"type": "integer", "enum": ids},
                "queries": {"type": "array", "items": {"type": "string"},
                            "minItems": 1, "maxItems": 2},
                "kind": {"type": "string",
                         "enum": ["local", "stock", "schematic", "cutout"]}}}}}}
    for row in _as_list(llm.json(QUERY_SYSTEM, user, schema=schema)):
        try:
            bid = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        qs = row.get("queries") or row.get("query") or []
        qs = [qs] if isinstance(qs, str) else [str(q).strip() for q in qs if str(q).strip()]
        kind = str(row.get("kind", "local")).lower()
        if bid in texts and qs:
            out[bid] = {"queries": qs[:2],
                        "kind": kind if kind in {"local", "stock", "schematic", "cutout"}
                        else "local"}
    return out


_META = frozenset(["media", "start", "clip", "image", "schematic", "photo", "diagram"])
SCENE_WORDS = frozenset([
    "view", "panorama", "panoramic", "landscape", "scene", "field", "crowd", "room", "sky",
    "horizon", "aerial", "overview", "montage", "collage", "control"])


def _asset_stems(asset: dict[str, Any]) -> set[str]:
    from src.stage3_plan_scenes import STOPWORDS, _stem

    toks = [*asset.get("tags", []), *re.findall(r"[A-Za-z0-9]+", asset.get("description", ""))]
    return {_stem(t) for t in toks
            if len(t) > 2 and t.lower() not in STOPWORDS and t.lower() not in _META
            and not t.isdigit()}


def keyword_prior(ctx: Any, text: str, weight: float, context: str = "",
                  power: float = 1.0) -> dict[str, float]:
    """Weak text prior from the sidecar/filename description (0..~weight, IDF-weighted).

    * rare shared terms count more than ones every asset has ("moon", "astronaut"), so generic
      lines do not drift to the most generic photos;
    * schematics get a 3x weight: image embeddings cannot read diagrams, their sidecar text can;
    * ``context`` (neighbouring beats' narration) adds a small topic-coherence term.
    """
    import math

    from src.stage3_plan_scenes import _stem, content_words

    if weight <= 0:
        return {}
    stems = {a["id"]: _asset_stems(a) for a in ctx.assets}
    n = max(len(stems), 1)
    df: dict[str, int] = {}
    for st in stems.values():
        for t in st:
            df[t] = df.get(t, 0) + 1

    def idf(t: str) -> float:
        return (math.log((n + 1) / (df.get(t, 0) + 1)) + 0.3) ** power

    q = {_stem(w) for w in content_words(text)}
    qc = {_stem(w) for w in content_words(context)} - q
    out: dict[str, float] = {}
    for a in ctx.assets:
        st = stems[a["id"]]
        direct = min(sum(idf(t) for t in q & st) / 6.0, 1.0)
        topic = min(sum(idf(t) for t in qc & st) / 9.0, 1.0)
        mult = 3.0 if a.get("asset_type") == "schematic" else 1.0
        out[a["id"]] = weight * mult * direct + weight * 0.4 * topic
    return out


def _doc_text(asset: dict[str, Any]) -> str:
    """Sidecar description without the leading ``type | filename |`` boilerplate."""
    d = re.sub(r"^(clip|image|schematic) \|[^|]*\|", "", asset.get("description", ""))
    return " ".join(d.split())[:400]


def description_prior(ctx: Any, texts: dict[int, str], cfg: dict[str, Any],
                      weight: float = 0.10) -> dict[int, dict[str, float]]:
    """Text-to-text similarity (local embedding model) between each narration line and every
    asset's sidecar description: the main weak prior, and the only signal that can rank
    diagrams. Empty when no embedding model is reachable (IDF prior alone then applies)."""
    import httpx

    from src.semantic import OllamaBackend

    ocfg = cfg.get("ollama", {}) or {}
    be = OllamaBackend(ocfg.get("base_url", "http://localhost:11434"),
                       ocfg.get("caption_model", "qwen2.5vl:3b"),
                       ocfg.get("embed_model", "nomic-embed-text"), timeout=60.0)
    try:
        if not be.models():
            return {}
        docs = be.embed_texts_as_documents([_doc_text(a) for a in ctx.assets])
        ids = list(texts)
        qs = be.embed_queries([texts[i] for i in ids])
    except (httpx.HTTPError, KeyError, ValueError):
        return {}
    sims = qs @ docs.T
    out: dict[int, dict[str, float]] = {}
    for row, bid in zip(sims, ids, strict=True):
        z = (row - row.mean()) / (row.std() + 1e-6)
        out[bid] = {a["id"]: float(weight * np.clip((z[j] - 0.5) / 2.5, 0.0, 1.0))
                    for j, a in enumerate(ctx.assets)}
    return out


def text_overlap(asset: dict[str, Any], text: str) -> int:
    """Shared content-word stems between the narration and the asset's sidecar text."""
    from src.stage3_plan_scenes import _stem, content_words

    return len({_stem(w) for w in content_words(text)} & _asset_stems(asset))


def has_clear_subject(asset: dict[str, Any] | None) -> bool:
    """A still worth cutting out: an image whose description opens on one object/person, not
    on a scene/landscape/panorama."""
    if not asset or asset.get("asset_type") != "image":
        return False
    desc = re.sub(r"^image \|[^|]*\|", "", asset.get("description", ""))
    head = re.findall(r"[a-z]+", desc.lower())[:6]
    return bool(head) and not (set(head) & SCENE_WORDS)


def _sidecar_text(asset: dict[str, Any]) -> str:
    path = asset.get("filepath")
    if not path:
        return ""
    try:
        return Path(str(path) + ".txt").read_text()
    except OSError:
        return ""


def _ranges(text: str, key: str) -> list[tuple[float, float]]:
    """``key: 0-9.2, 30-35`` -> [(0, 9.2), (30, 35)] (sidecar hint lists)."""
    out: list[tuple[float, float]] = []
    for m in re.finditer(rf"\b{key}\s*[:=]\s*([0-9.,\s\-–]+)", text, re.IGNORECASE):
        for a, b in re.findall(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", m.group(1)):
            if float(b) > float(a):
                out.append((float(a), float(b)))
    return out


def sidecar_media_start(asset: dict[str, Any]) -> float | None:
    """Optional ``media_start: <seconds>`` line in the asset's sidecar (skip a title card)."""
    m = re.search(r"media_start\s*[:=]\s*(\d+(?:\.\d+)?)", _sidecar_text(asset), re.IGNORECASE)
    return float(m.group(1)) if m else None


def sidecar_state(asset: dict[str, Any]) -> str | None:
    """Optional ``state: ideal|flawed|before|after`` sidecar hint; a diagram whose file name says
    ``perfect`` counts as ``ideal`` when no hint is given."""
    m = re.search(r"\bstate\s*[:=]\s*(ideal|flawed|before|after)\b", _sidecar_text(asset),
                  re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return "ideal" if "perfect" in str(asset.get("id", "")).lower() else None


def sidecar_clip_hints(asset: dict[str, Any]) -> dict[str, Any]:
    """Optional clip hints from the sidecar: ``media_start`` / ``media_end`` (seconds),
    ``avoid: a-b, c-d`` (never show these ranges, e.g. a burned-in caption or studio intro)
    and ``segments: a-b, c-d`` (only these ranges are good)."""
    text = _sidecar_text(asset)
    end = re.search(r"media_end\s*[:=]\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    start = sidecar_media_start(asset)
    return {"start": start, "end": float(end.group(1)) if end else None,
            "avoid": _ranges(text, "avoid"), "segments": _ranges(text, "segments")}


def apply_clip_hints(start: float, length: float, dur: float | None,
                     hints: dict[str, Any]) -> tuple[float, float]:
    """Move the window [start, start+length] so it honours media_start/media_end, lies in a
    ``segments`` range and clear of every ``avoid`` range (best effort, shrinks if needed)."""
    lo = hints.get("start") or 0.0
    hi = hints.get("end") or dur or (start + length)
    if dur:
        hi = min(hi, dur)
    segs = [(max(a, lo), min(b, hi)) for a, b in hints.get("segments") or []]
    segs = [(a, b) for a, b in segs if b > a] or [(lo, hi)]
    # pick the segment closest to the requested start, then slide inside it
    a, b = min(segs, key=lambda r: 0 if r[0] <= start <= r[1] else min(abs(start - r[0]),
                                                                       abs(start - r[1])))
    length = min(length, b - a)
    st = min(max(start, a), b - length)
    for _ in range(4):
        hit = next(((x, y) for x, y in hints.get("avoid") or [] if st < y and st + length > x),
                   None)
        if hit is None:
            break
        after = hit[1]
        st = after if after + length <= b else max(a, hit[0] - length)
    return round(st, 3), round(st + length, 3)


SPLIT_PANEL_W, SPLIT_PANEL_H = 1136, 700  # diagram panel of split_schematic (1920x1080 frame)
DEFAULT_TEXT_FRAC = 0.03    # assumed diagram text height / image height when no sidecar hint
MIN_TEXT_PX = 14.0          # smallest legible text height on a 1080p frame


def schematic_scale_in_split(asset: dict[str, Any] | None, panel_w: int = SPLIT_PANEL_W - 32,
                             panel_h: int = SPLIT_PANEL_H - 32) -> float:
    """How much of its native size a diagram keeps in the split_schematic diagram panel
    (limited by whichever panel dimension binds)."""
    w = (asset or {}).get("width") or 1920
    h = (asset or {}).get("height") or 1080
    return min(panel_w / max(w, 1), panel_h / max(h, 1), 1.0)


def diagram_text_px(asset: dict[str, Any] | None) -> float:
    """Native text height of a diagram: sidecar ``text_px: N`` if given, else a fraction of
    the image height (the generated diagrams use ~3%)."""
    a = asset or {}
    try:
        m = re.search(r"text_px\s*[:=]\s*(\d+(?:\.\d+)?)",
                      Path(str(a.get("filepath")) + ".txt").read_text(), re.IGNORECASE)
        if m:
            return float(m.group(1))
    except OSError:
        pass
    return DEFAULT_TEXT_FRAC * (a.get("height") or 1080)


def split_legible(asset: dict[str, Any] | None, min_scale: float | None = None) -> bool:
    """Diagram stays readable in the split panel: scale >= min and text >= MIN_TEXT_PX."""
    sc = schematic_scale_in_split(asset)
    return sc >= (MIN_SPLIT_SCALE if min_scale is None else min_scale) \
        and diagram_text_px(asset) * sc >= MIN_TEXT_PX


MIN_SPLIT_SCALE = 0.5  # below this the diagram text is unreadable: show it fullscreen instead
SMALL_PHOTO_W = 1000


def _desc(asset: dict[str, Any] | None, hit: Hit) -> str:
    d = hit.caption or (asset or {}).get("description", "")
    return " ".join(str(d).split())[:140]


def llm_pick(llm: LLM, rows: list[dict[str, Any]], cache: Path | None = None
             ) -> dict[int, str]:
    """rows: {id, text, query, cands:[(asset_id, score, desc)]} -> {beat_id: asset_id}.
    Asked CHUNK beats at a time; a failing chunk just yields no picks (ranking wins).
    Answers are cached per (model, rows) so a re-plan with unchanged candidates is stable."""
    key = _cache_key("pick-v2", getattr(llm, "model", "") if llm.ok else "",
                     [(r["id"], r["text"], r["query"], [(a, f"{sc:.2f}", d) for a, sc, d in r["cands"]])
                      for r in rows])
    hit = _cached(cache, key)
    if isinstance(hit, dict) and hit:
        return {int(k): str(v) for k, v in hit.items()}
    out: dict[int, str] = {}
    for i in range(0, len(rows), CHUNK):
        blocks = []
        for r in rows[i: i + CHUNK]:
            cs = "\n".join(f'   - {a} (score {s:.2f}): {d}' for a, s, d in r["cands"])
            blocks.append(f'Beat {r["id"]}: "{r["text"]}"\n  query: {r["query"]}\n{cs}')
        user = ('Return {"picks":[{"id":0,"asset":"<candidate id>","reason":"short"}]}.\n\n'
                + "\n\n".join(blocks))
        if not llm.ok:
            break
        chunk = rows[i: i + CHUNK]
        schema = {"type": "object", "required": ["picks"], "properties": {"picks": {
            "type": "array", "minItems": len(chunk), "maxItems": len(chunk), "items": {
                "type": "object", "required": ["id", "asset"], "properties": {
                    "id": {"type": "integer", "enum": [r["id"] for r in chunk]},
                    "asset": {"type": "string",
                              "enum": sorted({a for r in chunk for a, _s, _d in r["cands"]})},
                    "reason": {"type": "string"}}}}}}
        for row in _as_list(llm.json(PICK_SYSTEM, user, max_tokens=600, schema=schema)):
            try:
                out[int(row["id"])] = str(row.get("asset", row.get("asset_id", ""))).strip()
            except (KeyError, TypeError, ValueError):
                continue
    if out and len(out) == len(rows):
        _store(cache, key, out)
    return out


def _cand_types(template: VisualTemplate, kind: str) -> tuple[str, ...]:
    if kind == "schematic" and template != VisualTemplate.CUTOUT_CALLOUT:
        return ("schematic",)
    return ("clip", "image")


def apply_semantic(plan: ScenePlan, ctx: Any, work: Path, config: dict[str, Any],
                   backend: SemanticBackend | None = None,
                   post_repair: Callable[[ScenePlan], ScenePlan] | None = None) -> ScenePlan:
    """Annotate/re-rank the beats of ``plan`` with content-based choices (in place)."""
    cfg = semantic_config(config)
    backend = backend or get_backend(config)
    if backend is None:
        console.print("  [yellow]No semantic backend: keyword matching used for asset choice[/]")
        return plan
    index = SemanticIndex.load(work, float(cfg.get("merge_margin") or 0.03))
    if index is None or index.tag != backend.tag:
        console.print("  [yellow]Semantic index missing or built by another backend (run "
                      "`semantic <episode>`): keyword matching used[/]")
        return plan
    k = int(cfg.get("top_k", 5))
    kpool = max(k, 8)  # extra candidates so reuse/diversity penalties have alternatives
    index.hints = {a["id"]: h for a in ctx.assets if (h := sidecar_media_start(a))}
    min_score = min_score_for(backend.name, config)
    margin = float(cfg.get("llm_pick_margin", 0.04 if backend.name == "siglip" else 0.08))

    beats = plan.beats
    texts = {b.id: ctx.text_between(b.start_time, b.end_time) or "" for b in beats}
    global RECENT
    RECENT = float(cfg.get("reuse_window", 22.0))
    cache_dir = work / "semantic_cache" if cfg.get("cache_queries", True) else None
    qllm = LLM(config, "query")  # query rewrite may use a bigger model than the picks
    ctx_summary = ""
    if qllm.ok and cfg.get("script_context", False):
        ctx_summary = script_context(qllm, texts, cache_dir / "script_ctx.json"
                                     if cache_dir else None)
    rewritten = rewrite_queries(qllm, texts, ctx_summary, bool(cfg.get("query_neighbours", False)),
                                cache_dir / "queries.json" if cache_dir else None) \
        if qllm.ok else {}
    if qllm.ok and not rewritten:
        console.print("  [yellow]LLM query rewrite failed; using narration text as query[/]")
    queries = {b.id: rewritten.get(b.id, {}).get("queries") or [texts[b.id]] for b in beats}
    if cfg.get("narration_query", False):  # multi-query retrieval: also the raw narration
        for b in beats:
            if texts[b.id] and rewritten.get(b.id):
                queries[b.id] = [*queries[b.id], texts[b.id]]
    llm = LLM(config, "pick")
    kinds = {b.id: rewritten.get(b.id, {}).get("kind", "local") for b in beats}

    try:
        flat = [(b.id, q) for b in beats for q in queries[b.id] if q.strip()]
        qv = backend.embed_queries([q for _, q in flat]) if flat else np.zeros((0, 1))
    finally:
        backend.free()  # give the VRAM back before the LLM picks
    by_beat: dict[int, list[np.ndarray]] = {}
    for (bid, _), v in zip(flat, qv, strict=True):
        by_beat.setdefault(bid, []).append(v)

    pw = float(cfg.get("prior_weight") or (0.06 if backend.name == "siglip" else 0.08))
    ids = [b.id for b in beats]
    priors = {}
    for n, b in enumerate(beats):
        near = [texts[i] for i in ids[max(0, n - 2): n + 3] if i != b.id]
        priors[b.id] = keyword_prior(ctx, texts[b.id] + " " + " ".join(queries[b.id]), pw,
                                     " ".join(near),
                                     float(cfg.get("prior_idf_power", 1.0)))

    if cfg.get("text_prior", True) and not is_offline():
        tp = description_prior(ctx, {b.id: texts[b.id] for b in beats if texts[b.id]}, cfg,
                               float(cfg.get("text_prior_weight", 0.10)))
        for bid, pr in tp.items():
            for aid, v in pr.items():
                priors[bid][aid] = priors[bid].get(aid, 0.0) + v

    def search(bid: int, types: tuple[str, ...], exclude: set[str]) -> list[Hit]:
        best: dict[str, Hit] = {}
        for v in by_beat.get(bid, []):
            for h in index.search(v, types, kpool, exclude, priors[bid]):
                if h.asset_id not in best or h.rank > best[h.asset_id].rank:
                    best[h.asset_id] = h
        return sorted(best.values(), key=lambda h: -h.rank)[:kpool]

    for b in beats:  # abstract narration never gets stock photos: show a diagram instead
        if b.primary_source == PrimarySource.STOCK and kinds[b.id] == "schematic" \
                and rewritten.get(b.id):
            b.primary_source = PrimarySource.LOCAL
            b.search_queries, b.stock_media = [], "any"
            b.stock_asset_id = b.license_type = b.media_path = None
            b.source_id = ""
            if b.visual_template not in (VisualTemplate.FULLSCREEN_CLIP,
                                         VisualTemplate.PUNCH_IN_EMPHASIS,
                                         VisualTemplate.SPLIT_SCHEMATIC):
                b.visual_template = VisualTemplate.FULLSCREEN_CLIP

    # candidates per local beat
    cands: dict[int, list[Hit]] = {}
    for b in beats:
        if b.primary_source == PrimarySource.STOCK:
            continue
        cands[b.id] = search(b.id, _cand_types(b.visual_template, kinds[b.id]), set())
    rows = []
    for b in beats:
        if cands.get(b.id):
            rows.append({"id": b.id, "text": texts[b.id], "query": queries[b.id][0],
                         "cands": [(h.asset_id, h.score, _desc(ctx.assets_by_id.get(h.asset_id),
                                                               h)) for h in cands[b.id][:k]]})
    dec = getattr(ctx, "decision", None)
    use_llm_pick = llm.ok and rows and cfg.get("llm_pick", True) \
        and getattr(ctx, "planner_mode", "llm") != "decision"
    picks = llm_pick(llm, rows, cache_dir / "picks.json" if cache_dir else None) \
        if use_llm_pick else {}
    if dec is not None and dec.enabled and by_beat:
        _decide_assets(dec, beats, ctx, texts, by_beat, index, priors,
                       kinds, config)
    DEBUG.clear()
    DEBUG["min_score"] = min_score
    DEBUG.update({"queries": queries, "kinds": dict(kinds), "picks": picks,
                  "cands": {bid: [(h.asset_id, round(h.rank, 4)) for h in hs]
                            for bid, hs in cands.items()}})

    last_used: dict[str, float] = {}
    last_overlay: dict[str, float] = {}
    uses: dict[str, int] = {}
    recent_templates: list[VisualTemplate] = []
    changed = 0

    hint_cache: dict[str, dict[str, Any]] = {}

    def fit(hit: Hit, b: Any) -> None:
        clip = ctx.assets_by_id.get(hit.asset_id, {})
        b.media_start, b.media_end = fit_window(hit, b.duration, clip.get("duration"))
        hints = hint_cache.get(hit.asset_id)
        if hints is None:
            hints = hint_cache[hit.asset_id] = sidecar_clip_hints(clip) if clip else {}
        if b.media_start is not None and any(hints.get(k) for k in
                                             ("start", "end", "avoid", "segments")):
            b.media_start, b.media_end = apply_clip_hints(
                b.media_start, b.media_end - b.media_start, clip.get("duration"), hints)

    def adj(h: Hit, now: float) -> float:
        """rank minus a per-asset reuse penalty (grows with every use, steeper when recent)."""
        pen = 0.02 * uses.get(h.asset_id, 0)
        if now - last_used.get(h.asset_id, -1e9) < RECENT:
            pen += 0.05
        return h.rank - pen

    gchoice: dict[int, Hit] = {}
    if cfg.get("global_assignment", True):
        gchoice = _global_choices(beats, ctx, texts, queries, kinds, by_beat, index, priors,
                                  picks, min_score, cfg, RECENT)
        DEBUG["global"] = {bid: h.asset_id for bid, h in gchoice.items()}
    planned: dict[str, int] = {}
    for h in gchoice.values():
        planned[h.asset_id] = planned.get(h.asset_id, 0) + 1

    def room(aid: str) -> bool:
        """Per-asset reuse cap, counting what the global plan still has in store for it."""
        t = (ctx.assets_by_id.get(aid) or {}).get("asset_type")
        cap = {"image": PHOTO_CAP, "schematic": DIAGRAM_CAP}.get(t, CLIP_CAP)
        return uses.get(aid, 0) + planned.get(aid, 0) < cap
    split_cap = max(1, round(float((cfg.get("variety") or {}).get("split", [0.08, 0.12])[1])
                             * len(beats)))

    for b in beats:
        b.visual_query = " | ".join(queries[b.id])[:200] or None
        hs = cands.get(b.id)
        if hs:
            hs = sorted(hs, key=lambda h: -adj(h, b.start_time))
            chosen = hs[0]
            want = picks.get(b.id)
            pick = next((h for h in hs[:k] if h.asset_id == want), None)
            if pick is not None and adj(pick, b.start_time) >= adj(hs[0], b.start_time) - margin:
                chosen = pick
            g = gchoice.get(b.id)
            if g is not None and b.visual_template != VisualTemplate.CUTOUT_CALLOUT:
                chosen = g  # the episode-wide assignment (diversity, caps, clip share)
                kinds[b.id] = "schematic" if g.asset_type == "schematic" else "local"
                sch = None
            else:
                sch = _diagram_override(b, chosen, search, ctx,
                                        texts[b.id] + " " + queries[b.id][0], adj, uses,
                                        min_score if cfg.get("schematic_fallback") else None,
                                        frozenset(a for a, t in last_used.items()
                                                  if b.start_time - t < RECENT))
            if sch is not None:  # the line talks about a diagram's content: show that diagram
                chosen, kinds[b.id] = sch, "schematic"
            if b.visual_template == VisualTemplate.CUTOUT_CALLOUT and kinds[b.id] != "schematic":
                changed += _assign_cutout(b, [h for h in hs if room(h.asset_id) or h.asset_id == chosen.asset_id]
                                          or hs, chosen, ctx, min_score, fit, texts[b.id])
            elif kinds[b.id] == "schematic" and chosen.asset_type == "schematic":
                side = [h for h in search(b.id, ("clip", "image"), set())
                        if h.score >= min_score and room(h.asset_id)
                        and b.start_time - last_used.get(h.asset_id, -1e9) >= RECENT]
                n_split = sum(x.visual_template == VisualTemplate.SPLIT_SCHEMATIC for x in beats
                              if x.end_time <= b.start_time)
                n = _assign_schematic(b, chosen, recent_templates, ctx, side[0] if side else None,
                                      texts[b.id], fit, split_ok=n_split < split_cap)
                changed += n
                if n == 0 and side:  # diagram refused (template rules): best footage instead
                    chosen = next((h for h in side if b.start_time - last_used.get(
                        h.asset_id, -1e9) >= RECENT), side[0])
                    b.source_id = chosen.asset_id
                    fit(chosen, b)
                    changed += 1
            else:
                if chosen.asset_id != b.source_id:
                    changed += 1
                b.source_id = chosen.asset_id
                if b.visual_template == VisualTemplate.SPLIT_SCHEMATIC:
                    _pick_overlay(b, by_beat, index, priors[b.id], last_overlay, b.start_time,
                                  min_score, ctx)
                fit(chosen, b)
            _small_photo_fix(b, ctx)
            _mark_defocus(b, ctx, texts[b.id])
            chosen_ids = {x for x in (b.source_id, b.overlay_asset) if x}
            b.match_score = round(chosen.score, 4)
            for x in chosen_ids:
                last_used[x] = b.end_time
                uses[x] = uses.get(x, 0) + 1
            reason = f"semantic {backend.name} {chosen.score:.2f}"
            if dec is not None and b.id in dec.p_asset:
                pa = dec.p_asset[b.id]
                aid = b.source_id if b.source_id in pa else chosen.asset_id
                if aid in pa:
                    top = max(pa, key=pa.__getitem__)
                    reason += f" | decision:{dec.name} asset p={pa[aid]:.2f}" + (
                        "" if top == aid else f" (top {pa[top]:.2f})")
            b.beat_reason = f"{b.beat_reason} | {reason}" if b.beat_reason else reason
        elif b.primary_source == PrimarySource.LOCAL and b.source_id in index.asset_ids():
            vs = by_beat.get(b.id)
            if vs:
                h = index.score_asset(b.source_id, vs[0])
                b.match_score = round(h.score, 4) if h else None
        recent_templates = (recent_templates + [b.visual_template])[-2:]
    if (cfg.get("variety") or {}).get("enabled", True) is not False:
        _rebalance_mix(beats, ctx, texts, queries, search, fit, min_score, cfg, work)
    if post_repair is not None:
        before = {b.id: (b.source_id, b.overlay_asset) for b in beats}
        plan = post_repair(plan)
        beats = plan.beats
        for b in beats:  # the repairer may swap assets: keep scores/trims truthful
            vs = by_beat.get(b.id)
            if b.id in before and before[b.id] != (b.source_id, b.overlay_asset):
                b.media_start = b.media_end = None
            subject = b.overlay_asset if b.visual_template == VisualTemplate.CUTOUT_CALLOUT \
                else b.source_id
            if vs and subject in index.asset_ids() and b.primary_source == PrimarySource.LOCAL:
                h = index.score_asset(subject, vs[0])
                b.match_score = round(h.score, 4) if h else None
    for b in beats:
        if b.visual_template == VisualTemplate.CUTOUT_CALLOUT:
            vs = by_beat.get(b.id)
            h = index.score_asset(b.overlay_asset, vs[0]) if vs and b.overlay_asset \
                and b.overlay_asset in index.asset_ids() else None
            if not (b.overlay_asset and has_clear_subject(ctx.assets_by_id.get(b.overlay_asset))
                    and h is not None and h.score >= min_score):
                _demote_cutout(b, ctx, b.overlay_asset or b.source_id, texts.get(b.id, ""))
        _small_photo_fix(b, ctx)
    if (cfg.get("diversity") or {}).get("enabled", True) is not False:
        try:
            from src.views import diversify_views

            vs = diversify_views(beats, ctx, index, by_beat, backend, cfg, work,
                                 clip_hints=_hinted_window)
            backend.free()
            if any(vs.values()):
                console.print(f"  Reuse views: {vs['crop']} detail crop(s), {vs['segment']} "
                              f"other clip segment(s), {vs['highlight']} highlight(s)")
        except Exception as e:  # noqa: BLE001 - views are optional
            console.print(f"  [yellow]Reuse views skipped ({e})[/]")
    finalize_callouts(beats, ctx, texts)
    try:
        import json

        (work / "semantic_debug.json").write_text(json.dumps(
            {"global": DEBUG.get("global", {}), "variety": DEBUG.get("variety", {}),
             "families": DEBUG.get("families", {}), "matrix": DEBUG.get("matrix", {})}, default=str))
    except OSError:
        pass
    low = sum(1 for b in beats if b.match_score is not None and b.match_score < min_score)
    console.print(f"  Semantic matching ({backend.name}): {changed} beat(s) re-assigned, "
                  f"{low} below min_score {min_score:.2f} (flagged for review)")
    return plan



PHOTO_CAP, DIAGRAM_CAP, CLIP_CAP = 3, 2, 3


def _hinted_window(start: float, length: float, dur: float, asset: dict[str, Any]
                   ) -> tuple[float, float]:
    """A candidate clip window moved to honour the sidecar ``media_start/end/avoid/segments``."""
    hints = sidecar_clip_hints(asset)
    if not any(hints.get(k) for k in ("start", "end", "avoid", "segments")):
        return start, start + length
    return apply_clip_hints(start, length, dur, hints)


def asset_families(index: SemanticIndex, ids: list[str], threshold: float = 0.9) -> dict[str, int]:
    """Near-duplicate family id per asset (mean frame embedding cosine > threshold)."""
    from src.assignment import families

    vecs = {}
    for a in ids:
        rows = [f["row"] for f in index.data["assets"][a]["frames"]]
        if rows:
            vecs[a] = index.vecs[rows].mean(axis=0)
    return families(vecs, threshold)


def _global_choices(beats: list[Any], ctx: Any, texts: dict[int, str],
                    queries: dict[int, list[str]], kinds: dict[int, str],
                    by_beat: dict[int, list[np.ndarray]], index: SemanticIndex,
                    priors: dict[int, dict[str, float]], picks: dict[int, str],
                    min_score: float, cfg: dict[str, Any], window: float) -> dict[int, Hit]:
    """Episode-wide beat x asset assignment (src.assignment) on the similarity matrix.

    Diagrams are ranked by sidecar/filename term overlap + the description embedding prior
    (image embeddings cannot read them); footage/photos by embedding score + prior. Clips get
    a content-aware bonus when the line describes an action/process; every asset has a reuse
    cap (photos 3, diagrams 2) and the 20-25 s no-reuse window is enforced."""
    from src.assignment import global_assign, is_action, state_penalty

    caps_cfg = cfg.get("reuse_caps") or {}
    caps_by = {"image": int(caps_cfg.get("photo", PHOTO_CAP)),
               "schematic": int(caps_cfg.get("diagram", DIAGRAM_CAP)),
               "clip": int(caps_cfg.get("clip", CLIP_CAP))}
    ids = [a for a in index.asset_ids() if a in ctx.assets_by_id]
    ids = [a for a in ids if (ctx.assets_by_id[a].get("asset_type")) in caps_by]
    rows = [b for b in beats if b.primary_source == PrimarySource.LOCAL
            and b.visual_template != VisualTemplate.CUTOUT_CALLOUT and by_beat.get(b.id)]
    if not rows or not ids:
        return {}
    col = {a: j for j, a in enumerate(ids)}
    types = [ctx.assets_by_id[a]["asset_type"] for a in ids]
    clip_bonus = float(cfg.get("clip_bonus", 0.07))
    dec = getattr(ctx, "decision", None)
    gate_bonus = 0.0
    if dec is not None:
        gate_bonus = float(dec.cfg["weight"]) * float(dec.cfg["gate_p"])
    states = {a: sidecar_state(ctx.assets_by_id[a]) for a in ids}
    mat = np.full((len(rows), len(ids)), -np.inf)
    hits: dict[tuple[int, str], Hit] = {}
    for r, b in enumerate(rows):
        act = is_action(texts[b.id])
        for v in by_beat[b.id]:
            for h in index.search(v, ("clip", "image", "schematic"), 999, None, priors[b.id]):
                if h.asset_id not in col:
                    continue
                a = ctx.assets_by_id[h.asset_id]
                pr = priors[b.id].get(h.asset_id, 0.0)
                dbonus = dec.asset_bonus(b.id, h.asset_id) if dec is not None else None
                if h.asset_type == "schematic":
                    if b.duration < 2.4:  # too short to read a diagram
                        continue
                    ov = text_overlap(a, texts[b.id] + " " + queries[b.id][0])
                    chosen_by_model = dbonus is not None and dbonus >= gate_bonus
                    if ov < 1 and not (kinds.get(b.id) == "schematic" and pr >= 0.03) \
                            and not chosen_by_model:
                        continue
                    val = 0.06 + 0.07 * min(ov, 3) + pr + 0.05 * (kinds.get(b.id) == "schematic")
                else:
                    val = h.rank + (clip_bonus if (act and h.asset_type == "clip")
                                    else 0.02 if h.asset_type == "clip" else 0.0)
                val += state_penalty(texts[b.id], states.get(h.asset_id))
                if dbonus:
                    val += dbonus
                if picks.get(b.id) == h.asset_id:
                    val += 0.04
                j = col[h.asset_id]
                if val > mat[r, j]:
                    mat[r, j] = val
                    hits[(b.id, h.asset_id)] = h
    div = cfg.get("diversity") or {}
    fam_of: dict[str, int] = {}
    if div.get("enabled", True) is not False:
        fam_of = asset_families(index, ids, float(div.get("near_dup", 0.9)))
    DEBUG["families"] = fam_of
    DEBUG["matrix"] = {"rows": [(b.id, b.start_time, b.end_time) for b in rows], "ids": ids,
                       "types": types, "mat": np.where(np.isfinite(mat), mat, -1e6).tolist()}
    pick = global_assign(mat, [caps_by[t] for t in types], [t == "clip" for t in types],
                         [b.start_time for b in rows], [b.end_time for b in rows], window,
                         float(cfg.get("min_clip_share", 0.25)), 0.7 * min_score, 0.15,
                         family=[fam_of[a] for a in ids] if fam_of else None,
                         family_cap=int(div.get("family_cap", 4)),
                         fresh_bonus=float(div.get("fresh_bonus", 0.10)) if fam_of else 0.0,
                         max_family_loss=float(div.get("max_family_loss", 0.06)))
    out: dict[int, Hit] = {}
    for b, j in zip(rows, pick, strict=True):
        if j is not None and (b.id, ids[j]) in hits:
            out[b.id] = hits[(b.id, ids[j])]
    return out


def _decide_assets(dec: Any, beats: list[Any], ctx: Any, texts: dict[int, str],
                   by_beat: dict[int, list[np.ndarray]], index: SemanticIndex,
                   priors: dict[int, dict[str, float]], kinds: dict[int, str],
                   config: dict[str, Any]) -> None:
    """Ask the decision model which of the top-K candidates fits each local beat; the result
    (dec.p_asset) is blended into the global assignment matrix."""
    from src.decision_planner import candidate_union

    total = ctx.duration
    rows = []
    for i, b in enumerate(beats):
        if b.primary_source != PrimarySource.LOCAL or not by_beat.get(b.id):
            continue
        cands = [(a, c) for a, c in candidate_union(
            index, by_beat[b.id], priors[b.id], ("clip", "image", "schematic"),
            int(dec.cfg["candidates"])) if a in ctx.assets_by_id]
        rows.append({"id": b.id, "cands": cands, "start": b.start_time, "end": b.end_time,
                     "state": dec.state_for(i, beats, texts, total)})
    t0 = time.monotonic()
    dec.decide_assets(rows, ctx.assets_by_id)
    console.print(f"  Decision model ({dec.name}) rated {len(dec.p_asset)}/{len(rows)} beats "
                  f"in {time.monotonic() - t0:.0f}s")


def _diagram_override(b: Any, chosen: Hit, search: Callable[..., list[Hit]], ctx: Any,
                      text: str, adj: Callable[[Hit, float], float],
                      uses: dict[str, int], weak_below: float | None = None,
                      recent_ids: frozenset[str] = frozenset()) -> Hit | None:
    """A diagram whose sidecar text shares >= 2 terms with the line (or 1 and it also outranks
    the best footage) beats footage: image embeddings cannot read diagrams, the text can."""
    if chosen.asset_type == "schematic":
        return None
    sch = search(b.id, ("schematic",), set())
    if not sch:
        return None

    def worth(h: Hit) -> float:  # shared terms, minus a penalty for recent / repeated use
        return (text_overlap(ctx.assets_by_id.get(h.asset_id, {}), text)
                - 0.4 * uses.get(h.asset_id, 0) - 20 * (h.rank - adj(h, b.start_time))
                - (5 if h.asset_id in recent_ids else 0))  # never the same diagram back-to-back

    top = max(sch, key=lambda h: (worth(h), h.rank))
    ov = text_overlap(ctx.assets_by_id.get(top.asset_id, {}), text)
    if worth(top) < 1:
        return None
    if ov >= 2 or (ov >= 1 and top.rank >= chosen.rank):
        return top
    if weak_below is not None and ov >= 1 and chosen.score < weak_below:
        return top  # no footage matches well and a diagram really shares terms with the line
    return None


def _pick_overlay(b: Any, by_beat: dict[int, list[np.ndarray]], index: SemanticIndex,
                  prior: dict[str, float], last: dict[str, float], now: float,
                  min_score: float, ctx: Any = None) -> None:
    """Concrete narration with a diagram panel: keep the diagram only if one really fits and is
    legible at panel size; otherwise full-frame footage."""
    hits: list[Hit] = []
    for v in by_beat.get(b.id, []):
        hits += index.search(v, ("schematic",), 3, None, prior)
    hits.sort(key=lambda h: -h.rank)
    fresh = [h for h in hits if now - last.get(h.asset_id, -1e9) >= RECENT]
    pick = (fresh or hits)[0] if hits else None
    legible = pick is not None and (
        ctx is None or split_legible(ctx.assets_by_id.get(pick.asset_id)))
    if pick is not None and pick.rank >= min_score and legible:
        b.overlay_asset = pick.asset_id
        last[pick.asset_id] = now
    else:  # no diagram that fits and stays readable: full-frame footage instead
        b.visual_template = VisualTemplate.FULLSCREEN_CLIP
        b.overlay_asset = None


def _assign_schematic(b: Any, hit: Hit, recent: list[VisualTemplate], ctx: Any,
                      side: Hit | None, text: str = "",
                      fit: Callable[[Hit, Any], None] | None = None,
                      split_ok: bool = True, strong: int = 3) -> int:
    """A diagram beat: SPLIT (related footage/photo left, diagram >= 60% of the width right)
    when a matching side asset exists and the diagram stays legible in its panel; FULLSCREEN
    (native size, no caption: it carries its own text) when the line is mostly about the
    diagram's content (>= ``strong`` shared terms), it would be unreadable, no side media
    matches or the split budget is spent."""
    asset = ctx.assets_by_id.get(hit.asset_id)
    about = text_overlap(asset or {}, text)
    legible = split_legible(asset)
    long_enough = b.duration >= 2.8 and b.beat_style.value not in ("fail", "hype")
    if about >= strong or not legible or side is None or not split_ok or not long_enough:
        b.visual_template = VisualTemplate.FULLSCREEN_CLIP
        b.source_id, b.overlay_asset = hit.asset_id, None
        b.media_start = b.media_end = None
        b.emphasis = False
        b.callout_text = None  # the diagram carries its own text
        return 1
    if b.visual_template != VisualTemplate.SPLIT_SCHEMATIC:
        if len(recent) == 2 and recent[0] == recent[1] == VisualTemplate.SPLIT_SCHEMATIC:
            return 0  # no 3 identical templates in a row
        if recent and recent[-1] == VisualTemplate.SPLIT_SCHEMATIC:
            return 0  # splits are spaced out
        if b.visual_template not in (VisualTemplate.FULLSCREEN_CLIP,
                                     VisualTemplate.PUNCH_IN_EMPHASIS):
            return 0
        b.visual_template = VisualTemplate.SPLIT_SCHEMATIC
    b.overlay_asset = hit.asset_id
    b.source_id = side.asset_id
    b.emphasis = False
    if fit is not None:
        fit(side, b)
    return 1


def _assign_cutout(b: Any, hs: list[Hit], chosen: Hit, ctx: Any, min_score: float,
                   fit: Callable[[Hit, Any], None], text: str = "") -> int:
    """Cut out ONLY the subject of this beat's own best match.

    The subject must be a still with a clear single subject that really matches the line; the
    background is the runner-up only when it also matches, else the same asset (blurred).
    Anything weaker becomes full-frame footage with a narration lower-third."""
    stills = [h for h in hs if h.asset_type == "image"
              and has_clear_subject(ctx.assets_by_id.get(h.asset_id))]
    subj = next((h for h in stills if h.score >= min_score + 0.02
                 and h.rank >= hs[0].rank - 0.06), None)
    if subj is None:
        _demote_cutout(b, ctx, chosen.asset_id, text)
        b.media_start = b.media_end = None
        fit(chosen, b)
        return 1
    rest = [h for h in hs if h.asset_id != subj.asset_id and h.score >= min_score
            and h.rank >= subj.rank - 0.05]
    changed = int(b.overlay_asset != subj.asset_id)
    b.overlay_asset = subj.asset_id
    if rest:
        b.source_id = rest[0].asset_id
        fit(rest[0], b)
    else:
        b.source_id = subj.asset_id
        b.media_start = b.media_end = None
    return changed


def _demote_cutout(b: Any, ctx: Any, asset_id: str, text: str = "") -> None:
    """No clear subject: fullscreen (punch for fail/hype) with the narration as lower-third."""
    punch = b.beat_style.value in ("fail", "hype")
    b.visual_template = VisualTemplate.PUNCH_IN_EMPHASIS if punch else VisualTemplate.FULLSCREEN_CLIP
    if asset_id:
        b.source_id = asset_id
    b.overlay_asset = None
    b.callout_text = b.callout_text or (ctx.callout(text) if text else "") or None


def _mark_defocus(b: Any, ctx: Any, text: str) -> None:
    """The line describes blur and the shown photo/clip is sharp: blur it on purpose at render
    time (flagged 'illustrative defocus' in the timeline). Never on diagrams or flawed-state
    assets."""
    from src.assignment import BLUR_RE, FIXED_RE

    a = ctx.assets_by_id.get(b.source_id or "") or {}
    b.illustrative_defocus = bool(
        BLUR_RE.search(text) and not FIXED_RE.search(text)
        and b.visual_template in (VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.PUNCH_IN_EMPHASIS)
        and a.get("asset_type") in ("image", "clip") and sidecar_state(a) not in ("flawed", "before")
        and not re.search(r"\b(blur|blurry|blurred|fuzzy)\b", a.get("description", ""), re.IGNORECASE))


def _small_photo_fix(b: Any, ctx: Any) -> None:
    """Low-resolution stills are never punched in (the zoom would upscale them again)."""
    a = ctx.assets_by_id.get(b.source_id) or {}
    if b.visual_template == VisualTemplate.PUNCH_IN_EMPHASIS and a.get("asset_type") == "image" \
            and (a.get("width") or 9999) < SMALL_PHOTO_W:
        b.visual_template = VisualTemplate.FULLSCREEN_CLIP


def finalize_callouts(beats: list[Any], ctx: Any, texts: dict[int, str]) -> None:
    """Global callout pass (src.callouts): LLM suggestions + noun-phrase extraction, validated
    against the narration and the shown asset's sidecar, ranked by script-level term salience,
    at most one per two beats, no duplicates, none on hype/fail payoffs unless they name the
    mishap. Beats without a good key-term phrase simply have no callout."""
    from src.callouts import assign

    ctx.callouts_done = True

    def shown(b: Any) -> frozenset[str]:
        st: set[str] = set()
        for aid in (b.source_id, b.overlay_asset):
            st |= _asset_stems(ctx.assets_by_id.get(aid or "") or {})
        return frozenset(st)

    def diagram_main(b: Any) -> bool:  # a fullscreen diagram already carries its own text
        a = ctx.assets_by_id.get(b.source_id or "") or {}
        return a.get("asset_type") == "schematic" and b.visual_template in (
            VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.PUNCH_IN_EMPHASIS)

    assign(beats, texts, ctx, getattr(ctx, "llm_callouts", None), shown, skip=diagram_main,
           max_ratio=float((getattr(ctx, "config", {}).get("callouts") or {}).get("max_ratio", 0.34)))


def _cutout_ok(asset_id: str, work: Path) -> bool:
    """Stage 5 already produced a clean matte for this still (quality-gate history)?"""
    import json

    try:
        man = json.loads((work / "cutouts_manifest.json").read_text())
        d = Path(man.get("cutouts_dir", ""))
        return d.is_dir() and any(asset_id in f.name for f in d.glob("cutout_*"))
    except (OSError, ValueError):
        return False


def _rebalance_mix(beats: list[Any], ctx: Any, texts: dict[int, str],
                   queries: dict[int, list[str]], search: Callable[..., list[Hit]],
                   fit: Callable[[Hit, Any], None], min_score: float, cfg: dict[str, Any],
                   work: Path) -> None:
    """Hit the target template mix (src.variety) with content-checked conversions only."""
    from src import variety

    assets = ctx.assets_by_id
    split_scale = float(cfg.get("min_split_scale", MIN_SPLIT_SCALE))
    stamp: dict[str, float] = {}  # asset id -> end time of its last use as overlay/subject
    for b in beats:
        if b.overlay_asset:
            stamp[b.overlay_asset] = b.end_time

    def total_uses(aid: str) -> int:
        return sum((x.source_id == aid) + (x.overlay_asset == aid) for x in beats)

    def fresh(aid: str, b: Any) -> bool:
        t = (assets.get(aid) or {}).get("asset_type")
        cap = DIAGRAM_CAP if t == "schematic" else PHOTO_CAP
        return b.start_time - stamp.get(aid, -1e9) >= RECENT and total_uses(aid) < cap

    def line(b: Any) -> str:
        return texts[b.id] + " " + " ".join(queries.get(b.id, []))

    def cutout(b: Any) -> variety.Option | None:
        if b.beat_style.value in ("fail", "hype") or b.duration < 2.2 \
                or b.primary_source != PrimarySource.LOCAL:
            return None
        hs = search(b.id, ("image",), set())
        for h in hs[:6]:
            a = assets.get(h.asset_id) or {}
            if not (has_clear_subject(a) and (a.get("width") or 9999) >= SMALL_PHOTO_W
                    and h.score >= min_score - 0.01 and fresh(h.asset_id, b)
                    and text_overlap(a, line(b)) >= 1):
                continue
            known = _cutout_ok(h.asset_id, work)

            def go(h: Hit = h, b: Any = b) -> None:
                bg = assets.get(b.source_id) or {}
                b.overlay_asset = h.asset_id
                if bg.get("asset_type") == "schematic" or not b.source_id:
                    b.source_id = h.asset_id
                    b.media_start = b.media_end = None
                stamp[h.asset_id] = b.end_time
                b.visual_template = VisualTemplate.CUTOUT_CALLOUT
                b.emphasis = False

            return (h.rank + 0.1 * text_overlap(assets.get(h.asset_id) or {}, line(b))
                    + (0.05 if known else 0.0), go)
        return None

    def split(b: Any) -> variety.Option | None:
        if b.duration < 2.8 or b.primary_source != PrimarySource.LOCAL \
                or b.beat_style.value in ("fail", "hype"):
            return None
        sch = [h for h in search(b.id, ("schematic",), set())[:3]
               if fresh(h.asset_id, b)
               and text_overlap(assets.get(h.asset_id) or {}, line(b)) >= 1
               and split_legible(assets.get(h.asset_id), split_scale)]
        if not sch:
            return None
        cur = assets.get(b.source_id) or {}
        left = None
        if cur.get("asset_type") != "schematic" and b.source_id:
            left = b.source_id
        else:
            side = [h for h in search(b.id, ("clip", "image"), set())[:3]
                    if h.score >= min_score]
            left = side[0].asset_id if side else None
        if left is None or left == sch[0].asset_id:
            return None
        top = sch[0]

        def go(top: Hit = top, left: str = left, b: Any = b) -> None:
            if left != b.source_id:
                b.source_id = left
                b.media_start = b.media_end = None
            b.overlay_asset = top.asset_id
            stamp[top.asset_id] = b.end_time
            b.visual_template = VisualTemplate.SPLIT_SCHEMATIC
            b.emphasis = False

        ov = text_overlap(assets.get(top.asset_id) or {}, line(b))
        return top.rank + 0.3 * ov + (0.2 if b.beat_style.value == "technical" else 0.0), go

    def punch(b: Any) -> variety.Option | None:
        a = assets.get(b.source_id) or {}
        if b.primary_source != PrimarySource.LOCAL or not a or a.get("asset_type") == "schematic" \
                or (a.get("asset_type") == "image" and (a.get("width") or 9999) < SMALL_PHOTO_W):
            return None
        weight = {"fail": 3.0, "hype": 3.0}.get(b.beat_style.value, 0.0) \
            + (2.0 if b.emphasis else 0.0) + (0.5 if b.beat_style.value == "deadpan" else 0.0)
        weight += 1.0 / max(b.duration, 1.0)  # short beats punch better

        def go(b: Any = b) -> None:
            b.visual_template = VisualTemplate.PUNCH_IN_EMPHASIS

        return weight, go

    def traced(name: str, fn: Callable[[Any], variety.Option | None]
               ) -> Callable[[Any], variety.Option | None]:
        def wrapped(b: Any) -> variety.Option | None:
            out = fn(b)
            DEBUG.setdefault("variety", {}).setdefault(name, {})[b.id] = \
                None if out is None else round(out[0], 3)
            return out
        return wrapped

    cutout, split, punch = traced("cutout", cutout), traced("split", split), traced("punch", punch)
    mix = variety.rebalance(beats, {VisualTemplate.CUTOUT_CALLOUT: cutout,
                                    VisualTemplate.SPLIT_SCHEMATIC: split,
                                    VisualTemplate.PUNCH_IN_EMPHASIS: punch},
                            cfg.get("variety"))
    console.print(f"  Template mix after rebalancing: {mix}")
