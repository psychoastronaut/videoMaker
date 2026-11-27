"""Stage 3: scene director.

Produces ``scene_plan.json`` either from a local OpenAI-compatible LLM (with validation-error
retries) or, when no LLM is reachable, from a deterministic planner that cuts on real word
boundaries/pauses using the pacing rules in ``config/default.yaml``. Both paths go through
``_validate_and_repair`` which guarantees full audio coverage, sane beat lengths and valid,
template-compatible asset references.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from src.common import (
    episode_dir as episode_dir_for,
)
from src.common import (
    is_offline,
    load_config,
    probe_duration,
    read_json,
    voice_path,
    write_json,
)
from src.schemas import BeatStyle, PrimarySource, SceneBeat, ScenePlan, VisualTemplate

console = Console()

STOPWORDS = frozenset(["a", "about", "above", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "let", "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "us", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why", "will", "with", "would", "you", "your", "yours", "okay", "yeah", "really", "actually", "basically", "gonna", "going", "get", "gets", "got", "one", "two", "three", "also", "next", "time", "thanks", "again", "still", "even", "every", "per", "need", "needs", "needed", "looked", "look", "went", "built", "supposed", "ever", "many", "much", "made", "make", "makes", "whole", "single", "watching", "did", "didn't", "doesn't", "don't", "isn't", "wasn't", "hadn't", "little", "bit", "lot", "way", "thing", "things", "something", "nothing", "anything", "everything", "nobody", "somebody", "everybody", "everyone", "someone", "anyone", "try", "tries", "tried", "beside", "near", "onto", "upon", "around", "behind", "among", "adding"])

STYLE_PRIORITY = [BeatStyle.FAIL, BeatStyle.HYPE, BeatStyle.DEADPAN, BeatStyle.TECHNICAL]

RECENT_WINDOW = 20.0  # seconds within which reusing the same asset is penalised
MIN_BEAT_DEFAULT = 1.2
MAX_BEAT_DEFAULT = 6.0


# --------------------------------------------------------------------------- entry point


def run(episode_id: str, work_dir: Path) -> dict[str, Any]:
    """Run Stage 3: generate the scene plan from transcript + assets."""
    transcript_path = work_dir / "transcript.json"
    assets_path = work_dir / "local_assets.json"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript not found. Run Stage 1 first: {transcript_path}")
    if not assets_path.is_file():
        raise FileNotFoundError(f"Asset index not found. Run Stage 2 first: {assets_path}")

    transcript = read_json(transcript_path)
    assets_data = read_json(assets_path)
    config = load_config()
    duration = _audio_duration(episode_id, transcript)
    if duration <= 0:
        raise ValueError("Voiceover duration is 0; cannot plan scenes")
    words = flatten_words(transcript)
    from src.media import _stock_policy
    from src.stockpool import load_pool

    pool = load_pool(work_dir, _stock_policy(work_dir))
    if pool:
        console.print(f"  Stock pool: {len(pool)} pre-fetched free media item(s)")
    from src.common import drop_burned, drop_excluded

    assets = drop_burned(drop_excluded(assets_data.get("assets", []), work_dir), work_dir)
    ctx = PlanContext(config, assets, words, duration, pool)

    plan = _try_llm_plan(transcript, ctx, episode_id)
    if plan is None:
        console.print("  [dim]Using deterministic planner[/]")
        plan = _generate_deterministic_plan(ctx, episode_id)
    else:
        console.print("  [green]LLM plan accepted[/] (repairing/normalising)")

    plan = _validate_and_repair(plan, ctx)
    try:
        from src.semantic_planner import apply_semantic

        plan = apply_semantic(plan, ctx, work_dir, config,
                              post_repair=lambda p: _repair_keep_trims(p, ctx))
    except Exception as e:  # noqa: BLE001 - semantic layer is optional; never break planning
        console.print(f"  [yellow]Semantic matching failed ({e}); keyword plan kept[/]")
    try:
        _topup_library(plan, ctx, episode_id, work_dir, config)
    except Exception as e:  # noqa: BLE001 - optional layer; never break planning
        console.print(f"  [yellow]Library top-up skipped ({e})[/]")
    if not getattr(ctx, "callouts_done", False):  # no semantic layer: same callout rules
        from src.semantic_planner import finalize_callouts

        finalize_callouts(plan.beats, ctx,
                          {b.id: ctx.text_between(b.start_time, b.end_time) for b in plan.beats})
    try:  # sidecar hints: audio / picture-in-picture / diagram highlights (unset fields only)
        from src.overlays import apply_sidecar_hints

        n_hints = apply_sidecar_hints(plan.beats, list(ctx.assets))
        if n_hints:
            console.print(f"  Sidecar hints applied to {n_hints} beats (audio/pip/highlights)")
    except Exception as e:  # noqa: BLE001 - optional layer
        console.print(f"  [yellow]Sidecar hints skipped ({e})[/]")
    plan_file = work_dir / "scene_plan.json"
    if plan_file.is_file():  # never lose a previous (possibly hand-edited) plan
        shutil.copy2(plan_file, work_dir / "scene_plan.prev.json")
        console.print("  [dim]Previous plan backed up to scene_plan.prev.json[/]")
    write_json(plan_file, plan.model_dump(mode="json"))

    styles: dict[str, int] = {}
    templates: dict[str, int] = {}
    for b in plan.beats:
        styles[b.beat_style.value] = styles.get(b.beat_style.value, 0) + 1
        templates[b.visual_template.value] = templates.get(b.visual_template.value, 0) + 1
    console.print(f"  Scene plan saved: {work_dir / 'scene_plan.json'}")
    console.print(f"  {len(plan.beats)} beats, total {plan.total_duration:.2f}s | "
                  f"templates {templates} | styles {styles}")
    return plan.model_dump(mode="json")


def _topup_library(plan: ScenePlan, ctx: PlanContext, episode_id: str, work_dir: Path,
                   config: dict[str, Any]) -> int:
    """Small library (distinct near-duplicate families < ``ratio`` per beat): fetch free stock
    for the weakest-matched fullscreen/punch beats and put the results on those beats. Honours
    ``stock_media.enabled`` (off: nothing happens, near-duplicate awareness + reuse views carry
    the variety) and ``stock_media.topup``. Stage 4 still verifies every item it resolves."""
    sm = config.get("stock_media", {})
    tc = sm.get("topup") or {}
    if sm.get("enabled", True) is False or tc.get("enabled", True) is False:
        return 0
    from src.common import is_offline
    from src.semantic_planner import DEBUG

    fams = DEBUG.get("families") or {}
    floor = float(DEBUG.get("min_score") or 0.1)
    n_assets = len(set(fams.values())) if fams else len(ctx.of_type("clip", "image", "schematic"))
    n = len(plan.beats)
    if n == 0 or is_offline() or n_assets >= float(tc.get("ratio", 1.0)) * n:
        return 0
    cap = int(tc.get("max_items", max(2, round(0.12 * n))))
    weak = sorted((b for i, b in enumerate(plan.beats)
                   if i > 0 and b.primary_source == PrimarySource.LOCAL and b.visual_query
                   and b.visual_template in (VisualTemplate.FULLSCREEN_CLIP,
                                             VisualTemplate.PUNCH_IN_EMPHASIS)
                   and b.match_score is not None
                   and b.match_score < float(tc.get("below", 1.2)) * floor), key=lambda b: b.match_score or 0.0)
    weak = weak[:cap]
    if not weak:
        return 0
    console.print(f"  Library small ({n_assets} distinct looks for {n} beats): top-up for "
                  f"{len(weak)} weakest beat(s)")
    from src.stage_prefetch import topup
    from src.stockpool import POOL_FILE

    queries = [b.visual_query.split(" | ")[0] for b in weak]
    added = topup(episode_id, work_dir, queries, cap)
    by_q = {it["for_query"]: it for it in added}
    done = 0
    for b, q in zip(weak, queries, strict=True):
        it = by_q.get(q)
        if it is None or not Path(it.get("local_path", "")).is_file():
            continue
        b.primary_source, b.source_id, b.overlay_asset = PrimarySource.STOCK, "", None
        b.stock_asset_id, b.media_path = it["id"], it["local_path"]
        b.license_type, b.search_queries, b.stock_media = it.get("license_type"), [it["query"]], "image"
        b.media_start = b.media_end = b.crop_region = None
        b.beat_reason = (b.beat_reason + "; library top-up: weakest local match").strip("; ")
        done += 1
    console.print(f"  Top-up: {done} beat(s) now use free stock ({work_dir / POOL_FILE})")
    return done


def _repair_keep_trims(plan: ScenePlan, ctx: PlanContext) -> ScenePlan:
    """Re-run template/asset compatibility repair after semantic assignment. A trim interval
    only makes sense for the asset it was fitted to, so it is dropped if the asset changed."""
    before = {b.id: b.source_id for b in plan.beats}
    plan = _validate_and_repair(plan, ctx, semantic=True)
    for b in plan.beats:
        dur = (ctx.assets_by_id.get(b.source_id) or {}).get("duration")
        if b.media_start is not None and (
                before.get(b.id) != b.source_id or not dur or b.media_end is None
                or b.media_end > dur + 0.05):
            b.media_start = b.media_end = None
    return plan


def _audio_duration(episode_id: str, transcript: dict[str, Any]) -> float:
    wav = voice_path(episode_dir_for(episode_id))
    d = probe_duration(wav) if wav.is_file() else None
    if d:
        return d
    if transcript.get("duration"):
        return float(transcript["duration"])
    segs = transcript.get("segments", [])
    return float(segs[-1]["end"]) if segs else 0.0


# --------------------------------------------------------------------------- shared context


@dataclass
class Word:
    text: str
    start: float
    end: float


def flatten_words(transcript: dict[str, Any]) -> list[Word]:
    """All words with timestamps; segments without word timing get proportional timing."""
    out: list[Word] = []
    for seg in transcript.get("segments", []):
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                text = str(w["word"]).strip()
                if text:
                    out.append(Word(text, float(w["start"]), float(w["end"])))
            continue
        tokens = str(seg.get("text", "")).split()
        if not tokens:
            continue
        weights = [len(t) + 1 for t in tokens]
        total = sum(weights)
        span = float(seg["end"]) - float(seg["start"])
        cur = float(seg["start"])
        for tok, wt in zip(tokens, weights, strict=True):
            d = span * wt / total
            out.append(Word(tok, cur, cur + d))
            cur += d
    return out


def _compile_keywords(kw_map: dict[str, list[str]]) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for name, kws in kw_map.items():
        if not kws:
            continue
        alt = "|".join(re.escape(k.lower()).replace(r"\ ", r"\s+") for k in
                       sorted(kws, key=len, reverse=True))
        patterns[name] = re.compile(rf"(?<![a-z0-9])(?:{alt})(?![a-z0-9])")
    return patterns


class PlanContext:
    """Config-derived rules + assets + words, shared by the planner and the repairer."""

    def __init__(self, config: dict[str, Any], assets: list[dict[str, Any]], words: list[Word],
                 duration: float, pool: list[dict[str, Any]] | None = None) -> None:
        self.callouts_done = False
        self.llm_callouts: list[tuple[float, float, str]] = []  # LLM callout suggestions
        self.stock_enabled = config.get("stock_media", {}).get("enabled", True) is not False
        self.pool = (pool or []) if self.stock_enabled else []  # pre-fetched free stock
        staging = config.get("staging", {})
        pacing = staging.get("beat_pacing", {})
        self.config = config
        self.assets = assets
        self.assets_by_id = {a["id"]: a for a in assets}
        self.words = words
        self.duration = duration
        self.min_beat = float(pacing.get("min_beat_duration", MIN_BEAT_DEFAULT))
        self.max_beat = float(pacing.get("max_beat_duration", MAX_BEAT_DEFAULT))
        self.durations: dict[str, tuple[float, float]] = {
            k: (float(v[0]), float(v[1])) for k, v in pacing.get("durations", {}).items()}
        for key, default in {"normal": (2.5, 4.5), "hold": (4.0, 6.0), "fail": (1.5, 3.0),
                             "hype": (1.5, 3.0), "technical": (3.0, 5.0),
                             "deadpan": (2.5, 4.5)}.items():
            self.durations.setdefault(key, default)
        tone_kw = {k: [str(x).lower() for x in v] for k, v in
                   pacing.get("tone_keywords", {}).items()}
        hold_kw = [str(x).lower() for x in staging.get("hold_on_keywords", [])]
        self.patterns = _compile_keywords({**tone_kw, "hold": hold_kw})
        self.stock_ratio = float(staging.get("stock_ratio", 0.25)) if self.stock_enabled else 0.0
        vocab: set[str] = set()
        for a in assets:
            vocab.update(_stem(t) for t in a.get("tags", []))
            vocab.update(_stem(t) for t in re.findall(r"[A-Za-z0-9]+", a.get("description", "")))
        for words_ in list(tone_kw.values()) + [hold_kw]:
            for phrase in words_:
                vocab.update(_stem(t) for t in re.findall(r"[a-z0-9]+", phrase))
        vocab -= {_stem(x) for x in ("clip", "image", "schematic")} | STOPWORDS | FILLER
        self.vocab = frozenset(vocab)
        # words that only carry the beat's mood (broke, finally, magic...) make poor queries
        ent_cfg = staging.get("entities", {})
        self.entities_enabled = bool(ent_cfg.get("enabled", True)) and self.stock_enabled
        self.entity_ratio = float(ent_cfg.get("stock_ratio", 0.35))
        self.pool_ratio = float(staging.get("pool_stock_ratio", 0.45))
        self.entity_keywords = entity_keyword_list(config)
        self.fail_words = frozenset(
            _stem(p) for p in tone_kw.get("fail", []) if " " not in p) - {_stem("short")}
        self.style_words = frozenset(
            _stem(t) for words_ in tone_kw.values() for phrase in words_
            for t in re.findall(r"[a-z0-9]+", phrase))

    def words_between(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if start - 1e-6 <= (w.start + w.end) / 2 < end + 1e-6]

    def pick_pool(self, text: str, used: set[str]) -> dict[str, Any] | None:
        """Unused on-theme (thematic / generic) pool image or clip, most related to the text."""
        from src.stockpool import _words

        cands = [p for p in self.pool if p["id"] not in used and not p.get("entity")
                 and p.get("exact", True)]
        if not cands:
            return None
        want = _words(text)
        return max(cands, key=lambda p: len(want & _words(
            p.get("query", "") + " " + " ".join(p.get("tags", [])))))

    def entity(self, words: list[Word]) -> tuple[str, bool] | None:
        """Concrete place/machine named in ``words``: (query, has_proper_name) or None.

        Uses Capitalised names mid-sentence (Tokyo, Eiffel Tower; acronyms and CamelCase such as
        PID/LiPo are not names) plus ``staging.entities.keywords``. The query is the name or
        keyword plus one neighbouring noun, 2-4 words ("Tokyo skyline", "CNC lathe").
        """
        if not self.entities_enabled or not words:
            return None
        return find_entity([w.text for w in words], self.entity_keywords)

    def stock_query(self, text: str) -> str:
        return stock_query(text, exclude=self.style_words)

    def callout(self, text: str, keyword: str = "", asset: dict[str, Any] | None = None) -> str:
        return narration_callout(text, self.vocab, keyword)

    def mishap_callout(self, text: str, keyword: str) -> str | None:
        """Label for full-frame fail/hype/technical beats: the moment from the narration
        (up to two content words before the keyword, one after), never an asset's name.
        None when the narration offers nothing good."""
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
        low = [t.lower() for t in tokens]
        kw = keyword.lower().split()
        content = [t.lower() not in STOPWORDS and t.lower() not in FILLER and len(t) > 2
                   for t in tokens]
        for i in range(len(low) - len(kw) + 1) if kw else []:
            if low[i: i + len(kw)] != kw:
                continue
            left: list[str] = []
            j, gap = i - 1, 0
            while j >= 0 and len(left) < 2 and gap <= 2:
                if content[j]:
                    left.insert(0, tokens[j])
                    gap = 0
                else:
                    gap += 1
                j -= 1
            right = [tokens[i + len(kw)]] if i + len(kw) < len(tokens) \
                and content[i + len(kw)] else []
            phrase = [*left, *tokens[i: i + len(kw)], *right]
            if len(phrase) >= 2 and any(content[i: i + len(kw)]):
                return " ".join(phrase[:5])
            break
        phrase_text = extract_callout(text, keyword, self.vocab, "")
        return None if phrase_text == "Key detail" else phrase_text

    def subject_callout(self, text: str, keyword: str, subject: dict[str, Any] | None,
                        existing: str | None = None) -> str:
        """Callout for a beat whose label sits next to ``subject`` (cutout / schematic).

        The narration phrase is kept only if it names the subject (shares a stem with its
        tags/description); otherwise the badge shows the subject's own label so it can never
        contradict what is on screen.
        """
        if existing and callout_from_narration(existing, text):
            return existing
        return narration_callout(text, self.vocab, keyword)

    # -- text helpers
    def text_between(self, start: float, end: float) -> str:
        mid = [w.text for w in self.words if start - 1e-6 <= (w.start + w.end) / 2 < end + 1e-6]
        return " ".join(mid)

    def classify(self, text: str) -> tuple[BeatStyle, str, bool, bool]:
        """(style, matched keyword, is_invisible_mechanics/technical, is_hold)."""
        t = text.lower().replace("’", "'")
        hits: dict[str, str] = {}
        for name, pat in self.patterns.items():
            m = pat.search(t)
            if m:
                hits[name] = m.group(0)
        style, kw = BeatStyle.NORMAL, ""
        for cand in STYLE_PRIORITY:
            if cand.value in hits:
                style, kw = cand, hits[cand.value]
                break
        technical = "technical" in hits or "invisible_mechanics" in hits
        if style == BeatStyle.NORMAL and technical:
            style = BeatStyle.TECHNICAL
            kw = hits.get("technical") or hits.get("invisible_mechanics", "")
        return style, kw, technical, "hold" in hits

    def target_range(self, style: BeatStyle, hold: bool) -> tuple[float, float]:
        return self.durations["hold" if hold else style.value]

    # -- asset helpers
    def of_type(self, *types: str) -> list[dict[str, Any]]:
        return [a for a in self.assets if a["asset_type"] in types]


def content_words(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    return [t for t in toks if t.lower() not in STOPWORDS and len(t) > 2]


def _stem(token: str) -> str:
    t = token.lower().strip("'-")
    for suffix in ("ing", "es", "s", "ed"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[: -len(suffix)]
    return t


def asset_score(asset: dict[str, Any], text: str) -> int:
    """Number of content-word stems of ``text`` found in the asset's tags/description."""
    stems = {_stem(w) for w in content_words(text)}
    tokens = set()
    for tag in asset.get("tags", []):
        tokens.add(_stem(tag))
    for tok in re.findall(r"[A-Za-z0-9]+", asset.get("description", "")):
        tokens.add(_stem(tok))
    tokens.discard(_stem(asset["asset_type"]))
    return len(stems & tokens)


FILLER = frozenset(["thousand", "hundred", "million", "times", "second", "seconds", "course", "theory", "kind", "sort", "exactly", "literally", "pop", "puff", "boom", "bang", "zap", "whoosh", "bzzt", "pow", "kaboom"])
_VERBISH = re.compile(r"(ed|ing|ly)$")
# common irregular verbs / light verbs that are never good search nouns
IRREGULAR_VERBS = frozenset(
    ["kept", "went", "got", "made", "took", "saw", "ran", "came", "gave", "held", "left", "put", "set", "said", "told", "found", "knew", "thought", "brought", "built", "sent", "spent", "stood", "sat", "grew", "threw", "broke", "chose", "wore", "wrote", "drew", "drove", "began"])


def asset_label(asset: dict[str, Any] | None) -> str:
    """Human label from an asset id/filename: ``clip_servo_bench_test`` -> ``Servo Bench Test``."""
    if not asset:
        return ""
    stem = re.sub(r"^(clip|image|schematic)_", "", asset["id"])
    return " ".join(w.capitalize() for w in re.split(r"[_\-\s]+", stem) if w)[:40]


def _ensure_words(phrase: str, lo: int = 2, hi: int = 5) -> list[str]:
    words = phrase.split()
    return words[:hi] if len(words) >= lo else []


def extract_callout(text: str, keyword: str = "", vocab: frozenset[str] | set[str] = frozenset(),
                    label: str = "") -> str:
    """A 2-5 word noun-phrase-like callout.

    Preference: a phrase around the matched style keyword; else the best run of content words
    (scored by overlap with the asset/keyword vocabulary, penalising verb/adverb endings and
    filler such as "thousand times"); else the label of the asset being shown; else the two
    longest content words. Never returns fewer than two words.
    """
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    is_content = [t.lower() not in STOPWORDS and t.lower() not in FILLER and len(t) > 2
                  and not t.isdigit() for t in tokens]
    if keyword:
        kw_tokens = keyword.lower().split()
        low = [t.lower() for t in tokens]
        for i in range(len(low) - len(kw_tokens) + 1):
            if low[i: i + len(kw_tokens)] == kw_tokens:
                a, b = i, i + len(kw_tokens)
                if a > 0 and is_content[a - 1]:
                    a -= 1
                if b < len(tokens) and is_content[b]:
                    b += 1
                words = _ensure_words(" ".join(tokens[a:b]))
                if words and any(is_content[i: i + len(kw_tokens)]):
                    return " ".join(words)
                break
    runs: list[list[str]] = []
    cur: list[str] = []
    for tok, ok in zip(tokens, is_content, strict=True):
        if ok:
            cur.append(tok)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    def score(phrase: list[str]) -> float:
        dom = sum(3 for w in phrase if _stem(w) in vocab)
        nouns = sum(1 for w in phrase if not _VERBISH.search(w.lower()))
        return dom + nouns + 0.5 * len(phrase)

    cands = []
    for run in runs:
        for n in (3, 2):
            for i in range(len(run) - n + 1):
                phrase = run[i: i + n]
                cands.append((score(phrase), phrase))
    best = max(cands, key=lambda c: c[0]) if cands else None
    if best and best[0] >= 4:  # has domain vocabulary and mostly noun-like words
        return " ".join(best[1])
    lab = _ensure_words(label)
    if lab:
        return " ".join(lab)
    if best:
        return " ".join(best[1])
    content = [t for t, ok in zip(tokens, is_content, strict=True) if ok]
    if len(label.split()) == 1:  # one-word asset name: keep it, never mix with narration words
        return f"{label.split()[0]} Detail"
    if len(content) >= 2:
        longest = sorted(content, key=len, reverse=True)[:2]
        return " ".join(t for t in content if t in longest)[:40]
    return "Key detail"


def _title_word(w: str) -> str:
    if w.isupper() or any(c.isdigit() for c in w):
        return w
    return w[:1].upper() + w[1:].lower()


def narration_callout(text: str, vocab: frozenset[str] | set[str] = frozenset(),
                      keyword: str = "") -> str:
    """2-5 meaningful words taken from the narration, Title Case. Never an asset/file name.

    Returns "" when the narration has no usable words (the caller then shows no callout)."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    ok = [t.lower() not in STOPWORDS and t.lower() not in FILLER
          and (len(t) > 2 or (t.isdigit() and len(t) == 2))
          and t.lower() not in IRREGULAR_VERBS | {"meant", "knew"} for t in tokens]
    runs: list[list[str]] = []
    cur: list[str] = []
    for tok, good in zip(tokens, ok, strict=True):
        if good:
            cur.append(tok)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    kw = set(keyword.lower().split())

    def score(ph: list[str]) -> float:
        dom = sum(3 for w in ph if _stem(w) in vocab)
        nouns = sum(1 for w in ph if not _VERBISH.search(w.lower()))
        kws = sum(2 for w in ph if w.lower() in kw)
        adj = sum(1 for w in ph if w.lower() in ADJECTIVES)
        end_bad = 2 if _VERBISH.search(ph[-1].lower()) else 0
        return dom + nouns + kws + 0.5 * len(ph) - 0.5 * adj - end_bad

    cands = [(score(r[i:i + n]), r[i:i + n]) for r in runs for n in (4, 3, 2)
             for i in range(len(r) - n + 1)]
    cands = [c for c in cands if c[1][-1].lower() not in ADJECTIVES]
    if cands:
        best = max(cands, key=lambda c: (c[0], len(c[1])))[1]
    else:
        flat = [t for t, g in zip(tokens, ok, strict=True)
                if g and t.lower() not in ADJECTIVES]
        if len(flat) < 2:
            return ""
        best = flat[:2]
    return " ".join(_title_word(w) for w in best[:5])


def callout_from_narration(callout: str, text: str) -> bool:
    """True when every word of ``callout`` (>= 2 words) occurs, stemmed, in the narration."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", callout)]
    if len(words) < 2 or len(words) > 5:
        return False
    stems = {_stem(w) for w in re.findall(r"[A-Za-z0-9]+", text)}
    return all(_stem(w) in stems for w in words)


ADJECTIVES = frozenset(
    ["dark", "clean", "good", "bad", "big", "small", "little", "whole", "perfect", "new", "old", "first", "last", "same", "real", "cool", "great", "nice", "long", "short", "high", "low", "hot", "cold", "fast", "slow", "tiny", "huge", "simple", "hard", "easy", "empty", "full", "bright", "quiet", "loud", "janky", "broken", "wrong", "right", "top", "bottom", "whole", "entire", "single"])
BRAND = frozenset(["psycho", "astronaut", "welcome", "subscribe", "channel", "video", "watching"])


def stock_query(text: str, max_words: int = 4, exclude: frozenset[str] = frozenset()) -> str:
    """Short stock-search query: 2-4 noun-like content words (no verbs/filler/stopwords)."""
    toks: list[str] = []
    for t in re.findall(r"[A-Za-z0-9]+", text):
        low = t.lower()
        if low in STOPWORDS or low in FILLER or low in BRAND or low in ADJECTIVES \
                or low in IRREGULAR_VERBS \
                or len(t) <= 2 or t.isdigit() \
                or _stem(t) in exclude or low in (x.lower() for x in toks):
            continue
        toks.append(t)
    # -ing words stay (soldering, wiring are B-roll nouns); -ed/-ly words go
    nouns = [t for t in toks if not re.search(r"(ed|ly)$", t.lower())] or toks
    if len(nouns) > max_words:
        keep = set(sorted(nouns, key=len, reverse=True)[:max_words])
        nouns = [t for t in nouns if t in keep]
    # a single generic word ("fuse", "survive") retrieves arbitrary footage: need >= 2 nouns
    return " ".join(nouns) if len(nouns) >= 2 else ""


DEFAULT_ENTITY_KEYWORDS = [
    "city", "skyline", "factory", "bridge", "rocket", "engine", "drone", "lathe", "power plant"]


def entity_keyword_list(config: dict[str, Any]) -> list[str]:
    """``staging.entities.keywords`` (longest phrases first)."""
    kws = config.get("staging", {}).get("entities", {}).get("keywords", DEFAULT_ENTITY_KEYWORDS)
    return sorted({str(k).lower() for k in kws}, key=lambda k: (-len(k.split()), -len(k)))


def _strip(tok: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", tok)


def _is_noun_like(tok: str) -> bool:
    low = tok.lower()
    return bool(len(tok) > 2 and low not in STOPWORDS and low not in FILLER
                and low not in ADJECTIVES and low not in IRREGULAR_VERBS
                and low not in BRAND and not re.search(r"(ed|ly)$", low) and not tok.isdigit())


def find_entity(raw_tokens: list[str], keywords: list[str]) -> tuple[str, bool] | None:
    """Pure entity detector used by the planner (see PlanContext.entity)."""
    toks = [_strip(t) for t in raw_tokens]
    low = [t.lower() for t in toks]
    starts = [i == 0 or bool(re.search(r"[.!?]$", raw_tokens[i - 1])) for i in range(len(toks))]

    def keyword_at(i: int) -> int:
        for kw in keywords:  # longest first
            parts = kw.split()
            if low[i: i + len(parts)] == parts:
                return len(parts)
        return 0

    kw_pos = [(i, n) for i in range(len(toks)) if (n := keyword_at(i))]
    # Capitalised proper-name runs
    names: list[tuple[int, int]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        proper = (bool(t) and t[0].isupper() and t[1:] == t[1:].lower() and len(t) > 2
                  and low[i] not in STOPWORDS and low[i] not in BRAND and low[i] not in FILLER)
        if proper and starts[i]:  # sentence-initial: only a name if a keyword follows it
            proper = i + 1 < len(toks) and keyword_at(i + 1) > 0
        if proper:
            j = i + 1
            while j < len(toks) and j - i < 2 and toks[j][:1].isupper() and toks[j][1:] == \
                    toks[j][1:].lower() and low[j] not in STOPWORDS:
                j += 1
            names.append((i, j))
            i = j
        else:
            i += 1

    def pick_noun(near: int, forward: bool) -> str | None:
        rng = range(near, min(near + 4, len(toks))) if forward \
            else range(near, max(near - 4, -1), -1)
        for k in rng:
            if _is_noun_like(toks[k]) and not (names and any(a <= k < b for a, b in names)):
                return toks[k]
            if toks[k] and re.search(r"[,;:]$", raw_tokens[k]) and forward:
                break
        return None

    if names:
        a, b = names[0]
        name = toks[a:b]
        after = kw_pos and next(((i, n) for i, n in kw_pos if i >= b), None)
        extra: list[str] = []
        if after and after[0] - b <= 3:
            extra = toks[after[0]: after[0] + after[1]]
        else:
            noun = pick_noun(b, True)
            if noun:
                extra = [noun]
            elif kw_pos:
                extra = toks[kw_pos[0][0]: kw_pos[0][0] + kw_pos[0][1]]
        q = [*name, *extra]
        return (" ".join(q[:4]), True) if len(q) >= 2 else None
    if kw_pos:
        i, n = kw_pos[0]
        kw = toks[i: i + n]
        before = toks[i - 1] if i > 0 and (toks[i - 1].isupper() or _is_noun_like(toks[i - 1])) \
            else None
        q = ([before] if before else []) + kw
        if len(q) < 2:
            noun = pick_noun(i + n, True)
            if noun:
                q = [*kw, noun]
        return (" ".join(q[:4]), False) if len(q) >= 2 else None
    return None


def stock_ok_for(template: VisualTemplate, style: BeatStyle) -> bool:
    """Stock is for plain B-roll only: fullscreen (or a cutout background) in a normal beat."""
    return template in (VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.CUTOUT_CALLOUT) \
        and style in (BeatStyle.NORMAL, BeatStyle.DEADPAN)


def clean_query(text: str, max_words: int = 6) -> str:
    """Stock search query: content words only, no punctuation/stopwords, <= 6 words."""
    words: list[str] = []
    for w in re.findall(r"[A-Za-z0-9]+", text):
        if w.lower() not in STOPWORDS and len(w) > 2 and w.lower() not in (x.lower() for x in words):
            words.append(w)
    return " ".join(words[:max_words])


# --------------------------------------------------------------------------- LLM planner

SYSTEM_PROMPT = (
    "You are the editor of the YouTube channel 'Psycho Astronaut'. Style: fast, experimental, "
    "engineering-focused, like Michael Reeves, Stuff Made Here, ElectroBOOM, Colin Furze and "
    "Allen Pan. Turn the voiceover transcript into a JSON edit plan (beats).\n\n"
    "RULES:\n"
    "- Beats must cover the whole voiceover contiguously: start at 0, each start_time equals "
    "the previous end_time, the last end_time equals the total duration.\n"
    "- Brisk pacing: 1.5-3s for fails/hype, 2.5-4.5s normal, 3-5s technical, never above 6s. "
    "Cut on pauses/word boundaries, not mid-word.\n"
    "- beat_style: 'fail' for hardware failures, janky workarounds and mishaps (smoke, "
    "short circuits, duct tape); 'hype' for reveals/successes; 'technical' when explaining "
    "invisible mechanics (signals, latency, code, circuits); 'deadpan' for dry asides; "
    "otherwise 'normal'.\n"
    "- Invisible mechanics -> 'split_schematic' (source_id = a clip or image shown on the left, "
    "overlay_asset = a schematic id shown on the right) or 'cutout_callout' (source_id = "
    "background, overlay_asset = an image id to cut out). Use short labels (2-5 words) in "
    "callout_text. 'punch_in_emphasis' for key moments/fails; 'fullscreen_clip' for B-roll.\n"
    "- Use ONLY asset ids from the list. Prefer local assets; avoid the same source in "
    "consecutive beats. Use primary_source 'stock' (source_id empty, search_queries with 2-6 "
    "plain words) only when no local asset fits.\n"
    "- beat_reason: a short reason for the cut. Return ONLY JSON."
)


def _build_user_prompt(transcript: dict[str, Any], ctx: PlanContext, episode_id: str) -> str:
    assets = "\n".join(
        f"- {a['id']} ({a['asset_type']}"
        + (f", {a['duration']:.1f}s" if a.get("duration") else "")
        + f"): {a.get('description', '')}" for a in ctx.assets) or "  (none)"
    pool = "\n".join(
        f"- {p['id']} ({p.get('media_type', 'image')}, {p.get('license_type')}): "
        f"{p.get('query', '')}" for p in ctx.pool)
    pool_block = (
        "\nFREE STOCK POOL (already downloaded, license-audited; secondary to local footage). "
        "To use one: primary_source \"stock\", stock_asset_id = its id, source_id empty, "
        "search_queries = [its query]. Use for establishing shots / B-roll / cutout backgrounds "
        "of the place or machine it shows:\n" + pool + "\n") if pool else ""
    segs = "\n".join(f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}"
                     for s in transcript.get("segments", []))
    return f"""Episode: {episode_id}
Total voiceover duration: {ctx.duration:.2f}s

TRANSCRIPT SEGMENTS:
{segs}

AVAILABLE LOCAL ASSETS:
{assets}
{pool_block}
Return JSON exactly like:
{{"episode_id": "{episode_id}", "beats": [
  {{"start_time": 0.0, "end_time": 2.8, "visual_template": "fullscreen_clip",
    "primary_source": "local", "source_id": "<asset id>", "overlay_asset": null,
    "callout_text": null, "beat_style": "normal", "emphasis": false,
    "beat_reason": "why this cut", "director_notes": "visual intent", "search_queries": []}}
]}}
visual_template: fullscreen_clip | split_schematic | cutout_callout | punch_in_emphasis.
beat_style: normal | deadpan | hype | technical | fail. primary_source: local | stock."""


def _list_models(base: str) -> list[str] | None:
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base.rstrip('/')}/models")
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def _model_size(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)b", name)
    return float(m.group(1)) if m else 0.0


def pick_model(wanted: str, available: list[str]) -> str | None:
    """Configured model if present; else the largest one of the same family (<=14B, default
    context variants first); else the largest available."""
    if not available:
        return None
    if wanted in available:
        return wanted
    family = wanted.split(":")[0]

    def rank(models: list[str]) -> str:
        plain = [m for m in models if not re.search(r"-\d+k$", m)] or models
        sane = [m for m in plain if _model_size(m) <= 14] or plain
        return max(sane, key=_model_size)

    same = [m for m in available if m.split(":")[0] == family]
    return rank(same or available)


def _extract_json(content: str) -> Any:
    content = content.strip()
    start = min((i for i in (content.find("{"), content.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON object found in LLM response")
    end = max(content.rfind("}"), content.rfind("]"))
    return json.loads(content[start: end + 1])


CHUNK_SYSTEM = (
    "You are the editor of a fast, engineering-focused YouTube channel. For each numbered beat "
    "(a narration line already cut to length) choose how it is shown. Reply with JSON only.")
_TEMPLATES_LLM = {"fullscreen_clip", "punch_in_emphasis"}
_TRANSITIONS_LLM = ("none", "crossfade", "dissolve", "fadeblack", "zoomin", "smoothleft",
                    "slideleft", "wipeleft")
_HARD = {"hype", "fail"}   # energetic beats are cut, not blended
_HARD_OK = {"none", "zoomin"}


def chunk_schema(ids: list[int]) -> dict[str, Any]:
    """JSON schema for one planning chunk (Ollama ``format``): exactly one row per beat id,
    enums for every categorical field."""
    row = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "enum": ids},
            "beat_style": {"type": "string", "enum": [s.value for s in BeatStyle]},
            "visual_template": {"type": "string", "enum": sorted(_TEMPLATES_LLM)},
            "transition": {"type": "string", "enum": list(_TRANSITIONS_LLM)},
            "callout": {"type": ["string", "null"]},
            "emphasis": {"type": "boolean"},
        },
        "required": ["id", "beat_style", "visual_template", "transition", "callout",
                     "emphasis"],
    }
    return {"type": "object", "required": ["beats"], "properties": {
        "beats": {"type": "array", "items": row, "minItems": len(ids), "maxItems": len(ids)}}}


def _chunk_prompt(group: list[SceneBeat], texts: dict[int, str],
                  prev: list[str] | None = None, hint: str = "") -> str:
    lines = "\n".join(
        f'{b.id}: "{texts[b.id]}" (now: {b.beat_style.value}, {b.visual_template.value})'
        for b in group)
    before = (f"The previous beats used templates: {', '.join(prev)}. " if prev else "")
    return (
        "beat_style: fail (mishap/failure), hype (reveal/success), technical (explaining "
        "mechanics/code/signals), deadpan (dry aside) or normal. visual_template: "
        "fullscreen_clip (b-roll) or punch_in_emphasis (key moment, at most 2 per 5 beats). "
        "transition = how the cut INTO that beat looks: none (hard cut; DEFAULT, use for "
        "most beats), crossfade or dissolve (rare, only between two long calm beats), fadeblack or zoomin "
        "(a new topic), smoothleft/slideleft/wipeleft (rare). "
        + (hint or "callout: 2-4 word Title Case noun-phrase label of a technical term from "
           "the line, or null. ") + "\n"
        "VARIETY: never use the same visual_template 3 beats in a row; use at least 2 "
        "different transitions in the group and never the same transition more than 3 times; "
        f"judge each line on its own meaning. {before}\n"
        'Return {"beats":[{"id":0,"beat_style":"normal","visual_template":"fullscreen_clip",'
        '"transition":"none","callout":null,"emphasis":false}]} with one entry per line, '
        "same ids.\n" + lines)


def _variety_problem(tmpls: list[str], trans: list[str], prev: list[str]) -> str:
    """'' when the choices are varied enough, else a message to feed back to the model."""
    seq = [*prev[-2:], *tmpls]
    for i in range(len(seq) - 2):
        if seq[i] == seq[i + 1] == seq[i + 2]:
            return (f"visual_template '{seq[i]}' is used 3 beats in a row; "
                    "alternate fullscreen_clip and punch_in_emphasis")
    if tmpls.count("punch_in_emphasis") > max(2, (len(tmpls) * 2) // 5):
        return "too many punch_in_emphasis beats; use it for at most 2 of 5 beats"
    if len(trans) >= 4 and len(set(trans)) < 2:
        return f"every transition is '{trans[0]}'; use at least 2 different transitions"
    worst = max((trans.count(t) for t in set(trans) if t != "none"), default=0)  # cuts are fine
    if len(trans) >= 4 and worst > 3:
        return "no transition may be used more than 3 times in a group"
    return ""


def _apply_chunk(group: list[SceneBeat], texts: dict[int, str], data: Any,
                 prev: list[str] | None = None, final: bool = False,
                 callouts: list[tuple[float, float, str]] | None = None) -> str:
    """Validate the model's rows for ``group`` and apply them; returns a problem or ''.
    A variety problem on the ``final`` attempt is repaired deterministically instead."""
    prev = prev or []
    rows = data.get("beats") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return 'expected {"beats": [...]}'
    by_id: dict[int, dict[str, Any]] = {}
    for r in rows:
        try:
            by_id[int(r["id"])] = r
        except (KeyError, TypeError, ValueError):
            return "every row needs an integer id"
    want = {b.id for b in group}
    if set(by_id) != want:
        return f"ids must be exactly {sorted(want)}, got {sorted(by_id)}"
    styles = {s.value for s in BeatStyle}
    for b in group:
        r = by_id[b.id]
        if str(r.get("beat_style", "")).lower() not in styles:
            return f"beat {b.id}: beat_style must be one of {sorted(styles)}"
        if "transition" in r and str(r["transition"]).lower() not in _TRANSITIONS_LLM:
            return f"beat {b.id}: transition must be one of {list(_TRANSITIONS_LLM)}"
    style_of = {b.id: str(by_id[b.id]["beat_style"]).lower() for b in group}
    tmpl_of: dict[int, str] = {}
    trans_of: dict[int, str] = {}
    for b in group:
        r = by_id[b.id]
        t = str(r.get("visual_template", ""))
        tmpl_of[b.id] = t if t in _TEMPLATES_LLM and b.visual_template.value in _TEMPLATES_LLM \
            else b.visual_template.value
        tr = str(r.get("transition", "")).lower()
        if style_of[b.id] in _HARD and tr not in _HARD_OK:
            tr = "none"   # energetic beats are cut
        trans_of[b.id] = tr
    tmpls = [tmpl_of[b.id] for b in group]
    trans = [trans_of[b.id] for b in group if trans_of.get(b.id)]
    problem = _variety_problem(tmpls, trans, prev)
    if problem and not final:
        return problem
    if problem:
        _repair_variety(group, tmpl_of, trans_of, style_of, prev)
    for b in group:
        r = by_id[b.id]
        b.beat_style = BeatStyle(style_of[b.id])
        if tmpl_of[b.id] in _TEMPLATES_LLM and b.visual_template.value in _TEMPLATES_LLM:
            b.visual_template = VisualTemplate(tmpl_of[b.id])
        if trans_of.get(b.id):
            b.transition = trans_of[b.id]
        co = r.get("callout", r.get("callout_text"))
        if isinstance(co, str) and callouts is not None and 2 <= len(co.split()) <= 5:
            # (start, end, text): beat ids are renumbered by the repair pass; ranked later
            callouts.append((b.start_time, b.end_time, " ".join(co.split())))
        if isinstance(r.get("emphasis"), bool):
            b.emphasis = r["emphasis"]
    return ""


def _repair_variety(group: list[SceneBeat], tmpl_of: dict[int, str], trans_of: dict[int, str],
                    style_of: dict[int, str], prev: list[str]) -> None:
    """Deterministic last resort for a low-variety answer from a small model."""
    seq = list(prev[-2:])
    for b in group:
        t = tmpl_of[b.id]
        if len(seq) >= 2 and seq[-1] == seq[-2] == t and t in _TEMPLATES_LLM \
                and b.visual_template.value in _TEMPLATES_LLM:
            t = "fullscreen_clip" if t == "punch_in_emphasis" else "punch_in_emphasis"
            tmpl_of[b.id] = t
        seq.append(t)
    ids = [b.id for b in group]
    if len(ids) >= 4:
        counts: dict[str, int] = {}
        for i in ids:
            counts[trans_of[i]] = counts.get(trans_of[i], 0) + 1
        for n, i in enumerate(ids):
            if trans_of[i] != "none" and counts[trans_of[i]] > 2:
                if style_of[i] in _HARD:
                    continue
                alt = ("dissolve", "crossfade", "fadeblack")[n % 3]
                counts[trans_of[i]] -= 1
                trans_of[i] = alt if alt != trans_of[i] else "none"
                counts[trans_of[i]] = counts.get(trans_of[i], 0) + 1
                if len(set(trans_of.values())) >= 2 and max(
                        (c for t, c in counts.items() if t != "none"), default=0) <= 3:
                    break


def _try_llm_plan(transcript: dict[str, Any], ctx: PlanContext, episode_id: str
                  ) -> ScenePlan | None:
    """Chunked LLM planning.

    The deterministic planner fixes the cuts (timing is never the model's job); the local LLM
    then decides style/template/transition/callout for 4-6 beats per call (schema-constrained
    on Ollama, small JSON, per-chunk timeout, retry with the validation error). A chunk that
    fails keeps its deterministic values, so one bad answer never costs the whole plan.
    Returns None only when no LLM is reachable.
    """
    if is_offline():
        return None
    llm = ctx.config.get("llm", {})
    candidates = [os.environ.get("LLM_BASE_URL") or llm.get("base_url", "")]
    if not (os.environ.get("LLM_BASE_URL") or llm.get("base_url")):
        candidates = ["http://localhost:11434/v1", "http://localhost:8080/v1"]
    base, models = "", None
    for url in candidates:
        if not url:
            continue
        models = _list_models(url)
        if models is not None:
            base = url.rstrip("/")
            break
    if not base or not models:
        console.print("  [dim]No LLM endpoint available[/]")
        return None
    wanted = os.environ.get("LLM_MODEL") or llm.get("model", "qwen2.5:14b-instruct")
    model = pick_model(wanted, models)
    if model is None:
        return None
    if model != wanted:
        console.print(f"  [yellow]LLM model '{wanted}' not available; using '{model}'[/]")
    mode = "structured" if native_api_url(base, llm) else "json_object"
    console.print(f"  LLM: {base} model={model} (chunked planning, {mode})")

    plan = _generate_deterministic_plan(ctx, episode_id)
    texts = {b.id: ctx.text_between(b.start_time, b.end_time) for b in plan.beats}
    from src.callouts import prompt_hint

    hint = prompt_hint(texts.values(), ctx.vocab)
    size = max(1, int(llm.get("chunk_beats", 5)))
    retries = int(llm.get("chunk_retries", 2))
    per_chunk = float(llm.get("chunk_timeout", min(float(llm.get("timeout", 150)), 60.0)))
    deadline = time.monotonic() + float(llm.get("plan_budget", 240))
    ok = failed = streak = rejected = 0
    started = time.monotonic()
    for i in range(0, len(plan.beats), size):
        group = plan.beats[i: i + size]
        if streak >= 3 or time.monotonic() >= deadline:
            failed += 1
            continue
        prev = [b.visual_template.value for b in plan.beats[max(i - 2, 0): i]]
        schema = chunk_schema([b.id for b in group])
        messages: list[dict[str, str]] = [
            {"role": "system", "content": CHUNK_SYSTEM},
            {"role": "user", "content": _chunk_prompt(group, texts, prev, hint)}]
        done = False
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=httpx.Timeout(per_chunk, connect=5.0)) as client:
                    content = llm_chat(
                        client, base, llm, model, messages, schema, max_tokens=700,
                        temperature=float(llm.get("temperature", 0.3)),
                        deadline=min(deadline, time.monotonic() + per_chunk + 1),
                        request_timeout=per_chunk)
            except (httpx.HTTPError, KeyError, ValueError) as e:
                console.print(f"  [yellow]chunk {i // size + 1}: LLM request failed "
                              f"({type(e).__name__})[/]")
                break
            try:
                problem = _apply_chunk(group, texts, _extract_json(content), prev,
                                       final=attempt >= min(retries, 1),
                                       callouts=ctx.llm_callouts)
            except ValueError as e:
                problem = str(e)[:300]
            if not problem:
                done = True
                break
            rejected += 1
            console.print(f"  [yellow]chunk {i // size + 1} attempt {attempt + 1} rejected: "
                          f"{problem[:120]}[/]")
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"Rejected: {problem}\n"
                          "Return the corrected JSON only."}]
        ok += done
        failed += not done
        console.print(f"  chunk {i // size + 1}/{-(-len(plan.beats) // size)} "
                      f"{'ok' if done else 'fallback'}")
        streak = 0 if done else streak + 1
    if plan.beats:
        plan.beats[0].transition = "none"
    console.print(f"  LLM chunks: {ok} ok, {failed} fell back to the deterministic plan; "
                  f"{rejected} rejected answers; {time.monotonic() - started:.0f}s")
    return ScenePlan(episode_id=episode_id, beats=plan.beats)


_NATIVE_BAD: set[str] = set()   # native base URLs that answered 404/405: use /v1 for the run


def native_api_url(base: str, llm_cfg: dict[str, Any]) -> str | None:
    """Ollama-native base ('http://host:11434') for structured output, or None when disabled
    or already known to be unsupported. Derived from the OpenAI-style base unless
    ``llm.native_api`` / ``LLM_NATIVE_API`` says otherwise."""
    if not llm_cfg.get("structured", True):
        return None
    explicit = os.environ.get("LLM_NATIVE_API") or llm_cfg.get("native_api")
    url = str(explicit).rstrip("/") if explicit else re.sub(r"/v1/?$", "", base.rstrip("/"))
    if not url or url == base.rstrip("/") or url in _NATIVE_BAD:
        return None
    return url


def native_payload(model: str, messages: list[dict[str, str]], schema: dict[str, Any],
                   llm_cfg: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    """Ollama /api/chat body: schema-constrained decoding (``format``), greedy, fixed ctx."""
    return {"model": model, "messages": messages, "stream": False, "format": schema,
            "keep_alive": llm_cfg.get("keep_alive", "10m"),
            "options": {"temperature": 0, "num_ctx": int(llm_cfg.get("num_ctx", 4096)),
                        "num_predict": max_tokens}}


def llm_chat(client: Any, base: str, llm_cfg: dict[str, Any], model: str,
             messages: list[dict[str, str]], schema: dict[str, Any] | None, *,
             max_tokens: int, temperature: float, deadline: float | None,
             request_timeout: float) -> str:
    """One chat call. With a ``schema`` and an Ollama server the native endpoint constrains
    the output to the schema; any other server (or a native 404) uses the OpenAI-compatible
    /v1/chat/completions with ``response_format=json_object`` as before."""
    native = native_api_url(base, llm_cfg) if schema else None
    if native and schema:
        try:
            return chat_with_backoff(
                client, f"{native}/api/chat",
                native_payload(model, messages, schema, llm_cfg, max_tokens), 0,
                deadline=deadline, request_timeout=request_timeout,
                extract=lambda body: body["message"]["content"])
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (400, 404, 405, 501):
                raise
            _NATIVE_BAD.add(native)
            console.print("  [dim]native structured output unavailable; using /v1[/]")
    payload: dict[str, Any] = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    return chat_with_backoff(client, f"{base}/chat/completions", payload, 0,
                             deadline=deadline, request_timeout=request_timeout)


TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


def chat_with_backoff(client: Any, url: str, payload: dict[str, Any], retries: int,
                      sleep: Callable[[float], None] = time.sleep, *,
                      deadline: float | None = None, request_timeout: float | None = None,
                      clock: Callable[[], float] = time.monotonic,
                      extract: Callable[[Any], str] | None = None) -> str:
    """POST a chat completion; transient failures (timeouts, connection errors, 5xx/429) are
    retried up to ``retries`` times with a short backoff (2s, 5s, 10s...). Raises the last
    error when exhausted; non-transient 4xx errors are raised immediately.

    ``deadline`` (a ``clock()`` value) is an overall budget: each request's timeout is capped
    to the time left and no retry starts once the budget cannot cover it.
    """
    delays = [2.0, 5.0, 10.0]
    last: Exception | None = None
    for attempt in range(retries + 1):
        kwargs: dict[str, Any] = {}
        if deadline is not None:
            left = deadline - clock()
            if left <= 1.0:
                raise last or httpx.ReadTimeout("LLM total time budget exhausted")
            kwargs["timeout"] = min(request_timeout, left) if request_timeout else left
        try:
            resp = client.post(url, json=payload, **kwargs)
            resp.raise_for_status()
            body = resp.json()
            return extract(body) if extract else body["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in TRANSIENT_STATUS:
                raise
            last = e
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e
        if attempt < retries:
            delay = delays[min(attempt, len(delays) - 1)]
            if deadline is not None and deadline - clock() <= delay + 1.0:
                break
            console.print(f"  [yellow]LLM transient error ({last}); retry {attempt + 1}/"
                          f"{retries} in {delay:.0f}s[/]")
            sleep(delay)
    assert last is not None
    raise last


def _semantic_problem(plan: ScenePlan, ctx: PlanContext) -> str:
    if not plan.beats:
        return "the plan has no beats"
    covered = sum(max(b.end_time - b.start_time, 0) for b in plan.beats)
    if ctx.duration and covered < 0.7 * ctx.duration:
        return (f"beats only cover {covered:.1f}s of the {ctx.duration:.1f}s voiceover; "
                "they must cover all of it contiguously")
    return ""


# --------------------------------------------------------------------------- deterministic planner


@dataclass
class Chunk:
    start: float
    end: float
    words: list[Word]
    style: BeatStyle = BeatStyle.NORMAL
    keyword: str = ""
    technical: bool = False
    hold: bool = False
    kws: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _cut_candidates(words: list[Word]) -> list[tuple[int, float, float]]:
    """(index of first word after the cut, cut time, weight) for every word boundary."""
    out = []
    for i in range(1, len(words)):
        prev, nxt = words[i - 1], words[i]
        gap = max(nxt.start - prev.end, 0.0)
        t = prev.end + gap / 2
        weight = gap
        if re.search(r"[.!?]$", prev.text):
            weight += 1.0
        elif re.search(r"[,;:—-]$", prev.text):
            weight += 0.3
        out.append((i, t, weight))
    return out


def _sentence_units(words: list[Word]) -> list[list[Word]]:
    units: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?]$", w.text) or (len(cur) > 1 and w.start - cur[-2].end > 0.8):
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return units


def _split_unit(words: list[Word], t0: float, t1: float, lo: float, hi: float, min_beat: float
                ) -> list[tuple[float, float, list[Word]]]:
    """Split one unit (time span t0..t1) into pieces near [lo, hi] seconds on word boundaries."""
    span = t1 - t0
    n = max(1, int(-(-span // hi)))  # ceil
    if n > 1 and span / n < max(lo, min_beat):
        n = max(1, int(span // max(lo, min_beat)))
    if n <= 1 or len(words) < 2:
        return [(t0, t1, words)]
    cands = _cut_candidates(words)
    cuts: list[int] = []
    prev_t = t0
    for k in range(1, n):
        target = t0 + span * k / n
        window = span / n * 0.35
        best = None
        for idx, t, weight in cands:
            if idx in cuts or abs(t - target) > window or t - prev_t < min_beat \
                    or t1 - t < min_beat:
                continue
            score = weight - abs(t - target) * 0.8
            if best is None or score > best[0]:
                best = (score, idx, t)
        if best is None:
            continue
        cuts.append(best[1])
        prev_t = best[2]
    cuts.sort()
    if not cuts:
        return [(t0, t1, words)]
    time_of = {idx: t for idx, t, _ in cands}
    pieces = []
    bounds = [0, *cuts, len(words)]
    times = [t0, *[time_of[i] for i in cuts], t1]
    for a, b, ta, tb in zip(bounds[:-1], bounds[1:], times[:-1], times[1:], strict=True):
        pieces.append((ta, tb, words[a:b]))
    return pieces


def _build_chunks(ctx: PlanContext) -> list[Chunk]:
    words = ctx.words
    if not words:
        return []
    units = _sentence_units(words)
    chunks: list[Chunk] = []
    unit_start = 0.0
    for ui, unit in enumerate(units):
        if ui == len(units) - 1:
            unit_end = ctx.duration
        else:
            nxt = units[ui + 1][0]
            unit_end = unit[-1].end + max(nxt.start - unit[-1].end, 0.0) / 2
        text = " ".join(w.text for w in unit)
        style, _, technical, hold = ctx.classify(text)
        lo, hi = ctx.target_range(style, hold)
        hi = min(hi, ctx.max_beat)
        carry: tuple[BeatStyle, str] | None = None
        for ta, tb, ws in _split_unit(unit, unit_start, unit_end, lo, hi, ctx.min_beat):
            piece_text = " ".join(w.text for w in ws)
            p_style, p_kw, p_tech, p_hold = ctx.classify(piece_text)
            if p_style != BeatStyle.NORMAL:
                carry = (p_style, p_kw)
            elif carry is not None:
                # the piece right after a keyword piece continues the same moment
                p_style, p_kw = carry
            chunks.append(Chunk(ta, tb, ws, p_style, p_kw, p_tech or technical, p_hold or hold))
        unit_start = unit_end
    return _merge_short(chunks, ctx)


def _merge_short(chunks: list[Chunk], ctx: PlanContext) -> list[Chunk]:
    """Merge beats shorter than ``min_beat`` into a neighbour (never beyond ``max_beat``)."""
    merged = list(chunks)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, c in enumerate(merged):
            if c.duration >= ctx.min_beat:
                continue
            options = []
            if i > 0:
                options.append((i - 1, i))
            if i < len(merged) - 1:
                options.append((i, i + 1))
            options = [(a, b) for a, b in options
                       if merged[b].end - merged[a].start <= ctx.max_beat + 1.0]
            if not options:
                continue
            a, b = min(options, key=lambda ab: merged[ab[1]].end - merged[ab[0]].start)
            ca, cb = merged[a], merged[b]
            winner = min((ca, cb), key=lambda x: STYLE_RANK[x.style])
            merged[a:b + 1] = [Chunk(ca.start, cb.end, ca.words + cb.words, winner.style,
                                     winner.keyword, ca.technical or cb.technical,
                                     ca.hold or cb.hold)]
            changed = True
            break
    return merged


STYLE_RANK = {BeatStyle.FAIL: 0, BeatStyle.HYPE: 1, BeatStyle.DEADPAN: 2,
              BeatStyle.TECHNICAL: 3, BeatStyle.NORMAL: 4}


@dataclass
class _PlannerState:
    templates: list[VisualTemplate] = field(default_factory=list)
    used: dict[str, int] = field(default_factory=dict)
    last_sources: set[str] = field(default_factory=set)
    stock_count: int = 0
    last_stock_index: int = -10
    now: float = 0.0  # start time of the beat being planned
    pool_used: set[str] = field(default_factory=set)  # stock-pool items already placed
    stride: int = 3  # minimum beats between thematic stock beats (spreads them out)
    style: BeatStyle = BeatStyle.NORMAL  # style of the beat being planned
    uses: list[tuple[float, str]] = field(default_factory=list)  # (beat start, asset id)

    def recent(self, asset_id: str) -> int:
        """Times ``asset_id`` was used in the last ~20s before ``now``."""
        return sum(1 for t, i in self.uses if i == asset_id and 0 <= self.now - t <= RECENT_WINDOW)


def _mood_penalty(ctx: PlanContext, asset: dict[str, Any], style: BeatStyle) -> int:
    """A smoke/fail-tagged clip must not illustrate a success/hype/normal line."""
    if style not in (BeatStyle.HYPE, BeatStyle.NORMAL):
        return 0
    stems = {_stem(t) for t in asset.get("tags", [])}
    stems.update(_stem(t) for t in re.findall(r"[A-Za-z0-9]+", asset.get("description", "")))
    return 3 if stems & ctx.fail_words else 0


def _pick_asset(ctx: PlanContext, types: tuple[str, ...], text: str, state: _PlannerState,
                exclude: set[str] | None = None) -> tuple[dict[str, Any] | None, int]:
    """Best asset of the given types: relevance first, then least used, avoid recent repeats."""
    cands = [a for a in ctx.of_type(*types) if not exclude or a["id"] not in exclude]
    if not cands:
        return None, 0
    fresh = [a for a in cands if a["id"] not in state.last_sources] or cands
    best = max(fresh, key=lambda a: (asset_score(a, text) - 2 * state.recent(a["id"])
                                     - _mood_penalty(ctx, a, state.style),
                                     -state.used.get(a["id"], 0), -ctx.assets.index(a)))
    return best, asset_score(best, text)


def _generate_deterministic_plan(ctx: PlanContext, episode_id: str) -> ScenePlan:
    """Plan from word timings + config keywords; cuts land on real pauses, no drift."""
    chunks = _build_chunks(ctx) or [Chunk(0.0, ctx.duration, [])]
    state = _PlannerState()
    beats: list[SceneBeat] = []
    has_entities = any(ctx.entity(c.words) for c in chunks)
    ratio = 0.0 if ctx.stock_ratio <= 0 else (  # stock_ratio 0 switches all stock off
        max(ctx.stock_ratio, ctx.entity_ratio, ctx.pool_ratio if ctx.pool else 0.0)
        if has_entities or ctx.pool else ctx.stock_ratio)
    stock_budget = round(ratio * len(chunks) + 0.01)
    state.stride = max(2, round(len(chunks) / max(stock_budget, 1)))
    has = {t: bool(ctx.of_type(t)) for t in ("clip", "image", "schematic")}
    visual_types = tuple(t for t in ("clip", "image") if has[t])

    for i, chunk in enumerate(chunks):
        style = chunk.style
        # --- template preference
        if chunk.hold:
            prefs = [VisualTemplate.SPLIT_SCHEMATIC, VisualTemplate.CUTOUT_CALLOUT]
        elif chunk.technical or style == BeatStyle.TECHNICAL:
            prefs = [VisualTemplate.SPLIT_SCHEMATIC, VisualTemplate.CUTOUT_CALLOUT,
                     VisualTemplate.FULLSCREEN_CLIP]
        elif style == BeatStyle.FAIL:
            prefs = [VisualTemplate.PUNCH_IN_EMPHASIS, VisualTemplate.CUTOUT_CALLOUT,
                     VisualTemplate.FULLSCREEN_CLIP]
        elif style == BeatStyle.HYPE:
            prefs = [VisualTemplate.PUNCH_IN_EMPHASIS, VisualTemplate.FULLSCREEN_CLIP,
                     VisualTemplate.CUTOUT_CALLOUT]
        else:
            prefs = [VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.CUTOUT_CALLOUT,
                     VisualTemplate.PUNCH_IN_EMPHASIS]
        available = []
        for t in prefs:
            if t == VisualTemplate.SPLIT_SCHEMATIC and not (has["schematic"] and visual_types):
                continue
            if t == VisualTemplate.CUTOUT_CALLOUT and not has["image"]:
                continue
            available.append(t)
        if not available:
            available = [VisualTemplate.FULLSCREEN_CLIP]
        last = state.templates[-1] if state.templates else None
        last2 = state.templates[-2:] if len(state.templates) >= 2 else []
        blocked: set[VisualTemplate] = set()
        if last is not None and last != VisualTemplate.FULLSCREEN_CLIP:
            blocked.add(last)
        if len(last2) == 2 and last2[0] == last2[1]:
            blocked.add(last2[0])
        template = next((t for t in available if t not in blocked), available[0])

        beat = _make_beat(ctx, chunk, template, state, i, stock_budget)
        beats.append(beat)
        state.templates.append(beat.visual_template)
        state.last_sources = {x for x in (beat.source_id, beat.overlay_asset) if x}
        state.uses.extend((beat.start_time, x) for x in state.last_sources)
        for x in state.last_sources:
            state.used[x] = state.used.get(x, 0) + 1
        if beat.primary_source == PrimarySource.STOCK:
            state.stock_count += 1
            state.last_stock_index = i
    _ensure_stock(ctx, beats)
    for b in beats:
        _attach_pool(ctx, b, ctx.text_between(b.start_time, b.end_time), state.pool_used)
    return ScenePlan(episode_id=episode_id, beats=beats)


def _ensure_stock(ctx: PlanContext, beats: list[SceneBeat]) -> None:
    """With stock enabled (stock_ratio > 0) at least one plain B-roll beat uses stock when some
    normal line has a good (>= 2 nouns) query and no relevant local asset."""
    if ctx.stock_ratio <= 0 or any(b.primary_source == PrimarySource.STOCK for b in beats):
        return
    best: tuple[int, int, SceneBeat, str] | None = None
    for i, b in enumerate(beats):
        if i == 0 or b.beat_style != BeatStyle.NORMAL or b.primary_source != PrimarySource.LOCAL \
                or b.visual_template == VisualTemplate.SPLIT_SCHEMATIC:
            continue
        text = ctx.text_between(b.start_time, b.end_time)
        query = ctx.stock_query(text)
        if not query or max((asset_score(a, text) for a in ctx.of_type("clip", "image")),
                            default=0) > 0:
            continue
        natural = b.visual_template in (VisualTemplate.FULLSCREEN_CLIP,
                                        VisualTemplate.CUTOUT_CALLOUT)
        rank = (int(natural), len(query.split()))
        if best is None or rank > (best[0], best[1]):
            best = (rank[0], rank[1], b, query)
    if best is None:
        return
    b, query = best[2], best[3]
    if b.visual_template == VisualTemplate.CUTOUT_CALLOUT and b.overlay_asset:
        b.source_id = ""  # licensed stock behind the local subject
    else:
        b.visual_template = VisualTemplate.FULLSCREEN_CLIP
        b.overlay_asset, b.source_id = None, ""
    b.primary_source = PrimarySource.STOCK
    b.search_queries = [query]
    b.callout_text = None if b.visual_template == VisualTemplate.FULLSCREEN_CLIP \
        else b.callout_text
    b.beat_reason = (b.beat_reason + "; stock B-roll for a line no local asset covers").strip("; ")


def _thematic_ok(ctx: PlanContext, state: _PlannerState, index: int, budget: int,
                 style: BeatStyle, weak_local: bool, background: bool = False) -> bool:
    """On-theme pool image instead of a weakly-relevant (or just reused) local asset."""
    return (bool(ctx.pool) and weak_local and index > 0 and state.stock_count < budget
            and index - state.last_stock_index >= state.stride
            and (style in (BeatStyle.NORMAL, BeatStyle.DEADPAN)
                 or (background and style == BeatStyle.TECHNICAL)))  # neutral backdrop only


def _attach_pool(ctx: PlanContext, beat: SceneBeat, text: str, used: set[str]) -> None:
    """Resolve a stock beat from the pre-fetched pool (so Stage 4 only verifies it)."""
    if beat.primary_source != PrimarySource.STOCK or not ctx.pool:
        return
    if beat.stock_asset_id and any(p["id"] == beat.stock_asset_id for p in ctx.pool):
        used.add(beat.stock_asset_id)
        return
    beat.stock_asset_id = beat.media_path = beat.license_type = None
    from src.stockpool import match_pool

    item = match_pool(ctx.pool, [*beat.search_queries, ctx.stock_query(text)],
                      images_only=beat.stock_media == "image", exclude=used)
    if item is None:
        return
    used.add(item["id"])
    beat.stock_asset_id = item["id"]
    beat.media_path = item["local_path"]
    beat.license_type = item.get("license_type")
    beat.search_queries = [item["query"]]  # Stage 4 reuses the pool item when queries match
    if not item.get("exact", True) and beat.visual_template == VisualTemplate.FULLSCREEN_CLIP:
        beat.callout_text = None  # a fallback-rung image must not be captioned as the entity


def _stock_allowed(ctx: PlanContext, text: str, state: _PlannerState, index: int,
                   budget: int) -> bool:
    return (state.stock_count < budget and index - state.last_stock_index >= 3 and index > 0
            and bool(ctx.stock_query(text)))


def _make_beat(ctx: PlanContext, chunk: Chunk, template: VisualTemplate, state: _PlannerState,
               index: int, stock_budget: int) -> SceneBeat:
    text = chunk.text
    style = chunk.style
    state.now = chunk.start
    state.style = style
    visual_types = tuple(t for t in ("clip", "image") if ctx.of_type(t))
    source_id, overlay, primary = "", None, PrimarySource.LOCAL
    queries: list[str] = []
    callout: str | None = None
    stock_media = "any"
    stock_ref: dict[str, Any] | None = None

    # Contextual B-roll: a real photo of the place/machine the line is about (never a punch
    # payoff, and fail/hype lines only when no local asset is relevant).
    entity = ctx.entity(chunk.words)
    best_local = max((asset_score(a, text) for a in ctx.of_type("clip", "image")), default=0)
    entity_ok = (
        entity is not None and index > 0 and state.stock_count < stock_budget
        and template in (VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.CUTOUT_CALLOUT,
                         VisualTemplate.SPLIT_SCHEMATIC)
        and not (style in (BeatStyle.FAIL, BeatStyle.HYPE) and best_local > 0)
        and (best_local == 0 or (entity is not None and entity[1])))  # named places beat tags

    if template == VisualTemplate.SPLIT_SCHEMATIC:
        schem, _ = _pick_asset(ctx, ("schematic",), text, state)
        left, _ = _pick_asset(ctx, visual_types, text, state,
                              exclude={schem["id"]} if schem else None)
        source_id = left["id"] if left else (schem["id"] if schem else "")
        overlay = schem["id"] if schem and left else None
        callout = ctx.subject_callout(text, chunk.keyword, schem)
        notes = "Left: footage; right: schematic that shows what the narration describes"
        if entity_ok and entity and schem:
            primary, source_id, overlay = PrimarySource.STOCK, "", schem["id"]
            queries, stock_media = [entity[0]], "image"
            notes = f"Left: photo of {entity[0]}; right: schematic of how it works"
    elif template == VisualTemplate.CUTOUT_CALLOUT:
        subject, _ = _pick_asset(ctx, ("image",), text, state)
        bg, bg_score = _pick_asset(ctx, ("clip",), text, state)
        source_id = (bg or subject or {}).get("id", "")
        overlay = subject["id"] if subject and bg else None
        if subject and entity_ok and entity:
            # establishing photo of the named place/machine behind the local subject
            primary, source_id, overlay = PrimarySource.STOCK, "", subject["id"]
            queries, stock_media = [entity[0]], "image"
        elif subject and _thematic_ok(
                ctx, state, index, stock_budget, style,
                bg is None or bg_score <= 1 or (bg_score <= 2 and state.recent(bg["id"]) > 0),
                background=True) \
                and (item := ctx.pick_pool(text, state.pool_used)) is not None:
            state.pool_used.add(item["id"])
            primary, source_id, overlay = PrimarySource.STOCK, "", subject["id"]
            queries = [item["query"]]
            stock_media = "image" if item.get("media_type", "image") == "image" else "any"
            stock_ref = item
        elif subject and (bg is None or bg_score == 0) and stock_ok_for(template, style) \
                and _stock_allowed(ctx, text, state, index, stock_budget):
            # no relevant local background: use licensed stock behind the local subject
            primary, source_id, overlay = PrimarySource.STOCK, "", subject["id"]
            queries = [ctx.stock_query(text)]
        callout = ctx.subject_callout(text, chunk.keyword, subject or bg)
        notes = "Cut-out subject over blurred footage with a short label"
    else:
        pool = ("clip", "image") if template == VisualTemplate.FULLSCREEN_CLIP else \
            ("image", "clip")
        pool = tuple(t for t in pool if ctx.of_type(t)) or ("schematic",)
        asset, score = _pick_asset(ctx, pool, text, state)
        # Stock is for plain B-roll only: never for fail/hype/punch payoffs, and never when any
        # local asset is relevant (even one used recently).
        best_any = max((asset_score(a, text) for a in ctx.of_type(*pool)), default=0)
        if entity_ok and entity and template == VisualTemplate.FULLSCREEN_CLIP:
            primary, queries, stock_media = PrimarySource.STOCK, [entity[0]], "image"
            callout = entity[0]  # the lower-third names what is on screen
            notes = f"Establishing photo of {entity[0]} (free-license stock image)"
        elif template == VisualTemplate.FULLSCREEN_CLIP and _thematic_ok(
                ctx, state, index, stock_budget, style,
                asset is None or score == 0 or (score <= 1 and state.recent(asset["id"]) > 0)) \
                and (item := ctx.pick_pool(text, state.pool_used)) is not None:
            state.pool_used.add(item["id"])
            primary, queries = PrimarySource.STOCK, [item["query"]]
            stock_media = "image" if item.get("media_type", "image") == "image" else "any"
            stock_ref = item
            notes = f"On-theme free-license B-roll: {item['query']}"
        else:
            wants_stock = asset is None or (
                stock_ok_for(template, style) and score == 0 and best_any == 0
                and _stock_allowed(ctx, text, state, index, stock_budget))
            if wants_stock and ctx.stock_query(text):
                primary = PrimarySource.STOCK
                queries = [ctx.stock_query(text)]
                notes = "No local asset matches this line; licensed stock (CC0/PD) or fallback card"
            else:
                source_id = asset["id"] if asset else ""
                notes = ("Punch-in to land the moment" if template ==
                         VisualTemplate.PUNCH_IN_EMPHASIS else "Footage that illustrates the line")
        if style in (BeatStyle.FAIL, BeatStyle.TECHNICAL, BeatStyle.HYPE) \
                and chunk.duration >= 2.0 and stock_media != "image":
            # label from the narration only, never a local asset's name
            callout = ctx.mishap_callout(text, chunk.keyword)

    reason = {
        BeatStyle.FAIL: f"mishap ('{chunk.keyword}'): punchy visual + red accent",
        BeatStyle.HYPE: f"payoff ('{chunk.keyword}'): fast, energetic move",
        BeatStyle.DEADPAN: f"dry aside ('{chunk.keyword}'): held static shot",
        BeatStyle.TECHNICAL: f"invisible mechanics ('{chunk.keyword}'): show the diagram",
        BeatStyle.NORMAL: "narration beat: B-roll with gentle movement",
    }[style]
    if chunk.hold and style == BeatStyle.NORMAL:
        reason = "diagram/schematic mentioned: hold longer so it can be read"
    return SceneBeat(
        start_time=chunk.start, end_time=chunk.end, visual_template=template,
        primary_source=primary, source_id=source_id, overlay_asset=overlay,
        callout_text=callout, beat_style=style, beat_reason=reason, director_notes=notes,
        search_queries=queries, stock_media=stock_media,
        stock_asset_id=stock_ref["id"] if stock_ref else None,
        media_path=stock_ref["local_path"] if stock_ref else None,
        license_type=stock_ref.get("license_type") if stock_ref else None,
        emphasis=style in (BeatStyle.FAIL, BeatStyle.HYPE) and template !=
        VisualTemplate.PUNCH_IN_EMPHASIS,
    )


# --------------------------------------------------------------------------- validation / repair


def _snap_to_words(t: float, ctx: PlanContext, tol: float = 0.35) -> float:
    """Snap a cut to the nearest word-gap midpoint within ``tol`` seconds."""
    best, best_d = t, tol
    for i in range(1, len(ctx.words)):
        a, b = ctx.words[i - 1], ctx.words[i]
        mid = (a.end + max(b.start, a.end)) / 2
        if abs(mid - t) < best_d:
            best, best_d = mid, abs(mid - t)
    return best


def _compatible_assets(ctx: PlanContext, beat: SceneBeat, text: str, used_before: set[str]
                       ) -> None:
    """Make asset references exist and fit the template (in place)."""
    ids = ctx.assets_by_id
    tmpl = beat.visual_template
    visual_types = ("clip", "image")
    state = _PlannerState(last_sources=used_before, style=beat.beat_style)

    if beat.overlay_asset and beat.overlay_asset not in ids:
        beat.overlay_asset = None

    entity_backed = (
        beat.primary_source == PrimarySource.STOCK and beat.stock_media == "image"
        and tmpl != VisualTemplate.PUNCH_IN_EMPHASIS
        and not (beat.beat_style in (BeatStyle.FAIL, BeatStyle.HYPE)
                 and max((asset_score(a, text) for a in ctx.of_type("clip", "image")),
                         default=0) > 0))
    if beat.primary_source == PrimarySource.STOCK and not (
            stock_ok_for(tmpl, beat.beat_style) or entity_backed):
        # stock is reserved for normal B-roll and entity-backed establishing photos
        beat.primary_source = PrimarySource.LOCAL
        beat.search_queries = []
        beat.stock_media = "any"
        beat.stock_asset_id = beat.license_type = beat.media_path = None

    if beat.primary_source == PrimarySource.STOCK:
        beat.source_id = ""
        queries = [clean_query(q) for q in beat.search_queries]
        queries = [q for q in queries if q]
        beat.search_queries = queries or ([ctx.stock_query(text)] if ctx.stock_query(text) else [])
        if not beat.search_queries:
            beat.primary_source = PrimarySource.LOCAL  # nothing to search for
        else:
            if tmpl == VisualTemplate.SPLIT_SCHEMATIC:
                ov = ids.get(beat.overlay_asset or "")
                if entity_backed and ov and ov["asset_type"] == "schematic":
                    return  # left panel = stock photo, right panel = local schematic
                beat.visual_template = VisualTemplate.FULLSCREEN_CLIP
                beat.overlay_asset = None
            return

    if beat.source_id and beat.source_id not in ids:
        beat.source_id = ""  # unknown id: replace below with a compatible asset

    if tmpl == VisualTemplate.SPLIT_SCHEMATIC:
        src = ids.get(beat.source_id)
        ov = ids.get(beat.overlay_asset or "")
        # Normalise: right = schematic, left = footage/image.
        if src and src["asset_type"] == "schematic" and not (ov and ov["asset_type"] ==
                                                              "schematic"):
            beat.overlay_asset, beat.source_id = beat.source_id, beat.overlay_asset or ""
            src, ov = ids.get(beat.source_id), ids.get(beat.overlay_asset or "")
        keep_clip = bool(ov and ov["asset_type"] == "clip" and beat.overlay_start is not None)
        if not ov or (ov["asset_type"] not in ("schematic", "image") and not keep_clip):
            schem, _ = _pick_asset(ctx, ("schematic",), text, state, exclude={beat.source_id})
            beat.overlay_asset = schem["id"] if schem else None
        if not beat.source_id:
            left, _ = _pick_asset(ctx, visual_types, text, state,
                                  exclude={beat.overlay_asset or ""})
            beat.source_id = left["id"] if left else ""
        if not beat.overlay_asset:
            # No schematic exists: degrade gracefully.
            beat.visual_template = (VisualTemplate.CUTOUT_CALLOUT if ctx.of_type("image")
                                    else VisualTemplate.FULLSCREEN_CLIP)
            tmpl = beat.visual_template
        elif not beat.source_id:  # only a schematic exists: show it as the main visual
            beat.source_id, beat.overlay_asset = beat.overlay_asset, None
            beat.visual_template = VisualTemplate.FULLSCREEN_CLIP
            tmpl = beat.visual_template

    if tmpl == VisualTemplate.CUTOUT_CALLOUT:
        if not beat.source_id:
            bg, _ = _pick_asset(ctx, visual_types, text, state)
            beat.source_id = bg["id"] if bg else ""
        # The subject must be a still image (never the background footage, never a schematic).
        ov = ids.get(beat.overlay_asset or "")
        if not ov or ov["asset_type"] != "image":
            beat.overlay_asset = None
            subj, _ = _pick_asset(ctx, ("image",), text, state, exclude={beat.source_id})
            src = ids.get(beat.source_id)
            if subj:
                beat.overlay_asset = subj["id"]
            elif not (src and src["asset_type"] == "image"):
                # nothing to cut out: keep the label as a lower-third over a punch/full frame
                beat.visual_template = (VisualTemplate.PUNCH_IN_EMPHASIS
                                        if beat.beat_style in (BeatStyle.FAIL, BeatStyle.HYPE)
                                        else VisualTemplate.FULLSCREEN_CLIP)
                tmpl = beat.visual_template
        if tmpl == VisualTemplate.CUTOUT_CALLOUT and not beat.callout_text:
            beat.callout_text = ctx.callout(text, "", ids.get(beat.overlay_asset or beat.source_id)) or None

    if tmpl in (VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.PUNCH_IN_EMPHASIS):
        beat.overlay_asset = None
        if not beat.source_id:
            asset, _ = _pick_asset(ctx, visual_types or ("schematic",), text, state)
            if asset:
                beat.source_id = asset["id"]
            elif stock_query(text):  # no local assets at all -> stock
                beat.primary_source = PrimarySource.STOCK
                beat.search_queries = [ctx.stock_query(text)]


def _alternative_template(ctx: PlanContext, beat: SceneBeat, avoid: VisualTemplate
                          ) -> VisualTemplate:
    """A feasible template other than ``avoid`` that suits the beat's style (no 3 in a row)."""
    style = beat.beat_style
    if beat.primary_source == PrimarySource.STOCK:
        order = [VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.PUNCH_IN_EMPHASIS]
    elif style == BeatStyle.TECHNICAL:
        order = [VisualTemplate.SPLIT_SCHEMATIC, VisualTemplate.CUTOUT_CALLOUT,
                 VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.PUNCH_IN_EMPHASIS]
    elif style in (BeatStyle.FAIL, BeatStyle.HYPE):
        order = [VisualTemplate.PUNCH_IN_EMPHASIS, VisualTemplate.CUTOUT_CALLOUT,
                 VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.SPLIT_SCHEMATIC]
    else:
        order = [VisualTemplate.FULLSCREEN_CLIP, VisualTemplate.CUTOUT_CALLOUT,
                 VisualTemplate.PUNCH_IN_EMPHASIS, VisualTemplate.SPLIT_SCHEMATIC]
    visual = bool(ctx.of_type("clip", "image"))
    feasible = {
        VisualTemplate.FULLSCREEN_CLIP: True,
        VisualTemplate.PUNCH_IN_EMPHASIS: True,
        VisualTemplate.CUTOUT_CALLOUT: bool(ctx.of_type("image")),
        VisualTemplate.SPLIT_SCHEMATIC: bool(ctx.of_type("schematic")) and visual,
    }
    for t in order:
        if t != avoid and feasible[t]:
            return t
    return avoid


def _style_cap(ctx: PlanContext, beat: SceneBeat, text: str) -> float:
    """Longest allowed beat: the style's target maximum (+0.5s), the global max for holds."""
    _, _, _, hold = ctx.classify(text) if text else (None, "", False, False)
    if hold:
        return ctx.max_beat
    return min(ctx.max_beat, ctx.durations[beat.beat_style.value][1] + 0.5)


def _fix_callout(ctx: PlanContext, beat: SceneBeat, text: str) -> None:
    """Callouts are 2-5 words; labels next to a subject must name that subject."""
    tmpl = beat.visual_template
    if tmpl in (VisualTemplate.CUTOUT_CALLOUT, VisualTemplate.SPLIT_SCHEMATIC):
        subject = ctx.assets_by_id.get(beat.overlay_asset or "") or ctx.assets_by_id.get(
            beat.source_id)
        beat.callout_text = ctx.subject_callout(text, "", subject, beat.callout_text) or None
    elif beat.callout_text and len(beat.callout_text.split()) == 1:
        beat.callout_text = ctx.mishap_callout(text, "") or beat.callout_text


def _avoid_repeat(ctx: PlanContext, beat: SceneBeat, text: str, prev: set[str]) -> None:
    """Swap a local source that repeats the previous beat's media for a same-type alternative."""
    state = _PlannerState(last_sources=set(prev), style=beat.beat_style)
    for attr in ("source_id", "overlay_asset"):
        current = getattr(beat, attr)
        if not current or current not in prev or current not in ctx.assets_by_id:
            continue
        kind = ctx.assets_by_id[current]["asset_type"]
        taken = {x for x in (beat.source_id, beat.overlay_asset) if x}
        alt, _ = _pick_asset(ctx, (kind,), text, state, exclude=taken | prev)
        if alt:
            setattr(beat, attr, alt["id"])


def _validate_and_repair(plan: ScenePlan, ctx: PlanContext, semantic: bool = False
                        ) -> ScenePlan:
    """Guarantee coverage of [0, duration], sane lengths, valid + compatible assets.

    ``semantic=True`` (the post-matching pass) keeps the content-based asset choices: the
    keyword-based repeat swap is skipped because the semantic planner already penalises reuse."""
    duration = ctx.duration
    beats = [b for b in sorted(plan.beats, key=lambda b: b.start_time)
             if b.end_time > b.start_time and b.start_time < duration]
    if not beats:  # nothing usable: plan from scratch
        return _generate_deterministic_plan(ctx, plan.episode_id)

    # 1. Contiguous coverage: gaps are absorbed by the previous beat, overlaps trimmed.
    beats[0].set_times(0.0, min(beats[0].end_time, duration))
    for i in range(1, len(beats)):
        prev = beats[i - 1]
        end = min(beats[i].end_time, duration)
        beats[i].set_times(prev.end_time, end)
    beats = [b for b in beats if b.end_time - b.start_time > 1e-3]
    if not beats:
        return _generate_deterministic_plan(ctx, plan.episode_id)
    beats[-1].set_times(beats[-1].start_time, duration)

    # 2. Snap interior cuts to real word boundaries (LLM timings are approximate).
    for i in range(len(beats) - 1):
        t = _snap_to_words(beats[i].end_time, ctx)
        if beats[i].start_time + 0.3 < t < beats[i + 1].end_time - 0.3:
            beats[i].set_times(beats[i].start_time, t)
            beats[i + 1].set_times(t, beats[i + 1].end_time)

    # 3. Merge too-short beats into the previous (or next) beat.
    i = 0
    while i < len(beats) and len(beats) > 1:
        b = beats[i]
        if b.end_time - b.start_time < ctx.min_beat:
            if i > 0:
                beats[i - 1].set_times(beats[i - 1].start_time, b.end_time)
                beats[i - 1].emphasis = beats[i - 1].emphasis or b.emphasis
            else:
                beats[1].set_times(b.start_time, beats[1].end_time)
            beats.pop(i)
            i = max(i - 1, 0)
            continue
        i += 1

    # 4. Split too-long beats (on pauses when words are known).
    out: list[SceneBeat] = []
    for b in beats:
        d = b.end_time - b.start_time
        cap = _style_cap(ctx, b, ctx.text_between(b.start_time, b.end_time))
        if d <= cap + 1e-6:
            out.append(b)
            continue
        n = int(-(-d // (cap - 0.5)))  # leave slack for snapping to pauses
        edges = [b.start_time]
        for k in range(1, n):
            edges.append(_snap_to_words(b.start_time + d * k / n, ctx, tol=0.4))
        edges.append(b.end_time)
        for k in range(n):
            piece = b.model_copy(deep=True)
            piece.set_times(edges[k], edges[k + 1])
            if k > 0:
                piece.emphasis = False
                piece.beat_reason = (b.beat_reason + " (continued)").strip()
            out.append(piece)
    beats = out

    # 5. Per-beat repair: style upgrade from keywords, assets, queries, reason.
    prev_sources: set[str] = set()
    pool_used: set[str] = set()
    recent: list[tuple[float, set[str]]] = []
    templates: list[VisualTemplate] = []
    for b in beats:
        text = ctx.text_between(b.start_time, b.end_time)
        if not semantic and len(templates) >= 2 \
                and templates[-1] == templates[-2] == b.visual_template:
            # (semantic mode: src.variety rebalances templates with content-aware subjects)
            b.visual_template = _alternative_template(ctx, b, templates[-1])
        if b.beat_style == BeatStyle.NORMAL and text:
            style, kw, _, _ = ctx.classify(text)
            if style != BeatStyle.NORMAL:
                b.beat_style = style
                b.tone = style.value
                if not b.beat_reason:
                    b.beat_reason = f"style set from keyword '{kw}'"
        _compatible_assets(ctx, b, text, prev_sources)
        _attach_pool(ctx, b, text, pool_used)
        near = [ids for t, ids in recent if b.start_time - t <= RECENT_WINDOW]
        if not semantic:
            _avoid_repeat(ctx, b, text, set().union(*near, *(ids for _, ids in recent[-2:])))
        _fix_callout(ctx, b, text)
        if not b.beat_reason:
            b.beat_reason = b.director_notes or (
                f"{b.beat_style.value} beat shown as {b.visual_template.value}")
        prev_sources = {x for x in (b.source_id, b.overlay_asset) if x}
        recent.append((b.start_time, prev_sources))
        templates.append(b.visual_template)
        b.set_times(b.start_time, b.end_time)

    _ensure_punch(beats)
    _ensure_hook(beats)
    plan.beats = beats
    for i, b in enumerate(plan.beats):
        b.id = i
    return plan


def _ensure_hook(beats: list[SceneBeat]) -> None:
    """Open on a punch-in: a plain full-frame first beat becomes punch_in_emphasis so the video
    has motion in the first seconds (cuts must stay on word gaps, so we do not split it)."""
    first = beats[0]
    if first.visual_template != VisualTemplate.FULLSCREEN_CLIP \
            or first.primary_source != PrimarySource.LOCAL or not first.source_id:
        return
    if len(beats) > 2 and all(b.visual_template == VisualTemplate.PUNCH_IN_EMPHASIS
                              for b in beats[1:3]):
        return  # would make three punch-ins in a row
    first.visual_template = VisualTemplate.PUNCH_IN_EMPHASIS
    first.beat_reason = (first.beat_reason + "; opens on a punch-in (hook)").strip("; ")


def _ensure_punch(beats: list[SceneBeat]) -> None:
    """Fail/emphasis moments exist -> at least one punch_in_emphasis beat in the plan."""
    if any(b.visual_template == VisualTemplate.PUNCH_IN_EMPHASIS for b in beats):
        return
    for b in beats:
        if (b.beat_style == BeatStyle.FAIL or b.emphasis) \
                and b.visual_template == VisualTemplate.FULLSCREEN_CLIP \
                and b.primary_source == PrimarySource.LOCAL and b.source_id:
            b.visual_template = VisualTemplate.PUNCH_IN_EMPHASIS
            b.emphasis = False
            b.beat_reason = (b.beat_reason + "; punch-in to land the moment").strip("; ")
            return
