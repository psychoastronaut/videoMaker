"""Stage 2b: free-stock pre-fetch (runs after transcript + asset index, before planning).

Reads ``raw/script.md`` and the transcript, extracts the places/machines/concrete nouns the
video talks about, and downloads a pool of free images (and short videos when no entity) into
``work/stock_cache/`` through the same license-checked fetcher Stage 4 uses. The result,
``work/stock_pool.json``, lists every item with its full license data; the planner and Stage 4
use it before searching again. Local footage keeps priority: sentences a local asset already
covers are skipped. Offline: nothing is fetched and the pool is empty.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console

from src.common import episode_dir, is_offline, load_config, read_json, write_json
from src.schemas import BeatStyle
from src.stage3_plan_scenes import (
    BRAND,
    PlanContext,
    Word,
    _sentence_units,
    asset_score,
    flatten_words,
)
from src.stage4_fetch_stock import StockFetcher, query_ladder
from src.stockpool import POOL_FILE

console = Console()


def _script_text(ep: Path) -> str:
    p = ep / "raw" / "script.md"
    if not p.is_file():
        p = ep / "raw" / "script.txt"
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def _script_sentences(text: str) -> list[list[Word]]:
    """Sentences of script.md as untimed Word lists (markdown headings dropped)."""
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
        toks = sent.split()
        if len(toks) >= 3:
            out.append([Word(t, 0.0, 0.0) for t in toks])
    return out


def extract_queries(transcript: dict[str, Any], script: str, assets: list[dict[str, Any]],
                    config: dict[str, Any]) -> list[dict[str, Any]]:
    """Ranked stock queries: entity photos first (named places first), then concrete-noun
    lines. Sentences already covered by a local asset, and fail/hype/technical lines (stock is
    not used for them), are skipped for generic queries."""
    words = flatten_words(transcript)
    duration = float(transcript.get("duration", 0) or (words[-1].end if words else 0))
    ctx = PlanContext(config, assets, words, duration)
    sentences = _sentence_units(words) + _script_sentences(script)
    found: dict[str, dict[str, Any]] = {}
    for sent in sentences:
        text = " ".join(w.text for w in sent)
        local = max((asset_score(a, text) for a in ctx.of_type("clip", "image")), default=0)
        ent = ctx.entity(sent)
        style = ctx.classify(text)[0]
        if ent and (local == 0 or ent[1]):
            query, strong = ent
            found.setdefault(query.lower(), {"query": query, "entity": True,
                                             "rank": (2 if strong else 1, len(query.split())),
                                             "text": text})
        elif local == 0 and style in (BeatStyle.NORMAL, BeatStyle.DEADPAN):
            query = ctx.stock_query(text)
            if query and query.lower() not in found and not any(
                    w.lower() in BRAND for w in query.split()):
                found[query.lower()] = {"query": query, "entity": False,
                                        "rank": (0, len(query.split())), "text": text}
    return sorted(found.values(), key=lambda c: c["rank"], reverse=True)


# trigger words in the script -> on-theme B-roll queries (always some B-roll, whatever the topic)
THEME_TRIGGERS: dict[str, list[str]] = {
    "solder": ["soldering iron", "soldering"], "circuit": ["circuit board"],
    "pcb": ["circuit board"], "board": ["circuit board"], "electronic": ["electronics"],
    "wire": ["electronics wires"], "workshop": ["workshop"], "bench": ["workshop"],
    "tool": ["workshop tools"], "factory": ["factory"], "assembly": ["factory"],
    "city": ["city skyline", "city night"], "skyline": ["city skyline"],
    "night": ["city night"], "lab": ["laboratory"], "robot": ["robot arm"],
    "motor": ["electric motor"], "machine": ["machine tool"], "lathe": ["machine tool"],
    "cnc": ["machine tool"], "battery": ["battery"], "print": ["3d printer"],
    "engineer": ["engineering"], "drone": ["drone"], "rocket": ["rocket launch"],
    "power": ["power plant"], "sensor": ["sensor electronics"], "mill": ["milling machine"],
    "weld": ["welding"], "metal": ["metal workshop"],
}
THEME_DEFAULTS = ["workshop", "electronics", "circuit board", "factory", "laboratory",
                  "machine tool", "engineering"]


def extract_themes(transcript: dict[str, Any], script: str, limit: int) -> list[str]:
    """Thematic B-roll queries ranked by how often the script mentions their trigger words,
    padded with generic engineering themes so the pool is never empty of B-roll."""
    text = " ".join([script, *(s.get("text", "") for s in transcript.get("segments", []))])
    toks = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]
    score: dict[str, int] = {}
    for tok in toks:
        for trig, queries in THEME_TRIGGERS.items():
            if tok.startswith(trig):
                for q in queries:
                    score[q] = score.get(q, 0) + 1
    ranked = sorted(score, key=lambda q: -score[q])
    out = list(dict.fromkeys([*ranked, *THEME_DEFAULTS]))
    return out[:limit]


def prefetch_signature(ep: Path, work: Path, policy: str) -> str:
    """What the pool depends on: script, transcript text, local asset ids, license policy."""
    parts: list[str] = [_script_text(ep), policy]
    tp = work / "transcript.json"
    if tp.is_file():
        parts.append(" ".join(s.get("text", "") for s in read_json(tp).get("segments", [])))
    ap = work / "local_assets.json"
    if ap.is_file():
        parts.append(",".join(sorted(a["id"] for a in read_json(ap).get("assets", []))))
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()


def run(episode_id: str, work_dir: Path) -> dict[str, Any]:
    """Run the pre-fetch: write ``stock_pool.json`` (possibly empty)."""
    config = load_config()
    stock_cfg = config.get("stock_media", {})
    pf = stock_cfg.get("prefetch", {})
    ep = episode_dir(episode_id)
    transcript = read_json(work_dir / "transcript.json") \
        if (work_dir / "transcript.json").is_file() else {"segments": []}
    assets = read_json(work_dir / "local_assets.json").get("assets", []) \
        if (work_dir / "local_assets.json").is_file() else []
    cache = work_dir / "stock_cache"
    cache.mkdir(parents=True, exist_ok=True)
    offline = is_offline() or stock_cfg.get("enabled", True) is False  # master switch
    fetcher = StockFetcher(stock_cfg, cache, offline)
    pool: dict[str, Any] = {"offline": offline, "policy": fetcher.policy_signature(),
                            "assets": []}

    if offline or not pf.get("enabled", True):
        why = "offline mode" if offline else "disabled in config"
        console.print(f"  [dim]Stock pre-fetch skipped ({why}); the pool is empty[/]")
    else:
        queries = extract_queries(transcript, _script_text(ep), assets, config)
        limit = int(pf.get("max_items", 24))
        n_theme = int(pf.get("thematic_items", 16))
        ent_q = [c for c in queries if c["entity"]]
        gen_q = [c for c in queries if not c["entity"]]
        console.print(f"  {len(ent_q)} entity + {len(gen_q)} generic query(ies); "
                      f"{n_theme} thematic; pool cap {limit}")

        def add(asset: Any, query: str, *, entity: bool, exact: bool, thematic: bool) -> None:
            fetcher.used_ids.add(asset.id)
            entry = asset.model_dump(mode="json")
            entry.update(beat_id=None, entity=entity, exact=exact, thematic=thematic,
                         title=query,
                         tags=[w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", query)])
            pool["assets"].append(entry)
            console.print(f"  pool: {asset.media_type} '{query}' <- {asset.source_platform} "
                          f"({asset.license_type})")

        # 1) entity photos (strict relevance gate; ladder rungs after the exact phrase are
        #    flagged exact=False so they are never captioned as the entity)
        reserve = min(n_theme, max(limit // 2, 0))
        for c in ent_q:
            if len(pool["assets"]) >= limit - reserve:
                break
            for q, exact in query_ladder(c["query"]):
                asset = fetcher.find(q, -1, prefer_video=False, need_seconds=6.0,
                                     images_only=True, relaxed=not exact)
                if asset:
                    add(asset, q if exact else c["query"] + " / " + q, entity=True,
                        exact=exact, thematic=False)
                    pool["assets"][-1]["query"] = c["query"] if exact else q
                    break
            else:
                console.print(f"  [dim]no verified free media for '{c['query']}'[/]")
        # 2) thematic B-roll (relaxed gate)
        themes = 0
        for q in extract_themes(transcript, _script_text(ep), n_theme * 2):
            if themes >= n_theme or len(pool["assets"]) >= limit:
                break
            asset = fetcher.find(q, -1, prefer_video=False, need_seconds=6.0,
                                 images_only=True, relaxed=True)
            if asset:
                add(asset, q, entity=False, exact=True, thematic=True)
                themes += 1
            else:
                console.print(f"  [dim]no verified free media for theme '{q}'[/]")
        # 3) generic concrete-noun lines with the remaining room
        for c in gen_q:
            if len(pool["assets"]) >= limit:
                break
            asset = fetcher.find(c["query"], -1, prefer_video=True, need_seconds=6.0,
                                 images_only=False)
            if asset:
                add(asset, c["query"], entity=False, exact=True, thematic=False)
    write_json(work_dir / POOL_FILE, pool)
    console.print(f"  Stock pool saved: {work_dir / POOL_FILE} ({len(pool['assets'])} items)")
    return pool


def topup(episode_id: str, work_dir: Path, queries: list[str], limit: int = 8
          ) -> list[dict[str, Any]]:
    """Library top-up: fetch free stock images for ``queries`` (the weakest-matched beats' visual
    queries, CC0/public-domain providers first - same relevance and licence gates as the
    prefetch) and APPEND them to ``stock_pool.json``. Returns the new pool items (aligned
    with the queries that found something: each item has ``for_query``). Offline / stock
    disabled: nothing is fetched."""
    config = load_config()
    stock_cfg = config.get("stock_media", {})
    if is_offline() or stock_cfg.get("enabled", True) is False:
        return []
    cache = work_dir / "stock_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fetcher = StockFetcher(stock_cfg, cache, False)
    pool_path = work_dir / POOL_FILE
    pool: dict[str, Any] = read_json(pool_path) if pool_path.is_file() else \
        {"offline": False, "policy": fetcher.policy_signature(), "assets": []}
    for it in pool.get("assets", []):
        fetcher.used_ids.add(it["id"])
    added: list[dict[str, Any]] = []
    for q in queries:
        if len(added) >= limit:
            break
        for qq, exact in query_ladder(q):
            asset = fetcher.find(qq, -1, prefer_video=False, need_seconds=6.0,
                                 images_only=True, relaxed=not exact)
            if asset:
                fetcher.used_ids.add(asset.id)
                entry = asset.model_dump(mode="json")
                entry.update(beat_id=None, entity=False, exact=exact, thematic=False,
                             title=qq, query=qq, for_query=q, topup=True,
                             tags=[w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", qq)])
                pool["assets"].append(entry)
                added.append(entry)
                console.print(f"  top-up: '{q}' <- {asset.source_platform} ({asset.license_type})")
                break
        else:
            console.print(f"  [dim]top-up: no verified free media for '{q}'[/]")
    if added:
        write_json(pool_path, pool)
    return added
