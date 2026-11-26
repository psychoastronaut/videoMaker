"""Stock pool (work/stock_pool.json): free media fetched *before* planning.

The pre-fetch stage fills the pool from the script/transcript; the planner (and the LLM prompt)
see it as available assets, and Stage 4 resolves beats from it before searching again. The
pool file carries the full license data of every item.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.common import read_json

POOL_FILE = "stock_pool.json"


def _stem(w: str) -> str:
    w = w.lower()
    return w[:-1] if w.endswith("s") and len(w) > 4 else w


def _words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[A-Za-z0-9]{3,}", text)}


def load_pool(work_dir: Path, allowed: Callable[[dict[str, Any]], bool] | None = None
              ) -> list[dict[str, Any]]:
    """Pool items that still exist on disk and pass the CURRENT license policy."""
    path = work_dir / POOL_FILE
    if not path.is_file():
        return []
    try:
        items = read_json(path).get("assets", [])
    except ValueError:
        return []
    return [a for a in items if Path(a.get("local_path", "")).is_file()
            and (allowed is None or allowed(a))]


def match_pool(pool: list[dict[str, Any]], queries: list[str], *, images_only: bool = False,
               exclude: set[str] | None = None) -> dict[str, Any] | None:
    """Best pool item for any of ``queries``: identical query wins; otherwise every word of a
    1-2 word query (two words of a longer one) must be among the item's query/tags."""
    best: tuple[int, dict[str, Any]] | None = None
    for item in pool:
        if exclude and item["id"] in exclude:
            continue
        if images_only and item.get("media_type", "image") != "image":
            continue
        tags = _words(" ".join([item.get("query", ""), *item.get("tags", [])]))
        for q in queries:
            if not q.strip():
                continue
            if q.strip().lower() == item.get("query", "").strip().lower():
                score = 100
            else:
                words = _words(q)
                need = min(len(words), 2)
                hits = len(words & tags)
                score = hits if words and hits >= need else 0
            if score and (best is None or score > best[0]):
                best = (score, item)
    return best[1] if best else None
