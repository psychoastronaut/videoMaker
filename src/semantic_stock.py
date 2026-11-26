"""Stage 2b/4 helper: verify stock candidates against the beat's visual query.

A downloaded (already license-checked) stock candidate is embedded and compared with the beat's
``visual_query``; below the backend's threshold it is rejected so Stage 4 re-queries (LLM
rewrite, max 2 attempts) and finally falls back to local media. Licence handling is untouched:
this only decides whether a candidate is *used*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from src.semantic import SemanticBackend, get_backend, min_score_for, semantic_config
from src.semantic_index import embed_stills

console = Console()
MAX_REQUERIES = 2


class StockVerifier:
    def __init__(self, work: Path, backend: SemanticBackend, config: dict[str, Any]) -> None:
        self.work, self.backend, self.config = work, backend, config
        cfg = semantic_config(config)
        self.threshold = float(cfg.get("stock_min_score") or min_score_for(backend.name, config))
        self.requeries = int(cfg.get("stock_requeries", MAX_REQUERIES))
        self._qcache: dict[str, np.ndarray] = {}
        self._llm: Any = None

    @classmethod
    def create(cls, work: Path, config: dict[str, Any]) -> StockVerifier | None:
        if not semantic_config(config).get("verify_stock", True):
            return None
        backend = get_backend(config, quiet=True)
        return cls(work, backend, config) if backend is not None else None

    def _qvec(self, query: str) -> np.ndarray:
        if query not in self._qcache:
            self._qcache[query] = self.backend.embed_queries([query])[0]
        return self._qcache[query]

    def score(self, media_path: str | Path, query: str) -> float | None:
        got = embed_stills(self.work, self.backend, {"c": Path(media_path)})
        if "c" not in got:
            return None
        return float(got["c"][0] @ self._qvec(query))

    def rank_pool(self, pool: list[dict[str, Any]], query: str, exclude: set[str],
                  images_only: bool = False) -> tuple[dict[str, Any], float] | None:
        """Best pool item by visual similarity (None below the threshold)."""
        items = {p["id"]: Path(p["local_path"]) for p in pool if p["id"] not in exclude
                 and (not images_only or p.get("media_type", "image") == "image")}
        if not items:
            return None
        vecs = embed_stills(self.work, self.backend, items)
        q = self._qvec(query)
        best = max(((float(v @ q), pid) for pid, (v, _) in vecs.items()), default=None)
        if best is None or best[0] < self.threshold:
            return None
        return next(p for p in pool if p["id"] == best[1]), best[0]

    def alt_query(self, original: str, narration_query: str, tried: list[str]) -> str | None:
        """Ask the LLM for a different 2-4 word stock search query (None if unavailable)."""
        from src.semantic_planner import LLM

        self.backend.free()  # VRAM back to the LLM
        if self._llm is None:
            self._llm = LLM(self.config)
        if not self._llm.ok:
            return None
        data = self._llm.json(
            "You write short stock-photo search queries. Reply with JSON only.",
            f'We need footage/photo of: "{narration_query}". These searches returned off-topic '
            f"results: {tried}. Write ONE new, different 2-4 word concrete search query "
            '(common objects/places, no abstract words). Return {"query":"..."}.', 200)
        q = str((data or {}).get("query", "")).strip() if isinstance(data, dict) else ""
        return q if q and q.lower() not in {t.lower() for t in tried} else None

    def close(self) -> None:
        self.backend.free()
