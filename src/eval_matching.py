"""Objective evaluation of sentence -> asset matching (baseline keyword matcher vs backends)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.console import Console

from src.common import read_json
from src.semantic import SemanticBackend, get_backend, semantic_config
from src.semantic_index import SemanticIndex, build_index

console = Console()
DEFAULT_PAIRS = Path(__file__).resolve().parent.parent / "tests" / "semantic" / "pairs_ep-real.json"


def baseline_rank(assets: list[dict[str, Any]], text: str) -> list[str]:
    """Current v1 matcher: keyword overlap with filename/sidecar description."""
    from src.stage3_plan_scenes import asset_score

    return [a["id"] for a in sorted(assets, key=lambda a: -asset_score(a, text))]


def backend_rank(index: SemanticIndex, backend: SemanticBackend, texts: list[str],
                 assets: list[dict[str, Any]] | None = None, prior_weight: float = 0.0
                 ) -> list[list[str]]:
    from src.semantic_planner import keyword_prior

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.assets = assets or []  # type: ignore[attr-defined]
    qv = backend.embed_queries(texts)
    return [[h.asset_id for h in index.search(
        q, k=1000, prior=keyword_prior(ctx, t, prior_weight) if prior_weight else None)]
        for q, t in zip(qv, texts, strict=True)]


def score(ranks: list[list[str]], pairs: list[dict[str, Any]]) -> dict[str, float]:
    top1 = sum(r[:1] and r[0] in p["expected"] for r, p in zip(ranks, pairs, strict=True))
    top3 = sum(any(x in p["expected"] for x in r[:3]) for r, p in zip(ranks, pairs, strict=True))
    n = max(len(pairs), 1)
    return {"top1": top1 / n, "top3": top3 / n, "n": len(pairs)}


def run(work: Path, backend_name: str = "all", pairs_path: Path | None = None,
        config: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    pairs = read_json(pairs_path or DEFAULT_PAIRS)["pairs"]
    assets = read_json(work / "local_assets.json")["assets"]
    texts = [p["text"] for p in pairs]
    results = {"baseline": score([baseline_rank(assets, t) for t in texts], pairs)}
    names = ["siglip", "ollama"] if backend_name == "all" else (
        [] if backend_name == "baseline" else [backend_name])
    for name in names:
        backend = get_backend(config, requested=name, quiet=True)
        if backend is None:
            console.print(f"  [yellow]{name}: not available, skipped[/]")
            continue
        t0 = time.time()
        try:
            sub = work / f"eval_{name}"  # per-backend scratch dir (files still cached by hash)
            sub.mkdir(exist_ok=True)
            build_index(sub, backend, assets, semantic_config(config))
            index = SemanticIndex.load(sub)
            if index is None:
                continue
            ranks = backend_rank(index, backend, texts)
            res = score(ranks, pairs)
            res["seconds"] = round(time.time() - t0, 1)
            results[name] = res
            pw = float(semantic_config(config).get("prior_weight")
                        or (0.04 if name == "siglip" else 0.06))
            results[f"{name}+prior"] = score(
                backend_rank(index, backend, texts, assets, pw), pairs)
            for r, p in zip(ranks, pairs, strict=True):
                if r[0] not in p["expected"]:
                    console.print(f"    [dim]{name} miss:[/] {p['text'][:55]!r} -> {r[0]}")
        finally:
            backend.free()
    t = f"{'matcher':10} top1   top3   (n={len(pairs)})"
    console.print(t)
    for k, v in results.items():
        console.print(f"{k:10} {v['top1']:.0%}   {v['top3']:.0%}"
                      + (f"   {v['seconds']}s" if "seconds" in v else ""))
    return results
