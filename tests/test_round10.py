"""Round 10: query ladder, thematic pool, pool share, Pixabay gate (network mocked)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import make_subject_image
from test_review_fixes import episode_with_assets, write_assets
from test_round9 import servo_assets, transcript

from src.common import ENV_OFFLINE, read_json, write_json
from src.schemas import StockAsset
from src.stage4_fetch_stock import StockFetcher, query_ladder, relevant
from src.stage_prefetch import extract_themes
from src.stage_prefetch import run as prefetch


def test_query_ladder_goes_phrase_head_hypernym() -> None:
    rungs = query_ladder("CNC lathe")
    assert rungs[0] == ("CNC lathe", True)
    assert ("lathe", False) in rungs and ("machine tool", False) in rungs
    assert all(not exact for _, exact in rungs[1:])
    assert query_ladder("Tokyo skyline")[-1][0] == "city skyline"
    assert query_ladder("Tokyo") == [("Tokyo", True)]


def test_extract_themes_ranks_script_terms_and_pads_with_defaults() -> None:
    themes = extract_themes({"segments": []}, "I was soldering a circuit board. Soldering again.", 8)
    assert themes[0] == "soldering iron" or themes[0] == "soldering"
    assert "circuit board" in themes and "workshop" in themes
    assert len(themes) == len(set(themes)) <= 8


def test_relaxed_gate_accepts_partial_matches_strict_does_not() -> None:
    assert not relevant("soldering iron", "Workshop bench with iron")
    assert relevant("workshop tools", "Old workshop", relaxed=True)


def test_prefetch_adds_thematic_items_flagged_not_entity(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _ep, work = episode_with_assets("pf10")
    write_json(work / "transcript.json", transcript())
    write_assets(work, servo_assets(tmp_path))
    monkeypatch.delenv(ENV_OFFLINE)
    calls: list[tuple[str, bool]] = []

    def fake_find(self: StockFetcher, query: str, beat_id: int, **kw: Any) -> StockAsset | None:
        calls.append((query, kw.get("relaxed", False)))
        if query == "Tokyo skyline":
            return None  # exact phrase misses -> ladder rungs are tried
        img = tmp_path / f"{len(calls)}.png"
        make_subject_image(img)
        return StockAsset(id=f"pixabay_{len(calls)}", source_platform="pixabay",
                          source_url="u", local_path=str(img),
                          license_type="Pixabay Content License", query=query,
                          media_type="image")

    monkeypatch.setattr(StockFetcher, "find", fake_find)
    pool = prefetch("pf10", work)
    items = pool["assets"]
    thematic = [a for a in items if a.get("thematic")]
    assert len(thematic) >= 7 and all(not a["entity"] and a["exact"] for a in thematic)
    rung = next(a for a in items if a["entity"])
    assert rung["exact"] is False and "tokyo" in rung["tags"]
    assert ("skyline", True) in calls  # head-noun rung used the relaxed gate
    assert len(items) <= 24
    assert read_json(work / "stock_pool.json")["assets"]
