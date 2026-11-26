"""Round-9: stock pre-fetch stage, stock pool in planner/LLM prompt, Stage 4 pool resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from conftest import make_subject_image
from test_planner import make_ctx
from test_review_fixes import episode_with_assets, write_assets
from test_round4 import set_policy
from test_round5 import local, stock_episode

from src import cli
from src.common import ENV_OFFLINE, read_json, write_json
from src.schemas import SceneBeat, ScenePlan, StockAsset
from src.stage3_plan_scenes import (
    PlanContext,
    _build_user_prompt,
    _generate_deterministic_plan,
    _validate_and_repair,
    flatten_words,
)
from src.stage4_fetch_stock import StockFetcher
from src.stage4_fetch_stock import run as stage4
from src.stage_prefetch import extract_queries, prefetch_signature
from src.stage_prefetch import run as prefetch
from src.stockpool import load_pool, match_pool

TEXT = ("Today we explore many interesting things about people and their daily habits. "
        "The Tokyo skyline glows at night over a sleepy suburb. "
        "A CNC lathe in a factory carves a brass part. "
        "The servo motor spins fast. "
        "The battery shorted and there was magic smoke everywhere in the workshop. "
        "A quiet garden table holds ripe tomatoes and fresh peppers.")


def transcript(text: str = TEXT) -> dict[str, Any]:
    segs, t = [], 0.0
    for i, sent in enumerate(s.strip() + "." for s in text.split(".") if s.strip()):
        words = []
        for tok in sent.split():
            words.append({"word": tok, "start": round(t, 3), "end": round(t + 0.3, 3)})
            t += 0.33
        segs.append({"id": i, "start": words[0]["start"], "end": words[-1]["end"], "text": sent,
                     "words": words})
        t += 0.4
    return {"duration": round(t, 2), "segments": segs, "full_text": text}


def servo_assets(tmp: Path) -> list[dict[str, Any]]:
    return [local(tmp, "servo_motor", "image", ["servo", "motor"])]


def pool_item(tmp: Path, pid: str, query: str, media: str = "image", platform: str = "wikimedia",
              lic: str = "CC0") -> dict[str, Any]:
    p = tmp / f"{pid}.png"
    make_subject_image(p)
    return {"id": pid, "query": query, "media_type": media, "license_type": lic,
            "source_platform": platform, "source_url": "u", "local_path": str(p),
            "tags": query.lower().split(), "title": query, "entity": True}


# ------------------------------------------------------------------ extraction


def test_extract_queries_ranks_entities_first_and_skips_covered_and_fail_lines(
        tmp_path: Path) -> None:
    cfg = make_ctx().config
    qs = extract_queries(transcript(), "", servo_assets(tmp_path), cfg)
    queries = [q["query"] for q in qs]
    assert queries[0] == "Tokyo skyline" and qs[0]["entity"]
    assert "CNC lathe" in queries
    assert not any("servo" in q.lower() for q in queries)        # local asset covers it
    assert not any("smoke" in q.lower() for q in queries)        # fail line: no generic stock
    assert any(not q["entity"] and "tomatoes" in q["query"].lower() for q in qs)


def test_extract_queries_reads_the_script_too(tmp_path: Path) -> None:
    script = "# Beat 1\nThe Eiffel Tower stands above a busy city every single day.\n"
    qs = extract_queries({"duration": 3, "segments": []}, script, [], make_ctx().config)
    assert qs and qs[0]["entity"] and qs[0]["query"].startswith("Eiffel Tower")


# ------------------------------------------------------------------ the stage


def test_offline_prefetch_writes_an_empty_pool(capsys: pytest.CaptureFixture[str]) -> None:
    _ep, work = episode_with_assets("pf1")
    write_json(work / "transcript.json", transcript())
    write_assets(work, [])
    prefetch("pf1", work)
    pool = read_json(work / "stock_pool.json")
    assert pool["assets"] == [] and pool["offline"] is True
    assert "offline" in capsys.readouterr().out


def test_prefetch_fetches_a_deduped_capped_pool_with_license_data(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _ep, work = episode_with_assets("pf2")
    write_json(work / "transcript.json", transcript())
    write_assets(work, servo_assets(tmp_path))
    plan_before = {"episode_id": "pf2", "beats": [{"start_time": 0, "end_time": 3}]}
    write_json(work / "scene_plan.json", plan_before)
    monkeypatch.delenv(ENV_OFFLINE)
    calls: list[tuple[str, bool]] = []

    def fake_find(self: StockFetcher, query: str, beat_id: int, **kw: Any) -> StockAsset | None:
        calls.append((query, kw["images_only"]))
        if "garden" in query.lower() or "tomatoes" in query.lower():
            return None
        img = tmp_path / f"{len(calls)}.png"
        make_subject_image(img)
        return StockAsset(id=f"wikimedia_{len(calls)}", source_platform="wikimedia",
                          source_url="u", local_path=str(img), license_type="CC0", query=query,
                          media_type="image", author="Ann")

    monkeypatch.setattr(StockFetcher, "find", fake_find)
    pool = prefetch("pf2", work)
    ids = [a["id"] for a in pool["assets"]]
    assert len(ids) == len(set(ids)) >= 2
    first = pool["assets"][0]
    assert first["query"] == "Tokyo skyline" and first["entity"] and first["license_type"] == "CC0"
    assert first["tags"] == ["tokyo", "skyline"] and first["author"] == "Ann"
    assert calls[0] == ("Tokyo skyline", True)                      # entities are image-only
    assert read_json(work / "scene_plan.json") == plan_before        # never touches the plan
    # cap
    monkeypatch.setattr("src.stage_prefetch.load_config", lambda: {
        **make_ctx().config, "stock_media": {"prefetch": {"max_items": 1}}})
    assert len(prefetch("pf2", work)["assets"]) == 1


def test_prefetch_is_stale_when_script_transcript_or_policy_changes(
        monkeypatch: pytest.MonkeyPatch) -> None:
    ep, work = episode_with_assets("pf3")
    write_json(work / "transcript.json", transcript())
    write_assets(work, [])
    (ep / "raw" / "script.md").write_text("The Tokyo skyline at night.")
    prefetch("pf3", work)
    cli._record_state("prefetch", work)
    assert cli.stale_reason("prefetch", ep, work) is None
    (ep / "raw" / "script.md").write_text("The Berlin skyline at dawn.")
    assert "changed" in (cli.stale_reason("prefetch", ep, work) or "")
    cli._record_state("prefetch", work)
    assert cli.stale_reason("prefetch", ep, work) is None
    set_policy(True)                                              # strict_cc0_only
    assert cli.stale_reason("prefetch", ep, work)
    cli._record_state("prefetch", work)
    monkeypatch.delenv(ENV_OFFLINE)                               # going online
    assert cli.stale_reason("prefetch", ep, work)
    assert prefetch_signature(ep, work, "a") != prefetch_signature(ep, work, "b")


def test_status_lists_the_prefetch_stage(capsys: pytest.CaptureFixture[str]) -> None:
    episode_with_assets("pf4")
    cli.status("pf4")
    assert "Prefetch Stock Pool" in capsys.readouterr().out
    keys = [s.key for s in cli.STAGES]
    assert keys.index("index") < keys.index("prefetch") < keys.index("plan")


# ------------------------------------------------------------------ planner / LLM / stage 4


def test_pool_matching_rules(tmp_path: Path) -> None:
    pool = [pool_item(tmp_path, "a", "Tokyo skyline"), pool_item(tmp_path, "v", "CNC lathe",
                                                                 media="video")]
    assert match_pool(pool, ["Tokyo skyline"])["id"] == "a"
    assert match_pool(pool, ["skyline Tokyo night"])["id"] == "a"       # >= 2 words overlap
    assert match_pool(pool, ["Paris skyline"]) is None                  # 2-word: both needed
    assert match_pool(pool, ["CNC lathe"], images_only=True) is None
    assert match_pool(pool, ["Tokyo skyline"], exclude={"a"}) is None


def test_pool_is_filtered_by_the_current_license_policy(tmp_path: Path) -> None:
    _ep, work = episode_with_assets("pf5")
    write_json(work / "stock_pool.json", {"assets": [
        pool_item(tmp_path, "w", "Tokyo skyline"),
        pool_item(tmp_path, "p", "CNC lathe", platform="pexels", lic="Pexels License")]})
    from src.media import _stock_policy
    assert {a["id"] for a in load_pool(work, _stock_policy(work))} == {"w", "p"}
    set_policy(True)
    assert {a["id"] for a in load_pool(work, _stock_policy(work))} == {"w"}


def test_planner_uses_pool_items_and_stage4_resolves_without_searching(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = make_ctx()
    t = transcript()
    assets = servo_assets(tmp_path)
    pool = [pool_item(tmp_path, "wikimedia_tokyo", "Tokyo skyline"),
            pool_item(tmp_path, "wikimedia_lathe", "CNC lathe")]
    ctx = PlanContext(base.config, assets, flatten_words(t), t["duration"], pool)
    plan = _validate_and_repair(_generate_deterministic_plan(ctx, "pf6"), ctx)
    stock = [b for b in plan.beats if b.primary_source.value == "stock"]
    by_q = {b.search_queries[0]: b for b in stock}
    assert by_q["Tokyo skyline"].stock_asset_id == "wikimedia_tokyo"
    assert by_q["Tokyo skyline"].media_path == pool[0]["local_path"]
    assert by_q["CNC lathe"].stock_asset_id == "wikimedia_lathe"
    # local footage keeps priority: the servo line has a relevant local asset -> no stock
    servo_beats = [b for b in plan.beats if "servo" in ctx.text_between(
        b.start_time, b.end_time).lower()]
    assert servo_beats and all(b.primary_source.value == "local" for b in servo_beats)

    # Stage 4 only verifies: no search is allowed to happen
    work = stock_episode("pf6", assets, [b.model_dump(mode="json") for b in plan.beats])
    write_json(work / "stock_pool.json", {"assets": pool})
    monkeypatch.delenv(ENV_OFFLINE)
    searched: list[str] = []
    monkeypatch.setattr(StockFetcher, "find", lambda self, q, *a, **k: searched.append(q))
    stage4("pf6", work)
    assert "Tokyo skyline" not in searched and "CNC lathe" not in searched  # pool, not search
    audit = read_json(work / "licenses_audit.json")["assets"]
    assert {a["id"] for a in audit} == {"wikimedia_tokyo", "wikimedia_lathe"}
    out = {b["search_queries"][0]: b for b in read_json(work / "scene_plan.json")["beats"]
           if b["primary_source"] == "stock"}
    assert out["Tokyo skyline"]["license_type"] == "CC0"


def test_stage4_takes_a_pool_match_for_beats_without_an_id(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pool = [pool_item(tmp_path, "wikimedia_tokyo", "Tokyo skyline")]
    beat = {"id": 0, "start_time": 0, "end_time": 4, "visual_template": "fullscreen_clip",
            "primary_source": "stock", "stock_media": "image", "beat_style": "normal",
            "search_queries": ["Tokyo skyline"]}
    work = stock_episode("pf7", servo_assets(tmp_path), [beat])
    write_json(work / "stock_pool.json", {"assets": pool})
    monkeypatch.delenv(ENV_OFFLINE)
    monkeypatch.setattr(StockFetcher, "find",
                        lambda *a, **k: pytest.fail("pool match expected"))
    stage4("pf7", work)
    assert read_json(work / "scene_plan.json")["beats"][0]["stock_asset_id"] == "wikimedia_tokyo"


def test_llm_prompt_lists_the_stock_pool(tmp_path: Path) -> None:
    base = make_ctx()
    t = transcript()
    pool = [pool_item(tmp_path, "wikimedia_tokyo", "Tokyo skyline")]
    ctx = PlanContext(base.config, [], flatten_words(t), t["duration"], pool)
    prompt = _build_user_prompt(t, ctx, "ep")
    assert "FREE STOCK POOL" in prompt and "wikimedia_tokyo" in prompt and "Tokyo skyline" in prompt
    empty = PlanContext(base.config, [], flatten_words(t), t["duration"])
    assert "FREE STOCK POOL" not in _build_user_prompt(t, empty, "ep")


def test_llm_beat_with_an_unknown_pool_id_is_resolved_or_cleared(tmp_path: Path) -> None:
    base = make_ctx()
    t = transcript()
    pool = [pool_item(tmp_path, "wikimedia_tokyo", "Tokyo skyline")]
    ctx = PlanContext(base.config, [], flatten_words(t), t["duration"], pool)
    raw = ScenePlan(beats=[SceneBeat(
        start_time=0, end_time=3.4, primary_source="stock", stock_media="image",
        stock_asset_id="made_up", search_queries=["Tokyo skyline"]),
        SceneBeat(start_time=3.4, end_time=t["duration"], primary_source="local")])
    beat = _validate_and_repair(raw, ctx).beats[0]
    assert beat.stock_asset_id == "wikimedia_tokyo"
    assert os.path.isfile(beat.media_path)
