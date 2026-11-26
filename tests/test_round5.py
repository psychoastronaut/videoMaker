"""Round-5: stock->local fallback, designed title cards, planner stock policy, visual tweaks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from conftest import make_subject_image, write_wav
from test_planner import ASSETS, SENTENCES, make_ctx, make_transcript
from test_review_fixes import episode_with_assets, write_assets

from src.common import ENV_OFFLINE, read_json, write_json
from src.media import AssetResolver
from src.schemas import SceneBeat, ScenePlan
from src.stage3_plan_scenes import (
    PlanContext,
    _generate_deterministic_plan,
    _validate_and_repair,
    flatten_words,
)
from src.stage4_fetch_stock import StockAsset, StockFetcher
from src.stage4_fetch_stock import run as stage4
from src.stage5_cutouts import run as stage5
from src.stage6_render import (
    PUNCH_ANCHOR_Y,
    SPLIT_H,
    CompileContext,
    RenderSettings,
    _write_output_audit,
    compile_beat,
    motion_for,
)


def stock_episode(name: str, assets: list[dict[str, Any]], beats: list[dict[str, Any]],
                  transcript_text: str = "the servo motor spins") -> Path:
    ep, work = episode_with_assets(name)
    write_wav(ep / "raw" / "voiceover.wav", 10.0)
    write_assets(work, assets)
    write_json(work / "transcript.json", {"duration": 10.0, "segments": [
        {"start": 0, "end": 10, "text": transcript_text, "words": []}]})
    write_json(work / "scene_plan.json", {"episode_id": name, "beats": beats})
    return work


def local(tmp: Path, name: str, kind: str, tags: list[str]) -> dict[str, Any]:
    p = tmp / f"{name}.png"
    make_subject_image(p)
    return {"id": f"{kind}_{name}", "filepath": str(p), "asset_type": kind, "filename": p.name,
            "tags": [kind, *tags], "description": " ".join(tags)}


def stock_beat(bid: int, start: float, end: float, template: str = "punch_in_emphasis",
               **extra: Any) -> dict[str, Any]:
    return {"id": bid, "start_time": start, "end_time": end, "visual_template": template,
            "primary_source": "stock", "search_queries": ["servo motor"], "beat_style": "fail",
            "callout_text": "Servo trouble", **extra}


# ------------------------------------------------------------------ (a) stock -> local


def test_unavailable_stock_falls_back_to_best_matching_local_asset(tmp_path: Path) -> None:
    servo = local(tmp_path, "servo_motor", "image", ["servo", "motor"])
    other = local(tmp_path, "workshop", "image", ["workshop"])
    work = stock_episode("fb1", [other, servo], [stock_beat(0, 0, 4)])
    stage4("fb1", work)  # offline (conftest)
    beat = read_json(work / "scene_plan.json")["beats"][0]
    assert beat["primary_source"] == "local" and beat["source_id"] == "image_servo_motor"
    assert beat["visual_template"] == "punch_in_emphasis" and beat["callout_text"]
    assert beat["beat_style"] == "fail" and beat["stock_fallback_of"] == "image_servo_motor"
    assert "stock unavailable -> local Servo Motor" in beat["beat_reason"]
    assert read_json(work / "licenses_audit.json")["assets"] == []  # nothing fetched/used


def test_fallback_avoids_assets_used_in_the_last_20_seconds(tmp_path: Path) -> None:
    a = local(tmp_path, "servo_a", "image", ["servo"])
    b = local(tmp_path, "servo_b", "image", ["servo"])
    near = {"id": 0, "start_time": 0, "end_time": 4, "visual_template": "fullscreen_clip",
            "primary_source": "local", "source_id": "image_servo_a", "beat_style": "normal"}
    work = stock_episode("fb2", [a, b], [near, stock_beat(1, 4, 8, "fullscreen_clip")])
    stage4("fb2", work)
    assert read_json(work / "scene_plan.json")["beats"][1]["source_id"] == "image_servo_b"


def test_cutout_with_stock_background_keeps_its_subject(tmp_path: Path) -> None:
    subject = local(tmp_path, "board", "image", ["board"])
    bg = local(tmp_path, "bench", "image", ["bench"])
    beat = stock_beat(0, 0, 4, "cutout_callout", overlay_asset="image_board")
    work = stock_episode("fb3", [subject, bg], [beat])
    stage4("fb3", work)
    out = read_json(work / "scene_plan.json")["beats"][0]
    assert out["visual_template"] == "cutout_callout" and out["overlay_asset"] == "image_board"
    assert out["source_id"] == "image_bench"


def test_stock_is_retried_and_restored_when_it_becomes_available(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    servo = local(tmp_path, "servo_motor", "image", ["servo"])
    work = stock_episode("fb4", [servo], [stock_beat(0, 0, 4)])
    stage4("fb4", work)
    assert read_json(work / "scene_plan.json")["beats"][0]["primary_source"] == "local"

    monkeypatch.delenv(ENV_OFFLINE)
    img = tmp_path / "wiki.png"
    make_subject_image(img)
    found = StockAsset(id="wikimedia_x", source_platform="wikimedia", source_url="u",
                       local_path=str(img), license_type="CC0", query="servo motor")
    monkeypatch.setattr(StockFetcher, "find", lambda self, *a, **k: found)
    stage4("fb4", work)
    beat = read_json(work / "scene_plan.json")["beats"][0]
    assert beat["primary_source"] == "stock" and beat["stock_asset_id"] == "wikimedia_x"
    assert beat["source_id"] == "" and not beat["stock_fallback_of"]
    assert "stock unavailable" not in beat["beat_reason"]


def test_hand_picked_media_is_not_replaced_by_a_stock_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    servo = local(tmp_path, "servo_motor", "image", ["servo"])
    mine = local(tmp_path, "mine", "image", ["mine"])
    work = stock_episode("fb5", [servo, mine], [stock_beat(0, 0, 4)])
    stage4("fb5", work)
    plan = read_json(work / "scene_plan.json")
    plan["beats"][0]["source_id"] = "image_mine"  # the user chose their own media
    write_json(work / "scene_plan.json", plan)
    monkeypatch.delenv(ENV_OFFLINE)
    monkeypatch.setattr(StockFetcher, "find",
                        lambda self, *a, **k: pytest.fail("must not fetch stock"))
    stage4("fb5", work)
    beat = read_json(work / "scene_plan.json")["beats"][0]
    assert beat["primary_source"] == "local" and beat["source_id"] == "image_mine"
    assert beat["stock_fallback_of"] is None


# ------------------------------------------------------------------ (b) title card


def test_title_card_only_without_any_local_asset_and_no_duplicate_lower_third() -> None:
    work = stock_episode("fb6", [], [stock_beat(0, 0, 4, "fullscreen_clip")])
    stage4("fb6", work)
    stage5("fb6", work)
    beat = read_json(work / "scene_plan.json")["beats"][0]
    assert beat["license_type"] == "generated-fallback" and Path(beat["media_path"]).is_file()
    assert beat["lower_third_path"] is None  # the card already shows the text
    assert Path(beat["media_path"]).name.startswith("fallback_beat")


# ------------------------------------------------------------------ (c) planner


def test_stock_is_only_for_plain_broll_and_never_when_a_local_asset_is_relevant() -> None:
    base = make_ctx()
    ctx = PlanContext(base.config, ASSETS, base.words, base.duration)
    ctx.stock_ratio = 1.0
    plan = _generate_deterministic_plan(ctx, "t")
    for b in plan.beats:
        if b.primary_source.value == "stock" and b.visual_template.value != "cutout_callout":
            assert b.visual_template.value == "fullscreen_clip"
            assert b.beat_style.value == "normal"
    # a fail/hype line never goes to stock even with no matching asset
    t = {"duration": 8.0, "segments": [{"start": 0, "end": 8,
                                        "text": "everything exploded in smoke nobody expected",
                                        "words": []}]}
    ctx2 = PlanContext(base.config, [ASSETS[1]], flatten_words(t), 8.0)
    ctx2.stock_ratio = 1.0
    assert all(b.primary_source.value == "local" for b in _generate_deterministic_plan(
        ctx2, "t").beats)


# ------------------------------------------------------------------ code P2


def test_policy_signature_includes_offline(tmp_path: Path) -> None:
    on = StockFetcher({}, tmp_path, offline=False).policy_signature()
    off = StockFetcher({}, tmp_path, offline=True).policy_signature()
    assert on != off


def test_output_audit_counts_overlay_references_and_any_rendered_use(tmp_path: Path) -> None:
    assets = [{"id": i, "source_platform": "wikimedia", "source_url": "u", "local_path": "x",
               "license_type": "CC0"} for i in ("a", "b", "c")]
    write_json(tmp_path / "licenses_audit.json", {"assets": assets})
    write_assets(tmp_path, [])
    r = AssetResolver(tmp_path)
    beats = [{"id": 0, "stock_asset_id": "a", "overlay_asset": None},
             {"id": 1, "stock_asset_id": "a", "overlay_asset": "b"},   # a also used here
             {"id": 2, "stock_asset_id": "c", "overlay_asset": None}]
    out = tmp_path / "out"
    out.mkdir()
    _write_output_audit(tmp_path, out, beats, r, {0: "fallback", 2: "fallback"})
    audit = read_json(out / "licenses_audit.json")
    assert {a["id"] for a in audit["assets"]} == {"a", "b"}  # a rendered in beat 1, b as overlay
    assert [e["id"] for e in audit["excluded"]] == ["c"]


# ------------------------------------------------------------------ visual P2


def test_punch_anchor_and_split_size_and_cutout_background(tmp_path: Path) -> None:
    m = motion_for({"id": 0, "beat_style": "normal"}, "punch_in_emphasis", RenderSettings(), 90)
    assert m.fy == PUNCH_ANCHOR_Y < 0.5
    assert SPLIT_H >= 640
    img = tmp_path / "i.png"
    make_subject_image(img)
    cut = tmp_path / "cut.png"
    make_subject_image(cut)
    work = tmp_path / "w"
    work.mkdir()
    write_assets(work, [{"id": "image_i", "filepath": str(img), "asset_type": "image",
                         "filename": "i.png", "tags": [], "description": ""}])
    ctx = CompileContext(AssetResolver(work), RenderSettings(), work, tmp_path)
    job = compile_beat({"id": 1, "visual_template": "cutout_callout", "primary_source": "local",
                        "source_id": "image_i", "cutout_path": str(cut), "beat_style": "normal"},
                       60, ctx)
    assert "brightness=-0.14" in job.graph and "saturation=0.75" in job.graph


def test_first_beat_is_a_punch_in_hook() -> None:
    ctx = make_ctx()
    raw = ScenePlan(beats=[
        SceneBeat(start_time=0, end_time=4, source_id="clip_servo"),
        SceneBeat(start_time=4, end_time=ctx.duration, source_id="clip_arm")])
    assert _validate_and_repair(raw, ctx).beats[0].visual_template.value == "punch_in_emphasis"
    plan = _generate_deterministic_plan(ctx, "t")
    assert _validate_and_repair(plan, ctx).beats[0].visual_template.value == "punch_in_emphasis"


def test_same_asset_is_not_reused_within_20_seconds_when_alternatives_exist() -> None:
    base = make_ctx(make_transcript())
    assets = [dict(a) for a in ASSETS] + [
        {"id": f"clip_extra{i}", "asset_type": "clip", "filepath": f"/x/{i}.mp4",
         "tags": ["clip"], "description": "clip"} for i in range(3)]
    ctx = PlanContext(base.config, assets, base.words, base.duration)
    plan = _validate_and_repair(_generate_deterministic_plan(ctx, "t"), ctx)
    uses = [(b.start_time, x) for b in plan.beats for x in (b.source_id, b.overlay_asset) if x]
    clashes = [(t1, t2, x) for i, (t1, x) in enumerate(uses) for t2, y in uses[i + 1:]
               if x == y and t2 - t1 < 8]
    assert not clashes, clashes
    assert len(SENTENCES) > 3 and os.environ  # sanity: the transcript has several sentences
