"""Round-6: plan signature after Stage 4, relevance gating, stock policy, callouts, authors."""

from __future__ import annotations

from pathlib import Path

from test_planner import ASSETS, make_ctx, neutral_transcript
from test_round5 import local, stock_beat, stock_episode

from src import cli
from src.common import read_json, write_json
from src.schemas import SceneBeat, ScenePlan, StockAsset, VisualTemplate
from src.stage3_plan_scenes import (
    PlanContext,
    _generate_deterministic_plan,
    _validate_and_repair,
    stock_ok_for,
    stock_query,
)


def test_offline_stage4_downgrade_is_not_a_hand_edit(tmp_path: Path) -> None:
    servo = local(tmp_path, "servo_motor", "image", ["servo"])
    work = stock_episode("sg1", [servo], [stock_beat(0, 0, 4, "fullscreen_clip",
                                                    beat_style="normal")])
    cli._record_state("plan", work)  # what Stage 3 does after writing the plan
    assert not cli.plan_is_hand_edited(work)

    cli._execute(cli.STAGE_BY_KEY["fetch"], "sg1", work)  # offline: downgrade to local media
    beat = read_json(work / "scene_plan.json")["beats"][0]
    assert beat["primary_source"] == "local" and beat["stock_fallback_of"]
    assert not cli.plan_is_hand_edited(work)  # `plan` / `run` may regenerate freely

    plan = read_json(work / "scene_plan.json")  # a REAL hand edit is still detected
    plan["beats"][0]["callout_text"] = "Hand edit"
    write_json(work / "scene_plan.json", plan)
    assert cli.plan_is_hand_edited(work)


def test_hand_edit_before_stage4_stays_protected(tmp_path: Path) -> None:
    servo = local(tmp_path, "servo_motor", "image", ["servo"])
    work = stock_episode("sg2", [servo], [stock_beat(0, 0, 4, "fullscreen_clip",
                                                    beat_style="normal")])
    cli._record_state("plan", work)
    plan = read_json(work / "scene_plan.json")
    plan["beats"][0]["callout_text"] = "Mine"
    write_json(work / "scene_plan.json", plan)
    cli._execute(cli.STAGE_BY_KEY["fetch"], "sg2", work)
    assert cli.plan_is_hand_edited(work)  # not re-stamped: the edit is still protected


def test_stock_queries_need_two_nouns_without_adjectives() -> None:
    assert stock_query("and the whole bench went dark") == ""
    assert stock_query("a clean scope") == ""
    assert stock_query("finally the dark workshop bench shows screwdrivers") != ""
    assert "dark" not in stock_query("the dark workshop bench shows screwdrivers").split()


def test_stock_only_for_normal_fullscreen_or_cutout_backgrounds() -> None:
    from src.schemas import BeatStyle
    assert stock_ok_for(VisualTemplate.FULLSCREEN_CLIP, BeatStyle.NORMAL)
    assert stock_ok_for(VisualTemplate.CUTOUT_CALLOUT, BeatStyle.NORMAL)
    for style in (BeatStyle.FAIL, BeatStyle.HYPE, BeatStyle.TECHNICAL):
        assert not stock_ok_for(VisualTemplate.FULLSCREEN_CLIP, style)
        assert not stock_ok_for(VisualTemplate.CUTOUT_CALLOUT, style)
    assert not stock_ok_for(VisualTemplate.PUNCH_IN_EMPHASIS, BeatStyle.NORMAL)
    assert not stock_ok_for(VisualTemplate.SPLIT_SCHEMATIC, BeatStyle.NORMAL)


def test_planner_and_repair_never_stock_fail_or_technical_beats() -> None:
    base = make_ctx()
    ctx = PlanContext(base.config, ASSETS, base.words, base.duration)
    ctx.stock_ratio = 1.0
    for b in _generate_deterministic_plan(ctx, "t").beats:
        if b.primary_source.value == "stock":
            assert stock_ok_for(b.visual_template, b.beat_style) or (
                b.stock_media == "image" and b.visual_template.value != "punch_in_emphasis")
    # LLM-style plan asking for stock on a fail cutout and a technical beat -> local media
    raw = ScenePlan(beats=[
        SceneBeat(start_time=0, end_time=base.duration / 2, beat_style="fail",
                  visual_template="cutout_callout", primary_source="stock",
                  overlay_asset="image_board", search_queries=["burnt circuit board"]),
        SceneBeat(start_time=base.duration / 2, end_time=base.duration, beat_style="technical",
                  visual_template="fullscreen_clip", primary_source="stock",
                  search_queries=["signal control loop"])])
    for b in _validate_and_repair(raw, ctx).beats:
        assert b.primary_source.value == "local" and b.source_id and not b.search_queries


def test_neutral_broll_may_still_use_stock() -> None:
    ctx = make_ctx(neutral_transcript())
    raw = ScenePlan(beats=[SceneBeat(start_time=0, end_time=ctx.duration,
                                     primary_source="stock", search_queries=["workshop tools"])])
    assert _validate_and_repair(raw, ctx).beats[0].primary_source.value == "stock"


def test_fail_callouts_come_from_the_mishap_phrase_not_asset_labels() -> None:
    ctx = make_ctx()
    out = ctx.mishap_callout("The battery pack had shorted, and my duct tape did not survive",
                             "shorted")
    assert out and "shorted" in out.lower() and 2 <= len(out.split()) <= 5
    assert "scope" not in (ctx.mishap_callout("a puff of magic smoke", "magic smoke") or "").lower()
    assert ctx.mishap_callout("okay then", "") is None
    plan = _generate_deterministic_plan(ctx, "t")
    for b in plan.beats:
        if b.visual_template.value in ("fullscreen_clip", "punch_in_emphasis") and b.callout_text:
            assert b.callout_text.lower() not in {"scope trace", "servo detail", "arm detail"}


def test_author_field_is_cleaned() -> None:
    def author(v: str | None) -> str | None:
        return StockAsset(id="a", source_platform="p", source_url="u", local_path="x",
                          license_type="CC0", author=v).author
    assert author("  Jane \\n Doe  ".replace("\\n", "\n")) == "Jane Doe"
    assert author("<a href='x'>Jane</a>\n<b>Doe</b>") == "Jane Doe"
    assert author("Unknown author") is None
    assert author("Unknown") is None
    assert author("   \n ") is None
    assert author(None) is None
    assert author("Jane Doe Unknown author") == "Jane Doe"
