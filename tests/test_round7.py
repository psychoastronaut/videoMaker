"""Round-7: guaranteed demo stock beat, query/callout hygiene, relevance, mood mismatch."""

from __future__ import annotations

from test_planner import ASSETS, make_ctx, neutral_transcript

from src.demo_generator import SENTENCES, _build_segments
from src.schemas import BeatStyle, PrimarySource
from src.stage3_plan_scenes import (
    PlanContext,
    _generate_deterministic_plan,
    _pick_asset,
    _PlannerState,
    _validate_and_repair,
    extract_callout,
    flatten_words,
    stock_query,
)
from src.stage4_fetch_stock import relevant


def test_demo_script_has_a_plain_broll_line_with_concrete_nouns() -> None:
    line = next(t for t, _ in SENTENCES if "soldering" in t)
    assert len(stock_query(line).split()) >= 2
    assert any(line == s["text"] for s in _build_segments())


def test_demo_plan_has_a_stock_beat_after_repair() -> None:
    """Same path the pipeline takes: demo transcript -> planner -> repair."""
    segs = _build_segments()
    transcript = {"duration": segs[-1]["end"] + 0.6, "segments": segs}
    base = make_ctx()
    assets = [  # the demo's local assets (ids/tags as indexed from the demo files)
        {"id": i, "asset_type": t, "filepath": f"/x/{i}", "tags": [t, *i.split("_")[1:]],
         "description": i.replace("_", " ")}
        for i, t in [("clip_servo_bench_test", "clip"), ("clip_motor_driver_smoke", "clip"),
                     ("clip_workshop_timelapse", "clip"), ("clip_robot_arm_demo", "clip"),
                     ("clip_scope_trace", "clip"), ("image_microcontroller_board", "image"),
                     ("image_servo_motor", "image"), ("image_battery_pack", "image"),
                     ("image_duct_tape_hack", "image"),
                     ("schematic_motor_driver_circuit", "schematic"),
                     ("schematic_control_loop_diagram", "schematic"),
                     ("schematic_signal_timing", "schematic")]]
    ctx = PlanContext(base.config, assets, flatten_words(transcript), transcript["duration"])
    plan = _validate_and_repair(_generate_deterministic_plan(ctx, "demo"), ctx)
    stock = [b for b in plan.beats if b.primary_source == PrimarySource.STOCK]
    assert stock and all(b.search_queries and len(b.search_queries[0].split()) >= 2
                         for b in stock)
    assert all(b.beat_style == BeatStyle.NORMAL for b in stock)


def test_planner_forces_one_stock_beat_when_a_normal_line_qualifies() -> None:
    ctx = make_ctx(neutral_transcript())
    ctx.stock_ratio = 0.01  # budget rounds to 0, but stock is enabled
    plan = _generate_deterministic_plan(ctx, "t")
    assert any(b.primary_source == PrimarySource.STOCK for b in plan.beats)
    ctx.stock_ratio = 0.0  # disabled -> never
    assert not any(b.primary_source == PrimarySource.STOCK
                   for b in _generate_deterministic_plan(ctx, "t").beats)


def test_no_stock_forced_when_every_line_has_a_relevant_local_asset() -> None:
    t = {"duration": 8.0, "segments": [{"start": 0, "end": 8,
                                        "text": "servo motors under the workshop table",
                                        "words": []}]}
    base = make_ctx()
    assets = [{"id": "clip_servo_workshop", "asset_type": "clip", "filepath": "/x",
               "tags": ["clip", "servo", "workshop"], "description": "servo workshop"}]
    ctx = PlanContext(base.config, assets, flatten_words(t), 8.0)
    assert not any(b.primary_source == PrimarySource.STOCK
                   for b in _generate_deterministic_plan(ctx, "t").beats)


def test_onomatopoeia_is_dropped_from_callouts_and_queries() -> None:
    assert "pop" not in extract_callout("There was a pop, a puff of magic smoke").lower().split()
    assert not {"boom", "bang"} & set(extract_callout("boom bang servo motor").lower().split())
    assert stock_query("pop puff boom soldering iron") == "soldering iron"


def test_stock_queries_drop_light_and_irregular_verbs() -> None:
    assert stock_query("the servo kept bench") == "servo bench"
    assert stock_query("she took the soldering iron") == "soldering iron"


def test_two_word_queries_need_both_words_in_title_and_support_in_tags() -> None:
    assert relevant("oscilloscope trace", "Oscilloscope trace on screen")  # good title, no tags
    assert relevant("oscilloscope trace", "Oscilloscope trace on screen", "lab, measurement,"
                    " trace")
    assert not relevant("oscilloscope trace", "Oscilloscope", "nothing about it")
    assert not relevant("oscilloscope trace", "Oscilloscope trace", "cats, dogs, garden")


def test_smoke_clip_is_not_chosen_for_a_hype_line_when_alternatives_exist() -> None:
    base = make_ctx()
    assets = [
        {"id": "clip_motor_smoke", "asset_type": "clip", "filepath": "/x/a",
         "tags": ["clip", "motor", "smoke"], "description": "motor smoke"},
        {"id": "clip_robot_arm", "asset_type": "clip", "filepath": "/x/b",
         "tags": ["clip", "robot", "arm"], "description": "robot arm"}]
    ctx = PlanContext(base.config, assets, base.words, base.duration)
    st = _PlannerState(style=BeatStyle.HYPE)
    picked, _ = _pick_asset(ctx, ("clip",), "the motor finally worked", st)
    assert picked and picked["id"] == "clip_robot_arm"
    st_fail = _PlannerState(style=BeatStyle.FAIL)
    picked, _ = _pick_asset(ctx, ("clip",), "the motor caught fire", st_fail)
    assert picked and picked["id"] == "clip_motor_smoke"  # fail lines still may use it
    assert ASSETS
