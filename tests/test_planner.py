from __future__ import annotations

import itertools

from src.common import load_config
from src.schemas import PrimarySource, SceneBeat, ScenePlan, VisualTemplate
from src.stage3_plan_scenes import (
    PlanContext,
    _generate_deterministic_plan,
    _validate_and_repair,
    clean_query,
    extract_callout,
    flatten_words,
)

ASSETS = [
    {"id": "clip_servo", "asset_type": "clip", "filepath": "/x/servo.mp4", "tags": ["clip", "servo"],
     "description": "clip servo"},
    {"id": "clip_arm", "asset_type": "clip", "filepath": "/x/arm.mp4", "tags": ["clip", "arm"],
     "description": "clip arm"},
    {"id": "image_board", "asset_type": "image", "filepath": "/x/board.png", "tags": ["image", "board"],
     "description": "image board"},
    {"id": "schematic_loop", "asset_type": "schematic", "filepath": "/x/loop.png",
     "tags": ["schematic", "loop"], "description": "schematic loop"},
]

SENTENCES = [
    "I built a robot arm that pours coffee and it works most days.",
    "The signal goes through the control loop a thousand times per second, every second.",
    "Then the battery shorted and there was magic smoke everywhere in the workshop today.",
    "Of course I wired it backwards, apparently reading datasheets is optional for me.",
    "It finally worked and the arm poured a perfect cup of coffee for everybody.",
    "Thanks for watching and see you in the next video about duct tape.",
]


def make_transcript(gap: float = 0.4) -> dict:
    segments, t = [], 0.0
    for i, text in enumerate(SENTENCES):
        tokens = text.split()
        words = []
        for tok in tokens:
            words.append({"word": tok, "start": round(t, 3), "end": round(t + 0.33, 3)})
            t += 0.36
        segments.append({"id": i, "start": words[0]["start"], "end": words[-1]["end"],
                         "text": text, "words": words})
        t += gap
    return {"duration": round(t, 3), "segments": segments, "full_text": " ".join(SENTENCES)}


NEUTRAL = [
    "A quiet workshop table holds screwdrivers pliers hammers and many little tools today.",
    "Another shelf stores cables connectors batteries and spare parts for later projects.",
    "Windows overlook the garden where tomatoes peppers and herbs grow every summer.",
]


def neutral_transcript() -> dict:
    """Narration with no fail/hype/technical keywords (so beats stay 'normal')."""
    segments, t = [], 0.0
    for i, text in enumerate(NEUTRAL):
        words = []
        for tok in text.split():
            words.append({"word": tok, "start": round(t, 3), "end": round(t + 0.33, 3)})
            t += 0.36
        segments.append({"id": i, "start": words[0]["start"], "end": words[-1]["end"],
                         "text": text, "words": words})
        t += 0.4
    return {"duration": round(t, 3), "segments": segments, "full_text": " ".join(NEUTRAL)}


def make_ctx(transcript: dict | None = None) -> PlanContext:
    transcript = transcript or make_transcript()
    return PlanContext(load_config(), ASSETS, flatten_words(transcript), transcript["duration"])


def test_deterministic_plan_covers_audio_without_gaps_or_drift() -> None:
    ctx = make_ctx()
    plan = _validate_and_repair(_generate_deterministic_plan(ctx, "t"), ctx)
    beats = plan.beats
    assert beats[0].start_time == 0.0
    assert abs(beats[-1].end_time - ctx.duration) < 1e-6
    for a, b in itertools.pairwise(beats):
        assert abs(a.end_time - b.start_time) < 1e-6
    for b in beats:
        assert ctx.min_beat - 1e-6 <= b.duration <= ctx.max_beat + 1e-6
        assert b.beat_reason
    assert plan.total_duration == beats[-1].end_time


def test_long_sentences_are_split_into_several_beats() -> None:
    ctx = make_ctx()
    plan = _generate_deterministic_plan(ctx, "t")
    assert len(plan.beats) > len(SENTENCES)


def test_styles_and_templates_follow_config_keywords() -> None:
    ctx = make_ctx()
    beats = _validate_and_repair(_generate_deterministic_plan(ctx, "t"), ctx).beats
    styles = {b.beat_style.value for b in beats}
    assert {"fail", "technical", "hype", "deadpan"} <= styles
    templates = {b.visual_template for b in beats}
    assert VisualTemplate.SPLIT_SCHEMATIC in templates
    assert VisualTemplate.PUNCH_IN_EMPHASIS in templates
    for b in beats:
        if b.visual_template == VisualTemplate.CUTOUT_CALLOUT:
            assert b.callout_text and 1 <= len(b.callout_text.split()) <= 5


def test_no_template_used_three_times_in_a_row() -> None:
    ctx = make_ctx()
    beats = _generate_deterministic_plan(ctx, "t").beats
    for a, b, c in zip(beats, beats[1:], beats[2:], strict=False):
        assert not (a.visual_template == b.visual_template == c.visual_template)


def test_repair_fills_gaps_splits_long_and_fixes_assets() -> None:
    ctx = make_ctx()
    d = ctx.duration
    raw = ScenePlan(beats=[
        SceneBeat(start_time=0.0, end_time=2.0, source_id="clip_does_not_exist"),
        SceneBeat(start_time=3.0, end_time=3.4, source_id="clip_arm"),  # gap before, too short
        SceneBeat(start_time=3.4, end_time=d - 5, source_id="clip_servo"),  # too long
        SceneBeat(start_time=d - 5, end_time=d - 1, visual_template="split_schematic",
                  source_id="schematic_loop", overlay_asset="bogus"),
    ])
    plan = _validate_and_repair(raw, ctx)
    ids = {a["id"] for a in ASSETS}
    assert plan.beats[0].start_time == 0.0
    assert abs(plan.beats[-1].end_time - d) < 1e-6
    for a, b in itertools.pairwise(plan.beats):
        assert abs(a.end_time - b.start_time) < 1e-6
    for b in plan.beats:
        assert b.duration <= ctx.max_beat + 1e-6
        assert b.duration >= ctx.min_beat - 1e-6
        if b.primary_source == PrimarySource.LOCAL:
            assert b.source_id in ids
        if b.visual_template == VisualTemplate.SPLIT_SCHEMATIC:
            assert b.overlay_asset in ids
            assert b.overlay_asset == "schematic_loop"
            assert b.source_id != "schematic_loop"


def test_repair_normalises_stock_queries() -> None:
    ctx = make_ctx(neutral_transcript())
    raw = ScenePlan(beats=[SceneBeat(
        start_time=0, end_time=ctx.duration, primary_source="stock", source_id="x",
        search_queries=["The big, red!! robot arm of the future is here today ok"])])
    beat = _validate_and_repair(raw, ctx).beats[0]
    assert beat.source_id == ""
    assert beat.search_queries
    assert all(len(q.split()) <= 6 and "!" not in q for q in beat.search_queries)


def test_stock_beat_without_query_gets_one_from_text() -> None:
    ctx = make_ctx(neutral_transcript())
    raw = ScenePlan(beats=[SceneBeat(start_time=0, end_time=ctx.duration,
                                     primary_source="stock")])
    beat = _validate_and_repair(raw, ctx).beats[0]
    assert beat.primary_source == PrimarySource.STOCK
    assert beat.search_queries and beat.search_queries[0]


def test_callout_and_query_helpers() -> None:
    assert extract_callout("There was a puff of magic smoke", "magic smoke") == "magic smoke"
    assert 2 <= len(extract_callout("the signal timing looked clean").split()) <= 5
    assert clean_query("Oh, the robot's arm!! of doom") == "robot arm doom"
