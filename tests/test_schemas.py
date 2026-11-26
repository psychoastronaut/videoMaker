from src.schemas import BeatStyle, SceneBeat, ScenePlan


def test_tone_is_alias_for_beat_style() -> None:
    beat = SceneBeat(start_time=0, end_time=2, tone="hype")
    assert beat.beat_style == BeatStyle.HYPE
    assert beat.tone == "hype"


def test_unknown_style_defaults_to_normal() -> None:
    beat = SceneBeat(start_time=0, end_time=2, beat_style="excited")
    assert beat.beat_style == BeatStyle.NORMAL


def test_beat_style_wins_over_tone_when_both_given() -> None:
    beat = SceneBeat(start_time=0, end_time=2, tone="hype", beat_style="fail")
    assert beat.beat_style == BeatStyle.FAIL
    assert beat.tone == "fail"


def test_callout_is_cleaned_and_limited_to_five_words() -> None:
    beat = SceneBeat(start_time=0, end_time=2, callout_text="  the *quick* brown fox, jumps over!! ")
    assert beat.callout_text == "the quick brown fox jumps"
    assert SceneBeat(start_time=0, end_time=2, callout_text="!!!").callout_text is None


def test_total_duration_is_serialized() -> None:
    plan = ScenePlan(beats=[SceneBeat(start_time=0, end_time=2), SceneBeat(start_time=2, end_time=5)])
    dumped = plan.model_dump(mode="json")
    assert dumped["total_duration"] == 5.0
    assert [b["id"] for b in dumped["beats"]] == [0, 1]
    # round trip tolerates the computed field
    assert ScenePlan.model_validate(dumped).total_duration == 5.0


def test_duration_follows_times() -> None:
    beat = SceneBeat(start_time=1.0, end_time=3.5, duration=99)
    assert beat.duration == 2.5
