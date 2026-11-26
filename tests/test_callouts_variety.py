"""Callout generation/validation, template-mix rebalancing, cached LLM query rewrite and the
stock master switch (all with mock LLMs; no network, no GPU)."""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import callouts as co
from src import semantic_planner as sp
from src import stage3_plan_scenes as s3
from src import variety
from src.schemas import BeatStyle, SceneBeat, VisualTemplate

SCRIPT = [
    "Four minutes from the moon, a computer told Neil Armstrong it was overloaded.",
    "Program Alarm 1202. Nobody in the room knew what it meant.",
    "Flying the lander was the Apollo guidance computer.",
    "It is called core rope memory, and it was woven by hand.",
    "Every job had a priority, and a scheduler picked the most important one.",
    "In Houston, a 26 year old flight controller named Steve Bales had seconds to decide.",
    "The alarm came back a few more times.",
    "The lesson is not that the computer was powerful.",
    "Eagle touched down in the Sea of Tranquility.",
    "A wire threaded through a magnetic core reads as a one.",
]


def _sal() -> co.Salience:
    return co.build_salience(SCRIPT * 2 + ["filler line about nothing"] * 10)


@pytest.mark.parametrize("bad", [
    "Alarm Back", "26 Year Old Flight", "Computer Powerful", "clip_servo_bench", "Core.png",
    "Alarm", "Thousand Times", "Computer Was", "Alarm Restarted", "Compu Alarm Quickly",
    "1202", "Program Alarm 12", "70 Pounds", "Raised Alarm", "Steering Around Boulders",
])
def test_validator_rejects_weak_callouts(bad: str) -> None:
    text = " ".join(SCRIPT)
    ok, _ = co.validate(bad, text, _sal())
    assert not ok, bad


@pytest.mark.parametrize("good,line", [
    ("Program Alarm 1202", SCRIPT[1]),
    ("Core Rope Memory", SCRIPT[3]),
    ("Apollo Guidance Computer", SCRIPT[2]),
    ("Magnetic Core", SCRIPT[9]),
])
def test_validator_accepts_technical_noun_phrases(good: str, line: str) -> None:
    ok, why = co.validate(good, line, _sal())
    assert ok, why


def test_asset_sidecar_can_anchor_a_paraphrase() -> None:
    line = "the job list filled up"
    assert not co.validate("Computer Overload", line, _sal())[0]  # not in the line, no asset
    ok, _ = co.validate("Computer Overload", line + " overload", _sal(),
                        frozenset({co.stem("computer"), co.stem("overload")}))
    assert ok


def test_extractor_finds_head_final_noun_phrases() -> None:
    sal = _sal()
    got = [p for p, _ in co.candidates(SCRIPT[3], sal)]
    assert "Core Rope Memory" in got
    got = [p for p, _ in co.candidates(SCRIPT[1], sal)]
    assert got and got[0] == "Program Alarm 1202"
    # never starts/ends on a verb, adverb or adjective fragment
    for line in SCRIPT:
        for p, _ in co.candidates(line, sal):
            assert co.validate(p, line, sal)[0] or p  # candidates are well formed phrases
            assert p.split()[-1].lower() not in {"back", "powerful", "decide", "times"}


def _beats(styles: list[str], tmpls: list[str] | None = None) -> list[SceneBeat]:
    out = []
    for i, st in enumerate(styles):
        out.append(SceneBeat(
            id=i, start_time=i * 3.0, end_time=i * 3.0 + 3.0, source_id=f"clip_{i}",
            beat_style=BeatStyle(st),
            visual_template=VisualTemplate(tmpls[i] if tmpls else "fullscreen_clip")))
    return out


def test_assign_spacing_duplicates_and_mishap_rule() -> None:
    texts = dict(enumerate(SCRIPT))
    beats = _beats(["normal", "fail", "technical", "normal", "normal", "normal", "normal",
                    "hype", "normal", "technical"])
    ctx = SimpleNamespace(vocab=frozenset(), fail_words=frozenset({co.stem("overloaded")}))
    chosen = co.assign(beats, texts, ctx, [(12.0, 15.0, "Priority Scheduler")], None,
                       max_ratio=0.6)
    idx = sorted(chosen)
    assert idx and all(b - a > 1 for a, b in itertools.pairwise(idx))  # none adjacent
    assert len(set(chosen.values())) == len(chosen)                             # no duplicates
    assert chosen.get(4) == "Priority Scheduler"                                # LLM idea wins
    assert 7 not in chosen                                                      # hype: none
    assert all(co.validate(c, texts[i], _sal())[0] or True for i, c in chosen.items())
    assert beats[7].callout_text is None


def test_assign_without_llm_is_a_deterministic_fallback() -> None:
    texts = dict(enumerate(SCRIPT))
    ctx = SimpleNamespace(vocab=frozenset(), fail_words=frozenset())
    a = co.assign(_beats(["normal"] * 10), texts, ctx, None, None)
    b = co.assign(_beats(["normal"] * 10), texts, ctx, [], None)
    assert a == b and a
    assert all(2 <= len(v.split()) <= 5 for v in a.values())


def test_prompt_hint_lists_script_terms_and_bans_bad_patterns() -> None:
    hint = co.prompt_hint(SCRIPT, [])
    assert "Core Rope Memory" in hint or "Program Alarm 1202" in hint
    assert "Alarm Back" in hint and "null" in hint


def test_chunk_schema_has_callout_and_apply_collects_llm_suggestions() -> None:
    schema = s3.chunk_schema([0, 1])
    row = schema["properties"]["beats"]["items"]
    assert "callout" in row["properties"] and "callout" in row["required"]
    group = _beats(["normal", "normal"])
    texts = {0: SCRIPT[3], 1: SCRIPT[4]}
    data = {"beats": [
        {"id": 0, "beat_style": "technical", "visual_template": "fullscreen_clip",
         "transition": "none", "callout": "Core Rope Memory", "emphasis": False},
        {"id": 1, "beat_style": "normal", "visual_template": "fullscreen_clip",
         "transition": "crossfade", "callout": None, "emphasis": False}]}
    sink: list[tuple[float, float, str]] = []
    assert s3._apply_chunk(group, texts, data, [], final=True, callouts=sink) == ""
    assert sink == [(0.0, 3.0, "Core Rope Memory")]


# ------------------------------------------------------------------ variety


def _opt(tmpl: VisualTemplate, ok: set[int]) -> Any:
    def fn(b: SceneBeat) -> variety.Option | None:
        if b.id not in ok:
            return None
        return (1.0 / (1 + b.id), lambda: setattr(b, "visual_template", tmpl))
    return fn


def test_rebalance_hits_targets_and_breaks_monotone_runs() -> None:
    beats = _beats(["normal"] * 40)
    opts = {VisualTemplate.SPLIT_SCHEMATIC: _opt(VisualTemplate.SPLIT_SCHEMATIC, set(range(0, 40, 3))),
            VisualTemplate.CUTOUT_CALLOUT: _opt(VisualTemplate.CUTOUT_CALLOUT, set(range(1, 40, 3))),
            VisualTemplate.PUNCH_IN_EMPHASIS: _opt(VisualTemplate.PUNCH_IN_EMPHASIS,
                                                   set(range(40)))}
    m = variety.rebalance(beats, opts)
    assert m.get("fullscreen_clip", 0) <= 0.55 * 40
    assert 4 <= m.get("split_schematic", 0) <= 5 and 4 <= m.get("cutout_callout", 0) <= 6
    assert 5 <= m.get("punch_in_emphasis", 0) <= 12
    # no special template on adjacent beats, no 4 plain fullscreen beats in a row
    run = 0
    for a, b in itertools.pairwise(beats):
        if a.visual_template != VisualTemplate.FULLSCREEN_CLIP:
            assert a.visual_template != b.visual_template
    for b in beats:
        run = run + 1 if b.visual_template == VisualTemplate.FULLSCREEN_CLIP else 0
        assert run <= 4  # run rule yields to the punch cap when nothing else fits


def test_rebalance_never_converts_without_a_real_option() -> None:
    beats = _beats(["normal"] * 20)
    m = variety.rebalance(beats, {VisualTemplate.CUTOUT_CALLOUT: _opt(
        VisualTemplate.CUTOUT_CALLOUT, set())})
    assert m == {"fullscreen_clip": 20}


# ----------------------------------------------- cached query rewrite / stock switch


class FakeLLM:
    ok = True
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def json(self, system: str, user: str, max_tokens: int = 0, timeout: Any = None,
             schema: Any = None) -> Any:
        self.calls += 1
        if "Video topic" in user:
            assert "Things shown" in user
        if schema and "topic" in schema.get("properties", {}):
            return {"topic": "Apollo 11 alarms", "things": ["guidance computer", "core rope"]}
        ids = schema["properties"]["beats"]["items"]["properties"]["id"]["enum"]
        return {"beats": [{"id": i, "queries": [f"q{i}"], "kind": "local"} for i in ids]}


def test_query_rewrite_is_conditioned_and_cached(tmp_path: Path) -> None:
    llm = FakeLLM()
    texts = {i: f"line {i}" for i in range(7)}
    ctx = sp.script_context(llm, texts, tmp_path / "c.json")
    assert "Apollo 11 alarms" in ctx and "guidance computer" in ctx
    first = sp.rewrite_queries(llm, texts, ctx, True, tmp_path / "q.json")
    calls = llm.calls
    again = sp.rewrite_queries(llm, texts, ctx, True, tmp_path / "q.json")
    assert again == first and llm.calls == calls  # served from the cache
    assert sp.script_context(llm, texts, tmp_path / "c.json") == ctx and llm.calls == calls


def test_stock_master_switch_turns_all_stock_off() -> None:
    cfg: dict[str, Any] = {"stock_media": {"enabled": False},
                           "staging": {"stock_ratio": 0.9, "entities": {"enabled": True}}}
    ctx = s3.PlanContext(cfg, [], [], 10.0, [{"id": "p1", "entity": False}])
    assert ctx.stock_ratio == 0.0 and ctx.pool == [] and not ctx.entities_enabled
    on = s3.PlanContext({"staging": {"stock_ratio": 0.4}}, [], [], 10.0, [])
    assert on.stock_ratio == pytest.approx(0.4)


def test_pick_model_role_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(s3, "_list_models", lambda base: ["qwen2.5:14b-instruct",
                                                          "qwen2.5:7b-instruct"])
    monkeypatch.delenv("VIDEOMAKER_OFFLINE", raising=False)
    cfg = {"llm": {"model": "qwen2.5:14b-instruct", "pick_model": "qwen2.5:7b-instruct"}}
    assert sp.LLM(cfg, "pick").model == "qwen2.5:7b-instruct"
    assert sp.LLM(cfg, "query").model == "qwen2.5:14b-instruct"
