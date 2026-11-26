"""Tuning round: narration callouts, legible diagrams, chunked LLM planning, retrieval priors,
small photos and the sidecar ``media_start`` hint."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from test_planner import make_ctx

from src import semantic_planner as sp
from src import stage3_plan_scenes as s3
from src.schemas import BeatStyle, SceneBeat, ScenePlan, VisualTemplate
from src.semantic_index import Hit, SemanticIndex

# ---------------------------------------------------------------- 1. callouts


def test_callout_comes_from_narration_title_case() -> None:
    out = s3.narration_callout("The computer told Neil the processor was overloaded.")
    words = out.split()
    assert 2 <= len(words) <= 5
    assert all(w[0].isupper() for w in words)
    assert s3.callout_from_narration(out, "The computer told Neil the processor was overloaded.")


def test_callout_never_a_filename_and_rejects_foreign_words() -> None:
    text = "the guidance computer kept running"
    assert not s3.callout_from_narration("Computer Told Neil", text)
    assert not s3.callout_from_narration("Clip Astronaut At Lm Cockpit", text)
    assert s3.narration_callout("Thanks for watching.") == ""


def test_finalize_callouts_rebuilds_filename_like_text() -> None:
    ctx = make_ctx()
    b = SceneBeat(id=0, start_time=0, end_time=3, visual_template="cutout_callout",
                  source_id="clip_servo", overlay_asset="image_board", callout_text="Computer Told")
    sp.finalize_callouts([b], ctx, {0: "the servo drives the arm joint"})
    assert s3.callout_from_narration(b.callout_text, "the servo drives the arm joint")


def _hs(*pairs: tuple[str, str, float]) -> list[Hit]:
    return [Hit(a, s, None, None, "", t, s) for a, t, s in pairs]


def test_cutout_without_clear_subject_becomes_fullscreen_with_lower_third() -> None:
    ctx = SimpleNamespace(
        assets_by_id={"image_view": {"id": "image_view", "asset_type": "image",
                                     "description": "image | x | Panoramic view of the lunar surface"}},
        callout=lambda t: "Lunar Surface")
    b = SceneBeat(id=0, start_time=0, end_time=3, visual_template="cutout_callout",
                  source_id="image_view", overlay_asset="image_view")
    hs = _hs(("image_view", "image", 0.3))
    calls: list[Any] = []
    sp._assign_cutout(b, hs, hs[0], ctx, 0.1, lambda h, bb: calls.append(h), "the lunar surface")
    assert b.visual_template == VisualTemplate.FULLSCREEN_CLIP
    assert b.overlay_asset is None
    assert b.callout_text == "Lunar Surface"


def test_cutout_subject_is_own_match_and_background_not_unrelated() -> None:
    ctx = SimpleNamespace(
        assets_by_id={"image_astro": {"id": "image_astro", "asset_type": "image",
                                      "description": "image | x | Astronaut in a white spacesuit"}},
        callout=lambda t: "Astronaut")
    b = SceneBeat(id=0, start_time=0, end_time=3, visual_template="cutout_callout",
                  source_id="clip_other", overlay_asset="image_other")
    hs = _hs(("image_astro", "image", 0.3), ("clip_far", "clip", 0.12))
    sp._assign_cutout(b, hs, hs[0], ctx, 0.1, lambda h, bb: None, "the astronaut")
    assert b.visual_template == VisualTemplate.CUTOUT_CALLOUT
    assert b.overlay_asset == "image_astro"
    assert b.source_id == "image_astro"  # weak runner-up is not used as an unrelated background


# ---------------------------------------------------------------- 2. schematics


def test_diagram_about_its_content_is_fullscreen_without_caption() -> None:
    schem = {"id": "s1", "asset_type": "schematic", "width": 1920, "height": 1080,
             "tags": ["schematic"], "description": "Flowchart: alarm 1202 overload restart"}
    ctx = SimpleNamespace(assets_by_id={"s1": schem}, callout=s3.narration_callout)
    b = SceneBeat(id=0, start_time=0, end_time=4, visual_template="fullscreen_clip",
                  source_id="clip_x")
    side = Hit("clip_x", 0.2, 0, 3, "", "clip", 0.2)
    n = sp._assign_schematic(b, Hit("s1", 0.1, None, None, "", "schematic", 0.1), [], ctx, side,
                             "the alarm overload restart flowchart", None)
    assert n == 1 and b.visual_template == VisualTemplate.FULLSCREEN_CLIP
    assert b.source_id == "s1" and b.overlay_asset is None and not b.callout_text  # carries its own text


def test_big_diagram_not_kept_in_split_panel() -> None:
    assert sp.schematic_scale_in_split({"width": 4000, "height": 2250}) < sp.MIN_SPLIT_SCALE
    assert sp.schematic_scale_in_split({"width": 640}) >= sp.MIN_SPLIT_SCALE


# ---------------------------------------------------------------- 3. chunked LLM


def _plan_ctx() -> Any:
    return make_ctx()


def test_chunked_llm_failure_falls_back_per_chunk(monkeypatch: Any) -> None:
    ctx = _plan_ctx()
    ctx.config.setdefault("llm", {})["chunk_beats"] = 3
    monkeypatch.delenv("VIDEOMAKER_OFFLINE", raising=False)
    monkeypatch.setattr(s3, "is_offline", lambda: False)
    monkeypatch.setattr(s3, "_list_models", lambda base: ["qwen2.5:7b-instruct"])
    base = s3._generate_deterministic_plan(ctx, "t")
    calls = {"n": 0}

    def fake(client: Any, url: str, payload: dict, retries: int, **kw: Any) -> str:
        calls["n"] += 1
        user = payload["messages"][-1]["content"]
        if calls["n"] == 1:
            return json.dumps({"beats": [{"id": 999}]})  # invalid -> retried with the error
        if calls["n"] == 2:
            raise s3.httpx.ReadTimeout("slow")  # chunk 1 times out -> deterministic
        ids = [int(x.split(":")[0]) for x in user.splitlines() if x[:1].isdigit()]
        return json.dumps({"beats": [{"id": i, "beat_style": "hype",
                                      "visual_template": "punch_in_emphasis",
                                      "callout_text": None} for i in ids]})

    monkeypatch.setattr(s3, "chat_with_backoff", fake)
    plan = s3._try_llm_plan({"segments": []}, ctx, "t")
    assert plan is not None and len(plan.beats) == len(base.beats)
    assert [b.beat_style for b in plan.beats[:3]] == [b.beat_style for b in base.beats[:3]]
    assert any(b.beat_style == BeatStyle.HYPE for b in plan.beats[3:])
    assert [(b.start_time, b.end_time) for b in plan.beats] == \
           [(b.start_time, b.end_time) for b in base.beats]


def test_semantic_llm_chunks_and_breaker() -> None:
    sizes: list[int] = []

    class FakeLLM(sp.LLM):
        def __init__(self) -> None:  # no endpoint
            self.ok = True
            self.fails = 0

        def json(self, system: str, user: str, max_tokens: int = 0, timeout: Any = None, schema: Any = None) -> Any:
            sizes.append(user.count('\n') )
            return None

    texts = {i: f"line {i}" for i in range(12)}
    assert sp.rewrite_queries(FakeLLM(), texts) == {}
    assert len(sizes) >= 3  # asked in chunks, not one giant call
    assert [len(c) for c in sp._chunks(texts)] == [5, 5, 2]


# ---------------------------------------------------------------- 4. retrieval priors


def _assets() -> list[dict[str, Any]]:
    mk = lambda i, t, d: {"id": i, "asset_type": t, "tags": [], "description": d}
    return ([mk(f"image_moon{n}", "image", f"astronaut on the moon {n}") for n in range(6)]
            + [mk("schem_alarm", "schematic", "Flowchart: alarm 1202 overload restart"),
               mk("image_dsky", "image", "DSKY display keyboard")])


def test_prior_boosts_schematic_text_match_and_downweights_generic_terms() -> None:
    ctx = SimpleNamespace(assets=_assets())
    pri = sp.keyword_prior(ctx, "the alarm 1202 overload", 0.06)
    assert pri["schem_alarm"] > 0.1  # schematic 3x weight
    gen = sp.keyword_prior(ctx, "astronaut on the moon", 0.06)
    assert gen["schem_alarm"] == 0 and gen["image_moon0"] < 0.03  # common terms: low IDF
    assert sp.keyword_prior(ctx, "alarm", 0.0) == {}


def test_context_adds_topic_coherence() -> None:
    ctx = SimpleNamespace(assets=_assets())
    plain = sp.keyword_prior(ctx, "and then it happened", 0.06)
    ctxd = sp.keyword_prior(ctx, "and then it happened", 0.06, "keyboard display DSKY alarm")
    assert plain["image_dsky"] == 0 < ctxd["image_dsky"]


# ---------------------------------------------------------------- 5. photos and media_start


def test_small_punch_photo_is_not_upscaled() -> None:
    ctx = SimpleNamespace(assets_by_id={"i": {"asset_type": "image", "width": 640}})
    b = SceneBeat(id=0, start_time=0, end_time=3, visual_template="punch_in_emphasis",
                  source_id="i")
    sp._small_photo_fix(b, ctx)
    assert b.visual_template == VisualTemplate.FULLSCREEN_CLIP


def test_sidecar_media_start_hint(tmp_path: Path) -> None:
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"")
    (tmp_path / "c.mp4.txt").write_text("Apollo clip\nmedia_start: 2.5\n")
    assert sp.sidecar_media_start({"filepath": str(clip)}) == 2.5
    assert sp.sidecar_media_start({"filepath": str(tmp_path / "none.mp4")}) is None


def test_index_ignores_shots_before_hint() -> None:
    frames = [{"row": 0, "shot": [0, 2], "caption": ""}, {"row": 1, "shot": [2, 5], "caption": ""}]
    vecs = np.array([[1.0, 0.0], [0.6, 0.8]])
    idx = SemanticIndex({"assets": {"c": {"type": "clip", "frames": frames}}}, vecs)
    q = np.array([1.0, 0.0])
    assert idx.score_asset("c", q).start == 0
    idx.hints = {"c": 2.0}
    assert idx.score_asset("c", q).start >= 2.0


def test_scene_plan_roundtrip_still_valid() -> None:
    ScenePlan.model_validate({"episode_id": "x", "beats": [{"start_time": 0, "end_time": 2}]})


def test_lower_third_kept_only_when_about_the_shown_asset() -> None:
    ctx = SimpleNamespace(assets_by_id={"c": {"id": "c", "asset_type": "clip", "tags": [],
                                              "description": "clip | x | Mission control consoles"}},
                          vocab=frozenset(), callout=s3.narration_callout)
    on = SceneBeat(id=0, start_time=0, end_time=3, source_id="c", callout_text="Control Room")
    off = SceneBeat(id=1, start_time=3, end_time=6, source_id="c", callout_text="Car Key")
    sp.finalize_callouts([on, off], ctx, {0: "the control room was quiet", 1: "your car key has more"})
    assert on.callout_text == "Control Room"
    assert off.callout_text is None


def test_narration_callout_keeps_numbers_and_skips_light_verbs() -> None:
    out = s3.narration_callout("a 24 year old software lead")
    assert "24" in out.split()
    assert "Meant" not in s3.narration_callout("nobody in the room knew what it meant")
