"""Round-8: entity-backed establishing photos (places, machines) as contextual B-roll."""

from __future__ import annotations

from pathlib import Path

from conftest import make_subject_image
from test_planner import ASSETS, make_ctx
from test_review_fixes import write_assets
from test_round5 import local, stock_beat, stock_episode

from src.common import read_json
from src.demo_generator import _build_segments
from src.schemas import PrimarySource, SceneBeat, ScenePlan
from src.stage3_plan_scenes import (
    PlanContext,
    _generate_deterministic_plan,
    _validate_and_repair,
    find_entity,
    flatten_words,
)
from src.stage4_fetch_stock import StockFetcher, relevant
from src.stage4_fetch_stock import run as stage4

KW = sorted(["city", "skyline", "factory", "lathe", "rocket", "power plant", "engine"],
            key=lambda k: (-len(k.split()), -len(k)))


def ent(text: str):
    return find_entity(text.split(), KW)


def test_entity_detection_names_places_and_machines() -> None:
    assert ent("The Tokyo skyline glows at night over a sleepy suburb.") == ("Tokyo skyline", True)
    assert ent("A CNC lathe in a factory carves a brass part.") == ("CNC lathe", False)
    assert ent("We visited Berlin and toured a power plant today") == ("Berlin power plant", True)
    assert ent("Launch day at a rocket test stand") == ("rocket test", False)
    q = ent("Pictures of the Eiffel Tower near a city")
    assert q and q[1] and q[0].startswith("Eiffel Tower")


def test_entity_detection_ignores_acronyms_brands_and_sentence_starts() -> None:
    for text in ["The PID loop for LiPo batteries", "Welcome to Psycho Astronaut today",
                 "Another shelf stores cables", "Every servo gets a signal",
                 "Thanks for watching"]:
        assert ent(text) is None, text


def test_planner_makes_establishing_photo_beats_from_the_narration() -> None:
    segs = _build_segments()
    transcript = {"duration": segs[-1]["end"] + 0.6, "segments": segs}
    base = make_ctx()
    assets = [{"id": f"{t}_{n}", "asset_type": t, "filepath": f"/x/{n}", "tags": [t, n],
               "description": n} for t, n in [("clip", "servo"), ("clip", "arm"),
                                              ("image", "board"), ("schematic", "loop")]]
    ctx = PlanContext(base.config, assets, flatten_words(transcript), transcript["duration"])
    plan = _validate_and_repair(_generate_deterministic_plan(ctx, "demo"), ctx)
    photos = [b for b in plan.beats if b.stock_media == "image"]
    queries = {b.search_queries[0] for b in photos}
    assert {"Tokyo skyline", "CNC lathe"} <= queries, queries
    for b in photos:
        assert b.primary_source == PrimarySource.STOCK and b.visual_template.value in (
            "fullscreen_clip", "cutout_callout", "split_schematic")
        assert 2 <= len(b.search_queries[0].split()) <= 4
    full = [b for b in photos if b.visual_template.value == "fullscreen_clip"]
    assert full and all(b.callout_text for b in full)  # lower-third names what is shown


def test_entity_stock_is_disabled_with_stock_ratio_zero_and_never_for_punch_payoffs() -> None:
    base = make_ctx()
    t = {"duration": 8.0, "segments": [{"start": 0, "end": 8, "words": [],
         "text": "The factory exploded in smoke near Tokyo airport"}]}
    ctx = PlanContext(base.config, ASSETS, flatten_words(t), 8.0)
    ctx.stock_ratio = 0.0
    assert not any(b.primary_source == PrimarySource.STOCK
                   for b in _generate_deterministic_plan(ctx, "t").beats)
    # repair: a fail beat that is a punch-in cannot be stock even if marked entity image
    raw = ScenePlan(beats=[SceneBeat(
        start_time=0, end_time=8, beat_style="fail", visual_template="punch_in_emphasis",
        primary_source="stock", stock_media="image", search_queries=["Tokyo airport"])])
    out = _validate_and_repair(raw, ctx).beats[0]
    assert out.primary_source == PrimarySource.LOCAL and out.stock_media == "any"


def test_repair_keeps_technical_split_with_a_stock_photo_on_the_left() -> None:
    ctx = make_ctx()
    raw = ScenePlan(beats=[SceneBeat(
        start_time=0, end_time=ctx.duration, beat_style="technical",
        visual_template="split_schematic", primary_source="stock", stock_media="image",
        overlay_asset="schematic_loop", search_queries=["CNC lathe"], callout_text="Control loop")])
    beat = _validate_and_repair(raw, ctx).beats[0]
    assert beat.primary_source == PrimarySource.STOCK and beat.overlay_asset == "schematic_loop"
    assert beat.visual_template.value == "split_schematic"


def test_image_intent_beats_only_query_photo_providers(tmp_path: Path) -> None:
    f = StockFetcher({"pexels_api_key": "k"}, tmp_path, offline=False)
    names = [p.__name__ for p in f._providers(False, images_only=True)]
    assert names and not any(n.endswith("video") for n in names)
    assert {"_openverse", "_wikimedia", "_pexels"} <= set(names)


def test_stage4_requests_images_only_for_entity_beats(monkeypatch, tmp_path: Path) -> None:
    import os
    os.environ.pop("VIDEOMAKER_OFFLINE")
    asset = local(tmp_path, "servo", "image", ["servo"])
    beat = stock_beat(0, 0, 4, "fullscreen_clip", beat_style="normal", stock_media="image",
                      search_queries=["Tokyo skyline"])
    work = stock_episode("en1", [asset], [beat])
    seen: dict = {}

    def fake_find(self, query, beat_id, **kw):
        seen.update(kw, query=query)

    monkeypatch.setattr(StockFetcher, "find", fake_find)
    stage4("en1", work)
    assert seen["images_only"] is True and seen["prefer_video"] is False
    # no result -> best local fallback keeps the beat
    out = read_json(work / "scene_plan.json")["beats"][0]
    assert out["primary_source"] == "local" and out["source_id"] == "image_servo"
    assert "stock unavailable" in out["beat_reason"]
    write_assets(work, [])


def test_proper_name_in_title_is_a_strong_match_for_entity_queries() -> None:
    assert relevant("Tokyo skyline", "Tokyo at dusk.jpg", strong_names=True)
    assert not relevant("Tokyo skyline", "Tokyo at dusk.jpg")  # ordinary gate: both words
    assert not relevant("Tokyo skyline", "Paris skyline", strong_names=True)
    assert not relevant("Tokyo skyline", "Tokyo logo", strong_names=True)  # logos still out


def test_stock_photos_must_be_landscape(tmp_path: Path) -> None:
    import httpx
    from PIL import Image
    f = StockFetcher({"min_resolution": 100}, tmp_path, offline=False)

    def serve(w: int, h: int):
        p = tmp_path / f"{w}x{h}.png"
        Image.new("RGB", (w, h), (9, 9, 9)).save(p)
        data = p.read_bytes()
        return httpx.MockTransport(lambda req: httpx.Response(200, content=data))

    with httpx.Client(transport=serve(400, 200)) as c:
        assert f._download(c, "http://x/land.png", "openverse")
    with httpx.Client(transport=serve(200, 400)) as c:
        assert f._download(c, "http://x/port.png", "openverse") is None
    make_subject_image(tmp_path / "unused.png")
