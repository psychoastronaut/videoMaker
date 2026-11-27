"""Fix mode (src/agent_fix.py): fake LLM / searcher / renderer; locate.py highlight safety."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import make_subject_image
from PIL import Image

from src import agent_fix as af
from src import cli, locate, overlays
from src.common import drop_excluded, episode_dir, excluded_ids, read_json, write_json
from src.stage6_render import highlight_trusted

SPEECH = ["the", "mirror", "was", "wrong", "in", "1990", ".", "nasa", "launched", "the", "telescope", "into", "orbit", ".", "then", "the", "error", "appeared", "on", "the", "first", "images", ".", "engineers", "studied", "the", "mirror", "again", ".", "finally", "astronauts", "repaired", "it", "in", "1993", "."]


def make_ep(tmp: Path, intro_branding: bool = True) -> str:
    ep = episode_dir("ep-fix")
    (ep / "work").mkdir(parents=True)
    raw = ep / "raw"
    for sub in ("clips", "images", "schematics"):
        (raw / sub).mkdir(parents=True)
    assets = []
    for aid, kind in (("image_old", "images"), ("image_new", "images"), ("image_other", "images")):
        p = raw / kind / f"{aid[6:]}.png"
        make_subject_image(p)
        assets.append({"id": aid, "filepath": str(p), "asset_type": "image",
                       "filename": p.name, "description": aid, "tags": [], "width": 640,
                       "height": 360})
    clip = raw / "clips" / "bad.mp4"
    clip.write_bytes(b"x")
    assets.append({"id": "clip_bad", "filepath": str(clip), "asset_type": "clip",
                   "filename": "bad.mp4", "duration": 10.0, "description": "bad", "tags": [],
                   "width": 1920, "height": 1080})
    write_json(ep / "work" / "local_assets.json", {"assets": assets, "skipped": []})
    words, t = [], 0.0
    for w in SPEECH:
        words.append({"word": w, "start": round(t, 2), "end": round(t + 0.4, 2)})
        t += 0.5
    write_json(ep / "work" / "transcript.json",
               {"duration": t, "segments": [{"id": 0, "start": 0, "end": t, "text": "",
                                             "words": words}]})
    n = len(SPEECH)
    cuts = [0.0, 5.5, 11.0, 16.5, 22.0, n * 0.5]
    beats = []
    srcs = ["clip_bad", "image_old", "clip_bad", "image_other", "image_old"]
    for i in range(5):
        beats.append({"id": i, "start_time": cuts[i], "end_time": cuts[i + 1],
                      "visual_template": "fullscreen_clip", "primary_source": "local",
                      "source_id": srcs[i], "callout_text": "Mirror" if i == 1 else None,
                      "media_start": 0.0 if srcs[i] == "clip_bad" else None,
                      "highlights": [{"shape": "box", "region": "top"}] if i == 3 else None})
    write_json(ep / "work" / "scene_plan.json", {"episode_id": "ep-fix", "beats": beats})
    return "ep-fix"


class FakeChat:
    def __init__(self, *actions_per_call: list[dict[str, Any]]) -> None:
        self.calls = list(actions_per_call)
        self.prompts: list[str] = []

    def __call__(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        self.prompts.append(messages[-1]["content"])
        return json.dumps({"actions": self.calls.pop(0), "explanation": "fake"})


class FakeSearcher:
    def search(self, query: str, types: tuple[str, ...], exclude: set[str],
               prior: dict[str, float], k: int = 5) -> list[dict[str, Any]]:
        order = ["image_new", "image_other", "image_old", "clip_bad"]
        return [{"asset": a, "score": 0.5 - i * 0.1, "start": None, "end": None, "caption": ""}
                for i, a in enumerate(order) if a not in exclude and a.split("_")[0] in
                {t for t in types}][:k]


def test_time_and_quote_location(tmp_path: Path) -> None:
    ep = make_ep(tmp_path)
    ctx = af.load_ctx(ep)
    assert ctx.intro == 3.0
    notes: list[str] = []
    # final video 0:08 = narration 5.0 s (3 s intro card) -> beat 0 (0-5.5)
    assert af.locate_beats(ctx, {"by": "video_time", "value": "0:08"}, notes)[0]["id"] == 0
    assert af.locate_beats(ctx, {"by": "narration_time", "value": "0:08"}, notes)[0]["id"] == 1
    # "1990" is said once, in beat 0; "1993" in the last beat; accents/punctuation ignored
    assert af.locate_beats(ctx, {"by": "quote", "value": "1990"}, notes)[0]["id"] == 0
    assert af.locate_beats(ctx, {"by": "quote", "value": "In 1993."}, notes)[0]["id"] == 3
    assert af.locate_beats(ctx, {"by": "quote", "value": "the mirror"}, notes)[0]["id"] == 0
    assert "said 2 times" in " ".join(notes)
    with pytest.raises(af.FixError):
        af.locate_beats(ctx, {"by": "video_time", "value": "0:01"}, notes)   # inside the intro
    with pytest.raises(af.FixError):
        af.locate_beats(ctx, {"by": "quote", "value": "zebra"}, notes)
    assert af.locate_beats(ctx, {"by": "asset", "value": "clip_bad"}, notes)[1]["id"] == 2


def test_replace_by_query_writes_plan_backup_log_and_protects_it(tmp_path: Path) -> None:
    ep = make_ep(tmp_path)
    work = episode_dir(ep) / "work"
    chat = FakeChat([{"type": "replace_asset", "beat": {"by": "quote", "value": "1990"},
                      "query": "telescope released from the shuttle"}])
    rendered: list[str] = []
    p = af.run_fix(ep, "quando eu falo 1990 troque o clipe", yes=True, chat=chat,
                   searcher=FakeSearcher(), renderer=lambda e: rendered.append(e) or work / "x.mp4")
    assert p is not None
    new = read_json(work / "scene_plan.json")
    # beat 0 showed clip_bad; the replacement is an image, trim reset, crop/highlight cleared
    assert new["beats"][0]["source_id"] == "image_new"
    assert new["beats"][0]["media_start"] is None
    assert read_json(work / "scene_plan.prev.json")["beats"][0]["source_id"] == "clip_bad"
    log = [json.loads(x) for x in (work / "fix_log.jsonl").read_text().splitlines()]
    assert log[-1]["instruction"].startswith("quando") and log[-1]["targets"][0]["asset"] == "clip_bad"
    assert rendered == [ep]
    # a fix marks the plan as hand-edited, so `run`/`plan` will not overwrite it
    assert cli.plan_is_hand_edited(work)
    # the first prompt carries the final-video time offset and the beat table
    assert "intro card" in chat.prompts[0] and "clip_bad" in chat.prompts[0]


def test_dry_run_changes_nothing_and_confirm_can_cancel(tmp_path: Path) -> None:
    ep = make_ep(tmp_path)
    work = episode_dir(ep) / "work"
    before = (work / "scene_plan.json").read_text()
    act = {"type": "replace_asset", "beat": {"by": "beat_id", "value": "1"}, "query": "x"}
    af.run_fix(ep, "x", dry_run=True, chat=FakeChat([act]), searcher=FakeSearcher())
    af.run_fix(ep, "x", confirm=lambda q: False, chat=FakeChat([act]), searcher=FakeSearcher())
    assert (work / "scene_plan.json").read_text() == before
    assert not (work / "fix_log.jsonl").exists()


def test_exclude_is_global_respected_and_undoable(tmp_path: Path) -> None:
    ep = make_ep(tmp_path)
    work = episode_dir(ep) / "work"
    # fix 1 replaces beat 0; fix 2 says "this clip" (@last) -> banned everywhere
    chat = FakeChat([{"type": "replace_asset", "beat": {"by": "beat_id", "value": "0"},
                      "query": "q"}],
                    [{"type": "exclude_asset", "asset": "@last"}])
    af.run_fix(ep, "troque", yes=True, chat=chat, searcher=FakeSearcher(), render=False)
    af.run_fix(ep, "remova esse clipe do projeto inteiro", yes=True, chat=chat,
               searcher=FakeSearcher(), render=False)
    plan = read_json(work / "scene_plan.json")
    assert all(b["source_id"] != "clip_bad" for b in plan["beats"])      # beat 2 replaced too
    assert "clip_bad" in excluded_ids(work)
    assert (episode_dir(ep) / "raw/clips/bad.mp4.txt").read_text().strip() == "exclude: true"
    assets = read_json(work / "local_assets.json")["assets"]
    assert "clip_bad" not in {a["id"] for a in drop_excluded(assets, work)}
    # undo restores the plan and removes the exclusion + sidecar line
    ctx = af.load_ctx(ep)
    assert af.undo_last(ctx)
    assert "clip_bad" not in excluded_ids(work)
    assert not (episode_dir(ep) / "raw/clips/bad.mp4.txt").exists()
    assert read_json(work / "scene_plan.json")["beats"][2]["source_id"] == "clip_bad"


def test_stage2_skips_excluded_assets(tmp_path: Path) -> None:
    from src import stage2_index_assets

    ep = make_ep(tmp_path)
    work = episode_dir(ep) / "work"
    (episode_dir(ep) / "raw/clips/bad.mp4").unlink()
    write_json(work / af.EXCLUDED_FILE, [{"id": "image_old", "file": "old.png"}])
    (episode_dir(ep) / "raw/images/other.png.txt").write_text("exclude: true\n")
    data = stage2_index_assets.run(ep, work)
    ids = {a["id"] for a in data["assets"]}
    assert ids == {"image_new"}


def test_crop_trim_callout_transition_template_validation(tmp_path: Path) -> None:
    ep = make_ep(tmp_path)
    ctx = af.load_ctx(ep)
    b2 = {"by": "beat_id", "value": "2"}
    p = af.apply_actions(ctx, [
        {"type": "set_zoom", "beat": {"by": "beat_id", "value": "1"}, "focus": "top-left", "zoom": 2},
        {"type": "set_trim", "beat": b2, "start": 2.0, "end": 9.0},
        {"type": "set_callout", "beat": b2, "text": "Wrong mirror shape here now extra"},
        {"type": "remove_callout", "beat": {"by": "beat_id", "value": "1"}},
        {"type": "set_transition", "beat": b2, "transition": "fade", "duration": 0.5},
        {"type": "change_template", "beat": b2, "template": "punch_in_emphasis"}], None)
    beats = p.plan["beats"]
    assert beats[1]["crop_region"] == [0.0, 0.0, 0.5, 0.5] and beats[1]["callout_text"] is None
    assert (beats[2]["media_start"], beats[2]["media_end"]) == (2.0, 9.0)
    assert beats[2]["callout_text"] == "Wrong mirror shape here now"
    assert beats[2]["transition"] == "crossfade" and beats[2]["visual_template"] == "punch_in_emphasis"
    for bad in ({"type": "set_trim", "beat": b2, "start": 50},
                {"type": "set_crop", "beat": b2, "region": [0.9, 0.9, 0.5, 0.5]},
                {"type": "set_transition", "beat": b2, "transition": "spin"},
                {"type": "change_template", "beat": b2, "template": "nope"},
                {"type": "move_overlay", "beat": b2, "corner": "tl"},
                {"type": "teleport", "beat": b2}):
        with pytest.raises(af.FixError):
            af.apply_actions(ctx, [bad], None)


def test_split_and_merge(tmp_path: Path) -> None:
    ep = make_ep(tmp_path)
    ctx = af.load_ctx(ep)
    p = af.apply_actions(ctx, [{"type": "split_beat", "beat": {"by": "beat_id", "value": "0"},
                                "at": {"by": "quote", "value": "launched"}}], None)
    from src.schemas import ScenePlan

    plan = ScenePlan.model_validate(p.plan)
    assert len(plan.beats) == 6 and plan.beats[0].end_time == plan.beats[1].start_time == 4.0
    assert plan.beats[1].media_start == 4.0     # clip continues where the first piece ended
    with pytest.raises(af.FixError):
        af.apply_actions(ctx, [{"type": "split_beat", "beat": {"by": "beat_id", "value": "0"},
                                "at": {"by": "narration_time", "value": "0.2"}}], None)
    q = af.apply_actions(ctx, [{"type": "merge_beats", "beat": {"by": "beat_id", "value": "0"}}],
                         None)
    assert len(q.plan["beats"]) == 4 and q.plan["beats"][0]["end_time"] == 11.0


def test_interactive_undo_and_render(tmp_path: Path, capsys: Any) -> None:
    ep = make_ep(tmp_path)
    work = episode_dir(ep) / "work"
    act = {"type": "replace_asset", "beat": {"by": "beat_id", "value": "1"}, "query": "q"}
    inputs = iter(["troque o beat 1", "y", "undo", "troque o beat 1", "y", "render", "quit"])
    rendered: list[str] = []
    af.interactive(ep, chat=FakeChat([act], [act]), searcher=FakeSearcher(),
                   renderer=lambda e: rendered.append(e) or work / "v.mp4",
                   read=lambda prompt: next(inputs))
    assert read_json(work / "scene_plan.json")["beats"][1]["source_id"] == "image_new"
    assert rendered == [ep]


def test_ollama_down_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"llm": {"base_url": "http://127.0.0.1:9/v1"}}
    with pytest.raises(af.FixError, match="ollama serve"):
        af.ollama_check(cfg, "qwen2.5:14b-instruct")


def test_review_flags_burned_year_and_bad_highlight(tmp_path: Path) -> None:
    ep = make_ep(tmp_path)
    work = episode_dir(ep) / "work"
    (episode_dir(ep) / "raw/clips/bad.mp4").write_bytes(b"x")

    def vlm(ctx: af.Ctx, frame: Path, narr: str) -> dict[str, Any] | None:
        if "1990" in narr:     # beat 0 shows a clip with a burned-in "2009" caption
            return {"matches": 6, "burned_text": "On May 11, 2009 Atlantis", "text_kind": "caption"}
        return {"matches": 9, "burned_text": "", "text_kind": "none"}

    # the clip frame cannot be grabbed from a fake mp4: use a still instead
    import src.agent_fix as m
    orig = m.asset_frame
    m.asset_frame = lambda ctx, b, tmp: Path(ctx.assets["image_old"]["filepath"])
    try:
        sus = af.review(ep, vlm=vlm, searcher=FakeSearcher(), render=False)
    finally:
        m.asset_frame = orig
    by = {s.beat: s for s in sus}
    assert "2009" in by[0].reasons[0] and by[0].action["type"] == "replace_asset"
    assert by[0].action["asset"] == "image_new"
    assert by[3].action["type"] == "remove_highlight"      # 'region: top' without a location
    assert (work / "fix_review.json").is_file()


# --------------------------------------------------------------------------- highlights


def test_unlocated_region_highlight_is_not_drawn_but_sidecar_and_numeric_are() -> None:
    assert highlight_trusted({"shape": "box", "region": "top"})[0] is False
    assert highlight_trusted({"shape": "box", "x": 0.1, "y": 0.4, "w": 0.2, "h": 0.1})[0]
    assert highlight_trusted(overlays.parse_highlight("top | Title"))[0]      # explicit sidecar
    assert highlight_trusted({"x": 0.1, "y": 0.4, "w": 0.0, "h": 0.1})[0] is False  # empty box
    assert af.highlight_problem({"highlights": [{"shape": "box", "region": "top"}]})


def test_no_reliable_location_means_no_highlight(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    img = tmp_path / "d.png"
    Image.new("RGB", (640, 360), (10, 10, 26)).save(img)
    monkeypatch.setattr(locate, "ocr_words", lambda p: None)
    monkeypatch.setattr(locate, "vlm_box", lambda *a, **k: None)
    assert locate.locate(img, "the null corrector said perfect") is None
    # OCR found words, but none the narration talks about -> still nothing
    monkeypatch.setattr(locate, "ocr_words", lambda p: [("Title", [0.4, 0.03, 0.2, 0.05])])
    assert locate.locate(img, "the null corrector said perfect") is None
    # a located box over the title band is rejected
    assert locate.sane_box([0.3, 0.02, 0.4, 0.05]) is None
    assert locate.sane_box([0.1, 0.1, 0.9, 0.9]) is None       # too big


def test_known_bbox_gives_a_rectangle_exactly_over_it(monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:
    img = tmp_path / "d.png"
    Image.new("RGB", (640, 360), (10, 10, 26)).save(img)
    words = [("Null", [0.10, 0.30, 0.06, 0.05]), ("corrector", [0.17, 0.30, 0.12, 0.05]),
             ("Title", [0.40, 0.03, 0.20, 0.05])]
    monkeypatch.setattr(locate, "ocr_words", lambda p: words)
    h = locate.locate(img, "the null corrector said it was perfect")
    assert h and h["source"] == "ocr" and highlight_trusted(h)[0]
    pad = locate.PAD
    want = (0.10 - pad, 0.30 - pad, 0.19 + 2 * pad, 0.05 + 2 * pad)
    assert (h["x"], h["y"], h["w"], h["h"]) == pytest.approx(want, abs=1e-4)
    # pixel box on a 1920x1080 picture
    x, y, w, hh = overlays.resolve_box(h, (0, 0, 1920, 1080), 1920)
    assert (x, y) == (round(want[0] * 1920), round(want[1] * 1080))
    assert (w, hh) == (round(want[2] * 1920), round(want[3] * 1080))
    # the drawn plate's outline sits exactly on that box once placed at (x + dx, y + dy)
    png, dx, dy = overlays.make_highlight(tmp_path / "hl.png", w, hh, "", (80, 200, 120), 4, 20)
    plate = Image.open(png).convert("RGBA")
    edge = plate.getpixel((-dx + 1, -dy + hh // 2))        # left edge of the box, mid height
    assert edge[3] == 255 and edge[:3] == (80, 200, 120)
    assert plate.getpixel((-dx + w // 2, -dy + hh // 2))[3] == 0     # inside: empty (no fill)
