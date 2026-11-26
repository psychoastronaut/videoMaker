"""Regression tests for reviewer findings (P1-1..3, P2-1..10)."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import make_subject_image, write_wav
from PIL import Image
from test_planner import ASSETS, make_ctx

from src import cli, stage1_transcribe, stage5_cutouts
from src.common import authoring_signature, episode_dir, write_json
from src.media import AssetResolver, ResolvedMedia
from src.schemas import BeatStyle, SceneBeat, ScenePlan, StockAsset, VisualTemplate
from src.stage3_plan_scenes import (
    PlanContext,
    _validate_and_repair,
    extract_callout,
    flatten_words,
)
from src.stage4_fetch_stock import StockFetcher
from src.stage6_render import (
    CompileContext,
    RenderSettings,
    _render_beat,
    compile_beat,
    prepare_beats,
)


def episode_with_assets(name: str = "rv") -> tuple[Path, Path]:
    ep = episode_dir(name)
    raw, work = ep / "raw", ep / "work"
    for sub in ("clips", "images", "schematics"):
        (raw / sub).mkdir(parents=True)
    work.mkdir()
    return ep, work


def write_assets(work: Path, assets: list[dict[str, Any]]) -> None:
    write_json(work / "local_assets.json", {"assets": assets})


def asset(path: Path, kind: str) -> dict[str, Any]:
    return {"id": f"{kind}_{path.stem}", "filepath": str(path), "asset_type": kind,
            "filename": path.name, "tags": [kind, path.stem], "description": path.stem}


# ------------------------------------------------------------------ P1-1


def test_llm_cutout_without_overlay_gets_an_image_subject_not_background() -> None:
    ctx = make_ctx()
    raw = ScenePlan(beats=[SceneBeat(start_time=0, end_time=ctx.duration,
                                     visual_template="cutout_callout", source_id="clip_servo")])
    beat = _validate_and_repair(raw, ctx).beats[0]
    assert beat.visual_template == VisualTemplate.CUTOUT_CALLOUT
    assert beat.overlay_asset == "image_board"


def test_schematic_is_never_a_cutout_subject() -> None:
    ctx = make_ctx()
    raw = ScenePlan(beats=[SceneBeat(start_time=0, end_time=ctx.duration,
                                     visual_template="cutout_callout", source_id="clip_servo",
                                     overlay_asset="schematic_loop")])
    assert _validate_and_repair(raw, ctx).beats[0].overlay_asset == "image_board"


def test_cutout_degrades_when_there_is_nothing_to_cut_out() -> None:
    transcript = {"duration": 8.0, "segments": [{"start": 0, "end": 8, "text": "hello world"}]}
    assets = [a for a in ASSETS if a["asset_type"] != "image"]
    ctx = PlanContext(make_ctx().config, assets, flatten_words(transcript), 8.0)
    raw = ScenePlan(beats=[SceneBeat(start_time=0, end_time=8.0, beat_style="fail",
                                     visual_template="cutout_callout", source_id="clip_servo",
                                     callout_text="Magic smoke")])
    beat = _validate_and_repair(raw, ctx).beats[0]
    assert beat.visual_template == VisualTemplate.PUNCH_IN_EMPHASIS
    assert beat.callout_text == "Magic smoke"  # becomes a lower-third


def test_stage5_never_uses_background_footage_as_subject(tmp_path: Path) -> None:
    work = tmp_path / "w"
    work.mkdir()
    clip = tmp_path / "bg.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=160x90:d=1",
                    str(clip)], check=True)
    img = tmp_path / "p.png"
    make_subject_image(img)
    write_assets(work, [asset(clip, "clip"), asset(img, "image")])
    r = AssetResolver(work)
    sub, why = stage5_cutouts.pick_subject({"source_id": "clip_bg"}, r)
    assert sub is None and "footage" in why
    sub, _ = stage5_cutouts.pick_subject({"source_id": "image_p"}, r)  # still image: ok
    assert sub and sub.asset_id == "image_p"


# ------------------------------------------------------------------ P1-2


def test_hand_edited_plan_is_protected_and_backed_up(monkeypatch: pytest.MonkeyPatch) -> None:
    ep, work = episode_with_assets()
    plan = {"episode_id": "rv", "beats": [{"start_time": 0, "end_time": 3,
                                           "visual_template": "fullscreen_clip",
                                           "source_id": "x", "beat_style": "normal"}]}
    write_json(work / "scene_plan.json", plan)
    cli._record_state("plan", work)
    assert not cli.plan_is_hand_edited(work)
    plan["beats"][0]["visual_template"] = "punch_in_emphasis"
    write_json(work / "scene_plan.json", plan)
    assert cli.plan_is_hand_edited(work)

    # new raw asset -> stage 3 is "stale" but must not run without --force
    write_json(work / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "a"}]})
    write_assets(work, [])
    ran: list[str] = []
    monkeypatch.setattr(cli, "_execute", lambda *a, **k: ran.append("x"))
    assert cli.stale_reason("plan", ep, work)
    cli._run_stage(cli.STAGE_BY_KEY["plan"], "rv", force=False)
    assert ran == []
    cli._run_stage(cli.STAGE_BY_KEY["plan"], "rv", force=True)
    assert ran == ["x"]


def test_stage3_backs_up_previous_plan() -> None:
    from src.stage3_plan_scenes import run as stage3

    ep, work = episode_with_assets()
    write_wav(ep / "raw" / "voiceover.wav", 6.0)
    write_json(work / "transcript.json", {"duration": 6.0, "segments": [
        {"start": 0, "end": 6, "text": "hello there world", "words": []}]})
    write_assets(work, [])
    write_json(work / "scene_plan.json", {"episode_id": "rv", "beats": [{"start_time": 0,
                                                                         "end_time": 1}]})
    stage3("rv", work)
    prev = work / "scene_plan.prev.json"
    assert prev.is_file() and '"end_time": 1' in prev.read_text()


def test_unrecorded_existing_plan_counts_as_hand_edited() -> None:
    _, work = episode_with_assets()
    write_json(work / "scene_plan.json", {"beats": [{"start_time": 0, "end_time": 2}]})
    assert cli.plan_is_hand_edited(work)  # e.g. a plan from before signatures existed
    assert authoring_signature({"beats": []})


# ------------------------------------------------------------------ P1-3 / P2-1 / P2-2


def run_stage5_with(tmp_assets: list[dict[str, Any]], beats: list[dict[str, Any]]):
    _ep, work = episode_with_assets("s5")
    write_assets(work, tmp_assets)
    write_json(work / "scene_plan.json", {"episode_id": "s5", "beats": beats})
    return stage5_cutouts.run("s5", work), work


def test_corrupt_image_does_not_crash_stage5(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: Path) -> None:
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"this is not a png")
    bad_clip = tmp_path / "broken.mp4"
    bad_clip.write_bytes(b"nope")
    good = tmp_path / "ok.png"
    make_subject_image(good)
    manifest, work = run_stage5_with(
        [asset(bad, "image"), asset(bad_clip, "clip"), asset(good, "image")],
        [{"start_time": 0, "end_time": 2, "visual_template": "cutout_callout",
          "source_id": "image_broken", "callout_text": "Broken image"},
         {"start_time": 2, "end_time": 4, "visual_template": "cutout_callout",
          "source_id": "image_ok", "overlay_asset": "clip_broken", "callout_text": "Bad clip"},
         {"start_time": 4, "end_time": 6, "visual_template": "cutout_callout",
          "source_id": "image_ok", "callout_text": "Fine subject"}])
    from src.common import read_json

    beats = read_json(work / "scene_plan.json")["beats"]
    assert beats[0]["cutout_path"] is None and beats[1]["cutout_path"] is None
    assert beats[2]["cutout_path"] and Path(beats[2]["cutout_path"]).is_file()
    assert all(b["badge_path"] for b in beats)  # labels still produced
    assert len(manifest["warnings"]) >= 2  # + card notice for the flat-background photo
    assert (work / "cutouts" / ".ready").is_file()


def test_low_coverage_cutout_falls_back_to_framed_original(monkeypatch: pytest.MonkeyPatch,
                                                           tmp_path: Path) -> None:
    img = tmp_path / "subject.png"
    make_subject_image(img)
    nearly_empty = Image.new("RGBA", (640, 360), (0, 0, 0, 0))
    nearly_empty.putpixel((5, 5), (255, 0, 0, 255))
    monkeypatch.setattr(stage5_cutouts, "_cached_matte",
                        lambda i, s, k, o, c: (nearly_empty, "fake"))
    warnings: list[str] = []
    out = stage5_cutouts._make_cutout(ResolvedMedia(str(img), False, "image_subject", "image"),
                                      0, tmp_path, {}, 1080, warnings.append)
    assert warnings and "framed card" in warnings[0]
    assert Image.open(out).width > 300  # the whole picture, not a speck


def test_raw_cutout_cache_does_not_collide_on_same_stem(tmp_path: Path) -> None:
    a, b = tmp_path / "a" / "motor.png", tmp_path / "b" / "motor.jpg"
    a.parent.mkdir()
    b.parent.mkdir()
    assert stage5_cutouts._path_key(str(a)) != stage5_cutouts._path_key(str(b))
    assert stage5_cutouts._path_key(str(a)) == stage5_cutouts._path_key(str(a))


# ------------------------------------------------------------------ P2-3 / P2-4


def test_synthetic_transcript_is_kept_but_force_retranscribes(monkeypatch: pytest.MonkeyPatch,
                                                              tmp_path: Path) -> None:
    ep, work = episode_with_assets("t1")
    write_wav(ep / "raw" / "voiceover.wav", 2.0)
    write_json(work / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "x"}],
                                          "synthetic": True})
    import os
    os.utime(work / "transcript.json", (1, 1))  # much older than the voiceover
    assert stage1_transcribe.run("t1", work)["synthetic"] is True

    class Boom(Exception):
        pass

    def fake_model(*_a: Any, **_k: Any) -> None:
        raise Boom

    monkeypatch.setattr(stage1_transcribe, "WhisperModel", fake_model)
    with pytest.raises(Boom):  # force reached the transcription step
        stage1_transcribe.run("t1", work, force=True)


def test_stale_real_transcript_is_redone_without_force(monkeypatch: pytest.MonkeyPatch) -> None:
    ep, work = episode_with_assets("t2")
    write_wav(ep / "raw" / "voiceover.wav", 2.0)
    write_json(work / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "x"}]})
    import os
    os.utime(work / "transcript.json", (1, 1))
    called: list[int] = []

    def fake_model(*_a: Any, **_k: Any) -> None:
        called.append(1)
        raise RuntimeError("stop")

    monkeypatch.setattr(stage1_transcribe, "WhisperModel", fake_model)
    with pytest.raises(RuntimeError):
        stage1_transcribe.run("t2", work)
    assert called


def test_float32_wav_duration_is_read(tmp_path: Path) -> None:
    ep, work = episode_with_assets("t3")
    wav = ep / "raw" / "voiceover.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=d=2",
                    "-c:a", "pcm_f32le", str(wav)], check=True)
    write_json(work / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "x"}],
                                          "synthetic": True})
    assert stage1_transcribe.run("t3", work)["segments"]  # wave.open would have raised


# ------------------------------------------------------------------ P2-5


def stock(platform: str, lic: str) -> StockAsset:
    return StockAsset(id="a", source_platform=platform, source_url="", local_path="x",
                      license_type=lic)


def test_reused_stock_is_revalidated_against_policy(tmp_path: Path) -> None:
    f = StockFetcher({}, tmp_path, offline=False)
    assert f.license_ok(stock("openverse", "CC0"))
    assert f.license_ok(stock("wikimedia", "Public domain"))
    assert not f.license_ok(stock("wikimedia", "CC BY-SA 4.0"))
    assert not f.license_ok(stock("openverse", "by"))
    assert f.license_ok(stock("pexels", "Pexels License"))  # platform licenses on by default
    assert not f.license_ok(stock("pexels", "CC0"))
    strict = StockFetcher({"strict_cc0_only": True}, tmp_path, offline=False)
    assert not strict.license_ok(stock("pexels", "Pexels License"))
    assert strict.license_ok(stock("wikimedia", "CC0"))
    assert f.license_ok(stock("generated", "generated-fallback"))


# ------------------------------------------------------------------ P2-6


@pytest.mark.parametrize("text,kw", [
    (("Every servo gets a signal from the control loop, which checks its position "
      "a thousand times per second."), "signal"),
    ("In theory, that is all you need.", "in theory"),
    ("Of course, I wired the motor driver backwards.", "of course"),
    ("and the whole bench went dark.", "magic smoke"),
    ("Okay, it spilled a little, but nobody ships on the first try.", ""),
    ("Thanks for watching.", ""),
])
def test_callouts_are_two_to_five_words_without_filler(text: str, kw: str) -> None:
    ctx = make_ctx()
    out = ctx.callout(text, kw, ASSETS[0])
    words = out.lower().split()
    assert len(words) == 0 or 2 <= len(words) <= 5, out
    assert not {"thousand", "times", "course", "theory"} & set(words), out
    assert out.lower() not in {"bench dark", "of course", "in theory"}


def test_callout_uses_asset_label_when_text_is_all_filler() -> None:
    assert extract_callout("In theory, that is all you need.",
                           label="Servo Motor") == "Servo Motor"


def test_repair_regenerates_one_word_callouts() -> None:
    ctx = make_ctx()
    raw = ScenePlan(beats=[SceneBeat(start_time=0, end_time=ctx.duration,
                                     visual_template="cutout_callout", source_id="clip_servo",
                                     overlay_asset="image_board", callout_text="Oops")])
    words = _validate_and_repair(raw, ctx).beats[0].callout_text.split()
    assert 2 <= len(words) <= 5


# ------------------------------------------------------------------ P2-7


def test_fail_and_hype_beats_are_capped_per_style() -> None:
    ctx = make_ctx()
    cap = ctx.durations["fail"][1] + 0.5
    raw = ScenePlan(beats=[SceneBeat(start_time=0, end_time=ctx.duration, beat_style="fail",
                                     source_id="clip_servo")])
    beats = _validate_and_repair(raw, ctx).beats
    assert len(beats) > 1
    assert all(b.duration <= cap + 1e-6 for b in beats if b.beat_style == BeatStyle.FAIL)


def test_a_punch_in_exists_when_there_are_fail_moments() -> None:
    ctx = make_ctx()
    raw = ScenePlan(beats=[
        SceneBeat(start_time=0, end_time=3, source_id="clip_arm"),
        SceneBeat(start_time=3, end_time=ctx.duration, beat_style="fail", source_id="clip_servo")])
    beats = _validate_and_repair(raw, ctx).beats
    assert any(b.visual_template == VisualTemplate.PUNCH_IN_EMPHASIS for b in beats)


# ------------------------------------------------------------------ P2-8 .. P2-10


def make_resolver(tmp_path: Path, items: list[dict[str, Any]]) -> AssetResolver:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    write_assets(work, items)
    return AssetResolver(work)


def test_render_time_validation_swaps_schematic_on_the_left(tmp_path: Path) -> None:
    sch, img = tmp_path / "s.png", tmp_path / "i.png"
    make_subject_image(sch)
    make_subject_image(img)
    r = make_resolver(tmp_path, [asset(sch, "schematic"), asset(img, "image")])
    beats = [{"id": 0, "start_time": 0, "end_time": 3, "visual_template": "split_schematic",
              "primary_source": "local", "source_id": "schematic_s", "overlay_asset": "image_i"},
             {"id": 1, "start_time": 3.5, "end_time": 6, "visual_template": "cutout_callout",
              "primary_source": "local", "source_id": "missing"}]
    warnings = prepare_beats(beats, r)
    assert beats[0]["source_id"] == "image_i" and beats[0]["overlay_asset"] == "schematic_s"
    assert beats[0]["start_time"] == 0 and beats[0]["end_time"] == 3  # timing untouched
    assert any("swapped" in w for w in warnings)
    assert any("not found" in w for w in warnings)
    assert any("gap/overlap" in w for w in warnings)


def test_nvenc_failure_mid_run_retries_on_cpu(monkeypatch: pytest.MonkeyPatch,
                                              tmp_path: Path) -> None:
    img = tmp_path / "i.png"
    make_subject_image(img)
    r = make_resolver(tmp_path, [asset(img, "image")])
    s = dataclasses.replace(RenderSettings(), codec="h264_nvenc")
    ctx = CompileContext(r, s, tmp_path / "work", tmp_path)
    seen: list[str] = []

    def fake_run(cmd: list[str], timeout: int = 0) -> subprocess.CompletedProcess[str]:
        codec = cmd[cmd.index("-c:v") + 1]
        seen.append(codec)
        return subprocess.CompletedProcess(cmd, 1 if codec == "h264_nvenc" else 0, "", "boom")

    monkeypatch.setattr("src.stage6_render._run", fake_run)
    beat = {"id": 0, "visual_template": "fullscreen_clip", "primary_source": "local",
            "source_id": "image_i", "beat_style": "normal"}
    assert _render_beat(beat, 30, tmp_path / "o.mp4", ctx) == ""
    assert seen == ["h264_nvenc", "libx264"]
    assert s.codec == "libx264" and ctx.codec_switched
    _render_beat(beat, 30, tmp_path / "o2.mp4", ctx)
    assert seen[-1] == "libx264"  # later beats go straight to CPU


def test_extreme_aspect_ratio_is_letterboxed(tmp_path: Path) -> None:
    pano = tmp_path / "pano.png"
    Image.new("RGB", (6000, 100), (200, 50, 50)).save(pano)
    r = make_resolver(tmp_path, [asset(pano, "image")])
    ctx = CompileContext(r, RenderSettings(), tmp_path / "work", tmp_path)
    beat = {"id": 0, "visual_template": "fullscreen_clip", "primary_source": "local",
            "source_id": "image_pano", "beat_style": "normal"}
    job = compile_beat(beat, 60, ctx)
    assert "force_original_aspect_ratio=decrease" in job.graph and "pad=" in job.graph
    assert "force_original_aspect_ratio=increase" not in job.graph
    assert any("extreme aspect" in w for w in ctx.warnings)
    cmd = ["ffmpeg", "-y", "-v", "error", *job.input_args, "-filter_complex", job.graph,
           "-map", "[vout]", "-frames:v", "5", str(tmp_path / "pano.mp4")]
    assert subprocess.run(cmd, check=False, capture_output=True, text=True).returncode == 0
