"""End-to-end smoke: stages 2-6 on a tiny episode, all templates + a missing-media fallback."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from conftest import make_schematic, make_subject_image, write_wav

from src import stage5_cutouts
from src.common import episode_dir, read_json, write_json
from src.stage2_index_assets import run as stage2
from src.stage4_fetch_stock import run as stage4
from src.stage5_cutouts import _fallback_cutout
from src.stage5_cutouts import run as stage5
from src.stage6_render import run as stage6


def probe(path: Path, entries: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json",
                          str(path)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def test_render_smoke(monkeypatch) -> None:
    monkeypatch.setattr(stage5_cutouts, "_cached_matte", lambda img, src, key, out, cfg:
                        (_fallback_cutout(img), "fallback-keying"))
    ep = episode_dir("smoke")
    raw, work = ep / "raw", ep / "work"
    for sub in ("clips", "images", "schematics"):
        (raw / sub).mkdir(parents=True)
    work.mkdir()
    write_wav(raw / "voiceover.wav", 7.0)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=640x360:r=25", "-t", "1.5", "-pix_fmt", "yuv420p",
                    str(raw / "clips" / "short_clip.mp4")], check=True)
    make_subject_image(raw / "images" / "board.png")
    make_schematic(raw / "schematics" / "loop.png")
    write_json(work / "transcript.json", {
        "duration": 7.0, "full_text": "x",
        "segments": [{"id": 0, "start": 0.0, "end": 7.0, "text": "Hello there.", "words": []}]})

    stage2(ep.name, work)
    beats = [
        {"start_time": 0, "end_time": 1.5, "visual_template": "fullscreen_clip",
         "source_id": "clip_short_clip", "beat_style": "hype"},
        {"start_time": 1.5, "end_time": 3.0, "visual_template": "split_schematic",
         "source_id": "clip_short_clip", "overlay_asset": "schematic_loop",
         "callout_text": "Control loop", "beat_style": "technical"},
        {"start_time": 3.0, "end_time": 4.5, "visual_template": "cutout_callout",
         "source_id": "image_board", "callout_text": "Magic smoke", "beat_style": "fail"},
        {"start_time": 4.5, "end_time": 5.5, "visual_template": "punch_in_emphasis",
         "source_id": "image_board", "callout_text": "Boom", "beat_style": "normal"},
        {"start_time": 5.5, "end_time": 7.0, "visual_template": "fullscreen_clip",
         "primary_source": "stock", "search_queries": ["fuse"], "beat_style": "deadpan"},
    ]
    write_json(work / "scene_plan.json", {"episode_id": ep.name, "beats": beats})

    stage4(ep.name, work)  # offline -> fallback card for the stock beat
    stage5(ep.name, work)
    plan = read_json(work / "scene_plan.json")
    assert plan["total_duration"] == 7.0
    assert plan["beats"][2]["cutout_path"] and Path(plan["beats"][2]["cutout_path"]).is_file()
    assert plan["beats"][1]["badge_path"] and plan["beats"][3]["lower_third_path"]
    # offline: the stock beat is downgraded to the best local asset, keeping its template
    b4 = plan["beats"][4]
    assert b4["primary_source"] == "local" and b4["source_id"] and b4["stock_fallback_of"]
    assert "stock unavailable -> local" in b4["beat_reason"]
    assert b4["visual_template"] == "fullscreen_clip" and not b4["license_type"]

    # break a beat on purpose: renderer must fall back to a card, not corrupt the output
    plan["beats"][3]["source_id"] = "image_missing"
    write_json(work / "scene_plan.json", plan)
    result = stage6(ep.name, work)

    out = ep / "out"
    info = probe(out / "final_cut.mp4", "stream=codec_name,width,height,pix_fmt,r_frame_rate")
    video = next(s for s in info["streams"] if s["codec_name"] == "h264")
    assert (video["width"], video["height"], video["pix_fmt"]) == (1920, 1080, "yuv420p")
    assert video["r_frame_rate"] == "30/1"
    audio = next(s for s in info["streams"] if s["codec_name"] == "aac")
    assert audio
    dur = float(probe(out / "final_cut.mp4", "format=duration")["format"]["duration"])
    from src.common import load_config
    branded = load_config().get("render", {}).get("branding", False)
    # 7.0 s voiceover + (3.0 s intro + 3.0 s outro card | 1.0 s outro hold)
    assert abs(dur - (7.0 + 3.0 + 3.0 if branded else 8.0)) <= 0.1
    adur = float(probe(out / "final_cut.mp4", "stream=duration")["streams"][1]["duration"])
    vdur = float(probe(out / "final_cut.mp4", "stream=duration")["streams"][0]["duration"])
    assert abs(adur - vdur) <= 1 / 30 + 0.03
    assert any("beat 3" in w and "fallback" in w for w in result["warnings"])
    timeline = (out / "timeline_summary.md").read_text()
    assert "stock unavailable -> local" in timeline and "fallback" in timeline
    assert not (out / "final_cut.srt").exists() and (out / "licenses_audit.json").is_file()
