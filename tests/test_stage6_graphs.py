from __future__ import annotations

import re
from pathlib import Path

from src.media import AssetResolver, ResolvedMedia
from src.stage6_render import (
    CompileContext,
    Motion,
    RenderSettings,
    _seconds_to_srt_time,
    _srt_cues,
    compile_beat,
    frame_spans,
    motion_chain,
    motion_for,
)


def make_ctx(tmp_path: Path) -> CompileContext:
    work = tmp_path / "work"
    work.mkdir()
    return CompileContext(AssetResolver(work), RenderSettings(), work, tmp_path)


def media(path: Path, video: bool = False) -> ResolvedMedia:
    path.write_bytes(b"x")
    return ResolvedMedia(str(path), video, path.stem, "image")


class FixedResolver(AssetResolver):
    def __init__(self, main: ResolvedMedia | None, overlay: ResolvedMedia | None = None) -> None:
        self.main, self.overlay = main, overlay

    def beat_main(self, beat):
        return self.main

    def beat_overlay(self, beat):
        return self.overlay


def refs(graph: str) -> set[int]:
    return {int(m) for m in re.findall(r"\[(\d+):v\]", graph)}


def test_every_input_is_used_exactly_once_and_template_graphs_are_labelled(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    clip = media(tmp_path / "c.mp4", video=True)
    img = media(tmp_path / "i.png")
    cut, badge, lt = (tmp_path / n for n in ("cut.png", "badge.png", "lt.png"))
    for p in (cut, badge, lt):
        p.write_bytes(b"x")
    ctx.resolver = FixedResolver(clip, img)
    base = {"id": 1, "beat_style": "normal", "callout_text": "magic smoke",
            "cutout_path": str(cut), "badge_path": str(badge), "lower_third_path": str(lt)}
    for template, n_inputs in [("fullscreen_clip", 2), ("punch_in_emphasis", 2),
                               ("cutout_callout", 3), ("split_schematic", 4)]:
        job = compile_beat({**base, "visual_template": template}, 60, ctx)
        assert job.n_inputs == n_inputs, template
        assert refs(job.graph) == set(range(n_inputs)), template
        assert job.graph.count("[vout]") == 1
        assert ";;" not in job.graph and not job.graph.endswith(";")
    assert not ctx.warnings


def test_overlay_inputs_are_real_inputs_with_loop(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    ctx.resolver = FixedResolver(media(tmp_path / "i.png"))
    p = tmp_path / "cut.png"
    p.write_bytes(b"x")
    job = compile_beat({"id": 0, "visual_template": "cutout_callout", "beat_style": "normal",
                        "cutout_path": str(p)}, 30, ctx)
    assert job.input_args.count("-loop") == 2
    assert "overlay=" in job.graph


def test_punch_is_ease_out_then_hold() -> None:
    s = RenderSettings()
    m = motion_for({"id": 0, "beat_style": "normal"}, "punch_in_emphasis", s, 90)
    assert m.kind == "punch"
    assert 1.15 <= m.end_zoom <= 1.25
    assert 6 <= m.ramp_frames <= 9  # ~0.25s at 30 fps
    expr = m.zoom_expr()
    assert "min(in/" in expr and "pow(" in expr
    assert "(t" not in expr and "t/" not in expr.replace("min(in/", "")


def test_deadpan_is_static_fail_micro_punches() -> None:
    s = RenderSettings()
    assert motion_for({"id": 0, "beat_style": "deadpan"}, "fullscreen_clip", s, 90).kind == "static"
    fail = motion_for({"id": 0, "beat_style": "fail"}, "fullscreen_clip", s, 90)
    assert fail.kind == "punch" and fail.end_zoom == s.micro_punch_zoom


def test_videos_are_conformed_and_stills_upscaled_before_zoompan() -> None:
    video = motion_chain(0, True, 1920, 1080, Motion("drift", 1.0, 1.05, total_frames=90), 30, "o")
    still = motion_chain(0, False, 1920, 1080, Motion("drift", 1.0, 1.05, total_frames=90), 30, "o")
    assert video.startswith("[0:v]fps=30,")
    assert "perspective=" in still and "eval=frame" in still and "zoompan" not in still
    assert "fps=" not in still
    assert "perspective" not in video  # no synthetic drift on video clips by default
    punch = motion_chain(0, True, 1920, 1080, Motion("punch", end_zoom=1.2, ramp_frames=7), 30, "o")
    assert "perspective=" in punch
    assert "perspective" in motion_chain(0, True, 1920, 1080, Motion("drift", 1.0, 1.05,
                                         total_frames=90), 30, "o", video_drift=True)
    static = motion_chain(0, False, 1920, 1080, Motion(), 30, "o")
    assert "perspective" not in static


def test_frame_spans_do_not_drift() -> None:
    beats = [{"end_time": 1.04}, {"end_time": 2.07}, {"end_time": 3.1}]
    spans = frame_spans(beats, 30, 3.1)
    assert sum(spans) == 93
    assert spans == [31, 31, 31]


def test_srt_time_rounds_and_carries() -> None:
    assert _seconds_to_srt_time(59.9996) == "00:01:00,000"
    assert _seconds_to_srt_time(3661.5) == "01:01:01,500"


def test_srt_cues_split_long_segments() -> None:
    words = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(30)]
    cues = _srt_cues({"segments": [{"start": 0, "end": 15, "text": "x", "words": words}]})
    assert len(cues) > 1
    assert all(end > start for start, end, _ in cues)
