"""Round-3 regressions: corrupt media hangs, codec switch on card path, warnings, variety."""

from __future__ import annotations

import dataclasses
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import make_subject_image
from test_planner import ASSETS, make_ctx
from test_review_fixes import asset, episode_with_assets, make_resolver

from src.common import write_json
from src.graphics import display_case
from src.media import AssetResolver, undecodable_reason
from src.schemas import SceneBeat, ScenePlan, VisualTemplate
from src.stage2_index_assets import run as stage2
from src.stage3_plan_scenes import _validate_and_repair, chat_with_backoff
from src.stage6_render import (
    TIMEOUT_RC,
    CompileContext,
    FilterJob,
    RenderSettings,
    _beat_command,
    _merge_stage5_warnings,
    _render_beat,
    _run,
    beat_timeout,
)


def test_stage2_skips_undecodable_files(tmp_path: Path) -> None:
    ep, work = episode_with_assets("c1")
    make_subject_image(ep / "raw" / "images" / "good.png")
    (ep / "raw" / "images" / "corrupt.png").write_bytes(b"junk bytes")
    (ep / "raw" / "clips" / "bad.mp4").write_bytes(b"not a video")
    result = stage2("c1", work)
    ids = [a["id"] for a in result["assets"]]
    assert ids == ["image_good"]
    assert {Path(s["file"]).name for s in result["skipped"]} == {"corrupt.png", "bad.mp4"}


def test_resolver_rejects_undecodable_media(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")
    assert undecodable_reason(bad)
    r = make_resolver(tmp_path, [asset(bad, "image")])
    assert r.local_asset("image_bad") is None
    assert r.beat_main({"primary_source": "stock", "media_path": str(bad)}) is None


def test_beat_command_aborts_on_first_error(tmp_path: Path) -> None:
    cmd = _beat_command(FilterJob(), 30, tmp_path / "o.mp4", RenderSettings())
    assert "-xerror" in cmd


def test_run_keeps_only_stderr_tail_and_kills_on_timeout() -> None:
    noisy = ["python3", "-c", "import sys; sys.stderr.write('x' * 5_000_000)"]
    r = _run(noisy, timeout=30)
    assert r.returncode == 0 and len(r.stderr) <= 6000
    t0 = time.time()
    r = _run(["sleep", "30"], timeout=1)
    assert r.returncode == TIMEOUT_RC and time.time() - t0 < 10
    assert beat_timeout(30, 30) == 60 and beat_timeout(3000, 30) == 1000


def test_corrupt_main_media_falls_back_to_card_quickly(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"junk" * 100)
    r = make_resolver(tmp_path, [asset(bad, "image")])
    ctx = CompileContext(r, RenderSettings(codec="libx264", x264_preset="ultrafast"),
                         tmp_path / "work", tmp_path)
    beat = {"id": 0, "visual_template": "fullscreen_clip", "primary_source": "local",
            "source_id": "image_corrupt", "beat_style": "normal"}
    t0 = time.time()
    note = _render_beat(beat, 30, tmp_path / "o.mp4", ctx)
    assert "undecodable" in note and (tmp_path / "o.mp4").stat().st_size > 0
    assert time.time() - t0 < 30


def _ctx_with_nvenc(tmp_path: Path) -> CompileContext:
    r = make_resolver(tmp_path, [])
    return CompileContext(r, dataclasses.replace(RenderSettings(), codec="h264_nvenc"),
                          tmp_path / "work", tmp_path)


def test_card_path_also_switches_codec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ctx = _ctx_with_nvenc(tmp_path)
    seen: list[str] = []

    def fake_run(cmd: list[str], timeout: float = 0) -> subprocess.CompletedProcess[str]:
        codec = cmd[cmd.index("-c:v") + 1]
        seen.append(codec)
        return subprocess.CompletedProcess(cmd, 1 if codec == "h264_nvenc" else 0, "", "x")

    monkeypatch.setattr("src.stage6_render._run", fake_run)
    beat = {"id": 0, "visual_template": "fullscreen_clip", "primary_source": "local",
            "source_id": "missing", "beat_style": "normal"}
    assert "fallback card" in _render_beat(beat, 30, tmp_path / "o.mp4", ctx)
    assert seen == ["h264_nvenc", "libx264"] and ctx.codec_switched


def test_no_cpu_retry_after_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ctx = _ctx_with_nvenc(tmp_path)
    img = tmp_path / "i.png"
    make_subject_image(img)
    ctx.resolver = make_resolver(tmp_path, [asset(img, "image")])
    seen: list[str] = []

    def fake_run(cmd: list[str], timeout: float = 0) -> subprocess.CompletedProcess[str]:
        seen.append(cmd[cmd.index("-c:v") + 1])
        return subprocess.CompletedProcess(cmd, TIMEOUT_RC, "", "TIMEOUT")

    monkeypatch.setattr("src.stage6_render._run", fake_run)
    beat = {"id": 0, "visual_template": "fullscreen_clip", "primary_source": "local",
            "source_id": "image_i", "beat_style": "normal"}
    with pytest.raises(RuntimeError):  # card also "times out" in this fake
        _render_beat(beat, 30, tmp_path / "o.mp4", ctx)
    assert "libx264" not in seen and not ctx.codec_switched


def test_stage5_reasons_replace_generic_cutout_note(tmp_path: Path) -> None:
    write_json(tmp_path / "cutouts_manifest.json",
               {"warnings": ["beat 3: cutout failed for x.png (UnidentifiedImageError)"]})
    out = _merge_stage5_warnings(
        ["beat 3: cutout missing (run stage 5), rendering without cutout", "other"], tmp_path)
    assert not any("cutout missing" in w for w in out)
    assert "stage 5: beat 3: cutout failed for x.png (UnidentifiedImageError)" in out
    assert "other" in out


def test_llm_plans_get_template_variety_and_source_alternation() -> None:
    ctx = make_ctx()
    n = 8
    step = ctx.duration / n
    raw = ScenePlan(beats=[
        SceneBeat(start_time=i * step, end_time=(i + 1) * step, beat_style="fail",
                  visual_template="punch_in_emphasis",
                  source_id="clip_servo" if i % 2 else "clip_arm") for i in range(n)])
    beats = _validate_and_repair(raw, ctx).beats
    for a, b, c in zip(beats, beats[1:], beats[2:], strict=False):
        assert not (a.visual_template == b.visual_template == c.visual_template)
    assert len({b.visual_template for b in beats}) >= 2


def test_cutout_badge_names_the_subject() -> None:
    ctx = make_ctx()
    subject = next(a for a in ASSETS if a["id"] == "image_board")
    # narration is about duct tape but the subject is the board -> label from the subject
    assert ctx.subject_callout("a roll of duct tape", "duct tape", subject) == "Duct Tape"  # narration only, never the asset label
    # narration names the subject -> narration phrase kept
    assert ctx.subject_callout("the tiny board talking", "", subject).lower().count("board")
    raw = ScenePlan(beats=[SceneBeat(
        start_time=0, end_time=ctx.duration, visual_template="cutout_callout",
        source_id="clip_servo", overlay_asset="image_board", callout_text="Duct tape")])
    beat = _validate_and_repair(raw, ctx).beats[0]
    assert beat.visual_template == VisualTemplate.CUTOUT_CALLOUT
    assert 2 <= len(beat.callout_text.split()) <= 5  # taken from the narration
    assert 2 <= len(beat.callout_text.split()) <= 5


def test_display_case_is_consistent() -> None:
    assert display_case("duct tape workaround") == "Duct Tape Workaround"
    assert display_case("PID loop for LiPo") == "PID Loop For LiPo"


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("x", request=httpx.Request("POST", "http://x"),
                                        response=self)

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "{}"}}]}


class _Client:
    def __init__(self, statuses: list[int | Exception]) -> None:
        self.statuses = list(statuses)
        self.calls = 0

    def post(self, *_a: Any, **_k: Any) -> _Resp:
        self.calls += 1
        item = self.statuses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)


def test_llm_transient_errors_are_retried_with_backoff() -> None:
    slept: list[float] = []
    c = _Client([500, httpx.ReadTimeout("slow"), 200])
    assert chat_with_backoff(c, "u", {}, 2, sleep=slept.append) == "{}"
    assert c.calls == 3 and slept == [2.0, 5.0]


def test_llm_gives_up_after_retries_and_skips_4xx_retry() -> None:
    c = _Client([503, 503, 503])
    with pytest.raises(httpx.HTTPStatusError):
        chat_with_backoff(c, "u", {}, 2, sleep=lambda _s: None)
    assert c.calls == 3
    c = _Client([400, 200])
    with pytest.raises(httpx.HTTPStatusError):
        chat_with_backoff(c, "u", {}, 2, sleep=lambda _s: None)
    assert c.calls == 1


def test_asset_resolver_class_is_importable() -> None:
    assert AssetResolver
