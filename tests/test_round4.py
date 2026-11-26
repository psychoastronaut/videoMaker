"""Round-4: stock policy invalidation/enforcement, license strictness, caps, truncation, LLM budget."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from conftest import make_subject_image
from test_planner import make_ctx, neutral_transcript
from test_review_fixes import episode_with_assets, write_assets

from src import cli, stock_video
from src.common import ENV_CONFIG, read_json, write_json
from src.media import AssetResolver, undecodable_reason
from src.stage2_index_assets import run as stage2
from src.stage3_plan_scenes import PlanContext, _generate_deterministic_plan, chat_with_backoff
from src.stage4_fetch_stock import _WIKI_PD, StockFetcher
from src.stage6_render import (
    CompileContext,
    RenderSettings,
    _render_beat,
    _stock_label,
    _write_output_audit,
)


def set_policy(strict: bool) -> None:
    import os
    path = Path(os.environ[ENV_CONFIG])
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("stock_media", {})["strict_cc0_only"] = strict
    path.write_text(yaml.safe_dump(cfg))


def pexels_episode(tmp_name: str = "pol") -> tuple[Path, dict[str, Any]]:
    ep, work = episode_with_assets(tmp_name)
    img = ep / "raw" / "images" / "p.png"
    make_subject_image(img)
    asset = {"id": "pexels_video_1", "source_platform": "pexels", "source_url": "u",
             "local_path": str(img), "license_type": "Pexels License", "media_type": "image"}
    write_json(work / "licenses_audit.json", {"assets": [asset]})
    write_assets(work, [])
    beat = {"id": 0, "start_time": 0, "end_time": 2, "visual_template": "fullscreen_clip",
            "primary_source": "stock", "stock_asset_id": "pexels_video_1",
            "media_path": str(img), "license_type": "Pexels License", "beat_style": "normal",
            "search_queries": ["robot"]}
    write_json(work / "scene_plan.json", {"episode_id": tmp_name, "beats": [beat]})
    return work, beat


# ------------------------------------------------------------------ P1: policy switch


def test_policy_change_makes_stage4_stale() -> None:
    work, _ = pexels_episode()
    ep = work.parent
    cli._record_state("fetch", work)
    assert cli.stale_reason("fetch", ep, work) is None
    set_policy(True)  # strict_cc0_only
    assert "license policy" in (cli.stale_reason("fetch", ep, work) or "")
    cli._record_state("fetch", work)
    assert cli.stale_reason("fetch", ep, work) is None
    set_policy(False)
    assert cli.stale_reason("fetch", ep, work)


def test_strict_switch_refuses_fetched_platform_media(tmp_path: Path) -> None:
    work, beat = pexels_episode("pol2")
    allowed = AssetResolver(work)
    assert allowed.beat_main(beat) is not None and not allowed.refused

    set_policy(True)
    strict = AssetResolver(work)
    assert "pexels_video_1" in strict.refused
    assert strict.beat_main(beat) is None and strict.stock_asset("pexels_video_1") is None
    assert "not allowed" in (strict.stock_refusal(beat) or "")

    ctx = CompileContext(strict, RenderSettings(codec="libx264", x264_preset="ultrafast"), work,
                         tmp_path)
    note = _render_beat(beat, 30, tmp_path / "o.mp4", ctx)
    assert "refused" in note and "fallback card" in note
    assert "REFUSED" in _stock_label(beat, strict)

    out = tmp_path / "out"
    out.mkdir()
    _write_output_audit(work, out, [beat], strict, {0: note})
    audit = read_json(out / "licenses_audit.json")
    assert audit["assets"] == []
    assert audit["excluded"][0]["id"] == "pexels_video_1"
    assert "not allowed" in audit["excluded"][0]["excluded_reason"]


def test_unaudited_stock_is_never_rendered() -> None:
    work, beat = pexels_episode("pol3")
    write_json(work / "licenses_audit.json", {"assets": []})
    r = AssetResolver(work)
    assert r.beat_main(beat) is None
    assert "not in licenses_audit" in (r.stock_refusal(beat) or "")


# ------------------------------------------------------------------ P2-1..3


def test_image_download_is_size_capped(tmp_path: Path) -> None:
    f = StockFetcher({"max_image_mb": 0.001, "min_resolution": 10}, tmp_path, offline=False)
    img = tmp_path / "big.png"
    make_subject_image(img)
    data = img.read_bytes()
    assert len(data) > 1048
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200, content=data, headers={"content-type": "image/png"}))
    with httpx.Client(transport=transport) as c:
        assert f._download(c, "http://x/a.png", "openverse") is None
    assert not list(tmp_path.glob("openverse_*"))
    roomy = StockFetcher({"min_resolution": 10}, tmp_path, offline=False)
    with httpx.Client(transport=transport) as c:
        got = roomy._download(c, "http://x/a.png", "openverse")
    assert got and got[0].suffix == ".png"


@pytest.mark.parametrize("url,ok", [
    ("https://creativecommons.org/publicdomain/zero/1.0/", True),
    ("http://creativecommons.org/publicdomain/zero/1.0", True),
    ("https://www.creativecommons.org/publicdomain/mark/1.0/", True),
    ("https://evil.example/creativecommons.org/publicdomain/zero/1.0", False),
    ("https://creativecommons.org/publicdomain/zero/1.0/evil", False),
    ("https://creativecommons.org/licenses/by/4.0/", False),
    ("", False), (None, False)])
def test_archive_license_url_is_anchored(url: str | None, ok: bool) -> None:
    assert stock_video.is_archive_pd(url) is ok


@pytest.mark.parametrize("lic,ok", [
    ("CC0", True), ("CC0 1.0", True), ("CC0 1.0 Universal", True), ("Public domain", True),
    ("PD-USGov-NASA", True),
    ("Public domain; CC BY-SA 4.0", False), ("CC0 or GFDL", False),
    ("Public domain (and CC BY 2.0)", False), ("CC BY-SA 4.0", False), ("GFDL", False)])
def test_wikimedia_license_is_whole_string(lic: str, ok: bool) -> None:
    assert bool(_WIKI_PD.match(lic.lower())) is ok


# ------------------------------------------------------------------ P2-4


def test_stock_beat_callouts_never_use_local_asset_labels() -> None:
    base = make_ctx(neutral_transcript())
    assets = [{"id": "clip_scope_trace", "asset_type": "clip", "filepath": "/x/s.mp4",
               "tags": ["clip", "scope", "trace"], "description": "clip scope trace"}]
    ctx = PlanContext(base.config, assets, base.words, base.duration)
    ctx.stock_ratio = 1.0
    plan = _generate_deterministic_plan(ctx, "t")
    stock = [b for b in plan.beats if b.primary_source.value == "stock"
             and b.visual_template.value in ("fullscreen_clip", "punch_in_emphasis")]
    assert stock
    for b in stock:
        assert b.callout_text is None or not {"scope", "trace"} & set(b.callout_text.lower().split())


# ------------------------------------------------------------------ P2-5


def test_stage2_skips_truncated_mp4(tmp_path: Path) -> None:
    ep, work = episode_with_assets("tr")
    good = ep / "raw" / "clips" / "good.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=320x180:r=30:d=4", "-c:v", "libx264", "-movflags", "+faststart",
                    str(good)], check=True)
    data = good.read_bytes()
    (ep / "raw" / "clips" / "half.mp4").write_bytes(data[: len(data) // 2])
    assert undecodable_reason(good) is None
    assert "truncated" in (undecodable_reason(ep / "raw" / "clips" / "half.mp4") or "")
    result = stage2("tr", work)
    assert [a["id"] for a in result["assets"]] == ["clip_good"]
    assert any("half.mp4" in s["file"] for s in result["skipped"])


# ------------------------------------------------------------------ P2-6


class _Resp:
    status_code = 500

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError("x", request=httpx.Request("POST", "http://x"),
                                    response=self)


class _Client:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def post(self, *_a: Any, **k: Any) -> _Resp:
        self.timeouts.append(k.get("timeout"))
        return _Resp()


def test_llm_total_budget_caps_timeouts_and_stops_retrying() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(d: float) -> None:
        slept.append(d)
        now[0] += d

    c = _Client()
    with pytest.raises(httpx.HTTPError):
        chat_with_backoff(c, "u", {}, retries=5, sleep=sleep, deadline=10.0,
                          request_timeout=150, clock=lambda: now[0])
    assert c.timeouts[0] == 10.0  # per-request timeout capped to the remaining budget
    assert sum(slept) <= 10 and len(c.timeouts) <= 3  # not all 6 attempts


def test_llm_budget_already_exhausted_fails_fast() -> None:
    c = _Client()
    with pytest.raises(httpx.TimeoutException):
        chat_with_backoff(c, "u", {}, 2, deadline=5.0, request_timeout=150, clock=lambda: 9.0)
    assert c.timeouts == []
