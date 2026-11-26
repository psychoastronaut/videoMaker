"""Stock video + image pipeline with all network access mocked."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_planner import ASSETS, make_ctx, neutral_transcript

from src import stock_video
from src.media import AssetResolver
from src.schemas import SceneBeat, ScenePlan
from src.stage3_plan_scenes import _generate_deterministic_plan, stock_query
from src.stage4_fetch_stock import StockFetcher
from src.stage6_render import _stock_label


def make_video(path: Path, w: int = 1280, h: int = 720, seconds: float = 6.0) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc2=s={w}x{h}:r=25:d={seconds}", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def src_video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "src.mp4")


@pytest.fixture
def fetcher(tmp_path: Path, src_video: Path, monkeypatch: pytest.MonkeyPatch) -> StockFetcher:
    cache = tmp_path / "cache"
    cache.mkdir()
    f = StockFetcher({"pexels_api_key": "k", "pixabay_api_key": "k"}, cache, offline=False)
    f._need = 4.0
    # the "download" copies a local file instead of touching the network
    monkeypatch.setattr(stock_video, "download_capped",
                        lambda client, url, dest, max_bytes: shutil.copy(src_video, dest) and True)
    return f


class FakeClient:
    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes

    def get(self, url: str, **_k: Any) -> Any:
        for key, payload in self.routes.items():
            if key in url:
                return type("R", (), {"status_code": 200, "json": lambda self, p=payload: p})()
        return type("R", (), {"status_code": 404, "json": lambda self: {}})()


def test_pick_height_file_prefers_tallest_within_bounds() -> None:
    files = [{"url": "a", "width": 3840, "height": 2160}, {"url": "b", "width": 1920, "height": 1080},
             {"url": "c", "width": 1280, "height": 720}, {"url": "d", "width": 640, "height": 360},
             {"url": "e", "width": 720, "height": 1280}]
    assert stock_video.pick_height_file(files, 720, 1080)["url"] == "b"
    assert stock_video.pick_height_file(files, 2000, 3000)["url"] == "a"
    assert stock_video.pick_height_file(files[3:4], 720, 1080) is None
    assert stock_video.pick_height_file(files[4:], 480, 1920) is None  # portrait rejected


def test_pexels_video_downloaded_verified_and_trimmed(fetcher: StockFetcher) -> None:
    payload = {"videos": [{"id": 77, "url": "https://pexels.com/v/77", "user": {"name": "Ann"},
                           "video_files": [
                               {"file_type": "video/mp4", "width": 640, "height": 360, "link": "x"},
                               {"file_type": "video/mp4", "width": 1280, "height": 720,
                                "link": "https://cdn/77.mp4"}]}]}
    asset = fetcher._pexels_video(FakeClient({"pexels.com/videos": payload}), "robot arm", 4)  # type: ignore[arg-type]
    assert asset and asset.media_type == "video" and asset.id == "pexels_video_77"
    assert asset.license_type == "Pexels License" and asset.author == "Ann"
    assert asset.duration is not None and asset.duration <= 4.5  # trimmed from 6s
    assert asset.height == 720 and Path(asset.local_path).is_file()
    assert not (Path(asset.local_path).parent / "pexels_video_77.src").exists()
    assert stock_video.probe_video(Path(asset.local_path))


def test_low_resolution_platform_video_is_rejected(fetcher: StockFetcher, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    small = make_video(tmp_path / "small.mp4", 640, 360)
    monkeypatch.setattr(stock_video, "download_capped",
                        lambda c, u, d, m: shutil.copy(small, d) and True)
    payload = {"hits": [{"id": 5, "pageURL": "p", "user": "u", "videos": {
        "large": {"url": "https://cdn/5.mp4", "width": 1920, "height": 1080, "size": 1000}}}]}
    # the API *claims* 1080p but the real file is 360p -> ffprobe verification rejects it
    assert fetcher._pixabay_video(FakeClient({"pixabay.com/api/videos": payload}), "q", 0) is None  # type: ignore[arg-type]


def test_pixabay_video_license_and_dedupe(fetcher: StockFetcher) -> None:
    payload = {"hits": [{"id": 9, "pageURL": "https://pixabay.com/v/9", "user": "bob",
                         "tags": "robot, arm", "videos": {
        "large": {"url": "https://cdn/9.mp4", "width": 1280, "height": 720, "size": 1000}}}]}
    client = FakeClient({"pixabay.com/api/videos": payload})
    asset = fetcher._pixabay_video(client, "robot arm", 1)  # type: ignore[arg-type]
    assert asset and asset.license_type == "Pixabay Content License"
    fetcher.used_ids.add(asset.id)
    assert fetcher._pixabay_video(client, "robot arm", 2) is None  # id never reused
    assert fetcher._pixabay_video(client, "volcano", 3) is None  # unrelated tags are dropped


def test_oversized_download_is_aborted(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"x" * 5000))
    with httpx.Client(transport=transport) as client:
        dest = tmp_path / "v.mp4"
        assert not stock_video.download_capped(client, "http://x/v.mp4", dest, 1000)
        assert not dest.exists() and not list(tmp_path.glob("*.part"))
        assert stock_video.download_capped(client, "http://x/v.mp4", dest, 10_000)


def test_undecodable_video_download_is_rejected(fetcher: StockFetcher,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stock_video, "download_capped",
                        lambda c, u, d, m: Path(d).write_bytes(b"not a video") and True)
    payload = {"videos": [{"id": 1, "url": "u", "video_files": [
        {"file_type": "video/mp4", "width": 1920, "height": 1080, "link": "https://cdn/1.mp4"}]}]}
    assert fetcher._pexels_video(FakeClient({"pexels.com/videos": payload}), "q", 0) is None  # type: ignore[arg-type]


def test_wikimedia_video_requires_explicit_pd(fetcher: StockFetcher) -> None:
    def page(lic: str, idx: int) -> dict[str, Any]:
        return {"title": f"File:Robot arm {idx}.webm", "index": idx, "imageinfo": [{
            "url": f"https://upload/v{idx}.webm", "mime": "video/webm", "size": 1000,
            "descriptionurl": "https://commons/x",
            "extmetadata": {"LicenseShortName": {"value": lic}}}]}
    payload = {"query": {"pages": {"1": page("CC BY-SA 4.0", 1), "2": page("Public domain", 2)}}}
    asset = fetcher._wikimedia_video(FakeClient({"api.php": payload}), "robot", 0)  # type: ignore[arg-type]
    assert asset and asset.license_type == "Public domain"  # the CC BY-SA entry is skipped
    only_sa = {"query": {"pages": {"1": page("CC BY-SA 4.0", 1)}}}
    assert fetcher._wikimedia_video(FakeClient({"api.php": only_sa}), "robot", 0) is None  # type: ignore[arg-type]


def test_archive_video_requires_public_domain_licenseurl(fetcher: StockFetcher) -> None:
    def routes(lic_url: str) -> dict[str, Any]:
        return {"advancedsearch": {"response": {"docs": [{"identifier": "vid1",
                                                         "title": "Robot arm film"}]}},
                "metadata/vid1": {"metadata": {"licenseurl": lic_url},
                                  "files": [{"name": "a.mp4", "size": "1000"}]}}
    ok = fetcher._archive_video(
        FakeClient(routes("https://creativecommons.org/publicdomain/zero/1.0/")), "robot arm", 0)  # type: ignore[arg-type]
    assert ok and ok.license_type == "CC0" and ok.source_platform == "archive"
    assert fetcher.license_ok(ok)
    bad = fetcher._archive_video(
        FakeClient(routes("https://creativecommons.org/licenses/by/4.0/")), "robot arm", 0)  # type: ignore[arg-type]
    assert bad is None


def test_provider_order_video_first_for_fullscreen_and_images_first_for_cutouts(
        tmp_path: Path) -> None:
    f = StockFetcher({"pexels_api_key": "k"}, tmp_path, offline=False)
    vid_first = [p.__name__ for p in f._providers(True)]
    img_first = [p.__name__ for p in f._providers(False)]
    assert vid_first[0].endswith("video") and img_first[0] == "_wikimedia"
    assert vid_first.index("_wikimedia_video") < vid_first.index("_pexels_video")  # CC0 first
    nov = StockFetcher({"video": {"enabled": False}}, tmp_path, offline=False)
    assert not any(p.__name__.endswith("video") for p in nov._providers(True))


def test_stock_query_is_two_to_four_nouns() -> None:
    q = stock_query("The arm lifted the cup, swung over, and poured perfectly into the mug")
    assert 1 <= len(q.split()) <= 4
    assert "the" not in q.lower().split()
    assert stock_query("Next time I am adding a fuse") == "adding fuse" or len(
        stock_query("Next time I am adding a fuse").split()) <= 4


def test_planner_sends_unmatched_beats_to_stock_with_short_queries() -> None:
    ctx = make_ctx(neutral_transcript())
    ctx.stock_ratio = 0.5
    plan = _generate_deterministic_plan(ctx, "t")
    stock = [b for b in plan.beats if b.primary_source.value == "stock"]
    assert stock
    for b in stock:
        assert b.search_queries and len(b.search_queries[0].split()) <= 4
    assert len(stock) <= round(0.5 * len(plan.beats)) + 1


def test_timeline_flags_platform_licenses(tmp_path: Path) -> None:
    work = tmp_path / "w"
    work.mkdir()
    (work / "licenses_audit.json").write_text(json.dumps({"assets": [
        {"id": "pexels_video_1", "source_platform": "pexels", "source_url": "u",
         "local_path": "x", "license_type": "Pexels License", "media_type": "video"},
        {"id": "wikimedia_a", "source_platform": "wikimedia", "source_url": "u",
         "local_path": "x", "license_type": "CC0"},
        {"id": "fb", "source_platform": "generated", "source_url": "", "local_path": "x",
         "license_type": "generated-fallback"}]}))
    r = AssetResolver(work)
    label = lambda i, lic: _stock_label({"stock_asset_id": i, "license_type": lic}, r)
    assert "free-license (non-CC0)" in label("pexels_video_1", "Pexels License")
    assert "video" in label("pexels_video_1", "Pexels License")
    assert label("wikimedia_a", "CC0") == "stock image (CC0)"
    assert "fallback card" in label("fb", "generated-fallback")


def test_repair_keeps_stock_cutout_background_with_local_subject() -> None:
    ctx = make_ctx(neutral_transcript())
    raw = ScenePlan(beats=[SceneBeat(
        start_time=0, end_time=ctx.duration, visual_template="cutout_callout",
        primary_source="stock", overlay_asset="image_board", search_queries=["workshop bench"],
        callout_text="Board Detail")])
    beat = _validate_and_repair_one(raw, ctx)
    assert beat.primary_source.value == "stock" and beat.overlay_asset == "image_board"
    assert beat.visual_template.value == "cutout_callout" and ASSETS


def _validate_and_repair_one(plan: ScenePlan, ctx: Any) -> SceneBeat:
    from src.stage3_plan_scenes import _validate_and_repair
    return _validate_and_repair(plan, ctx).beats[0]


def test_relevance_filter() -> None:
    from src.stage4_fetch_stock import relevant
    assert relevant("robot arm", "File:Robot ARM demo.webm")
    assert relevant("servo motors", "a small servo motor")
    assert not relevant("servo motor", "servo only")  # 1-2 word queries need every word
    assert relevant("robot arm cup", "a robot arm lifting", "")  # long: two matches suffice
    assert not relevant("finally worked", "On the Air 1937 vintage radio")
    assert not relevant("robot", None, "")
    # off-topic hits seen online: one shared word is not enough
    assert stock_query("and the whole bench went dark") == ""  # adjective dropped, 1 noun left
    assert stock_query("a clean scope") == ""
    assert not relevant("clean scope engine", "QCSEE Experimental Engine logo with cartoon fish")


def test_relevance_rejects_logos_icons_maps_clipart() -> None:
    from src.stage4_fetch_stock import relevant
    for junk in ("Servo motor logo", "servo motor icon.svg", "Servo motor clipart",
                 "Servo motor cartoon", "Servo motor map"):
        assert not relevant("servo motor", junk), junk
    assert relevant("world map", "A world map of cables")  # the query asks for a map


def test_stock_query_drops_mood_and_brand_words() -> None:
    ctx = make_ctx()
    q = ctx.stock_query("Welcome to Psycho Astronaut, it finally worked")
    assert "psycho" not in q.lower() and "finally" not in q.lower()
