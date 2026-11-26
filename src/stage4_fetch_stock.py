"""Stage 4: free-license stock retriever (images and video) with a mandatory license audit.

For every ``primary_source == "stock"`` beat the fetcher queries, in order of preference for
the beat's template (video first for fullscreen/punch beats, images first for cutout
backgrounds):
- CC0 / Public Domain only: Openverse images (post-checked ``cc0``/``pdm``), Wikimedia Commons
  images and video (explicit CC0/PD license strings only), Internet Archive video (explicit
  public-domain ``licenseurl``).
- Free *platform* licenses (not CC0): Pexels and Pixabay photos/videos, only when an API key is
  set, ``allow_platform_licenses`` is true and ``strict_cc0_only`` is false. The exact platform
  license is recorded and flagged in the timeline.
Videos are size-capped, verified with ffprobe and trimmed to what the beat needs.
Beats without a verified result get a generated fallback card. The resolved asset is written
back into ``scene_plan.json`` (``stock_asset_id``, ``media_path``, ``license_type``) and every
download is recorded in ``licenses_audit.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from rich.console import Console

from src import graphics, stock_video
from src.common import USER_AGENT, is_offline, load_config, read_json, write_json
from src.schemas import LicenseAudit, PrimarySource, SceneBeat, ScenePlan, StockAsset
from src.stage3_plan_scenes import asset_label, asset_score, flatten_words
from src.stock_video import NASA_LICENSE, PEXELS_LICENSE, PIXABAY_LICENSE
from src.stockpool import load_pool, match_pool

console = Console()

OPENVERSE_OK = {"cc0", "pdm"}
# Wikimedia license strings that are explicitly CC0 / public domain (lower-cased, whole value).
# Whole-string match: a combined/dual license ("Public domain; CC BY-SA 4.0") never passes.
_WIKI_PD = re.compile(
    r"^(cc0( 1\.0( universal)?)?|cc-zero|public domain( mark 1\.0)?|pd(-[a-z0-9.\-]+)?|"
    r"pdm( 1\.0)?)$")
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


def run(episode_id: str, work_dir: Path) -> dict[str, Any]:
    """Run Stage 4: resolve stock beats and write the license audit."""
    config = load_config()
    stock_cfg = config.get("stock_media", {})
    cache_dir = work_dir / "stock_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    plan_path = work_dir / "scene_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Scene plan not found: {plan_path}")
    plan = ScenePlan.model_validate(read_json(plan_path))

    previous: dict[str, StockAsset] = {}
    audit_path = work_dir / "licenses_audit.json"
    if audit_path.is_file():
        for a in LicenseAudit.model_validate(read_json(audit_path)).assets:
            previous[a.id] = a

    offline = is_offline()
    fetcher = StockFetcher(stock_cfg, cache_dir, offline)
    # pre-fetched pool items (already downloaded + license data) are resolved before searching
    pool = load_pool(work_dir, lambda a: fetcher.license_ok(StockAsset.model_validate(a)))
    for item in pool:
        previous.setdefault(item["id"], StockAsset.model_validate(item))
    pool_used: set[str] = {b.stock_asset_id for b in plan.beats
                           if b.stock_asset_id and b.stock_asset_id in {p["id"] for p in pool}}
    audit: list[StockAsset] = []
    local_assets = read_json(work_dir / "local_assets.json").get("assets", []) \
        if (work_dir / "local_assets.json").is_file() else []
    words = flatten_words(read_json(work_dir / "transcript.json")) \
        if (work_dir / "transcript.json").is_file() else []
    pending = [b for b in plan.beats if b.primary_source.value == "stock"]
    verifier = None
    if pending and not offline:
        try:
            from src.semantic_stock import StockVerifier

            verifier = StockVerifier.create(work_dir, config)
        except Exception as e:  # noqa: BLE001 - verification is optional
            console.print(f"  [yellow]Stock visual verification unavailable: {e}[/]")
        if verifier is None:
            console.print("  [dim]Stock: keyword relevance only (no semantic backend)[/]")
    if pending:
        console.print(f"  {len(pending)} stock beat(s)"
                      + (" (offline: falling back to local media)" if offline else ""))

    for beat in plan.beats:
        retry = False
        if beat.primary_source.value != "stock":
            # a beat downgraded to local media earlier keeps its stock intent until edited
            retry = bool(beat.stock_fallback_of and beat.source_id == beat.stock_fallback_of
                         and beat.search_queries and not offline)
            if not retry:
                if beat.stock_fallback_of and beat.source_id != beat.stock_fallback_of:
                    beat.stock_fallback_of = None  # the user picked their own media
                beat.stock_asset_id = beat.license_type = beat.media_path = None
                continue
        if verifier and beat.visual_query and beat.stock_asset_id in {p["id"] for p in pool}:
            # Stage 3 attached a pool item by keywords: keep it only if it looks like the query
            vq0 = beat.visual_query.split(" | ")[0]
            item0 = next(p for p in pool if p["id"] == beat.stock_asset_id)
            sc0 = _safe(verifier.score, item0["local_path"], vq0)
            if sc0 is not None and sc0 < verifier.threshold:
                console.print(f"  [yellow]beat {beat.id}: pool item '{item0['id']}' rejected "
                              f"(similarity {sc0:.2f} < {verifier.threshold:.2f} to '{vq0}')[/]")
                pool_used.discard(item0["id"])
                beat.stock_asset_id = beat.license_type = beat.media_path = None
            elif sc0 is not None:
                beat.match_score = round(sc0, 4)
        queries = beat.search_queries or [beat.callout_text or f"beat {beat.id}"]

        reused = previous.get(beat.stock_asset_id or "")
        if reused and reused.license_type != "generated-fallback" \
                and Path(reused.local_path).is_file() and reused.query in queries \
                and fetcher.license_ok(reused):
            asset: StockAsset | None = reused
            asset.beat_id = beat.id
        else:
            asset = None
            if not beat.stock_asset_id or beat.stock_asset_id not in {p["id"] for p in pool}:
                item = match_pool(pool, queries, images_only=beat.stock_media == "image",
                                  exclude=pool_used)
                if verifier and beat.visual_query:
                    # visual similarity decides: best pool item for the query, if good enough
                    ranked = _safe(verifier.rank_pool, pool, beat.visual_query.split(" | ")[0],
                                   pool_used, beat.stock_media == "image")
                    item = ranked[0] if ranked else None
                    if ranked:
                        beat.match_score = round(ranked[1], 4)
                if item is not None:  # a pool item covers this beat: no new search needed
                    pool_used.add(item["id"])
                    asset = StockAsset.model_validate(item)
                    asset.beat_id = beat.id
                    beat.search_queries = [item["query"]]
            images_only = beat.stock_media == "image"  # entity establishing shots are photos
            prefer_video = not images_only and beat.visual_template.value in (
                "fullscreen_clip", "punch_in_emphasis", "split_schematic")
            tried: list[str] = []
            todo = list(queries) if asset is None else []
            if todo and verifier and beat.visual_query:
                vq1 = beat.visual_query.split(" | ")[0]
                if vq1.lower() not in {q.lower() for q in todo} and len(vq1.split()) <= 6:
                    todo.append(vq1)  # the narration-derived query is a valid search too
            requeries = verifier.requeries if verifier else 0
            while todo:
                q = todo.pop(0)
                tried.append(q)
                asset = fetcher.find(q, beat.id, prefer_video=prefer_video,
                                     need_seconds=beat.duration + 1.0, images_only=images_only)
                if asset and verifier and beat.visual_query:
                    vq = beat.visual_query.split(" | ")[0]
                    sc = _safe(verifier.score, asset.local_path, vq)
                    if sc is not None and sc < verifier.threshold:
                        console.print(f"  [yellow]beat {beat.id}: '{asset.filename}' rejected "
                                      f"(similarity {sc:.2f} < {verifier.threshold:.2f} to "
                                      f"'{vq}')[/]")
                        fetcher.used_ids.add(asset.id)
                        asset = None
                        if not todo and requeries > 0:
                            requeries -= 1
                            alt = _safe(verifier.alt_query, q, vq, list(tried))
                            if alt:
                                todo.append(alt)
                                beat.search_queries = [*beat.search_queries, alt]
                        continue
                    if sc is not None:
                        beat.match_score = round(sc, 4)
                if asset:
                    break
            if asset is None and images_only and not offline:
                # query ladder: head noun, hypernym (generic stand-ins, never captioned)
                for q in queries[:1]:
                    for rung, _ in query_ladder(q)[1:]:
                        asset = fetcher.find(rung, beat.id, prefer_video=False,
                                             images_only=True, relaxed=True)
                        if asset:
                            if beat.visual_template.value == "fullscreen_clip":
                                beat.callout_text = None
                            break
        if asset is None:
            if retry:
                continue  # still nothing: keep the local fallback chosen earlier
            local = _local_fallback(beat, plan, local_assets, words)
            if local is not None:
                _downgrade_to_local(beat, local)
                console.print(f"  [yellow]beat {beat.id}: no verified free media for "
                              f"'{queries[0]}'; using local {local['id']}[/]")
                continue
            asset = _fallback_asset(cache_dir, queries[0], beat.id,
                                    title=beat.callout_text, style=beat.beat_style.value)
            console.print(f"  [yellow]beat {beat.id}: no free media and no local asset; "
                          "using a title card[/]")
        else:
            console.print(f"  beat {beat.id}: {asset.media_type} from "
                          f"{asset.source_platform} '{asset.filename}' ({asset.license_type})")
        if retry:  # stock found after an earlier downgrade: switch back to stock
            beat.primary_source = PrimarySource.STOCK
            beat.source_id = ""
            beat.stock_fallback_of = None
            beat.beat_reason = re.sub(r"(^|; )stock unavailable -> local .*$", "", beat.beat_reason)
        fetcher.used_ids.add(asset.id)
        beat.stock_asset_id = asset.id
        beat.media_path = asset.local_path
        beat.license_type = asset.license_type
        audit.append(asset)

    if verifier:
        verifier.close()
    write_json(plan_path, plan.model_dump(mode="json"))
    result = LicenseAudit(assets=audit)
    write_json(audit_path, result.model_dump(mode="json"))  # last: marks the stage complete
    console.print(f"  License audit saved: {audit_path} ({len(audit)} assets) {result.summary}")
    return result.model_dump(mode="json")


def _safe(fn: Any, *args: Any) -> Any:
    """Run an optional semantic check; any failure means 'no opinion' (keyword logic stands)."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [dim]semantic stock check skipped: {e}[/]")
        return None


RECENT_WINDOW = 20.0  # seconds within which the same local asset is avoided


def _local_fallback(beat: SceneBeat, plan: ScenePlan, assets: list[dict[str, Any]],
                    words: list[Any]) -> dict[str, Any] | None:
    """Best local clip/image for a stock beat with no media: relevance to the narration and
    queries first, then not used in the surrounding ~20s, clips before stills."""
    cands = [a for a in assets if a["asset_type"] in ("clip", "image")]
    if beat.visual_template.value == "cutout_callout" and beat.overlay_asset:
        others = [a for a in cands if a["id"] != beat.overlay_asset]
        cands = others or cands  # a still of the subject itself is the last resort
    if not cands:
        return None
    text = " ".join([w.text for w in words
                     if beat.start_time <= (w.start + w.end) / 2 < beat.end_time]
                    + beat.search_queries)
    recent: dict[str, int] = {}
    for b in plan.beats:
        if b is beat or abs(b.start_time - beat.start_time) > RECENT_WINDOW:
            continue
        weight = 5 if abs(b.id - beat.id) == 1 else 1  # never repeat the neighbouring beat
        for ident in (b.source_id, b.overlay_asset):
            if ident:
                recent[ident] = recent.get(ident, 0) + weight
    return max(cands, key=lambda a: (asset_score(a, text) - 2 * recent.get(a["id"], 0),
                                     a["asset_type"] == "clip", a["id"]))


def _downgrade_to_local(beat: SceneBeat, asset: dict[str, Any]) -> None:
    """Keep the beat's template/motion/callout; swap the unavailable stock for local media."""
    beat.primary_source = PrimarySource.LOCAL
    beat.source_id = asset["id"]
    beat.stock_asset_id = beat.license_type = beat.media_path = None
    beat.stock_fallback_of = asset["id"]
    note = f"stock unavailable -> local {asset_label(asset)}"
    base = re.sub(r"(^|; )stock unavailable -> local .*$", "", beat.beat_reason)
    beat.beat_reason = f"{base}; {note}" if base else note


def _fallback_asset(cache_dir: Path, query: str, beat_id: int, title: str | None = None,
                    style: str = "normal") -> StockAsset:
    """Designed title card (no debug text) for a stock beat when no media of any kind exists."""
    card = cache_dir / f"fallback_beat{beat_id:03d}.png"
    graphics.make_title_card(card, title or query, style)
    digest = hashlib.md5(f"{beat_id}:{query}".encode()).hexdigest()[:8]
    return StockAsset(
        id=f"fallback_{digest}", source_platform="generated", source_url="",
        local_path=str(card), license_type="generated-fallback", license_url="",
        width=1920, height=1080, filename=card.name, beat_id=beat_id, query=query,
        downloaded_at=datetime.now(UTC).isoformat())


class StockFetcher:
    """Tries each allowed provider in order and returns the first verified, usable image."""

    def __init__(self, cfg: dict[str, Any], cache_dir: Path, offline: bool) -> None:
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.offline = offline
        self.min_res = int(cfg.get("min_resolution", 1280))
        self.max_results = int(cfg.get("max_results_per_query", 5))
        self.timeout = float(cfg.get("download_timeout", 30))
        self.pexels_key = cfg.get("pexels_api_key") or os.environ.get("PEXELS_API_KEY", "")
        self.pixabay_key = cfg.get("pixabay_api_key") or os.environ.get("PIXABAY_API_KEY", "")
        self.openverse_key = cfg.get("openverse_api_key") or os.environ.get(
            "OPENVERSE_API_KEY", "")
        self.strict = bool(cfg.get("strict_cc0_only", False))
        self.allow_platform = bool(cfg.get("allow_platform_licenses", True)) and not self.strict
        vcfg = cfg.get("video", {})
        self.video_enabled = bool(vcfg.get("enabled", True))
        self.min_h = int(vcfg.get("min_height", 720))
        self.min_h_open = int(vcfg.get("min_height_open", 480))
        self.max_h = int(vcfg.get("max_height", 1080))
        self.max_bytes = int(float(vcfg.get("max_file_mb", 80)) * 1024 * 1024)
        self.max_seconds = float(vcfg.get("max_clip_seconds", 12))
        self.max_image_bytes = int(float(cfg.get("max_image_mb", 25)) * 1024 * 1024)
        self.use_archive = bool(vcfg.get("archive_org", True))
        self.used_ids: set[str] = set()
        self._need = 6.0
        self._prefer_video = True
        self._images_only = False
        self._relaxed = False
        self._last_wiki = 0.0
        self.smithsonian_key = (cfg.get("smithsonian_api_key")
                                or os.environ.get("SMITHSONIAN_API_KEY", "") or "DEMO_KEY")
        self.accepted = {str(x).lower() for x in cfg.get(
            "accepted_openverse_licenses", OPENVERSE_OK)} & OPENVERSE_OK or OPENVERSE_OK

    def license_ok(self, asset: StockAsset) -> bool:
        """Re-check an (already downloaded) asset against the *current* license policy."""
        lic, platform = asset.license_type, asset.source_platform
        if platform == "generated":
            return lic == "generated-fallback"
        if platform == "openverse":
            return lic.lower() in self.accepted
        if platform == "wikimedia":
            return bool(_WIKI_PD.match(lic.lower()))
        if platform == "nasa":
            return lic == NASA_LICENSE
        if platform == "smithsonian":
            return lic == "CC0"
        if platform == "archive":
            return bool(stock_video.is_archive_pd(asset.license_url))
        if platform == "pexels":
            return self.allow_platform and lic == PEXELS_LICENSE
        if platform == "pixabay":
            return self.allow_platform and lic == PIXABAY_LICENSE
        return False

    def policy_signature(self) -> str:
        """Hash of everything that decides which stock media is allowed. Stored with the
        Stage 4 stamp so changing the policy re-runs the stage."""
        payload = {
            "strict": self.strict, "platform": self.allow_platform,
            "accepted": sorted(self.accepted), "video": self.video_enabled,
            "archive": self.use_archive,
            "providers": sorted(p.__name__ for p in self._providers(True)),
            "offline": self.offline,
            "min": [self.min_res, self.min_h, self.min_h_open],
            "keys": [bool(self.pexels_key), bool(self.pixabay_key), bool(self.openverse_key),
                     self.smithsonian_key != "DEMO_KEY"],
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _providers(self, prefer_video: bool, images_only: bool = False) -> list[Any]:
        videos: list[Any] = []
        if self.video_enabled and not images_only:
            videos = [self._wikimedia_video, self._nasa_video]
            if self.use_archive:
                videos.append(self._archive_video)
            if self.allow_platform:
                videos += [self._pexels_video, self._pixabay_video]
        images = [self._wikimedia, self._nasa, self._openverse, self._smithsonian]
        if self.allow_platform:
            images += [self._pexels, self._pixabay]
        # Platform providers have far better HD coverage; keep CC0/PD first within a group.
        return videos + images if prefer_video else images + videos

    def find(self, query: str, beat_id: int, *, prefer_video: bool = True,
             need_seconds: float = 6.0, images_only: bool = False,
             relaxed: bool = False) -> StockAsset | None:
        if self.offline:
            return None
        self._need = min(self.max_seconds, max(need_seconds, 3.0))
        self._prefer_video = prefer_video
        self._images_only = images_only
        self._relaxed = relaxed
        with httpx.Client(timeout=self.timeout, follow_redirects=True,
                          headers={"User-Agent": USER_AGENT}) as client:
            for provider in self._providers(prefer_video, images_only):
                try:
                    asset = provider(client, query, beat_id)
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                    console.print(f"  [dim]{provider.__name__.lstrip('_')} failed: {e}[/]")
                    continue
                if asset and asset.id not in self.used_ids:
                    return asset
                time.sleep(0.2)
        return None

    # -- video providers -------------------------------------------------------------------

    def _video_asset(self, client: httpx.Client, *, source: str, vid: str, url: str,
                     source_url: str, license_type: str, license_url: str | None,
                     author: str | None, beat_id: int, query: str, min_h: int
                     ) -> StockAsset | None:
        """Download (capped), verify with ffprobe, trim -> StockAsset, or None."""
        asset_id = f"{source}_video_{vid}"
        if asset_id in self.used_ids:
            return None
        raw = self.cache_dir / f"{asset_id}.src"
        trimmed = self.cache_dir / f"{asset_id}_t{round(self._need)}.mp4"
        if not trimmed.is_file():
            if not raw.is_file() and not stock_video.download_capped(client, url, raw,
                                                                    self.max_bytes):
                return None
            info = stock_video.probe_video(raw)
            if not info or info[1] < min_h or info[0] < info[1]:
                raw.unlink(missing_ok=True)
                return None
            if not stock_video.trim_video(raw, trimmed, self._need, self.max_h):
                return None
            raw.unlink(missing_ok=True)  # keep only the trimmed copy
        info = stock_video.probe_video(trimmed)
        if not info:
            return None
        return StockAsset(
            id=asset_id, source_platform=source, source_url=source_url, local_path=str(trimmed),
            license_type=license_type, license_url=license_url, width=info[0], height=info[1],
            author=author, filename=trimmed.name, beat_id=beat_id, query=query,
            media_type="video", duration=round(info[2], 2), downloaded_at=_now())

    def _wikimedia_video(self, client: httpx.Client, query: str, beat_id: int
                         ) -> StockAsset | None:
        api = self.cfg.get("wikimedia_api_url", "https://commons.wikimedia.org/w/api.php")
        resp = client.get(api, params={
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
            "gsrsearch": f"{query} filetype:video", "gsrlimit": self.max_results * 3,
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata"})
        if resp.status_code != 200:
            return None
        pages = resp.json().get("query", {}).get("pages", {})
        for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("video/webm", "video/ogg") or not info.get("url"):
                continue
            if (info.get("size") or 0) > self.max_bytes:
                continue
            meta = info.get("extmetadata", {})
            lic = _wiki_license(meta)
            if not _WIKI_PD.match(lic.lower()):
                continue
            title = page.get("title", "")
            if not relevant(query, title, meta.get("ImageDescription", {}).get("value", ""),
                            strong_names=self._images_only, relaxed=self._relaxed):
                continue
            author = re.sub(r"<[^>]+>", "", str(meta.get("Artist", {}).get("value", ""))) or None
            asset = self._video_asset(
                client, source="wikimedia",
                vid=hashlib.md5(title.encode()).hexdigest()[:10], url=info["url"],
                source_url=info.get("descriptionurl", info["url"]), license_type=lic,
                license_url=meta.get("LicenseUrl", {}).get("value")
                or f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                author=author, beat_id=beat_id, query=query, min_h=self.min_h_open)
            if asset:
                return asset
        return None

    def _archive_video(self, client: httpx.Client, query: str, beat_id: int
                       ) -> StockAsset | None:
        resp = client.get("https://archive.org/advancedsearch.php", params={
            "q": f"({query}) AND mediatype:movies AND (licenseurl:*publicdomain* OR "
                 "licenseurl:*zero*)",
            "fl[]": ["identifier", "title", "creator", "description"], "rows": self.max_results,
            "output": "json"})
        if resp.status_code != 200:
            return None
        for doc in resp.json().get("response", {}).get("docs", []):
            ident = doc.get("identifier")
            if not ident or not relevant(query, doc.get("title"), doc.get("description"),
                                               strong_names=self._images_only, relaxed=self._relaxed):
                continue
            meta_resp = client.get(f"https://archive.org/metadata/{ident}")
            if meta_resp.status_code != 200:
                continue
            meta = meta_resp.json()
            lic_url = str(meta.get("metadata", {}).get("licenseurl", ""))
            if not stock_video.is_archive_pd(lic_url):  # post-check the real metadata
                continue
            files = [f for f in meta.get("files", [])
                     if str(f.get("name", "")).lower().endswith(".mp4")
                     and int(f.get("size") or 0) <= self.max_bytes]
            if not files:
                continue
            f = max(files, key=lambda x: int(x.get("size") or 0))
            lic = "CC0" if "zero" in lic_url.lower() else "Public Domain Mark"
            asset = self._video_asset(
                client, source="archive", vid=ident,
                url=f"https://archive.org/download/{ident}/{f['name']}",
                source_url=f"https://archive.org/details/{ident}", license_type=lic,
                license_url=lic_url, author=str(doc.get("creator") or "") or None,
                beat_id=beat_id, query=query, min_h=self.min_h_open)
            if asset:
                return asset
        return None

    def _pexels_video(self, client: httpx.Client, query: str, beat_id: int
                      ) -> StockAsset | None:
        if not self.pexels_key:
            return None
        resp = client.get("https://api.pexels.com/videos/search",
                          headers={"Authorization": self.pexels_key},
                          params={"query": query, "per_page": self.max_results,
                                  "orientation": "landscape", "size": "medium"})
        if resp.status_code != 200:
            return None
        for v in resp.json().get("videos", []):
            f = stock_video.pick_height_file(stock_video.pexels_files(v), self.min_h, self.max_h)
            if not f:
                continue
            asset = self._video_asset(
                client, source="pexels", vid=str(v.get("id")), url=f["url"],
                source_url=v.get("url", f["url"]), license_type=PEXELS_LICENSE,
                license_url="https://www.pexels.com/license/",
                author=(v.get("user") or {}).get("name"), beat_id=beat_id, query=query,
                min_h=self.min_h)
            if asset:
                return asset
        return None

    def _pixabay_video(self, client: httpx.Client, query: str, beat_id: int
                       ) -> StockAsset | None:
        if not self.pixabay_key:
            return None
        resp = client.get("https://pixabay.com/api/videos/", params={
            "key": self.pixabay_key, "q": query, "per_page": max(self.max_results, 3)})
        if resp.status_code != 200:
            return None
        for hit in resp.json().get("hits", []):
            f = stock_video.pick_height_file(stock_video.pixabay_files(hit), self.min_h,
                                             self.max_h)
            if not f or (f.get("size") and f["size"] > self.max_bytes) \
                    or not relevant(query, hit.get("tags"), strong_names=self._images_only, relaxed=self._relaxed):
                continue
            asset = self._video_asset(
                client, source="pixabay", vid=str(hit.get("id")), url=f["url"],
                source_url=hit.get("pageURL", f["url"]), license_type=PIXABAY_LICENSE,
                license_url="https://pixabay.com/service/license-summary/",
                author=hit.get("user"), beat_id=beat_id, query=query, min_h=self.min_h)
            if asset:
                return asset
        return None

    # -- providers -------------------------------------------------------------------------

    def _openverse(self, client: httpx.Client, query: str, beat_id: int) -> StockAsset | None:
        api = self.cfg.get("openverse_api_url", "https://api.openverse.org/v1/").rstrip("/")
        headers = {"Authorization": f"Bearer {self.openverse_key}"} if self.openverse_key else {}
        resp = client.get(f"{api}/images/", headers=headers, params={
            "q": query, "license": ",".join(sorted(self.accepted)), "page_size":
            self.max_results * 2, "size": "large"})
        if resp.status_code != 200:
            return None
        for r in resp.json().get("results", []):
            lic = str(r.get("license", "")).lower()
            if lic not in self.accepted:  # post-check: never trust the query filter alone
                continue
            url = r.get("url")
            tags = " ".join(str(t.get("name", "")) for t in r.get("tags") or [])
            if not url or not relevant(query, r.get("title"), tags,
                                                       strong_names=self._images_only, relaxed=self._relaxed):
                continue
            got = self._download(client, url, "openverse")
            if not got:
                continue
            path, w, h = got
            return StockAsset(
                id=f"openverse_{r.get('id', hashlib.md5(url.encode()).hexdigest()[:10])}",
                source_platform="openverse", source_url=r.get("foreign_landing_url") or url,
                local_path=str(path), license_type=lic.upper() if lic == "cc0" else "PDM",
                license_url=r.get("license_url"), width=w, height=h,
                thumb_url=r.get("thumbnail"), author=r.get("creator"), filename=path.name,
                beat_id=beat_id, query=query, downloaded_at=_now())
        return None

    def _wiki_pages(self, client: httpx.Client, search: str, limit: int, extra: dict[str, Any]
                    ) -> list[dict[str, Any]]:
        """One Commons File-namespace search (throttled; retried when rate-limited)."""
        api = self.cfg.get("wikimedia_api_url", "https://commons.wikimedia.org/w/api.php")
        params: dict[str, Any] = {
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
            "gsrsearch": search, "gsrlimit": limit, "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata", **extra}
        for delay in (0.0, 2.0, 5.0):
            wait = self._last_wiki + 1.0 - time.monotonic()
            time.sleep(max(delay, wait, 0.0))
            self._last_wiki = time.monotonic()
            resp = client.get(api, params=params)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith(
                    "application/json"):
                pages = resp.json().get("query", {}).get("pages", {})
                return sorted(pages.values(), key=lambda p: p.get("index", 0))
        return []

    def _wikimedia(self, client: httpx.Client, query: str, beat_id: int) -> StockAsset | None:
        """Commons images (File namespace). Most files are share-alike, so search the plain
        query (no category suffix, which kills recall) and post-filter on the extmetadata
        license: only explicit CC0/PD, bitmaps, landscape and >= min_resolution."""
        for search in (f"{query} filetype:bitmap", query):
            for page in self._wiki_pages(client, search, 50, {"iiurlwidth": 2560}):
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime") not in ALLOWED_MIME:
                    continue
                w0, h0 = info.get("width") or 0, info.get("height") or 1
                if w0 < h0 or max(w0, h0) < self.min_res:
                    continue
                meta = info.get("extmetadata", {})
                lic = _wiki_license(meta)
                if not _WIKI_PD.match(lic.lower()):
                    continue
                title = page.get("title", "")
                url = info.get("thumburl") or info.get("url")
                aid = f"wikimedia_{hashlib.md5(title.encode()).hexdigest()[:10]}"
                if not url or aid in self.used_ids or not relevant(
                        query, title, meta.get("ImageDescription", {}).get("value", ""),
                        strong_names=self._images_only, relaxed=self._relaxed):
                    continue
                got = self._download(client, url, "wikimedia")
                if not got:
                    continue
                path, w, h = got
                author = re.sub(r"<[^>]+>", "", str(meta.get("Artist", {}).get("value", ""))) \
                    or None
                return StockAsset(
                    id=aid, source_platform="wikimedia",
                    source_url=info.get("descriptionurl", url), local_path=str(path),
                    license_type=lic,
                    license_url=meta.get("LicenseUrl", {}).get("value")
                    or f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                    width=w, height=h, author=author, filename=path.name, beat_id=beat_id,
                    query=query, downloaded_at=_now())
        return None

    def _nasa_items(self, client: httpx.Client, query: str, media: str) -> list[dict[str, Any]]:
        resp = client.get("https://images-api.nasa.gov/search",
                          params={"q": query, "media_type": media})
        if resp.status_code != 200:
            return []
        return resp.json().get("collection", {}).get("items", [])[: self.max_results * 4]

    def _nasa_asset_url(self, client: httpx.Client, nasa_id: str, wanted: tuple[str, ...]
                        ) -> str | None:
        resp = client.get(f"https://images-api.nasa.gov/asset/{nasa_id}")
        if resp.status_code != 200:
            return None
        hrefs = [i.get("href", "") for i in resp.json().get("collection", {}).get("items", [])]
        for tag in wanted:
            hit = next((h for h in hrefs if tag in h.lower()), None)
            if hit:
                return hit.replace("http://", "https://")
        return None

    def _nasa(self, client: httpx.Client, query: str, beat_id: int) -> StockAsset | None:
        """NASA Image and Video Library: no key, public domain (NASA media guidelines)."""
        for item in self._nasa_items(client, query, "image"):
            data = (item.get("data") or [{}])[0]
            nid = data.get("nasa_id")
            aid = f"nasa_{nid}"
            if not nid or aid in self.used_ids or not relevant(
                    query, data.get("title"), data.get("description"),
                    " ".join(data.get("keywords") or []), strong_names=self._images_only,
                    relaxed=self._relaxed):
                continue
            url = self._nasa_asset_url(client, nid, ("~large.", "~orig.", "~medium."))
            got = self._download(client, url, "nasa") if url else None
            if not got:
                continue
            path, w, h = got
            return StockAsset(
                id=aid, source_platform="nasa", source_url=f"https://images.nasa.gov/details/{nid}",
                local_path=str(path), license_type=NASA_LICENSE,
                license_url="https://www.nasa.gov/nasa-brand-center/images-and-media/",
                width=w, height=h, author=data.get("photographer") or data.get("center"),
                filename=path.name, beat_id=beat_id, query=query, downloaded_at=_now())
        return None

    def _nasa_video(self, client: httpx.Client, query: str, beat_id: int) -> StockAsset | None:
        for item in self._nasa_items(client, query, "video"):
            data = (item.get("data") or [{}])[0]
            nid = data.get("nasa_id")
            if not nid or f"nasa_video_{nid}" in self.used_ids or not relevant(
                    query, data.get("title"), data.get("description"),
                    " ".join(data.get("keywords") or []), relaxed=self._relaxed):
                continue
            url = self._nasa_asset_url(client, nid, ("~medium.mp4", "~large.mp4", "~mobile.mp4"))
            if not url:
                continue
            asset = self._video_asset(
                client, source="nasa", vid=nid, url=url,
                source_url=f"https://images.nasa.gov/details/{nid}", license_type=NASA_LICENSE,
                license_url="https://www.nasa.gov/nasa-brand-center/images-and-media/",
                author=data.get("photographer") or data.get("center"), beat_id=beat_id,
                query=query, min_h=self.min_h_open)
            if asset:
                return asset
        return None

    def _smithsonian(self, client: httpx.Client, query: str, beat_id: int) -> StockAsset | None:
        """Smithsonian Open Access (CC0). Uses SMITHSONIAN_API_KEY (free at api.data.gov) or
        the heavily rate-limited DEMO_KEY."""
        resp = client.get("https://api.si.edu/openaccess/api/v1.0/search", params={
            "q": f'{query} AND online_media_type:"Images"', "api_key": self.smithsonian_key,
            "rows": self.max_results * 3})
        if resp.status_code != 200:
            return None
        for row in resp.json().get("response", {}).get("rows", []):
            desc = row.get("content", {}).get("descriptiveNonRepeating", {})
            for m in (desc.get("online_media") or {}).get("media", []):
                if m.get("type") != "Images" or (m.get("usage") or {}).get("access") != "CC0":
                    continue
                url = m.get("content")
                aid = f"smithsonian_{hashlib.md5(str(url).encode()).hexdigest()[:10]}"
                if not url or aid in self.used_ids or not relevant(
                        query, row.get("title"), "", strong_names=self._images_only,
                        relaxed=self._relaxed):
                    continue
                got = self._download(client, url, "smithsonian")
                if not got:
                    continue
                path, w, h = got
                return StockAsset(
                    id=aid, source_platform="smithsonian",
                    source_url=desc.get("record_link") or desc.get("guid") or str(url),
                    local_path=str(path), license_type="CC0",
                    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                    width=w, height=h, filename=path.name, beat_id=beat_id, query=query,
                    downloaded_at=_now())
        return None

    def _pexels(self, client: httpx.Client, query: str, beat_id: int) -> StockAsset | None:
        if not self.pexels_key:
            return None
        resp = client.get("https://api.pexels.com/v1/search",
                          headers={"Authorization": self.pexels_key},
                          params={"query": query, "per_page": self.max_results,
                                  "orientation": "landscape"})
        if resp.status_code != 200:
            return None
        for photo in resp.json().get("photos", []):
            url = photo.get("src", {}).get("large2x") or photo.get("src", {}).get("original")
            got = self._download(client, url, "pexels") if url else None
            if not got:
                continue
            path, w, h = got
            return StockAsset(
                id=f"pexels_{photo.get('id')}", source_platform="pexels",
                source_url=photo.get("url", url), local_path=str(path),
                license_type=PEXELS_LICENSE, license_url="https://www.pexels.com/license/",
                width=w, height=h, author=photo.get("photographer"), filename=path.name,
                beat_id=beat_id, query=query, downloaded_at=_now())
        return None

    def _pixabay(self, client: httpx.Client, query: str, beat_id: int) -> StockAsset | None:
        if not self.pixabay_key:
            return None
        resp = client.get("https://pixabay.com/api/", params={
            "key": self.pixabay_key, "q": query, "image_type": "photo",
            "per_page": max(self.max_results, 3), "orientation": "horizontal"})
        if resp.status_code != 200:
            return None
        for hit in resp.json().get("hits", []):
            url = hit.get("largeImageURL")
            if not relevant(query, hit.get("tags"), strong_names=self._images_only,
                            relaxed=self._relaxed):
                continue
            got = self._download(client, url, "pixabay") if url else None
            if not got:
                continue
            path, w, h = got
            return StockAsset(
                id=f"pixabay_{hit.get('id')}", source_platform="pixabay",
                source_url=hit.get("pageURL", url), local_path=str(path),
                license_type=PIXABAY_LICENSE,
                license_url="https://pixabay.com/service/license-summary/", width=w, height=h,
                author=hit.get("user"), filename=path.name, beat_id=beat_id, query=query,
                downloaded_at=_now())
        return None

    # -- download --------------------------------------------------------------------------

    def _download(self, client: httpx.Client, url: str, source: str
                  ) -> tuple[Path, int, int] | None:
        """Size-capped download, then verify it is a decodable image of sufficient size."""
        key = hashlib.md5(f"{source}:{url}".encode()).hexdigest()[:12]
        cached = next((p for p in self.cache_dir.glob(f"{source}_{key}.*")
                       if p.suffix not in (".part", ".dl")), None)
        if cached is None:
            tmp = self.cache_dir / f"{source}_{key}.dl"
            if not stock_video.download_capped(client, url, tmp, self.max_image_bytes):
                return None
            cached = tmp
        try:
            with Image.open(cached) as img:
                img.verify()
            with Image.open(cached) as img:
                w, h = img.size
                fmt = (img.format or "JPEG").lower()
        except (OSError, ValueError):
            cached.unlink(missing_ok=True)
            return None
        if w < h:  # B-roll is 16:9: portrait photos would be cropped to a sliver
            cached.unlink(missing_ok=True) if cached.suffix == ".dl" else None
            return None
        if max(w, h) < self.min_res:
            cached.unlink(missing_ok=True) if cached.suffix == ".dl" else None
            return None
        if cached.suffix == ".dl":
            final = self.cache_dir / f"{source}_{key}.{ {'jpeg': 'jpg'}.get(fmt, fmt) }"
            cached.replace(final)
            cached = final
        return self._maybe_upscale(cached, w, h)

    def _maybe_upscale(self, path: Path, w: int, h: int) -> tuple[Path, int, int]:
        """Optional AI upscale of a stock still below the upscale threshold (never fatal)."""
        try:
            from src import upscale
            cfg = upscale.upscale_config(load_config())
            if max(w, h) >= int(cfg.get("min_long_side", 1400)):
                return path, w, h
            new = Path(upscale.upscale_stock_image(path, self.cache_dir, load_config()))
            if new != path and new.is_file():
                with Image.open(new) as img:
                    return new, *img.size
        except Exception as e:  # noqa: BLE001 - extra is optional; keep the original
            console.print(f"  [dim]stock upscale skipped: {type(e).__name__}[/]")
        return path, w, h


HYPERNYMS = {
    "lathe": "machine tool", "cnc": "machine tool", "mill": "machine tool",
    "milling": "machine tool", "drill": "machine tool", "press": "machine tool",
    "soldering": "electronics workshop", "solder": "electronics workshop",
    "circuit": "electronics", "pcb": "electronics", "board": "electronics",
    "robot": "industrial robot", "arm": "industrial robot", "drone": "aircraft",
    "factory": "factory floor", "assembly": "factory floor", "warehouse": "factory floor",
    "skyline": "city skyline", "city": "city skyline", "bridge": "city skyline",
    "harbor": "port", "port": "cargo ship", "reactor": "power plant", "turbine": "power plant",
    "engine": "machinery", "workshop": "workshop", "laboratory": "laboratory",
}
_LADDER_NOISE = frozenset(["the", "and", "for", "with", "from", "into", "this", "that"])


def query_ladder(query: str) -> list[tuple[str, bool]]:
    """Fallback queries for an entity, best first: (query, exact). Full phrase, head noun
    alone (for a bare place name the phrase is kept), a hypernym ("machine tool", "workshop",
    "factory floor"). Only the first rung is exact; the rest are generic stand-ins that the
    planner must not caption as the entity."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if w.lower() not in _LADDER_NOISE]
    rungs: list[tuple[str, bool]] = [(query, True)]
    if len(words) >= 2:
        head = words[-1]
        rungs.append((head, False))
    for w in reversed(words):
        hyper = HYPERNYMS.get(w.lower())
        if hyper:
            rungs.append((hyper, False))
            break
    else:
        if len(words) >= 1 and not any(w[:1].isupper() for w in words):
            rungs.append(("workshop", False))
    seen: set[str] = set()
    return [(q, e) for q, e in rungs if not (q.lower() in seen or seen.add(q.lower()))]


def _stem(word: str) -> str:
    w = word.lower()
    return w[:-1] if w.endswith("s") and len(w) > 4 else w


# Titles/descriptions of this kind are never useful B-roll (logos, clip art, maps, ...).
_WEAK_MEDIA = re.compile(
    r"\b(logo|icon|emblem|seal|map|clip ?art|cartoon|svg|vector|illustration|diagram of|"
    r"poster|badge|flag|infographic)\b", re.IGNORECASE)
_QUERY_STOP = frozenset(["the", "and", "for", "with", "from", "into", "over", "this", "that", "are", "was", "were", "has", "have"])


def relevant(query: str, title: Any = None, *others: Any, strong_names: bool = False,
             relaxed: bool = False) -> bool:
    """Is a candidate (title, then tags/description) really about the query?

    1-2 word queries need every word in the title; with two words, tags/description (when the
    provider gives any) must also mention at least one of them. Longer queries need two matching
    words overall. Logos, icons, maps, clip art and similar are rejected unless the query asks
    for them. Full-text search returns loosely related items; this keeps off-topic footage out."""
    extra = " ".join(str(t) for t in others if t).lower()
    head = str(title or "").lower()
    hay = f"{head} {extra}"
    weak = _WEAK_MEDIA.search(hay)
    if weak and not re.search(rf"\b{re.escape(weak.group(1).lower())}", query.lower()):
        return False
    words = {_stem(w) for w in re.findall(r"[A-Za-z0-9]{3,}", query)
             if w.lower() not in _QUERY_STOP}
    if not words:
        return False

    def tokens(t: str) -> set[str]:
        return {_stem(w) for w in re.findall(r"[a-z0-9]{3,}", t)}

    # Entity queries only: a Capitalised name from the query (Tokyo, Eiffel) in the title is a
    # strong match even if the second word is absent.
    names = {_stem(w) for w in re.findall(r"\b[A-Z][a-z]{2,}\b", query)
             if w.lower() not in _QUERY_STOP}
    if strong_names and names and names <= tokens(head):
        return True
    if relaxed:  # thematic B-roll: any query word in title/tags/description
        return bool(words & tokens(hay))
    if len(words) <= 2:
        if not words <= tokens(head):
            return False
        return len(words) < 2 or not extra.strip() or bool(words & tokens(extra))
    return len(words & tokens(hay)) >= 2


def _wiki_license(meta: dict[str, Any]) -> str:
    """License short name from Wikimedia extmetadata ('' if absent)."""
    for key in ("LicenseShortName", "License"):
        if isinstance(meta.get(key), dict):
            return str(meta[key].get("value", "")).strip()
    return ""


def _now() -> str:
    return datetime.now(UTC).isoformat()
