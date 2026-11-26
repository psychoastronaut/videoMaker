from __future__ import annotations

from pathlib import Path

import pytest

from src.stage4_fetch_stock import _WIKI_PD, StockFetcher, _fallback_asset


@pytest.mark.parametrize("lic", ["CC0", "CC0 1.0", "Public domain", "PD-USGov-NASA", "PD-old-70"])
def test_wikimedia_accepts_only_explicit_cc0_pd(lic: str) -> None:
    assert _WIKI_PD.match(lic.lower())


@pytest.mark.parametrize("lic", ["CC BY-SA 4.0", "CC BY 2.0", "Commons", "GFDL", "Attribution",
                                 "Wikimedia Commons license", ""])
def test_wikimedia_rejects_everything_else(lic: str) -> None:
    assert not _WIKI_PD.match(lic.lower())


class _Resp:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def get(self, *_a, **_k) -> _Resp:
        return _Resp(self.payload)


def test_openverse_post_check_rejects_non_cc0(tmp_path: Path, monkeypatch) -> None:
    fetcher = StockFetcher({}, tmp_path, offline=False)
    downloads: list[str] = []
    monkeypatch.setattr(fetcher, "_download",
                        lambda c, url, src: downloads.append(url) or None)
    payload = {"results": [
        {"id": "1", "license": "by-sa", "url": "http://x/1.jpg"},
        {"id": "2", "license": "by", "url": "http://x/2.jpg"},
    ]}
    assert fetcher._openverse(_Client(payload), "q", 0) is None  # type: ignore[arg-type]
    assert downloads == []  # never even downloaded a non-CC0/PDM result


def test_platform_providers_need_key_flag_and_not_strict(tmp_path: Path) -> None:
    names = lambda f: {p.__name__ for p in f._providers(True)}
    default = StockFetcher({"pexels_api_key": "k"}, tmp_path, offline=False)
    assert default.allow_platform and "_pexels_video" in names(default)
    strict = StockFetcher({"pexels_api_key": "k", "strict_cc0_only": True}, tmp_path,
                          offline=False)
    assert not strict.allow_platform
    assert names(strict).isdisjoint({"_pexels", "_pexels_video", "_pixabay", "_pixabay_video"})
    off = StockFetcher({"allow_platform_licenses": False}, tmp_path, offline=False)
    assert not off.allow_platform


def test_provider_without_key_returns_nothing(tmp_path: Path) -> None:
    f = StockFetcher({}, tmp_path, offline=False)
    f.pexels_key = f.pixabay_key = ""
    assert f._pexels(None, "q", 0) is None and f._pexels_video(None, "q", 0) is None  # type: ignore[arg-type]
    assert f._pixabay(None, "q", 0) is None and f._pixabay_video(None, "q", 0) is None  # type: ignore[arg-type]


def test_offline_never_queries(tmp_path: Path) -> None:
    assert StockFetcher({}, tmp_path, offline=True).find("anything", 0) is None


def test_fallback_card_is_audited_and_per_beat(tmp_path: Path) -> None:
    a = _fallback_asset(tmp_path, "robot arm", 3)
    assert a.license_type == "generated-fallback"
    assert a.beat_id == 3
    assert Path(a.local_path).is_file()
    assert a.local_path.endswith("fallback_beat003.png")
