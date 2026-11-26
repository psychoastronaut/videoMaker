"""Upscaling: sizing/skip rules, tiled feathering, index integration, resolver preference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src import upscale
from src.media import AssetResolver

CFG = dict(upscale.DEFAULTS)


class FakeUpscaler(upscale.Upscaler):
    """Nearest-neighbour 4x instead of the network (no torch / weights needed)."""

    def __init__(self, cfg=None, tile=64):
        super().__init__(cfg or CFG)
        self.tile = tile
        self.device = "cpu"

    def _run_tile(self, tile):
        return np.repeat(np.repeat(tile, 4, axis=0), 4, axis=1).astype(np.float32) / 255.0


def _photo(path: Path, size=(300, 200)) -> Path:
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def test_target_size_caps():
    assert upscale.target_size(600, 400, CFG) == (1200, 800)
    assert upscale.target_size(1200, 800, CFG) == (1920, 1280)
    assert upscale.target_size(1920, 1080, CFG) == (1920, 1080)


def test_skip_rules(tmp_path):
    photo = _photo(tmp_path / "a.png")
    assert upscale.skip_reason(photo, "image", CFG) is None
    assert upscale.skip_reason(photo, "schematic", CFG) == "schematic"
    big = _photo(tmp_path / "big.png", (1500, 100))
    assert "already" in upscale.skip_reason(big, "image", CFG)
    flat = tmp_path / "flat.png"
    Image.new("RGB", (400, 300), (10, 20, 200)).save(flat)
    assert upscale.skip_reason(flat, "image", CFG) == "flat graphic"
    assert upscale.skip_reason(flat, "image", {**CFG, "flat_graphics": True}) is None


def test_tiled_feathering_matches_untiled():
    rgb = np.random.default_rng(0).integers(0, 255, (150, 170, 3), dtype=np.uint8)
    tiled = FakeUpscaler(tile=64)._enhance(rgb, 64)
    whole = FakeUpscaler(tile=512)._enhance(rgb, 512)
    assert tiled.shape == whole.shape == (600, 680, 3)
    assert np.allclose(tiled, whole, atol=1e-5)


def test_alpha_and_exif_orientation(tmp_path):
    src = tmp_path / "t.png"
    arr = np.zeros((80, 120, 4), np.uint8)
    arr[..., :3] = 200
    arr[:40, :, 3] = 255
    Image.fromarray(arr, "RGBA").save(src)
    eng = FakeUpscaler(dict(CFG, blend=1.0))
    info = eng.upscale(src, tmp_path / "o.png")
    out = Image.open(tmp_path / "o.png")
    assert out.mode == "RGBA" and out.size == (info["width"], info["height"]) == (240, 160)
    assert out.getchannel("A").getpixel((10, 10)) == 255
    assert out.getchannel("A").getpixel((10, 150)) == 0
    exif = Image.Exif()
    exif[274] = 6  # rotated 90 degrees
    rot = tmp_path / "r.jpg"
    Image.new("RGB", (300, 200), (90, 90, 90)).save(rot, exif=exif)
    eng.upscale(rot, tmp_path / "r_up.png")
    assert Image.open(tmp_path / "r_up.png").size == (400, 600)  # portrait after transpose


def test_apply_to_index_and_resolver(tmp_path, monkeypatch):
    src = _photo(tmp_path / "p.png")
    work = tmp_path / "work"
    work.mkdir()
    index = {"assets": [{"id": "image_p", "filepath": str(src), "asset_type": "image",
                         "filename": "p.png", "width": 300, "height": 200}]}
    monkeypatch.setattr(upscale, "Upscaler", lambda cfg: _Ctx(FakeUpscaler(cfg)))
    log: list[str] = []
    summary = upscale.apply_to_index(index, work, {"upscale": {"enabled": True}}, log=log.append)
    a = index["assets"][0]
    assert len(summary["upscaled"]) == 1
    assert Path(a["upscaled_path"]).is_file() and a["upscaled_from"] == str(src)
    assert (a["width"], a["height"], a["original_width"]) == (600, 400, 300)
    # idempotent: a second run is served from the cache without building an engine
    monkeypatch.setattr(upscale, "Upscaler", lambda cfg: (_ for _ in ()).throw(AssertionError()))
    again = upscale.apply_to_index(index, work, {"upscale": {"enabled": True}}, log=log.append)
    assert again["upscaled"][0]["cached"] is True
    # resolver prefers the upscaled copy and falls back to the original when it is gone
    (work / "local_assets.json").write_text(json.dumps(index))
    assert AssetResolver(work).local_asset("image_p").path == a["upscaled_path"]
    Path(a["upscaled_path"]).unlink()
    assert AssetResolver(work).local_asset("image_p").path == str(src)


class _Ctx:
    def __init__(self, eng):
        self.eng = eng

    def __enter__(self):
        return self.eng

    def close(self):
        pass


def test_disabled_and_missing_deps(tmp_path, monkeypatch):
    index = {"assets": [{"id": "x", "filepath": "nope", "asset_type": "image"}]}
    assert upscale.apply_to_index(index, tmp_path, {"upscale": {"enabled": False}})["upscaled"] == []
    monkeypatch.setattr(upscale, "deps_available", lambda: False)
    msgs: list[str] = []
    upscale.apply_to_index(index, tmp_path, {}, log=msgs.append)
    assert msgs and "skipped" in msgs[0]


def test_offline_missing_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOMAKER_OFFLINE", "1")
    try:
        upscale.ensure_model({"model_dir": str(tmp_path)})
        raise AssertionError("expected UpscaleUnavailable")
    except upscale.UpscaleUnavailable:
        pass
