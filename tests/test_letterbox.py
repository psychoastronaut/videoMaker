import numpy as np

from src.letterbox import crop_filter, detect_crop


def _frame(top=135, bot=135, text=True, h=1080, w=1920, content=120):
    f = np.zeros((h, w), np.uint8)
    f[top:h - bot] = content
    f[top:h - bot, ::7] = 60  # textured content
    if text:  # light caption text inside both bars (minority of pixels)
        for y0, y1 in ((40, 90), (h - 95, h - 45)):
            f[y0:y1, 300:1500:3] = 240
    return f


def test_detects_bars_with_text():
    assert detect_crop([_frame() for _ in range(5)]) == [0, 135, 1920, 810]


def test_no_bars():
    assert detect_crop([_frame(0, 0, text=False)] * 5) is None


def test_dark_scene_without_edge_is_not_cropped():
    f = np.zeros((1080, 1920), np.uint8)
    f += np.linspace(0, 20, 1080, dtype=np.uint8)[:, None]  # smooth dark gradient
    assert detect_crop([f] * 5) is None


def test_lone_dark_band_ignored():
    assert detect_crop([_frame(200, 0, text=False)] * 5) is None


def test_majority_vote_tolerates_one_bad_frame():
    frames = [_frame() for _ in range(4)] + [_frame(0, 0, text=False)]
    assert detect_crop(frames) == [0, 135, 1920, 810]


def test_crop_filter_scales_to_actual_file():
    assert crop_filter([0, 135, 1920, 810], (960, 540), (1920, 1080)) == "crop=960:404:0:68"


def test_resolver_returns_cropped_copy(tmp_path):
    import json
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return
    from src.media import AssetResolver

    img = tmp_path / "bars.png"
    from PIL import Image
    Image.fromarray(_frame(text=False)).save(img)
    (tmp_path / "local_assets.json").write_text(json.dumps({"assets": [{
        "id": "image_bars", "filepath": str(img), "asset_type": "image", "width": 1920,
        "height": 1080, "crop_box": [0, 135, 1920, 810]}]}))
    m = AssetResolver(tmp_path).local_asset("image_bars")
    assert m is not None and m.path != str(img)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0:s=x", m.path], capture_output=True, text=True, check=False).stdout
    assert out.strip() == "1920x810"
