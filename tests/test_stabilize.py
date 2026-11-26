from pathlib import Path

from src import stabilize
from src.media import AssetResolver


def test_config_defaults_and_override() -> None:
    cfg = stabilize.stabilize_config({"stabilize": {"enabled": False, "smoothing": 9}})
    assert cfg["enabled"] is False and cfg["smoothing"] == 9 and cfg["threshold"] > 0


def test_disabled_does_nothing(tmp_path: Path) -> None:
    idx = {"assets": [{"id": "clip_a", "asset_type": "clip", "filepath": str(tmp_path / "x.mp4")}]}
    out = stabilize.apply_to_index(idx, tmp_path, {"stabilize": {"enabled": False}})
    assert not out["stabilized"] and "stabilized_path" not in idx["assets"][0]


def test_cache_key_changes_with_settings(tmp_path: Path) -> None:
    f = tmp_path / "c.mp4"
    f.write_bytes(b"abc" * 100)
    a = stabilize.cached_path(f, tmp_path, stabilize.stabilize_config({}))
    b = stabilize.cached_path(f, tmp_path, stabilize.stabilize_config({"stabilize": {"smoothing": 3}}))
    assert a != b and a.parent == tmp_path


def test_resolver_prefers_stabilized_path(tmp_path: Path) -> None:
    import json
    import subprocess

    src, st = tmp_path / "src.mp4", tmp_path / "st.mp4"
    for p in (src, st):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc=size=64x64:rate=10:duration=1", str(p)], check=True)
    (tmp_path / "local_assets.json").write_text(json.dumps({"assets": [
        {"id": "clip_a", "asset_type": "clip", "filepath": str(src), "stabilized_path": str(st)}]}))
    media = AssetResolver(tmp_path).local_asset("clip_a")
    assert media is not None and media.path == str(st)
