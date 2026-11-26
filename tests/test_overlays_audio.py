import subprocess

from src import overlays
from src import stage6_render as s6
from src.schemas import SceneBeat


def test_new_beat_fields_backward_compatible():
    b = SceneBeat.model_validate({"id": 1, "start_time": 0, "end_time": 2})
    assert b.clip_audio is None and b.overlay_mode is None and b.highlights is None
    b = SceneBeat.model_validate({"clip_audio": "DUCK", "overlay_mode": "inset",
                                  "highlights": [{"region": "center"}, "junk"]})
    assert (b.clip_audio, b.overlay_mode, len(b.highlights)) == ("duck", "pip", 1)


def test_highlight_parsing_and_boxes():
    h = overlays.parse_highlight("0.5,0.3,0.2,0.1 | Latency link | 1.5")
    assert h == {"shape": "box", "x": 0.5, "y": 0.3, "w": 0.2, "h": 0.1,
                 "label": "Latency link", "start": 1.5}
    assert overlays.parse_highlight("top-left")["region"] == "top-left"
    assert overlays.parse_highlight("nonsense") is None
    assert overlays.resolve_box(h, (0, 100, 1000, 500), 1920) == (500, 250, 200, 50)
    assert overlays.resolve_box({"x": 960, "y": 540, "w": 96, "h": 54}, (0, 0, 960, 540), 960) \
        == (480, 270, 48, 27)


def test_pip_geometry_inside_safe_margin():
    w, h, x, y, b = overlays.pip_geometry(1920, 1080, 16 / 9, "br")
    assert x + w + 2 * b <= 1920 - 96 + 1 and y + h + 2 * b <= 1080 - 54 + 1


def test_clip_audio_filter_runs(tmp_path):
    def gen(name, f, d):
        p = tmp_path / name
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"sine=f={f}:d={d}", str(p)], check=True)
        return p

    voice, clip = gen("v.wav", 200, 4), gen("c.wav", 600, 2)
    for duck in (True, False):
        ev = [s6.ClipAudio(str(clip), 1.0, 2.0, 0.0, 1.0, duck)]
        out = tmp_path / f"mix{duck}.wav"
        r = s6._mix_clip_audio(voice, ev, out, 4.0, s6.RenderSettings())
        assert r == out and out.stat().st_size > 0
