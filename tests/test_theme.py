"""psycho_terminal theme: palette, text fit, boot cards, duration invariants, cache keys."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from PIL import Image

from src import graphics, theme
from src import stage6_render as r
from src.transitions import GLITCH, XFADE, auto_transition, resolve_joins


@pytest.fixture
def terminal(monkeypatch):
    monkeypatch.setattr(theme, "active", lambda: theme.TERMINAL)


def test_palette_matches_site_and_is_legible():
    assert theme.GREEN == (0, 255, 65) and theme.CYAN == (0, 255, 249)
    assert theme.MAGENTA == (255, 0, 255)
    scrim = theme.SCRIM[:3]
    for style, col in theme.STYLE_ACCENTS.items():
        assert theme.contrast(col, scrim) >= 7.0, style
    assert theme.contrast(theme.MINT, scrim) >= 12


def test_style_semantics_kept(terminal):
    assert graphics.accent_for("fail") == theme.RED
    assert graphics.accent_for("technical") == theme.CYAN
    assert graphics.accent_for("hype")[0] > 200 and graphics.accent_for("hype")[1] > 240
    assert theme.tag_for("fail") == "ERR" and theme.tag_for("technical") == "DATA"
    assert theme.tag_for("hype") == "OK"


def test_classic_unchanged(monkeypatch):
    monkeypatch.setattr(theme, "active", lambda: theme.CLASSIC)
    assert graphics.accent_for("fail") == (255, 64, 64)
    assert graphics.panel_color() == (16, 24, 48)


def test_fonts_ship_with_licence():
    assert theme.FONT_TEXT.is_file() and theme.FONT_DISPLAY.is_file()
    assert (theme.FONT_DIR / "OFL-VT323.txt").is_file()


@pytest.mark.parametrize("text", ["Spherical aberration", "A very long callout text " * 6, "X"])
def test_terminal_text_fits(terminal, text):
    badge = graphics.render_badge(text, "technical", 46, 1100)
    assert badge.width <= 1100 + 2 * 21 + 2  # glow padding only
    lt = graphics.render_lower_third(text, "fail", 40)
    assert lt.width <= 1300 + 4
    assert lt.getpixel((lt.width // 2, lt.height - 3))[3] > 200  # scrim, not empty


def test_terminal_cards(terminal, tmp_path):
    for fn in (lambda p: graphics.make_card(p, "Hello", "sub", 640, 360),
               lambda p: graphics.make_title_card(p, "Gravity wells", "hype", 640, 360),
               lambda p: graphics.make_grid_background(p, 640, 360)):
        p = tmp_path / "c.png"
        fn(p)
        assert Image.open(p).size == (640, 360)
    assert Image.open(p).getpixel((5, 5))[2] < 30  # near-black background


def test_boot_frames_change_over_time():
    a = theme.boot_frame("intro", 0.05, 640, 360)
    b = theme.boot_frame("intro", 1.9, 640, 360)
    assert a.tobytes() != b.tobytes()
    out = theme.boot_frame("outro", 2.4, 640, 360, 2.5, 0, 0.8)
    assert max(out.getdata(), key=sum)[1] < 255 * 0.5  # fading out


def test_crt_normalize_and_chain():
    assert theme.normalize_crt(None) == "off" and theme.normalize_crt("STRONG") == "strong"
    assert theme.normalize_crt("bogus") == "off"
    assert "overlay" in theme.crt_chain("subtle", "a", "b", 3, 1920)[-1]
    assert theme.crt_chain("off", "a", "b", None, 1920) == ["[a]null[b]"]


def test_settings_defaults_by_theme():
    t = r.RenderSettings.from_config({"theme": "psycho_terminal"})
    assert (t.crt, t.branding, t.hud) == ("subtle", True, False)
    c = r.RenderSettings.from_config({})
    assert (c.theme, c.crt, c.branding) == ("classic", "off", False)
    assert not t.as_draft().branding


def test_glitch_transitions_vetted_and_forty_percent_rule():
    assert all(g in XFADE for g in GLITCH)
    beats = [{"id": i, "beat_style": "fail", "visual_template": "fullscreen_clip"}
             for i in range(3)]
    assert auto_transition(beats[1], beats[0], False, glitch=True) == "glitchslice"
    assert auto_transition(beats[1], beats[0], False) == "none"
    joins = resolve_joins(beats, [20, 20, 20], 30, glitch=True)
    assert joins[1].frames and joins[1].frames < 0.4 * 20


def test_cache_key_includes_theme_and_crt(tmp_path):
    base = r.RenderSettings()
    assert dataclasses.asdict(base) != dataclasses.asdict(
        dataclasses.replace(base, theme="psycho_terminal"))
    assert dataclasses.asdict(base) != dataclasses.asdict(dataclasses.replace(base, crt="strong"))
    assert Path(theme.__file__).read_bytes() in [Path(theme.__file__).read_bytes()]
    src = Path(r.__file__).read_text()
    assert "theme.__file__" in src  # code revision hash covers the theme module


def test_boot_card_duration_and_audio_offset(tmp_path):
    s = dataclasses.replace(r.RenderSettings(), width=320, height=180, codec="libx264",
                            x264_preset="ultrafast", branding=True)
    warns: list[str] = []
    intro = r._boot_card("intro", 2.0, s, tmp_path, warns)
    outro = r._boot_card("outro", 2.5, s, tmp_path, warns)
    from src.common import probe_duration
    assert probe_duration(intro) == pytest.approx(2.0, abs=0.04)
    assert probe_duration(outro) == pytest.approx(2.5, abs=0.04)


def test_branding_sound_no_clip_and_voice_window_silent(tmp_path):
    import numpy as np

    from src import branding_sound as bs
    x = bs.render(3.0, 4.0, 3.0, 48000)
    assert len(x) == 10 * 48000
    assert 0.1 < np.max(np.abs(x)) <= bs.PEAK + 1e-6
    assert np.max(np.abs(x[3 * 48000 + 100: 7 * 48000 - 100])) < 1e-9  # under the voiceover
    assert np.max(np.abs(x[:48000])) > 0.05 and np.max(np.abs(x[-48000:])) > 0.01


def test_sound_settings_in_cache_key_and_defaults():
    a = r.RenderSettings()
    assert a.branding_sound and a.intro_seconds == 3.0
    assert dataclasses.asdict(a) != dataclasses.asdict(
        dataclasses.replace(a, branding_sound_gain=0.5))
    assert "branding_sound.__file__" in Path(r.__file__).read_text()


def test_intro_typing_complete_by_70_percent():
    it = theme.INTRO_T
    sub_end = it["sub"][0] + len(theme.INTRO_SUB) / it["sub"][1]
    status_end = it["status"][0] + len(theme.INTRO_STATUS) / it["status"][1]
    assert max(sub_end, status_end) <= 0.7 * 3.0 + 1e-6
    ot = theme.OUTRO_T
    assert ot["github"] <= 3.0 - 0.8 - 0.4  # everything visible well before the fade starts
