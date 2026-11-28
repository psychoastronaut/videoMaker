"""Stage 5b music: synthesizer, sources/licences, ducked mix (measured with ffmpeg), QA."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from src import music, music_synth
from src.common import episode_dir, read_json
from tests.conftest import write_wav


def _stereo_wav(path: Path, data: np.ndarray, sr: int = 48000) -> None:
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1] if data.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _rms_db(path: Path, start: float, dur: float) -> float:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-t", str(dur),
                        "-i", str(path), "-af", "astats=metadata=0", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    import re

    vals = re.findall(r"RMS level dB:\s*(-?[0-9.inf]+)", r.stderr)
    return float(vals[-1])  # "Overall" block comes last


# --------------------------------------------------------------------------- synth


def test_synth_is_deterministic_and_seeded() -> None:
    a = music_synth.synth(12, 5, "tense")
    b = music_synth.synth(12, 5, "tense")
    c = music_synth.synth(12, 6, "tense")
    assert a.shape == (12 * 48000, 2) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert float(np.max(np.abs(a))) <= 0.71


def test_synth_structure_fade_and_moods() -> None:
    x = music_synth.synth(30, 11, "calm")
    head, mid, tail = (float(np.sqrt(np.mean(x[s:e] ** 2)))
                       for s, e in ((0, 24000), (14 * 48000, 16 * 48000), (-24000, None)))
    assert head < mid * 0.3 and tail < mid * 0.3  # fade in / fade out
    for mood in music_synth.MOODS:
        spec = music_synth.make_spec(3, mood)
        assert 80 <= spec.bpm <= 100 and spec.scale[2] == 3  # minor third: minor modes
    assert not np.array_equal(music_synth.synth(8, 3, "tense"), music_synth.synth(8, 3, "upbeat"))


def test_auto_mood_from_beat_styles() -> None:
    assert music.auto_mood(None) == "tense"
    hype = {"beats": [{"beat_style": "hype"}] * 4 + [{"beat_style": "normal"}] * 6}
    assert music.auto_mood(hype) == "upbeat"
    plain = {"beats": [{"beat_style": "normal"}] * 10}
    assert music.auto_mood(plain) == "calm"


# --------------------------------------------------------------------------- licences


def test_license_policy_never_accepts_cc_by() -> None:
    for ok in ("CC0", "cc0 1.0", "Public Domain", "pdm", "Public Domain Mark 1.0"):
        assert music.is_free_license(ok)
    for bad in ("CC BY 4.0", "by", "CC-BY-SA 3.0", "by-sa", "cc by-nc", None, "", "all rights"):
        assert not music.is_free_license(bad)


class _Resp:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.status_code = 200
        self._r = results

    def json(self) -> dict[str, Any]:
        return {"results": self._r}


class _FakeClient:
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any] | None = None) -> _Resp:
        self.params.append(params or {})
        lic = (params or {}).get("license")
        mk = lambda i, lic_, dur: {
            "id": f"{lic_}{i}", "license": lic_, "duration": dur, "title": "Dark ambient pad",
            "tags": [{"name": "ambient"}], "url": f"https://x/{lic_}{i}.mp3",
            "creator": "someone", "license_url": "u", "category": None}
        return _Resp([mk(1, lic, 200_000), mk(2, lic, 20_000), mk(3, "by", 300_000)])


def test_openverse_filters_license_and_duration() -> None:
    client = _FakeClient()
    cands = music.openverse_candidates(client, "tense", music.MusicCfg())  # type: ignore[arg-type]
    assert cands and all(c["license"] in ("CC0", "PDM") for c in cands)
    assert all(c["duration"] >= 60 for c in cands)
    assert not any(c["id"].startswith("by") for c in cands)
    assert {p["license"] for p in client.params} == {"cc0", "pdm"}
    strict = _FakeClient()
    music.openverse_candidates(strict, "tense", music.MusicCfg(strict_cc0_only=True))  # type: ignore[arg-type]
    assert {p["license"] for p in strict.params} == {"cc0"}


def _episode(tmp: Path, name: str = "ep-m", seconds: float = 6.0) -> Path:
    ep = episode_dir(name)
    (ep / "raw").mkdir(parents=True)
    (ep / "work").mkdir()
    (ep / "out").mkdir()
    write_wav(ep / "raw" / "voiceover.wav", seconds)
    return ep


def test_offline_fallback_to_synth_and_audit(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    info = music.run("ep-m", ep / "work")  # offline env from conftest, no episode/library music
    assert info["source"] == "synth" and (ep / "work" / "music.wav").is_file()
    audit = read_json(ep / "out" / "licenses_audit.json")["music"]
    assert audit["source"] == "synth" and "original" in audit["license"]
    meta = read_json(ep / "work" / "music.json")
    assert meta["level_lufs"] == -29.0 and meta["mood"] in music_synth.MOODS
    assert music.stale_reason(ep, ep / "work") is None
    (ep / "raw" / "voiceover.wav").touch()
    assert music.stale_reason(ep, ep / "work") == "voiceover newer than music"


def test_episode_track_uses_sidecar_license_and_strict_refuses(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    folder = ep / "raw" / "music"
    folder.mkdir()
    _stereo_wav(folder / "mine.wav", np.random.default_rng(0).standard_normal((48000 * 4, 2)) * 0.1)
    (folder / "mine.wav.txt").write_text("license: CC0\nauthor: me\n")
    info = music.run("ep-m", ep / "work")
    assert info["source"] == "episode" and info["license"] == "CC0"
    assert read_json(ep / "out" / "licenses_audit.json")["music"]["author"] == "me"
    (folder / "mine.wav.txt").write_text("license: CC BY 4.0\n")
    cfg = music.MusicCfg(strict_cc0_only=True)
    assert music._local_track(folder, "episode", 1, cfg) is None
    assert music._local_track(folder, "episode", 1, music.MusicCfg())["free_license"] is False


def test_loop_crossfade_covers_duration() -> None:
    x = np.ones((1000, 2), np.float32)
    y = music.loop_to(x, 5000, 100)
    assert y.shape == (5000, 2) and 0.7 < float(y.min()) and float(y.max()) < 1.5


# --------------------------------------------------------------------------- mixing / QA


def _speech(pattern: list[tuple[float, bool]], sr: int = 48000) -> np.ndarray:
    parts = []
    for dur, on in pattern:
        t = np.arange(int(dur * sr)) / sr
        parts.append(np.sin(2 * np.pi * 300 * t) * 0.25 if on else np.zeros_like(t))
    return np.concatenate(parts)


def test_short_pauses_do_not_move_the_music_gain() -> None:
    cfg = music.MusicCfg()
    pat = [(0.5, False)] + [x for _ in range(6) for x in ((1.2, True), (0.4 + 0.07 * _, False))]
    act = music.speech_activity(_speech(pat), 48000, cfg.hold)
    g = music.duck_gain_db(act, cfg)
    mid = g[int(2.0 / music.FR): int((len(g) * music.FR - 1.0) / music.FR)]
    assert float(np.ptp(mid)) <= 1.0  # constant (+-0.5 dB) across 0.4-0.8 s gaps
    assert abs(float(mid.mean()) + cfg.duck_db) <= 1.0
    assert music.pumping_db(g, act) <= 1.0 + 1e-6


def test_long_pause_releases_smoothly_and_ducks_again() -> None:
    cfg = music.MusicCfg()
    act = music.speech_activity(_speech([(3, True), (4, False), (3, True)]), 48000, cfg.hold)
    g = music.duck_gain_db(act, cfg)
    at = lambda sec: float(g[int(sec / music.FR)])
    assert abs(at(2.5) + cfg.duck_db) < 0.2 and abs(at(6.9)) < 0.5 + cfg.duck_db  # ducked / rising
    assert at(6.0) > -cfg.duck_db + 1.0  # rose in the pause
    assert at(9.5) < -cfg.duck_db + 0.5  # ducked again under the second phrase
    assert float(np.max(np.abs(np.diff(g)))) < 2.5  # no steps (a step would be 6 dB in 50 ms)


def test_ducked_stem_is_quieter_under_voice_measured_by_ffmpeg(tmp_path: Path) -> None:
    cfg = music.MusicCfg(duck_db=6.0, intro_gain=1.0, attack=0.2, release=1.0)
    sr = 48000
    voice = _speech([(6, True), (6, False)])
    act = music.speech_activity(voice, sr, cfg.hold)
    g = music.duck_gain_db(act, cfg)
    t = np.arange(12 * sr) / sr
    gain = np.interp(t, (np.arange(len(g)) + 0.5) * music.FR, g)
    mus = (np.sin(2 * np.pi * 110 * t) * 0.1 * 10 ** (gain / 20))[:, None].repeat(2, 1)
    _stereo_wav(tmp_path / "d.wav", mus)
    under_voice = _rms_db(tmp_path / "d.wav", 2.0, 3.0)
    in_pause = _rms_db(tmp_path / "d.wav", 9.0, 2.0)
    assert 4.5 <= in_pause - under_voice <= 7.5, (under_voice, in_pause)


def test_qa_music_flags_loud_music(tmp_path: Path) -> None:
    sr = 48000
    t = np.arange(6 * sr) / sr
    v = np.sin(2 * np.pi * 250 * t) * 0.2
    _stereo_wav(tmp_path / "v.wav", np.stack([v, v], 1))
    quiet = np.sin(2 * np.pi * 90 * t) * 0.02
    _stereo_wav(tmp_path / "mq.wav", np.stack([quiet, quiet], 1))
    loud = quiet * 8
    _stereo_wav(tmp_path / "ml.wav", np.stack([loud, loud], 1))
    ok = music.qa_music(tmp_path / "v.wav", tmp_path / "mq.wav")
    bad = music.qa_music(tmp_path / "v.wav", tmp_path / "ml.wav")
    assert ok["passed"] and ok["music_vs_voice_db_p95"] <= -18
    assert not bad["passed"]


def test_mux_with_music_end_to_end(tmp_path: Path) -> None:
    """The real ``_mux`` graph: voice + ducked music in one pass, stems for the QA."""
    from src.common import load_config
    from src.stage6_render import RenderSettings, _mux

    s = RenderSettings.from_config(load_config())
    sr = 48000
    dur = 8.0
    t = np.arange(int(dur * sr)) / sr
    voice = np.where(t < 4, np.sin(2 * np.pi * 300 * t) * 0.3, 0.0)
    _stereo_wav(tmp_path / "vo.wav", np.stack([voice, voice], 1))
    mus = music_synth.synth(dur + 2, 4, "calm")
    _stereo_wav(tmp_path / "music.wav", mus[: int((dur + 2) * sr)] * 0.06)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"color=c=black:s=64x64:r=30:d={dur + 2}", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(tmp_path / "v.mp4")], check=True)
    concat = tmp_path / "concat.txt"
    concat.write_text(f"file '{tmp_path / 'v.mp4'}'\n")
    out = tmp_path / "out.mp4"
    stems = (tmp_path / "sv.wav", tmp_path / "sm.wav")
    cfg = music.MusicCfg()
    _mux(concat, tmp_path / "vo.wav", out, dur, s, None, [], lead=1.0, tail_extra=1.0,
         music_mix=(tmp_path / "music.wav", cfg), stems=stems)
    assert out.is_file() and all(p.is_file() for p in stems)
    rep = music.qa_music(*stems)
    assert rep["passed"] and rep["speech_blocks"] >= 3
