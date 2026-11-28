"""Voice chain: filter building, noise estimate and an end-to-end synthetic run."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from src import audio_clean
from src.common import episode_dir, voice_path


def _synth(path: Path) -> None:
    """Speech-like bursts (harmonic tone, 150 Hz) + 2 s gaps + white noise + 60 Hz hum."""
    sr = 22050
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * 12)) / sr
    env = ((t % 4) < 2).astype(float)  # 2 s voice, 2 s silence
    voice = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in (1, 2, 3, 4)) * 0.12 * env
    x = voice + rng.normal(0, 0.008, t.size) + 0.01 * np.sin(2 * np.pi * 60 * t)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    import wave
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def test_presets_build_filters():
    for name in audio_clean.PRESETS:
        s = audio_clean.settings({"audio": {"preset": name}})
        f = audio_clean.build_filter(s)
        for part in ("highpass", "afftdn", "deesser", "acompressor", "silenceremove",
                     "alimiter", "loudnorm=I=-16"):
            assert part in f
    with pytest.raises(ValueError):
        audio_clean.settings({"audio": {"preset": "nope"}})


def test_rnnoise_only_if_model_exists(tmp_path):
    s = audio_clean.settings({"audio": {"rnnoise_model": str(tmp_path / "x.rnnn")}})
    assert "arnndn" not in audio_clean.build_filter(s)
    (tmp_path / "x.rnnn").write_text("m")
    assert "arnndn" in audio_clean.build_filter(s)


def test_noise_floor_db():
    x = np.concatenate([np.full(16000, 0.5), np.full(16000, 0.001)]).astype(np.float32)
    assert audio_clean.noise_floor_db(x) < -55


def test_auto_enabled_and_voice_path(hermetic_env):
    ep = episode_dir("e1")
    (ep / "raw").mkdir(parents=True)
    assert not audio_clean.is_enabled("e1")
    (ep / "raw" / "voiceover.wav").write_bytes(b"x")
    assert audio_clean.is_enabled("e1")
    assert voice_path(ep) == ep / "raw" / "voiceover.wav"
    (ep / "work").mkdir()
    (ep / "work" / "voiceover_clean.wav").write_bytes(b"x")
    assert voice_path(ep) == ep / "work" / "voiceover_clean.wav"


def test_end_to_end(hermetic_env):
    ep = episode_dir("e2")
    (ep / "raw").mkdir(parents=True)
    _synth(ep / "raw" / "voiceover.wav")
    rep = audio_clean.run("e2", ep / "work", report=True)
    assert (ep / "raw" / "voiceover.orig.wav").is_file()
    assert (ep / "work" / "audio_report.json").is_file()
    assert abs(rep["after"]["lufs"] - (-16)) < 1.0
    assert rep["after"]["true_peak"] <= -1.0
    assert rep["noise_reduction_db"] > 6
    assert rep["removed_seconds"] > 1.0
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
                          "-of", "csv=p=0", str(ep / "work" / "voiceover_clean.wav")],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "48000"


def test_natural_is_default_and_gentle_on_timbre():
    s = audio_clean.settings({})
    assert s["preset"] == "natural"
    f = audio_clean.build_filter(s)
    assert "f=250" not in f and "highshelf=f=8000" in f  # no mud cut, air restored
    assert "ratio=2" in f
    # voice-zone notches are shallow, high-Q; hum below the voice keeps the full depth
    peaks = [{"freq_hz": 298.5, "prominence_db": 17.4, "width_hz": 11.6},
             {"freq_hz": 58.9, "prominence_db": 14.3, "width_hz": 10.2},
             {"freq_hz": 500.0, "prominence_db": 10.0, "width_hz": 8.0}]
    from src import audio_analysis as aa
    out = aa.notch_filters(aa.limit_voice_zone(peaks, None, 9.0))
    assert len(out) == 2 and "g=-9.0" in out[0] and "g=-20.3" in out[1]


def test_carve_cuts_voice_band_only():
    import numpy as np

    from src import music
    sr = 48000
    t = np.arange(sr) / sr
    x = np.stack([np.sin(2 * np.pi * f * t) for f in (80, 1000)], 1).astype(np.float32)
    y = music.carve(x, music.MusicCfg(), sr)
    db = 20 * np.log10(np.sqrt((y[sr // 4:-sr // 4] ** 2).mean(0)) / np.sqrt((x[sr // 4:-sr // 4] ** 2).mean(0)))
    assert abs(db[0]) < 1.0 and -4.6 < db[1] < -3.4
