"""Noise analysis (tonal peaks, notches), two-stage denoise chain, verification and A/B."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from src import audio_analysis as aa
from src import audio_clean
from src.common import episode_dir

SR = 22050


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.normal(size=n))
    spec /= np.sqrt(np.maximum(np.arange(spec.size), 1))
    y = np.fft.irfft(spec, n)
    return y / np.std(y)


def _synth(path: Path, tone_hz: float = 300.0, seconds: int = 20) -> None:
    """Voice-like bursts (140 Hz harmonics, 1.6 s on / 1.4 s off) + pink noise + a tone."""
    rng = np.random.default_rng(1)
    t = np.arange(SR * seconds) / SR
    env = ((t % 3) < 1.6).astype(float)
    voice = sum(np.sin(2 * np.pi * 137 * k * t) / k for k in range(1, 12)) * 0.08 * env
    x = voice + 0.006 * _pink(t.size, rng) + 0.006 * np.sin(2 * np.pi * tone_hz * t)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def test_detects_tone_of_any_frequency(tmp_path):
    for hz in (300.0, 437.0):
        _synth(tmp_path / "a.wav", hz)
        x = aa.load_mono(tmp_path / "a.wav")
        f, p, noise = aa.noise_spectrum(x)
        assert noise.any()
        peaks = aa.find_tonal_peaks(f, p)
        assert peaks and abs(peaks[0]["freq_hz"] - hz) < 4
        assert peaks[0]["prominence_db"] > 12


def test_notch_filters_format():
    n = aa.notch_filters([{"freq_hz": 298.5, "prominence_db": 17.0, "width_hz": 11.0}])
    assert n[0].startswith("equalizer=f=298.5:t=h:w=") and ":g=-" in n[0]


def test_chain_order_and_modes():
    s = audio_clean.settings({"audio": {"rnnoise_model": __file__}})
    prof = {"notches": ["equalizer=f=300:t=h:w=9:g=-20"], "peaks": [], "gate_db": -60.0,
            "floor_db": -65.0}
    f = audio_clean.build_filter(s, prof=prof)
    order = ["highpass", "equalizer=f=300", "afftdn", "arnndn", "agate", "equalizer=f=250",
             "acompressor", "silenceremove", "loudnorm"]
    pos = [f.index(k) for k in order]
    assert pos == sorted(pos)
    assert "makeup=1" in f  # no gain before loudnorm
    off = audio_clean.build_filter({**s, "denoise": "off"}, prof=prof)
    assert "afftdn" not in off and "arnndn" not in off
    assert "arnndn" not in audio_clean.build_filter({**s, "denoise": "afftdn"}, prof=prof)
    assert "afftdn" not in audio_clean.build_filter({**s, "denoise": "rnnoise"}, prof=prof)
    with pytest.raises(ValueError):
        audio_clean.settings({"audio": {"denoise": "deepfilternet"}})


def test_rnnoise_mode_without_model_degrades_to_afftdn():
    s = audio_clean.settings({"audio": {"denoise": "rnnoise"}})
    assert "afftdn" in audio_clean.build_filter(s)


def test_end_to_end_tone_and_speech(hermetic_env):
    ep = episode_dir("dn")
    (ep / "raw").mkdir(parents=True)
    _synth(ep / "raw" / "voiceover.wav")
    (ep / "work").mkdir()
    # afftdn only: RNNoise (trained on real speech) would also remove pure synthetic sines
    s = audio_clean.settings({"audio": {"denoise": "afftdn"}})
    rep = audio_clean.process(ep / "raw" / "voiceover.wav", ep / "work" / "voiceover_clean.wav", s)
    (ep / "work" / "audio_report.json").write_text(json.dumps(rep))
    print(audio_clean.format_report(rep))
    na, v = rep["noise_analysis"], rep["verification"]
    assert any(abs(q["freq_hz"] - 300) < 4 for q in na["tonal_peaks"])
    assert na["notch_filters"]
    t = v["tones"][next(k for k in v["tones"] if abs(float(k) - 300) < 4)]
    assert t["after_rel_voice_db"] < t["before_rel_voice_db"] - 10
    assert v["floor_improvement_rel_voice_db"] > 5
    assert v["speech_band_change_db"] > -1.0
    assert v["after"]["snr_db"] > v["before"]["snr_db"] + 5
    assert json.loads((ep / "work" / "audio_report.json").read_text())["verification"]
    assert abs(rep["after"]["lufs"] - (-16)) < 1.0


def test_ab_outputs(hermetic_env):
    ep = episode_dir("ab")
    (ep / "raw").mkdir(parents=True)
    _synth(ep / "raw" / "voiceover.wav")
    audio_clean.run("ab", ep / "work", ab=True)
    d = ep / "work" / "audio_ab"
    for n in ("original", "atual", "novo"):
        assert (d / f"{n}.wav").stat().st_size > 10_000
        assert (d / f"spectrogram_{n}.png").is_file()
