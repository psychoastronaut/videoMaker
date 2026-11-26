"""Synthesized, licence-free sound design for the boot intro/outro (numpy only, no downloads).

Intro: CRT power-on thunk + fading hum, soft key ticks while lines type, a rising lock chirp when
``SYSTEM ONLINE`` completes and a short glitch burst into the first beat.  Outro: ticks while the
lines type and a descending power-down tone.  Peaks stay around -12 dBFS (well under the voice).
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from src import theme

PEAK = 0.25  # linear peak of the whole track (about -12 dBFS), before ``branding_sound_gain``


def _env(n: int, attack: float, decay: float, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / max(decay, 1e-4))


def _add(buf: np.ndarray, sig: np.ndarray, at: float, sr: int, gain: float = 1.0) -> None:
    i = int(at * sr)
    if i >= len(buf) or i < 0:
        return
    sig = sig[: len(buf) - i]
    buf[i:i + len(sig)] += sig * gain


def thunk(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    low = np.sin(2 * np.pi * (70 - 35 * t / 0.45) * t) * _env(n, 0.003, 0.12, sr)
    click = rng.standard_normal(n) * _env(n, 0.0005, 0.01, sr) * 0.35
    return low + click


def hum(sr: int, dur: float) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    sig = (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
           + 0.15 * np.sin(2 * np.pi * 150 * t))
    return sig * np.minimum(t / 0.3, 1.0) * np.clip((dur - t) / 0.5, 0.0, 1.0)


def tick(sr: int, rng: np.random.Generator, f: float = 1800.0) -> np.ndarray:
    n = int(0.018 * sr)
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * f * t) * 0.6 + rng.standard_normal(n) * 0.25) * _env(
        n, 0.0004, 0.004, sr)


def chirp(sr: int, f0: float, f1: float, dur: float) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f0 + (f1 - f0) * (t / dur) ** 1.5
    phase = 2 * np.pi * np.cumsum(f) / sr
    env = np.minimum(t / 0.01, 1.0) * np.clip((dur - t) / 0.05, 0.0, 1.0)
    return (np.sin(phase) + 0.3 * np.sin(2 * phase)) * env


def glitch_burst(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    gate = (np.sin(2 * np.pi * 37 * t) + np.sin(2 * np.pi * 91 * t) > 0.2).astype(float)
    crush = np.round(noise * 4) / 4  # coarse quantisation = digital texture
    return crush * gate * np.clip(t / dur * 2, 0, 1) * np.clip((dur - t) / 0.03, 0, 1)


def _typing_ticks(buf, sr, rng, text: str, start: float, cps: float, every: int = 2,
                  gain: float = 0.5) -> None:
    for i in range(0, len(text), every):
        _add(buf, tick(sr, rng, 1500 + 300 * rng.random()), start + i / cps, sr,
             gain * (0.7 + 0.3 * rng.random()))


def render(intro: float, body: float, outro: float, sr: int = 48000, seed: int = 7) -> np.ndarray:
    """Mono track of ``intro + body + outro`` seconds; silence over the voiceover."""
    rng = np.random.default_rng(seed)
    total = intro + body + outro
    buf = np.zeros(round(total * sr))
    it, ot = theme.INTRO_T, theme.OUTRO_T
    # intro
    _add(buf, hum(sr, intro - 0.15), 0.0, sr, 0.10)
    _add(buf, thunk(sr, rng), 0.0, sr, 0.9)
    _typing_ticks(buf, sr, rng, theme.INTRO_SUB, *it["sub"])
    sub_end = it["sub"][0] + len(theme.INTRO_SUB) / it["sub"][1]
    _add(buf, chirp(sr, 500, 2200, 0.22), sub_end, sr, 0.45)
    _typing_ticks(buf, sr, rng, theme.INTRO_STATUS, *it["status"], every=3, gain=0.35)
    g = it["glitch_out"]
    _add(buf, glitch_burst(sr, g, rng), intro - g, sr, 0.35)
    # outro
    o = intro + body
    _add(buf, hum(sr, outro - 0.2), o, sr, 0.07)
    for key, text in (("l1", theme.OUTRO_L1), ("l2", theme.OUTRO_L2), ("l3", theme.OUTRO_L3)):
        _typing_ticks(buf, sr, rng, text, o + ot[key][0], ot[key][1], every=3, gain=0.4)
    _add(buf, chirp(sr, 700, 1500, 0.12), o + ot["github"], sr, 0.35)
    n = int(1.0 * sr)
    t = np.arange(n) / sr
    f = 900 * np.exp(-3.4 * t) + 40
    down = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.clip((1.0 - t) / 0.3, 0, 1) \
        * np.minimum(t / 0.02, 1)
    _add(buf, down, o + outro - 1.05, sr, 0.5)
    _add(buf, thunk(sr, rng), o + outro - 0.5, sr, 0.5)
    peak = float(np.max(np.abs(buf))) or 1.0
    return buf / peak * PEAK


def write_wav(path: Path, intro: float, body: float, outro: float, sr: int = 48000) -> Path:
    mono = render(intro, body, outro, sr)
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2")
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(stereo.tobytes())
    return path
