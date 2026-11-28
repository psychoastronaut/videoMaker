"""Procedural background music (numpy/scipy only): 100% original, no licence risk.

Dark, discreet ambient/synthwave for the "psycho_terminal" channel: detuned saw pad, soft
sub-bass, sparse echoing arpeggio, light hats, a small algorithmic reverb. Minor modes, 80-100
BPM, a deterministic result for a given ``(duration, seed, mood)``, sections that add/remove
layers (intro pad -> bass -> arp -> hats -> break ...) and a fade in/out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

MOODS = ("tense", "calm", "upbeat")

# semitone offsets of the natural minor / phrygian / dorian scales
_SCALES = {"tense": (0, 1, 3, 5, 7, 8, 10), "calm": (0, 2, 3, 5, 7, 8, 10),
           "upbeat": (0, 2, 3, 5, 7, 9, 10)}
# chord progressions as scale degrees (0 = tonic), two bars per chord
_PROGS = {"tense": ((0, 5, 3, 6), (0, 1, 5, 6), (0, 3, 1, 4)),
          "calm": ((0, 5, 2, 6), (0, 3, 5, 4), (0, 6, 5, 6)),
          "upbeat": ((0, 5, 2, 6), (0, 6, 5, 4), (0, 3, 6, 4))}
_BPM = {"tense": (86, 94), "calm": (80, 84), "upbeat": (94, 100)}
# layer plan per 8-bar section; cycles after the intro, always ends with "outro"
_SECTIONS = {"tense": ("intro", "bass", "arp", "break", "arp", "full"),
             "calm": ("intro", "bass", "pad", "arp", "pad", "bass"),
             "upbeat": ("intro", "bass", "arp", "full", "break", "full")}
_LAYERS = {"intro": {"pad"}, "pad": {"pad"}, "bass": {"pad", "sub"},
           "arp": {"pad", "sub", "arp"}, "break": {"pad", "arp"},
           "full": {"pad", "sub", "arp", "hat"}, "outro": {"pad", "sub"}}


@dataclass(frozen=True)
class Spec:
    bpm: float
    root: int            # MIDI note of the tonic (octave 1-2)
    scale: tuple[int, ...]
    prog: tuple[int, ...]
    mood: str


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def make_spec(seed: int, mood: str) -> Spec:
    mood = mood if mood in MOODS else "tense"
    rng = np.random.default_rng(seed)
    lo, hi = _BPM[mood]
    prog = _PROGS[mood][int(rng.integers(len(_PROGS[mood])))]
    return Spec(float(rng.integers(lo, hi + 1)), int(rng.integers(33, 40)),  # A1..D#2
                _SCALES[mood], prog, mood)


def _degree(spec: Spec, deg: int, octave: int = 0) -> int:
    """MIDI note of scale degree ``deg`` (may exceed one octave) above the tonic."""
    n = len(spec.scale)
    return spec.root + spec.scale[deg % n] + 12 * (deg // n + octave)


def _chord(spec: Spec, deg: int) -> list[int]:
    return [_degree(spec, deg + k) for k in (0, 2, 4, 6)]


def _lp(x: np.ndarray, fc: float, sr: int, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, fc, "low", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def _hp(x: np.ndarray, fc: float, sr: int, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, fc, "high", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def _add(buf: np.ndarray, sig: np.ndarray, at: int) -> None:
    if at >= len(buf) or at < 0:
        return
    sig = sig[: len(buf) - at]
    buf[at:at + len(sig)] += sig


def _saw(f: float, n: int, sr: int, phase: float) -> np.ndarray:
    ph = (phase + f * np.arange(n) / sr) % 1.0
    return (2.0 * ph - 1.0).astype(np.float32)


def _adsr(n: int, sr: int, a: float, r: float) -> np.ndarray:
    t = np.arange(n) / sr
    return (np.minimum(t / max(a, 1e-3), 1.0)
            * np.clip((n / sr - t) / max(r, 1e-3), 0.0, 1.0)).astype(np.float32)


def _pad(spec: Spec, chords: list[tuple[int, int, int]], total: int, sr: int,
         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Detuned saw pad, one buffer per channel (opposite detune = stereo width)."""
    out = [np.zeros(total, np.float32), np.zeros(total, np.float32)]
    for start, length, deg in chords:
        n = length + int(1.6 * sr)  # release tail overlaps the next chord
        env = _adsr(n, sr, 1.4, 1.6)
        for midi in _chord(spec, deg)[:3] + [_degree(spec, deg + 2, 1)]:
            f = _hz(midi + 12)
            for ch, sign in ((0, 1), (1, -1)):
                voice = np.zeros(n, np.float32)
                for cents in (-9, 0, 9):
                    voice += _saw(f * 2 ** (sign * cents / 1200), n, sr, float(rng.random()))
                _add(out[ch], voice * env * 0.2, start)
    pad = [_lp(_hp(c, 90, sr, 1), 1500, sr) for c in out]
    t = np.arange(total) / sr
    lfo = (0.8 + 0.2 * np.sin(2 * np.pi * 0.07 * t)).astype(np.float32)  # slow swell
    return pad[0] * lfo, pad[1] * lfo


def _sub(spec: Spec, chords: list[tuple[int, int, int]], total: int, sr: int) -> np.ndarray:
    out = np.zeros(total, np.float32)
    for start, length, deg in chords:
        n = length + int(0.6 * sr)
        f = _hz(_degree(spec, deg))
        t = np.arange(n) / sr
        body = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
        _add(out, (body * _adsr(n, sr, 0.25, 0.6)).astype(np.float32) * 0.28, start)
    return _lp(np.tanh(out * 1.6), 220, sr)


def _arp_notes(spec: Spec, chords: list[tuple[int, int, int]], total: int, sr: int,
               rng: np.random.Generator, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros(total, np.float32)
    right = np.zeros(total, np.float32)
    step = int(60.0 / spec.bpm / 2 * sr)
    n = int(0.6 * sr)
    t = np.arange(n) / sr
    dens = 0.4 if spec.mood != "upbeat" else 0.58
    k = 0
    for start, length, deg in chords:
        tones = [_degree(spec, deg + j, 2) for j in (0, 2, 4, 1, 6)]
        for i in range(length // step):
            at = start + i * step
            roll = rng.random()   # draw always so the stream does not depend on `active`
            pick = int(rng.integers(len(tones)))
            vel = 0.5 + 0.5 * float(rng.random())
            if roll > dens or not active[min(at, total - 1)]:
                continue
            f = _hz(tones[pick])
            pluck = ((np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)
                      + 0.15 * _saw(f, n, sr, 0.0)) * np.exp(-t / 0.17)).astype(np.float32)
            _add(left if k % 2 == 0 else right, pluck * vel * 0.2, at)
            k += 1
    return _lp(left, 2600, sr), _lp(right, 2600, sr)


def _hats(spec: Spec, total: int, sr: int, rng: np.random.Generator,
          active: np.ndarray) -> np.ndarray:
    out = np.zeros(total, np.float32)
    beat = int(60.0 / spec.bpm * sr)
    n = int(0.05 * sr)
    env = np.exp(-np.arange(n) / sr / 0.012).astype(np.float32)
    i = 0
    while (at := int((i + 0.5) * beat)) < total:  # off-beat 8ths
        noise = rng.standard_normal(n).astype(np.float32)
        vel = 0.4 + 0.6 * float(rng.random())
        if active[at]:
            _add(out, noise * env * vel * 0.05, at)
        i += 1
    return _hp(out, 7000, sr)


def _delay(x: np.ndarray, sr: int, bpm: float, fb: float = 0.38, taps: int = 4) -> np.ndarray:
    """Dotted-8th feedback delay (echoes only; add to the dry signal)."""
    d = int(60.0 / bpm * 0.75 * sr)
    out = np.zeros_like(x)
    for k in range(1, taps + 1):
        if k * d >= len(x):
            break
        out[k * d:] += _lp(x, 2600, sr, 1)[:len(x) - k * d] * fb ** k
    return out


def _reverb_ir(sr: int, seed: int, secs: float = 2.4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(secs * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-t / (secs / 5.5))
    ir = _lp(ir.astype(np.float32), 4500, sr, 1)
    ir[: int(0.012 * sr)] *= np.linspace(0, 1, int(0.012 * sr))  # soft pre-attack
    return ir / np.sqrt(np.sum(ir ** 2))


def _reverb(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    return signal.oaconvolve(x, ir, mode="full")[: len(x)].astype(np.float32)


def synth(duration: float, seed: int = 7, mood: str = "tense", sr: int = 48000,
          spec: Spec | None = None) -> np.ndarray:
    """Stereo float32 array ``(n, 2)``, peak ~ -3 dBFS, with fade in/out."""
    spec = spec or make_spec(seed, mood)
    rng = np.random.default_rng(seed + 101)
    total = int(duration * sr)
    bar = int(4 * 60.0 / spec.bpm * sr)
    chord_len = 2 * bar
    n_chords = total // chord_len + 2
    chords = [(i * chord_len, chord_len, spec.prog[i % len(spec.prog)]) for i in range(n_chords)]
    # the last chord before the end resolves to the tonic
    last = max(total // chord_len - 1, 0)
    chords = [(s, ln, 0 if i >= last else d) for i, (s, ln, d) in enumerate(chords)
              if s < total]

    # section map: 8 bars each, the last ~10 s are the outro (pad + sub only)
    plan = _SECTIONS[spec.mood]
    sec_len = 8 * bar
    layer_on = {k: np.zeros(total, bool) for k in ("pad", "sub", "arp", "hat")}
    n_sec = total // sec_len + 1
    for si in range(n_sec):
        name = plan[si % len(plan)] if si < len(plan) else plan[1 + (si - 1) % (len(plan) - 1)]
        a, b = si * sec_len, min((si + 1) * sec_len, total)
        for layer in _LAYERS[name]:
            layer_on[layer][a:b] = True
    outro_from = max(total - int(12 * sr), 0)
    for layer in ("arp", "hat"):
        layer_on[layer][outro_from:] = False

    def gate(layer: str, xfade: float = 1.5) -> np.ndarray:
        k = max(int(xfade * sr), 1)  # smooth on/off so layers fade between sections
        c = np.concatenate([[0.0], np.cumsum(layer_on[layer].astype(np.float64))])
        i = np.arange(total)
        lo, hi = np.clip(i - k // 2, 0, total), np.clip(i + k // 2, 0, total)
        return ((c[hi] - c[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)

    pad_l, pad_r = _pad(spec, chords, total, sr, rng)
    g = gate("pad", 2.5)
    pad_l, pad_r = pad_l * g, pad_r * g
    sub = _sub(spec, chords, total, sr) * gate("sub", 2.0)
    arp_l, arp_r = _arp_notes(spec, chords, total, sr, rng, layer_on["arp"])
    hat = _hats(spec, total, sr, rng, layer_on["hat"]) if spec.mood != "calm" else \
        np.zeros(total, np.float32)

    ir_l, ir_r = _reverb_ir(sr, seed), _reverb_ir(sr, seed + 1)
    wet_l = _reverb(pad_l * 0.6 + arp_l + _delay(arp_l, sr, spec.bpm), ir_l)
    wet_r = _reverb(pad_r * 0.6 + arp_r + _delay(arp_r, sr, spec.bpm), ir_r)
    left = pad_l + sub + arp_l + _delay(arp_l, sr, spec.bpm) * 0.8 + hat * 0.8 + wet_l * 0.55
    right = pad_r + sub + arp_r + _delay(arp_r, sr, spec.bpm) * 0.8 + hat * 1.0 + wet_r * 0.55
    out = np.stack([left, right], axis=1).astype(np.float32)

    # fades: slow in (4 s), longer out (6 s) so the track ends on the tail of the reverb
    t = np.arange(total) / sr
    env = np.minimum(t / 4.0, 1.0) * np.clip((duration - t) / 6.0, 0.0, 1.0)
    out *= (env ** 1.5)[:, None].astype(np.float32)
    peak = float(np.max(np.abs(out))) or 1.0
    return out / peak * 0.7
