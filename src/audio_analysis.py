"""Objective voice-over analysis: noise profile, tonal peaks (hum/fan), floor, SNR, speech band.

Everything is plain numpy/scipy on a 16 kHz mono float signal; no network, no models.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

SR = 16000
WIN = 0.05  # 50 ms frames for level statistics
NFFT = 4096  # 3.9 Hz bins at 16 kHz
# afftdn custom band-noise centres (Hz), 15 bands
MONO = "pan=mono|c0=0.5*c0+0.5*c1"  # unity-gain downmix (plain -ac 1 adds +3 dB on dual mono)
AFFTDN_BANDS = (20, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000)


def channels(path: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=channels", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False)
    try:
        return int(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 1


def load_mono(path: Path, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-af", MONO if channels(path) == 2 else "anull",
                          "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def frame_db(x: np.ndarray, sr: int = SR, win: float = WIN) -> np.ndarray:
    """RMS dBFS of consecutive windows."""
    n = int(sr * win)
    if x.size < n:
        return np.array([-120.0])
    fr = x[: x.size // n * n].reshape(-1, n)
    return 20 * np.log10(np.maximum(np.sqrt(np.mean(fr ** 2, axis=1)), 1e-7))


def _split(db: np.ndarray, noise_pct: float = 10, voice_pct: float = 60) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of noise-only frames (quietest) and voice frames (loud)."""
    live = db > -110  # ignore digital silence
    ref = db[live] if live.any() else db
    lo = np.percentile(ref, noise_pct)
    hi = max(np.percentile(ref, voice_pct), lo + 12)
    return (db <= lo) & live, db >= hi


def noise_spectrum(x: np.ndarray, sr: int = SR, win: float = WIN,
                   noise_pct: float = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean power spectrum (freqs, dB/bin, noise-frame mask) over the quietest frames.

    Quiet frames are grouped in runs; runs are concatenated with a Hann-windowed Welch
    estimate so we never analyse across a voiced gap.
    """
    db = frame_db(x, sr, win)
    n = int(sr * win)
    noise, _ = _split(db, noise_pct)
    segs, run = [], []
    for i, flag in enumerate(noise):
        if flag:
            run.append(i)
        elif run:
            segs.append(run)
            run = []
    if run:
        segs.append(run)
    psds = []
    for r in segs:
        seg = x[r[0] * n:(r[-1] + 1) * n]
        if seg.size >= 1024:
            f, p = signal.welch(seg, sr, nperseg=min(NFFT, seg.size), noverlap=None)
            if p.size == NFFT // 2 + 1:
                psds.append(p)
    if not psds:  # fall back to the whole signal's quiet frames, stitched
        idx = np.where(noise)[0]
        seg = np.concatenate([x[i * n:(i + 1) * n] for i in idx]) if idx.size else x
        f, p = signal.welch(seg, sr, nperseg=min(NFFT, max(seg.size, 256)))
        psds = [p]
    f = np.fft.rfftfreq(NFFT, 1 / sr) if psds[0].size == NFFT // 2 + 1 else f
    mean = np.mean(psds, axis=0)
    return f, 10 * np.log10(np.maximum(mean, 1e-20)), noise


def find_tonal_peaks(f: np.ndarray, pdb: np.ndarray, fmin: float = 45, fmax: float = 3500,
                     min_prom: float = 8.0, max_peaks: int = 8) -> list[dict[str, float]]:
    """Narrow spectral lines standing out from the local broadband floor (hum, fan, whine)."""
    df = f[1] - f[0]
    k = max(round(150 / df) | 1, 5)  # ~150 Hz median window = local floor
    pad = np.pad(pdb, k // 2, mode="edge")
    base = signal.medfilt(pad, k)[k // 2:-(k // 2)]
    rel = pdb - base
    idx, props = signal.find_peaks(rel, height=min_prom, prominence=min_prom * 0.75, width=(1, int(40 / df)))
    out = []
    for i, h, w in zip(idx, props["peak_heights"], props["widths"], strict=True):
        if not fmin <= f[i] <= fmax:
            continue
        # parabolic interpolation for the real frequency
        if 0 < i < len(pdb) - 1:
            a, b, c = pdb[i - 1], pdb[i], pdb[i + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        else:
            d = 0.0
        out.append({"freq_hz": round(float(f[i] + d * df), 1), "prominence_db": round(float(h), 1),
                    "width_hz": round(float(max(w * df, df)), 1)})
    out.sort(key=lambda p: -p["prominence_db"])
    return out[:max_peaks]


VOICE_ZONE = (85.0, 1000.0)  # fundamentals/low harmonics of speech: notches here tint the timbre


def limit_voice_zone(peaks: list[dict[str, float]], x: np.ndarray, voice_cut_db: float) -> list[dict[str, float]]:
    """Mark each peak inside the voice zone with ``max_cut_db`` (depth cap) and a higher Q.

    A line is only worth a (shallow) notch there if it stays stationary: its excess over the
    local floor in the quiet frames must be at least 12 dB. Outside the zone the cap is 26 dB."""
    out = []
    for q in peaks:
        q = dict(q)
        if VOICE_ZONE[0] <= q["freq_hz"] <= VOICE_ZONE[1] and voice_cut_db < 26:
            if q["prominence_db"] < 12.0:
                continue
            q["max_cut_db"] = voice_cut_db
            q["width_hz"] = min(q["width_hz"], 8.0)
        out.append(q)
    return out


def notch_filters(peaks: list[dict[str, float]], max_cut_db: float = 26.0) -> list[str]:
    """ffmpeg ``equalizer`` notches (Q ~ f / width, clamped to 4..40) for each tonal peak."""
    out = []
    for p in peaks:
        f = p["freq_hz"]
        bw = min(max(p["width_hz"] * 0.8, 5.0), f / 4)  # narrow: Welch window already widens the line
        gain = min(p["prominence_db"] + 6, p.get("max_cut_db", max_cut_db))
        out.append(f"equalizer=f={f:g}:t=h:w={bw:.1f}:g=-{gain:.1f}")
    return out


def longest_noise_run(x: np.ndarray, sr: int = SR, min_s: float = 0.4) -> tuple[float, float] | None:
    """(start, duration) in seconds of the longest stretch of pure noise (frames within 5 dB of
    the 10th-percentile level)."""
    db = frame_db(x, sr)
    noise = (db <= np.percentile(db, 10) + 5) & (db > -110)
    best, i = (0, 0), 0
    while i < noise.size:
        if noise[i]:
            j = i
            while j < noise.size and noise[j]:
                j += 1
            best = max(best, (j - i, i), key=lambda t: t[0])
            i = j
        else:
            i += 1
    n, st = best
    return (round(st * WIN + 0.02, 3), round(n * WIN - 0.04, 3)) if n * WIN >= min_s else None


def band_energy_db(x: np.ndarray, mask: np.ndarray, lo: float, hi: float, sr: int = SR,
                   win: float = WIN) -> float:
    """Mean power (dB) in [lo,hi] Hz over the frames selected by ``mask``."""
    n = int(sr * win)
    sos = signal.butter(4, [lo, hi], "bandpass", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x)
    fr = y[: mask.size * n].reshape(-1, n)[mask[: y.size // n]]
    if fr.size == 0:
        return -120.0
    return float(10 * np.log10(max(np.mean(fr ** 2), 1e-14)))


def analyze(x: np.ndarray, peaks_hz: list[float] | None = None, sr: int = SR) -> dict[str, Any]:
    """Absolute/relative noise floor, SNR, tonal energy and speech-band energy of a signal."""
    db = frame_db(x, sr)
    noise, voice = _split(db)
    floor = float(np.mean(db[noise])) if noise.any() else float(np.percentile(db, 10))
    vdb = float(np.percentile(db, 90))
    res: dict[str, Any] = {
        "noise_floor_dbfs": round(floor, 2), "voice_p90_dbfs": round(vdb, 2),
        "noise_rel_voice_db": round(floor - vdb, 2), "snr_db": round(vdb - floor, 2),
        "speech_band_db": round(band_energy_db(x, voice, 300, 4000, sr), 2)}
    tones = {}
    for hz in peaks_hz or []:
        tones[f"{hz:g}"] = round(band_energy_db(x, noise, max(hz - 4, 20), hz + 4, sr), 2)
    res["tonal_energy_db"] = tones
    return res
