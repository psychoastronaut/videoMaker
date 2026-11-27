"""Voice chain: local ffmpeg-only clean-up of the voice-over, run before Stage 1.

raw/voiceover.* -> work/voiceover_clean.wav (original kept as raw/voiceover.orig.*).
Chain: highpass, afftdn (+ optional arnndn), gate, voice EQ, deesser, compressor,
silence shortening, limiter, 2-pass loudnorm. Silence shortening changes timings, so
Stage 1 (and render) must use the cleaned file (see ``voice_path`` in src.common).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from src import audio_analysis as aa
from src.audio_analysis import MONO
from src.common import (
    episode_dir,
    is_offline,
    load_config,
    probe_duration,
    voice_path,
    write_json,
)

console = Console()

AUDIO_EXTS = ("wav", "mp3", "m4a", "flac", "ogg", "aac", "opus")

# per-preset knobs: highpass Hz, afftdn noise reduction dB, gate threshold, mud cut dB,
# presence boost dB, deesser intensity, compressor ratio, silence threshold dB
PRESETS: dict[str, dict[str, float]] = {
    "gentle": {"hp": 70, "nr": 8, "gate": 0.0, "mud": -1.5, "pres": 1.5, "deess": 0.3,
               "ratio": 2.0, "sil_db": -50},
    "podcast": {"hp": 80, "nr": 14, "gate": 0.01, "mud": -3.0, "pres": 2.5, "deess": 0.5,
                "ratio": 3.0, "sil_db": -45},
    "aggressive": {"hp": 100, "nr": 24, "gate": 0.03, "mud": -4.5, "pres": 3.5, "deess": 0.7,
                   "ratio": 4.0, "sil_db": -40},
}

DEFAULTS: dict[str, Any] = {
    "enabled": "auto", "preset": "podcast", "target_lufs": -16.0, "true_peak": -1.5,
    "max_pause": 0.8, "keep_pause": 0.35, "rnnoise_model": None, "hum_hz": 60,
    "denoise": "auto", "denoise_strength": 0.6, "notch": "auto", "expander_db": 14,
}
DENOISE_MODES = ("auto", "rnnoise", "afftdn", "off")
# RNNoise model (GregorR/rnnoise-models, "somnolent-hogwash": speech vs fans/AC/computer noise).
# The repository states the models are not creative work and carry no copyright claim.
MODEL_URL = ("https://raw.githubusercontent.com/GregorR/rnnoise-models/master/"
             "somnolent-hogwash-2018-09-01/sh.rnnn")
MODEL_DIR = "~/.cache/videomaker/models"


def settings(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    s = dict(DEFAULTS)
    s.update((cfg if cfg is not None else load_config()).get("audio") or {})
    if s["preset"] not in PRESETS:
        raise ValueError(f"audio.preset must be one of {sorted(PRESETS)}, got {s['preset']!r}")
    s["denoise"] = str(s["denoise"]).strip().lower()
    if s["denoise"] not in DENOISE_MODES:
        raise ValueError(f"audio.denoise must be one of {DENOISE_MODES}, got {s['denoise']!r}")
    return s


def find_source(raw: Path) -> Path | None:
    """The original voice-over: raw/voiceover.orig.* if saved, else raw/voiceover.*."""
    for stem in ("voiceover.orig", "voiceover"):
        for ext in AUDIO_EXTS:
            p = raw / f"{stem}.{ext}"
            if p.is_file():
                return p
    return None


def is_enabled(episode_id: str, cfg: dict[str, Any] | None = None) -> bool:
    en = settings(cfg)["enabled"]
    if isinstance(en, str):
        en = en.strip().lower()
        if en == "auto":
            return find_source(episode_dir(episode_id) / "raw") is not None
        return en in {"true", "yes", "on", "1"}
    return bool(en)


def _strength(s: dict[str, Any]) -> float:
    return min(max(float(s.get("denoise_strength", 0.6)), 0.0), 1.0)


def denoise_filters(s: dict[str, Any], prof: dict[str, Any] | None = None) -> list[str]:
    """Stage (a) afftdn with the measured noise profile, stage (b) RNNoise (arnndn)."""
    mode, st = str(s.get("denoise", "auto")).lower(), _strength(s)
    if mode == "off" or st <= 0:
        return []
    p = PRESETS[s["preset"]]
    f: list[str] = []
    if mode in ("auto", "afftdn"):
        nr = min(max(p["nr"] * st / 0.6, 4.0), 20.0)
        if prof and prof.get("nf") is not None:  # level calibrated on the file's own noise clip
            f.append(f"afftdn=nr={nr:.1f}:nf={prof['nf']:.0f}:rf=-70:tn=0")
        else:  # no clean clip: let afftdn track the noise itself
            f.append(f"afftdn=nr={nr:.1f}:nf=-45:tn=1")
    model = s.get("rnnoise_model")
    if mode in ("auto", "rnnoise") and model and Path(str(model)).expanduser().is_file():
        mix = min(max(st * 0.65, 0.1), 1.0)
        f += ["aresample=48000", f"arnndn=m={Path(str(model)).expanduser()}:mix={mix:.2f}"]
    elif mode == "rnnoise" and not f:  # no model available: degrade to afftdn, never to nothing
        f.append(f"afftdn=nr={min(max(p['nr'] * st / 0.6, 4.0), 20.0):.1f}:nf=-45:tn=1")
    return f


def gate_filter(s: dict[str, Any], prof: dict[str, Any] | None = None) -> str | None:
    """Soft downward expander: lowers the residue between words (never cuts it); slow release
    so the tail of a word is not chopped."""
    p = PRESETS[s["preset"]]
    if p["gate"] <= 0:
        return None
    thr = p["gate"]
    if prof and prof.get("gate_db") is not None:
        thr = 10 ** (float(prof["gate_db"]) / 20)
    elif s.get("noise_floor_db") is not None:
        thr = 10 ** ((float(s["noise_floor_db"]) + 6 + 10 * p["gate"]) / 20)
    thr = min(max(thr, 0.0005), 0.1)
    rng = 10 ** (-abs(float(s.get("expander_db", 14))) / 20)
    return f"agate=threshold={thr:.5f}:ratio=2.5:attack=15:release=300:range={rng:.4f}:knee=2.5"


def build_filter(s: dict[str, Any], measured: dict[str, str] | None = None, loud: bool = True,
                 prof: dict[str, Any] | None = None, part: str = "all",
                 shorten: bool = True) -> str:
    """The ffmpeg -af chain for settings ``s``.

    ``part``: "pre" = mono, highpass, notches, denoise; "post" = downward expander, EQ,
    de-esser, compressor, silence shortening, loudnorm (``measured`` = pass-1 values);
    "all" = both. ``prof`` is the noise analysis (notches, afftdn band profile, gate level).
    """
    p = PRESETS[s["preset"]]
    f: list[str] = []
    if part in ("all", "pre"):
        if not prof or prof.get("downmix", True):
            f.append(MONO)
        f.append(f"highpass=f={p['hp']:g}")
        if prof and prof.get("notches"):
            f += list(prof["notches"])
        elif not prof and s.get("hum_hz") and p["nr"] > 8:  # no analysis: mains hum fallback
            f += [f"equalizer=f={float(s['hum_hz']) * k:g}:t=q:w=30:g=-20" for k in (1, 2)]
        f += denoise_filters(s, prof)
    if part == "pre":
        return ",".join(f)
    g = gate_filter(s, prof)
    if g:
        f.append(g)
    sil = p["sil_db"]
    if prof and prof.get("floor_db") is not None:
        sil = min(max(sil, float(prof["floor_db"]) + 6), -30)
    elif s.get("noise_floor_db") is not None:
        sil = min(max(sil, float(s["noise_floor_db"]) + 6), -30)
    f += [
        f"equalizer=f=250:t=q:w=1:g={p['mud']:g}",
        f"equalizer=f=4000:t=q:w=0.8:g={p['pres']:g}",
        f"deesser=i={p['deess']:g}:m=0.5:f=0.5",
        # no makeup: the only gain stage is loudnorm, after denoise and expander
        f"acompressor=threshold=-20dB:ratio={p['ratio']:g}:attack=15:release=200:makeup=1",
    ]
    if shorten:  # silences longer than max_pause shrink to keep_pause
        f.append(f"silenceremove=stop_periods=-1:stop_duration={float(s['max_pause']):g}"
                 f":stop_threshold={sil:g}dB:stop_silence={float(s['keep_pause']):g}:detection=rms")
    f.append("alimiter=limit=0.89:attack=5:release=50:level=disabled")
    i, tp = float(s["target_lufs"]), float(s["true_peak"])
    if loud:
        ln = f"loudnorm=I={i:g}:TP={tp:g}:LRA=11"
        if measured:
            ln += (f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
                   f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
                   f":offset={measured['target_offset']}:linear=true")
        f.append(ln + (":print_format=json" if not measured else ""))
    return ",".join(f)


def build_legacy_filter(s: dict[str, Any], measured: dict[str, str] | None = None,
                        shorten: bool = False, downmix: bool = True) -> str:
    """The previous chain (fixed afftdn, 60 Hz notches, gate, makeup gain) for A/B listening."""
    p = PRESETS[s["preset"]]
    f = [MONO, f"highpass=f={p['hp']:g}"] if downmix else [f"highpass=f={p['hp']:g}"]
    if p["nr"] > 8:
        f += [f"equalizer=f={60 * k}:t=q:w=30:g=-20" for k in (1, 2)]
    f.append(f"afftdn=nr={p['nr']:g}:nf=-45:tn=1")
    if p["gate"] > 0:
        f.append(f"agate=threshold={p['gate']}:ratio=3:attack=10:release=250:range=0.05")
    f += [f"equalizer=f=250:t=q:w=1:g={p['mud']:g}", f"equalizer=f=4000:t=q:w=0.8:g={p['pres']:g}",
          f"deesser=i={p['deess']:g}:m=0.5:f=0.5",
          f"acompressor=threshold=-20dB:ratio={p['ratio']:g}:attack=15:release=200:makeup=3"]
    if shorten:
        f.append(f"silenceremove=stop_periods=-1:stop_duration={float(s['max_pause']):g}"
                 f":stop_threshold={p['sil_db']:g}dB:stop_silence={float(s['keep_pause']):g}:detection=rms")
    f.append("alimiter=limit=0.89:attack=5:release=50:level=disabled")
    ln = f"loudnorm=I={float(s['target_lufs']):g}:TP={float(s['true_peak']):g}:LRA=11"
    if measured:
        ln += (f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
               f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
               f":offset={measured['target_offset']}:linear=true")
    return ",".join([*f, ln + (":print_format=json" if not measured else "")])


def _ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", *args],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-600:]}")
    return r


def _loudnorm_json(stderr: str) -> dict[str, str]:
    return json.loads(stderr[stderr.rindex("{"):stderr.rindex("}") + 1])


def measure(path: Path) -> dict[str, float]:
    """LUFS, sample peak (dBFS), estimated noise floor (dBFS), duration (on the mono downmix)."""
    mono = MONO if aa.channels(path) == 2 else "anull"
    r = _ffmpeg(["-i", str(path), "-af", f"{mono},loudnorm=print_format=json", "-f", "null", "-"])
    j = _loudnorm_json(r.stderr)
    x = aa.load_mono(path).astype(np.float32)
    peak = 20 * np.log10(max(float(np.max(np.abs(x))) if x.size else 0.0, 1e-9))
    nf = noise_floor_db(x)
    return {"lufs": float(j["input_i"]), "true_peak": float(j["input_tp"]),
            "peak_db": round(float(peak), 2), "noise_floor_db": nf,
            "noise_rel_lufs": round(nf - float(j["input_i"]), 2),
            "duration": round(probe_duration(path) or 0.0, 3)}


def noise_floor_db(x: np.ndarray, sr: int = 16000, win: float = 0.05) -> float:
    """Noise floor estimate: 10th percentile of 50 ms RMS windows, dBFS."""
    n = int(sr * win)
    if x.size < n:
        return -120.0
    frames = x[: x.size // n * n].reshape(-1, n)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return round(float(20 * np.log10(max(np.percentile(rms, 10), 1e-7))), 2)


def ensure_rnnoise_model(s: dict[str, Any]) -> Path | None:
    """The .rnnn model: ``audio.rnnoise_model`` if set, else downloaded once into the cache.

    Returns None (and the chain degrades to afftdn only) when offline or the download fails.
    """
    if s.get("rnnoise_model"):
        p = Path(str(s["rnnoise_model"])).expanduser()
        return p if p.is_file() else None
    p = Path(MODEL_DIR).expanduser() / "sh.rnnn"
    if p.is_file() and p.stat().st_size > 10_000:
        return p
    if is_offline():
        return None
    try:
        import urllib.request
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MODEL_URL, timeout=30) as r:
            data = r.read()
        if len(data) < 10_000:
            return None
        tmp = p.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(p)
        return p
    except Exception as e:  # noqa: BLE001  no network: keep going without RNNoise
        console.print(f"  [yellow]RNNoise model unavailable ({e.__class__.__name__}); afftdn only[/]")
        return None


def analyze_noise(src: Path, s: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Noise profile of the source: tonal peaks + notches, afftdn band profile, noise segments."""
    x = aa.load_mono(src)
    f, pdb, noise = aa.noise_spectrum(x)
    notch = s.get("notch", "auto")
    if isinstance(notch, (list, tuple)):  # explicit frequencies
        peaks = [{"freq_hz": float(h), "prominence_db": 15.0, "width_hz": 8.0} for h in notch]
    elif str(notch).lower() == "off":
        peaks = []
    else:
        peaks = [q for q in aa.find_tonal_peaks(f, pdb) if q["prominence_db"] >= 9.0]
    db = aa.frame_db(x)
    floor = float(np.mean(db[noise])) if noise.any() else float(np.percentile(db, 10))
    prof = {"notches": aa.notch_filters(peaks), "peaks": peaks, "floor_db": round(floor, 2),
            "downmix": aa.channels(src) == 2, "noise_clip": aa.longest_noise_run(x),
            "nf": None, "noise_seconds": round(float(noise.sum()) * aa.WIN, 2)}
    if prof["noise_clip"] and s["denoise"] in ("auto", "afftdn") and _strength(s) > 0:
        prof["nf"] = calibrate_nf(src, prof, s)
    return prof, x


def calibrate_nf(src: Path, prof: dict[str, Any], s: dict[str, Any]) -> float | None:
    """afftdn's noise-floor level for THIS file: the lowest ``nf`` whose afftdn run removes the
    target amount (strength-scaled, 3-12 dB) from the pure-noise clip. Fast: the clip is ~2 s."""
    t0, d = prof["noise_clip"]
    pre = ",".join(([MONO] if prof.get("downmix", True) else []) + [f"highpass={PRESETS[s['preset']]['hp']:g}",
                                                                   *prof["notches"]])
    nr = min(max(PRESETS[s["preset"]]["nr"] * _strength(s) / 0.6, 4.0), 20.0)
    target = min(max(6 * _strength(s) / 0.6, 3.0), 12.0)

    def level(af: str) -> float:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t0:.3f}", "-t", f"{d:.3f}",
                              "-i", str(src), "-af", af, "-ar", "48000", "-f", "f32le", "-"],
                             capture_output=True, check=False).stdout
        y = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
        return float(20 * np.log10(max(np.sqrt(np.mean(y ** 2)), 1e-9))) if y.size else -120.0

    try:
        base, best = level(pre), (-20, 0.0)
        for nf in range(-70, -19, 5):
            red = base - level(f"{pre},afftdn=nr={nr:.1f}:nf={nf}:rf=-70:tn=0")
            if red >= target:
                return float(nf)
            best = max(best, (nf, red), key=lambda t: t[1])
        return float(best[0]) if best[1] >= 3 else None
    except (OSError, ValueError):
        return None


def _run_chain(src: Path, af: str, dst: Path, rate: int = 48000, codec: str = "pcm_s16le") -> None:
    _ffmpeg(["-i", str(src), "-af", af, "-ar", str(rate), "-ac", "1", "-c:a", codec, str(dst)])


def render(src: Path, dst: Path, s: dict[str, Any], prof: dict[str, Any], shorten: bool = True,
           legacy: bool = False) -> dict[str, Any]:
    """Two-stage render: notches + denoise -> temp (analysed for the expander threshold) ->
    expander, EQ, compression, silence shortening, 2-pass loudnorm -> dst."""
    tmp = dst.with_suffix(".tmp.wav")
    info: dict[str, Any] = {}
    try:
        if legacy:
            dm = bool(prof.get("downmix", True))
            r = _ffmpeg(["-i", str(src), "-af", build_legacy_filter(s, shorten=shorten, downmix=dm),
                         "-f", "null", "-"])
            m = _loudnorm_json(r.stderr)
            _run_chain(src, build_legacy_filter(s, m, shorten, dm), tmp)
        else:
            dn = dst.with_suffix(".dn.wav")
            _run_chain(src, build_filter(s, prof=prof, part="pre"), dn, codec="pcm_f32le")
            y = aa.load_mono(dn)
            db = aa.frame_db(y)
            noise, _ = aa._split(db)
            floor = float(np.mean(db[noise])) if noise.any() else float(np.percentile(db, 10))
            vp90 = float(np.percentile(db, 90))
            # expander knee ~8 dB above the denoised floor, but well below the voice
            prof = {**prof, "floor_db": floor, "gate_db": min(floor + float(s.get("expander_margin_db", 12)), vp90 - 30)}
            info = {"denoised_floor_dbfs": round(floor, 2), "gate_threshold_dbfs": round(prof["gate_db"], 2)}
            full = build_filter(s, prof=prof, part="post", shorten=shorten)
            r = _ffmpeg(["-i", str(dn), "-af", full, "-f", "null", "-"])
            m = _loudnorm_json(r.stderr)
            _run_chain(dn, build_filter(s, m, prof=prof, part="post", shorten=shorten), tmp)
            info["_dn"] = dn
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return info


def verify(x: np.ndarray, dn: Path, y: np.ndarray, s: dict[str, Any], prof: dict[str, Any],
           gate_db: float | None) -> dict[str, Any]:
    """Objective before/after numbers: floor (absolute + relative), SNR, tones, speech band."""
    hz = [q["freq_hz"] for q in prof["peaks"]]
    before, after = aa.analyze(x, hz), aa.analyze(y, hz)
    # speech preserved: 300-4000 Hz energy over the ORIGINAL's voice frames, same level scale,
    # after notches + denoise + expander (no EQ / gain)
    d = aa.load_mono(dn)
    _, voice = aa._split(aa.frame_db(x))
    g = gate_filter(s, {**prof, "gate_db": gate_db}) if gate_db is not None else None
    if g:
        gated = dn.with_suffix(".g.wav")
        _run_chain(dn, g, gated, codec="pcm_f32le")
        d = aa.load_mono(gated)
        gated.unlink(missing_ok=True)
    n = min(x.size, d.size)
    loss = aa.band_energy_db(d[:n], voice, 300, 4000) - aa.band_energy_db(x[:n], voice, 300, 4000)
    tones = {k: {"before_db": v, "after_db": after["tonal_energy_db"][k],
                 "before_rel_voice_db": round(v - before["voice_p90_dbfs"], 1),
                 "after_rel_voice_db": round(after["tonal_energy_db"][k] - after["voice_p90_dbfs"], 1)}
             for k, v in before["tonal_energy_db"].items()}
    return {"before": before, "after": after, "tones": tones,
            "speech_band_change_db": round(float(loss), 2),
            "floor_improvement_rel_voice_db": round(before["noise_rel_voice_db"] - after["noise_rel_voice_db"], 2)}


def process(src: Path, dst: Path, s: dict[str, Any]) -> dict[str, Any]:
    """Analyse the noise, run the chain, return the before/after report."""
    before = measure(src)
    s = {**s, "noise_floor_db": before["noise_floor_db"]}
    if s["denoise"] in ("auto", "rnnoise") and _strength(s) > 0:
        model = ensure_rnnoise_model(s)
        s["rnnoise_model"] = str(model) if model else None
    prof, x = analyze_noise(src, s)
    info = render(src, dst, s, prof)
    dn = info.pop("_dn")
    try:
        v = verify(x, dn, aa.load_mono(dst), s, prof, info.get("gate_threshold_dbfs"))
    finally:
        dn.unlink(missing_ok=True)
    after = measure(dst)
    return {"preset": s["preset"], "denoise": s["denoise"], "denoise_strength": _strength(s),
            "rnnoise_model": s.get("rnnoise_model"), "source": str(src), "output": str(dst),
            "before": before, "after": after,
            "removed_seconds": round(before["duration"] - after["duration"], 3),
            "noise_reduction_db": round(before["noise_rel_lufs"] - after["noise_rel_lufs"], 2),
            "noise_analysis": {"noise_seconds": prof["noise_seconds"], "floor_dbfs": prof["floor_db"],
                               "tonal_peaks": prof["peaks"], "notch_filters": prof["notches"],
                               "noise_clip_s": prof["noise_clip"], "afftdn_nf": prof["nf"], **info},
            "verification": v}


def _vloud(src: Path, af: str, dst: Path, s: dict[str, Any]) -> None:
    """Level-match ``src`` to the target LUFS with a linear gain (A/B fairness)."""
    ln = f"loudnorm=I={float(s['target_lufs']):g}:TP=-1.5:LRA=11"
    r = _ffmpeg(["-i", str(src), "-af", f"{af},{ln}:print_format=json", "-f", "null", "-"])
    m = _loudnorm_json(r.stderr)
    ln += (f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
           f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    _run_chain(src, f"{af},{ln}", dst)


def ab_windows(x: np.ndarray, n: int = 3, length: float = 10.0) -> list[float]:
    """Start times of ``n`` windows with plenty of voice and at least one pause (noise audible)."""
    db = aa.frame_db(x)
    noise, voice = aa._split(db, 15)
    w = int(length / aa.WIN)
    if db.size < w + 2:
        return [0.0]
    starts, scores = np.arange(0, db.size - w), []
    for i in starts:
        v, q = voice[i:i + w].mean(), noise[i:i + w].mean()
        scores.append(v - abs(q - 0.12) if q > 0.03 else -1.0)
    scores_a, picks = np.array(scores), []
    for k in range(n):  # one window per third of the recording
        lo, hi = k * len(starts) // n, (k + 1) * len(starts) // n
        picks.append(float(starts[lo + int(np.argmax(scores_a[lo:hi]))] * aa.WIN))
    return picks


def make_ab(src: Path, out_dir: Path, s: dict[str, Any], n: int = 3, length: float = 10.0) -> dict[str, Any]:
    """original.wav / atual.wav / novo.wav (n x 10 s excerpts with voice, same timeline, level
    matched, silence shortening off) and before/after spectrogram PNGs, in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s = dict(s)
    if s["denoise"] in ("auto", "rnnoise") and _strength(s) > 0:
        model = ensure_rnnoise_model(s)
        s["rnnoise_model"] = str(model) if model else None
    prof, x = analyze_noise(src, s)
    starts = ab_windows(x, n, length)
    full = {"original": out_dir / "_original_full.wav", "atual": out_dir / "_atual_full.wav",
            "novo": out_dir / "_novo_full.wav"}
    _vloud(src, MONO, full["original"], s)
    tmp = out_dir / "_legacy.wav"
    render(src, tmp, s, prof, shorten=False, legacy=True)
    _vloud(tmp, "anull", full["atual"], s)
    tmp2 = out_dir / "_new.wav"
    info = render(src, tmp2, s, prof, shorten=False)
    info.pop("_dn").unlink(missing_ok=True)
    _vloud(tmp2, "anull", full["novo"], s)
    for t in (tmp, tmp2):
        t.unlink(missing_ok=True)
    for name, f in full.items():
        parts, labels = [], []
        for k, st in enumerate(starts):
            parts.append(f"[0:a]atrim={st:.2f}:{st + length:.2f},asetpts=PTS-STARTPTS,apad=pad_dur=1[a{k}]")
            labels.append(f"[a{k}]")
        fg = ";".join(parts) + f";{''.join(labels)}concat=n={len(starts)}:v=0:a=1[o]"
        _ffmpeg(["-i", str(f), "-filter_complex", fg, "-map", "[o]", "-ar", "48000", "-ac", "1",
                 "-c:a", "pcm_s16le", str(out_dir / f"{name}.wav")])
        f.unlink(missing_ok=True)
        _ffmpeg(["-i", str(out_dir / f"{name}.wav"), "-lavfi",
                 "showspectrumpic=s=1400x520:legend=1:fscale=log:scale=log:start=40:stop=8000",
                 "-frames:v", "1", str(out_dir / f"spectrogram_{name}.png")])
    meta = {"windows_start_s": starts, "window_s": length, "gap_between_clips_s": 1,
            "note": "same timeline in all three files, level-matched, silence shortening off"}
    write_json(out_dir / "ab.json", meta)
    return meta


def format_report(rep: dict[str, Any]) -> str:
    b, a = rep["before"], rep["after"]
    v, na = rep["verification"], rep["noise_analysis"]
    vb, va = v["before"], v["after"]
    lines = [
        (f"  preset {rep['preset']}, denoise {rep['denoise']} (strength {rep['denoise_strength']:g})"
         f"{', RNNoise on' if rep.get('rnnoise_model') else ''}"),
        f"  LUFS (mono)  {b['lufs']:7.1f} -> {a['lufs']:7.1f}",
        f"  true peak    {b['true_peak']:7.1f} -> {a['true_peak']:7.1f} dBTP",
        (f"  noise floor  {vb['noise_floor_dbfs']:7.1f} -> {va['noise_floor_dbfs']:7.1f} dBFS (absolute)"),
        (f"  floor vs voice {vb['noise_rel_voice_db']:6.1f} -> {va['noise_rel_voice_db']:6.1f} dB"
         f" ({v['floor_improvement_rel_voice_db']:+.1f} dB better)"),
        f"  SNR          {vb['snr_db']:7.1f} -> {va['snr_db']:7.1f} dB",
        f"  speech 300-4k change {v['speech_band_change_db']:+.2f} dB (notches + denoise + expander)",
        (f"  duration     {b['duration']:7.1f} -> {a['duration']:7.1f} s"
         f" (removed {rep['removed_seconds']:.1f} s)")]
    for q in na["tonal_peaks"]:
        t = v["tones"].get(f"{q['freq_hz']:g}")
        if t:
            lines.append(f"  tone {q['freq_hz']:7.1f} Hz (+{q['prominence_db']:.0f} dB): "
                         f"{t['before_rel_voice_db']:6.1f} -> {t['after_rel_voice_db']:6.1f} dB re voice")
    return "\n".join(lines)


def run(episode_id: str, work_dir: Path, force: bool = False, report: bool = False,
        preset: str | None = None, ab: bool = False) -> dict[str, Any] | None:
    """Audio stage: clean the voice-over into work/voiceover_clean.wav."""
    cfg = load_config()
    s = settings(cfg)
    if preset:
        s["preset"] = preset
        settings({"audio": s})
    raw = episode_dir(episode_id) / "raw"
    if not is_enabled(episode_id, cfg):
        console.print("  [dim]audio.enabled is off or raw/voiceover.* missing; skipped[/]")
        return None
    src = find_source(raw)
    if src is None:
        raise FileNotFoundError(f"Voiceover not found in {raw}")
    if ".orig." not in src.name:  # keep the untouched original next to it
        keep = raw / f"voiceover.orig{src.suffix}"
        if not keep.exists():
            shutil.copy2(src, keep)
        src = keep
    work_dir.mkdir(parents=True, exist_ok=True)
    dst = work_dir / "voiceover_clean.wav"
    if ab:
        meta = make_ab(src, work_dir / "audio_ab", s)
        console.print(f"  A/B excerpts + spectrograms: {work_dir / 'audio_ab'} (windows {meta['windows_start_s']})")
        return meta
    rep = process(src, dst, s)
    write_json(work_dir / "audio_report.json", rep)
    console.print(f"  Clean voice-over: {dst}")
    if report:
        console.print(format_report(rep))
    return rep


__all__ = ["build_filter", "measure", "process", "run", "voice_path"]
