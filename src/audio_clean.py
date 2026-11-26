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

from src.common import episode_dir, load_config, probe_duration, voice_path, write_json

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
}


def settings(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    s = dict(DEFAULTS)
    s.update((cfg if cfg is not None else load_config()).get("audio") or {})
    if s["preset"] not in PRESETS:
        raise ValueError(f"audio.preset must be one of {sorted(PRESETS)}, got {s['preset']!r}")
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


def build_filter(s: dict[str, Any], measured: dict[str, str] | None = None,
                 loud: bool = True) -> str:
    """The ffmpeg -af chain for the settings ``s`` (``measured`` = loudnorm pass-1 values)."""
    p = PRESETS[s["preset"]]
    f = [f"highpass=f={p['hp']:g}"]
    hum = s.get("hum_hz")
    if hum and p["nr"] > 8:  # mains hum notches (fundamental + 2nd harmonic)
        f += [f"equalizer=f={float(hum) * k:g}:t=q:w=30:g=-20" for k in (1, 2)]
    f += [f"afftdn=nr={p['nr']:g}:nf=-45:tn=1"]
    model = s.get("rnnoise_model")
    if model and Path(str(model)).expanduser().is_file():
        f.append(f"arnndn=m={Path(str(model)).expanduser()}")
    if p["gate"] > 0:
        thr = p["gate"]
        if s.get("noise_floor_db") is not None:  # adaptive: a few dB above the measured floor
            thr = 10 ** ((float(s["noise_floor_db"]) + 6 + 10 * p["gate"]) / 20)
        thr = min(max(thr, 0.002), 0.1)
        f.append(f"agate=threshold={thr:.4f}:ratio=3:attack=10:release=250:range=0.05")
    sil = p["sil_db"]
    if s.get("noise_floor_db") is not None:  # never below the measured floor (+6 dB margin)
        sil = min(max(sil, float(s["noise_floor_db"]) + 6), -30)
    f += [
        f"equalizer=f=250:t=q:w=1:g={p['mud']:g}",
        f"equalizer=f=4000:t=q:w=0.8:g={p['pres']:g}",
        f"deesser=i={p['deess']:g}:m=0.5:f=0.5",
        f"acompressor=threshold=-20dB:ratio={p['ratio']:g}:attack=15:release=200:makeup=3",
        # silences longer than max_pause shrink to keep_pause (natural pauses stay intact)
        (f"silenceremove=stop_periods=-1:stop_duration={float(s['max_pause']):g}"
         f":stop_threshold={sil:g}dB:stop_silence={float(s['keep_pause']):g}:detection=rms"),
        "alimiter=limit=0.89:attack=5:release=50:level=disabled",
    ]
    i, tp = float(s["target_lufs"]), float(s["true_peak"])
    if loud:
        ln = f"loudnorm=I={i:g}:TP={tp:g}:LRA=11"
        if measured:
            ln += (f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
                   f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
                   f":offset={measured['target_offset']}:linear=true")
        f.append(ln + (":print_format=json" if not measured else ""))
    return ",".join(f)


def _ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", *args],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-600:]}")
    return r


def _loudnorm_json(stderr: str) -> dict[str, str]:
    return json.loads(stderr[stderr.rindex("{"):stderr.rindex("}") + 1])


def measure(path: Path) -> dict[str, float]:
    """LUFS, sample peak (dBFS), estimated noise floor (dBFS), duration."""
    r = _ffmpeg(["-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"])
    j = _loudnorm_json(r.stderr)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    peak = 20 * np.log10(max(float(np.max(np.abs(x))) if x.size else 0.0, 1e-9))
    return {"lufs": float(j["input_i"]), "true_peak": float(j["input_tp"]),
            "peak_db": round(float(peak), 2), "noise_floor_db": noise_floor_db(x),
            "noise_rel_lufs": round(noise_floor_db(x) - float(j["input_i"]), 2),
            "duration": round(probe_duration(path) or 0.0, 3)}


def noise_floor_db(x: np.ndarray, sr: int = 16000, win: float = 0.05) -> float:
    """Noise floor estimate: 10th percentile of 50 ms RMS windows, dBFS."""
    n = int(sr * win)
    if x.size < n:
        return -120.0
    frames = x[: x.size // n * n].reshape(-1, n)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return round(float(20 * np.log10(max(np.percentile(rms, 10), 1e-7))), 2)


def process(src: Path, dst: Path, s: dict[str, Any]) -> dict[str, Any]:
    """Run the chain (loudnorm measured in a first pass), return the before/after report."""
    before = measure(src)
    s = {**s, "noise_floor_db": before["noise_floor_db"]}
    tmp = dst.with_suffix(".tmp.wav")
    r = _ffmpeg(["-i", str(src), "-af", build_filter(s), "-f", "null", "-"])
    m = _loudnorm_json(r.stderr)
    _ffmpeg(["-i", str(src), "-af", build_filter(s, measured=m), "-ar", "48000", "-ac", "1",
             "-c:a", "pcm_s16le", str(tmp)])
    tmp.replace(dst)
    after = measure(dst)
    return {"preset": s["preset"], "source": str(src), "output": str(dst), "before": before,
            "after": after,
            "removed_seconds": round(before["duration"] - after["duration"], 3),
            "noise_reduction_db": round(before["noise_rel_lufs"] - after["noise_rel_lufs"], 2)}


def format_report(rep: dict[str, Any]) -> str:
    b, a = rep["before"], rep["after"]
    return "\n".join([
        f"  preset {rep['preset']}",
        f"  LUFS         {b['lufs']:7.1f} -> {a['lufs']:7.1f}",
        f"  true peak    {b['true_peak']:7.1f} -> {a['true_peak']:7.1f} dBTP",
        (f"  noise floor  {b['noise_floor_db']:7.1f} -> {a['noise_floor_db']:7.1f} dBFS"
         f" (relative to voice level: {-rep['noise_reduction_db']:+.1f} dB)"),
        (f"  duration     {b['duration']:7.1f} -> {a['duration']:7.1f} s"
         f" (removed {rep['removed_seconds']:.1f} s)")])


def run(episode_id: str, work_dir: Path, force: bool = False, report: bool = False,
        preset: str | None = None) -> dict[str, Any] | None:
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
    rep = process(src, dst, s)
    write_json(work_dir / "audio_report.json", rep)
    console.print(f"  Clean voice-over: {dst}")
    if report:
        console.print(format_report(rep))
    return rep


__all__ = ["build_filter", "measure", "process", "run", "voice_path"]
