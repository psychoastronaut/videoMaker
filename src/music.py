"""Stage 5b: background music (``work/music.wav``) with a licence audit.

Sources (``music.source``): ``episode`` (raw/music/*), ``library`` (assets/music/*), ``openverse``
(CC0 audio via the Openverse API; also Internet Archive / Wikimedia Commons when public domain),
``synth`` (src/music_synth.py, original, always available), ``off``; ``auto`` tries them in that
order and falls back to the synthesizer. Never CC-BY/CC-BY-SA. The track is looped/cross-faded
to the full timeline (intro card + voice-over + outro card) and normalised to
``music.level_lufs``; the render ducks it under the voice (``mix_filter``).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from rich.console import Console

from src import music_synth
from src.common import (
    PROJECT_ROOT,
    USER_AGENT,
    is_offline,
    load_config,
    probe_duration,
    read_json,
    voice_path,
    write_json,
)

console = Console()
SR = 48000
SOURCES = ("auto", "episode", "library", "openverse", "synth", "off")
MOODS = ("auto", *music_synth.MOODS)
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}
OPENVERSE_API = "https://api.openverse.org/v1/audio/"
CACHE_DIR = Path("~/.cache/videomaker/music").expanduser()
FREE_LICENSES = {"cc0", "pdm", "cc0 1.0", "public domain", "public domain mark 1.0", "pd"}
MAX_DOWNLOAD = 60 * 1024 * 1024
MIN_SECONDS = 60.0
_QUERIES = {"tense": ["dark ambient", "drone ambient pad", "dark synthwave"],
            "calm": ["ambient pad", "calm ambient music", "soundscape ambient"],
            "upbeat": ["synthwave", "electronic ambient beat", "chill electronic"]}
_MUSICISH = ("music", "ambient", "synthwave", "soundtrack", "bgm", "pad", "chill", "drone",
             "atmosphere", "instrumental", "loop", "electronic")
_NOT_MUSIC = ("field recording", "speech", "voice", "kick", "snare", "hat", "sfx", "rain",
              "traffic", "footstep", "scream", "phone", "ringtone", "flying", "sample pack")


@dataclass
class MusicCfg:
    enabled: str = "auto"          # auto | true | false
    source: str = "auto"
    mood: str = "auto"
    level_lufs: float = -22.0      # integrated loudness of the track before ducking
    duck_db: float = 10.0
    intro_gain: float = 0.35       # linear factor over the intro card, ramps to 1 by its end
    seed: int | None = None
    volume_in_pauses: float = 2.0  # dB the music rises over long pauses
    strict_cc0_only: bool = False
    openverse: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MusicCfg:
        m = config.get("music") or {}
        en = m.get("enabled", "auto")
        en = "false" if en is False else "true" if en is True else str(en).lower()
        src = str(m.get("source", "auto")).lower()
        mood = str(m.get("mood", "auto")).lower()
        stock = config.get("stock_media") or {}
        return cls(
            enabled=en if en in ("auto", "true", "false") else "auto",
            source=src if src in SOURCES else "auto",
            mood=mood if mood in MOODS else "auto",
            level_lufs=float(m.get("level_lufs", -22.0)),
            duck_db=max(float(m.get("duck_db", 10.0)), 0.0),
            intro_gain=min(max(float(m.get("intro_gain", 0.35)), 0.0), 1.0),
            seed=None if m.get("seed") is None else int(m["seed"]),
            volume_in_pauses=float(m.get("volume_in_pauses", 2.0)),
            strict_cc0_only=bool(m.get("strict_cc0_only", stock.get("strict_cc0_only", False))),
            openverse=bool(m.get("openverse", True)))

    @property
    def off(self) -> bool:
        return self.enabled == "false" or self.source == "off"


def episode_seed(ep: Path, cfg: MusicCfg) -> int:
    if cfg.seed is not None:
        return cfg.seed
    return int(hashlib.sha1(ep.name.encode()).hexdigest()[:8], 16) % 100000


# --------------------------------------------------------------------------- timeline / mood


def timeline_seconds(config: dict[str, Any], voice_dur: float) -> tuple[float, float]:
    """``(lead, total)``: intro-card length and the full rendered duration (as Stage 6 does)."""
    from src.stage6_render import RenderSettings

    s = RenderSettings.from_config(config)
    fps = s.fps
    if s.branding:
        lead = round(s.intro_seconds * fps) / fps
        tail = round(s.outro_seconds * fps) / fps
        return lead, round(voice_dur * fps) / fps + lead + tail
    hold = s.outro_hold if s.outro else 0.0
    return 0.0, round((voice_dur + hold) * fps) / fps


def auto_mood(plan: dict[str, Any] | None) -> str:
    """Mood from the share of beat styles: lots of hype/fail -> upbeat, mostly plain -> calm."""
    beats = (plan or {}).get("beats") or []
    if not beats:
        return "tense"
    n = len(beats)
    share = {k: sum(1 for b in beats if b.get("beat_style") == k) / n
             for k in ("hype", "fail", "technical", "normal", "deadpan")}
    if share["hype"] + share["fail"] >= 0.3:
        return "upbeat"
    if share["hype"] + share["fail"] < 0.08 and share["technical"] < 0.3:
        return "calm"
    return "tense"


# --------------------------------------------------------------------------- audio io


def decode(path: Path, sr: int = SR) -> np.ndarray:
    """Any audio file -> float32 ``(n, 2)`` at ``sr`` (ffmpeg)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar",
                        str(sr), "-f", "f32le", "-"], capture_output=True, check=False,
                       timeout=300)
    if r.returncode != 0 or not r.stdout:
        raise ValueError(f"cannot decode {path.name}")
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).copy()


def write_wav(path: Path, data: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def measure_lufs(path: Path, window: tuple[float, float] | None = None) -> float | None:
    """Integrated LUFS (ebur128) of a file, optionally of ``(start, duration)`` only."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats"]
    if window:
        cmd += ["-ss", f"{window[0]:.2f}", "-t", f"{window[1]:.2f}"]
    cmd += ["-i", str(path), "-af", "ebur128", "-f", "null", "-"]
    try:
        err = subprocess.run(cmd, capture_output=True, text=True, check=False,
                             timeout=300).stderr or ""
    except (subprocess.SubprocessError, OSError):
        return None
    m = re.findall(r"\bI:\s+(-?[0-9.]+) LUFS", err[err.rfind("Summary:"):])
    return float(m[-1]) if m else None


def loop_to(x: np.ndarray, n: int, fade: int) -> np.ndarray:
    """Repeat ``x`` to ``n`` samples with equal-power cross-fades between repeats."""
    if len(x) >= n:
        return x[:n].copy()
    fade = min(fade, len(x) // 3)
    out = x.copy()
    t = np.linspace(0.0, np.pi / 2, fade, dtype=np.float32)[:, None]
    while len(out) < n:
        out[-fade:] = out[-fade:] * np.cos(t) + x[:fade] * np.sin(t)
        out = np.concatenate([out, x[fade:]])
    return out[:n]


def normalise(x: np.ndarray, target_lufs: float, tmp: Path) -> tuple[np.ndarray, float | None]:
    """Gain ``x`` so its integrated loudness is ``target_lufs`` (peak-safe); returns (x, LUFS)."""
    write_wav(tmp, x / max(float(np.max(np.abs(x))), 1e-6) * 0.5)
    cur = measure_lufs(tmp)
    if cur is None:
        return x, None
    y = x / max(float(np.max(np.abs(x))), 1e-6) * 0.5 * 10 ** ((target_lufs - cur) / 20)
    peak = float(np.max(np.abs(y)))
    if peak > 0.95:
        y *= 0.95 / peak
    return y.astype(np.float32), target_lufs


# --------------------------------------------------------------------------- licences / sources


def read_sidecar(path: Path) -> dict[str, str]:
    """``key: value`` lines of ``<file>.txt`` or ``<stem>.txt`` (license, author, url, title)."""
    for cand in (Path(str(path) + ".txt"), path.with_suffix(".txt")):
        if cand.is_file():
            out = {}
            for line in cand.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.match(r"^\s*([A-Za-z_]+)\s*[:=]\s*(.+?)\s*$", line)
                if m:
                    out[m.group(1).lower()] = m.group(2)
            return out
    return {}


def is_free_license(lic: str | None) -> bool:
    """CC0 / public domain only (CC-BY, CC-BY-SA, NC, ND and unknown licences never pass)."""
    return bool(lic) and re.sub(r"\s+", " ", str(lic).strip().lower()) in FREE_LICENSES | {
        "cc-0", "cc zero", "pd-self", "public-domain", "publicdomain", "dominio publico"}


def _local_track(folder: Path, source: str, seed: int, cfg: MusicCfg) -> dict[str, Any] | None:
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXT) \
        if folder.is_dir() else []
    ok: list[tuple[Path, dict[str, str]]] = []
    for p in files:
        side = read_sidecar(p)
        lic = side.get("license") or side.get("licence")
        if cfg.strict_cc0_only and not is_free_license(lic):
            console.print(f"  [yellow]music: {p.name} skipped (strict_cc0_only, licence "
                          f"{lic or 'undeclared'})[/]")
            continue
        ok.append((p, side))
    if not ok:
        return None
    p, side = ok[seed % len(ok)]
    lic = side.get("license") or side.get("licence") or "undeclared (user-supplied)"
    return {"source": source, "path": str(p), "title": side.get("title") or p.stem,
            "author": side.get("author") or side.get("creator") or "", "license": lic,
            "license_url": side.get("license_url") or "", "url": side.get("url") or "",
            "free_license": is_free_license(lic)}


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)


def _musicish(r: dict[str, Any]) -> bool:
    text = (str(r.get("title", "")) + " " + " ".join(
        t.get("name", "") for t in r.get("tags") or [])).lower()
    return any(k in text for k in _MUSICISH) and not any(k in text for k in _NOT_MUSIC)


def _download(client: httpx.Client, url: str, dest: Path) -> bool:
    from src.stock_video import download_capped

    return dest.is_file() or download_capped(client, url, dest, MAX_DOWNLOAD)


def openverse_candidates(client: httpx.Client, mood: str, cfg: MusicCfg) -> list[dict[str, Any]]:
    """CC0 / PDM audio >= 60 s from Openverse (music category when tagged, else music-ish tags)."""
    licenses = ["cc0"] if cfg.strict_cc0_only else ["cc0", "pdm"]
    found: dict[str, dict[str, Any]] = {}
    for q in _QUERIES[mood]:
        for lic in licenses:  # comma-joined values return nothing on the audio endpoint
            try:
                resp = client.get(OPENVERSE_API, params={
                    "q": q, "license": lic, "page_size": 20, "duration": "medium,long",
                    "mature": "false"})
                results = resp.json().get("results", []) if resp.status_code == 200 else []
            except (httpx.HTTPError, ValueError):
                continue
            for r in results:
                dur = (r.get("duration") or 0) / 1000
                if (r.get("license") or "").lower() not in {"cc0", "pdm"} or dur < MIN_SECONDS:
                    continue
                if r.get("category") not in (None, "music") or not _musicish(r) or not r.get("url"):
                    continue
                found[r["id"]] = {
                    "source": "openverse", "id": r["id"], "url": r["url"], "title": r.get("title"),
                    "author": r.get("creator") or "", "license": r["license"].upper(),
                    "license_url": r.get("license_url") or "",
                    "landing_url": r.get("foreign_landing_url") or "", "duration": dur,
                    "provider": r.get("source") or "", "free_license": True}
    return sorted(found.values(), key=lambda c: c["id"])


def archive_candidates(client: httpx.Client, mood: str) -> list[dict[str, Any]]:
    """Public-domain / CC0 audio items of the Internet Archive (licenseurl must match exactly)."""
    from src.stock_video import is_archive_pd

    out: list[dict[str, Any]] = []
    subj = {"tense": "ambient", "calm": "ambient", "upbeat": "electronic"}[mood]
    try:
        q = (f'mediatype:audio AND subject:{subj} AND (licenseurl:*publicdomain* OR '
             'licenseurl:*zero*)')
        rows = client.get("https://archive.org/advancedsearch.php", params={
            "q": q, "fl[]": "identifier", "rows": 15, "output": "json"}).json()
        ids = [d["identifier"] for d in rows["response"]["docs"]]
    except (httpx.HTTPError, ValueError, KeyError):
        return out
    for ident in ids:
        try:
            meta = client.get(f"https://archive.org/metadata/{ident}").json()
        except (httpx.HTTPError, ValueError):
            continue
        md = meta.get("metadata", {})
        if not is_archive_pd(md.get("licenseurl")):
            continue
        for f in meta.get("files", []):
            if f.get("format") in ("VBR MP3", "MP3") and float(f.get("length") or 0) >= MIN_SECONDS:
                out.append({"source": "archive.org", "id": f"ia_{ident}", "title": md.get("title"),
                            "url": f"https://archive.org/download/{ident}/{f['name']}",
                            "author": str(md.get("creator") or ""), "license": "public domain",
                            "license_url": md["licenseurl"].strip(),
                            "landing_url": f"https://archive.org/details/{ident}",
                            "duration": float(f["length"]), "free_license": True})
                break
    return out


def commons_candidates(client: httpx.Client, mood: str) -> list[dict[str, Any]]:
    """Wikimedia Commons audio whose licence is exactly CC0 / public domain."""
    from src.stage4_fetch_stock import _WIKI_PD

    out: list[dict[str, Any]] = []
    try:
        data = client.get("https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
            "gsrsearch": f"filetype:audio {_QUERIES[mood][0]}", "gsrlimit": 15,
            "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size"}).json()
        pages = list((data.get("query") or {}).get("pages", {}).values())
    except (httpx.HTTPError, ValueError):
        return out
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        lic = (meta.get("LicenseShortName", {}).get("value") or "").strip()
        if not _WIKI_PD.match(lic.lower()) or not info.get("url"):
            continue
        if info.get("size", 0) < 1_000_000:  # ~ >1 min at modest bitrates; real length checked later
            continue
        out.append({"source": "wikimedia", "id": f"wm_{page.get('pageid')}",
                    "title": page.get("title"), "url": info["url"], "license": lic,
                    "author": re.sub(r"<[^>]+>", "", str(meta.get("Artist", {}).get("value", ""))),
                    "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                    "landing_url": info.get("descriptionurl", ""), "free_license": True})
    return out


def fetch_online(mood: str, seed: int, cfg: MusicCfg, need: float) -> dict[str, Any] | None:
    """First downloadable, decodable CC0/PD track (Openverse, then Archive, then Commons)."""
    if is_offline() or not cfg.openverse:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with _client() as client:
            for finder in (lambda: openverse_candidates(client, mood, cfg),
                           lambda: archive_candidates(client, mood),
                           lambda: commons_candidates(client, mood)):
                cands = [c for c in finder() if is_free_license(c["license"])
                         and not (cfg.strict_cc0_only and c["license"].lower() not in
                                  ("cc0", "cc0 1.0"))]
                random.Random(seed).shuffle(cands)
                for c in cands[:6]:
                    dest = CACHE_DIR / f"{hashlib.sha1(c['id'].encode()).hexdigest()[:16]}.audio"
                    if not _download(client, c["url"], dest):
                        continue
                    dur = probe_duration(dest) or 0.0
                    if dur < MIN_SECONDS:
                        continue
                    return {**c, "path": str(dest), "duration": dur}
    except (httpx.HTTPError, OSError) as e:
        console.print(f"  [yellow]music: online search failed ({e}); using the synthesizer[/]")
    return None


# --------------------------------------------------------------------------- audit


def write_audit(ep: Path, info: dict[str, Any]) -> None:
    """work/music_audit.json, merged into out/licenses_audit.json under ``music``."""
    entry = {k: info.get(k) for k in ("source", "title", "author", "license", "license_url",
                                      "url", "landing_url", "provider", "id", "path",
                                      "free_license", "synth_seed", "mood", "bpm")
             if info.get(k) not in (None, "")}
    write_json(ep / "work" / "music_audit.json", entry)
    out = ep / "out" / "licenses_audit.json"
    try:
        data = read_json(out) if out.is_file() else {}
        if not isinstance(data, dict):
            return
        data["music"] = entry
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, data)
    except (OSError, ValueError):
        pass


# --------------------------------------------------------------------------- stage


def signature(ep: Path, cfg: MusicCfg, total: float) -> str:
    files = []
    for folder in (ep / "raw" / "music", PROJECT_ROOT / "assets" / "music"):
        if folder.is_dir():
            files += [f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}"
                      for p in sorted(folder.glob("*")) if p.is_file()]
    key = json.dumps([cfg.source, cfg.mood, cfg.level_lufs, cfg.seed, cfg.strict_cc0_only,
                      cfg.openverse, round(total, 1), files], sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def is_enabled(cfg: MusicCfg) -> bool:
    return not cfg.off


def stale_reason(ep: Path, work: Path) -> str | None:
    """Why Stage 5b must (re)run, or None (up to date / disabled / nothing to mix)."""
    config = load_config()
    cfg = MusicCfg.from_config(config)
    voice = voice_path(ep)
    if cfg.off or not voice.is_file():
        return None
    art, meta = work / "music.wav", work / "music.json"
    if not art.is_file() or not meta.is_file():
        return "no music track"
    if voice.stat().st_mtime > art.stat().st_mtime:
        return "voiceover newer than music"
    dur = probe_duration(voice) or 0.0
    try:
        if read_json(meta).get("signature") != signature(ep, cfg, timeline_seconds(config, dur)[1]):
            return "music settings/sources changed"
    except (OSError, ValueError):
        return "music.json is corrupt"
    return None


def _pick(ep: Path, cfg: MusicCfg, mood: str, seed: int, total: float,
          source: str) -> tuple[dict[str, Any] | None, str]:
    order = {"auto": ["episode", "library", "openverse", "synth"], "episode": ["episode"],
             "library": ["library"], "openverse": ["openverse"], "synth": ["synth"]}[source]
    for name in order:
        if name == "episode":
            t = _local_track(ep / "raw" / "music", "episode", seed, cfg)
        elif name == "library":
            t = _local_track(PROJECT_ROOT / "assets" / "music", "library", seed, cfg)
        elif name == "openverse":
            t = fetch_online(mood, seed, cfg, total)
        else:
            return None, "synth"
        if t:
            return t, name
        console.print(f"  [dim]music: no usable track from {name}[/]")
    return None, "synth"


def build(ep: Path, cfg: MusicCfg, source: str | None = None, mood: str | None = None,
          ) -> dict[str, Any]:
    """Create work/music.wav + work/music.json (+ audit). Returns the music.json record."""
    config = load_config()
    work = ep / "work"
    work.mkdir(parents=True, exist_ok=True)
    if source:
        cfg.source = source
    if mood:
        cfg.mood = mood
    voice = voice_path(ep)
    if not voice.is_file():
        raise FileNotFoundError(f"voice-over not found: {voice}")
    vdur = probe_duration(voice) or 0.0
    lead, total = timeline_seconds(config, vdur)
    try:
        plan = read_json(work / "scene_plan.json")
    except (OSError, ValueError):
        plan = None
    mood_name = cfg.mood if cfg.mood != "auto" else auto_mood(plan)
    seed = episode_seed(ep, cfg)
    track, _ = _pick(ep, cfg, mood_name, seed, total, cfg.source)
    n = int(total * SR)
    info: dict[str, Any]
    if track is not None:
        try:
            audio = loop_to(decode(Path(track["path"])), n, 2 * SR)
            t = np.arange(n) / SR
            audio = audio * (np.minimum(t / 3.0, 1.0)
                             * np.clip((total - t) / 5.0, 0.0, 1.0))[:, None].astype(np.float32)
            info = dict(track)
        except (ValueError, OSError) as e:
            console.print(f"  [yellow]music: {e}; using the synthesizer[/]")
            track = None
    if track is None:
        spec = music_synth.make_spec(seed, mood_name)
        audio = music_synth.synth(total, seed, mood_name, SR, spec)
        info = {"source": "synth", "title": f"psycho_terminal synth ({mood_name})",
                "author": "videoMaker procedural synthesizer", "license": "original (generated)",
                "free_license": True, "synth_seed": seed, "bpm": spec.bpm}
    audio, lufs = normalise(audio, cfg.level_lufs, work / ".music_tmp.wav")
    (work / ".music_tmp.wav").unlink(missing_ok=True)
    write_wav(work / "music.wav", audio)
    info.update({"mood": mood_name, "duration": round(total, 3), "lead": lead,
                 "level_lufs": lufs, "signature": signature(ep, cfg, total)})
    write_json(work / "music.json", info)
    write_audit(ep, info)
    console.print(f"  Music: {info['source']} / {mood_name} / {info['license']} -> "
                  f"{work / 'music.wav'} ({total:.1f}s, {lufs} LUFS)")
    return info


# --------------------------------------------------------------------------- mixing


def mix_filter(cfg: MusicCfg, music_in: str, key_in: str, out: str, lead: float, total: float,
               stem: str | None = None) -> str:
    """Filter chain: music (fade in/out, intro ramp, pause boost) ducked by the voice ``key_in``.

    ``music_in`` is the music input label (e.g. ``2:a``); ``out`` the ducked music label.
    The compressor threshold sits ~12 dB below typical speech, so ``ratio`` is chosen so that
    speech lowers the music by about ``duck_db`` (attack 20 ms, release 400 ms).
    """
    ratio = min(max(1.0 / max(1.0 - cfg.duck_db / 12.0, 0.05), 1.0), 20.0)
    fin, fout = 3.0, min(4.0, max(total / 4, 1.0))
    ramp = (f"volume='min(1,{cfg.intro_gain:.3f}+{1 - cfg.intro_gain:.3f}*t/{lead:.3f})':eval=frame,"
            if lead > 0 and cfg.intro_gain < 1 else "")
    boost = 10 ** (cfg.volume_in_pauses / 20)
    chain = (f"[{music_in}]aresample={SR},aformat=channel_layouts=stereo,{ramp}"
             f"volume={boost:.3f},afade=t=in:st=0:d={fin:.2f},"
             f"afade=t=out:st={max(total - fout, 0):.3f}:d={fout:.2f}[mm];"
             # the key is levelled first (hard compression) so the ducking depth does not depend
             # on how loud each phrase is; the music compressor then sits ~12 dB below its level (-35 dB)
             f"[{key_in}]acompressor=threshold=0.0316:ratio=20:attack=5:release=120:makeup=1[kk];"
             f"[mm][kk]sidechaincompress=threshold=0.002:ratio={ratio:.2f}:attack=20:"
             f"release=400:makeup=1:level_sc=1")
    return chain + (f",asplit=2[{out}][{stem}]" if stem else f"[{out}]")


def mix_preview(ep: Path, cfg: MusicCfg, seconds: float = 20.0) -> Path:
    """work/music_preview.wav: ``seconds`` of voice (loudnormed to -16 LUFS) + ducked music."""
    work = ep / "work"
    info = read_json(work / "music.json")
    lead = float(info.get("lead") or 0.0)
    voice = voice_path(ep)
    vdur = probe_duration(voice) or seconds
    # the densest speech window: start where the voice begins (after the intro card)
    start_v = 0.0
    dur = min(seconds, vdur)
    out = work / "music_preview.wav"
    fc = (f"[0:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample={SR},adelay={round(lead * 1000)}:all=1,"
          f"apad=whole_dur={lead + dur:.3f},atrim=0:{lead + dur:.3f},asplit=2[vo][vk];"
          + mix_filter(cfg, "1:a", "vk", "md", lead, float(info["duration"])) + ";"
          "[vo][md]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95:level=disabled"
          "[aout]")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start_v:.2f}",
           "-t", f"{dur:.2f}", "-i", str(voice), "-i", str(work / "music.wav"),
           "-filter_complex", fc, "-map", "[aout]", "-t", f"{lead + dur:.2f}", "-ar", str(SR),
           "-ac", "2", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"preview mix failed: {r.stderr[-300:]}")
    return out


# --------------------------------------------------------------------------- QA


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    return data.reshape(-1, ch).mean(axis=1), sr


def qa_music(voice_stem: Path, music_stem: Path, final_lufs: float | None = None,
             max_delta_db: float = -12.0) -> dict[str, Any]:
    """Speech intelligibility: per 100 ms frame where the voice speaks, music dB - voice dB."""
    v, sr = _read_mono(voice_stem)
    m, _ = _read_mono(music_stem)
    fr = sr // 10  # 100 ms frames
    n = min(len(v), len(m)) // fr

    def frame_db(x: np.ndarray) -> np.ndarray:
        return 10 * np.log10(np.mean(x[: n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)

    v_db, m_db = frame_db(v), frame_db(m)
    # speech = frames within 15 dB of the voice's typical level (room noise between words and
    # silent pauses do not count)
    ref = float(np.percentile(v_db, 90)) if n else 0.0
    speech = (v_db > ref - 15) & (v_db > -45) if n else np.zeros(0, bool)
    deltas = (m_db - v_db)[speech].tolist()
    rep: dict[str, Any] = {
        "voice_lufs": measure_lufs(voice_stem), "music_lufs": measure_lufs(music_stem),
        "final_lufs": final_lufs, "speech_blocks": len(deltas),
        "music_vs_voice_db_mean": round(float(np.mean(deltas)), 1) if deltas else None,
        "music_vs_voice_db_p95": round(float(np.percentile(deltas, 95)), 1) if deltas else None,
        "max_allowed_db": max_delta_db}
    rep["passed"] = not deltas or rep["music_vs_voice_db_p95"] <= max_delta_db
    return rep


def run(episode_id: str, work_dir: Path, source: str | None = None, mood: str | None = None,
        preview: bool = False) -> dict[str, Any]:
    """Stage 5b entry point (``music <ep> [--source] [--mood] [--preview]``)."""
    from src.common import episode_dir

    ep = episode_dir(episode_id)
    cfg = MusicCfg.from_config(load_config())
    if (cfg.off and not source) or source == "off":
        console.print("  music disabled (music.enabled: false / source: off)")
        return {"source": "off"}
    info = build(ep, cfg, source, mood)
    if preview:
        p = mix_preview(ep, cfg)
        console.print(f"  Preview (voice + music, 20 s): {p}")
        info["preview"] = str(p)
    return info
