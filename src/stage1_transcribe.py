"""Stage 1: Voice-Over Transcription & Beat Alignment.

Uses faster-whisper (CUDA) to transcribe voiceover audio with word-level
timestamps, detect natural pauses, and compute cut boundaries.
"""

from __future__ import annotations

import json

# Monkey-patch av.open to handle metadata_errors parameter (Python 3.14 compat)
# Must be done before any other imports that might use av
import sys
from pathlib import Path

if "av" not in sys.modules:
    import av
    _orig_av_open_func = av.open
    def _patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open_func(*args, **kwargs)
    av.open = _patched_av_open

    import av.container.core
    _orig_av_core_open = av.container.core.open
    def _patched_av_core_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_core_open(*args, **kwargs)
    av.container.core.open = _patched_av_core_open

from faster_whisper import WhisperModel
from rich.console import Console

from src.common import episode_dir, load_config, probe_duration, voice_path, write_json
from src.schemas import Segment, Transcript, WordTimestamp

console = Console()


def run(episode_id: str, work_dir: Path, force: bool = False) -> dict:
    """Run Stage 1: transcribe voiceover and compute cut boundaries.

    ``force`` re-transcribes even when an up-to-date (or synthetic demo) transcript exists.
    """
    audio_path = voice_path(episode_dir(episode_id))

    if not audio_path.exists():
        raise FileNotFoundError(f"Voiceover not found: {audio_path}")

    config = load_config()
    trans_config = config.get("transcription", {})

    console.print(f"  Loading audio: {audio_path}")
    duration = probe_duration(audio_path)
    if not duration:
        raise ValueError(f"Could not read audio duration (unsupported/corrupt file?): {audio_path}")
    console.print(f"  Audio duration: {duration:.1f}s")

    output_path = work_dir / "transcript.json"

    # Reuse an existing transcript (e.g. from the demo generator or a manual fix) unless forced.
    # A synthetic (demo) transcript is always kept: its audio is not real speech.
    if output_path.exists() and not force:
        with open(output_path) as f:
            existing = json.load(f)
        fresh = output_path.stat().st_mtime >= audio_path.stat().st_mtime
        if existing.get("segments") and (fresh or existing.get("synthetic")):
            console.print(f"  [dim]Transcript up to date ({len(existing['segments'])} segments), "
                          "skipping transcription (use `transcribe --force` to redo)[/]")
            return existing

    model_size = trans_config.get("model_size", "small")
    model_dir = Path(trans_config.get("model_dir", "~/.cache/videomaker/whisper")).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Try CUDA first, fall back to CPU if it is unavailable or runs out of memory.
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16",
                             download_root=str(model_dir))
        console.print(f"  Loaded Whisper model: {model_size} (CUDA)")
    except Exception as e:  # noqa: BLE001 - ctranslate2 raises RuntimeError/ValueError variants
        console.print(f"  [yellow]CUDA unavailable ({str(e)[:80]}), falling back to CPU[/]")
        model = WhisperModel(model_size, device="cpu", compute_type="int8",
                             download_root=str(model_dir))
        console.print(f"  Loaded Whisper model: {model_size} (CPU)")

    segments_out, info = model.transcribe(
        str(audio_path),
        language=trans_config.get("language", "en"),
        beam_size=trans_config.get("beam_size", 5),
        vad_filter=trans_config.get("vad_filter", True),
        vad_parameters={
            "min_silence_duration_ms": trans_config.get("min_silence_duration_ms", 600),
            "threshold": trans_config.get("vad_threshold", 0.4),
        } if trans_config.get("vad_filter", True) else None,
        word_timestamps=True,
        temperature=trans_config.get("temperature", 0.0),
    )

    segments: list[Segment] = []
    all_words: list[WordTimestamp] = []
    word_counter = 0

    for seg_idx, seg in enumerate(segments_out):
        words = []
        for w in seg.words:
            wt = WordTimestamp(word=w.word, start=w.start, end=w.end)
            words.append(wt)
            all_words.append(wt)
            word_counter += 1

        segment = Segment(
            id=seg_idx,
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=words,
        )
        segments.append(segment)

    console.print(f"  Transcribed {len(segments)} segments, {word_counter} words")

    # Detect natural pauses for cut boundaries
    cut_boundaries = _compute_cut_boundaries(segments, config)

    transcript = Transcript(
        full_text=" ".join(s.text for s in segments),
        language=info.language or "en",
        duration=duration,
        segments=segments,
        cut_boundaries=cut_boundaries,
    )

    write_json(output_path, transcript.model_dump(mode="json"))

    console.print(f"  Transcript saved: {output_path}")
    console.print(f"  Cut boundaries: {[f'{t:.1f}s' for t in cut_boundaries]}")

    return transcript.model_dump(mode="json")


def _compute_cut_boundaries(segments: list[Segment], config: dict) -> list[float]:
    """Compute cut boundaries based on segment timing and word gaps."""
    pacing = config.get("staging", {}).get("cut_pacing", {})
    min_pace = pacing.get("min_seconds", 3.0)

    # Detect natural pauses from word-level timestamps (gaps > 0.3s indicate pauses)
    pause_points: list[float] = []
    for seg in segments:
        words = seg.words
        for i in range(1, len(words)):
            gap = words[i].start - words[i - 1].end
            if gap > 0.3:
                pause_points.append(round((words[i - 1].end + words[i].start) / 2, 2))

    boundaries: list[float] = []
    for seg in segments:
        # Segment end is the natural cut; snap to the nearest pause within 1s for a clean cut.
        cut_time = seg.end
        if pause_points:
            nearest = min(pause_points, key=lambda p: abs(p - cut_time))
            if abs(nearest - cut_time) < 1.0:
                cut_time = nearest
        boundaries.append(round(cut_time, 2))

    # Enforce minimum spacing; the final boundary (end of speech) is always kept.
    filtered: list[float] = []
    for i, b in enumerate(boundaries):
        is_last = i == len(boundaries) - 1
        if not filtered or b - filtered[-1] >= min_pace:
            filtered.append(b)
        elif is_last:
            filtered[-1] = b
    return filtered
