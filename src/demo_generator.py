"""Demo episode generator: synthetic assets for an engineering/hardware episode.

The theme (a robot arm that keeps failing) mirrors the ``ep01-sample`` flavour: hardware,
schematics, janky workarounds and failure moments, so the demo exercises every template, every
beat style and at least one stock beat. Everything is generated offline with Pillow/numpy/ffmpeg.
"""

from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.common import write_json
from src.graphics import FONT_BOLD, load_font

# (text, pause_before_seconds). Word timings are derived from the text length.
SENTENCES: list[tuple[str, float]] = [
    ("I built a robot arm that is supposed to pour my coffee.", 0.0),
    ("Welcome to Psycho Astronaut, where the hardware is janky and the budget is zero.", 0.45),
    ("The brain is a tiny microcontroller board talking to three servo motors.", 0.4),
    ("A soldering iron rests beside a workbench vise and a coffee mug.", 0.4),
    ("The Tokyo skyline glows at night over a sleepy suburb.", 0.4),
    (("Every servo gets a signal from the control loop, which checks its position "
      "a thousand times per second."), 0.4),
    ("In theory, that is all you need.", 0.35),
    ("Of course, I wired the motor driver backwards.", 0.4),
    ("There was a pop, a puff of magic smoke, and the whole bench went dark.", 0.45),
    ("The battery pack had shorted, and my duct tape workaround did not survive.", 0.4),
    ("So I rebuilt the circuit from the schematic and checked every single connection.", 0.45),
    ("This time the signal timing looked clean on the scope.", 0.4),
    ("And then it finally worked.", 0.4),
    ("The arm lifted the cup, swung over, and poured perfectly.", 0.35),
    ("Okay, it spilled a little, but nobody ships on the first try.", 0.4),
    ("A CNC lathe in a factory carves a brass part for the next version.", 0.4),
    ("Next time I am adding a fuse. Thanks for watching.", 0.4),
]

WORDS_PER_SECOND = 2.9
COMMA_PAUSE = 0.16


def create_demo_episode(episode_dir: Path, work_dir: Path) -> None:
    """Generate a complete synthetic test episode (raw assets, voiceover, transcript)."""
    raw = episode_dir / "raw"
    for sub in ("clips", "images", "schematics"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    segments = _build_segments()
    duration = segments[-1]["end"] + 0.6
    _generate_voiceover(raw, segments, duration)
    _generate_script(raw)
    _generate_test_clips(raw / "clips")
    _generate_test_images(raw / "images")
    _generate_test_schematics(raw / "schematics")
    _write_transcript(raw, work_dir, segments, duration)


# --------------------------------------------------------------------------- transcript/audio


def _build_segments() -> list[dict]:
    """Segments with word-level timestamps (speech-rate model with comma pauses)."""
    segments: list[dict] = []
    t = 0.3
    for idx, (text, pause) in enumerate(SENTENCES):
        t += pause
        tokens = text.split()
        weights = [len(re.sub(r"\W", "", w)) + 2 for w in tokens]
        speech = len(tokens) / WORDS_PER_SECOND
        unit = speech / sum(weights)
        words = []
        cur = t
        for w, wt in zip(tokens, weights, strict=True):
            dur = wt * unit
            words.append({"word": w, "start": round(cur, 3), "end": round(cur + dur, 3)})
            cur += dur
            if w.endswith(","):
                cur += COMMA_PAUSE
        segments.append({"id": idx, "start": words[0]["start"], "end": words[-1]["end"],
                         "text": text, "words": words})
        t = words[-1]["end"]
    return segments


def _write_transcript(raw: Path, work_dir: Path, segments: list[dict], duration: float) -> None:
    transcript = {
        "full_text": " ".join(s["text"] for s in segments),
        "language": "en",
        "duration": round(duration, 3),
        "segments": segments,
        "cut_boundaries": [s["end"] for s in segments],
        "synthetic": True,
    }
    write_json(raw / "transcript.json", transcript)
    write_json(work_dir / "transcript.json", transcript)  # written after the wav: stays fresh
    print(f"  Generated transcript: {work_dir / 'transcript.json'}")


def _generate_voiceover(raw: Path, segments: list[dict], duration: float) -> None:
    """Speech-like audio: one voiced burst per word, silence elsewhere."""
    rate = 16000
    n = int(rate * duration)
    wave_data = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(7)
    for seg in segments:
        for i, w in enumerate(seg["words"]):
            a, b = int(w["start"] * rate), int(w["end"] * rate)
            if b <= a:
                continue
            t = np.arange(b - a) / rate
            f0 = 118 + 22 * np.sin(i * 1.7 + seg["id"])
            tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
            env = np.hanning(b - a) ** 0.6
            wave_data[a:b] += tone * env * 0.25
    wave_data += rng.normal(0, 0.002, n)
    peak = np.max(np.abs(wave_data))
    wave_data = wave_data / peak * 0.7 if peak else wave_data
    with wave.open(str(raw / "voiceover.wav"), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.int16(wave_data * 32767).tobytes())
    print(f"  Generated voiceover: {raw / 'voiceover.wav'} ({duration:.1f}s)")


def _generate_script(raw: Path) -> None:
    lines = ["# Episode: The Coffee-Pouring Robot Arm (demo)", ""]
    for i, (text, _) in enumerate(SENTENCES, 1):
        lines += [f"## Beat {i}", text, ""]
    (raw / "script.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- clips

# (filename, seconds, lavfi source, label). One clip (2.5s) is shorter than most beats so the
# renderer's looping is exercised.
CLIPS = [
    ("servo_bench_test.mp4", 7.0, "testsrc2=s=1920x1080:r=30", "SERVO BENCH TEST"),
    ("motor_driver_smoke.mp4", 6.0,
     "gradients=s=1920x1080:r=30:speed=0.04:c0=0x1a0505:c1=0xff5a1f:c2=0x2a0a0a:nb_colors=3",
     "MOTOR DRIVER - SMOKE"),
    ("workshop_timelapse.mp4", 2.5, "rgbtestsrc=s=1920x1080:r=30", "WORKSHOP TIMELAPSE"),
    ("robot_arm_demo.mp4", 7.0,
     "gradients=s=1920x1080:r=30:speed=0.03:c0=0x06223a:c1=0x2fa84f:c2=0x0b1830:nb_colors=3",
     "ROBOT ARM DEMO"),
    ("scope_trace.mp4", 6.0, "color=c=0x07130c:s=1920x1080:r=30", "SCOPE TRACE"),
]


def _generate_test_clips(clips_dir: Path) -> None:
    for filename, seconds, source, label in CLIPS:
        vf = [
            (f"drawtext=fontfile={FONT_BOLD}:text='{label}':x=(w-text_w)/2:y=h*0.16:fontsize=64:"
             "fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=18"),
        ]
        if filename.startswith("scope"):
            vf.insert(0, "drawgrid=w=120:h=120:t=2:c=0x1d5a35@0.8")
            vf.insert(1, "drawbox=x='mod(t*320,1800)':y=480:w=90:h=90:color=0x3dff8a:t=fill")
        if filename.startswith("servo"):
            vf.append("hue=h=t*20")
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", source, "-t", str(seconds),
               "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-an", str(clips_dir / filename)]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"demo clip {filename} failed: {result.stderr[-400:]}")
        print(f"  Generated clip: {filename} ({seconds}s)")


# --------------------------------------------------------------------------- images

LIGHT_BG = (232, 234, 238)


def _generate_test_images(images_dir: Path) -> None:
    """Product-photo style subjects on a flat background (so cutouts have something to cut)."""
    for name, painter in [
        ("microcontroller_board.png", _paint_board),
        ("servo_motor.png", _paint_servo),
        ("battery_pack.png", _paint_battery),
        ("duct_tape_hack.png", _paint_tape),
    ]:
        img = Image.new("RGB", (1920, 1080), LIGHT_BG)
        painter(ImageDraw.Draw(img))
        img.save(images_dir / name)
        print(f"  Generated image: {name}")


def _paint_board(d: ImageDraw.ImageDraw) -> None:
    d.rounded_rectangle([560, 320, 1360, 760], radius=24, fill=(18, 110, 70),
                        outline=(8, 60, 40), width=8)
    d.rectangle([800, 440, 1080, 640], fill=(25, 25, 30))
    d.text((830, 520), "MCU", font=load_font(70), fill=(210, 210, 215))
    for x in range(600, 1330, 50):
        d.rectangle([x, 330, x + 22, 372], fill=(210, 180, 70))
        d.rectangle([x, 708, x + 22, 750], fill=(210, 180, 70))
    d.rectangle([520, 470, 600, 610], fill=(150, 150, 160))


def _paint_servo(d: ImageDraw.ImageDraw) -> None:
    d.rounded_rectangle([640, 400, 1180, 700], radius=18, fill=(40, 70, 170),
                        outline=(20, 30, 90), width=8)
    d.rectangle([1180, 500, 1320, 600], fill=(40, 70, 170), outline=(20, 30, 90), width=8)
    d.ellipse([1240, 470, 1400, 630], fill=(235, 235, 240), outline=(90, 90, 100), width=8)
    d.ellipse([1290, 520, 1350, 580], fill=(90, 90, 100))
    d.text((720, 505), "SERVO", font=load_font(90), fill=(235, 240, 255))
    for y, c in [(560, (200, 40, 40)), (600, (240, 190, 30)), (640, (60, 60, 60))]:
        d.line([(640, y), (500, y + 60)], fill=c, width=14)


def _paint_battery(d: ImageDraw.ImageDraw) -> None:
    d.rounded_rectangle([560, 380, 1260, 700], radius=30, fill=(34, 34, 40),
                        outline=(10, 10, 12), width=8)
    d.rectangle([660, 440, 1160, 640], fill=(245, 190, 20))
    d.text((700, 500), "LiPo 11.1V", font=load_font(64), fill=(30, 30, 30))
    d.line([(1260, 480), (1480, 440)], fill=(200, 30, 30), width=18)
    d.line([(1260, 600), (1480, 640)], fill=(20, 20, 20), width=18)


def _paint_tape(d: ImageDraw.ImageDraw) -> None:
    d.ellipse([620, 330, 1060, 770], fill=(120, 124, 132), outline=(70, 72, 80), width=10)
    d.ellipse([760, 470, 920, 630], fill=LIGHT_BG, outline=(70, 72, 80), width=10)
    d.polygon([(1040, 450), (1420, 400), (1440, 470), (1060, 540)], fill=(130, 134, 142))
    d.polygon([(1060, 560), (1440, 520), (1460, 590), (1080, 650)], fill=(112, 116, 124))


# --------------------------------------------------------------------------- schematics

SCHEM_BG = (10, 10, 26)
BLUE = (74, 144, 217)
RED = (217, 74, 74)
GREEN = (84, 214, 140)
WHITE = (235, 240, 255)


def _generate_test_schematics(schem_dir: Path) -> None:
    for name, painter in [
        ("motor_driver_circuit.png", _paint_driver),
        ("control_loop_diagram.png", _paint_loop),
        ("signal_timing.png", _paint_timing),
    ]:
        art = Image.new("RGB", (1920, 1080), SCHEM_BG)
        painter(ImageDraw.Draw(art))
        # Everything (labels included) lives in the central safe area: 14% inset all round.
        sw, sh = int(1920 * 0.76), int(1080 * 0.76)
        art = art.resize((sw, sh), Image.LANCZOS)
        img = Image.new("RGB", (1920, 1080), SCHEM_BG)
        d = ImageDraw.Draw(img)
        for x in range(0, 1920, 40):
            d.line([(x, 0), (x, 1080)], fill=(22, 26, 56), width=1)
        for y in range(0, 1080, 40):
            d.line([(0, y), (1920, y)], fill=(22, 26, 56), width=1)
        img.paste(art, ((1920 - sw) // 2, (1080 - sh) // 2))
        img.save(schem_dir / name)
        print(f"  Generated schematic: {name}")


def _box(d: ImageDraw.ImageDraw, rect: list[int], text: str, color: tuple[int, int, int]) -> None:
    d.rounded_rectangle(rect, radius=18, outline=color, width=10, fill=(14, 18, 44))
    font = load_font(120)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    d.text((cx - (x1 - x0) // 2 - x0, cy - (y1 - y0) // 2 - y0), text, font=font, fill=WHITE)


def _arrow(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int],
           color: tuple[int, int, int]) -> None:
    d.line([a, b], fill=color, width=14)
    ax, ay = b
    sign = 1 if b[0] >= a[0] else -1
    d.polygon([(ax, ay), (ax - 46 * sign, ay - 28), (ax - 46 * sign, ay + 28)], fill=color)


def _paint_driver(d: ImageDraw.ImageDraw) -> None:
    _box(d, [60, 330, 800, 750], "BATTERY", RED)
    _box(d, [1120, 330, 1860, 750], "DRIVER", BLUE)
    _arrow(d, (800, 450), (1120, 450), RED)
    _arrow(d, (800, 640), (1120, 640), GREEN)
    d.text((885, 340), "VCC", font=load_font(80), fill=RED)
    d.text((885, 660), "GND", font=load_font(80), fill=GREEN)
    d.text((60, 90), "MOTOR DRIVER WIRING", font=load_font(130), fill=BLUE)


def _paint_loop(d: ImageDraw.ImageDraw) -> None:
    _box(d, [40, 300, 580, 560], "TARGET", BLUE)
    _box(d, [700, 300, 1180, 560], "PID", BLUE)
    _box(d, [1300, 300, 1880, 560], "SERVO", BLUE)
    _arrow(d, (580, 430), (700, 430), WHITE)
    _arrow(d, (1180, 430), (1300, 430), WHITE)
    d.line([(1590, 560), (1590, 820), (940, 820), (940, 560)], fill=GREEN, width=14)
    d.polygon([(940, 560), (912, 610), (968, 610)], fill=GREEN)
    d.text((720, 850), "FEEDBACK", font=load_font(110), fill=GREEN)
    d.text((40, 90), "CONTROL LOOP", font=load_font(120), fill=BLUE)


def _paint_timing(d: ImageDraw.ImageDraw) -> None:
    d.text((60, 90), "PWM SIGNAL TIMING", font=load_font(130), fill=BLUE)
    x, y_hi, y_lo = 140, 420, 700
    pts = []
    for period in range(6):
        pts += [(x, y_lo), (x, y_hi), (x + 150, y_hi), (x + 150, y_lo), (x + 300, y_lo)]
        x += 300
    d.line(pts, fill=GREEN, width=10)
    d.text((140, 800), "20 ms period | 1.5 ms pulse", font=load_font(96), fill=WHITE)
