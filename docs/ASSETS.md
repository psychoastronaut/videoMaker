# Assets

## Folders

```
episodes/<ep>/raw/
  voiceover.wav          # also .mp3/.m4a/.flac; fixed name voiceover.*
  script.md | script.txt # optional; feeds entity detection and stock prefetch
  clips/                 # video: .mp4 .mov .avi .mkv .webm   (id: clip_<stem>)
  images/                # photos: .png .jpg .jpeg .webp .bmp (id: image_<stem>)
  schematics/            # diagrams: .png .jpg .jpeg .webp    (id: schematic_<stem>)
```

Only `clips`, `images` and `schematics` are scanned, non-recursively. Other folders (for
example `raw/bench/` created by `new --template diy`) are ignored. Asset ids are
`<type>_<file stem>`; inspect the exact ids with `inspect <ep> assets`. Descriptive file names
help matching (`servo_burned_driver.jpg`, not `IMG_0042.jpg`). Clips need no audio unless you use `audio:`.

## Sidecars

A sidecar is `<file>.<ext>.txt` next to the asset (`servo_test.mp4.txt`). Free text lines at the
top describe what is on screen (this feeds the matcher). Lines of the form `key: value` are
hints. Keep only what you need. The authoritative annotated template is
`raw/_TEMPLATE.txt.example` (written by `new`):

```
Close-up of a servo sweeping 0-180 degrees on the bench, wires visible, no text on screen.

kind: screen            # screen|tablet|face -> picture-in-picture candidate; terminal -> wider split layout
pip_for: clip_main_demo # asset id this screen/face clip may accompany (or `pip: <asset id>` on the main asset)
audio: ambient          # speech (duck under voice) | ambient (keep) | none (mute); default mute
media_start: 2.5        # seconds: skip a title card / clapper
media_end: 40           # seconds: ignore the tail
avoid: 0-3, 55-60       # ranges never shown (burned-in captions, studio intro)
segments: 5-12, 20-28   # ONLY these ranges are good (clips)
highlight: 0.1,0.2,0.3,0.2 | Servo horn | 1.5   # x,y,w,h (fractions, or px at 1920x1080) | label | start s
arrow: center | Signal wire | 3.0               # same, drawn as an arrow (regions: center, top-left, ...)
state: ideal            # ideal|flawed|before|after (a flawed diagram is not shown for "correct" wording)
text_px: 18             # diagrams: text height in px for the split_schematic legibility check (default 3 % of height)
```

| Key | Applies to | Effect |
|---|---|---|
| free text | all | Description used for semantic/keyword matching |
| `kind` | clips, images | `screen`/`tablet`/`face` make PiP candidates; `terminal` selects `render.split_layout.terminal` (912 px left panel) |
| `pip_for` / `pip` | PiP assets | Links a screen/face clip to its main asset |
| `audio` | clips | `speech` ducks the narration, `ambient` mixes the clip audio, `none` mutes (maps to beat `clip_audio` duck/on/off) |
| `media_start` / `media_end` | clips | Trim window in seconds |
| `avoid` | clips | Comma-separated `a-b` ranges that are never shown |
| `segments` | clips | Only these ranges are used |
| `highlight` / `arrow` | diagrams, photos | Pulsing box/arrow with a label; the next highlight is used on each reuse |
| `state` | diagrams | `ideal`, `flawed`, `before`, `after`; penalizes showing the wrong state for the narration |
| `text_px` | diagrams | Text height used by the legibility rule (scale >= 0.5 and text >= 14 px) |

## Good practice

- Describe the picture, not the story: "wide shot of a CNC lathe cutting aluminium".
- 25-40 distinct assets for a 3-5 minute video; the planner caps reuse (photo 3, diagram 2, clip 3).
- Photos below 1400 px long side are upscaled if the `[upscale]` extra is installed.
- Use `avoid:` for burned-in captions and letterbox bars are cropped automatically.
- Diagrams: large (>= 1920 px wide), high contrast, text >= 18 px; otherwise they are not shown in a split.
- Only use media you own or that is CC0/public-domain/licensed for your use.
- Large media is git-ignored by default (`episodes/*/raw/**/*.mp4`, images); sidecars are kept.
