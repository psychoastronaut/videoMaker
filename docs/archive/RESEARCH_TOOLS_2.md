# RESEARCH_TOOLS_2 (2026-11-25) - tools for process speed and output quality
Scratch/evidence: (local scratch dir) (venv vdepth = py3.12, project .venv untouched).
Stars/pushed from GitHub API today; licences of CODE and WEIGHTS listed separately. Items marked (unverified) come from memory, not checked today.

## Validated on this machine
- QA ffmpeg (ep-callout-test/final_cut.mp4, 123 s): blackdetect+freezedetect+silencedetect+ebur128 in ONE pass = 7 s. Found: 1 black gap 67.80-67.93 s, 19 "freezes" (all photo beats: n=0.003 too sensitive for slow pans; use only to flag >4 s), I=-16.2 LUFS, LRA 2.6, TP -3.3 dBFS, no silence >1.5 s. ffmpeg here has NO libvmaf (ssim/psnr present) -> VMAF needs a build; low value for a generator (no reference).
- VLM critique qwen2.5vl:3b (json mode, 2-7 s/frame, +~4 GB VRAM; another process already held ~9 GB at test time): 3 frames. Frame 45 (blurry 1960s footage): flagged "blurry/low-res" (right) but also "text overlapping" (hallucinated). Frame 80 (diagram): real defect = "Landing guidance keeps running" touches its box borders; model reported a different, wrong sentence as clipped. Verdict: 3b = NOT reliable for clipping/overlap; ok only for blur/ugly-crop triage. Use deterministic checks (PIL textbbox vs box in graphics.py) for text; try qwen2.5vl:7b (Apache-2.0, ~6 GB Q4, not pulled/tested) only for "does frame match narration" on flagged beats.
- 2.5D parallax, photo apollo11_aldrin_walking...jpg, 3 s 1080p30: (a) DepthFlow 1.0.1 (pip works on 3.12, headless OpenGL OK): ~21 s first run (model download), 2.6 s render of 90 frames; parallax visible (astronaut vs ground) but stretches edges on portrait photos -> pre-compose 16:9 first. (b) Own minimal Depth-Anything-V2-Small (transformers, fp16) + cv2.remap + NVENC pipe: load 1.9 s, depth 0.71 s, 136 MB VRAM, 3.4 s render for 90 frames. Both: /scratchpad/research3/depth_push.mp4, mini.mp4; frames in frames/push_pair.jpg, mini_pair.jpg. Visually convincing, subtle at 70 px shift; no disocclusion artefacts at that amplitude. Depth map cacheable per image.
- Interpolation: ffmpeg minterpolate mci 24->30 on 4 s 1080p = 80 s (CPU, single thread; skip). rife-ncnn-vulkan v4.6 on 96 frames 720p -> 120 frames: 13.6 s (Vulkan, MIT; interpolated-frame quality not eyeballed). `framerate` filter = instant blend, not true interpolation.
- NVENC: 6 concurrent h264_nvenc encodes ran with no session error; 6x20 s took 10.7 s vs 2.6 s each serial (15.8 s) => only ~1.5x gain (CPU decode/filter + one NVENC engine). Use 3 workers max. ffmpeg has scale_cuda, overlay_cuda, nlmeans_vulkan/opencl, arnndn, afftdn, sidechaincompress, lut3d, hqdn3d.
- Per-beat cache feasibility: stage6_render.py already renders each beat to tmp_dir/beat_NNN.mp4 in a sequential loop (L769-786) then xfades/concats. Cache = key sha256(beat JSON w/o volatile keys + frames + head + tail + RenderSettings + (path,size,mtime) of every referenced media/cutout/grid) -> reuse segment if file exists. ~40 lines + ThreadPoolExecutor(3) around _render_beat. No external tool needed (joblib/doit/DVC/Snakemake are overkill; doit/Hamilton only if the DAG grows).

## Ranked shortlist
PROCESS: 1) per-beat content-hash segment cache + 3-way parallel render (high impact, 0.5 d, low risk, own code). 2) Draft mode: render at 640x360, ultrafast/NVENC, skip xfade-heavy extras, for plan iteration (high, 0.5 d, low). 3) ffmpeg QA gate after render: black/freeze(>4 s)/silence/ebur128 + PIL text-fit check at compile time -> qa_report.json (med-high, 0.5 d, low). 4) Structured JSONL run log with per-stage/beat timings + hashes (med, 0.25 d). 5) VLM triage with qwen2.5vl:7b on flagged frames only (low-med, 1 d, risk: false positives; test first).
QUALITY: 1) Depth-parallax for stills (Depth-Anything-V2-Small + remap, or DepthFlow) as new motion in photo beats (highest visual win, 1-2 d, med risk: artefacts on thin structures; keep per-beat fallback to current perspective motion). 2) Music bed CC0 + sidechain ducking + TTS voice chain (highpass/acompressor/equalizer) (med-high, 0.5 d, low). 3) OWLv2/Grounding-DINO (Apache) open-vocab box to aim punch-ins at 'the lunar module' + PaddleOCR/EasyOCR (Apache) to avoid cropping burned-in text (med, 2 d, med; VRAM contention with Ollama -> load/unload). 4) RIFE v4.6 for slow-mo only and 24->30 clips (low-med, 1 d, licence MIT). 5) Colour match across mixed footage: ffmpeg colorbalance/curves per-beat stats + light film grain (noise filter) (med, 1 d; avoid color-matcher: GPL-3).

## Reviewed table (code licence / weights licence / stars / last push / verdict)
| Item | Licence | Stars, push | Verdict |
| DepthFlow | AGPL-3.0 (code) | 1.6k, 2026-11-13 | borrow idea/shader; AGPL: running CLI for own videos probably OK but do not bundle/redistribute (unverified legal reading) |
| Depth-Anything-V2 | code Apache-2.0; weights Small Apache-2.0, Base/Large/Giant CC-BY-NC-4.0 | 9.2k, 2026-03 | ADOPT Small only |
| Depth Pro (Apple) | weights apple-amlr (research) | - | skip |
| MiDaS/DPT-hybrid | MIT/Apache weights | - | fallback only |
| 3d-ken-burns | non-commercial research (NOASSERTION) | 1.6k | skip |
| RIFE (hzwer) / rife-ncnn-vulkan | MIT / MIT | 5.7k 2025-09 / 1.1k 2024-01 | adopt narrow (slow-mo) |
| FILM (google) | Apache-2.0 | 3.2k 2024-08 | skip (TF, heavy) |
| minterpolate | LGPL ffmpeg | - | skip (80 s/4 s) |
| Ultralytics YOLO | AGPL-3.0 | 63k | skip |
| RT-DETR (lyuwenyu) | Apache-2.0 | 5.7k | borrow if closed-set needed |
| OWLv2 / Grounding-DINO / Florence-2 | Apache-2.0 / Apache-2.0 / MIT (weights same on HF) | 10.8k (GDINO, stale 2024) | adopt OWLv2 or Florence-2 (via transformers) |
| PaddleOCR / EasyOCR / Tesseract | Apache-2.0 | 93k / 30k | adopt EasyOCR or PaddleOCR (weights Apache) |
| DeepFilterNet | MIT or Apache-2.0 code+weights | 5.1k, 2026-11 | optional; TTS already clean -> skip for now |
| RNNoise / ffmpeg arnndn, afftdn | BSD-3 | 6k | borrow via ffmpeg |
| Kokoro-82M | code Apache-2.0, weights Apache-2.0 | 10k | already in HF cache; keep as default |
| Piper | rhasspy MIT (archived 2025-08) -> piper1-gpl GPL-3; per-voice dataset licences vary | 11k | skip (GPL / voice licences) |
| pyloudnorm | MIT | 0.8k | skip (ffmpeg ebur128 enough) |
| VMAF | BSD+patent | 5.7k | skip (ffmpeg lacks libvmaf; no reference) |
| doit / Snakemake / Hamilton / joblib / DVC | MIT/MIT/Apache/BSD/Apache | 2-4k | skip; own hash cache |
| Lottie (rlottie NOASSERTION, lottie-web MIT) | mixed | 1.4k | skip; use Pillow/Skia-free graphics.py + Cairo/Pango only if kerning is a problem |
| Manim | MIT | 43k | borrow for diagram animation (later) |
| color-matcher | GPL-3 | 0.7k | skip |
| qwen2.5vl:7b / gemma3 vision | Apache-2.0 / Gemma terms (restrictive-ish) | - | qwen 7b test; skip gemma |
| Music/SFX: Freesound (CC0 filter, API key), Pixabay (own content licence, not CC0, Content-ID risk), FMA (mixed), Incompetech (CC-BY, attribution) | (unverified, from memory) | - | Freesound CC0 + Openverse CC0 audio; record licence per track in the existing audit |

## Recommended integration order
1 cache+parallel (Process 1) -> 2 draft mode -> 3 QA gate (cheap, catches the text-in-box bug seen at 80 s) -> 4 music+ducking+voice chain -> 5 depth-parallax as optional template motion (flag `render.parallax: true`, cache depth maps in work/depth/) -> 6 OWLv2+OCR-aware punch-in -> 7 RIFE slow-mo -> 8 VLM-7b triage.
Not tested: qwen2.5vl:7b, RIFE output quality, DepthFlow on 3.14, music APIs, TTS comparison (Kokoro already local).
