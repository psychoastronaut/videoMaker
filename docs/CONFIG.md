# Configuration

`config/default.yaml` holds every default. An episode `config.yaml` (and `--config path.yaml`)
is deep-merged over it, so list only overrides. Environment variables: `LLM_BASE_URL`,
`LLM_MODEL`, `LLM_NATIVE_API`, `VIDEOMAKER_OFFLINE=1` (no network/LLM: deterministic planner),
and the API keys below.

| Section | Key settings (default) |
|---|---|
| `theme` | `psycho_terminal` (neon terminal look) or `classic` (blue). Re-run `cutouts` after switching |
| `pipeline` | `resolution` 1920x1080, `fps` 30, `video_codec: h264_nvenc` (auto x264 fallback), `crf` 18 (x264 -crf / NVENC -cq), `preset` p4, `x264_preset` medium, audio AAC 192k |
| `llm` | `base_url` `http://localhost:11434/v1`, `model` `qwen2.5:14b-instruct` (use `qwen2.5:7b-instruct` on 8 GB), `structured: true` (Ollama `/api/chat` JSON schema), `num_ctx` 4096, `chunk_beats` 5, `timeout` 150, `total_timeout` 300; optional `query_model` / `pick_model` |
| `staging` | `cut_pacing`, `beat_pacing` (min 1.9 s, max 6.0 s, per-style durations, `tone_keywords`), `pool_stock_ratio` 0.45, `stock_ratio` 0.25, `entities.keywords` (add your topic words) |
| `semantic` | `backend: auto|siglip|ollama|none`, `global_assignment`, `reuse_window` 22, `reuse_caps`, `diversity.*`, `min_clip_share` 0.25, `top_k` 5, `min_score`, `verify_stock`, `siglip.model` (`google/siglip2-base-patch16-224`; so400m for more VRAM), `ollama.caption_model`, `ollama.embed_model`; `variety` and `callouts.max_ratio` tune the template mix and labels |
| `transcription` | `model_size: small` (try `medium` with 12 GB), `language: en`, VAD options, `model_dir` |
| `assets` | Supported extensions per type |
| `stock_media` | `enabled`, `strict_cc0_only`, `allow_platform_licenses`, `video.*` (min_height 720, max_file_mb 80, max_clip_seconds 12), `prefetch.*` (max_items 24, thematic_items 16), `topup.*`, `min_resolution` 1280, `accepted_openverse_licenses: [cc0, pdm]` |
| `cutouts` | `models` fallback chain, `max_cutout_height_ratio` 0.45, font sizes |
| `render` | `crt: off|subtle|strong`, `branding` (boot intro/outro cards), `intro_seconds`, `outro_seconds`, `glitch_transitions`, `hud`, `transitions` + `transition_duration` 0.35, `workers` 3, `loudness_target` -16, `generate_srt`, `split_layout`, motion/punch zoom values |
| `audio` | `enabled: auto|true|false`, `preset: natural|podcast|gentle|aggressive` (natural), `target_lufs` -16, `max_pause` 0.8, `keep_pause` 0.35, `hum_hz` 60 (only used when no analysis is available), `denoise: auto\|rnnoise\|afftdn\|off`, `denoise_strength` 0.6, `notch: auto\|off\|[Hz]`, `expander_db` 14, `expander_margin_db` 12, `rnnoise_model` (default: auto-download of `sh.rnnn`, see below) |
| `music` | `enabled: auto|true|false`, `source: auto|episode|library|openverse|synth|off`, `mood`, `level_lufs` -29, `duck_db` 6, `carve_db` -4 (`carve_lo` 200 / `carve_hi` 3000 Hz), `hold` 2, `attack` 0.25, `release` 2.5, `intro_gain` 0.35, `volume_in_pauses` 0, `seed`, `openverse` |
| `stabilize` | `enabled: auto|true|false`, `threshold` 1.25, `smoothing` 15, `optzoom` |
| `upscale` | `enabled: auto|true|false`, `min_long_side` 1400, `max_long_side` 1920, `max_factor` 2.0, `blend` 0.7, `tile` 256 |

Recommended starting points: 12 GB GPU keep defaults; 8 GB use a 7b LLM, `semantic.siglip.device: auto`,
`render.workers: 2`; no GPU set `pipeline.video_codec: libx264` and `transcription.model_size: small`;
local-only `stock_media.enabled: false`; legally cautious `stock_media.strict_cc0_only: true`.

## Keys (`.env`)

Copy `.env.example` to `.env` (git-ignored, read automatically). Each key may also be set in the
environment or under `stock_media.*` in the config.

| Variable | Used for |
|---|---|
| `OPENVERSE_API_KEY` | Optional; raises Openverse rate limits |
| `PEXELS_API_KEY` | Pexels photos/videos (platform license, non-CC0) |
| `PIXABAY_API_KEY` | Pixabay photos/videos (platform license, non-CC0) |
| `SMITHSONIAN_API_KEY` | Smithsonian Open Access (CC0); free at api.data.gov, otherwise a rate-limited shared demo key is used |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_NATIVE_API` | Point at another OpenAI-compatible server/model |
| `VIDEOMAKER_OFFLINE` | `1` disables network and LLM use |

NASA, Wikimedia Commons and the Internet Archive need no key.

## Audio credits: RNNoise model

`audio.denoise: auto|rnnoise` uses `arnndn` (FFmpeg) with the `somnolent-hogwash` model
(`sh.rnnn`) from https://github.com/GregorR/rnnoise-models, downloaded once into
`~/.cache/videomaker/models` (skipped when `VIDEOMAKER_OFFLINE` is set or there is no network;
the chain then uses afftdn only). The repository states its models are not creative work and
claim no copyright (its tools are separate); RNNoise itself is BSD-3-Clause (Xiph.Org / Jean-Marc
Valin). Set `audio.rnnoise_model` to use another `.rnnn`.
