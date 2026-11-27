# Usage

All commands run from the repo root. `--config` is a global option and goes BEFORE the command.
An episode `config.yaml` is deep-merged over `config/default.yaml` (list only overrides).

```bash
python -m src.cli --config episodes/<ep>/config.yaml <command> <ep>
```

## 1. Create the episode

```bash
python -m src.cli new --list-templates
python -m src.cli new my-ep --template diy --theme psycho_terminal --no-stock
```

`new` options: `--template/-t blank|tutorial|diy`, `--theme`, `--stock/--no-stock`
(`stock_media.enabled`), `--from <ep>` (copy another episode's `config.yaml`), `--force` (fill in
missing files of an existing folder). It creates `raw/{clips,images,schematics}`, `raw/voiceover/`,
`raw/script.txt`, `raw/_TEMPLATE.txt.example`, `work/`, `out/`, `config.yaml` and a checklist
`README.md`. Only `clips`, `images`, `schematics` are scanned by Stage 2.

## 2. Add assets and the voice-over

Put media and sidecars in `episodes/my-ep/raw/` (see [ASSETS.md](ASSETS.md)) and the narration as
`raw/voiceover.wav` (also `.mp3/.m4a/.flac`). Optional script: `raw/script.md` or `raw/script.txt`
(used for entity detection and stock prefetch).

## 3. Voice chain (Stage 0)

```bash
python -m src.cli audio my-ep --report            # before/after numbers (see below)
python -m src.cli audio my-ep --preset gentle     # podcast | gentle | aggressive
python -m src.cli audio my-ep --ab                # work/audio_ab/: listen and compare
```

The original is kept as `raw/voiceover.orig.*`; the result is `work/voiceover_clean.wav`
(mono, 48 kHz). Timings downstream refer to the shortened audio. All local.

Chain: highpass -> automatic notches -> denoise (afftdn, then RNNoise) -> soft downward
expander -> EQ -> de-esser -> compressor (no makeup) -> silence shortening -> limiter ->
2-pass loudnorm. The only gain stage is the last one, so residue is never boosted by an earlier
stage and the expander threshold is set from the measured, already denoised floor.

* Noise analysis: the quietest frames (50 ms RMS) are used as pure-noise segments; their mean
  spectrum is searched for narrow lines (hum, fan, whine at ANY frequency, harmonics included).
  Each line gets a narrow `equalizer` notch (width ~ the measured line, up to -26 dB). Found
  lines are listed in `work/audio_report.json` (`noise_analysis.tonal_peaks`, `notch_filters`).
  `audio.notch: auto | off | [Hz, ...]` (a list forces those frequencies).
* Denoise `audio.denoise: auto | rnnoise | afftdn | off`, `audio.denoise_strength` 0..1
  (default 0.6). `auto` = (a) `afftdn` whose noise level is calibrated on the longest
  pure-noise clip of this file, then (b) `arnndn` (RNNoise) at a conservative mix
  (0.65 x strength). The RNNoise model (`sh.rnnn`, "somnolent-hogwash" from
  GregorR/rnnoise-models, no copyright claimed) is downloaded once to
  `~/.cache/videomaker/models`; offline or on failure the chain silently uses afftdn only.
  `audio.rnnoise_model` points at another `.rnnn`. DeepFilterNet is not wired in (its pip
  package pins old torch and a Rust build; it would risk the environment): if you want it,
  run it yourself on `raw/voiceover.orig.*` and use the result as the source.
* Expander: `agate` as a soft downward expander (ratio 2.5, 15 ms attack, 300 ms release,
  knee 2.5, reduction limited to `audio.expander_db`, default 14 dB), threshold
  `audio.expander_margin_db` (12) above the denoised floor. It lowers the residue between words,
  it never mutes it.
* Loudness: `audio.target_lufs` (-16) is measured on the mono file (a dual-mono stereo source
  reads 3 LU higher than the same samples as mono; plain `-ac 1` would also add 3 dB, so the
  downmix is a unity-gain `pan`).
* `--report` / `work/audio_report.json` (`verification`): noise floor in absolute dBFS and
  relative to the voice (p90 of frame levels), SNR, energy of each tonal peak before/after
  (absolute and relative to the voice), and the 300-4000 Hz speech energy change in voiced
  frames after notches + denoise + expander (should stay within about 1 dB).
* `--ab` writes `work/audio_ab/original.wav`, `atual.wav` (previous chain) and `novo.wav`: three
  10 s excerpts with voice and a pause each (same timeline, 1 s apart, level matched, silence
  shortening off so they line up), `spectrogram_{original,atual,novo}.png` and `ab.json`
  with the excerpt start times.
* Tuning: if the voice sounds watery or "chirpy", lower `denoise_strength` (try 0.4); if noise
  is still audible, raise it (0.8) or use `--preset aggressive`.

## 4. Run the pipeline

```bash
python -m src.cli --config episodes/my-ep/config.yaml status my-ep
python -m src.cli --config episodes/my-ep/config.yaml run my-ep
python -m src.cli --config episodes/my-ep/config.yaml run my-ep --force            # re-run every stage
python -m src.cli --config episodes/my-ep/config.yaml run my-ep --from-stage plan  # skip stages before
```

`--from-stage` takes `audio|transcribe|index|semantic|prefetch|plan|fetch|cutouts|render`. `run` skips stages whose inputs did not change.

Single stages (always executed): `transcribe [-f]`, `index`, `semantic`, `upscale [-f]`,
`stabilize [-f] [--all]`, `prefetch`, `plan [-f]`, `fetch`, `cutouts`, `render`.

## 5. Plan: review and edit

```bash
python -m src.cli inspect my-ep plan          # also: transcript | assets | licenses
python -m src.cli plan my-ep --force          # regenerate; backs up a hand-edited plan first
```

`work/scene_plan.json` is the source of truth. Edit it by hand (fields in
[ARCHITECTURE.md](ARCHITECTURE.md#scene-plan-schema)), then `render`. A hand-edited plan is
never overwritten by `run` or `plan` without `--force`. Read `out/timeline_summary.md` to find
low-score beats.

## 6. Render

```bash
python -m src.cli --config episodes/my-ep/config.yaml render my-ep --draft   # out/draft.mp4, 640x360, fast
python -m src.cli --config episodes/my-ep/config.yaml render my-ep           # out/final_cut.mp4
```

Unchanged beats come from `work/render_cache`; editing one beat re-renders only that beat. Each
render writes `out/qa_report.json`.

## 7. Evaluate matching

```bash
python -m src.cli eval-matching my-ep --backend all          # baseline | siglip | ollama | all
python -m src.cli eval-matching my-ep --pairs path/to/pairs.json
```

Needs `work/local_assets.json` and labelled pairs (default `tests/semantic/pairs_<ep>.json`).

## 8. Dashboard

```bash
python -m src.cli --config episodes/my-ep/config.yaml dashboard my-ep --run                  # run + live TUI
python -m src.cli --config episodes/my-ep/config.yaml dashboard my-ep --run --force --from-stage plan
python -m src.cli --config episodes/my-ep/config.yaml dashboard my-ep --run --record out.cast --snapshot-at-end end.png
python -m src.cli --config episodes/my-ep/config.yaml dashboard my-ep --follow               # attach to a running run/render
python -m src.cli --config episodes/my-ep/config.yaml dashboard my-ep --replay --speed 2 --video
python -m src.cli --config episodes/my-ep/config.yaml dashboard my-ep --snapshot snap.png --size 140x44 --theme psycho
```

- `--theme slate|psycho`; `--snapshot` writes `.svg|.png|.txt` without a terminal.
- `--preview auto|kitty|iterm|sixel|blocks|quadrants`: real images in Kitty/iTerm2/sixel (chafa)
  terminals, otherwise character blocks. Example: `--preview kitty`.
- Ctrl-C stops the pipeline cleanly (exit 130); the pipeline's exit code is propagated.

### Dashboard preview quality

The preview panel adapts to the terminal (`--preview auto`, the default):

| Terminal | Mode | Result |
|---|---|---|
| Kitty, Ghostty | `kitty` | Full-resolution image of the current beat inside the panel |
| iTerm2, WezTerm | `iterm` | Full-resolution image (re-sent on each refresh) |
| Any terminal with `chafa` | `sixel` | Sixel image |
| Anything else, tmux, no tty | `quadrants` | 2x2 pixels per character cell (blocky, works everywhere) |

`--record` and `--snapshot` files (`.cast`, `.png`, `.svg`, `.txt`) always use the character-cell
renderer, because terminal graphics escapes cannot be stored in them. The live screen keeps the
detected protocol. Force a mode with `--preview kitty` (or `quadrants`, `blocks`).

Character-cell fallback (`quadrants`), 160x45 snapshot:

![Dashboard preview, quadrants fallback](img/preview-quadrants.png)

Inside tmux the graphics protocols are disabled by default; run the dashboard in a plain Kitty
window for full resolution.


## Fix mode (agentic point corrections)

`fix` edits `work/scene_plan.json` from a plain-language instruction (PT or EN). A local LLM
(Ollama `/api/chat` with a JSON schema; the model from `llm.model`) only fills an action list;
code locates the beat, picks assets from the SigLIP/semantic index, validates, shows a diff and
asks before writing. 100% local; no network. If Ollama is down it stops with a clear message.

```bash
python -m src.cli --config episodes/ep-hubble/config.yaml fix ep-hubble \
  "no 0:08 aparece um clipe de 2009 com legenda, troque por algo do Hubble sendo liberado"
python -m src.cli ... fix ep-hubble "remova esse clipe do projeto inteiro" --yes
python -m src.cli ... fix ep-hubble "no 0:30 dê zoom no canto superior esquerdo" --dry-run
python -m src.cli ... fix ep-hubble --interactive     # several fixes, `undo`, `render`, `quit`
python -m src.cli ... fix ep-hubble --review [--apply] [--scan-text]
```

* Actions: `replace_asset` (by name or by semantic query; top-k outside the bad asset, avoiding
  assets used nearby), `exclude_asset` (project-wide), `change_template`, `set_crop`/`set_zoom`
  (`crop_region`, stills and clips), `set_trim`, `set_callout`/`remove_callout`,
  `set_transition`, `split_beat`/`merge_beats`, `move_overlay` (PiP corner), `remove_highlight`.
* Locating the beat: final-video time ("no 0:08": plan time + the branding intro card; silence
  removal already happened in the voice chain, so the plan timeline is the narration timeline),
  narration time, a spoken quote ("quando eu falo 1990"), a beat id, or an asset. "esse clipe"
  refers to the asset the previous fix touched (`work/fix_log.jsonl`).
* Safety: diff first (beat, time, speech, before -> after), `--yes` to skip the prompt,
  `--dry-run` to only look. Backup `work/scene_plan.prev.json`, undo stack `work/fix_undo/`,
  log `work/fix_log.jsonl`. The plan is marked hand-edited, so `run` never overwrites it
  (`plan --force` regenerates). Only local, already-licensed assets are chosen; stock is never
  fetched and the fix render runs with `VIDEOMAKER_OFFLINE=1`.
* Exclusion: `exclude: true` is appended to the asset's sidecar `.txt` and the id is listed in
  `work/excluded_assets.json`. Stages 2, 2a and 3 and the semantic retrieval skip such assets;
  every beat that used it is re-assigned in the same fix.
* After applying, only the changed beats are re-rendered (`work/render_cache`), a before/after
  contact sheet is written to `work/fix_preview.png`, and the video path is printed.
  `--no-render` edits the plan only.
* `--review`: per beat, takes a frame of the beat's asset, asks the VLM (`qwen2.5vl:3b`)
  whether it fits the speech and whether there is burned-in text (subtitle, watermark, a year
  the narration does not say), and flags highlights that have no real position. Suspects are
  listed with a reason and a proposed fix (`work/fix_review.json`); `--apply` applies them
  after confirmation. `--scan-text` scans every asset and records `burned_text` in
  `local_assets.json` / `work/burned_text.json`; the planner then avoids those assets (or crops
  a thin caption band away via `crop_box`).

### Automatic diagram highlights

A highlight box is drawn only when its position is real: an explicit `highlight:` line in the
sidecar, or a box found by local OCR (optional `easyocr` / `pytesseract`) or by the VLM and
validated by reading the cropped region back, and in both cases matching the narration. Boxes
are padded, never cover the title/caption band and are capped at 30 % of the picture. No
reliable location means no highlight. Old plans with a bare `region: top|bottom` highlight are
skipped at render time (warning `highlight skipped`).

## Other commands

`create-demo [ep]` builds a synthetic offline episode; `inspect`, `status` as above.
