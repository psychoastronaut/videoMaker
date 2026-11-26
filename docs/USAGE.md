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
python -m src.cli audio my-ep --report            # before/after LUFS, peak, noise floor
python -m src.cli audio my-ep --preset gentle     # podcast | gentle | aggressive
```

The original is kept as `raw/voiceover.orig.*`; the result is `work/voiceover_clean.wav`.
Timings downstream refer to the shortened audio.

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

## Other commands

`create-demo [ep]` builds a synthetic offline episode; `inspect`, `status` as above.
