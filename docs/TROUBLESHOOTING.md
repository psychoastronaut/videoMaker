# Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Ollama 500 / timeouts in Stage 3 | Model not loaded or out of VRAM. `ollama ps`, `ollama pull <model>`, try `llm.model: qwen2.5:7b-instruct`, raise `llm.timeout`. A failed chunk falls back to the deterministic plan (see `fallback` in the dashboard). `VIDEOMAKER_OFFLINE=1` skips the LLM |
| CUDA out of memory | Ollama and SigLIP2/rembg/Real-ESRGAN share the GPU. Models free VRAM after use and fall back to CPU on OOM. Stop other GPU apps, use a smaller LLM, set `semantic.siglip.device: cpu`, lower `render.workers` |
| `ffmpeg` has no NVENC / NVENC fails | The renderer falls back to libx264 automatically. Force it with `pipeline.video_codec: libx264`; tune `x264_preset` / `crf`. Check `ffmpeg -encoders \| grep nvenc` |
| vidstab missing | Stabilization uses an OpenCV fallback |
| A cut-out became a framed card | The Stage 5 quality gate rejected the alpha (busy background, low contrast). Use a photo with a clear subject, or pick another asset in `scene_plan.json` (`overlay_asset`), then `cutouts` and `render` |
| The same image keeps repeating | Library too small. Add assets, keep `semantic.diversity.enabled: true`, raise `stock_media.topup`, or edit `source_id` in the plan. Check `reuse_caps` |
| Stock images off-topic | Keep `semantic.verify_stock: true`, add `staging.entities.keywords`, write a script (`raw/script.md`), set `stock_media.strict_cc0_only: true`, or disable stock with `stock_media.enabled: false` |
| Wrong asset for a line | Add/improve sidecar descriptions, edit `scene_plan.json`, then `render`. Measure with `eval-matching` |
| Plan edits were overwritten | Use `plan --force` only when you want to regenerate (the old plan is backed up to `work/scene_plan.prev.json`) |
| Audio too quiet/noisy/choppy | `audio <ep> --report`; try `--preset gentle` or `aggressive`; tune `audio.max_pause`/`keep_pause`. Final loudness is -16 LUFS, TP -1.5 |
| Audio/video drift | Re-run `audio` and then `run --force --from-stage transcribe`; the branded intro delays the voice-over by design |
| Black/frozen segments in the QA report | Static diagrams over 4 s trigger the freeze check; check `out/qa_report.json` and `timeline_summary.md` |
| Dashboard preview is blocky | Use a Kitty/iTerm2/sixel terminal and `--preview kitty|iterm|sixel` |
