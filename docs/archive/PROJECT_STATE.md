# Psycho Astronaut — estado do projeto (arquivo de contexto, 2026-11-25)

Resumo consolidado para retomar o trabalho sem o histórico da conversa. Detalhes de uso: `README.md`; configuração: `config/default.yaml`; planos antigos: `docs/IMPROVEMENT_PLAN.md` (v1, concluído) e `docs/IMPROVEMENT_PLAN_V2.md` (semântico, implementado).

## Pipeline atual (Python 3.14, `.venv`, comandos `.venv/bin/python -m src.cli ...`)
`run | transcribe | index | semantic | upscale | stabilize | prefetch | plan | fetch | cutouts | render | status | inspect | eval-matching | create-demo`
1. Stage 1 faster-whisper (palavras + pausas) → `work/transcript.json`
2. Stage 2 índice local (ffprobe, sidecar `.txt`, `media_start:` opcional) + upscale Real-ESRGAN (fotos < 1400 px, extra `[upscale]`) + estabilização vidstab (clipes trêmulos, `shake_score`) + índice semântico (SigLIP2 ou Ollama qwen2.5vl/nomic-embed-text; extra `[semantic]`)
2b. Prefetch de stock (pool temático + entidades → `work/stock_pool.json`)
3. Stage 3 planner: cortes determinísticos + LLM local em blocos de 5 beats com saída estruturada (Ollama `/api/chat` + JSON schema), reescrita de consultas visuais, escolha entre top‑k, callouts (`src/callouts.py`), variedade (`src/variety.py`), transições (`src/transitions.py`); fallback determinístico por bloco
4. Stage 4 stock gratuito com auditoria de licença (Openverse CC0/PDM, Wikimedia CC0/PD, Archive, NASA, Pixabay/Pexels com chave; `strict_cc0_only`, `stock_media.enabled`), verificação visual por similaridade
5. Stage 5 cutouts: rembg `birefnet-general-lite` → fallback isnet/u2net, limpeza de alfa, portão de qualidade que troca recorte ruim por cartão com moldura (`cutouts_manifest.json`)
6. Stage 6 FFmpeg: 4 templates (fullscreen, punch_in, split_schematic, cutout_callout), movimento por `perspective` sem tremor, transições xfade (9 tipos), NVENC→x264, loudnorm 2 passes, intro/outro fade (+1 s de pausa final), `timeline_summary.md`; legenda `.srt` só com `render.generate_srt: true`
Proteção: plano editado à mão não é sobrescrito sem `--force`; invalidação por mtime/assinatura.
Tudo local: faster-whisper, Ollama (qwen2.5 3b/7b/14b-instruct, qwen2.5vl:3b, nomic-embed-text), SigLIP2, rembg. Sem APIs de nuvem. GPU RTX 3060 12 GB compartilhada com Ollama (embedder carrega só durante o uso).
Chaves: `PIXABAY_API_KEY` no ambiente (sem Pexels/Smithsonian). `.env` é lido automaticamente.

## Estado de qualidade (testes: ~262 passando, ruff limpo)
- Acerto de matching no `ep-serious` (47 beats, gabarito `raw/scene_map.json`): ≈ 19–23% asset previsto / 46–67% aceitável; ruído ±8 pts. Maior ganho veio de corrigir o reparo do Stage 3 que sobrescrevia escolhas semânticas.
- Mistura de templates (ep-callout-test): 66% fullscreen, 23% punch-in, 11% cutout, **0% split** (esquemáticos de 1920 px reprovam a regra de legibilidade `semantic.min_split_scale`; 0.35 daria ~4 splits, não renderizado).
- Callouts melhoraram ("Program Alarm 1202"), ainda alguns fracos ("Room Left", "Car Key").
- Tremor: fotos 0,61→0,04 px; clipes 2,05→0,71.
- Feedback do usuário pendente: "a versão anterior estava melhor que ep-callout-test" — versão/aspecto NÃO esclarecidos (candidatas: ep-shake-test, ep-tr-test, ep-serious-plan; mudanças do callout-test: mais punch-ins 4→11, mais cutouts/cartões, callouts novos, matching 14b).

## Episódios (em `episodes/`)
- Manter: `ep-serious` (+`-v1`), `ep-serious-plan`, `ep-tr-test`, `ep-shake-test`, `ep-callout-test` (comparação), `ep-hubble` (novo, mídia curada em andamento).
- Descartáveis: `ep-demo`, `ep-final`, `ep-pix`, `ep-real-v2`, `ep01-sample`, `ep01-blackholes` (demos sintéticos/antigos). `ep-real` (TTS + mídia aleatória) serve só de histórico.
- Demo sintético: `create-demo <id>`.

## Pendências / próximos passos
1. Esclarecer qual versão anterior era melhor que `ep-callout-test` e calibrar punch-ins/cutouts (variety) em função disso.
2. `split_schematic` em 0%: testar `semantic.min_split_scale: 0.35` ou novo layout (diagrama grande + painel lateral).
3. Callouts ainda irregulares; cutouts viram cartões com frequência (variety não enxerga o portão do Stage 5).
4. Avaliar `ep-hubble` (roteiro novo, ≥30 assets curados, gabarito) com o pipeline completo.
5. Possível: ferramentas de estilo (analisar vídeos de referência — ver avaliação de `JuneYaooo/rea-video`: apenas ideia a emprestar), música de fundo CC0 com ducking (`sidechaincompress`), animação de diagramas (Graphviz/Manim), stable-ts para alinhamento por palavra.
6. Qualidade de código: README/config já documentam as novas seções; nenhum repositório git (sem versionamento).

## Referências de pesquisa (não adotadas / licença)
Descartados: VideoSeek (AGPL), Scene Scout (GPL), Smart-Broll/FlonaAI (OpenAI), RMBG‑2.0/MusicGen/F5‑TTS/XTTS (pesos não comerciais), Remotion (licença paga acima de porte). Adotados: SigLIP2, PySceneDetect, spandrel+Real‑ESRGAN, BiRefNet via rembg, ffmpeg xfade/vidstab, Ollama structured outputs.

## Rodada 2026-11-25 (ep-hubble-fix)
Feito: callouts suprimidos em diagramas fullscreen (+ aviso `overlap:` no Stage 6); atribuicao global de assets (`src/assignment.py`: caps foto 3/diagrama 2/clipe 3, janela 22 s, bonus de clipe para acao, share minimo de clipes 25 %, penalidade de estado `state: ideal|flawed|before|after` e defocus ilustrativo em linhas de "blurry"); split novo (midia 640 px + diagrama 1136 px, legibilidade por escala e altura de texto); dicas de sidecar `media_end`, `avoid`, `segments`, `text_px`, `state`; panoramas com pan, retratos em cartao, cartoes de cutout sobre grade escura; cache por beat em `work/render_cache`, render paralelo (3), `render --draft`, QA gate (`out/qa_report.json`), `work/run_log.jsonl`.
Resultado ep-hubble-fix (14b, siglip): gabarito intended 33 % / acceptable 47 % (base 35/53: o custo da diversidade, cenas que repetem o mesmo diagrama), clipes 13/43, splits 5/43, reuso max 3, cold 105 s (antes 233 s), warm (1 beat) 50 s, draft 80 s.
Pendente: aviso de legenda queimada (avoid espacial), diagramas estaticos >4 s disparam o QA freeze, callouts fracos ("Lens Sat"), draft ainda caro (filtros a 1080p).

## Tema psycho_terminal (2026-11-25)
`theme: psycho_terminal` (src/theme.py, assets/fonts, README) + render.crt/branding/hud/glitch_transitions; intro 2.0 s + outro 2.5 s boot cards. Teste: episodes/ep-theme-test (render 110.17 s = 105.67 + 2 + 2.5). Diagramas raster pre-feitos mantem o azul classico (so a moldura/CRT mudam).
