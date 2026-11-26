# Psycho Astronaut — Plano de implementação e melhoria (v1)

Base: análise do código v0 em 2026-11-24. Regra geral: **manter estágios, CLI, estrutura de episódios e regras de licença**; corrigir e evoluir de forma incremental. Sem stubs/TODOs.

## Critérios de aceite (definição de "pronto")
1. `python -m src.cli create-demo ep-demo && python -m src.cli run ep-demo` roda offline sem erro e gera `out/final_cut.mp4` (1920x1080, 30fps, yuv420p, h264, áudio AAC 48k, duração == voiceover ±0.1s), `out/licenses_audit.json`, `out/timeline_summary.md`, `out/final_cut.srt`.
2. O demo exercita **os 4 templates** (`fullscreen_clip`, `split_schematic`, `cutout_callout`, `punch_in_emphasis`), **todos os beat_styles** (`normal`, `deadpan`, `hype`, `technical`, `fail`) e pelo menos 1 beat `stock` (offline → card de fallback auditado).
3. Frames extraídos do meio de cada beat mostram visualmente o template correto (overlay/cutout/split/punch-in visíveis e legíveis).
4. Nenhum beat falha silenciosamente: erro de render de beat → fallback explícito (card) + aviso no log + nota no timeline_summary; nunca corrompe o concat.
5. Editar `work/scene_plan.json` à mão e rodar `render ep-demo` reflete a mudança (re-roda Stage 5 se necessário + Stage 6).
6. `pytest` passa (testes unitários + 1 smoke de render curto). `ruff check src` limpo.

## P0 — Bugs que impedem o vídeo pretendido
1. **Schema (`schemas.py`)**
   - `SceneBeat`: adicionar `cutout_path`, `badge_path`, `lower_third_path`, `media_path` (resolvido), `stock_asset_id`, `license_type`, `beat_reason` (motivo curto do corte), `emphasis: bool`.
   - Unificar `tone`/`beat_style`: `beat_style: BeatStyle` é a fonte da verdade; `tone` vira alias/compat (aceita no input, normaliza para enum).
   - `ScenePlan.total_duration` como `computed_field` (serializado). Validar `callout_text` (2–5 palavras, corta/limpa).
2. **Stage 5 persiste** os caminhos gerados de volta em `scene_plan.json` (via schema). `rembg`: usar `new_session(model)` (cache de sessão), remover kwargs inválidos; fallback por placeholder só se falhar, com aviso.
3. **Stage 6 — reescrever o compilador de filter_complex por template**, com entradas corretas:
   - Normalização comum: `scale=W:H:force_original_aspect_ratio=increase,crop=W:H,setsar=1,fps=30,format=yuv420p`.
   - Imagens: `-loop 1 -t dur`, `zoompan` suave (Ken Burns lento, direção alternada por beat). Vídeos: **sem zoompan**; enquadramento dinâmico sutil via scale/crop animado; clipes curtos → `-stream_loop -1` ou `tpad=stop_mode=clone`; clipes sem áudio OK (vídeo apenas).
   - `punch_in_emphasis` **snappy**: zoom ease-out rápido (~0.2–0.3s) de 1.0→~1.15–1.2 e segura (sem drift contínuo nem tremor), crop centrado/ponto de interesse.
   - `split_schematic`: base em grid escuro; mídia principal à esquerda (~58%) e esquema em painel à direita com borda/realce, label curto.
   - `cutout_callout`: fundo (clip/imagem desfocado/escurecido levemente) + cutout PNG escalado (≤ 45% altura) com sombra + badge legível; no máximo 1 badge por beat; animação de entrada curta (fade/slide ≤ 0.25s).
   - Overlays (cutout, badge, lower-third) passados como `-i` reais com `overlay` e `enable`/fade.
   - `beat_style` afeta o enquadramento: `fail` → acento vermelho no badge + micro-punch; `hype` → movimentos um pouco mais rápidos; `technical` → acento ciano, enquadramento estável; `deadpan` → estático/hold (sem movimento).
   - Intermediários codificados uma vez com qualidade final (NVENC `-cq` ou libx264 `-crf` do config), concat com `-c:v copy`; áudio: `loudnorm` (alvo do config, sem o `aresample` errado), `-ar 48000`, vídeo/áudio alinhados (pad do último beat se necessário, sem `-shortest` cortando narração).
   - Falha de beat → re-render como card de fallback; temp dir limpo (`TemporaryDirectory`), logs claros com o comando ffmpeg em caso de erro.
   - Resolver mídia por `source_id` exato via `local_assets.json`; para stock, via `stock_asset_id`/`media_path`. Sem substring fuzzy.
4. **Stage 3 — LLM e planner determinístico**
   - Config `llm.base_url`, `llm.model` (default `qwen2.5:14b-instruct`), env override; se o modelo não existir, auto-selecionar um disponível de `/v1/models` e logar. Retry (até 2) re-enviando os erros de validação. Prompt orientado a estilo (Reeves / Stuff Made Here / ElectroBOOM / Furze / Allen Pan): ritmo brisk, marcar falhas/gambiarras/mishaps, mecânicas invisíveis → `split_schematic`/`cutout_callout` com labels de 2–5 palavras.
   - Planner determinístico (fallback): **dividir segmentos longos em fronteiras de palavra/pausa** com durações alvo por estilo lidas de `staging.beat_pacing` (hype/fail ~1.5–3s, normal ~2.5–4.5s, technical ~3–5s, hold de demo até 6s); timeline ancorada nos tempos reais das palavras (sem drift); keywords de estilo vindas do config (fail: broke, failed, smoke, fire, oops, janky, duct tape, workaround, exploded, short circuit…; hype; technical/invisible: signal, latency, voltage, current, model, code, circuit, algorithm…); variar templates, evitar mesma fonte consecutiva; callout extraído do trecho (2–5 palavras); `beat_reason` sempre preenchido.
   - `_validate_and_repair`: cobertura total [0, duração do áudio] sem gaps/overlaps, duração mínima (1.2s) e máxima, recalcula `duration`, IDs locais existentes (substitui por asset compatível em vez de só apagar), compatibilidade template↔tipo de asset, queries de stock limpas (sem pontuação/stopwords, ≤ 6 palavras), stock sem query → query derivada do texto.
5. **Stage 4 ligado ao render**: cada beat stock recebe `stock_asset_id`/`media_path` do melhor resultado; fallback card gerado **por beat** e linkado; bug do md5 vs id corrigido.

## P1 — Qualidade/robustez
6. **Licenças estritas**: Openverse com post-check (`cc0`/`pdm` apenas); Wikimedia aceita só CC0/Public domain explícitos (remover substring "commons"), com `User-Agent` genérico; Pexels/Pixabay somente se chave presente **e** `stock_media.allow_platform_licenses: true`, gravando licença exata ("Pexels License"/"Pixabay Content License") — default `false` para manter "strict CC0/PD"; checar `min_resolution`; auditoria com filename, plataforma, URL, licença, license_url, autor (se houver), timestamp, beat_id.
7. **CLI**: `--force` em `run`; comandos por estágio (`transcribe`, `index`, `plan`, `fetch`, `cutouts`, `render`); invalidação por mtime (artefato a jusante mais velho que o de montante → re-roda); `render` sempre re-renderiza e re-roda Stage 5 se o plano mudou; `--config` opcional; loader simples de `.env`.
8. **timeline_summary.md**: duração total real; tabela `# | Start | End | Dur | Template | Source (local / CC0 stock + licença) | Style | Asset | Callout | Reason`; seção de avisos (fallbacks, assets ausentes, beats curtos/longos); dica de como editar `scene_plan.json` e re-renderizar.
9. **Stage 1**: corrigir filtro de espaçamento mínimo dos cut boundaries; cache do modelo Whisper compartilhado (`~/.cache` ou `config.transcription.model_dir`); usar `vad_filter` do config.

## P2 — Higiene
10. README.md (uso, config, fluxo de edição manual), corrigir `PEPELS_API_KEY`, remover deps não usadas (`aiohttp`, `requests`, `pydub`, `ffmpeg-python` se não usadas), `tests/` com pytest.
11. Demo generator coerente com `ep01-sample` (tema engenharia/hardware com momentos de falha) e garantindo os critérios de aceite acima; termos técnicos do planner não hardcoded para órbita.

## Processo
- Subagente **coder** (Sonnet) implementa P0→P1→P2 e verifica com o demo.
- Subagente **reviewer** revisa o diff contra `scratchpad/baseline`, roda testes/demo, extrai frames e aprova ou lista problemas (P0/P1/P2).
- Iterar até o reviewer aprovar sem P0/P1 pendentes.
