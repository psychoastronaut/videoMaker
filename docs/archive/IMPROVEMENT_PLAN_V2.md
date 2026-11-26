# Psycho Astronaut — Plano V2: edição guiada pelo conteúdo (áudio/transcrição ↔ imagem)

## Problema
No `ep-real` as escolhas de imagem e de corte não têm relação com o que a narração diz. Causa: o Stage 2 descreve assets só por nome de arquivo/sidecar, e o Stage 3 escolhe por palavra-chave. Nada "vê" o conteúdo visual nem compara o sentido da frase com ele. O stock também é escolhido só por texto de título/tags.

## Decisão de arquitetura
Nenhuma ferramenta pronta resolve isso localmente e com licença compatível (VideoSeek AGPL, Scene Scout GPL, Smart-Broll/FlonaAI usam OpenAI, ai-video-editor não tem matching visual). **Construir um módulo próprio** com peças open-source validadas nesta máquina e copiar só técnicas, nunca código GPL/AGPL.

Peças validadas (Python 3.14, RTX 3060): `google/siglip2-base-patch16-224` fp16 (730 MB VRAM, ~100 fps) e `siglip2-so400m-patch14-384` (2,2 GB, ~26 fps) com top-1 sensato em 5 consultas reais do `ep-real`; PySceneDetect (BSD) para cortes de cena; `qwen2.5vl:3b` no Ollama (já baixado, ~2,6 GB, ~2 s/imagem) para legendas; qwen2.5 7b para planejar.
Não validado: frases abstratas (latência, PID) — por isso o LLM reescreve cada frase em consultas visuais concretas.

## Técnicas emprestadas
1. Pontuar clipe pelo **melhor frame / média dos top-k** (MaxFrame), nunca média de todos.
2. **Shot detection** → 1–3 keyframes por shot; guardar início/fim para o render cortar o melhor intervalo.
3. Fundir frames adjacentes de score alto em um intervalo; pré-processar a ~480p.
4. Cache de embeddings por hash do arquivo (`.npy` + JSON); cosseno por força bruta (sem FAISS).
5. **LLM reescreve cada frase em 1–2 consultas visuais em inglês**; o embedder pontua.
6. Rank = similaridade + penalidade de diversidade + keyword/sidecar como prior fraco.
7. Alinhar a escolha do LLM às timestamps de palavra (técnica do videoclipper).
8. Cada linha classificada como full-frame / overlay / só áudio (b-roll-skills).

## Fases

### F0 — Avaliação objetiva (antes de mudar)
- Montar `tests/semantic/` com ~30 pares frase→asset esperado a partir de `ep-real` (rótulos manuais) e um script `eval_matching` que mede top-1/top-3.
- Métrica de edição: para cada beat do `timeline_summary.md`, score de similaridade frase↔mídia escolhida (baseline v1 vs v2).

### F1 — Índice semântico (Stage 2)
- `src/semantic.py`: carregar SigLIP2 (modelo configurável `semantic.model`, fallback CPU, descarregar da GPU após uso), embeddings de imagem/texto (usar `.pooler_output`, `padding='max_length', max_length=64`; fixar versão do `transformers`).
- `src/semantic_index.py`: PySceneDetect em clipes, 1–3 keyframes/shot, embeddings + timestamps; imagens/esquemas direto. Cache por hash em `work/semantic/`. Saída: `work/semantic_index.json` + `.npy`.
- Legenda opcional por shot com `qwen2.5vl:3b` (`semantic.captions: true`), gravada no índice como texto extra para o LLM; executada em lote com o planner ocioso.
- Dependências em extra opcional `pyproject [semantic]` (torch, transformers, scenedetect, opencv-python-headless); pipeline continua funcionando sem ela (cai no matching atual com aviso).

### F2 — Planner guiado por conteúdo (Stage 3)
- Para cada segmento/beat: LLM gera `visual_queries` (1–2 frases concretas em inglês, + tipo: local/stock/schematic/cutout). Embedder recupera top-k (k≈5) de assets locais **com intervalo de trim** (`media_start`, `media_end`) e do pool de stock.
- LLM recebe só os k candidatos (legenda + score) e escolhe/justifica; validação garante `source_id` válido, trim dentro da duração, diversidade (sem reutilizar o mesmo asset em ~20 s, sem 3 templates seguidos).
- Planner determinístico (fallback sem LLM) usa o mesmo ranking por embedding em vez de keywords.
- Novos campos em `SceneBeat`: `visual_query`, `match_score`, `media_start`, `media_end`; `timeline_summary.md` passa a mostrar a consulta, o score e o motivo; beats com score < limiar são marcados para revisão.
- Cortes alinhados às pausas/palavras (já existente) + ritmo por estilo, agora também usando o conteúdo (troca de assunto → corte).

### F3 — Stock com verificação visual (Stages 2b/4)
- Para cada beat/entidade: buscar N candidatos (já existe), baixar thumbnail/preview, **embedar e reordenar por similaridade com a `visual_query`**; rejeitar abaixo do limiar; se o melhor for fraco, nova consulta reescrita pelo LLM (máx. 2 tentativas) e só então fallback local.
- Substitui o filtro de relevância por palavras (resolve casos como "bench dark" → estação de trem).
- Pool temático também ranqueado por similaridade com o roteiro inteiro.

### F4 — Render (Stage 6)
- Usar `media_start/end` para recortar o melhor intervalo do clipe (início do shot relevante), mantendo loop/pad existentes.
- Nenhuma mudança de templates; apenas trim correto e logs de score.

### F5 — Verificação e polimento
- `eval_matching` e métrica de edição devem melhorar vs. baseline; teste e2e com `ep-real` (Whisper real + LLM local 7b) e revisão visual de frames por revisor independente.
- Config `semantic.*`, README, comando `index --semantic`/`status` mostrando o estágio; testes unitários com embedder mockado e um teste leve com o modelo base.
- Orçamento de GPU: embedder (0,7–2,5 GB) em processo/etapa separada e liberado antes das chamadas pesadas do Ollama; com o 14b carregado usar `base` ou CPU.

## Riscos
- Evidência de qualidade pequena (5 consultas, ~27 imagens); abstrações dependem da reescrita pelo LLM. Mitigação: F0 mede antes/depois.
- GPU compartilhada com o Ollama; swap de modelos pode lentificar. Mitigação: etapas sequenciais, modelo base.
- Download de modelos no primeiro uso (rede única); licenças: modelos Google SigLIP2 (Apache-2.0 nas model cards — conferir), não copiar código GPL/AGPL.
- `qwen2.5vl:3b` legenda genérica: usar como texto auxiliar, não como fonte principal.

## Critérios de aceite
1. `eval_matching` top-3 ≥ 80% no conjunto rotulado do `ep-real` (a calibrar em F0).
2. Em vídeo real (`ep-real`), revisão por frames: ≥ 80% dos beats com visual coerente com a frase; nenhum stock fora de tema.
3. Pipeline roda sem o extra `[semantic]` (fallback) e offline após o primeiro download dos modelos.
4. pytest/ruff limpos; plano editado à mão continua protegido; `timeline_summary.md` mostra query, score e fonte por beat.

## Processo
Sonnet coder implementa F0→F5 em ordem; um subagente revisor valida F2/F3 com frames; reportar ao usuário o antes/depois no `ep-real`.

## Tuning round (callouts, diagrams, chunked planning, retrieval)

- Callouts are 2-5 Title Case words copied from the beat's narration, never an asset/file name;
  a lower-third on footage is kept only if it shares a term with the shown asset.
- Cutouts only cut out the beat's own best-matching still with a clear single subject; the
  background is the runner-up only if it also matches (else the same still, blurred). Otherwise
  the beat is fullscreen/punch with a lower-third.
- Diagrams: fullscreen at native size (contain, no zoom, lower-third) whenever the line shares
  terms with the sidecar text or the diagram would fall under 50% scale in the split panel
  (`MIN_SPLIT_SCALE`); the renderer warns when diagram text is shown below 50% of native.
- Planning is chunked: the deterministic planner fixes the cuts, the LLM refines 5 beats per call
  (`llm.chunk_beats`, `llm.chunk_timeout`, `llm.chunk_retries`, `llm.plan_budget`); a failed chunk
  keeps its deterministic values. Query rewrite and asset picks are chunked the same way.
- Retrieval: IDF-weighted sidecar-term prior (schematics x3), neighbour-beat topic term,
  embedding similarity narration vs sidecar description (`semantic.text_prior`,
  `semantic.text_prior_weight`, needs `nomic-embed-text`), per-asset reuse penalty.
- Stills narrower than 1000 px are framed (<=1.5x, blurred fill) instead of upscaled fullscreen;
  never punched in. Optional sidecar line `media_start: <seconds>` skips a title card.
