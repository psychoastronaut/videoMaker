# Psycho Astronaut — Episódio 0: Manifesto (roteiro de gravação)

Duração alvo: ~100 s de voz. Idioma: inglês. Tom: deadpan, seco, sem "voz de locutor".
`/` = pausa curta (0,3 s). `//` = pausa longa (0,7 s, é onde o corte respira). **negrito** = ênfase leve.

## Dicas rápidas
- Grave em sala com tapete/cortinas, microfone a ~15 cm, levemente fora do eixo da boca.
- Faça 3 tomadas completas. O pipeline limpa ruído, volume e silêncios longos.
- Grave um "room tone" de 10 s em silêncio no começo (útil para calibrar o denoise).
- Leia como quem conta uma piada técnica: ritmo rápido, final de frase caindo.
- Nomes: *Faster Whisper* ("fáster uísper"), *SigLIP two* ("síglip tu"), *Qwen* ("tchuen"), *Ollama* ("olâma"), *Rembg BiRefNet* ("rem-bê-gê bi-ref-nét"), *Real ESRGAN* ("riau és-érgan"), *Vidstab* ("vid-stab"), *NVENC* ("en-vênk").

## Roteiro

**[1] Abertura**
Welcome to **Psycho Astronaut**. // Episode zero. / The manifesto. //

**[2] Premissa**
This channel is experimental engineering / with a budget of **zero** dollars. //
We call it conscious jank. / Duct tape, / but with version control. //

**[3] Regra 1**
Rule one. / Everything runs **locally**. / No cloud. / No subscriptions. / One graphics card. //

**[4] A revelação**
This video was not edited by a person. // It was assembled by a **Python pipeline**. //

**[5] Pipeline (uma frase por etapa; aqui entram os splits)**
Step one. / Faster Whisper listens to my voice / and writes down every word.
Step two. / An index reads every asset I own, / and labels it.
Step three. / SigLIP two matches each sentence to a picture, / by meaning. //
Then a local Qwen model, running on Ollama, / plans the cuts, / five beats at a time.
Hungarian assignment makes sure the same photo never shows up twice. / **Mostly.** //

**[6] Stock + cutouts**
Stock footage? / Only public domain, with a license audit. / **Lawyers, relax.**
Backgrounds are removed by Rembg BiRefNet. / A quality gate rejects the ugly ones.
Real ESRGAN upscales the blurry photos. / Vidstab calms the shaky clips. //

**[7] Render**
Then FFmpeg renders everything on NVENC, / with cache and transitions.
Loudnorm fixes the volume. / The terminal theme adds scanlines and glitch. //

**[7b] Prova: o vídeo do Hubble (entram clipes/imagens do ep-hubble)**
Proof? / This is a video about the Hubble servicing mission. / Same pipeline. / Real NASA footage, / all public domain. //
It is not perfect. / Sometimes I say **telescope**, / and it shows a decorative mirror / with flowers. //

**[8] Falha (punch-in nos logs de erro)**
And yes, / it crashes sometimes. //
Out of memory. / **Of course.** / That is the whole show. //

**[8b] O que ainda é humano**
But it is not fully automated. / Not even **close**. //
The ideas are human. / The script is human. / The voice is human, / sorry. //
The pipeline only does the boring part. / I decide what to say. / The machine decides where the pictures go. //

**[9] Fecho**
All of it, / on a single RTX three thousand sixty. //
The code is open. / The footage is public domain. / The budget is **zero**. //
Subscribe, / or don't. / The pipeline will not take it personally.

## Depois de gravar
1. Salve como `raw/voiceover.wav` (ou .mp3/.m4a); o original fica em `raw/voiceover.orig.*`.
2. `.venv/bin/python -m src.cli --config episodes/ep-manifesto/config.yaml audio ep-manifesto --report`
3. Refaça o plano e o render (os tempos mudam com a sua voz; o plano atual foi escrito à mão para a voz placeholder).
