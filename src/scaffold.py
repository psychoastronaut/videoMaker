"""Episode scaffolding (`new <ep>`): the full, empty folder architecture plus a lean config.yaml."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Stage 2 only scans these three (non-recursive); everything else under raw/ is ignored by the indexer.
SCANNED_DIRS = ("clips", "images", "schematics")
TEMPLATES = {
    "blank": "Folders + lean config + checklist; no opinion about the topic",
    "tutorial": "Software/how-to episode: screen recordings (kind: screen), terminal split layout, callouts on",
    "diy": "Hardware/bench episode: raw/bench staging folder, sidecar tips for bench clips (audio:, avoid:, media_start:)",
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

SIDECAR_TEMPLATE = """\
# Sidecar for a clip/image/schematic. Copy to <file>.<ext>.txt (e.g. servo_test.mp4.txt) and keep ONLY what you need.
# The first free-text lines describe the asset (what is on screen); they feed the matcher. Lines below are hints.
Close-up of a servo sweeping 0-180 degrees on the bench, wires visible, no text on screen.

kind: screen            # screen|tablet|face -> picture-in-picture candidate (see also pip_for:)
pip_for: clip_main_demo # asset id this screen/face clip may accompany
audio: ambient          # speech (duck under voice) | ambient (keep) | none (mute); default mute
media_start: 2.5        # seconds: skip a title card / clapper
media_end: 40           # seconds: ignore the tail
avoid: 0-3, 55-60       # ranges never shown (burned-in captions, studio intro)
segments: 5-12, 20-28   # ONLY these ranges are good (clips)
highlight: 0.1,0.2,0.3,0.2 | Servo horn | 1.5   # diagrams/photos: x,y,w,h (fractions or px@1920x1080) | label | start s
arrow: center | Signal wire | 3.0               # same, drawn as an arrow (regions: center, top-left, ...)
state: ideal            # ideal|flawed|before|after (a flawed diagram is not shown for the 'correct' wording)
text_px: 18             # diagrams: text height in px (legibility check for split_schematic; default 3% of height)
"""

_SCRIPT_TEMPLATE = """\
# {ep} - roteiro
# Uma frase por linha ou paragrafos curtos. Linhas com # sao ignoradas pelo planner.
# Cite lugares, maquinas e nomes proprios: o prefetch busca midia livre para eles.

[Gancho - 10s]


[Desenvolvimento]


[Fechamento]

"""

_VOICEOVER_README = """\
Coloque aqui a narracao: voiceover.wav (ou .mp3/.m4a/.flac). Nome fixo: `voiceover.*`.
`python -m src.cli audio <ep>` limpa a voz (original preservado como voiceover.orig.*) e gera work/voiceover_clean.wav.
Arquivos voiceover*.wav sao ignorados pelo git deste episodio.
"""

_GITIGNORE = """\
work/
out/
raw/voiceover*.wav
raw/voiceover*.mp3
raw/voiceover*.m4a
raw/voiceover*.flac
raw/**/*.mp4
raw/**/*.mov
raw/**/*.mkv
raw/**/*.webm
raw/**/*.psd
raw/**/*.xcf
"""

_CHECKLIST = """\
# {ep}

Template: `{template}`. Config: `config.yaml` (so overrides sobre `config/default.yaml`).

## Checklist
- [ ] Roteiro em `raw/script.txt`
- [ ] Voz gravada em `raw/voiceover.wav` (limpar: `audio {ep}`)
- [ ] Clipes em `raw/clips/` (B-roll, sem audio necessario){extra_assets}
- [ ] Fotos em `raw/images/`; diagramas em `raw/schematics/`
- [ ] Sidecars `<arquivo>.txt` com descricao e dicas (veja `raw/_TEMPLATE.txt.example`)
- [ ] `status {ep}` sem pendencias criticas
- [ ] `run {ep}` e revisao de `out/timeline_summary.md`

## Comandos
```bash
python -m src.cli audio {ep}                      # voz
python -m src.cli --config episodes/{ep}/config.yaml status {ep}
python -m src.cli --config episodes/{ep}/config.yaml run {ep}
python -m src.cli --config episodes/{ep}/config.yaml dashboard {ep} --run
python -m src.cli --config episodes/{ep}/config.yaml render {ep} --draft   # rascunho rapido
```
{notes}"""

_NOTES = {
    "blank": "",
    "tutorial": (
        "\n## Dicas (tutorial)\n- Grave a tela em `raw/clips/` e use `kind: screen` no sidecar (PiP ao lado do rosto/bancada).\n"
        "- Diagramas de terminal: `kind: terminal` no sidecar usa o layout de split mais largo.\n"
        "- Texto pequeno na tela? Informe `text_px:` no sidecar para o check de legibilidade.\n"
    ),
    "diy": (
        "\n## Dicas (DIY / bancada)\n- Despeje o material bruto em `raw/bench/` (NAO varrido); corte e copie o que presta para `raw/clips/`.\n"
        "- Sidecar de clipe de bancada: `audio: ambient` (som de servo/solda) ou `speech` (voce falando, ducking),\n"
        "  `media_start:` para pular o clap, `avoid:` para mao na frente da camera, `segments:` para os bons trechos.\n"
        "- Fotos da peca: nome descritivo (`placa_esp32_frente.jpg`) ajuda o matcher.\n"
    ),
}


@dataclass
class ScaffoldResult:
    ep: Path
    created: list[str]
    skipped: list[str]


def validate_id(ep_id: str) -> None:
    if not _ID_RE.match(ep_id):
        raise ValueError(f"invalid episode id {ep_id!r}: use letters, digits, '.', '_' or '-'")


def build_config_text(ep_id: str, template: str = "blank", theme: str = "psycho_terminal",
                      stock: bool = True) -> str:
    """Lean episode overrides. Keys not present here fall back to config/default.yaml (deep-merged)."""
    extra = ""
    if template == "tutorial":
        extra = "render:\n  generate_srt: true      # legendas .srt (tutorial: util para acessibilidade)\n"
    elif template == "diy":
        extra = "stabilize:\n  enabled: auto           # clipes de bancada costumam ser tremidos\n"
    return f"""\
# {ep_id} - overrides sobre config/default.yaml (deep-merge: so liste o que muda; o resto vem do default).
# Uso: python -m src.cli --config episodes/{ep_id}/config.yaml <comando> {ep_id}
theme: {theme}                # psycho_terminal | (veja src/theme.py)

stock_media:
  enabled: {str(stock).lower()}              # false = so midia propria (raw/), nada de download
  # strict_cc0_only: true     # so CC0/dominio publico
  # video: {{enabled: false}}   # sem stock em video

semantic:
  backend: auto               # siglip | ollama | none
  # min_clip_share: 0.25      # parcela minima de beats com clipe
  # reuse_caps: {{photo: 3, diagram: 2, clip: 3}}

llm:
  model: qwen2.5:14b-instruct # 7b-instruct = mais rapido, 14b = melhor matching
  # structured: true

audio:
  enabled: auto               # auto = liga se raw/voiceover.* existir
  preset: podcast             # podcast | gentle | aggressive
  # target_lufs: -16.0

render:
  # crt: {{enabled: true}}
  # branding: {{enabled: true}}
  # generate_srt: false
{extra}"""


def _write(path: Path, text: str, force: bool, res: ScaffoldResult, root: Path) -> None:
    rel = str(path.relative_to(root))
    if path.exists() and not force:
        res.skipped.append(rel)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    res.created.append(rel)


def scaffold(ep_id: str, episodes_root: Path, *, template: str = "blank", theme: str = "psycho_terminal",
             stock: bool = True, from_ep: str | None = None, force: bool = False) -> ScaffoldResult:
    validate_id(ep_id)
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r} ({'|'.join(TEMPLATES)})")
    ep = episodes_root / ep_id
    if ep.exists() and any(ep.iterdir()) and not force:
        raise FileExistsError(f"{ep} already exists and is not empty (use --force to fill in missing files)")
    src_cfg = None
    if from_ep:
        src_cfg = episodes_root / from_ep / "config.yaml"
        if not src_cfg.is_file():
            raise FileNotFoundError(f"--from {from_ep}: {src_cfg} not found")
    res = ScaffoldResult(ep, [], [])
    dirs = [f"raw/{d}" for d in SCANNED_DIRS] + ["raw/voiceover", "work", "out"]
    if template == "diy":
        dirs.append("raw/bench")
    for d in dirs:
        p = ep / d
        p.mkdir(parents=True, exist_ok=True)
        if d != "raw/voiceover":
            keep = p / ".gitkeep"
            if not keep.exists():
                keep.write_text("")
                res.created.append(f"{d}/.gitkeep")
    _write(ep / "raw/voiceover/README.txt", _VOICEOVER_README, force, res, ep)
    _write(ep / "raw/script.txt", _SCRIPT_TEMPLATE.format(ep=ep_id), force, res, ep)
    _write(ep / "raw/_TEMPLATE.txt.example", SIDECAR_TEMPLATE, force, res, ep)
    if template == "diy":
        _write(ep / "raw/bench/README.txt",
               "Material bruto de bancada (nao varrido pelo Stage 2). Corte/copie o que presta para ../clips/.\n",
               force, res, ep)
    cfg_text = (src_cfg.read_text(encoding="utf-8") if src_cfg
                else build_config_text(ep_id, template, theme, stock))
    _write(ep / "config.yaml", cfg_text, force, res, ep)
    extra = " (+ `raw/bench/` para o bruto)" if template == "diy" else (
        " (gravacoes de tela: `kind: screen`)" if template == "tutorial" else "")
    _write(ep / "README.md", _CHECKLIST.format(ep=ep_id, template=template, extra_assets=extra,
                                               notes=_NOTES[template]), force, res, ep)
    _write(ep / ".gitignore", _GITIGNORE, force, res, ep)
    return res


def next_steps(ep_id: str) -> str:
    cfg = f"episodes/{ep_id}/config.yaml"
    return "\n".join([
        f"1. Escreva o roteiro: episodes/{ep_id}/raw/script.txt",
        f"2. Grave a voz em episodes/{ep_id}/raw/voiceover.wav e limpe: .venv/bin/python -m src.cli audio {ep_id}",
        f"3. Coloque clipes/fotos/diagramas em episodes/{ep_id}/raw/{{clips,images,schematics}}/ (+ sidecars .txt)",
        f"4. Rode: .venv/bin/python -m src.cli --config {cfg} run {ep_id}",
        f"   ou com painel: .venv/bin/python -m src.cli --config {cfg} dashboard {ep_id} --run",
        f"5. Veja o que falta: .venv/bin/python -m src.cli --config {cfg} status {ep_id}",
    ])


def missing_items(ep: Path) -> list[str]:
    """What an episode still needs before `run` can work (empty list = ready)."""
    raw = ep / "raw"
    out = []
    if not any(raw.glob("voiceover.*")) and not (ep / "work" / "voiceover_clean.wav").exists():
        out.append("voz: raw/voiceover.wav (ou .mp3/.m4a/.flac)")
    script = raw / "script.txt"
    if not (raw / "script.md").is_file() and not (
            script.is_file() and any(ln.strip() and not ln.lstrip().startswith(("#", "[")) for ln in
                                     script.read_text(encoding="utf-8").splitlines())):
        out.append("roteiro: raw/script.txt (opcional mas melhora o prefetch de stock)")
    n = {d: sum(1 for f in (raw / d).glob("*") if f.is_file() and not f.name.startswith(".")
                and not f.name.endswith((".txt", ".example"))) for d in SCANNED_DIRS if (raw / d).is_dir()}
    if not sum(n.values()):
        out.append("assets: nenhum arquivo em raw/clips, raw/images ou raw/schematics (so stock sera usado)")
    return out
