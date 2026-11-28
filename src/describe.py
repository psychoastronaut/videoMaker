"""`describe`: short factual descriptions for assets without (or with a weak) sidecar text.

Descriptions feed the decision planner (text only) and the keyword/semantic priors. A vision
model (default ``qwen2.5vl:3b``) looks at the picture (clips: 1-3 frames; diagrams: the image,
text included) and writes 1-2 sentences. They go to the sidecar as a ``description:`` line.
User text (free lines in the sidecar) is never overwritten without ``--force``; with it the old
text is saved to ``<sidecar>.bak`` first.
"""

from __future__ import annotations

import base64
import io
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from src.common import read_json

console = Console()

PROMPT = (
    "Describe this {kind} for a video editor's asset catalogue in 1-2 short factual sentences "
    "(max 40 words). Name the visible objects, people/roles, action and setting; give the year "
    "or visible text ONLY if clearly readable (never guess dates, places or names). No vague adjectives (beautiful, stunning, "
    "interesting), no opinions, no guessing of facts not visible, no preface. Plain sentence(s).")
VAGUE = re.compile(r"\b(beautiful|stunning|amazing|interesting|nice|great|awesome|impressive|"
                   r"breathtaking|captivating|vibrant|iconic)\b", re.IGNORECASE)
KEY_RE = re.compile(r"^[A-Za-z_]+\s*:")


@dataclass
class Status:
    asset_id: str
    path: Path
    sidecar: Path
    state: str          # ok | weak | missing
    text: str           # user free text ('' when none)
    generated: str = "" # existing `description:` line, if any


def sidecar_parts(path: Path) -> tuple[str, str, list[str]]:
    """(free text, description: value, other lines) of a sidecar file."""
    free: list[str] = []
    desc = ""
    other: list[str] = []
    if path.is_file():
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.match(r"^description\s*:", ln.strip(), re.IGNORECASE):
                desc = ln.split(":", 1)[1].strip()
            elif KEY_RE.match(ln.strip()):
                other.append(ln)
            elif ln.strip():
                free.append(ln.strip())
    return " ".join(free), desc, other


def classify(asset: dict[str, Any]) -> Status:
    path = Path(asset["filepath"])
    side = Path(str(path) + ".txt")
    free, desc, _ = sidecar_parts(side)
    words = re.findall(r"[A-Za-z]{3,}", free or desc)
    stem = set(re.findall(r"[a-z]{3,}", path.stem.lower()))
    novel = {w.lower() for w in words} - stem
    if not (free or desc):
        state = "missing"
    elif len(words) < 5 or len(novel) < 3 or (free and not desc and VAGUE.search(free)
                                                and len(words) < 8):
        state = "weak"
    else:
        state = "ok"
    return Status(asset["id"], path, side, state, free, desc)


def frames_for(asset: dict[str, Any], tmp: Path) -> list[bytes]:
    """JPEG bytes (<= 768 px) of the picture, or 1-3 frames of a clip."""
    from PIL import Image

    from src.semantic_index import extract_frame

    path = Path(asset["filepath"])
    srcs: list[Path] = []
    if asset.get("asset_type") == "clip":
        dur = float(asset.get("duration") or 0.0) or 3.0
        for n, f in enumerate((0.2, 0.5, 0.8) if dur > 6 else (0.5,)):
            out = tmp / f"{asset['id']}_{n}.jpg"
            if extract_frame(path, dur * f, out):
                srcs.append(out)
    else:
        srcs = [path]
    out_b: list[bytes] = []
    for p in srcs:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((768, 768))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            out_b.append(buf.getvalue())
    return out_b


def vlm_describe(client: httpx.Client, base: str, model: str, asset: dict[str, Any],
                 tmp: Path, timeout: float = 120.0) -> str:
    kind = {"clip": "video clip (frames from it)", "image": "photograph",
            "schematic": "diagram (read its labels)"}.get(asset.get("asset_type", ""), "image")
    imgs = frames_for(asset, tmp)
    if not imgs:
        return ""
    payload = {"model": model, "stream": False, "keep_alive": "2m",
               "messages": [{"role": "user", "content": PROMPT.format(kind=kind),
                             "images": [base64.b64encode(b).decode() for b in imgs]}],
               "options": {"temperature": 0.1, "num_predict": 110, "num_ctx": 4096}}
    if "qwen3" in model.lower():  # thinking-capable families: answer directly
        payload["think"] = False
    r = client.post(f"{base}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    text = " ".join(str(r.json()["message"]["content"]).split())
    text = re.sub(r"^(this|the) (image|photo|photograph|video|clip|frame|diagram)s?( shows| depicts"
                  r"| is| appears to be)?[:,]?\s*", "", text, flags=re.IGNORECASE)
    return trim_sentences(text[:1].upper() + text[1:]) if text else ""


def trim_sentences(text: str, n: int = 2, max_words: int = 45) -> str:
    """At most ``n`` sentences / ``max_words`` words (small VLMs ramble)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(parts[:n])
    w = out.split()
    if len(w) > max_words:
        out = " ".join(w[:max_words]).rstrip(",;:") + "."
    return out


def write_description(side: Path, text: str, force: bool = False) -> str:
    """Put ``description: text`` at the end of the sidecar; returns what happened.
    Existing free text is kept unless ``force`` (then backed up to <sidecar>.bak and replaced)."""
    free, old, other = sidecar_parts(side)
    if free and not force:
        return "skipped (user text)"
    if free and force and side.is_file():
        side.with_name(side.name + ".bak").write_text(side.read_text(encoding="utf-8"),
                                                      encoding="utf-8")
    keep = other
    side.write_text("\n".join([*keep, f"description: {text}"]) + "\n", encoding="utf-8")
    return "replaced" if (old or free) else "written"


def run(work: Path, vlm: str = "qwen2.5vl:3b", write: bool = False, force: bool = False,
        base: str = "http://localhost:11434", only: list[str] | None = None,
        list_only: bool = False) -> list[dict[str, Any]]:
    """List assets by description quality; generate (and optionally write) descriptions for the
    missing/weak ones (all of them with ``force``)."""
    assets = read_json(work / "local_assets.json")["assets"]
    sts = [(a, classify(a)) for a in assets]
    t = Table(title="Asset descriptions")
    for c in ("id", "state", "text"):
        t.add_column(c)
    for _a, s in sts:
        t.add_row(s.asset_id, {"ok": "ok", "weak": "[yellow]weak[/]",
                               "missing": "[red]missing[/]"}[s.state],
                  (s.text or s.generated)[:70])
    console.print(t)
    todo = [(a, s) for a, s in sts if (force or s.state != "ok") and (not only or a["id"] in only)]
    missing = [s.asset_id for _a, s in sts if s.state == "missing"]
    console.print(f"  {len(missing)} without description, "
                  f"{sum(s.state == 'weak' for _a, s in sts)} weak, "
                  f"{sum(s.state == 'ok' for _a, s in sts)} ok")
    results: list[dict[str, Any]] = []
    if list_only or not todo:
        return results
    console.print(f"  Describing {len(todo)} asset(s) with {vlm}"
                  f"{'' if write else ' (dry run: use --write to save)'}")
    try:
        with httpx.Client() as client, tempfile.TemporaryDirectory() as td:
            for a, s in todo:
                try:
                    text = vlm_describe(client, base, vlm, a, Path(td))
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    console.print(f"  [yellow]{s.asset_id}: {type(e).__name__} {e}[/]")
                    continue
                if not text:
                    continue
                note = ""
                if write:
                    note = write_description(s.sidecar, text, force)
                results.append({"id": s.asset_id, "text": text, "result": note or "dry-run"})
                console.print(f"  [bold]{s.asset_id}[/] {note}\n    {text}")
    finally:
        try:  # give the VRAM back
            httpx.post(f"{base}/api/generate", json={"model": vlm, "keep_alive": 0}, timeout=15.0)
        except httpx.HTTPError:
            pass
    if write and results:
        console.print("  Sidecars updated: run `index` (and `semantic`) to refresh the matcher.")
    return results
