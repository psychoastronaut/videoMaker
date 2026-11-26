"""Local semantic matching backends (no cloud APIs).

One interface, two interchangeable implementations:

* ``SigLIPBackend``  -- SigLIP2 image/text embeddings (optional extra ``[semantic]``).
* ``OllamaBackend``  -- ``qwen2.5vl`` captions per keyframe + an Ollama text-embedding model
  (``nomic-embed-text``) comparing captions with visual queries. No extra Python deps.

``get_backend`` resolves ``semantic.backend: auto|ollama|siglip|none`` (auto = siglip when
installed, else ollama when reachable, else none). A backend maps keyframe images and visual
queries into one cosine space: ``embed_frames`` -> (vectors, captions), ``embed_queries``.
Both return L2-normalised float32 arrays so similarity is a plain dot product.
"""

from __future__ import annotations

import base64
import importlib.util
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from rich.console import Console

from src.common import is_offline, load_config

console = Console()

OLLAMA_URL = "http://localhost:11434"
# Reasonable acceptance thresholds on the raw cosine (calibrated with `eval-matching`).
DEFAULT_MIN_SCORE = {"siglip": 0.10, "ollama": 0.55}


def normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-8)


class SemanticBackend(ABC):
    name = "none"
    model = ""

    @property
    def tag(self) -> str:
        """Cache key component: changes when backend/model/prompt changes."""
        return f"{self.name}-{self.model}".replace("/", "_").replace(":", "_")

    @abstractmethod
    def embed_frames(self, paths: list[Path]) -> tuple[np.ndarray, list[str]]:
        """Embed keyframe images. Returns (N x D vectors, N captions ('' if none))."""

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed visual queries (N x D)."""

    def free(self) -> None:  # release GPU memory before heavy LLM calls
        return None

    @property
    def min_score(self) -> float:
        return DEFAULT_MIN_SCORE.get(self.name, 0.0)


# --------------------------------------------------------------------------- SigLIP2


class SigLIPBackend(SemanticBackend):
    name = "siglip"

    def __init__(self, model: str = "google/siglip2-base-patch16-224", device: str = "auto"
                 ) -> None:
        self.model = model
        self.device_pref = device
        self._m: Any = None
        self._p: Any = None
        self._device = "cpu"

    @staticmethod
    def available() -> bool:
        return all(importlib.util.find_spec(m) for m in ("torch", "transformers"))

    def _load(self) -> None:
        if self._m is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        want_cuda = self.device_pref in ("auto", "cuda") and torch.cuda.is_available()
        self._p = AutoProcessor.from_pretrained(self.model)
        if want_cuda:
            try:
                self._m = AutoModel.from_pretrained(
                    self.model, dtype=torch.float16).to("cuda").eval()
                self._device = "cuda"
                return
            except (RuntimeError, torch.cuda.OutOfMemoryError):  # incl. AcceleratorError
                console.print("  [yellow]SigLIP: GPU out of memory, using CPU[/]")
                self.free()
                self._p = AutoProcessor.from_pretrained(self.model)  # free() dropped it
        self._m = AutoModel.from_pretrained(self.model, dtype=torch.float32).eval()
        self._device = "cpu"

    def free(self) -> None:
        self._m = None
        self._p = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _feat(out: Any) -> Any:
        return out if hasattr(out, "shape") else out.pooler_output  # transformers>=5

    def _run(self, fn: Any, batch: int, items: list[Any], **kw: Any) -> np.ndarray:

        outs = []
        for i in range(0, len(items), batch):
            x = fn(items[i:i + batch], **kw)
            outs.append(x)
        return np.concatenate(outs) if outs else np.zeros((0, 1), np.float32)

    def _enc_images(self, imgs: list[Any]) -> np.ndarray:
        import torch

        x = self._p(images=imgs, return_tensors="pt").to(self._device)
        if self._device == "cuda":
            x["pixel_values"] = x["pixel_values"].half()
        with torch.no_grad():
            f = self._feat(self._m.get_image_features(**x))
        return f.float().cpu().numpy()

    def _enc_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        x = self._p(text=texts, padding="max_length", max_length=64, truncation=True,
                    return_tensors="pt").to(self._device)
        with torch.no_grad():
            f = self._feat(self._m.get_text_features(**x))
        return f.float().cpu().numpy()

    def _guard(self, fn: Any, items: list[Any], batch: int) -> np.ndarray:
        import torch

        self._load()
        try:
            return self._run(fn, batch, items)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e):
                raise
            console.print("  [yellow]SigLIP: GPU out of memory, retrying on CPU[/]")
            self.device_pref = "cpu"
            self.free()
            self._load()
            return self._run(fn, batch, items)

    def embed_frames(self, paths: list[Path]) -> tuple[np.ndarray, list[str]]:
        from PIL import Image

        def enc(items: list[Path]) -> np.ndarray:
            ims = []
            for p in items:
                with Image.open(p) as im:
                    ims.append(im.convert("RGB"))
            return self._enc_images(ims)

        vecs = self._guard(lambda items, **_: enc(items), list(paths), 8)
        return normalize(vecs), [""] * len(paths)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        vecs = self._guard(lambda items, **_: self._enc_texts(items), list(texts), 32)
        return normalize(vecs)


# --------------------------------------------------------------------------- Ollama


CAPTION_PROMPT = ("Describe this image in one sentence for a video editor. Name the concrete "
                  "objects, the setting and any action. No opinions, no preamble.")


class OllamaBackend(SemanticBackend):
    name = "ollama"

    def __init__(self, base_url: str = OLLAMA_URL, caption_model: str = "qwen2.5vl:3b",
                 embed_model: str = "nomic-embed-text", timeout: float = 120.0) -> None:
        self.base = base_url.rstrip("/").removesuffix("/v1")
        self.caption_model = caption_model
        self.embed_model = embed_model
        self.timeout = timeout
        self.model = f"{caption_model}+{embed_model}"

    def models(self) -> list[str] | None:
        try:
            r = httpx.get(f"{self.base}/api/tags", timeout=3.0)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    def available(self) -> bool:
        names = self.models()
        if not names:
            return False
        have = {n.split(":")[0] if ":" not in self.embed_model else n for n in names}
        return (self.caption_model in names or self.caption_model.split(":")[0] in have) and (
            self.embed_model in names or f"{self.embed_model}:latest" in names
            or self.embed_model in have)

    def caption(self, path: Path) -> str:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((640, 640))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
        payload = {
            "model": self.caption_model, "stream": False, "keep_alive": "2m",
            "messages": [{"role": "user", "content": CAPTION_PROMPT,
                          "images": [base64.b64encode(buf.getvalue()).decode()]}],
            "options": {"temperature": 0.1, "num_predict": 70},
        }
        try:
            r = httpx.post(f"{self.base}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return " ".join(str(r.json()["message"]["content"]).split())
        except (httpx.HTTPError, KeyError, ValueError) as e:
            console.print(f"  [yellow]caption failed for {path.name}: {e}[/]")
            return ""

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        inputs = [f"{prefix}{t}" if t else f"{prefix}image" for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(inputs), 32):
            r = httpx.post(f"{self.base}/api/embed",
                           json={"model": self.embed_model, "input": inputs[i:i + 32],
                                 "keep_alive": "2m"}, timeout=self.timeout)
            r.raise_for_status()
            out += r.json()["embeddings"]
        return normalize(np.array(out, dtype=np.float32))

    def embed_frames(self, paths: list[Path]) -> tuple[np.ndarray, list[str]]:
        caps = [self.caption(p) for p in paths]
        return self._embed(caps, "search_document: "), caps

    def embed_texts_as_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "search_document: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "search_query: ")

    def free(self) -> None:  # unload both models so the planner LLM gets the VRAM
        for model in (self.caption_model, self.embed_model):
            try:
                httpx.post(f"{self.base}/api/generate", json={"model": model, "keep_alive": 0},
                           timeout=10.0)
            except httpx.HTTPError:
                pass


# --------------------------------------------------------------------------- resolution


def semantic_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return (config if config is not None else load_config()).get("semantic", {}) or {}


def min_score_for(backend_name: str, config: dict[str, Any] | None = None) -> float:
    """Review threshold: ``semantic.min_score`` (number, or {backend: number})."""
    v = semantic_config(config).get("min_score")
    if isinstance(v, dict):
        v = v.get(backend_name)
    return float(v) if v is not None else DEFAULT_MIN_SCORE.get(backend_name, 0.0)


def get_backend(config: dict[str, Any] | None = None, requested: str | None = None,
                quiet: bool = False) -> SemanticBackend | None:
    """Resolve the configured backend, or None (with a clear warning) when unavailable."""
    cfg = semantic_config(config)
    want = (requested or cfg.get("backend", "auto")).lower()

    def warn(msg: str) -> None:
        if not quiet:
            console.print(f"  [yellow]{msg}[/]")

    if want == "none" or not cfg.get("enabled", True) and requested is None:
        return None
    if want in ("auto", "siglip"):
        if SigLIPBackend.available():
            sg = cfg.get("siglip", {}) or {}
            return SigLIPBackend(sg.get("model", "google/siglip2-base-patch16-224"),
                                 sg.get("device", "auto"))
        if want == "siglip":
            warn("semantic.backend=siglip but torch/transformers are not installed "
                 "(pip install -e '.[semantic]'); semantic matching disabled")
            return None
    if want in ("auto", "ollama"):
        ol = cfg.get("ollama", {}) or {}
        be = OllamaBackend(ol.get("base_url", OLLAMA_URL), ol.get("caption_model", "qwen2.5vl:3b"),
                           ol.get("embed_model", "nomic-embed-text"),
                           float(ol.get("timeout", 120)))
        if is_offline() and "localhost" not in be.base and "127.0.0.1" not in be.base:
            return None
        if be.available():
            return be
        warn("Ollama semantic backend unavailable (server down or qwen2.5vl/nomic-embed-text "
             "not pulled)")
    warn("No semantic backend available: falling back to keyword matching")
    return None
