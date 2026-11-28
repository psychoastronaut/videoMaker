"""Client for Jev-style decision models (``POST /v1/systemone``, Ollama >= 0.35).

A decision model reads a ``state`` (text or JSON) once and answers many typed questions with
calibrated probabilities: ``choice`` (pick one option, 2-26 on Ollama), ``noul`` (true/false
probability) and ``score`` (probability-weighted rubric level). No reasoning step, text only.

``DecisionClient.ask`` adds what the planner needs around the raw endpoint: batching (at most
``max_questions`` questions per call), wide choices (more options than the model accepts are
narrowed in rounds), retries with backoff, an on-disk cache keyed by a hash of the request, and
unloading the model at the end so the GPU is free for the next stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

console = Console()

DEFAULT_BASE_URL = "http://localhost:11434"
KNOWN_MODELS = ("nimble", "tev1", "tev1:0.8b", "tev1:4b")
MAX_QUESTIONS = 64
MAX_OPTIONS = 26  # Ollama's documented limit for choice/score options


class DecisionError(RuntimeError):
    """The decision endpoint failed (bad request, bad answer, retries exhausted)."""


class DecisionUnavailable(DecisionError):
    """No usable endpoint: Ollama too old, model not pulled, server down."""


@dataclass(frozen=True)
class Question:
    """One named question. ``criteria``: ``{option: description|None}`` for choice, an ordered
    list of level descriptions for score, optional ``{"true": ..., "false": ...}`` for noul."""
    type: str
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None

    def body(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out


def choice(instructions: str, criteria: dict[str, str | None]) -> Question:
    return Question("choice", instructions, criteria)


def noul(instructions: str, criteria: dict[str, str | None] | None = None) -> Question:
    return Question("noul", instructions, criteria)


def score(instructions: str, levels: list[str]) -> Question:
    return Question("score", instructions, levels)


@dataclass
class Answer:
    """Typed answer. ``probabilities`` keys are option names (choice) or level indexes (score)."""
    type: str
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    noul: float | None = None
    score: float | None = None
    legend: dict[str, str] = field(default_factory=dict)

    @property
    def p(self) -> float:
        """Probability of the answer: top choice, true for noul."""
        if self.type == "noul":
            return float(self.noul or 0.0)
        return float(self.probabilities.get(self.choice or "", 0.0))


def parse_answer(raw: Any, expected: Question | None = None) -> Answer:
    """Validate one answer object from the response; raises DecisionError when malformed."""
    if not isinstance(raw, dict):
        raise DecisionError(f"answer is not an object: {raw!r}")
    kind = str(raw.get("type") or (expected.type if expected else ""))
    if expected is not None and kind != expected.type:
        raise DecisionError(f"answer type {kind!r} != question type {expected.type!r}")
    try:
        if kind == "noul":
            val = float(raw["noul"])
            if not 0.0 <= val <= 1.0:
                raise DecisionError(f"noul probability out of range: {val}")
            return Answer("noul", noul=val)
        probs = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
        conf = float(raw.get("confidence") or 0.0)
        if kind == "choice":
            ch = raw.get("choice")
            if ch is None and probs:
                ch = max(probs, key=probs.__getitem__)
            if ch is None:
                raise DecisionError("choice answer without choice")
            ch = str(ch)
            if not probs:
                probs = {ch: 1.0}
            if expected is not None and isinstance(expected.criteria, dict) \
                    and ch not in expected.criteria:
                raise DecisionError(f"choice {ch!r} is not an option")
            return Answer("choice", choice=ch, probabilities=probs, confidence=conf)
        if kind == "score":
            return Answer("score", score=float(raw["score"]), probabilities=probs, confidence=conf,
                          legend={str(k): str(v) for k, v in (raw.get("legend") or {}).items()})
    except (KeyError, TypeError, ValueError) as e:
        raise DecisionError(f"malformed {kind} answer: {e}") from e
    raise DecisionError(f"unknown answer type {kind!r}")


def normalise(probs: dict[str, float]) -> dict[str, float]:
    s = sum(max(v, 0.0) for v in probs.values())
    return {k: max(v, 0.0) / s for k, v in probs.items()} if s > 0 else dict(probs)


def entropy_confidence(probs: dict[str, float]) -> float:
    """1 - normalised entropy, in 0..1 (used when a wide choice is merged from rounds)."""
    p = [v for v in normalise(probs).values() if v > 0]
    if len(p) < 2:
        return 1.0
    return 1.0 - (-sum(x * math.log(x) for x in p) / math.log(len(p)))


def resolve_model(cfg: dict[str, Any]) -> tuple[str, str]:
    """(model name, base url) from ``decision.model`` / ``decision.base_url``. ``model`` may be
    ``nimble|tev1|tev1:0.8b|<any ollama tag>`` or a URL of another Jev-compatible server (then
    the URL is the base and the model name is ``decision.model_name``, default ``jev``)."""
    model = str(cfg.get("model") or "nimble")
    base = str(cfg.get("base_url") or DEFAULT_BASE_URL)
    if model.startswith(("http://", "https://")):
        return str(cfg.get("model_name") or "jev"), model
    return model, base


class DecisionClient:
    def __init__(self, cfg: dict[str, Any] | None = None, cache_dir: Path | None = None,
                 sleep: Any = time.sleep) -> None:
        cfg = cfg or {}
        self.model, base = resolve_model(cfg)
        base = base.rstrip("/")
        self.base = base.removesuffix("/v1")
        self.url = f"{self.base}/v1/systemone"
        self.timeout = float(cfg.get("timeout", 120))
        self.retries = int(cfg.get("retries", 2))
        self.keep_alive = cfg.get("keep_alive", "5m")
        self.max_questions = max(1, min(int(cfg.get("max_questions", MAX_QUESTIONS)), MAX_QUESTIONS))
        self.max_options = max(2, min(int(cfg.get("max_options", MAX_OPTIONS)), 255))
        self.unload_when_done = bool(cfg.get("unload", True))
        self.cache_dir = cache_dir if cfg.get("cache", True) else None
        self._sleep = sleep
        self.stats = {"calls": 0, "questions": 0, "cache_hits": 0, "retries": 0, "failures": 0,
                      "seconds": 0.0, "input_tokens": 0}
        self._used = False

    # ------------------------------------------------------------------ cache

    def _key(self, state: Any, questions: dict[str, Question]) -> str:
        blob = json.dumps({"m": self.model, "s": state,
                           "q": {k: q.body() for k, q in questions.items()}},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        try:
            return json.loads((self.cache_dir / f"{key}.json").read_text())
        except (OSError, ValueError):
            return None

    def _cache_put(self, key: str, answers: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(answers))
        except OSError:
            pass

    # ------------------------------------------------------------------ transport

    def _post(self, state: Any, questions: dict[str, Question]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "state": state,
                                "questions": {k: q.body() for k, q in questions.items()}}
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = time.monotonic()
            try:
                with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=5.0)) as c:
                    r = c.post(self.url, json=body)
                if r.status_code in (404, 405, 501):
                    raise DecisionUnavailable(
                        f"{self.url} answered {r.status_code} (Ollama >= 0.35 and "
                        f"`ollama pull {self.model}` needed): {r.text[:120]}")
                if r.status_code in (408, 425, 429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request,
                                                response=r)
                if r.status_code >= 400:
                    raise DecisionError(f"HTTP {r.status_code}: {r.text[:200]}")
                data = r.json()
                if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
                    raise DecisionError("response has no 'answers' object")
                self.stats["calls"] += 1
                self.stats["questions"] += len(questions)
                self.stats["seconds"] += time.monotonic() - t0
                self.stats["input_tokens"] += int(
                    (data.get("usage") or {}).get("input_tokens") or 0)
                return data["answers"]
            except (httpx.TransportError, httpx.HTTPStatusError) as e:  # transient
                last = e
                if isinstance(e, httpx.ConnectError) and attempt >= 1:
                    break
                self.stats["retries"] += 1
                self._sleep(min(2.0 ** attempt, 8.0))
            except ValueError as e:  # invalid JSON
                last = e
                self.stats["retries"] += 1
                self._sleep(0.5)
        self.stats["failures"] += 1
        if isinstance(last, httpx.ConnectError):
            raise DecisionUnavailable(f"cannot reach {self.url}") from last
        raise DecisionError(f"{type(last).__name__}: {last}") from last

    # ------------------------------------------------------------------ public API

    def ask(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        """Answer every question about ``state``. Splits into batches of ``max_questions``;
        choice questions with more than ``max_options`` options are answered in rounds. Raises
        DecisionError/DecisionUnavailable when a batch cannot be answered."""
        narrow = {k: q for k, q in questions.items()
                  if q.type != "choice" or len(q.criteria or {}) <= self.max_options}
        wide = {k: q for k, q in questions.items() if k not in narrow}
        out: dict[str, Answer] = {}
        names = list(narrow)
        for i in range(0, len(names), self.max_questions):
            batch = {k: narrow[k] for k in names[i: i + self.max_questions]}
            out.update(self._ask_batch(state, batch))
        for k, q in wide.items():
            out[k] = self._ask_wide(state, q)
        return out

    def _ask_batch(self, state: Any, batch: dict[str, Question]) -> dict[str, Answer]:
        self._used = True
        key = self._key(state, batch)
        raw = self._cache_get(key)
        if raw is not None:
            self.stats["cache_hits"] += 1
        else:
            raw = self._post(state, batch)
        answers: dict[str, Answer] = {}
        for name, q in batch.items():
            if name not in raw:
                raise DecisionError(f"answer for {name!r} missing")
            answers[name] = parse_answer(raw[name], q)
        self._cache_put(key, {k: raw[k] for k in batch})
        return answers

    def _ask_wide(self, state: Any, q: Question) -> Answer:
        """Choice with too many options: groups of <= max_options-1 -> the best few of each
        group meet in a final round. Non-finalists keep a small share of their group mass."""
        crit = dict(q.criteria or {})
        size = max(2, self.max_options - 1)
        keys = list(crit)
        groups = [keys[i: i + size] for i in range(0, len(keys), size)]
        qs = {f"g{n}": choice(q.instructions, {k: crit[k] for k in g})
              for n, g in enumerate(groups)}
        round1 = self.ask(state, qs)
        per_group = max(2, self.max_options // len(groups))
        finalists: list[str] = []
        local: dict[str, float] = {}
        for ans in round1.values():
            top = sorted(ans.probabilities.items(), key=lambda kv: -kv[1])[:per_group]
            finalists += [k for k, _ in top]
            local.update(ans.probabilities)
        finalists = finalists[: self.max_options]
        final = self._ask_batch(state, {"final": choice(
            q.instructions, {k: crit[k] for k in finalists})})["final"]
        probs = {k: 0.02 * local.get(k, 0.0) / max(len(groups), 1) for k in crit}
        for k, v in final.probabilities.items():
            probs[k] = v
        probs = normalise(probs)
        best = max(probs, key=probs.__getitem__)
        return Answer("choice", choice=best, probabilities=probs,
                      confidence=entropy_confidence(probs))

    def loaded_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base}/api/ps", timeout=5.0)
            return [m.get("name", "") for m in r.json().get("models", [])]
        except (httpx.HTTPError, ValueError, AttributeError):
            return []

    def unload(self, model: str | None = None) -> None:
        """Free the VRAM of ``model`` (default: the decision model) on an Ollama server."""
        name = model or self.model
        try:
            httpx.post(f"{self.base}/api/generate", json={"model": name, "keep_alive": 0},
                       timeout=15.0)
        except httpx.HTTPError:
            pass

    def free_others(self) -> list[str]:
        """Unload every OTHER model Ollama holds (one big model on the GPU at a time)."""
        freed = []
        for name in self.loaded_models():
            if name and name.removesuffix(":latest") != self.model.removesuffix(":latest"):
                self.unload(name)
                freed.append(name)
        return freed

    def close(self) -> None:
        if self._used and self.unload_when_done:
            self.unload()
            self._used = False

    def summary(self) -> str:
        s = self.stats
        return (f"{s['calls']} call(s), {s['questions']} question(s), {s['cache_hits']} cached, "
                f"{s['retries']} retried, {s['failures']} failed, {s['seconds']:.1f}s")
