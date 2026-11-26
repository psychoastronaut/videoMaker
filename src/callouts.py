"""Callout (lower-third / badge label) generation, validation and global selection.

Callouts are short noun-phrase labels in the engineering-YouTuber style ("Program Alarm 1202",
"Core Rope Memory", "Priority Scheduler"), not narration fragments. Sources, best first:

1. the local LLM's ``callout`` field in the chunked planning call (few-shot prompt built from
   the script's own key terms, see :func:`key_terms_hint`);
2. the deterministic noun-phrase extractor (POS-lite heuristics + term salience against the
   whole script, a TF-IDF over the narration lines).

Every candidate goes through :func:`validate` (2-5 words, anchored to the narration or to the
shown asset's sidecar text, no dangling verb/adverb/adjective, no truncated word, no file name,
no age/number-only fragment, at least one salient technical term). :func:`assign` then picks
callouts for the *whole video*: only key-term beats get one, never on hype/fail payoffs unless
the phrase names the mishap, never on consecutive beats, never duplicated.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

# -ing/-ed words that are nouns in an engineering script
NOUN_ING = frozenset(
    ["landing", "wiring", "warning", "docking", "scheduling", "engineering", "computing", "programming", "processing", "routing", "coding", "timing", "tracking", "mapping", "tuning", "sensing", "switching", "buffering", "building", "housing", "bearing", "ceiling", "heading", "spring", "string", "ring", "king", "thing", "nothing", "wing"])
NOUN_ED = frozenset(["speed", "need", "seed", "hundred", "bed", "shed", "sled", "red"])
# words that read as verbs / adverbs / particles / filler even without a telltale ending
WEAK = frozenset(
    ["back", "away", "again", "off", "out", "up", "down", "more", "most", "over", "just", "only", "also", "still", "even", "very", "long", "first", "next", "last", "then", "than", "once", "ever", "never", "always", "often", "soon", "later", "early", "late", "ago", "already", "enough", "quite", "rather", "almost", "nearly", "really", "actually", "basically", "literally", "exactly", "clearly", "simply", "finally", "suddenly", "together", "apart", "forward", "ahead", "around", "along", "across", "behind", "inside", "outside", "however", "maybe", "perhaps", "indeed", "instead", "thus", "hence", "know", "knew", "known", "read", "reads", "raise", "raises", "restart", "restarts", "steal", "steals", "stole", "take", "takes", "took", "taken", "said", "say", "says", "become", "became", "happen", "happens", "use", "uses", "used", "call", "calls", "keep", "keeps", "kept", "tell", "tells", "told", "drop", "drops", "give", "gives", "gave", "get", "gets", "got", "go", "goes", "went", "gone", "come", "comes", "came", "make", "makes", "made", "let", "lets", "put", "puts", "run", "runs", "ran", "saw", "see", "sees", "seen", "show", "shows", "shown", "wove", "weave", "woven", "thread", "threw", "throw", "throws", "fall", "falls", "fell", "fill", "fills", "filled", "find", "finds", "found", "think", "thought", "want", "wants", "wanted", "stay", "stays", "try", "tries", "tried", "work", "works", "worked", "meant", "mean", "means", "cannot", "predict", "expect", "expects", "ones", "others", "another", "other", "predict", "seems", "seem", "seemed", "turn", "turns", "turned", "decide", "decided", "decides", "beat", "beats", "pass", "passes", "powerful", "useful", "useless", "careful", "beautiful", "wonderful", "different", "important", "interesting", "possible", "impossible", "simple", "clever", "honest", "calm", "young", "old", "new", "good", "bad", "big", "small", "many", "much", "more", "lot", "lots", "every", "each", "some", "any"])
GENERIC = frozenset(
    ["thing", "things", "stuff", "way", "ways", "time", "times", "part", "parts", "bit", "bits", "kind", "sort", "lot", "lots", "fraction", "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "year", "years", "week", "weeks", "month", "months", "someone", "everyone", "anyone", "nobody", "people", "man", "men", "guy", "guys"])
AGE = re.compile(r"\b\d+\s*[- ]?(year|years|yr|old)\b", re.IGNORECASE)
FILELIKE = re.compile(r"[_/\\]|\.\w{2,4}\b|\b(?:clip|image|schematic|png|jpg|mp4)\b", re.IGNORECASE)
ACRONYM = re.compile(r"^[A-Z]{2,}[0-9]*$")
SHORT_OK = frozenset(["ai", "pid", "pcb", "gpio", "adc", "dac", "i2c", "spi", "uart", "can", "led", "gps", "imu", "rpm", "psi", "agc", "lm", "cm"])
_ADJ_END = re.compile(r"(ful|ous|ive|able|ible|less|ish|ical|ary)$")
_VERB_END = re.compile(r"(ed|ing|ly)$")
MAX_CHARS = 30


def stem(token: str) -> str:
    t = token.lower().strip("'-")
    for suffix in ("ing", "es", "s", "ed"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[: -len(suffix)]
    return t


def _stopwords() -> frozenset[str]:
    from src.stage3_plan_scenes import FILLER, STOPWORDS

    return frozenset(STOPWORDS | FILLER)


def title_word(w: str) -> str:
    if w.isupper() or any(c.isdigit() for c in w):
        return w
    return w[:1].upper() + w[1:].lower()


def title_case(phrase: str) -> str:
    return " ".join(title_word(w) for w in phrase.split())


def is_nounish(tok: str, stop: frozenset[str]) -> bool:
    """POS-lite: a token that can head or modify a noun phrase."""
    low = tok.lower()
    if low in stop or low in WEAK or low in GENERIC:
        return False
    if low.isdigit():
        return False
    if len(low) < 3 and low not in SHORT_OK:
        return False
    return not (_VERB_END.search(low) and low not in NOUN_ING and low not in NOUN_ED)


@dataclass
class Salience:
    """Term salience of the whole script: TF-IDF over narration lines + domain boosts."""

    df: dict[str, int] = field(default_factory=dict)
    n: int = 1
    vocab: frozenset[str] = frozenset()
    proper: frozenset[str] = frozenset()
    script_stems: frozenset[str] = frozenset()

    @property
    def tech_min(self) -> float:  # smaller scripts have smaller IDFs
        return min(1.3, 0.5 * math.log(self.n + 1))

    def idf(self, token: str) -> float:
        return math.log((self.n + 1) / (self.df.get(stem(token), 0) + 0.5))

    def score(self, token: str) -> float:
        s = stem(token)
        v = self.idf(token)
        if s in self.vocab:
            v += 0.5
        if s in self.proper:
            v += 0.6
        return v


def build_salience(texts: Iterable[str], vocab: Iterable[str] = ()) -> Salience:
    lines = [t for t in texts if t]
    df: dict[str, int] = {}
    proper: set[str] = set()
    allst: set[str] = set()
    for t in lines:
        toks = TOKEN.findall(t)
        for i, tok in enumerate(toks):
            if i > 0 and tok[:1].isupper() and not tok.isupper() and tok.lower() != "i":
                proper.add(stem(tok))
        sts = {stem(x) for x in toks}
        allst |= sts
        for s in sts:
            df[s] = df.get(s, 0) + 1
    return Salience(df, max(len(lines), 1), frozenset(vocab), frozenset(proper),
                    frozenset(allst))


def key_terms(sal: Salience, texts: Iterable[str], limit: int = 14) -> list[str]:
    """The script's most salient noun phrases (for the LLM prompt's few-shot hint)."""
    stop = _stopwords()
    seen: dict[str, float] = {}
    for t in texts:
        for phrase, sc in candidates(t, sal, stop=stop, limit=3):
            seen[phrase] = max(seen.get(phrase, 0.0), sc)
    return [p for p, _ in sorted(seen.items(), key=lambda kv: -kv[1])[:limit]]


def _runs(text: str, stop: frozenset[str]) -> list[list[str]]:
    """Maximal runs of noun-ish tokens, broken by punctuation, function words and verbs.
    An English noun phrase is head-final, so a candidate must end where its run ends."""
    runs: list[list[str]] = []
    cur: list[str] = []
    last_end = 0
    for m in TOKEN.finditer(text):
        tok = m.group(0)
        gap = text[last_end:m.start()]
        if cur and re.search(r"[,.;:!?\u2014()]|\s-\s", gap):
            runs.append(cur)
            cur = []
        last_end = m.end()
        if is_nounish(tok, stop):
            cur.append(tok)
        elif cur and tok.isdigit() and re.fullmatch(r"\d{3,4}", tok) \
                and not re.fullmatch(r"(19|20)\d\d", tok):
            cur.append(tok)  # designator: "Program Alarm 1202"
            runs.append(cur)
            cur = []
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def candidates(text: str, sal: Salience, stop: frozenset[str] | None = None,
               limit: int = 6) -> list[tuple[str, float]]:
    """Noun-phrase candidates of one narration line, best first (Title Case): the 2-4 word
    suffixes of each noun-ish run (the head noun is the run's last word)."""
    stop = stop or _stopwords()
    out: dict[str, float] = {}
    for run in _runs(text, stop):
        for n in (2, 3, 4):
            if len(run) < n:
                break
            phrase = run[-n:]
            des = phrase[-1].isdigit()
            sc = _score_phrase(phrase[:-1] if des else phrase, sal, designator=des)
            if sc is not None and (len(phrase) >= 2):
                out[title_case(" ".join(phrase))] = sc + (1.0 if des else 0.0)
    # a longer phrase beats its own suffix ("Core Rope Memory" over "Rope Memory") unless it
    # scores much worse
    for p in list(out):
        for q in list(out):
            if p != q and p in out and q in out and q.lower().endswith(" " + p.lower()) \
                    and out[q] >= out[p] - 1.2:
                del out[p]
    ranked = sorted(out.items(), key=lambda kv: -kv[1])
    return ranked[:limit]


def _score_phrase(phrase: list[str], sal: Salience, designator: bool = False) -> float | None:
    words = [w for w in phrase]
    if words[-1].lower() in ADJ_LAST or _ADJ_END.search(words[-1].lower()):
        return None
    if len(words) >= 2 and words[0].lower() in ADJ_LAST and len(words) == 2 and not designator:
        return None  # "Hard Copy"/"Young Engineers": a bare adjective + noun is rarely a term
    sc = [sal.score(w) for w in words]
    if max(sc) < sal.tech_min:  # nothing technical/rare in it
        return None
    base = sum(sc) / len(sc) + 0.5 * max(sc)
    size_bonus = {1: -1.0, 2: 0.7, 3: 0.6, 4: 0.0}.get(len(words) + (1 if designator else 0), -1)
    if len(" ".join(words)) > MAX_CHARS:
        return None
    return base + size_bonus


ADJ_LAST = frozenset(
    ["dark", "clean", "good", "bad", "big", "small", "little", "whole", "perfect", "new", "old", "first", "last", "same", "real", "cool", "great", "nice", "long", "short", "high", "low", "hot", "cold", "fast", "slow", "tiny", "huge", "simple", "hard", "easy", "empty", "full", "bright", "quiet", "loud", "janky", "broken", "wrong", "right", "top", "bottom", "entire", "single", "young", "calm", "honest"])


def validate(co: str, text: str, sal: Salience, asset_stems: frozenset[str] = frozenset(),
             stop: frozenset[str] | None = None) -> tuple[bool, str]:
    """(ok, reason) - the deterministic gate every callout must pass, whoever wrote it."""
    stop = stop or _stopwords()
    co = (co or "").strip()
    if not co:
        return False, "empty"
    if FILELIKE.search(co) or AGE.search(co):
        return False, "filename/age"
    if len(co) > MAX_CHARS + 6:
        return False, "too long"
    words = TOKEN.findall(co)
    if not 2 <= len(words) <= 5 or " ".join(words) != " ".join(co.split()):
        return False, "word count/format"
    low = [w.lower() for w in words]
    if low[0] in stop or low[0] in WEAK or low[0] in GENERIC:
        return False, "starts with function/verb word"
    if _VERB_END.search(low[0]) and low[0] not in NOUN_ING and low[0] not in NOUN_ED:
        return False, "starts with a verb form"
    if any(w.isdigit() for w in low[:-1]):
        return False, "number inside the phrase"
    last = low[-1]
    if last.isdigit():
        if len(words) < 2 or not (3 <= len(last) <= 4) or re.fullmatch(r"(19|20)\d\d", last) \
                or not is_nounish(words[-2], stop):
            return False, "dangling number"
    elif (last in stop or last in WEAK or last in GENERIC or last in ADJ_LAST
          or (_VERB_END.search(last) and last not in NOUN_ING and last not in NOUN_ED)
          or _ADJ_END.search(last)):
        return False, "dangling verb/adverb/adjective"
    for w, lw in zip(words, low, strict=True):
        if lw.isdigit():
            continue
        if len(lw) < 3 and lw not in SHORT_OK and not ACRONYM.match(w):
            return False, "truncated word"
        if lw in {"thousand", "hundred", "million", "times", "kind", "sort"}:
            return False, "filler"
    st = [stem(w) for w in words if not w.isdigit()]
    anchored = [s for s in st if s in sal.script_stems or s in asset_stems]
    if len(anchored) < len(st) - (1 if len(st) >= 3 else 0):
        return False, "not anchored to the narration"
    nouns = [w for w in words if is_nounish(w, stop)]
    if not nouns or max(sal.score(w) for w in nouns) < sal.tech_min:
        return False, "no technical term"
    text_st = {stem(x) for x in TOKEN.findall(text)}
    if not (set(st) & (text_st | asset_stems)):
        return False, "unrelated to this beat"
    return True, "ok"


def phrase_score(co: str, sal: Salience, text: str, asset_stems: frozenset[str],
                 from_llm: bool) -> float:
    ws = [w for w in TOKEN.findall(co) if not w.isdigit()]
    sc = sum(sal.score(w) for w in ws) / max(len(ws), 1) + 0.5 * max(
        (sal.score(w) for w in ws), default=0.0)
    sc += {2: 0.7, 3: 0.6, 4: 0.1}.get(len(ws), -0.4)
    sc += 1.5 if from_llm else 0.0
    sts = {stem(w) for w in ws}
    sc += 0.8 * bool(sts & asset_stems)
    sc += 0.4 * (sts <= {stem(x) for x in TOKEN.findall(text)})  # literally in the line
    return sc


@dataclass
class Choice:
    beat_id: int
    phrase: str
    score: float
    forced: bool = False


def assign(beats: list[Any], texts: dict[int, str], ctx: Any,
           llm_cands: list[tuple[float, float, str]] | None = None, asset_stems: Any = None,
           max_ratio: float = 0.34, min_duration: float = 1.5, min_score: float = 2.6,
           sal: Salience | None = None, skip: Any = None) -> dict[int, str]:
    """Set ``callout_text`` on the key-term beats (and clear it everywhere else).

    ``asset_stems(beat) -> frozenset`` returns the stems of what the beat shows. Returns the
    {beat_id: callout} mapping that was applied."""
    stop = _stopwords()
    sal = sal or build_salience(texts.values(), getattr(ctx, "vocab", ()))
    fail = getattr(ctx, "fail_words", frozenset())
    options: dict[int, list[tuple[float, str]]] = {}
    for b in beats:
        if skip is not None and skip(b):  # e.g. a diagram that carries its own text
            options[b.id] = []
            continue
        text = texts.get(b.id, "")
        ast = asset_stems(b) if asset_stems else frozenset()
        cands: dict[str, bool] = {}
        mine = next((c for s0, e0, c in (llm_cands or [])
                     if b.start_time - 0.3 <= (s0 + e0) / 2 <= b.end_time + 0.3), None)
        for src, llm in ((mine, True), (b.callout_text, False)):
            if src:
                cands.setdefault(title_case(" ".join(src.split())), llm)
        for p, _ in candidates(text, sal, stop=stop, limit=5):
            cands.setdefault(p, False)
        scored = []
        for p, llm in cands.items():
            ok, _why = validate(p, text, sal, ast, stop)
            if not ok:
                continue
            scored.append((phrase_score(p, sal, text, ast, llm), p))
        scored.sort(reverse=True)
        options[b.id] = scored
    chosen: dict[int, str] = {}
    used_words: list[set[str]] = []
    forced = [b for b in beats if b.visual_template.value in ("cutout_callout",
                                                              "split_schematic")]

    def usable(b: Any, sc: float, p: str, forced_beat: bool) -> bool:
        if b.duration < min_duration and not forced_beat:
            return False
        sts = {stem(w) for w in TOKEN.findall(p)}
        if any(len(sts & u) / len(sts | u) >= 0.5 for u in used_words):
            return False  # duplicate / near duplicate of an earlier callout
        if b.beat_style.value in ("fail", "hype") and not forced_beat:
            return bool(sts & fail) and sc >= min_score - 0.4  # only a phrase naming the mishap
        return sc >= (min_score - 0.8 if forced_beat else min_score)

    def commit(b: Any, p: str) -> None:
        chosen[b.id] = p
        used_words.append({stem(w) for w in TOKEN.findall(p)})

    for b in forced:  # badge templates need a label: best valid phrase if any
        for sc, p in options[b.id]:
            if usable(b, sc, p, True):
                commit(b, p)
                break
    cap = max(len(chosen), round(max_ratio * len(beats)))
    pool = sorted(((sc, b, p) for b in beats if b.id not in chosen for sc, p in options[b.id][:3]),
                  key=lambda x: -x[0])
    idx = {b.id: i for i, b in enumerate(beats)}
    for sc, b, p in pool:
        if len(chosen) >= cap:
            break
        if b.id in chosen or not usable(b, sc, p, False):
            continue
        i = idx[b.id]
        if any(idx[o] in (i - 1, i + 1) for o in chosen):  # never on consecutive beats
            continue
        commit(b, p)
    for b in beats:
        b.callout_text = chosen.get(b.id)
    return chosen


def prompt_hint(texts: Iterable[str], vocab: Iterable[str] = ()) -> str:
    """Few-shot line for the planning prompt, derived from this script's own key terms."""
    texts = [t for t in texts if t]
    if not texts:
        return ""
    sal = build_salience(texts, vocab)
    terms = key_terms(sal, texts, 10)
    if not terms:
        return ""
    return ("callout: a punchy 2-4 word Title Case noun-phrase LABEL naming the technical thing "
            "or key term on screen, like a YouTube engineer's lower-third. Good examples for "
            f"this script: {', '.join(terms[:6])}. Use words from the line; never a verb, "
            "adverb, age, bare number, filler or sentence fragment (bad: 'Alarm Back', "
            "'26 Year Old Flight', 'Computer Powerful'). Not every beat needs one: use null "
            "for hype/fail payoffs, small talk and anything without a key term. Never two "
            "callouts in a row. ")
