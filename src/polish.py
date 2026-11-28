"""Polish pass: editing rules applied to a scene plan (new or existing).

Rules (each is a pure function over the list of beat dicts, so the same code serves the
planner, the ``polish`` command and the QA/``fix --review`` checks):

1. transitions: no side-to-side movement (slide/smooth/wipe) -> cut/crossfade/dissolve only;
   glitch joins only when the theme asks for them;
2. reading time: a diagram/graph beat stays up ``max(3.5, 0.35 s per legible word + 1)`` s
   (cap 9 s); the time comes from the next beat (the diagram may stay on during the following
   phrase), absorbed short neighbours or, at most 1 s, the previous beat;
3. minimum beat length (2.2 s, 3 s for diagrams): extend from neighbours with slack, else merge;
4. no repeated asset/near-duplicate in the final ``tail_window`` s versus the previous
   ``repeat_window`` s (replace with a fitting unused asset, else merge the beat);
5. overlays/cutouts/PiPs must be cited by the narration, otherwise they are removed.

Total duration and contiguity are preserved: only boundaries move or beats merge.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from src.transitions import GLITCH, MOTION

DEFAULTS: dict[str, Any] = {
    "min_beat": 2.2,
    "min_text_beat": 3.0,
    "read_min": 3.5,
    "read_per_word": 0.35,
    "read_extra": 1.0,
    "read_max": 9.0,
    "max_prev_take": 1.0,       # seconds a diagram may start before its sentence
    "tail_window": 25.0,
    "repeat_window": 20.0,
    "overlay_terms_min": 1,
    "generic_terms": ["hubble", "telescope", "space", "nasa", "image", "photo"],
}
FS, SPLIT, CUT, PUNCH = "fullscreen_clip", "split_schematic", "cutout_callout", "punch_in_emphasis"

_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with", "which", "while", "than", "then", "there", "these", "those", "up", "down", "over", "under", "about", "after", "before", "between", "both", "each", "other", "one", "two", "three", "diagram", "dark", "blue", "labelled", "labeled", "label", "text", "says", "drawn", "shows", "shown", "row", "rows", "box", "boxes", "left", "right"])


def settings(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("polish") or {})
    return out


def terms(text: str, generic: frozenset[str] = frozenset()) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) < 3 or w in _STOP or w in generic:
            continue
        for suf in ("ing", "es", "s", "ed"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def _desc(asset: dict[str, Any] | None) -> str:
    if not asset:
        return ""
    return f"{asset.get('description') or ''} {' '.join(asset.get('tags') or [])}"


def diagram_words(asset: dict[str, Any] | None) -> int:
    """Legible words of a diagram, from its sidecar text (after the last ``|``): quoted labels
    plus capitalised/upper-case words; without any, half of the content words."""
    if not asset or asset.get("asset_type") != "schematic":
        return 0
    tail = str(asset.get("description") or "").split("|")[-1].split(":", 1)[-1]
    quoted = re.findall(r"'([^']{2,})'|\"([^\"]{2,})\"", tail)
    n = sum(len(re.findall(r"[\w.%~µ]+", a or b)) for a, b in quoted)
    rest = re.sub(r"'[^']*'|\"[^\"]*\"", " ", tail)
    n += sum(1 for w in re.findall(r"[A-Za-z0-9]+", rest)[1:] if w[0].isupper() or w.isdigit())
    return n or round(0.5 * len(terms(tail)))


def read_need(beat: dict[str, Any], assets: dict[str, dict[str, Any]], c: dict[str, Any]) -> float:
    """Seconds a beat must stay on screen so its diagram can be read (0 = no diagram)."""
    ids = [beat.get("source_id")] if beat.get("visual_template") in (FS, PUNCH) else []
    if beat.get("visual_template") == SPLIT:
        ids.append(beat.get("overlay_asset"))
    n = [diagram_words(assets.get(i)) for i in ids if i and (assets.get(i) or {}).get("asset_type") == "schematic"]
    if not n:
        return 0.0
    return min(float(c["read_max"]), max(float(c["read_min"]),
                                         float(c["read_per_word"]) * max(n) + float(c["read_extra"])))


def floor_of(beat: dict[str, Any], assets: dict[str, dict[str, Any]], c: dict[str, Any]) -> float:
    need = read_need(beat, assets, c)
    return max(float(c["min_text_beat"]), need) if need else float(c["min_beat"])


def _dur(b: dict[str, Any]) -> float:
    return float(b["end_time"]) - float(b["start_time"])


def _sync(b: dict[str, Any]) -> None:
    b["start_time"] = round(float(b["start_time"]), 3)
    b["end_time"] = round(float(b["end_time"]), 3)
    b["duration"] = round(b["end_time"] - b["start_time"], 3)


def speech(words: list[dict[str, Any]], b: dict[str, Any], pad: float = 0.0) -> str:
    return " ".join(str(w["word"]).strip() for w in words
                    if b["start_time"] - pad - 1e-3 <= float(w["start"]) < b["end_time"] + pad - 1e-3)


def asset_ids(b: dict[str, Any]) -> list[str]:
    return [x for x in (b.get("source_id"), b.get("overlay_asset"), b.get("pip_asset")) if x]


def family(asset_id: str) -> str:
    """Near-duplicate key from the file name (year, ``_small`` and the type prefix dropped)."""
    s = re.sub(r"^(image|clip|schematic)_", "", asset_id.lower())
    s = re.sub(r"(^|_)(19|20)\d\d(?=_|$)|_small(?=_|$)", "", s)
    return s.strip("_")


@dataclass
class Report:
    changes: list[tuple[int, str, str]] = field(default_factory=list)   # (beat id, kind, text)
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)

    def add(self, beat: int, kind: str, text: str) -> None:
        self.changes.append((beat, kind, text))

    def count(self, kind: str) -> int:
        return sum(1 for _, k, _ in self.changes if k == kind)


# --------------------------------------------------------------------------- boundary helpers


def _snap(words: list[dict[str, Any]], t: float, lo: float, hi: float, up: bool = True) -> float:
    """Word start near ``t`` inside [lo, hi] (>= t when ``up``); else ``t`` clamped."""
    t = min(max(t, lo), hi)
    cands = [float(w["start"]) for w in words if lo <= float(w["start"]) <= hi
             and (float(w["start"]) >= t - 1e-6 if up else abs(float(w["start"]) - t) <= 0.4)]
    if not cands:
        return t
    best = min(cands, key=lambda x: abs(x - t))
    return best if abs(best - t) <= 0.45 else t


def _move(beats: list[dict[str, Any]], i: int, t: float) -> None:
    """Move the boundary between beat i and i+1 to ``t``."""
    delta = t - beats[i]["end_time"]
    beats[i]["end_time"] = t
    beats[i + 1]["start_time"] = t
    _sync(beats[i])
    _sync(beats[i + 1])
    _stretch(beats[i], delta)


def _stretch(b: dict[str, Any], delta: float) -> None:
    if b.get("media_end") is not None and delta > 0:
        b["media_end"] = round(float(b["media_end"]) + delta, 3)


def _absorb(beats: list[dict[str, Any]], i: int, into: int) -> dict[str, Any]:
    """Delete beat i; its time goes to the neighbour ``into`` (i-1 or i+1)."""
    gone = beats.pop(i)
    j = min(i, into)
    if into < i:
        beats[j]["end_time"] = gone["end_time"]
        _stretch(beats[j], _dur(gone))
    else:
        beats[j]["start_time"] = gone["start_time"]
    _sync(beats[j])
    return gone


# --------------------------------------------------------------------------- 1. transitions


def fix_transitions(beats: list[dict[str, Any]], rep: Report, glitch: bool = False) -> None:
    for i, b in enumerate(beats):
        t = b.get("transition")
        if t in MOTION:
            new = "dissolve" if b.get("beat_style") == "technical" else "crossfade"
            b["transition"] = new
            b["transition_duration"] = None
            rep.add(b["id"], "transition", f"{t} -> {new}")
        elif t in GLITCH and not glitch:
            b["transition"] = "none"
            b["transition_duration"] = None
            rep.add(b["id"], "transition", f"{t} -> none (theme has no glitch)")
    if beats and beats[0].get("transition") not in (None, "none"):
        beats[0]["transition"] = "none"


# --------------------------------------------------------------------------- 5. overlays


def overlay_cited(b: dict[str, Any], asset_id: str | None, assets: dict[str, dict[str, Any]],
                  words: list[dict[str, Any]], c: dict[str, Any]) -> bool:
    generic = frozenset(c["generic_terms"])
    subj = terms(_desc(assets.get(asset_id or "")) or str(asset_id).replace("_", " "), generic)
    heard = terms(speech(words, b, pad=0.4), generic)
    return len(subj & heard) >= int(c["overlay_terms_min"])


def fix_overlays(beats: list[dict[str, Any]], assets: dict[str, dict[str, Any]],
                 words: list[dict[str, Any]], c: dict[str, Any], rep: Report) -> None:
    for b in beats:
        if b.get("visual_template") == CUT and b.get("overlay_asset") \
                and not overlay_cited(b, b["overlay_asset"], assets, words, c):
            gone = b["overlay_asset"]
            if not b.get("source_id") or b["source_id"] == gone:
                b["source_id"] = gone
            b.update(visual_template=FS, overlay_asset=None, cutout_path=None, badge_path=None,
                     emphasis=False)
            rep.add(b["id"], "overlay", f"cutout of {gone} not mentioned in the narration -> "
                                         "plain fullscreen")
        if b.get("overlay_mode") == "pip" and b.get("pip_asset") \
                and not overlay_cited(b, b["pip_asset"], assets, words, c):
            rep.add(b["id"], "overlay", f"picture-in-picture {b['pip_asset']} not mentioned -> removed")
            b.update(overlay_mode=None, pip_asset=None, pip_corner=None, pip_start=None,
                     pip_delay=None)


# --------------------------------------------------------------------------- 2. reading time


def _floor(beats: list[dict[str, Any]], i: int, assets: dict[str, dict[str, Any]],
           c: dict[str, Any]) -> float:
    """Hard floor a beat can be squeezed to (the reading ``need`` is a target, not a floor)."""
    return float(c["min_text_beat"] if read_need(beats[i], assets, c) else c["min_beat"])


def fix_reading(beats: list[dict[str, Any]], assets: dict[str, dict[str, Any]],
                words: list[dict[str, Any]], c: dict[str, Any], rep: Report) -> None:
    i = 0
    while i < len(beats):
        need = read_need(beats[i], assets, c)
        deficit = need - _dur(beats[i])
        if need and deficit > 0.05:
            start_dur = _dur(beats[i])
            # 1) the following phrase: shrink the next beat down to its floor, absorbing it
            #    whole when it is already shorter than a minimal beat
            while deficit > 0.05 and i + 1 < len(beats):
                j = i + 1
                nxt_dur, nxt_floor = _dur(beats[j]), _floor(beats, j, assets, c)
                if read_need(beats[j], assets, c) == 0 and nxt_dur < float(c["min_beat"]):
                    gone = _absorb(beats, j, i)
                    rep.add(gone["id"], "merge", f"short beat absorbed by diagram beat {beats[i]['id']}")
                    deficit = need - _dur(beats[i])
                    continue
                if read_need(beats[j], assets, c):  # never rob another diagram of its time
                    nxt_floor = max(nxt_floor, min(_dur(beats[j]), read_need(beats[j], assets, c)))
                avail = nxt_dur - nxt_floor
                if avail <= 0.05:
                    break
                t = beats[i]["end_time"] + min(avail, deficit)
                t = _snap(words, t, beats[i]["end_time"], beats[j]["end_time"] - nxt_floor)
                _move(beats, i, t)
                deficit = need - _dur(beats[i])
                break
            # 2) at most max_prev_take s from the previous beat
            if deficit > 0.05 and i > 0:
                pf = _floor(beats, i - 1, assets, c)
                if read_need(beats[i - 1], assets, c):
                    pf = max(pf, min(_dur(beats[i - 1]), read_need(beats[i - 1], assets, c)))
                avail = min(_dur(beats[i - 1]) - pf, float(c["max_prev_take"]))
                if avail > 0.05:
                    t = beats[i]["start_time"] - min(avail, deficit)
                    t = _snap(words, t, beats[i - 1]["start_time"] + pf,
                              beats[i]["start_time"], up=False)
                    _move(beats, i - 1, t)
            gained = _dur(beats[i]) - start_dur
            if gained > 0.05:
                rep.add(beats[i]["id"], "read",
                        f"diagram on screen {start_dur:.1f}s -> {_dur(beats[i]):.1f}s "
                        f"(needs {need:.1f}s)")
        i += 1


# --------------------------------------------------------------------------- 3. minimum length


def fix_short(beats: list[dict[str, Any]], assets: dict[str, dict[str, Any]],
              words: list[dict[str, Any]], c: dict[str, Any], rep: Report) -> None:
    skip: set[int] = set()
    for _ in range(4 * len(beats) + 4):
        if len(beats) < 2:
            return
        idx = next((i for i, b in enumerate(beats) if b["id"] not in skip
                    and _dur(b) < _floor(beats, i, assets, c) - 0.05), None)
        if idx is None:
            return
        i = idx
        fl = _floor(beats, i, assets, c)
        deficit = fl - _dur(beats[i])
        nxt = _dur(beats[i + 1]) - _floor(beats, i + 1, assets, c) if i + 1 < len(beats) else 0.0
        prv = _dur(beats[i - 1]) - _floor(beats, i - 1, assets, c) if i > 0 else 0.0
        nxt, prv = max(nxt, 0.0), max(prv, 0.0)
        if nxt + prv >= deficit - 0.05:
            old = _dur(beats[i])
            take_n = min(nxt, deficit)
            if take_n > 0.05:
                t = _snap(words, beats[i]["end_time"] + take_n, beats[i]["end_time"],
                          beats[i + 1]["end_time"] - _floor(beats, i + 1, assets, c))
                _move(beats, i, t)
            rest = fl - _dur(beats[i])
            if rest > 0.05 and i > 0:
                t = _snap(words, beats[i]["start_time"] - min(prv, rest),
                          beats[i - 1]["start_time"] + _floor(beats, i - 1, assets, c),
                          beats[i]["start_time"], up=False)
                _move(beats, i - 1, t)
            rep.add(beats[i]["id"], "short", f"beat {old:.1f}s -> {_dur(beats[i]):.1f}s (min {fl:.1f}s)")
            if _dur(beats[i]) >= fl - 0.05:
                continue
        if read_need(beats[i], assets, c):
            # a diagram is never merged away: take a short plain neighbour into it, else keep
            small = [j for j in (i + 1, i - 1) if 0 <= j < len(beats)
                     and not read_need(beats[j], assets, c) and _dur(beats[j]) <= fl]
            if small:
                gone = _absorb(beats, small[0], i)
                rep.add(gone["id"], "merge", f"short beat absorbed by diagram beat {beats[i if small[0] > i else i - 1]['id']}")
            else:
                skip.add(beats[i]["id"])
            continue
        # cannot be lengthened: merge into the better neighbour (a diagram beat first, else longer)
        cands = [j for j in (i - 1, i + 1) if 0 <= j < len(beats)]
        into = max(cands, key=lambda j: (_dur(beats[j]) + _dur(beats[i]) <= float(c["read_max"]),
                                         read_need(beats[j], assets, c) > 0, _dur(beats[j])))
        gone = _absorb(beats, i, into)
        rep.add(gone["id"], "merge", f"short beat ({_dur(gone):.1f}s) merged into beat "
                                     f"{beats[min(i, into)]['id']}")


# --------------------------------------------------------------------------- 4. tail repeats


def _horizon(beats: list[dict[str, Any]], c: dict[str, Any]) -> float:
    """Earliest time an ending beat must not share assets with: the ``repeat_window`` before
    the start of the final ``tail_window``."""
    return beats[-1]["end_time"] - float(c["tail_window"]) - float(c["repeat_window"])


def _conflicts(beats: list[dict[str, Any]], i: int, horizon: float) -> set[str]:
    fams = {family(a): a for a in asset_ids(beats[i])}
    out: set[str] = set()
    for j in range(i):
        if beats[j]["end_time"] > horizon:
            for a in asset_ids(beats[j]):
                if family(a) in fams:
                    out.add(fams[family(a)])
    return out


def _replacement(beats: list[dict[str, Any]], i: int, bad: str, assets: dict[str, dict[str, Any]],
                 words: list[dict[str, Any]], c: dict[str, Any]) -> str | None:
    kind = (assets.get(bad) or {}).get("asset_type")
    if kind == "schematic":
        return None
    hz = _horizon(beats, c)
    near = {family(a) for b in beats if b["end_time"] > hz for a in asset_ids(b)}
    used = {family(a) for b in beats for a in asset_ids(b)}
    generic = frozenset(c["generic_terms"])
    heard = terms(speech(words, beats[i], pad=0.4), generic)
    best, score = None, (0, 0.0)
    for aid, a in assets.items():
        if a.get("asset_type") not in ("clip", "image") or family(aid) in near:
            continue
        if a.get("asset_type") == "clip" and float(a.get("duration") or 9) < _dur(beats[i]) - 0.5:
            continue
        ov = len(terms(_desc(a) + " " + aid.replace("_", " "), generic) & heard)
        key = (ov, 1.0 if family(aid) not in used else 0.0)
        if ov >= 1 and key > score:
            best, score = aid, key
    return best


def fix_tail_repeats(beats: list[dict[str, Any]], assets: dict[str, dict[str, Any]],
                     words: list[dict[str, Any]], c: dict[str, Any], rep: Report) -> None:
    if not beats:
        return
    win = float(c["repeat_window"])
    i = 0
    while i < len(beats):
        hz = _horizon(beats, c)
        tail_start = beats[-1]["end_time"] - float(c["tail_window"])
        if beats[i]["start_time"] < tail_start - 1e-6:
            i += 1
            continue
        bad = _conflicts(beats, i, hz)
        if not bad:
            i += 1
            continue
        b = beats[i]
        done = True
        for a in sorted(bad):
            new = _replacement(beats, i, a, assets, words, c)
            if new is None:
                done = False
                break
            for k in ("source_id", "overlay_asset", "pip_asset"):
                if b.get(k) == a:
                    b[k] = new
            b.update(media_start=None, media_end=None, crop_region=None, highlights=None,
                     media_path=None, stock_asset_id=None, cutout_path=None, badge_path=None)
            if b.get("primary_source") != "local":
                b["primary_source"] = "local"
            rep.add(b["id"], "repeat", f"{a} repeated from the last {win:.0f}s before the ending -> {new}")
        if done:
            i += 1
            continue
        if len(beats) < 2:
            return
        cands = [j for j in (i - 1, i + 1) if 0 <= j < len(beats)]
        clean = [j for j in cands if not (set(map(family, asset_ids(beats[j])))
                                          & set(map(family, bad)))]
        into = max(clean or cands, key=lambda j: _dur(beats[j]))
        gone = _absorb(beats, i, into)
        rep.add(gone["id"], "repeat", f"{', '.join(sorted(bad))} repeated in the ending, no "
                                       f"fresh fit -> beat merged into beat "
                                       f"{beats[min(i, into)]['id']}")


# --------------------------------------------------------------------------- driver / audit


def polish_beats(beats: list[dict[str, Any]], assets: dict[str, dict[str, Any]],
                 words: list[dict[str, Any]], cfg: dict[str, Any] | None = None, *,
                 glitch: bool = False) -> tuple[list[dict[str, Any]], Report]:
    """Apply every rule; returns (new beats, report). Total duration is unchanged."""
    c = settings(cfg)
    rep = Report()
    out = copy.deepcopy(beats)
    rep.before = stats(out, assets, c)
    fix_transitions(out, rep, glitch)
    fix_overlays(out, assets, words, c, rep)
    fix_reading(out, assets, words, c, rep)
    fix_short(out, assets, words, c, rep)
    fix_tail_repeats(out, assets, words, c, rep)
    fix_short(out, assets, words, c, rep)
    for i, b in enumerate(out):
        b["id"] = i
        _sync(b)
    rep.after = stats(out, assets, c)
    return out, rep


def stats(beats: list[dict[str, Any]], assets: dict[str, dict[str, Any]], c: dict[str, Any]) -> dict[str, Any]:
    tr: dict[str, int] = {}
    for b in beats:
        t = b.get("transition") or "auto"
        tr[t] = tr.get(t, 0) + 1
    return {"beats": len(beats), "transitions": tr,
            "diagrams": {b["id"]: [round(_dur(b), 1), round(read_need(b, assets, c), 1)]
                         for b in beats if read_need(b, assets, c)}}


def audit(beats: list[dict[str, Any]], assets: dict[str, dict[str, Any]],
          words: list[dict[str, Any]], cfg: dict[str, Any] | None = None, *,
          glitch: bool = False) -> list[tuple[int, str, str]]:
    """Read-only checks (QA / ``fix --review``): (beat id, kind, message)."""
    c = settings(cfg)
    out: list[tuple[int, str, str]] = []
    total = beats[-1]["end_time"] if beats else 0.0
    for i, b in enumerate(beats):
        d, fl, need = _dur(b), floor_of(b, assets, c), read_need(b, assets, c)
        if need and d < need - 0.3:
            out.append((b["id"], "read_time", f"diagram on screen {d:.1f}s, needs ~{need:.1f}s to read"))
        elif d < fl - 0.05:
            out.append((b["id"], "short_beat", f"beat of {d:.1f}s (minimum {fl:.1f}s)"))
        t = b.get("transition")
        if t in MOTION or (t in GLITCH and not glitch):
            out.append((b["id"], "motion_transition", f"'{t}' moves the picture sideways; use cut/fade"))
        if b.get("visual_template") == CUT and b.get("overlay_asset") \
                and not overlay_cited(b, b["overlay_asset"], assets, words, c):
            out.append((b["id"], "static_overlay", ("cutout not mentioned in the narration "
                                                    "(it would just sit there)")))
        if b["start_time"] >= total - float(c["tail_window"]) - 1e-6:
            bad = _conflicts(beats, i, _horizon(beats, c))
            if bad:
                out.append((b["id"], "tail_repeat", (f"{', '.join(sorted(bad))} repeats an asset "
                                                      f"from the last {c['repeat_window']:.0f}s in the ending")))
    return out


def assets_by_id(local_assets: dict[str, Any] | list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lst = local_assets.get("assets", []) if isinstance(local_assets, dict) else local_assets
    return {a["id"]: a for a in lst}


def transcript_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return [w for seg in transcript.get("segments", []) for w in (seg.get("words") or [])]


# --------------------------------------------------------------------------- command


def _load(episode_id: str) -> tuple[Any, dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from src.common import drop_burned, drop_excluded, episode_dir, load_config, read_json

    ep = episode_dir(episode_id)
    work = ep / "work"
    if not (work / "scene_plan.json").is_file():
        raise FileNotFoundError(f"No scene plan in {work}: run `plan {episode_id}` first.")
    plan = read_json(work / "scene_plan.json")
    tr = read_json(work / "transcript.json") if (work / "transcript.json").is_file() else {}
    raw = read_json(work / "local_assets.json").get("assets", []) \
        if (work / "local_assets.json").is_file() else []
    raw = drop_burned(drop_excluded(raw, work), work)
    return work, plan, assets_by_id(raw), transcript_words(tr), load_config()


def run(episode_id: str, dry_run: bool = False) -> Report:
    """``polish <ep>``: apply the rules to the existing plan, show a readable diff, back up to
    ``scene_plan.prev.json``, mark the plan hand-edited (like ``fix``)."""
    import shutil

    from rich.console import Console
    from rich.table import Table

    from src.agent_fix import mark_hand_edited
    from src.common import write_json
    from src.schemas import ScenePlan

    con = Console()
    work, plan, assets, words, cfg = _load(episode_id)
    old = plan["beats"]
    glitch = bool((cfg.get("render") or {}).get("glitch_transitions"))
    new, rep = polish_beats(old, assets, words, cfg, glitch=glitch)
    t = Table(title=f"Polish: {len(rep.changes)} change(s)", show_lines=True)
    for col in ("Beat", "Rule", "Change"):
        t.add_column(col, overflow="fold")
    for bid, kind, text in rep.changes:
        t.add_row(str(bid), kind, text)
    con.print(t)
    con.print(f"  beats {rep.before['beats']} -> {rep.after['beats']}; transitions "
              f"{rep.before['transitions']} -> {rep.after['transitions']}")
    con.print(f"  diagram beats [seconds, needed] before {rep.before['diagrams']}")
    con.print(f"  diagram beats [seconds, needed] after  {rep.after['diagrams']}")
    if dry_run or not rep.changes:
        con.print("  [dim]dry run: nothing written[/]" if dry_run else "  Nothing to polish.")
        return rep
    total_before = old[-1]["end_time"]
    out = ScenePlan.model_validate({**plan, "beats": new}).model_dump(mode="json")
    assert abs(out["beats"][-1]["end_time"] - total_before) < 1e-6, "polish changed the total duration"
    shutil.copy2(work / "scene_plan.json", work / "scene_plan.prev.json")
    write_json(work / "scene_plan.json", out)
    mark_hand_edited(work)
    con.print("  Plan written (backup: work/scene_plan.prev.json). Run `render` to apply.")
    return rep
