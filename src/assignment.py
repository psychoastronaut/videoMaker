"""Global, diversity-aware assignment of local assets to beats.

Independent per-beat picks drift to a handful of "best" assets (the same photo seven times,
clips never used). This module solves the whole episode at once on the beat x asset
similarity matrix: Hungarian assignment over capacity slots (each extra use of an asset costs
more), then a repair pass for the no-reuse window and a minimum video-clip share.
Pure numpy/scipy: no I/O, deterministic, unit-testable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

NEG = -1e6
SLOT_PENALTY = (0.0, 0.08, 0.20, 0.40)  # extra cost of the 1st, 2nd, 3rd... use of an asset
FRESH_BONUS = 0.10    # reward for putting a never-used asset on screen
FAMILY_CAP = 4        # total uses of one near-duplicate family
NEAR_DUP = 0.9        # cosine of two assets' embeddings above which they look the same


def families(vecs: dict[str, np.ndarray], threshold: float = NEAR_DUP) -> dict[str, int]:
    """Near-duplicate families: assets whose mean embeddings have cosine > ``threshold`` are
    chained (union-find) into one family id. Returns {asset_id: family_id}."""
    ids = list(vecs)
    if not ids:
        return {}
    m = np.stack([np.asarray(vecs[a], dtype=float) for a in ids])
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    sim = m @ m.T
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim[i, j] > threshold:
                parent[find(i)] = find(j)
    return {a: find(i) for i, a in enumerate(ids)}

ACTION_WORDS = {*["install", "installed", "installing", "repair", "repaired", "repairing", "fix", "fixed", "fixing", "launch", "launched", "deploy", "deployed", "deploying", "walk", "walked", "spacewalk", "spacewalks", "capture", "captured", "grab", "grabbed", "approach", "approached", "release", "released", "orbit", "orbiting", "assemble", "assembled", "assembling", "build", "built", "test", "tested", "testing", "grind", "polish", "polished", "lift", "lifted", "rotate", "open", "opened", "replace", "replaced", "attach", "attached", "swap", "swapped", "work", "worked", "working", "move", "moved", "flew", "fly", "flying", "float", "floating", "train", "trained", "training", "servicing", "mission", "crew", "astronaut", "astronauts", "arm", "robotic", "unfold", "fold"]}


def is_action(text: str) -> bool:
    """Narration that describes an action/process/event (footage beats a still)."""
    return len({w.lower() for w in re.findall(r"[A-Za-z]+", text)} & ACTION_WORDS) >= 1


def _conflict(starts: Sequence[float], ends: Sequence[float], i: int, j: int,
              window: float) -> bool:
    gap = starts[j] - ends[i] if starts[j] >= starts[i] else starts[i] - ends[j]
    return gap < window


def global_assign(score: np.ndarray, caps: Sequence[int], is_clip: Sequence[bool],
                  starts: Sequence[float], ends: Sequence[float], window: float = 22.0,
                  clip_target: float = 0.25, clip_floor: float = 0.0,
                  max_clip_loss: float = 0.12, family: Sequence[int] | None = None,
                  family_cap: int = FAMILY_CAP, fresh_bonus: float = FRESH_BONUS,
                  max_family_loss: float = 0.06) -> list[int | None]:
    """Assign one asset column to every beat row; ``score`` is NEG/-inf where ineligible.

    * every asset a is used at most ``caps[a]`` times, with growing cost per extra use;
    * the same asset is not reused within ``window`` seconds (best effort in the repair);
    * at least ``clip_target`` of the beats get a clip when clips exist whose score is
      >= ``clip_floor`` and not more than ``max_clip_loss`` below the beat's current choice.
    * ``family`` (one id per column): near-duplicate assets count as ONE for the no-reuse
      window and for ``family_cap``; a family-only clash (different assets) is only resolved when
      the replacement loses at most ``max_family_loss`` of score (accuracy first);
    * a never-used asset gets ``fresh_bonus`` (unused-asset bonus).
    Returns the chosen column per row (None when the row has no eligible asset)."""
    from scipy.optimize import linear_sum_assignment

    s = np.where(np.isfinite(score), score, NEG).astype(float)
    n, m = s.shape
    if n == 0 or m == 0:
        return [None] * n
    slots = [(a, k) for a in range(m) for k in range(max(int(caps[a]), 0))]
    if not slots:
        return [None] * n
    fam = list(family) if family is not None else list(range(m))
    val = np.stack([s[:, a] - SLOT_PENALTY[min(k, len(SLOT_PENALTY) - 1)]
                    + (fresh_bonus if k == 0 else 0.0) for a, k in slots],
                   axis=1)
    rows, cols = linear_sum_assignment(-val)
    pick: list[int | None] = [None] * n
    for r, c in zip(rows, cols, strict=True):
        a = slots[c][0]
        if s[r, a] > NEG / 2:
            pick[r] = a

    uses = [0] * m
    for a in pick:
        if a is not None:
            uses[a] += 1

    def ok_for(i: int, a: int, ignore: int | None = None) -> bool:
        if s[i, a] <= NEG / 2 or (uses[a] >= caps[a] and pick[i] != a):
            return False
        fam_uses = sum(1 for j in range(n) if j != i and pick[j] is not None
                       and fam[pick[j]] == fam[a])
        if fam_uses >= family_cap:
            return False
        return not any(j != i and j != ignore and pick[j] is not None and fam[pick[j]] == fam[a]
                       and _conflict(starts, ends, i, j, window) for j in range(n))

    # 1) no-reuse window: move the weaker of two close uses to its best allowed alternative
    stuck: set[tuple[int, int]] = set()
    for _ in range(3 * n):
        bad = [(i, j) for i in range(n) for j in range(i + 1, n)
               if pick[i] is not None and pick[j] is not None and fam[pick[i]] == fam[pick[j]]
               and (i, j) not in stuck and _conflict(starts, ends, i, j, window)]
        if not bad:
            break
        i, j = bad[0]
        loser = j if s[j, pick[j]] <= s[i, pick[i]] else i

        def alt_key(r: int, a: int) -> float:
            return -(s[r, a] + (fresh_bonus if uses[a] == 0 else 0.0))

        def movable(r: int, same: bool) -> list[int]:
            cur_r = pick[r]
            alts_r = sorted((a for a in range(m) if a != cur_r and ok_for(r, a)),
                            key=lambda a: alt_key(r, a))
            if not same:  # near-duplicate clash only: do not trade accuracy for it
                alts_r = [a for a in alts_r if s[r, cur_r] - s[r, a] <= max_family_loss]
            return alts_r
        same = pick[i] == pick[j]
        alts = movable(loser, same)
        if not alts:  # the other one may move instead
            other = i if loser == j else j
            alts = movable(other, same)
            loser = other if alts else loser
        if not alts:
            stuck.add((i, j))
            continue
        cur = pick[loser]
        uses[cur] -= 1
        pick[loser] = alts[0]
        uses[alts[0]] += 1

    # 2) minimum clip share
    clip = list(is_clip)
    need = int(np.ceil(clip_target * n)) - sum(1 for a in pick if a is not None and clip[a])
    while need > 0:
        best: tuple[float, int, int] | None = None
        for i in range(n):
            cur = pick[i]
            if cur is None or clip[cur]:
                continue
            for a in range(m):
                if not clip[a] or s[i, a] < max(clip_floor, NEG / 2) or not ok_for(i, a):
                    continue
                loss = s[i, cur] - s[i, a]
                if loss <= max_clip_loss and (best is None or loss < best[0]):
                    best = (loss, i, a)
        if best is None:
            break
        _, i, a = best
        uses[pick[i]] -= 1  # type: ignore[index]
        pick[i] = a
        uses[a] += 1
        need -= 1
    return pick


DEFECT_RE = re.compile(r"\b(blurry|blurred|blur|fuzzy|flawed|flaw|wrong|out of focus|aberration|"
                       r"error|mistake|defect|before|distorted|misshapen)\b", re.IGNORECASE)
FIXED_RE = re.compile(r"\b(sharp|fixed|corrected|correction|after|restored|razor|crisp|"
                      r"cancelled|canceled)\b", re.IGNORECASE)
BLUR_RE = re.compile(r"\b(blurry|blurred|blur|fuzzy|out of focus|soft)\b", re.IGNORECASE)
IDEAL, FLAWED = {"ideal", "after"}, {"flawed", "before"}


def state_penalty(text: str, state: str | None, weight: float = 0.15) -> float:
    """Negative adjustment: an ideal/after-state asset on a defect line (the narration says the
    image was blurry, the screen must not show the PERFECT mirror), a flawed/before-state asset
    on a 'fixed' line."""
    if not state:
        return 0.0
    defect, fixed = bool(DEFECT_RE.search(text)), bool(FIXED_RE.search(text))
    if defect and not fixed and state in IDEAL:
        return -weight
    if fixed and not defect and state in FLAWED:
        return -weight
    if defect and not fixed and state in FLAWED:
        return weight / 3  # a flawed-state asset on a defect line is what we want
    return 0.0
