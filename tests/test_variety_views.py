"""Reuse variety: near-duplicate families, fresh-asset bonus, virtual sub-asset views."""

import numpy as np

from src.assignment import families, global_assign
from src.schemas import SceneBeat
from src.views import candidate_regions, crop_scale, free_window, iou, pick_region


def test_families_chain_near_duplicates():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.99, 0.05, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    f = families({"a": a, "b": b, "c": c}, 0.9)
    assert f["a"] == f["b"] != f["c"]


def test_family_counts_as_one_for_window():
    # beats 0 and 1 are 1 s apart; assets 0 and 1 are near-duplicates, asset 2 is different
    s = np.array([[0.9, 0.88, 0.80], [0.9, 0.88, 0.80]])
    out = global_assign(s, [3, 3, 3], [False] * 3, [0.0, 4.0], [3.0, 7.0], 22.0,
                        clip_target=0.0, family=[0, 0, 1], max_family_loss=0.2)
    assert {out[0], out[1]} != {0, 1} and not (out[0] in (0, 1) and out[1] in (0, 1))


def test_family_swap_never_costs_more_than_cap():
    s = np.array([[0.9, 0.88, 0.2], [0.9, 0.88, 0.2]])
    out = global_assign(s, [3, 3, 3], [False] * 3, [0.0, 4.0], [3.0, 7.0], 22.0,
                        clip_target=0.0, family=[0, 0, 1], max_family_loss=0.05)
    assert out[0] in (0, 1) and out[1] in (0, 1)  # accuracy kept: the only alternative is poor


def test_fresh_bonus_prefers_unused_asset():
    s = np.array([[0.50, 0.49]])
    assert global_assign(s, [3, 3], [False] * 2, [0.0], [3.0], clip_target=0.0,
                         fresh_bonus=0.0)[0] == 0
    s2 = np.array([[0.50, 0.49], [0.60, 0.0]])  # row 1 takes asset 0 -> asset 1 is the free one
    out = global_assign(s2, [3, 3], [False] * 2, [0.0, 40.0], [3.0, 43.0], clip_target=0.0)
    assert out == [1, 0]


def test_crop_regions_are_distinct_and_sharp():
    assert crop_scale(3000) == 0.6 and crop_scale(1920) is None
    regs = candidate_regions(0.6)
    assert all(0 <= r[0] and r[0] + r[2] <= 1.001 for r in regs)
    assert iou(regs[0], regs[1]) < 0.45
    first = pick_region([0.1, 0.9, 0.5, 0.2, 0.3, 0.0, 0.0], regs, [[0, 0, 1, 1]])
    second = pick_region([0.1, 0.9, 0.5, 0.2, 0.3, 0.0, 0.0], regs, [[0, 0, 1, 1], first])
    assert first == regs[1] and second != first and iou(first, second) <= 0.45


def test_free_window_avoids_used_segment():
    win = free_window([(0.0, 4.0)], 4.0, 12.0, [(0.0, 4.0, 0.9), (4.0, 12.0, 0.5)])
    assert win == (4.0, 8.0)
    assert free_window([(0.0, 4.0)], 4.0, 5.0, [(0.0, 5.0, 0.9)]) is None


def test_crop_region_schema_validation():
    assert SceneBeat(crop_region=[0.4, 0.0, 0.6, 0.6]).crop_region == [0.4, 0.0, 0.6, 0.6]
    assert SceneBeat(crop_region=[0.8, 0.0, 0.6, 0.6]).crop_region is None
    assert SceneBeat(crop_region="bad").crop_region is None
