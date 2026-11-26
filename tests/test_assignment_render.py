"""Global assignment, sidecar clip hints, render cache/draft/QA helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import semantic_planner as sp
from src.assignment import global_assign, is_action
from src.stage6_render import (
    CompileContext,
    RenderSettings,
    _prune_cache,
    beat_cache_key,
    compile_beat,
)


def test_assignment_respects_caps_and_window() -> None:
    # 6 beats all prefer asset 0; caps 2, window 10 s -> spread over the other assets
    s = np.array([[0.9, 0.5, 0.4, 0.3]] * 6)
    starts = [0, 4, 8, 30, 34, 38]
    ends = [4, 8, 12, 34, 38, 42]
    pick = global_assign(s, [2, 2, 2, 2], [False] * 4, starts, ends, window=10, clip_target=0)
    assert all(p is not None for p in pick)
    assert max(pick.count(a) for a in range(4)) <= 2
    for i in range(6):
        for j in range(i + 1, 6):
            if pick[i] == pick[j]:
                assert starts[j] - ends[i] >= 10


def test_assignment_minimum_clip_share() -> None:
    s = np.array([[0.5, 0.45], [0.5, 0.44], [0.5, 0.43], [0.5, 0.42]])
    # asset 0 = photo (cap 4), asset 1 = clip (cap 4)
    pick = global_assign(s, [4, 4], [False, True], [0, 10, 20, 30], [5, 15, 25, 35],
                         window=1, clip_target=0.5)
    assert sum(p == 1 for p in pick) >= 2


def test_action_detection() -> None:
    assert is_action("Astronauts installed the new camera during the spacewalk")
    assert not is_action("The mirror was ground to a precise curve")


def test_clip_hints_avoid_segments_and_end() -> None:
    h = {"start": None, "end": 20.0, "avoid": [(0.0, 9.0)], "segments": []}
    assert sp.apply_clip_hints(2.0, 3.0, 30.0, h) == (9.0, 12.0)
    h2 = {"start": None, "end": None, "avoid": [], "segments": [(10.0, 14.0)]}
    st, en = sp.apply_clip_hints(1.0, 3.0, 30.0, h2)
    assert 10.0 <= st and en <= 14.0
    assert sp.apply_clip_hints(18.0, 4.0, 30.0, h)[1] <= 20.0


def test_sidecar_hints_parse(tmp_path: Path) -> None:
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"")
    (tmp_path / "c.mp4.txt").write_text("opens on a studio. media_start: 9.2 media_end: 20 "
                                        "avoid: 12-13.5, 15-16 segments: 9-14")
    hints = sp.sidecar_clip_hints({"filepath": str(clip)})
    assert hints["start"] == 9.2 and hints["end"] == 20.0
    assert hints["avoid"] == [(12.0, 13.5), (15.0, 16.0)] and hints["segments"] == [(9.0, 14.0)]


def test_split_legibility_for_1600_diagram() -> None:
    a = {"width": 1600, "height": 900}
    assert sp.schematic_scale_in_split(a) > 0.6 and sp.split_legible(a)


def test_cache_key_changes_with_beat_fields(tmp_path: Path) -> None:
    from PIL import Image
    from test_review_fixes import asset, make_resolver

    img = tmp_path / "a.png"
    Image.new("RGB", (1920, 1080), (10, 10, 10)).save(img)
    ctx = CompileContext(make_resolver(tmp_path, [asset(img, "image")]), RenderSettings(),
                         tmp_path, tmp_path)
    beat = {"id": 0, "visual_template": "fullscreen_clip", "primary_source": "local",
            "source_id": "image_a", "beat_style": "normal", "start_time": 0, "end_time": 3}
    k1 = beat_cache_key(beat, 90, 0, 0, ctx, "r")
    assert k1 == beat_cache_key(dict(beat), 90, 0, 0, ctx, "r")
    assert k1 != beat_cache_key({**beat, "callout_text": "X"}, 90, 0, 0, ctx, "r")
    assert k1 != beat_cache_key(beat, 91, 0, 0, ctx, "r")
    assert k1 != beat_cache_key(beat, 90, 5, 0, ctx, "r")
    assert k1 != beat_cache_key(beat, 90, 0, 0, ctx, "r2")
    assert compile_beat(beat, 90, ctx).graph


def test_prune_cache_keeps_used(tmp_path: Path) -> None:
    import os
    import time

    a, b = tmp_path / "beat_a.mp4", tmp_path / "beat_b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    old = time.time() - 40 * 86400
    os.utime(a, (old, old))
    os.utime(b, (old, old))
    _prune_cache(tmp_path, {a})
    assert a.exists() and not b.exists()
    json.dumps({})


def test_defect_line_penalises_ideal_diagram_and_marks_defocus(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from src.assignment import state_penalty
    from src.schemas import SceneBeat

    line = "The first image came back blurry, the mirror was wrong"
    assert state_penalty(line, "ideal") < 0 < state_penalty(line, "flawed")
    assert state_penalty("now the image is sharp and corrected", "flawed") < 0
    assert state_penalty("a neutral line", "ideal") == 0
    side = tmp_path / "d.png"
    side.write_bytes(b"")
    (tmp_path / "d.png.txt").write_text("diagram. state: flawed")
    assert sp.sidecar_state({"id": "x", "filepath": str(side)}) == "flawed"
    assert sp.sidecar_state({"id": "schematic_perfect_mirror", "filepath": "/nope"}) == "ideal"
    photo = {"id": "image_a", "asset_type": "image", "description": "image | a | Hubble"}
    ctx = SimpleNamespace(assets_by_id={"image_a": photo})
    b = SceneBeat(id=0, start_time=0, end_time=3, visual_template="fullscreen_clip",
                  source_id="image_a")
    sp._mark_defocus(b, ctx, line)
    assert b.illustrative_defocus
    sp._mark_defocus(b, ctx, "the corrected image is sharp")
    assert not b.illustrative_defocus
