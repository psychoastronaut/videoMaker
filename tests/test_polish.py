from src import polish
from src.stage3_plan_scenes import _TRANSITIONS_LLM
from src.transitions import MOTION, resolve_joins

DIAGRAM = ("schematic | three tests diagram | Diagram on dark blue: rows, 'Null corrector says "
           "PERFECT', 'Test 2 says FLAW', 'Lesson: investigate disagreement'")
ASSETS = {
    "schematic_a": {"id": "schematic_a", "asset_type": "schematic", "description": DIAGRAM},
    "clip_a": {"id": "clip_a", "asset_type": "clip", "duration": 20, "description": "clip | a"},
    "image_astro": {"id": "image_astro", "asset_type": "image",
                    "description": "image | Astronaut Thomas Akers in a white spacesuit"},
    "image_mirror": {"id": "image_mirror", "asset_type": "image",
                     "description": "image | mirror polishing swirl"},
}


ASSETS.update({f"clip_{i}": {"id": f"clip_{i}", "asset_type": "clip", "duration": 30,
                             "description": f"clip | nothing {i}"} for i in range(12)})


def beat(i, a, b, src=None, tmpl="fullscreen_clip", **kw):
    src = src or f"clip_{i}"
    d = {"id": i, "start_time": a, "end_time": b, "duration": b - a, "visual_template": tmpl,
         "source_id": src, "overlay_asset": None, "primary_source": "local", "beat_style": "normal",
         "transition": "none"}
    d.update(kw)
    return d


def words(*pairs):
    return [{"word": w, "start": t} for t, w in pairs]


def total(bs):
    return bs[-1]["end_time"] - bs[0]["start_time"]


def contiguous(bs):
    return all(abs(bs[i]["end_time"] - bs[i + 1]["start_time"]) < 1e-6 for i in range(len(bs) - 1))


def test_motion_transitions_not_in_planner_and_rewritten():
    assert not set(MOTION) & set(_TRANSITIONS_LLM)
    bs = [beat(0, 0, 4), beat(1, 4, 8, transition="slideleft"),
          beat(2, 8, 12, transition="smoothleft", beat_style="technical"),
          beat(3, 12, 16, transition="wipeleft")]
    out, rep = polish.polish_beats(bs, ASSETS, [])
    assert [b["transition"] for b in out] == ["none", "crossfade", "dissolve", "crossfade"]
    assert rep.count("transition") == 3 and total(out) == 16


def test_resolve_joins_never_moves_the_picture():
    bs = [beat(0, 0, 4), beat(1, 4, 8, transition="slideleft", id=1)]
    j = resolve_joins(bs, [120, 120], 30)
    assert j[1].name == "crossfade"
    assert resolve_joins(bs, [120, 120], 30, allow_motion=True)[1].name == "slideleft"


def test_diagram_gets_reading_time_from_next_beat():
    bs = [beat(0, 0, 4), beat(1, 4, 6.0, src="schematic_a"), beat(2, 6.0, 12), beat(3, 12, 16)]
    need = polish.read_need(bs[1], ASSETS, polish.settings(None))
    assert 3.5 < need <= 9
    out, rep = polish.polish_beats(bs, ASSETS, [])
    d = out[1]["end_time"] - out[1]["start_time"]
    assert d >= need - 0.01 and rep.count("read") == 1
    assert out[0]["end_time"] == 4 or out[0]["end_time"] >= 3.9   # previous beat barely moves
    assert total(out) == 16 and contiguous(out)


def test_diagram_boundary_snaps_to_a_word():
    ws = words((4.0, "a"), (6.0, "b"), (7.3, "c"), (9.0, "d"))
    bs = [beat(0, 0, 4), beat(1, 4, 6, src="schematic_a"), beat(2, 6, 14)]
    out, _ = polish.polish_beats(bs, ASSETS, ws)
    assert out[1]["end_time"] in {7.3, 9.0}


def test_short_beat_extended_or_merged():
    bs = [beat(0, 0, 5), beat(1, 5, 6.0), beat(2, 6.0, 12)]
    out, rep = polish.polish_beats(bs, ASSETS, [])
    assert len(out) == 3 and all(b["end_time"] - b["start_time"] >= 2.2 - 0.01 for b in out)
    bs = [beat(0, 0, 2.3), beat(1, 2.3, 3.3), beat(2, 3.3, 5.6)]
    out, rep = polish.polish_beats(bs, ASSETS, [])
    assert len(out) == 2 and rep.count("merge") == 1 and total(out) == 5.6 and contiguous(out)


def test_short_diagram_is_not_merged_away():
    bs = [beat(0, 0, 3), beat(1, 3, 5.5, src="schematic_a"), beat(2, 5.5, 8.5)]
    out, _ = polish.polish_beats(bs, ASSETS, [])
    assert any(b["source_id"] == "schematic_a" for b in out)


def test_tail_repeat_replaced_by_fresh_asset_or_merged():
    assets = dict(ASSETS)
    assets["image_other_mirror"] = {"id": "image_other_mirror", "asset_type": "image",
                                    "description": "image | a mirror in a lab"}
    bs = [beat(0, 0, 40, src="image_mirror"), beat(1, 40, 60, src="clip_1"),
          beat(2, 60, 64, src="image_mirror")]
    ws = words((60.0, "the"), (60.5, "mirror"))
    out, rep = polish.polish_beats(bs, assets, ws)
    assert out[-1]["source_id"] == "image_other_mirror" and rep.count("repeat") == 1
    out, rep = polish.polish_beats(bs, ASSETS, [])      # nothing fits: merged into the neighbour
    assert len(out) == 2 and total(out) == 64 and contiguous(out)
    assert not polish.audit(out, ASSETS, [])


def test_uncited_cutout_removed_cited_kept():
    cut = beat(0, 0, 4, src="clip_9", tmpl="cutout_callout", overlay_asset="image_astro",
               cutout_path="x.png", callout_text="Astronaut")
    ws = words((0.5, "mirror"), (1.0, "grinding"))
    out, rep = polish.polish_beats([cut, beat(1, 4, 8)], ASSETS, ws)
    assert out[0]["visual_template"] == "fullscreen_clip" and out[0]["cutout_path"] is None
    assert rep.count("overlay") == 1
    ws = words((0.5, "astronauts"), (1.0, "swapped"))
    out, rep = polish.polish_beats([dict(cut), beat(1, 4, 8)], ASSETS, ws)
    assert out[0]["visual_template"] == "cutout_callout" and rep.count("overlay") == 0


def test_audit_flags_each_problem():
    bs = [beat(0, 0, 4), beat(1, 4, 5.0, src="schematic_a", transition="slideleft"),
          beat(2, 5.0, 6.0), beat(3, 6.0, 60, tmpl="cutout_callout", src="clip_9",
                                  overlay_asset="image_astro"),
          beat(4, 60, 64, src="clip_9")]
    kinds = {k for _, k, _ in polish.audit(bs, ASSETS, words((7.0, "mirror")))}
    assert {"read_time", "motion_transition", "short_beat", "static_overlay",
            "tail_repeat"} <= kinds
