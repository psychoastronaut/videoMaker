"""Dashboard tests: render to a recorded Console, replay progression, snapshot export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src import dashboard as dash


def _episode(tmp_path: Path) -> Path:
    ep = tmp_path / "ep-x"
    (ep / "work").mkdir(parents=True)
    (ep / "out").mkdir()
    beats = [{"id": i, "start_time": i * 3.0, "end_time": i * 3.0 + 3, "visual_template":
              ["fullscreen_clip", "punch_in_emphasis", "cutout_callout"][i % 3], "source_id": f"clip_a{i}",
              "match_score": 0.5, "callout_text": f"Call {i}" if i % 2 else None, "beat_style": "normal",
              "visual_query": "q"} for i in range(12)]
    (ep / "work" / "scene_plan.json").write_text(json.dumps({"beats": beats, "total_duration": 36.0}))
    (ep / "work" / "transcript.json").write_text(json.dumps(
        {"duration": 36, "segments": [{"words": [1, 2, 3]}]}))
    (ep / "work" / "local_assets.json").write_text(json.dumps({"assets": [{"asset_type": "clip"}] * 4}))
    (ep / "work" / "cutouts_manifest.json").write_text(json.dumps(
        {"cutout_count": 2, "badge_count": 1, "cutout_decisions": [{"mode": "cutout"}, {"mode": "card"}],
         "warnings": []}))
    (ep / "work" / "run_log.jsonl").write_text(
        "\n".join(json.dumps(e) for e in [
            {"ts": "2026-01-01T10:00:00", "stage": "transcribe", "seconds": 4.0, "status": "ok"},
            {"ts": "2026-01-01T10:00:09", "stage": "plan", "seconds": 0.3, "status": "error", "error": "boom"},
            {"ts": "2026-01-01T10:01:00", "stage": "render", "seconds": 30.0, "status": "ok"}]))
    return ep


def _text(v: dash.View, theme: str, w: int, h: int) -> str:
    con = dash.build_console(w, h)
    con.print(dash.render(v, dash.THEMES[theme], w, h))
    return con.export_text(styles=False)


@pytest.mark.parametrize("theme", ["slate", "psycho"])
@pytest.mark.parametrize("size", [(120, 40), (160, 45)])
def test_render_fits_and_has_sections(tmp_path: Path, theme: str, size: tuple[int, int]) -> None:
    d = dash.load_data(_episode(tmp_path), "ep-x")
    out = _text(dash.static_view(d, {}), theme, *size)
    lines = out.splitlines()
    assert len(lines) == size[1]
    assert all(len(ln) <= size[0] for ln in lines)
    for token in ("AGENT PIPELINE", "Voice chain", "Render", "PREVIEW", "GPU", "session log", "loop", "✓ done"):
        assert token in out
    assert "Call 5" in out or "Call 3" in out


def test_status_and_error(tmp_path: Path) -> None:
    d = dash.load_data(_episode(tmp_path), "ep-x")
    views = dash.stage_views(d)
    assert views[1].status == "done" and views[1].counters[0] == ("words", 3)
    assert views[0].status == "idle"
    assert views[6].status == "idle"          # no final_cut.mp4
    assert views[3].status == "done"          # plan exists, so the old error is irrelevant


def test_gpu_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dash, "gpu_info", lambda: None)
    d = dash.load_data(_episode(tmp_path), "ep-x")
    assert "no nvidia-smi" in _text(dash.static_view(d, {}), "slate", 120, 40)


def test_replay_progresses(tmp_path: Path) -> None:
    d = dash.load_data(_episode(tmp_path), "ep-x")
    v0, done0 = dash.replay_view(d, {}, 0.0, 1.0, False)
    assert not done0 and all(s.status == "idle" for s in v0.stages)
    v1, _ = dash.replay_view(d, {}, 2.0, 1.0, False)
    assert any(s.status != "idle" for s in v1.stages)
    vend, done = dash.replay_view(d, {}, 1000.0, 1.0, False)
    assert done and vend.shown_beats == 12 and vend.mode == "replay"
    # speed shortens the timeline
    _, done_fast = dash.replay_view(d, {}, 20.0, 50.0, False)
    assert done_fast


def test_halfblock_and_preview(tmp_path: Path) -> None:
    img = tmp_path / "f.jpg"
    Image.new("RGB", (64, 36), (200, 10, 10)).save(img)
    th = dash.THEMES["slate"]
    con = dash.build_console(20, 6)
    con.print(dash.HalfBlock(img, 16, 4, th))
    txt = con.export_text()
    assert any(ch in txt for ch in "▀█▘")
    d = dash.load_data(_episode(tmp_path), "ep-x")
    v = dash.static_view(d, {"beat": {i: img for i in range(12)}})
    assert any(ch in _text(v, "psycho", 120, 40) for ch in "▀█▘▝▖▗▌▐▄")


@pytest.mark.parametrize("name", ["a.svg", "a.png", "a.txt"])
def test_snapshot_export(tmp_path: Path, name: str) -> None:
    d = dash.load_data(_episode(tmp_path), "ep-x")
    out = dash.export_snapshot(dash.static_view(d, {}), dash.THEMES["psycho"], tmp_path / name, 120, 40)
    assert out.stat().st_size > 500
    if name.endswith(".png"):
        assert Image.open(out).size[0] > 1000
    if name.endswith(".svg"):
        assert "<svg" in out.read_text()


def test_run_state_parses_pipeline_output(tmp_path: Path) -> None:
    rs = dash.RunState()
    for ln in [
        "╭────╮", "│ Skipping: Stage 1: Transcribe (up to date) │",
        "  Stage 2: Index Assets: inputs changed",
        "  Found 13 video clips", "  Asset index saved: /x/local_assets.json (45 assets)",
        "Stage 2: Index Assets complete",
        "  Stage 3: Plan Scenes: forced", "  chunk 2/9 ok",
        "  beat 03/25 punch_in_emphasis   2.50s none ok (1.0s)",
        "  beat 3: cutout cutout_003_x.png",
        "Error in Stage 3: Plan Scenes:",
    ]:
        rs.feed(ln)
    assert rs.status == {1: "done", 2: "done", 3: "error"} and 1 in rs.skipped
    assert rs.counts["clips"] == 13 and rs.counts["assets"] == 45 and rs.counts["cutouts"] == 1
    assert rs.chunk == (2, 9) and rs.beat == (3, 25) and rs.cur_beat == 2
    d = dash.load_data(_episode(tmp_path), "ep-x")
    views = dash.live_stage_views(rs, d)
    assert [v.status for v in views[1:4]] == ["done", "done", "error"]
    assert views[0].status == "idle"


def test_run_state_running_and_eta(tmp_path: Path) -> None:
    rs = dash.RunState()
    rs.feed("  Stage 3: Plan Scenes: forced")
    rs.feed("  chunk 2/8 ok")
    d = dash.load_data(_episode(tmp_path), "ep-x")
    v = dash.live_stage_views(rs, d)[3]
    assert v.status == "running" and ("chunk", "2/8") in v.counters
    assert v.loop == (2, 8)


def test_cpu_info_and_run_exit_code(tmp_path: Path) -> None:
    assert dash.cpu_info() is None or len(dash.cpu_info() or (0, 0)) == 2
    from typer.testing import CliRunner

    from src.cli import app

    r = CliRunner().invoke(app, ["run", "x", "--from-stage", "bogus"])
    assert r.exit_code == 1
