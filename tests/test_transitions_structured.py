"""Ollama structured output (native /api/chat + schema) and Stage 6 transitions."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from src import stage3_plan_scenes as s3
from src.common import authoring_signature
from src.schemas import BeatStyle, SceneBeat
from src.stage6_render import CompileContext, RenderSettings, _assemble, _run
from src.transitions import XFADE, Join, auto_transition, resolve_joins

# ----------------------------------------------------------------------------- structured


def _beats(n: int) -> list[SceneBeat]:
    return [SceneBeat(id=i, start_time=i * 3.0, end_time=i * 3.0 + 3.0) for i in range(n)]


def test_chunk_schema_enums_and_counts():
    sc = s3.chunk_schema([5, 6, 7])
    arr = sc["properties"]["beats"]
    assert arr["minItems"] == arr["maxItems"] == 3
    props = arr["items"]["properties"]
    assert props["id"]["enum"] == [5, 6, 7]
    assert set(props["beat_style"]["enum"]) == {s.value for s in BeatStyle}
    assert "transition" in props and "none" in props["transition"]["enum"]
    assert set(props["transition"]["enum"]) <= set(XFADE)


def test_native_payload_and_url():
    cfg = {"structured": True, "num_ctx": 2048}
    assert s3.native_api_url("http://h:11434/v1", cfg) == "http://h:11434"
    assert s3.native_api_url("http://h:11434/v1", {"structured": False}) is None
    assert s3.native_api_url("http://h:8080", cfg) is None
    p = s3.native_payload("m", [{"role": "user", "content": "x"}], {"type": "object"}, cfg, 700)
    assert p["format"] == {"type": "object"} and p["stream"] is False
    assert p["options"]["temperature"] == 0 and p["options"]["num_ctx"] == 2048


def test_llm_chat_native_then_fallback(monkeypatch):
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        body = json.loads(req.content)
        if req.url.path == "/api/chat":
            if req.url.host == "nonative":
                return httpx.Response(404)
            assert body["format"]["type"] == "object" and body["options"]["temperature"] == 0
            return httpx.Response(200, json={"message": {"content": '{"beats": []}'}})
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    s3._NATIVE_BAD.clear()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    schema = {"type": "object"}
    kw = {"max_tokens": 10, "temperature": 0.2, "deadline": None, "request_timeout": 5.0}
    assert s3.llm_chat(client, "http://ok:11434/v1", {}, "m", [], schema, **kw) == '{"beats": []}'
    assert s3.llm_chat(client, "http://nonative:11434/v1", {}, "m", [], schema, **kw) == "{}"
    assert calls == ["/api/chat", "/api/chat", "/v1/chat/completions"]
    # a second call skips the dead native endpoint; no schema = plain /v1
    assert s3.llm_chat(client, "http://nonative:11434/v1", {}, "m", [], schema, **kw) == "{}"
    assert s3.llm_chat(client, "http://ok:11434/v1", {}, "m", [], None, **kw) == "{}"
    assert calls[3:] == ["/v1/chat/completions", "/v1/chat/completions"]
    s3._NATIVE_BAD.clear()


def _rows(tmpl: list[str], trans: list[str]) -> dict:
    return {"beats": [{"id": i, "beat_style": "normal", "visual_template": t, "transition": tr,
                       "callout_text": None, "emphasis": False}
                      for i, (t, tr) in enumerate(zip(tmpl, trans, strict=True))]}


def test_variety_rejected_then_repaired():
    texts = {i: "a b c" for i in range(5)}
    low = _rows(["punch_in_emphasis"] * 5, ["none"] * 5)
    group = _beats(5)
    problem = s3._apply_chunk(group, texts, low)
    assert "3 beats in a row" in problem
    assert all(b.transition is None for b in group)          # nothing applied on rejection
    assert s3._apply_chunk(group, texts, low, final=True) == ""  # final attempt repairs
    tm = [b.visual_template.value for b in group]
    assert not any(tm[i] == tm[i + 1] == tm[i + 2] for i in range(3))


def test_hype_beats_are_cut_and_bad_enum_rejected():
    texts = {0: "x", 1: "y"}
    data = _rows(["fullscreen_clip", "punch_in_emphasis"], ["dissolve", "crossfade"])
    data["beats"][1]["beat_style"] = "hype"
    g = _beats(2)
    assert s3._apply_chunk(g, texts, data, final=True) == ""
    assert g[1].transition == "none"
    data["beats"][0]["transition"] = "spin"
    assert "transition must be" in s3._apply_chunk(_beats(2), texts, data)


# ----------------------------------------------------------------------------- selection


def _b(i: int, style: str = "normal", tmpl: str = "fullscreen_clip", **kw) -> dict:
    return {"id": i, "beat_style": style, "visual_template": tmpl, "emphasis": False, "duration": 4.0, **kw}


def test_auto_rules():
    assert auto_transition(_b(0), None, False) == "none"
    assert auto_transition(_b(1, "hype"), _b(0), False) == "none"
    assert auto_transition(_b(1, "fail"), _b(0), True) == "zoomin"
    assert auto_transition(_b(1, "technical"), _b(0), False) == "dissolve"
    assert auto_transition(_b(1, "deadpan"), _b(0), False) == "crossfade"
    assert auto_transition(_b(1), _b(0), True) == "fadeblack"
    assert auto_transition(_b(1, tmpl="punch_in_emphasis"), _b(0, "technical"), False) == "none"


def test_resolve_joins_limits():
    beats = [_b(0), _b(1, "technical"), _b(2, "technical", transition="slideleft"),
             _b(3, "technical", transition_duration=2.0), _b(4, "technical"),
             _b(5, "technical", transition="none")]
    spans = [90, 90, 90, 90, 12, 90]   # beat 4 is 0.4 s: 0.4*12=4.8 frames max
    j = resolve_joins(beats, spans, 30)
    assert j[0] == Join()                       # never on the first beat
    assert j[1].name == "dissolve" and j[1].frames == round(0.35 * 1.15 * 30)
    assert j[2].name == "slideleft"
    assert j[3] == Join()                       # 2.0s >> 40% of 90 frames
    assert j[4] == Join()                       # shorter neighbour too short
    assert j[5] == Join()                       # explicit none
    assert resolve_joins(beats, spans, 30, enabled=False) == [Join()] * 6
    for i in range(1, 6):                       # < 40% invariant
        assert j[i].frames < 0.4 * min(spans[i - 1], spans[i]) or j[i].frames == 0


def test_resolve_joins_group_cap():
    beats = [_b(i, "technical") for i in range(10)]
    j = resolve_joins(beats, [90] * 10, 30, max_group=4)
    assert [x.frames > 0 for x in j] == [False, True, True, True, False, True, True, True,
                                          False, True]


def test_schema_roundtrip_and_signature():
    b = SceneBeat(id=1, transition="Cross Fade", transition_duration=9)
    assert b.transition is None or b.transition in XFADE
    b = SceneBeat(id=1, transition="fade", transition_duration=9)
    assert b.transition == "crossfade" and b.transition_duration == 1.5
    assert SceneBeat(id=1, transition="bogus").transition is None
    base = {"beats": [{"id": 0, "start_time": 0}]}
    edited = {"beats": [{"id": 0, "start_time": 0, "transition": "dissolve"}]}
    assert authoring_signature(base) == authoring_signature({"beats": [{"id": 0, "start_time": 0,
                                                                        "transition": None}]})
    assert authoring_signature(base) != authoring_signature(edited)


# ----------------------------------------------------------------------------- duration


def _ffprobe_frames(p: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


@pytest.mark.skipif(shutil.which("ffmpeg") is None,
                    reason="ffmpeg missing")
def test_transitions_keep_total_frames(tmp_path):
    fps = 30
    s = RenderSettings(width=320, height=180, fps=fps, codec="libx264", x264_preset="ultrafast")
    ctx = CompileContext(None, s, tmp_path, tmp_path)  # type: ignore[arg-type]
    styles = ["normal", "technical", "technical", "hype", "deadpan", "normal"]
    spans = [60, 75, 90, 45, 80, 60]
    beats = [_b(i, st) for i, st in enumerate(styles)]
    joins = resolve_joins(beats, spans, fps)
    assert sum(1 for j in joins if j.frames) >= 3
    done = []
    for i, (b, sp) in enumerate(zip(beats, spans, strict=True)):
        tail = joins[i + 1].frames if i + 1 < len(beats) else 0
        f = tmp_path / f"b{i}.mp4"
        r = _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                  f"testsrc2=s=320x180:r={fps}", "-frames:v", str(sp + tail), "-r", str(fps),
                  *s.encoder_args(), str(f)])
        assert r.returncode == 0, r.stderr
        b["_join"] = joins[i].name if joins[i].frames else "none"
        done.append((b, sp + tail, f, joins[i].frames, tail))
    parts = _assemble(done, {i: sp for i, sp in enumerate(spans)}, ctx, tmp_path, [])
    cat = tmp_path / "c.txt"
    cat.write_text("".join(f"file '{p}'\n" for p in parts))
    out = tmp_path / "o.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                    str(cat), "-c", "copy", str(out)], check=True)
    assert abs(_ffprobe_frames(out) - sum(spans)) <= 1
