"""Decision client + planner against a fake /v1/systemone server."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.decision import (
    DecisionClient,
    DecisionError,
    DecisionUnavailable,
    choice,
    noul,
    parse_answer,
    resolve_model,
    score,
)
from src.decision_planner import (
    DecisionPlanner,
    asset_description,
    candidate_union,
    decision_config,
    planner_mode,
)
from src.schemas import BeatStyle


class Fake:
    """Programmable server: handler(body) -> (status, json). Records requests."""

    def __init__(self, handler):
        self.requests: list[dict] = []
        self.unloaded: list[str] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, obj):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/generate":
                    outer.unloaded.append(body["model"])
                    return self._send(200, {})
                outer.requests.append(body)
                status, obj = handler(body)
                self._send(status, obj)

            def do_GET(self):
                self._send(200, {"models": [{"name": "qwen2.5:14b-instruct"}, {"name": "nimble"}]})

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()


def answer_for(q, pick=None):
    if q["type"] == "choice":
        keys = list(q["criteria"])
        top = pick if pick in keys else keys[0]
        probs = {k: (0.7 if k == top else 0.3 / max(len(keys) - 1, 1)) for k in keys}
        return {"type": "choice", "choice": top, "probabilities": probs, "confidence": 0.5}
    if q["type"] == "noul":
        return {"type": "noul", "noul": 0.9}
    return {"type": "score", "score": 1.2, "legend": {"0": "a"}, "probabilities": {"0": 1.0},
            "confidence": 0.1}


def ok_handler(body):
    return 200, {"model": body["model"], "answers": {k: answer_for(q) for k, q in
                                                      body["questions"].items()},
                 "usage": {"input_tokens": 10, "output_tokens": 1}}


@pytest.fixture
def server():
    made = []

    def make(handler=ok_handler):
        f = Fake(handler)
        made.append(f)
        return f
    yield make
    for f in made:
        f.stop()


def client(f, tmp_path=None, **kw):
    cfg = {"base_url": f.url, "timeout": 5, "retries": 2, **kw}
    return DecisionClient(cfg, tmp_path / "cache" if tmp_path else None, sleep=lambda s: None)


def test_parse_typed_answers():
    a = parse_answer({"type": "choice", "choice": "x", "probabilities": {"x": 0.9, "y": 0.1},
                      "confidence": 0.8}, choice("q", {"x": None, "y": None}))
    assert a.choice == "x" and a.p == 0.9
    assert parse_answer({"type": "noul", "noul": 0.25}).p == 0.25
    s = parse_answer({"type": "score", "score": 0.6, "legend": {"0": "lo"},
                      "probabilities": {"0": 1}, "confidence": 0.2}, score("q", ["lo", "hi"]))
    assert s.score == 0.6 and s.legend == {"0": "lo"}
    with pytest.raises(DecisionError):
        parse_answer({"type": "choice", "choice": "z"}, choice("q", {"x": None}))
    with pytest.raises(DecisionError):
        parse_answer({"type": "noul", "noul": 1.5})
    with pytest.raises(DecisionError):
        parse_answer({"type": "noul", "noul": 0.5}, choice("q", {"x": None}))


def test_resolve_model():
    assert resolve_model({}) == ("nimble", "http://localhost:11434")
    assert resolve_model({"model": "tev1:0.8b"})[0] == "tev1:0.8b"
    assert resolve_model({"model": "https://kev.example/api"}) == ("jev", "https://kev.example/api")


def test_request_shape_and_batching(server):
    f = server()
    c = client(f)
    qs = {f"q{i}": noul(f"question {i}") for i in range(70)}
    out = c.ask({"text": "hi"}, qs)
    assert len(out) == 70 and len(f.requests) == 2
    assert [len(r["questions"]) for r in f.requests] == [64, 6]
    r = f.requests[0]
    assert r["model"] == "nimble" and r["state"] == {"text": "hi"}
    assert r["questions"]["q0"] == {"type": "noul", "instructions": "question 0"}
    assert c.stats["questions"] == 70


def test_cache_by_request_hash(server, tmp_path):
    f = server()
    c = client(f, tmp_path)
    qs = {"a": choice("pick", {"x": "first", "y": None})}
    c.ask("s", qs)
    c.ask("s", qs)
    assert len(f.requests) == 1 and c.stats["cache_hits"] == 1
    c.ask("other state", qs)
    assert len(f.requests) == 2
    c.ask("s", {"a": choice("pick", {"x": "changed", "y": None})})
    assert len(f.requests) == 3
    assert client(f, tmp_path, cache=False).cache_dir is None


def test_retries_then_success(server):
    n = {"i": 0}

    def flaky(body):
        n["i"] += 1
        return (503, {}) if n["i"] < 3 else ok_handler(body)
    f = server(flaky)
    c = client(f)
    assert c.ask("s", {"a": noul("q")})["a"].p == 0.9
    assert c.stats["retries"] == 2


def test_retries_exhausted_and_bad_request(server):
    f = server(lambda b: (500, {}))
    with pytest.raises(DecisionError):
        client(f, retries=1).ask("s", {"a": noul("q")})
    g = server(lambda b: (400, {"error": "bad"}))
    c = client(g)
    with pytest.raises(DecisionError):
        c.ask("s", {"a": noul("q")})
    assert len(g.requests) == 1  # 400 is not retried


def test_unavailable_404_and_connection_refused(server):
    f = server(lambda b: (404, {"error": "not found"}))
    with pytest.raises(DecisionUnavailable):
        client(f).ask("s", {"a": noul("q")})
    c = DecisionClient({"base_url": "http://127.0.0.1:9", "retries": 1, "timeout": 1},
                       sleep=lambda s: None)
    with pytest.raises(DecisionUnavailable):
        c.ask("s", {"a": noul("q")})


def test_malformed_answers(server):
    f = server(lambda b: (200, {"answers": {"a": {"type": "noul"}}}))
    with pytest.raises(DecisionError):
        client(f).ask("s", {"a": noul("q")})
    g = server(lambda b: (200, {"answers": {}}))
    with pytest.raises(DecisionError):
        client(g).ask("s", {"a": noul("q")})


def test_wide_choice_is_narrowed_in_rounds(server):
    def pick_opt30(body):
        qs = body["questions"]
        assert all(len(q["criteria"]) <= 26 for q in qs.values())
        return 200, {"answers": {k: answer_for(q, "opt30") for k, q in qs.items()}}
    f = server(pick_opt30)
    crit = {f"opt{i}": f"option {i}" for i in range(60)}
    a = client(f).ask("s", {"w": choice("which", crit)})["w"]
    assert a.choice == "opt30" and abs(sum(a.probabilities.values()) - 1) < 1e-6
    assert len(f.requests) == 2  # one call with the groups, one final round


def test_unload_and_free_others(server):
    f = server()
    c = client(f)
    c.ask("s", {"a": noul("q")})
    assert c.free_others() == ["qwen2.5:14b-instruct"]
    c.close()
    assert f.unloaded == ["qwen2.5:14b-instruct", "nimble"]
    c.close()  # idempotent
    assert f.unloaded.count("nimble") == 1


def beats(n=3):
    from src.schemas import SceneBeat
    return [SceneBeat(id=i, start_time=i * 4.0, end_time=i * 4.0 + 4.0,
                      visual_template="fullscreen_clip", source_id="clip_a") for i in range(n)]


def test_beat_decisions_threshold_and_reason(server, tmp_path):
    def h(body):
        ans = {}
        for k, q in body["questions"].items():
            if k == "style":
                ans[k] = answer_for(q, "hype")  # p = 0.7
            elif k == "transition":
                ans[k] = answer_for(q, "fadeblack")
            elif k == "template":
                a = answer_for(q, "punch_in")
                a["probabilities"] = {kk: (0.4 if kk == "punch_in" else 0.2) for kk in q["criteria"]}
                ans[k] = a  # below threshold: template kept
            else:
                ans[k] = answer_for(q)
        return 200, {"answers": ans}
    f = server(h)
    dp = DecisionPlanner({"decision": {"base_url": f.url, "retries": 0}}, tmp_path)
    bs = beats()
    texts = {b.id: f"line {b.id}" for b in bs}
    counts = dp.apply_beat_decisions(bs, dp.decide_beats(bs, texts, 100.0))
    assert bs[1].beat_style.value == "hype"
    assert bs[1].visual_template.value == "fullscreen_clip"  # p 0.4 < 0.55
    assert bs[0].transition in (None, "none")                # first beat has no incoming join
    assert bs[1].transition in (None, "none")                        # hype is cut, never blended
    assert "decision:nimble style=hype p=0.70" in bs[1].beat_reason
    assert counts["style"] == 3 and counts["template"] == 0
    st = f.requests[0]["state"]
    assert st["narration"] == "line 0" and st["next_line"] == "line 1"
    assert "opening" in st["position"] and f.requests[1]["state"]["previous_line"] == "line 0"
    assert set(f.requests[0]["questions"]) == {"style", "template", "transition", "cutout_named",
                                               "callout_worthy", "topic_change"}
    assert "slide" not in json.dumps(f.requests[0]["questions"]["transition"]).lower()
    assert "wipe" not in json.dumps(f.requests[0]["questions"]["transition"]).lower()


def test_low_confidence_keeps_current(server, tmp_path):
    def h(body):
        out = {}
        for k, q in body["questions"].items():
            a = answer_for(q)
            if q["type"] == "choice":
                n = len(q["criteria"])
                a["probabilities"] = {kk: 1 / n for kk in q["criteria"]}
            out[k] = a
        return 200, {"answers": out}
    dp = DecisionPlanner({"decision": {"base_url": server(h).url}}, tmp_path)
    bs = beats()
    for b in bs:
        b.beat_style = BeatStyle.TECHNICAL
    n = dp.apply_beat_decisions(bs, dp.decide_beats(bs, {b.id: "x" for b in bs}, 12.0))
    assert n == {"style": 0, "template": 0, "transition": 0, "callout_off": 0}
    assert all(b.beat_style.value == "technical" and not b.beat_reason for b in bs)


def test_server_down_keeps_plan(tmp_path):
    dp = DecisionPlanner({"decision": {"base_url": "http://127.0.0.1:9", "retries": 0,
                                       "timeout": 1}}, tmp_path)
    bs = beats()
    out = dp.decide_beats(bs, {b.id: "x" for b in bs}, 12.0)
    assert out == {} and dp.enabled is False
    assert dp.decide_assets([{"id": 0, "state": "s", "cands": [("a", ""), ("b", "")]}], {}) == {}


def test_asset_probabilities_and_bonus(server, tmp_path):
    def h(body):
        q = body["questions"]["asset"]
        assert len(q["criteria"]) == 3
        return 200, {"answers": {"asset": answer_for(q, "image_b")}}
    f = server(h)
    dp = DecisionPlanner({"decision": {"base_url": f.url, "weight": 0.2}}, tmp_path)
    assets = {"clip_a": {"asset_type": "clip", "description": "clip | a | rocket launch"},
              "image_b": {"asset_type": "image", "description": "image | b | mirror ground"},
              "schematic_c": {"asset_type": "schematic", "description": "schematic | c | rays"}}
    p = dp.decide_assets([{"id": 7, "state": {"narration": "x"},
                           "cands": [("clip_a", ""), ("image_b", "a mirror"), ("schematic_c", "")]}],
                         assets)
    assert abs(sum(p[7].values()) - 1) < 1e-6 and max(p[7], key=p[7].get) == "image_b"
    assert dp.asset_bonus(7, "image_b") == pytest.approx(0.2)
    assert 0 < dp.asset_bonus(7, "clip_a") < 0.2
    assert dp.asset_bonus(7, "nope") is None and dp.asset_bonus(8, "clip_a") is None
    desc = f.requests[0]["questions"]["asset"]["criteria"]
    assert desc["image_b"].startswith("photo: mirror ground") and "caption: a mirror" in desc["image_b"]
    assert desc["clip_a"].startswith("video clip:") and desc["schematic_c"].startswith("diagram:")


def test_asset_description_prefers_sidecar(tmp_path):
    img = tmp_path / "a.jpg"
    (tmp_path / "a.jpg.txt").write_text("Hubble in the payload bay, 1993.\n\nkind: screen\n")
    d = asset_description({"asset_type": "image", "filepath": str(img), "description": "image | zzz"})
    assert d == "photo: Hubble in the payload bay, 1993."
    (tmp_path / "b.jpg.txt").write_text("description: Two astronauts on a spacewalk.\n")
    assert "Two astronauts" in asset_description({"asset_type": "image",
                                                  "filepath": str(tmp_path / "b.jpg")})


def test_candidate_union_mixes_embedding_and_prior():
    class Idx:
        def search(self, v, types, k, exclude, prior):
            hs = [SimpleNamespace(asset_id=f"a{i}", score=1 - i / 10, caption=f"cap{i}")
                  for i in range(10)]
            return hs[:k]
    got = candidate_union(Idx(), [np.zeros(2)], {"zz": 0.5, "yy": 0.2, "a9": 0.0}, ("clip",), 5)
    ids = [a for a, _ in got]
    assert ids[:3] == ["a0", "a1", "a2"] and "zz" in ids and "yy" in ids and len(ids) == 5


def test_planner_mode_and_config(monkeypatch):
    assert planner_mode({}) == "llm"
    assert planner_mode({"llm": {"planner": "hybrid"}}) == "hybrid"
    assert planner_mode({"llm": {"planner": "bogus"}}) == "llm"
    monkeypatch.setenv("VIDEOMAKER_PLANNER", "decision")
    assert planner_mode({"llm": {"planner": "llm"}}) == "decision"
    c = decision_config({"decision": {"thresholds": {"style": 0.9}}})
    assert c["thresholds"]["style"] == 0.9 and c["thresholds"]["template"] == 0.55


def test_stage3_decision_mode_end_to_end(server, tmp_path, monkeypatch):
    """`llm.planner: decision`: no chat LLM, beat-level overrides recorded in beat_reason."""
    import yaml

    from src.common import ENV_CONFIG, ENV_OFFLINE, episode_dir, read_json
    from src.stage3_plan_scenes import run
    from tests.test_planner import ASSETS, make_transcript

    def h(body):
        out = {}
        for k, q in body["questions"].items():
            out[k] = answer_for(q, "deadpan" if k == "style" else None)
        return 200, {"answers": out}
    f = server(h)
    cfg = yaml.safe_load(Path(os.environ[ENV_CONFIG]).read_text())
    cfg["llm"]["planner"] = "decision"
    cfg["decision"] = {"base_url": f.url, "retries": 0}
    cfg["semantic"] = {"backend": "none"}
    p = tmp_path / "dec.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv(ENV_CONFIG, str(p))
    monkeypatch.setenv(ENV_OFFLINE, "0")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")  # a chat LLM must not be needed
    ep = episode_dir("dec-ep")
    work = ep / "work"
    work.mkdir(parents=True)
    tr = make_transcript()
    (work / "transcript.json").write_text(json.dumps(tr))
    (work / "local_assets.json").write_text(json.dumps({"assets": ASSETS}))
    plan = run("dec-ep", work)
    assert plan["beats"]
    assert any("decision:nimble" in (b.get("beat_reason") or "") for b in plan["beats"])
    assert any(b["beat_style"] == "deadpan" for b in plan["beats"])
    assert f.unloaded == ["qwen2.5:14b-instruct", "nimble"]  # others freed first, itself last
    assert read_json(work / "decision_stats.json")["questions"] > 0
