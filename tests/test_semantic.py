"""Semantic matching with a mocked backend (no models, no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import semantic, semantic_index, semantic_planner
from src.common import authoring_signature, write_json
from src.schemas import SceneBeat, ScenePlan, VisualTemplate
from src.semantic import SemanticBackend, normalize
from src.semantic_index import Hit, SemanticIndex, fit_window

TOPICS = ["robot", "smoke", "lathe", "diagram"]


def vec(topic: str) -> np.ndarray:
    v = np.full(len(TOPICS), 0.05, np.float32)
    for i, t in enumerate(TOPICS):
        if t in topic:
            v[i] = 1.0
    return normalize(v)[0]


class FakeBackend(SemanticBackend):
    name = "fake"
    model = "m"

    def embed_frames(self, paths):
        return np.array([vec(Path(p).stem) for p in paths]), [Path(p).stem for p in paths]

    def embed_queries(self, texts):
        return np.array([vec(t) for t in texts])


def make_index() -> SemanticIndex:
    frames_clip = [
        {"t": 1.0, "shot": [0.0, 2.0], "shot_index": 0, "caption": "smoke", "row": 0},
        {"t": 3.0, "shot": [2.0, 4.0], "shot_index": 1, "caption": "robot", "row": 1},
        {"t": 5.0, "shot": [4.0, 6.0], "shot_index": 2, "caption": "robot", "row": 2},
        {"t": 7.0, "shot": [6.0, 8.0], "shot_index": 3, "caption": "lathe", "row": 3},
    ]
    vecs = np.array([vec("smoke"), vec("robot"), vec("robot"), vec("lathe"),
                     vec("lathe"), vec("diagram")])
    data = {"backend": "fake", "tag": "fake-m", "assets": {
        "clip_a": {"type": "clip", "duration": 8.0, "frames": frames_clip},
        "image_lathe": {"type": "image", "duration": None, "frames": [
            {"t": None, "shot": None, "shot_index": 0, "caption": "lathe", "row": 4}]},
        "schematic_d": {"type": "schematic", "duration": None, "frames": [
            {"t": None, "shot": None, "shot_index": 0, "caption": "diagram", "row": 5}]},
    }}
    return SemanticIndex(data, vecs, merge_margin=0.03)


def test_maxframe_and_trim_interval_merges_adjacent_frames() -> None:
    hit = make_index().score_asset("clip_a", vec("robot"))
    assert hit is not None and hit.score > 0.9
    assert (hit.start, hit.end) == (2.0, 6.0)  # both robot shots merged, smoke/lathe not


def test_search_filters_types_and_orders() -> None:
    idx = make_index()
    top = idx.search(vec("lathe"), ("clip", "image"), k=2)
    assert {h.asset_id for h in top} == {"clip_a", "image_lathe"}
    assert all(h.asset_type != "schematic" for h in top)
    assert idx.search(vec("diagram"), ("schematic",), 3)[0].asset_id == "schematic_d"


def test_prior_changes_rank_not_score() -> None:
    idx = make_index()
    hits = idx.search(vec("lathe"), ("clip", "image"), 2, prior={"clip_a": 0.5})
    assert hits[0].asset_id == "clip_a"
    assert hits[0].score < hits[0].rank


def test_fit_window_inside_clip() -> None:
    hit = Hit("c", 0.5, 2.0, 6.0)
    s, e = fit_window(hit, 3.0, 8.0)
    assert 0 <= s < e <= 8.0 and e - s == pytest.approx(3.3)
    assert fit_window(Hit("c", 0.5, 7.5, 8.0), 3.0, 8.0)[1] == pytest.approx(8.0)
    assert fit_window(Hit("i", 0.5), 3.0, None) == (None, None)


def test_fixed_shots_cover_duration() -> None:
    shots = semantic_index.fixed_shots(10.0)
    assert shots[0][0] == 0 and shots[-1][1] == pytest.approx(10.0)


def test_build_index_caches_by_hash(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "robot.png"
    Image.new("RGB", (32, 32), "red").save(p)
    assets = [{"id": "image_robot", "filepath": str(p), "asset_type": "image"}]
    work = tmp_path / "work"
    work.mkdir()
    data = semantic_index.build_index(work, FakeBackend(), assets)
    assert data["assets"]["image_robot"]["frames"][0]["caption"] == "robot"
    calls = []
    b2 = FakeBackend()
    b2.embed_frames = lambda paths: calls.append(paths) or (np.zeros((1, 4)), [""])  # type: ignore
    semantic_index.build_index(work, b2, assets)
    assert not calls  # served from work/semantic cache
    assert SemanticIndex.load(work) is not None


def _ctx() -> SimpleNamespace:
    assets = [{"id": "clip_a", "asset_type": "clip", "duration": 8.0, "tags": [], "description": ""},
              {"id": "image_lathe", "asset_type": "image", "tags": [], "description": ""},
              {"id": "schematic_d", "asset_type": "schematic", "tags": [], "description": ""}]
    return SimpleNamespace(
        assets=assets, assets_by_id={a["id"]: a for a in assets},
        text_between=lambda s, e: {0.0: "the robot arm moves", 4.0: "smoke from the lathe"}[s])


def test_apply_semantic_reassigns_scores_and_trims(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDEOMAKER_OFFLINE", "1")  # no LLM: deterministic path
    idx = make_index()
    write_json(tmp_path / "semantic_index.json", idx.data)
    np.save(tmp_path / "semantic_index.npy", idx.vecs)
    plan = ScenePlan(beats=[
        SceneBeat(id=0, start_time=0, end_time=4, source_id="image_lathe"),
        SceneBeat(id=1, start_time=4, end_time=8, source_id="clip_a",
                  visual_template=VisualTemplate.FULLSCREEN_CLIP)])
    plan = semantic_planner.apply_semantic(plan, _ctx(), tmp_path, {}, FakeBackend())
    b0, b1 = plan.beats
    assert b0.source_id == "clip_a" and b0.match_score and b0.match_score > 0.8
    assert b0.media_start is not None and 0 <= b0.media_start < b0.media_end <= 8.0
    assert b1.source_id == "image_lathe"  # smoke/lathe query; clip_a was just used < 20s ago
    assert b1.visual_query and b0.visual_query == "the robot arm moves"


def test_no_backend_leaves_plan_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_planner, "get_backend", lambda *a, **k: None)
    plan = ScenePlan(beats=[SceneBeat(id=0, start_time=0, end_time=3, source_id="x")])
    out = semantic_planner.apply_semantic(plan, _ctx(), tmp_path, {})
    assert out.beats[0].source_id == "x" and out.beats[0].visual_query is None


def test_get_backend_none_and_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    assert semantic.get_backend({"semantic": {"backend": "none"}}) is None
    monkeypatch.setattr(semantic.SigLIPBackend, "available", staticmethod(lambda: False))
    monkeypatch.setattr(semantic.OllamaBackend, "available", lambda self: False)
    assert semantic.get_backend({"semantic": {"backend": "auto"}}, quiet=True) is None
    assert semantic.get_backend({"semantic": {"backend": "siglip"}}, quiet=True) is None


def test_trim_fields_are_backward_compatible_in_signature() -> None:
    old = {"beats": [{"id": 0, "start_time": 0, "end_time": 1}]}
    with_none = {"beats": [{"id": 0, "start_time": 0, "end_time": 1, "media_start": None}]}
    assert authoring_signature(old) == authoring_signature(with_none)
    trimmed = {"beats": [{"id": 0, "start_time": 0, "end_time": 1, "media_start": 1.0}]}
    assert authoring_signature(old) != authoring_signature(trimmed)


@pytest.mark.skipif(not importlib.util.find_spec("torch")
                    or not importlib.util.find_spec("transformers"),
                    reason="semantic extra not installed")
def test_real_siglip_ranks_a_red_square_for_red(tmp_path: Path) -> None:
    from PIL import Image

    try:
        be = semantic.SigLIPBackend(device="cpu")
        paths = []
        for name, col in (("red", (230, 20, 20)), ("blue", (20, 20, 230))):
            p = tmp_path / f"{name}.png"
            Image.new("RGB", (224, 224), col).save(p)
            paths.append(p)
        v, _ = be.embed_frames(paths)
        q = be.embed_queries(["a plain red image"])
    except OSError:
        pytest.skip("SigLIP weights not cached / offline")
    assert (v @ q[0]).argmax() == 0


def test_stock_verifier_rejects_off_topic_and_ranks_pool(tmp_path: Path) -> None:
    from PIL import Image

    from src.semantic_stock import StockVerifier

    robot, smoke = tmp_path / "robot.png", tmp_path / "smoke.png"
    for p, col in ((robot, "red"), (smoke, "gray")):  # distinct bytes -> distinct cache keys
        Image.new("RGB", (16, 16), col).save(p)
    ver = StockVerifier(tmp_path, FakeBackend(), {"semantic": {"stock_min_score": 0.8}})
    assert ver.score(robot, "robot arm") > 0.8
    assert ver.score(smoke, "robot arm") < 0.8
    pool = [{"id": "a", "local_path": str(smoke)}, {"id": "b", "local_path": str(robot)}]
    best = ver.rank_pool(pool, "a robot", set())
    assert best is not None and best[0]["id"] == "b"
    assert ver.rank_pool(pool, "lathe", set()) is None  # nothing close enough
