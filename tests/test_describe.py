import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

from src import describe as d


def mk(tmp_path, name, text=None, kind="image"):
    p = tmp_path / f"{name}.jpg"
    Image.new("RGB", (64, 48), (200, 30, 30)).save(p)
    if text is not None:
        (tmp_path / f"{name}.jpg.txt").write_text(text)
    return {"id": f"{kind}_{name}", "filepath": str(p), "asset_type": kind}


def test_classify_missing_weak_ok(tmp_path):
    assert d.classify(mk(tmp_path, "a")).state == "missing"
    assert d.classify(mk(tmp_path, "mirror_blank", "mirror blank")).state == "weak"
    assert d.classify(mk(tmp_path, "b", "Two astronauts install a new camera inside the "
                         "Hubble payload bay during the 1993 repair mission.")).state == "ok"
    assert d.classify(mk(tmp_path, "c", "kind: screen\n")).state == "missing"


def test_write_keeps_user_text_and_hints(tmp_path):
    a = mk(tmp_path, "u", "Two astronauts install a camera in the payload bay of Hubble.\n"
           "audio: ambient\n")
    side = tmp_path / "u.jpg.txt"
    assert d.write_description(side, "New text.") == "skipped (user text)"
    assert "New text" not in side.read_text()
    assert d.write_description(side, "New text.", force=True) == "replaced"
    assert (tmp_path / "u.jpg.txt.bak").read_text().startswith("Two astronauts")
    assert side.read_text() == "audio: ambient\ndescription: New text.\n"
    assert d.classify(a).generated == "New text."
    b = mk(tmp_path, "n")
    assert d.write_description(tmp_path / "n.jpg.txt", "A red square.") == "written"
    assert d.write_description(tmp_path / "n.jpg.txt", "A blue square.") == "replaced"  # own line
    assert (tmp_path / "n.jpg.txt").read_text() == "description: A blue square.\n"
    assert d.classify(b).state == "weak"  # 3 words


def test_run_with_fake_vlm(tmp_path):
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            out = {"message": {"content": "The image shows a red square on a plain background."}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    work = tmp_path / "work"
    work.mkdir()
    keep = mk(tmp_path, "keep", "A long enough user sentence about a red mirror in a lab.")
    miss = mk(tmp_path, "miss")
    (work / "local_assets.json").write_text(json.dumps({"assets": [keep, miss]}))
    base = f"http://127.0.0.1:{srv.server_port}"
    dry = d.run(work, "fake:1b", write=False, base=base)
    assert [r["id"] for r in dry] == ["image_miss"] and not (tmp_path / "miss.jpg.txt").exists()
    res = d.run(work, "fake:1b", write=True, base=base)
    assert res[0]["result"] == "written"
    assert (tmp_path / "miss.jpg.txt").read_text() == "description: A red square on a plain background.\n"
    assert any(b.get("keep_alive") == 0 for b in seen)  # model unloaded at the end
    assert d.run(work, "fake:1b", write=True, base=base, list_only=True) == []
    srv.shutdown()
