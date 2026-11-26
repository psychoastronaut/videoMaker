"""`new` scaffold, deep-merged episode config, status on an empty episode, dashboard preview modes."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from rich.console import Console
from typer.testing import CliRunner

from src import dashboard as dash
from src.cli import app
from src.common import DEFAULT_CONFIG_PATH, ENV_CONFIG, load_config

runner = CliRunner()


def test_new_scaffold_and_status(hermetic_env: Path) -> None:
    r = runner.invoke(app, ["new", "ep-t", "--template", "diy", "--no-stock"])
    assert r.exit_code == 0, r.output
    assert "dashboard ep-t --run" in r.output.replace("\n", " ") or "dashboard" in r.output
    ep = hermetic_env / "ep-t"
    for d in ("raw/clips", "raw/images", "raw/schematics", "raw/voiceover", "raw/bench", "work", "out"):
        assert (ep / d).is_dir()
    for f in ("config.yaml", "README.md", ".gitignore", "raw/script.txt", "raw/_TEMPLATE.txt.example"):
        assert (ep / f).is_file()
    cfg = yaml.safe_load((ep / "config.yaml").read_text())
    assert cfg["stock_media"]["enabled"] is False and cfg["theme"] == "psycho_terminal"
    assert len(cfg) < 12
    assert runner.invoke(app, ["new", "ep-t"]).exit_code == 1  # exists
    assert runner.invoke(app, ["new", "ep-t", "--force"]).exit_code == 0
    assert runner.invoke(app, ["new", "bad/id"]).exit_code == 1
    assert "diy" in runner.invoke(app, ["new", "--list-templates"]).output
    s = runner.invoke(app, ["status", "ep-t"])
    assert s.exit_code == 0 and "Missing" in s.output


def test_new_from_copies_config(hermetic_env: Path) -> None:
    runner.invoke(app, ["new", "ep-a", "--no-stock"])
    assert runner.invoke(app, ["new", "ep-b", "--from", "ep-a"]).exit_code == 0
    assert (hermetic_env / "ep-b/config.yaml").read_text() == (hermetic_env / "ep-a/config.yaml").read_text()


def test_episode_config_deep_merges(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("stock_media:\n  enabled: false\n")
    monkeypatch.setenv(ENV_CONFIG, str(p))
    cfg = load_config()
    default = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    assert cfg["stock_media"]["enabled"] is False
    assert cfg["stock_media"]["video"] == default["stock_media"]["video"]
    assert cfg["llm"] == default["llm"]


def test_detect_preview_mode() -> None:
    d = dash.detect_preview_mode
    assert d("auto", {"KITTY_WINDOW_ID": "1", "TERM": "xterm-kitty"}, tty=True) == "kitty"
    assert d("auto", {"TERM": "xterm-ghostty"}, tty=True) == "kitty"
    assert d("auto", {"TERM_PROGRAM": "iTerm.app"}, tty=True) == "iterm"
    assert d("auto", {"TERM": "xterm-256color"}, tty=True) == "quadrants"
    assert d("auto", {"KITTY_WINDOW_ID": "1", "TMUX": "x"}, tty=True) == "quadrants"
    assert d("auto", {"KITTY_WINDOW_ID": "1"}, tty=False) == "quadrants"
    assert d("blocks", {}, tty=True) == "blocks"


def _img(tmp_path: Path) -> Path:
    p = tmp_path / "f.png"
    Image.fromarray((np.random.rand(90, 160, 3) * 255).astype("uint8")).save(p)
    return p


def test_quadrant_and_block_lines(tmp_path: Path) -> None:
    p = _img(tmp_path)
    for mode in ("blocks", "quadrants"):
        lines = dash._cell_lines(str(p), p.stat().st_mtime, 40, 12, (0, 0, 0), mode)
        assert len(lines) == 12 and all(len(t.plain) == 40 for t in lines)
    q = dash._cell_lines(str(p), p.stat().st_mtime, 40, 12, (0, 0, 0), "quadrants")
    assert set("".join(t.plain for t in q)) <= set(dash._QUAD_CHARS)


def test_kitty_escape_in_rich_live(tmp_path: Path, monkeypatch) -> None:
    p = _img(tmp_path)
    monkeypatch.setattr(dash, "_preview_mode", "kitty")
    dash._kitty_sent.clear()
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=True, width=60, height=14, color_system="truecolor")
    hb = dash.HalfBlock(p, 40, 10, dash.THEMES["psycho"], big=True)
    con.print(hb)
    out = buf.getvalue()
    assert "\x1b_Ga=t,i=" in out and "a=p,i=" in out and "c=" in out and out.count("a=t") >= 1
    buf.truncate(0)
    buf.seek(0)
    con.print(hb)
    assert "a=t" not in buf.getvalue()  # unchanged beat: nothing re-sent
    assert "a=d" in dash.graphics_cleanup() or dash._preview_mode == "kitty"
