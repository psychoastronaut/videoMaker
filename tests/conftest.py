"""Shared fixtures: hermetic episodes dir, offline mode, CPU-only quick encode config."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from src.common import DEFAULT_CONFIG_PATH, ENV_CONFIG, ENV_EPISODES, ENV_OFFLINE


@pytest.fixture(autouse=True)
def hermetic_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Episodes live in tmp_path, network/LLM off, libx264 ultrafast for speed."""
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    monkeypatch.setenv(ENV_EPISODES, str(episodes))
    monkeypatch.setenv(ENV_OFFLINE, "1")
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    cfg["pipeline"]["video_codec"] = "libx264"
    cfg["pipeline"]["x264_preset"] = "ultrafast"
    cfg["pipeline"]["crf"] = 28
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv(ENV_CONFIG, str(cfg_path))
    return episodes


def write_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    t = np.arange(int(rate * seconds)) / rate
    data = (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())


def make_subject_image(path: Path) -> None:
    img = Image.new("RGB", (640, 360), (230, 230, 235))
    ImageDraw.Draw(img).rectangle([200, 100, 440, 260], fill=(30, 120, 200))
    img.save(path)


def make_schematic(path: Path) -> None:
    img = Image.new("RGB", (640, 360), (10, 10, 26))
    ImageDraw.Draw(img).rectangle([100, 100, 540, 260], outline=(74, 144, 217), width=6)
    img.save(path)
