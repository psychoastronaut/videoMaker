"""Mask post-processing and quality gate of Stage 5 (synthetic masks, no rembg needed)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src import stage5_cutouts as s5


def _blob_alpha() -> np.ndarray:
    a = np.zeros((200, 300), np.float32)
    a[60:160, 90:210] = 1.0
    return a


def test_refine_kills_haze_fragments_and_fills_holes() -> None:
    a = _blob_alpha()
    a[:50, :] = 0.12                      # low-alpha haze above the subject
    a[10:14, 10:14] = 1.0                 # floating fragment
    a[100:104, 130:134] = 0.0             # small hole
    out = s5.refine_alpha(a)
    assert out[:45].max() == 0            # haze and fragment gone
    assert out[102, 132] > 0.95           # hole filled
    assert out[110, 150] > 0.99           # interior solid
    assert 0 < out[60, 150] < 1           # feathered edge


def test_refine_keeps_two_large_components() -> None:
    a = _blob_alpha()
    a[70:150, 20:70] = 1.0
    out = s5.refine_alpha(a)
    assert out[100, 40] > 0.99 and out[100, 150] > 0.99


def test_defringe_replaces_edge_colour() -> None:
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[:] = (255, 255, 255)              # white old background bleeding in
    rgb[5:15, 5:15] = (200, 0, 0)
    alpha = np.zeros((20, 20), np.float32)
    alpha[5:15, 5:15] = 1.0
    alpha[4, 5:15] = 0.5
    rgb[4, 5:15] = (255, 255, 255)
    out = s5.defringe(rgb, alpha)
    assert tuple(out[4, 8]) == (200, 0, 0)


def test_gate_accepts_clean_and_rejects_bad_masks() -> None:
    cfg: dict = {}
    clean = np.zeros((200, 300, 4), np.uint8)
    clean[60:160, 90:210] = (10, 20, 30, 255)
    assert s5.clean_cutout(Image.fromarray(clean), cfg)[2] == ""

    empty = np.zeros((200, 300, 4), np.uint8)
    assert "too small" in s5.clean_cutout(Image.fromarray(empty), cfg)[2]

    full = np.full((200, 300, 4), 255, np.uint8)
    assert "nothing to remove" in s5.clean_cutout(Image.fromarray(full), cfg)[2]

    hazy = clean.copy()
    hazy[..., 3] = np.where(hazy[..., 3] > 0, 255, 0)
    hazy[0:50, :, 3] = 90                 # big uncertain area
    hazy[50:200, 0:300, 3] = np.maximum(hazy[50:200, 0:300, 3], 120)
    assert s5.clean_cutout(Image.fromarray(hazy), cfg)[2] != ""

    frag = np.zeros((300, 300, 4), np.uint8)
    for i in range(6):
        frag[10 + i * 45:40 + i * 45, 100:130, 3] = 255
    assert "fragmented" in s5.clean_cutout(Image.fromarray(frag), cfg)[2]


def test_dark_and_uniform_backgrounds_detected() -> None:
    dark = Image.new("RGB", (400, 300), (3, 3, 6))
    dark.paste((250, 250, 250), (150, 100, 250, 200))
    assert "dark" in s5.dark_uniform_background(dark)
    flat = Image.new("RGB", (400, 300), (200, 200, 205))
    assert "uniform" in s5.dark_uniform_background(flat)
    noisy = Image.fromarray(np.random.default_rng(0).integers(60, 220, (300, 400, 3),
                                                              dtype=np.uint8))
    assert s5.dark_uniform_background(noisy) == ""


def test_card_is_rounded_rgba() -> None:
    card = s5.make_card_image(Image.new("RGB", (640, 360), (90, 90, 90)))
    assert card.mode == "RGBA" and card.getpixel((0, 0))[3] == 0
    assert card.getpixel((card.width // 2, card.height // 2))[3] == 255
