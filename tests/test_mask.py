"""Unit tests for ROI detection and masking."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dic.config import MaskConfig
from dic.mask import auto_mask, build_mask, local_std, otsu_threshold, resolve_user_mask
from dic.synth import DotPattern, render


def test_local_std_of_constant_is_zero():
    img = np.full((40, 40), 77.0)
    assert np.allclose(local_std(img, 5), 0.0)


def test_local_std_of_noise_is_approximately_sigma():
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 3.0, (120, 120))
    std = local_std(img, 15)
    assert 2.0 < np.median(std) < 4.5


def test_otsu_separates_two_modes():
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(2.0, 0.5, 5000), rng.normal(30.0, 3.0, 5000)])
    t = otsu_threshold(v)
    assert 3.0 < t < 25.0


def test_auto_mask_finds_speckled_region_and_ignores_background():
    pat = DotPattern.generate((300, 300), n_dots=700, seed=5, region=(80, 60, 240, 250))
    img = render(pat, None, noise_sigma=2.0, seed=1)
    cfg = MaskConfig(window=21)
    mask = auto_mask(img, cfg, subset_size=21)
    # the detected area must be close to the speckled rectangle (80..240) x (60..250)
    assert 0.15 < mask.mean() < 0.55
    # a point deep inside the background must be excluded
    assert not mask[10, 10]
    assert not mask[290, 290]
    # a point inside the speckled rectangle must be included
    assert mask[150, 150]


def test_auto_mask_keeps_a_fully_speckled_frame():
    pat = DotPattern.generate((200, 200), n_dots=700, seed=6)
    img = render(pat, None, noise_sigma=2.0, seed=1)
    mask = auto_mask(img, MaskConfig(), subset_size=21)
    assert mask.mean() > 0.9


def test_build_mask_user_mode(tmp_path):
    m = np.zeros((50, 50), dtype=np.uint8)
    m[10:40, 10:40] = 255
    p = tmp_path / "mask0001.png"
    Image.fromarray(m).save(p)
    ref = np.zeros((50, 50))
    mask, source = build_mask(ref, MaskConfig(mode="user"), 11, user_mask_path=p)
    assert "user mask" in source
    assert mask[25, 25]
    assert not mask[0, 0]


def test_build_mask_directory_picks_the_reference_frame(tmp_path):
    for i in range(3):
        m = np.full((40, 40), 255, np.uint8)
        m[: i * 5, :] = 0
        Image.fromarray(m).save(tmp_path / f"mask{i + 1:04d}.png")
    ref = np.zeros((40, 40))
    mask, _ = build_mask(ref, MaskConfig(mode="user"), 5, user_mask_path=tmp_path, user_mask_index=0)
    assert mask.all()  # the first mask is fully valid


def test_build_mask_none_mode():
    ref = np.zeros((30, 30))
    mask, source = build_mask(ref, MaskConfig(mode="none"), 9)
    assert mask.all()
    assert "none" in source


def test_resolve_user_mask_shape_mismatch(tmp_path):
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(tmp_path / "m.png")
    with pytest.raises(ValueError):
        resolve_user_mask(tmp_path / "m.png", (20, 20))
