"""Unit tests for image I/O: natural sorting, grayscale conversion, validation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dic.io import (
    ImageLoadError,
    discover_images,
    load_mask,
    load_sequence,
    natural_key,
    to_grayscale,
)


def test_natural_sort_orders_numbers_numerically():
    names = ["image10.png", "image2.png", "image1.png", "image20.png"]
    assert sorted(names, key=natural_key) == [
        "image1.png",
        "image2.png",
        "image10.png",
        "image20.png",
    ]


def test_natural_sort_handles_arbitrary_prefixes_and_suffixes():
    names = ["a_12_b.tif", "a_2_b.tif", "z_1_b.tif"]
    assert sorted(names, key=natural_key) == ["a_2_b.tif", "a_12_b.tif", "z_1_b.tif"]


def test_grayscale_rgb_uses_rec601_weights():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = (255, 0, 0)
    img[0, 1] = (0, 255, 0)
    img[1, 0] = (0, 0, 255)
    img[1, 1] = (255, 255, 255)
    gray = to_grayscale(img)
    assert gray[0, 0] == pytest.approx(0.299 * 255)
    assert gray[0, 1] == pytest.approx(0.587 * 255)
    assert gray[1, 0] == pytest.approx(0.114 * 255)
    assert gray[1, 1] == pytest.approx(255.0)


def test_grayscale_rgba_drops_alpha():
    img = np.zeros((2, 2, 4), dtype=np.uint8)
    img[..., 3] = 7
    img[0, 0, 0] = 10
    assert to_grayscale(img)[0, 0] == pytest.approx(0.299 * 10)


def test_discover_requires_two_images(tmp_path):
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / "image0001.png")
    with pytest.raises(ImageLoadError):
        discover_images(tmp_path)


def test_discover_missing_directory(tmp_path):
    with pytest.raises(ImageLoadError):
        discover_images(tmp_path / "nope")


def test_load_sequence_rejects_inconsistent_sizes(tmp_path):
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / "image0001.png")
    Image.fromarray(np.zeros((9, 9), np.uint8)).save(tmp_path / "image0002.png")
    with pytest.raises(ImageLoadError):
        load_sequence(tmp_path)


def test_load_sequence_reports_hashes_and_shape(tmp_path):
    for i in range(3):
        Image.fromarray(np.full((6, 7), i * 10, np.uint8)).save(tmp_path / f"image{i + 1:04d}.png")
    seq = load_sequence(tmp_path)
    assert len(seq) == 3
    assert seq.shape == (6, 7)
    assert all(len(h) == 64 for h in seq.hashes)
    assert seq.images[0][0, 0] == pytest.approx(0.0)


def test_load_mask_convention(tmp_path):
    m = np.zeros((5, 5), np.uint8)
    m[1:4, 1:4] = 255
    Image.fromarray(m).save(tmp_path / "mask0001.png")
    out = load_mask(tmp_path / "mask0001.png")
    assert out.dtype == bool
    assert out.sum() == 9


def test_discover_warns_on_numbering_gaps(tmp_path, caplog):
    import logging

    for n in (1, 2, 5):
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / f"image{n:04d}.png")
    with caplog.at_level(logging.WARNING):
        files = discover_images(tmp_path)
    assert len(files) == 3
    assert any("not contiguous" in r.message for r in caplog.records)


def test_discover_does_not_warn_on_contiguous_numbering(tmp_path, caplog):
    import logging

    for n in (1, 2, 3):
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / f"image{n:04d}.png")
    with caplog.at_level(logging.WARNING):
        discover_images(tmp_path)
    assert not any("not contiguous" in r.message for r in caplog.records)
