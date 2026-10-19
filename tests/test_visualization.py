"""Tests for the overlay presentation.

Regression test for the reported problem that the animation's background image never changed:
by default the field must be drawn on the *current* frame's image at the material positions
``X + u``, so the specimen visibly deforms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dic.config import RunConfig
from dic.pipeline import run_pipeline
from dic.synth import DotPattern, render, stretch
from dic.visualization import (
    _frame_image,
    _reference_image,
    _view,
    render_all_figures,
)


def _sequence(tmp_path: Path, n: int = 4, size: int = 180) -> Path:
    pat = DotPattern.generate((size, size), n_dots=300, seed=0)
    d = tmp_path / "seq"
    d.mkdir(exist_ok=True)
    for i in range(n):
        D = stretch(0.06 * i, -0.02 * i, np.array([size / 2, size / 2]))
        img = render(pat, D, noise_sigma=2.0, seed=100 + i)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(d / f"image{i + 1:04d}.png")
    return d


def _cfg() -> RunConfig:
    cfg = RunConfig()
    cfg.correlation.subset_size = 21
    cfg.correlation.step = 10
    cfg.output.figures = False
    cfg.output.extras = False
    cfg.uncertainty.noise_floor = False
    return cfg


def test_default_overlay_is_the_deformed_frame(tmp_path):
    d = _sequence(tmp_path)
    cfg = _cfg()
    res = run_pipeline(d, cfg)
    last = res.frames[-1]
    base, X, Y, deformed = _view(res, last, cfg.output)
    assert deformed
    # the base image must be the *current* frame, not the reference
    assert base is _frame_image(res, last.index)
    assert base is not _reference_image(res)
    # the field must be plotted at the material positions X + u
    Xr, Yr = np.meshgrid(np.unique(res.xs), np.unique(res.ys))
    assert np.nanmax(np.abs(last.ux)) > 1.0  # the field is non-zero
    assert np.nanmax(np.abs(X - Xr)) > 1.0
    assert np.nanmax(np.abs(Y - Yr)) > 0.1


def test_reference_overlay_uses_the_undeformed_image(tmp_path):
    d = _sequence(tmp_path)
    cfg = _cfg()
    cfg.output.overlay_frame = "reference"
    res = run_pipeline(d, cfg)
    last = res.frames[-1]
    base, X, Y, deformed = _view(res, last, cfg.output)
    assert not deformed
    assert base is _reference_image(res)
    Xr, Yr = np.meshgrid(np.unique(res.xs), np.unique(res.ys))
    assert np.allclose(X, Xr) and np.allclose(Y, Yr)


@pytest.mark.slow
def test_animation_background_changes_with_the_specimen(tmp_path):
    d = _sequence(tmp_path)
    cfg = _cfg()
    cfg.output.extras = True
    cfg.output.animation_format = "gif"
    res = run_pipeline(d, cfg)
    out = tmp_path / "out"
    written = render_all_figures(res, out)
    gifs = [p for p in written["extras"] if str(p).endswith(".gif")]
    assert gifs, "no animation was produced"
    im = Image.open(gifs[0])
    frames = []
    for i in (0, im.n_frames - 1):
        im.seek(i)
        frames.append(np.asarray(im.convert("L"), dtype=float))
    diff = float(np.mean(np.abs(frames[0] - frames[1])))
    # the background must change noticeably, not just the contours
    assert diff > 5.0
