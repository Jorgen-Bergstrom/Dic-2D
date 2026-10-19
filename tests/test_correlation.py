"""Unit and integration tests for the correlation core."""

from __future__ import annotations

import numpy as np
import pytest

from dic.config import CorrelationConfig
from dic.correlation import SubsetCorrelator, estimate_global_affine, make_grid
from dic.synth import AffineDeformation, render, rotation, shear, stretch

CENTER = np.array([110.0, 110.0])


def _cfg(**kw) -> CorrelationConfig:
    base = dict(subset_size=31, step=10, prefilter_sigma=0.7)
    base.update(kw)
    return CorrelationConfig(**base)


def test_make_grid_respects_mask():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    xs, ys = make_grid((100, 100), mask, subset_size=11, step=5)
    assert xs.size > 0
    # every subset footprint must be inside the mask
    r = 5
    assert xs.min() >= 20 + r
    assert xs.max() <= 79 - r
    assert ys.min() >= 20 + r
    assert ys.max() <= 79 - r


def test_make_grid_none_mask_uses_whole_image():
    xs, ys = make_grid((100, 100), None, subset_size=11, step=10)
    assert xs.min() == 5
    assert xs.max() <= 94
    assert xs.max() >= 85


@pytest.mark.parametrize("tx,ty", [(0.0, 0.0), (0.37, -0.61), (3.25, 2.75), (-4.4, 1.1)])
def test_recover_subpixel_translation(speckle_pattern, tx, ty):
    ref = render(speckle_pattern, None, noise_sigma=0.0)
    cur = render(speckle_pattern, AffineDeformation(np.eye(2), np.array([tx, ty])), noise_sigma=0.0)
    cfg = _cfg()
    xs, ys = make_grid(ref.shape, None, cfg.subset_size, cfg.step)
    corr = SubsetCorrelator(ref, xs, ys, cfg)
    p0, _ = corr.initial_guess(cur, None, None)
    res = corr.correlate_robust(cur, p0)
    m = res.valid
    assert m.mean() > 0.9
    assert abs(np.mean(res.u[m]) - tx) < 0.02
    assert abs(np.mean(res.v[m]) - ty) < 0.02
    assert np.sqrt(np.mean((res.u[m] - tx) ** 2)) < 0.03


def test_recover_affine_stretch(speckle_pattern):
    ref = render(speckle_pattern, None, noise_sigma=0.0)
    D = stretch(0.15, -0.04, CENTER)
    cur = render(speckle_pattern, D, noise_sigma=0.0)
    cfg = _cfg()
    xs, ys = make_grid(ref.shape, None, cfg.subset_size, cfg.step)
    corr = SubsetCorrelator(ref, xs, ys, cfg)
    p0, _ = corr.initial_guess(cur, None, None)
    res = corr.correlate_robust(cur, p0)
    m = res.valid
    assert m.mean() > 0.75
    assert abs(np.nanmean(res.ux[m]) - 0.15) < 5e-3
    assert abs(np.nanmean(res.vy[m]) + 0.04) < 5e-3


def test_recover_rigid_rotation_single_step(speckle_pattern):
    ref = render(speckle_pattern, None, noise_sigma=0.0)
    D = rotation(np.radians(25.0), CENTER)
    cur = render(speckle_pattern, D, noise_sigma=0.0)
    cfg = _cfg()
    xs, ys = make_grid(ref.shape, None, cfg.subset_size, cfg.step)
    corr = SubsetCorrelator(ref, xs, ys, cfg)
    p0, ga = corr.initial_guess(cur, None, None)
    assert ga is not None  # coarse search must have been used
    res = corr.correlate_robust(cur, p0)
    m = res.valid
    assert m.mean() > 0.75
    # the affine parameters must reproduce the rotation matrix F = R(25 deg)
    ang = np.radians(25.0)
    assert np.nanmean(res.ux[m]) == pytest.approx(np.cos(ang) - 1.0, abs=5e-3)
    assert np.nanmean(res.vy[m]) == pytest.approx(np.cos(ang) - 1.0, abs=5e-3)
    assert np.nanmean(res.uy[m]) == pytest.approx(-np.sin(ang), abs=5e-3)
    assert np.nanmean(res.vx[m]) == pytest.approx(np.sin(ang), abs=5e-3)


def test_estimate_global_affine_recovers_rotation():
    from dic.synth import DotPattern

    pat = DotPattern.generate((200, 200), n_dots=300, seed=2, softness=1.0)
    ref = render(pat, None, noise_sigma=0.0)
    c = np.array([100.0, 100.0])
    ang = np.radians(20.0)
    cur = render(pat, rotation(ang, c), noise_sigma=0.0)
    A, t = estimate_global_affine(ref, cur, _cfg())
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    assert np.max(np.abs(A - R)) < 0.02
    assert np.allclose(t, c - R @ c, atol=2.0)


def test_out_of_frame_points_are_nan_not_fabricated(speckle_pattern):
    ref = render(speckle_pattern, None, noise_sigma=0.0)
    # push everything far to the right so most subsets leave the image
    cur = render(speckle_pattern, AffineDeformation(np.eye(2), np.array([150.0, 0.0])), noise_sigma=0.0)
    cfg = _cfg()
    xs, ys = make_grid(ref.shape, None, cfg.subset_size, cfg.step)
    corr = SubsetCorrelator(ref, xs, ys, cfg)
    res = corr.correlate_robust(cur, np.zeros((xs.size, 6)))
    assert (~res.valid).any()
    assert np.isnan(res.p[~res.valid]).all()
    assert np.isnan(res.zncc[~res.valid]).all()


def test_shear_recovered_without_outliers(speckle_pattern):
    ref = render(speckle_pattern, None, noise_sigma=1.5)
    D = shear(0.08, CENTER)
    cur = render(speckle_pattern, D, noise_sigma=1.5, seed=5)
    cfg = _cfg()
    xs, ys = make_grid(ref.shape, None, cfg.subset_size, cfg.step)
    corr = SubsetCorrelator(ref, xs, ys, cfg)
    p0, _ = corr.initial_guess(cur, None, None)
    res = corr.correlate_robust(cur, p0)
    m = res.valid
    from dic.synth import ground_truth_fields

    gt = ground_truth_fields(D, xs, ys)
    err = np.hypot(res.u[m] - gt["ux"][m], res.v[m] - gt["uy"][m])
    assert np.max(err) < 0.5
