"""Unit tests for uncertainty quantification and the noise floor."""

from __future__ import annotations

import numpy as np

from dic.config import CorrelationConfig, StrainConfig
from dic.correlation import SubsetCorrelator, make_grid
from dic.synth import render
from dic.uncertainty import compute_noise_floor, estimate_noise_from_correlation


def _cfg():
    return CorrelationConfig(subset_size=21, step=8, prefilter_sigma=0.7)


def test_residual_noise_estimate_is_close_to_truth(speckle_pattern):
    ref = render(speckle_pattern, None, noise_sigma=2.0, seed=1)
    cur = render(speckle_pattern, None, noise_sigma=2.0, seed=2)
    cfg = _cfg()
    xs, ys = make_grid(ref.shape, None, cfg.subset_size, cfg.step)
    corr = SubsetCorrelator(ref, xs, ys, cfg)
    res = corr.correlate_robust(cur, np.zeros((xs.size, 6)))
    est = estimate_noise_from_correlation(corr, [res.residual], [res.valid])
    assert est is not None
    assert 1.0 < est < 3.5


def test_noise_floor_is_small_on_zero_motion(speckle_pattern):
    ref = render(speckle_pattern, None, noise_sigma=2.0, seed=1)
    cfg = _cfg()
    xs, ys = make_grid(ref.shape, None, cfg.subset_size, cfg.step)
    corr = SubsetCorrelator(ref, xs, ys, cfg)
    u_rms, v_rms, s_rms, n = compute_noise_floor(
        ref, corr, cfg, StrainConfig(window=3), noise_sigma=2.0, seed=0, realisations=2
    )
    assert n > 0
    assert u_rms < 0.05
    assert v_rms < 0.05
    assert np.isfinite(s_rms)
    assert s_rms < 2e-3


def test_noise_floor_grows_with_noise(speckle_pattern):
    ref = render(speckle_pattern, None, noise_sigma=0.0)
    cfg = _cfg()
    xs, ys = make_grid(ref.shape, None, cfg.subset_size, cfg.step)
    corr = SubsetCorrelator(ref, xs, ys, cfg)
    lo = compute_noise_floor(ref, corr, cfg, StrainConfig(window=3), noise_sigma=0.5, seed=0)[0]
    hi = compute_noise_floor(ref, corr, cfg, StrainConfig(window=3), noise_sigma=4.0, seed=0)[0]
    assert hi > lo
