"""Unit tests for the B-spline interpolator and its exact derivatives."""

from __future__ import annotations

import numpy as np
import pytest

from dic.interpolation import (
    SplineInterpolator,
    estimate_image_noise,
    gaussian_prefilter,
    prefilter_noise_gain,
)

YY, XX = np.mgrid[0:200, 0:200].astype(float)


def _field(x, y):
    return np.sin(0.11 * x) * np.cos(0.07 * y) + 0.3 * x


def _dx(x, y):
    return 0.11 * np.cos(0.11 * x) * np.cos(0.07 * y) + 0.3


def _dy(x, y):
    return -0.07 * np.sin(0.11 * x) * np.sin(0.07 * y)


@pytest.mark.parametrize("order", [1, 3, 5])
def test_values_match_analytic(order):
    f = _field(XX, YY)
    interp = SplineInterpolator(f, order=order)
    rng = np.random.default_rng(0)
    y = rng.uniform(30, 170, 1500)
    x = rng.uniform(30, 170, 1500)
    err = np.max(np.abs(interp.values(y, x) - _field(x, y)))
    tol = {1: 5e-3, 3: 1e-5, 5: 1e-6}[order]
    assert err < tol


@pytest.mark.parametrize("order", [1, 3, 5])
def test_gradients_match_analytic(order):
    f = _field(XX, YY)
    interp = SplineInterpolator(f, order=order)
    rng = np.random.default_rng(1)
    y = rng.uniform(30, 170, 1500)
    x = rng.uniform(30, 170, 1500)
    gx, gy = interp.gradients(y, x)
    tol = {1: 5e-2, 3: 1e-3, 5: 1e-4}[order]
    assert np.max(np.abs(gx - _dx(x, y))) < tol
    assert np.max(np.abs(gy - _dy(x, y))) < tol


def test_interpolator_is_exact_on_a_linear_ramp():
    f = 2.5 * XX + 3.0 * YY
    interp = SplineInterpolator(f, order=3)
    y = np.array([42.3, 95.7, 120.1])
    x = np.array([37.9, 74.4, 130.2])
    assert np.allclose(interp.values(y, x), 2.5 * x + 3.0 * y, atol=1e-6)
    gx, gy = interp.gradients(y, x)
    assert np.allclose(gx, 2.5, atol=1e-6)
    assert np.allclose(gy, 3.0, atol=1e-6)


def test_gaussian_prefilter_zero_sigma_is_identity():
    img = np.random.default_rng(0).random((10, 10))
    assert np.allclose(gaussian_prefilter(img, 0.0), img)


def test_prefilter_noise_gain_is_less_than_one_and_decreasing():
    assert prefilter_noise_gain(0.0) == pytest.approx(1.0)
    g1 = prefilter_noise_gain(1.0)
    g2 = prefilter_noise_gain(2.0)
    assert 0.0 < g1 < 1.0
    assert g2 < g1


def test_noise_estimator_recovers_known_sigma():
    rng = np.random.default_rng(4)
    base = 100.0 + 20.0 * np.sin(0.02 * XX)
    noisy = base + rng.normal(0.0, 3.0, base.shape)
    est = estimate_image_noise(noisy)
    assert 2.0 < est < 4.5
