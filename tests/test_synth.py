"""Unit tests for the synthetic generator and its analytic ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from dic.synth import (
    AffineDeformation,
    DotPattern,
    NonUniformDeformation,
    ground_truth_fields,
    render,
    rotation,
    stretch,
)


def test_affine_deformation_and_jacobian():
    F = np.array([[1.1, 0.05], [-0.02, 0.9]])
    t = np.array([3.0, -4.0])
    D = AffineDeformation(F, t, np.array([10.0, 20.0]))
    pts = np.array([[0.0, 0.0], [5.0, 5.0]])
    moved = D(pts)
    assert np.allclose(moved, (pts - np.array([10.0, 20.0])) @ F.T + np.array([10.0, 20.0]) + t)
    assert np.allclose(D.jacobian(pts), F)


def test_nonuniform_jacobian_matches_finite_difference():
    D = NonUniformDeformation(a=3.0, b=2.0, kx=0.02, ky=0.015,
                              shear=(0.01, -0.005, 0.004, -0.02), center=np.zeros(2))
    pts = np.array([[100.0, 50.0], [30.0, 200.0]])
    h = 1e-4
    J = D.jacobian(pts)
    for k in range(pts.shape[0]):
        for j in range(2):
            dp = np.zeros(2)
            dp[j] = h
            fd = (D(pts[k] + dp) - D(pts[k] - dp)) / (2 * h)
            assert np.allclose(J[k, :, j], fd, atol=1e-6)


def test_ground_truth_fields_are_consistent_with_displacement():
    D = stretch(0.2, -0.05, np.array([50.0, 50.0]))
    xs = np.array([10.0, 60.0])
    ys = np.array([20.0, 70.0])
    gt = ground_truth_fields(D, xs, ys)
    assert np.allclose(gt["ux"], 0.2 * (xs - 50.0))
    assert np.allclose(gt["uy"], -0.05 * (ys - 50.0))
    assert np.allclose(gt["exx"], 0.2)
    assert np.allclose(gt["eyy"], -0.05)
    # Green-Lagrange for a uniaxial stretch
    assert np.allclose(gt["Exx"], 0.5 * (1.2 ** 2 - 1))


def test_render_deformation_moves_the_pattern():
    # pattern confined to the centre so nothing leaves the frame when it shifts
    pat = DotPattern.generate((300, 300), n_dots=200, seed=0, softness=1.0,
                              region=(80, 80, 220, 220))
    ref = render(pat, None)
    shifted = render(pat, AffineDeformation(np.eye(2), np.array([10.0, 0.0])))
    ys, xs = np.mgrid[0:300, 0:300]
    w = np.clip(ref - ref.min(), 0, None)
    w2 = np.clip(shifted - shifted.min(), 0, None)
    cx1 = (w * xs).sum() / w.sum()
    cx2 = (w2 * xs).sum() / w2.sum()
    assert cx2 - cx1 == pytest.approx(10.0, abs=0.2)


def test_rotation_preserves_pattern_mass():
    pat = DotPattern.generate((100, 100), n_dots=50, seed=1, softness=1.0)
    ref = render(pat, None)
    rot = render(pat, rotation(np.radians(10.0), np.array([50.0, 50.0])))
    # a rotation about the centre should keep the mean roughly equal (edges differ)
    assert abs(ref.mean() - rot.mean()) < 5.0


def test_render_noise_sigma_is_reproducible():
    pat = DotPattern.generate((60, 60), n_dots=20, seed=2)
    a = render(pat, None, noise_sigma=2.0, seed=7)
    b = render(pat, None, noise_sigma=2.0, seed=7)
    c = render(pat, None, noise_sigma=2.0, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_render_illumination_gain_and_offset():
    pat = DotPattern.generate((60, 60), n_dots=30, seed=3)
    base = render(pat, None, clip=False)
    lit = render(pat, None, gain=1.1, offset=5.0, clip=False)
    assert np.allclose(lit, 1.1 * base + 5.0)
