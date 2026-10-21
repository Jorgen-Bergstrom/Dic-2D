"""Unit tests for the virtual-strain-gauge (plane-fit) strain estimator."""

from __future__ import annotations

import numpy as np
import pytest

from dic.config import StrainConfig
from dic.strain import compute_strains, fit_gradients, gradient_kernels


def test_gradient_kernels_are_exact_for_a_plane():
    """The plane-fit kernels must reproduce the exact gradient of a linear field."""
    window, dx, dy = 5, 2.0, 3.0
    kx, ky = gradient_kernels(window, dx, dy)
    assert kx.shape == (window, window)
    assert kx.sum() == pytest.approx(0.0, abs=1e-12)
    assert ky.sum() == pytest.approx(0.0, abs=1e-12)
    # a plane z = a + b x + c y  ->  kernels must give exactly b and c at the centre
    h = window // 2
    i, j = np.meshgrid(np.arange(-h, h + 1), np.arange(-h, h + 1))
    z = 1.0 + 2.0 * (i * dx) - 0.5 * (j * dy)
    assert np.sum(kx * z) == pytest.approx(2.0, abs=1e-10)
    assert np.sum(ky * z) == pytest.approx(-0.5, abs=1e-10)


def test_fit_gradients_recovers_analytic_gradient_of_a_linear_field():
    ny, nx = 20, 25
    dx, dy = 4.0, 4.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    X = xs * dx
    Y = ys * dy
    field = 3.0 * X - 1.5 * Y + 7.0
    valid = np.ones_like(field, dtype=bool)
    gx, gy, ok = fit_gradients(field, valid, 5, dx, dy)
    interior = ok & (xs > 2) & (xs < nx - 3) & (ys > 2) & (ys < ny - 3)
    assert np.allclose(gx[interior], 3.0, atol=1e-9)
    assert np.allclose(gy[interior], -1.5, atol=1e-9)


def test_fit_gradients_reports_nan_near_invalid_samples():
    ny, nx = 15, 15
    field = np.zeros((ny, nx))
    valid = np.ones((ny, nx), dtype=bool)
    valid[7, 7] = False
    gx, gy, ok = fit_gradients(field, valid, 3, 1.0, 1.0)
    assert not ok[7, 7]
    assert not ok[6, 6]  # within the window of the invalid sample
    # a full window is required, so the outer ring of the grid has no strain
    assert not ok[0, 0]
    assert ok[3, 3]


def test_compute_strains_nominal_and_true_agree_at_small_strain():
    ny, nx = 30, 30
    dx = dy = 2.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    ex, ey, gam = 1e-3, -5e-4, 2e-4
    ux = ex * xs * dx + 0.5 * gam * ys * dy
    uy = ey * ys * dy + 0.5 * gam * xs * dx
    valid = np.ones_like(ux, dtype=bool)
    sr = compute_strains(ux, uy, valid, dx, dy, StrainConfig(window=5), subset_size=31)
    interior = sr.valid
    assert np.nanmean(sr["Eeng_xx"][interior]) == pytest.approx(ex, abs=1e-9)
    assert np.nanmean(sr["Eeng_yy"][interior]) == pytest.approx(ey, abs=1e-9)
    assert np.nanmean(sr["Eeng_gamma_xy"][interior]) == pytest.approx(gam, abs=1e-9)
    assert np.nanmean(sr["Eeng_xy"][interior]) == pytest.approx(0.5 * gam, abs=1e-9)
    # at small strain the true strain equals the nominal strain to first order
    assert np.nanmean(sr["Etr_xx"][interior]) == pytest.approx(ex, abs=1e-6)
    assert np.nanmean(sr["Etr_yy"][interior]) == pytest.approx(ey, abs=1e-6)
    assert np.nanmean(sr["Etr_gamma_xy"][interior]) == pytest.approx(gam, abs=1e-6)


def test_compute_strains_true_differs_from_nominal_at_large_strain():
    ny, nx = 30, 30
    dx = dy = 1.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    ex = 0.4
    ux = ex * xs * dx
    uy = np.zeros_like(ux)
    valid = np.ones_like(ux, dtype=bool)
    sr = compute_strains(ux, uy, valid, dx, dy, StrainConfig(window=5), subset_size=21)
    interior = sr.valid
    # nominal strain is lambda - 1; true strain is ln(lambda)
    assert np.nanmean(sr["Eeng_xx"][interior]) == pytest.approx(0.4, abs=1e-9)
    assert np.nanmean(sr["Eeng_yy"][interior]) == pytest.approx(0.0, abs=1e-9)
    assert np.nanmean(sr["Eeng_1"][interior]) == pytest.approx(0.4, abs=1e-9)
    assert np.nanmean(sr["Eeng_2"][interior]) == pytest.approx(0.0, abs=1e-9)
    assert np.nanmean(sr["Etr_xx"][interior]) == pytest.approx(np.log(1.4), abs=1e-6)
    assert np.nanmean(sr["Etr_yy"][interior]) == pytest.approx(0.0, abs=1e-9)
    assert np.nanmean(sr["Etr_1"][interior]) == pytest.approx(np.log(1.4), abs=1e-6)
    assert np.nanmean(sr["Etr_2"][interior]) == pytest.approx(0.0, abs=1e-9)
    assert np.nanmean(sr["Etr_eq"][interior]) == pytest.approx(
        2.0 / np.sqrt(3.0) * np.log(1.4), abs=1e-6
    )


def test_rotation_produces_zero_nominal_strain_and_correct_angle():
    ny, nx = 40, 40
    dx = dy = 1.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    cx, cy = 20.0, 20.0
    theta = np.radians(5.0)
    X = xs * dx - cx
    Y = ys * dy - cy
    ux = (np.cos(theta) - 1) * X - np.sin(theta) * Y
    uy = np.sin(theta) * X + (np.cos(theta) - 1) * Y
    valid = np.ones_like(ux, dtype=bool)
    sr = compute_strains(ux, uy, valid, dx, dy, StrainConfig(window=5), subset_size=21)
    interior = sr.valid
    # nominal strain is V - I and true strain is ln(V), so a rigid rotation gives zero
    assert abs(np.nanmean(sr["Eeng_xx"][interior])) < 1e-9
    assert abs(np.nanmean(sr["Eeng_yy"][interior])) < 1e-9
    assert abs(np.nanmean(sr["Eeng_gamma_xy"][interior])) < 1e-9
    assert abs(np.nanmean(sr["Eeng_eq"][interior])) < 1e-9
    assert abs(np.nanmean(sr["Etr_xx"][interior])) < 1e-9
    assert abs(np.nanmean(sr["Etr_yy"][interior])) < 1e-9
    assert abs(np.nanmean(sr["Etr_gamma_xy"][interior])) < 1e-9
    assert abs(np.nanmean(sr["Etr_eq"][interior])) < 1e-9
    assert np.nanmean(sr["theta_deg"][interior]) == pytest.approx(5.0, abs=1e-6)


def test_simple_shear_uses_left_stretch_not_sym_F():
    ny, nx = 30, 30
    dx = dy = 1.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    gamma = 0.5
    ux = gamma * ys * dy
    uy = np.zeros_like(ux)
    valid = np.ones_like(ux, dtype=bool)
    sr = compute_strains(ux, uy, valid, dx, dy, StrainConfig(window=5), subset_size=21)
    interior = sr.valid

    F = np.array([[1.0, gamma], [0.0, 1.0]])
    b = F @ F.T
    eigvals, eigvecs = np.linalg.eigh(b)
    V = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    T = eigvecs @ np.diag(0.5 * np.log(eigvals)) @ eigvecs.T

    assert np.nanmean(sr["Eeng_xx"][interior]) == pytest.approx(V[0, 0] - 1.0, abs=1e-9)
    assert np.nanmean(sr["Eeng_yy"][interior]) == pytest.approx(V[1, 1] - 1.0, abs=1e-9)
    assert np.nanmean(sr["Eeng_xy"][interior]) == pytest.approx(V[0, 1], abs=1e-9)
    assert np.nanmean(sr["Eeng_gamma_xy"][interior]) == pytest.approx(2.0 * V[0, 1], abs=1e-9)
    assert np.nanmean(sr["Etr_xx"][interior]) == pytest.approx(T[0, 0], abs=1e-9)
    assert np.nanmean(sr["Etr_yy"][interior]) == pytest.approx(T[1, 1], abs=1e-9)
    assert np.nanmean(sr["Etr_gamma_xy"][interior]) == pytest.approx(2.0 * T[0, 1], abs=1e-9)
    assert np.nanmean(sr["Etr_1"][interior]) == pytest.approx(0.5 * np.log(eigvals[1]), abs=1e-9)
    assert np.nanmean(sr["Etr_2"][interior]) == pytest.approx(0.5 * np.log(eigvals[0]), abs=1e-9)
    # sym(F) - I would give Eeng_xx = 0; the nominal tensor does not
    assert abs(np.nanmean(sr["Eeng_xx"][interior])) > 1e-3


def test_equivalent_strain_is_zero_for_zero_strain():
    ny, nx = 20, 20
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    valid = np.ones((ny, nx), dtype=bool)
    sr = compute_strains(ux, uy, valid, 1.0, 1.0, StrainConfig(window=3), subset_size=11)
    assert np.nanmax(np.abs(sr["Eeng_eq"])) == pytest.approx(0.0, abs=1e-12)
    assert np.nanmax(np.abs(sr["Etr_eq"])) == pytest.approx(0.0, abs=1e-12)
