"""Unit tests for the affine kinematic model and warp composition."""

from __future__ import annotations

import numpy as np
import pytest

from dic.kinematics import (
    affine_from_global,
    compose_inverse_update,
    extrapolate_params,
    left_stretch,
    matrix_to_params,
    params_to_matrix,
    true_strain,
    warp_points,
)


def test_params_matrix_roundtrip():
    p = np.array([[1.0, -2.0, 0.1, 0.02, -0.03, 0.2]])
    A, t = params_to_matrix(p)
    p2 = matrix_to_params(A, t)
    assert np.allclose(p, p2)


def test_warp_points_matches_manual_affine():
    p = np.array([[3.0, -1.0, 0.05, 0.01, 0.0, -0.02]])
    x0 = np.array([100.0])
    y0 = np.array([50.0])
    xi_x = np.array([-1.0, 0.0, 1.0])
    xi_y = np.array([0.0, 0.0, 0.0])
    Xt, Yt = warp_points(p, x0, y0, xi_x, xi_y)
    assert np.allclose(Xt, 100.0 + 3.0 + 1.05 * xi_x)
    assert np.allclose(Yt, 50.0 - 1.0 + 0.0 * xi_x)


def test_compose_inverse_update_converges_to_target_translation():
    # start at zero, apply the exact inverse increment once -> should land on the target
    p = np.zeros((1, 6))
    dp = np.array([[2.0, -3.0, 0.0, 0.0, 0.0, 0.0]])
    out = compose_inverse_update(p, dp)
    assert out[0, 0] == pytest.approx(-2.0)
    assert out[0, 1] == pytest.approx(3.0)


def test_affine_from_global_matches_direct_map():
    A_g = np.array([[1.02, 0.03], [-0.01, 0.97]])
    t_g = np.array([4.0, -5.0])
    x0 = np.array([10.0, 200.0])
    y0 = np.array([20.0, 300.0])
    p = affine_from_global(A_g, t_g, x0, y0)
    xi_x = np.array([0.0, 2.0, -2.0])
    xi_y = np.array([0.0, 1.0, 3.0])
    Xt, Yt = warp_points(p, x0, y0, xi_x, xi_y)
    for k in range(x0.size):
        for j in range(xi_x.size):
            ref = np.array([x0[k] + xi_x[j], y0[k] + xi_y[j]])
            want = A_g @ ref + t_g
            assert Xt[k, j] == pytest.approx(want[0])
            assert Yt[k, j] == pytest.approx(want[1])


def test_extrapolate_params_linear():
    p1 = np.array([[1.0, 1.0, 0, 0, 0, 0]])
    p2 = np.array([[2.0, 2.0, 0, 0, 0, 0]])
    out = extrapolate_params(p1, p2)
    assert out[0, 0] == pytest.approx(3.0)
    assert np.allclose(extrapolate_params(None, p2), p2)


def test_left_stretch_matches_eigen_decomposition():
    F = np.array([[1.2, 0.3], [-0.1, 0.9]])
    V11, V12, V22 = left_stretch(F[0, 0], F[0, 1], F[1, 0], F[1, 1])
    b = F @ F.T
    eigvals, eigvecs = np.linalg.eigh(b)
    V = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    assert V11 == pytest.approx(V[0, 0])
    assert V12 == pytest.approx(V[0, 1])
    assert V22 == pytest.approx(V[1, 1])


def test_left_stretch_of_rotation_is_identity():
    theta = np.radians(30.0)
    c, s = np.cos(theta), np.sin(theta)
    V11, V12, V22 = left_stretch(c, -s, s, c)
    assert V11 == pytest.approx(1.0)
    assert V12 == pytest.approx(0.0, abs=1e-15)
    assert V22 == pytest.approx(1.0)


def test_true_strain_matches_matrix_logarithm():
    F = np.array([[1.2, 0.3], [-0.1, 0.9]])
    T11, T12, T22 = true_strain(F[0, 0], F[0, 1], F[1, 0], F[1, 1])
    b = F @ F.T
    eigvals, eigvecs = np.linalg.eigh(b)
    T = eigvecs @ np.diag(0.5 * np.log(eigvals)) @ eigvecs.T
    assert T11 == pytest.approx(T[0, 0])
    assert T12 == pytest.approx(T[0, 1])
    assert T22 == pytest.approx(T[1, 1])


def test_true_strain_of_rotation_is_zero():
    theta = np.radians(30.0)
    c, s = np.cos(theta), np.sin(theta)
    T11, T12, T22 = true_strain(c, -s, s, c)
    assert T11 == pytest.approx(0.0, abs=1e-15)
    assert T12 == pytest.approx(0.0, abs=1e-15)
    assert T22 == pytest.approx(0.0, abs=1e-15)
