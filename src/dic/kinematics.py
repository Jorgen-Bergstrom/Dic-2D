"""Affine kinematic model and warp composition for IC-GN.

A first-order (affine) warp maps a local subset coordinate :math:`\\xi = (x-x_0, y-y_0)` to the
deformed image:

.. math::

    \\begin{pmatrix} X_t \\\\ Y_t \\end{pmatrix}
    = \\begin{pmatrix} x_0 \\\\ y_0 \\end{pmatrix}
    + \\begin{pmatrix} u \\\\ v \\end{pmatrix}
    + \\begin{bmatrix} 1+u_x & u_y \\\\ v_x & 1+v_y \\end{bmatrix}
      \\begin{pmatrix} \\xi_x \\\\ \\xi_y \\end{pmatrix}

The six parameters are ``p = (u, v, u_x, u_y, v_x, v_y)``.  IC-GN composes an inverse incremental
warp into the current warp::

    A_new = A_old @ inv(A_delta)
    t_new = t_old - A_new @ t_delta

All functions operate on arrays of points (one row per point of interest).
"""

from __future__ import annotations

import numpy as np

#: Number of affine parameters.
N_PARAMS = 6


def params_to_matrix(
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a parameter array into affine matrices and translation vectors.

    Parameters
    ----------
    p : numpy.ndarray
        Shape ``(N, 6)`` or ``(6,)`` with columns ``(u, v, u_x, u_y, v_x, v_y)``.

    Returns
    -------
    A : numpy.ndarray
        Shape ``(N, 2, 2)``.
    t : numpy.ndarray
        Shape ``(N, 2)``.
    """
    p = np.atleast_2d(p)
    n = p.shape[0]
    A = np.zeros((n, 2, 2))
    A[:, 0, 0] = 1.0 + p[:, 2]
    A[:, 0, 1] = p[:, 3]
    A[:, 1, 0] = p[:, 4]
    A[:, 1, 1] = 1.0 + p[:, 5]
    t = p[:, :2]
    return A, t


def matrix_to_params(A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of :func:`params_to_matrix`."""
    A = np.atleast_3d(A)
    t = np.atleast_2d(t)
    return np.stack(
        [t[:, 0], t[:, 1], A[:, 0, 0] - 1.0, A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] - 1.0],
        axis=1,
    )


def warp_points(
    p: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    xi_x: np.ndarray,
    xi_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map local subset coordinates to target coordinates.

    Parameters
    ----------
    p : numpy.ndarray
        Shape ``(N, 6)`` parameters.
    x0, y0 : numpy.ndarray
        Shape ``(N,)`` reference point coordinates.
    xi_x, xi_y : numpy.ndarray
        Shape ``(nsub,)`` local subset coordinates.

    Returns
    -------
    Xt, Yt : numpy.ndarray
        Shape ``(N, nsub)`` target coordinates.
    """
    A, t = params_to_matrix(p)
    a11, a12 = A[:, 0, 0:1], A[:, 0, 1:2]
    a21, a22 = A[:, 1, 0:1], A[:, 1, 1:2]
    xx = xi_x[None, :]
    yy = xi_y[None, :]
    Xt = x0[:, None] + t[:, 0:1] + a11 * xx + a12 * yy
    Yt = y0[:, None] + t[:, 1:2] + a21 * xx + a22 * yy
    return Xt, Yt


def compose_inverse_update(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """Apply the IC-GN inverse-compositional update ``p <- p o dp^{-1}``.

    Parameters
    ----------
    p, dp : numpy.ndarray
        Shape ``(N, 6)`` current and incremental parameters.

    Returns
    -------
    numpy.ndarray
        Updated parameters, shape ``(N, 6)``.
    """
    A, t = params_to_matrix(p)
    Ad, td = params_to_matrix(dp)
    Ad_inv = np.linalg.inv(Ad)
    A_new = A @ Ad_inv
    t_new = t - np.einsum("nij,nj->ni", A_new, td)
    return matrix_to_params(A_new, t_new)


def affine_from_global(
    A_g: np.ndarray, t_g: np.ndarray, x0: np.ndarray, y0: np.ndarray
) -> np.ndarray:
    """Convert a global affine map into per-point local warp parameters.

    The global map is ``x_def = A_g @ x_ref + t_g``; the local warp about ``x0`` is
    ``A = A_g`` and ``t = A_g @ x0 + t_g - x0``.

    Parameters
    ----------
    A_g : numpy.ndarray
        Shape ``(2, 2)``.
    t_g : numpy.ndarray
        Shape ``(2,)``.
    x0, y0 : numpy.ndarray
        Shape ``(N,)`` reference coordinates.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 6)``.
    """
    A_g = np.asarray(A_g, dtype=np.float64)
    t_g = np.asarray(t_g, dtype=np.float64)
    pts = np.stack([x0, y0], axis=1)  # (N, 2)
    moved = pts @ A_g.T + t_g
    diff = moved - pts
    n = x0.size
    p = np.zeros((n, N_PARAMS))
    p[:, 0] = diff[:, 0]
    p[:, 1] = diff[:, 1]
    p[:, 2] = A_g[0, 0] - 1.0
    p[:, 3] = A_g[0, 1]
    p[:, 4] = A_g[1, 0]
    p[:, 5] = A_g[1, 1] - 1.0
    return p


def extrapolate_params(p_prev2: np.ndarray | None, p_prev: np.ndarray) -> np.ndarray:
    """Linear temporal extrapolation of the warp parameters.

    Returns ``2*p_prev - p_prev2`` when two previous frames are available, else ``p_prev``.
    """
    if p_prev2 is None:
        return p_prev.copy()
    return 2.0 * p_prev - p_prev2
