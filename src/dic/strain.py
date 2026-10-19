"""Strain estimation from the displacement field (requirement R7).

A first-order polynomial (plane) is fitted locally to the displacement samples on the correlation
grid -- the *virtual strain gauge* (VSG).  The linear coefficients of the fit are the displacement
gradients.  Because the grid is regular the fit is a fixed convolution, so the whole field is
processed in one pass.

Sign / coordinate conventions (image frame, x to the right, y downwards):

* ``exx = du/dx``, ``eyy = dv/dy``, ``gamma_xy = du/dy + dv/dx`` (engineering shear).
* ``F = I + grad(u)`` is the deformation gradient in the image frame.
* Green-Lagrange strain ``E = 1/2 (F^T F - I)``.
* The engineering (small-strain) tensor equals ``sym(F) - I`` and is only valid for small strain;
  the caller is warned when it exceeds ``StrainConfig.engineering_validity_limit``.

The effective spatial resolution of the strain is the virtual strain gauge length
``L_VSG = (window - 1) * step + subset`` (iDICs Good Practices Guide, Ch. 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import StrainConfig

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class StrainResult:
    """Strain fields on the correlation grid, each of shape ``(ny, nx)``.

    Attributes
    ----------
    fields : dict of str -> numpy.ndarray
        Strain components.  Invalid points are ``NaN``.
    valid : numpy.ndarray
        Boolean validity of the strain (all VSG samples valid).
    gauge_length_px : float
        Effective virtual strain gauge length in pixels.
    """

    fields: dict[str, np.ndarray]
    valid: np.ndarray
    gauge_length_px: float

    def __getitem__(self, key: str) -> np.ndarray:
        return self.fields[key]

    def keys(self) -> list[str]:
        """Names of the available strain fields."""
        return list(self.fields.keys())


def gradient_kernels(window: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane-fit gradient kernels for a *window* x *window* neighbourhood.

    Returns two ``(window, window)`` kernels ``(kx, ky)`` such that
    ``dz/dx = correlate(z, kx)`` and ``dz/dy = correlate(z, ky)``.

    Parameters
    ----------
    window : int
        Odd number of samples per axis.
    dx, dy : float
        Grid spacings in px.
    """
    if window % 2 == 0 or window < 3:
        raise ValueError("Strain window must be an odd integer >= 3.")
    h = window // 2
    i, j = np.meshgrid(np.arange(-h, h + 1), np.arange(-h, h + 1))
    M = np.stack([np.ones(i.size), i.ravel() * dx, j.ravel() * dy], axis=1)
    pinv = np.linalg.pinv(M)  # (3, k^2)
    kx = pinv[1].reshape(window, window)
    ky = pinv[2].reshape(window, window)
    return kx, ky


def fit_gradients(
    field: np.ndarray,
    valid: np.ndarray,
    window: int,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local plane-fit gradients of *field* on a regular grid.

    Parameters
    ----------
    field : numpy.ndarray
        Scalar field, shape ``(ny, nx)``; may contain ``NaN``.
    valid : numpy.ndarray
        Boolean validity, same shape.
    window : int
        Odd number of samples per axis in the fitting window.
    dx, dy : float
        Grid spacings in px.

    Returns
    -------
    gx, gy : numpy.ndarray
        Gradients along x and y (``NaN`` where the window is incomplete).
    valid_out : numpy.ndarray
        Boolean; ``True`` where the whole window was valid.
    """
    kx, ky = gradient_kernels(window, dx, dy)
    z = np.where(valid, np.nan_to_num(field, nan=0.0), 0.0)
    w = valid.astype(np.float64)
    count = ndimage.correlate(w, np.ones((window, window)), mode="constant", cval=0.0)
    valid_out = count >= (window * window) - 1e-6
    gx = ndimage.correlate(z, kx, mode="constant", cval=0.0)
    gy = ndimage.correlate(z, ky, mode="constant", cval=0.0)
    gx = np.where(valid_out, gx, np.nan)
    gy = np.where(valid_out, gy, np.nan)
    return gx, gy, valid_out


def compute_strains(
    ux: np.ndarray,
    uy: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    cfg: StrainConfig,
    *,
    subset_size: int = 1,
) -> StrainResult:
    """Compute engineering and finite strain fields from a displacement field.

    Parameters
    ----------
    ux, uy : numpy.ndarray
        Displacement components on the correlation grid, shape ``(ny, nx)`` (px).
    valid : numpy.ndarray
        Boolean validity of the displacement, shape ``(ny, nx)``.
    dx, dy : float
        Grid spacings (px).
    cfg : StrainConfig
        Strain parameters.
    subset_size : int
        Subset size (used only for the reported gauge length).

    Returns
    -------
    StrainResult
    """
    window = cfg.window
    du_dx, du_dy, v1 = fit_gradients(ux, valid, window, dx, dy)
    dv_dx, dv_dy, v2 = fit_gradients(uy, valid, window, dx, dy)
    valid_out = v1 & v2

    # Deformation gradient F = I + grad(u)
    F11 = 1.0 + du_dx
    F12 = du_dy
    F21 = dv_dx
    F22 = 1.0 + dv_dy

    # Engineering (small) strain = sym(F) - I
    exx = du_dx
    eyy = dv_dy
    gamma_xy = du_dy + dv_dx
    exy = 0.5 * gamma_xy

    # Green-Lagrange strain
    Exx = 0.5 * (F11 * F11 + F21 * F21 - 1.0)
    Eyy = 0.5 * (F12 * F12 + F22 * F22 - 1.0)
    Exy = 0.5 * (F11 * F12 + F21 * F22)
    Egamma_xy = 2.0 * Exy

    # Rotation: small-angle and finite (polar decomposition of F)
    omega = 0.5 * (dv_dx - du_dy)
    theta = _finite_rotation(F11, F12, F21, F22)

    # Principal engineering strains and directions
    e_mean = 0.5 * (exx + eyy)
    e_rad = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy * exy)
    e1 = e_mean + e_rad
    e2 = e_mean - e_rad
    theta_p = 0.5 * np.arctan2(gamma_xy, exx - eyy)

    # Principal Green-Lagrange strains
    E_mean = 0.5 * (Exx + Eyy)
    E_rad = np.sqrt((0.5 * (Exx - Eyy)) ** 2 + Exy * Exy)
    E1 = E_mean + E_rad
    E2 = E_mean - E_rad

    # Equivalent (von Mises) strain, plane-strain incompressibility eps_zz = -(eps_xx+eps_yy)
    eps_eq = np.sqrt(
        (2.0 / 3.0) * (exx * exx + eyy * eyy + (exx + eyy) ** 2 + 2.0 * exy * exy)
    )
    E_eq = np.sqrt(
        (2.0 / 3.0) * (Exx * Exx + Eyy * Eyy + (Exx + Eyy) ** 2 + 2.0 * Exy * Exy)
    )

    fields = {
        "du_dx": du_dx,
        "du_dy": du_dy,
        "dv_dx": dv_dx,
        "dv_dy": dv_dy,
        "exx": exx,
        "eyy": eyy,
        "exy": exy,
        "gamma_xy": gamma_xy,
        "Exx": Exx,
        "Eyy": Eyy,
        "Exy": Exy,
        "Egamma_xy": Egamma_xy,
        "omega": omega,
        "theta_deg": np.degrees(theta),
        "e1": e1,
        "e2": e2,
        "theta_p_deg": np.degrees(theta_p),
        "E1": E1,
        "E2": E2,
        "eps_eq": eps_eq,
        "E_eq": E_eq,
    }

    finite = np.isfinite(exx) & np.isfinite(eyy)
    if finite.any():
        peak = float(np.nanmax(np.abs(np.stack([exx[finite], eyy[finite], gamma_xy[finite]]))))
        if peak > cfg.engineering_validity_limit:
            logger.warning(
                "Engineering strain reaches %.4f, above the small-strain validity limit %.4f. "
                "Use the Green-Lagrange fields (Exx, Eyy, Exy) at this deformation level.",
                peak,
                cfg.engineering_validity_limit,
            )

    gauge = (window - 1) * max(dx, dy) + subset_size
    return StrainResult(fields=fields, valid=valid_out, gauge_length_px=float(gauge))


def _finite_rotation(F11, F12, F21, F22) -> np.ndarray:
    """Finite rotation angle (rad) from the polar decomposition of F.

    ``R = F (F^T F)^{-1/2}`` and the angle is ``atan2(R21, R11)``.  For a symmetric 2x2
    ``C = F^T F`` the inverse square root has a closed form.
    """
    C11 = F11 * F11 + F21 * F21
    C12 = F11 * F12 + F21 * F22
    C22 = F12 * F12 + F22 * F22
    det = C11 * C22 - C12 * C12
    sqdet = np.sqrt(np.clip(det, 0.0, None))
    s = np.sqrt(np.clip(C11 + C22 + 2.0 * sqdet, _EPS, None))
    ch11 = (C22 + sqdet) / s
    ch12 = -C12 / s
    R11 = F11 * ch11 + F12 * ch12
    R21 = F21 * ch11 + F22 * ch12
    return np.arctan2(R21, R11)
