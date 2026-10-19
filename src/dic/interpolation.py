"""Sub-pixel interpolation: prefiltered B-splines with *exact* derivatives.

The deformed image is sampled at arbitrary sub-pixel coordinates by IC-GN, and the reference
gradient is needed for the steepest-descent images.  Both are obtained from a prefiltered
B-spline (order 3 = bicubic, order 5 = biquintic) using ``scipy.ndimage.map_coordinates``.

The derivative of an order-*n* B-spline expansion is exact and cheap:

.. math::

    \\frac{d}{dx} B_n(x) = B_{n-1}(x + \\tfrac12) - B_{n-1}(x - \\tfrac12)

so the gradient is the difference of two order-``n-1`` evaluations on the *same* coefficient
array.  This avoids finite-difference approximations entirely.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class SplineInterpolator:
    """Prefiltered B-spline interpolator over a 2-D image.

    Parameters
    ----------
    image : numpy.ndarray
        2-D image.
    order : int, default 3
        B-spline order: 1 = linear, 3 = bicubic, 5 = biquintic.
    mode : str, default "mirror"
        Boundary mode for evaluation (out-of-bounds samples are additionally rejected by the
        caller's explicit bounds check).
    """

    def __init__(self, image: np.ndarray, order: int = 3, mode: str = "mirror") -> None:
        self.order = int(order)
        self.mode = mode
        self.image = np.ascontiguousarray(image, dtype=np.float64)
        if self.order >= 2:
            self.coeffs = ndimage.spline_filter(
                self.image, order=self.order, mode="mirror", output=np.float64
            )
        else:
            # linear/nearest interpolation needs no prefilter (coefficients == samples)
            self.coeffs = self.image
        self.shape = self.image.shape

    # -- sampling ---------------------------------------------------------------------
    def values(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Sample the spline at flat coordinate arrays *y* (row) and *x* (col)."""
        coords = np.empty((2, y.size), dtype=np.float64)
        coords[0] = y.ravel()
        coords[1] = x.ravel()
        out = ndimage.map_coordinates(
            self.coeffs, coords, order=self.order, prefilter=False, mode=self.mode
        )
        return out

    def gradients(self, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return exact ``(d/dx, d/dy)`` of the spline at flat coordinates.

        Returns
        -------
        gx, gy : numpy.ndarray
            Derivatives along the column (x) and row (y) axes.
        """
        o = self.order
        if o < 1:
            raise ValueError("Gradient requires spline order >= 1.")
        c = self.coeffs
        half = 0.5
        coords = np.empty((2, y.size), dtype=np.float64)

        coords[0] = y.ravel()
        coords[1] = (x.ravel() + half)
        gx = ndimage.map_coordinates(c, coords, order=o - 1, prefilter=False, mode=self.mode)
        coords[1] = (x.ravel() - half)
        gx -= ndimage.map_coordinates(c, coords, order=o - 1, prefilter=False, mode=self.mode)

        coords[0] = (y.ravel() + half)
        coords[1] = x.ravel()
        gy = ndimage.map_coordinates(c, coords, order=o - 1, prefilter=False, mode=self.mode)
        coords[0] = (y.ravel() - half)
        gy -= ndimage.map_coordinates(c, coords, order=o - 1, prefilter=False, mode=self.mode)
        return gx, gy


def gaussian_prefilter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Return a Gaussian-smoothed copy of *image* (sigma in px; 0 disables)."""
    if sigma <= 0:
        return np.asarray(image, dtype=np.float64)
    return ndimage.gaussian_filter(
        np.asarray(image, dtype=np.float64), sigma, mode="mirror"
    )


def prefilter_noise_gain(sigma: float, truncate: float = 4.0) -> float:
    """Noise attenuation factor of :func:`gaussian_prefilter`.

    White noise of unit variance filtered by a normalised 2-D Gaussian kernel ``g`` comes out
    with variance ``sum(g^2)``.  The factor returned here is ``sqrt(sum(g^2))`` (1-D), so the
    *original* noise is ``sigma_measured / gain**2`` for a separable 2-D filter.  It is used to
    undo the pre-filter when the image noise is estimated from the matching residual.
    """
    if sigma <= 0:
        return 1.0
    n = int(2 * np.ceil(truncate * sigma) + 1)
    k = np.zeros(n)
    k[n // 2] = 1.0
    k = ndimage.gaussian_filter1d(k, sigma, mode="constant", truncate=truncate)
    return float(np.sqrt(np.sum(k * k)))


def estimate_image_noise(image: np.ndarray) -> float:
    """Estimate additive Gaussian image noise (grey levels) with Immerkær's method.

    Uses the 3x3 Laplacian mask and a robust mean of its absolute response:

    .. math::

        \\hat\\sigma = \\sqrt{\\tfrac{\\pi}{2}}\\,
                      \\frac{1}{6 (W-2)(H-2)} \\sum |I * M|.

    Parameters
    ----------
    image : numpy.ndarray
        2-D image.

    Returns
    -------
    float
        Estimated grey-level standard deviation of the noise.
    """
    img = np.asarray(image, dtype=np.float64)
    mask = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    resp = ndimage.convolve(img, mask, mode="mirror")
    h, w = img.shape
    if h < 3 or w < 3:
        return 0.0
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(resp[1:-1, 1:-1])) / 6.0)
