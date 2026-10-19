"""Uncertainty quantification and the empirical noise floor (requirements R15, R17).

Two complementary pieces of information are produced:

1. **Per-point uncertainty** (computed in :mod:`dic.correlation`): the parameter covariance of
   the IC-GN solution is ``Cov(dp) = sigma_r^2 * H^{-1}`` where ``sigma_r`` is the standard
   deviation of the normalised matching residual and ``H`` the normalised correlation Hessian.
   ``sigma_u`` and ``sigma_v`` are the square roots of the first two diagonal entries, in pixels.

2. **Empirical noise floor**: the pipeline is run on a synthetic pair of *zero-motion* images
   carrying the same additive noise as the user's data, and the apparent displacement and strain
   it produces are measured.  This is the iDICs-recommended noise-floor analysis and it needs no
   ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import CorrelationConfig, StrainConfig
from .correlation import SubsetCorrelator
from .interpolation import estimate_image_noise, prefilter_noise_gain
from .strain import compute_strains

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyReport:
    """Summary of the uncertainty analysis for a run."""

    image_noise_sigma: float
    image_noise_method: str
    noise_floor_u_rms_px: float
    noise_floor_v_rms_px: float
    noise_floor_strain_rms: float
    noise_floor_points: int
    mean_sigma_u_px: float
    mean_sigma_v_px: float

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "image_noise_sigma_grey": self.image_noise_sigma,
            "image_noise_method": self.image_noise_method,
            "noise_floor_u_rms_px": self.noise_floor_u_rms_px,
            "noise_floor_v_rms_px": self.noise_floor_v_rms_px,
            "noise_floor_strain_rms": self.noise_floor_strain_rms,
            "noise_floor_points": self.noise_floor_points,
            "mean_sigma_u_px": self.mean_sigma_u_px,
            "mean_sigma_v_px": self.mean_sigma_v_px,
        }


def estimate_noise_from_correlation(
    corr: SubsetCorrelator, residuals: list[np.ndarray], valid: list[np.ndarray]
) -> float | None:
    """Estimate image noise (grey levels) from the DIC matching residual.

    The normalised residual ``f_norm - g_norm`` has standard deviation ``sigma_r``; multiplying
    by the reference subset norm ``df`` returns it to grey levels, and dividing by ``sqrt(2)``
    (two independent noisy images) gives the per-image noise.  Because the correlation runs on
    Gaussian-prefiltered images, the result is divided by the pre-filter's noise gain to report
    the noise of the *original* images.

    Returns ``None`` if no usable residuals are available.
    """
    gain = prefilter_noise_gain(corr.cfg.prefilter_sigma) ** 2
    vals = []
    for r, m in zip(residuals, valid, strict=False):
        if r is None or not np.any(m):
            continue
        vals.append(r[m] * corr.df[m])
    if not vals:
        return None
    pooled = np.concatenate(vals)
    if pooled.size == 0:
        return None
    return float(np.median(pooled) / (np.sqrt(2.0) * max(gain, 1e-9)))


def compute_noise_floor(
    reference: np.ndarray,
    corr: SubsetCorrelator,
    cfg: CorrelationConfig,
    strain_cfg: StrainConfig,
    *,
    noise_sigma: float,
    seed: int,
    realisations: int = 2,
    max_points: int = 900,
) -> tuple[float, float, float, int]:
    """Measure the displacement and strain noise floor on zero-motion synthetic data.

    Parameters
    ----------
    reference : numpy.ndarray
        The user's reference image (its speckle pattern is used as-is).
    corr : SubsetCorrelator
        Correlator for the full grid (used only for its grid coordinates and spacings).
    cfg, strain_cfg : configs
    noise_sigma : float
        Additive noise to inject (grey levels).
    seed : int
        RNG seed (reproducibility).
    realisations : int
        Number of independent noise realisations.
    max_points : int
        Subsample the grid to at most this many points (speed).

    Returns
    -------
    u_rms, v_rms, strain_rms, n_points
    """
    rng = np.random.default_rng(seed)
    # keep a *regular* sub-grid (so the strain window still works)
    ix = np.searchsorted(np.unique(corr.xs), corr.xs)
    iy = np.searchsorted(np.unique(corr.ys), corr.ys)
    k = 1
    while (np.sum((ix % k == 0) & (iy % k == 0)) > max_points) and k < 64:
        k += 1
    sel = np.flatnonzero((ix % k == 0) & (iy % k == 0))
    xs = corr.xs[sel]
    ys = corr.ys[sel]

    u_rms, v_rms, s_rms = [], [], []
    for _ in range(max(1, realisations)):
        ref_n = reference + rng.normal(0.0, noise_sigma, reference.shape)
        cur_n = reference + rng.normal(0.0, noise_sigma, reference.shape)
        c = SubsetCorrelator(ref_n, xs, ys, cfg)
        res = c.correlate_robust(cur_n, np.zeros((xs.size, 6)))
        m = res.valid
        if m.sum() < 5:
            continue
        u_rms.append(float(np.sqrt(np.mean(res.u[m] ** 2))))
        v_rms.append(float(np.sqrt(np.mean(res.v[m] ** 2))))
        # strain noise floor on the same grid
        if xs.size >= 25:
            ux = _to_grid(res.u, xs, ys, m)
            uy = _to_grid(res.v, xs, ys, m)
            gy = _to_grid(m.astype(float), xs, ys, m) > 0.5
            if gy.shape[0] >= 5 and gy.shape[1] >= 5:
                sr = compute_strains(ux, uy, gy, _spacing(xs), _spacing(ys), strain_cfg)
                e = sr["exx"]
                if np.isfinite(e).any():
                    s_rms.append(float(np.sqrt(np.nanmean(e[np.isfinite(e)] ** 2))))

    if not u_rms:
        logger.warning("Noise floor could not be measured (no valid points).")
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.mean(u_rms)),
        float(np.mean(v_rms)),
        float(np.mean(s_rms)) if s_rms else float("nan"),
        int(xs.size),
    )


def _spacing(v: np.ndarray) -> float:
    u = np.unique(v)
    return float(np.median(np.diff(u))) if u.size > 1 else 1.0


def _to_grid(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Place a flat point array back on its regular grid (NaN where invalid)."""
    ux = np.unique(xs)
    uy = np.unique(ys)
    grid = np.full((uy.size, ux.size), np.nan)
    ix = np.searchsorted(ux, xs)
    iy = np.searchsorted(uy, ys)
    vals = np.where(valid, values, np.nan)
    grid[iy, ix] = vals
    return grid


def image_noise_estimate(
    reference: np.ndarray,
    corr: SubsetCorrelator,
    residuals: list[np.ndarray],
    valid: list[np.ndarray],
    override: float = 0.0,
) -> tuple[float, str]:
    """Best available estimate of the additive image noise (grey levels)."""
    if override > 0:
        return float(override), "user-specified"
    res = estimate_noise_from_correlation(corr, residuals, valid)
    if res is not None and np.isfinite(res) and res > 0:
        return float(res), "DIC matching residual"
    return float(estimate_image_noise(reference)), "Immerkaer Laplacian estimator"
