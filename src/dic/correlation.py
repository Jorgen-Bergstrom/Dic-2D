"""Subset-based correlation core: inverse-compositional Gauss-Newton (IC-GN), ZNSSD, affine.

Design notes
------------
* Everything is vectorised over the points of interest: one IC-GN iteration processes every
  point simultaneously.
* Reference-side quantities (subset intensities, exact B-spline gradients, steepest-descent
  images and the constant Hessian) are computed **once** for the whole sequence and reused.
* Initial guesses come from a temporal prediction (linear extrapolation of the previous frames)
  with a hierarchical multi-scale affine fallback for large inter-frame motion.
* A *re-seed* pass (reliability-guided, in the spirit of Ncorr's RG-DIC) recovers points that
  converged to a wrong local minimum by restarting them from their neighbours' solution.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import CorrelationConfig
from .interpolation import SplineInterpolator, gaussian_prefilter
from .kinematics import (
    N_PARAMS,
    affine_from_global,
    compose_inverse_update,
    extrapolate_params,
    warp_points,
)

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class FrameResult:
    """Correlation result for a set of grid points.

    All arrays have shape ``(N,)`` (one entry per point).  Invalid points are marked by
    ``valid == False`` and their numeric fields are ``NaN`` -- never fabricated.
    """

    p: np.ndarray
    zncc: np.ndarray
    sigma_u: np.ndarray
    sigma_v: np.ndarray
    residual: np.ndarray
    sssig: np.ndarray
    iterations: np.ndarray
    valid: np.ndarray
    global_affine: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def u(self) -> np.ndarray:
        return self.p[:, 0]

    @property
    def v(self) -> np.ndarray:
        return self.p[:, 1]

    @property
    def ux(self) -> np.ndarray:
        return self.p[:, 2]

    @property
    def uy(self) -> np.ndarray:
        return self.p[:, 3]

    @property
    def vx(self) -> np.ndarray:
        return self.p[:, 4]

    @property
    def vy(self) -> np.ndarray:
        return self.p[:, 5]


# --------------------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------------------
def make_grid(
    shape: tuple[int, int],
    mask: np.ndarray | None,
    subset_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the correlation-point grid.

    Points are placed on a regular grid with spacing *step* such that the **entire** subset
    footprint lies inside the (eroded) mask and inside the image.  Requiring the full footprint
    keeps the Hessian identical for every point; the price is a border of missing data near the
    ROI edge, which is standard practice.
    """
    h, w = shape
    r = subset_size // 2
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    structure = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    ys_all = np.arange(r, h - r, step)
    xs_all = np.arange(r, w - r, step)
    xs, ys = np.meshgrid(xs_all, ys_all)
    xs = xs.ravel()
    ys = ys.ravel()
    keep = eroded[ys, xs]
    return xs[keep].astype(np.float64), ys[keep].astype(np.float64)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _zncc_subsets(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """ZNCC between stacked subsets; last axis is the sample axis."""
    fc = f - f.mean(-1, keepdims=True)
    gc = g - g.mean(-1, keepdims=True)
    num = np.sum(fc * gc, axis=-1)
    den = np.sqrt(np.sum(fc * fc, axis=-1) * np.sum(gc * gc, axis=-1))
    return num / np.where(den > _EPS, den, np.nan)


def _downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased integer downsample by *factor*."""
    if factor == 1:
        return np.asarray(image, dtype=np.float64)
    smoothed = ndimage.gaussian_filter(np.asarray(image, float), sigma=0.5 * factor, mode="mirror")
    return ndimage.zoom(smoothed, 1.0 / factor, order=1, mode="nearest", prefilter=False)


def _robust_affine_fit(
    src: np.ndarray, dst: np.ndarray, iters: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine fit ``dst = A @ src + t`` with iterative outlier rejection."""
    keep = np.ones(src.shape[0], dtype=bool)
    A_g = np.eye(2)
    t_g = np.zeros(2)
    for _ in range(iters):
        s = src[keep]
        d = dst[keep]
        if s.shape[0] < 3:
            break
        M = np.concatenate([s, np.ones((s.shape[0], 1))], axis=1)
        coef, *_ = np.linalg.lstsq(M, d, rcond=None)
        A_g = coef[:2, :].T
        t_g = coef[2, :]
        pred = s @ A_g.T + t_g
        resid = np.linalg.norm(pred - d, axis=1)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) + 1e-9
        keep_sub = resid <= med + 4.0 * 1.4826 * mad
        if keep_sub.sum() < 3:
            break
        idx = np.flatnonzero(keep)
        new_keep = np.zeros_like(keep)
        new_keep[idx[keep_sub]] = True
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return A_g, t_g


def _match_level(
    ref_l: np.ndarray,
    cur_l: np.ndarray,
    A_g: np.ndarray,
    t_l: np.ndarray,
    *,
    grid: int,
    subset: int,
    search: int,
    min_zncc: float,
    mask_l: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Match a grid of points at one pyramid level and refit the affine map.

    The predicted position of a grid point ``p`` is ``A_g @ p + t_l``; an integer offset within
    ``+/- search`` is searched to maximise ZNCC, and the resulting correspondences are fitted.

    Returns
    -------
    A, t, score
        The refitted affine map and the median ZNCC of the matched points (``-inf`` if the fit
        could not be performed).
    """
    h, w = ref_l.shape
    r = subset // 2
    ng = max(3, grid)
    if w <= 2 * r + 2 or h <= 2 * r + 2:
        return A_g, t_l, -np.inf

    xs = np.linspace(r, w - 1 - r, ng)
    ys = np.linspace(r, h - 1 - r, ng)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()
    ix0 = np.rint(gx).astype(int)
    iy0 = np.rint(gy).astype(int)
    if mask_l is not None:
        inside = mask_l[iy0, ix0]
        if inside.sum() >= 3:
            gx, gy, ix0, iy0 = gx[inside], gy[inside], ix0[inside], iy0[inside]
    pts = np.stack([gx, gy], axis=1)

    # predicted positions
    pred = pts @ A_g.T + t_l
    px = np.rint(pred[:, 0]).astype(int)
    py = np.rint(pred[:, 1]).astype(int)

    off = np.arange(-r, r + 1)
    ox, oy = np.meshgrid(off, off)
    ox = ox.ravel()
    oy = oy.ravel()
    # integer reference grid positions (the reference subset is sampled exactly there)
    ix, iy = ix0, iy0
    rx = ix[:, None] + ox[None, :]
    ry = iy[:, None] + oy[None, :]
    if rx.min() < 0 or ry.min() < 0 or rx.max() >= w or ry.max() >= h:
        return A_g, t_l, -np.inf
    f_sub = ref_l[ry, rx]

    best = np.full(gx.size, -np.inf)
    best_dx = np.zeros(gx.size)
    best_dy = np.zeros(gx.size)
    for dy in range(-search, search + 1):
        cy = py + dy
        for dx in range(-search, search + 1):
            cx = px + dx
            ok = (cx >= r) & (cx < w - r) & (cy >= r) & (cy < h - r)
            if not ok.any():
                continue
            ccx = np.where(ok, cx, r)
            ccy = np.where(ok, cy, r)
            g_sub = cur_l[ccy[:, None] + oy[None, :], ccx[:, None] + ox[None, :]]
            z = np.nan_to_num(_zncc_subsets(f_sub, g_sub), nan=-np.inf)
            z = np.where(ok, z, -np.inf)
            better = z > best
            best = np.where(better, z, best)
            best_dx = np.where(better, dx, best_dx)
            best_dy = np.where(better, dy, best_dy)

    good = np.isfinite(best) & (best >= min_zncc)
    if good.sum() < 3:
        return A_g, t_l, -np.inf
    # matched subset centres are the rounded predictions plus the best integer offset
    dst = np.stack([px, py], axis=1).astype(np.float64) + np.stack([best_dx, best_dy], axis=1)
    src = np.stack([ix, iy], axis=1).astype(np.float64)
    A_new, t_new = _robust_affine_fit(src[good], dst[good])
    score = float(np.median(best[good]))
    return A_new, t_new, score


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def _rotate_like(image: np.ndarray, angle_rad: float, center_xy: np.ndarray) -> np.ndarray:
    """Return ``image`` rotated by *angle_rad* about *center_xy* (content rotation).

    ``out[x] = image[R (x - c) + c]``, i.e. a feature that was at ``X`` in the reference moves to
    ``R (X - c) + c``.  Used to undo a large rigid rotation before template matching.

    ``R`` and ``center_xy`` are in (x, y) order; ``ndimage.affine_transform`` needs (row=y,
    col=x) order, so the matrix and offset are converted with the swap ``P``.
    """
    if angle_rad == 0.0:
        return image
    R = _rotation_matrix(angle_rad)  # (x, y) order
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    M = P @ R @ P  # (row, col) order
    c = np.asarray(center_xy, dtype=float)
    offset = (c - R @ c)[::-1]  # (row, col) order
    return ndimage.affine_transform(
        image, M, offset=offset, order=1, mode="constant", cval=float(np.mean(image))
    )


def _match_hierarchy(
    reference: np.ndarray,
    current: np.ndarray,
    cfg: CorrelationConfig,
    angle_rad: float,
    level: int,
    subset: int,
    min_zncc: float,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the multi-scale match for one candidate rotation angle."""
    A_g = np.eye(2)
    t = np.zeros(2)
    score = -np.inf
    height, width = reference.shape
    center_full = np.array([width, height], dtype=float) / 2.0
    r = subset // 2
    for lev in range(level, -1, -1):
        factor = 2 ** lev
        ref_l = _downsample(reference, factor)
        cur_l = _downsample(current, factor)
        cur_l = _rotate_like(cur_l, angle_rad, center_full / factor)
        mask_l = None
        if mask is not None:
            m = _downsample(mask.astype(float), factor) > 0.5
            mask_l = ndimage.binary_erosion(m, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
            if not mask_l.any():
                mask_l = None
        # brute-force radius at the coarsest level, a small refinement radius in between, and
        # the user-configurable --search-radius at full resolution
        if lev == level:
            search = cfg.coarse_search
        elif lev == 0:
            search = max(2, int(cfg.search_radius))
        else:
            search = 2
        t_l = t / factor
        A_g, t_l, score = _match_level(
            ref_l, cur_l, A_g, t_l,
            grid=cfg.coarse_grid, subset=subset, search=search, min_zncc=min_zncc,
            mask_l=mask_l,
        )
        t = t_l * factor
    if angle_rad != 0.0:
        R = _rotation_matrix(angle_rad)
        A_g = R @ A_g
        t = R @ (t - center_full) + center_full
    return A_g, t, score


def estimate_global_affine(
    reference: np.ndarray,
    current: np.ndarray,
    cfg: CorrelationConfig,
    *,
    subset: int = 11,
    min_zncc: float = 0.25,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hierarchical multi-scale estimate of the global affine map reference -> current.

    Starting from the coarsest pyramid level (brute-force integer search,
    +/-``coarse_search`` px) the affine map is refined level by level with a small +/-2 px
    search.  The result is an initial guess good to roughly a pixel at full resolution, which is
    what IC-GN needs to avoid local minima after large motion.

    Because template matching decorrelates under large rigid rotation, a coarse search over
    candidate rotation angles (``coarse_angle_step`` up to ``coarse_angle_max``) is added when
    the un-rotated match scores poorly.
    """
    level = max(1, int(cfg.coarse_level))
    smallest = min(reference.shape)
    # adapt the level so that the coarsest image stays large enough for the subset
    while level > 1 and smallest / (2 ** level) < 48:
        level -= 1

    A_g, t, score = _match_hierarchy(
        reference, current, cfg, 0.0, level, subset, min_zncc, mask
    )
    if score >= cfg.coarse_good_score:
        logger.debug("Coarse affine (no rotation) score=%.3f A=%s t=%s", score, A_g.tolist(), t.tolist())
        return A_g, t

    step = float(cfg.coarse_angle_step)
    amax = float(cfg.coarse_angle_max)
    if step <= 0 or amax <= 0:
        return A_g, t

    best_score = score
    best = (A_g, t)
    angles = list(np.arange(-amax, amax + 0.5 * step, step))
    for deg in angles:
        if abs(deg) < 1e-9:
            continue
        A_c, t_c, sc = _match_hierarchy(
            reference, current, cfg, np.radians(deg), level, subset, min_zncc, mask
        )
        if sc > best_score:
            best_score = sc
            best = (A_c, t_c)
    logger.info(
        "Coarse multi-scale search: best score %.3f (no-rotation score %.3f) over %d candidate "
        "rotations.",
        best_score,
        score,
        len(angles),
    )
    return best


# --------------------------------------------------------------------------------------
# IC-GN core
# --------------------------------------------------------------------------------------
def _icgn_core(
    frame_interp: SplineInterpolator,
    f_norm: np.ndarray,
    df: np.ndarray,
    SD: np.ndarray,
    hessian_inv: np.ndarray,
    sssig: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    xi_x: np.ndarray,
    xi_y: np.ndarray,
    shape: tuple[int, int],
    cfg: CorrelationConfig,
    p0: np.ndarray,
) -> FrameResult:
    """Run IC-GN for the supplied points.  See :meth:`SubsetCorrelator.correlate`."""
    height, width = shape
    n = xs.size
    n_sub = xi_x.size
    p = np.array(p0, dtype=np.float64, copy=True)
    converged = np.zeros(n, dtype=bool)
    iterations = np.zeros(n, dtype=int)
    tol = cfg.convergence_tol
    tol_strain = cfg.convergence_tol

    def evaluate(pp: np.ndarray):
        Xt, Yt = warp_points(pp, xs, ys, xi_x, xi_y)
        inb = (
            (Xt >= 0.0) & (Xt <= width - 1.0) & (Yt >= 0.0) & (Yt <= height - 1.0)
        ).all(axis=1)
        g = frame_interp.values(Yt.ravel(), Xt.ravel()).reshape(n, n_sub)
        g_mean = g.mean(axis=1)
        g_c = g - g_mean[:, None]
        dg = np.sqrt(np.sum(g_c * g_c, axis=1))
        dg = np.where(dg > _EPS, dg, _EPS)
        return g_c / dg[:, None], inb

    for _ in range(cfg.max_iterations):
        g_norm, inb = evaluate(p)
        r = f_norm - g_norm
        b = np.einsum("nsk,ns->nk", SD, r) / df[:, None]
        dp = -np.einsum("nkl,nl->nk", hessian_inv, b)
        step_disp = np.abs(dp[:, :2]).max(axis=1)
        step_strain = np.abs(dp[:, 2:]).max(axis=1)
        converged |= inb & (step_disp < tol) & (step_strain < tol_strain)
        iterations[~converged] += 1
        dp[~inb] = 0.0
        p = compose_inverse_update(p, dp)
        if not np.any(~converged & inb):
            break

    g_norm, inb = evaluate(p)
    zncc = np.sum(f_norm * g_norm, axis=1)
    resid = f_norm - g_norm
    sigma_r = resid.std(axis=1)
    sigma_u = sigma_r * np.sqrt(np.abs(hessian_inv[:, 0, 0]))
    sigma_v = sigma_r * np.sqrt(np.abs(hessian_inv[:, 1, 1]))
    valid = (
        inb
        & converged
        & np.isfinite(zncc)
        & (zncc >= cfg.min_zncc)
        & (sssig >= cfg.min_sssig)
    )
    p_out = p.copy()
    p_out[~valid] = np.nan
    return FrameResult(
        p=p_out,
        zncc=np.where(valid, zncc, np.nan),
        sigma_u=np.where(valid, sigma_u, np.nan),
        sigma_v=np.where(valid, sigma_v, np.nan),
        residual=np.where(valid, sigma_r, np.nan),
        sssig=sssig.copy(),
        iterations=iterations,
        valid=valid,
    )


class SubsetCorrelator:
    """IC-GN correlator for a fixed reference image and grid.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference (undeformed) image.
    xs, ys : numpy.ndarray
        Grid point coordinates (column x, row y).
    cfg : CorrelationConfig
        Correlation parameters.
    """

    def __init__(
        self,
        reference: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        cfg: CorrelationConfig,
        mask: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.mask = mask
        self.xs = np.asarray(xs, dtype=np.float64)
        self.ys = np.asarray(ys, dtype=np.float64)
        self.n_points = self.xs.size
        self.image_shape = reference.shape
        self.height, self.width = reference.shape

        ref = gaussian_prefilter(reference, cfg.prefilter_sigma)
        self.reference = ref
        r = cfg.subset_size // 2
        off = np.arange(-r, r + 1, dtype=np.float64)
        xi_x, xi_y = np.meshgrid(off, off)
        self.xi_x = xi_x.ravel()
        self.xi_y = xi_y.ravel()
        self.n_sub = self.xi_x.size

        interp = SplineInterpolator(ref, order=cfg.ref_interp_order)
        Y = self.ys[:, None] + self.xi_y[None, :]
        X = self.xs[:, None] + self.xi_x[None, :]
        f = interp.values(Y.ravel(), X.ravel()).reshape(self.n_points, self.n_sub)
        gx, gy = interp.gradients(Y.ravel(), X.ravel())
        gx = gx.reshape(self.n_points, self.n_sub)
        gy = gy.reshape(self.n_points, self.n_sub)

        self.f = f
        self.f_mean = f.mean(axis=1)
        self.df = np.sqrt(np.sum((f - self.f_mean[:, None]) ** 2, axis=1))
        self.df = np.where(self.df > _EPS, self.df, _EPS)
        self.f_norm = (f - self.f_mean[:, None]) / self.df[:, None]

        SD = np.empty((self.n_points, self.n_sub, N_PARAMS))
        SD[:, :, 0] = gx
        SD[:, :, 1] = gy
        SD[:, :, 2] = gx * self.xi_x[None, :]
        SD[:, :, 3] = gx * self.xi_y[None, :]
        SD[:, :, 4] = gy * self.xi_x[None, :]
        SD[:, :, 5] = gy * self.xi_y[None, :]
        self.SD = SD
        self.sssig = np.sum(gx * gx + gy * gy, axis=1)

        H = np.einsum("nsk,nsl->nkl", SD, SD) / (self.df[:, None, None] ** 2)
        self.hessian = H
        self.hessian_inv = np.linalg.pinv(H)

        self._cur_interp: SplineInterpolator | None = None
        self._cur_frame: np.ndarray | None = None
        self._tree = cKDTree(np.stack([self.xs, self.ys], axis=1))

    # -- interpolator cache ------------------------------------------------------------
    def _interp_for(self, frame: np.ndarray) -> SplineInterpolator:
        """Return (and cache) the interpolator for *frame*.

        The cache is keyed on the **object identity** of the frame array.  The pipeline reuses
        the same array object for all the IC-GN passes of one frame (including the re-seed
        passes), and passes a distinct array for every new frame, so identity is both correct and
        immune to the classic "stale cache" bug of an integer key.
        """
        if self._cur_frame is frame and self._cur_interp is not None:
            return self._cur_interp
        img = gaussian_prefilter(frame, self.cfg.prefilter_sigma)
        interp = SplineInterpolator(img, order=self.cfg.interp_order)
        self._cur_frame = frame
        self._cur_interp = interp
        return interp

    # -- public API --------------------------------------------------------------------
    def correlate(
        self,
        frame: np.ndarray,
        p0: np.ndarray,
        *,
        indices: np.ndarray | None = None,
    ) -> FrameResult:
        """Run IC-GN for every grid point (or the given *indices*) against *frame*."""
        interp = self._interp_for(frame)
        if indices is None:
            return _icgn_core(
                interp, self.f_norm, self.df, self.SD, self.hessian_inv, self.sssig,
                self.xs, self.ys, self.xi_x, self.xi_y, self.image_shape, self.cfg, p0,
            )
        idx = np.asarray(indices)
        return _icgn_core(
            interp,
            self.f_norm[idx],
            self.df[idx],
            self.SD[idx],
            self.hessian_inv[idx],
            self.sssig[idx],
            self.xs[idx],
            self.ys[idx],
            self.xi_x,
            self.xi_y,
            self.image_shape,
            self.cfg,
            p0,
        )

    def correlate_robust(
        self,
        frame: np.ndarray,
        p0: np.ndarray,
        *,
        reseed_rounds: int = 3,
        outlier_px: float = 0.75,
        reseed_zncc: float = 0.9,
        neighbours: int = 9,
    ) -> FrameResult:
        """Correlate with reliability-guided re-seeding of failed points.

        A point is *suspicious* if it is invalid, if its ZNCC is below ``reseed_zncc``, or if it
        disagrees with the local median of its neighbours by more than
        ``max(outlier_px, 4 * local_spread)`` -- the second term makes the test scale-aware so a
        strongly graded displacement field is not mistaken for a field of outliers.  Suspicious
        points are restarted from the neighbour median (the RG-DIC idea).  Points that remain
        poor are left invalid (``NaN``).
        """
        res = self.correlate(frame, p0)
        for _ in range(reseed_rounds):
            med, spread = self._neighbour_stats(res.p, res.valid, neighbours)
            dev = np.hypot(res.u - med[:, 0], res.v - med[:, 1])
            thresh = np.maximum(outlier_px, 4.0 * spread)
            low_quality = np.nan_to_num(res.zncc, nan=-1.0) < reseed_zncc
            suspicious = (~res.valid) | low_quality | (np.isfinite(dev) & (dev > thresh))
            has_nb = np.isfinite(med[:, 0])
            targets = suspicious & has_nb
            if not targets.any():
                break
            idx = np.flatnonzero(targets)
            p_seed = med[idx].copy()
            sub = self.correlate(frame, p_seed, indices=idx)
            # keep the better (higher ZNCC) of the old and new solutions
            old_z = np.nan_to_num(res.zncc[idx], nan=-np.inf)
            new_z = np.nan_to_num(sub.zncc, nan=-np.inf)
            take = new_z > old_z
            take_idx = idx[take]
            if take_idx.size == 0:
                break
            res.p[take_idx] = sub.p[take]
            res.zncc[take_idx] = sub.zncc[take]
            res.sigma_u[take_idx] = sub.sigma_u[take]
            res.sigma_v[take_idx] = sub.sigma_v[take]
            res.residual[take_idx] = sub.residual[take]
            res.iterations[take_idx] = sub.iterations[take]
            res.valid[take_idx] = sub.valid[take]
        return res

    def _neighbour_stats(
        self, p: np.ndarray, valid: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Local median and robust spread of the warp parameters over *k* nearest neighbours.

        Invalid points are ``NaN`` in ``p`` and are simply ignored by the NaN-aware median, so a
        point with no valid neighbour returns ``NaN`` and is never re-seeded.
        """
        kk = int(min(k, max(3, int(valid.sum()))))
        _, nbr = self._tree.query(np.stack([self.xs, self.ys], axis=1), k=kk)
        nbr = np.atleast_2d(nbr)
        vals = p[nbr]  # (N, kk, 6), NaN where the neighbour is invalid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(vals, axis=1)
            spread_u = 1.4826 * np.nanmedian(np.abs(vals[:, :, 0] - med[:, 0:1]), axis=1)
            spread_v = 1.4826 * np.nanmedian(np.abs(vals[:, :, 1] - med[:, 1:2]), axis=1)
        spread = np.fmax(np.nan_to_num(spread_u, nan=np.inf), np.nan_to_num(spread_v, nan=np.inf))
        return med, spread

    # -- initial guess -----------------------------------------------------------------
    def initial_guess(
        self,
        frame: np.ndarray,
        p_prev: np.ndarray | None,
        p_prev2: np.ndarray | None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
        """Return the initial guess for *frame* and any coarse affine used.

        Strategy (``cfg.init_strategy``):

        * ``temporal`` - linear extrapolation of the previous frames' parameters.
        * ``coarse``   - hierarchical multi-scale affine estimate.
        * ``auto``     - temporal, with a coarse fallback when the median ZNCC is poor.
        """
        strategy = self.cfg.init_strategy
        interp = self._interp_for(frame)

        def _median_zncc(pp: np.ndarray) -> float:
            g_norm, inb = self._eval_norm(pp, interp)
            z = np.sum(self.f_norm * g_norm, axis=1)
            z = np.where(inb, z, np.nan)
            return float(np.nanmedian(z)) if np.isfinite(z).any() else -1.0

        if p_prev is None:
            if strategy in ("coarse", "auto"):
                A_g, t_g = estimate_global_affine(
                    self.reference, frame, self.cfg, mask=self.mask
                )
                return affine_from_global(A_g, t_g, self.xs, self.ys), (A_g, t_g)
            return np.zeros((self.n_points, N_PARAMS)), None

        p0 = extrapolate_params(p_prev2, p_prev)
        if strategy == "temporal":
            return p0, None

        med = _median_zncc(p0)
        if strategy == "auto" and med >= self.cfg.init_fallback_zncc:
            return p0, None

        logger.info(
            "Temporal guess median ZNCC %.3f < %.2f; running coarse multi-scale search.",
            med,
            self.cfg.init_fallback_zncc,
        )
        A_g, t_g = estimate_global_affine(self.reference, frame, self.cfg, mask=self.mask)
        p_coarse = affine_from_global(A_g, t_g, self.xs, self.ys)
        # never accept a coarse guess that is worse than the temporal one
        if strategy == "auto" and _median_zncc(p_coarse) < med:
            logger.info("Coarse guess is worse than the temporal guess; keeping the temporal one.")
            return p0, None
        return p_coarse, (A_g, t_g)

    def _eval_norm(self, p: np.ndarray, interp: SplineInterpolator):
        Xt, Yt = warp_points(p, self.xs, self.ys, self.xi_x, self.xi_y)
        inb = (
            (Xt >= 0.0)
            & (Xt <= self.width - 1.0)
            & (Yt >= 0.0)
            & (Yt <= self.height - 1.0)
        ).all(axis=1)
        g = interp.values(Yt.ravel(), Xt.ravel()).reshape(self.n_points, self.n_sub)
        g_c = g - g.mean(axis=1)[:, None]
        dg = np.sqrt(np.sum(g_c * g_c, axis=1))
        dg = np.where(dg > _EPS, dg, _EPS)
        return g_c / dg[:, None], inb
