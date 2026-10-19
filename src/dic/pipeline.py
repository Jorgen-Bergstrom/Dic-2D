"""End-to-end pipeline: reference -> correlation -> strain -> uncertainty.

The public entry point is :func:`run_pipeline`, which takes a directory of images and a
:class:`~dic.config.RunConfig` and returns a :class:`RunResult` holding every field, quality
measure and timing.  Writing files is delegated to :mod:`dic.outputs` so the pipeline stays
importable and testable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import RunConfig
from .correlation import FrameResult, SubsetCorrelator, make_grid
from .io import load_sequence
from .mask import build_mask
from .strain import StrainResult, compute_strains
from .uncertainty import UncertaintyReport, compute_noise_floor, image_noise_estimate

logger = logging.getLogger(__name__)


@dataclass
class FrameFields:
    """All fields and quality measures for one frame."""

    index: int
    name: str
    time: float
    t_load: float
    ux: np.ndarray
    uy: np.ndarray
    valid: np.ndarray
    zncc: np.ndarray
    sigma_u: np.ndarray
    sigma_v: np.ndarray
    residual: np.ndarray
    strain: StrainResult
    iterations: np.ndarray
    global_affine: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def coverage(self) -> float:
        return float(np.mean(self.valid))

    @property
    def mean_sigma_u(self) -> float:
        vals = self.sigma_u[self.valid]
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

    @property
    def mean_sigma_v(self) -> float:
        vals = self.sigma_v[self.valid]
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


@dataclass
class RunResult:
    """Everything a run produces, in memory."""

    cfg: RunConfig
    image_paths: list[Path]
    image_hashes: list[str]
    shape: tuple[int, int]
    xs: np.ndarray
    ys: np.ndarray
    ny: int
    nx: int
    dx: float
    dy: float
    mask: np.ndarray
    mask_source: str
    frames: list[FrameFields] = field(default_factory=list)
    uncertainty: UncertaintyReport | None = None
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def _to_grid(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Scatter a flat point array onto its regular ``(ny, nx)`` grid (NaN where absent)."""
    ux = np.unique(xs)
    uy = np.unique(ys)
    grid = np.full((uy.size, ux.size), np.nan)
    ix = np.searchsorted(ux, xs)
    iy = np.searchsorted(uy, ys)
    grid[iy, ix] = values
    if grid.shape != (ny, nx):
        out = np.full((ny, nx), np.nan)
        out[: grid.shape[0], : grid.shape[1]] = grid
        return out
    return grid


def _frame_from_result(
    index: int,
    name: str,
    res: FrameResult,
    cfg: RunConfig,
    xs: np.ndarray,
    ys: np.ndarray,
    ny: int,
    nx: int,
    dx: float,
    dy: float,
    time: float,
    t_load: float,
) -> FrameFields:
    ux_g = _to_grid(res.u, xs, ys, ny, nx)
    uy_g = _to_grid(res.v, xs, ys, ny, nx)
    valid_g = _to_grid(res.valid.astype(float), xs, ys, ny, nx) > 0.5
    strain = compute_strains(
        ux_g, uy_g, valid_g, dx, dy, cfg.strain, subset_size=cfg.correlation.subset_size
    )
    return FrameFields(
        index=index,
        name=name,
        time=time,
        t_load=t_load,
        ux=res.u,
        uy=res.v,
        valid=res.valid,
        zncc=res.zncc,
        sigma_u=res.sigma_u,
        sigma_v=res.sigma_v,
        residual=res.residual,
        strain=strain,
        iterations=res.iterations,
        global_affine=res.global_affine,
    )


def run_pipeline(
    input_dir: str | Path,
    cfg: RunConfig,
    *,
    user_mask_path: str | Path | None = None,
    user_mask_index: int = 0,
    max_images: int | None = None,
    progress: callable | None = None,
) -> RunResult:
    """Run the full DIC analysis.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Directory of images (read-only).
    cfg : RunConfig
        Complete run configuration.
    user_mask_path : str or pathlib.Path, optional
        User-supplied mask file or directory.
    user_mask_index : int
        Which mask to use when *user_mask_path* is a directory.
    max_images : int, optional
        Limit the number of frames processed (testing).
    progress : callable, optional
        Called as ``progress(index, total, message)``.

    Returns
    -------
    RunResult
    """
    t_start = time.time()
    seq = load_sequence(input_dir, max_images=max_images)
    n = len(seq)
    logger.debug("%d frames to process", n)

    reference = seq.images[0]
    mask, mask_source = build_mask(
        reference,
        cfg.mask,
        cfg.correlation.subset_size,
        user_mask_path=user_mask_path,
        user_mask_index=user_mask_index,
    )
    xs, ys = make_grid(seq.shape, mask, cfg.correlation.subset_size, cfg.correlation.step)
    if xs.size < 9:
        raise RuntimeError(
            f"Only {xs.size} correlation points fit inside the ROI. Enlarge the ROI, reduce "
            f"--subset, or reduce --step."
        )
    nx = int(np.unique(xs).size)
    ny = int(np.unique(ys).size)
    dx = float(np.median(np.diff(np.unique(xs)))) if nx > 1 else 1.0
    dy = float(np.median(np.diff(np.unique(ys)))) if ny > 1 else 1.0
    logger.info(
        "Correlation grid: %d x %d = %d points (step %.1f x %.1f px), ROI coverage %.1f%%",
        nx,
        ny,
        xs.size,
        dx,
        dy,
        100.0 * mask.mean(),
    )

    result = RunResult(
        cfg=cfg,
        image_paths=seq.paths,
        image_hashes=seq.hashes,
        shape=seq.shape,
        xs=xs,
        ys=ys,
        ny=ny,
        nx=nx,
        dx=dx,
        dy=dy,
        mask=mask,
        mask_source=mask_source,
    )

    t_corr_start = time.time()
    if cfg.correlation.reference == "incremental":
        frames = _run_incremental(seq, cfg, xs, ys, ny, nx, dx, dy, mask, progress)
    else:
        frames = _run_direct(seq, cfg, xs, ys, ny, nx, dx, dy, mask, progress)
    result.frames = frames
    result.timings["correlation_s"] = time.time() - t_corr_start

    # ---- coverage gate (R16) ---------------------------------------------------------
    for f in frames:
        if f.coverage < cfg.min_coverage:
            msg = (
                f"Frame {f.index} ({f.name}): only {100 * f.coverage:.1f}% of points are valid, "
                f"below --min-coverage {100 * cfg.min_coverage:.0f}%."
            )
            logger.error(msg)
            result.warnings.append(msg)

    # ---- uncertainty ----------------------------------------------------------------
    if cfg.uncertainty.per_point or cfg.uncertainty.noise_floor:
        t_unc = time.time()
        corr = SubsetCorrelator(reference, xs, ys, cfg.correlation, mask=mask)
        # skip the reference frame itself (its residual is identically zero by construction)
        residuals = [f.residual for f in frames[1:]]
        valids = [f.valid for f in frames[1:]]
        noise_sigma, noise_method = image_noise_estimate(
            reference, corr, residuals, valids, cfg.uncertainty.image_noise_sigma
        )
        logger.info("Estimated image noise: %.3f grey levels (%s)", noise_sigma, noise_method)
        nf_u = nf_v = nf_s = float("nan")
        nf_points = 0
        if cfg.uncertainty.noise_floor and np.isfinite(noise_sigma):
            nf_u, nf_v, nf_s, nf_points = compute_noise_floor(
                reference,
                corr,
                cfg.correlation,
                cfg.strain,
                noise_sigma=noise_sigma,
                seed=cfg.seed,
                realisations=cfg.uncertainty.noise_floor_realisations,
            )
        su = [f.mean_sigma_u for f in frames]
        sv = [f.mean_sigma_v for f in frames]
        result.uncertainty = UncertaintyReport(
            image_noise_sigma=float(noise_sigma),
            image_noise_method=noise_method,
            noise_floor_u_rms_px=float(nf_u),
            noise_floor_v_rms_px=float(nf_v),
            noise_floor_strain_rms=float(nf_s),
            noise_floor_points=int(nf_points),
            mean_sigma_u_px=float(np.nanmean(su)) if su else float("nan"),
            mean_sigma_v_px=float(np.nanmean(sv)) if sv else float("nan"),
        )
        result.timings["uncertainty_s"] = time.time() - t_unc

    result.timings["total_s"] = time.time() - t_start
    return result


def _run_direct(seq, cfg, xs, ys, ny, nx, dx, dy, mask, progress):
    """Reference = the first image for every frame (default)."""
    n = len(seq)
    corr = SubsetCorrelator(seq.images[0], xs, ys, cfg.correlation, mask=mask)
    frames: list[FrameFields] = []
    p_prev: np.ndarray | None = None
    p_prev2: np.ndarray | None = None
    t_prev = time.time()
    for i in range(n):
        if progress:
            elapsed = time.time() - t_prev
            eta = elapsed * (n - i - 1)
            progress(i, n, f"frame {i + 1}/{n} (ETA ~{eta:.0f}s)")
        t0 = time.time()
        if i == 0:
            p0 = np.zeros((xs.size, 6))
            res = corr.correlate(seq.images[0], p0)
        else:
            p0, ga = corr.initial_guess(seq.images[i], p_prev, p_prev2)
            res = corr.correlate_robust(seq.images[i], p0)
            res.global_affine = ga
        ff = _frame_from_result(
            i,
            seq.paths[i].name,
            res,
            cfg,
            xs,
            ys,
            ny,
            nx,
            dx,
            dy,
            time=i * cfg.dt,
            t_load=i / max(n - 1, 1),
        )
        frames.append(ff)
        p_prev2, p_prev = p_prev, res.p
        logger.debug("frame %d done in %.2fs, coverage %.3f", i, time.time() - t0, ff.coverage)
        t_prev = time.time()
    return frames


def _run_incremental(seq, cfg, xs, ys, ny, nx, dx, dy, mask, progress):
    """Reference updating with accumulation (for strongly decorrelating sequences).

    The reference is promoted to the previous frame every ``update_interval`` frames (or sooner
    if the median ZNCC collapses).  Total displacement and the deformation gradient are
    accumulated in the original reference frame, so the reported fields stay in the undeformed
    configuration even though the subsets move.
    """
    n = len(seq)
    n_pts = xs.size
    interval = max(1, int(getattr(cfg.correlation, "update_interval", 1)))
    d_total = np.zeros((n_pts, 2))
    F_total = np.tile(np.eye(2), (n_pts, 1, 1))
    ref_idx = 0
    last_zncc = 1.0
    frames: list[FrameFields] = []
    t_prev = time.time()

    for i in range(n):
        if progress:
            elapsed = time.time() - t_prev
            eta = elapsed * (n - i - 1)
            progress(i, n, f"frame {i + 1}/{n} (ETA ~{eta:.0f}s)")
        if i == 0:
            strain = compute_strains(
                _to_grid(np.zeros(n_pts), xs, ys, ny, nx),
                _to_grid(np.zeros(n_pts), xs, ys, ny, nx),
                _to_grid(np.ones(n_pts), xs, ys, ny, nx) > 0.5,
                dx,
                dy,
                cfg.strain,
                subset_size=cfg.correlation.subset_size,
            )
            frames.append(
                FrameFields(
                    index=0,
                    name=seq.paths[0].name,
                    time=0.0,
                    t_load=0.0,
                    ux=np.zeros(n_pts),
                    uy=np.zeros(n_pts),
                    valid=np.ones(n_pts, dtype=bool),
                    zncc=np.ones(n_pts),
                    sigma_u=np.full(n_pts, np.nan),
                    sigma_v=np.full(n_pts, np.nan),
                    residual=np.zeros(n_pts),
                    strain=strain,
                    iterations=np.zeros(n_pts, dtype=int),
                )
            )
            t_prev = time.time()
            continue

        if (i - ref_idx) >= interval or (last_zncc < cfg.correlation.update_zncc and i - 1 > ref_idx):
            ref_idx = i - 1

        x_ref = xs + d_total[:, 0]
        y_ref = ys + d_total[:, 1]
        r = cfg.correlation.subset_size // 2
        h, w = seq.shape
        ref_ok = (
            (x_ref >= r) & (x_ref <= w - 1 - r) & (y_ref >= r) & (y_ref <= h - 1 - r)
        )

        corr = SubsetCorrelator(seq.images[ref_idx], x_ref, y_ref, cfg.correlation, mask=mask)
        p0, ga = corr.initial_guess(seq.images[i], None, None)
        res = corr.correlate_robust(seq.images[i], p0)
        res.global_affine = ga

        d_inc = res.p[:, :2]
        F_inc = np.tile(np.eye(2), (n_pts, 1, 1))
        F_inc[:, 0, 0] = 1.0 + res.p[:, 2]
        F_inc[:, 0, 1] = res.p[:, 3]
        F_inc[:, 1, 0] = res.p[:, 4]
        F_inc[:, 1, 1] = 1.0 + res.p[:, 5]

        valid = res.valid & ref_ok
        d_total = np.where(valid[:, None], d_total + np.nan_to_num(d_inc), d_total)
        F_total = np.where(valid[:, None, None], F_inc @ F_total, F_total)

        # Build a FrameFields whose affine parameters encode the total deformation gradient
        p_total = np.zeros((n_pts, 6))
        p_total[:, 0] = d_total[:, 0]
        p_total[:, 1] = d_total[:, 1]
        p_total[:, 2] = F_total[:, 0, 0] - 1.0
        p_total[:, 3] = F_total[:, 0, 1]
        p_total[:, 4] = F_total[:, 1, 0]
        p_total[:, 5] = F_total[:, 1, 1] - 1.0
        merged = FrameResult(
            p=np.where(valid[:, None], p_total, np.nan),
            zncc=res.zncc,
            sigma_u=res.sigma_u,
            sigma_v=res.sigma_v,
            residual=res.residual,
            sssig=res.sssig,
            iterations=res.iterations,
            valid=valid,
            global_affine=ga,
        )
        last_zncc = float(np.nanmedian(res.zncc[valid])) if valid.any() else 0.0
        ff = _frame_from_result(
            i, seq.paths[i].name, merged, cfg, xs, ys, ny, nx, dx, dy,
            time=i * cfg.dt, t_load=i / max(n - 1, 1),
        )
        frames.append(ff)
        t_prev = time.time()
    return frames
