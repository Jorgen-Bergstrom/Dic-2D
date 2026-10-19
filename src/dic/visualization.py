"""Visualization: per-frame overlays, animations, curves and profiles (R10, R11, R14).

Design choices (justified in ``DECISIONS.md``):

* By default every field is drawn as a filled contour map **superimposed on the current
  (deformed) image**, with the field placed at the material positions ``X + u``.  The specimen
  therefore visibly deforms from frame to frame and the field follows the material -- which is
  what makes the per-frame figures and the animations informative.  ``--overlay reference``
  switches to the classic fixed-reference presentation (the undeformed image, field at the
  reference positions), which is useful when you want to compare fields in a common frame.
* The displacement figure shows ``u_x``, ``u_y`` and ``|u|``; the third panel carries a subsampled
  displacement-vector (quiver) overlay.
* The strain figure shows the engineering components (top row) next to the Green-Lagrange
  components (bottom row) so the small-strain-versus-finite-strain distinction is visible at a
  glance.
* A growing-field animation, mean/max-versus-time curves, line profiles and an uncertainty map
  are added because a mechanical-testing engineer uses them to check a test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .config import OutputConfig
from .pipeline import FrameFields, RunResult

logger = logging.getLogger(__name__)


def _field_grid(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Scatter flat values onto the regular ``(ny, nx)`` grid."""
    ux = np.unique(xs)
    uy = np.unique(ys)
    g = np.full((uy.size, ux.size), np.nan)
    g[np.searchsorted(uy, ys), np.searchsorted(ux, xs)] = values
    return g


def _overlay(ax, base, X, Y, field, title, unit, levels, alpha):
    ax.imshow(base, cmap="gray", origin="upper")
    finite = np.isfinite(field)
    if finite.any():
        lo, hi = np.nanpercentile(field, [1, 99])
        if not np.isfinite(lo) or lo == hi:
            lo, hi = np.nanmin(field), np.nanmax(field) + 1e-9
        lv = np.linspace(lo, hi, levels)
        cs = ax.contourf(X, Y, field, levels=lv, cmap="turbo", alpha=alpha, extend="both")
        cb = ax.figure.colorbar(cs, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(unit)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.5, base.shape[1] - 0.5)
    ax.set_ylim(base.shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _grid_interp(result: RunResult, values: np.ndarray):
    """RegularGridInterpolator over the correlation grid for one flat field."""
    from scipy.interpolate import RegularGridInterpolator

    nx = np.unique(result.xs)
    ny = np.unique(result.ys)
    grid = _field_grid(values, result.xs, result.ys, result.ny, result.nx)
    return RegularGridInterpolator((ny, nx), grid, bounds_error=False, fill_value=np.nan)


def _roi_boundary(result: RunResult, frame: FrameFields, deformed: bool):
    """Return the ROI boundary as (x, y) arrays, optionally displaced to the current frame.

    The mask lives in the reference frame, so in the deformed presentation its boundary is
    warped by the measured displacement field.  This keeps requirement R10 ("mask shown") true
    in both presentations.
    """
    from scipy import ndimage

    mask = result.mask
    if mask.all():
        # a full-frame ROI: show the image border, which is the boundary that actually moves
        edge = np.zeros_like(mask)
        edge[0, :] = edge[-1, :] = True
        edge[:, 0] = edge[:, -1] = True
    else:
        edge = mask ^ ndimage.binary_erosion(mask, border_value=0)
    ys, xs = np.nonzero(edge)
    if not deformed or xs.size == 0:
        return xs.astype(float), ys.astype(float)
    iu = _grid_interp(result, frame.ux)
    iv = _grid_interp(result, frame.uy)
    pts = np.stack([ys, xs], axis=1)
    with np.errstate(invalid="ignore"):
        return xs + iu(pts), ys + iv(pts)


def _view(result: RunResult, frame: FrameFields, cfg: OutputConfig):
    """Base image and plotting coordinates for one frame.

    ``cfg.overlay_frame == "deformed"`` (default) draws the *current* image and places the field
    at the material (deformed) positions ``X + u``, so the specimen visibly deforms and the
    field follows the material.  ``"reference"`` draws the undeformed reference image with the
    field at the reference positions, the classic fixed-reference presentation.

    Returns
    -------
    base, X, Y, deformed : the base image, the plotting coordinates and whether they are deformed.
    """
    X, Y = np.meshgrid(np.unique(result.xs), np.unique(result.ys))
    if getattr(cfg, "overlay_frame", "deformed") == "reference":
        return _reference_image(result), X, Y, False
    ux = _field_grid(frame.ux, result.xs, result.ys, result.ny, result.nx)
    uy = _field_grid(frame.uy, result.xs, result.ys, result.ny, result.nx)
    return _frame_image(result, frame.index), X + ux, Y + uy, True


def _draw_roi(ax, result: RunResult, frame: FrameFields, deformed: bool) -> None:
    try:
        x, y = _roi_boundary(result, frame, deformed)
        if x.size:
            ax.plot(x, y, linestyle="none", marker=".", markersize=1.0, color="w", alpha=0.7)
    except Exception:  # pragma: no cover - cosmetic only
        pass


def plot_displacement(
    result: RunResult, frame: FrameFields, path: Path, cfg: OutputConfig
) -> Path:
    """Displacement overlay for one frame: ux, uy and |u| + quiver."""
    base, X, Y, deformed = _view(result, frame, cfg)
    ux = _field_grid(frame.ux, result.xs, result.ys, result.ny, result.nx)
    uy = _field_grid(frame.uy, result.xs, result.ys, result.ny, result.nx)
    mag = np.hypot(ux, uy)
    unit = f"displacement [{result.cfg.unit}]"
    scale = result.cfg.scale

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    _overlay(axes[0], base, X, Y, ux * scale, f"$u_x$  (frame {frame.index + 1})", unit,
             cfg.contour_levels, cfg.overlay_alpha)
    _overlay(axes[1], base, X, Y, uy * scale, f"$u_y$  (frame {frame.index + 1})", unit,
             cfg.contour_levels, cfg.overlay_alpha)
    _overlay(axes[2], base, X, Y, mag * scale, f"$|u|$ + vectors (frame {frame.index + 1})",
             unit, cfg.contour_levels, cfg.overlay_alpha)
    # quiver, subsampled to keep it legible; arrows start at the material points
    step = max(1, int(np.ceil(result.nx / 30)))
    sx = slice(None, None, step)
    qx = (ux * scale)[sx, sx]
    qy = (uy * scale)[sx, sx]
    if np.isfinite(qx).any() and np.nanmax(np.hypot(qx, qy)) > 0:
        axes[2].quiver(
            X[sx, sx], Y[sx, sx], qx, qy,
            color="k", alpha=0.7, angles="xy", width=0.0025,
        )
    for ax in axes:
        _draw_roi(ax, result, frame, deformed)
    view = "field at deformed positions" if deformed else "field at reference positions"
    fig.suptitle(
        f"2D DIC displacement - {frame.name}  (t={frame.time:.4g} s, load {frame.t_load:.3f}), "
        f"coverage {100 * frame.coverage:.1f}%, {view}",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi)
    plt.close(fig)
    return path


def plot_strain(result: RunResult, frame: FrameFields, path: Path, cfg: OutputConfig) -> Path:
    """Strain overlay for one frame: engineering (top) and Green-Lagrange (bottom)."""
    base, X, Y, deformed = _view(result, frame, cfg)
    s = frame.strain
    panels = [
        (s["exx"], r"$\varepsilon_{xx}$ (engineering)", "strain [-]"),
        (s["eyy"], r"$\varepsilon_{yy}$ (engineering)", "strain [-]"),
        (s["gamma_xy"], r"$\gamma_{xy}$ (engineering)", "strain [-]"),
        (s["Exx"], r"$E_{xx}$ (Green-Lagrange)", "strain [-]"),
        (s["Eyy"], r"$E_{yy}$ (Green-Lagrange)", "strain [-]"),
        (s["Egamma_xy"], r"$\Gamma_{xy}$ (Green-Lagrange)", "strain [-]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    for ax, (field, title, unit) in zip(axes.ravel(), panels, strict=False):
        _overlay(ax, base, X, Y, field, title, unit, cfg.contour_levels, cfg.overlay_alpha)
        _draw_roi(ax, result, frame, deformed)
    eng_peak = np.nanmax(np.abs(np.stack([s["exx"], s["eyy"], s["gamma_xy"]])))
    note = (
        f" | engineering strain outside small-strain validity ({eng_peak:.3f} > "
        f"{result.cfg.strain.engineering_validity_limit}) - read the Green-Lagrange row"
        if np.isfinite(eng_peak) and eng_peak > result.cfg.strain.engineering_validity_limit
        else ""
    )
    fig.suptitle(
        f"2D DIC strain - {frame.name}  (load {frame.t_load:.3f}), VSG "
        f"{s.gauge_length_px:.0f} px{note}",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi)
    plt.close(fig)
    return path


def _reference_image(result: RunResult) -> np.ndarray:
    """Load the reference (first) image for the overlay base (cached on the result object)."""
    cached = getattr(result, "_ref_image", None)
    if cached is not None:
        return cached
    img = _read_image(result.image_paths[0])
    result._ref_image = img
    return img


def _read_image(path) -> np.ndarray:
    """Read one image file as float64 grayscale."""
    from PIL import Image

    from .io import to_grayscale

    with Image.open(path) as im:
        return to_grayscale(np.asarray(im))


def _frame_image(result: RunResult, index: int, max_cache: int = 2) -> np.ndarray:
    """Load the image of frame *index*, with a tiny cache.

    The cache is deliberately small: a full 1024² float64 image is 8 MB, so caching every frame
    of a long sequence would waste hundreds of megabytes.
    """
    cache = getattr(result, "_frame_cache", None)
    if cache is None:
        cache = {}
        result._frame_cache = cache
    if index in cache:
        return cache[index]
    img = _read_image(result.image_paths[index])
    cache[index] = img
    while len(cache) > max(1, max_cache):
        cache.pop(next(iter(cache)))
    return img


# --------------------------------------------------------------------------------------
# Extra visualizations (R14)
# --------------------------------------------------------------------------------------
def plot_curves(result: RunResult, path: Path, cfg: OutputConfig) -> Path:
    """Mean/max displacement and strain versus time (with the uncertainty band)."""
    t = np.array([f.time for f in result.frames])
    t_load = np.array([f.t_load for f in result.frames])
    u_mean, v_mean, u_max, umag_max, cov = [], [], [], [], []
    exx, eyy, eq = [], [], []
    for f in result.frames:
        m = f.valid
        u = f.ux[m]
        v = f.uy[m]
        u_mean.append(np.mean(u) if m.any() else np.nan)
        v_mean.append(np.mean(v) if m.any() else np.nan)
        u_max.append(np.max(np.abs(u)) if m.any() else np.nan)
        umag_max.append(np.max(np.hypot(u, v)) if m.any() else np.nan)
        cov.append(f.coverage)
        exx.append(np.nanmean(f.strain["exx"]))
        eyy.append(np.nanmean(f.strain["eyy"]))
        eq.append(np.nanmean(f.strain["eps_eq"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(t, u_mean, "-o", ms=3, label=r"mean $u_x$")
    axes[0, 0].plot(t, v_mean, "-s", ms=3, label=r"mean $u_y$")
    axes[0, 0].plot(t, u_max, "--", label=r"max $|u_x|$")
    axes[0, 0].plot(t, umag_max, ":", label=r"max $|u|$")
    axes[0, 0].set_xlabel(f"time [{ 's' if result.cfg.dt else 'frame' }]")
    axes[0, 0].set_ylabel(f"displacement [{result.cfg.unit}]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Displacement versus time")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, np.array(exx) * 100, "-o", ms=3, label=r"mean $\varepsilon_{xx}$")
    axes[0, 1].plot(t, np.array(eyy) * 100, "-s", ms=3, label=r"mean $\varepsilon_{yy}$")
    axes[0, 1].plot(t, np.array(eq) * 100, "-^", ms=3, label=r"mean $\varepsilon_{eq}$")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("strain [%]")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Mean strain versus time")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t_load, np.asarray(cov) * 100, "-o", ms=3, color="tab:green")
    axes[1, 0].set_xlabel("load parameter t")
    axes[1, 0].set_ylabel("valid coverage [%]")
    axes[1, 0].set_title("Valid-point coverage")
    axes[1, 0].grid(alpha=0.3)

    if result.uncertainty is not None:
        su = [f.mean_sigma_u for f in result.frames]
        sv = [f.mean_sigma_v for f in result.frames]
        axes[1, 1].plot(t, su, "-o", ms=3, label=r"mean $\sigma_{u}$")
        axes[1, 1].plot(t, sv, "-s", ms=3, label=r"mean $\sigma_{v}$")
        nf = result.uncertainty.noise_floor_u_rms_px
        if np.isfinite(nf):
            axes[1, 1].axhline(nf, color="k", ls="--", label="measured noise floor")
        axes[1, 1].set_xlabel("time")
        axes[1, 1].set_ylabel("per-point uncertainty [px]")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_title("Displacement uncertainty")
        axes[1, 1].grid(alpha=0.3)
    fig.suptitle("2D DIC time series", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi)
    plt.close(fig)
    return path


def plot_profiles(result: RunResult, path: Path, cfg: OutputConfig) -> Path:
    """Horizontal and vertical line profiles through the last frame."""
    f = result.frames[-1]
    ref = _reference_image(result)
    h, w = ref.shape
    # central row and column of the grid
    nx_vals = np.unique(result.xs)
    ny_vals = np.unique(result.ys)
    jmid = result.nx // 2
    imid = result.ny // 2
    ux = _field_grid(f.ux, result.xs, result.ys, result.ny, result.nx)
    uy = _field_grid(f.uy, result.xs, result.ys, result.ny, result.nx)
    exx = f.strain["exx"]
    eyy = f.strain["eyy"]
    scale = result.cfg.scale

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(nx_vals, ux[imid] * scale, "-o", ms=3, label=r"$u_x$")
    axes[0, 0].plot(nx_vals, uy[imid] * scale, "-s", ms=3, label=r"$u_y$")
    axes[0, 0].set_title(f"Horizontal profile at y={ny_vals[imid]:.0f} px (frame {f.index + 1})")
    axes[0, 0].set_xlabel("x [px]")
    axes[0, 0].set_ylabel(f"displacement [{result.cfg.unit}]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ny_vals, ux[:, jmid] * scale, "-o", ms=3, label=r"$u_x$")
    axes[0, 1].plot(ny_vals, uy[:, jmid] * scale, "-s", ms=3, label=r"$u_y$")
    axes[0, 1].set_title(f"Vertical profile at x={nx_vals[jmid]:.0f} px (frame {f.index + 1})")
    axes[0, 1].set_xlabel("y [px]")
    axes[0, 1].set_ylabel(f"displacement [{result.cfg.unit}]")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(nx_vals, exx[imid] * 100, "-o", ms=3, label=r"$\varepsilon_{xx}$")
    axes[1, 0].plot(nx_vals, eyy[imid] * 100, "-s", ms=3, label=r"$\varepsilon_{yy}$")
    axes[1, 0].set_xlabel("x [px]")
    axes[1, 0].set_ylabel("strain [%]")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(ny_vals, exx[:, jmid] * 100, "-o", ms=3, label=r"$\varepsilon_{xx}$")
    axes[1, 1].plot(ny_vals, eyy[:, jmid] * 100, "-s", ms=3, label=r"$\varepsilon_{yy}$")
    axes[1, 1].set_xlabel("y [px]")
    axes[1, 1].set_ylabel("strain [%]")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle("2D DIC line profiles", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi)
    plt.close(fig)
    return path


def plot_uncertainty(result: RunResult, path: Path, cfg: OutputConfig) -> Path:
    """Per-point uncertainty map for a representative frame plus the noise-floor summary."""
    if result.uncertainty is None:
        return path
    # choose the frame with the largest deformation for a representative map
    f = max(result.frames, key=lambda fr: np.nanmax(np.hypot(fr.ux, fr.uy)) if fr.valid.any() else 0)
    base, X, Y, deformed = _view(result, f, cfg)
    su = _field_grid(f.sigma_u, result.xs, result.ys, result.ny, result.nx)
    sv = _field_grid(f.sigma_v, result.xs, result.ys, result.ny, result.nx)
    zncc = _field_grid(f.zncc, result.xs, result.ys, result.ny, result.nx)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    _overlay(axes[0], base, X, Y, su, r"$\sigma_u$ per point", "px", cfg.contour_levels, 0.7)
    _overlay(axes[1], base, X, Y, sv, r"$\sigma_v$ per point", "px", cfg.contour_levels, 0.7)
    _overlay(axes[2], base, X, Y, zncc, "ZNCC (correlation quality)", "-", cfg.contour_levels, 0.7)
    for ax in axes:
        _draw_roi(ax, result, f, deformed)
    u = result.uncertainty
    fig.suptitle(
        f"Uncertainty map - frame {f.index + 1} | image noise {u.image_noise_sigma:.2f} grey "
        f"({u.image_noise_method}) | noise floor u RMS {u.noise_floor_u_rms_px:.4f} px, "
        f"v RMS {u.noise_floor_v_rms_px:.4f} px",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi)
    plt.close(fig)
    return path


def animate(result: RunResult, field: str, path: Path, cfg: OutputConfig) -> Path:
    """Animate a field growing over time.

    With the default ``overlay_frame = "deformed"`` the base image is the **current frame**, so
    the specimen visibly deforms as the animation runs and the field follows the material.

    Parameters
    ----------
    field : str
        ``"umag"`` for displacement magnitude, or a strain field name such as ``"exx"``.
    """
    Xr, Yr = np.meshgrid(np.unique(result.xs), np.unique(result.ys))
    deformed = getattr(cfg, "overlay_frame", "deformed") != "reference"
    coords: list[tuple[np.ndarray, np.ndarray]] = []
    arrays: list[np.ndarray] = []
    for f in result.frames:
        ux = _field_grid(f.ux, result.xs, result.ys, result.ny, result.nx)
        uy = _field_grid(f.uy, result.xs, result.ys, result.ny, result.nx)
        if field == "umag":
            arr = np.hypot(ux, uy)
        else:
            arr = f.strain[field]
        arrays.append(arr)
        coords.append((Xr + ux, Yr + uy) if deformed else (Xr, Yr))

    finite_min = [np.nanmin(a) for a in arrays if np.isfinite(a).any()]
    finite_max = [np.nanmax(a) for a in arrays if np.isfinite(a).any()]
    if not finite_min:
        return path
    vmin = float(np.percentile(finite_min, 1))
    vmax = float(np.percentile(finite_max, 99))
    if not np.isfinite(vmin) or vmin == vmax:
        vmax = vmin + 1e-9
    levels = np.linspace(vmin, vmax, cfg.contour_levels)

    ref_base = _reference_image(result) if not deformed else None
    height, width = result.shape
    label = "displacement magnitude [px]" if field == "umag" else f"{field} [-]"

    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)

    def update(k):
        base = ref_base if ref_base is not None else _frame_image(result, k, max_cache=1)
        Xk, Yk = coords[k]
        ax.clear()
        ax.imshow(base, cmap="gray", origin="upper")
        ax.contourf(Xk, Yk, arrays[k], levels=levels, cmap="turbo",
                    alpha=cfg.overlay_alpha, extend="both")
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label} - frame {k + 1}/{len(arrays)}", fontsize=10)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(arrays), blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = cfg.animation_fps
    if cfg.animation_format == "mp4":
        try:
            ani.save(path, fps=fps, dpi=90)
        except Exception as exc:  # pragma: no cover - ffmpeg missing
            logger.warning("mp4 animation failed (%s); falling back to GIF.", exc)
            path = path.with_suffix(".gif")
            ani.save(path, dpi=90, writer=animation.PillowWriter(fps=fps))
    else:
        ani.save(path, dpi=90, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


_WORKER_STATE: dict[str, object] = {}


def _worker_init(result: RunResult, cfg: OutputConfig) -> None:
    """Store the shared state in each worker process (avoids re-pickling per task)."""
    _WORKER_STATE["result"] = result
    _WORKER_STATE["cfg"] = cfg


def _worker_render(kind_and_index: tuple[str, int]) -> str:
    """Render one figure in a worker process."""
    kind, i = kind_and_index
    result = _WORKER_STATE["result"]  # type: ignore[assignment]
    cfg = _WORKER_STATE["cfg"]  # type: ignore[assignment]
    frame = result.frames[i]  # type: ignore[attr-defined]
    if kind == "displacement":
        path = Path(_WORKER_STATE["out"]) / "figures" / "displacement" / f"disp_{frame.index + 1:04d}.png"
        return str(plot_displacement(result, frame, path, cfg))  # type: ignore[arg-type]
    path = Path(_WORKER_STATE["out"]) / "figures" / "strain" / f"strain_{frame.index + 1:04d}.png"
    return str(plot_strain(result, frame, path, cfg))  # type: ignore[arg-type]


def render_all_figures(
    result: RunResult,
    output_dir: str | Path,
    *,
    progress: callable | None = None,
) -> dict[str, list[Path]]:
    """Render the per-frame overlays plus every extra visualization.

    Parameters
    ----------
    result : RunResult
    output_dir : str or pathlib.Path
    progress : callable, optional
        Progress callback ``progress(index, total, message)``.

    Notes
    -----
    ``RunConfig.jobs`` > 1 renders the per-frame figures in that many worker processes (the
    correlation itself is already vectorised and is not parallelised across frames because the
    temporal initial guess makes frames sequential).
    """
    output_dir = Path(output_dir)
    cfg = result.cfg.output
    written: dict[str, list[Path]] = {"displacement": [], "strain": [], "extras": []}
    if not cfg.figures and not cfg.extras:
        return written

    disp_dir = output_dir / "figures" / "displacement"
    strain_dir = output_dir / "figures" / "strain"
    jobs = max(1, int(result.cfg.jobs))
    if cfg.figures and jobs > 1 and result.n_frames > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        _WORKER_STATE["out"] = output_dir
        tasks = [("displacement", i) for i in range(result.n_frames)]
        tasks += [("strain", i) for i in range(result.n_frames)]
        ctx = mp.get_context("fork")
        try:
            with ProcessPoolExecutor(
                max_workers=jobs, mp_context=ctx,
                initializer=_worker_init, initargs=(result, cfg),
            ) as ex:
                paths = list(ex.map(_worker_render, tasks))
            written["displacement"] = [Path(p) for p in paths[: result.n_frames]]
            written["strain"] = [Path(p) for p in paths[result.n_frames:]]
        except Exception as exc:  # pragma: no cover - fall back to serial
            logger.warning("Parallel figure rendering failed (%s); rendering serially.", exc)
            written["displacement"] = []
            written["strain"] = []
    if cfg.figures and not written["displacement"]:
        for i, f in enumerate(result.frames):
            if progress:
                progress(i, result.n_frames, f"figures {i + 1}/{result.n_frames}")
            written["displacement"].append(
                plot_displacement(result, f, disp_dir / f"disp_{f.index + 1:04d}.png", cfg)
            )
            written["strain"].append(
                plot_strain(result, f, strain_dir / f"strain_{f.index + 1:04d}.png", cfg)
            )

    if cfg.extras:
        extra_dir = output_dir / "extras"
        written["extras"].append(plot_curves(result, extra_dir / "curves.png", cfg))
        written["extras"].append(plot_profiles(result, extra_dir / "profiles.png", cfg))
        if result.uncertainty is not None:
            written["extras"].append(plot_uncertainty(result, extra_dir / "uncertainty.png", cfg))
        anim_dir = output_dir / "animations"
        suffix = ".mp4" if cfg.animation_format == "mp4" else ".gif"
        written["extras"].append(
            animate(result, "umag", anim_dir / f"displacement{suffix}", cfg)
        )
        written["extras"].append(animate(result, "exx", anim_dir / f"exx{suffix}", cfg))
    return written
