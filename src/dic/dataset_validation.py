"""Validation against a provided dataset with analytic ground truth (requirement 5.5a).

Reads the ``ground_truth_grid.csv`` / ``ground_truth.csv`` written by the dataset generator,
interpolates the recovered fields onto the ground-truth grid, and reports RMS / maximum errors,
coverage and the noise floor.  The conventions documented in the dataset's ``metadata.json`` are
respected:

* coordinates are pixel centres in the **reference** image (``X_px``, ``Y_px``);
* ``ux_px = x - X``, ``uy_px = y - Y``;
* the comparable strain components are ``eps_xx_img``, ``eps_yy_img``, ``gamma_xy_img``
  (components of ``sym(J) - I`` in the image frame), which correspond exactly to the engineering
  strain reported here;
* ``in_ref`` / ``in_def`` flag whether a point is inside the reference / deformed image; a point
  is only comparable when both are 1.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .pipeline import RunResult

logger = logging.getLogger(__name__)


def load_ground_truth_grid(path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """Load ``ground_truth_grid.csv`` into a per-frame dict of arrays."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    by_frame: dict[int, dict[str, list]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            idx = int(row["image_index"])
            d = by_frame.setdefault(idx, {k: [] for k in (
                "X", "Y", "ux", "uy", "exx", "eyy", "gxy", "in_ref", "in_def")})
            d["X"].append(float(row["X_px"]))
            d["Y"].append(float(row["Y_px"]))
            d["ux"].append(float(row["ux_px"]))
            d["uy"].append(float(row["uy_px"]))
            d["exx"].append(float(row["eps_xx_img"]))
            d["eyy"].append(float(row["eps_yy_img"]))
            d["gxy"].append(float(row["gamma_xy_img"]))
            d["in_ref"].append(int(row["in_ref"]))
            d["in_def"].append(int(row["in_def"]))
    return {
        k: {kk: np.asarray(vv, dtype=float) for kk, vv in v.items()} for k, v in by_frame.items()
    }


def load_ground_truth_images(path: str | Path) -> dict[int, dict[str, float]]:
    """Load the per-frame scalar ground truth from ``ground_truth.csv``."""
    path = Path(path)
    out: dict[int, dict[str, float]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            idx = int(row["image_index"])
            out[idx] = {k: float(v) for k, v in row.items() if k not in ("image",)}
    return out


def _grid_interpolators(result: RunResult) -> dict[str, RegularGridInterpolator]:
    """Interpolators for the recovered fields over the regular correlation grid."""
    nx = np.unique(result.xs)
    ny = np.unique(result.ys)
    out = {}
    for name in ("ux", "uy", "exx", "eyy", "gamma_xy"):
        field = np.full((ny.size, nx.size), np.nan)
        out[name] = field
    out["_nx"] = nx
    out["_ny"] = ny
    return out


def compare_with_ground_truth(
    result: RunResult,
    gt_grid: dict[int, dict[str, np.ndarray]],
) -> dict[str, object]:
    """Compare the run against the analytic ground truth on the ground-truth grid.

    Returns a dict with per-frame rows and aggregate statistics.
    """
    nx = np.unique(result.xs)
    ny = np.unique(result.ys)
    xmin, xmax = nx.min(), nx.max()
    ymin, ymax = ny.min(), ny.max()

    rows: list[dict[str, float]] = []
    for f in result.frames:
        gt = gt_grid.get(f.index)
        if gt is None:
            continue
        # recovered fields on the correlation grid
        ux = _to_grid(f.ux, result.xs, result.ys, ny, nx)
        uy = _to_grid(f.uy, result.xs, result.ys, ny, nx)
        exx = f.strain["exx"]
        eyy = f.strain["eyy"]
        gxy = f.strain["gamma_xy"]
        interp = {
            "ux": RegularGridInterpolator((ny, nx), ux, bounds_error=False, fill_value=np.nan),
            "uy": RegularGridInterpolator((ny, nx), uy, bounds_error=False, fill_value=np.nan),
            "exx": RegularGridInterpolator((ny, nx), exx, bounds_error=False, fill_value=np.nan),
            "eyy": RegularGridInterpolator((ny, nx), eyy, bounds_error=False, fill_value=np.nan),
            "gamma_xy": RegularGridInterpolator((ny, nx), gxy, bounds_error=False, fill_value=np.nan),
        }
        pts = np.stack([gt["Y"], gt["X"]], axis=1)
        got = {k: interp[k](pts) for k in interp}

        # comparable = inside the reference image, inside the deformed image, and inside our grid
        inside_grid = (
            (gt["X"] >= xmin) & (gt["X"] <= xmax) & (gt["Y"] >= ymin) & (gt["Y"] <= ymax)
        )
        comparable = (gt["in_ref"] > 0.5) & (gt["in_def"] > 0.5) & inside_grid
        recovered = np.isfinite(got["ux"]) & np.isfinite(got["uy"])
        both = comparable & recovered

        row: dict[str, float] = {"frame": float(f.index + 1)}
        if both.any():
            eu = got["ux"][both] - gt["ux"][both]
            ev = got["uy"][both] - gt["uy"][both]
            row["u_rms"] = float(np.sqrt(np.mean(eu ** 2)))
            row["u_max"] = float(np.max(np.abs(eu)))
            row["v_rms"] = float(np.sqrt(np.mean(ev ** 2)))
            row["v_max"] = float(np.max(np.abs(ev)))
            row["bias_u"] = float(np.mean(eu))
            row["bias_v"] = float(np.mean(ev))
        strain_ok = both & np.isfinite(got["exx"]) & np.isfinite(got["eyy"]) & np.isfinite(got["gamma_xy"])
        if strain_ok.any():
            ee = got["exx"][strain_ok] - gt["exx"][strain_ok]
            ey = got["eyy"][strain_ok] - gt["eyy"][strain_ok]
            gg = got["gamma_xy"][strain_ok] - gt["gxy"][strain_ok]
            row["exx_rms"] = float(np.sqrt(np.mean(ee ** 2)))
            row["exx_max"] = float(np.max(np.abs(ee)))
            row["eyy_rms"] = float(np.sqrt(np.mean(ey ** 2)))
            row["eyy_max"] = float(np.max(np.abs(ey)))
            row["gxy_rms"] = float(np.sqrt(np.mean(gg ** 2)))
            row["gxy_max"] = float(np.max(np.abs(gg)))
            rel = np.abs(ee) / np.maximum(np.abs(gt["exx"][strain_ok]), 1e-3)
            row["exx_rel_rms"] = float(np.sqrt(np.mean(rel ** 2)))
        row["coverage"] = float(recovered[comparable].mean()) if comparable.any() else 0.0
        row["n_comparable"] = float(comparable.sum())
        rows.append(row)

    agg = {}
    keys = set().union(*[r.keys() for r in rows]) if rows else set()
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        agg[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return {"rows": rows, "aggregate": agg}


def _to_grid(
    values: np.ndarray, xs: np.ndarray, ys: np.ndarray, ny: int, nx: int
) -> np.ndarray:
    ux = np.unique(xs)
    uy = np.unique(ys)
    g = np.full((uy.size, ux.size), np.nan)
    g[np.searchsorted(uy, ys), np.searchsorted(ux, xs)] = values
    return g


def format_dataset_section(
    result: RunResult,
    comparison: dict[str, object],
    gt_images: dict[int, dict[str, float]] | None = None,
) -> str:
    """Render the provided-dataset validation as a Markdown section."""
    rows = comparison["rows"]  # type: ignore[index]
    agg = comparison["aggregate"]  # type: ignore[index]
    lines: list[str] = []
    add = lines.append
    add("## A. Provided dataset (`Data01/`) against analytic ground truth\n")
    add(f"* {result.n_frames} frames, {result.shape[1]}x{result.shape[0]} px, "
        f"grid {result.nx} x {result.ny} = {result.xs.size} points.")
    add(f"* Subset {result.cfg.correlation.subset_size} px, step {result.cfg.correlation.step} px, "
        f"pre-filter sigma {result.cfg.correlation.prefilter_sigma} px, "
        f"strain window {result.cfg.strain.window}x{result.cfg.strain.window} "
        f"(VSG {result.frames[-1].strain.gauge_length_px:.0f} px).")
    add(f"* ROI: {result.mask_source}.")
    add("* Comparison uses `ground_truth_grid.csv`, restricted to points with `in_ref = in_def = 1` "
        "(visible in both images) and inside the correlation grid. Late frames legitimately lose "
        "the outer field of view, so the comparable region shrinks - this is correct behaviour, "
        "not a failure.")
    add("")
    add("### Aggregate (mean over frames where comparable)\n")
    add("| Metric | Value |")
    add("|---|---|")
    for label, key in [
        ("Displacement u RMS [px]", "u_rms"),
        ("Displacement u max [px]", "u_max"),
        ("Displacement v RMS [px]", "v_rms"),
        ("Displacement v max [px]", "v_max"),
        ("Bias u [px]", "bias_u"),
        ("Bias v [px]", "bias_v"),
        ("Strain exx RMS [-]", "exx_rms"),
        ("Strain exx max abs [-]", "exx_max"),
        ("Strain eyy RMS [-]", "eyy_rms"),
        ("Strain eyy max abs [-]", "eyy_max"),
        ("Strain gamma_xy RMS [-]", "gxy_rms"),
        ("Strain gamma_xy max abs [-]", "gxy_max"),
        ("Relative exx error RMS", "exx_rel_rms"),
        ("Coverage of comparable points", "coverage"),
    ]:
        v = agg.get(key, float("nan"))
        add(f"| {label} | {_fmt(v)} |")
    add("")
    if result.uncertainty is not None:
        u = result.uncertainty
        add(f"* Image noise (estimated): {_fmt(u.image_noise_sigma, 3)} grey levels "
            f"({u.image_noise_method}).")
        add(f"* **Measured noise floor** (zero motion, same settings): "
            f"u RMS {_fmt(u.noise_floor_u_rms_px)} px, v RMS {_fmt(u.noise_floor_v_rms_px)} px, "
            f"strain RMS {_fmt(u.noise_floor_strain_rms)}.")
        add("")
    add("### Per-frame errors\n")
    add("| Frame | Comparable pts | Coverage | u RMS | u max | v RMS | v max | exx RMS | exx max | eyy RMS | gamma RMS |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    step = max(1, len(rows) // 25)
    for i in range(0, len(rows), step):
        r = rows[i]
        add(
            f"| {int(r['frame'])} | {int(r.get('n_comparable', 0))} | "
            f"{100 * r.get('coverage', float('nan')):.1f}% | "
            f"{_fmt(r.get('u_rms'))} | {_fmt(r.get('u_max'))} | {_fmt(r.get('v_rms'))} | "
            f"{_fmt(r.get('v_max'))} | {_fmt(r.get('exx_rms'))} | {_fmt(r.get('exx_max'))} | "
            f"{_fmt(r.get('eyy_rms'))} | {_fmt(r.get('gxy_rms'))} |"
        )
    add("")
    # compare the recovered mean strain with the generator's own mean (ground_truth.csv)
    if gt_images is not None:
        mine = np.array([np.nanmean(f.strain["exx"]) for f in result.frames])
        theirs = np.array([gt_images.get(f.index, {}).get("eps_xx_img_mean", np.nan)
                           for f in result.frames])
        ok = np.isfinite(mine) & np.isfinite(theirs)
        if ok.any():
            add(f"* Mean `exx` versus the generator's own image-mean `eps_xx_img_mean`: "
                f"RMS difference {_fmt(np.sqrt(np.mean((mine[ok] - theirs[ok]) ** 2)))} "
                f"(final frame: recovered {_fmt(mine[ok][-1])}, truth {_fmt(theirs[ok][-1])}).")
            add("")
    add("### Interpretation\n")
    u_rms = agg.get("u_rms", float("nan"))
    e_rms = agg.get("exx_rms", float("nan"))
    if np.isfinite(u_rms):
        add(f"* Displacement RMS error {_fmt(u_rms)} px against the analytic field; the analytic "
            f"deformation is affine, for which the affine shape function is exact, so this is the "
            f"combined interpolation-bias and noise floor.")
    if np.isfinite(e_rms):
        add(f"* Strain RMS error {_fmt(e_rms)} absolute on a strain field reaching 0.40 - i.e. "
            f"about {_fmt(100 * e_rms)} percentage points.")
    add("* The reported coverage is the fraction of *comparable* ground-truth points where the "
        "pipeline produced a valid (non-NaN) result.")
    add("")
    return "\n".join(lines)


def _fmt(v, nd: int = 4) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}g}"
