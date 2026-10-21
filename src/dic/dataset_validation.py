"""Validation against a provided dataset with analytic ground truth (requirement 5.5a).

Reads the ``ground_truth_grid.csv`` / ``ground_truth.csv`` written by the dataset generator,
interpolates the recovered fields onto the ground-truth grid, and reports RMS / maximum errors,
coverage and the noise floor.  The conventions documented in the dataset's ``metadata.json`` are
respected:

* coordinates are pixel centres in the **reference** image (``X_px``, ``Y_px``);
* ``ux_px = x - X``, ``uy_px = y - Y``;
* the comparable strain components are the nominal (Biot) strains ``Eeng_xx``, ``Eeng_yy`` and
  ``Eeng_gamma_xy`` in the image frame, computed from the ground-truth deformation gradient
  ``J``.  If the file has no ``J`` columns, the generator's engineering components
  (``eps_xx_img`` in the provided dataset, or ``Eeng_xx`` in generated files) are used as a
  fallback (the two coincide only in the small-strain limit);
* ``in_ref`` / ``in_def`` flag whether a point is inside the reference / deformed image; a point
  is only comparable when both are 1.  Files without these flags (the demo generator) are treated
  as fully visible.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .kinematics import left_stretch
from .pipeline import RunResult

logger = logging.getLogger(__name__)


def _float(row: dict[str, str], *keys: str, default: float = np.nan) -> float:
    """First finite value among *keys* in a CSV row."""
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return default


def load_ground_truth_grid(path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """Load ``ground_truth_grid.csv`` into a per-frame dict of arrays.

    The nominal strain is derived from the ``J11..J22`` columns when they are present;
    otherwise the generator's engineering components are kept as a fallback.  Both the
    provided-dataset headers (``eps_*_img``) and the demo-generator headers (``Eeng_*``) are
    accepted.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    by_frame: dict[int, dict[str, list]] = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        frame_key = "image_index" if "image_index" in (reader.fieldnames or []) else "frame"
        for row in reader:
            idx = int(row[frame_key])
            d = by_frame.setdefault(idx, {k: [] for k in (
                "X", "Y", "ux", "uy", "Eeng_xx_eng", "Eeng_yy_eng", "Eeng_gamma_xy_eng",
                "J11", "J12", "J21", "J22", "in_ref", "in_def")})
            d["X"].append(_float(row, "X_px"))
            d["Y"].append(_float(row, "Y_px"))
            d["ux"].append(_float(row, "ux_px"))
            d["uy"].append(_float(row, "uy_px"))
            d["Eeng_xx_eng"].append(_float(row, "eps_xx_img", "Eeng_xx"))
            d["Eeng_yy_eng"].append(_float(row, "eps_yy_img", "Eeng_yy"))
            d["Eeng_gamma_xy_eng"].append(_float(row, "gamma_xy_img", "Eeng_gamma_xy"))
            for key in ("J11", "J12", "J21", "J22"):
                d[key].append(_float(row, key))
            d["in_ref"].append(_float(row, "in_ref", default=1.0))
            d["in_def"].append(_float(row, "in_def", default=1.0))
    out: dict[int, dict[str, np.ndarray]] = {}
    for idx, values in by_frame.items():
        arrays = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        if all(np.isfinite(arrays[k]).all() for k in ("J11", "J12", "J21", "J22")):
            V11, V12, V22 = left_stretch(
                arrays["J11"], arrays["J12"], arrays["J21"], arrays["J22"]
            )
            arrays["Eeng_xx"] = V11 - 1.0
            arrays["Eeng_yy"] = V22 - 1.0
            arrays["Eeng_gamma_xy"] = 2.0 * V12
        else:
            arrays["Eeng_xx"] = arrays["Eeng_xx_eng"]
            arrays["Eeng_yy"] = arrays["Eeng_yy_eng"]
            arrays["Eeng_gamma_xy"] = arrays["Eeng_gamma_xy_eng"]
        out[idx] = arrays
    return out


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
    for name in ("ux", "uy", "Eeng_xx", "Eeng_yy", "Eeng_gamma_xy"):
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
        Eeng_xx = f.strain["Eeng_xx"]
        Eeng_yy = f.strain["Eeng_yy"]
        Eeng_gamma_xy = f.strain["Eeng_gamma_xy"]
        interp = {
            "ux": RegularGridInterpolator((ny, nx), ux, bounds_error=False, fill_value=np.nan),
            "uy": RegularGridInterpolator((ny, nx), uy, bounds_error=False, fill_value=np.nan),
            "Eeng_xx": RegularGridInterpolator(
                (ny, nx), Eeng_xx, bounds_error=False, fill_value=np.nan
            ),
            "Eeng_yy": RegularGridInterpolator(
                (ny, nx), Eeng_yy, bounds_error=False, fill_value=np.nan
            ),
            "Eeng_gamma_xy": RegularGridInterpolator(
                (ny, nx), Eeng_gamma_xy, bounds_error=False, fill_value=np.nan
            ),
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
        strain_ok = (
            both
            & np.isfinite(got["Eeng_xx"])
            & np.isfinite(got["Eeng_yy"])
            & np.isfinite(got["Eeng_gamma_xy"])
        )
        if strain_ok.any():
            ee = got["Eeng_xx"][strain_ok] - gt["Eeng_xx"][strain_ok]
            ey = got["Eeng_yy"][strain_ok] - gt["Eeng_yy"][strain_ok]
            gg = (
                got["Eeng_gamma_xy"][strain_ok]
                - gt["Eeng_gamma_xy"][strain_ok]
            )
            row["Eeng_xx_rms"] = float(np.sqrt(np.mean(ee ** 2)))
            row["Eeng_xx_max"] = float(np.max(np.abs(ee)))
            row["Eeng_yy_rms"] = float(np.sqrt(np.mean(ey ** 2)))
            row["Eeng_yy_max"] = float(np.max(np.abs(ey)))
            row["Eeng_gamma_xy_rms"] = float(np.sqrt(np.mean(gg ** 2)))
            row["Eeng_gamma_xy_max"] = float(np.max(np.abs(gg)))
            rel = np.abs(ee) / np.maximum(np.abs(gt["Eeng_xx"][strain_ok]), 1e-3)
            row["Eeng_xx_rel_rms"] = float(np.sqrt(np.mean(rel ** 2)))
        row["coverage"] = float(recovered[comparable].mean()) if comparable.any() else 0.0
        row["n_comparable"] = float(comparable.sum())
        rows.append(row)

    agg = {}
    keys = set().union(*[r.keys() for r in rows]) if rows else set()
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        agg[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return {"rows": rows, "aggregate": agg}


def _nominal_Eeng_xx_mean(row: dict[str, float]) -> float:
    """Image-mean nominal ``Eeng_xx`` from a ``ground_truth.csv`` row.

    ``F11..F22`` are present in the provided dataset and describe the affine deformation.
    If they are absent, fall back to the generator's engineering mean (small-strain limit).
    """
    if all(key in row for key in ("F11", "F12", "F21", "F22")):
        V11, _, _ = left_stretch(row["F11"], row["F12"], row["F21"], row["F22"])
        return float(V11 - 1.0)
    return float(row.get("eps_xx_img_mean", row.get("Eeng_xx_mean", np.nan)))


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
        ("Strain Eeng_xx RMS [-]", "Eeng_xx_rms"),
        ("Strain Eeng_xx max abs [-]", "Eeng_xx_max"),
        ("Strain Eeng_yy RMS [-]", "Eeng_yy_rms"),
        ("Strain Eeng_yy max abs [-]", "Eeng_yy_max"),
        ("Strain Eeng_gamma_xy RMS [-]", "Eeng_gamma_xy_rms"),
        ("Strain Eeng_gamma_xy max abs [-]", "Eeng_gamma_xy_max"),
        ("Relative Eeng_xx error RMS", "Eeng_xx_rel_rms"),
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
    add(
        "| Frame | Comparable pts | Coverage | u RMS | u max | v RMS | v max | "
        "Eeng_xx RMS | Eeng_xx max | Eeng_yy RMS | Eeng_gamma_xy RMS |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    step = max(1, len(rows) // 25)
    for i in range(0, len(rows), step):
        r = rows[i]
        add(
            f"| {int(r['frame'])} | {int(r.get('n_comparable', 0))} | "
            f"{100 * r.get('coverage', float('nan')):.1f}% | "
            f"{_fmt(r.get('u_rms'))} | {_fmt(r.get('u_max'))} | {_fmt(r.get('v_rms'))} | "
            f"{_fmt(r.get('v_max'))} | {_fmt(r.get('Eeng_xx_rms'))} | "
            f"{_fmt(r.get('Eeng_xx_max'))} | {_fmt(r.get('Eeng_yy_rms'))} | "
            f"{_fmt(r.get('Eeng_gamma_xy_rms'))} |"
        )
    add("")
    # compare the recovered mean strain with the generator's own mean (ground_truth.csv)
    if gt_images is not None:
        mine = np.array([np.nanmean(f.strain["Eeng_xx"]) for f in result.frames])
        theirs = np.array([_nominal_Eeng_xx_mean(gt_images.get(f.index, {}))
                           for f in result.frames])
        ok = np.isfinite(mine) & np.isfinite(theirs)
        if ok.any():
            add(
                f"* Mean `Eeng_xx` (nominal) versus the generator's own image-mean nominal "
                f"strain: RMS difference {_fmt(np.sqrt(np.mean((mine[ok] - theirs[ok]) ** 2)))} "
                f"(final frame: recovered {_fmt(mine[ok][-1])}, truth {_fmt(theirs[ok][-1])})."
            )
            add("")
    add("### Interpretation\n")
    u_rms = agg.get("u_rms", float("nan"))
    e_rms = agg.get("Eeng_xx_rms", float("nan"))
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
