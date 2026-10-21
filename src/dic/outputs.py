"""Machine-readable output: parameters dump, mask, per-frame fields and time series (R12).

Everything a user needs to post-process in Python/MATLAB without reading pixels off figures:

* ``params.json``    - the exact configuration, package versions, input file names and SHA-256.
* ``mask_used.png`` / ``mask_used.npz`` - the ROI actually used.
* ``fields/frame_XXXX.npz`` - per-frame arrays on the correlation grid.
* ``displacement_timeseries.csv`` / ``strain_timeseries.csv`` - per-frame summaries.
* ``grid.csv`` - the correlation-grid coordinates and the mask validity.
* ``fields_long.csv`` - optional long-form per-point dump.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .pipeline import RunResult

logger = logging.getLogger(__name__)

#: Fields written to every per-frame NPZ archive.
NPZ_FIELDS = (
    "ux",
    "uy",
    "valid",
    "zncc",
    "sigma_u",
    "sigma_v",
    "residual",
    "iterations",
)
STRAIN_FIELDS = (
    "du_dx",
    "du_dy",
    "dv_dx",
    "dv_dy",
    "Eeng_xx",
    "Eeng_yy",
    "Eeng_xy",
    "Eeng_gamma_xy",
    "Eeng_1",
    "Eeng_2",
    "Eeng_eq",
    "Etr_xx",
    "Etr_yy",
    "Etr_xy",
    "Etr_gamma_xy",
    "Etr_1",
    "Etr_2",
    "Etr_eq",
    "omega",
    "theta_deg",
    "theta_p_deg",
)


def package_versions() -> dict[str, str]:
    """Versions of the runtime and the scientific stack (reproducibility)."""
    import matplotlib
    import PIL
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dic": __version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__,
    }


def write_params(result: RunResult, output_dir: Path) -> Path:
    """Write ``params.json``: settings, versions and input provenance."""
    payload = {
        "config": result.cfg.to_dict(),
        "versions": package_versions(),
        "input": {
            "n_images": result.n_frames,
            "shape": list(result.shape),
            "files": [
                {"name": p.name, "sha256": h}
                for p, h in zip(result.image_paths, result.image_hashes, strict=False)
            ],
        },
        "grid": {
            "n_points": int(result.xs.size),
            "nx": result.nx,
            "ny": result.ny,
            "dx_px": result.dx,
            "dy_px": result.dy,
        },
        "mask": {"source": result.mask_source, "coverage": float(result.mask.mean())},
    }
    path = output_dir / "params.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def write_mask(result: RunResult, output_dir: Path) -> Path:
    """Write the mask used, as PNG (for inspection) and NPZ (for reuse)."""
    from PIL import Image

    mask_dir = output_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((result.mask * 255).astype(np.uint8)).save(mask_dir / "mask_used.png")
    np.savez_compressed(mask_dir / "mask_used.npz", mask=result.mask)
    return mask_dir / "mask_used.png"


def write_grid(result: RunResult, output_dir: Path) -> Path:
    """Write the correlation grid coordinates and validity."""
    path = output_dir / "grid.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["index", "x_px", "y_px"])
        for i in range(result.xs.size):
            wr.writerow([i, f"{result.xs[i]:.4f}", f"{result.ys[i]:.4f}"])
    return path


def write_frames(result: RunResult, output_dir: Path) -> list[Path]:
    """Write one NPZ per frame with displacement, quality and strain fields."""
    fields_dir = output_dir / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f in result.frames:
        arrays = {
            "x_px": result.xs,
            "y_px": result.ys,
            "valid": f.valid,
            "zncc": f.zncc,
            "sigma_u": f.sigma_u,
            "sigma_v": f.sigma_v,
            "residual": f.residual,
            "iterations": f.iterations,
        }
        for name in NPZ_FIELDS:
            if name in arrays:
                continue
            arrays[name] = getattr(f, name, np.full(result.xs.size, np.nan))
        # strain fields are stored on the (ny, nx) grid
        for name in STRAIN_FIELDS:
            arrays[f"strain_{name}"] = f.strain[name]
        arrays["strain_valid"] = f.strain.valid
        path = fields_dir / f"frame_{f.index + 1:04d}.npz"
        np.savez_compressed(path, **arrays)
        written.append(path)
    return written


def write_timeseries(result: RunResult, output_dir: Path) -> tuple[Path, Path]:
    """Write per-frame displacement and strain summary CSVs."""
    dpath = output_dir / "displacement_timeseries.csv"
    spath = output_dir / "strain_timeseries.csv"
    with open(dpath, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "frame",
                "image",
                "time_s",
                "t_load",
                "coverage",
                "u_mean_px",
                "v_mean_px",
                "u_max_px",
                "v_max_px",
                "umag_mean_px",
                "umag_max_px",
                "zncc_median",
                "sigma_u_mean_px",
                "sigma_v_mean_px",
            ]
        )
        for f in result.frames:
            m = f.valid
            u = f.ux[m]
            v = f.uy[m]
            mag = np.hypot(u, v)
            wr.writerow(
                [
                    f.index,
                    f.name,
                    f"{f.time:.6g}",
                    f"{f.t_load:.6g}",
                    f"{f.coverage:.5f}",
                    f"{np.mean(u):.6g}" if m.any() else "nan",
                    f"{np.mean(v):.6g}" if m.any() else "nan",
                    f"{np.max(np.abs(u)):.6g}" if m.any() else "nan",
                    f"{np.max(np.abs(v)):.6g}" if m.any() else "nan",
                    f"{np.mean(mag):.6g}" if m.any() else "nan",
                    f"{np.max(mag):.6g}" if m.any() else "nan",
                    f"{np.nanmedian(f.zncc):.6g}",
                    f"{f.mean_sigma_u:.6g}",
                    f"{f.mean_sigma_v:.6g}",
                ]
            )
    with open(spath, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "frame",
                "image",
                "time_s",
                "t_load",
                "strain_coverage",
                "Eeng_xx_mean",
                "Eeng_yy_mean",
                "Eeng_gamma_xy_mean",
                "Eeng_eq_mean",
                "Eeng_eq_max",
                "Eeng_xx_max_abs",
                "Eeng_yy_max_abs",
                "Etr_xx_mean",
                "Etr_yy_mean",
            ]
        )
        for f in result.frames:
            s = f.strain
            wr.writerow(
                [
                    f.index,
                    f.name,
                    f"{f.time:.6g}",
                    f"{f.t_load:.6g}",
                    f"{np.mean(s.valid):.5f}",
                    _fmt(s["Eeng_xx"]),
                    _fmt(s["Eeng_yy"]),
                    _fmt(s["Eeng_gamma_xy"]),
                    _fmt(s["Eeng_eq"]),
                    _fmt(s["Eeng_eq"], np.nanmax),
                    _fmt(s["Eeng_xx"], lambda a: np.nanmax(np.abs(a))),
                    _fmt(s["Eeng_yy"], lambda a: np.nanmax(np.abs(a))),
                    _fmt(s["Etr_xx"]),
                    _fmt(s["Etr_yy"]),
                ]
            )
    return dpath, spath


def _fmt(field: np.ndarray, fn=np.nanmean) -> str:
    vals = field[np.isfinite(field)]
    if vals.size == 0:
        return "nan"
    return f"{fn(field):.6g}"


def write_long_csv(result: RunResult, output_dir: Path) -> Path:
    """Write a long-form per-point CSV (optional, can be large)."""
    path = output_dir / "fields_long.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            ["frame", "point", "x_px", "y_px", "ux_px", "uy_px", "valid", "zncc", "sigma_u", "sigma_v"]
        )
        for f in result.frames:
            for i in range(result.xs.size):
                wr.writerow(
                    [
                        f.index,
                        i,
                        f"{result.xs[i]:.3f}",
                        f"{result.ys[i]:.3f}",
                        f"{f.ux[i]:.6g}",
                        f"{f.uy[i]:.6g}",
                        int(f.valid[i]),
                        f"{f.zncc[i]:.6g}",
                        f"{f.sigma_u[i]:.6g}",
                        f"{f.sigma_v[i]:.6g}",
                    ]
                )
    return path


def write_all(result: RunResult, output_dir: str | Path) -> dict[str, object]:
    """Write every machine-readable output.  Returns a dict of paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, object] = {}
    written["params"] = write_params(result, output_dir)
    written["mask"] = write_mask(result, output_dir)
    written["grid"] = write_grid(result, output_dir)
    written["frames"] = write_frames(result, output_dir) if result.cfg.output.save_npz else []
    dpath, spath = write_timeseries(result, output_dir)
    written["displacement_timeseries"] = dpath
    written["strain_timeseries"] = spath
    if result.cfg.output.save_csv:
        written["fields_long"] = write_long_csv(result, output_dir)
    logger.info("Wrote machine-readable outputs to %s", output_dir)
    return written
